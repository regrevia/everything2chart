"""Automatic piano fingering, sparse by default.

The cost model is a port of the published approach behind Blake West's
``performer`` (MIT) and Kanma's ``piano_fingering`` (MIT), which is the same
model with a Python wrapper.  Its numbers come from regression against real
fingerings, so they are reused rather than re-invented:

* Moving a hand while keeping one finger costs a fixed reposition (~4) plus a
  per-semitone term -- this is what stops every note being played by the thumb.
* Two adjacent fingers walking stepwise in the same direction as the music are
  almost free, because the hand barely moves.
* Thumb-under / thumb-over crossings are priced by their own curve: cheap at
  the interval they are actually used for, expensive everywhere else.
* Thumb and pinky on a black key are taxed.

What is ours: input is ``NoteEvent`` (no note-off pairing), a note is only
fingered when the strike is physically reachable, and ``mode="key"`` prints
numbers only where a player needs to read them.

Fingering is an annotation. It never changes pitch, timing or duration.
"""

from __future__ import annotations

import re
from itertools import combinations

from .types import NoteEvent

# Semitone reach of a finger pair. 1=thumb .. 5=pinky.
_FING_DIST: dict[tuple[int, int], float] = {
    (1, 1): 0, (1, 2): 2, (1, 3): 3.5, (1, 4): 5, (1, 5): 7,
    (2, 1): 2, (2, 2): 0, (2, 3): 2, (2, 4): 3.5, (2, 5): 5,
    (3, 1): 3.5, (3, 2): 2, (3, 3): 0, (3, 4): 2, (3, 5): 3.5,
    (4, 1): 5, (4, 2): 3.5, (4, 3): 2, (4, 4): 0, (4, 5): 2,
    (5, 1): 7, (5, 2): 5, (5, 3): 3.5, (5, 4): 2, (5, 5): 0,
}

_STRETCH: dict[tuple[int, int], float] = {
    (1, 1): 0.8, (1, 2): 1.15, (1, 3): 1.4, (1, 4): 1.45, (1, 5): 1.6,
    (2, 1): 1.15, (2, 2): 0.6, (2, 3): 0.9, (2, 4): 1.15, (2, 5): 1.3,
    (3, 1): 1.4, (3, 2): 0.9, (3, 3): 0.6, (3, 4): 0.9, (3, 5): 1.15,
    (4, 1): 1.45, (4, 2): 1.15, (4, 3): 0.9, (4, 4): 0.7, (4, 5): 0.7,
    (5, 1): 1.6, (5, 2): 1.3, (5, 3): 1.15, (5, 4): 0.8, (5, 5): 0.6,
}

_ASC_THUMB_STRETCH = {(2, 1): 0.95, (3, 1): 1.0, (4, 1): 0.95, (5, 1): 0.95}
_DESC_THUMB_STRETCH = {(1, 2): 1.0, (1, 3): 1.0, (1, 4): 0.9, (1, 5): 0.95}

# Cost of covering N semitones with ONE finger: a fixed lift-and-place plus a
# per-semitone distance term.  Index 0 is unused.
_FIXED_MOVE = 4.0
_MOVE_RAW: tuple[float, ...] = (
    0, 0, 0.5, 1.8, 3, 5, 7, 8, 8.9, 9.7, 10.5, 11, 11.4,
    11.8, 12.2, 12.5, 12.8, 13.1, 13.4, 13.7, 14, 14.3, 14.6, 14.9, 15.2,
)

_BLACK = frozenset({1, 3, 6, 8, 10})
MAX_FINGERS = 5
_MOVE_CUTOFF = 7.5
# Middle C: below it the right hand reaches awkwardly, above it the left.
_REGISTER_CENTRE = 60


def _move(semis: int) -> float:
    """Reposition cost: fixed lift plus distance."""
    n = abs(int(semis))
    if n == 0:
        return 0.0
    if n < len(_MOVE_RAW):
        return _MOVE_RAW[n] + _FIXED_MOVE
    return _MOVE_RAW[-1] + _FIXED_MOVE + (n - 24) / 5.0


def _thumb_cross(x: float) -> float:
    """The regression curve the reference uses for thumb passings."""
    return (0.0002185873295 * x ** 7 - 0.008611946279 * x ** 6
            + 0.1323250066 * x ** 5 - 1.002729677 * x ** 4
            + 3.884106308 * x ** 3 - 6.723075747 * x ** 2
            + 1.581196785 * x + 7.711241722)


def _in_position(x: float) -> float:
    """Cost of a reachable finger pair inside one position (x = slack)."""
    return (-0.0000006589793725 * x ** 10 - 0.000002336381414 * x ** 9
            + 0.00009925769823 * x ** 8 + 0.0001763353131 * x ** 7
            - 0.004660305277 * x ** 6 - 0.004290746384 * x ** 5
            + 0.06855725903 * x ** 4 + 0.03719817227 * x ** 3
            + 0.4554696705 * x ** 2 - 0.08305450359 * x + 0.3020594956)


def _colour(p1: int, p2: int, f1: int, f2: int) -> float:
    """Pinky or thumb landing on (or leaving) a black key is harder."""
    b1, b2 = (p1 % 12) in _BLACK, (p2 % 12) in _BLACK
    if not b1 and b2:
        if f2 in (1, 5) or f1 == f2:
            return 4.0
    if b1 and not b2:
        if f1 in (1, 5):
            return 4.0
        if f1 == f2:
            return -1.0
    return 0.0


def _pair_cost(p1: int, f1: int, p2: int, f2: int) -> float:
    """Cost of moving from pitch p1 on f1 to pitch p2 on f2."""
    if p1 == p2:
        return 0.0 if f1 == f2 else _FIXED_MOVE
    note_d = abs(p2 - p1)
    fing_d = _FING_DIST[(f1, f2)]
    ascending = p2 > p1

    # Same finger, new pitch: the hand itself must travel.
    if f1 == f2:
        return _move(note_d) + _colour(p1, p2, f1, f2)

    # Ascending onto the thumb (thumb-under) / descending off the thumb.
    if ascending and f2 == 1:
        x = (note_d + fing_d) / _ASC_THUMB_STRETCH.get((f1, f2), 1.0)
        if x > 10:
            return _move(int(round(note_d + fing_d)) + _FIXED_MOVE)
        return _thumb_cross(x) + (8.0 if (p2 % 12) in _BLACK and (p1 % 12) not in _BLACK else 0.0)
    if not ascending and f1 == 1:
        x = (note_d + fing_d) / _DESC_THUMB_STRETCH.get((f1, f2), 1.0)
        if x > 10:
            return _move(int(round(note_d + fing_d)) + _FIXED_MOVE)
        return _thumb_cross(x) + (8.0 if (p1 % 12) in _BLACK and (p2 % 12) not in _BLACK else 0.0)

    # Direction of fingers fights direction of music: hand must be replaced.
    if ascending and f2 < f1:
        return _move(int(round(note_d + fing_d))) + _colour(p1, p2, f1, f2)
    if not ascending and f2 > f1:
        return _move(int(round(note_d + fing_d))) + _colour(p1, p2, f1, f2)

    # Fingers and music agree: cheap while the gap is inside the hand.
    stretch = _STRETCH[(f1, f2)]
    x = abs(note_d - fing_d) / stretch if stretch else 0.0
    if x > _MOVE_CUTOFF:
        return _move(int(round(note_d - fing_d))) + _colour(p1, p2, f1, f2)
    cost = _in_position(x) + _colour(p1, p2, f1, f2)
    if x > 6.8:
        cost += (x - 6.8) * 3.0
    return cost


# The widest interval a pair of neighbouring fingers can hold, in semitones.
# Performer never rules a span out, it only prices it; a hard limit is still
# needed here because the search must not emit a grip a hand cannot make.  These
# are the intervals a player stretches to, not the comfortable ones: 1-5 takes
# an octave, 2-5 an octave, 1-4 a major seventh -- all ordinary in the literature.
_MAX_SPAN: dict[tuple[int, int], int] = {
    (1, 2): 4, (1, 3): 7, (1, 4): 9, (1, 5): 12,
    (2, 3): 4, (2, 4): 6, (2, 5): 9,
    (3, 4): 3, (3, 5): 7,
    (4, 5): 4,
}


def _span_ok(f1: int, f2: int, semis: int) -> bool:
    """Can these two fingers hold this many semitones apart?"""
    lo, hi = min(f1, f2), max(f1, f2)
    if lo == hi:
        return semis == 0
    return semis <= _MAX_SPAN.get((lo, hi), 0)


def _shapes(pitches: list[int]) -> list[tuple[int, ...]]:
    """Physically legal finger assignments for one strike."""
    n = len(pitches)
    if n == 0 or n > MAX_FINGERS:
        return []
    if n == 1:
        return [(f,) for f in range(1, 6)]
    out: list[tuple[int, ...]] = []
    for combo in combinations(range(1, 6), n):
        if not _span_ok(combo[0], combo[-1], pitches[-1] - pitches[0]):
            continue
        if all(_span_ok(combo[i], combo[i + 1], pitches[i + 1] - pitches[i])
               for i in range(n - 1)):
            out.append(combo)
    return out


def _strikes(events: list[NoteEvent], qtol: float = 0.02) -> list[tuple[float, list[int]]]:
    """Group simultaneous notes into strikes, in time order."""
    grouped: list[tuple[float, list[int]]] = []
    for e in sorted(events, key=lambda x: (x.start, x.pitch)):
        if grouped and abs(e.start - grouped[-1][0]) <= qtol:
            grouped[-1][1].append(e.pitch)
        else:
            grouped.append((e.start, [e.pitch]))
    for _, ps in grouped:
        ps.sort()
    return grouped


def _orient(pitches: list[int], hand: str) -> tuple[list[int], list[int]]:
    """Put a strike into model space: ascending pitch, thumb on the low note.

    A right hand plays in that space directly.  A left hand's thumb is on the
    *high* side, so its pitches are mirrored (negated) -- that makes an
    ascending left-hand scale look like a descending right-hand one, which is
    exactly the shape the cost model was tuned on.  Returns the pitches the
    model sees and the real pitch each of them came from.
    """
    if hand == "left":
        real = list(reversed(pitches))
        return [-p for p in real], real
    return list(pitches), list(pitches)


def assign(events: list[NoteEvent], hand: str = "right") -> dict[tuple[int, int], int]:
    """Viterbi over strikes: cheapest legal finger path through the piece."""
    strikes = _strikes(events)
    if not strikes:
        return {}

    model = [_orient(pitches, hand) for _, pitches in strikes]
    shapes: list[list[tuple[int, ...]]] = []
    for seen, _real in model:
        cand = _shapes(seen)
        if not cand:                       # unreachable strike: keep the run alive
            cand = [(1,) * len(seen)]
        shapes.append(cand)

    inf = float("inf")
    score: list[list[float]] = [[inf] * len(s) for s in shapes]
    back: list[list[int]] = [[-1] * len(s) for s in shapes]
    for k in range(len(shapes[0])):
        score[0][k] = 0.0

    for i in range(1, len(strikes)):
        prev_p = model[i - 1][0]
        cur_p = model[i][0]
        for k, cur in enumerate(shapes[i]):
            best, best_j = inf, -1
            for j, prev in enumerate(shapes[i - 1]):
                if score[i - 1][j] == inf:
                    continue
                total = score[i - 1][j] + _transition(prev_p, prev, cur_p, cur, hand)
                if total < best:
                    best, best_j = total, j
            score[i][k] = best
            back[i][k] = best_j

    last = min(range(len(shapes[-1])), key=lambda k: score[-1][k])
    path = [last]
    for i in range(len(strikes) - 1, 0, -1):
        nxt = back[i][path[-1]]
        path.append(nxt if nxt >= 0 else 0)
    path.reverse()

    out: dict[tuple[int, int], int] = {}
    for i, ((start, _pitches), k) in enumerate(zip(strikes, path)):
        for pitch, finger in zip(model[i][1], shapes[i][k]):
            out[(round(start, 2), pitch)] = finger
    return out


def _transition(prev_p: list[int], prev_f: tuple[int, ...],
                cur_p: list[int], cur_f: tuple[int, ...],
                hand: str = "right") -> float:
    """Cost of one hand moving from one strike to the next, in model space."""
    total = 0.0
    # Walk the current notes against the previous ones by rank: the low note of
    # one strike leads to the low note of the next.
    pairs = 0
    for j, cp in enumerate(cur_p):
        for i, pp in enumerate(prev_p):
            pf = prev_f[min(i, len(prev_f) - 1)]
            total += _pair_cost(pp, pf, cp, cur_f[j])
            pairs += 1
    if pairs:
        total /= pairs
    # A chord held inside one position is cheap; make the wide ones pay.
    if len(cur_p) > 1:
        outer = _FING_DIST[(cur_f[0], cur_f[-1])]
        gap = (cur_p[-1] - cur_p[0]) - outer
        if gap > 0:
            total += gap * 2.0
    # Playing low on the keyboard is harder for the right hand and easy for the
    # left, and the other way round at the top. Without this a tune written at
    # the centre of the keyboard fingers the same either way.
    # Pitches reach here in model space, where the left hand's are mirrored.
    if hand == "right":
        total += max(0, _REGISTER_CENTRE - min(cur_p))
    else:
        total += max(0, -min(cur_p) - _REGISTER_CENTRE)
    return total


def mark_key_points(events: list[NoteEvent], fingers: dict[tuple[int, int], int],
                    hand: str = "right") -> dict[tuple[int, int], int]:
    """Keep only the finger numbers a player has to read.

    The digits are the positions the hand cannot infer from the shape of the
    music: the first note of a phrase, the note where the thumb moves in or
    out, a leap past a fifth, a chord whose grip differs from the one before
    it, and a strike that needs a hand wider than an octave.  A stepwise run
    inside one position is dropped -- its fingers walk in order, so printing
    every one of them is noise that makes the score harder, not easier.
    """
    strikes = _strikes(events)
    end_at: dict[float, float] = {}
    for e in events:
        k = round(e.start, 2)
        end_at[k] = max(end_at.get(k, 0.0), e.start + max(e.dur, 0.0))

    kept: dict[tuple[int, int], int] = {}
    prev_pat: tuple[int, ...] | None = None
    prev_pitch: int | None = None
    prev_end = 0.0
    for idx, (start, pitches) in enumerate(strikes):
        key = round(start, 2)
        pat = tuple(fingers[(key, p)] for p in pitches if (key, p) in fingers)
        if len(pat) != len(pitches):
            continue
        span = pitches[-1] - pitches[0]
        keep = (
            idx == 0
            or prev_pat is None
            or start - prev_end > 0.5                     # new phrase after a rest
            or span >= 9                                  # wider than an octave
            or (prev_pitch is not None and abs(pitches[0] - prev_pitch) > 7)
            or _crosses(prev_pat, pat, pitches[0], prev_pitch)
            or _regrips(prev_pat, pat)
        )
        if keep:
            for p, f in zip(pitches, pat):
                kept[(key, p)] = f
        prev_pat, prev_pitch = pat, pitches[0]
        prev_end = end_at.get(key, start)
    return kept


def _crosses(prev: tuple[int, ...], cur: tuple[int, ...], pitch: int,
             prev_pitch: int | None) -> bool:
    """True where the hand runs out of fingers and the thumb goes under.

    A crossing is the only place a stepwise run stops being self-evident: the
    pitch keeps going the same way while the finger numbers turn around.  The
    note *after* the crossing is ordinary again, so it is not marked.
    """
    if prev_pitch is None or not prev or not cur:
        return False
    if 1 not in (prev[0], cur[0]):
        return False
    step = pitch - prev_pitch
    turn = cur[0] - prev[0]
    return step * turn < 0


def _regrips(prev: tuple[int, ...], cur: tuple[int, ...]) -> bool:
    """True where the hand has to take a different shape.

    Compared as offsets from the lowest finger, not as absolute numbers: the
    same grip moved along the keyboard (1-2-3, then 2-3-4) is one shape the
    player already knows, while 1-2-3 becoming 1-3-5 is a new grip to read.
    """
    if not prev or not cur or len(prev) != len(cur):
        return False
    return tuple(f - prev[0] for f in prev) != tuple(f - cur[0] for f in cur)


def annotate(arr, mode: str = "key") -> dict:
    """Attach finger numbers to an arrangement in place. Returns a report."""
    report = {"mode": mode, "right": 0, "left": 0,
              "right_total": len(arr.right), "left_total": len(arr.left)}
    for hand, events in (("right", arr.right), ("left", arr.left)):
        if not events:
            continue
        if mode == "none":
            for e in events:
                e.finger = 0
            continue
        fingers = assign(events, hand)
        if mode == "key":
            fingers = mark_key_points(events, fingers, hand)
        marked = 0
        for e in events:
            f = fingers.get((round(e.start, 2), e.pitch), 0)
            e.finger = f
            marked += 1 if f else 0
        report[hand] = marked
    return report


def verify_export(arr, musicxml_text: str, mode: str = "key") -> dict:
    """Compare the finger numbers in the staff file with the arrangement.

    Re-runnable evidence: walks the strikes in time order, reads the number
    carried by each note, and checks that the staff file agrees with it and
    that every chord is monotone in its hand's direction.
    """
    idx: dict[tuple[float, int], int] = {}
    for events in (arr.right, arr.left):
        for e in events:
            idx[(round(e.start, 2), e.pitch)] = getattr(e, "finger", 0)

    marked_strikes = 0
    non_monotone = 0
    carried = 0
    for hand, events in (("right", arr.right), ("left", arr.left)):
        for start, pitches in _strikes(events):
            start = round(start, 2)
            fingers = [idx.get((start, p), 0) for p in pitches]
            fingers = [f for f in fingers if f]
            if not fingers:
                continue
            marked_strikes += 1
            carried += len(fingers)
            if len(fingers) < 2:
                continue
            ordered = sorted(fingers, reverse=(hand == "left"))
            if fingers != ordered:
                non_monotone += 1

    in_file = len(re.findall(r"<fingering[^>]*>(\d)</fingering>", musicxml_text))
    return {
        "mode": mode,
        "in_arrangement": carried,
        "in_staff_file": in_file,
        "marked_strikes": marked_strikes,
        "non_monotone_strikes": non_monotone,
        "agrees": in_file == carried and non_monotone == 0,
    }
