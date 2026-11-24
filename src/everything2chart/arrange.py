"""The core of this project: turn separated stems into a two-hand piano
arrangement that a human can actually play at the chosen difficulty.

Rules (see PLAN.md §8):
  * right hand = melodic line, re-voiced into a playable register
  * left hand  = bass root + chord tones, strictly below the right hand
  * everything is laid on a bar grid with a *density budget* per preset, so
    "easy" never produces an unplayable wall of notes
  * easy keeps melody contour + bass skeleton + harmonic function, which is
    what makes it still sound like the source track
"""

from __future__ import annotations

import logging

import numpy as np

from .config import load_presets
from .notation import tail_beats
from .types import Arrangement, ChordEvent, NoteEvent, SongInfo

log = logging.getLogger(__name__)

PC_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
# the other way to spell each pitch class, so a key name is never refused for
# being written `C#m` instead of `Dbm`, or `Gb` instead of `F#`
PC_NAMES_ALT = ["B#", "C#", "D", "D#", "Fb", "E#", "Gb", "G", "G#", "A", "A#", "Cb"]


# --------------------------------------------------------------------------- helpers

def fold_into(pitch: int, lo: int, hi: int) -> int:
    """Move a pitch by octaves into [lo, hi]; clamp if the range is narrower."""
    p = int(pitch)
    while p < lo:
        p += 12
    while p > hi:
        p -= 12
    return int(np.clip(p, lo, hi))


def collapse_octaves(notes: list[NoteEvent]) -> list[NoteEvent]:
    """Detectors confuse octaves far more than wrong notes. When a jump is within
    one semitone of an octave, move the note into the previous note's octave."""
    out: list[NoteEvent] = []
    for n in notes:
        if out:
            step = n.pitch - out[-1].pitch
            for k in (-2, -1, 1, 2):
                if abs(step - 12 * k) <= 1:
                    n.pitch -= 12 * k
                    break
        out.append(n)
    return out


OCTAVE_STICKINESS = 12.0   # cost of changing octave between two adjacent notes
# The window is a preference, not a wall: a note already inside [lo, hi] keeps its
# written octave unless leaving it wins by more than this. Without the credit the
# fold drifts down whenever the melody arrives high -- on the fast bench clips it
# landed a median ninth below the truth (67 vs 79) even though every pitch class
# was right, because `stickiness` alone makes a low note look as cheap as a high
# one. Measured on four clips x the melody floor: arranged right-hand note-F1
# 0.436/0.561/0.421/0.208 -> 0.792/0.749/0.779/0.519, identical to the pitch-class
# ceiling, and every value from 0.5 to 6 gives the same table, so this is a
# tie-break rather than a knob.
OCTAVE_ANCHOR = 1.0


def fold_line(pitches: list[int], lo: int, hi: int,
              stickiness: float = OCTAVE_STICKINESS) -> list[int]:
    """Fold a melody into [lo, hi] the way a player would rewrite it by ear.

    Picking the octave nearest the previous note is not enough: two notes a fifth
    apart sit closer in the *other* octave, so the line keeps hopping register even
    though every interval is right.  What a player actually does is stay put and
    only cross an octave when the alternative is worse by a lot -- so charge a
    fixed price for changing octave.  The result keeps the contour and stops
    wandering.
    """
    out: list[int] = []
    prev: int | None = None
    prev_k = 0
    for pitch in pitches:
        p = int(pitch)
        # every octave placement that lands inside the window
        cands = [(k, p + 12 * k) for k in range(-5, 6)
                 if lo <= p + 12 * k <= hi]
        if not cands:
            # window narrower than an octave: absolute fold, nothing to choose
            q = p
            while q < lo:
                q += 12
            while q > hi:
                q -= 12
            out.append(int(np.clip(q, lo, hi)))
            prev = out[-1]
            continue
        if prev is None:
            best_k, best_p = min(cands, key=lambda kp: (abs(kp[1] - p), abs(kp[0])))
        else:
            best_k, best_p = min(
                cands,
                key=lambda kp: abs(kp[1] - prev)
                + (stickiness if kp[0] != prev_k else 0.0)
                - (OCTAVE_ANCHOR if kp[0] == 0 else 0.0))
        out.append(int(best_p))
        prev, prev_k = int(best_p), best_k
    return out


# How many keys one hand can hold at once.  These are the limits the printed
# score has to keep in every tier; a denser tier widens its range and its
# rhythm, it does not grow extra fingers.
MAX_KEYS_RIGHT = 2
MAX_KEYS_LEFT = 6


def _thin_overlaps(notes: list[NoteEvent], max_sim: int,
                   prefer_high: bool = True) -> list[NoteEvent]:
    if max_sim <= 0:
        return []
    ordered = sorted(notes, key=lambda n: (n.start, -n.pitch if prefer_high else n.pitch))
    kept: list[NoteEvent] = []
    for n in ordered:
        active = [k for k in kept if k.end > n.start + 1e-6 and k.start < n.end - 1e-6]
        if len(active) < max_sim:
            kept.append(n)
            continue
        if not prefer_high:
            continue
        lowest_idx = min(range(len(active)), key=lambda i: active[i].pitch)
        if n.pitch > active[lowest_idx].pitch:
            kept.remove(active[lowest_idx])
            kept.append(n)
    return sorted(kept, key=lambda n: n.start)


def _clip_to_beats(notes: list[NoteEvent], spq: float) -> list[NoteEvent]:
    """Cut every note at the first bar-beat that strikes again.

    简谱 writes one column per beat and holds a sounding note with ``-``; a beat
    taken by a ``-`` cannot also carry a new strike, so a note that is still
    ringing when the beat after it starts striking has to give up that beat.
    Without this the score would claim a length the notation cannot play, and
    the printed sheet would disagree with the staff and the MIDI -- the one
    thing this project is not allowed to do.
    """
    beat_of = [int(n.start / spq + 1e-9) for n in notes]
    striking = sorted(set(beat_of))
    nxt = {b: (striking[i + 1] if i + 1 < len(striking) else None)
           for i, b in enumerate(striking)}
    # A later strike inside the same beat needs the same cut: one written beat
    # can only say one thing (a column, possibly a chord), so a note that is
    # still ringing when a second key of the same hand goes down has no way to
    # be printed -- and the page would contradict the MIDI.
    onsets = sorted(n.start for n in notes)
    grid = spq / 4
    out = []
    for n, b in zip(notes, beat_of):
        end = n.end
        c = nxt[b]
        later = next((t for t in onsets if t > n.start + 1e-6), None)
        beat_end = c * spq if c is not None and c * spq > n.start else None
        cut = min([t for t in (beat_end, later) if t is not None], default=None)
        if cut is not None:
            end = min(end, cut)
            # a note in the last slot of its beat has almost no room left
            if end - n.start < grid:
                end = n.start + grid
        out.append(NoteEvent(n.pitch, n.start, end - n.start, n.velocity,
                             n.stem, n.confidence))
    return out


def _quantise_hand(notes: list[NoteEvent], spq: float, sub: int = 4) -> list[NoteEvent]:
    """Snap onsets and lengths to the sixteenth grid the score is written on."""
    grid = spq / sub
    out = []
    for n in notes:
        start = round(n.start / grid) * grid
        dur = max(grid, round(n.dur / grid) * grid)
        out.append(NoteEvent(n.pitch, start, dur, n.velocity, n.stem, n.confidence))
    return out


def _share_beat_length(notes: list[NoteEvent], spq: float) -> list[NoteEvent]:
    """Make the notes of one written beat share the shortest of their lengths."""
    shortest: dict[int, float] = {}
    counts: dict[int, int] = {}
    for n in notes:
        key = int(n.start / spq + 1e-9)
        counts[key] = counts.get(key, 0) + 1
        shortest[key] = min(shortest.get(key, float("inf")), n.dur)
    out = []
    for n in notes:
        key = int(n.start / spq + 1e-9)
        dur = shortest[key] if counts[key] > 1 else n.dur
        out.append(NoteEvent(n.pitch, n.start, dur, n.velocity, n.stem,
                             n.confidence))
    return out


def _cut_at_next_strike(notes: list[NoteEvent], spq: float) -> list[NoteEvent]:
    """A held key cannot survive a later strike of the same hand in 简谱.

    ``-`` means "everything this hand is sounding keeps sounding", so the moment
    the same hand presses another key the held note has no way to be written
    again and the page silently shortens it.  Cutting the note where the next
    strike lands is what a player does anyway -- the new key press lifts the old
    one -- and it makes the printed length the sounding length.
    """
    onsets = sorted(n.start for n in notes)
    out: list[NoteEvent] = []
    for n in notes:
        end = n.end
        nxt = next((t for t in onsets if t > n.start + 1e-6), None)
        if nxt is None or nxt >= end:
            out.append(n)
            continue
        # Cut at the strike, then strike it again there: a held bass note is
        # normally re-struck rather than damped, and repeating the key keeps the
        # harmony sounding instead of leaving the bar thinner than the recording.
        out.append(NoteEvent(n.pitch, n.start, max(nxt - n.start, spq / 4),
                             n.velocity, n.stem, n.confidence))
        t = nxt
        while t < end - 1e-6:
            # A re-strike inside the beat that already prints this note has no
            # column of its own: the beat would then say two things at once.
            if int(t / spq + 1e-9) == int(n.start / spq + 1e-9):
                t = next((u for u in onsets if u > t + 1e-6), None) or end
                continue
            nxt2 = next((u for u in onsets if u > t + 1e-6), None)
            seg = (min(nxt2, end) if nxt2 is not None else end) - t
            out.append(NoteEvent(n.pitch, t, max(seg, spq / 4),
                                 max(30, int(n.velocity * 0.85)), n.stem,
                                 n.confidence))
            if nxt2 is None:
                break
            t = nxt2
    return out


def _one_length_per_strike(notes: list[NoteEvent], spq: float) -> list[NoteEvent]:
    """One written beat states one length, so every key struck on that beat has
    to take it.

    ``_share_beat_length`` does this for whole-beat groups, but a 简谱 column is
    read as "these notes last this long" and a beat rarely holds all its notes
    for the same time.  Shortening the long ones to the shortest under the
    strike keeps the page honest -- it can be played exactly as printed -- and
    the next strike comes sooner than the sound did, which is what a player
    correcting early does, not a wrong note.
    """
    groups: dict[int, list[NoteEvent]] = {}
    for n in notes:
        groups.setdefault(int(n.start / spq + 1e-9), []).append(n)
    out: list[NoteEvent] = []
    for beat in sorted(groups):
        group = groups[beat]
        if len(group) > 1:
            shortest = min(n.dur for n in group)
            group = [NoteEvent(n.pitch, n.start, shortest, n.velocity, n.stem,
                               n.confidence) for n in group]
        out.extend(group)
    return out


def _collapse_beat_onsets(notes: list[NoteEvent], spq: float,
                          sub: int = 4) -> list[NoteEvent]:
    """One strike per written slot per hand, at the slot's first onset.

    简谱 prints one column per slot -- ``sub`` per beat, sixteenths at the
    default 4 -- and every column shares a single duration, so a second onset
    inside the same slot would have to borrow the first one's length.  Pulling
    the different pitches of a slot onto that slot's earliest onset keeps one
    column and turns them into the chord the hand would take together.  A
    repeated pitch is left alone: that is a re-struck melody note, not a chord
    tone, and dropping it would erase part of the tune.

    The slot, not the beat, is the window: a beat-wide collapse also welds
    eighths onto each other and turns a walking eighth-note line into quarter
    notes, which is a rhythm the page can print but the recording never played.
    """
    step = spq / max(1, sub)
    first: dict[int, float] = {}
    for n in notes:
        b = int(n.start / step + 1e-6)
        if b not in first or n.start < first[b]:
            first[b] = n.start
    seen: set[tuple[int, int]] = set()
    out = []
    for n in notes:
        b = int(n.start / step + 1e-6)
        key = (b, n.pitch)
        t = first[b]
        if key in seen or abs(t - n.start) < 1e-9:
            out.append(n)
        else:
            out.append(NoteEvent(n.pitch, t, max(n.dur - (n.start - t), spq / 4),
                                 n.velocity, n.stem, n.confidence))
        seen.add(key)
    return sorted(out, key=lambda n: (n.start, n.pitch))


def _end_before_strike_beat(notes: list[NoteEvent], spq: float) -> list[NoteEvent]:
    """End a held note where the next beat presses a key.

    简谱 gives one cell per beat, and a cell that presses a key cannot also say
    "-", so a note that rings past such a beat loses the rest of its length on
    the page: the reader sees a shorter note than the arrangement holds.  Ending
    it at that beat keeps the printed length and the sounding length the same,
    which is also what the hand does -- the new key press releases the old one.
    """
    out: list[NoteEvent] = []
    for n in notes:
        end = n.end
        nxt = next((t for t in sorted(x.start for x in notes)
                    if t > n.start + 1e-6), None)
        if nxt is not None and nxt <= end + 1e-9:
            beat = int(nxt / spq + 1e-9) * spq
            if n.start + spq / 8 < beat < end:
                end = beat
        out.append(n if abs(end - n.end) < 1e-9 else
                   NoteEvent(n.pitch, n.start, end - n.start,
                             n.velocity, n.stem, n.confidence))
    return sorted(out, key=lambda n: (n.start, n.pitch))


def _dedupe_beats(notes: list[NoteEvent], spq: float,
                  sub: int = 4) -> list[NoteEvent]:
    """One key, one note per written slot.

    The page writes a pitch once per slot, so two notes of the same pitch in
    the same slot cannot both be printed: the reader takes them for one note
    and the round trip loses a length.  The slot is the finest value either
    notation can engrave (``sub`` per beat, sixteenths at the default 4), not
    the whole beat -- a beat-wide window also swallows the eighth-note repeat
    of one key, and on a melody of repeated notes that is the rhythm itself.
    """
    step = spq / max(1, sub)
    best: dict[tuple[int, int], NoteEvent] = {}
    order: list[tuple[int, int]] = []
    for n in notes:
        key = (int(n.start / step + 1e-6), n.pitch)
        cur = best.get(key)
        if cur is None:
            best[key] = n
            order.append(key)
        elif n.dur > cur.dur:
            best[key] = n
    kept = {id(best[k]) for k in order}
    out = [n for n in notes if id(n) in kept]
    out.sort(key=lambda n: (n.start, n.pitch))
    return out


def _dedupe_unisons(notes: list[NoteEvent]) -> list[NoteEvent]:
    """Drop a second strike of the same pitch at the same instant.

    It sounds no different from the first one, and a notation that can only put
    one length on a strike would otherwise have to guess which one was meant.
    The longer of the two is kept.
    """
    best: dict[tuple[int, int], NoteEvent] = {}
    order: list[tuple[int, int]] = []
    for n in notes:
        key = (int(round(n.start * 1000)), n.pitch)
        if key not in best:
            best[key] = n
            order.append(key)
        elif n.dur > best[key].dur:
            best[key] = n
    return [best[k] for k in order]


def _avoid_collisions(lh: list[NoteEvent], rh: list[NoteEvent],
                      min_gap: int = 2) -> list[NoteEvent]:
    if not rh:
        return lh
    out: list[NoteEvent] = []
    for n in lh:
        limit = 127
        for r in rh:
            if r.end > n.start and r.start < n.end:
                limit = min(limit, r.pitch - min_gap)
        if n.pitch >= limit:
            if limit < 21:
                continue
            n.pitch = limit
        if any(o.pitch == n.pitch and abs(o.start - n.start) < 1e-3 for o in out):
            continue
        out.append(n)
    return out


def _fold_into(near: list[NoteEvent], n: NoteEvent, rng: tuple[int, int],
               spq: float) -> NoteEvent | None:
    """Move a note by octaves until it sits under the left hand, keeping its class.

    Returns None when every octave of the class already has a key of its own in
    that written beat: the score cannot print two notes of one beat on one key,
    and printing it twice would only be read back as one note of the other
    length, i.e. a page that disagrees with itself.
    """
    lo, hi = rng
    lo = min(lo, min((m.pitch for m in near), default=lo))
    hi = max(hi, max((m.pitch for m in near), default=hi))
    beat = int(n.start / spq + 1e-6)
    taken = {(int(m.start / spq + 1e-6), m.pitch) for m in near}
    base = n.pitch
    while base < lo:
        base += 12
    while base > hi:
        base -= 12
    for shift in (0, 12, -12, 24, -24):
        pitch = base + shift
        if not lo <= pitch <= hi:
            continue
        if (beat, pitch) in taken:
            continue
        return NoteEvent(pitch=pitch, start=n.start, dur=n.dur,
                         velocity=n.velocity, stem="arranger",
                         confidence=n.confidence)
    return None


def _split_overlaps(notes: list[NoteEvent], max_sim: int, prefer_high: bool = True,
                    ) -> tuple[list[NoteEvent], list[NoteEvent]]:
    """Thin a hand to `max_sim` keys and give back what it had to let go.

    A voice the right hand cannot press is still a note the recording plays.
    Handing it to the left keeps the harmony the listener hears while the page
    keeps both limits; only the register moves.
    """
    kept = _thin_overlaps(notes, max_sim, prefer_high=prefer_high)
    keep = {id(n) for n in kept}
    return kept, [n for n in notes if id(n) not in keep]


def _thin_chords(lh: list[NoteEvent], cap: int) -> list[NoteEvent]:
    """No more than `cap` left-hand notes sound at the very same instant.

    A dense bass passage makes the arranger stack a chord on every eighth; a few
    overlapping eighths under a triad is eight-odd keys at one moment, which no
    hand plays and which smears the bass into a rumble.  Only the notes that
    start at this instant are candidates -- a note already sounding is not the
    offender, the new stack on top of it is.  The lowest note and the highest are
    what the ear uses to place a chord, so shed from the middle, and shed the
    upper note of any semitone pair first.
    """
    order = sorted(lh, key=lambda n: (round(n.start, 4), n.pitch))
    drop: set[int] = set()
    starts = sorted({round(n.start, 4) for n in order})
    for t in starts:
        fresh = [n for n in order if round(n.start, 4) == t and id(n) not in drop]
        # `t` is the rounded start; comparing raw starts against it would miss
        # the very notes that start here and make the budget below grow by the
        # size of the chord instead of shrinking, i.e. the cap would never bite.
        sounding = [n for n in order if n.start - 1e-6 <= t < n.start + n.dur
                    and id(n) not in drop]
        held = [n for n in sounding if round(n.start, 4) != t]
        room = cap - len(held)
        if room >= len(fresh):
            continue
        if room <= 0:
            drop.update(id(n) for n in fresh)
            continue
        fresh.sort(key=lambda n: n.pitch)
        keep = [fresh[0], fresh[-1]]
        for n in fresh[1:-1]:
            if any(abs(n.pitch - k.pitch) == 1 for k in keep):
                drop.add(id(n))
        rest = [n for n in fresh[1:-1] if id(n) not in drop]
        rest.sort(key=lambda n: (abs(n.pitch - fresh[0].pitch) + abs(n.pitch - fresh[-1].pitch),
                                 n.pitch))
        for n in rest:
            if len(keep) < room:
                keep.append(n)
            else:
                drop.add(id(n))
        if len(keep) > room:
            for n in sorted(keep, key=lambda n: n.pitch)[1:-1]:
                if len(keep) <= room:
                    break
                drop.add(id(n))
                keep.remove(n)
    return [n for n in lh if id(n) not in drop]


def _legato(rh: list[NoteEvent], max_gap: float = 0.22, max_leap: int = 7) -> list[NoteEvent]:
    """Bridge tiny melody dropouts so the right hand reads as a line, not dots."""
    for a, b in zip(rh, rh[1:]):
        gap = b.start - a.end
        if 0 < gap <= max_gap and abs(b.pitch - a.pitch) <= max_leap:
            a.dur += gap
    return rh


# --------------------------------------------------------------------------- harmony

def harmonic_timeline(chords: list[ChordEvent], bass: list[NoteEvent],
                      info: SongInfo) -> list[tuple[float, float, int]]:
    """(start, end, root_pc) on a half-bar grid, root taken from the bass line.

    ``estimate_chords`` flips almost every beat, which is far denser than the
    harmony the arrangement should be coloured by, and its roots only match the
    real bass line about 40% of the time.  Both hands want one stable root per
    half bar, so the bass stem - weighting each note by how long it sounds - is
    the witness here; the chord estimate is only the fallback.
    """
    def chord_at(t: float) -> int | None:
        for c in chords:
            if c.start <= t < c.end:
                return c.root
        return chords[-1].root if chords else None

    def bass_roots(t0: float, t1: float) -> int | None:
        weight: dict[int, float] = {}
        for b in bass:
            overlap = min(b.end, t1) - max(b.start, t0)
            if overlap <= 0:
                continue
            pc = b.pitch % 12
            weight[pc] = weight.get(pc, 0.0) + overlap * (b.dur ** 0.5)
        if not weight:
            return None
        return max(weight, key=weight.get)

    end_all = max([c.end for c in chords] + [b.end for b in bass] + [0.0])
    step = 60.0 / max(info.bpm, 1.0) * info.beats_per_bar / 2.0
    out: list[tuple[float, float, int]] = []
    t = 0.0
    prev: int | None = None
    while t < end_all - 1e-6 and step > 0:
        t1 = min(t + step, end_all)
        root = bass_roots(t, t1)
        if root is None:
            root = chord_at(t)
        if root is None:
            root = prev if prev is not None else 0
        out.append((t, t1, root))
        prev = root
        t = t1

    # a single segment disagreeing with both neighbours is noise, not a real change
    for i in range(1, len(out) - 1):
        if out[i][2] != out[i - 1][2] and out[i - 1][2] == out[i + 1][2]:
            out[i] = (out[i][0], out[i][1], out[i - 1][2])
    return out


def _voice(root_pc: int, voice: int, lo: int, hi: int) -> int:
    """Root pitch class placed in the register at/above lo, then the interval added."""
    base = root_pc
    while base < lo:
        base += 12
    p = base + voice
    while p > hi:
        p -= 12
    return int(np.clip(p, lo, hi))


def bass_skeleton(bass: list[NoteEvent], info: SongInfo, grid: float,
                  lo: int, hi: int, transpose: int = 0) -> list[tuple[float, float, int]]:
    """Quantise the real bass line onto `grid`; one note per slot, lowest wins.

    This is what keeps an easy arrangement recognisable: the left hand is not a
    generic block-chord pattern, it is the actual bass of the track thinned down.
    """
    if not bass:
        return []
    slots: dict[int, NoteEvent] = {}
    for n in bass:
        k = int(round(n.start / grid))
        cur = slots.get(k)
        if cur is None or (n.dur > cur.dur * 1.5) or (abs(n.dur - cur.dur) < 1e-6 and n.pitch < cur.pitch):
            slots[k] = n
    keys = sorted(slots)
    out: list[tuple[float, float, int]] = []
    for i, k in enumerate(keys):
        n = slots[k]
        start = k * grid
        nxt = keys[i + 1] * grid if i + 1 < len(keys) else min(info.duration, start + grid * 2)
        dur = max(nxt - start, grid * 0.9)
        if dur > grid * 4:                       # long silences: stop the note early
            dur = grid * 4
        # The page is written `transpose` semitones from the recording, so the
        # bass has to move with it: without this the bass stayed in the source
        # key while `harmony_at` returned the notated pc, and the `t > pitch`
        # filter below dropped a number of chord tones that depended on the tier.
        out.append((start, min(dur, info.duration - start),
                    fold_into(n.pitch + transpose, lo, hi)))
    return out


def _audio_pcs(y, sr: int, lo: float, hi: float, n_pcs: int = 3) -> list[int]:
    """The strongest pitch classes of the source itself over [lo, hi).

    Used only in bars the transcriber produced nothing for: the arranger's chord
    estimate is a smooth harmonic reading and can name a different chord than the
    one the recording actually holds, while the recording's own spectrum cannot.
    """
    import numpy as np
    try:
        import librosa
    except Exception:      # pragma: no cover - librosa is a hard dependency
        return []
    a, z = int(lo * sr), int(hi * sr)
    if z - a < sr // 8:
        return []
    seg = np.asarray(y)[a:z]
    if seg.ndim > 1:
        seg = seg.mean(axis=1)
    seg = seg.astype(np.float32)
    if not np.isfinite(seg).all() or float(np.abs(seg).max()) < 1e-4:
        return []
    c = librosa.feature.chroma_cqt(y=seg, sr=sr).mean(axis=1)
    if not np.isfinite(c).any() or float(c.sum()) <= 0:
        return []
    order = np.argsort(c)[::-1]
    top = float(c[order[0]]) or 1.0
    return [int(i) for i in order if c[i] >= 0.45 * top][:n_pcs]


def _score_bar_count(info: SongInfo, spq: float, onsets=()) -> int:
    """How many bars the score will actually have.

    The arranger budgets its fills, accents and missing chord tones per bar,
    and it used to count them from the file duration: ``duration / bar + 1``.
    On an eight-bar take that is nine, so the last bar of every budget went to
    a ninth bar that the renderer then clipped away, and the accents spent on
    it never reached the bar the listener hears.  Both sides now ask
    `notation.tail_beats`, so a bar the arranger writes is a bar that prints.
    """
    bpb = info.beats_per_bar
    beats = tail_beats(onsets, info.duration, bpb, spq)
    return max(1, int(np.ceil(beats / bpb)))


def _build_left_hand(harmony: list[tuple[float, float, int]], preset: dict,
                     info: SongInfo, rh_min: int, pcs: set[int] | None = None,
                     transpose: int = 0,
                     bass: list[NoteEvent] | None = None,
                     bars: int | None = None) -> list[NoteEvent]:
    """Skeleton + chord tones for the left hand.

    Pitch-class spaces, which is where the eight bugs in this function's history
    all came from:

    * ``harmony``, ``bass``, ``bass_skeleton()`` and ``_anchor_bass_roots()``
      speak in the **source** key.
    * ``pcs`` (the set allowed by ``snap_to_key``), ``NoteEvent.pitch`` and
      everything the page prints speak in the **notated** key.
    * ``transpose`` -- which is ``shift`` at the call site -- is the only bridge.
      Add it when going source -> notated; subtract it when reading a notated
      value back. Every pitch this function returns is already notated.
    """
    lh = preset["left_hand"]
    lo, hi = lh["range"]
    hi = min(hi, rh_min - 2)
    voices: list[int] = lh["voices"]
    budget = lh["events_per_bar"]
    spq = 60.0 / info.bpm if info.bpm else 0.5
    bar = spq * info.beats_per_bar
    total_bars = bars or int(info.duration / bar) + 1

    def harmony_at(t: float) -> int:
        for s_, e_, pc in harmony:
            if s_ <= t < e_:
                return (pc + transpose) % 12
        return harmony[-1][2] if harmony else 0

    # how fine the left hand runs, and how many chord tones ride on top of the bass
    if budget <= 3:
        grid, extra, max_notes = bar / 2, [7], 3
    elif budget <= 7:
        grid, extra, max_notes = spq, [7], 3
    else:
        grid, extra, max_notes = spq / 2, [3, 4, 7], 4

    skel = bass_skeleton(bass or [], info, grid, lo, hi, transpose) if bass else []

    out: list[NoteEvent] = []
    if skel:
        for start, dur, pitch in skel:
            out.append(NoteEvent(pitch=pitch, start=start, dur=dur, velocity=72,
                                 stem="arranger"))
            # Chord tones are the harmony root plus the preset's intervals, placed
            # relative to the bass rather than to the hand's register: built as
            # absolute pitches they got clipped against `hi` by an amount that
            # depended on where the tier's key put the bass, so the three tiers
            # ended up with different left-hand note counts.
            # Each slot takes a fixed number of chord tones.  They are built on
            # the harmony root, not on the bass note: the bass may be a chromatic
            # passing note and building intervals above it drags the whole chord
            # out of the key.  What used to make the count differ per tier was
            # filtering the tones against the bass pitch, so a tier whose key put
            # the bass higher lost more of them; the interval set now decides the
            # count and `fold_into` only picks the octave.
            room = max_notes - 1
            tones = sorted({_voice(harmony_at(start), v, lo, hi)
                            for v in set(v % 12 for v in voices + extra)})[:room]
            for k, t in enumerate(tones):
                out.append(NoteEvent(pitch=t, start=start, dur=dur * 0.92,
                                     velocity=60 - 4 * k, stem="arranger"))
            if grid <= spq / 2 and dur >= spq * 0.9:   # hard: octave doubles on the beat
                out.append(NoteEvent(pitch=fold_into(pitch - 12, lo, hi), start=start,
                                     dur=dur * 0.95, velocity=66, stem="arranger"))

    else:
        for b in range(total_bars):
            t0 = b * bar
            if t0 >= info.duration:
                break
            roots = [harmony_at(t0 + k * bar / 4) for k in range(4)]
            main = max(set(roots), key=roots.count)
            step = bar / (4 if budget <= 7 else 8)
            steps = int(round(bar / step))
            for k in range(steps):
                t = t0 + k * step
                if t >= info.duration:
                    break
                dur = min(step * 0.95, info.duration - t)
                for v_i, v in enumerate(voices[:max_notes]):
                    out.append(NoteEvent(pitch=_voice(main, v, lo, hi), start=t, dur=dur,
                                         velocity=72 if v_i == 0 else 60 - 4 * v_i,
                                         stem="arranger"))
    if bass:
        out = _anchor_bass_roots(out, bass, info, lo, hi, transpose, pcs,
                                 bars=bars)
    return out


# --------------------------------------------------------------------------- key handling

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
EASY_KEYS = ["C", "G", "F", "Am", "Em", "Dm"]          # 0 or 1 black key


def _tonic_pc(key: str) -> int:
    """Pitch class of a key name, in either enharmonic spelling.

    `PC_NAMES` carries one spelling per pitch class (a mix of sharps and flats),
    so a key written the other way -- `C#m`, `G#m`, `Gb`, `Cb` -- used to miss
    the lookup and raise.  Configuring any of those as `global.target_key`
    crashed the build.  Both tables are accepted here; an unknown name is still
    an error, never a silent `C`.
    """
    name = key[:-1] if key.endswith("m") and len(key) > 1 else key
    for i, spelled in enumerate(zip(PC_NAMES, PC_NAMES_ALT)):
        if name in spelled:
            return i
    raise ValueError(f"unknown key signature {key!r}")


def scale_pcs(key: str) -> set[int]:
    tonic = _tonic_pc(key)
    base = MINOR_SCALE if key.endswith("m") else MAJOR_SCALE
    return {(tonic + i) % 12 for i in base}


BLACK_PCS = {1, 3, 6, 8, 10}


def _shifted_name(key: str, semis: int) -> str:
    """The name of `key` after `semis` semitones, mode unchanged."""
    name = key[:-1] if key.endswith("m") else key
    tonic = (_tonic_pc(key) + semis) % 12
    for spelled in (PC_NAMES[tonic], PC_NAMES_ALT[tonic]):
        if spelled is not None:
            return spelled + ("m" if key.endswith("m") else "")
    return PC_NAMES[tonic] + ("m" if key.endswith("m") else "")


def ranked_plan(key: str, pitches, rank: int) -> tuple[int, str]:
    """Shift for tier `rank`, ordered by the black keys it puts under the hand.

    The tiers used to be ordered by *key signature* -- C major has none, so it
    looked like the easiest thing to read.  Measured against the recording that
    was backwards: a piece written around a key other than C carries accidentals,
    and naming it C turns every one of those into a black key.  On the two
    samples here the "no accidentals" tier landed 27 and 70 black-key notes
    against 2 and 0 for a key with a real signature, so the tier advertised as
    easiest was the hardest to play.  A signature is only a proxy; the number of
    black keys actually under the fingers is the thing being promised, so rank on
    that first and use the signature to break ties.
    """
    tonic = _tonic_pc(key)
    mode = MINOR_SCALE if key.endswith("m") else MAJOR_SCALE
    rows = []
    for semis in range(-6, 7):
        black = sum(1 for p in pitches if (p + semis) % 12 in BLACK_PCS)
        signature = len([p for p in mode if (p + tonic + semis) % 12 in BLACK_PCS])
        rows.append((black, signature, abs(semis), semis))
    rows.sort()
    # Distinct keys per tier: equal scores are broken by the smaller shift so the
    # three tiers never collapse onto the same notation.
    seen = []
    for row in rows:
        if all(row[3] != taken for taken in seen):
            seen.append(row[3])
        if len(seen) > rank:
            break
    semis = seen[min(rank, len(seen) - 1)]
    return semis, _shifted_name(key, semis)


def _shift_to(key: str, target: str) -> int:
    """Smallest signed semitone shift that writes `key` as `target`."""
    shift = (_tonic_pc(target) - _tonic_pc(key)) % 12
    return shift if shift <= 6 else shift - 12


def transpose_plan(key: str, allowed_black: int,
                   target: str | None = None, pitches=None,
                   rank: int | None = None) -> tuple[int, str]:
    """Semitones to shift so the score is notated in `target`.

    A tier is allowed to differ from another in exactly two things: how complex
    the key signature is, and how many black keys that signature puts under the
    hand -- the second is not an independent choice, it follows from the first.
    So a tier names its key outright and the shift is whatever writes the
    recording in that key; the notes, the rhythm and the density stay the same
    across tiers, which is what makes one transcription three readable versions
    of the same piece instead of three different reductions.
    """
    if rank is not None and pitches:
        return ranked_plan(key, pitches, rank)
    if target:
        return _shift_to(key, target), target
    current = len(scale_pcs(key) & {1, 3, 6, 8, 10})
    if current <= allowed_black:
        return 0, key
    best = None
    for target in EASY_KEYS:
        black = len(scale_pcs(target) & {1, 3, 6, 8, 10})
        if black > allowed_black:
            continue
        shift = (_tonic_pc(target) - _tonic_pc(key)) % 12
        signed = shift if shift <= 6 else shift - 12
        if signed == 0:
            continue
        score = (abs(signed), black)
        if best is None or score < best[0]:
            best = (score, signed, target)
    return (best[1], best[2]) if best else (0, key)


def _anchor_bass_roots(out: list[NoteEvent], bass: list[NoteEvent], info: SongInfo,
                       lo: int, hi: int, transpose: int,
                       pcs: set[int] | None = None,
                       bars: int | None = None) -> list[NoteEvent]:
    """Every bar must state the pitch class of the bass stem's dominant note.

    This is the single cheapest thing that keeps a sparse arrangement sounding like
    the track: the ear tracks the bass root far more than the inner chord tones, so a
    thinned-out left hand that omits the root stops being recognisable even when the
    melody is perfect. Runs after key snapping on purpose - the root wins over the
    key signature when the two disagree.
    """
    bar = 60.0 / info.bpm * info.beats_per_bar if info.bpm else 2.0
    for b in range(bars or int(info.duration / bar) + 1):
        t0, t1 = b * bar, (b + 1) * bar
        seg = [n for n in bass if n.start < t1 and n.start + n.dur > t0]
        here = [n for n in out if t0 <= n.start < t1]
        if not seg or not here:
            continue
        # an in-key bass note that is nearly as long outranks a chromatic one:
        # the root has to be heard, but a beginner should not read accidentals for it
        def weight(n: NoteEvent) -> float:
            pc = (n.pitch + transpose) % 12
            return n.dur * (2.0 if (pcs is None or pc in pcs) else 1.0)
        dom = max(seg, key=weight)
        want = (dom.pitch + transpose) % 12
        if any(n.pitch % 12 == want for n in here):
            continue
        cand = [p for p in range(lo, hi + 1) if p % 12 == want]
        if not cand:
            continue
        low = min(here, key=lambda n: n.pitch)
        low.pitch = min(cand, key=lambda p: abs(p - low.pitch))
    return out


def _anchor_roots(lh: list[NoteEvent], harmony, info: SongInfo,
                  lh_cfg: dict, transpose: int,
                  bars: int | None = None) -> list[NoteEvent]:
    """Make every bar's left hand state that bar's chord root.

    A skeleton thinned to the beat grid, a thinning pass, or a share-the-beat
    pass can each drop the root and leave a third and a fifth ringing, which is
    the chord minus the note that names it.  The harmony track is what the
    reduction is supposed to be spelling, so the lowest note of the bar is moved
    onto the root when the bar has no note of that class at all.

    Only the lowest note moves: it is the one the ear hears as the bass, and
    moving an inner voice would change the chord rather than complete it.
    """
    if not lh or not harmony:
        return lh
    bar = 60.0 / info.bpm * info.beats_per_bar if info.bpm else 2.0
    lo, hi = lh_cfg["range"]
    total = bars or int(info.duration / bar) + 1

    def root_at(t: float) -> int | None:
        # Source key's pitch-class space: this is compared against `src`, which
        # comes straight out of the recording.  Returning the notated pc here
        # made the `root in src` test depend on the key, so which note a silent
        # bar gets -- and whether it later collapses as a duplicate -- changed
        # with `transpose`, and the three tiers parted ways.
        for s_, e_, pc in harmony:
            if s_ <= t < e_:
                return pc % 12
        return harmony[-1][2] % 12 if harmony else None

    for b in range(total):
        t0, t1 = b * bar, (b + 1) * bar
        here = [n for n in lh if t0 <= n.start < t1]
        if not here:
            continue
        root = root_at(t0)
        if root is None:
            continue
        if any(n.pitch % 12 == root for n in here):
            continue
        low = min(here, key=lambda n: n.pitch)
        cand = [p for p in range(lo, hi + 1) if p % 12 == root]
        if not cand:
            continue
        low.pitch = min(cand, key=lambda p: abs(p - low.pitch))
    return lh


def _right_harmony(rh: list[NoteEvent], harmony: list[tuple[float, float, int]],
                   info: SongInfo, pcs: set[int], lo: int, hi: int, transpose: int,
                   max_extra: int, min_note_dur: float) -> list[NoteEvent]:
    """Flesh the melody out with chord tones underneath it.

    A bare top line is why a reduction of an orchestral track can carry the tune and
    still sound thin: the source has strings and brass holding the harmony, and the
    piano has to hand something to the fingers. Only notes that the melody itself
    leaves ringing (a sustained note) get support, so the texture stays legitimately
    playable rather than becoming a chord wall.
    """
    if max_extra <= 0 or not rh or not harmony:
        return rh

    def root_at(t: float) -> int:
        for s_, e_, pc in harmony:
            if s_ <= t < e_:
                return (pc + transpose) % 12
        return (harmony[-1][2] + transpose) % 12

    out = list(rh)
    for n in rh:
        if n.dur < min_note_dur:
            continue
        root = root_at(n.start)
        # chord tones in the key, ranked by how well they sit under the melody
        # Which intervals may sit under the melody is decided by the melody and
        # the chord, never by the key: the key only says how a tone is spelled.
        # Gating on `pcs` made the number of added tones a function of the key,
        # so the three tiers -- which share every preset value but `target_key`
        # -- came out with different left-hand density.  Rank in-key tones first
        # so the spelling stays readable, but never let the key change the count.
        cands = sorted((p for p in range(max(lo, n.pitch - 11), n.pitch)
                        if (p % 12 - root) % 12 in (0, 3, 4, 7, 8)),
                       key=lambda p: (p % 12 not in pcs, abs(p - n.pitch)))
        for k, p in enumerate(cands[:max_extra]):
            out.append(NoteEvent(pitch=p, start=n.start, dur=n.dur * 0.95,
                                 velocity=max(38, n.velocity - 18 - 6 * k),
                                 stem="arranger", confidence=n.confidence))
    return out


def snap_to_key(pitch: int, pcs: set[int], max_dist: int = 1) -> int:
    """Keep a left-hand note inside the key signature unless it is a real chromatic step."""
    if pitch % 12 in pcs:
        return pitch
    for d in range(1, max_dist + 1):
        if (pitch - d) % 12 in pcs:
            return pitch - d
        if (pitch + d) % 12 in pcs:
            return pitch + d
    return pitch


# --------------------------------------------------------------------------- main

def _release(hold: float, info: SongInfo) -> float:
    """Stop a filled left-hand note one beat before the next event.

    The release has to land on a value the page can print.  The old
    ``hold * 0.96`` cut a four-beat hold to 3.84 beats; the sixteenth grid
    rounds that to 3.75, and music21 spells 3.75 as a *triple-dotted half
    note* -- correct, unplayable at sight, and it hit 670 of the 1502
    left-hand notes.  Leaving out a whole beat gives 3.0, a plain dotted half
    that ends on beat 4: the bar keeps a natural breath before the next
    chord, the note count is unchanged, and no tie is needed to read it.
    """
    spq = 60.0 / info.bpm if info.bpm else 0.5
    return max(spq / 4, hold - spq)


def _fill_silent_bars(rh: list[NoteEvent], lh: list[NoteEvent], info: SongInfo,
                      spq: float, lh_cfg: dict, harmony, transpose: int,
                      ref: tuple | None,
                      source_key: str | None = None,
                      bars: int | None = None) -> list[NoteEvent]:
    """Hold a chord under every bar the accompaniment leaves thin.

    A transcription-driven left hand follows the notes it was given, so a bar
    where the separator heard little gets few or no keys -- and on the page that
    reads as a rest in the middle of a song that never stopped.  The recording is
    the authority on what was ringing: whatever classes it sustains across the
    bar and the hand does not already hold get pressed at the bar line, so the
    texture reads as accompaniment instead of as a gap.

    Bars the hand already fills for most of their length are left alone; the tail
    of the recording is too, since a decay after the last note is the end of the
    piece and not a hole.
    """
    if not lh or not ref:
        return lh
    bar = spq * info.beats_per_bar
    lo, hi = lh_cfg["range"]
    hi = min(hi, min((n.pitch for n in rh), default=hi + 2) - 2)
    total_bars = bars or int(info.duration / bar) + 1
    budget = lh_cfg["events_per_bar"]
    max_notes = 3 if budget <= 7 else 4
    last_sound = max([n.start + n.dur for n in list(rh) + list(lh)], default=0.0)
    # The recording is the authority on what was ringing, but it also rings with
    # whatever the bass instrument brushed past.  Copying a class the piece is not
    # in puts a stray accidental on nearly every bar, so the key is the filter:
    # a class outside it has to be worth the ink to earn a place under the hand.
    in_key = None
    if source_key:
        # `src` above is read off the recording, i.e. it is already in the
        # source key's own pitch-class space, and `scale_pcs` returns absolute
        # pitch classes for that key.  Both sides are therefore in the same
        # space: adding `transpose` here would compare the recording against
        # the *notated* key and call every real chord tone a borrowed class.
        in_key = scale_pcs(source_key)

    extra: list[NoteEvent] = []
    for b in range(total_bars):
        t0 = b * bar
        if t0 >= info.duration or t0 >= last_sound:
            continue
        held = [n for n in lh if n.start < t0 + bar and n.start + n.dur > t0]
        hold = min(bar, info.duration - t0)
        src = _strong_pitch_classes(ref[0], ref[1], t0, t0 + hold)
        if not src:
            continue
        have = {(n.pitch - transpose) % 12 for n in held}
        pool = src
        if in_key:
            inside = [p for p in src if p in in_key]
            outside = [p for p in src if p not in in_key]
            # One borrowed class per bar at most: the recording really does ring
            # outside the key now and then, and dropping every one of them costs
            # more similarity than the accidental is worth -- but a bar that
            # prints two or three of them is no longer readable under the hand.
            pool = inside + outside[:1]
        missing = [p for p in pool if p not in have][:max_notes]
        for i, pc in enumerate(missing):
            # `pc` came out of the recording, so it is in the source key's
            # pitch-class space; the page is written in the notated key.
            extra.append(NoteEvent(pitch=_voice((pc + transpose) % 12, i, lo, hi), start=t0,
                                   dur=_release(hold, info), velocity=max(34, 46 - 4 * i),
                                   stem="arranger"))
    if not extra:
        return lh
    return _dedupe_unisons(_share_beat_length(_clip_to_beats(
        _share_beat_length(_quantise_hand(lh + extra, spq), spq), spq), spq))


def _cover_empty_bars(lh: list[NoteEvent], rh: list[NoteEvent], info: SongInfo,
                      spq: float, lh_cfg: dict, harmony, transpose: int,
                      ref: tuple | None,
                      bars: int | None = None) -> list[NoteEvent]:
    """Put the bass back under a bar the left hand leaves completely open.

    `_cover_missing_tones` can only thicken a chord that is already there, so a
    bar with no left-hand key at all gets nothing from it -- and those bars are
    heard as a hole in the accompaniment even when the right hand keeps singing.
    The recording says what was ringing; when its sustained classes include the
    bar's root (or, failing that, whatever it holds most strongly), one low key
    press states what the ear is already hearing.

    This adds an onset where there was none, so it runs before the last pass
    that only edits notes already written: the new note then gets quantised,
    clipped and capped like every other.
    """
    if not ref or not lh:
        return lh
    bar = spq * info.beats_per_bar
    lo, hi = lh_cfg["range"]
    hi = min(hi, min((n.pitch for n in rh), default=hi + 2) - 2)
    total = bars or int(info.duration / bar) + 1
    have: set[int] = set()
    for n in lh:
        for b in range(int(n.start // bar), int((n.start + n.dur - 1e-6) // bar) + 1):
            have.add(b)
    last_sound = max([n.start + n.dur for n in list(rh) + list(lh)], default=0.0)

    def root_at(t: float) -> int | None:
        for s_, e_, pc in harmony:
            if s_ <= t < e_:
                return (pc + transpose) % 12
        return (harmony[-1][2] + transpose) % 12 if harmony else None

    extra: list[NoteEvent] = []
    for b in range(total):
        t0 = b * bar
        if t0 >= info.duration or b in have or t0 >= last_sound:
            continue
        hold = min(bar, info.duration - t0)
        src = _strong_pitch_classes(ref[0], ref[1], t0, t0 + hold)
        if not src:
            continue
        root = root_at(t0)
        pick = root if root is not None and root in src else src[0]
        pitch = _voice(pick, 0, lo, hi)
        extra.append(NoteEvent(pitch=pitch, start=t0, dur=_release(hold, info),
                               velocity=44, stem="arranger"))
    if not extra:
        return lh
    return _dedupe_unisons(_share_beat_length(_clip_to_beats(
        _share_beat_length(_quantise_hand(lh + extra, spq), spq), spq), spq))


def _strong_pitch_classes(y, sr: int, lo: float, hi: float,
                          frac: float = 0.6, rel: float = 0.5,
                          floor: float = 0.0) -> list[int]:
    """Pitch classes the recording actually sustains through a stretch of time.

    A bar mean blurs a struck cymbal into a harmonic tone: one loud transient
    across the bar lifts a pitch class that is silent the rest of the time.  A
    class only counts when it stays above `rel` of its own peak for `frac` of
    the stretch, which keeps transients out and keeps held chords in.

    `rel` alone is per class, so a class with almost no energy passes it just as
    easily as the tonic does -- its own peak is tiny.  `floor` rejects classes
    that stay weak against the loudest class of the same stretch.
    """
    import librosa
    import numpy as np
    seg, sr = librosa.to_mono(y), sr
    a0, a1 = int(lo * sr), int(hi * sr)
    seg = y[..., a0:a1] if y.ndim > 1 else y[a0:a1]
    if seg.size < 2048:
        return []
    c = librosa.feature.chroma_cqt(y=librosa.util.normalize(seg), sr=sr, hop_length=1024)
    if c.size == 0:
        return []
    mean = c.mean(axis=1)
    top = float(mean.max()) if mean.size else 0.0
    out = []
    for i in range(c.shape[0]):
        row = c[i]
        if row.max() <= 0:
            continue
        if floor and top > 0 and float(mean[i]) < floor * top:
            continue
        if float((row >= rel * row.max()).mean()) >= frac:
            out.append(int(i))
    out.sort(key=lambda i: -mean[i])
    return out


def _cover_missing_tones(rh: list[NoteEvent], lh: list[NoteEvent], info: SongInfo,
                         spq: float, lh_cfg: dict, transpose: int,
                         ref: tuple | None, tier: str,
                         bars: int | None = None) -> list[NoteEvent]:
    """Voice the recording's sustained pitch classes the left hand leaves out.

    A bar can be loud and still lack what the listener hears as the chord: the
    skeleton holds root and fifth, so the third -- the note that names the
    chord -- is the one most often missing.  Adding it is not padding, it is
    saying the chord that was already playing.

    Every added tone lands on a strike the bar already has, so the number of
    key presses never rises; only the chord under the hand gets thicker, which
    is what a pianist does when a chord sounds thin.  Classes must be sustained
    in the recording and absent from the hand, at most two per bar.
    """
    if not ref or not lh:
        return lh
    bar = spq * info.beats_per_bar
    lo, hi = lh_cfg["range"]
    hi = min(hi, min((n.pitch for n in rh), default=hi + 2) - 2)
    cap = 2
    total = bars or int(info.duration / bar) + 1
    present: dict[int, list[NoteEvent]] = {}
    for n in lh:
        for b in range(int(n.start // bar), int((n.start + n.dur - 1e-6) // bar) + 1):
            present.setdefault(b, []).append(n)

    extra: list[NoteEvent] = []
    for b in range(total):
        t0 = b * bar
        if t0 >= info.duration or b not in present:
            continue
        hold = min(bar, info.duration - t0)
        src = _strong_pitch_classes(ref[0], ref[1], t0, t0 + hold)
        if not src:
            continue
        room = [n.pitch for n in present[b]]
        have = {(p - transpose) % 12 for p in room}
        candidates = [p for p in src if p not in have]
        at = min(n.start for n in present[b])
        # Clashes only matter between tones struck together: a passing tone a
        # semitone away later in the bar is normal writing, not a rub.
        struck = {(n.pitch - transpose) % 12 for n in present[b]
                  if abs(n.start - at) < 1e-6}
        added: list[int] = []
        for p in candidates:
            if len(added) >= cap:
                break
            # A filling class adjacent to one already present does not supply the
            # missing third, it supplies a rub: the chroma reports a neighbour of
            # a loud real tone as a class of its own.  Compare pitch classes, not
            # sounding pitches -- two octaves of C are not a clash.
            if any(min((p - h) % 12, (h - p) % 12) <= 1
                   for h in struck | set(added)):
                continue
            added.append(p)
        for p in added:
            pitch = _voice((p + transpose) % 12, 0, lo, hi)
            while pitch < min(room):
                pitch += 12
            pitch = min(max(pitch, lo), hi)
            extra.append(NoteEvent(pitch=pitch, start=at, dur=_release(hold, info),
                                   velocity=48, stem="arranger"))
    if not extra:
        return lh
    return _dedupe_unisons(_share_beat_length(_clip_to_beats(
        _share_beat_length(_quantise_hand(lh + extra, spq), spq), spq), spq))


meta_spread: dict[str, int] = {}
# Widest a left-hand chord may be, in semitones.  An octave is what a hand takes
# without rolling it, and the measurements put the delivered chords at 14 to 17
# semitones -- larger than the six-key cap, which only asked whether the hand
# could reach at all.
SPAN_CAP = 12
# Which keys to keep when a strike asks for more than the hand takes.  "inner"
# walks up from the bass, which makes a compact chord; "outer" keeps the bass and
# then reaches for the highest available tone, which is the bass-plus-top voicing
# a pianist actually plays when the inner parts would only double the melody.
# Measured, not assumed: see PLAN 10.13.
CAP_KEEP = "harmonic"
# How many of the keys the cap took away were the only instance of their pitch
# class on that strike (real harmony lost) rather than a doubling of one already
# sounding (nothing audible lost).  Reported in meta so the cost can be seen.
META_CAP_DETAIL: dict[str, int] = {}
# Harmonic rank of a written pitch class against the bar's root, best first:
# root, then the fifth, then the third, then whatever else is in the key.  When
# the cap has to take a key away it should take the least load-bearing one, and
# pitch order has no idea which that is -- measured, it throws away 235 of 243
# keys that were the only statement of their pitch class.
_HARM_RANK = {0: 0, 7: 1, 3: 2, 4: 2, 2: 3, 5: 3, 9: 3, 11: 3, 1: 4, 6: 4, 8: 4, 10: 4}


def _cap_strikes(lh: list[NoteEvent], cap: int,
                 span_cap: int = SPAN_CAP,
                 root_at=None) -> tuple[list[NoteEvent], int]:
    """Keep at most `cap` keys under the left hand at any instant.

    The fills add a bar's missing chord tones onto the strike the bar already
    has, which is right for the harmony and wrong for the hand: with the fills
    on, one strike can ask for six keys inside a ninth while the preset gives
    the left hand two, and the only gate left is the six-key hard cap -- a
    "can the hand span it" number, not a playability rule.

    Moving the surplus to another beat was measured and rejected: chroma is
    compared instant by instant, so a tone played where the recording has no
    such tone scores as a wrong pitch, and shifting cost 5.5 points
    (0.7364 -> 0.6809) for nothing the listener can hear.

    What survives here is chosen by pitch class, not by pitch.  Dropping an
    octave doubling of a tone the strike already sounds changes nothing anyone
    hears; dropping the only note of its pitch class thins the chord.  So the
    lowest key stays -- it is the root the bar is built on -- and the rest are
    taken in ascending order while they keep introducing a new pitch class.
    Only if that still leaves the strike over `cap` does a duplicated pitch
    class get dropped, highest first.

    Returns the new hand and how many keys the cap took away.
    """
    if cap <= 0 or not lh:
        return lh, 0
    groups: dict[float, list[NoteEvent]] = {}
    for n in lh:
        groups.setdefault(round(n.start, 6), []).append(n)
    out: list[NoteEvent] = []
    dropped = 0
    for _t in sorted(groups):
        group = sorted(groups[_t], key=lambda n: n.pitch)
        # The span rule applies even to a strike that is under the key cap: a
        # three-key chord is still unplayable if it is a twelfth wide, and that
        # early return is exactly how the wide ones survived.
        if len(group) <= cap and group[-1].pitch - group[0].pitch <= span_cap:
            out.extend(group)
            continue
        keep = [group[0]]
        seen = {group[0].pitch % 12}
        lo = group[0].pitch
        rest = group[1:]
        if CAP_KEEP == "outer" and len(group) > cap:
            rest = group[-1:0:-1]
        elif CAP_KEEP == "harmonic" and len(group) > cap and root_at is not None:
            r = root_at(_t)
            rest = sorted(rest, key=lambda n: (_HARM_RANK.get((n.pitch - r) % 12, 9),
                                               n.pitch))
        for n in rest:
            if len(keep) >= cap:
                break
            if n.pitch % 12 not in seen and n.pitch - lo <= span_cap:
                keep.append(n)
                seen.add(n.pitch % 12)
        # Every remaining key doubles a pitch class the strike already has, so
        # filling up to `cap` with them costs no harmony at all.
        for n in group:
            if len(keep) >= cap:
                break
            if n not in keep and n.pitch - lo <= span_cap:
                keep.append(n)
        gone = [n for n in group if n not in keep]
        dropped += len(gone)
        META_CAP_DETAIL["dup"] = META_CAP_DETAIL.get("dup", 0) + sum(
            1 for n in gone if (n.pitch % 12) in {k.pitch % 12 for k in keep})
        META_CAP_DETAIL["uniq"] = META_CAP_DETAIL.get("uniq", 0) + sum(
            1 for n in gone if (n.pitch % 12) not in {k.pitch % 12 for k in keep})
        out.extend(sorted(keep, key=lambda n: n.pitch))
    out.sort(key=lambda n: n.start)
    return out, dropped


def build_arrangement(
    melody: list[NoteEvent],
    bass: list[NoteEvent],
    chords: list[ChordEvent],
    info: SongInfo,
    preset_name: str = "easy",
    presets: dict | None = None,
    transpose: int = 0,
    ref_audio: "tuple | None" = None,
) -> Arrangement:
    presets = presets or load_presets()
    if preset_name not in presets:
        raise KeyError(f"unknown preset {preset_name!r}, have {list(presets)}")
    preset = presets[preset_name]
    # Every bar line, slot and density budget below is counted from beat 0, so
    # beat 0 has to be at t=0.  A take that starts 90 ms late, or with a
    # pickup, otherwise puts the whole piece off the bar grid and the last bar
    # comes out one beat short or one beat long.  Rebase the input on the beat
    # origin and put it back on the output, so a duration is unchanged and the
    # reference audio stays addressable at the times the bars name.
    origin = float(getattr(info, "beat_origin", 0.0) or 0.0)
    if origin:
        def _shift(notes, delta):
            return [NoteEvent(pitch=n.pitch, start=n.start + delta, dur=n.dur,
                              velocity=n.velocity, stem=n.stem,
                              confidence=n.confidence, finger=n.finger)
                    for n in notes]
        melody = _shift(melody, -origin)
        bass = _shift(bass, -origin)
        chords = [ChordEvent(start=max(0.0, c.start - origin),
                             end=max(0.0, c.end - origin), root=c.root,
                             quality=c.quality, confidence=c.confidence)
                  for c in chords]
        if ref_audio:
            y, sr = ref_audio
            pad = int(round(origin * sr))
            if pad > 0:
                ref_audio = (np.concatenate([np.zeros(pad, dtype=y.dtype), y]), sr)
        info = SongInfo(path=info.path, duration=info.duration,
                        bpm=info.bpm,
                        beats=[b - origin for b in info.beats],
                        beats_per_bar=info.beats_per_bar, beat_origin=0.0)
    spq = 60.0 / info.bpm if info.bpm else 0.5
    # How many bars the score will have, decided the same way the renderer
    # decides it so every per-bar budget lands on a bar that prints.  Taken
    # from the transcribed onsets, which is what both hands are built out of.
    score_bars = _score_bar_count(info, spq,
                                  [n.start for n in melody] + [n.start for n in bass])
    rh_cfg = preset["right_hand"]
    lo_rh, hi_rh = rh_cfg["range"]

    melody = collapse_octaves(melody)
    key, _ = estimate_key(original_pitch_classes(melody, bass))
    shift, notated_key = transpose_plan(
        key, preset["global"].get("allowed_black_keys", 12),
        preset["global"].get("target_key"),
        pitches=[n.pitch for n in melody] + [n.pitch for n in bass],
        rank=preset["global"].get("black_key_rank"))
    # The tiers may differ in exactly one thing: this semitone count.  So the
    # arrangement is built in the source key, where no stage can see which key
    # it will be printed in, and shifted once at the very end.  Building in the
    # notated key instead let the key leak into the result: the left-hand
    # snap and the chord tones were computed from the notated key's scale, so
    # the same source note could be snapped a semitone and could create or
    # destroy a unison, and the three tiers ended up with different notes.
    shift = transpose + shift
    pcs = scale_pcs(key)
    transpose = 0

    kept = [n for n in melody if n.dur >= rh_cfg["min_note"]]
    folded = fold_line([n.pitch + transpose for n in kept], lo_rh, hi_rh)
    rh = [NoteEvent(pitch=f, start=n.start, dur=n.dur, velocity=n.velocity,
                    stem=n.stem, confidence=n.confidence)
          for n, f in zip(kept, folded)]
    rh, rh_spill = _split_overlaps(rh, min(rh_cfg["max_simultaneous"], MAX_KEYS_RIGHT))
    rh.sort(key=lambda n: n.start)
    if rh_cfg.get("legato", True):
        rh = _legato(rh, max_gap=rh_cfg.get("max_gap", 0.22),
                     max_leap=rh_cfg.get("max_leap", 7))

    harmony = harmonic_timeline(chords, bass, info)
    rh = _right_harmony(rh, harmony, info, pcs, lo_rh, hi_rh, transpose,
                        rh_cfg.get("harmony_notes", 0),
                        rh_cfg.get("harmony_min_note_dur", 0.5))
    rh, spill2 = _split_overlaps(rh, min(rh_cfg["max_simultaneous"], MAX_KEYS_RIGHT))
    rh_spill += spill2
    rh.sort(key=lambda n: n.start)
    rh_min = min((n.pitch for n in rh), default=hi_rh)
    lh = _build_left_hand(harmony, preset, info, rh_min, pcs=pcs,
                          transpose=transpose, bass=bass, bars=score_bars)
    if rh_spill:
        folded = [_fold_into(lh, n, preset["left_hand"]["range"], spq) for n in rh_spill]
        lh = lh + [f for f in folded if f is not None]
    lh = _thin_overlaps(lh, min(preset["left_hand"]["max_simultaneous"],
                                MAX_KEYS_LEFT), prefer_high=False)
    lh = _avoid_collisions(lh, rh)
    # order matters: put every strike on the written grid, give the notes of one
    # written beat one shared length, and only then cut each back at the next
    # strike -- clipping before the shared length would undo the clipping.
    # a "-" holds every sounding note of the hand, so one beat can only print one
    # length: after clipping, beats with several notes take the shortest, which
    # is both what a hand does (a new key press lifts the old one) and exactly
    # what the text can say.  Single-note beats keep their own length.
    rh = _dedupe_unisons(_share_beat_length(_clip_to_beats(
        _share_beat_length(_quantise_hand(rh, spq), spq), spq), spq))
    lh = _dedupe_unisons(_share_beat_length(_clip_to_beats(
        _share_beat_length(_quantise_hand(lh, spq), spq), spq), spq))
    lh = _fill_silent_bars(rh, lh, info, spq, preset["left_hand"],
                           harmony, transpose, ref_audio, key, bars=score_bars)
    lh = _cover_missing_tones(rh, lh, info, spq, preset["left_hand"],
                             transpose, ref_audio, preset_name,
                             bars=score_bars)
    lh = _cover_empty_bars(lh, rh, info, spq, preset["left_hand"],
                           harmony, transpose, ref_audio, bars=score_bars)
    lh = _anchor_roots(lh, harmony, info, preset["left_hand"], transpose,
                       bars=score_bars)
    # Spelling is the only thing the key is allowed to decide here.  Every stage
    # above writes notes from the transcription, the harmony timeline or the
    # chord tones, and each of them can be a semitone off the key; snapping each
    # stage on its own let the later ones reintroduce out-of-key notes, and how
    # many depended on the tier's key.  One pass at the end, after every stage
    # that can add a note, makes the left hand equally in-key on all three.
    lh = [NoteEvent(pitch=snap_to_key(n.pitch, pcs), start=n.start, dur=n.dur,
                    velocity=n.velocity, stem=n.stem, confidence=n.confidence)
          for n in lh]
    # Last word on how many keys one hand presses at once: transcription density
    # in the bass is not a promise the page has to keep.  It has to run after the
    # shared beat lengths, because that step lengthens notes and is what creates
    # the overlaps in the first place.
    lh = _dedupe_unisons(_thin_chords(lh, MAX_KEYS_LEFT))
    # Re-striking a held note can land on a strike the page already prints, so
    # the same key at the same instant can appear twice; one key press is one
    # printed note, and the longest reading of it is the one to keep.
    # The two hand limits are the rule the score has to keep whatever tier it
    # is: one hand may hold six keys, the other two.  A dense tier widens the
    # range it plays in, not how many fingers it has.
    rh = _thin_chords(_collapse_beat_onsets(_end_before_strike_beat(_dedupe_beats(
        _dedupe_unisons(_thin_chords(_one_length_per_strike(_cut_at_next_strike(
            _collapse_beat_onsets(rh, spq), spq), spq),
            MAX_KEYS_RIGHT)), spq), spq), spq), MAX_KEYS_RIGHT)
    lh = _thin_chords(_collapse_beat_onsets(_end_before_strike_beat(_dedupe_beats(
        _dedupe_unisons(_thin_chords(_one_length_per_strike(_cut_at_next_strike(
            _collapse_beat_onsets(lh, spq), spq), spq),
            MAX_KEYS_LEFT)), spq), spq), spq), MAX_KEYS_LEFT)
    rh.sort(key=lambda n: n.start)
    lh.sort(key=lambda n: n.start)
    hand_cap = min(preset["left_hand"].get("max_simultaneous", MAX_KEYS_LEFT),
                   MAX_KEYS_LEFT)
    # harmony speaks in the source key; the strikes about to be capped are
    # written pitches, so the root is shifted into the notated key once here.
    _harm = [(a, b, (r + transpose) % 12) for a, b, r in harmony]

    def _root_at(t: float) -> int:
        for a, b, r in _harm:
            if a <= t < b:
                return r
        return _harm[0][2] if _harm else 0

    lh, capped = _cap_strikes(lh, hand_cap, root_at=_root_at)

    # One shift, applied to finished pitches: the tiers cannot differ by more.
    if shift:
        rh = [NoteEvent(pitch=n.pitch + shift, start=n.start, dur=n.dur,
                        velocity=n.velocity, stem=n.stem, confidence=n.confidence)
              for n in rh]
        lh = [NoteEvent(pitch=n.pitch + shift, start=n.start, dur=n.dur,
                        velocity=n.velocity, stem=n.stem, confidence=n.confidence)
              for n in lh]

    if origin:
        rh = _shift(rh, origin)
        lh = _shift(lh, origin)
    # Two different quantities used to share one name here, which is how the
    # tier order got inverted: `accidental_pitch_classes` is how many pitch
    # classes of the *key signature* sit on black keys (G major -> F#, so 1).
    # It says nothing about the notes.  A piece written around a key other than
    # C turns its accidentals into black keys when notated in C, so the tier
    # with the cleanest signature -- 0 by this measure -- had 27 and 70 black
    # keys actually under the hand against 2 and 0 for the tier with one.
    # The played count is what the tier promises, so it gets its own field.
    accidental_pc = len(scale_pcs(notated_key) & BLACK_PCS)
    black_key_notes = sum(1 for n in rh + lh if n.pitch % 12 in BLACK_PCS)
    # Drop the notes that fall outside the score.  Both renderers already clip
    # them -- a note opening after the last bar line cannot be printed -- so
    # they were only ever a phantom: the pickle counted five notes of clip1
    # that no reader would ever see, and a note-count comparison against the
    # printed notation disagreed for no real reason.  The detector's tail
    # blip is exactly what :func:`tail_beats` exists to exclude, so it must not
    # survive into the score either.
    score_beats = int(score_bars * info.beats_per_bar)
    end = score_beats * spq - 1e-9
    rh = [n for n in rh if n.start < end]
    lh = [n for n in lh if n.start < end]
    return Arrangement(
        right=rh, left=lh, info=info, preset=preset_name, key_signature=notated_key,
        meta={"transpose": shift, "source_key": key, "score_beats": score_beats,
              "accidental_pitch_classes": accidental_pc,
              "key_signature_accidentals": accidental_pc,
              "black_key_notes": black_key_notes,
              "preset_label": preset.get("label", preset_name),
              "capped_strikes": capped,
              "capped_dup": META_CAP_DETAIL.get("dup", 0),
              "capped_uniq": META_CAP_DETAIL.get("uniq", 0),
              "notes_per_bar_lh": round(len(lh) / max(info.duration / (60 / info.bpm)
                                                     / info.beats_per_bar, 1), 2),
              "notes_per_bar_rh": round(len(rh) / max(info.duration / (60 / info.bpm)
                                                     / info.beats_per_bar, 1), 2)},
    )


def original_pitch_classes(melody: list[NoteEvent], bass: list[NoteEvent]) -> np.ndarray:
    hist = np.zeros(12)
    for n in melody:
        hist[n.pitch % 12] += max(n.dur, 0.05) * (1.0 if n.stem == "other" else 0.7)
    for n in bass:
        hist[n.pitch % 12] += max(n.dur, 0.05) * 1.2
    return hist


def estimate_key(hist: np.ndarray) -> tuple[str, int]:
    """Krumhansl-Schmuckler correlation; returns key name and accidental pcs count."""
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    if hist.sum() == 0:
        return "C", 0
    h = hist / hist.sum()
    best, best_key = -2.0, "C"
    for root in range(12):
        for prof, suffix in ((major, ""), (minor, "m")):
            r = np.corrcoef(np.roll(h, -root), prof)[0, 1]
            if r > best:
                best, best_key = r, PC_NAMES[root] + suffix
    # accidental load of the notated key: count black keys in its scale
    scale_pcs = {0, 2, 4, 5, 7, 9, 11} if not best_key.endswith("m") else {0, 2, 3, 5, 7, 8, 10}
    tonic = PC_NAMES.index(best_key[:-1] if best_key.endswith("m") else best_key)
    if best_key.endswith("m"):
        tonic = (tonic + 3) % 12
    black = len([p for p in scale_pcs if (p + tonic) % 12 in (1, 3, 6, 8, 10)])
    return best_key, black
