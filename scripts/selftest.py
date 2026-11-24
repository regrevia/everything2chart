#!/usr/bin/env python3
"""Regression gate for the numbered-notation spellings that broke silently.

Every defect this file pins was found by a person listening to the output, not
by the code: a bar that read a quarter beat short, a beat with a hole in the
middle of it, a left-hand chord printed with no right-hand event beside it.
The audit scripts measure those defects, but measuring is not the same as
failing -- `scripts/bar_audit.py` reads a cell as ``max(right, left)``, so a
beat one hand spells short is invisible there as long as the other hand spells
all of it.  This file is the failing version of those questions, built on
hand-written arrangements so it needs no audio, no demucs and no cache.

Two rules hold for every case, and they are the contract the legend promises:

    every bar sums to its metre, measured per hand, per bar
    every cell sums to a whole beat, measured per hand, per cell

and a third that keeps the first two honest:

    reading the printed notation back yields exactly the score's pitches

a rest contributes no pitch, so the third rule is what separates "the width was
repaired" from "the width was padded with invented notes".

Usage:
    PYTHONPATH=src .venv/bin/python scripts/selftest.py [--verbose]
    PYTHONPATH=src .venv/bin/python scripts/selftest.py --list
Exit code is the number of failed cases, so it can gate a release.
"""
from __future__ import annotations

import argparse
import io
import sys
import tempfile
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))

from bar_audit import hand_beats  # noqa: E402
from everything2chart.notation import render_jianpu  # noqa: E402
from everything2chart.types import Arrangement, NoteEvent, SongInfo  # noqa: E402
from jianpu_roundtrip import read_jianpu  # noqa: E402

BPM = 120.0          # one beat = exactly 0.5 s, so beat N starts at N * 0.5
SPQ = 60.0 / BPM


def arr(right, left, beats: int | None = None, bpb: int = 4,
        key: str = "C") -> Arrangement:
    """An arrangement written in BEATS rather than seconds.

    ``right``/``left`` are ``(pitch, start_beat, length_in_beats)`` triples so a
    case reads as the rhythm it is testing.  ``beats`` overrides the score
    length; leaving it out lets the renderer decide, which is itself a rule
    worth exercising.
    """
    def build(spec):
        return [NoteEvent(pitch=p, start=b * SPQ, dur=d * SPQ) for p, b, d in spec]

    longest = max((b + d for _, b, d in list(right) + list(left)), default=1.0)
    info = SongInfo(path="selftest", duration=longest * SPQ, bpm=BPM,
                    beats=[i * SPQ for i in range(int(longest) + 1)], beats_per_bar=bpb)
    meta = {} if beats is None else {"score_beats": beats}
    return Arrangement(right=build(right), left=build(left), info=info,
                       preset="test", key_signature=key, meta=meta)


def bars_of(text: str) -> list[list[str]]:
    return [ln.strip().strip("|").split("|")
            for ln in text.splitlines() if ln.startswith("|")]


# ---------------------------------------------------------------- invariants

def check_bars(text: str, bpb: int) -> list[str]:
    """Every bar sums to its metre; only the last bar may be short.

    The two hands are held to different rules on purpose, because the notation
    treats them differently and pretending otherwise makes the gate lie.

    The right hand carries the metre.  It is the one printed outside the
    parentheses, and a bar where it sums short is exactly the "少 half a beat"
    the reader hears -- clip1's fourth bar did this at 3.0 of 4.

    The left hand is folded accompaniment, printed inside ``(...)`` only on the
    beats it sounds.  It must never *overrun* the metre, but when it falls
    silent the cell simply has no left-hand symbol, so a short left hand reads
    as "no accompaniment here" rather than as a hole.  On every real
    arrangement measured (clip1 and clip2, all three tiers, 12 bars each) the
    left hand does sum to the metre in every bar, via holds -- so this is not a
    licence to be sloppy, it is the one case the printed text cannot answer.
    """
    bad: list[str] = []
    rows = bars_of(text)
    for i, row in enumerate(rows, 1):
        right = left = 0.0
        for cell in row:
            r, l = hand_beats(cell)
            right += r
            left += l
        final = i == len(rows)
        if abs(right - bpb) > 1e-6 and not (final and right < bpb and right > 0):
            bad.append(f"bar {i} right hand sums to {right:g}, metre is {bpb}")
        if left > bpb + 1e-6:
            bad.append(f"bar {i} left hand overruns the metre: {left:g} > {bpb}")
    return bad


def check_cells(text: str) -> list[str]:
    """Every cell is a whole beat in each hand, or an untouched beat in it."""
    bad: list[str] = []
    for i, row in enumerate(bars_of(text), 1):
        for j, cell in enumerate(row, 1):
            for hand, v in zip(("right", "left"), hand_beats(cell)):
                if 1e-6 < v < 1.0 - 1e-6:
                    bad.append(f"bar {i} beat {j} {hand} spells {v:g} of a beat: {cell.strip()!r}")
    return bad


def check_roundtrip(text: str, a: Arrangement) -> list[str]:
    """The printed tokens name exactly the score's pitches -- rests add none."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.jianpu.txt"
        p.write_text(text, encoding="utf-8")
        buf = io.StringIO()
        with redirect_stdout(buf):
            printed = sorted(read_jianpu(p))
    scored = sorted(n.pitch for n in list(a.right) + list(a.left))
    if printed == scored:
        return []
    # Compare as multisets: a repeated pitch counts twice, so a duplicated
    # token is visible here instead of cancelling out of a set difference.
    got, want = Counter(printed), Counter(scored)
    extra = sorted((got - want).elements())
    missing = sorted((want - got).elements())
    return [f"printed {len(printed)} pitches {extra} not in score; "
            f"score {len(scored)} pitches {missing} not printed"]


# -------------------------------------------------------------------- cases
#
# Each case returns (arrangement, bpb, expected_substrings).  The substrings are
# the point: they pin the spelling that was wrong, so a future refactor that
# re-breaks it fails here rather than on someone's speakers.

def case_two_half_beats() -> tuple[Arrangement, int, tuple[str, ...]]:
    """Two notes half a beat apart must not print as adjacent sixteenths.

    This is the bug that cost clip1 four bars: `6,__ 5,__` puts the second note
    a quarter beat after the first, so a reader plays the bar a quarter beat
    short.  The gap has to be spelled.
    """
    a = arr(right=[(76, 0.0, 0.25), (74, 0.5, 0.25)], left=[(48, 0.0, 1.0)], beats=4)
    return a, 4, ("0__",)


def case_late_strike() -> tuple[Arrangement, int, tuple[str, ...]]:
    """A note entering on the last sixteenth still leaves a full beat."""
    a = arr(right=[(72, 0.75, 0.25)], left=[(48, 0.0, 1.0)], beats=4)
    return a, 4, ("0.",)


def case_silent_right_hand() -> tuple[Arrangement, int, tuple[str, ...]]:
    """A beat the right hand rests through must still have a right-hand event.

    Printing `([1,+2,])` gives the right hand no symbol at all for that beat, so
    the cell is not readable as a full beat -- clip1's fourth bar did this.
    """
    a = arr(right=[(72, 0.0, 1.0)], left=[(48, 0.0, 2.0)], beats=4)
    return a, 4, ("0(",)


def case_silent_left_hand() -> tuple[Arrangement, int, tuple[str, ...]]:
    """The mirror case: a beat the left hand rests through needs no symbol."""
    a = arr(right=[(72, 0.0, 2.0)], left=[(48, 0.0, 1.0)], beats=4)
    return a, 4, ()


def case_hold_over_bar_line() -> tuple[Arrangement, int, tuple[str, ...]]:
    """A note sounding into the next bar is written `-` there, not struck again."""
    a = arr(right=[(72, 0.0, 5.0)], left=[(48, 0.0, 5.0)], beats=8)
    return a, 4, ("-",)


def case_short_final_bar() -> tuple[Arrangement, int, tuple[str, ...]]:
    """An incomplete final bar is music, not a bar that failed to fill.

    Six beats of 4/4 is one full bar and a half bar.  The renderer must print
    both, and the width check must not call the half bar a defect -- only the
    last bar is allowed to be short.
    """
    a = arr(right=[(72, 0.0, 1.0), (74, 5.0, 1.0)], left=[(48, 0.0, 1.0)], beats=6)
    return a, 4, ()


def case_dotted_note() -> tuple[Arrangement, int, tuple[str, ...]]:
    """A dotted quarter is spelled with a dot and pads out its own beat."""
    a = arr(right=[(72, 0.0, 1.5), (76, 1.5, 0.5)], left=[(48, 0.0, 2.0)], beats=4)
    return a, 4, (".",)


def case_octave_marks() -> tuple[Arrangement, int, tuple[str, ...]]:
    """Register is carried by `'` and `,`, and both must survive rendering."""
    a = arr(right=[(88, 0.0, 1.0), (84, 1.0, 1.0)], left=[(40, 0.0, 1.0), (48, 1.0, 1.0)],
            beats=4)
    return a, 4, ("'", ",")


def case_three_four() -> tuple[Arrangement, int, tuple[str, ...]]:
    """The metre is the arrangement's, not a constant four."""
    a = arr(right=[(72, 0.0, 1.0), (74, 2.0, 1.0)], left=[(48, 0.0, 1.0)], beats=6, bpb=3)
    return a, 3, ()


def case_dense_eighth_run() -> tuple[Arrangement, int, tuple[str, ...]]:
    """Eight eighths in a bar, with the left hand changing under each pair.

    Two notes inside one beat must print inside one cell, and neither hand may
    drop a note just because the cell is busy.
    """
    right = [(72 + (i % 3), 0.5 * i, 0.5) for i in range(16)]
    left = [(48 + i, 2.0 * (i // 2), 2.0) for i in range(4)]
    return arr(right=right, left=left, beats=8), 4, ()


def case_chord_strike() -> tuple[Arrangement, int, tuple[str, ...]]:
    """Simultaneous notes on one beat print as one bracketed strike."""
    a = arr(right=[(72, 0.0, 1.0), (76, 0.0, 1.0)], left=[(48, 0.0, 1.0), (55, 0.0, 1.0)],
            beats=4)
    return a, 4, ("[", "+")


CASES = {name[len("case_"):]: fn for name, fn in sorted(globals().items())
         if name.startswith("case_") and callable(fn)}


# --------------------------------------------------------------------- runner

def run(names: list[str], verbose: bool) -> int:
    failed = 0
    for name in names:
        a, bpb, wants = CASES[name]()
        text = render_jianpu(a, beats_per_bar=bpb)
        problems = check_bars(text, bpb) + check_cells(text) + check_roundtrip(text, a)
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "x.jianpu.txt").write_text(text, encoding="utf-8")
        rows = bars_of(text)
        prints = {c.strip().lstrip("0123456789") for row in rows for c in row}
        for want in wants:
            if want not in text and not any(want in cell for cell in prints):
                problems.append(f"expected {want!r} in the printed notation")
        if problems:
            failed += 1
            print(f"FAIL {name}")
            for p in problems:
                print(f"       {p}")
            for row in rows:
                print("       | " + " | ".join(c.strip() for c in row) + " |")
        elif verbose:
            print(f"ok   {name}: {len(rows)} bars")
            for row in rows:
                print("       | " + " | ".join(c.strip() for c in row) + " |")
    total = len(names)
    print(f"\n{total - failed}/{total} cases pass"
          + ("" if failed == 0 else f"  --  {failed} FAILED"))
    return failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", help="case names to run; default all")
    ap.add_argument("--list", action="store_true", help="list case names")
    ap.add_argument("--verbose", "-v", action="store_true", help="show passing bars too")
    a = ap.parse_args()
    if a.list:
        for name in CASES:
            doc = (CASES[name].__doc__ or "").strip().splitlines()[0]
            print(f"{name:<26} {doc}")
        return 0
    names = a.cases or list(CASES)
    unknown = [n for n in names if n not in CASES]
    if unknown:
        raise SystemExit(f"unknown case(s): {', '.join(unknown)}; try --list")
    return run(names, a.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
