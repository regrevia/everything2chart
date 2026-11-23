"""Per-bar width audit: does every bar hold exactly its metre?

The complaint this answers is "a 4/4 tune sometimes has one beat too many, or
half a beat too few".  That is not the bar COUNT -- a wrong count is a whole
bar missing -- it is a bar whose contents do not sum to the metre, which the
reader hears as a hole or a stutter at the bar line.  So both renderers are
measured bar by bar:

* staff (MusicXML) -- every ``<measure>``'s filled quarter length, read by
  music21, next to the metre in force.
* numbered notation -- every cell of every row, in beats, from the legend the
  header prints: a bare number is one beat, ``_`` halves it, two halves a
  quarter, ``-`` holds the previous one more beat.

A bar is short when its length is not a whole number of beats, or when it is
not a full bar and is not the last one.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---- numbered notation ----------------------------------------------------

def _beats_of(tok: str) -> float:
    """Length of one printed cell, in beats."""
    t = tok.strip("()[]")
    if not t:
        return 0.0
    if set(t) <= {"-"}:
        return float(len(t))          # "-" holds the previous note one beat
    b = 1.0
    for ch in t:
        if ch == "_":
            b /= 2.0
        elif ch == ".":
            b *= 1.5
    return b


def _token_beats(tok: str) -> float:
    """Beats one token occupies.

    ``[a+b]`` is a chord: every member is struck at the same instant, so the
    token lasts as long as its LONGEST member, not the sum.  Summing them made
    every chord read as a full bar over and hid the real deficits.
    """
    inner = tok.strip()
    if inner.startswith("[") and inner.endswith("]"):
        return max((_beats_of(m) for m in inner[1:-1].split("+")), default=0.0)
    return _beats_of(inner)


def _side(cell: str) -> float:
    """Beats written for one hand inside one beat cell."""
    return sum(_token_beats(t) for t in cell.split())


def cell_beats(cell: str) -> float:
    """Beats consumed by one shown beat.

    ``5_ 6_([3,._+7,._])`` is an eighth plus an eighth with a left-hand chord
    under them, which is ONE beat: the bracketed part is the other hand at the
    same instant, not a longer note.  Adding the two would count every
    accompanied beat twice and report every bar as over-full.
    """
    if "(" not in cell:
        return _side(cell)
    rh = cell.split("(")[0]
    lh = cell[cell.index("(") + 1:cell.rindex(")")] if ")" in cell else ""
    return max(_side(rh), _side(lh))


def jianpu_bars(path: Path) -> list[float]:
    bars: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = line.strip().strip("|").split("|")
        bars.append(sum(cell_beats(c) for c in cells))
    return bars


# ---- staff ----------------------------------------------------------------


def staff_bars(path: Path) -> list[float]:
    from music21 import converter

    score = converter.parse(str(path))
    out: list[float] = []
    for part in score.parts:
        for m in part.getElementsByClass("Measure"):
            out.append(float(m.duration.quarterLength))
    return out


def report(name: str, lens: list[float], bpb: float) -> dict:
    full = round(bpb, 4)
    wrong = [i for i, v in enumerate(lens) if abs(v - full) > 1e-6]
    return {
        "arm": name,
        "bars": len(lens),
        "beats_per_bar": bpb,
        "full": len(lens) - len(wrong),
        "wrong_bar_index": [i + 1 for i in wrong],
        "wrong_bar_beats": [round(lens[i], 4) for i in wrong],
        "detail": [round(v, 4) for v in lens],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="output dirs, one per arm")
    ap.add_argument("--stem", default="clip1-easy")
    ap.add_argument("--bpb", type=float, default=4.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rows = []
    for d in a.dirs:
        p = Path(d)
        jp, mx = p / f"{a.stem}.jianpu.txt", p / f"{a.stem}.musicxml"
        if not jp.exists() or not mx.exists():
            print(f"skip {d}: no {a.stem} artifacts", file=sys.stderr)
            continue
        j = jianpu_bars(jp)
        s = staff_bars(mx)
        rows.append(report(f"{p.name}/jianpu", j, a.bpb))
        rows.append(report(f"{p.name}/staff(rh+lh)", s, a.bpb))
    if a.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        return 0
    for r in rows:
        print(f"{r['arm']:<26} {r['bars']:>3} bars  "
              f"{r['full']:>3} exactly {r['beats_per_bar']:g} beats"
              f"   off: {r['wrong_bar_beats']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
