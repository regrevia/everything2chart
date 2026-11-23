#!/usr/bin/env python3
"""Read a numbered-notation file back into notes and compare with the score.

The legend printed in the jianpu header is a contract, so it can be inverted:
every remaining token is a note the reader will play, and every note the score
holds must appear as a token.  This is what distinguishes padding a short bar
with rests from padding it with music -- the bar widths can be made to add up
either way, and only the token-to-note comparison says which one happened.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/jianpu_roundtrip.py <arr.pkl> [arr.pkl ...]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

_SOLFA_TO_PC = {"1": 0, "2": 2, "3": 4, "4": 5, "5": 7, "6": 9, "7": 11}


def _pitch_of(tok: str, tonic: int) -> int | None:
    """Return the MIDI pitch a printed token names, or None for a rest/hold."""
    t = tok.strip()
    if not t or set(t) <= {"-"}:
        return None
    if t.startswith("0"):
        return None
    # The length marks are written after the octave marks ("#4'._"), so they
    # have to come off first or the apostrophes look like they are not there.
    t = t.rstrip("_.-")
    up = len(t) - len(t.rstrip("'"))
    down = len(t) - len(t.rstrip(","))
    t = t.rstrip("'").rstrip(",")
    res = 0
    if t.startswith("b"):
        res, t = -1, t[1:]
    elif t.startswith("#"):
        res, t = 1, t[0:] and t[1:]
    digit = t[0] if t else ""
    if digit not in _SOLFA_TO_PC:
        return None
    pc = (_SOLFA_TO_PC[digit] + res) % 12
    octave = up - down
    # `tonic` is a pitch class, and degree 1 with no mark sits in the octave
    # that starts at middle C -- the same convention the writer uses.
    return 60 + tonic + octave * 12 + pc


def _notes_in_side(side: str, tonic: int) -> list[int]:
    out = []
    for tok in side.split():
        t = tok.strip()
        if not t or t.startswith("0") or set(t) <= {"-"}:
            continue
        if t.startswith("[") and t.endswith("]"):
            for m in t[1:-1].split("+"):
                p = _pitch_of(m, tonic)
                if p is not None:
                    out.append(p)
        else:
            p = _pitch_of(t, tonic)
            if p is not None:
                out.append(p)
    return out


def read_jianpu(path: Path) -> list[int]:
    """Every pitch the numbered notation prints, in order."""
    pitches: list[int] = []
    tonic = None
    for line in path.read_text().splitlines():
        if "1=" in line and tonic is None:
            try:
                name = line.split("1=")[1].split()[0].strip()
                tonic = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7,
                         "A": 9, "B": 11}.get(name[0])
                if name[1:2] in ("#", "b"):
                    tonic = (tonic + (1 if name[1] == "#" else -1)) % 12
            except Exception:
                pass
            continue
        if not line.startswith("|"):
            continue
        if tonic is None:
            raise SystemExit("no 1=<tonic> header, cannot read pitches")
        for cell in line.strip().strip("|").split("|"):
            cell = cell.strip()
            rh, lh = cell, ""
            if "(" in cell:
                i = cell.index("(")
                rh, lh = cell[:i], cell[i + 1:cell.rindex(")")] if ")" in cell else cell[i + 1:]
            pitches += _notes_in_side(rh, tonic)
            pitches += _notes_in_side(lh, tonic)
    return pitches


def score_pitches(pkl: Path) -> list[int]:
    arr = pickle.loads(pkl.read_bytes())
    return [n.pitch for n in list(arr.right) + list(arr.left)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pickles", nargs="+")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    bad = 0
    for p in a.pickles:
        pkl = Path(p)
        jp = pkl.with_suffix("").with_suffix(".jianpu.txt")
        if not jp.exists():
            print(f"skip {pkl}: no {jp.name}", file=sys.stderr)
            continue
        printed = sorted(read_jianpu(jp))
        scored = sorted(score_pitches(pkl))
        same = printed == scored
        # Sounding notes only: a rest token must never add a pitch, so a match
        # here proves the padding was rests and not invented music.
        row = {"arm": str(pkl.parent.name) + "/" + pkl.stem,
               "score_notes": len(scored), "printed_notes": len(printed),
               "identical": same}
        if not same:
            extra = sorted(set(printed) - set(scored))
            missing = sorted(set(scored) - set(printed))
            row["printed_not_scored"] = extra
            row["scored_not_printed"] = missing
            bad += 1
        if a.json:
            print(json.dumps(row, ensure_ascii=False))
        else:
            print(f"{row['arm']:<34} score {row['score_notes']:>3}  "
                  f"printed {row['printed_notes']:>3}  "
                  f"{'identical' if same else 'MISMATCH  extra=%s missing=%s' % (row.get('printed_not_scored'), row.get('scored_not_printed'))}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
