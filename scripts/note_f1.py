"""Score an arrangement against known notes: precision, recall, F1.

The number competitors quote is note F1 under an onset tolerance.  This
computes it the same way so the claim is comparable, and also reports the
pitch-class-only figure, which is what the chroma metric really sees.

    PYTHONPATH=src .venv/bin/python scripts/note_f1.py \
        bench/clip1.truth.json out/bench1/clip1-easy.arr.pkl --tolerance 0.05

A shown note is written in the tier's key, so its pitch is shifted from the
recorded one; `stats['transpose']` is subtracted before matching.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def _truth(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["notes"]


def _pred(arr, transpose: int, role: str | None) -> list[dict]:
    out: list[dict] = []
    hands = []
    if role in (None, "all", "rh"):
        hands.append(("rh", arr.right))
    if role in (None, "all", "lh"):
        hands.append(("lh", arr.left))
    for name, notes in hands:
        for e in notes:
            out.append({"pitch": int(e.pitch) - transpose,
                        "start": float(e.start), "dur": float(e.dur), "role": name})
    return out


def match(ref: list[dict], hyp: list[dict], tol: float) -> dict:
    """Greedy one-to-one on exact pitch, nearest onset first."""
    used = [False] * len(hyp)
    hit = 0
    for r in sorted(ref, key=lambda x: x["start"]):
        best, best_d = -1, tol
        for i, h in enumerate(hyp):
            if used[i] or h["pitch"] != r["pitch"]:
                continue
            d = abs(h["start"] - r["start"])
            if d <= best_d:
                best, best_d = i, d
        if best >= 0:
            used[best] = True
            hit += 1
    p = hit / len(hyp) if hyp else 0.0
    rc = hit / len(ref) if ref else 0.0
    return {"precision": round(p, 4), "recall": round(rc, 4),
            "f1": round(2 * p * rc / (p + rc), 4) if p + rc else 0.0,
            "matched": hit, "ref": len(ref), "hyp": len(hyp)}


def match_pc(ref: list[dict], hyp: list[dict], tol: float) -> dict:
    """Pitch class only: octave errors stop counting, harmony errors still do."""
    ref2 = [dict(r, pitch=r["pitch"] % 12) for r in ref]
    hyp2 = [dict(h, pitch=h["pitch"] % 12) for h in hyp]
    return match(ref2, hyp2, tol)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("truth", help="*.truth.json written by bench_synth.py")
    ap.add_argument("arr", help="*.arr.pkl produced by the pipeline")
    ap.add_argument("--tolerance", type=float, default=0.05, help="onset window, seconds")
    ap.add_argument("--role", default="all", choices=["all", "rh", "lh"])
    args = ap.parse_args()

    arr = pickle.loads(Path(args.arr).read_bytes())
    tr = int(arr.meta.get("transpose", 0))
    ref = _truth(args.truth)
    if args.role != "all":
        ref = [r for r in ref if r.get("role") == args.role]
    hyp = _pred(arr, tr, args.role)

    out = {"transpose": tr, "tolerance": args.tolerance, "role": args.role,
           "note": match(ref, hyp, args.tolerance),
           "pitch_class": match_pc(ref, hyp, args.tolerance)}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
