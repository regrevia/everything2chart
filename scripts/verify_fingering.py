#!/usr/bin/env python3
"""Check that a score's finger numbers match the arrangement behind it.

The finger numbers in a staff file are written by music21, and the ones in the
arrangement are assigned by the cost model.  If those two ever drift apart the
score still looks plausible -- a wrong digit on one note is invisible in a
picture -- so the check has to be mechanical.

Usage::

    python tools/verify_fingering.py out/mcp-015/full1-easy.arr.pkl --mode all

It prints the same dict the pipeline writes to ``report.json`` as
``fingering_check``: how many numbers the arrangement holds, how many reached
the staff file, how many strikes are marked, and how many of those are not
monotone in their hand's direction.  ``agrees`` is true only when the counts
match and nothing is misordered.

Exit code is 0 when they agree, 1 when they do not, so this can gate a release.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pianochart import fingering  # noqa: E402


def _staff_path(pkl: Path) -> Path:
    """The staff file the run that wrote this pickle also wrote.

    A directory accumulates staff files from earlier runs with other stems.  The
    stem cannot be trusted on its own (an early run leaves a bare ``<stem>-tier``
    file that a later one does not overwrite), and guessing between them would
    silently check the wrong file -- the exact false pass this tool exists to
    prevent.  So take the newest, and always print which one was used.
    """
    # The pickle and the staff file of one run share their stem, so an
    # explicit pairing beats a timestamp: ``--preset all`` writes three
    # pickles and three staff files a second apart, and the newest one is
    # whichever tier happened to finish last, not this one.
    beside = sorted(pkl.parent.glob("*.musicxml"))
    if not beside:
        raise SystemExit(f"no .musicxml beside {pkl}")
    own = pkl.name.split(".arr.")[0]
    same = [c for c in beside if c.name.split(".musicxml")[0] == own]
    if len(same) == 1:
        return same[0]
    candidates = same or beside
    return max(candidates, key=lambda c: c.stat().st_mtime)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arrangement", type=Path, help="the .arr.pkl a run wrote")
    ap.add_argument("--mode", default="key", choices=("key", "all", "none"),
                    help="fingering mode to re-check (default key)")
    ap.add_argument("--musicxml", type=Path, help="override the staff file")
    args = ap.parse_args()

    arr = pickle.loads(args.arrangement.read_bytes())
    staff = args.musicxml or _staff_path(args.arrangement)
    text = staff.read_text(encoding="utf-8")

    # The pickle carries whatever numbers the run wrote.  Clearing them first is
    # what makes this a check rather than a read-back of its own output.
    for events in (arr.right, arr.left):
        for e in events:
            e.finger = 0
    fingering.annotate(arr, args.mode)

    result = fingering.verify_export(arr, text, args.mode)
    result["arrangement"] = str(args.arrangement)
    result["staff_file"] = str(staff)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["agrees"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
