"""Build a benchmark clip whose notes we know exactly, so accuracy can be scored.

Competitors quote note F1 against reference MIDI.  We cannot score against real
recordings — nobody hands out ground truth for them — so this generates one:
a deterministic piece with a melody, block chords and a bass line, rendered
through a piano-ish additive synth.  Ground truth is the note list that built
the audio, so precision and recall are exact rather than estimated.

    PYTHONPATH=src .venv/bin/python scripts/bench_synth.py bench/clip1 --seed 1

Writes `<stem>.wav` and `<stem>.truth.json`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

SR = 44100

# A natural minor; the arranger picks the printed key itself, and chroma scoring
# rolls the tier transposition back, so the source key only has to be musical.
SCALE = [0, 2, 3, 5, 7, 8, 10]


def _synth_note(f0: float, dur: float) -> np.ndarray:
    """Harmonic stack with a piano-ish decay; richer than eval.synth on purpose."""
    n = max(int(dur * SR), 64)
    t = np.arange(n) / SR
    env = np.exp(-t * 2.2) * (1 - np.exp(-t * 400)) + 0.02 * np.exp(-t * 0.8)
    sig = np.zeros(n)
    for k, amp in enumerate((1.0, 0.45, 0.28, 0.14, 0.08), start=1):
        if f0 * k > SR / 2.2:
            break
        sig += amp * np.sin(2 * np.pi * f0 * k * t) * np.exp(-t * 0.6 * k)
    return (sig * env).astype(np.float32)


def build(seed: int = 1, bars: int = 8, bpm: float = 96.0) -> tuple[np.ndarray, list[dict]]:
    rnd = random.Random(seed)
    spb = 60.0 / bpm
    eighth = spb / 2
    total = bars * 4 * spb
    n = int(total * SR) + SR
    mix = np.zeros(n, dtype=np.float32)
    truth: list[dict] = []

    def place(pitch: int, start: float, dur: float, vel: int) -> None:
        if pitch < 21 or pitch > 108:
            return
        f0 = 440.0 * 2 ** ((pitch - 69) / 12)
        seg = _synth_note(f0, min(dur * 1.6, total - start))
        i = int(start * SR)
        end = min(i + len(seg), n)
        mix[i:end] += seg[: end - i] * (vel / 127.0) * 0.22
        truth.append({"pitch": pitch, "start": round(start, 4),
                      "dur": round(dur, 4), "role": "rh" if pitch >= 60 else "lh",
                      "velocity": vel})

    # Melody: stepwise motion inside the scale, one note per eighth, some rests.
    degree = 5                       # start on the fifth above the tonic
    t = 0.0
    for _ in range(bars * 8):
        if rnd.random() < 0.12:      # a rest keeps the line from sounding like a scale
            t += eighth
            continue
        degree += rnd.choice((-1, -1, 0, 1, 1, 2, -2))
        degree = max(0, min(len(SCALE) * 2 - 1, degree))
        pitch = 69 + SCALE[degree % 7] + 12 * (degree // 7)   # A4 tonic
        pitch = max(60, min(79, pitch))                        # inside the RH window
        place(pitch, t, eighth * rnd.choice((1, 1, 2)), rnd.randint(70, 100))
        t += eighth

    # Left hand: block chords on beats 1 and 3, root doubled an octave down.
    for bar in range(bars):
        for beat in (0, 2):
            root_pc = rnd.choice((0, 0, 3, 5, 7, 8, 10))
            start = bar * 4 * spb + beat * spb
            root = 45 + root_pc            # A2 area
            third = 3 if root_pc in (0, 3, 8) else 4
            for pitch in (root - 12, root, root + third, root + 7):
                place(pitch, start, spb * rnd.choice((1.0, 1.5, 2.0)), rnd.randint(60, 85))

    mix = np.clip(mix, -1.0, 1.0)
    return mix, truth


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="output path without the suffix, e.g. bench/clip1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bars", type=int, default=8)
    ap.add_argument("--bpm", type=float, default=96.0)
    args = ap.parse_args()

    import soundfile as sf

    mix, truth = build(args.seed, args.bars, args.bpm)
    stem = Path(args.stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(stem.with_suffix(".wav")), mix, SR, subtype="PCM_16")
    stem.with_suffix(".truth.json").write_text(
        json.dumps({"sr": SR, "bpm": args.bpm, "seed": args.seed,
                    "duration": round(len(mix) / SR, 3), "notes": truth},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"wav": str(stem.with_suffix(".wav")),
                      "notes": len(truth),
                      "duration": round(len(mix) / SR, 2)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
