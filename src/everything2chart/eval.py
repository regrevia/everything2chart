"""Objective checks for an arrangement (M0 level, no human ear needed).

Two numbers matter:
  * chroma similarity  — does the piano reduction still carry the same harmony
  * melody contour     — is the right hand still the tune (octave/step errors)

Both are cheap and let us compare presets against the source mix.
"""

from __future__ import annotations

import numpy as np

from .types import Arrangement

SR = 22050


def synth(arr: Arrangement, sr: int = SR) -> np.ndarray:
    """Cheap additive synth of the arrangement, for comparison only."""
    n = int(arr.info.duration * sr) + sr
    out = np.zeros(n, dtype=np.float32)
    for e in arr.right + arr.left:
        f0 = 440.0 * 2 ** ((e.pitch - 69) / 12)
        start = int(e.start * sr)
        dur = int(max(e.dur, 0.05) * sr)
        if start >= n:
            continue
        t = np.arange(dur) / sr
        env = np.exp(-t * 3.0) * (1 - np.exp(-t * 200))
        sig = (np.sin(2 * np.pi * f0 * t) * 0.6
               + np.sin(4 * np.pi * f0 * t) * 0.25
               + np.sin(6 * np.pi * f0 * t) * 0.1)
        seg = (sig * env * (e.velocity / 127.0) * 0.25).astype(np.float32)
        end = min(start + dur, n)
        out[start:end] += seg[: end - start]
    return out


def _chroma(y: np.ndarray, sr: int, hop: int = 1024) -> np.ndarray:
    import librosa

    c = librosa.feature.chroma_cqt(y=librosa.util.normalize(y), sr=sr, hop_length=hop)
    return c


def compare(arr: Arrangement, source: np.ndarray, sr: int = SR,
            compensate_transpose: bool = True) -> dict:
    """source must be mono, 22.05 kHz is fine (it is resampled by caller)."""
    import librosa

    syn = synth(arr, sr)
    if sr != SR:
        syn = librosa.resample(syn, orig_sr=SR, target_sr=sr)
    c_ref = _chroma(source, sr)
    c_syn = _chroma(syn, sr)
    if compensate_transpose:
        # The page is written in another key on purpose, so the notes it prints
        # are not the notes that were recorded.  Chroma has no octave, so a
        # rotation undoes the transposition exactly: row i takes what the
        # recording calls i + transpose.  Comparing without this scores every
        # tier against the wrong key and punishes the easiest one hardest.
        tr = int(arr.meta.get("transpose", 0)) % 12
        if tr:
            c_syn = np.roll(c_syn, -tr, axis=0)
    k = min(c_ref.shape[1], c_syn.shape[1])
    a, b = c_ref[:, :k], c_syn[:, :k]
    num = (a * b).sum(axis=0)
    den = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-9
    cos = num / den
    active = np.linalg.norm(c_ref, axis=0) > np.percentile(np.linalg.norm(c_ref, axis=0), 20)
    return {
        "chroma_cos_mean": float(np.nanmean(cos[active])),
        "chroma_cos_p10": float(np.nanpercentile(cos[active], 10)),
        "windows": int(active.sum()),
    }


def melody_contour(arr: Arrangement, melody_pitch: np.ndarray | None = None) -> dict:
    """How conjunct is the right hand? Big share of steps <= 4 semitones is good."""
    if len(arr.right) < 3:
        return {"steps": 0, "conjunct_ratio": 0.0, "mean_abs_step": 0.0}
    ps = [n.pitch for n in arr.right]
    steps = np.abs((np.diff(ps) + 6) % 12 - 6)   # octave registrations are not leaps
    return {
        "steps": int(len(steps)),
        "conjunct_ratio": float((steps <= 4).mean()),
        "mean_abs_step": float(steps.mean()),
    }
