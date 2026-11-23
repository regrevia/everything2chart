"""Audio I/O: everything goes through ffmpeg, so any container/codec works."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def ffmpeg_to_wav(src: str | Path, dst: str | Path, sr: int = 44100, channels: int = 2) -> Path:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-ac", str(channels), "-ar", str(sr), "-c:a", "pcm_s16le", str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


def load_mono_stereo(path: str | Path, sr: int = 44100) -> tuple[np.ndarray, int]:
    """Return (audio[samples, channels], sr), float32 in [-1, 1]."""
    path = Path(path)
    if path.suffix.lower() not in {".wav", ".flac", ".aiff"}:
        wav = path.with_suffix(".wav")
        if not wav.exists() or wav.stat().st_mtime < path.stat().st_mtime:
            ffmpeg_to_wav(path, wav, sr=sr, channels=2)
        path = wav
    y, real_sr = sf.read(str(path), always_2d=True, dtype="float32")
    if real_sr != sr:
        # resample through ffmpeg, keeps the rest of the pipeline simple
        tmp = path.with_name(path.stem + f"_{sr}.wav")
        ffmpeg_to_wav(path, tmp, sr=sr, channels=y.shape[1])
        y, real_sr = sf.read(str(tmp), always_2d=True, dtype="float32")
    return y, real_sr


def save_wav(path: str | Path, y: np.ndarray, sr: int = 44100) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(y, -1.0, 1.0), sr, subtype="PCM_16")
    return path
