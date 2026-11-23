"""Source separation wrapper around Demucs (MIT).

We do not fork Demucs: we only call `apply_model` with a locally stored
checkpoint, and hand the stems back as plain numpy arrays keyed by stem name.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)

DEFAULT_REPO = Path(__file__).resolve().parents[2] / "models" / "demucs_repo"


def _pick_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(name: str = "htdemucs", repo: Path | None = None):
    from demucs.pretrained import get_model

    repo = Path(repo) if repo else DEFAULT_REPO
    if not repo.is_dir():
        raise FileNotFoundError(
            f"demucs weights repo not found: {repo}\n"
            "put <sig>-<hash>.th + <name>.yaml there (see README) or pass --demucs-repo")
    return get_model(name, repo=repo)


@torch.no_grad()
def separate(
    audio: np.ndarray,
    model_name: str = "htdemucs",
    repo: Path | None = None,
    device: str | None = None,
    shifts: int = 0,
    overlap: float = 0.25,
) -> dict[str, np.ndarray]:
    """audio: float32 [samples, channels] -> {stem: float32 [samples, channels]}."""
    from demucs.apply import apply_model

    dev = _pick_device(device)
    model = load_model(model_name, repo).to(dev)
    model.eval()

    mix = torch.from_numpy(audio.T).float().to(dev)  # [channels, samples]
    if mix.shape[0] == 1:
        mix = mix.repeat(2, 1)
    ref = mix.mean(0)
    mix = (mix - ref.mean()) / (ref.std() + 1e-8)

    log.info("separating with %s on %s ...", model_name, dev)
    out = apply_model(model, mix[None], device=dev, shifts=shifts,
                      split=True, overlap=overlap, progress=True)[0]
    out = out * ref.std() + ref.mean()
    out = out.clamp(-1, 1).cpu().numpy()

    stems = {name: out[i].T.copy() for i, name in enumerate(model.sources)}
    if dev == "mps":
        torch.mps.empty_cache()
    return stems
