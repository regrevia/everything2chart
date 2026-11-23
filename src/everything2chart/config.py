"""Tiny config layer: where the repo sits, how presets are loaded."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
PRESETS_PATH = CONFIG_DIR / "presets.yaml"
CACHE_DIR = REPO_ROOT / "cache"
MODELS_DIR = REPO_ROOT / "models"
WORK_DIR = REPO_ROOT / "work"
OUT_DIR = REPO_ROOT / "out"


@lru_cache(maxsize=4)
def load_presets(path: str | None = None) -> dict:
    with open(path or PRESETS_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # Entries whose name starts with "_" exist only to hold a YAML anchor the
    # two or three real tiers merge from; they are not tiers themselves.
    return {k: v for k, v in data.items() if not str(k).startswith("_")}


def preset(name: str) -> dict:
    presets = load_presets()
    if name not in presets:
        raise KeyError(f"unknown preset {name!r}, have {list(presets)}")
    return presets[name]
