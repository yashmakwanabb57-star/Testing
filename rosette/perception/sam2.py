"""Offline loading of SAM 2.1 (transformers Sam2Model). No network at run time."""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .settings import Settings

REPOS = {"small": "facebook/sam2.1-hiera-small", "tiny": "facebook/sam2.1-hiera-tiny"}


class MissingModel(Exception):
    pass


@dataclass(frozen=True)
class Loaded:
    model: Any
    processor: Any
    device: str
    model_id: str


def local_dir(models_dir: Path, size: str) -> Path:
    return Path(models_dir) / REPOS[size].replace("/", "__")


@functools.lru_cache(maxsize=None)
def _load(models_dir: Path, size: str, device: str) -> Loaded:
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import Sam2Model, Sam2Processor

    path = local_dir(models_dir, size)
    model = Sam2Model.from_pretrained(path, local_files_only=True)
    processor = Sam2Processor.from_pretrained(path, local_files_only=True)
    model.to(device).eval()
    return Loaded(model, processor, device, REPOS[size])


def load(settings: Settings) -> Loaded:
    size = settings.model
    if size not in REPOS:
        raise ValueError(f"ROSETTE_SAM2={size!r} cannot be loaded; allowed values: small, tiny")
    path = local_dir(settings.models_dir, size)
    if not (path / "model.safetensors").exists():
        raise MissingModel(
            f"SAM 2.1 {size} weights not found in {path}. "
            f"Fetch them once with: .venv/Scripts/python.exe tools/fetch_models.py --model {size}"
        )
    os.environ["HF_HUB_OFFLINE"] = "1"
    from .device import pick

    device, _reason = pick(settings.device)
    return _load(Path(settings.models_dir), size, device)
