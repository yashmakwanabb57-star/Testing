"""Perception settings from the environment. Never imports torch."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

DEVICES = ("auto", "cpu", "cuda")
MODELS = ("small", "tiny", "off")
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


@dataclass(frozen=True)
class Settings:
    device: str
    model: str
    models_dir: Path


def _choice(env: Mapping[str, str], name: str, allowed: tuple[str, ...], default: str) -> str:
    value = env.get(name, default).strip().lower() or default
    if value not in allowed:
        raise ValueError(f"{name}={value!r} is invalid; allowed values: {', '.join(allowed)}")
    return value


def load(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    raw_dir = env.get("ROSETTE_MODELS_DIR", "").strip()
    return Settings(
        device=_choice(env, "ROSETTE_DEVICE", DEVICES, "auto"),
        model=_choice(env, "ROSETTE_SAM2", MODELS, "small"),
        models_dir=Path(raw_dir) if raw_dir else DEFAULT_MODELS_DIR,
    )
