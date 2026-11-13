"""Perception settings from the environment. Never imports torch."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

DEVICES = ("auto", "cpu", "cuda")
MODELS = ("small", "tiny", "off")
READER_MODES = ("on", "off", "auto")
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


@dataclass(frozen=True)
class Settings:
    device: str
    model: str
    models_dir: Path
    reader: str = "auto"
    reader_port: int = 8089
    reader_threads: int = 14


def _choice(env: Mapping[str, str], name: str, allowed: tuple[str, ...], default: str) -> str:
    value = env.get(name, default).strip().lower() or default
    if value not in allowed:
        raise ValueError(f"{name}={value!r} is invalid; allowed values: {', '.join(allowed)}")
    return value


def _int(env: Mapping[str, str], name: str, default: int, low: int = 1, high: int | None = None) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is invalid; expected a whole number") from None
    if value < low or (high is not None and value > high):
        span = f"{low} to {high}" if high is not None else f"at least {low}"
        raise ValueError(f"{name}={raw!r} is invalid; expected a whole number {span}")
    return value


def load(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    raw_dir = env.get("ROSETTE_MODELS_DIR", "").strip()
    return Settings(
        device=_choice(env, "ROSETTE_DEVICE", DEVICES, "auto"),
        model=_choice(env, "ROSETTE_SAM2", MODELS, "small"),
        models_dir=Path(raw_dir) if raw_dir else DEFAULT_MODELS_DIR,
        reader=_choice(env, "ROSETTE_READER", READER_MODES, "auto"),
        reader_port=_int(env, "ROSETTE_READER_PORT", 8089, 1024, 65535),
        reader_threads=_int(env, "ROSETTE_READER_THREADS", 14),
    )
