"""Editable intermediate model: color groups and geometric parts."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Literal

Role = Literal["head", "ring", "points", "center", "tail_stripe", "unknown"]

MAX_COUNT = 200
_COUNT_PARAMS = ("count", "tails_count", "tail_index", "stripe_index")
_LIST_PARAMS = ("stripes", "stripes_bottom")


def check_params(params: dict) -> None:
    """Raise ValueError for params that would hang or break the builders: counts must be
    ints in 1..MAX_COUNT (indices 0..MAX_COUNT), numbers must be finite, stripe lists short."""
    for k, v in params.items():
        if k in _COUNT_PARAMS:
            lo = 0 if k.endswith("index") else 1
            if (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                    or int(v) != v or not lo <= v <= MAX_COUNT):
                raise ValueError(f"param {k} out of range")
        elif k in _LIST_PARAMS:
            if isinstance(v, str) and v.count(",") >= MAX_COUNT:
                raise ValueError(f"param {k} too long")
        elif isinstance(v, float) and not math.isfinite(v):
            raise ValueError(f"param {k} not finite")


@dataclass
class ColorGroup:
    id: int
    rgb: tuple[int, int, int]
    pattern: bool
    texture: float
    area: float
    category: int | None
    confidence: float


@dataclass
class Part:
    id: str
    role: Role
    params: dict[str, float | int | str]
    color_group: int
    z: int
    confidence: float


@dataclass
class RosetteModel:
    parts: list[Part] = field(default_factory=list)
    groups: list[ColorGroup] = field(default_factory=list)
    rose: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        for g in d["groups"]:
            g["rgb"] = list(g["rgb"])
        return d

    @classmethod
    def from_json(cls, d: dict) -> "RosetteModel":
        groups = [ColorGroup(**{**g, "rgb": tuple(g["rgb"])}) for g in d.get("groups", [])]
        parts = [Part(**p) for p in d.get("parts", [])]
        for q in parts:
            check_params(q.params)
        return cls(parts=parts, groups=groups, rose=bool(d.get("rose", False)),
                   warnings=list(d.get("warnings", [])))
