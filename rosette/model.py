"""Editable intermediate model: color groups and geometric parts."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Literal

Role = Literal["head", "ring", "points", "center", "tail_stripe", "unknown", "template"]

MAX_COUNT = 200
_COUNT_PARAMS = ("count", "tails_count", "tail_index", "stripe_index")
_LIST_PARAMS = ("stripes", "stripes_bottom")


_POSITIVE = ("r", "r_inner", "r_outer", "width_ratio", "top_width", "bottom_width", "edge_width")
_CANVAS = {"cx": 600, "cy": 850}  # viewBox 0 0 600 850


def _number(params: dict, k: str) -> float:
    v = params[k]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"param {k} must be a finite number")
    return float(v)


def check_params(params: dict) -> None:
    """Raise ValueError for params that would hang or break the builders: counts must be
    ints in 1..MAX_COUNT (indices 0..MAX_COUNT), numbers must be finite, stripe lists short.
    Geometry must not be degenerate: radii, width_ratio and tail widths > 0, r_inner <
    r_outer, top_y < bottom_y, and the center (cx, cy) on the 600x850 canvas. Every
    analyze output on the 76 reference photos passes these rules."""
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
    for k in _POSITIVE:
        if k in params and not _number(params, k) > 0:
            raise ValueError(f"param {k} must be above 0")
    for k, hi in _CANVAS.items():
        if k in params and not 0 <= _number(params, k) <= hi:
            raise ValueError(f"param {k} is off the canvas")
    if "r_inner" in params and "r_outer" in params and not params["r_inner"] < params["r_outer"]:
        raise ValueError("param r_inner must be below r_outer")
    if "top_y" in params and "bottom_y" in params and not _number(params, "top_y") < _number(params, "bottom_y"):
        raise ValueError("param top_y must be above bottom_y")


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
    # Parts kept in the model but not drawn: with a library head (role "template", H7)
    # the detected head parts wait here so the user can switch back to them, and so do the
    # other library candidates of the head card (pipeline.use_head).
    alternatives: list[Part] = field(default_factory=list)
    # Color groups only the parts in alternatives use (not shown while they are hidden).
    alternative_groups: list[ColorGroup] = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        for g in d["groups"] + d["alternative_groups"]:
            g["rgb"] = list(g["rgb"])
        return d

    @classmethod
    def from_json(cls, d: dict) -> "RosetteModel":
        groups = [ColorGroup(**{**g, "rgb": tuple(g["rgb"])}) for g in d.get("groups", [])]
        parts = [Part(**p) for p in d.get("parts", [])]
        alternatives = [Part(**p) for p in d.get("alternatives", [])]
        alt_groups = [ColorGroup(**{**g, "rgb": tuple(g["rgb"])}) for g in d.get("alternative_groups", [])]
        for q in parts + alternatives:
            check_params(q.params)
        return cls(parts=parts, groups=groups, rose=bool(d.get("rose", False)),
                   warnings=list(d.get("warnings", [])), alternatives=alternatives,
                   alternative_groups=alt_groups)
