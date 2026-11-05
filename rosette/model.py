"""Editable intermediate model: color groups and geometric parts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Role = Literal["head", "ring", "points", "center", "tail_stripe", "unknown"]


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
        return cls(parts=parts, groups=groups, rose=bool(d.get("rose", False)),
                   warnings=list(d.get("warnings", [])))
