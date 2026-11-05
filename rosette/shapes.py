"""Shape dataclass shared by geometry builders and the SVG writer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Shape:
    tag: Literal["path", "polygon", "polyline", "circle", "ellipse"]
    attrs: dict[str, str]
    category: Literal[1, 2, 3, 4] | None
    pattern: bool = False
    role: str = ""
    z: int = 0
    kind: Literal["color", "outline", "rose", "circel"] = "color"


def fmt(x: float) -> str:
    """Format a number with at most 3 decimals, trailing zeros stripped."""
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s
