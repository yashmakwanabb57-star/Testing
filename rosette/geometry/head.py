from __future__ import annotations

import math

from ..shapes import Shape
from .common import TOOTH, points_attr, polar


def pinked_circle(cx: float, cy: float, r: float, tooth: float = TOOTH) -> Shape:
    """Circle with a zigzag rim: tips on radius r, valleys at r - tooth."""
    n = max(3, round(2 * math.pi * r / tooth))
    pts = []
    for i in range(n):
        pts.append(polar(cx, cy, r, 360.0 * i / n - 90))
        pts.append(polar(cx, cy, r - tooth, 360.0 * (i + 0.5) / n - 90))
    return Shape("polygon", {"points": points_attr(pts)}, None)
