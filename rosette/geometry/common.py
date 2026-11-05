"""Shared helpers: polar math, pinked (zigzag) edges, point formatting."""
from __future__ import annotations

import math

from ..shapes import fmt

TOOTH = 5.383

Pt = tuple[float, float]


def polar(cx: float, cy: float, r: float, deg: float) -> Pt:
    """Angle measured from +x, increasing clockwise on screen (SVG y points down)."""
    a = math.radians(deg)
    return (cx + r * math.cos(a), cy + r * math.sin(a))


def points_attr(pts: list[Pt]) -> str:
    return " ".join(f"{fmt(x)},{fmt(y)}" for x, y in pts)


def pinked_edge(p0: Pt, p1: Pt, tooth: float, inward: Pt) -> list[Pt]:
    """Zigzag from p0 to p1 (both included): tips on the line, valleys `tooth`
    deep toward the `inward` vector. Tooth width is adjusted to fit evenly."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    n = max(1, round(length / tooth))
    pts: list[Pt] = [p0]
    for i in range(n):
        t_mid = (i + 0.5) / n
        t_end = (i + 1) / n
        pts.append((p0[0] + dx * t_mid + inward[0] * tooth,
                    p0[1] + dy * t_mid + inward[1] * tooth))
        pts.append((p0[0] + dx * t_end, p0[1] + dy * t_end))
    return pts


def area(pts: list[Pt]) -> float:
    s = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        s += x0 * y1 - x1 * y0
    return abs(s) / 2
