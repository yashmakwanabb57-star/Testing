"""Shared helpers: polar math, pinked (zigzag) edges, point formatting."""
from __future__ import annotations

import math

from ..shapes import fmt

TOOTH = 5.383
# A pleated edge is a smooth wave, as the designer draws it (A10-15B close up: rounded crests and dips,
# about 18 units apart and 4 deep on the head; the owner's check): crests SCALLOP_PERIOD apart, SCALLOP_DEPTH
# from crest to dip (canvas units), a cosine between.
SCALLOP_PERIOD = 3.4 * TOOTH
SCALLOP_DEPTH = 0.75 * TOOTH
SCALLOP_POINTS = 12  # points per wave


def scallop_edge(cx: float, cy: float, r: float, period: float = SCALLOP_PERIOD,
                 depth: float = SCALLOP_DEPTH) -> list:
    """Points of a wavy circle: crests reaching r, dips at r - depth, a smooth cosine between."""
    n = max(3, round(2 * math.pi * r / period))
    pts = []
    for i in range(n):
        for j in range(SCALLOP_POINTS):
            u = j / SCALLOP_POINTS
            pts.append(polar(cx, cy, r - depth / 2 + depth / 2 * math.cos(2 * math.pi * u), 360.0 * (i + u) / n - 90))
    return pts

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
