from __future__ import annotations

import math
from typing import Literal

from ..shapes import Shape
from .common import Pt, points_attr, polar


def _to_world(cx, cy, theta_deg, radial, lateral) -> Pt:
    """radial = distance along the point's axis, lateral = offset perpendicular."""
    a = math.radians(theta_deg)
    ca, sa = math.cos(a), math.sin(a)
    return (cx + radial * ca - lateral * sa, cy + radial * sa + lateral * ca)


def _outline(kind: str, r_in: float, r_out: float, w: float,
             inner_ratio: float = 1.0) -> list[tuple[float, float]]:
    """(radial, lateral) vertices, symmetric about the axis. A bow is a ribbon loop:
    a block w wide at its outer end and inner_ratio * w at its inner end."""
    mid = (r_in + r_out) / 2
    h = w / 2
    if kind == "diamond":
        return [(r_in, 0), (mid, h), (r_out, 0), (mid, -h)]
    if kind == "bow":
        hi = h * inner_ratio
        return [(r_in, -hi), (r_in, hi), (r_out, h), (r_out, -h)]
    if kind == "petal":
        n = 8
        side = [(r_in + (r_out - r_in) * t / n, h * math.sin(math.pi * t / n))
                for t in range(1, n)]
        return [(r_in, 0)] + side + [(r_out, 0)] + [(r, -l) for r, l in reversed(side)]
    raise ValueError(f"unknown point kind {kind!r}")


def points_ring(cx: float, cy: float, kind: Literal["star", "diamond", "bow", "petal"],
                count: int, r_inner: float, r_outer: float,
                rotation_deg: float, width_ratio: float, inner_ratio: float = 1.0) -> list[Shape]:
    """Angles run clockwise on screen from +x; point k is centered on
    rotation_deg + k*360/count. Width = width_ratio * (r_outer - r_inner); a bow's
    inner end is inner_ratio times as wide (default 1: a block of constant width)."""
    if kind == "star":
        pts = []
        for k in range(count):
            pts.append(polar(cx, cy, r_outer, rotation_deg + 360.0 * k / count))
            pts.append(polar(cx, cy, r_inner, rotation_deg + 360.0 * (k + 0.5) / count))
        return [Shape("polygon", {"points": points_attr(pts)}, None)]
    w = width_ratio * (r_outer - r_inner)
    shapes = []
    for k in range(count):
        th = rotation_deg + 360.0 * k / count
        pts = [_to_world(cx, cy, th, r, l) for r, l in _outline(kind, r_inner, r_outer, w, inner_ratio)]
        shapes.append(Shape("polygon", {"points": points_attr(pts)}, None))
    return shapes
