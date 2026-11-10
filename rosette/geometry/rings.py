from __future__ import annotations

import math

from ..shapes import Shape, fmt
from .common import TOOTH, polar

# Points on the inner circle of a pinked ring (a straight even-odd path).
PINKED_RING_INNER_POINTS = 96


def _circle_sub(cx: float, cy: float, r: float) -> str:
    a, b = fmt(r), fmt(2 * r)
    return (f"M{fmt(cx + r)},{fmt(cy)}a{a},{a},0,1,0,-{b},0"
            f"a{a},{a},0,1,0,{b},0z")


def ring(cx: float, cy: float, r_outer: float, r_inner: float) -> Shape:
    d = _circle_sub(cx, cy, r_outer) + _circle_sub(cx, cy, r_inner)
    return Shape("path", {"d": d, "fill-rule": "evenodd"}, None)


def _loop_d(pts) -> str:
    return "M" + " ".join(f"{fmt(x)},{fmt(y)}" for x, y in pts) + "z"


def pinked_ring(cx: float, cy: float, r_outer: float, r_inner: float, tooth: float = TOOTH) -> Shape:
    """Annulus whose outer edge is pinked (tips on r_outer, valleys r_outer - tooth): a
    pleated ring (H11). A straight even-odd path, so the occlusion cuts can read it."""
    n = max(3, round(2 * math.pi * r_outer / tooth))
    outer = []
    for i in range(n):
        outer.append(polar(cx, cy, r_outer, 360.0 * i / n - 90))
        outer.append(polar(cx, cy, r_outer - tooth, 360.0 * (i + 0.5) / n - 90))
    m = PINKED_RING_INNER_POINTS
    inner = [polar(cx, cy, r_inner, 360.0 * i / m) for i in range(m)]
    return Shape("path", {"d": _loop_d(outer) + _loop_d(inner), "fill-rule": "evenodd"}, None)
