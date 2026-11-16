from __future__ import annotations

import math

from ..shapes import Shape, fmt
from .common import TOOTH, polar, scallop_edge

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
    outer = scallop_edge(cx, cy, r_outer)  # scallops, as the designer draws a pleated edge
    m = PINKED_RING_INNER_POINTS
    inner = [polar(cx, cy, r_inner, 360.0 * i / m) for i in range(m)]
    return Shape("path", {"d": _loop_d(outer) + _loop_d(inner), "fill-rule": "evenodd"}, None)
