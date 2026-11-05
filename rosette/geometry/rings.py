from __future__ import annotations

from ..shapes import Shape, fmt


def _circle_sub(cx: float, cy: float, r: float) -> str:
    a, b = fmt(r), fmt(2 * r)
    return (f"M{fmt(cx + r)},{fmt(cy)}a{a},{a},0,1,0,-{b},0"
            f"a{a},{a},0,1,0,{b},0z")


def ring(cx: float, cy: float, r_outer: float, r_inner: float) -> Shape:
    d = _circle_sub(cx, cy, r_outer) + _circle_sub(cx, cy, r_inner)
    return Shape("path", {"d": d, "fill-rule": "evenodd"}, None)
