from __future__ import annotations

import math

from ..shapes import Shape
from .common import TOOTH, Pt, pinked_edge, points_attr

CX = 300.0


def tails(top_y: float, bottom_y: float, count: int, top_width: float, bottom_width: float,
          spread_deg: float, stripes: list[float], tooth: float = TOOTH) -> list[list[Shape]]:
    """Tails fan out from the pivot (300, top_y); tail 0 is leftmost. Each stripe is a
    closed polygon whose bottom edge is pinked (tips on the trapezoid bottom, valleys
    `tooth` above it); neighbouring stripes share their straight side edges."""
    total = sum(stripes)
    cum = [0.0]
    for f in stripes:
        cum.append(cum[-1] + f / total)
    out: list[list[Shape]] = []
    for k in range(count):
        ang = 0.0 if count == 1 else -spread_deg / 2 + k * spread_deg / (count - 1)
        a = math.radians(ang)
        ca, sa = math.cos(a), math.sin(a)

        def rot(p: Pt) -> Pt:
            dx, dy = p[0] - CX, p[1] - top_y
            return (CX + dx * ca - dy * sa, top_y + dx * sa + dy * ca)

        def at(t: float, y: float, w: float) -> Pt:
            return (CX - w / 2 + w * t, y)

        row: list[Shape] = []
        for i in range(len(stripes)):
            t0, t1 = cum[i], cum[i + 1]
            tl, tr = at(t0, top_y, top_width), at(t1, top_y, top_width)
            bl, br = at(t0, bottom_y, bottom_width), at(t1, bottom_y, bottom_width)
            bottom = pinked_edge(br, bl, tooth, (0.0, -1.0))
            pts = [rot(p) for p in [tl, tr] + bottom]
            row.append(Shape("polygon", {"points": points_attr(pts)}, None))
        out.append(row)
    return out
