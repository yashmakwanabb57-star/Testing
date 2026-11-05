from __future__ import annotations

import math

from ..shapes import Shape
from .common import TOOTH, Pt, pinked_edge, points_attr

CX = 300.0


def tails(top_y: float, bottom_y: float, count: int, top_width: float, bottom_width: float,
          spread_deg: float, stripes: list[float], tooth: float = TOOTH) -> list[list[Shape]]:
    """Tail axes fan out from (300, top_y); tail 0 is leftmost. Tails are not rotated: all bottoms lie on y=bottom_y. Each stripe is a
    closed polygon whose bottom edge is pinked (tips on the trapezoid bottom, valleys
    `tooth` above it); neighbouring stripes share their straight side edges."""
    total = sum(stripes)
    cum = [0.0]
    for f in stripes:
        cum.append(cum[-1] + f / total)
    out: list[list[Shape]] = []
    for k in range(count):
        ang = 0.0 if count == 1 else -spread_deg / 2 + k * spread_deg / (count - 1)
        slope = math.tan(math.radians(ang))  # negative for the leftmost tail
        ax_top = CX + 0.0 * slope
        ax_bot = CX + (bottom_y - top_y) * slope

        def lerp(t: float, ax: float, w: float, y: float) -> Pt:
            return (ax - w / 2 + w * t, y)

        row: list[Shape] = []
        for i in range(len(stripes)):
            t0, t1 = cum[i], cum[i + 1]
            tl, tr = lerp(t0, ax_top, top_width, top_y), lerp(t1, ax_top, top_width, top_y)
            bl, br = lerp(t0, ax_bot, bottom_width, bottom_y), lerp(t1, ax_bot, bottom_width, bottom_y)
            pts = [tl, tr] + pinked_edge(br, bl, tooth, (0.0, -1.0))
            row.append(Shape("polygon", {"points": points_attr(pts)}, None))
        out.append(row)
    return out
