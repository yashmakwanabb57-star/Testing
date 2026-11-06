from __future__ import annotations

import math

from ..shapes import Shape
from .common import TOOTH, Pt, pinked_edge, points_attr

CX = 300.0


def tails(top_y: float, bottom_y: float, count: int, top_width: float, bottom_width: float,
          spread_deg: float, stripes: list[float], tooth: float = TOOTH,
          stripes_bottom: list[float] | None = None,
          tip_y: float | list[float] | None = None) -> list[list[Shape]]:
    """Tail axes fan out from (300, top_y); tail 0 is leftmost. Tails are not rotated: all bottoms lie on y=bottom_y. Each stripe is a
    closed polygon whose bottom edge is pinked (tips on the trapezoid bottom, valleys
    `tooth` above it); neighbouring stripes share their straight side edges.
    `stripes` are the stripe shares across the top edge, `stripes_bottom` (same
    count; default `stripes`) across the bottom edge: a ribbon of constant width on
    a widening tail takes a smaller share at the bottom.

    `tip_y` (one value, or one per tail; above top_y) closes each tail's top in a
    point on its axis at that height instead of the flat edge at top_y (H5): every
    stripe runs from its top corners at top_y (full width) up to the shared apex,
    so the stripes stay gapless and the designer's pointed tail top is kept."""
    def cumulative(fr: list[float]) -> list[float]:
        total = sum(fr)
        out = [0.0]
        for f in fr:
            out.append(out[-1] + f / total)
        return out

    cum = cumulative(stripes)
    cum_b = cumulative(stripes_bottom) if stripes_bottom else cum
    if len(cum_b) != len(cum):
        raise ValueError("stripes and stripes_bottom differ in length")
    tips = tip_y if isinstance(tip_y, list) else [tip_y] * count
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
            tl, tr = lerp(cum[i], ax_top, top_width, top_y), lerp(cum[i + 1], ax_top, top_width, top_y)
            bl = lerp(cum_b[i], ax_bot, bottom_width, bottom_y)
            br = lerp(cum_b[i + 1], ax_bot, bottom_width, bottom_y)
            top = [tl, tr]
            if tips[k] is not None and tips[k] < top_y:
                top = [tl, (ax_top + (tips[k] - top_y) * slope, tips[k]), tr]
            pts = top + pinked_edge(br, bl, tooth, (0.0, -1.0))
            row.append(Shape("polygon", {"points": points_attr(pts)}, None))
        out.append(row)
    return out
