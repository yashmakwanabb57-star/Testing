from __future__ import annotations

import math
from typing import Literal

from ..shapes import Shape
from .common import Pt, points_attr, polar

# Lateral width profiles of the designer's points kinds (width / widest width at
# t = 0, 0.1, ..., 1 from the inner to the outer end), medians over the 79 designer
# SVGs (H4 mining, `.superpowers/sdd/2026-11-05-rosette-svg-generator/h4-report.md`):
# - bow = ribbon loop block: the 10 loop layers (8 files) whose base is visible at
#   the ring are 0.85 of their widest width there and widen to full width by 0.7 of
#   the length; the outer end is a shallow convex curve (END_SAG);
# - diamond = kite (41 layers, 22 files): straight sides from pointed ends to the
#   widest point at 0.4 of the length (median peak; quartiles 0.3-0.5). The median of
#   the profiles has rounded shoulders only because the peaks differ; each designer
#   kite is a straight-sided polygon;
# - petal (14 layers, 13 files): a wide inner end (mostly under the center), widest at
#   half the length, a straight taper to a pointed tip. The mined petals are not
#   consistent (width/spacing quartiles 0.32-0.98 at t=0.1), so the profile was kept only
#   because it beats the old sine lobe on the 14 photos whose model has a petal layer:
#   head IoU mean 0.7702 vs 0.7606 (better on 6, worse on 2: S40-15 -0.004, S58-17
#   -0.007, equal on 6), whole shape 0.8897 vs 0.8866, color 0.8291 vs 0.8265.
_T = tuple(i / 10 for i in range(11))
LOOP_PROFILE = (0.85, 0.85, 0.89, 0.92, 0.94, 0.96, 0.98, 1.0, 1.0, 1.0, 1.0)
KITE_PEAK = 0.4
KITE_PROFILE = tuple(round(t / KITE_PEAK if t <= KITE_PEAK else (1 - t) / (1 - KITE_PEAK), 4) for t in _T)
PETAL_PROFILE = (0.75, 0.75, 0.86, 0.96, 0.98, 1.0, 0.83, 0.60, 0.41, 0.23, 0.0)
# A loop's outer end is a shallow convex curve: r_outer is its radius on the axis and
# its corners lie END_SAG times a straight cut's sag (sqrt(r^2 + h^2) - r) further from
# the head center (designer loops: median 0.6 over 21 loop layers; an arc around the
# head center would be 0). Measured on the photo, r_outer reaches the corners, but
# drawing the corners there shortens the loops against the designer's, which reach the
# head rim (S5-15 color 0.910 -> 0.868); the axis convention is the pre-H4 one.
END_SAG = 0.6
# Points on the end curve.
ARC_POINTS = 9
# Edge strips of a two-tone loop (designer: 14 strip layers on loops in 13 files): the
# strip's center lies EDGE_OUTSIDE strip widths outside the loop's side edge (median
# 0.19, quartiles 0.00-0.34).
EDGE_OUTSIDE = 0.19


def _to_world(cx, cy, theta_deg, radial, lateral) -> Pt:
    """radial = distance along the point's axis, lateral = offset perpendicular."""
    a = math.radians(theta_deg)
    ca, sa = math.cos(a), math.sin(a)
    return (cx + radial * ca - lateral * sa, cy + radial * sa + lateral * ca)


def _profile(kind: str, inner_ratio: float | None) -> tuple[float, ...]:
    if kind == "bow":
        if inner_ratio is None:
            return LOOP_PROFILE
        # an edited base width: the loop profile rescaled between it and full width
        base = LOOP_PROFILE[0]
        return tuple(inner_ratio + (p - base) * (1 - inner_ratio) / (1 - base) for p in LOOP_PROFILE)
    if kind == "diamond":
        return KITE_PROFILE
    if kind == "petal":
        return PETAL_PROFILE
    raise ValueError(f"unknown point kind {kind!r}")


def _side(kind: str, r_in: float, r_out: float, h: float, inner_ratio: float | None,
          offset: float = 0.0) -> list[tuple[float, float]]:
    """(radial, lateral) points of one side (lateral >= 0) from the inner to the outer
    end, the profile's half-width plus `offset`. A bow's side stops at the corner of
    its convex outer end (`_end_radial`, END_SAG)."""
    prof = _profile(kind, inner_ratio)
    pts = [(r_in + (r_out - r_in) * t, max(0.0, h * p + offset)) for t, p in zip(_T, prof)]
    if kind != "bow":
        return pts
    lat = pts[-1][1]
    corner = _end_radial(r_out, h, lat)
    return [q for q in pts[:-1] if q[0] < corner] + [(corner, lat)]


def _end_radial(r_out: float, h: float, l: float) -> float:
    """Radial coordinate of a loop's outer end at lateral offset l (half-width h)."""
    corner = r_out + END_SAG * (math.hypot(r_out, h) - r_out)
    corner_radial = math.sqrt(max(corner * corner - h * h, 0.0))
    x = min(abs(l) / h, 1.0) if h > 0 else 0.0
    return r_out + (corner_radial - r_out) * x * x


def _outline(kind: str, r_in: float, r_out: float, w: float,
             inner_ratio: float | None = None) -> list[tuple[float, float]]:
    """(radial, lateral) vertices, symmetric about the axis: out along one side, back
    along the other (a bow crosses its outer end on an arc around the head center)."""
    side = _side(kind, r_in, r_out, w / 2, inner_ratio)
    out = [(r, -l) for r, l in side]
    if kind == "bow":
        h = side[-1][1]
        n = ARC_POINTS - 1
        out += [(_end_radial(r_out, h, l), l) for l in (-h + 2 * h * i / n for i in range(1, n))]
    back = [(r, l) for r, l in reversed(side)]
    return out + [q for q in back if q[1] > 0]


def _edge_strips(kind: str, r_in: float, r_out: float, w: float, inner_ratio: float | None,
                 e: float) -> list[list[tuple[float, float]]]:
    """(radial, lateral) outlines of the two strips along a lobe's sides, e wide."""
    lo = _side(kind, r_in, r_out, w / 2, inner_ratio, -(0.5 - EDGE_OUTSIDE) * e)
    hi = _side(kind, r_in, r_out, w / 2, inner_ratio, (0.5 + EDGE_OUTSIDE) * e)
    one = lo + list(reversed(hi))
    return [one, [(r, -l) for r, l in reversed(one)]]


def points_ring(cx: float, cy: float, kind: Literal["star", "diamond", "bow", "petal"],
                count: int, r_inner: float, r_outer: float,
                rotation_deg: float, width_ratio: float, inner_ratio: float | None = None,
                edge_width: float | None = None) -> list[Shape]:
    """Angles run clockwise on screen from +x; point k is centered on
    rotation_deg + k*360/count. Width = width_ratio * (r_outer - r_inner) is the widest
    lateral width; the width profile is the designer's for the kind (LOOP_PROFILE,
    KITE_PROFILE, PETAL_PROFILE). An explicit inner_ratio sets a bow's base width
    (share of the widest; LOOP_PROFILE[0] = 0.85 when unset), the profile rescaled. With edge_width the shapes are
    the two edge strips along each point's sides instead (two-tone loops)."""
    if kind == "star":
        pts = []
        for k in range(count):
            pts.append(polar(cx, cy, r_outer, rotation_deg + 360.0 * k / count))
            pts.append(polar(cx, cy, r_inner, rotation_deg + 360.0 * (k + 0.5) / count))
        return [Shape("polygon", {"points": points_attr(pts)}, None)]
    w = width_ratio * (r_outer - r_inner)
    if edge_width:
        outlines_ = _edge_strips(kind, r_inner, r_outer, w, inner_ratio, edge_width)
    else:
        outlines_ = [_outline(kind, r_inner, r_outer, w, inner_ratio)]
    shapes = []
    for k in range(count):
        th = rotation_deg + 360.0 * k / count
        for o in outlines_:
            pts = [_to_world(cx, cy, th, r, l) for r, l in o]
            shapes.append(Shape("polygon", {"points": points_attr(pts)}, None))
    return shapes
