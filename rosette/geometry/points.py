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
LOOP_CURVE_BELOW = 0.6  # a bow base narrower than this share of its widest widens on a curve
LOOP_PROFILE = (0.85, 0.85, 0.89, 0.92, 0.94, 0.96, 0.98, 1.0, 1.0, 1.0, 1.0)
KITE_PEAK = 0.4
KITE_PROFILE = tuple(round(t / KITE_PEAK if t <= KITE_PEAK else (1 - t) / (1 - KITE_PEAK), 4) for t in _T)
# A lens kite (R37, S11-15's designer leaves over the 12-point star): a triangle standing on the
# center, widest at its base, tapering almost straight to its tip.
LENS_TAPER = 0.9  # S11-15's designer leaves: width 0.40 at the base to 0 at the tip, nearly straight
LEAF_LENS_TAPER = 0.75  # R83, S11-15's designer leaves on the 12-point star: sides bowing out (widths at 0.45, 0.6, 0.75, 0.9 of the tip)
LENS_PROFILE = tuple(round(math.sin(math.pi * t) ** 0.75, 4) for t in tuple(i / 10 for i in range(11)))
PETAL_PROFILE = (0.75, 0.75, 0.86, 0.96, 0.98, 1.0, 0.83, 0.60, 0.41, 0.23, 0.0)
# A satin petal flower (R56, S36-15's designer petals, median of its six, widths measured across
# the axis from the rose's center): from the rose's edge, widest at a third of the length, then a
# nearly straight taper to a sharp tip.
FLOWER_PROFILE = (0.70, 0.82, 0.93, 0.99, 0.94, 0.85, 0.70, 0.55, 0.37, 0.19, 0.0)
# A leaf petal (R64, S42-15's designer petals, median of its six, from the rose's center): full width
# just past the rose's edge, then a straight taper to a sharp tip.
LEAF_PETAL_PROFILE = (0.55, 0.72, 0.99, 1.0, 0.90, 0.80, 0.66, 0.50, 0.34, 0.17, 0.0)
# A satin flower's petal (designer S47-15's petals, widths over its tip radius every 0.05 of the length
# from the center, / 0.69): the bottom a smooth line widening from the center with a slight curve into
# its widest at 0.55 (no bump at the rose's edge), then nearly straight sides to a sharp tip (the owner's
# check). The back petals are the same, half covered by the front ones.
SATIN_FRONT_PROFILE = (0.02, 0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82, 0.91, 0.97, 1.0, 0.97,
                       0.9, 0.8, 0.68, 0.56, 0.42, 0.28, 0.14, 0.0)
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
# A leaf star (R30, A14-17's designer star): each arm fills its share of the circle at the
# valleys and narrows to its tip with sides bowing outward. Share of the arm's angular span
# (360 / count) it covers at t = 0, 0.1, ..., 1 from the valleys to the tip, measured on the
# designer file (circles every 6 units from r 50 to 110 around the head center).
LEAF_COVER = (1.0, 0.91, 0.76, 0.62, 0.51, 0.40, 0.30, 0.21, 0.13, 0.05, 0.0)
# A petal flower (R36, S10-19's designer flower between the loops): fuller petals, measured the
# same way on that file (valleys 0.63, tips 1.0 of its radius).
ROUND_COVER = (1.0, 0.97, 0.91, 0.83, 0.73, 0.62, 0.50, 0.38, 0.25, 0.12, 0.0)
# A diamond point (R81, S2-15's designer points, rays from the rose: 15 degrees either side at 0.65 of the tips'
# reach, 10 at 0.75, 5 at 0.84): inward-curved sides to a sharp tip (the owner's check).
DIAMOND_COVER = (0.87, 0.735, 0.609, 0.492, 0.384, 0.287, 0.201, 0.127, 0.066, 0.022, 0.0)  # 0.87 (1 - t)^1.6: sides curving in to the tip, wide below (under the loops)


def _to_world(cx, cy, theta_deg, radial, lateral) -> Pt:
    """radial = distance along the point's axis, lateral = offset perpendicular."""
    a = math.radians(theta_deg)
    ca, sa = math.cos(a), math.sin(a)
    return (cx + radial * ca - lateral * sa, cy + radial * sa + lateral * ca)


def _profile(kind: str, inner_ratio: float | None, lens: int = 0) -> tuple[float, ...]:
    if kind == "bow":
        if inner_ratio is None:
            return LOOP_PROFILE
        if inner_ratio < LOOP_CURVE_BELOW:
            # a loop narrowed hard toward the center (R44/R45: the designer's rose loops) widens
            # on a smooth curve, no corner where the straight profile would level off
            return tuple(round(inner_ratio + (1 - inner_ratio) * math.sin(math.pi / 2 * t), 4) for t in _T)
        # an edited base width: the loop profile rescaled between it and full width
        base = LOOP_PROFILE[0]
        return tuple(inner_ratio + (p - base) * (1 - inner_ratio) / (1 - base) for p in LOOP_PROFILE)
    if kind == "diamond":
        return LENS_PROFILE if lens else KITE_PROFILE
    if kind == "petal":
        return PETAL_PROFILE
    if kind == "flower":
        return FLOWER_PROFILE
    if kind == "leaf_petal":
        return LEAF_PETAL_PROFILE
    if kind == "satin_front":
        return SATIN_FRONT_PROFILE
    raise ValueError(f"unknown point kind {kind!r}")


def _side(kind: str, r_in: float, r_out: float, h: float, inner_ratio: float | None,
          offset: float = 0.0, lens: int = 0) -> list[tuple[float, float]]:
    """(radial, lateral) points of one side (lateral >= 0) from the inner to the outer
    end, the profile's half-width plus `offset`. A bow's side stops at the corner of
    its convex outer end (`_end_radial`, END_SAG)."""
    prof = _profile(kind, inner_ratio, lens)
    ts = _T if not lens else tuple(i / 30 for i in range(31))
    if not lens and len(prof) != len(ts):  # a finer measured profile (SATIN_FRONT_PROFILE: 21 points)
        ts = tuple(i / (len(prof) - 1) for i in range(len(prof)))
    if lens:  # a triangle on the center: widest at its base, tapering to its tip (2: convex sides, R83)
        prof = tuple((1 - t) ** (LEAF_LENS_TAPER if int(lens) == 2 else LENS_TAPER) for t in ts)
    pts = [(r_in + (r_out - r_in) * t, max(0.0, h * p + offset)) for t, p in zip(ts, prof)]
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
             inner_ratio: float | None = None, lens: int = 0) -> list[tuple[float, float]]:
    """(radial, lateral) vertices, symmetric about the axis: out along one side, back
    along the other (a bow crosses its outer end on an arc around the head center)."""
    side = _side(kind, r_in, r_out, w / 2, inner_ratio, lens=lens)
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


# A split arm (R70, S48-17's designer arms, traced from its rendered outline): two arms of a pair meet on
# the axis between their tips. Vertices as (radius / the tip's, degrees off the axis toward the tip): the
# tip, the notch on the axis, down the axis, the base, out along 31.7 degrees to the back edge, which runs
# to the tip (the designer's steps along it are the corner pieces lying on it; the owner's check).
SPLIT_ARM = ((1.0, 16.9), (0.731, 0.0), (0.583, 0.0), (0.437, 22.0), (0.478, 31.7), (0.66, 31.7))


def split_arms(cx: float, cy: float, count: int, r_tip: float, axis_deg: float, side: int) -> list[Shape]:
    """One SPLIT_ARM per axis (axis_deg + k * 360 / count), its tip on the `side` (+1 / -1) of the axis.
    The designer's band copies are the same arms at a smaller r_tip (R70)."""
    out = []
    for k in range(count):
        a0 = axis_deg + 360.0 * k / count
        pts = [polar(cx, cy, r * r_tip, a0 + side * d) for r, d in SPLIT_ARM]
        out.append(Shape("polygon", {"points": points_attr(pts)}, None))
    return out


# A side fold (R76, designer S63-17's banded arms, traced from its outlines about the arms' own center): a fold
# whose tip leans 20 degrees off the axis, one arm running in to the axis, the other out; the notch is the next
# band's tip (0.897 of this one's). Each band is drawn whole (filled in under its notch, where the next band lies
# on it) and runs on in under the corners and the flower, so no step between the arm ends and the corners
# shows the head through (the owner's check: no gaps). Vertices as
# (radius / the tip's, degrees off the axis toward the tip's side). The innermost band is whole.
SIDE_FOLD = ((1.0, 20.0), (0.708, 31.0), (0.58, 33.0), (0.56, 14.0), (0.625, 8.4), (0.648, 5.4))
SIDE_FOLD_LAST = ((1.0, 20.0), (0.677, 32.9), (0.58, 33.0), (0.56, 14.0), (0.629, 9.2), (0.71, 9.2))


def side_folds(cx: float, cy: float, count: int, r_tip: float, axis_deg: float, side: int, last: bool) -> list[Shape]:
    """One SIDE_FOLD (SIDE_FOLD_LAST) per axis (axis_deg + k * 360 / count), its tip on the `side` of the axis."""
    shape = SIDE_FOLD_LAST if last else SIDE_FOLD
    return [Shape("polygon", {"points": points_attr([polar(cx, cy, r * r_tip, axis_deg + 360.0 * k / count + side * d)
                                                     for r, d in shape])}, None) for k in range(count)]


# A straight loop (R77, designer S58-17's loops, traced across the loop at right angles to its axis): the
# sides run nearly parallel from the ruffle to a round end; its edge strip is a wider copy under it that stops
# short of the end. (distance along the axis, half-width), both / the star's tip.
STRAIGHT_LOOP = ((0.44, 0.146), (0.486, 0.144), (0.519, 0.162), (0.551, 0.179), (0.584, 0.181), (0.65, 0.182),
                 (0.85, 0.185), (0.863, 0.184), (0.875, 0.124), (0.882, 0.058), (0.886, 0.0))
STRAIGHT_LOOP_EDGE = ((0.44, 0.146), (0.551, 0.179), (0.584, 0.2), (0.616, 0.223), (0.853, 0.227), (0.853, 0.0))
LOBES = {"straight_loop": STRAIGHT_LOOP, "straight_loop_edge": STRAIGHT_LOOP_EDGE}


def lobes(cx: float, cy: float, count: int, r_tip: float, rotation_deg: float, profile: str) -> list[Shape]:
    """One symmetric lobe per axis (rotation_deg + k * 360 / count) from a LOBES (along, half-width) table."""
    half = LOBES[profile]
    side = [(a * r_tip, h * r_tip) for a, h in half]
    outline = side + [(a, -h) for a, h in reversed(side) if h]
    out = []
    for k in range(count):
        th = rotation_deg + 360.0 * k / count
        out.append(Shape("polygon", {"points": points_attr([_to_world(cx, cy, th, a, h) for a, h in outline])}, None))
    return out


def dots(cx: float, cy: float, count: int, r_mid: float, r_dot: float, rotation_deg: float,
         flat: bool = False) -> list[Shape]:
    """Small hexagons (R76: designer S63-17's small roses, a corner on the axis; R77, flat: S58-17's, a flat side
    on the axis) centered r_mid out on each axis."""
    out = []
    for k in range(count):
        th = rotation_deg + 360.0 * k / count
        x, y = polar(cx, cy, r_mid, th)
        out.append(Shape("polygon", {"points": points_attr([polar(x, y, r_dot, th + 30.0 * flat + 60.0 * j) for j in range(6)])}, None))
    return out


def chevrons(cx: float, cy: float, count: int, r_tip: float, length: float, half: float, r_notch: float,
             rotation_deg: float) -> list[Shape]:
    """One folded point per axis (rotation_deg + k * 360 / count), as designer S68-19 draws its folds (R74):
    a V from the tip, its arms ending `length` down the axis `half` off it and cut square to the arm, its
    inner edge parallel to the outer one from the notch on the axis at r_notch, where the next fold's tip
    sits (the folds of a point are one V moved along the axis). r_notch 0: the innermost fold, a triangle."""
    T, A = (r_tip, 0.0), (r_tip - length, half)
    if r_notch > 0:
        n = math.hypot(length, half)
        ux, uy = -length / n, half / n  # along the arm, tip to end
        w = (r_tip - r_notch) * uy  # the strip between the outer edge and its parallel through the notch
        I = (A[0] - uy * w, A[1] + ux * w)  # square across the arm, toward the axis
        side = [T, A, I, (r_notch, 0.0), (I[0], -I[1]), (A[0], -A[1])]
    else:
        side = [T, A, (A[0], -A[1])]
    out = []
    for k in range(count):
        th = rotation_deg + 360.0 * k / count
        out.append(Shape("polygon", {"points": points_attr([_to_world(cx, cy, th, r, l) for r, l in side])}, None))
    return out


def points_ring(cx: float, cy: float, kind: Literal["star", "diamond", "bow", "petal"],
                count: int, r_inner: float, r_outer: float,
                rotation_deg: float, width_ratio: float, inner_ratio: float | None = None,
                edge_width: float | None = None, leaf: int = 0, lens: int = 0,
                flower: int = 0) -> list[Shape]:
    """Angles run clockwise on screen from +x; point k is centered on
    rotation_deg + k*360/count. Width = width_ratio * (r_outer - r_inner) is the widest
    lateral width; the width profile is the designer's for the kind (LOOP_PROFILE,
    KITE_PROFILE, PETAL_PROFILE). An explicit inner_ratio sets a bow's base width
    (share of the widest; LOOP_PROFILE[0] = 0.85 when unset), the profile rescaled. With edge_width the shapes are
    the two edge strips along each point's sides instead (two-tone loops)."""
    if kind == "star" and leaf:
        return [_leaf_star(cx, cy, count, r_inner, r_outer, rotation_deg,
                           {2: ROUND_COVER, 3: DIAMOND_COVER}.get(int(leaf), LEAF_COVER))]
    if kind == "star":
        pts = []
        for k in range(count):
            pts.append(polar(cx, cy, r_outer, rotation_deg + 360.0 * k / count))
            pts.append(polar(cx, cy, r_inner, rotation_deg + 360.0 * (k + 0.5) / count))
        return [Shape("polygon", {"points": points_attr(pts)}, None)]
    w = width_ratio * (r_outer - r_inner)
    if kind == "petal" and flower:  # R56: the satin flower's petal (FLOWER_PROFILE); R64: 2, the leaf petal; R69: 4
        kind = {2: "leaf_petal", 4: "satin_front"}.get(int(flower), "flower")  # type: ignore[assignment]
    if edge_width:
        outlines_ = _edge_strips(kind, r_inner, r_outer, w, inner_ratio, edge_width)
    else:
        outlines_ = [_outline(kind, r_inner, r_outer, w, inner_ratio, lens)]
    shapes = []
    for k in range(count):
        th = rotation_deg + 360.0 * k / count
        for o in outlines_:
            pts = [_to_world(cx, cy, th, r, l) for r, l in o]
            shapes.append(Shape("polygon", {"points": points_attr(pts)}, None))
    return shapes


def _leaf_star(cx: float, cy: float, count: int, r_inner: float, r_outer: float, rotation_deg: float,
               cover: tuple = LEAF_COVER) -> Shape:
    """One polygon: a star whose arms run from the valleys (r_inner) to the tips (r_outer),
    covering LEAF_COVER of their angular span, the valleys shared by neighbouring arms."""
    span = 360.0 / count
    pts: list[Pt] = []
    for k in range(count):
        th = rotation_deg + span * k
        side = [(r_inner + (r_outer - r_inner) * t, c * span / 2) for t, c in zip(_T, cover)]
        arm = [(r, th - h) for r, h in side] + [(r, th + h) for r, h in reversed(side[:-1])]
        pts += [polar(cx, cy, r, a) for r, a in arm[:-1]]  # the last valley starts the next arm
    return Shape("polygon", {"points": points_attr(pts)}, None)
