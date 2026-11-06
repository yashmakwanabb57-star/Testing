"""RosetteModel -> list[Shape] for the SVG writer."""
from __future__ import annotations

from .geometry import disc, outlines, pinked_circle, place_rose, points_ring, ring, tails
import math

from shapely import Point, unary_union

from .geometry.clip import clip_occluded, to_geom
from .geometry.tails import CX as TAIL_CX
from .model import Part, RosetteModel
from .shapes import Shape

# Tail tops (H5). The reference files never draw a flat tail top: in 60 of 78 files every
# stripe's top row is under 0.06 head radii wide (median 0.015 R over 511 stripes); the top
# is either a point (213) or a slope (298). Our stripes keep full width down from top_y and
# close in one apex per tail on the tail axis, TAIL_TIP_DEPTH head radii below the head
# center (median top of the sloped reference stripes 0.88 R, quartiles 0.68-0.97). The apex
# stays at least TAIL_TIP_MIN head radii above top_y (median reference taper 0.05 R) and
# must lie inside the drawn head (any head shape covers it); when the head does not reach
# that depth the apex moves to the nearest covered depth (TAIL_TIP_STEP steps) and warns;
# when no depth is covered, or there is no head, it stays and warns that it shows.
TAIL_TIP_DEPTH, TAIL_TIP_MIN, TAIL_TIP_STEP = 0.88, 0.2, 0.01
TAIL_TIP_WARNING = "Tail top moved to stay under the head - check the tail tops against the photo."
TAIL_UNCOVERED_WARNING = "Tail top is not under any head shape - check the tail tops against the photo."
_HEAD_ROLES = ("head", "ring", "points", "center")

_TAIL_KEYS = ("tails_count", "top_y", "bottom_y", "top_width", "bottom_width",
              "spread_deg", "stripes", "stripes_bottom")


def _parse_stripes(s) -> list[float]:
    return [float(x) for x in str(s).split(",") if x.strip()]


def _tail_tips(model: RosetteModel, head: list[Shape], key: tuple) -> list[float]:
    """Apex height per tail (see TAIL_TIP_DEPTH); warns on the model when one moved
    (TAIL_TIP_WARNING) or lies under no head shape (TAIL_UNCOVERED_WARNING). Depths are
    in head radii: the drawn head's largest distance from its center; without a head
    half the tail's top width stands in for it."""
    count, top_y, tw, spread = int(key[0]), float(key[1]), float(key[3]), float(key[5])
    slopes = [0.0 if count == 1 else math.tan(math.radians(-spread / 2 + k * spread / (count - 1)))
              for k in range(count)]
    hp = next((q.params for r in ("center", "ring", "head", "points") for q in model.parts if q.role == r), None)

    def warn(msg: str) -> None:
        if msg not in model.warnings:
            model.warnings.append(msg)

    if not head or hp is None:
        warn(TAIL_UNCOVERED_WARNING)
        return [top_y - TAIL_TIP_MIN * tw / 2] * count
    sil = unary_union([to_geom(s) for s in head])
    cx, cy = float(hp["cx"]), float(hp["cy"])
    r = max(math.hypot(x - cx, y - cy) for g in getattr(sil, "geoms", [sil]) for x, y in g.exterior.coords)
    y_max = top_y - TAIL_TIP_MIN * r
    want = min(cy + TAIL_TIP_DEPTH * r, y_max)
    y_lo = min(cy, y_max)
    out = []
    for slope in slopes:
        cands = [want]
        for i in range(1, int(math.ceil(2 / TAIL_TIP_STEP)) + 1):
            for y in (want - i * TAIL_TIP_STEP * r, want + i * TAIL_TIP_STEP * r):
                if y_lo <= y <= y_max:
                    cands.append(y)
        y = next((y for y in cands if sil.contains(Point(TAIL_CX + (y - top_y) * slope, y))), None)
        if y is None:
            warn(TAIL_UNCOVERED_WARNING)
            y = want
        elif y != want:
            warn(TAIL_TIP_WARNING)
        out.append(y)
    return out


def build(model: RosetteModel) -> list[Shape]:
    pairs = color_shapes(model)
    colors = clip_occluded([s for s, _ in pairs])  # Color 1 is painted first whatever the stacking
    result = list(colors)
    result += outlines(colors)

    center = next((p for p in model.parts if p.role == "center"), None)
    if model.rose and center is not None:
        c = center.params
        rose = place_rose(c["cx"], c["cy"], 0.8 * c["r"])
        result += rose
        circel = disc(c["cx"], c["cy"], c["r"])
        circel.kind, circel.role, circel.z = "circel", "center", center.z
        result.append(circel)
    return result


def color_shapes(model: RosetteModel) -> list[tuple[Shape, Part]]:
    """(color shape, its part) for every part, in intended paint order (z, stable),
    before the occlusion cuts."""
    groups = {g.id: g for g in model.groups}
    colors: list[tuple[Shape, Part]] = []

    def finish(part: Part, shapes: list[Shape]) -> None:
        g = groups.get(part.color_group)
        if g is None:
            raise ValueError(f"part {part.id}: unknown color group {part.color_group}")
        if g.category is None:
            raise ValueError(f"part {part.id}: color group {g.id} has no category; run assignment first")
        for s in shapes:
            s.category = g.category
            s.pattern = g.pattern
            s.role = part.role
            s.z = part.z
            s.kind = "color"
            colors.append((s, part))

    tail_groups: dict[tuple, list[Part]] = {}
    for p in model.parts:
        q = p.params
        if p.role == "head":
            s = pinked_circle(q["cx"], q["cy"], q["r"]) if int(q.get("pinked", 1)) else disc(q["cx"], q["cy"], q["r"])
            finish(p, [s])
        elif p.role == "ring":
            finish(p, [ring(q["cx"], q["cy"], q["r_outer"], q["r_inner"])])
        elif p.role == "points":
            ir, ew = q.get("inner_ratio"), q.get("edge_width")
            finish(p, points_ring(q["cx"], q["cy"], q["kind"], int(q["count"]), q["r_inner"],
                                  q["r_outer"], q["rotation_deg"], q["width_ratio"],
                                  None if ir is None else float(ir), None if ew is None else float(ew)))
        elif p.role == "center":
            finish(p, [disc(q["cx"], q["cy"], q["r"])])
        elif p.role == "tail_stripe":
            tail_groups.setdefault(tuple(q.get(k, "") for k in _TAIL_KEYS), []).append(p)
        else:
            msg = f"part {p.id}: unrecognised shape, drawn as a plain disc"
            if msg not in model.warnings:
                model.warnings.append(msg)
            finish(p, [disc(q["cx"], q["cy"], q["r"])])

    head = [s for s, _ in colors if s.role in _HEAD_ROLES or (s.role == "unknown" and s.z < 60)]
    for key, parts in tail_groups.items():
        top_y, bottom_y, tw, bw, spread = key[1], key[2], key[3], key[4], key[5]
        polys = tails(top_y, bottom_y, int(key[0]), tw, bw, spread, _parse_stripes(key[6]),
                      stripes_bottom=_parse_stripes(key[7]) or None,
                      tip_y=_tail_tips(model, head, key))
        for p in parts:
            ti, si = int(p.params["tail_index"]), int(p.params["stripe_index"])
            try:
                shape = polys[ti][si]
            except IndexError:
                raise ValueError(f"part {p.id}: tail {ti} stripe {si} out of range") from None
            finish(p, [shape])

    colors.sort(key=lambda t: t[0].z)
    return colors
