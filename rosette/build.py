"""RosetteModel -> list[Shape] for the SVG writer."""
from __future__ import annotations

from shapely import LineString, Polygon, make_valid, set_precision, unary_union

from .geometry import disc, outlines, pinked_circle, place_rose, points_ring, ring, tails
from .geometry.clip import SIMPLIFY, clip_occluded, from_geom, to_geom
from .model import Part, RosetteModel
from .shapes import Shape

# Tail tops (H5). The reference files never draw a free tail top: in 60 of 78 files every
# stripe's top row is under 0.06 head radii wide; of 511 stripe tops 298 follow the lower
# outline of the head shapes over them and 213 end in a point under one. Our stripes are
# drawn at full width from top_y down as before, run on up their side lines into every
# gap between the head shapes that opens onto the stripe's top and is closed above by
# the head (a hole inside the head stays empty, a gap open to the outside is not filled),
# and cut to the outside of every head shape whatever the layers, as the reference tails
# are: every top edge then lies on a head shape's outline, in the shop and on outline
# pages that draw hidden edges too. Where the head does not close a stripe's top edge it
# stays flat at top_y and the model warns.
TAIL_UNCOVERED_WARNING = "Tail top is not under any head shape - check the tail tops against the photo."
_HEAD_ROLES = ("head", "ring", "points", "center")

_TAIL_KEYS = ("tails_count", "top_y", "bottom_y", "top_width", "bottom_width",
              "spread_deg", "stripes", "stripes_bottom")


def _parse_stripes(s) -> list[float]:
    return [float(x) for x in str(s).split(",") if x.strip()]


def _under_head(stripe: Shape, top_y: float, bottom_y: float, head) -> tuple[Shape, bool]:
    """The stripe run up its side lines into every gap between the head shapes that opens
    onto its top edge and is closed above by the head, and cut to the outside of the head
    (see the tail tops note). (shape, True when its top edge at top_y lies wholly under
    the head or in such gaps)."""
    pts = [tuple(float(v) for v in p.split(",")) for p in stripe.attrs["points"].split()]
    tl, tr, br, bl = pts[0], pts[1], pts[2], pts[-1]
    if head is None or head.is_empty:
        return stripe, False
    up = min(head.bounds[1], top_y) - 1.0
    # side lines x = a + b * t (t = 0 at top_y, 1 at bottom_y); they may meet above top_y
    dl, dr = bl[0] - tl[0], br[0] - tr[0]
    t = (up - top_y) / (bottom_y - top_y)
    if dl != dr and t < (tl[0] - tr[0]) / (dr - dl) < 0:  # they meet between `up` and top_y
        tm = (tl[0] - tr[0]) / (dr - dl)
        ring_ = [(tl[0] + dl * tm, top_y + tm * (bottom_y - top_y)), tr, tl]
    else:
        ring_ = [(tl[0] + dl * t, up), (tr[0] + dr * t, up), tr, tl]
    ext = set_precision(Polygon(ring_), 1e-3)
    top = LineString([tl, tr])
    gaps = [g for g in getattr(ext.difference(head), "geoms", [ext.difference(head)])
            if isinstance(g, Polygon) and g.distance(top) < 1e-6 and g.bounds[1] > up + 1e-6]
    covered = top.difference(unary_union([head, *gaps]).buffer(0.01)).length < 0.5
    shape = from_geom(unary_union([Polygon(pts), *gaps]).difference(head).simplify(SIMPLIFY), stripe)
    return (shape or stripe), covered


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
    # on a 0.001 grid: unions of pinked and cut shapes can trip GEOS on near-coincident edges
    head_geom = set_precision(unary_union([make_valid(to_geom(s)) for s in head]), 1e-3) if head else None
    for key, parts in tail_groups.items():
        top_y, bottom_y, tw, bw, spread = key[1], key[2], key[3], key[4], key[5]
        polys = tails(top_y, bottom_y, int(key[0]), tw, bw, spread, _parse_stripes(key[6]),
                      stripes_bottom=_parse_stripes(key[7]) or None)
        for p in parts:
            ti, si = int(p.params["tail_index"]), int(p.params["stripe_index"])
            try:
                shape = polys[ti][si]
            except IndexError:
                raise ValueError(f"part {p.id}: tail {ti} stripe {si} out of range") from None
            shape, covered = _under_head(shape, float(top_y), float(bottom_y), head_geom)
            if not covered and TAIL_UNCOVERED_WARNING not in model.warnings:
                model.warnings.append(TAIL_UNCOVERED_WARNING)
            finish(p, [shape])

    colors.sort(key=lambda t: t[0].z)
    return colors
