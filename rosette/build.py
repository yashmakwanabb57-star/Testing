"""RosetteModel -> list[Shape] for the SVG writer."""
from __future__ import annotations

import math

from shapely import LineString, Polygon, make_valid, set_precision, unary_union
from shapely.affinity import affine_transform

from .geometry import disc, outlines, pinked_circle, pinked_ring, place_rose, points_ring, ring, tails
from .geometry.clip import SIMPLIFY, clip_visible, from_geom, to_geom, xor_all
from .model import Part, RosetteModel
from .shapes import Shape
from .vision.templates import template_by_id

# Tail tops (H5). The reference files never draw a free tail top: in 60 of 78 files every
# stripe's top row is under 0.06 head radii wide; of 511 stripe tops 298 follow the lower
# outline of the head shapes over them and 213 end in a point under one. Our stripes are
# drawn at full width from top_y down as before, run on up their side lines into every
# gap between the head shapes that opens onto the stripe's top and is closed above by
# the head (a gap open upward, past the head's top, is not filled; one opening sideways
# past the stripe's outer side line is filled up to that line), and cut to the outside
# of every head shape whatever the layers, as the reference tails are: every top edge
# then lies on a head shape's outline, in the shop and on outline pages that draw
# hidden edges too. Where the head does not close a stripe's top edge it
# stays flat at top_y and the model warns.
TAIL_UNCOVERED_WARNING = "Tail top is not under any head shape - check the tail tops against the photo."
# A stripe that lies wholly under the head is not drawn (never restored uncut).
TAIL_HIDDEN_WARNING = ("Tail stripe {part} lies wholly under the head and is not drawn - "
                       "check the tails against the photo.")
# Grid (canvas units) the extension above a stripe is snapped to before it is cut.
_GRID = 1e-3

# Library heads (H7): a `template` part is drawn as the reference head's pieces placed
# on its head circle, piece i (the reference's paint order) at z TEMPLATE_Z + i, so the
# occlusion cuts stack them exactly as the reference does whatever Colors they map to.
TEMPLATE_Z = 10

_TAIL_KEYS = ("tails_count", "top_y", "bottom_y", "top_width", "bottom_width",
              "spread_deg", "stripes", "stripes_bottom")


def _parse_stripes(s) -> list[float]:
    return [float(x) for x in str(s).split(",") if x.strip()]


def _under_head(stripe: Shape, top_y: float, bottom_y: float, head) -> tuple[Shape | None, bool]:
    """The stripe run up its side lines into every gap between the head shapes that opens
    onto its top edge and is closed above by the head, and cut to the outside of the head
    (see the tail tops note). (shape, True when its top edge at top_y lies wholly under
    the head or in such gaps; shape None when nothing of it lies outside the head)."""
    pts = [tuple(float(v) for v in p.split(",")) for p in stripe.attrs["points"].split()]
    tl, tr, br, bl = pts[0], pts[1], pts[2], pts[-1]
    if head is None or head.is_empty:
        return stripe, False
    up = math.floor((min(head.bounds[1], top_y) - 1.0) / _GRID) * _GRID  # on ext's grid
    # side lines x = a + b * t (t = 0 at top_y, 1 at bottom_y); they may meet above top_y
    dl, dr = bl[0] - tl[0], br[0] - tr[0]
    t = (up - top_y) / (bottom_y - top_y)
    if dl != dr and t < (tl[0] - tr[0]) / (dr - dl) < 0:  # they meet between `up` and top_y
        tm = (tl[0] - tr[0]) / (dr - dl)
        ring_ = [(tl[0] + dl * tm, top_y + tm * (bottom_y - top_y)), tr, tl]
    else:
        ring_ = [(tl[0] + dl * t, up), (tr[0] + dr * t, up), tr, tl]
    ext = set_precision(Polygon(ring_), _GRID)
    top = LineString([tl, tr])
    gaps = [g for g in getattr(ext.difference(head), "geoms", [ext.difference(head)])
            if isinstance(g, Polygon) and g.distance(top) < 1e-6 and g.bounds[1] > up + _GRID / 2]
    covered = top.difference(unary_union([head, *gaps]).buffer(0.01)).length < 0.5
    shape = from_geom(unary_union([Polygon(pts), *gaps]).difference(head).simplify(SIMPLIFY), stripe)
    return shape, covered


def _template(part: Part):
    t = template_by_id(str(part.params.get("template_id", "")))
    if t is None:
        raise ValueError(f"part {part.id}: unknown head template {part.params.get('template_id')}")
    return t


def _template_matrix(q: dict) -> list[float]:
    """Head radii -> canvas units: scale by r, turn by rotation_deg, move to (cx, cy)."""
    r, a = float(q["r"]), math.radians(float(q.get("rotation_deg", 0.0)))
    c, s = math.cos(a) * r, math.sin(a) * r
    return [c, -s, s, c, float(q["cx"]), float(q["cy"])]


def template_groups(part: Part) -> dict[int, int]:
    """Reference category -> color group of a template part (params group_1..group_4;
    a category without one takes the part's color_group)."""
    q = part.params
    return {c: int(q.get(f"group_{c}", part.color_group)) for c in range(1, 5)}


def template_pieces(part: Part) -> list[tuple[Shape, int]]:
    """(shape, color group) per reference piece of a template part, in paint order; the
    shapes carry role (the piece's), z and kind but no category yet."""
    m = _template_matrix(part.params)
    groups = template_groups(part)
    pieces = _template(part).pieces
    # a piece drawn again later with the same outline is wholly hidden (some reference
    # files repeat an element): only the last copy is drawn, so no duplicate is written
    keys = [tuple(tuple(r.reshape(-1).tolist()) for r in pc.rings) for pc in pieces]
    out = []
    for i, pc in enumerate(pieces):
        if keys[i] in keys[i + 1:]:
            continue
        geom = xor_all([Polygon(ring_).buffer(0) for ring_ in pc.rings])
        if geom is None:
            continue
        shape = from_geom(affine_transform(geom, m), Shape("polygon", {}, None, role=pc.role, z=TEMPLATE_Z + i))
        if shape is not None:
            out.append((shape, groups[pc.category]))
    return out


def template_center(part: Part) -> tuple[float, float, float] | None:
    """(cx, cy, r) for the rose of a template head: its center pieces' reach, or None."""
    reach = [float((ring_ ** 2).sum(axis=1).max() ** 0.5) for pc in _template(part).pieces
             if pc.role == "center" for ring_ in pc.rings[:1]]
    if not reach:
        return None
    q = part.params
    return float(q["cx"]), float(q["cy"]), max(reach) * float(q["r"])


def _rose_center(model: RosetteModel) -> tuple[float, float, float, int] | None:
    center = next((p for p in model.parts if p.role == "center"), None)
    if center is not None:
        c = center.params
        return float(c["cx"]), float(c["cy"]), float(c["r"]), center.z
    tmpl = next((p for p in model.parts if p.role == "template"), None)
    if tmpl is None:
        return None
    hidden = next((p for p in model.alternatives if p.role == "center"), None)
    at = template_center(tmpl)
    if at is None and hidden is not None:  # the detected center, moved onto the template
        at = float(tmpl.params["cx"]), float(tmpl.params["cy"]), float(hidden.params["r"])
    if at is None:
        return None
    return (*at, TEMPLATE_Z + len(_template(tmpl).pieces))


def build(model: RosetteModel) -> list[Shape]:
    pairs = color_shapes(model)
    colors = clip_visible([s for s, _ in pairs])  # visible pieces, whatever the Colors (H14)
    result = list(colors)
    result += outlines(colors)

    center = _rose_center(model)
    if model.rose and center is not None:
        cx, cy, r, z = center
        rose = place_rose(cx, cy, 0.8 * r)
        result += rose
        circel = disc(cx, cy, r)
        circel.kind, circel.role, circel.z = "circel", "center", z
        result.append(circel)
    return result


def color_shapes(model: RosetteModel) -> list[tuple[Shape, Part]]:
    """(color shape, its part) for every part, in intended paint order (z, stable),
    before the occlusion cuts."""
    groups = {g.id: g for g in model.groups}
    colors: list[tuple[Shape, Part]] = []

    def group(part: Part, gid: int):
        g = groups.get(gid)
        if g is None:
            raise ValueError(f"part {part.id}: unknown color group {gid}")
        if g.category is None:
            raise ValueError(f"part {part.id}: color group {g.id} has no category; run assignment first")
        return g

    def finish(part: Part, shapes: list[Shape]) -> None:
        g = group(part, part.color_group)
        for s in shapes:
            s.category = g.category
            s.pattern = g.pattern
            s.role = part.role
            s.z = part.z
            s.kind = "color"
            colors.append((s, part))

    head_parts: list[int] = []  # ids of the parts whose shapes make the head

    tail_groups: dict[tuple, list[Part]] = {}
    for p in model.parts:
        q = p.params
        if p.role != "tail_stripe" and not (p.role == "unknown" and p.z >= 60):
            head_parts.append(id(p))
        if p.role == "template":
            for s, gid in template_pieces(p):
                g = group(p, gid)
                s.category, s.pattern, s.kind = g.category, g.pattern, "color"
                colors.append((s, p))
        elif p.role == "head":
            s = pinked_circle(q["cx"], q["cy"], q["r"]) if int(q.get("pinked", 1)) else disc(q["cx"], q["cy"], q["r"])
            finish(p, [s])
        elif p.role == "ring":
            ring_of = pinked_ring if int(q.get("pinked", 0)) else ring
            finish(p, [ring_of(q["cx"], q["cy"], q["r_outer"], q["r_inner"])])
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

    head = [s for s, p in colors if id(p) in head_parts]
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
            if shape is None:
                msg = TAIL_HIDDEN_WARNING.format(part=p.id)
                if msg not in model.warnings:
                    model.warnings.append(msg)
                continue
            finish(p, [shape])

    colors.sort(key=lambda t: t[0].z)
    return colors
