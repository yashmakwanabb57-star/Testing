"""RosetteModel -> list[Shape] for the SVG writer."""
from __future__ import annotations

import math

import numpy as np
from shapely import LineString, Point, Polygon, make_valid, set_precision, unary_union
from shapely.affinity import affine_transform

from .geometry import disc, outlines, pinked_circle, pinked_ring, place_rose, points_ring, ring, tails
from .geometry.points import LEAF_LENS_TAPER, LENS_TAPER, LOOP_PROFILE, chevrons, dots, lobes, side_folds, split_arms
from .geometry.common import polar
from .geometry.clip import SIMPLIFY, clip_occluded, clip_visible, from_geom, to_geom, xor_all
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

# R26: step (canvas units) by which a tail whose top the head leaves open is raised.
RAISE_STEP = 5.0
# R88: a tail read running up over the head's outer bands (tail param from_r, head radii: where it tucks
# under, A10-15's inner ring) is drawn over every head shape reaching past (from_r + TAIL_FROM_PAD) head radii
# and under the shapes inside it, its top run up its side lines to the head's center (the designer files cut
# the outer bands around such a tail; the owner's check).
TAIL_FROM_PAD = 0.04
TAIL_TUCK_SHARE = 0.5

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


def _raised_stripe(key: tuple, ti: int, si: int, new_top: float) -> Shape | None:
    """R26: stripe (ti, si) of a one-tail group `key` with its top moved up to new_top: every
    stripe boundary runs on along its own straight line (bottom point to old top point), so
    the raised stripes share their edges. None when a stripe would vanish on the way."""
    top_y, bottom_y, tw, bw = float(key[1]), float(key[2]), float(key[3]), float(key[4])
    f_top, f_bot = _parse_stripes(key[6]), _parse_stripes(key[7]) or None
    old = tails(top_y, bottom_y, 1, tw, bw, float(key[5]), f_top, stripes_bottom=f_bot)[0]
    pts = [[tuple(float(v) for v in q.split(",")) for q in s.attrs["points"].split()] for s in old]
    tops = [pts[0][0][0]] + [ps[1][0] for ps in pts]  # boundary x at the old top, left to right
    bots = [pts[0][-1][0]] + [ps[2][0] for ps in pts]  # and at the bottom
    k = (bottom_y - new_top) / (bottom_y - top_y)
    xs = [xb + (xt - xb) * k for xt, xb in zip(tops, bots)]
    widths = [b - a for a, b in zip(xs, xs[1:])]
    if min(widths) <= 0:
        return None
    w2 = xs[-1] - xs[0]
    polys = tails(new_top, bottom_y, 1, w2, bw, float(key[5]), [w / w2 for w in widths], stripes_bottom=f_bot)
    try:
        return polys[ti][si]
    except IndexError:
        return None


def _tail_over_head(colors, head_parts, circle, from_r: float, key: tuple, idx, polys, bottom_y: float):
    """R88: (cut stripes, tail z) for a one-tail group running over the head's outer bands, or None
    when the head has no shapes on both sides of the tuck radius."""
    cx, cy, r = circle
    lim = (from_r + TAIL_FROM_PAD) * r
    inner, outer_z = [], []
    head = []
    for s, hp in colors:
        if id(hp) not in head_parts:
            continue
        g = to_geom(s)
        pts = np.asarray(g.convex_hull.exterior.coords) if hasattr(g.convex_hull, "exterior") else np.zeros((1, 2))
        head.append((s, g, float(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy).max())))
    for s, g, reach in head:
        # the band a ring shows: out from the largest smaller shape stacked above it (rings are drawn as
        # discs out from the center, the inner ones on top) or from its own hole
        hole = max([float(g.distance(Point(cx, cy)))] + [rr for t, _, rr in head if t.z > s.z and rr < reach])
        # a ring whose band's outer part (TAIL_TUCK_SHARE of its width) holds the tuck radius covers the tail's
        # top: a tail tucks under a whole band (A2-15: its white tail read a little into the white ruffle round
        # the badge, the designer's tucked under it; A10-15's runs most of the way over its white band)
        straddled = s.role == "ring" and reach - TAIL_TUCK_SHARE * (reach - hole) < from_r * r < reach
        if reach <= lim or straddled or s.role == "center":  # the center (a badge, the rose) is always on top
            inner.append((s, g))
        elif s.role in ("head", "ring"):
            outer_z.append(s.z)
        else:  # a points layer outside: the tail shows between its points (gaps), never over them
            return None
    if not inner or not outer_z:
        return None
    zi = min(s.z for s, _ in inner)
    tail_z = (max(outer_z) + zi) / 2 if max(outer_z) < zi else zi - 0.5
    inner_geom = unary_union([g for _, g in inner])
    # the top runs up the side lines as far as every stripe lasts (to the center at most); it must reach in
    # under the inner shapes, past the tuck radius, or the tail is left as it is
    y, raised = cy, None
    while y < cy + from_r * r:
        raised = [_raised_stripe(key, ti, si, y) for ti, si in idx]
        if all(x is not None for x in raised):
            break
        raised, y = None, y + RAISE_STEP
    if raised is None:
        return None
    cut = [(from_geom(to_geom(x).difference(inner_geom).simplify(SIMPLIFY), x), True) for x in raised]
    return cut, tail_z


# A loop (or a lens leaf, S11-15) starting at or inside the center disc has its straight base's corners under the disc: its base
# moves in until they lie at LOOP_TUCK of the disc's radius, its widest width kept (S4-15 in the app, the
# owner's check: the light loops' base corners stuck out 5 units past the badge, a notch at each loop).
LOOP_TUCK = 0.9


def _tucked_loop(q: dict, center_r: float | None) -> tuple[float, float]:
    """(r_inner, width_ratio) to draw a points part with (see LOOP_TUCK)."""
    r_in, r_out, wr = float(q["r_inner"]), float(q["r_outer"]), float(q["width_ratio"])
    if q.get("kind") == "bow":
        base = float(q.get("inner_ratio") or LOOP_PROFILE[0])
    elif q.get("kind") == "diamond" and int(q.get("lens", 0)):  # a lens leaf is widest at its base (S11-15)
        base = 1.0
    else:
        return r_in, wr
    if q.get("edge_width") or center_r is None or r_in > center_r + 1e-6:
        return r_in, wr
    w = wr * (r_out - r_in)
    if base == 1.0:  # a lens: width w * ((r_out - r) / (r_out - r_in)) ** k, so a lens run further in with its width
        # scaled to match is the same shape outside: step in until its base corners lie under the disc
        k = LEAF_LENS_TAPER if int(q.get("lens", 0)) == 2 else LENS_TAPER
        r2 = r_in
        while r2 > 0:
            w2 = w * ((r_out - r2) / (r_out - r_in)) ** k
            if math.hypot(r2, w2 / 2) <= LOOP_TUCK * center_r:
                return r2, w2 / (r_out - r2)
            r2 -= 1.0
        return r_in, wr
    half_base = base * w / 2
    if math.hypot(r_in, half_base) <= LOOP_TUCK * center_r or half_base >= LOOP_TUCK * center_r:
        return r_in, wr
    r2 = math.sqrt((LOOP_TUCK * center_r) ** 2 - half_base ** 2)
    return r2, w / (r_out - r2)


# A diamond inside a star of another color, on its points (same count and turn, reaching less far, drawn
# over it), is an inlay: the star's points shrunk evenly by INLAY_BORDER of the star's reach, so a band of
# the star's color shows down both sides and round the tip (S35-19: white diamonds in lilac points, the
# owner's check; a kite's sides had run into the point's sides by the loops, no band there).
INLAY_BORDER = 0.03
INLAY_TURN = 2.0  # degrees


def _inlays(parts: list) -> dict:
    """{id(diamond part): its inlay shape} for diamonds lying in a star's points (see INLAY_BORDER)."""
    out = {}
    stars = [p for p in parts if p.role == "points" and p.params.get("kind") == "star"]
    for d in parts:
        q = d.params
        if d.role != "points" or q.get("kind") != "diamond" or int(q.get("lens", 0)) or q.get("edge_width"):
            continue
        n = int(q["count"])
        for st in stars:
            sq = st.params
            turn = (float(sq["rotation_deg"]) - float(q["rotation_deg"])) % (360.0 / n)
            if (int(sq["count"]) != n or st.color_group == d.color_group or st.z > d.z
                    or min(turn, 360.0 / n - turn) > INLAY_TURN or float(sq["r_outer"]) <= float(q["r_outer"])
                    or abs(float(sq["cx"]) - float(q["cx"])) > 1 or abs(float(sq["cy"]) - float(q["cy"])) > 1):
                continue
            star = to_geom(points_ring(sq["cx"], sq["cy"], "star", n, sq["r_inner"], sq["r_outer"],
                                       sq["rotation_deg"], sq["width_ratio"])[0])
            inset = star.buffer(-INLAY_BORDER * float(sq["r_outer"]), join_style="mitre")
            keep = inset.difference(Point(float(q["cx"]), float(q["cy"])).buffer(float(q["r_inner"]), quad_segs=32))
            shape = from_geom(keep.simplify(SIMPLIFY), Shape("polygon", {"points": ""}, None))
            if shape is not None:
                out[id(d)] = shape
            break
    return out


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


# A library head's badge (its largest center piece) within BADGE_MAX head radii and no further than
# BADGE_ROUND times its equal-area radius at its farthest corner (a hexagon: 1.10) is drawn as a circle out
# to that corner, the rose inside it, as our own heads (S56-17B, the owner's check); S48-17's large
# square middle (0.82, 1.23) stays as drawn.
BADGE_MAX = 0.5
BADGE_ROUND = 1.15


def _round_badge(eq: float, reach: float) -> bool:
    return eq <= BADGE_MAX and reach <= BADGE_ROUND * eq


def _badge(pieces) -> tuple[int, float, float, float, float] | None:
    """(index, x, y, equal-area radius, farthest-corner radius) of a library head's badge, head radii: its
    largest center piece; with none, a round piece (_round_badge) with its middle within DISC_NEAR of the
    head's (S80-22's hexagon badge is classed as points; the rose had fallen back to the photo's center,
    below it)."""
    def stats(pc):
        poly = Polygon(pc.rings[0])
        if not poly.is_valid or poly.area <= 0:
            return None
        pts = np.asarray(pc.rings[0], float)
        c = poly.centroid
        return poly.area, c.x, c.y, (poly.area / math.pi) ** 0.5, float(np.hypot(pts[:, 0] - c.x, pts[:, 1] - c.y).max())
    found = [(st, i) for i, pc in enumerate(pieces) if pc.role == "center" and pc.rings and (st := stats(pc))]
    if not found:
        found = [(st, i) for i, pc in enumerate(pieces) if pc.role != "head" and pc.rings and (st := stats(pc))
                 and _round_badge(st[3], st[4]) and math.hypot(st[1], st[2]) <= DISC_NEAR]
    if not found:
        return None
    (_, x, y, eq, reach), i = max(found)
    return i, x, y, eq, reach


def template_pieces(part: Part) -> list[tuple[Shape, int]]:
    """(shape, color group) per reference piece of a template part, in paint order; the
    shapes carry role (the piece's), z and kind but no category yet."""
    m = _template_matrix(part.params)
    groups = template_groups(part)
    pieces = _template(part).pieces
    # a piece drawn again later with the same outline is wholly hidden (some reference
    # files repeat an element): only the last copy is drawn, so no duplicate is written
    keys = [tuple(tuple(r.reshape(-1).tolist()) for r in pc.rings) for pc in pieces]
    # the badge (the largest center piece) is drawn as a circle on its middle, the rose inside it,
    # as our own heads draw it (S56-17B in the app, the owner's check: a hexagon badge)
    found = _badge(pieces)
    badge = found[0] if found else None
    out = []
    for i, pc in enumerate(pieces):
        if keys[i] in keys[i + 1:]:
            continue
        if i == badge:
            at = template_center(part)
            if at is not None and plain_disc(part) is None and _round_badge(found[3], found[4]):
                c = disc(*at)  # out to the badge's farthest corner: it covers all it did (the loops' ends under it)
                c.role, c.z = "center", TEMPLATE_Z + len(pieces)  # on top, as our own heads' centers (S80-22's petals)
                out.append((c, groups[pc.category]))
                continue
        geom = xor_all([Polygon(ring_).buffer(0) for ring_ in pc.rings])
        if geom is None:
            continue
        shape = from_geom(affine_transform(geom, m), Shape("polygon", {}, None, role=pc.role, z=TEMPLATE_Z + i))
        if shape is not None:
            out.append((shape, groups[pc.category]))
    return out


# R80: a library head that is only a pleated disc and one badge (A1-15's) is drawn as our own: a disc
# with a pleated edge all round and the badge on its middle, both at the pieces' own radii (sqrt(area /
# pi)); the traced outline was pleated only along the top and the badge sat low (S65-18, the owner's
# check).
def plain_disc(part: Part) -> tuple[float, float, int, int] | None:
    """(disc radius, badge radius, disc category, badge category) in canvas units for a template part
    made of one head piece and one center piece, or None."""
    pieces = _template(part).pieces
    if len(pieces) != 2 or sorted(pc.role for pc in pieces) != ["center", "head"]:
        return None
    head = next(pc for pc in pieces if pc.role == "head")
    badge = next(pc for pc in pieces if pc.role == "center")
    r = float(part.params["r"])
    rh = (Polygon(head.rings[0]).area / math.pi) ** 0.5 * r
    rc = (Polygon(badge.rings[0]).area / math.pi) ** 0.5 * r
    return rh, rc, head.category, badge.category


def template_center(part: Part) -> tuple[float, float, float] | None:
    """(cx, cy, r) for the rose of a template head: its largest center piece's own middle and
    radius (sqrt(area / pi)), or None. A badge off the head's middle (A1-15's, low) keeps its rose on
    it (S65-18, the owner's check); before, the head's middle and the piece's farthest reach."""
    from shapely.geometry import Polygon
    q = part.params
    plain = plain_disc(part)
    if plain is not None:  # R80: the badge on the disc's middle
        return float(q["cx"]), float(q["cy"]), plain[1]
    pieces = _template(part).pieces
    found = _badge(pieces)
    if found is not None:
        # a badge drawn as a circle out to its farthest corner (template_pieces, BADGE_*): the rose inside it
        _, dx, dy, eq, reach = found
        r = float(q["r"])
        rr = reach if _round_badge(eq, reach) else eq
        return float(q["cx"]) + dx * r, float(q["cy"]) + dy * r, rr * r
    disc_ = _center_disc(pieces)
    if disc_ is None:
        return None
    (dx, dy), dr = disc_
    r = float(q["r"])
    return float(q["cx"]) + dx * r, float(q["cy"]) + dy * r, dr * r


# R27: a library head whose round middle piece is classed as points, not center (S36-15's
# badge disc, 0.18 head radii above the head circle's middle): the rose sits on that disc. A
# piece is that disc when its outline fills >= DISC_FILL of the circle through its bounding
# box, its box is square within DISC_SQUARE, its radius is at most DISC_MAX_R head radii and its
# middle lies within DISC_NEAR head radii of the head's; the nearest such piece wins.
DISC_FILL, DISC_SQUARE, DISC_MAX_R, DISC_NEAR = 0.9, 0.1, 0.6, 0.35


def _center_disc(pieces) -> tuple[tuple[float, float], float] | None:
    """R27: ((x, y) middle, radius) of a template's round middle piece, in head radii, or None."""
    best = None
    for pc in pieces:
        if not pc.rings:
            continue
        ring_ = np.asarray(pc.rings[0], dtype=float)
        lo, hi = ring_.min(axis=0), ring_.max(axis=0)
        w, h = hi - lo
        if max(w, h) <= 0 or abs(w - h) > DISC_SQUARE * max(w, h):
            continue
        x, y = ring_[:, 0], ring_[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
        rad = max(w, h) / 2
        mid = (lo + hi) / 2
        if area < DISC_FILL * np.pi * rad * rad or rad > DISC_MAX_R or float(np.hypot(*mid)) > DISC_NEAR:
            continue
        if best is None or float(np.hypot(*mid)) < float(np.hypot(*best[0])):
            best = ((float(mid[0]), float(mid[1])), float(rad))
    return best


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


def _center_outline(colors: list[Shape], cx: float, cy: float) -> Shape | None:
    """The center the rose sits on, drawn as the rose-off outline (st_circel) when it is not a circle: a
    library head's hexagon badge (S56-17B in the app, the owner's check: the circle stuck out past it)."""
    at = Point(cx, cy)
    for s in reversed(colors):
        if s.role != "center" or s.tag == "circle":
            continue
        g = to_geom(s)
        if g.contains(at):
            return from_geom(g, Shape("polygon", {"points": ""}, None))
    return None


def build(model: RosetteModel, pieces: bool = True) -> list[Shape]:
    """Shapes for the writer. pieces: every color shape cut to its visible piece (H14, the
    written files); False keeps overlapping shapes whole where the shop's layer order
    already stacks them (`clip_occluded`), for seam-free fill renders such as the head fit."""
    pairs = color_shapes(model)
    shapes = [s for s, _ in pairs]
    colors = clip_visible(shapes) if pieces else clip_occluded(shapes)
    result = list(colors)
    result += outlines(colors)

    for p in model.parts:  # R76: the small roses carry the rose drawing too
        q = p.params
        if p.role == "points" and q.get("kind") == "dot" and int(q.get("rose", 0)):
            for k in range(int(q["count"])):
                x, y = polar(q["cx"], q["cy"], q["r_mid"], q["rotation_deg"] + 360.0 * k / int(q["count"]))
                result += place_rose(x, y, (1.0 if int(q.get("flat", 0)) else 0.8) * q["r_dot"])  # R77: S58-17's fill the hexagon
    center = _rose_center(model)
    if model.rose and center is not None:
        cx, cy, r, z = center
        rose = place_rose(cx, cy, r)  # the drawing's corners touch the center's circle (designer A10-15B; the owner's check)
        result += rose
        circel = _center_outline(colors, cx, cy) or disc(cx, cy, r)
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
    center_r = next((float(p.params["r"]) for p in model.parts if p.role == "center" and "r" in p.params), None)
    inlays = _inlays(model.parts)

    tail_groups: dict[tuple, list[Part]] = {}
    for p in model.parts:
        q = p.params
        if p.role != "tail_stripe" and not (p.role == "unknown" and p.z >= 60):
            head_parts.append(id(p))
        if p.role == "template" and plain_disc(p) is not None:  # R80
            rh, rc, kh, kc = plain_disc(p)
            grp = template_groups(p)
            for s, gid, role, z in ((pinked_circle(q["cx"], q["cy"], rh), grp[kh], "head", TEMPLATE_Z),
                                    (disc(q["cx"], q["cy"], rc), grp[kc], "center", TEMPLATE_Z + 1)):
                g = group(p, gid)
                s.category, s.pattern, s.kind, s.role, s.z = g.category, g.pattern, "color", role, z
                colors.append((s, p))
        elif p.role == "template":
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
        elif p.role == "points" and q.get("kind") == "side_fold":  # R76
            finish(p, side_folds(q["cx"], q["cy"], int(q["count"]), q["r_outer"], q["axis_deg"], int(q["side"]),
                                 bool(int(q.get("last", 0)))))
        elif p.role == "points" and q.get("kind") == "lobe":  # R77
            finish(p, lobes(q["cx"], q["cy"], int(q["count"]), q["r_tip"], q["rotation_deg"], str(q["profile"])))
        elif p.role == "points" and q.get("kind") == "dot":  # R76
            finish(p, dots(q["cx"], q["cy"], int(q["count"]), q["r_mid"], q["r_dot"], q["rotation_deg"],
                           bool(int(q.get("flat", 0)))))
        elif p.role == "points" and q.get("kind") == "chevron":  # R74
            finish(p, chevrons(q["cx"], q["cy"], int(q["count"]), q["r_outer"], q["r_outer"] - q["r_inner"],
                               q["fold_half"], q.get("fold_notch", 0.0), q["rotation_deg"]))
        elif p.role == "points" and q.get("kind") == "split_arm":  # R70
            finish(p, split_arms(q["cx"], q["cy"], int(q["count"]), q["r_outer"], q["axis_deg"], int(q["side"])))
        elif p.role == "points" and id(p) in inlays:  # an inlay: its star's points inset (S35-19)
            finish(p, [inlays[id(p)]])
        elif p.role == "points":
            ir, ew = q.get("inner_ratio"), q.get("edge_width")
            r_in, wr = _tucked_loop(q, center_r)
            finish(p, points_ring(q["cx"], q["cy"], q["kind"], int(q["count"]), r_in,
                                  q["r_outer"], q["rotation_deg"], wr,
                                  None if ir is None else float(ir), None if ew is None else float(ew),
                                  leaf=int(q.get("leaf", 0)), lens=int(q.get("lens", 0)),
                                  flower=int(q.get("flower", 0))))
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
    head_cy = next((float(p.params["cy"]) for p in model.parts if p.role in ("template", "head", "center")
                    and "cy" in p.params), None)
    head_circle = next(((float(p.params["cx"]), float(p.params["cy"]), float(p.params["r"])) for p in model.parts
                        if p.role in ("head", "template") and {"cx", "cy", "r"} <= set(p.params)), None)
    if head_circle is None:  # a head of rings only (A2-15): the largest ring is the head circle
        rings = [p for p in model.parts if p.role == "ring" and "r_outer" in p.params]
        if rings:
            big = max(rings, key=lambda p: float(p.params["r_outer"]))
            head_circle = (float(big.params["cx"]), float(big.params["cy"]), float(big.params["r_outer"]))
    for key, parts in tail_groups.items():
        top_y, bottom_y, tw, bw, spread = key[1], key[2], key[3], key[4], key[5]
        polys = tails(top_y, bottom_y, int(key[0]), tw, bw, spread, _parse_stripes(key[6]),
                      stripes_bottom=_parse_stripes(key[7]) or None)
        idx = []
        for p in parts:
            ti, si = int(p.params["tail_index"]), int(p.params["stripe_index"])
            try:
                polys[ti][si]
            except IndexError:
                raise ValueError(f"part {p.id}: tail {ti} stripe {si} out of range") from None
            idx.append((ti, si))
        cut = [_under_head(polys[ti][si], float(top_y), float(bottom_y), head_geom) for ti, si in idx]
        if (not all(c for _, c in cut) and int(key[0]) == 1 and head_cy is not None
                and head_cy < float(top_y)):
            # R26: a stripe top lies in a gap the head does not close (a library head drawn
            # smaller than the photo's: S36-15). The whole tail runs on up its side lines to the
            # lowest top (RAISE_STEP at a time, no higher than the head's center) at which the head
            # covers every stripe top, as the reference tails run up under the head; the gaps
            # between its shapes then show the stripes.
            y = float(top_y)
            while y - RAISE_STEP >= head_cy:  # the lowest top, step by step, the head closes
                y -= RAISE_STEP
                raised = [_raised_stripe(key, ti, si, y) for ti, si in idx]
                if any(r is None for r in raised):
                    break
                cut2 = [_under_head(r, y, float(bottom_y), head_geom) for r in raised]
                if all(c for _, c in cut2):
                    cut = cut2
                    break
        tail_z = None
        from_r = max((float(p.params.get("from_r") or 0.0) for p in parts), default=0.0)
        if from_r and head_circle is not None and int(key[0]) == 1:
            over = _tail_over_head(colors, head_parts, head_circle, from_r, key, idx, polys, float(bottom_y))
            if over is not None:
                cut, tail_z = over
        for p, (shape, covered) in zip(parts, cut):
            if not covered and TAIL_UNCOVERED_WARNING not in model.warnings:
                model.warnings.append(TAIL_UNCOVERED_WARNING)
            if shape is None:
                msg = TAIL_HIDDEN_WARNING.format(part=p.id)
                if msg not in model.warnings:
                    model.warnings.append(msg)
                continue
            finish(p, [shape])
            if tail_z is not None:
                shape.z = tail_z

    colors.sort(key=lambda t: t[0].z)
    return colors
