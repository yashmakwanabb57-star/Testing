import pytest

from rosette.build import build, color_shapes
from rosette.model import ColorGroup, Part, RosetteModel
from rosette.validate import validate
from rosette.writer import write_svg


def s35_like(rose=False):
    groups = [ColorGroup(1, (200, 30, 30), False, 0.1, 0.3, 1, 0.9),
              ColorGroup(2, (30, 30, 200), True, 0.9, 0.3, 2, 0.9),
              ColorGroup(3, (30, 200, 30), False, 0.1, 0.4, 3, 0.9)]
    parts = [
        Part("head", "head", {"cx": 300, "cy": 300, "r": 150, "pinked": 1}, 1, 10, 1.0),
        Part("pts", "points", {"cx": 300, "cy": 300, "kind": "star", "count": 12, "r_inner": 90,
                               "r_outer": 130, "rotation_deg": -90, "width_ratio": 0.5}, 2, 20, 1.0),
        Part("ctr", "center", {"cx": 300, "cy": 300, "r": 50}, 3, 50, 1.0),
    ]
    for i in range(3):
        parts.append(Part(f"t{i}", "tail_stripe", {
            "tail_index": i, "stripe_index": 0, "tails_count": 3, "top_y": 300, "bottom_y": 800,
            "top_width": 60, "bottom_width": 90, "spread_deg": 30, "stripes": "1"}, 1, 0, 1.0))
    return RosetteModel(parts, groups, rose=rose)


def test_build_s35_like_model_validates():
    for rose in (False, True):
        assert validate(write_svg(build(s35_like(rose)))) == []


def test_rose_adds_rose_and_circel():
    shapes = build(s35_like(True))
    assert any(s.kind == "rose" for s in shapes)
    assert sum(s.kind == "circel" for s in shapes) == 1
    assert not any(s.kind in ("rose", "circel") for s in build(s35_like(False)))


def test_unknown_part_becomes_flagged_disc():
    m = RosetteModel([Part("u", "unknown", {"cx": 100, "cy": 100, "r": 20}, 1, 5, 0.2)],
                     [ColorGroup(1, (0, 0, 0), False, 0, 1, 1, 1)])
    shapes = [s for s in build(m) if s.kind == "color"]
    assert len(shapes) == 1 and shapes[0].tag == "circle"
    build(m)
    assert len(m.warnings) == 1


def test_unassigned_group_raises():
    m = s35_like()
    m.groups[0].category = None
    with pytest.raises(ValueError):
        build(m)


def _layered_cat_at(model, pts):
    """Category seen at canvas points when the written SVG is drawn the way the shop
    draws it: Layer_1 (Color 1) first, then Colors 2, 3, 4 on top."""
    from tools.rasterize import DEFAULT_PALETTE, rasterize
    img = rasterize(write_svg(build(model)), 850, DEFAULT_PALETTE)
    pal = {c: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for c, h in DEFAULT_PALETTE.items()}
    out = []
    for x, y in pts:
        px = tuple(int(v) for v in img[y, x])
        out.append(min(pal, key=lambda c: sum((a - b) ** 2 for a, b in zip(pal[c], px))))
    return out


def test_color1_part_above_other_colors_stays_visible():
    # Color 1 is always painted first (Layer_1), so a Color 1 part meant to lie on
    # top (head over the tail tops, center over a star) would be hidden; the lower
    # parts are cut where such a part covers them (S28-15: Color 1 center under a
    # Color 3 star body; tail tops showing on the head).
    groups = [ColorGroup(1, (200, 30, 30), False, 0.1, 0.5, 1, 0.9),
              ColorGroup(2, (30, 30, 200), False, 0.1, 0.2, 2, 0.9),
              ColorGroup(3, (30, 200, 30), False, 0.1, 0.3, 3, 0.9)]
    parts = [
        Part("head", "head", {"cx": 300, "cy": 200, "r": 150, "pinked": 0}, 1, 10, 1.0),
        Part("pts", "points", {"cx": 300, "cy": 200, "kind": "star", "count": 6, "r_inner": 80,
                               "r_outer": 130, "rotation_deg": -90, "width_ratio": 0.5}, 3, 20, 1.0),
        Part("ctr", "center", {"cx": 300, "cy": 200, "r": 40}, 1, 50, 1.0),
        Part("t0", "tail_stripe", {"tail_index": 0, "stripe_index": 0, "tails_count": 1, "top_y": 200,
                                   "bottom_y": 800, "top_width": 120, "bottom_width": 160,
                                   "spread_deg": 0, "stripes": "1"}, 2, 0, 1.0),
    ]
    m = RosetteModel(parts, groups)
    center, star_body, head_over_tail, tail = (300, 200), (300, 140), (300, 330), (300, 600)
    assert _layered_cat_at(m, [center, star_body, head_over_tail, tail]) == [1, 3, 1, 2]
    assert validate(write_svg(build(m))) == []


def test_clip_only_by_strictly_higher_parts():
    from rosette.geometry import disc
    from rosette.geometry.clip import clip_occluded
    low = disc(300, 300, 50)
    low.category, low.z = 2, 20
    same = disc(300, 300, 20)
    same.category, same.z = 1, 20
    out = clip_occluded([low, same])
    assert out[0] is low  # equal z: no stacking intent, nothing cut


def test_clip_rejects_unknown_color_shape():
    from rosette.geometry.clip import clip_occluded
    from rosette.shapes import Shape
    odd = Shape("path", {"d": "M0,0 L10,0 L10,10z"}, 2, z=0)
    top = Shape("circle", {"cx": "5", "cy": "5", "r": "3"}, 1, z=5)
    with pytest.raises(ValueError):
        clip_occluded([odd, top])


def _tail_tops(shapes):
    from rosette.geometry.clip import to_geom
    out = []
    for s in shapes:
        if s.kind == "color" and s.role == "tail_stripe":
            g = to_geom(s)
            xs, ys = g.exterior.coords.xy if hasattr(g, "exterior") else (None, None)
            i = min(range(len(ys)), key=lambda k: ys[k])
            out.append((xs[i], ys[i]))
    return out


def ringed(top_y, ring_out=150.0):
    groups = [ColorGroup(1, (200, 30, 30), False, 0.1, 0.3, 1, 0.9),
              ColorGroup(2, (30, 30, 200), False, 0.9, 0.3, 2, 0.9)]
    parts = [Part("ring", "ring", {"cx": 300, "cy": 200, "r_outer": ring_out, "r_inner": 60}, 2, 30, 1.0),
             Part("ctr", "center", {"cx": 300, "cy": 200, "r": 60}, 2, 50, 1.0)]  # tails Color 1: not cut
    for i in range(2):
        parts.append(Part(f"t{i}", "tail_stripe", {
            "tail_index": 0, "stripe_index": i, "tails_count": 1, "top_y": top_y, "bottom_y": 800,
            "top_width": 120, "bottom_width": 160, "spread_deg": 0, "stripes": "1,1"}, 1, 0, 1.0))
    return RosetteModel(parts, groups)


@pytest.mark.parametrize("tail_color, head_color", [(2, 1), (1, 2), (1, 1)])
def test_tail_stripes_never_show_inside_the_head(tail_color, head_color):
    # H5 (A7-15): the shop paints Layer_1 first and draws the st6 line art over
    # everything: a stripe painted after the head is cut to the head's outside, and no
    # stripe outline runs inside the head whatever the layers
    from shapely import Point
    from rosette.geometry.clip import to_geom
    groups = [ColorGroup(1, (200, 30, 30), False, 0.1, 0.3, 1, 0.9),
              ColorGroup(2, (30, 30, 200), False, 0.9, 0.3, 2, 0.9)]
    parts = [Part("head", "head", {"cx": 300, "cy": 200, "r": 120, "pinked": 1}, head_color, 10, 1.0)]
    for i in range(3):
        parts.append(Part(f"t{i}", "tail_stripe", {
            "tail_index": 0, "stripe_index": i, "tails_count": 1, "top_y": 300, "bottom_y": 800,
            "top_width": 120, "bottom_width": 160, "spread_deg": 0, "stripes": "1,1,1"}, tail_color, 0, 1.0))
    shapes = build(RosetteModel(parts, groups))
    head = to_geom(next(s for s in shapes if s.kind == "color" and s.role == "head")).buffer(-0.5)
    fills = [s for s in shapes if s.role == "tail_stripe" and s.kind == "color"]
    assert len(fills) == 3
    if tail_color > head_color:
        assert all(to_geom(s).intersection(head).area < 1e-6 for s in fills)
    lines = [s for s in shapes if s.role == "tail_stripe" and s.kind == "outline"]
    pts = []
    for s in lines:  # vertices and segment midpoints
        v = [tuple(map(float, p.split(","))) for p in s.attrs["points"].split()]
        pts += v + [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(v, v[1:])]
    assert pts and not any(head.contains(Point(p)) for p in pts)
    assert validate(write_svg(shapes)) == []


def _union_above(shapes, top_y):
    from shapely import box, unary_union
    from rosette.geometry.clip import to_geom
    head = unary_union([to_geom(s) for s in shapes if s.kind == "color" and s.role != "tail_stripe"])
    return head, box(0, 0, 600, top_y)


def kites_and_loops(top_y, stripe_color=1):
    """A center disc, four loops on the diagonals and four kites on the axes (one
    pointing down at the tail), all Color 2; one tail of three stripes."""
    groups = [ColorGroup(1, (200, 30, 30), False, 0.1, 0.3, 1, 0.9),
              ColorGroup(2, (30, 30, 200), False, 0.9, 0.3, 2, 0.9)]
    parts = [Part("ctr", "center", {"cx": 300, "cy": 200, "r": 60}, 2, 50, 1.0),
             Part("loops", "points", {"cx": 300, "cy": 200, "kind": "bow", "count": 4, "r_inner": 60,
                                      "r_outer": 140, "rotation_deg": 45, "width_ratio": 0.9}, 2, 21, 1.0),
             Part("kites", "points", {"cx": 300, "cy": 200, "kind": "diamond", "count": 4, "r_inner": 60,
                                      "r_outer": 150, "rotation_deg": 90, "width_ratio": 0.3}, 2, 22, 1.0)]
    for i in range(3):
        parts.append(Part(f"t{i}", "tail_stripe", {
            "tail_index": 0, "stripe_index": i, "tails_count": 1, "top_y": top_y, "bottom_y": 800,
            "top_width": 120, "bottom_width": 160, "spread_deg": 0, "stripes": "1,1,1"}, stripe_color, 0, 1.0))
    return RosetteModel(parts, groups)


def _on_side(p, top_y=320):
    """On one of the stripe side lines of kites_and_loops' tail (x at top_y / at 800)."""
    t = (p[1] - top_y) / (800 - top_y)
    return any(abs(p[0] - (xt + (xb - xt) * t)) < 0.5
               for xt, xb in ((240, 220), (280, 300 - 80 / 3), (320, 300 + 80 / 3), (360, 380)))


def test_tail_top_follows_the_heads_lower_outline():
    # H5 fix round 1 (S35-19): above top_y a stripe runs up its side lines into the head's
    # shadow (everything at or below some head shape), so in the gaps between the kite and
    # the loops it reaches the shapes' lower edges; no free top edge, no flat edge at top_y
    from shapely import Point
    m = kites_and_loops(top_y=320)
    shapes = build(m)
    head, above = _union_above(shapes, 320)
    lines = [s for s in shapes if s.role == "tail_stripe" and s.kind == "outline"]
    edge = head.boundary
    for s in lines:
        v = [tuple(map(float, p.split(","))) for p in s.attrs["points"].split()]
        for a, b in zip(v, v[1:]):
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            if mid[1] < 319.5:  # above top_y: on the head's outline or a side line
                assert edge.distance(Point(mid)) < 0.5 or (_on_side(a) and _on_side(b)), (a, b)
            assert not (abs(a[1] - 320) < 1e-6 and abs(b[1] - 320) < 1e-6), (a, b)  # no flat top
    from rosette.geometry.clip import to_geom
    fills = [to_geom(s) for s in shapes if s.role == "tail_stripe" and s.kind == "color"]
    gap = sum(f.intersection(above).difference(head).area for f in fills)
    assert gap > 10  # the stripes fill the gaps below the head's lower outline
    # and stop at the head's outline, as the reference tails do: nothing of a stripe lies
    # under a head shape (outline pages draw every edge, hidden or not)
    assert sum(f.intersection(head).area for f in fills) < 1.0  # simplify slivers
    assert not m.warnings
    assert validate(write_svg(shapes)) == []


def test_color2_stripe_ends_along_a_color1_pinked_disc():
    # A7-15: a Color 2 stripe under a Color 1 pinked head disc painted before it ends
    # along the disc's scallop, also when top_y lies below the disc
    from rosette.geometry.clip import to_geom
    for top_y in (300, 345):
        m = ringed(top_y=top_y)
        m.parts[0] = Part("head", "head", {"cx": 300, "cy": 200, "r": 130, "pinked": 1}, 1, 10, 1.0)
        for q in m.parts[2:]:
            q.color_group = 2
        shapes = build(m)
        disc = to_geom(next(s for s in shapes if s.kind == "color" and s.role == "head"))
        fills = [to_geom(s) for s in shapes if s.role == "tail_stripe" and s.kind == "color"]
        assert all(f.intersection(disc).area < 1e-3 for f in fills)
        assert min(f.bounds[1] for f in fills) < 330  # reaches up to the scallop
        assert all(f.distance(disc) < 1e-3 for f in fills)
        assert not m.warnings


def test_tail_with_nothing_over_it_keeps_a_flat_top_and_warns():
    from rosette.build import TAIL_UNCOVERED_WARNING
    m = ringed(top_y=340)
    m.parts = m.parts[2:]
    shapes = build(m)
    tops = [min(float(p.split(",")[1]) for p in s.attrs["points"].split())
            for s in shapes if s.role == "tail_stripe" and s.kind == "color"]
    assert tops == [340, 340]
    assert TAIL_UNCOVERED_WARNING in m.warnings



def test_a_stripe_narrowing_downward_keeps_its_bottom():
    # a constant-width ribbon on a widening tail takes a smaller share at the bottom: its
    # side lines meet below the tail, and the stripe must still end at bottom_y
    from rosette.geometry.clip import to_geom
    m = kites_and_loops(top_y=320)
    for q in m.parts[3:]:
        q.params["stripes_bottom"] = "2,1,2"
    shapes = build(m)
    fills = [to_geom(s) for s in shapes if s.role == "tail_stripe" and s.kind == "color"]
    assert max(f.bounds[3] for f in fills) <= 800 + 1e-6
