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


def test_tail_top_is_a_point_inside_the_head_at_the_mined_depth():
    from rosette.build import TAIL_TIP_DEPTH, TAIL_TIP_MIN
    for top_y in (400, 340):  # apex at the mined depth, or TAIL_TIP_MIN above a shallow top_y
        m = ringed(top_y=top_y)
        shapes = [t for t, _ in color_shapes(m)]  # before the cuts
        x, y = _tail_tops(shapes)[0]
        assert abs(x - 300) < 1e-6 and abs(y - min(200 + TAIL_TIP_DEPTH * 150, top_y - TAIL_TIP_MIN * 150)) < 0.5
    m = ringed(top_y=400)
    shapes = [t for t, _ in color_shapes(m)]  # before the cuts
    tops = _tail_tops(shapes)
    assert len(tops) == 2 and len({(round(x, 2), round(y, 2)) for x, y in tops}) == 1
    x, y = tops[0]
    assert abs(x - 300) < 1e-6 and abs(y - (200 + TAIL_TIP_DEPTH * 150)) < 0.5
    assert y < 400 - TAIL_TIP_MIN * 150 + 1e-6
    assert not m.warnings
    assert validate(write_svg(build(m))) == []


def test_tail_top_moves_to_the_nearest_covered_depth_and_warns():
    # kites on the diagonals leave the bottom of the head open below the center disc:
    # the mined depth is not covered, so the apex moves up to the disc and the model warns
    from rosette.build import TAIL_TIP_DEPTH
    m = ringed(top_y=340)
    m.parts[0] = Part("pts", "points", {"cx": 300, "cy": 200, "kind": "diamond", "count": 4, "r_inner": 50,
                                        "r_outer": 150, "rotation_deg": 45, "width_ratio": 0.5}, 2, 20, 1.0)
    shapes = [t for t, _ in color_shapes(m)]  # before the cuts
    x, y = _tail_tops(shapes)[0]
    assert abs(x - 300) < 1e-6 and 250 < y <= 260 + 1e-6 < 200 + TAIL_TIP_DEPTH * 150
    assert any("tail top" in w.lower() for w in m.warnings)
    assert validate(write_svg(build(m))) == []


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
    pts = [tuple(map(float, p.split(","))) for s in lines for p in s.attrs["points"].split()]
    assert pts and not any(head.contains(Point(p)) for p in pts)
    assert validate(write_svg(shapes)) == []
