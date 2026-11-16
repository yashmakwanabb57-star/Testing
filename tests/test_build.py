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


def test_clip_at_equal_z_follows_the_paint_order():
    from rosette.geometry import disc
    from rosette.geometry.clip import clip_visible
    low = disc(300, 300, 50)
    low.category, low.z = 2, 20
    same = disc(300, 300, 20)
    same.category, same.z = 1, 20
    out = clip_visible([low, same])
    assert out[0] is low  # equal z: the shop paints Color 2 last, so it shows whole
    assert len(out) == 1  # and the Color 1 disc under it is hidden


def test_clip_rejects_unknown_color_shape():
    from rosette.geometry.clip import clip_visible
    from rosette.shapes import Shape
    odd = Shape("path", {"d": "M0,0 L10,0 L10,10z"}, 2, z=0)
    top = Shape("circle", {"cx": "5", "cy": "5", "r": "3"}, 1, z=5)
    with pytest.raises(ValueError):
        clip_visible([odd, top])


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



def _tail_tops(shapes):
    return [min(float(p.split(",")[1]) for p in s.attrs["points"].split())
            for s in shapes if s.role == "tail_stripe" and s.kind == "color"]


def test_head_beside_the_tail_leaves_its_top_flat_and_warns():
    # a head is present but does not close the stripes' top edge (it lies off the tail
    # axis): the tops stay flat at top_y and the model warns
    from rosette.build import TAIL_UNCOVERED_WARNING
    m = ringed(top_y=340)
    for q in m.parts[:2]:
        q.params["cx"] = 100
    shapes = build(m)
    assert _tail_tops(shapes) == [340, 340]
    assert TAIL_UNCOVERED_WARNING in m.warnings


def test_off_grid_top_y_with_the_head_below_still_warns():
    # `up` from an off-grid top_y: the extension snapped to the 0.001 grid must not
    # count as a gap closed above by the head
    from rosette.build import TAIL_UNCOVERED_WARNING
    m = ringed(top_y=100.3337)
    for q in m.parts[:2]:
        q.params["cy"] = 400  # the head lies below the tail top
    build(m)
    assert TAIL_UNCOVERED_WARNING in m.warnings


def test_stripe_wholly_inside_the_head_is_dropped_and_warned():
    # cutting a stripe to the outside of the head leaves nothing: it is not drawn
    # (never restored uncut), and the model says so
    from rosette.build import TAIL_HIDDEN_WARNING
    m = ringed(top_y=340)
    m.parts[0].params.update(cy=560, r_outer=400, r_inner=60)  # a ring over the whole tail
    m.parts[1].params["cy"] = 560
    shapes = build(m)
    assert _tail_tops(shapes) == []
    assert any(TAIL_HIDDEN_WARNING.format(part="t0") == w for w in m.warnings)
    assert validate(write_svg(shapes)) == []


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


def test_raised_stripe_keeps_its_side_lines():
    # R26: a one-tail stripe moved up to a new top runs on along its own side lines
    from rosette.build import _raised_stripe
    from rosette.geometry.tails import tails
    key = (1, 315.0, 825.0, 166.0, 324.0, 0.0, "0.18,0.64,0.18", "0.33,0.34,0.33")
    old = tails(315.0, 825.0, 1, 166.0, 324.0, 0.0, [0.18, 0.64, 0.18], stripes_bottom=[0.33, 0.34, 0.33])[0][1]
    new = _raised_stripe(key, 0, 1, 155.0)
    pts_old = [tuple(float(v) for v in p.split(",")) for p in old.attrs["points"].split()]
    pts_new = [tuple(float(v) for v in p.split(",")) for p in new.attrs["points"].split()]
    (tl, tr, br, bl), (ntl, ntr, nbr, nbl) = pts_old[:2] + [pts_old[2], pts_old[-1]], pts_new[:2] + [pts_new[2], pts_new[-1]]
    assert ntl[1] == 155.0 and ntr[1] == 155.0
    assert (nbl, nbr) == (bl, br)  # same bottom
    for top, bot, ntop in ((tl, bl, ntl), (tr, br, ntr)):  # the old top lies on the new side line
        x_at = ntop[0] + (bot[0] - ntop[0]) * (top[1] - ntop[1]) / (bot[1] - ntop[1])
        assert abs(x_at - top[0]) < 1e-3  # points are written to 3 decimals


def test_raised_stripe_refuses_when_a_stripe_vanishes():
    from rosette.build import _raised_stripe
    key = (1, 315.0, 825.0, 166.0, 324.0, 0.0, "0.02,0.96,0.02", "0.4,0.2,0.4")
    assert _raised_stripe(key, 0, 0, 25.0) is None


@pytest.mark.slow
def test_s36_15_tail_runs_up_under_a_smaller_library_head(monkeypatch):
    monkeypatch.setenv("ROSETTE_READER", "off")
    monkeypatch.setenv("ROSETTE_SAM2", "off")
    from pathlib import Path
    from rosette.pipeline import analyze_with_context, render
    from rosette.vision.templates import design_key
    photo = Path(__file__).resolve().parent.parent / "reference" / "photos" / "S36-15.png"
    model, _, _ = analyze_with_context(photo.read_bytes(), exclude_key=design_key("S36-15"))
    render(model)
    assert not any("Tail top is not under" in w for w in model.warnings)


def test_rose_sits_on_a_library_heads_round_middle_piece():
    # R27: S36-15's library head has no center piece; its badge disc (classed as points)
    # lies 0.18 head radii above the head circle's middle: the rose goes on that disc
    from rosette.build import template_center
    from rosette.model import Part
    t = Part("template", "template", {"template_id": "S36-15", "cx": 300.0, "cy": 200.0, "r": 100.0,
                                      "rotation_deg": 0.0}, 0, 10, 0.9)
    cx, cy, r = template_center(t)
    assert cx == pytest.approx(301.3, abs=0.2) and cy == pytest.approx(181.9, abs=0.2)
    assert r == pytest.approx(37.0, abs=0.2)


def test_center_disc_ignores_petals_and_far_pieces():
    from rosette.build import _center_disc
    from types import SimpleNamespace
    import numpy as np
    th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    disc_ = SimpleNamespace(rings=[np.c_[0.3 * np.cos(th), 0.3 * np.sin(th) - 0.1]])
    petal = SimpleNamespace(rings=[np.array([[0.0, 0.0], [0.6, 0.1], [0.0, 0.2]])])
    far = SimpleNamespace(rings=[np.c_[0.1 * np.cos(th) + 0.7, 0.1 * np.sin(th)]])
    (mx, my), r = _center_disc([petal, far, disc_])
    assert (round(mx, 3), round(my, 3), round(r, 3)) == (0.0, -0.1, 0.3)
    assert _center_disc([petal, far]) is None
