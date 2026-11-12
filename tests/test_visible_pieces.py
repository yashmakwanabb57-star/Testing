"""H14: every color shape is drawn as its visible piece, cut by everything stacked above
it (higher z; equal z: painted later), so no outline crosses another, as in the
reference files."""
import re

import pytest
from lxml import etree
from shapely import Point

from rosette.build import build
from rosette.geometry import disc, ring
from rosette.geometry.clip import clip_visible, to_geom
from rosette.model import ColorGroup, Part, RosetteModel
from rosette.shapes import Shape
from rosette.validate import validate
from rosette.writer import write_svg

SVG = "{http://www.w3.org/2000/svg}"


def _sq(x0, y0, x1, y1, cat, z, role="r"):
    return Shape("polygon", {"points": f"{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"}, cat, role=role, z=z)


def _overlaps(shapes, tol=0.5):
    """(i, j, area) of color shape pairs whose fills overlap by more than `tol` units^2."""
    cs = [s for s in shapes if s.kind == "color"]
    gs = [to_geom(s) for s in cs]
    return [(i, j, gs[i].intersection(gs[j]).area) for i in range(len(gs)) for j in range(i + 1, len(gs))
            if gs[i].intersection(gs[j]).area > tol]


def groups3():
    return [ColorGroup(1, (200, 30, 30), False, 0.1, 0.3, 1, 0.9),
            ColorGroup(2, (30, 30, 200), True, 0.9, 0.3, 2, 0.9),
            ColorGroup(3, (30, 200, 30), False, 0.1, 0.4, 3, 0.9)]


def loops_under_star():
    """Color 1 loops (z 21) under a Color 2 star (z 25) on a Color 3 head disc, with a
    Color 3 center on top: the shop paints Color 1 first, so before H14 only the star
    and center cut nothing of the loops' outline."""
    parts = [
        Part("head", "head", {"cx": 300, "cy": 200, "r": 150, "pinked": 1}, 3, 10, 1.0),
        Part("loops", "points", {"cx": 300, "cy": 200, "kind": "bow", "count": 6, "r_inner": 40,
                                 "r_outer": 140, "rotation_deg": -90, "width_ratio": 0.45}, 1, 21, 1.0),
        Part("star", "points", {"cx": 300, "cy": 200, "kind": "star", "count": 6, "r_inner": 50,
                                "r_outer": 120, "rotation_deg": -60, "width_ratio": 0.5}, 2, 25, 1.0),
        Part("ctr", "center", {"cx": 300, "cy": 200, "r": 45}, 3, 50, 1.0),
    ]
    for i in range(2):
        parts.append(Part(f"t{i}", "tail_stripe", {
            "tail_index": i, "stripe_index": 0, "tails_count": 2, "top_y": 260, "bottom_y": 800,
            "top_width": 80, "bottom_width": 110, "spread_deg": 20, "stripes": "1"}, 2, 0, 1.0))
    return RosetteModel(parts, groups3())


def test_a_loop_under_a_star_is_cut_by_the_star():
    shapes = build(loops_under_star())
    star = next(to_geom(s) for s in shapes if s.kind == "color" and s.z == 25)
    loops = [to_geom(s) for s in shapes if s.kind == "color" and s.z == 21]
    assert loops and all(g.intersection(star).area < 0.5 for g in loops)
    # the loops still show between the star's arms
    assert sum(g.area for g in loops) > 1000


def test_no_two_color_shapes_overlap_after_build():
    assert _overlaps(build(loops_under_star())) == []


def test_build_with_visible_pieces_validates():
    assert validate(write_svg(build(loops_under_star()))) == []


def test_a_ring_under_a_center_keeps_its_hole():
    r = ring(300, 300, 100, 60)
    r.category, r.z, r.role = 2, 30, "ring"
    c = disc(300, 300, 70)
    c.category, c.z, c.role = 2, 50, "center"
    out = clip_visible([r, c])
    g = to_geom(out[0])
    assert not g.contains(Point(300, 300))  # the hole stays a hole
    assert g.area == pytest.approx(3.14159 * (100 ** 2 - 70 ** 2), rel=0.01)
    assert out[1] is c


def test_a_ring_nothing_covers_stays_a_ring_path():
    r = ring(300, 300, 100, 60)
    r.category, r.z = 2, 30
    c = disc(300, 300, 60)  # sits in the hole: touches, covers nothing
    c.category, c.z = 1, 50
    out = clip_visible([r, c])
    assert out[0] is r


def test_a_fully_covered_shape_disappears():
    low = _sq(10, 10, 20, 20, 3, 5, "low")
    top = _sq(0, 0, 50, 50, 1, 9, "top")
    out = clip_visible([low, top])
    assert [s.role for s in out] == ["top"]


def test_equal_z_is_cut_by_the_shape_painted_later():
    # the shop paints Color 1 before Color 2: at equal z the Color 2 shape shows
    a = _sq(0, 0, 100, 100, 2, 20, "a")
    b = _sq(50, 0, 150, 100, 1, 20, "b")
    out = clip_visible([a, b])
    gb = to_geom(next(s for s in out if s.role == "b"))
    assert gb.area == pytest.approx(5000)
    assert next(s for s in out if s.role == "a") is a
    # same Color, equal z: the later one in the file shows
    c = _sq(50, 0, 150, 100, 2, 20, "c")
    out = clip_visible([a, c])
    assert to_geom(out[0]).area == pytest.approx(5000) and out[1] is c


def test_higher_z_cuts_whatever_the_colors():
    low = _sq(0, 0, 100, 100, 1, 10, "low")  # Color 1 under Color 3: before H14 not cut
    top = _sq(50, 0, 150, 100, 3, 20, "top")
    out = clip_visible([low, top])
    assert to_geom(out[0]).area == pytest.approx(5000)


def test_pieces_are_valid_and_slivers_are_dropped():
    low = _sq(0, 0, 100, 100, 2, 10, "low")
    top = _sq(0.2, -1, 101, 101, 1, 20, "top")  # leaves a 0.2 wide sliver of `low`
    out = clip_visible([low, top])
    assert [s.role for s in out] == ["top"]
    for s in clip_visible([s for s in build(loops_under_star()) if s.kind == "color"]):
        assert to_geom(s).is_valid


def test_pattern_twin_is_the_solid_twins_geometry():
    svg = write_svg(build(loops_under_star()))
    root = etree.fromstring(svg.encode())
    def els(gid):
        g = next(e for e in root.iter() if e.get("id") == gid)
        return [(e.tag, e.get("d") or e.get("points")) for e in g if isinstance(e.tag, str)
                and e.tag != f"{SVG}defs"]
    assert els("Layer_3_2") == els("Layer_2_1_2") and els("Layer_3_2")
    assert els("Layer_3_3") == els("Layer_2_1_3") and els("Layer_3_3")


def _z_order_render(shapes):
    """The intended picture: whole shapes painted by z (equal z: Color, then file order)."""
    import io

    import numpy as np
    import resvg_py
    from PIL import Image
    from tools.rasterize import DEFAULT_PALETTE
    body = []
    for _, _, _, s in sorted((s.z, s.category, i, s) for i, s in enumerate(shapes)):
        attrs = " ".join(f'{k}="{v}"' for k, v in s.attrs.items())
        body.append(f'<{s.tag} {attrs} fill="{DEFAULT_PALETTE[s.category]}"/>')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 850">' + "".join(body) + "</svg>")
    png = resvg_py.svg_to_bytes(svg_string=svg, height=850, background="#ffffff")
    return np.asarray(Image.open(io.BytesIO(bytes(png))).convert("RGB"))


def test_colored_rendering_is_the_intended_stacking():
    # cut shapes in the shop's layer order show what the whole shapes stacked by z show
    import numpy as np
    from rosette.build import color_shapes
    from tools.rasterize import DEFAULT_PALETTE, rasterize
    m = loops_under_star()
    a = _z_order_render([s for s, _ in color_shapes(m)])
    b = rasterize(write_svg([s for s in build(loops_under_star()) if s.kind == "color"]), 850, DEFAULT_PALETTE)
    # (antialiased seams between abutting pieces blend the two colors: not a color change)
    assert np.mean(np.abs(a.astype(int) - b.astype(int)).sum(axis=2) > 200) < 0.0005


def test_written_pieces_are_simple_rings():
    # cut pieces are snapped to the 3-decimal grid before they are written, so no ring
    # crosses or touches itself after rounding
    from shapely import LinearRing
    from rosette.geometry.clip import _NUM
    svg = write_svg(build(loops_under_star()))
    root = etree.fromstring(svg.encode())
    n = 0
    for e in root.iter():
        if not isinstance(e.tag, str) or not re.match(r"st\d+_\d", e.get("class") or ""):
            continue
        subs = [e.get("points")] if e.get("points") else (e.get("d") or "").split("M")[1:]
        for sub in subs:
            if sub and "a" not in sub:
                v = [float(x) for x in re.findall(_NUM, sub)]
                assert LinearRing(list(zip(v[0::2], v[1::2]))).is_simple
                n += 1
    assert n > 10


def test_xor_all_is_the_even_odd_fill():
    from shapely import Polygon
    from rosette.geometry.clip import xor_all
    sq = [Polygon([(-k, -k), (k, -k), (k, k), (-k, k)]) for k in (10, 8, 6, 4, 2)]
    assert xor_all(sq).area == pytest.approx(400 - 256 + 144 - 64 + 16)
    assert xor_all([]) is None
