import math

from lxml import etree

from rosette.geometry import (disc, outlines, pinked_circle, place_rose, points_ring,
                              ring, tails)
from rosette.geometry.common import area
from rosette.shapes import Shape
from rosette.validate import validate
from rosette.writer import write_svg


def pts_of(shape):
    v = [float(x) for x in shape.attrs["points"].replace(",", " ").split()]
    return list(zip(v[0::2], v[1::2]))


def test_pinked_circle_closed_and_tooth():
    s = pinked_circle(300, 167.372, 100)
    assert s.tag == "polygon" and s.category is None
    pts = pts_of(s)
    tips = pts[0::2]
    d = math.dist(tips[0], tips[1])
    assert abs(d - 5.383) / 5.383 < 0.05
    assert all(abs(math.dist(p, (300, 167.372)) - 100) < 0.01 for p in tips)
    svg = write_svg([Shape(s.tag, s.attrs, 1)])
    assert validate(svg, simulate_check=False) == []


def test_ring_evenodd_two_subpaths():
    s = ring(300, 167, 80, 60)
    assert s.tag == "path" and s.attrs["fill-rule"] == "evenodd"
    assert s.attrs["d"].count("M") == 2 and s.attrs["d"].count("z") == 2
    assert disc(300, 167, 42.53).attrs["r"] == "42.53"


def test_points_ring_rotational_symmetry():
    for kind in ("diamond", "bow", "petal"):
        shapes = points_ring(300, 167, kind, 8, 50, 90, 10, 0.5)
        assert len(shapes) == 8
        for k, s in enumerate(shapes):
            p = pts_of(s)
            mx = sum(x for x, _ in p) / len(p)
            my = sum(y for _, y in p) / len(p)
            ang = math.degrees(math.atan2(my - 167, mx - 300)) % 360
            want = (10 + k * 45) % 360
            diff = abs((ang - want + 180) % 360 - 180)
            assert diff < 0.5, (kind, k, ang, want)
    star = points_ring(300, 167, "star", 8, 50, 90, 0, 0.5)
    assert len(star) == 1 and len(pts_of(star[0])) == 16


def make_tails():
    return tails(200, 820, 5, 60, 90, 40, [0.3, 0.4, 0.3])


def test_tails_mirror_symmetry():
    t = make_tails()
    assert len(t) == 5 and all(len(row) == 3 for row in t)
    for k in range(5):
        for i in range(3):
            o = pts_of(t[k][i])
            m = pts_of(t[4 - k][2 - i])
            assert len(o) == len(m)
            mir = [(600 - x, y) for x, y in o]
            exp = [mir[1], mir[0]] + mir[2:][::-1]
            for (ex, ey), (mx, my) in zip(exp, m):
                assert abs(ex - mx) < 0.01 and abs(ey - my) < 0.01
    assert t[0][0].category is None


def test_tails_index_zero_leftmost():
    t = make_tails()
    means = [sum(x for s in row for x, _ in pts_of(s)) / sum(len(pts_of(s)) for s in row)
             for row in t]
    assert means[0] < 300
    assert all(a < b for a, b in zip(means, means[1:]))


def test_tails_bottoms_on_one_line():
    for row in make_tails():
        for s in row:
            for x, y in pts_of(s)[2:]:
                assert min(abs(y - 820), abs(y - (820 - 5.383))) < 0.01


def test_tails_stripes_cover_width():
    t = tails(200, 820, 3, 60, 90, 30, [0.25, 0.5, 0.25])
    total = sum(area(pts_of(s)) for s in t[1])
    trap = (60 + 90) / 2 * 620
    assert abs(total - trap) / trap < 0.01


def test_all_inside_canvas():
    shapes = [pinked_circle(300, 167.372, 140), ring(300, 167, 120, 100), disc(300, 167, 42)]
    shapes += points_ring(300, 167, "petal", 12, 45, 95, 0, 0.6)
    shapes += [s for row in make_tails() for s in row]
    for s in shapes:
        if s.tag == "polygon":
            for x, y in pts_of(s):
                assert 0 <= x <= 600 and 0 <= y <= 850
        elif s.tag == "circle":
            r = float(s.attrs["r"])
            assert 0 <= float(s.attrs["cx"]) - r and float(s.attrs["cx"]) + r <= 600
    svg = write_svg([Shape(s.tag, s.attrs, 1, role=str(i)) for i, s in enumerate(shapes)])
    assert validate(svg, simulate_check=False) == []


def test_outlines():
    base = [pinked_circle(300, 167, 100), ring(300, 167, 80, 60), disc(300, 167, 40)]
    out = outlines(base)
    assert len(out) == 3 and all(o.kind == "outline" for o in out)
    p = out[0].attrs["points"].split()
    assert p[0] == p[-1]


def test_place_rose_scaled():
    shapes = place_rose(300, 167.372, 40)
    assert len(shapes) == 3 and all(s.kind == "rose" for s in shapes)
    from rosette.geometry.rose import _NUM, _path_points
    import re
    pts = []
    for s in shapes:
        a, b, c, d, e, f = [float(x) for x in re.search(r"matrix\((.*)\)", s.attrs["transform"]).group(1).split()]
        raw = _path_points(s.attrs["d"]) if "d" in s.attrs else None
        if raw is None:
            v = [float(x) for x in _NUM.findall(s.attrs["points"])]
            raw = list(zip(v[0::2], v[1::2]))
        pts += [(a * x + e, d * y + f) for x, y in raw]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    assert abs((max(xs) + min(xs)) / 2 - 300) < 0.5
    assert abs((max(ys) + min(ys)) / 2 - 167.372) < 0.5
    r = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) / 2
    assert abs(r - 40) / 40 < 0.05


def test_tails_stripes_bottom_fractions():
    # a centre ribbon of constant width on a widening tail: its share of the tail
    # differs at the top and at the bottom (S26-15, S46-15)
    (left, mid, right), = tails(200, 820, 1, 60, 120, 0, [1, 1, 1], stripes_bottom=[1, 1, 4])
    top = sorted(x for x, y in pts_of(mid) if abs(y - 200) < 1e-6)
    bottom = sorted(x for x, y in pts_of(mid) if abs(y - 820) < 1e-6)
    assert top == [290, 310]
    assert bottom[0] == 300 - 60 + 20 and bottom[-1] == 300 - 60 + 40
    assert validate(write_svg([Shape(s.tag, s.attrs, 1) for s in (left, mid, right)])) == []


def width_along_x(p, x):
    """Width across y of the polygon p at abscissa x (a lobe pointing along +x)."""
    ys = []
    for (x0, y0), (x1, y1) in zip(p, p[1:] + p[:1]):
        if min(x0, x1) <= x <= max(x0, x1) and x0 != x1:
            ys.append(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    return max(ys) - min(ys)


def test_bow_is_the_designer_loop_block():
    # Designer loops (H4 mining, 10 loop layers whose base is visible): the base at the
    # ring is 0.85 of the widest width, the block widens slowly to full width by 0.7
    # of its length and ends in a shallow convex curve: its corners lie 0.6 of a
    # straight cut's sag further from the head center than its axis end (21 layers).
    p = pts_of(points_ring(300, 167, "bow", 6, 50, 90, 0, 0.5)[0])
    w = 0.5 * 40
    assert abs(width_along_x(p, 350.01) - 0.85 * w) < 0.3
    assert abs(width_along_x(p, 370) - 0.96 * w) < 0.3
    assert abs(width_along_x(p, 380) - w) < 0.3
    assert max(x for x, _ in p) == 90 + 300  # the axis end at r_outer
    sag = math.hypot(90, 10) - 90
    corner = max(math.dist(q, (300, 167)) for q in p)
    assert abs(corner - (90 + 0.6 * sag)) < 1e-3
    outer = [(x, y) for x, y in p if x > 386]
    assert len(outer) >= 7  # a curve, not a straight cut


def test_diamond_is_the_designer_kite():
    # Designer kites (43 layers, 22 files): straight sides from pointed ends to the
    # widest point at 0.4 of the length.
    p = pts_of(points_ring(300, 167, "diamond", 6, 50, 90, 0, 0.5)[0])
    w = 0.5 * 40
    assert abs(width_along_x(p, 366) - w) < 0.3
    assert abs(width_along_x(p, 358) - 0.5 * w) < 0.3
    assert abs(width_along_x(p, 382) - w / 3) < 0.3
    xs = [x for x, _ in p]
    assert min(xs) == 350 and max(xs) == 390


def test_petal_is_the_designer_petal():
    # Designer petals (16 layers, 13 files): a wide inner end (usually under the center),
    # widest at half the length, then a straight taper to a pointed tip.
    p = pts_of(points_ring(300, 167, "petal", 6, 50, 90, 0, 0.5)[0])
    w = 0.5 * 40
    assert abs(width_along_x(p, 350.01) - 0.75 * w) < 0.3
    assert abs(width_along_x(p, 370) - w) < 0.3
    assert abs(width_along_x(p, 386) - 0.23 * w) < 0.5
    assert max(x for x, _ in p) == 390


def test_edge_strips_follow_the_loop_sides():
    # Two-tone loops (designer: 14 strip layers on loops in 13 files): a strip along each
    # side of every loop, edge_width wide, its center 0.19 strip widths outside the edge.
    host = pts_of(points_ring(300, 167, "bow", 6, 50, 90, 0, 0.5)[0])
    strips = points_ring(300, 167, "bow", 6, 50, 90, 0, 0.5, edge_width=2.0)
    assert len(strips) == 12
    a, b = (pts_of(s) for s in strips[:2])
    half = width_along_x(host, 370) / 2
    for p in (a, b):
        assert abs(width_along_x(p, 370) - 2.0) < 0.05
        ys = [y - 167 for x, y in p if abs(x - 370) < 3]
        assert ys
    lo = min(abs(y - 167) for x, y in a + b if abs(x - 370) < 1e-6)
    hi = max(abs(y - 167) for x, y in a + b if abs(x - 370) < 1e-6)
    assert abs(lo - (half - 0.31 * 2)) < 0.05 and abs(hi - (half + 0.69 * 2)) < 0.05


def test_edited_inner_ratio_changes_only_the_loop_base():
    # An edited inner_ratio sets the base width; the rest of the designer loop profile
    # and the convex outer end stay (the UI shows the drawn 0.85 when none is set).
    p = pts_of(points_ring(300, 167, "bow", 6, 50, 90, 0, 0.5, inner_ratio=0.6)[0])
    q = pts_of(points_ring(300, 167, "bow", 6, 50, 90, 0, 0.5)[0])
    same = pts_of(points_ring(300, 167, "bow", 6, 50, 90, 0, 0.5, inner_ratio=0.85)[0])
    w = 0.5 * 40
    assert same == q
    assert abs(width_along_x(p, 350.01) - 0.6 * w) < 0.3
    assert abs(width_along_x(p, 380) - w) < 0.3  # full width from 0.7 of the length on
    assert max(x for x, _ in p) == 390
    sag = math.hypot(90, 10) - 90
    assert abs(max(math.dist(v, (300, 167)) for v in p) - (90 + 0.6 * sag)) < 1e-3
    # between the base and full width it follows the loop profile's shape
    assert width_along_x(q, 360) > width_along_x(p, 360) > 0.6 * w


def test_tail_stripes_converge_to_a_point_above_top_y():
    # H5: every stripe's top ends in the tail's apex (no flat top edge); the stripes
    # still reach full width at top_y and keep their side lines and pinked bottom
    t = tails(200, 820, 3, 60, 90, 30, [0.3, 0.4, 0.3], tip_y=170)
    for k, row in enumerate(t):
        apexes = set()
        for s in row:
            p = pts_of(s)
            top = min(y for _, y in p)
            assert abs(top - 170) < 1e-6
            assert sum(abs(y - top) < 1e-6 for _, y in p) == 1  # one top point, no edge
            apexes.add(next((round(x, 3), round(y, 3)) for x, y in p if abs(y - top) < 1e-6))
            assert sum(abs(y - 200) < 1e-6 for _, y in p) == 2  # full width at top_y
        assert len(apexes) == 1  # the stripes of a tail share the apex: no gaps between them
    (mid,) = {next((x, y) for x, y in pts_of(s) if abs(y - 170) < 1e-6) for s in t[1]}
    assert abs(mid[0] - 300) < 1e-6
    total = sum(area(pts_of(s)) for s in t[1])
    trap = (60 + 90) / 2 * 620 + 60 * 30 / 2
    assert abs(total - trap) / trap < 0.01


def _outline_points(o):
    pts = [tuple(map(float, p.split(","))) for p in o.attrs["points"].split()]
    mids = [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(pts, pts[1:])]
    return pts + mids


def _poly(pts, cat, role):
    from rosette.geometry.common import points_attr
    return Shape("polygon", {"points": points_attr(pts)}, cat, role=role, kind="color")


def test_outline_traces_only_edges_no_later_shape_hides():
    # H5 (A7-15): st6 line art is drawn over everything, so an outline must leave out
    # the edges that a shape painted later (same Color: later in the layer) covers
    from shapely import Point, Polygon
    lower = _poly([(0, 0), (100, 0), (100, 100), (0, 100)], 1, "lower")
    upper = _poly([(50, 50), (150, 50), (150, 150), (50, 150)], 1, "upper")
    out = outlines([lower, upper])
    inner = Polygon([(50, 50), (150, 50), (150, 150), (50, 150)]).buffer(-0.5)
    pts = [p for o in out if o.role == "lower" for p in _outline_points(o)]
    assert pts and not any(inner.contains(Point(p)) for p in pts)
    assert any(abs(x - 100) < 1e-6 and 0 < y < 50 for x, y in pts)  # its visible edges stay
    up = [o for o in out if o.role == "upper"]
    assert len(up) == 1 and Polygon(_outline_points(up[0])[:5]).area == 10000  # whole outline
    # the shop paints Layer_1 (Color 1) before Layer_2_1: a Color 2 shape listed first
    # still hides the Color 1 one
    up2 = _poly([(50, 50), (150, 50), (150, 150), (50, 150)], 2, "upper")
    out = outlines([up2, lower])
    assert not [o for o in out if o.role == "lower" and any(inner.contains(Point(p)) for p in _outline_points(o))]


def test_stripe_outline_stops_at_a_pinked_head_painted_later():
    from shapely import Point
    from rosette.geometry.clip import to_geom
    head = pinked_circle(300, 200, 120)
    head.category, head.kind, head.role = 1, "color", "head"
    t = tails(250, 800, 1, 120, 160, 0, [1, 1, 1], tip_y=230)
    stripes = [s for s in t[0]]
    for s in stripes:
        s.category, s.kind, s.role = 1, "color", "tail_stripe"
    out = outlines(stripes + [head])
    inner = to_geom(head).buffer(-0.5)
    pts = [p for o in out if o.role == "tail_stripe" for p in _outline_points(o)]
    assert pts and not any(inner.contains(Point(p)) for p in pts)
