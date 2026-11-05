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
