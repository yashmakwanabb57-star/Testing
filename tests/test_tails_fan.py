import numpy as np
import pytest

from rosette.build import build
from rosette.model import ColorGroup, RosetteModel
from rosette.validate import validate
from rosette.vision.parts import detect_parts
from rosette.vision.prepare import Prepared
from rosette.writer import write_svg

COLORS = {0: (30, 50, 160), 1: (160, 188, 205), 2: (200, 40, 40), 3: (240, 200, 60)}


def photo(boundaries, labels_lr, head_label=0):
    """Head disc and one tail whose stripe boundaries are straight lines x = x0 + s*(y-551)
    (boundaries: [(x0, s)] left to right, sides included), stripes labelled labels_lr."""
    h, w = 1200, 800
    yy, xx = np.mgrid[:h, :w]
    labels = np.full((h, w), -1, np.int32)
    head = (xx - 400) ** 2 + (yy - 300) ** 2 <= 250 ** 2
    labels[head] = head_label
    rows = (yy > 550) & (yy < 1150)
    for (x0, s0), (x1, s1), g in zip(boundaries, boundaries[1:], labels_lr):
        labels[rows & (xx >= x0 + s0 * (yy - 551)) & (xx < x1 + s1 * (yy - 551))] = g
    rgb = np.full((h, w, 3), 255, np.uint8)
    for g, c in COLORS.items():
        rgb[labels == g] = c
    p = Prepared(rgb=rgb, mask=labels >= 0, axis_x=400.0, head=(400.0, 300.0, 250.0),
                 tails_top_y=551.0, bottom_y=1149.0, scale=(1149 - 50) / 800)
    groups = [ColorGroup(g, c, False, 1.0, 0.2, None, 0.0) for g, c in COLORS.items()]
    return p, labels, groups


def tails_of(parts):
    return sorted((q for q in parts if q.role == "tail_stripe"),
                  key=lambda q: (q.params["tail_index"], q.params["stripe_index"]))


def test_asymmetric_diverging_ribbons_become_a_fan():
    # A13-15B: light | glitter | royal, the left inner boundary straight down, the right
    # one leaning out like the sides: three ribbons fanning out, not one striped tail
    b = [(250, -0.16), (350, 0.0), (430, 0.14), (550, 0.16)]
    p, labels, groups = photo(b, [1, 2, 3])
    t = tails_of(detect_parts(p, labels, groups))
    assert [int(q.params["tails_count"]) for q in t] == [3, 3, 3]
    assert [q.color_group for q in t] == [1, 2, 3]
    assert all(q.params["stripes"] == "1.0000" for q in t)
    assert float(t[0].params["spread_deg"]) > 10
    assert t[0].params["top_width"] == t[0].params["bottom_width"]
    # the fan builds and validates: three tails, each with its own pinked bottom
    for i, g in enumerate(groups):
        g.category = 1 + i % 4
    model = RosetteModel(parts=detect_parts(p, labels, groups), groups=groups)
    shapes = build(model)
    fills = [s for s in shapes if s.role == "tail_stripe" and s.kind == "color"]
    assert len(fills) == 3
    from shapely import Polygon, unary_union
    from rosette.geometry.clip import to_geom
    # tails meet, no gap between them (H14: they are visible pieces that abut, so they are
    # merged within 0.01 units)
    union = unary_union([to_geom(s).buffer(0.01) for s in fills])
    assert isinstance(union, Polygon) and not union.interiors
    bottoms = [max(float(v.split(",")[1]) for v in s.attrs["points"].split()) for s in fills]
    assert bottoms == [pytest.approx(bottoms[0])] * 3  # every tail ends on the same pinked line
    assert validate(write_svg(shapes)) == []


def test_mirror_symmetric_fan_stays_one_striped_tail():
    # outer tails fanning over a constant-width center ribbon: the designer's usual tail
    b = [(250, -0.16), (330, -0.08), (470, 0.08), (550, 0.16)]
    p, labels, groups = photo(b, [1, 2, 1])
    t = tails_of(detect_parts(p, labels, groups))
    assert {int(q.params["tails_count"]) for q in t} == {1}
    assert len(t) == 3 and min(q.confidence for q in t) >= 0.6


def test_asymmetric_tail_with_repeating_colors_is_flagged_not_fanned():
    b = [(250, -0.16), (330, 0.0), (430, 0.14), (550, 0.16)]
    p, labels, groups = photo(b, [1, 2, 1])
    t = tails_of(detect_parts(p, labels, groups))
    assert {int(q.params["tails_count"]) for q in t} == {1}
    assert max(q.confidence for q in t) < 0.6


def test_parallel_stripes_stay_one_tail():
    b = [(250, 0.0), (350, 0.0), (430, 0.0), (550, 0.0)]
    p, labels, groups = photo(b, [1, 2, 3])
    t = tails_of(detect_parts(p, labels, groups))
    assert {int(q.params["tails_count"]) for q in t} == {1} and len(t) == 3


def test_fan_stacking_follows_the_boundaries():
    # the left inner boundary runs with the middle ribbon's axis (middle over left), the
    # right one with the right ribbon's (right over middle): z rises left to right
    b = [(250, -0.16), (350, 0.0), (430, 0.14), (550, 0.16)]
    p, labels, groups = photo(b, [1, 2, 3])
    t = tails_of(detect_parts(p, labels, groups))
    z = [q.z for q in t]
    assert z[0] < z[1] < z[2]


def test_fan_z_from_boundary_angles():
    from rosette.vision.parts_tails import fan_z
    axes = [-8.5, 0.0, 8.5]  # the tails' axis angles
    assert fan_z([0.0, 8.0], axes) == [0, 1, 2]
    assert fan_z([-1.0, 1.0], axes) == [0, 1, 0]  # middle on top
    assert fan_z([-8.0, 8.0], axes) == [1, 0, 1]  # middle under both


def bottom_spans(t):
    """(left, right) x of each tail's bottom edge, from its own params as geometry.tails
    places it (axes fan out from (300, top_y))."""
    import math
    out = []
    for q in t:
        p = q.params
        n, k, s = int(p["tails_count"]), int(p["tail_index"]), float(p["spread_deg"])
        ang = 0.0 if n == 1 else -s / 2 + k * s / (n - 1)
        c = 300 + (p["bottom_y"] - p["top_y"]) * math.tan(math.radians(ang))
        out.append((c - p["bottom_width"] / 2, c + p["bottom_width"] / 2))
    return out


def test_fan_tails_take_the_measured_bottom_shares():
    # unequal ribbons (bottom shares 0.40 / 0.33 / 0.27): each tail its own width, the
    # middle one centered on the axis, bottoms edge to edge (no gap, no overlap)
    b = [(250, -0.16), (350, 0.0), (430, 0.14), (550, 0.16)]
    p, labels, groups = photo(b, [1, 2, 3])
    t = tails_of(detect_parts(p, labels, groups))
    assert [q.color_group for q in t] == [1, 2, 3]
    spans = bottom_spans(t)
    widths = [r - l for l, r in spans]
    xs = [154.3, 350.0, 513.7, 645.7]  # photo px at the bottom row
    shares = [(b - a) / (xs[-1] - xs[0]) for a, b in zip(xs, xs[1:])]
    assert [w / sum(widths) for w in widths] == [pytest.approx(v, abs=0.03) for v in shares]
    assert (spans[1][0] + spans[1][1]) / 2 == pytest.approx(300, abs=0.5)
    assert spans[0][1] == pytest.approx(spans[1][0], abs=0.01)
    assert spans[1][1] == pytest.approx(spans[2][0], abs=0.01)
    keys = {(q.params["top_width"], q.params["bottom_width"], q.params["spread_deg"]) for q in t}
    assert len(keys) == 3  # build draws every tail with its own geometry
    assert all(q.params["top_width"] == q.params["bottom_width"] for q in t)  # ribbons
    for i, g in enumerate(groups):
        g.category = 1 + i % 4
    shapes = build(RosetteModel(parts=detect_parts(p, labels, groups), groups=groups))
    fills = [s for s in shapes if s.role == "tail_stripe" and s.kind == "color"]
    assert len(fills) == 3 and validate(write_svg(shapes)) == []
