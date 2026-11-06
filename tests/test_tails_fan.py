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
    union = unary_union([to_geom(s) for s in fills])
    assert isinstance(union, Polygon) and not union.interiors  # tails meet, no gap between them
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
    assert fan_z([0.0, 8.0], 17.0) == [0, 1, 2]
    assert fan_z([-1.0, 1.0], 17.0) == [0, 1, 0]  # middle on top
    assert fan_z([-8.0, 8.0], 17.0) == [1, 0, 1]  # middle under both
