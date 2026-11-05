import pytest

from rosette.build import build
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
