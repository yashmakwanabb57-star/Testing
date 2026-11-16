from rosette.model import Part
from rosette.vision.parts_head import _badge_disc


def _ring(i, g, a, b):
    return Part(f"ring-{i}", "ring", {"cx": 300.0, "cy": 180.0, "r_inner": a, "r_outer": b, "pinked": 1}, g, 30 + i, 0.8)


def _a1_15():
    # A1-15: the badge read as a dot, a gold ring, the pink print and the gold rim; pleats beyond
    return [Part("head", "head", {"cx": 300.0, "cy": 180.0, "r": 157.0, "pinked": 1}, 0, 10, 0.9),
            Part("center", "center", {"cx": 300.0, "cy": 180.0, "r": 7.5}, 1, 50, 0.9),
            _ring(0, 1, 60.5, 80.5), _ring(1, 0, 49.5, 57.5), _ring(2, 1, 10.5, 24.5)]


def test_a_badge_read_as_a_dot_and_rings_is_one_center_disc():
    parts = _a1_15()
    _badge_disc(parts, head_label=0)
    center = next(q for q in parts if q.role == "center")
    # the gold rim (to 0.51 of the head radius) is the badge's edge: one disc out to it
    assert center.params["r"] == 80.5 and center.color_group == 1 and "pinked" not in center.params
    assert not [q for q in parts if q.role == "ring"]


def test_rings_beyond_the_badge_zone_stay_rings():
    parts = _a1_15() + [_ring(3, 0, 82.0, 100.0)]  # 0.64 of the head radius: pleats
    _badge_disc(parts, head_label=0)
    assert [q.params["r_outer"] for q in parts if q.role == "ring"] == [100.0]


def test_a_proper_center_disc_is_left_alone():
    parts = _a1_15()
    parts[1].params["r"] = 40.0
    _badge_disc(parts, head_label=0)
    assert len([q for q in parts if q.role == "ring"]) == 3


def test_a_star_around_the_dot_is_a_layered_head_not_a_badge():
    parts = _a1_15() + [Part("points-0", "points", {"kind": "star", "count": 6, "r_inner": 55.5, "r_outer": 105.5,
                                                    "rotation_deg": 0.0, "width_ratio": 0.6}, 2, 20, 0.8)]
    _badge_disc(parts, head_label=0)
    assert next(q for q in parts if q.role == "center").params["r"] == 7.5
