from types import SimpleNamespace

import numpy as np

from rosette.model import Part


def _flower(n: int = 6, lobes: bool = True):
    """A 600x600 photo, 1 px per canvas unit: a red head (0) with n pointed lobes reaching 210
    around a white rose (1) of radius 70 at (300, 230), tails (0) running down from the head."""
    h = w = 600
    yy, xx = np.mgrid[:h, :w]
    cx, cy = 300.0, 230.0
    r = np.hypot(xx - cx, yy - cy)
    th = np.arctan2(yy - cy, xx - cx)
    reach = 130 + 80 * np.abs(np.cos(n * th / 2)) ** 3 if lobes else np.full_like(r, 170.0)
    mask = r <= reach
    mask |= (np.abs(xx - cx) <= 90) & (yy >= cy) & (yy < 590)  # the tails
    labels = np.full((h, w), -1, dtype=int)
    labels[mask] = 0
    labels[r <= 70] = 1
    p = SimpleNamespace(axis_x=cx, scale=1.0, mask=mask, head=(cx, cy + 40.0, 220.0))
    return p, labels


def _rules_parts():
    return [Part("unknown-0", "unknown", {"cx": 300.0, "cy": 250.0, "r": 78.0}, 0, 20, 0.3),
            Part("center", "center", {"cx": 300.0, "cy": 260.0, "r": 30.0}, 1, 50, 0.5),
            Part("ring-0", "ring", {"cx": 300.0, "cy": 260.0, "r_inner": 74.0, "r_outer": 110.0}, 0, 30, 0.5),
            Part("tail-0", "tail_stripe", {"x0": 210.0}, 0, 1, 0.9)]


def test_r56_a_one_color_lobed_head_round_a_rose_is_a_petal_flower():
    from rosette.pipeline import _petal_flower
    from rosette.vision.layer_classify import KEEP_PARAM
    p, labels = _flower(6)
    out = _petal_flower(p, labels, _rules_parts())
    assert out is not None
    roles = sorted(q.role for q in out)
    assert roles == ["center", "points", "tail_stripe"]
    pet = next(q for q in out if q.role == "points")
    c = next(q for q in out if q.role == "center")
    assert pet.params["kind"] == "petal" and pet.params["count"] == 6 and pet.color_group == 0
    assert pet.params["tips"] == 1 and pet.params[KEEP_PARAM] == 1
    assert abs(pet.params["r_outer"] - 210) < 8
    assert abs(((pet.params["rotation_deg"] + 30) % 60) - 30) < 4  # a tip on +x, every 60 degrees
    from rosette.vision.parts_common import Frame
    assert c.color_group == 1 and abs(c.params["cy"] - Frame.of(p).y(230)) < 3 and abs(c.params["r"] - pet.params["r_outer"] / 2.55) < 1
    assert pet.params["r_inner"] < c.params["r"]  # the petals tuck under the rose
    assert pet.params["cy"] == c.params["cy"] and isinstance(c.params["cy"], float)
    assert pet.z < c.z


def test_r56_a_round_head_or_a_head_with_layers_stays():
    from rosette.pipeline import _petal_flower
    p, labels = _flower(lobes=False)
    assert _petal_flower(p, labels, _rules_parts()) is None
    p, labels = _flower(6)
    parts = _rules_parts() + [Part("points-0", "points", {"cx": 300.0, "cy": 230.0, "kind": "bow", "count": 6,
                                                         "r_inner": 70.0, "r_outer": 200.0, "rotation_deg": 0.0,
                                                         "width_ratio": 0.5}, 0, 20, 0.9)]
    assert _petal_flower(p, labels, parts) is None


def _two_color_flower():
    """Six lobes alternating red (0) and white (1) around a red rose (0) joined to the red petals."""
    p, labels = _flower(6)
    h, w = labels.shape
    yy, xx = np.mgrid[:h, :w]
    th = np.degrees(np.arctan2(yy - 230.0, xx - 300.0)) % 360
    lobe = np.rint(th / 60.0).astype(int) % 6
    r = np.hypot(xx - 300.0, yy - 230.0)
    head = p.mask & (r <= 215)
    labels[head] = np.where(lobe[head] % 2 == 0, 0, 1)
    labels[r <= 70] = 0
    return p, labels


def test_r57_alternating_petals_of_two_colors_round_a_rose():
    from rosette.pipeline import _petal_flower
    p, labels = _two_color_flower()
    out = _petal_flower(p, labels, _rules_parts())
    assert out is not None
    pets = sorted((q for q in out if q.role == "points"), key=lambda q: q.z)
    c = next(q for q in out if q.role == "center")
    assert [q.params["count"] for q in pets] == [3, 3]
    assert [q.color_group for q in pets] == [0, 1]  # the rose's color underneath
    assert c.color_group == 0 and c.z > pets[-1].z
    d = (pets[1].params["rotation_deg"] - pets[0].params["rotation_deg"]) % 120
    assert abs(d - 60) < 5
    assert abs(((pets[0].params["rotation_deg"] + 30) % 60) - 30) < 6  # a red tip on +x


def _s37_parts(kind="bow"):
    geo = {"cx": 300.0, "cy": 172.0}
    return [Part("head", "head", dict(geo, r=146.0), 0, 10, 0.9),
            Part("points-1", "points", dict(geo, kind=kind, count=8, r_inner=94.5, r_outer=127.5, rotation_deg=1.7,
                                            width_ratio=0.8), 1, 20, 0.9),
            Part("center", "center", dict(geo, r=34.5), 0, 50, 0.9),
            Part("ring-0", "ring", dict(geo, r_inner=34.5, r_outer=94.5), 0, 30, 0.9)]


def test_r58_glitter_points_round_a_pleated_ring_are_a_star_under_a_satin_star():
    # S37-15, the owner's check: gold glitter points between red satin points, on red pleats
    from rosette.pipeline import _star_under_satin_star
    from rosette.vision.layer_classify import KEEP_PARAM
    for kind in ("bow", "star", "diamond"):
        parts = _s37_parts(kind)
        assert _star_under_satin_star(parts)
        ids = {q.id: q for q in parts}
        gold, red = ids["points-1"], ids["points-satin"]
        ro = round(0.95 * 146.0, 3)  # the measured 127.5 is short of the tips
        assert gold.params["kind"] == "star" and gold.params["r_outer"] == ro
        assert red.params["kind"] == "star" and red.color_group == 0 and red.params["count"] == 8
        assert abs(red.params["r_outer"] - 0.88 * ro) < 0.01 and abs(red.params["r_inner"] - 0.53 * ro) < 0.01
        assert abs(((red.params["rotation_deg"] - 1.7) % 45) - 22.5) < 0.01
        assert gold.z < red.z < ids["ring-0"].z
        assert ids["ring-0"].params["r_outer"] == round(0.6 * 146.0, 3)  # the red points rise from its edge
        assert red.params["r_inner"] < ids["ring-0"].params["r_outer"]
        assert gold.params[KEEP_PARAM] == 1 and red.params[KEEP_PARAM] == 1


def test_r58_needs_the_head_colored_ring_inside():
    from rosette.pipeline import _star_under_satin_star
    parts = [q for q in _s37_parts() if q.id != "ring-0"]
    assert not _star_under_satin_star(parts)
    parts = _s37_parts()
    parts[1].color_group = 0  # points of the head's color: not this layout
    assert not _star_under_satin_star(parts)


def test_r58_a_thin_rim_of_the_glitter_color_round_the_center_is_the_rose():
    # S37-15, the owner's check: the center is one rose of the gold's color (designer: Color 2, 0.3 of the head)
    from rosette.pipeline import _star_under_satin_star
    geo = {"cx": 300.0, "cy": 172.0}
    parts = _s37_parts() + [Part("ring-1", "ring", dict(geo, r_inner=37.5, r_outer=44.5), 1, 31, 0.9)]
    assert _star_under_satin_star(parts)
    c = next(q for q in parts if q.role == "center")
    assert c.color_group == 1 and c.params["r"] == 44.5
    assert not any(q.id == "ring-1" for q in parts)
    assert any(q.id == "ring-0" for q in parts)
