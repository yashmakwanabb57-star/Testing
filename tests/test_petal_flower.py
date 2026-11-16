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


def _hue_rgb(h_deg: float, v: float = 200.0):
    """An orange of hue h (degrees) at brightness v."""
    import math
    h = math.radians(h_deg)
    # R, G, B with 2R - G - B and sqrt(3)(G - B) in the ratio of cos h : sin h, B = 20
    b = 20.0
    k = (v - b)
    g = b + k * (2 * math.sin(h)) / (math.sqrt(3) * math.cos(h) + math.sin(h))
    return (v, g, b)


def _satin_star_photo():
    """Pleats of hue 26 and a six-point star of hue 16 (both one group, 0), light pleat rim and
    tails of hue 33 (group 2), glitter loops between the star's points (group 1)."""
    h = w = 600
    yy, xx = np.mgrid[:h, :w]
    cx, cy, hr = 300.0, 230.0, 180.0
    r = np.hypot(xx - cx, yy - cy)
    th = np.degrees(np.arctan2(yy - cy, xx - cx))
    mask = r <= hr
    mask |= (np.abs(xx - cx) <= 120) & (yy >= cy) & (yy < 590)
    rgb = np.zeros((h, w, 3))
    labels = np.full((h, w), -1, dtype=int)
    labels[mask] = 0
    rgb[mask] = _hue_rgb(26)
    rim = mask & (r > 0.92 * hr) & (r <= hr)
    labels[rim] = 2; rgb[rim] = _hue_rgb(33)
    tail_out = mask & (yy > cy + hr) & (np.abs(xx - cx) > 60)
    labels[tail_out] = 2; rgb[tail_out] = _hue_rgb(33)
    tail_mid = mask & (yy > cy + hr) & (np.abs(xx - cx) <= 60)
    rgb[tail_mid] = _hue_rgb(16)
    loops = mask & (r > 0.45 * hr) & (r < 0.9 * hr) & (np.abs(((th + 30) % 60) - 30) < 12)
    labels[loops] = 1; rgb[loops] = (110, 75, 40)
    phi = np.radians(th - 30)
    star_r = 0.35 * hr + (0.73 - 0.35) * hr * np.abs(np.cos(3 * phi)) ** 4
    star = mask & (r <= star_r) & (r > 0.25 * hr)
    labels[star] = 0; rgb[star] = _hue_rgb(16)
    rose = r <= 0.25 * hr
    labels[rose] = 0; rgb[rose] = _hue_rgb(24)
    p = SimpleNamespace(axis_x=cx, scale=1.0, mask=mask, head=(cx, cy, hr), rgb=rgb.astype(np.uint8))
    from rosette.vision.parts_common import Frame
    fr = Frame.of(p)
    geo = {"cx": 300.0, "cy": round(float(fr.y(cy)), 3)}
    parts = [Part("head", "head", dict(geo, r=hr), 0, 10, 0.9),
             Part("points-1", "points", dict(geo, kind="bow", count=6, r_inner=0.45 * hr, r_outer=0.9 * hr,
                                             rotation_deg=0.0, width_ratio=0.6), 1, 21, 0.9),
             Part("center", "center", dict(geo, r=17.5), 0, 50, 0.5),
             Part("ring-0", "ring", dict(geo, r_inner=31.5, r_outer=70.5), 0, 30, 0.5)]
    return p, labels, parts, hr


def test_r59_a_dark_satin_star_on_lighter_pleats_of_one_group():
    from rosette.pipeline import _satin_star_on_pleats
    p, labels, parts, hr = _satin_star_photo()
    out = _satin_star_on_pleats(p, labels, parts)
    assert out is not None
    ids = {q.id: q for q in out}
    star = ids["points-star"]
    assert star.color_group == 0 and star.params["count"] == 6 and star.params["kind"] == "star"
    assert abs(((star.params["rotation_deg"] - 30) + 30) % 60 - 30) < 4  # tips between the loops
    assert abs(star.params["r_outer"] / hr - 0.73) < 0.06
    assert ids["head"].color_group == 2 and ids["center"].color_group == 2
    assert abs(ids["center"].params["r"] / hr - 0.33) < 0.01
    assert "ring-0" not in ids
    loops, edge = ids["points-1"], ids["edge-points-1"]
    assert loops.z < edge.z < star.z  # the loops and their edges run in under the star
    assert edge.color_group == 0 and edge.params["edge_width"] == round(0.06 * hr, 3)
    assert loops.params["r_inner"] == round(0.33 * hr, 3)
    assert loops.params["r_outer"] <= round(0.84 * hr, 3)  # a band of pleats round the loops
    width = 0.6 * (0.9 - 0.45) * hr
    assert abs(loops.params["width_ratio"] * (loops.params["r_outer"] - loops.params["r_inner"]) - width) < 0.5


def test_r59_a_star_of_the_pleats_own_hue_stays():
    from rosette.pipeline import _satin_star_on_pleats
    p, labels, parts, hr = _satin_star_photo()
    p.rgb[...] = np.where(p.rgb.sum(-1, keepdims=True) > 0, np.array(_hue_rgb(26), dtype=np.uint8), 0)
    assert _satin_star_on_pleats(p, labels, parts) is None


def test_r60_a_star_under_a_shorter_star_takes_its_tips_from_its_color_runs():
    # s39-15, the owner's check: the white star's tips reach 0.92 of the head, read at 0.79
    from rosette.pipeline import _under_star_tips
    h = w = 600
    yy, xx = np.mgrid[:h, :w]
    cx, cy, hr = 300.0, 230.0, 150.0
    r = np.hypot(xx - cx, yy - cy)
    th = np.degrees(np.arctan2(yy - cy, xx - cx))
    mask = r <= hr
    labels = np.full((h, w), -1, dtype=int)
    labels[mask] = 0
    phi = np.radians(th - 30)
    white = mask & (r <= 0.45 * hr + (0.92 - 0.45) * hr * np.abs(np.cos(3 * phi)) ** 6)
    labels[white] = 1
    gold = mask & (r <= 0.45 * hr + (0.76 - 0.45) * hr * np.abs(np.cos(3 * np.radians(th))) ** 6)
    labels[gold] = 2
    p = SimpleNamespace(axis_x=cx, scale=1.0, mask=mask, head=(cx, cy, hr))
    from rosette.vision.parts_common import Frame
    geo = {"cx": 300.0, "cy": round(float(Frame.of(p).y(cy)), 3)}
    under = Part("points-1", "points", dict(geo, kind="star", count=6, r_inner=0.4 * hr, r_outer=0.79 * hr, rotation_deg=30.0), 1, 20, 0.9)
    top = Part("points-2", "points", dict(geo, kind="star", count=6, r_inner=0.38 * hr, r_outer=0.76 * hr, rotation_deg=0.0), 2, 21, 0.9)
    parts = [Part("head", "head", dict(geo, r=hr), 0, 10, 0.9), under, top]
    assert _under_star_tips(parts, p, labels)
    assert abs(under.params["r_outer"] / hr - 0.92) < 0.03
    assert top.params["r_outer"] == 0.76 * hr
    assert under.params["r_inner"] == round(0.577 * under.params["r_outer"], 3)
    assert top.params["r_inner"] == round(0.577 * top.params["r_outer"], 3)
    ids = {q.id: q for q in parts}
    yi, bi = ids["points-1-inner"], ids["points-2-inner"]  # a smaller star of each on top, in order
    assert under.z < top.z < yi.z < bi.z and yi.color_group == 1 and bi.color_group == 2
    assert abs(yi.params["r_outer"] - 0.80 * under.params["r_outer"]) < 0.01
    assert abs(bi.params["r_outer"] - 0.77 * top.params["r_outer"]) < 0.01
    assert yi.params["rotation_deg"] == under.params["rotation_deg"] and bi.params["rotation_deg"] == top.params["rotation_deg"]


def _s40_photo():
    """Orange pleats (group 0, hue ~21) with six black glitter points (1) at 0, 60, ... and six light
    orange satin points (hue ~36) between them that the groups mostly lost to the pleats (0)."""
    h = w = 600
    yy, xx = np.mgrid[:h, :w]
    cx, cy, hr = 300.0, 230.0, 150.0
    r = np.hypot(xx - cx, yy - cy)
    th = np.degrees(np.arctan2(yy - cy, xx - cx))
    mask = r <= hr
    labels = np.full((h, w), -1, dtype=int)
    rgb = np.zeros((h, w, 3))
    labels[mask] = 0; rgb[mask] = _hue_rgb(21, 222)
    glit = mask & (r > 0.45 * hr) & (r < 0.45 * hr + 0.3 * hr * np.abs(np.cos(3 * np.radians(th))) ** 6)
    labels[glit] = 1; rgb[glit] = (14, 12, 11)
    sat = mask & (r > 0.58 * hr) & (r < 0.58 * hr + 0.35 * hr * np.abs(np.cos(3 * np.radians(th - 30))) ** 6)
    rgb[sat] = _hue_rgb(36, 235)
    labels[sat & (th < -60) & (th > -120)] = 2  # one satin point found as its own group
    labels[(r < 0.29 * hr)] = 1  # the badge read into the black group
    p = SimpleNamespace(axis_x=cx, scale=1.0, mask=mask, head=(cx, cy, hr), rgb=rgb.astype(np.uint8))
    from rosette.vision.parts_common import Frame
    geo = {"cx": 300.0, "cy": round(float(Frame.of(p).y(cy)), 3)}
    parts = [Part("points-1", "points", dict(geo, kind="petal", count=6, r_inner=0.29 * hr, r_outer=0.7 * hr,
                                             rotation_deg=0.0, width_ratio=0.6), 1, 22, 0.9),
             Part("center", "center", dict(geo, r=0.29 * hr), 1, 50, 0.5),
             Part("ring-0", "ring", dict(geo, r_inner=0.7 * hr, r_outer=hr), 0, 30, 0.5),
             Part("ring-1", "ring", dict(geo, r_inner=0.29 * hr, r_outer=0.58 * hr), 0, 31, 0.5),
             Part("unknown-0", "unknown", dict(geo, r=30.0), 0, 20, 0.3)]
    return p, labels, parts, hr


def test_r61_glitter_points_between_satin_points_of_a_third_color():
    from rosette.pipeline import _satin_star_third_color
    p, labels, parts, hr = _s40_photo()
    out = _satin_star_third_color(p, labels, parts)
    assert out is not None
    ids = {q.id: q for q in out}
    head, glit, sat, ring, c = ids["head"], ids["points-1"], ids["points-satin"], ids["ring-1"], ids["center"]
    assert head.color_group == 0 and head.params["r"] == hr
    assert glit.params["kind"] == "star" and sat.params["kind"] == "star" and sat.color_group == 2
    assert sat.z < glit.z < ring.z < c.z  # the glitter star on the satin star
    assert abs(sat.params["r_outer"] / glit.params["r_outer"] - 0.93 / 0.78) < 0.01
    assert abs(((sat.params["rotation_deg"] - glit.params["rotation_deg"]) % 60) - 30) < 0.01
    assert c.color_group == 2 and "ring-0" not in ids and "unknown-0" not in ids


def test_r61_needs_a_satin_hue_between_the_points():
    from rosette.pipeline import _satin_star_third_color
    p, labels, parts, hr = _s40_photo()
    p.rgb[...] = np.where(p.rgb.sum(-1, keepdims=True) > 100, np.array(_hue_rgb(21, 222), dtype=np.uint8), p.rgb)
    assert _satin_star_third_color(p, labels, parts) is None
