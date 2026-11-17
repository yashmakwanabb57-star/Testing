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


def _s41_parts(hr=155.0):
    geo = {"cx": 300.0, "cy": 180.0}
    return [Part("head", "head", dict(geo, r=hr), 0, 10, 0.9),
            Part("points-1", "points", dict(geo, kind="star", count=6, r_inner=77.5, r_outer=112.5, rotation_deg=57.3), 2, 20, 0.9),
            Part("front-2", "points", dict(geo, kind="star", count=6, r_inner=62.5, r_outer=82.5, rotation_deg=25.3), 0, 22, 0.9),
            Part("front-3", "points", dict(geo, kind="star", count=6, r_inner=62.5, r_outer=116.5, rotation_deg=25.3), 1, 21, 0.9),
            Part("center", "center", dict(geo, r=44.5), 1, 50, 0.9),
            Part("ring-0", "ring", dict(geo, r_inner=44.5, r_outer=63.5), 0, 30, 0.9)]


def test_r62_three_nested_stars_stack_as_the_designer():
    # S41-15, the owner's check: white star, glitter star on it, a wine star of the pleats' color on top
    from rosette.pipeline import _three_nested_stars
    from rosette.vision.layer_classify import KEEP_PARAM
    parts = _s41_parts()
    assert _three_nested_stars(parts)
    ids = {q.id: q for q in parts}
    white, glit, wine = ids["front-3"], ids["points-1"], ids["front-2"]
    assert white.z < glit.z < wine.z < ids["center"].z
    assert [round(q.params["r_outer"] / 155.0, 2) for q in (white, glit, wine)] == [0.91, 0.8, 0.67]
    assert all(q.params["r_inner"] == round(0.577 * q.params["r_outer"], 3) for q in (white, glit, wine))
    assert all(q.params[KEEP_PARAM] == 1 and q.params["tips"] == 1 for q in (white, glit, wine))
    assert "ring-0" not in ids and ids["center"].params["r"] == round(0.34 * 155.0, 3)
    assert white.params["rotation_deg"] == 30.0 and wine.params["rotation_deg"] == 30.0 and glit.params["rotation_deg"] == 0.0


def test_r62_needs_two_stars_on_one_axis_and_one_half_a_step_off():
    from rosette.pipeline import _three_nested_stars
    parts = _s41_parts()
    parts[1].params["rotation_deg"] = 25.3  # all three on one axis
    assert not _three_nested_stars(parts)


def _s4_photo():
    """A light head (group 0, lightness ~160) with six dark loops (~130) showing at the rim at 0, 60,
    ...; most of the dark read into the light group, a little as its own group (1)."""
    h = w = 600
    yy, xx = np.mgrid[:h, :w]
    cx, cy, hr = 300.0, 230.0, 160.0
    r = np.hypot(xx - cx, yy - cy)
    th = np.degrees(np.arctan2(yy - cy, xx - cx))
    mask = (r <= hr) | ((np.abs(xx - cx) <= 90) & (yy >= cy) & (yy < 590))
    labels = np.full((h, w), -1, dtype=int); labels[mask] = 0
    rgb = np.zeros((h, w, 3)); rgb[mask] = (140, 133, 212)
    dark = mask & (r > 0.8 * hr) & (r <= hr) & (np.abs(((th + 30) % 60) - 30) < 10)
    rgb[dark] = (99, 90, 209)
    labels[dark & (th > -20) & (th < 20)] = 1
    tail_out = mask & (yy > cy + hr) & (np.abs(xx - cx) > 50)
    labels[tail_out] = 1; rgb[tail_out] = (99, 90, 209)
    p = SimpleNamespace(axis_x=cx, scale=1.0, mask=mask, head=(cx, cy, hr), rgb=rgb.astype(np.uint8))
    from rosette.vision.parts_common import Frame
    geo = {"cx": 300.0, "cy": round(float(Frame.of(p).y(cy)), 3)}
    parts = [Part("head", "head", dict(geo, r=hr), 0, 10, 0.9), Part("unknown-1", "unknown", dict(geo, r=28.0), 1, 20, 0.3)]
    return p, labels, parts, hr


def test_r63_dark_loops_at_the_rim_under_light_loops():
    from rosette.pipeline import _two_tone_loops
    p, labels, parts, hr = _s4_photo()
    out = _two_tone_loops(p, labels, parts)
    assert out is not None
    ids = {q.id: q for q in out}
    dark, light, c = ids["points-dark"], ids["points-light"], ids["center"]
    assert "head" not in ids and "unknown-1" not in ids
    assert dark.color_group == 1 and light.color_group == 0 and c.color_group == 1
    assert dark.params["count"] == 6 and light.params["count"] == 6 and dark.params["kind"] == "bow"
    assert abs(((dark.params["rotation_deg"] + 30) % 60) - 30) < 3
    assert abs(((light.params["rotation_deg"] - dark.params["rotation_deg"]) % 60) - 30) < 0.01
    ring = ids["ring-light"]
    assert dark.z < ring.z < light.z < c.z and ring.color_group == 0
    assert abs(c.params["r"] - 0.39 * hr) < 0.01
    assert ring.params["r_inner"] < c.params["r"] and light.params["r_inner"] < c.params["r"]  # under the rose
    assert abs(ring.params["r_outer"] - 0.66 * hr) < 0.01  # the dark loops show only past it


def test_r63_a_head_of_one_lightness_stays():
    from rosette.pipeline import _two_tone_loops
    p, labels, parts, hr = _s4_photo()
    yy = np.arange(600)[:, None]
    p.rgb[(yy <= 230 + hr) & p.mask] = (140, 133, 212)  # the whole head one lightness, the dark only in the tails
    assert _two_tone_loops(p, labels, parts) is None


def _s42_parts(hr=142.8):
    geo = {"cx": 300.0, "cy": 168.6}
    return [Part("head", "head", dict(geo, r=hr), 0, 10, 0.9),
            Part("points-1", "points", dict(geo, kind="bow", count=6, r_inner=86.5, r_outer=108.5, rotation_deg=34.4,
                                            width_ratio=0.8), 1, 20, 0.9),
            Part("center", "center", dict(geo, r=50.5), 1, 50, 0.9),
            Part("ring-0", "ring", dict(geo, r_inner=50.5, r_outer=86.5), 0, 30, 0.9)]


def test_r64_star_tips_round_a_ring_of_the_head_color_are_a_star_under_petals():
    # S42-15, the owner's check: white star, light blue petals of the pleats' color on it, white rose
    from rosette.pipeline import _star_under_petals
    from rosette.vision.layer_classify import KEEP_PARAM
    parts = _s42_parts()
    assert _star_under_petals(parts)
    ids = {q.id: q for q in parts}
    star, pet, c = ids["points-1"], ids["points-petals"], ids["center"]
    hr = 142.8
    assert star.params["kind"] == "star" and star.color_group == 1 and star.params["rotation_deg"] == 30.0
    assert abs(star.params["r_outer"] / hr - 0.85) < 0.01 and star.params["r_inner"] == round(0.577 * star.params["r_outer"], 3)
    assert pet.params["kind"] == "petal" and pet.params["flower"] == 2 and pet.color_group == 0
    assert pet.params["rotation_deg"] == 0.0 and abs(pet.params["r_outer"] / hr - 0.71) < 0.01
    assert pet.params["r_inner"] < c.params["r"]  # the petals tuck under the rose
    assert star.z < pet.z < c.z and "ring-0" not in ids
    assert c.color_group == 1 and star.params[KEEP_PARAM] == 1 and pet.params[KEEP_PARAM] == 1


def test_r64_needs_the_center_of_the_star_color():
    from rosette.pipeline import _star_under_petals
    parts = _s42_parts()
    parts[2].color_group = 0
    assert not _star_under_petals(parts)


def test_r64_the_apps_reading_a_head_color_star_for_the_petals():
    from rosette.pipeline import _star_under_petals
    parts = [q for q in _s42_parts() if q.id != "ring-0"]
    geo = {"cx": 300.0, "cy": 168.6}
    parts.append(Part("front-2", "points", dict(geo, kind="star", count=6, r_inner=71.0, r_outer=103.0, rotation_deg=5.4), 0, 20, 0.9))
    assert _star_under_petals(parts)
    ids = {q.id: q for q in parts}
    assert "front-2" not in ids and ids["points-petals"].params["width_ratio"] == round(0.635 / 0.7, 4)


def _s43_parts(hr=144.6):
    geo = {"cx": 300.0, "cy": 169.3}
    return [Part("head", "head", dict(geo, r=hr), 0, 10, 0.9),
            Part("points-1", "points", dict(geo, kind="bow", count=6, r_inner=86.5, r_outer=126.5, rotation_deg=2.9,
                                            width_ratio=0.8), 1, 20, 0.9),
            Part("center", "center", dict(geo, r=50.5), 2, 50, 0.9),
            Part("ring-0", "ring", dict(geo, r_inner=50.5, r_outer=86.5), 0, 30, 0.9)]


def test_r65_loops_round_a_ring_of_the_head_color_run_in_under_petals_edged_in_the_rose_color():
    # S43-15, the owner's check: glitter loops edged white under light blue petals, a white rose
    from rosette.pipeline import _loops_under_petals
    parts = _s43_parts()
    assert _loops_under_petals(parts)
    ids = {q.id: q for q in parts}
    hr = 144.6
    loops, edge, pet, c = ids["points-1"], ids["edge-points-1"], ids["points-petals"], ids["center"]
    assert loops.z < edge.z < pet.z < c.z and "ring-0" not in ids
    assert edge.color_group == 2 and abs(edge.params["edge_width"] - 0.057 * hr) < 0.01
    assert loops.params["r_inner"] < c.params["r"] and loops.params["rotation_deg"] == 0.0
    assert abs(loops.params["width_ratio"] * (loops.params["r_outer"] - loops.params["r_inner"]) - 0.48 * hr) < 0.5
    assert pet.color_group == 0 and pet.params["flower"] == 2 and pet.params["rotation_deg"] == 30.0
    assert abs(pet.params["r_outer"] / hr - 0.758) < 0.01 and abs(c.params["r"] / hr - 0.345) < 0.01


def test_r65_needs_a_center_of_a_third_color():
    from rosette.pipeline import _loops_under_petals
    parts = _s43_parts()
    parts[2].color_group = 1
    assert not _loops_under_petals(parts)


def test_r65_the_apps_reading_a_head_color_star_for_the_petals():
    from rosette.pipeline import _loops_under_petals
    parts = [q for q in _s43_parts() if q.id != "ring-0"]
    parts.append(Part("front-2", "points", {"cx": 300.0, "cy": 169.3, "kind": "star", "count": 6, "r_inner": 60.0,
                                            "r_outer": 100.0, "rotation_deg": 31.0}, 0, 21, 0.9))
    assert _loops_under_petals(parts)
    assert "front-2" not in {q.id for q in parts}


def _s44_parts(hr=131.5):
    geo = {"cx": 300.0, "cy": 155.0}
    def pt(pid, g, z, kind, ro, rot):
        return Part(pid, "points", dict(geo, kind=kind, count=4, r_inner=51.5, r_outer=ro, rotation_deg=rot), g, z, 0.9)
    return [Part("head", "head", dict(geo, r=hr), 0, 10, 0.9),
            pt("points-1", 1, 24, "petal", 105.5, 46.2), pt("front-2", 0, 21, "star", 101.5, 3.0),
            pt("front-3", 2, 20, "star", 116.5, 3.0), pt("points-7", 2, 22, "star", 87.5, 3.0), pt("points-6", 0, 23, "star", 71.5, 3.0),
            Part("center", "center", dict(geo, r=38.5), 1, 50, 0.9),
            Part("ring-0", "ring", dict(geo, r_inner=38.5, r_outer=51.5), 0, 30, 0.9)]


def test_r66_a_cross_of_banded_four_point_stars():
    # s44-15, the owner's check: gold banded star on the axes, cream banded star on the diagonals on it
    from rosette.pipeline import _banded_cross_stars
    parts = _s44_parts()
    assert _banded_cross_stars(parts)
    pts = sorted((q for q in parts if q.role == "points"), key=lambda q: q.z)
    hr = 131.5
    got = [(q.color_group, q.params["count"], round(q.params["r_outer"] / hr, 3), q.params["rotation_deg"]) for q in pts]
    assert got == [(2, 4, 0.898, 0.0), (0, 4, 0.748, 0.0), (2, 4, 0.623, 0.0),
                   (1, 4, 0.895, 45.0), (1, 4, 0.748, 45.0), (1, 4, 0.622, 45.0), (0, 8, 0.52, 0.0)]
    assert all(q.params["tips"] == 1 for q in pts)
    ids = {q.id: q for q in parts}
    assert "ring-0" not in ids and ids["center"].color_group == 1 and ids["center"].z > pts[-1].z


def test_r66_needs_the_diagonal_layer():
    from rosette.pipeline import _banded_cross_stars
    parts = [q for q in _s44_parts() if q.id != "points-1"]
    assert not _banded_cross_stars(parts)


def _s45_parts(hr=129.3):
    geo = {"cx": 300.0, "cy": 154.7}
    return [Part("head", "head", dict(geo, r=hr), 0, 10, 0.9),
            Part("points-1", "points", dict(geo, kind="bow", count=8, r_inner=30.9, r_outer=101.5, rotation_deg=3.1,
                                            width_ratio=0.5), 1, 20, 0.9),
            Part("center", "center", dict(geo, r=37.5), 0, 50, 0.9),
            Part("ring-0", "ring", dict(geo, r_inner=63.5, r_outer=75.5), 2, 30, 0.9)]


def test_r67_an_eight_point_star_round_a_glitter_ring_a_ruffle_and_a_small_star():
    # S45-15, the owner's check: light blue star, glitter pleated ring, light blue ruffle, small glitter star, blue rose
    from rosette.pipeline import _star_round_rings
    parts = _s45_parts()
    assert _star_round_rings(parts)
    ids = {q.id: q for q in parts}
    hr = 129.3
    big, rx, ry, small, c = ids["points-1"], ids["ring-0"], ids["ring-inner"], ids["points-small"], ids["center"]
    assert big.params["kind"] == "star" and big.color_group == 1 and abs(big.params["r_outer"] / hr - 0.83) < 0.01
    assert big.params["rotation_deg"] == 0.0 and abs(big.params["r_inner"] / hr - 0.607) < 0.01
    assert rx.color_group == 2 and abs(rx.params["r_inner"] / hr - 0.44) < 0.01 and abs(rx.params["r_outer"] / hr - 0.6) < 0.01
    assert ry.color_group == 1 and abs(ry.params["r_inner"] / hr - 0.27) < 0.01
    assert small.color_group == 2 and small.params["count"] == 8 and abs(small.params["r_outer"] / hr - 0.424) < 0.01
    assert big.z < rx.z < ry.z < small.z < c.z and c.color_group == 0 and abs(c.params["r"] / hr - 0.269) < 0.01


def test_r67_needs_the_ring_of_a_third_color():
    from rosette.pipeline import _star_round_rings
    parts = _s45_parts()
    parts[3].color_group = 1
    assert not _star_round_rings(parts)


def _s46_parts(hr=128.5):
    geo = {"cx": 300.0, "cy": 150.8}
    return [Part("head", "head", dict(geo, r=hr), 0, 10, 0.9),
            Part("points-1", "points", dict(geo, kind="star", count=6, r_inner=54.5, r_outer=103.5, rotation_deg=34.3), 1, 21, 0.9),
            Part("points-2", "points", dict(geo, kind="bow", count=6, r_inner=54.5, r_outer=103.5, rotation_deg=4.4,
                                            width_ratio=0.6), 2, 20, 0.9),
            Part("center", "center", dict(geo, r=35.5), 0, 50, 0.9)]


def test_r68_loops_between_an_outer_and_an_inner_star_of_one_color():
    # S46-15, the owner's check: gold star, green-edged glitter loops on it, a smaller gold star on them
    from rosette.pipeline import _loops_between_stars
    parts = _s46_parts()
    assert _loops_between_stars(parts)
    ids = {q.id: q for q in parts}
    hr = 128.5
    outer, loops, edge, inner, c = ids["points-1"], ids["points-2"], ids["edge-points-2"], ids["points-1-inner"], ids["center"]
    assert outer.z < loops.z < edge.z < inner.z < c.z
    assert outer.params["rotation_deg"] == 30.0 and loops.params["rotation_deg"] == 0.0 and inner.params["rotation_deg"] == 30.0
    assert abs(outer.params["r_outer"] / hr - 0.90) < 0.01 and abs(inner.params["r_outer"] / hr - 0.70) < 0.01
    assert abs(inner.params["r_inner"] / hr - 0.432) < 0.01 and edge.color_group == 0 and inner.color_group == 1
    assert abs(loops.params["r_outer"] / hr - 0.845) < 0.01
    assert abs(loops.params["width_ratio"] * (loops.params["r_outer"] - loops.params["r_inner"]) - 0.26 * hr) < 0.5
    assert abs(c.params["r"] / hr - 0.27) < 0.01


def test_r68_needs_the_loops_half_a_step_off():
    from rosette.pipeline import _loops_between_stars
    parts = _s46_parts()
    parts[2].params["rotation_deg"] = 34.3
    assert not _loops_between_stars(parts)


def _s47_photo(pointed=True):
    """Blue pleats (0), six pink petals (1) round a white rose (2); the petals pointed or square-ended."""
    h = w = 600
    yy, xx = np.mgrid[:h, :w]
    cx, cy, hr = 300.0, 230.0, 160.0
    r = np.hypot(xx - cx, yy - cy)
    th = np.degrees(np.arctan2(yy - cy, xx - cx))
    mask = r <= hr
    labels = np.full((h, w), -1, dtype=int); labels[mask] = 0
    d = np.abs(((th + 30) % 60) - 30)  # degrees off the nearest petal axis
    if pointed:
        half = np.where(r < 0.6 * 120, 30.0, 30.0 * np.clip((120 - r) / (0.4 * 120), 0, 1))
    else:
        half = np.full_like(r, 22.0)
    pet = mask & (r <= 120) & (d <= half)
    labels[pet] = 1
    labels[r <= 60] = 2
    p = SimpleNamespace(axis_x=cx, scale=1.0, mask=mask, head=(cx, cy, hr))
    from rosette.vision.parts_common import Frame
    geo = {"cx": 300.0, "cy": round(float(Frame.of(p).y(cy)), 3)}
    parts = [Part("head", "head", dict(geo, r=hr), 0, 10, 0.9),
             Part("points-1", "points", dict(geo, kind="bow", count=6, r_inner=36.0, r_outer=112.5, rotation_deg=1.8,
                                             width_ratio=0.6), 1, 20, 0.9),
             Part("center", "center", dict(geo, r=60.5), 2, 50, 0.9)]
    return p, labels, parts, hr


def test_r69_pointed_loops_round_a_rose_are_leaf_petals():
    # S47-15, the owner's check: pink satin petals with pointed tips, not square loops
    from rosette.pipeline import _pointed_loops_as_petals
    p, labels, parts, hr = _s47_photo(True)
    assert _pointed_loops_as_petals(parts, p, labels)
    pets = sorted((q for q in parts if q.role == "points"), key=lambda q: q.z)
    pet = pets[0]
    assert len(pets) == 2 and all(q.params["count"] == 3 for q in pets)
    assert [q.params["flower"] for q in pets] == [4, 4]  # one petal shape, the front three on the back three
    assert pet.params["kind"] == "petal" and pet.params["rotation_deg"] == 0.0 and pets[1].params["rotation_deg"] == 60.0
    assert abs(pet.params["r_outer"] / hr - 0.81) < 0.01
    assert pet.params["r_inner"] < next(q for q in parts if q.role == "center").params["r"]


def test_r69_square_loops_stay():
    from rosette.pipeline import _pointed_loops_as_petals
    p, labels, parts, hr = _s47_photo(False)
    assert not _pointed_loops_as_petals(parts, p, labels)
    assert parts[1].params["kind"] == "bow"


def _s48_parts():
    geo = {"cx": 300.0, "cy": 148.0}
    def pt(pid, g, z, kind, ro, rot):
        return Part(pid, "points", dict(geo, kind=kind, count=4, r_inner=62.5, r_outer=ro, rotation_deg=rot, width_ratio=0.6), g, z, 0.9)
    return [pt("points-0", 1, 20, "star", 119.5, 47.4), pt("points-1", 2, 22, "petal", 122.5, 18.9),
            pt("points-2", 0, 21, "petal", 122.5, 75.4), Part("center", "center", dict(geo, r=26.5), 0, 50, 0.9)]


def test_r70_two_banded_stars_thirty_degrees_apart_under_squares_and_a_star():
    # S48-17, the owner's check: dark and light banded stars, white banded squares, a white star, the rose
    from rosette.pipeline import _split_star_flower
    parts = _s48_parts()
    assert _split_star_flower(parts)
    pts = sorted((q for q in parts if q.role == "points"), key=lambda q: q.z)
    assert [q.color_group for q in pts] == [1, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1]  # the squares, the arms on them, the middle
    assert [q.params["kind"] for q in pts] == ["star"] * 4 + ["split_arm"] * 6 + ["star"]
    assert [q.params.get("axis_deg") for q in pts[4:10]] == [0.0] * 3 + [90.0] * 3
    assert [round(q.params["r_outer"] / pts[4].params["r_outer"], 2) for q in pts[4:7]] == [1.0, 0.88, 0.73]  # scaled bands
    assert [q.params["count"] for q in pts] == [4] * 10 + [8]  # the middle: one eight-point star
    assert all(q.params["tips"] == 1 for q in pts)
    c = next(q for q in parts if q.role == "center")
    assert c.color_group == 0 and c.z > pts[-1].z


def test_r70_needs_the_thirty_degree_pair():
    from rosette.pipeline import _split_star_flower
    parts = _s48_parts()
    parts[1].params["rotation_deg"] = 45.0
    assert not _split_star_flower(parts)


def _s49_parts(hr=124.6):
    geo = {"cx": 300.0, "cy": 150.2}
    return [Part("head", "head", dict(geo, r=hr), 0, 10, 0.9),
            Part("points-1", "points", dict(geo, kind="diamond", count=12, r_inner=46.5, r_outer=94.5, rotation_deg=14.4,
                                            width_ratio=0.5), 2, 20, 0.9),
            Part("points-2", "points", dict(geo, kind="star", count=12, r_inner=57.5, r_outer=84.5, rotation_deg=29.0), 1, 21, 0.9),
            Part("center", "center", dict(geo, r=20.5), 0, 50, 0.9),
            Part("ring-0", "ring", dict(geo, r_inner=35.5, r_outer=46.5), 0, 30, 0.9)]


def test_r71_a_twelve_point_star_on_a_pleated_ring_with_its_tips_at_the_rim():
    # S49-17, the owner's check: white points at the rim, a dark green ring, a white star on it, a light green ruffle
    from rosette.pipeline import _star_on_pleated_ring
    parts = _s49_parts()
    assert _star_on_pleated_ring(parts)
    ids = {q.id: q for q in parts}
    hr = 124.6
    big, ring, inner, ruffle, c = ids["points-2"], ids["ring-band"], ids["points-2-inner"], ids["ring-0"], ids["center"]
    assert big.z < ring.z < inner.z < ruffle.z < c.z
    assert big.color_group == 1 and inner.color_group == 1 and ring.color_group == 2 and ruffle.color_group == 0
    assert abs(big.params["r_outer"] / hr - 0.88) < 0.01 and abs(inner.params["r_outer"] / hr - 0.65) < 0.01
    assert abs(inner.params["r_inner"] / hr - 0.36) < 0.01 and big.params["rotation_deg"] == inner.params["rotation_deg"]
    assert abs(ring.params["r_inner"] / hr - 0.38) < 0.01 and abs(ring.params["r_outer"] / hr - 0.78) < 0.01
    assert abs(ruffle.params["r_inner"] / hr - 0.25) < 0.01 and abs(ruffle.params["r_outer"] / hr - 0.38) < 0.01
    assert "points-1" not in ids and "leaf" not in big.params and inner.params["leaf"] == 2  # leaves inside only


def test_r71_needs_two_twelve_point_layers():
    from rosette.pipeline import _star_on_pleated_ring
    parts = _s49_parts()
    parts[1].params["count"] = 8
    assert not _star_on_pleated_ring(parts)


def _a10b_parts(hr=154.4):
    geo = {"cx": 300.0, "cy": 176.5}
    return [Part("head", "head", dict(geo, r=hr), 0, 10, 0.9),
            Part("unknown-1", "unknown", {"cx": 356.6, "cy": 123.2, "r": 50.0}, 2, 20, 0.3),
            Part("center", "center", dict(geo, r=43.5), 1, 50, 0.9),
            Part("ring-0", "ring", dict(geo, r_inner=43.5, r_outer=97.5), 1, 30, 0.9),
            Part("ring-1", "ring", dict(geo, r_inner=43.5, r_outer=80.5), 0, 31, 0.9)]


def test_r78_pleated_rings_of_three_colors_round_a_rose():
    # A10-15B, the owner's check: pink pleats, a glitter ring, a white ring, a pink ring, a white rose
    from rosette.pipeline import _three_color_rings
    parts = _a10b_parts()
    assert _three_color_rings(parts)
    rings = sorted((q for q in parts if q.role == "ring"), key=lambda q: -q.params["r_outer"])
    hr = 154.4
    assert [q.color_group for q in rings] == [2, 1, 0]
    assert [round(q.params["r_outer"] / hr, 2) for q in rings] == [0.87, 0.7, 0.56]
    assert all(q.params["pinked"] == 1 for q in rings) and rings[0].z < rings[1].z < rings[2].z
    c = next(q for q in parts if q.role == "center")
    assert c.color_group == 1 and abs(c.params["r"] / hr - 0.36) < 0.01
    assert not any(q.role == "unknown" for q in parts)


def test_r78_needs_a_third_color():
    from rosette.pipeline import _three_color_rings
    parts = [q for q in _a10b_parts() if q.role != "unknown"]
    assert not _three_color_rings(parts)


def _a12_parts(hr=166.2):
    geo = {"cx": 300.0, "cy": 188.6}
    return [Part("head", "head", dict(geo, r=hr), 0, 10, 0.9),
            Part("unknown-1", "unknown", {"cx": 290.2, "cy": 170.5, "r": 38.2}, 1, 20, 0.3),
            Part("center", "center", dict(geo, r=58.5), 0, 50, 0.9)]


def test_r79_a_blob_of_a_second_color_at_the_badge_is_a_pleated_ring_round_it():
    # A12-15, the owner's check: wine pleats, a peach pleated ring, a wine badge
    from rosette.pipeline import _ring_round_badge
    parts = _a12_parts()
    assert _ring_round_badge(parts)
    ids = {q.id: q for q in parts}
    hr = 166.2
    ring, c = ids["ring-badge"], ids["center"]
    assert ring.color_group == 1 and ring.params["pinked"] == 1 and ring.z < c.z
    assert abs(ring.params["r_outer"] / hr - 0.60) < 0.01 and abs(c.params["r"] / hr - 0.38) < 0.01
    assert ring.params["r_inner"] == c.params["r"] and "unknown-1" not in ids


def test_r79_a_blob_away_from_the_middle_stays():
    from rosette.pipeline import _ring_round_badge
    parts = _a12_parts()
    parts[1].params.update(cx=400.0, cy=250.0)
    assert not _ring_round_badge(parts)


def _s2_parts():
    geo = {"cx": 300.0, "cy": 168.5}
    return [Part("points-0", "points", dict(geo, kind="bow", count=4, r_inner=26.6, r_outer=143.5, rotation_deg=42.6,
                                            width_ratio=0.6), 1, 21, 0.9),
            Part("points-1", "points", dict(geo, kind="star", count=4, r_inner=53.2, r_outer=143.5, rotation_deg=88.7), 0, 20, 0.9),
            Part("edge-points-0", "points", dict(geo, kind="bow", count=4, r_inner=26.6, r_outer=143.5, rotation_deg=42.6,
                                                 width_ratio=0.6, edge_width=6.0), 0, 22, 0.9),
            Part("center", "center", dict(geo, r=53.2), 0, 50, 0.9)]


def test_r81_four_edged_loops_between_leaf_points_of_the_head_color():
    # S2-15, the owner's check: pink loops edged wine, wine leaf points between, a wine rose; built whole (no library head)
    from rosette.pipeline import _loops_between_leaf_points
    parts = _s2_parts()
    assert _loops_between_leaf_points(parts)
    ids = {q.id: q for q in parts}
    T = 143.5 / 0.88
    star, loops, edge, c = ids["points-1"], ids["points-0"], ids["edge-points-0"], ids["center"]
    assert "leaf" not in star.params and star.params["rotation_deg"] == 0.0 and abs(star.params["r_outer"] - T) < 0.01
    assert loops.params["rotation_deg"] == 45.0 and abs(loops.params["r_outer"] / T - 0.90) < 0.01
    assert abs(loops.params["width_ratio"] * (loops.params["r_outer"] - loops.params["r_inner"]) / T - 0.54) < 0.01
    assert abs(c.params["r"] / T - 0.35) < 0.01 and star.z < loops.z < edge.z < c.z
    assert all(q.params.get("tips") == 1 for q in (star, loops, edge))


def test_r81_needs_four_loops():
    from rosette.pipeline import _loops_between_leaf_points
    parts = _s2_parts()
    for q in parts[:3]:
        q.params["count"] = 6
    assert not _loops_between_leaf_points(parts)


def test_r81_edges_read_as_a_loop_layer_behind():
    # S2-15B: the loops' white edges read as a second, wider loop layer behind the loops
    from rosette.pipeline import _loops_between_leaf_points
    parts = _s2_parts()
    edge = parts[2]
    edge.params.pop("edge_width"); edge.color_group = 2; edge.z = 19
    assert _loops_between_leaf_points(parts)
    assert edge.params["edge_width"] > 1 and edge.color_group == 2 and parts[0].z < edge.z


def test_r82_a_ring_inside_a_ring_of_its_color_is_dropped():
    # S53-19, the owner's check: the white ring read twice drew an extra wavy border and circle line
    from rosette.pipeline import _drop_nested_rings
    geo = {"cx": 300.0, "cy": 150.0}
    parts = [Part("ring-0", "ring", dict(geo, r_inner=75.5, r_outer=98.5, pinked=1), 1, 30, 0.9),
             Part("ring-1", "ring", dict(geo, r_inner=83.5, r_outer=95.5, pinked=1), 1, 31, 0.9),
             Part("ring-2", "ring", dict(geo, r_inner=50.5, r_outer=81.5, pinked=1), 2, 32, 0.9)]
    assert _drop_nested_rings(parts)
    assert [q.id for q in parts] == ["ring-0", "ring-2"]
    assert not _drop_nested_rings(parts)


def _s11_parts(hr=142.5):
    geo = {"cx": 300.0, "cy": 166.6}
    return [Part("head", "head", dict(geo, r=hr), 0, 10, 0.9),
            Part("points-1", "points", dict(geo, kind="star", count=6, r_inner=62.1, r_outer=108.5, rotation_deg=32.5), 1, 20, 0.9),
            Part("points-2", "points", dict(geo, kind="diamond", count=6, r_inner=40.8, r_outer=108.5, rotation_deg=2.5,
                                            width_ratio=0.6), 1, 21, 0.9),
            Part("center", "center", dict(geo, r=48.5), 0, 50, 0.9)]


def test_r83_kites_between_star_points_of_their_color_are_leaves_on_a_twelve_point_star():
    # S11-15, the owner's check: a twelve-point star with a narrow leaf on every other point
    from rosette.pipeline import _leaves_on_twelve_star
    parts = _s11_parts()
    assert _leaves_on_twelve_star(parts)
    ids = {q.id: q for q in parts}
    hr = 142.5
    star, leaves, c = ids["points-1"], ids["points-2"], ids["center"]
    assert star.params["count"] == 12 and abs(star.params["r_outer"] / hr - 0.916) < 0.01
    assert abs(star.params["r_inner"] / hr - 0.586) < 0.01 and star.params["rotation_deg"] == 0.0
    assert leaves.params["lens"] == 2 and leaves.params["rotation_deg"] == 0.0
    assert abs(leaves.params["r_inner"] / hr - 0.375) < 0.01 and abs(leaves.params["r_outer"] / hr - 0.918) < 0.01
    assert star.z < leaves.z < c.z and abs(c.params["r"] / hr - 0.388) < 0.01


def test_r83_needs_one_color():
    from rosette.pipeline import _leaves_on_twelve_star
    parts = _s11_parts()
    parts[2].color_group = 2
    assert not _leaves_on_twelve_star(parts)


def _s27_parts():
    geo = {"cx": 300.0, "cy": 165.0}
    return [Part("points-0", "points", dict(geo, kind="star", count=4, r_inner=78.9, r_outer=139.5, rotation_deg=49.4), 0, 21, 0.9),
            Part("points-1", "points", dict(geo, kind="bow", count=4, r_inner=53.5, r_outer=139.5, rotation_deg=5.9, width_ratio=0.8), 1, 20, 0.9),
            Part("center", "center", dict(geo, r=53.5), 1, 50, 0.9),
            Part("ring-0", "ring", dict(geo, r_inner=53.5, r_outer=92.5), 0, 30, 0.9)]


def test_r84_eight_loops_in_two_colors_round_a_ruffle_are_built_whole():
    # S27-15, the owner's check: the app's library head lost the ruffle; the photo's head is kept whole and upright
    from rosette.pipeline import _two_color_loops_round_ruffle
    parts = _s27_parts()
    assert _two_color_loops_round_ruffle(parts)
    ids = {q.id: q for q in parts}
    assert ids["points-1"].params["rotation_deg"] == 0.0 and ids["points-0"].params["rotation_deg"] == 45.0
    assert all(ids[k].params["tips"] == 1 for k in ("points-0", "points-1"))
    assert ids["ring-0"].params["pinked"] == 1 and ids["points-0"].params["kind"] == "bow"
    assert ids["points-1"].z < ids["points-0"].z  # the ruffle's color on top


def test_r84_needs_the_ruffle():
    from rosette.pipeline import _two_color_loops_round_ruffle
    parts = [q for q in _s27_parts() if q.role != "ring"]
    assert not _two_color_loops_round_ruffle(parts)


def _s5_parts():
    geo = {"cx": 300.0, "cy": 159.0}
    return [Part("head", "head", dict(geo, r=139.4), 0, 10, 0.9),
            Part("points-1", "points", dict(geo, kind="bow", count=4, r_inner=90.5, r_outer=133.5, rotation_deg=46.6, width_ratio=1.9), 1, 20, 0.9),
            Part("unknown-2", "unknown", dict(cx=302.8, cy=100.9, r=34.7), 2, 20, 0.5),
            Part("center", "center", dict(geo, r=56.5), 0, 50, 0.9),
            Part("ring-0", "ring", dict(geo, r_inner=57.5, r_outer=90.5), 2, 30, 0.9)]


def test_r85_disc_of_the_center_color_under_edged_loops_is_a_second_loop_layer():
    # S5-15: black loops on the axes (read as a disc), gold-edged glitter loops on the diagonals, a gold ruffle
    from rosette.pipeline import _edged_loops_on_loops
    parts = _s5_parts()
    assert _edged_loops_on_loops(parts)
    ids = {q.id: q for q in parts}
    assert "head" not in ids and "unknown-2" not in ids
    axis, edge, top = ids["points-axis"], ids["points-1-edge"], ids["points-1"]
    assert axis.color_group == 0 and edge.color_group == 2 and top.color_group == 1
    assert axis.params["rotation_deg"] == 0.0 and top.params["rotation_deg"] == 45.0 == edge.params["rotation_deg"]
    assert axis.z < edge.z < top.z and ids["ring-0"].params["pinked"] == 1
    assert edge.params["width_ratio"] > top.params["width_ratio"]


def test_r85_needs_the_disc_in_the_center_color():
    from rosette.pipeline import _edged_loops_on_loops
    parts = _s5_parts()
    parts[0].color_group = 2
    assert not _edged_loops_on_loops(parts)


def _plain_tail(**kw):
    p = {"tail_index": 0, "stripe_index": 0, "tails_count": 1, "top_y": 287.9, "bottom_y": 825.0, "top_width": 194.7,
         "bottom_width": 335.4, "spread_deg": 0.0, "stripes": "1.0000", "stripes_bottom": "1.0000"}
    p.update(kw)
    return Part("tail-0-0", "tail_stripe", p, 0, 9, 0.9)


def test_r86_a_plain_one_color_tail_is_two_ribbons():
    # A1-15, the owner's check: the designer's tail is two ribbons, the front one's edge a line from 0.28 of
    # the top to the middle of the bottom
    from rosette.pipeline import _two_ribbon_tail
    parts = [Part("head", "head", {"cx": 300.0, "cy": 180.0, "r": 157.0}, 0, 10, 0.9), _plain_tail()]
    assert _two_ribbon_tail(parts)
    st = sorted((q for q in parts if q.role == "tail_stripe"), key=lambda q: q.params["stripe_index"])
    assert [q.id for q in st] == ["tail-0-0", "tail-0-1"] and {q.color_group for q in st} == {0}
    assert all(q.params["stripes"] == "0.2800,0.7200" and q.params["stripes_bottom"] == "0.5000,0.5000" for q in st)


def test_r86_leaves_striped_or_several_tails():
    from rosette.pipeline import _two_ribbon_tail
    assert not _two_ribbon_tail([_plain_tail(stripes="0.5000,0.5000", stripes_bottom="0.5000,0.5000")])
    assert not _two_ribbon_tail([_plain_tail(tails_count=2)])


def test_r87_a_smooth_head_round_pleated_rings_is_pleated_too():
    # A9-15, the owner's check: three pleated bands, the outer one read smooth (its pleats are fine)
    from rosette.pipeline import _pleated_head_rim
    geo = {"cx": 300.0, "cy": 167.0}
    parts = [Part("head", "head", dict(geo, r=142.9, pinked=0), 0, 10, 0.9),
             Part("ring-0", "ring", dict(geo, r_inner=54.5, r_outer=113.5, pinked=1), 1, 30, 0.9),
             Part("center", "center", dict(geo, r=54.5), 1, 50, 0.9)]
    assert _pleated_head_rim(parts) and parts[0].params["pinked"] == 1


def test_r87_leaves_a_smooth_head_without_pleated_rings():
    from rosette.pipeline import _pleated_head_rim
    geo = {"cx": 300.0, "cy": 167.0}
    parts = [Part("head", "head", dict(geo, r=142.9, pinked=0), 0, 10, 0.9),
             Part("ring-0", "ring", dict(geo, r_inner=54.5, r_outer=113.5, pinked=0), 1, 30, 0.9)]
    assert not _pleated_head_rim(parts) and parts[0].params["pinked"] == 0
