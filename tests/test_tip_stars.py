"""R17: a points layer read as blocks whose lobes are the tips of a star covered by other layers
becomes that star; true blocks and free kites stay. Synthetic polar labels only."""
from types import SimpleNamespace

import numpy as np

from rosette.model import Part
from rosette.vision import parts_head as H
from rosette.vision.parts_common import Frame

T = H.RING_THETA
FR = Frame(axis_x=0.0, top=0.0, scale=1.0)
RADII = np.arange(0.5, 150, 1.0)
WIN = np.ones(T, bool)
TH = np.linspace(0, 360, T, endpoint=False)


def _off_axis(n, rot):
    """Angle (radians) of each column from its nearest axis rot + k 360/n."""
    step = 360.0 / n
    d = (TH - rot) % step
    return np.radians(np.minimum(d, step - d))


def _star_mask(n, r_tip, r_valley, rot=0.0):
    side = H._star_side(np.minimum(_off_axis(n, rot), np.pi / n), r_tip, r_valley, n)
    return RADII[:, None] <= side[None, :]


def _block_mask(n, r0, r1, half_width, rot=0.0):
    lateral = RADII[:, None] * np.sin(_off_axis(n, rot))[None, :]
    return (lateral <= half_width) & (RADII[:, None] >= r0) & (RADII[:, None] <= r1)


def _points(pid, g, kind, n, ri, ro, rot=0.0, wr=0.6, z=20):
    return Part(pid, "points", {"cx": 300.0, "cy": 200.0, "kind": kind, "count": n, "r_inner": ri,
                                "r_outer": ro, "rotation_deg": rot, "width_ratio": wr}, g, z, 0.8)


def _ring(g, ri, ro):
    return Part(f"ring-{g}", "ring", {"cx": 300.0, "cy": 200.0, "r_inner": ri, "r_outer": ro}, g, 30, 0.9)


def _head(g):
    return Part("head", "head", {"cx": 300.0, "cy": 200.0, "r": 140.0, "pinked": 1}, g, 10, 0.9)


def _star_under_ring():
    """An 8-point star (label 1, tips 120, valleys 50) under a ring (label 0, rows 40-75) on the
    head disc (5): only its tips show beyond the ring."""
    pol = np.full((len(RADII), T), 5, np.int32)
    pol[_star_mask(8, 120.0, 50.0)] = 1
    pol[40:75] = 0
    return pol


def test_cut_a_is_off_by_default():
    """TIP_CUT_A is off (the color gate, docs/pipeline.md R17): tips beyond a ring stay as read."""
    assert H.TIP_CUT_A is False
    pol = _star_under_ring()
    lobes = _points("points-1", 1, "bow", 8, 75.0, 119.0)
    parts = [_head(5), lobes, _ring(0, 40.0, 75.0)]
    assert not H._tips_to_stars(parts, pol, RADII, WIN, FR, set(), None, None)
    assert lobes.params["kind"] == "bow"


def test_star_tips_beyond_a_ring_become_one_star(monkeypatch):
    monkeypatch.setattr(H, "TIP_CUT_A", True)
    pol = _star_under_ring()
    lobes = _points("points-1", 1, "bow", 8, 75.0, 119.0)
    parts = [_head(5), lobes, _ring(0, 40.0, 75.0)]
    f = H._tip_lobes(lobes, parts, pol, RADII, WIN, FR)
    assert H._is_tip_layer(f) == "A"
    assert H._tips_to_stars(parts, pol, RADII, WIN, FR, set(), None, None)
    q = lobes.params
    assert q["kind"] == "star" and q["count"] == 8 and q["rotation_deg"] == 0.0
    assert abs(q["r_outer"] - 119.5) <= 1.5
    # the valleys lie under the ring (hidden), near the true star's 50
    assert 40.0 <= q["r_inner"] <= 75.0
    assert abs(q["r_inner"] - 50.0) <= 8.0
    assert lobes.color_group == 1 and lobes.z == 20


def test_true_blocks_under_a_ring_stay_blocks(monkeypatch):
    """Loop blocks (constant width 24 units, rows 75-120) running under the same ring."""
    monkeypatch.setattr(H, "TIP_CUT_A", True)
    pol = np.full((len(RADII), T), 5, np.int32)
    pol[_block_mask(6, 50.0, 120.0, 12.0)] = 1
    pol[40:75] = 0
    lobes = _points("points-1", 1, "bow", 6, 75.0, 120.0)
    parts = [_head(5), lobes, _ring(0, 40.0, 75.0)]
    assert H._is_tip_layer(H._tip_lobes(lobes, parts, pol, RADII, WIN, FR)) is None
    assert not H._tips_to_stars(parts, pol, RADII, WIN, FR, set(), None, None)
    assert lobes.params["kind"] == "bow"


def test_free_kites_on_the_disc_stay_kites(monkeypatch):
    """Kites (label 1) lying on the head disc: their inner ends narrow to a point on the disc,
    nothing drawn above them cuts them."""
    monkeypatch.setattr(H, "TIP_CUT_A", True)
    pol = np.full((len(RADII), T), 5, np.int32)
    lateral = RADII[:, None] * np.sin(_off_axis(6, 0.0))[None, :]
    t = (RADII - 60.0) / 60.0
    half = np.where(t <= 0.4, t / 0.4, (1 - t) / 0.6) * 15.0
    pol[(lateral <= half[:, None]) & (t[:, None] >= 0) & (t[:, None] <= 1)] = 1
    lobes = _points("points-1", 1, "diamond", 6, 60.0, 120.0)
    parts = [_head(5), lobes]
    assert H._is_tip_layer(H._tip_lobes(lobes, parts, pol, RADII, WIN, FR)) is None
    assert not H._tips_to_stars(parts, pol, RADII, WIN, FR, set(), None, None)


def test_star_tips_in_the_valleys_of_a_star_drawn_above_become_one_star():
    """A white 6-point star (1) behind a gold one (2) turned half a step: the white shows as tips
    in the gold star's valleys (rule B)."""
    pol = np.full((len(RADII), T), 5, np.int32)
    pol[_star_mask(6, 110.0, 45.0, rot=30.0)] = 1
    pol[_star_mask(6, 105.0, 55.0, rot=0.0)] = 2
    pol[:40] = 0
    white = _points("points-1", 1, "bow", 6, 50.0, 109.0, rot=30.0, z=20)
    gold = _points("points-2", 2, "star", 6, 55.0, 105.0, rot=0.0, z=21)
    parts = [_head(5), white, gold, Part("center", "center", {"cx": 300.0, "cy": 200.0, "r": 40.0}, 0, 50, 1.0)]
    f = H._tip_lobes(white, parts, pol, RADII, WIN, FR)
    assert f["interleaved"]
    assert H._is_tip_layer(f) == "B"
    assert H._tips_to_stars(parts, pol, RADII, WIN, FR, set(), None, None)
    assert white.params["kind"] == "star" and white.params["count"] == 6
    assert gold.params["kind"] == "star" and gold.params["r_outer"] == 105.0


def test_a_layer_hosting_trims_is_left_alone(monkeypatch):
    monkeypatch.setattr(H, "TIP_CUT_A", True)
    pol = _star_under_ring()
    lobes = _points("points-1", 1, "bow", 8, 75.0, 119.0)
    strips = _points("points-2", 3, "bow", 8, 75.0, 119.0)
    parts = [_head(5), lobes, strips, _ring(0, 40.0, 75.0)]
    assert not H._tips_to_stars(parts, pol, RADII, WIN, FR, {id(strips)}, None, None)
    assert lobes.params["kind"] == "bow"


def test_pieces_holding_the_lobes_give_the_same_kind_and_count(monkeypatch):
    """With a SAM 2 piece of the lobes' color over the star (in a photo frame of scale 1), the star
    is read from the piece (source "piece"), with the rules' kind and count."""
    monkeypatch.setattr(H, "TIP_CUT_A", True)
    pol = _star_under_ring()
    cx = cy = 200.0
    yy, xx = np.mgrid[0:400, 0:400]
    r = np.hypot(xx - cx, yy - cy)
    th = np.degrees(np.arctan2(yy - cy, xx - cx)) % 360
    step = 45.0
    d = th % step
    phi = np.radians(np.minimum(d, step - d))
    mask = (r <= H._star_side(phi, 122.0, 50.0, 8)) & (r >= 75.0)
    piece = SimpleNamespace(mask=mask, in_head=True, color_group=1, area=int(mask.sum()))
    p = SimpleNamespace(head=(cx, cy, 140.0))
    lobes = _points("points-1", 1, "bow", 8, 75.0, 119.0)
    parts = [_head(5), lobes, _ring(0, 40.0, 75.0)]
    assert H._tips_to_stars(parts, pol, RADII, WIN, FR, set(), [piece], p)
    q = lobes.params
    assert q["kind"] == "star" and q["count"] == 8 and q["source"] == "piece"
    assert abs(q["r_outer"] - 121.5) <= 1.5


def _interleaved(gold_z=21, gold_ri=55.0):
    pol = np.full((len(RADII), T), 5, np.int32)
    pol[_star_mask(6, 110.0, 45.0, rot=30.0)] = 1
    pol[_star_mask(6, 105.0, 55.0, rot=0.0)] = 2
    pol[:40] = 0
    white = _points("points-1", 1, "bow", 6, 50.0, 109.0, rot=30.0, z=20)
    gold = _points("points-2", 2, "star", 6, gold_ri, 105.0, rot=0.0, z=gold_z)
    center = Part("center", "center", {"cx": 300.0, "cy": 200.0, "r": 40.0}, 0, 50, 1.0)
    return pol, white, [_head(5), white, gold, center]


def test_rule_b_places_tips_and_valleys_from_the_side_fit():
    pol, white, parts = _interleaved()
    assert H._tips_to_stars(parts, pol, RADII, WIN, FR, set(), None, None)
    assert abs(white.params["r_outer"] - 110.0) <= 1.5
    assert abs(white.params["r_inner"] - 45.0) <= 1.5


def test_rule_b_needs_the_interleaved_star_above():
    """The same tips with the gold star drawn BELOW the white one: nothing cuts them."""
    pol, white, parts = _interleaved(gold_z=19)
    f = H._tip_lobes(white, parts, pol, RADII, WIN, FR)
    assert not f["interleaved"] and H._is_tip_layer(f) is None
    assert not H._tips_to_stars(parts, pol, RADII, WIN, FR, set(), None, None)
    assert white.params["kind"] == "bow"


def test_rule_b_needs_a_drawn_cover_beside_the_lobe():
    """The gold star drawn only from 100 out: the labels beside the white lobe where it narrows are
    no drawn cover (side under TIP_SIDE)."""
    pol, white, parts = _interleaved(gold_ri=100.0)
    f = H._tip_lobes(white, parts, pol, RADII, WIN, FR)
    assert f["interleaved"] and f["side"] < H.TIP_SIDE
    assert H._is_tip_layer(f) is None


def test_rule_b_refuses_without_side_evidence():
    """Tips widest at their first row (nothing beside a narrower lobe to read) on axes interleaved
    with a higher star: no evidence of a cover beside them, so B refuses (not side 1.0 by default)."""
    pol = np.full((len(RADII), T), 5, np.int32)
    pol[_star_mask(6, 110.0, 30.0, rot=30.0) & (RADII[:, None] >= 60.0)] = 1
    pol[:60] = 0
    white = _points("points-1", 1, "bow", 6, 60.0, 109.0, rot=30.0, z=20)
    gold = _points("points-2", 2, "star", 6, 20.0, 40.0, rot=0.0, z=21)
    center = Part("center", "center", {"cx": 300.0, "cy": 200.0, "r": 60.0}, 0, 50, 1.0)
    parts = [_head(5), white, gold, center]
    f = H._tip_lobes(white, parts, pol, RADII, WIN, FR)
    assert f["interleaved"] and f["peak"] == 0.0 and f["side"] == 0.0
    assert H._is_tip_layer(f) is None


def test_rounded_flank_refused_by_bulge_alone(monkeypatch):
    """Lobes widest at the ring and rounded outward (a petal's convex flank): every bar but
    TIP_BULGE passes, and the layer stays as read."""
    monkeypatch.setattr(H, "TIP_CUT_A", True)
    pol = np.full((len(RADII), T), 5, np.int32)
    lateral = RADII[:, None] * np.sin(_off_axis(6, 0.0))[None, :]
    t = np.clip((RADII - 60.0) / 60.0, 0, None)
    half = 14.0 * np.sqrt(np.clip(1 - (0.95 * t) ** 2, 0, None))
    pol[(lateral <= half[:, None]) & (RADII[:, None] >= 50.0) & (RADII[:, None] <= 120.0)] = 1
    pol[40:62] = 0
    lobes = _points("points-1", 1, "bow", 6, 62.0, 120.0)
    parts = [_head(5), lobes, _ring(0, 40.0, 62.0)]
    f = H._tip_lobes(lobes, parts, pol, RADII, WIN, FR)
    assert f["sym"] >= H.SPLIT_FLOOR and f["peak"] <= H.TIP_PEAK and f["end"] <= H.TIP_END
    assert f["inner"] >= H.TIP_CUT_WIDE and f["cover"] >= H.COVER_CUT_SHARE
    assert f["bulge"] > H.TIP_BULGE
    assert H._is_tip_layer(f) is None


def test_a_piece_reaching_past_the_lobe_falls_back_to_the_labels(monkeypatch):
    """A piece of the lobes' color that holds the lobe but runs on to 138 (merged with another
    region): its tip is not the lobe's, so the tips come from the labels."""
    monkeypatch.setattr(H, "TIP_CUT_A", True)
    pol = _star_under_ring()
    cx = cy = 200.0
    yy, xx = np.mgrid[0:400, 0:400]
    r = np.hypot(xx - cx, yy - cy)
    th = np.degrees(np.arctan2(yy - cy, xx - cx)) % 360
    d = th % 45.0
    phi = np.radians(np.minimum(d, 45.0 - d))
    mask = (r <= H._star_side(phi, 138.0, 50.0, 8)) & (r >= 75.0)
    piece = SimpleNamespace(mask=mask, in_head=True, color_group=1, area=int(mask.sum()))
    p = SimpleNamespace(head=(cx, cy, 140.0))
    lobes = _points("points-1", 1, "bow", 8, 75.0, 119.0)
    parts = [_head(5), lobes, _ring(0, 40.0, 75.0)]
    assert H._tips_to_stars(parts, pol, RADII, WIN, FR, set(), [piece], p)
    q = lobes.params
    assert q["kind"] == "star" and "source" not in q
    assert abs(q["r_outer"] - 119.5) <= 1.5
