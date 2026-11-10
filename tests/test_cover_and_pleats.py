"""H11: points layers start at the cover, a star-shaped unknown under a star is a points
layer, convex 'stars' and high-count pleats are rings. Synthetic labels only."""
import numpy as np

from rosette.build import build
from rosette.model import ColorGroup, Part, RosetteModel
from rosette.vision import parts_head as H
from rosette.vision.parts import detect_parts
from rosette.vision.parts_common import Frame
from rosette.vision.prepare import Prepared
from rosette.writer import write_svg

T = H.RING_THETA
FR = Frame(axis_x=0.0, top=0.0, scale=1.0)
RADII = np.arange(0.5, 150, 1.0)
WIN = np.ones(T, bool)


def _axis_cols(n, rot=0.0, half=6):
    cols = set()
    for k in range(n):
        c = int(round((rot + 360.0 * k / n) / 360.0 * T))
        cols.update((c + d) % T for d in range(-half, half + 1))
    return sorted(cols)


def _points(g, kind, n, ri, ro, rot=0.0, wr=0.3):
    return Part(f"points-{g}", "points", {"cx": 300.0, "cy": 200.0, "kind": kind, "count": n, "r_inner": ri,
                                          "r_outer": ro, "rotation_deg": rot, "width_ratio": wr}, g, 20, 0.8)


def _ring(g, ri, ro):
    return Part(f"ring-{g}", "ring", {"cx": 300.0, "cy": 200.0, "r_inner": ri, "r_outer": ro}, g, 30, 0.9)


def _pol_petals_under(cover_label):
    """12 petals (label 1) visible from row 70 out, a full cover (cover_label) over rows
    50-70 in front of them, a ring (label 0) at rows 30-50, the head disc (5) elsewhere."""
    pol = np.full((len(RADII), T), 5, np.int32)
    pol[70:120][:, _axis_cols(12)] = 1
    pol[50:70] = cover_label
    pol[30:50] = 0
    return pol


def test_petals_starting_on_the_head_disc_keep_their_visible_start():
    parts = [_ring(0, 30.0, 50.0), _points(1, "petal", 12, 70.0, 120.0)]
    H._cover_start(parts, _pol_petals_under(5), RADII, WIN, FR, head_label=5, trims=set())
    assert parts[1].params["r_inner"] == 70.0


def test_petals_under_six_loop_blocks_drawn_above_them_start_at_the_ring():
    """12 petals (label 1, z 22) whose bases are hidden by a layer of six wide loop
    blocks (label 4, z 24: ending further in, drawn on top) that fill rows 50-70."""
    parts = [_ring(0, 30.0, 50.0), _points(1, "petal", 12, 70.0, 120.0)]
    parts[1].z = 22
    loops = _points(4, "bow", 6, 50.0, 70.0, wr=1.2)
    loops.z = 24
    H._cover_start(parts + [loops], _pol_petals_under(4), RADII, WIN, FR, head_label=5, trims=set())
    assert parts[1].params["r_inner"] == 50.0
    # the width stays: 0.3 * 50 units over the longer lobe
    assert abs(parts[1].params["width_ratio"] - 0.3 * 50 / 70) < 1e-3


def test_petals_over_a_layer_drawn_below_them_keep_their_visible_start():
    parts = [_ring(0, 30.0, 50.0), _points(1, "petal", 12, 70.0, 120.0)]
    parts[1].z = 24
    under = _points(4, "bow", 6, 50.0, 140.0, wr=1.2)
    under.z = 20
    H._cover_start(parts + [under], _pol_petals_under(4), RADII, WIN, FR, head_label=5, trims=set())
    assert parts[1].params["r_inner"] == 70.0


def test_petals_whose_cover_reaches_no_ring_keep_their_visible_start():
    pol = _pol_petals_under(4)
    pol[30:50] = 4  # the cover runs on to the middle: no ring or center to start at
    parts = [_points(1, "petal", 12, 70.0, 120.0), _ring(4, 30.0, 70.0)]
    parts[1].role = "unknown"
    H._cover_start(parts, pol, RADII, WIN, FR, head_label=5, trims=set())
    assert parts[0].params["r_inner"] == 70.0


def test_stars_keep_their_valleys():
    pol = np.full((len(RADII), T), 5, np.int32)
    pol[30:50] = 0
    pol[50:80] = 4
    pol[80:120][:, _axis_cols(12, half=10)] = 1
    parts = [_ring(0, 30.0, 50.0), _ring(4, 50.0, 80.0), _points(1, "star", 12, 80.0, 120.0)]
    H._cover_start(parts, pol, RADII, WIN, FR, head_label=5, trims=set())
    assert parts[2].params["r_inner"] == 80.0


def _star_outline_pol(n=6, r_valley=60, r_tip=110, rot=0.0, seed=0):
    """Label 1 filling an n-point star outline from row 40, its inside blotched with
    label 2 (print, glare) unevenly, so that its pixels are not rotationally symmetric."""
    th = np.linspace(0, 2 * np.pi, T, endpoint=False)
    phase = np.abs(((np.degrees(th) - rot) % (360 / n)) - 180 / n) / (180 / n)  # 1 at a tip, 0 in a valley
    edge = r_valley + (r_tip - r_valley) * phase
    rows = np.arange(len(RADII))[:, None]
    pol = np.full((len(RADII), T), 5, np.int32)
    pol[(rows >= 40) & (rows < edge[None])] = 1
    rng = np.random.default_rng(seed)
    blot = rng.random((len(RADII), T)) < 0.95
    blot[:, T // 2:] = False
    pol[(pol == 1) & blot & (rows < edge[None] - 4)] = 2  # a thin rim of the outline stays
    return pol


def _layers_with_unknown(count, r_outer, unknown_rows=(40, 115)):
    layers = [("points", 3, 0.8, {"kind": "bow", "count": count, "r_inner": 50.0, "r_outer": r_outer,
                                   "rotation_deg": 30.0, "width_ratio": 0.6}, 1.0, int(r_outer)),
              ("unknown", 1, 0.2, {"cx": 300.0, "cy": 200.0, "r": 60.0}, 1.0, unknown_rows[1])]
    meta = [dict(zone=(40, 150), rows=(50, int(r_outer))), dict(zone=(40, 150), rows=unknown_rows)]
    return layers, meta


def test_an_unknown_star_outline_inside_a_star_is_a_points_layer():
    pol = _star_outline_pol()
    _, share, _ = H._symmetry(pol[40:115] == 1, WIN)
    assert share < H.LOW_SYM  # the pixels alone do not show it
    layers, meta = _layers_with_unknown(6, 120.0)
    H._star_from_outline(layers, meta, pol, RADII, WIN, FR, 300.0, 200.0)
    role, g, conf, q = layers[1][:4]
    assert role == "points" and g == 1
    assert q["kind"] == "star" and q["count"] == 6
    assert abs(q["r_outer"] - 110) <= 3 and abs(q["r_inner"] - 60) <= 5
    assert min(q["rotation_deg"] % 60, 60 - q["rotation_deg"] % 60) <= 2
    assert conf <= H.BLOB_UNCONFIRMED_CONF  # flagged


def test_an_unknown_star_outline_without_a_points_layer_of_its_symmetry_stays_unknown():
    pol = _star_outline_pol()
    for count in (4, 5, 8, 18):
        layers, meta = _layers_with_unknown(count, 120.0)
        H._star_from_outline(layers, meta, pol, RADII, WIN, FR, 300.0, 200.0)
        assert layers[1][0] == "unknown"


def test_a_star_outline_reaching_far_beyond_the_points_stays_unknown():
    pol = _star_outline_pol(r_tip=140)
    layers, meta = _layers_with_unknown(6, 100.0, unknown_rows=(40, 145))
    H._star_from_outline(layers, meta, pol, RADII, WIN, FR, 300.0, 200.0)
    assert layers[1][0] == "unknown"


def test_a_star_without_notches_is_not_a_star():
    # 4-point "stars" whose valleys lie outside the square through their tips
    assert H._convex_star({"kind": "star", "count": 4, "r_inner": 99.5, "r_outer": 133.5})
    assert H._convex_star({"kind": "star", "count": 4, "r_inner": 95.5, "r_outer": 131.5})
    assert not H._convex_star({"kind": "star", "count": 4, "r_inner": 50.0, "r_outer": 133.5})
    assert not H._convex_star({"kind": "star", "count": 6, "r_inner": 81.5, "r_outer": 129.5})
    assert not H._convex_star({"kind": "petal", "count": 4, "r_inner": 99.5, "r_outer": 133.5})


def _photo(labels, R, cx=450.0, cy=330.0):
    return Prepared(rgb=np.zeros(labels.shape + (3,), np.uint8), mask=labels >= 0, axis_x=cx, head=(cx, cy, R),
                    tails_top_y=cy + 0.5 * R, bottom_y=float(labels.shape[0] - 51), scale=2.0)


def _grid(h=1200, w=900, cx=450.0, cy=330.0):
    yy, xx = np.mgrid[:h, :w].astype(float)
    dx, dy = xx - cx, yy - cy
    return yy, xx, np.hypot(dx, dy), np.arctan2(dy, dx)


def test_a_4_point_star_read_without_notches_gets_its_valleys_at_the_disc_over_it():
    """A 4-point star (label 1) from 0.3 R to 0.95 R behind a disc (label 2) to 0.6 R:
    visible only beyond the disc, its lobes are widest at the disc's edge, so its valleys
    read there and the star would be a square; they lie under the disc."""
    yy, xx, r, th = _grid()
    R = 300.0
    labels = np.full(r.shape, -1, np.int32)
    labels[(yy > 330 + 0.5 * R) & (yy < 1150) & (np.abs(xx - 450) < 120)] = 3
    ph = np.abs(((np.degrees(th) % 90) - 45) / 45)  # 1 on the axes, 0 between
    labels[r <= (0.3 + 0.65 * ph) * R] = 1
    labels[r <= 0.6 * R] = 2
    labels[r <= 0.2 * R] = 0
    parts = [q for q in detect_parts(_photo(labels, R), labels) if q.role != "tail_stripe"]
    stars = [q for q in parts if q.role == "points" and q.color_group == 1]
    assert len(stars) == 1 and stars[0].params["kind"] == "star" and stars[0].params["count"] == 4
    assert not H._convex_star(stars[0].params)
    assert stars[0].params["r_inner"] <= 0.6 * R / 2 + 2


def test_a_convex_star_is_never_drawn_as_a_star():
    yy, xx, r, th = _grid()
    R = 300.0
    labels = np.full(r.shape, -1, np.int32)
    labels[(yy > 330 + 0.5 * R) & (yy < 1150) & (np.abs(xx - 450) < 120)] = 3
    # an octagon-like 4-lobe shape of label 1 (lobes wider than their spacing) behind a
    # disc of label 2: no concave notches
    ph = np.abs(((np.degrees(th) % 90) - 45) / 45)  # 1 on the axes, 0 between
    labels[r <= (0.8 + 0.15 * ph) * R] = 1
    labels[r <= 0.6 * R] = 2
    labels[r <= 0.3 * R] = 0
    parts = [q for q in detect_parts(_photo(labels, R), labels) if q.role != "tail_stripe"]
    assert not any(q.role == "points" and q.params.get("kind") == "star" and H._convex_star(q.params)
                   for q in parts)


def _pleats_labels(n=24, fill=1.0):
    """A head disc (label 0) to R, n pleats (label 1) between a band of label 2 to
    0.65 R and the disc's band from 0.9 R, each a kite filling `fill` of its spacing at
    mid radius, a center (label 3) to 0.3 R and tails (4)."""
    yy, xx, r, th = _grid()
    R = 300.0
    labels = np.full(r.shape, -1, np.int32)
    labels[(yy > 330 + 0.5 * R) & (yy < 1150) & (np.abs(xx - 450) < 120)] = 4
    labels[r <= R] = 0
    ph = ((np.degrees(th) % (360 / n)) - 180 / n) / (180 / n)  # -1..1 across one pleat
    t = np.clip((r - 0.65 * R) / (0.25 * R), 0, 1)
    half = fill * (1 - np.abs(2 * t - 1))
    labels[(r > 0.65 * R) & (r < 0.9 * R) & (np.abs(ph) <= half)] = 1
    labels[r <= 0.65 * R] = 2
    labels[r <= 0.3 * R] = 3
    return labels, R


def test_pleat_rule_reads_spacing_filling_high_count_layers():
    bands = [H._Band(10, 20, 2, 1.0), H._Band(40, 50, 0, 1.0)]
    q = {"kind": "diamond", "count": 24, "r_inner": 72.5, "r_outer": 127.5, "width_ratio": 0.451}
    assert H._pleat_layer(q, (20, 40), bands)  # 24 kites filling 0.96 of their spacing
    assert not H._pleat_layer(dict(q, count=12, width_ratio=0.9), (20, 40), bands)  # too few lobes
    assert not H._pleat_layer(dict(q, width_ratio=0.2), (20, 40), bands)  # narrow points with gaps
    assert not H._pleat_layer(q, (20, 45), bands)  # open on the outside


def test_a_24_pleat_annulus_is_a_pinked_ring():
    labels, R = _pleats_labels()
    parts = [q for q in detect_parts(_photo(labels, R), labels) if q.role != "tail_stripe"]
    assert not any(q.role == "points" and q.color_group == 1 for q in parts)
    pinked = [q for q in parts if q.role == "ring" and q.params.get("pinked")]
    assert len(pinked) == 1 and pinked[0].color_group == 1 and pinked[0].params["pleats"] == 24
    assert pinked[0].params["r_inner"] < 0.7 * R / 2 + 5 and pinked[0].params["r_outer"] > 0.85 * R / 2


def test_pinked_ring_builds_and_writes():
    m = RosetteModel(parts=[Part("ring-0", "ring", {"cx": 300.0, "cy": 200.0, "r_inner": 40.0, "r_outer": 80.0,
                                                    "pinked": 1}, 1, 30, 0.9)],
                     groups=[ColorGroup(id=1, rgb=(200, 0, 0), pattern=False, texture=0.0, area=1.0, category=1, confidence=1.0)])
    svg = write_svg(build(m))
    assert 'fill-rule="evenodd"' in svg
    plain = RosetteModel(parts=[Part("ring-0", "ring", {"cx": 300.0, "cy": 200.0, "r_inner": 40.0,
                                                        "r_outer": 80.0}, 1, 30, 0.9)], groups=m.groups)
    assert len(write_svg(build(m))) > len(write_svg(build(plain)))  # the zigzag edge


def test_a_star_that_cannot_get_hidden_valleys_becomes_a_flagged_plain_ring(monkeypatch):
    """The annulus fallback (_hidden_valleys finds no hidden notches; with the current
    constants it is not reached on the 76 photos): a plain ring, flagged."""
    monkeypatch.setattr(H, "_hidden_valleys", lambda params, zone_start: None)
    yy, xx, r, th = _grid()
    R = 300.0
    labels = np.full(r.shape, -1, np.int32)
    labels[(yy > 330 + 0.5 * R) & (yy < 1150) & (np.abs(xx - 450) < 120)] = 3
    ph = np.abs(((np.degrees(th) % 90) - 45) / 45)
    labels[r <= (0.3 + 0.65 * ph) * R] = 1
    labels[r <= 0.72 * R] = 2  # the disc hides the star up to 0.75 of its tips: read without notches
    labels[r <= 0.2 * R] = 0
    parts = [q for q in detect_parts(_photo(labels, R), labels) if q.role != "tail_stripe"]
    assert not any(q.role == "points" and q.color_group == 1 for q in parts)
    rings = [q for q in parts if q.role == "ring" and q.color_group == 1]
    assert len(rings) == 1 and not rings[0].params.get("pinked")
    assert rings[0].confidence <= H.BLOB_UNCONFIRMED_CONF


def _pleat_pair(rot2, r_inner2=50.0):
    p1 = {"kind": "diamond", "count": 17, "r_inner": 50.0, "r_outer": 80.0, "rotation_deg": 3.9, "width_ratio": 0.8}
    p2 = dict(p1, kind="petal", r_inner=r_inner2, rotation_deg=rot2)
    layers = [("points", 0, 0.45, p1, 1.0, 80), ("points", 1, 0.45, p2, 1.0, 80)]
    meta = [dict(zone=(40, 90)), dict(zone=(40, 90))]
    return layers, meta


def test_an_interleaved_two_tone_pleat_pair_stays_points():
    spacing = 360 / 17
    layers, meta = _pleat_pair(3.9 + spacing / 2)  # lobes between each other's
    assert H._alternating(0, layers, meta) and H._alternating(1, layers, meta)
    layers, meta = _pleat_pair(3.9)  # on top of each other: not alternating
    assert not H._alternating(0, layers, meta)
    layers, meta = _pleat_pair(3.9 + spacing / 2, r_inner2=60.0)  # other radii (nested rings)
    assert not H._alternating(0, layers, meta)


def test_a_16_point_star_is_never_read_as_pleats():
    bands = [H._Band(10, 20, 2, 1.0), H._Band(40, 50, 0, 1.0)]
    q = {"kind": "star", "count": 16, "r_inner": 72.5, "r_outer": 127.5, "width_ratio": 1.0}
    assert not H._pleat_layer(q, (20, 40), bands)
    assert H._pleat_layer(dict(q, kind="diamond"), (20, 40), bands)
