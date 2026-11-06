"""H4: points layers drawn the way the designer draws them (loops, edge strips,
backing tips, specks). Synthetic label images only."""
import numpy as np

from rosette.vision import parts_head as H
from rosette.vision.parts import detect_parts
from rosette.vision.prepare import Prepared

# Lobe width profile of the S35-19 photo's loops (inner -> outer, share of the widest):
# the pleated ring hides the base, the folded loop end narrows the tip.
PHOTO_LOOP = [0.26, 0.56, 0.77, 0.92, 1.0, 0.99, 0.92, 0.86, 0.65, 0.50, 0.05]


def _profile(values, n=80):
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(values)), values)


def test_wide_lobe_keeping_its_width_is_a_loop_block():
    w = 90 * _profile(PHOTO_LOOP)
    assert H._kind(w) == "petal"  # by its width profile alone a pointed oval
    # covering 0.6 of the angular spacing and >= 0.85 of its widest over most radii
    assert H._kind(w, np.full(len(w), 150.0)) == "bow"
    # the same profile on a narrow lobe (0.3 of the spacing) stays a petal
    assert H._kind(w, np.full(len(w), 300.0)) == "petal"
    # a sine lobe has no plateau: a petal however wide
    sine = 90 * np.sin(np.pi * np.linspace(0.02, 0.98, 80))
    assert H._kind(sine, np.full(80, 150.0)) == "petal"


def loops_photo(n=6, profile=PHOTO_LOOP, edge_px=0, backing_px=0, teeth=True, edge_tip=None, seam_px=0):
    """Labels of a head: center (2) r < 0.3 R, ring (0) to 0.5 R, n loops (1) from
    0.47 R (just under the ring, whose pleat teeth reach 0.53 R) to 0.9 R with the lateral `profile`, and a tail (3) below.
    edge_px: a band of label 0 that wide along both loop sides. backing_px: n diamonds
    (4) between the loops, 0.5-0.85 R, on a label-0 copy reaching backing_px further.
    edge_tip=(side_px, tip_px): the same diamonds on a copy of a fifth label (5) showing
    side_px along their sides and tip_px beyond their tips. seam_px: the edge band lies
    that far off the loop sides (an antialiased seam of another label, S30-15)."""
    h, w = 1200, 900
    cx, cy, R = 450.0, 330.0, 300.0
    yy, xx = np.mgrid[:h, :w].astype(float)
    dx, dy = xx - cx, yy - cy
    r = np.hypot(dx, dy)
    labels = np.full((h, w), -1, np.int32)
    labels[(yy > cy + 0.5 * R) & (yy < 1150) & (np.abs(xx - cx) < 120 + (yy - cy) * 0.1)] = 3
    prof = np.asarray(profile, float)

    def lobes(rot, r0, r1, half, prof, pad=0.0, tip_pad=0.0):
        out = np.zeros_like(r, bool)
        for k in range(n):
            a = np.radians(rot + k * 360 / n)
            ax = dx * np.cos(a) + dy * np.sin(a)
            lat = -dx * np.sin(a) + dy * np.cos(a)
            t = np.clip((ax - r0) / (r1 - r0), 0, 1)
            hw = half * np.interp(t, np.linspace(0, 1, len(prof)), prof) + pad
            out |= (ax >= r0) & (ax <= r1 + tip_pad) & (np.abs(lat) <= hw)
        return out
    if backing_px:
        kite = [0, 0.4, 0.7, 0.97, 1, 0.97, 0.8, 0.6, 0.4, 0.2, 0]
        labels[lobes(180 / n, 0.5 * R, 0.85 * R, 0.06 * R, kite, backing_px / 3, backing_px)] = 0
        labels[lobes(180 / n, 0.5 * R, 0.85 * R, 0.06 * R, kite)] = 4
    if edge_tip:
        kite = [0, 0.4, 0.7, 0.97, 1, 0.97, 0.8, 0.6, 0.4, 0.2, 0]
        labels[lobes(180 / n, 0.5 * R, 0.85 * R, 0.06 * R, kite, edge_tip[0], edge_tip[1])] = 5
        labels[lobes(180 / n, 0.5 * R, 0.85 * R, 0.06 * R, kite)] = 4
    if edge_px:
        labels[lobes(0, 0.47 * R, 0.9 * R, 0.2 * R, prof, edge_px + seam_px)] = 0
    if seam_px:
        labels[lobes(0, 0.47 * R, 0.9 * R, 0.2 * R, prof, seam_px)] = -1
    labels[lobes(0, 0.47 * R, 0.9 * R, 0.2 * R, prof)] = 1
    labels[r <= 0.5 * R] = 0
    bins = ((np.degrees(np.arctan2(dy, dx)) % 360) * 2).astype(int) % 720
    reach = np.random.default_rng(1).uniform(0.49, 0.555, 720)  # hand-pleated teeth
    if teeth:
        labels[r <= reach[bins] * R] = 0
    labels[r <= 0.3 * R] = 2
    p = Prepared(rgb=np.zeros((h, w, 3), np.uint8), mask=labels >= 0, axis_x=cx, head=(cx, cy, R),
                 tails_top_y=cy + 0.5 * R, bottom_y=1149.0, scale=2.0)
    return p, labels


def test_photo_loops_are_loop_blocks_starting_at_the_ring():
    p, labels = loops_photo()
    parts = detect_parts(p, labels)
    loops = [q for q in parts if q.role == "points" and q.color_group == 1]
    rings = [q for q in parts if q.role == "ring"]
    assert len(loops) == 1 and len(rings) == 1
    q = loops[0].params
    assert q["kind"] == "bow" and q["count"] == 6
    # the loops run under the ring (its pleat teeth): they start where the ring ends
    assert rings[0].params["r_outer"] > 77
    assert q["r_inner"] == rings[0].params["r_outer"]
    # drawn with the designer's loop profile, not a photo-measured inner width
    assert "inner_ratio" not in q


def _head_parts(**kw):
    p, labels = loops_photo(**kw)
    return [q for q in detect_parts(p, labels) if q.role != "tail_stripe"]


def test_contrasting_band_along_both_loop_sides_becomes_edge_strips():
    # S35-19: lilac edges along the dark loops. The designer draws them as two strips
    # per loop (14 strip layers on loops in 13 files), 0.108 of the loop's width.
    parts = _head_parts(edge_px=8)
    loops = [q for q in parts if q.role == "points" and q.color_group == 1]
    strips = [q for q in parts if q.role == "points" and q.color_group == 0]
    assert len(loops) == 1 and len(strips) == 1
    lp, sp = loops[0].params, strips[0].params
    assert sp["kind"] == "bow" and sp["count"] == lp["count"]
    assert sp["rotation_deg"] == lp["rotation_deg"]
    assert (sp["r_inner"], sp["r_outer"], sp["width_ratio"]) == (lp["r_inner"], lp["r_outer"], lp["width_ratio"])
    w = lp["width_ratio"] * (lp["r_outer"] - lp["r_inner"])
    assert abs(sp["edge_width"] - 0.108 * w) < 0.05
    assert strips[0].z > loops[0].z  # painted over the loop's sides
    # the ring the band's color also forms stays a ring, grown over its pleats as before
    rings = [q for q in parts if q.role == "ring"]
    assert [q.color_group for q in rings] == [0] and rings[0].params["r_outer"] > 77
    assert not [q for q in parts if q.role == "unknown" and q.color_group == 0]


def test_edge_band_off_the_loop_by_a_seam_is_still_edge_strips():
    # S30-15: the pink edges of the dark loops are read with a 1-3 sample lilac seam
    # between loop and band (antialiasing); the band is still the loops' edge strips,
    # as the reference draws them (12 strips, 2 per loop)
    parts = _head_parts(edge_px=8, seam_px=3)
    strips = [q for q in parts if q.role == "points" and q.color_group == 0]
    assert len(strips) == 1 and strips[0].params.get("edge_width")


def test_no_edge_strips_without_a_band():
    parts = _head_parts()
    assert not [q for q in parts if "edge_width" in q.params]


def test_backing_showing_beyond_the_points_becomes_tips_behind_them():
    # S35-19: a lilac frame shows around the white diamonds' tips. The designer draws
    # the backing as a frame behind each point (11 files), never a disc under everything.
    parts = _head_parts(backing_px=12)
    pts = [q for q in parts if q.role == "points" and q.color_group == 4]
    back = [q for q in parts if q.role == "points" and q.color_group == 0]
    assert len(pts) == 1 and len(back) == 1
    pp, bp = pts[0].params, back[0].params
    assert bp["kind"] == pp["kind"] and bp["count"] == pp["count"]
    assert bp["rotation_deg"] == pp["rotation_deg"]
    # the backing reaches its measured tip: 0.85 R + 12 px = 133.5 units
    assert abs(bp["r_outer"] - 133.5) < 2 and bp["r_outer"] > pp["r_outer"] + 5
    assert bp["r_inner"] == pp["r_inner"] and abs(bp["width_ratio"] - pp["width_ratio"]) < 0.01
    assert back[0].z < pts[0].z
    rings = [q for q in parts if q.role == "ring"]
    assert [q.color_group for q in rings] == [0] and rings[0].params["r_outer"] > 77
    # the points' thin inner tips vanish in the ring's pleats: they start at the ring
    assert pp["r_inner"] == rings[0].params["r_outer"] == bp["r_inner"]
    assert not [q for q in parts if q.role == "unknown" and q.color_group == 0]


def test_band_along_the_sides_and_beyond_the_tips_is_a_same_kind_backing(monkeypatch):
    # S30-15: a pink band along the lilac kites' sides and a pink tip beyond them. The
    # reference files draw such trims as a same-kind backing (17 files; a kite behind
    # each kite in 7): one larger kite of the band's color behind each kite, same count
    # and angles, out by the measured margins. (The synthetic band is read with high
    # confidence; photos read it below EXPLAIN_MAX_CONF, S30-15 0.5.)
    from rosette.vision import parts_trims
    monkeypatch.setattr(parts_trims, "EXPLAIN_MAX_CONF", 1.0)
    parts = _head_parts(edge_tip=(6, 10))
    host = [q for q in parts if q.role == "points" and q.color_group == 4]
    back = [q for q in parts if q.role == "points" and q.color_group == 5]
    assert len(host) == 1 and len(back) == 1
    hp, bp = host[0].params, back[0].params
    assert bp["kind"] == hp["kind"] and bp["count"] == hp["count"]
    assert bp["rotation_deg"] == hp["rotation_deg"] and bp["r_inner"] == hp["r_inner"]
    # tip 10 px = 5 units beyond the drawn tip (0.85 R = 127.5); sides 6 px = 3 units each
    assert abs(bp["r_outer"] - 132.5) < 1.5
    w_h = hp["width_ratio"] * (hp["r_outer"] - hp["r_inner"])
    w_b = bp["width_ratio"] * (bp["r_outer"] - bp["r_inner"])
    assert abs((w_b - w_h) / 2 - 3) < 1.5
    assert back[0].z < host[0].z
    assert "edge_width" not in bp


def test_specks_of_a_band_are_folded_into_it_not_drawn_as_discs():
    # S35-19: the pinked edges of the center and of the ring leave small pieces of both
    # colors in the thin zone between them; they were drawn as two discs (unknown,
    # confidence 0.06). They are the bands' own fringe: no disc, and the ring closes
    # the gap down to the center.
    p, labels = loops_photo(teeth=False)
    h, w = labels.shape
    yy, xx = np.mgrid[:h, :w].astype(float)
    r = np.hypot(xx - 450.0, yy - 330.0)
    bins = ((np.degrees(np.arctan2(yy - 330.0, xx - 450.0)) % 360) * 2).astype(int) % 720
    rng = np.random.default_rng(2)
    inner_reach = rng.uniform(0.29, 0.36, 720)[bins] * 300  # the center's pinked edge
    outer_reach = rng.uniform(0.33, 0.40, 720)[bins] * 300  # the ring's pleats
    seam = (r > 0.3 * 300) & (r < 0.4 * 300)
    labels[seam] = -1
    labels[seam & (r >= outer_reach)] = 0
    labels[seam & (r < inner_reach)] = 2
    labels[(r > 0.3 * 300) & (r <= 0.5 * 300) & ~seam] = 0
    p = Prepared(rgb=p.rgb, mask=labels >= 0, axis_x=p.axis_x, head=p.head,
                 tails_top_y=p.tails_top_y, bottom_y=p.bottom_y, scale=p.scale)
    parts = [q for q in detect_parts(p, labels) if q.role != "tail_stripe"]
    assert not [q for q in parts if q.role == "unknown"], parts
    ring = [q for q in parts if q.role == "ring"]
    center = [q for q in parts if q.role == "center"]
    assert len(ring) == 1 and len(center) == 1
    assert ring[0].params["r_inner"] <= center[0].params["r"] + 1


def test_a_trim_never_outlives_its_host(monkeypatch):
    # Layer A (a low-confidence loop) hosts B's edge strips while A itself reads as the
    # backing of C: replacing A in the same pass would leave strips without their loop.
    from rosette.vision import parts_trims as T
    from rosette.vision.parts_common import Frame
    pol = np.full((40, 720), -1, np.int32)
    pol[10:20, 100:110] = 0  # B's pixels
    pol[10:20, 300:310] = 1  # A's pixels
    win = np.ones(720, bool)
    radii = np.arange(40, dtype=float) + 0.5
    zone = dict(zone=(0, 40), rows=(0, 30), eats=None, inner=None)
    a = ("points", 1, 0.5, dict(kind="bow", count=6, r_inner=5.0, r_outer=25.0, rotation_deg=0.0,
                                width_ratio=0.5), 100.0, 25)
    b = ("unknown", 0, 0.1, dict(cx=300, cy=150, r=5.0), 50.0, 25)
    c = ("points", 2, 0.9, dict(kind="diamond", count=6, r_inner=5.0, r_outer=20.0, rotation_deg=30.0,
                                width_ratio=0.4), 120.0, 20)
    monkeypatch.setattr(T, "_edge_bands", lambda pol, i0, i1, g, h, p, win, *a, **k: (1.0, pol == h))
    monkeypatch.setattr(T, "_backing_tips", lambda pol, radii, i_out, h, p, win, fr:
                        (1.0, 35, pol == h) if (h, p["kind"]) == (1, "diamond") else (0.0, i_out, pol < -1))
    layers, meta = T.explain_trims([a, b, c], [dict(zone) for _ in range(3)], pol, radii, win,
                                   Frame(axis_x=0.0, top=0.0, scale=1.0), 8.0)
    loops = [p for role, g, s, p, ar, rb in layers if g == 1 and p.get("kind") == "bow" and "edge_width" not in p]
    assert loops, layers  # the host loop stays
    assert [p for role, g, s, p, ar, rb in layers if "edge_width" in p]  # B became A's strips


def test_points_z_inner_layers_on_top_and_loops_over_stars():
    # H5 stacking mined from the references: of the points pieces that overlap or are
    # cut along each other, the one ending further in lies on top in 92 % (1292 of 1400
    # pairs with distinct ends); loop over star holds 89 % (63 of 71 pairs, 10 files)
    # whichever ends further out, so a star goes under every loop
    from rosette.vision.parts_head import points_z
    z = points_z([("points", "bow", 100), ("points", "diamond", 80)])
    assert z[1] > z[0]  # kite ending further in is on top
    z = points_z([("points", "bow", 100), ("points", "star", 80)])
    assert z[0] > z[1]  # the loop stays over the star
    z = points_z([("points", "bow", 100), ("points", "star", 120), ("points", "diamond", 70)])
    assert z[2] > z[0] > z[1]
    z = points_z([("points", "bow", 100), ("points", "star", 80), ("points", "diamond", 90)])
    assert z[2] > z[0] > z[1]  # kite over loop (inner on top), loop over star
    assert all(20 <= v <= 29 for v in z)


def test_points_z_stars_keep_their_own_order_under_loops():
    from rosette.vision.parts_head import points_z
    z = points_z([("points", "bow", 100), ("points", "star", 80), ("points", "star", 90)])
    assert z[0] > z[1] > z[2]  # both stars under the loop, the inner star over the outer
