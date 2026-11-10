"""H12: a points layer whose symmetry changes with radius is split (a 6-point star plus
the loops' edge strips at the rim, S29-15); points layers are stacked as the photo shows
them where they overlap. Synthetic labels only."""
import numpy as np

from rosette.model import Part
from rosette.vision import parts_head as H
from rosette.vision.parts import detect_parts
from rosette.vision.parts_common import Frame, polar_sample
from rosette.vision.parts_trims import EXPLAIN_MAX_CONF as H_EXPLAIN
from rosette.vision.prepare import Prepared

R = 300.0
CX, CY = 450.0, 330.0


def _grid(h=1200, w=900):
    yy, xx = np.mgrid[:h, :w].astype(float)
    dx, dy = xx - CX, yy - CY
    return yy, xx, dx, dy, np.hypot(dx, dy), np.arctan2(dy, dx)


def _photo(labels):
    return Prepared(rgb=np.zeros(labels.shape + (3,), np.uint8), mask=labels >= 0, axis_x=CX, head=(CX, CY, R),
                    tails_top_y=CY + 0.5 * R, bottom_y=float(labels.shape[0] - 51), scale=2.0)


def _star(th, n, rot, r_valley, r_tip):
    """Radius of an n-point star's outline at angle th."""
    phase = np.abs(((np.degrees(th) - rot) % (360 / n)) - 180 / n) / (180 / n)  # 1 at a tip
    return r_valley + (r_tip - r_valley) * phase


def _blocks(dx, dy, n, rot, r0, r1, half):
    out = np.zeros(dx.shape, bool)
    for k in range(n):
        a = np.radians(rot + k * 360 / n)
        ax = dx * np.cos(a) + dy * np.sin(a)
        lat = -dx * np.sin(a) + dy * np.cos(a)
        out |= (ax >= r0) & (ax <= r1) & (np.abs(lat) <= half)
    return out


def _tails(labels, yy, xx):
    labels[(yy > CY + 0.5 * R) & (yy < 1150) & (np.abs(xx - CX) < 110)] = 3


def star_over_loops(strips=True, star_on_top=True):
    """Tails (3), six loop blocks (1) on the axes 0 + 60 k from 0.4 R to 0.98 R, a pink
    (4) 6-point star on the axes 30 + 60 k (valleys 0.5 R, tips 0.85 R) and, with
    `strips`, pink strips along both sides of each loop from 0.55 R out (centered about
    15 degrees off the loop axes at the rim: 12 lobes there), a ring (0) to 0.4 R and a
    center (2) to 0.25 R."""
    yy, xx, dx, dy, r, th = _grid()
    labels = np.full(r.shape, -1, np.int32)
    _tails(labels, yy, xx)
    half = 0.19 * R
    loops = _blocks(dx, dy, 6, 0, 0.4 * R, 0.98 * R, half)
    edge = _blocks(dx, dy, 6, 0, 0.55 * R, 0.98 * R, 0.3 * R) & ~loops
    star = r <= _star(th, 6, 30, 0.5 * R, 0.85 * R)
    if star_on_top:
        labels[loops] = 1
        if strips:
            labels[edge] = 4
        labels[star] = 4
    else:
        labels[star] = 4
        if strips:
            labels[edge] = 4
        labels[loops] = 1
    labels[r <= 0.4 * R] = 0
    labels[r <= 0.25 * R] = 2
    return labels


def _head(labels):
    return [q for q in detect_parts(_photo(labels), labels) if q.role != "tail_stripe"]


def test_band_orders_read_the_count_per_radius():
    labels = star_over_loops()
    radii = np.arange(0.5 * R, 0.98 * R, 2.0)
    pol = polar_sample(labels, CX, CY, radii, H.RING_THETA, fill=-1)
    theta = np.degrees(np.linspace(0, 2 * np.pi, H.RING_THETA, endpoint=False))
    win = np.abs(theta - 90.0) > H.TAIL_SECTOR_DEG
    _, orders = H.band_orders(pol == 4, win)
    assert orders[0] == 6 and orders[-1] == 12
    assert H.radial_split(orders) is not None


def test_radial_split_rules():
    assert H.radial_split([0, 6, 6, 6, 6, 6, 12, 12]) == (6, 12, 6)
    assert H.radial_split([12] * 8) is None  # a plain 12-point layer
    assert H.radial_split([6] * 8) is None
    assert H.radial_split([0, 0, 6, 0, 0, 0, 0, 12]) is None  # one inner band is not enough
    assert H.radial_split([6, 3, 3, 6, 6, 6, 6, 6]) is None  # the outer count shows inside too
    assert H.radial_split([12, 12, 6, 6, 6, 6, 6, 6]) is None  # finer inside, not outside
    assert H.radial_split([6, 6, 6, 6, 6, 6, 9, 9]) is None  # not a multiple
    # the largest inner count dividing c2, not the bands' noisy majority
    assert H.radial_split([6, 6, 4, 4, 4, 4, 12, 12]) == (6, 12, 6)
    assert H.radial_split([6, 5, 5, 5, 5, 6, 12, 12]) == (6, 12, 6)


def test_a_six_point_star_with_twelve_strips_at_the_rim_is_split():
    parts = _head(star_over_loops())
    pink = [q for q in parts if q.role == "points" and q.color_group == 4]
    loops = [q for q in parts if q.role == "points" and q.color_group == 1]
    assert len(loops) == 1 and loops[0].params["count"] == 6
    stars = [q for q in pink if "edge_width" not in q.params]
    strips = [q for q in pink if "edge_width" in q.params]
    assert len(stars) == 1 and stars[0].params["count"] == 6 and stars[0].params["kind"] == "star"
    assert abs((stars[0].params["rotation_deg"] - 30 + 30) % 60 - 30) < 3
    assert abs(stars[0].params["r_outer"] - 0.85 * R / 2) < 6  # its tips, not where the 12 lobes start
    # the rim lobes are the loops' edge strips
    assert len(strips) == 1 and strips[0].params["count"] == 6
    assert strips[0].params["rotation_deg"] == loops[0].params["rotation_deg"]


def test_a_plain_twelve_point_star_is_not_split():
    yy, xx, dx, dy, r, th = _grid()
    labels = np.full(r.shape, -1, np.int32)
    _tails(labels, yy, xx)
    labels[r <= _star(th, 12, 0, 0.55 * R, 0.95 * R)] = 1
    labels[r <= 0.4 * R] = 0
    labels[r <= 0.25 * R] = 2
    pts = [q for q in _head(labels) if q.role == "points"]
    assert len(pts) == 1 and pts[0].params["count"] == 12


def test_a_six_point_star_over_a_twelve_pleat_ring_is_not_split_by_the_pleats():
    """A 6-point star (1) over a ring of 12 pleats of its own color (deep kites to
    0.6 R) on a band of a second color: the star stays one layer of 6."""
    yy, xx, dx, dy, r, th = _grid()
    labels = np.full(r.shape, -1, np.int32)
    _tails(labels, yy, xx)
    labels[r <= 0.6 * R] = 5
    labels[r <= _star(th, 12, 15, 0.3 * R, 0.6 * R)] = 1
    labels[r <= _star(th, 6, 30, 0.5 * R, 0.95 * R)] = 1
    labels[r <= 0.3 * R] = 0
    labels[r <= 0.2 * R] = 2
    pts = [q for q in _head(labels) if q.role == "points" and q.color_group == 1]
    assert [q.params["count"] for q in pts] == [6]


def _pts(g, kind, n, ri, ro, rot, wr, z):
    return Part(f"points-{g}", "points", {"cx": 300.0, "cy": 200.0, "kind": kind, "count": n, "r_inner": ri,
                                          "r_outer": ro, "rotation_deg": rot, "width_ratio": wr}, g, z, 0.8)


def _painted(parts_bottom_up, fr, shape=(1200, 900)):
    """Labels painted from the parts' drawn shapes, bottom first (canvas -> px by fr)."""
    from shapely import contains_xy
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(float)
    x = 300.0 + (xx - fr.axis_x) / fr.scale
    y = 25.0 + (yy - fr.top) / fr.scale
    labels = np.full(shape, -1, np.int32)
    for q in parts_bottom_up:
        labels[contains_xy(H._part_geom(q), x, y)] = q.color_group
    return labels


def test_the_layer_the_photo_shows_on_top_is_stacked_on_top():
    """The H5 prior puts loops over a star; the labels show the star over the loops, so
    the star goes over them. Only the z values change: the two keep their set of z."""
    fr = Frame(axis_x=CX, top=float(CY - R), scale=2.0)
    cy = 25.0 + R / 2
    prior = H.points_z([("points", "bow", 140.0), ("points", "star", 120.0)])
    assert prior[0] > prior[1]  # loops over the star
    loops = _pts(1, "bow", 6, 40.0, 140.0, 0.0, 0.5, prior[0])
    star = _pts(4, "star", 6, 50.0, 120.0, 30.0, 0.5, prior[1])
    for q in (loops, star):
        q.params["cy"] = cy
    labels = _painted([loops, star], fr)
    H.photo_stacking([loops, star], labels, _photo(labels), fr, set())
    assert star.z > loops.z and sorted((star.z, loops.z)) == sorted(prior)
    # painted the other way round, the prior stands
    loops.z, star.z = prior
    labels = _painted([star, loops], fr)
    H.photo_stacking([loops, star], labels, _photo(labels), fr, set())
    assert (loops.z, star.z) == tuple(prior)


def test_loops_the_photo_shows_on_top_stay_on_top():
    parts = _head(star_over_loops(strips=False, star_on_top=False))
    star = next(q for q in parts if q.role == "points" and q.color_group == 4)
    loops = next(q for q in parts if q.role == "points" and q.color_group == 1)
    assert loops.z > star.z


def test_photo_stacking_keeps_the_prior_without_overlap():
    yy, xx, dx, dy, r, th = _grid()
    labels = np.full(r.shape, -1, np.int32)
    fr = Frame(axis_x=CX, top=float(CY - R), scale=2.0)
    a = _pts(1, "bow", 6, 60.0, 140.0, 0.0, 0.2, 21)
    b = _pts(4, "bow", 6, 60.0, 140.0, 30.0, 0.2, 20)
    H.photo_stacking([a, b], labels, _photo(labels), fr, set())
    assert (a.z, b.z) == (21, 20)


def test_photo_stacking_keeps_the_prior_when_the_labels_are_split():
    """Overlap present but its labels are half one color, half the other: the prior stands."""
    yy, xx, dx, dy, r, th = _grid()
    fr = Frame(axis_x=CX, top=float(CY - R), scale=2.0)
    labels = np.where(xx < CX, 1, 4).astype(np.int32)
    a = _pts(1, "bow", 6, 30.0, 140.0, 0.0, 0.5, 21)
    b = _pts(4, "bow", 6, 30.0, 140.0, 20.0, 0.5, 20)
    H.photo_stacking([a, b], labels, _photo(labels), fr, set())
    assert (a.z, b.z) == (21, 20)
    labels[:] = 4  # every overlap sample is b's color: b goes on top
    H.photo_stacking([a, b], labels, _photo(labels), fr, set())
    assert b.z > a.z


def test_a_loop_block_is_as_wide_as_it_shows_at_mid_radius():
    """Loops (1) plus wedges of their own color between their bases: the mean coverage
    reads the loops wider than they are; the drawn width is the block's own at mid radius."""
    from test_points_layers import loops_photo
    p, labels = loops_photo(profile=[1.0] * 11)
    yy, xx = np.mgrid[:labels.shape[0], :labels.shape[1]].astype(float)
    dx, dy = xx - 450.0, yy - 330.0
    r = np.hypot(dx, dy)
    wedge = _blocks(dx, dy, 6, 30, 0.6 * R, 0.8 * R, 0.03 * R) & (labels != 3)
    labels[wedge] = 1
    loops = [q for q in detect_parts(p, labels) if q.role == "points" and q.color_group == 1]
    assert len(loops) == 1 and loops[0].params["kind"] == "bow"
    a = loops[0].params
    width = a["width_ratio"] * (a["r_outer"] - a["r_inner"])
    assert abs(width - 0.4 * R / 2) < 3  # the block is 0.4 R (60 units) wide


def _fake_trims(host_label, victim):
    """An explain_trims stand-in: every points layer `victim(layer)` picks is replaced by a
    trim copy of the first layer of `host_label` (meta: the host's, trim=True)."""
    def run(layers, meta, *args):
        host = next(k for k, lay in enumerate(layers) if lay[1] == host_label and lay[0] == "points")
        out, out_meta, trims = [], [], []
        for lay, mm in zip(layers, meta):
            if lay[0] == "points" and victim(lay):
                trims.append((("points", lay[1], lay[2], dict(layers[host][3], edge_width=1.0), 0.0, layers[host][5]),
                              dict(meta[host], trim=True)))
                continue
            out.append(lay)
            out_meta.append(mm)
        for t, tm in trims:
            out.append(t)
            out_meta.append(tm)
        return out, out_meta
    return run


def _forced_split(whole, m):
    inner = ("points", whole[1], whole[2], dict(whole[3], r_outer=80.0), 1.0, 80)
    outer = ("points", whole[1], 0.5, dict(whole[3], count=12, kind="bow"), 0.0, 100)

    def split(layers, meta, *args):
        out, out_meta = [], []
        for lay, mm in zip(layers, meta):
            if lay is whole:
                out += [inner, outer]
                out_meta += [dict(mm, rows=(10, 80), split=0, whole=(lay, mm)),
                             dict(mm, rows=(80, 100), eats=None, split=0, outer=True)]
            else:
                out.append(lay)
                out_meta.append(mm)
        return out, out_meta
    return split


def _undo_case(monkeypatch, whole, others, fake):
    m = dict(zone=(0, 120), rows=(10, 100), eats=None, inner=None)
    layers = [whole] + others
    meta = [dict(m) for _ in layers]
    monkeypatch.setattr(H, "_star_from_outline", lambda *a: None)
    monkeypatch.setattr(H, "explain_trims", fake)
    expected = fake(list(layers), list(meta))
    monkeypatch.setattr(H, "_split_by_radius", _forced_split(whole, m))
    got = H._split_and_explain(layers, meta, None, None, None, None, 300.0, 200.0)
    return got, expected


def test_an_undone_split_leaves_the_layers_as_they_were_with_a_trim_hosting_inner_part(monkeypatch):
    """S56-17B: loops whose notched ends read 12 at the rim; the outer part explains no
    other layer, so the split is undone. Here the inner part also hosts another layer's
    trims: after the undo the trims are those the whole layer hosts, as without a split."""
    whole = ("points", 1, 0.8, {"kind": "bow", "count": 6, "r_outer": 100.0}, 1.0, 100)
    band = ("points", 3, 0.4, {"kind": "bow", "count": 6, "r_outer": 90.0}, 1.0, 90)
    got, expected = _undo_case(monkeypatch, whole, [band], _fake_trims(1, lambda lay: lay[1] == 3))
    assert got == expected
    assert not any(mm.get("outer") for mm in got[1])


def test_an_undone_split_leaves_the_layers_as_they_were_when_its_inner_part_is_a_trim(monkeypatch):
    """The inner part is read under EXPLAIN_MAX_CONF and replaced by another layer's trims;
    the outer part stays its own layer, so the split is undone and the whole layer goes
    through the trims rule again as it would without the split."""
    whole = ("points", 1, 0.5, {"kind": "star", "count": 6, "r_outer": 100.0}, 1.0, 100)
    other = ("points", 2, 0.9, {"kind": "bow", "count": 6, "r_outer": 110.0}, 1.0, 110)
    fake = _fake_trims(2, lambda lay: lay[1] == 1 and lay[2] < H_EXPLAIN and lay[3]["count"] == 6)
    got, expected = _undo_case(monkeypatch, whole, [other], fake)
    assert got == expected
    assert [lay[3].get("count") for lay in got[0]] == [6, 6]


def test_a_split_whose_outer_part_became_trims_stands(monkeypatch):
    whole = ("points", 1, 0.8, {"kind": "star", "count": 6, "r_outer": 100.0}, 1.0, 100)
    other = ("points", 2, 0.9, {"kind": "bow", "count": 6, "r_outer": 110.0}, 1.0, 110)
    fake = _fake_trims(2, lambda lay: lay[1] == 1 and lay[3]["count"] == 12)
    got, _ = _undo_case(monkeypatch, whole, [other], fake)
    assert [(lay[1], lay[3].get("r_outer"), "edge_width" in lay[3]) for lay in got[0]] == [
        (1, 80.0, False), (2, 110.0, False), (1, 110.0, True)]


def test_of_two_nested_stars_the_photo_cannot_order_the_smaller_is_on_top():
    """Two concentric 6-point stars of different colors on the same axes, the overlap's
    labels split half and half: the star with the shorter tips lies on top; only the z
    values change."""
    yy, xx, dx, dy, r, th = _grid()
    fr = Frame(axis_x=CX, top=float(CY - R), scale=2.0)
    labels = np.where(xx < CX, 1, 4).astype(np.int32)
    big = _pts(1, "star", 6, 80.0, 130.0, 30.0, 1.0, 21)
    small = _pts(4, "star", 6, 60.0, 110.0, 31.0, 1.0, 20)
    H.photo_stacking([big, small], labels, _photo(labels), fr, set())
    assert small.z > big.z
    assert (big.params["r_inner"], big.params["r_outer"]) == (80.0, 130.0)
    # when the photo decides, it wins: every overlap sample is the big star's color
    big.z, small.z = 21, 20
    labels[:] = 1
    H.photo_stacking([big, small], labels, _photo(labels), fr, set())
    assert big.z > small.z


def test_the_cover_step_reads_the_order_the_photo_gives():
    """12 petals (1, prior z 22 over a 12-lobe layer 4 at z 21) whose bases the photo shows
    under layer 4's lobes: the photo puts layer 4 on top, and only then are the petals
    read as starting under it, at the ring."""
    fr = Frame(axis_x=CX, top=float(CY - R), scale=2.0)
    cy = 25.0 + R / 2
    ring = Part("ring-0", "ring", {"cx": 300.0, "cy": cy, "r_inner": 30.0, "r_outer": 50.0}, 0, 30, 0.9)
    petals = _pts(1, "petal", 12, 70.0, 120.0, 0.0, 0.5, 22)
    cover = _pts(4, "bow", 12, 50.0, 125.0, 0.0, 0.25, 21)
    for q in (petals, cover):
        q.params["cy"] = cy
    labels = _painted([petals, cover, ring], fr)
    radii = np.arange(0.5, 150, 1.0) * fr.scale
    pol = polar_sample(labels, CX, fr.py(cy), radii, H.RING_THETA, fill=-1)
    theta = np.degrees(np.linspace(0, 2 * np.pi, H.RING_THETA, endpoint=False))
    win = np.abs(theta - 90.0) > H.TAIL_SECTOR_DEG
    parts = [ring, petals, cover]
    H._finish_points(parts, labels, _photo(labels), fr, pol, radii, win, -2, set())
    assert cover.z > petals.z
    assert petals.params["r_inner"] == 50.0
