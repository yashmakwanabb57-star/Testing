"""H12: a points layer whose symmetry changes with radius is split (a 6-point star plus
the loops' edge strips at the rim, S29-15); points layers are stacked as the photo shows
them where they overlap. Synthetic labels only."""
import numpy as np

from rosette.model import Part
from rosette.vision import parts_head as H
from rosette.vision.parts import detect_parts
from rosette.vision.parts_common import Frame, polar_sample
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


def test_a_six_point_star_with_twelve_strips_at_the_rim_is_split():
    parts = _head(star_over_loops())
    pink = [q for q in parts if q.role == "points" and q.color_group == 4]
    loops = [q for q in parts if q.role == "points" and q.color_group == 1]
    assert len(loops) == 1 and loops[0].params["count"] == 6
    stars = [q for q in pink if "edge_width" not in q.params]
    strips = [q for q in pink if "edge_width" in q.params]
    assert len(stars) == 1 and stars[0].params["count"] == 6 and stars[0].params["kind"] == "star"
    assert abs((stars[0].params["rotation_deg"] - 30 + 30) % 60 - 30) < 3
    assert stars[0].params["r_outer"] <= 0.9 * R / 2
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


def test_a_split_whose_rim_lobes_flank_no_other_layer_is_undone():
    """S56-17B: loops whose notched ends read 12 at the rim. Their outer part explains no
    other layer's trims, so the layer is put back whole (and trims hosted by its inner
    part go with it); a split whose outer part became trims stands."""
    whole = ("points", 1, 0.8, {"kind": "star", "count": 6}, 1.0, 100)
    m = dict(zone=(0, 120), rows=(10, 100), eats=None, inner=None)
    inner = ("points", 1, 0.8, {"kind": "star", "count": 6}, 1.0, 80)
    outer = ("points", 1, 0.5, {"kind": "bow", "count": 12}, 0.0, 100)
    other = ("points", 2, 0.9, {"kind": "bow", "count": 6}, 1.0, 110)
    on_inner = ("points", 3, 0.5, {"kind": "star", "count": 6, "edge_width": 1.0}, 0.0, 80)
    layers = [other, inner, outer, on_inner]
    meta = [dict(m), dict(m, split=1, whole=(whole, m)), dict(m, split=1, outer=True),
            dict(m, split=1, whole=(whole, m), trim=True)]
    out, _ = H._keep_explained_splits(layers, meta)
    assert out == [other, whole]
    # the outer part was replaced by the other layer's strips: the split stands
    strips = ("points", 1, 0.5, {"kind": "bow", "count": 6, "edge_width": 2.0}, 0.0, 110)
    layers = [other, inner, strips]
    meta = [dict(m), dict(m, split=1, whole=(whole, m)), dict(m, trim=True)]
    out, _ = H._keep_explained_splits(layers, meta)
    assert out == layers
