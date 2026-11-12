"""H13: a head standing on a star (S53-19: a dark-blue 8-point star under the rings, glitter
triangles on four of its arms) gets a star-shaped base in the color of its tips, drawn
below every other head part. Synthetic labels only."""
import numpy as np

from rosette.vision import parts_head as H
from rosette.vision.parts import detect_parts
from rosette.vision.prepare import Prepared

R = 300.0
CX, CY = 450.0, 330.0
SCALE = 2.0  # px per canvas unit: R is 150 units


def _grid(h=1200, w=900):
    yy, xx = np.mgrid[:h, :w].astype(float)
    dx, dy = xx - CX, yy - CY
    return yy, xx, dx, dy, np.hypot(dx, dy), np.arctan2(dy, dx)


def _photo(labels):
    return Prepared(rgb=np.zeros(labels.shape + (3,), np.uint8), mask=labels >= 0, axis_x=CX, head=(CX, CY, R),
                    tails_top_y=CY + 0.5 * R, bottom_y=float(labels.shape[0] - 51), scale=SCALE)


def _star(th, n, rot, r_valley, r_tip):
    """Radius of an n-point star's outline (straight sides) at angle th."""
    a = np.pi / n
    phi = np.abs(((np.degrees(th) - rot) % (360 / n)) - 180 / n)
    phi = np.radians(180 / n - phi)  # angle from the nearest tip axis
    return r_tip * r_valley * np.sin(a) / (r_valley * np.sin(a - phi) + r_tip * np.sin(phi))


def _arms(dx, dy, n, rot, r0, r1, half_at_r0):
    """Triangles on the axes rot + k 360/n narrowing from half_at_r0 at r0 to a point at r1."""
    out = np.zeros(dx.shape, bool)
    for k in range(n):
        a = np.radians(rot + k * 360 / n)
        ax = dx * np.cos(a) + dy * np.sin(a)
        lat = -dx * np.sin(a) + dy * np.cos(a)
        out |= (ax >= r0) & (ax <= r1) & (np.abs(lat) <= half_at_r0 * (r1 - ax) / (r1 - r0))
    return out


def _head(labels):
    return [q for q in detect_parts(_photo(labels), labels) if q.role != "tail_stripe"]


def star_head(base="star", triangles=True, tips_two_tone=False):  # noqa: C901
    """Tails (3); a base (1): an 8-point star (tips 0.95 R on 0 + 45 k, valleys 0.55 R) or a
    disc to 0.95 R; with `triangles`, glitter triangles (4) filling the four axis arms from
    0.6 R to 0.88 R (the arms' tips stay blue); a white ring (0) 0.3-0.65 R over the
    valleys and a center (2) to 0.3 R."""
    yy, xx, dx, dy, r, th = _grid()
    labels = np.full(r.shape, -1, np.int32)
    labels[(yy > CY + 0.5 * R) & (yy < 1150) & (np.abs(xx - CX) < 110)] = 3
    if base == "star":
        body = r <= _star(th, 8, 0, 0.55 * R, 0.95 * R)
    elif base == "round":  # rounded lobes on 22.5 + 45 k (none cut by the tails): not a star's sides
        body = r <= 0.55 * R + 0.4 * R * np.cos(np.radians((np.degrees(th) % 45) - 22.5) * 4) ** 2
    else:
        body = r <= 0.95 * R
    labels[body] = 1
    if tips_two_tone:  # the diagonal arms in another color: two tip colors
        labels[body & _arms(dx, dy, 4, 45, 0.5 * R, 0.95 * R, 0.4 * R)] = 4
    if triangles:
        tri = _arms(dx, dy, 4, 0, 0.6 * R, 0.88 * R, 0.17 * R) & body
        labels[tri] = 4
    labels[r <= 0.65 * R] = 0
    labels[r <= 0.3 * R] = 2
    return labels


def _base(parts):
    return [q for q in parts if q.id == "base"]


def test_a_star_under_the_rings_is_the_heads_base_in_its_tip_color():
    parts = _head(star_head())
    base = _base(parts)
    assert len(base) == 1
    b = base[0]
    assert b.role == "points" and b.color_group == 1
    p = b.params
    assert p["kind"] == "star" and p["count"] == 8
    assert min(p["rotation_deg"] % 45, 45 - p["rotation_deg"] % 45) < 3
    assert abs(p["r_outer"] - 0.95 * R / SCALE) < 5
    assert p["r_inner"] < 0.65 * R / SCALE + 3  # the valleys are under the ring
    assert abs(p["r_inner"] - 0.55 * R / SCALE) < 12  # the sides follow the arms
    assert all(b.z < q.z for q in parts if q is not b)
    # the four-count reading of the same color (the arms the triangles leave) gives way to the base
    assert not [q for q in parts if q.role == "points" and q.color_group == 1 and q is not b]
    tri = [q for q in parts if q.role == "points" and q.color_group == 4]
    assert len(tri) == 1 and tri[0].params["count"] == 4


def test_a_round_base_stays_the_head_disc():
    parts = _head(star_head(base="disc", triangles=False))
    assert not _base(parts)
    assert [q.color_group for q in parts if q.role == "head"] == [1]


def test_a_star_already_read_as_a_points_layer_gets_no_base():
    parts = _head(star_head(triangles=False))
    assert not _base(parts)
    blue = [q for q in parts if q.role == "points" and q.color_group == 1]
    assert len(blue) == 1 and blue[0].params["count"] == 8


def test_tips_of_two_colors_give_no_base():
    parts = _head(star_head(triangles=False, tips_two_tone=True))
    assert not _base(parts)


def test_outline_symmetry_reads_the_tips():
    th = np.linspace(0, 2 * np.pi, H.RING_THETA, endpoint=False)
    x = _star(th, 8, 10, 100, 190)
    n, sc, rot = H._outline_symmetry(x, np.ones(len(x), bool))
    assert n == 8 and sc > 0.9 and abs(rot - 10) < 1


def _call_base(parts, trims=(), labels=None):
    from rosette.vision.parts_common import Frame
    labels = star_head(triangles=False) if labels is None else labels
    p = _photo(labels)
    theta = np.degrees(np.linspace(0, 2 * np.pi, H.RING_THETA, endpoint=False))
    win = np.abs(theta - 90.0) > H.TAIL_SECTOR_DEG
    H._base_star(parts, p, labels, Frame.of(p), win, {id(q) for q in trims})
    return [q for q in parts if q.id == "base"]


def _layer(g, n, rot, ro, kind="star"):
    from rosette.model import Part
    return Part(f"points-{g}-{n}", "points", {"cx": 300.0, "cy": 200.0, "kind": kind, "count": n, "r_inner": 60.0,
                                               "r_outer": ro, "rotation_deg": rot, "width_ratio": 1.0}, g, 20, 0.9)


def test_the_base_rule_alone_adds_the_star_when_nothing_draws_the_tips():
    assert len(_call_base([])) == 1


def test_tips_drawn_by_trims_or_by_another_colors_layer_get_no_base():
    # S80-22: a backing copy (a trim) of the tip color reaches the tips; S29-15: the tips
    # are the ends of loops of another color whose edge strips hold the tip color
    trim = _layer(1, 8, 0.0, 140.0)
    assert not _call_base([trim], trims=[trim])
    assert not _call_base([_layer(0, 8, 1.0, 140.0, kind="bow")])
    # another color's layer off the tip axes does not draw them
    assert len(_call_base([_layer(0, 8, 22.5, 140.0, kind="bow")])) == 1


DEPTH = 0.035


def pleated_bands():
    """S53-19's rings: a white center (2) to 0.25 R, then blue (0), glitter (1) and white (2)
    pleated bands whose edges are zigzags (the pleats: 24 teeth of DEPTH), on a head disc (4)
    to 0.95 R; tails (3)."""
    yy, xx, dx, dy, r, th = _grid()
    labels = np.full(r.shape, -1, np.int32)
    labels[(yy > CY + 0.5 * R) & (yy < 1150) & (np.abs(xx - CX) < 110)] = 3

    def edge(r0, n, rot=0.0):
        return r0 + DEPTH * R * np.abs(((np.degrees(th) - rot) * n / 360.0) % 1.0 - 0.5) * 2

    labels[r <= 0.95 * R] = 4
    labels[r <= edge(0.66 * R, 24)] = 2
    labels[r <= edge(0.48 * R, 24, 7)] = 1
    labels[r <= edge(0.32 * R, 24)] = 0
    labels[r <= edge(0.22 * R, 24, 4)] = 2
    return labels


def test_nested_pleated_bands_are_concentric_rings_that_do_not_overlap():
    rings = sorted((q for q in _head(pleated_bands()) if q.role == "ring"), key=lambda q: q.params["r_outer"])
    assert [q.color_group for q in rings] == [0, 1, 2]
    for a, b in zip(rings, rings[1:]):  # only a pinked edge's teeth may lie under the next ring
        assert b.params["r_inner"] >= a.params["r_outer"] - 7.0
        assert b.z < a.z  # the inner ring on top
    unit = R / SCALE
    for q, (lo, hi) in zip(rings, [(0.22, 0.36), (0.32, 0.52), (0.48, 0.70)]):
        assert lo * unit - 8 <= q.params["r_inner"] and q.params["r_outer"] <= hi * unit + 4, q.params


def test_ring_overlap_resolution_is_what_separates_the_bands(monkeypatch):
    monkeypatch.setattr(H, "_rings_by_dominance", lambda *a, **k: None)
    rings = sorted((q for q in _head(pleated_bands()) if q.role == "ring"), key=lambda q: q.params["r_outer"])
    white = rings[-1]  # without it the outer white ring runs in to the center (S53-19)
    assert white.color_group == 2 and white.params["r_inner"] < 0.25 * R / SCALE


class _Fr:
    scale = 1.0

    @staticmethod
    def d(v):
        return float(v)


class _Center:
    i1 = 10


def _profile(runs, n=120):
    """dom / domc of a radial profile: [(i0, i1, label)] held runs, every other row unheld."""
    dom, domc = np.zeros(n, int), np.full(n, 0.3)
    for a, b, g in runs:
        dom[a:b], domc[a:b] = g, 0.9
    return dom, domc


def _resolve(found, runs, center=True):
    dom, domc = _profile(runs)
    H._rings_by_dominance(found, dom, domc, _Fr, np.arange(len(dom)) + 0.5, _Center if center else None)
    return found


def test_rings_meet_on_an_odd_gap_and_the_outer_one_reaches_under_the_teeth():
    teeth = int(np.ceil(H.TOOTH / H.RADIAL_STEP_UNITS))
    blue, glitter = [50, 10, 50, 0, 0.5, 18], [50, 20, 50, 1, 0.5, 18]
    _resolve([blue, glitter], [(0, 10, 2), (10, 28, 0), (29, 50, 1)])  # one unheld row, 28
    assert blue[2] == 29  # the boundary row goes to one side
    assert glitter[1] == blue[2] - teeth


def test_a_ring_whose_color_holds_no_band_under_it_stays_as_measured():
    white, blue = [84, 35, 84, 2, 0.5, 24], [60, 35, 60, 0, 0.5, 18]
    _resolve([white, blue], [(0, 10, 3), (21, 33, 2), (36, 60, 0)])
    assert (white[1], white[2]) == (35, 84)


def test_identical_rings_of_one_color_are_kept_once():
    found = [[50, 12, 50, 0, 0.5, 18], [50, 15, 48, 0, 0.5, 18], [70, 20, 70, 1, 0.5, 18]]
    _resolve(found, [(0, 10, 2), (10, 40, 0), (40, 70, 1)])
    assert sorted((r[1], r[2], r[3]) for r in found if r[3] == 0) == [(10, 40, 0)]


def test_ring_extension_is_bounded_on_both_sides():
    # no center band: the unheld rows inside the innermost ring and outside the outermost are
    # not taken; the gap between two rings is split
    blue, glitter = [60, 5, 60, 0, 0.5, 0], [70, 30, 70, 1, 0.5, 0]
    _resolve([blue, glitter], [(20, 40, 0), (44, 60, 1)], center=False)
    assert blue[1] == 20 and glitter[2] == 60
    assert blue[2] == glitter[1] == 42


def test_a_base_whose_outline_does_not_fit_a_star_is_flagged():
    from rosette.assign import FLAG_BELOW
    sharp = _call_base([])
    assert len(sharp) == 1 and sharp[0].confidence >= FLAG_BELOW
    rounded = _call_base([], labels=star_head(base="round", triangles=False))
    assert len(rounded) == 1 and rounded[0].confidence < FLAG_BELOW


def backed_star():
    """S69-19: a navy (0) 6-point star on top (tips 0.6 R, valleys 0.35 R) over a white (1)
    6-point star (tips 0.88 R, valleys 0.5 R) whose sides show a thin navy border: a navy
    backing star behind it (tips 0.95 R, sides about 4 units outside the white ones), on the
    same axes 30 + 60 k; a ring (2) to 0.3 R and a center (4) to 0.2 R; tails (3)."""
    yy, xx, dx, dy, r, th = _grid()
    labels = np.full(r.shape, -1, np.int32)
    labels[(yy > CY + 0.5 * R) & (yy < 1150) & (np.abs(xx - CX) < 110)] = 3
    labels[r <= _star(th, 6, 30, 0.53 * R, 0.95 * R)] = 0
    labels[r <= _star(th, 6, 30, 0.5 * R, 0.88 * R)] = 1
    labels[r <= _star(th, 6, 30, 0.35 * R, 0.6 * R)] = 0
    labels[r <= 0.3 * R] = 2
    labels[r <= 0.2 * R] = 4
    return labels


def test_a_thin_border_along_a_stars_sides_is_its_backing_not_part_of_the_front_star():
    parts = _head(backed_star())
    navy = sorted((q for q in parts if q.role == "points" and q.color_group == 0), key=lambda q: q.params["r_outer"])
    white = [q for q in parts if q.role == "points" and q.color_group == 1]
    assert len(white) == 1 and len(navy) == 2
    front, back = navy
    unit = R / SCALE
    assert front.params["r_outer"] <= 0.6 * unit + 5  # the front star ends at its own tips
    assert back.params["count"] == 6 and back.params["r_outer"] >= 0.92 * unit
    assert abs(back.params["rotation_deg"] - white[0].params["rotation_deg"]) < 1
    assert back.z < white[0].z < front.z  # the backing behind the white star, the front star on it
