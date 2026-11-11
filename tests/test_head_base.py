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


def star_head(base="star", triangles=True, tips_two_tone=False):
    """Tails (3); a base (1): an 8-point star (tips 0.95 R on 0 + 45 k, valleys 0.55 R) or a
    disc to 0.95 R; with `triangles`, glitter triangles (4) filling the four axis arms from
    0.6 R to 0.88 R (the arms' tips stay blue); a white ring (0) 0.3-0.65 R over the
    valleys and a center (2) to 0.3 R."""
    yy, xx, dx, dy, r, th = _grid()
    labels = np.full(r.shape, -1, np.int32)
    labels[(yy > CY + 0.5 * R) & (yy < 1150) & (np.abs(xx - CX) < 110)] = 3
    if base == "star":
        body = r <= _star(th, 8, 0, 0.55 * R, 0.95 * R)
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


def _call_base(parts, trims=()):
    from rosette.model import Part  # noqa: F401
    from rosette.vision.parts_common import Frame
    labels = star_head(triangles=False)
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
