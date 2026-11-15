import numpy as np

from rosette.model import Part
from rosette.vision.parts_head import _star_on_ring


class _Fr:
    scale = 1.0

    def d(self, v):
        return v


def _kite(g, rot, z):
    return Part(f"points-{g}", "points", {"cx": 300.0, "cy": 160.0, "kind": "diamond", "count": 6, "r_inner": 30.0,
                                          "r_outer": 110.0, "rotation_deg": rot, "width_ratio": 0.8}, g, z, 0.6)


def _pol(body_to=60):
    """Black (2) full to body_to, then 6 black arms on light orange (1) out to 110."""
    radii = np.arange(0.5, 130.0, 1.0)
    th = np.arange(720) * 0.5
    pol = np.full((len(radii), 720), 0, np.int32)
    for i, r in enumerate(radii):
        if r < body_to:
            pol[i] = 2
        elif r < 110:
            half = 30.0 * (110 - r) / (110 - body_to)  # arms narrowing to the tips
            d = np.abs((th % 60) - 30.0)
            pol[i] = np.where(30.0 - d <= half, 2, 1)
    return pol, radii


def test_interleaved_kites_on_a_full_body_become_a_star_on_a_ring():
    pol, radii = _pol()
    a, b = _kite(2, 0.0, 21), _kite(1, 30.0, 20)
    parts = [a, b]
    _star_on_ring(parts, pol, radii, np.ones(720, bool), _Fr())
    assert a.params["kind"] == "star" and 46 <= a.params["r_inner"] <= 55 and a.params.get("leaf") == 1  # 0.8 of where the arms part (R30)
    assert b.role == "ring" and b.params["r_inner"] == a.params["r_inner"] and b.z < a.z


def test_kites_without_a_full_body_are_left_alone():
    pol, radii = _pol(body_to=0)
    pol[:30] = 1  # inside the pair: the other color, no star body
    a, b = _kite(2, 0.0, 21), _kite(1, 30.0, 20)
    _star_on_ring([a, b], pol, radii, np.ones(720, bool), _Fr())
    assert a.params["kind"] == "diamond" and b.role == "points"


def test_leaf_star_is_one_valid_polygon_with_bulging_arms():
    from shapely import Point, Polygon
    from rosette.geometry.points import points_ring
    straight = points_ring(300, 160, "star", 6, 50.0, 110.0, 0.0, 1.0)[0]
    leaf = points_ring(300, 160, "star", 6, 50.0, 110.0, 0.0, 1.0, leaf=True)[0]
    pl = [tuple(float(v) for v in p.split(",")) for p in leaf.attrs["points"].split()]
    ps = [tuple(float(v) for v in p.split(",")) for p in straight.attrs["points"].split()]
    gl, gs = Polygon(pl), Polygon(ps)
    assert gl.is_valid and gl.area > gs.area  # the arms bow outward
    assert gl.contains(Point(300, 160)) and abs(gl.bounds[2] - 410.0) < 0.01  # body and a tip at r_outer
