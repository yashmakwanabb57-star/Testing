from types import SimpleNamespace

import numpy as np

from rosette.model import Part
from rosette.pipeline import BADGE_PLEATS, _pleats_round_badge


def _photo(mixed: bool):
    """A 400x400 photo, canvas 1 px per unit, center at canvas (300, 200): a wine badge (1) of
    radius 50, then a band to 60 that is pink (2) and wine shade in 10-degree sectors, or plain
    pink, then the pink star body."""
    h = w = 400
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = 200.0, 200.0
    r = np.hypot(xx - cx, yy - cy)
    ang = (np.degrees(np.arctan2(yy - cy, xx - cx)) % 360) // 10
    labels = np.full((h, w), 2, dtype=int)
    labels[r <= 50] = 1
    band = (r > 50) & (r <= 60)
    if mixed:
        labels[band & (ang % 2 == 0)] = 1
    mask = np.zeros((h, w), bool)
    mask[25:, :] = True
    p = SimpleNamespace(axis_x=cx, scale=1.0, mask=mask, head=(cx, cy, 190.0))
    return p, labels


def _parts():
    geo = {"cx": 300.0, "cy": 200.0}
    return [
        Part("head", "head", dict(geo, r=190.0), 0, 0, 0.9),
        Part("points-1", "points", dict(geo, kind="star", count=6, r_inner=82.0, r_outer=130.0, rotation_deg=30.0), 1, 22, 0.9),
        Part("points-2", "points", dict(geo, kind="star", count=6, r_inner=60.0, r_outer=110.0, rotation_deg=30.0), 2, 23, 0.9),
        Part("center", "center", dict(geo, r=50.0), 1, 50, 0.9),
    ]


def test_r54_pleats_round_a_badge_become_a_pinked_ring():
    p, labels = _photo(mixed=True)
    parts = _parts()
    assert _pleats_round_badge(parts, p, labels)
    ring = next(q for q in parts if q.role == "ring")
    assert ring.color_group == 2 and ring.params["pinked"] == 1
    assert ring.params["r_inner"] == 50.0
    assert ring.params["r_outer"] == round(min(BADGE_PLEATS * 50.0, 82.0), 3)
    assert 23 < ring.z < 50


def test_r54_a_plain_star_round_the_badge_stays():
    p, labels = _photo(mixed=False)
    parts = _parts()
    assert not _pleats_round_badge(parts, p, labels)
    assert not any(q.role == "ring" for q in parts)


def test_r42_long_star_under_its_body_ring_goes_to_the_bottom_and_is_kept():
    # S33-15, the owner's check: the violet star (its body the pleated ring) lies under both
    # other stars; the app's classifier read its tips as kites
    from rosette.pipeline import _long_star_under
    from rosette.vision.layer_classify import KEEP_PARAM
    geo = {"cx": 300.0, "cy": 200.0}
    long_ = Part("points-0", "points", dict(geo, kind="star", count=6, r_inner=72.0, r_outer=150.0, rotation_deg=30.0), 0, 21, 0.9)
    blue = Part("points-2", "points", dict(geo, kind="star", count=6, r_inner=81.0, r_outer=132.0, rotation_deg=0.0), 1, 20, 0.9)
    edge = Part("points-1", "points", dict(geo, kind="star", count=6, r_inner=81.0, r_outer=113.0, rotation_deg=0.0), 2, 22, 0.9)
    parts = [long_, edge, blue]
    _long_star_under(parts, long_)
    assert long_.z < blue.z < edge.z and long_.params[KEEP_PARAM] == 1
    z = long_.z
    _long_star_under(parts, long_)  # already at the bottom: stays
    assert long_.z == z


def test_r55_points_between_an_edged_star_go_under_the_star():
    # s34-15, the owner's check: bottom to top the lavender edge star, the black points, the white star
    from rosette.pipeline import _points_under_edged_star
    geo = {"cx": 300.0, "cy": 200.0}
    star = Part("points-0", "points", dict(geo, kind="star", count=6, r_inner=92.5, r_outer=135.5, rotation_deg=32.6), 1, 21, 0.9)
    pts = Part("points-1", "points", dict(geo, kind="diamond", count=6, r_inner=41.5, r_outer=139.5, rotation_deg=2.2), 2, 22, 0.9)
    edge = Part("points-2", "points", dict(geo, kind="star", count=6, r_inner=92.5, r_outer=158.5, rotation_deg=32.6), 0, 20, 0.9)
    parts = [star, pts, edge]
    assert _points_under_edged_star(parts)
    assert edge.z < pts.z < star.z
    assert star.params["r_inner"] == round(92.5 * 135.5 / 158.5, 3)  # the edge star scaled down
    assert not _points_under_edged_star(parts)  # already in order: no change


def test_r55_leaves_loops_and_stars_between():
    from rosette.pipeline import _points_under_edged_star
    geo = {"cx": 300.0, "cy": 200.0}
    for kind in ("bow", "star", "petal"):
        star = Part("points-0", "points", dict(geo, kind="star", count=6, r_inner=92.5, r_outer=135.5, rotation_deg=32.6), 1, 21, 0.9)
        mid = Part("points-1", "points", dict(geo, kind=kind, count=6, r_inner=41.5, r_outer=139.5, rotation_deg=2.2), 2, 22, 0.9)
        edge = Part("points-2", "points", dict(geo, kind="star", count=6, r_inner=92.5, r_outer=158.5, rotation_deg=32.6), 0, 20, 0.9)
        assert not _points_under_edged_star([star, mid, edge])


def test_r55_needs_the_edge_star():
    from rosette.pipeline import _points_under_edged_star
    geo = {"cx": 300.0, "cy": 200.0}
    star = Part("points-0", "points", dict(geo, kind="star", count=6, r_inner=92.5, r_outer=135.5, rotation_deg=32.6), 1, 21, 0.9)
    pts = Part("points-1", "points", dict(geo, kind="diamond", count=6, r_inner=41.5, r_outer=139.5, rotation_deg=2.2), 2, 22, 0.9)
    assert not _points_under_edged_star([star, pts])
    assert pts.z == 22
