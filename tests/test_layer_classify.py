"""ML-1 Task 3: the classifier at run time and the pleated-ring rule (synthetic)."""
import numpy as np

from rosette.model import Part
from rosette.vision import layer_classify as lc
from rosette.vision.parts_head import PLEAT_TEXTURE, pleat_texture
from rosette.vision.prepare import Prepared


def _prepared(pleats: bool) -> Prepared:
    h = w = 400
    c, r = 200.0, 150.0
    yy, xx = np.mgrid[:h, :w]
    d, t = np.hypot(xx - c, yy - c), np.arctan2(yy - c, xx - c)
    gray = np.full((h, w), 200.0)
    band = (d >= 80) & (d <= 120)
    if pleats:  # folds every 6 degrees: light and dark stripes across the band
        gray[band] = 170 + 40 * np.cos(60 * t[band])
    else:  # satin: one slow highlight
        gray[band] = 180 + 20 * np.cos(t[band])
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.uint8)
    return Prepared(rgb=rgb, mask=d <= r, axis_x=c, head=(c, c, r), tails_top_y=c + r, bottom_y=h - 1, scale=1.0), band


def test_pleat_texture_tells_pleated_from_smooth_bands():
    p, band = _prepared(True)
    assert pleat_texture(p, band, 80, 120) > PLEAT_TEXTURE
    p, band = _prepared(False)
    assert pleat_texture(p, band, 80, 120) < PLEAT_TEXTURE
    assert pleat_texture(p, np.zeros_like(band), 80, 120) == 0.0  # nothing of the colour shows


class _Stub:
    """A LayerModel stand-in with fixed answers."""
    def __init__(self, kinds=None, counts=None, gate=0.6):
        self.kinds, self.counts, self.gate = kinds or [], counts or [], gate

    def predict_kind(self, rows):
        return self.kinds[:len(rows)]

    def predict_count(self, rows):
        return self.counts[:len(rows)]


def _pts(pid, kind, n=6, g=1, rot=0.0, r_out=100.0, **extra):
    p = dict(cx=300, cy=200, kind=kind, count=n, r_inner=50, r_outer=r_out, rotation_deg=rot, width_ratio=0.6)
    p.update(extra)
    return Part(pid, "points", p, g, 20, 0.9)


def _rows(parts):
    return [dict(part=q, features={}, rules=lc.rules_class(q, parts), members=[q.color_group]) for q in parts]


def test_kinds_change_within_scope_above_the_bar_only():
    ring = Part("ring-0", "ring", dict(cx=300, cy=200, r_inner=20, r_outer=40), 1, 30, 1)
    loops = _pts("points-0", "bow", g=2)
    edge = _pts("points-1", "bow", g=3, edge_width=3.0)
    kite = _pts("points-2", "diamond", g=4, rot=30.0)
    center = Part("center", "center", dict(cx=300, cy=200, r=20), 5, 50, 1)
    parts = [ring, loops, edge, kite, center]
    stub = _Stub(kinds=[("pinked_ring", 0.9), ("kite", 0.9), ("loop", 0.95), ("star", 0.5), ("ring", 0.99)])
    changed, warnings = lc.apply_kinds(_rows(parts), stub, parts)
    assert ring.params["pinked"] == 1
    assert loops.params["kind"] == "diamond"
    assert "edge_width" not in edge.params and edge.params["kind"] == "bow"
    assert kite.params["kind"] == "diamond"  # below the bar: the rules stay, flagged
    assert changed == ["ring-0", "points-0", "points-1"]
    assert warnings == [lc.KIND_WARNING.format(pid="points-2")]
    assert "center" not in changed  # a role change (center -> ring) is never made nor flagged


def test_kinds_that_need_geometry_are_flagged_not_made():
    loops = _pts("points-0", "bow", g=1)
    kite = _pts("points-1", "diamond", g=2, rot=30.0)
    parts = [loops, kite]
    changed, warnings = lc.apply_kinds(_rows(parts), _Stub(kinds=[("edge", 0.99), ("backing", 0.99)]), parts)
    assert changed == [] and len(warnings) == 2  # no host for the backing, an edge needs one
    host = _pts("points-2", "petal", g=3, r_out=80.0)
    back = _pts("points-3", "diamond", g=4, r_out=110.0)
    parts = [host, back]
    changed, _ = lc.apply_kinds(_rows(parts), _Stub(kinds=[("petal", 0.99), ("backing", 0.99)]), parts)
    assert changed == ["points-3"] and back.params["kind"] == "petal"
    assert lc.rules_class(back, parts) == "backing"


def _unwrap(n, rot):
    radii = np.arange(0.5, 160, 1.0)
    th = np.degrees(np.linspace(0, 2 * np.pi, 720, endpoint=False))
    pol = np.zeros((len(radii), 720), np.int32)
    on = np.mod(th - rot + 8, 360.0 / n) < 16
    pol[np.ix_((radii >= 50) & (radii <= 100), on)] = 1
    return lc.HeadUnwrap((300, 200), (300, 200), 150, 150, 1.0, radii, pol, np.abs(th - 90) > 35)


def test_count_replaces_the_rules_and_trims_follow():
    loops = _pts("points-0", "bow", n=4, g=1)
    edge = _pts("points-1", "bow", n=4, g=2, edge_width=3.0)
    other = _pts("points-2", "diamond", n=5, g=3, rot=20.0)
    parts = [loops, edge, other]
    rows = _rows(parts)
    u = _unwrap(6, 10.0)
    changed, warnings = lc.apply_counts(rows, u, _Stub(counts=[(6, 0.9), (5, 0.8)]), parts)
    assert warnings == []
    assert changed == ["points-0", "points-1"]
    assert loops.params["count"] == 6 and edge.params["count"] == 6
    assert abs(loops.params["rotation_deg"] - 10.0) < 1.0 and edge.params["rotation_deg"] == loops.params["rotation_deg"]
    assert abs(loops.params["width_ratio"] - 0.4) < 1e-9  # 0.6 * 4 / 6
    assert other.params["count"] == 5 and other.params["rotation_deg"] == 20.0


def test_count_below_its_bar_is_flagged_not_made():
    loops = _pts("points-0", "bow", n=4, g=1)
    stub = _Stub(counts=[(6, 0.4)])
    stub.count_gate = 0.5
    changed, warnings = lc.apply_counts(_rows([loops]), _unwrap(6, 10.0), stub, [loops])
    assert changed == [] and loops.params["count"] == 4
    assert warnings == [lc.COUNT_WARNING.format(pid="points-0")]


def test_no_saved_model_changes_nothing(monkeypatch):
    monkeypatch.setattr(lc, "model_without", lambda key: None)
    assert lc.classify_head_layers(None, None, None, {}) == []
