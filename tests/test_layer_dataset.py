"""ML-1 Task 1: layer features, the rules' classes and the reference matching (synthetic)."""
import numpy as np

from rosette.model import Part
from rosette.vision.layer_features import COUNTS, feature_names, layer_features
from tools import build_layer_dataset as bd

T = 720


def _win(sector: bool = True) -> np.ndarray:
    th = np.degrees(np.linspace(0, 2 * np.pi, T, endpoint=False))
    return np.abs(th - 90.0) > 35.0 if sector else np.ones(T, bool)


def _lobed(n: int, radii: np.ndarray, r0: float, r1: float, width_deg: float, label: int = 1) -> np.ndarray:
    """Polar labels: label on n lobes of constant angular width between r0 and r1, 0 elsewhere."""
    th = np.degrees(np.linspace(0, 2 * np.pi, T, endpoint=False))
    on = (np.mod(th, 360.0 / n) < width_deg)
    pol = np.zeros((len(radii), T), np.int32)
    rows = (radii >= r0) & (radii <= r1)
    pol[np.ix_(rows, on)] = label
    return pol


def test_lobed_layer_reads_its_symmetry_and_runs():
    radii = np.arange(0.5, 110, 1.0)
    for n in (5, 6, 8, 12):
        pol = _lobed(n, radii, 40, 90, 360.0 / n / 3)
        f = layer_features(pol, radii, _win(), [1], 40, 90, 100, 1.0, category=2)
        assert f[f"sym_{n}"] > 0.9
        assert f["sym_strong_n"] in (n, 2 * n) or f["sym_strong_n"] % n == 0
        assert abs(f["runs_med"] - n) <= 2  # the tails sector hides one or two
        assert abs(f["fill"] - 1 / 3) < 0.05
        assert f["cat_2"] == 1.0 and f["cat_1"] == 0.0
        assert f["full_rows"] == 0.0


def test_full_band_and_surroundings():
    radii = np.arange(0.5, 110, 1.0)
    pol = np.full((len(radii), T), 3, np.int32)
    pol[(radii >= 40) & (radii <= 60)] = 1
    f = layer_features(pol, radii, _win(), [1], 40, 60, 100, 1.0)
    assert f["full_rows"] == 1.0 and f["cov_4"] > 0.99
    assert f["enclosed"] == 1.0 and f["same_in"] == 0.0
    pol[radii < 40] = 1  # the same color inside: a core
    f = layer_features(pol, radii, _win(), [1], 40, 60, 100, 1.0)
    assert f["same_in"] == 1.0 and f["enclosed"] == 0.0


def test_merged_labels_count_as_one_color():
    radii = np.arange(0.5, 110, 1.0)
    pol = _lobed(6, radii, 40, 90, 20)
    pol[pol == 1] = 4
    f = layer_features(pol, radii, _win(), [1, 4], 40, 90, 100, 1.0)
    assert f["sym_6"] > 0.9


def test_feature_names_are_stable():
    names = feature_names()
    assert names[:3] == ["r_in", "r_out", "thick"]
    assert all(f"sym_{n}" in names for n in COUNTS)
    assert len(names) == len(set(names))


def _pts(pid, kind, count, g, rot=0.0, r_out=100.0, **extra):
    p = dict(cx=300, cy=200, kind=kind, count=count, r_inner=50, r_outer=r_out, rotation_deg=rot, width_ratio=0.5)
    p.update(extra)
    return Part(pid, "points", p, g, 20, 0.9)


def test_rules_class_of_parts():
    loops = _pts("points-0", "bow", 6, 1)
    edge = _pts("points-1", "bow", 6, 2, edge_width=3.0)
    kite = _pts("points-2", "diamond", 6, 3, r_out=90.0)
    back = _pts("points-3", "diamond", 6, 4, rot=2.0, r_out=110.0)
    base = _pts("base", "star", 8, 5)
    parts = [loops, edge, kite, back, base,
             Part("ring-0", "ring", dict(cx=300, cy=200, r_inner=20, r_outer=40, pinked=1, pleats=24), 1, 30, 1),
             Part("ring-1", "ring", dict(cx=300, cy=200, r_inner=40, r_outer=50), 1, 31, 1),
             Part("head", "head", dict(cx=300, cy=200, r=130, pinked=0), 1, 10, 1),
             Part("center", "center", dict(cx=300, cy=200, r=20), 1, 50, 1),
             Part("unknown-0", "unknown", dict(cx=300, cy=200, r=5), 1, 60, 0.2)]
    got = [bd.rules_class(q, parts) for q in parts]
    assert got == ["loop", "edge", "kite", "backing", "base_star", "pinked_ring", "ring", "disc", "center", "none"]
    assert bd.rules_count(kite, "kite") == 6 and bd.rules_count(parts[5], "pinked_ring") == 0


def _layer(cls, cat, r_in=0.2, r_out=0.8):
    return dict(cls=cls, category=cat, r_in=r_in, r_out=r_out, count=0)


def test_match_takes_the_reference_layer_seen_on_the_pixels():
    layers = [_layer("disc", 1), _layer("star", 2), _layer("loop", 3)]
    top = np.full((10, 10), -1)
    top[:, :5] = 1
    top[:, 5:8] = 2
    seen = np.zeros((10, 10), bool)
    seen[:, 3:9] = True  # 20 px star, 30 px loop
    L, share = bd.match(seen, top, layers, cat=None)
    assert L["cls"] == "loop" and abs(share - 0.5) < 1e-9
    # the same Color counts fully, another Color OTHER_COLOR of its share
    L, _ = bd.match(seen, top, layers, cat=2)
    assert L["cls"] == "star"
    seen[:] = False
    seen[0, 0] = True
    top[0, 0] = -1
    assert bd.match(seen, top, layers)[0] is None  # background only: no match


def test_trim_in_reference_by_band_and_color():
    layers = [_layer("edge", 2, 0.5, 0.9), _layer("backing", 3, 0.7, 1.0)]
    assert bd.trim_in_reference("edge", 2, (0.45, 0.95), layers)["cls"] == "edge"
    assert bd.trim_in_reference("edge", 3, (0.45, 0.95), layers) is None
    assert bd.trim_in_reference("backing", 3, (0.1, 0.3), layers) is None


def _polygon_mask(shape, pts):
    import cv2
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m.astype(bool)


def test_piece_shape_tells_stars_kites_and_pleated_annuli():
    sh, c, R = (500, 500), 250.0, 200.0
    head = (c, c, R)
    n = 8  # a star polygon: 8 lobes, no full circle
    a = np.radians(np.arange(2 * n) * 180.0 / n)
    rad = np.where(np.arange(2 * n) % 2 == 0, 180.0, 90.0)
    star = _polygon_mask(sh, np.c_[c + rad * np.cos(a), c + rad * np.sin(a)])
    lobes, pleated = bd.piece_shape(star, head, 1.0)
    assert lobes == n and not pleated
    kite = _polygon_mask(sh, np.array([[c + 60, c], [c + 120, c - 15], [c + 180, c], [c + 120, c + 15]]))
    assert bd.piece_shape(kite, head, 1.0) == (1, False)
    th = np.radians(np.arange(0, 360, 0.5))  # an annulus with a scalloped outer edge (36 scallops)
    outer = 150 + 2.5 * np.cos(36 * th)
    yy, xx = np.mgrid[:sh[0], :sh[1]]
    d, t = np.hypot(xx - c, yy - c), np.arctan2(yy - c, xx - c)
    ann = (d >= 110) & (d <= 150 + 2.5 * np.cos(36 * t))
    lobes, pleated = bd.piece_shape(ann, head, 1.0)
    assert pleated


def test_layer_order_counts_lobes_cut_in_pieces_and_hidden_behind_the_tails():
    sh, c = (500, 500), 250.0
    head = (c, c, 200.0)
    yy, xx = np.mgrid[:sh[0], :sh[1]]
    d, t = np.hypot(xx - c, yy - c), np.degrees(np.arctan2(yy - c, xx - c)) % 360
    band = (d >= 80) & (d <= 180)
    loops = band & (np.mod(t, 60) < 24) & ~(np.abs(np.mod(t, 60) - 12) < 1.5)  # 6 loops, each cut in two
    assert bd.layer_order(loops, head) == 6
    kites = band & (np.mod(t, 45) < 12) & ~((t > 80) & (t < 100))  # 8, the one under the tails left out
    assert bd.layer_order(kites, head) == 8
