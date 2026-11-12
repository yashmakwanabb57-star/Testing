"""ML-1 Task 2: the learned layer classifier (synthetic data) and the saved model."""
import json
from pathlib import Path

import numpy as np
import pytest

from rosette import layer_model as lm
from rosette.vision.layer_features import COUNTS, feature_names, layer_features
from tools import train_layers as tl

T = 720


def _features(kind: str, n: int, rng) -> dict:
    """Feature dict of a synthetic layer: 'loop' wide lobes, 'kite' narrow lobes, 'ring' a band."""
    radii = np.arange(0.5, 110, 1.0)
    th = np.degrees(np.linspace(0, 2 * np.pi, T, endpoint=False))
    pol = np.zeros((len(radii), T), np.int32)
    r0, r1 = 40 + rng.uniform(-3, 3), 90 + rng.uniform(-3, 3)
    rows = (radii >= r0) & (radii <= r1)
    if kind == "ring":
        pol[rows] = 1
    else:
        share = 0.6 if kind == "loop" else 0.2
        rot = rng.uniform(0, 360)
        on = np.mod(th - rot, 360.0 / n) < share * 360.0 / n
        pol[np.ix_(rows, on)] = 1
    win = np.abs(th - 90.0) > 35.0
    f = layer_features(pol, radii, win, [1], r0, r1, 100, 1.0, category=1)
    f.update(n_over=0.0, d_out=0.0, inside_other=0.0, same_n_over=0.0)
    return f


def _data(m=40, seed=0):
    rng = np.random.default_rng(seed)
    rows, y, c = [], [], []
    for i in range(m):
        kind = ("loop", "kite", "ring")[i % 3]
        n = int(rng.choice([4, 5, 6, 8, 10, 12])) if kind != "ring" else 0
        rows.append(_features(kind, n, rng))
        y.append(kind)
        c.append(n)
    return rows, y, c


def test_model_learns_synthetic_kinds_and_counts():
    rows, y, c = _data()
    names = list(rows[0].keys())
    m = lm.train(rows, y, c, names)
    test, ty, tc = _data(12, seed=1)
    kinds = m.predict_kind(test)
    assert [k for k, _ in kinds] == ty
    assert all(0 <= p <= 1 for _, p in kinds)
    counts = m.predict_count([t for t, k in zip(test, ty) if k != "ring"])
    assert [n for n, _ in counts] == [n for n, k in zip(tc, ty) if k != "ring"]


def test_count_candidates_cover_every_count():
    rows, _, _ = _data(3)
    assert lm.count_matrix(rows[0]).shape[0] == len(COUNTS)


def test_gate_is_the_lowest_bar_with_the_precision():
    probs = [0.95] * 30 + [0.7] * 10 + [0.5] * 20
    right = [True] * 29 + [False] + [True] * 9 + [False] + [True] * 5 + [False] * 15
    bar, prec, cov = tl.gate(probs, right, precision=0.9, min_layers=20)
    assert bar == 0.7 and abs(prec - 38 / 40) < 1e-9 and abs(cov - 40 / 60) < 1e-9
    assert tl.gate([0.9] * 30, [False] * 30, 0.9, 20)[0] > 1.0  # never right: no bar


def test_rules_and_model_per_class():
    got = tl.per_class(["loop", "loop", "kite"], ["loop", "kite", "kite"])
    assert got["loop"] == (1, 2) and got["kite"] == (1, 1) and got["star"] == (0, 0)


DATA = Path(lm.__file__).resolve().parent / "data"


def test_saved_model_reproduces_its_stored_prediction():
    check = json.loads((DATA / "layer_model_check.json").read_text(encoding="utf-8"))
    m = lm.load(DATA / "layer_model.joblib")
    assert m is not None and m.train_hash == check["train_hash"]
    assert set(feature_names()) <= set(m.features)
    (k, p), (n, s) = m.predict_kind([check["features"]])[0], m.predict_count([check["features"]])[0]
    assert k == check["kind"] and p == pytest.approx(check["prob"])
    assert n == check["count"] and s == pytest.approx(check["count_score"])


def test_saved_model_was_trained_on_the_dataset():
    if tl.CONFIRMED.exists():
        pytest.skip("local confirmed layers: retrain with tools/train_layers.py")
    m = lm.load(DATA / "layer_model.joblib")
    assert m.train_hash == lm.data_hash()


def test_bars_are_measured_on_the_layers_they_apply_to():
    rows = [dict(rules="loop", label="loop", count="6"), dict(rules="center", label="center", count="0"),
            dict(rules="pinked_ring", label="ring", count="0"), dict(rules="edge", label="edge", count="6")]
    assert tl.kind_scope(rows) == [0, 2, 3]  # the center is never changed by the kind model
    assert tl.count_scope(rows) == [0]  # trims follow their host's count


def test_held_out_rows_leave_confirmed_downloads_out():
    rows = [dict(label="loop", source="ref"), dict(label="loop", source="confirmed"), dict(label="", source="ref")]
    assert tl.held_out_rows(rows) == [rows[0]]


def test_held_out_evaluation_groups_by_design(monkeypatch):
    seen = []
    monkeypatch.setattr(tl, "fit_kind", lambda x, y: seen.append(len(y)) or None)
    monkeypatch.setattr(tl, "fit_count", lambda f, c: None)

    class M:
        def __init__(self, **k):
            pass

        def predict_kind(self, rows):
            return [("loop", 1.0)] * len(rows)

        def predict_count(self, rows):
            return [(6, 1.0)] * len(rows)

    monkeypatch.setattr(tl, "LayerModel", M)
    rows = [dict(photo=p, design=d, rules="loop", role="points", rules_count="6", label="loop", count="6", a="0")
            for p, d in (("S36-15", "S36"), ("S36-15B", "S36"), ("S1-15", "S1"))]
    tl.loo(rows, ["a"])
    assert sorted(seen) == [1, 2]  # S36 held out with both its photos
