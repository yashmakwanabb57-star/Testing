"""Learned head-layer classifier (ML-1): the kind and lobe count of a measured head layer.

Trained by `tools/train_layers.py` on `rosette/data/layers.csv` (built by
`tools/build_layer_dataset.py`: the 76 reference photos' detected head layers, labelled
from their reference SVGs) and saved with joblib to `rosette/data/layer_model.joblib`
together with the training set's sha256, the feature names and the confidence bar.
Runs locally on the features of `rosette.vision.layer_features`; no network.

- kind: a random forest over the layer features (class weights balance the rare kinds);
  `predict_kind` gives the class and its probability.
- count: every candidate n in 2..24 is scored by a second forest from features of that n
  (its symmetry score against the best, its harmonic, the runs seen on circles, whether a
  multiple or half of it scores as high) and the best-scoring n wins (`predict_count`):
  a ranking, so counts unseen in training are still reachable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rosette.vision.layer_features import COUNTS, HARMONICS

DATA = Path(__file__).resolve().parent / "data"
MODEL_PATH = DATA / "layer_model.joblib"
CLASSES = ("disc", "pinked_ring", "ring", "loop", "kite", "petal", "star", "base_star", "backing", "edge",
           "center", "none")
POINTS_CLASSES = ("loop", "kite", "petal", "star", "base_star", "backing", "edge")
# Forest sizes and seed (deterministic training).
KIND_TREES, COUNT_TREES, SEED = 400, 300, 0
# Layer features also given to the count scorer.
COUNT_CONTEXT = ("r_in", "r_out", "thick", "fill", "full_rows", "hf_share", "w_peak", "w_full", "runs_med")


ROLES = ("head", "ring", "points", "center", "unknown")


def rules_features(rules: str, role: str, rules_count: int) -> dict[str, float]:
    """What the rule-based head made of the layer, as features: its class and role one-hot
    and its count (the rules' reading is one input among the measurements)."""
    f = {f"rules_{c}": float(rules == c) for c in CLASSES}
    f.update({f"role_{r}": float(role == r) for r in ROLES})
    f["rules_count"] = float(rules_count)
    return f


def count_candidate_features(f: dict[str, float], n: int) -> list[float]:
    """Features of candidate count n for a layer's feature dict f."""
    sym = np.array([f[f"sym_{k}"] for k in COUNTS])
    s = f[f"sym_{n}"]
    harm = np.array([f[f"harm_{k}"] for k in range(1, HARMONICS + 1)])
    runs = np.array([f[f"runs_{k}"] for k in range(5)])
    return [
        float(n), s, s - sym.max(), float((sym > s).sum()),
        f[f"sym_{2 * n}"] if 2 * n <= COUNTS[-1] else -1.0,
        f[f"sym_{n // 2}"] if n % 2 == 0 and n // 2 >= 2 else -1.0,
        float(n == f["sym_strong_n"]),
        float(max([k for k in COUNTS if f[f"sym_{k}"] >= 0.4], default=0) == n),
        harm[n - 1] / max(harm.max(), 1e-9),
        float(np.abs(runs - n).min() / n), abs(f["runs_med"] - n) / n, abs(f["runs_med"] - 2 * n) / n,
        float(np.mean(np.abs(runs - n) <= 1)),
        float(f.get("rules_count", 0.0) == n),
    ] + [f[k] for k in COUNT_CONTEXT]


def count_matrix(f: dict[str, float]) -> np.ndarray:
    return np.array([count_candidate_features(f, n) for n in COUNTS])


@dataclass
class LayerModel:
    kind: object
    count: object
    features: list[str]
    train_hash: str = ""
    gate: float = 1.01
    info: dict = field(default_factory=dict)

    def _x(self, rows: list[dict]) -> np.ndarray:
        return np.array([[float(r[k]) for k in self.features] for r in rows])

    def predict_kind(self, rows: list[dict]) -> list[tuple[str, float]]:
        """[(class, probability)] per feature dict."""
        if not rows:
            return []
        pr = self.kind.predict_proba(self._x(rows))
        cl = list(self.kind.classes_)
        return [(str(cl[int(np.argmax(p))]), float(p.max())) for p in pr]

    def predict_count(self, rows: list[dict]) -> list[tuple[int, float]]:
        """[(count, score)] per feature dict: the candidate the count scorer rates highest."""
        out = []
        for r in rows:
            p = self.count.predict_proba(count_matrix(r))[:, list(self.count.classes_).index(1)]
            k = int(np.argmax(p))
            out.append((COUNTS[k], float(p[k])))
        return out


def fit_kind(x: np.ndarray, y: list[str]):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=KIND_TREES, class_weight="balanced_subsample",
                                 min_samples_leaf=1, random_state=SEED, n_jobs=-1)
    clf.fit(x, y)
    clf.n_jobs = 1  # one layer at a time at run time: no worker start-up
    return clf


def fit_count(rows: list[dict], counts: list[int]):
    """The candidate scorer over every (layer, n) pair; target: n is the labelled count."""
    from sklearn.ensemble import RandomForestClassifier
    xs, ys = [], []
    for r, c in zip(rows, counts):
        xs.append(count_matrix(r))
        ys += [int(n == c) for n in COUNTS]
    clf = RandomForestClassifier(n_estimators=COUNT_TREES, class_weight="balanced_subsample",
                                 min_samples_leaf=2, random_state=SEED, n_jobs=-1)
    clf.fit(np.vstack(xs), ys)
    clf.n_jobs = 1
    return clf


def train(rows: list[dict], labels: list[str], counts: list[int], features: list[str]) -> LayerModel:
    """Model over feature dicts `rows`: kind from every labelled row, count from the rows of
    a points class with a count in COUNTS."""
    x = np.array([[float(r[k]) for k in features] for r in rows])
    kind = fit_kind(x, labels)
    cr = [(r, c) for r, c, y in zip(rows, counts, labels) if y in POINTS_CLASSES and c in COUNTS]
    count = fit_count([r for r, _ in cr], [c for _, c in cr]) if cr else None
    return LayerModel(kind=kind, count=count, features=list(features))


def save(model: LayerModel, path: Path = MODEL_PATH) -> None:
    import joblib
    joblib.dump(model, path, compress=3)


_CACHE: dict[str, LayerModel] = {}


def load(path: Path = MODEL_PATH) -> LayerModel | None:
    """The saved model, or None when there is none (the rules then decide alone)."""
    key = str(path)
    if key not in _CACHE:
        if not path.exists():
            return None
        import joblib
        _CACHE[key] = joblib.load(path)
    return _CACHE[key]
