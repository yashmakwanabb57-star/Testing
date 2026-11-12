"""Train and evaluate the learned head-layer classifier (ML-1).

Usage: .venv/Scripts/python.exe tools/train_layers.py [--eval-only]

Reads `rosette/data/layers.csv` (tools/build_layer_dataset.py), evaluates leave one
DESIGN out (every layer of the held-out design's photos is predicted by a model trained on
the other designs, as tools/score.py's new-image mode groups them; never leave-one-layer-out,
which would leak a photo's other layers; confirmed downloads never enter it) for the model
and, on the same labels, the rules (`rules`, `rules_count` columns: what the rule-based
head drew), prints per-class accuracy, confusion matrices and count accuracy, derives
the bars (the lowest held-out probability / count score from which the model is right on at
least GATE_PRECISION of the layers the bar is applied to: ring and point-shape layers for the
kind, non-trim points layers for the count) and, unless --eval-only, trains on every
labelled layer and saves `rosette/data/layer_model.joblib` (with the training set's sha256,
the bar and the LOO summary) and `rosette/data/layer_model_check.json` (one stored
prediction the tests reproduce).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # run as a script
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from rosette.layer_model import (CLASSES, CONFIRMED, COUNTS, DATASET, META, MODEL_PATH, data_hash,  # noqa: E402
                                 POINTS_CLASSES, LayerModel, as_features, fit_count, fit_kind, read_rows,
                                 rules_features, save, train)
from rosette.vision.layer_classify import RING_CLASSES, SHAPE_CLASSES  # noqa: E402

CHECK = ROOT / "rosette" / "data" / "layer_model_check.json"
# The model decides a layer when its probability is at or above the bar where its LOO
# precision (right among the layers at or above the bar) reaches this share.
GATE_PRECISION = 0.9
# Rules classes whose count the count scorer may replace (non-trim points layers).
COUNT_RULES = ("loop", "kite", "petal", "star", "base_star")
# Fewest LOO layers above a candidate bar for its precision to count.
GATE_MIN_LAYERS = 20


def feature_columns(rows: list[dict]) -> list[str]:
    return [k for k in rows[0] if k not in META]


def labelled(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["label"]]


def held_out_rows(rows: list[dict]) -> list[dict]:
    """The rows a held-out evaluation may use: labelled reference rows, never confirmed
    downloads (their design is not always known)."""
    return [r for r in labelled(rows) if r.get("source") != "confirmed"]


def model_columns(cols: list[str], with_rules: bool) -> list[str]:
    """The kind model's inputs: the measured columns, and with_rules the rules' reading."""
    extra = list(rules_features("none", "points", 0)) if with_rules else []
    return list(cols) + extra


def loo(rows: list[dict], cols: list[str]) -> list[dict]:
    """Per labelled row: the model's held-out kind, probability and count, trained without
    every photo of the row's design (`design` column: the scoring's grouping, so a variant
    such as S36-15B never teaches its S36-15 twin); `cols` are the kind model's inputs."""
    feats = as_features(rows, [c for c in cols if c in rows[0]])
    x = np.array([[f[k] for k in cols] for f in feats])
    y = [r["label"] for r in rows]
    counts = [int(r["count"]) for r in rows]
    groups = [r.get("design") or r["photo"] for r in rows]
    out = [dict() for _ in rows]
    for g in sorted(set(groups)):
        test = [i for i, d in enumerate(groups) if d == g]
        tr = [i for i, d in enumerate(groups) if d != g]
        kind = fit_kind(x[tr], [y[i] for i in tr])
        ct = [i for i in tr if y[i] in POINTS_CLASSES and counts[i] in COUNTS]
        count = fit_count([feats[i] for i in ct], [counts[i] for i in ct])
        m = LayerModel(kind=kind, count=count, features=cols)
        tf = [feats[i] for i in test]
        for i, (k, p), (c, cp) in zip(test, m.predict_kind(tf), m.predict_count(tf)):
            out[i] = dict(kind=k, prob=p, count=c, count_score=cp)
    return out


def kind_scope(rows: list[dict]) -> list[int]:
    """Rows the kind bar is applied to: the rules read a plain or pinked ring or a point
    shape (layer_classify.RING_CLASSES, SHAPE_CLASSES)."""
    return [i for i, r in enumerate(rows) if r["rules"] in RING_CLASSES + SHAPE_CLASSES]


def count_scope(rows: list[dict]) -> list[int]:
    """Rows the count bar is applied to: non-trim points layers of the rules with a
    labelled count."""
    return [i for i, r in enumerate(rows) if r["rules"] in COUNT_RULES and r["label"] in POINTS_CLASSES
            and int(r["count"]) in COUNTS]


def bars(rows: list[dict], res: list[dict]) -> dict:
    """The kind and count bars (`gate`) on their scopes, with precision and coverage."""
    ks, cs = kind_scope(rows), count_scope(rows)
    kb = gate([res[i]["prob"] for i in ks], [res[i]["kind"] == rows[i]["label"] for i in ks])
    cb = gate([res[i]["count_score"] for i in cs], [res[i]["count"] == int(rows[i]["count"]) for i in cs])
    return dict(kind=kb, count=cb, kind_n=len(ks), count_n=len(cs))


def per_class(y: list[str], pred: list[str]) -> dict[str, tuple[int, int]]:
    """{class: (right, n)} by the label."""
    out = {}
    for c in CLASSES:
        idx = [i for i, t in enumerate(y) if t == c]
        out[c] = (sum(pred[i] == c for i in idx), len(idx))
    return out


def confusion(y: list[str], pred: list[str]) -> list[str]:
    cl = [c for c in CLASSES if c in y or c in pred]
    ab = {c: c[:6] for c in cl}
    lines = ["label \\ pred " + " ".join(f"{ab[c]:>6}" for c in cl)]
    cnt = Counter(zip(y, pred))
    for t in cl:
        if t in y:
            lines.append(f"{t:13} " + " ".join(f"{cnt.get((t, p), 0) or '.':>6}" for p in cl))
    return lines


def gate(probs: list[float], right: list[bool], precision: float = GATE_PRECISION,
         min_layers: int = GATE_MIN_LAYERS) -> tuple[float, float, float]:
    """(bar, precision at it, share of layers at or above it): the lowest probability bar
    whose LOO precision at or above it is >= `precision` over >= `min_layers` layers and
    stays so at every higher bar with that many layers. (1.01, nan, 0) when none."""
    order = np.argsort(probs, kind="stable")[::-1]
    p = np.array(probs, float)[order]
    r = np.array(right, float)[order]
    n = len(p)
    if n < min_layers:
        return 1.01, float("nan"), 0.0
    prec = np.cumsum(r) / np.arange(1, n + 1)
    run_min = np.minimum.accumulate(prec[min_layers - 1:])  # precision at every higher bar
    best = 0
    for k in range(min_layers, n + 1):
        if run_min[k - min_layers] < precision:
            break
        if k == n or p[k - 1] > p[k]:  # a bar cannot split equal probabilities
            best = k
    if best == 0:
        return 1.01, float("nan"), 0.0
    return float(p[best - 1]), float(prec[best - 1]), best / n


def report(rows: list[dict], res: list[dict]) -> list[str]:
    y = [r["label"] for r in rows]
    rk = [r["rules"] for r in rows]
    mk = [x["kind"] for x in res]
    lines = [f"layers {len(rows)} on {len(set(r['photo'] for r in rows))} photos, "
             f"{len(set(r.get('design') or r['photo'] for r in rows))} designs (leave one design out)", ""]
    pr, pm = per_class(y, rk), per_class(y, mk)
    lines.append(f"{'class':12} {'n':>4} {'rules':>7} {'model':>7}")
    for c in CLASSES:
        n = pr[c][1]
        if n:
            lines.append(f"{c:12} {n:4d} {pr[c][0] / n:7.3f} {pm[c][0] / n:7.3f}")
    ar = np.mean([a == b for a, b in zip(y, rk)])
    am = np.mean([a == b for a, b in zip(y, mk)])
    bal_r = np.mean([pr[c][0] / pr[c][1] for c in CLASSES if pr[c][1]])
    bal_m = np.mean([pm[c][0] / pm[c][1] for c in CLASSES if pm[c][1]])
    lines += [f"{'kind all':12} {len(y):4d} {ar:7.3f} {am:7.3f}",
              f"{'balanced':12} {'':4} {bal_r:7.3f} {bal_m:7.3f}", ""]
    pts = [i for i, t in enumerate(y) if t in POINTS_CLASSES and int(rows[i]["count"]) in COUNTS]
    cr = np.mean([int(rows[i]["rules_count"]) == int(rows[i]["count"]) for i in pts]) if pts else float("nan")
    cm = np.mean([res[i]["count"] == int(rows[i]["count"]) for i in pts]) if pts else float("nan")
    both_r = np.mean([rk[i] == y[i] and int(rows[i]["rules_count"]) == int(rows[i]["count"]) for i in pts])
    both_m = np.mean([mk[i] == y[i] and res[i]["count"] == int(rows[i]["count"]) for i in pts])
    lines += [f"count (points labels, {len(pts)} layers): rules {cr:.3f} model {cm:.3f}",
              f"kind and count both right on them: rules {both_r:.3f} model {both_m:.3f}", ""]
    for c in POINTS_CLASSES:
        idx = [i for i in pts if y[i] == c]
        if idx:
            lines.append(f"  count {c:10} {len(idx):4d} rules "
                         f"{np.mean([int(rows[i]['rules_count']) == int(rows[i]['count']) for i in idx]):.3f} "
                         f"model {np.mean([res[i]['count'] == int(rows[i]['count']) for i in idx]):.3f}")
    lines += ["", "rules confusion:"] + confusion(y, rk) + ["", "model confusion:"] + confusion(y, mk)
    b = bars(rows, res)
    (bar, prec, cov), (cbar, cprec, ccov) = b["kind"], b["count"]
    lines += ["", f"kind bar (on the {b['kind_n']} ring / point-shape layers): probability >= {bar:.3f} -> "
                  f"held-out precision {prec:.3f} on {cov:.1%} of them",
              f"count bar (on the {b['count_n']} non-trim points layers): score >= {cbar:.3f} -> "
              f"held-out precision {cprec:.3f} on {ccov:.1%} of them"]
    ks = kind_scope(rows)
    above = [i for i in ks if res[i]["prob"] >= bar]
    if above:
        lines.append(f"  rules on the layers above the kind bar: {np.mean([rk[i] == y[i] for i in above]):.3f}; "
                     f"model and rules disagree on {sum(mk[i] != rk[i] for i in above)}, model right on "
                     f"{sum(mk[i] != rk[i] and mk[i] == y[i] for i in above)}")
    gated = [mk[i] if i in set(above) else rk[i] for i in range(len(y))]
    lines.append(f"  gated kind (model above the bar on its scope, else rules): "
                 f"{np.mean([a == b for a, b in zip(y, gated)]):.3f}")
    return lines


def main(argv: list[str]) -> int:
    rows = read_rows(DATASET)
    extra = read_rows(CONFIRMED) if CONFIRMED.exists() else []
    cols = feature_columns(rows)
    lab = held_out_rows(rows)
    results = {}
    for with_rules in (False, True):
        mc = model_columns(cols, with_rules)
        res = loo(lab, mc)
        results[with_rules] = (mc, res)
        print(f"==== model inputs: measurements{' + the rules reading' if with_rules else ' only'}")
        print(chr(10).join(report(lab, res)))
        print()
    # the variant kept: the one with the higher LOO kind accuracy
    acc = {w: np.mean([x["kind"] == r["label"] for x, r in zip(res, lab)]) for w, (_, res) in results.items()}
    with_rules = acc[True] >= acc[False]
    mc, res = results[with_rules]
    print(f"kept: {'measurements + rules' if with_rules else 'measurements only'} (LOO kind {acc[with_rules]:.3f})")
    if "--eval-only" in argv:
        return 0
    allrows = lab + labelled(extra)
    feats = as_features(allrows, cols)
    m = train(feats, [r["label"] for r in allrows], [int(r["count"]) for r in allrows], mc)
    m.train_hash = data_hash()
    b = bars(lab, res)
    (bar, prec, cov), (cbar, cprec, ccov) = b["kind"], b["count"]
    m.gate, m.count_gate = bar, cbar
    m.info = dict(loo_layers=len(lab), loo_designs=len(set(r["design"] for r in lab)), gate_precision=prec,
                  gate_coverage=cov, count_gate_precision=cprec, count_gate_coverage=ccov,
                  confirmed_layers=len(labelled(extra)), with_rules=bool(with_rules), loo_kind=float(acc[with_rules]))
    save(m)
    probe = feats[0]
    (k, p), (c, cs) = m.predict_kind([probe])[0], m.predict_count([probe])[0]
    CHECK.write_bytes((json.dumps(dict(features=probe, kind=k, prob=p, count=c, count_score=cs,
                                       train_hash=m.train_hash), indent=1) + chr(10)).encode("utf-8"))
    print(f"saved {MODEL_PATH.relative_to(ROOT)} (train hash {m.train_hash[:12]}, kind bar {bar:.3f}, "
          f"count bar {cbar:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
