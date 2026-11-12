"""The learned layer classifier at run time (ML-1): kind and count of the rule-based head layers.

`classify_head_layers(p, labels, model, merged)` runs after the rule-based head is detected,
tidied and assigned (`pipeline.detect`), on the same rows the classifier was trained on
(`head_layer_rows`, shared with `tools/build_layer_dataset.py`):

- count: the count scorer's n replaces the rules' count of every points layer that is not a
  trim when its score is at or above the count bar (`LayerModel.count_gate`, from the
  leave-one-photo-out precision; below it a different count adds COUNT_WARNING); the
  layer's rotation is re-read for the new n (phase of the n-th harmonic, as the
  detector does) and its width ratio scaled by old / new count; the edge strips and backings
  on its axes follow it.
- kind: at or above the model's bar (`LayerModel.gate`, from the leave-one-photo-out
  precision) the predicted class replaces the rules' class, but only between plain and
  pinked ring and among the point shapes (loop, kite, petal, star, backing, edge, base
  star), and only where the change is a parameter change: loop / kite / petal / star set
  the kind (an edge strip loses its strip width), backing takes the kind of the host it
  sits behind (another Color, same count and axes, shorter). Making a layer an edge strip
  or the base star, or a backing without a host, needs geometry the model does not give.
  Below the bar, or where the change cannot be made, a disagreement within that scope
  adds KIND_WARNING for the part. The role is never changed.

Without a saved model (`rosette.layer_model.load` -> None) nothing changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rosette.layer_model import POINTS_CLASSES, LayerModel, model_without, rules_features
from rosette.vision.layer_features import layer_features
from rosette.vision.parts_common import Frame, polar_sample, r3
from rosette.vision.parts_head import RING_THETA, TAIL_SECTOR_DEG

# Unknown parts count as head layers when their center lies within this many head radii.
UNKNOWN_IN = 1.05
# Radial step of the polar unwrap (canvas units; parts_head.RADIAL_STEP_UNITS).
STEP_UNITS = 1.0
# Axes within this many degrees are the same axes (tools.stacking.BACK_AXIS_DEG).
AXIS_DEG = 4.0
KIND_WARNING = "Part {pid}: the layer kind is uncertain - check the head against the photo."
COUNT_WARNING = "Part {pid}: the point count is uncertain - check the number of points against the photo."
RING_CLASSES = ("ring", "pinked_ring")
SHAPE_CLASSES = ("loop", "kite", "petal", "star", "backing", "edge", "base_star")
_KIND_PARAM = {"loop": "bow", "kite": "diamond", "petal": "petal", "star": "star"}


def rules_class(part, parts) -> str:
    """Class of a rule-based head part (layer_model.CLASSES)."""
    a = part.params
    if part.role == "head":
        return "disc"
    if part.role == "ring":
        return "pinked_ring" if int(a.get("pinked", 0)) else "ring"
    if part.role == "center":
        return "center"
    if part.role != "points":
        return "none"
    if part.id == "base":
        return "base_star"
    if a.get("edge_width"):
        return "edge"
    if _host(part, parts) is not None:
        return "backing"
    return {"bow": "loop", "diamond": "kite", "petal": "petal", "star": "star"}.get(str(a.get("kind")), "none")


def _same_axes(a: dict, b: dict) -> bool:
    return abs((float(a["rotation_deg"]) - float(b["rotation_deg"]) + 180) % 360 - 180) <= AXIS_DEG


def _host(part, parts, kind=None):
    """The points layer `part` is a backing of: another Color, the same kind (or `kind`),
    count and axes, a shorter reach; or None."""
    a = part.params
    for q in parts:
        b = q.params
        if (q is not part and q.role == "points" and q.color_group != part.color_group
                and b.get("kind") == (kind or a.get("kind")) and not b.get("edge_width")
                and int(b.get("count", 0)) == int(a["count"]) and _same_axes(a, b)
                and float(a["r_outer"]) > float(b["r_outer"])):
            return q
    return None


def rules_count(part, cls: str) -> int:
    return int(part.params.get("count", 0)) if cls in POINTS_CLASSES else 0


def extent(part, center: tuple[float, float]) -> tuple[float, float]:
    """Radial extent (canvas units) of a detected head part."""
    a = part.params
    if part.role in ("points", "ring"):
        return float(a["r_inner"]), float(a["r_outer"])
    if part.role in ("head", "center"):
        return 0.0, float(a["r"])
    d = float(np.hypot(float(a["cx"]) - center[0], float(a["cy"]) - center[1]))
    return max(0.0, d - float(a["r"])), d + float(a["r"])


def context_features(k: int, extents: list[tuple[float, float]], counts: list[float]) -> dict[str, float]:
    """Numbers about the layers overlapping layer k (extents in head radii; counts: their
    best symmetry n): how many overlap it, by how much the nearest other outer edge lies
    inside or beyond its own, whether one encloses it and whether one shares its count."""
    a0, a1 = extents[k]
    over = [j for j, (b0, b1) in enumerate(extents) if j != k and min(a1, b1) - max(a0, b0) > 0.02]
    d_out = min((a1 - extents[j][1] for j in over), key=abs, default=0.0)
    return {"n_over": float(len(over)), "d_out": float(d_out),
            "inside_other": float(any(extents[j][0] <= a0 + 0.01 and extents[j][1] >= a1 - 0.01 for j in over)),
            "same_n_over": float(any(counts[j] == counts[k] for j in over))}


@dataclass
class HeadUnwrap:
    """The polar label unwrap the rows were measured on."""
    center: tuple[float, float]  # canvas units
    center_px: tuple[float, float]
    r_px: float
    head_r: float  # canvas units
    scale: float
    radii: np.ndarray
    pol: np.ndarray
    win: np.ndarray


def head_layer_rows(p, labels: np.ndarray, model, merged: dict[int, int]) -> tuple[list[dict], HeadUnwrap | None]:
    """Rows (part, features incl. context and the rules' reading, rules class and count,
    extent in head radii, label members) of the model's head layers, and the unwrap."""
    fr = Frame.of(p)
    parts = [q for q in model.parts if q.role in ("head", "ring", "points", "center")]
    if not parts:
        return [], None
    ccx, ccy = float(parts[0].params["cx"]), float(parts[0].params["cy"])
    cx_px = fr.axis_x + (ccx - 300.0) * fr.scale
    cy_px = fr.py(ccy)
    r_px = cy_px - (p.head[1] - p.head[2])  # refine_center keeps the head top
    head_r = r_px / fr.scale
    for q in model.parts:
        if (q.role == "unknown"
                and np.hypot(float(q.params["cx"]) - ccx, float(q.params["cy"]) - ccy) <= UNKNOWN_IN * head_r):
            parts.append(q)
    step = fr.scale * STEP_UNITS
    radii = np.arange(step / 2, 1.1 * r_px, step)
    pol = polar_sample(labels.astype(np.int32), cx_px, cy_px, radii, RING_THETA, fill=-1)
    theta = np.degrees(np.linspace(0, 2 * np.pi, RING_THETA, endpoint=False))
    win = np.abs(theta - 90.0) > TAIL_SECTOR_DEG
    groups = {g.id: g for g in model.groups}
    rows = []
    for q in parts:
        g = q.color_group
        members = [g] + [a for a, b in merged.items() if b == g]
        r0, r1 = extent(q, (ccx, ccy))
        grp = groups.get(g)
        f = layer_features(pol, radii, win, members, r0 * fr.scale, r1 * fr.scale, r_px, fr.scale,
                           grp.category if grp else None, grp.pattern if grp else False,
                           grp.texture if grp else 0.0)
        cls = rules_class(q, model.parts)
        rows.append(dict(part=q, features=f, rules=cls, rules_count=rules_count(q, cls),
                         extent=(r0 / head_r, r1 / head_r), members=members))
    extents = [r["extent"] for r in rows]
    counts = [r["features"]["sym_best_n"] for r in rows]
    for k, r in enumerate(rows):
        r["features"].update(context_features(k, extents, counts))
        r["features"].update(rules_features(r["rules"], r["part"].role, r["rules_count"]))
    return rows, HeadUnwrap((ccx, ccy), (cx_px, cy_px), r_px, head_r, fr.scale, radii, pol, win)


def rotation_for(u: HeadUnwrap, members, r_inner: float, r_outer: float, n: int) -> float:
    """Rotation (deg) of an n-lobed layer of `members` between r_inner..r_outer (canvas
    units): the phase of the n-th harmonic over the visible angles (parts_head._symmetry)."""
    i0 = int(np.searchsorted(u.radii, r_inner * u.scale))
    i1 = max(i0 + 1, int(np.searchsorted(u.radii, r_outer * u.scale, side="right")))
    x = np.isin(u.pol[i0:i1], members).astype(float)
    th = np.linspace(0, 2 * np.pi, x.shape[1], endpoint=False)
    c = (x[:, u.win] * np.exp(-1j * n * th[u.win])[None]).sum()
    return float((-np.degrees(np.angle(c)) / n) % (360.0 / n))


def _set_count(part, n: int, rot: float) -> None:
    a = part.params
    old = int(a["count"])
    a["width_ratio"] = r3(float(a["width_ratio"]) * old / n)
    a["count"], a["rotation_deg"] = int(n), r3(rot)


def apply_counts(rows: list[dict], u: HeadUnwrap, lm: LayerModel, parts) -> tuple[list[str], list[str]]:
    """The count scorer's count for every non-trim points layer where its score is at or
    above the model's count bar (`LayerModel.count_gate`); its trims follow. Below the bar a
    different count adds COUNT_WARNING. Returns (ids changed, warnings)."""
    hosts = [r for r in rows if r["part"].role == "points" and r["rules"] not in ("edge", "backing")]
    changed, warnings = [], []
    bar = float(getattr(lm, "count_gate", 0.0))
    for r, (n, score) in zip(hosts, lm.predict_count([r["features"] for r in hosts])):
        q = r["part"]
        old = int(q.params["count"])
        if n == old:
            continue
        if score < bar:
            warnings.append(COUNT_WARNING.format(pid=q.id))
            continue
        before = dict(q.params)
        trims = [t["part"] for t in rows if t["rules"] in ("edge", "backing") and t["part"] is not q
                 and int(t["part"].params["count"]) == old and _same_axes(t["part"].params, before)]
        rot = rotation_for(u, r["members"], float(q.params["r_inner"]), float(q.params["r_outer"]), n)
        _set_count(q, n, rot)
        changed.append(q.id)
        for t in trims:
            _set_count(t, n, rot)
            changed.append(t.id)
    return changed, warnings


def apply_kinds(rows: list[dict], lm: LayerModel, parts) -> tuple[list[str], list[str]]:
    """The kind model's class where it may decide (module doc). Returns (ids changed,
    warnings)."""
    changed, warnings = [], []
    for r, (cls, prob) in zip(rows, lm.predict_kind([r["features"] for r in rows])):
        q, rc = r["part"], r["rules"]
        scope = RING_CLASSES if rc in RING_CLASSES else SHAPE_CLASSES if rc in SHAPE_CLASSES else ()
        if cls == rc or cls not in scope:
            continue
        done = prob >= lm.gate and _change(q, rc, cls, parts)
        if done:
            changed.append(q.id)
        else:
            warnings.append(KIND_WARNING.format(pid=q.id))
    return changed, warnings


def _change(q, old: str, new: str, parts) -> bool:
    """Make part q of class `old` class `new` when it is a parameter change."""
    a = q.params
    if new in RING_CLASSES:
        if new == "pinked_ring":
            a["pinked"] = 1
        else:
            a.pop("pinked", None)
            a.pop("pleats", None)
        return True
    if new in _KIND_PARAM:
        a["kind"] = _KIND_PARAM[new]
        a.pop("edge_width", None)
        return True
    if new == "backing" and old != "edge":
        for kind in ("bow", "diamond", "petal", "star"):
            host = _host(q, parts, kind)
            if host is not None:
                a["kind"] = host.params["kind"]
                return True
    return False


def classify_head_layers(p, labels: np.ndarray, model, merged: dict[int, int],
                         lm: LayerModel | None = None, exclude_key: str | None = None) -> list[str]:
    """Applies the given layer model, else the saved one (trained without the design
    `exclude_key` when given: evaluation only), to the model's head layers in place;
    returns the warnings to add. No model: no change."""
    lm = lm if lm is not None else model_without(exclude_key)
    if lm is None:
        return []
    rows, u = head_layer_rows(p, labels, model, merged)
    if not rows:
        return []
    warnings = []
    if lm.count is not None:
        warnings += apply_counts(rows, u, lm, model.parts)[1]
    warnings += apply_kinds(rows, lm, model.parts)[1]
    return warnings
