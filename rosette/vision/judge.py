"""ML-4 judge: every candidate head rendered in the photo's frame and scored against the photo.

Candidate 0 is the rules head (spec ruling 1). Each candidate is built with `rosette.build`
(whole stacked shapes, as `pipeline._head_render`), rendered once at `RENDER_H` px with
`roles.render_categories` and resampled onto the photo's pixels through the frame the model
was measured in (`parts_common.Frame`: axis, head top, px per canvas unit; the same mapping
as `pipeline.head_fit`). The class map holds group id + 1 per pixel (0 none), so it compares
directly with the photo's `labels` (group ids, -1 off the rosette).

Three terms, each in [0, 1], over the rows above `p.tails_top_y` only (the tails are ignored):
- color: share of pixels where the candidate's class equals the photo's label, over pixels
  where either is set (the photo's inside its mask), as `head_fit.agreement`;
- edge: share of the photo's Canny edges (grey image, `CANNY_LOW`/`CANNY_HIGH`, inside the
  mask) within `EDGE_TOL_PX` of a render boundary (a class change in the class map);
- piece: mean over the pieces passed in of the IoU between the piece and the connected
  region of one class in the class map that overlaps it most (the candidate's visible part
  that covers it; derived from the class map, no extra render).
A term without evidence (no pieces, no photo edges) is dropped and the weights renormalised.
Nothing is cached across calls.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import cv2
import numpy as np

from rosette.build import build
from rosette.model import RosetteModel
from rosette.roles import PAD_FRAC, render_categories
from rosette.vision.parts_common import CANVAS_AXIS_X, CANVAS_TOP_Y, Frame
from rosette.writer import write_svg

# Weights of the three terms (Task 6 may re-mine them against the photos, spec ruling 3).
W_COLOR, W_EDGE, W_PIECE = 0.5, 0.3, 0.2
# Best minus runner-up under this keeps candidate 0 and flags the decision.
JUDGE_MARGIN = 0.01
RENDER_H = 400  # px, the render height of one candidate (spec ruling 4)
CANNY_LOW, CANNY_HIGH = 50, 150
EDGE_TOL_PX = 3
# Canvas size (writer viewBox 0 0 600 850).
CANVAS_W, CANVAS_H = 600.0, 850.0
_COLORS = {1: "#ff0000", 2: "#0000ff", 3: "#00c000", 4: "#ff00ff"}  # as pipeline._FIT_COLORS
_COLOR_TOL = 90  # summed RGB distance to a category colour (as pipeline._head_render)


@dataclass
class Verdict:
    winner: int
    scores: list[float]
    terms: list[dict]
    margin: float
    labels: list[str] = field(default_factory=list)  # one description per candidate
    changed: list[str] = field(default_factory=list)  # part ids the winner changed against candidate 0


def _part_groups(q) -> set[int]:
    if q.role == "template":
        return {q.color_group} | {int(v) for k, v in q.params.items() if k.startswith("group_")}
    return {q.color_group}


def _canvas_render(model: RosetteModel, height: int) -> tuple[np.ndarray, int] | None:
    """(group id + 1 per padded render pixel (0 none), pad) of the head parts, or None when
    the head has no parts or more groups than render colours."""
    head = [q for q in model.parts if q.role != "tail_stripe"]
    gids = sorted({g for q in head for g in _part_groups(q)})
    if not head or len(gids) > len(_COLORS):
        return None
    cat = {gid: i + 1 for i, gid in enumerate(gids)}
    groups = [copy.deepcopy(g) for g in model.groups if g.id in cat]
    for g in groups:
        g.category = cat[g.id]
    # whole stacked shapes, not visible pieces: abutting pieces leave antialiased seams (H14)
    svg = write_svg(build(RosetteModel(parts=copy.deepcopy(head), groups=groups), pieces=False))
    img = render_categories(svg, height, _COLORS).astype(np.int32)
    ref = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in _COLORS.values()])
    d = np.abs(img[:, :, None, :] - ref[None, None]).sum(axis=-1)
    lut = np.array([0] + [g + 1 for g in gids])
    cls = lut[np.where(d.min(axis=-1) < _COLOR_TOL, d.argmin(axis=-1) + 1, 0)]
    return cls, int(round(PAD_FRAC * height))


def _rows_above(p) -> int:
    return int(min(max(np.ceil(p.tails_top_y), 0), p.mask.shape[0]))


def render_classes(model: RosetteModel, p, height: int = RENDER_H) -> np.ndarray:
    """Class map in the photo's frame (shape of p.mask): group id + 1 of the candidate's head
    per pixel, 0 none; rows from p.tails_top_y down are 0 (the tails are not judged)."""
    out = np.zeros(p.mask.shape, np.int32)
    r = _canvas_render(model, height)
    top_rows = _rows_above(p)
    if r is None or top_rows == 0:
        return out
    img, pad = r
    fr = Frame.of(p)
    kx = (img.shape[1] - 2 * pad) / CANVAS_W
    ky = (img.shape[0] - 2 * pad) / CANVAS_H
    xs = np.arange(p.mask.shape[1], dtype=float)
    ys = np.arange(top_rows, dtype=float)
    cols = np.floor(pad + (CANVAS_AXIS_X + (xs - fr.axis_x) / fr.scale) * kx).astype(np.int64)
    rows = np.floor(pad + (CANVAS_TOP_Y + (ys - fr.top) / fr.scale) * ky).astype(np.int64)
    okc = (cols >= 0) & (cols < img.shape[1])
    okr = (rows >= 0) & (rows < img.shape[0])
    sub = img[np.clip(rows, 0, img.shape[0] - 1)][:, np.clip(cols, 0, img.shape[1] - 1)]
    out[:top_rows] = np.where(okr[:, None] & okc[None, :], sub, 0)
    return out


def _photo_side(p, labels: np.ndarray) -> dict:
    """Photo-only data shared by every candidate of one judge call."""
    h, w = p.mask.shape
    top_rows = _rows_above(p)
    above = np.zeros((h, w), bool)
    above[:top_rows] = True
    region = p.mask & above
    ph = np.where(region, labels + 1, 0).astype(np.int32)
    # Canny on the crop above the tails only, so no tails pixel reaches an edge through the
    # Sobel window or the hysteresis
    edges = np.zeros((h, w), bool)
    if top_rows:
        grey = cv2.cvtColor(np.ascontiguousarray(p.rgb[:top_rows]), cv2.COLOR_RGB2GRAY)
        edges[:top_rows] = cv2.Canny(grey, CANNY_LOW, CANNY_HIGH) > 0
    edges &= region
    return {"above": above, "ph": ph, "edges": edges}


def _boundaries(cm: np.ndarray) -> np.ndarray:
    """Pixels on either side of a class change (4-neighbours)."""
    b = np.zeros(cm.shape, bool)
    dx = cm[:, 1:] != cm[:, :-1]
    dy = cm[1:, :] != cm[:-1, :]
    b[:, 1:] |= dx
    b[:, :-1] |= dx
    b[1:, :] |= dy
    b[:-1, :] |= dy
    return b


def _components(cm: np.ndarray) -> np.ndarray:
    """One id per connected region of one class (8-connected), 0 for class 0."""
    comp = np.zeros(cm.shape, np.int32)
    nxt = 0
    for c in np.unique(cm):
        if c == 0:
            continue
        n, lab = cv2.connectedComponents((cm == c).astype(np.uint8), connectivity=8)
        sel = lab > 0
        comp[sel] = lab[sel] + nxt
        nxt += n - 1
    return comp


def _piece_agreement(cm: np.ndarray, above: np.ndarray, pieces) -> float | None:
    """Mean IoU of each piece (its part above the tails) with the class-map region that
    overlaps it most; None without a piece above the tails."""
    ious = []
    comp = None
    for pc in pieces:
        m = pc.mask & above
        if not m.any():
            continue
        if comp is None:
            comp = _components(cm)
        ids, counts = np.unique(comp[m], return_counts=True)
        keep = ids > 0
        if not keep.any():
            ious.append(0.0)
            continue
        part = comp == int(ids[keep][np.argmax(counts[keep])])
        ious.append(int((m & part).sum()) / int((m | part).sum()))
    return float(np.mean(ious)) if ious else None


def score(model: RosetteModel, p, labels: np.ndarray, pieces, cache: dict | None = None) -> tuple[float, dict]:
    """(weighted score, terms) of one candidate against the photo; terms has "color", "edge"
    and "piece" (each in [0, 1]; a term without evidence is left out and the weights of the
    others renormalised). cache: a dict shared by the candidates of one judge call for the
    photo-side data (never across calls)."""
    cache = {} if cache is None else cache
    if "photo" not in cache:
        cache["photo"] = _photo_side(p, labels)
    ph = cache["photo"]
    cm = render_classes(model, p)
    terms: dict[str, float] = {}

    union = (ph["ph"] > 0) | (cm > 0)
    n = int(union.sum())
    terms["color"] = float(((ph["ph"] == cm) & union).sum() / n) if n else 0.0

    edges = ph["edges"]
    if edges.any():
        b = _boundaries(cm) & ph["above"]
        if b.any():
            dist = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 5)
            terms["edge"] = float((dist[edges] <= EDGE_TOL_PX).sum() / int(edges.sum()))
        else:
            terms["edge"] = 0.0

    pa = _piece_agreement(cm, ph["above"], pieces or [])
    if pa is not None:
        terms["piece"] = pa

    weights = {"color": W_COLOR, "edge": W_EDGE, "piece": W_PIECE}
    total = sum(weights[k] for k in terms)
    return float(sum(weights[k] * v for k, v in terms.items()) / total), terms


def _part_key(q) -> tuple:
    return (q.role, q.color_group, q.z, tuple(sorted((k, repr(v)) for k, v in q.params.items())))


def changed_parts(rules: RosetteModel, other: RosetteModel) -> list[str]:
    """Ids of the parts of `other` that differ from `rules` (role, colour group, z or params)
    or that `rules` has not."""
    base = {q.id: _part_key(q) for q in rules.parts}
    return [q.id for q in other.parts if base.get(q.id) != _part_key(q)]


def judge(candidates: list[RosetteModel], p, labels: np.ndarray, pieces,
          decisions: list[str] | None = None) -> Verdict:
    """Score every candidate (each rendered once) and choose: the best wins when it leads the
    runner-up by JUDGE_MARGIN or more, else candidate 0 (the rules head). decisions: one
    description per candidate for the flag lines (default "candidate <i>")."""
    if not candidates:
        raise ValueError("judge needs at least one candidate")
    if decisions is not None and len(decisions) != len(candidates):
        raise ValueError("judge needs one decision description per candidate")
    names = list(decisions) if decisions is not None else [f"candidate {i}" for i in range(len(candidates))]
    cache: dict = {}
    scores, terms = [], []
    for m in candidates:
        s, t = score(m, p, labels, pieces, cache)
        scores.append(s)
        terms.append(t)
    if len(candidates) == 1:
        return Verdict(winner=0, scores=scores, terms=terms, margin=1.0, labels=names)
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    margin = float(scores[order[0]] - scores[order[1]])
    winner = order[0] if margin >= JUDGE_MARGIN else 0
    return Verdict(winner=winner, scores=scores, terms=terms, margin=margin, labels=names,
                   changed=changed_parts(candidates[0], candidates[winner]) if winner else [])


def flag(model: RosetteModel, verdict: Verdict) -> RosetteModel:
    """A copy of the chosen model with the verdict written on it: decided, the parts the winner
    changed get confidence min(confidence, 0.5 + 10 * margin) and an info line names the
    choice; undecided (margin under JUDGE_MARGIN), a warning says the rules' reading is kept.
    A single candidate adds nothing."""
    out = copy.deepcopy(model)
    n = len(verdict.scores)
    if n < 2:
        return out
    names = verdict.labels or [f"candidate {i}" for i in range(n)]
    if verdict.margin < JUDGE_MARGIN:
        rival = max(range(1, n), key=lambda i: (verdict.scores[i], -i))
        out.warnings.append(f"judge undecided: {names[0]} or {names[rival]} kept as the rules read it")
        return out
    cap = 0.5 + 10 * verdict.margin
    changed = set(verdict.changed)
    for q in out.parts:
        if q.id in changed:
            q.confidence = min(q.confidence, cap)
    out.info.append(f"judge: {names[0]} -> {names[verdict.winner]} (margin {verdict.margin:.3f})")
    return out
