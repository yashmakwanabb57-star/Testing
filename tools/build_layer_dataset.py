"""Head-layer dataset for the learned layer classifier (ML-1).

Usage: .venv/Scripts/python.exe tools/build_layer_dataset.py [--cache DIR] [IDs...]
       (writes rosette/data/layers.csv; with IDs, prints their rows only; --cache keeps
       each photo's detection and reference layers as pickles in DIR for reruns)

For every reference photo the pipeline's detection (`rosette.pipeline.detect`: prepare,
color groups, the rule-based head parts, tidy, assign) gives the head layers. Each one
(head disc, ring, points layer, base star, center, unknown inside the head) gets:
- its measured features (`rosette.vision.layer_features`) read off the photo's labels
  between its radial extent, plus a few context numbers (layers overlapping it);
- the rules' class (`rules_class`: what the rule-based head drew) and count;
- a label: the class and count of the reference SVG layer it overlaps most (`ref_layers`,
  `match`), compared in head radii around each file's own head center. A detected layer
  overlapping no reference layer (IoU under MATCH_MIN) is `none`.

Classes: disc (head disc), pinked_ring, ring, loop, kite, petal, star, base_star,
backing, edge (edge strip), center, none.

Reference files that differ from their photo (`docs/pipeline.md`, Known limitations and
the H11-H13 rulings) are handled by `CURATED`: where the docs confirm the rules' reading
of the photo it is the label (source "rules"), elsewhere the layer is left out.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # run as a script
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from shapely import contains_xy  # noqa: E402

from rosette.roles import category, circular_runs, classify_masks, drawn_frame, element_masks, polar_mask  # noqa: E402
from rosette.vision.layer_features import COUNTS, layer_features, symmetry_scores  # noqa: E402
from rosette.vision.parts_common import Frame, polar_sample  # noqa: E402
from rosette.vision.parts_head import RING_THETA, TAIL_SECTOR_DEG, _part_geom  # noqa: E402
from tools.stacking import mark_backings, piece_kind  # noqa: E402
from tools.truth import PINKED_DEPTH, _rim_depth  # noqa: E402

REF = ROOT / "reference"
OUT = ROOT / "rosette" / "data" / "layers.csv"
CLASSES = ("disc", "pinked_ring", "ring", "loop", "kite", "petal", "star", "base_star", "backing", "edge",
           "center", "none")
# Classes whose layers have a lobe count.
POINTS_CLASSES = ("loop", "kite", "petal", "star", "base_star", "backing", "edge")
# Comparison grid: GRID points per side over +-GRID_R head radii.
GRID, GRID_R = 241, 1.25
# A detected layer whose best match score (see `match`) is under this overlaps nothing: none.
MATCH_MIN = 0.25
# A reference layer of another Color counts this share of its IoU.
OTHER_COLOR = 0.5
# Reference points pieces of one Color and kind join one layer when their radial extents
# share at least GROUP_SHARE of the shorter one (designers place lobes by hand).
GROUP_SHARE = 0.5
# Turned self-correlation from which a reference layer's order counts (parts_head.SYM_T).
ORDER_T = 0.6
# A star layer under every other head element, with no head disc, reaching at least
# BASE_R head radii, is the head's base star (H13).
BASE_R = 0.85
# A reference points piece covering whole circles (FULL_ROW, FULL_ROWS_MIN) whose outline
# varies by PLEAT_DEPTH units (min, max) is a pleated annulus: a pinked ring (`piece_shape`).
FULL_ROW, FULL_ROWS_MIN, PLEAT_DEPTH = 0.75, 0.1, (1.0, 12.0)
# A reference points layer starting within this many head radii of the middle is the
# center drawn lobed (roles.CENTER_RMIN).
CENTER_R = 0.1
# Unknown parts count as head layers when their center lies within this many head radii.
UNKNOWN_IN = 1.05
# The photo pixels a detected part is matched by are those of its color inside its radial
# extent and within DRAWN_GROW_UNITS of its drawn shape, unless that keeps under
# DRAWN_MIN_SHARE of them (a drawn shape far off its color: the color's pixels decide).
DRAWN_GROW_UNITS, DRAWN_MIN_SHARE = 3.0, 0.3
# Radial step of the polar unwrap (canvas units; parts_head.RADIAL_STEP_UNITS).
STEP_UNITS = 1.0

# Reference-differs photos (docs/pipeline.md). "rules": the docs confirm the rule-based
# head's reading of the photo, so the rules' class/count is the label, except the parts
# listed in "exclude" (by role:kind). "exclude": layers left out (by the label's class or
# the rules' class), "all" for every head layer.
CURATED: dict[str, dict] = {
    # H12/H13: pink star over the dark border star over loops with pink edges, ring,
    # center is the photo's construction; the pink pleated ring still reads as a plain
    # dark ring (Known limitations), so the rings are left out.
    "S29-15": {"source": "rules", "exclude": {"ring", "pinked_ring", "none"}},
    # H11/H13 rulings: the rule-based head (blue base star, glitter 4-point star on its
    # axis arms, three pinked rings, badge) fits the photo; the reference's 8 glitter
    # points are not on it.
    "S53-19": {"source": "rules", "exclude": {"none"}},
    # Known limitations: the photo's two 6-point stars are 4-point layers in the
    # reference and in the rules; neither reading is right.
    "s44-15": {"source": "ref", "exclude": set(POINTS_CLASSES)},
}


# ---------------------------------------------------------------- the rules' reading

def rules_class(part, parts) -> str:
    """Class of a rule-based head part (`CLASSES`)."""
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
    for q in parts:  # a copy of another Color's points (same kind, count, axes), reaching further
        b = q.params
        if (q is not part and q.role == "points" and q.color_group != part.color_group
                and b.get("kind") == a.get("kind")
                and int(b.get("count", 0)) == int(a["count"]) and not b.get("edge_width")
                and abs((float(a["rotation_deg"]) - float(b["rotation_deg"]) + 180) % 360 - 180) <= 4.0
                and float(a["r_outer"]) > float(b["r_outer"])):
            return "backing"
    return {"bow": "loop", "diamond": "kite", "petal": "petal", "star": "star"}.get(str(a.get("kind")), "none")


def rules_count(part, cls: str) -> int:
    return int(part.params.get("count", 0)) if cls in POINTS_CLASSES else 0


# ---------------------------------------------------------------- reference layers

def ref_layers(svg: str) -> tuple[list[dict], np.ndarray]:
    """(layers, top) of a reference file. Each layer: cls, count, category, r_in / r_out
    (head radii), order (first element index, paint order). top: on the head-radii grid
    (`_grid`), the index of the layer seen on top at each point (-1: background, tails or
    an unclassified piece)."""
    f = drawn_frame(svg)
    cx, cy, r = f.head
    masks = element_masks(svg)
    index = {id(el): k for k, (el, _) in enumerate(masks)}
    every = classify_masks(masks, f)
    pieces = [(el, role, m) for el, role, m in every if role in ("head", "ring", "points", "center")]
    piece_layer: dict[int, int] = {}
    kinds = mark_backings([(category(el), piece_kind(m, role, f.head), m) for el, role, m in pieces], f.head)
    layers: list[dict] = []
    for (el, role, m), kind in zip(pieces, kinds):
        ys, xs = np.nonzero(m)
        d = np.hypot(xs - cx, ys - cy) / r
        r_in, r_out = float(np.percentile(d, 1)), float(np.percentile(d, 99))
        cat = category(el)
        if role == "head":
            cls = "disc"
        elif role == "center":
            cls = "center"
        elif role == "ring":
            cls = "pinked_ring" if _rim_depth(m) / f.scale >= PINKED_DEPTH else "ring"
        else:
            lobes, pleated = piece_shape(m, (cx, cy, r), f.scale)
            cls = {"loop": "loop", "kite": "kite", "petal": "petal", "star": "star", "edge": "edge",
                   "backing": "backing"}.get(kind, "none")
            if pleated:  # a pleated annulus drawn as one toothed shape
                cls = "pinked_ring"
            if cls == "none":
                continue
        same = None
        for L in layers:
            if L["cls"] != cls or L["category"] != cat:
                continue
            if cls in POINTS_CLASSES:
                shared = min(L["r_out"], r_out) - max(L["r_in"], r_in)
                if shared >= GROUP_SHARE * min(L["r_out"] - L["r_in"], r_out - r_in):
                    same = L
                    break
            elif abs(L["r_out"] - r_out) <= 0.02 and abs(L["r_in"] - r_in) <= 0.02:
                same = L
                break
        if same is None:
            layers.append(dict(cls=cls, category=cat, r_in=r_in, r_out=r_out, order=index[id(el)], pieces=1,
                               mask=m.copy()))
            same = layers[-1]
        else:
            same["mask"] |= m
            same["r_in"], same["r_out"] = min(same["r_in"], r_in), max(same["r_out"], r_out)
            same["order"] = min(same["order"], index[id(el)])
            same["pieces"] += 1
        piece_layer[id(m)] = layers.index(same)
    for L in layers:
        mask = L.pop("mask")
        L["count"] = layer_order(mask, f.head) if L["cls"] in POINTS_CLASSES else 0
    if not any(L["cls"] == "disc" for L in layers):  # H13: the base star under everything
        first = min((L["order"] for L in layers), default=None)
        for L in layers:
            if L["cls"] == "star" and L["order"] == first and L["r_out"] >= BASE_R:
                L["cls"] = "base_star"
    top = np.full(_grid()[0].shape, -1, int)
    for el, _, m in sorted(every, key=lambda t: index[id(t[0])]):  # paint order
        top[ref_grid(m, f.head)] = piece_layer.get(id(m), -1)
    return layers, top


def layer_order(mask: np.ndarray, head: tuple[float, float, float]) -> int:
    """Lobe count of a reference points layer (all its pieces): the rotational order of
    its polar mask, the largest n in 2..24 whose turned self-correlation over the angles
    outside the tails sector is >= ORDER_T (as the detector reads the photo; a loop drawn
    cut in two pieces around a star tip still counts once), else the best-scoring n."""
    cx, cy, r = head
    ys, xs = np.nonzero(mask)
    d = np.hypot(xs - cx, ys - cy)
    radii = np.linspace(d.min(), d.max(), 40)
    pol = polar_mask(mask, cx, cy, radii, RING_THETA)
    th = np.degrees(np.linspace(0, 2 * np.pi, RING_THETA, endpoint=False))
    sym = symmetry_scores(pol, np.abs(th - 90.0) > TAIL_SECTOR_DEG)
    strong = [n for n, sc in zip(COUNTS, sym) if sc >= ORDER_T]
    return int(max(strong)) if strong else int(COUNTS[int(np.argmax(sym))])


def piece_shape(mask: np.ndarray, head: tuple[float, float, float], scale: float) -> tuple[int, bool]:
    """(lobes, pleated annulus) of one reference piece. lobes: the median number of runs
    on the circles it covers partly (a single loop or kite: 1, a star polygon: its points).
    A pleated annulus covers whole circles (>= FULL_ROW of the angles outside the tails
    sector, which designers often leave open, on >= FULL_ROWS_MIN of its radii) and its
    outline, freed of its slow trend, varies by PLEAT_DEPTH units (scallops, not star
    tips)."""
    cx, cy, r = head
    ys, xs = np.nonzero(mask)
    d = np.hypot(xs - cx, ys - cy)
    radii = np.linspace(d.min(), d.max(), 50)[1:-1]
    if len(radii) == 0:
        return 0, False
    pol = polar_mask(mask, cx, cy, radii)
    th = np.degrees(np.linspace(0, 2 * np.pi, pol.shape[1], endpoint=False))
    win = np.abs(th - 90.0) > TAIL_SECTOR_DEG
    cov = pol[:, win].mean(axis=1)
    part = (cov > 0.02) & (cov < FULL_ROW)
    runs = [circular_runs(row) for row, ok in zip(pol, part) if ok]
    lobes = int(np.median(runs)) if runs else 1
    if (cov >= FULL_ROW).mean() < FULL_ROWS_MIN:
        return lobes, False
    fine = np.linspace(d.min(), d.max(), 200)
    pf = polar_mask(mask, cx, cy, fine, 1440)
    th = np.degrees(np.linspace(0, 2 * np.pi, 1440, endpoint=False))
    ok = pf.any(axis=0) & (np.abs(th - 90.0) > TAIL_SECTOR_DEG + 10)
    if ok.sum() < 100:
        return lobes, False
    outer = np.where(pf.any(axis=0), len(fine) - 1 - np.argmax(pf[::-1], axis=0), 0)
    rad = fine[outer]
    from scipy import ndimage
    hp = rad - ndimage.median_filter(rad, size=73, mode="wrap")
    depth = float(np.percentile(hp[ok], 95) - np.percentile(hp[ok], 5)) / scale
    return lobes, PLEAT_DEPTH[0] <= depth <= PLEAT_DEPTH[1]


def _grid() -> tuple[np.ndarray, np.ndarray]:
    u = np.linspace(-GRID_R, GRID_R, GRID)
    return np.meshgrid(u, u)


def ref_grid(mask: np.ndarray, head: tuple[float, float, float]) -> np.ndarray:
    """A render-px mask sampled on the head-radii grid."""
    cx, cy, r = head
    uu, vv = _grid()
    xs = np.rint(cx + uu * r).astype(int)
    ys = np.rint(cy + vv * r).astype(int)
    h, w = mask.shape
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    out = np.zeros(uu.shape, bool)
    out[ok] = mask[ys[ok], xs[ok]]
    return out


def part_grid(part, center: tuple[float, float], head_r_units: float) -> np.ndarray | None:
    """A detected points, ring or center part's drawn shape (canvas units, before any
    occlusion) on the head-radii grid; None for other roles."""
    if part.role not in ("points", "ring", "center"):
        return None
    g = _part_geom(part)
    if g is None:
        return None
    uu, vv = _grid()
    return contains_xy(g, center[0] + uu * head_r_units, center[1] + vv * head_r_units)


def match(seen: np.ndarray, top: np.ndarray, layers: list[dict], cat: int | None = None) -> tuple[dict | None, float]:
    """(reference layer seen at most of a detected layer's pixels, that share). `seen`: the
    grid points where the photo shows the detected layer's color inside its radial extent
    (near its drawn shape); `top`: the reference layer on top per grid point
    (`ref_layers`). Visible is compared with visible: a star polygon or a full backing copy
    drawn under other layers does not claim what it hides. A layer of another Color than
    `cat` counts OTHER_COLOR of its share. Under MATCH_MIN: no match."""
    n = int(seen.sum())
    if n == 0 or not layers:
        return None, 0.0
    counts = np.bincount(top[seen] + 1, minlength=len(layers) + 1)[1:] / n
    w = np.array([1.0 if cat is None or L["category"] == cat else OTHER_COLOR for L in layers])
    k = int(np.argmax(counts * w))
    share = float(counts[k] * w[k])
    if share < MATCH_MIN:
        return None, share
    return layers[k], share


def trim_in_reference(cls: str, cat: int | None, extent: tuple[float, float], layers: list[dict]) -> dict | None:
    """A reference layer of trim class `cls` (edge strips, backing) and Color `cat` whose
    radial extent overlaps `extent` (head radii), or None. Trims are thin or show only
    beyond their host's tips, so a hand-drawn file rarely puts them on the photo's pixels:
    a detected trim is matched by the reference having such a trim in that band."""
    a0, a1 = extent
    for L in layers:
        if L["cls"] == cls and L["category"] == cat and min(a1, L["r_out"]) - max(a0, L["r_in"]) > 0.05:
            return L
    return None


# ---------------------------------------------------------------- the photo side

def _extent(part, head_r_units: float, center: tuple[float, float]) -> tuple[float, float]:
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


def _cached(cache: Path | None, name: str, make):
    """make(), pickled under cache/name when a cache directory is given (development reruns)."""
    if cache is None:
        return make()
    import pickle
    f = cache / name
    if f.exists():
        return pickle.loads(f.read_bytes())
    v = make()
    cache.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(v))
    return v


def photo_layers(sid: str, cache: Path | None = None) -> list[dict]:
    """Rows (features, rules class and count, part id) of one photo's head layers."""
    from rosette.pipeline import detect
    d = _cached(cache, f"{sid}.det.pkl", lambda: detect((REF / "photos" / f"{sid}.png").read_bytes()))
    return rows_of_detection(d)


def head_parts(model) -> list:
    head = [q for q in model.parts if q.role in ("head", "ring", "points", "center", "unknown")]
    return head


def rows_of_detection(d) -> list[dict]:
    """Feature rows of a Detection's head layers (see photo_layers)."""
    p, labels, model, merged = d.prepared, d.labels, d.model, d.merged
    fr = Frame.of(p)
    parts = [q for q in model.parts if q.role in ("head", "ring", "points", "center")]
    if not parts:
        return []
    ccx, ccy = float(parts[0].params["cx"]), float(parts[0].params["cy"])
    cx_px = fr.axis_x + (ccx - 300.0) * fr.scale
    cy_px = fr.py(ccy)
    r_px = cy_px - (p.head[1] - p.head[2])  # refine_center keeps the head top
    head_r = r_px / fr.scale
    for q in model.parts:
        if q.role == "unknown" and np.hypot(float(q.params["cx"]) - ccx, float(q.params["cy"]) - ccy) <= UNKNOWN_IN * head_r:
            parts.append(q)
    step = fr.scale * STEP_UNITS
    radii = np.arange(step / 2, 1.1 * r_px, step)
    pol = polar_sample(labels.astype(np.int32), cx_px, cy_px, radii, RING_THETA, fill=-1)
    theta = np.degrees(np.linspace(0, 2 * np.pi, RING_THETA, endpoint=False))
    win = np.abs(theta - 90.0) > TAIL_SECTOR_DEG
    groups = {g.id: g for g in model.groups}
    uu, vv = _grid()
    h, w = labels.shape
    gx = np.clip(np.rint(cx_px + uu * r_px).astype(int), 0, w - 1)
    gy = np.clip(np.rint(cy_px + vv * r_px).astype(int), 0, h - 1)
    inside = (cx_px + uu * r_px >= 0) & (cx_px + uu * r_px < w) & (cy_px + vv * r_px >= 0) & (cy_px + vv * r_px < h)
    glab = np.where(inside, labels[gy, gx], -1)
    rho = np.hypot(uu, vv)
    rows = []
    for q in parts:
        g = q.color_group
        members = [g] + [a for a, b in merged.items() if b == g]
        r0, r1 = _extent(q, head_r, (ccx, ccy))
        grp = groups.get(g)
        f = layer_features(pol, radii, win, members, r0 * fr.scale, r1 * fr.scale, r_px, fr.scale,
                           grp.category if grp else None, grp.pattern if grp else False,
                           grp.texture if grp else 0.0)
        cls = rules_class(q, model.parts)
        if q.role == "unknown":  # its disc
            a = q.params
            region = np.hypot(uu - (float(a["cx"]) - ccx) / head_r, vv - (float(a["cy"]) - ccy) / head_r) <= float(a["r"]) / head_r
        else:
            region = (rho >= r0 / head_r) & (rho <= r1 / head_r)
        seen = np.isin(glab, members) & region & (rho <= UNKNOWN_IN)
        drawn = part_grid(q, (ccx, ccy), head_r)
        if drawn is not None:  # where the part is drawn (same-color trims share a band)
            grow = max(1, int(round(DRAWN_GROW_UNITS / head_r / (2 * GRID_R / (GRID - 1)))))
            near = seen & ndimage.binary_dilation(drawn, iterations=grow)
            if near.sum() >= DRAWN_MIN_SHARE * seen.sum():
                seen = near
        rows.append(dict(part=q, features=f, rules=cls, rules_count=rules_count(q, cls),
                         extent=(r0 / head_r, r1 / head_r), seen=seen))
    extents = [r["extent"] for r in rows]
    counts = [r["features"]["sym_best_n"] for r in rows]
    for k, r in enumerate(rows):
        r["features"].update(context_features(k, extents, counts))
    for r in rows:
        r["center"], r["head_r"] = (ccx, ccy), head_r
    return rows


def label_rows(sid: str, rows: list[dict], cache: Path | None = None) -> None:
    """Adds label, count, iou (the match share) and source to the rows (in place); excluded rows get label ''."""
    svg = (REF / "svg" / f"{sid}.svg").read_text(encoding="utf-8")
    layers, top = _cached(cache, f"{sid}.ref.pkl", lambda: ref_layers(svg))
    for L in layers:  # a lobed badge from the middle out is the center (a rosette badge)
        if L["cls"] in POINTS_CLASSES and L["r_in"] <= CENTER_R:
            L["cls"], L["count"] = "center", 0
    cur = CURATED.get(sid, {})
    for r in rows:
        q = r["part"]
        cat = next((c for c in (1, 2, 3, 4) if r["features"][f"cat_{c}"]), None)
        L, share = match(r["seen"], top, layers, cat)
        if r["rules"] in ("edge", "backing"):
            T = trim_in_reference(r["rules"], cat, r["extent"], layers)
            if T is not None:
                L, share = T, 1.0
        r["iou"] = share
        if cur.get("source") == "rules":
            r["label"], r["count"], r["source"] = r["rules"], r["rules_count"], "rules"
        else:
            r["label"] = L["cls"] if L else "none"
            r["count"] = int(L["count"]) if L else 0
            r["source"] = "ref"
            if q.role == "ring" and L is not None and L["cls"] == "disc":
                r["label"], r["source"] = "ring", "ref-disc"  # the head disc showing as a band: drawn alike
        if r["label"] in cur.get("exclude", ()) or r["rules"] in cur.get("exclude", ()):
            r["label"], r["source"] = "", "excluded"


def photo_ids() -> list[str]:
    return sorted(p.stem for p in (REF / "photos").glob("*.png") if (REF / "svg" / f"{p.stem}.svg").exists())


META = ("photo", "part", "role", "rules", "rules_count", "label", "count", "iou", "source")


def to_records(sid: str, rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        rec = dict(photo=sid, part=r["part"].id, role=r["part"].role, rules=r["rules"],
                   rules_count=r["rules_count"], label=r["label"], count=r["count"],
                   iou=round(float(r["iou"]), 4), source=r["source"])
        rec.update({k: round(float(v), 6) for k, v in r["features"].items()})
        out.append(rec)
    return out


def write_csv(records: list[dict], path: Path) -> None:
    keys = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(records)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str]) -> int:
    cache = None
    if "--cache" in argv:  # --cache DIR: keep detections and reference layers between runs
        k = argv.index("--cache")
        cache = Path(argv[k + 1])
        argv = argv[:k] + argv[k + 2:]
    ids = argv or photo_ids()
    records = []
    for sid in ids:
        rows = photo_layers(sid, cache)
        label_rows(sid, rows, cache)
        recs = to_records(sid, rows)
        records += recs
        print(sid, " ".join(f"{r['part']}:{r['rules']}/{r['label'] or '-'}" for r in recs), flush=True)
    if not argv:
        write_csv(records, OUT)
        from collections import Counter
        c = Counter(r["label"] for r in records if r["label"])
        print(f"{len(records)} layers, {sum(c.values())} labelled ->", OUT.relative_to(ROOT))
        for k in CLASSES:
            print(f"  {k:12} {c.get(k, 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
