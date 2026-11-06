"""Head template library: prototype, NOT wired into the pipeline (H1 analysis).

Idea: the 79 designer SVGs are the space of head shapes this designer makes. Each
designer head (every non-tail color piece, with its category, in paint order) is
extracted as a template normalized to its head circle. For a photo, the template whose
head best matches the photo's head segmentation is chosen and placed on the photo's
head circle (center and radius), replacing the detected head parts.

Matching uses classical features only, on a common GRID x GRID raster in head-radius
units (u, v in [-EXTENT, EXTENT], v down) restricted to v <= WINDOW_V (the tails cover
the lower head in photos and are not part of a template):
  partition  one-to-one mapping template category -> photo color group by overlap
             (Hungarian); share of the foreground union where the mapped label agrees
  iou        silhouette IoU
  profile    mean |outer radius difference| along rays in the window (head radii)
  lobes      lobe count of the outer radius profile (peaks over the arc, per 360 deg)
score = W_PARTITION * partition + W_IOU * iou + W_PROFILE * (1 - min(1, profile / PROFILE_SCALE))
        + W_LOBES * exp(-|lobes_photo - lobes_template| / LOBE_SCALE)

Evaluation (main): head agreement measured the way tools/score.py measures
color_agreement, but only over designer head pieces: each piece's visible region
(minus pieces painted after it, under MIN_ELEMENT_SHARE of the rosette skipped)
votes correct when the generated category holding most of its pixels equals the
designer category; area-weighted share correct. The generated raster is aligned to
the designer raster by matching head circles (center and radius; the whole-rosette
figure, reported alongside, uses score.py's bounding-box alignment). Two numberings are
reported: 'pipe' uses the pipeline's Color numbering (template categories reach a
Color through the photo group they overlap most), 'geo' takes the best relabeling of
generated categories (geometry only, numbering ignored).

Usage:
  .venv/Scripts/python.exe -m tools.head_templates [--cache DIR] [IDs...]
prints per ID (pipe/geo/pix): current head agreement, leave-one-out template (same design and its
variants removed: design_key) and full-library template, both numberings, the chosen
template, plus totals.
"""
from __future__ import annotations

import itertools
import math
import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # run as a script
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from rosette.assign import assign  # noqa: E402
from rosette.build import build  # noqa: E402
from rosette.model import RosetteModel  # noqa: E402
from rosette.roles import RENDER_H, category, classify_components, render_categories  # noqa: E402
from rosette.vision.colors import pixel_groups  # noqa: E402
from rosette.vision.parts import detect_parts  # noqa: E402
from rosette.vision.parts_common import Frame  # noqa: E402
from rosette.vision.prepare import prepare  # noqa: E402
from rosette.writer import write_svg  # noqa: E402
from tools.rasterize import DEFAULT_PALETTE, rasterize  # noqa: E402
from tools.score import MIN_ELEMENT_SHARE, categories  # noqa: E402

REF = ROOT / "reference"
# Common raster: GRID x GRID samples over [-EXTENT, EXTENT] head radii.
GRID, EXTENT = 128, 1.9
# Only rows above this many head radii below the head center are compared.
WINDOW_V = 0.55
# Roles of designer pieces that belong to the head (everything but tail stripes).
HEAD_ROLES = ("head", "ring", "points", "center", "unknown")
ALL_ROLES = HEAD_ROLES + ("tail_stripe",)
# Feature weights and scales of the template score.
W_PARTITION, W_IOU, W_PROFILE, W_LOBES = 0.4, 0.3, 0.2, 0.1
PROFILE_SCALE, LOBE_SCALE = 0.4, 2.0
# Rays sampled for the outer radius profile, and radial samples per ray.
N_RAYS, N_RHO = 180, 96
# Library files that are not products (unknown provenance / layout test).
NOT_PRODUCTS = ("TEST", "D1")
_COLORS = dict(DEFAULT_PALETTE)


def design_key(sid: str) -> str:
    """Same design across variants and sizes: S36-15B, S36-15, S36-19 -> 'S36';
    S11a-15 -> 'S11'."""
    m = re.match(r"^([A-Za-z])(\d+)", sid)
    return f"{m.group(1).upper()}{int(m.group(2))}" if m else sid.upper()


# ---------------------------------------------------------------- grids


def _grid_xy(cx: float, cy: float, r: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(-EXTENT, EXTENT, GRID)
    return cx + t[None, :] * r, cy + t[:, None] * r


def sample(img: np.ndarray, cx: float, cy: float, r: float, fill=0) -> np.ndarray:
    """Nearest-neighbour samples of img on the head grid (fill outside the image)."""
    xs, ys = _grid_xy(cx, cy, r)
    xi, yi = np.rint(xs).astype(int), np.rint(ys).astype(int)
    xi, yi = np.broadcast_to(xi, (GRID, GRID)), np.broadcast_to(yi, (GRID, GRID))
    h, w = img.shape[:2]
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.full((GRID, GRID), fill, dtype=img.dtype)
    out[ok] = img[yi[ok], xi[ok]]
    return out


def window() -> np.ndarray:
    v = np.linspace(-EXTENT, EXTENT, GRID)
    return np.broadcast_to((v <= WINDOW_V)[:, None], (GRID, GRID))


_WIN = window()


# ---------------------------------------------------------------- templates


@dataclass
class Template:
    sid: str
    key: str
    grid: np.ndarray  # (GRID, GRID) uint8 category 0..4 of the head pieces
    head: tuple[float, float, float]  # head circle in the padded RENDER_H frame
    image: np.ndarray  # padded RENDER_H render of the head pieces, categories 0..4
    full: np.ndarray  # padded RENDER_H render of the whole design, categories 0..4
    pieces: list = field(default_factory=list)  # [(category, role, visible bool mask)]
    n_categories: int = 0


def extract_template(sid: str, svg: str) -> Template:
    f, items = classify_components(svg)
    shape = items[0][2].shape if items else (1, 1)
    image = np.zeros(shape, np.uint8)
    for el, role, m in items:  # document order is paint order
        if role in HEAD_ROLES:
            image[m] = category(el)
    later = np.zeros(shape, bool)
    pieces = []
    for el, role, m in reversed(items):
        pieces.append((category(el), role, m & ~later))
        later |= m
    pieces.reverse()
    full = categories(render_categories(svg, RENDER_H, _COLORS))
    cx, cy, r = f.head
    grid = sample(image, cx, cy, r)
    return Template(sid, design_key(sid), grid, f.head, image, full, pieces,
                    len({c for c in np.unique(grid) if c}))


def library(cache: Path | None = None) -> list[Template]:
    out = []
    for p in sorted((REF / "svg").glob("*.svg")):
        if p.stem in NOT_PRODUCTS:
            continue
        cp = cache / "templates" / f"{p.stem}.pkl" if cache else None
        if cp and cp.exists():
            out.append(Template(**pickle.loads(cp.read_bytes())))
            continue
        t = extract_template(p.stem, p.read_text(encoding="utf-8"))
        if cp:
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_bytes(pickle.dumps(dict(t.__dict__)))  # plain dict: loadable from any __main__
        out.append(t)
    return out


# ---------------------------------------------------------------- features


def _outer_profile(fg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(angles, outer radius in head radii) of the rays that stay inside the window."""
    th = np.linspace(-np.pi, np.pi, N_RAYS, endpoint=False)
    rho = np.linspace(0, EXTENT, N_RHO)
    keep = np.sin(th) * EXTENT <= WINDOW_V
    th = th[keep]
    c = (GRID - 1) / 2
    k = (GRID - 1) / (2 * EXTENT)
    xs = np.rint(c + np.cos(th)[:, None] * rho[None] * k).astype(int).clip(0, GRID - 1)
    ys = np.rint(c + np.sin(th)[:, None] * rho[None] * k).astype(int).clip(0, GRID - 1)
    hit = fg[ys, xs]
    idx = np.where(hit.any(axis=1), N_RHO - 1 - np.argmax(hit[:, ::-1], axis=1), 0)
    return th, rho[idx]


def lobes(profile: np.ndarray, arc_deg: float) -> float:
    """Peaks of the (circularly smoothed) outer profile, scaled to a full turn."""
    if len(profile) < 5:
        return 0.0
    p = np.convolve(np.r_[profile[-2:], profile, profile[:2]], np.ones(5) / 5, mode="valid")
    base = float(np.median(p))
    peaks = 0
    for i in range(1, len(p) - 1):
        if p[i] >= p[i - 1] and p[i] > p[i + 1] and p[i] > base + 0.04:
            peaks += 1
    return peaks * 360.0 / max(arc_deg, 1.0)


def partition(labels: np.ndarray, tgrid: np.ndarray) -> tuple[float, dict[int, int]]:
    """Map template categories onto photo labels one-to-one by overlap (Hungarian; extra
    template categories then go to their most-overlapping label) and return the share of
    the window's foreground union where the mapped label agrees, and the map.
    labels: photo group id per cell, -1 background."""
    win = _WIN
    pf, tf = (labels >= 0) & win, (tgrid > 0) & win
    union = pf | tf
    if not union.any():
        return 0.0, {}
    cats = [int(c) for c in np.unique(tgrid[tf])]
    groups = [int(g) for g in np.unique(labels[pf])]
    if not cats or not groups:
        return 0.0, {}
    ov = np.array([[float(((tgrid == c) & (labels == g) & win).sum()) for g in groups] for c in cats])
    rows, cols = linear_sum_assignment(-ov)
    mapping = {cats[i]: groups[j] for i, j in zip(rows, cols)}
    for i, c in enumerate(cats):
        if c not in mapping:
            mapping[c] = groups[int(ov[i].argmax())]
    mapped = np.full(tgrid.shape, -2, np.int32)
    for c, g in mapping.items():
        mapped[tgrid == c] = g
    agree = (mapped == labels) & union
    return float(agree.sum() / union.sum()), mapping


def features(labels: np.ndarray, tgrid: np.ndarray) -> dict:
    win = _WIN
    pf, tf = (labels >= 0) & win, (tgrid > 0) & win
    union = (pf | tf).sum()
    iou = float((pf & tf).sum() / union) if union else 0.0
    th, rp = _outer_profile(pf)
    _, rt = _outer_profile(tf)
    arc = math.degrees(len(th) * 2 * math.pi / N_RAYS)
    part, mapping = partition(labels, tgrid)
    lp, lt = lobes(rp, arc), lobes(rt, arc)
    prof = float(np.abs(rp - rt).mean())
    total = (W_PARTITION * part + W_IOU * iou + W_PROFILE * (1 - min(1.0, prof / PROFILE_SCALE))
             + W_LOBES * math.exp(-abs(lp - lt) / LOBE_SCALE))
    return dict(score=total, partition=part, iou=iou, profile=prof, lobes_photo=lp, lobes_template=lt,
                mapping=mapping)


def choose(labels: np.ndarray, lib: list[Template], exclude_key: str | None = None) -> tuple[Template, dict]:
    best = None
    for t in lib:
        if exclude_key is not None and t.key == exclude_key:
            continue
        f = features(labels, t.grid)
        if best is None or f["score"] > best[1]["score"]:
            best = (t, f)
    return best


# ---------------------------------------------------------------- photos


def photo_record(sid: str, cache: Path | None = None) -> dict:
    """What the pipeline sees of one photo: head circle, labels, mask, frame and the
    assigned model (same steps as rosette.pipeline.analyze, keeping the labels)."""
    cp = cache / "photos" / f"{sid}.pkl" if cache else None
    if cp and cp.exists():
        return pickle.loads(cp.read_bytes())
    p = prepare((REF / "photos" / f"{sid}.png").read_bytes())
    labels, groups = pixel_groups(p)
    parts = detect_parts(p, labels)
    used = {q.color_group for q in parts}
    model = assign(RosetteModel(parts=parts, groups=[g for g in groups if g.id in used],
                                rose=any(q.role == "center" for q in parts)))
    fr = Frame.of(p)
    rec = dict(sid=sid, head=p.head, mask=np.packbits(p.mask), shape=p.mask.shape,
               labels=labels.astype(np.int16), model=model, frame=(fr.axis_x, fr.top, fr.scale),
               svg=write_svg(build(model)))
    if cp:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_bytes(pickle.dumps(rec))
    return rec


def photo_grid(rec: dict) -> np.ndarray:
    mask = np.unpackbits(rec["mask"])[: rec["shape"][0] * rec["shape"][1]].reshape(rec["shape"]).astype(bool)
    lab = np.where(mask, rec["labels"], -1).astype(np.int16)
    cx, cy, r = rec["head"]
    return sample(lab, cx, cy, r, fill=-1)


# ---------------------------------------------------------------- placement and scoring


def _render(svg: str) -> np.ndarray:
    return categories(rasterize(svg, RENDER_H, _COLORS))


def tails_only_svg(model: RosetteModel) -> str:
    m = RosetteModel(parts=[q for q in model.parts if q.role == "tail_stripe"], groups=model.groups,
                     rose=False, warnings=[])
    return write_svg(build(m))


def generated_head(rec: dict) -> tuple[float, float, float]:
    """The photo's head circle in generated-raster pixels (RENDER_H rows, 600x850 canvas)."""
    ax, top, sc = rec["frame"]
    k = RENDER_H / 850.0
    cx, cy, r = rec["head"]
    return (300 + (cx - ax) / sc) * k, (25 + (cy - top) / sc) * k, r / sc * k


def placed(rec: dict, t: Template, mapping: dict[int, int]) -> np.ndarray:
    """Generated category raster (RENDER_H rows): our tails, the template head on top,
    template categories turned into pipeline Colors through the photo groups."""
    model = rec["model"]
    cat_of = {g.id: g.category for g in model.groups}
    head_part = next((q for q in model.parts if q.role == "head"), None)
    default = cat_of.get(head_part.color_group) if head_part else None
    if default is None:
        default = min((c for c in cat_of.values() if c), default=1)
    lut = np.zeros(5, np.uint8)
    for c in range(1, 5):
        lut[c] = cat_of.get(mapping.get(c), None) or default
    img = _render(tails_only_svg(model))
    hx, hy, hr = generated_head(rec)
    tx, ty, tr = t.head
    s = tr / hr  # template px per generated px
    m = np.float32([[s, 0, tx - s * hx], [0, s, ty - s * hy]])
    warped = cv2.warpAffine(t.image, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                            borderValue=0)
    out = img.copy()
    sel = warped > 0
    out[sel] = lut[warped[sel]]
    return out


def _bbox(cat: np.ndarray):
    ys, xs = np.nonzero(cat)
    if not len(ys):
        return slice(0, 1), slice(0, 1)
    return slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)


def aligned(truth: Template, gen: np.ndarray) -> np.ndarray:
    """Generated raster resized onto the designer's bounding box (score.py alignment)."""
    box = _bbox(truth.full)
    h, w = box[0].stop - box[0].start, box[1].stop - box[1].start
    g = cv2.resize(gen[_bbox(gen)], (w, h), interpolation=cv2.INTER_NEAREST)
    out = np.zeros_like(truth.full)
    out[box] = g
    return out


def aligned_head(truth: Template, gen: np.ndarray, head: tuple[float, float, float]) -> np.ndarray:
    """Generated raster moved and scaled so its head circle lands on the designer's head
    circle: judges the head's shape apart from the rosette's overall proportions (the
    designer often draws the head larger or the tails shorter than the photo shows)."""
    hx, hy, hr = head
    tx, ty, tr = truth.head
    s = hr / tr  # generated px per designer px
    m = np.float32([[s, 0, hx - s * tx], [0, s, hy - s * ty]])
    return cv2.warpAffine(gen, m, (truth.full.shape[1], truth.full.shape[0]),
                          flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0)


def agreement(truth: Template, gen_aligned: np.ndarray, roles=HEAD_ROLES) -> tuple[float, float, float]:
    """(pipe, geo, pix): area-weighted share of designer pieces (of these roles) whose
    visible region is won by the designer category (score.py's color_agreement rule);
    geo allows the best relabeling of generated categories; pix is the stricter pixel
    share (generated category == designer category per pixel), best relabeling."""
    min_px = MIN_ELEMENT_SHARE * (truth.full > 0).sum()
    rows = []
    for cat, role, vis in truth.pieces:
        a = float(vis.sum())
        if role not in roles or a < min_px:
            continue
        rows.append((cat, a, np.bincount(gen_aligned[vis], minlength=5)))
    if not rows:
        return 0.0, 0.0, 0.0
    total = sum(a for _, a, _ in rows)
    pipe = sum(a for cat, a, cnt in rows if int(cnt.argmax()) == cat) / total
    geo = pix = 0.0
    for perm in itertools.permutations(range(1, 5)):
        lut = np.array([0, *perm])  # generated label g -> lut[g]
        right = px = 0.0
        for cat, a, cnt in rows:
            relab = np.bincount(lut, weights=cnt, minlength=5)
            right += a if int(relab.argmax()) == cat else 0.0
            px += float(relab[cat])
        geo, pix = max(geo, right / total), max(pix, px / total)
    return pipe, geo, pix


# ---------------------------------------------------------------- main


def evaluate(ids: list[str], cache: Path | None) -> list[dict]:
    lib = library(cache)
    by_sid = {t.sid: t for t in lib}
    rows = []
    for sid in ids:
        truth = by_sid.get(sid) or extract_template(sid, (REF / "svg" / f"{sid}.svg").read_text(encoding="utf-8"))
        rec = photo_record(sid, cache)
        labels = photo_grid(rec)
        head = generated_head(rec)
        cur = _render(rec["svg"])
        row = dict(id=sid, cur=agreement(truth, aligned_head(truth, cur, head)),
                   cur_all=agreement(truth, aligned(truth, cur), roles=ALL_ROLES))
        for mode, excl in (("loo", truth.key), ("full", None)):
            t, f = choose(labels, lib, excl)
            gen = placed(rec, t, f["mapping"])
            row[mode] = agreement(truth, aligned_head(truth, gen, head))
            row[mode + "_all"] = agreement(truth, aligned(truth, gen), roles=ALL_ROLES)
            row[mode + "_pick"] = t.sid
            row[mode + "_feat"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()
                                   if k != "mapping"}
        # self-match: the photo's own designer head placed with the same machinery
        own = features(labels, truth.grid)
        row["own"] = agreement(truth, aligned_head(truth, placed(rec, truth, own["mapping"]), head))
        row["own_score"] = round(own["score"], 3)
        rows.append(row)
        fmt = lambda v: "/".join(f"{x:.3f}" for x in v)  # noqa: E731
        print(f"{sid:9} cur {fmt(row['cur'])}  loo {fmt(row['loo'])} [{row['loo_pick']}]  "
              f"full {fmt(row['full'])} [{row['full_pick']}]  own {fmt(row['own'])}", flush=True)
    return rows


def main(argv: list[str]) -> int:
    cache = None
    if "--cache" in argv:
        i = argv.index("--cache")
        cache = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    ids = argv or sorted(p.stem for p in (REF / "photos").glob("*.png") if (REF / "svg" / f"{p.stem}.svg").exists())
    rows = evaluate(ids, cache)
    n = len(rows)
    for k in ("cur", "loo", "full", "own"):
        pipe, geo, pix = (np.mean([r[k][i] for r in rows]) for i in range(3))
        print(f"mean head agreement {k:4}: pipe {pipe:.3f}  geo {geo:.3f}  pix {pix:.3f}  (n={n})")
    for k in ("cur", "loo", "full"):
        print(f"mean whole-rosette agreement {k:4}: pipe {np.mean([r[k + '_all'][0] for r in rows]):.3f}  "
              f"geo {np.mean([r[k + '_all'][1] for r in rows]):.3f}")
    for i, name in ((1, "geo"), (2, "pix")):
        print(f"LOO better than current ({name}): {sum(r['loo'][i] > r['cur'][i] + 1e-9 for r in rows)}/{n}, "
              f"worse: {sum(r['loo'][i] < r['cur'][i] - 1e-9 for r in rows)}/{n}")
    print(f"full library picks own design: {sum(design_key(r['full_pick']) == design_key(r['id']) for r in rows)}/{n}")
    if cache:
        (cache / "head_templates_rows.pkl").write_bytes(pickle.dumps(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
