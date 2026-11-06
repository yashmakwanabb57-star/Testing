"""Photo vs designer scoreboard: the regression gate for pipeline changes.

Usage: .venv/Scripts/python.exe tools/score.py [IDs...] [--sheet DIR] [--debug]

For each canonical ID (reference/canonical.txt) with a photo: run the pipeline on
the photo, rasterize the generated SVG and the designer SVG with category colors
(solid configuration, Layer_3 hidden, outlines hidden) at the same height, crop each
to its non-white bounding box, resize the generated crop onto the designer crop
(nearest neighbor) and compare:

  color_agreement  per part region: each designer color element of the solid
                   configuration (everything outside the shop's hidden pattern layer,
                   rosette.roles.pattern_layer; old-layout files keep color parts in
                   Layer_2, Layer_2_1_1 or unnamed groups) is rendered alone in the
                   same frame, minus the elements painted after it (its visible
                   region; under MIN_ELEMENT_SHARE of the rosette area skipped); the
                   generated category holding the most pixels of that region decides
                   correct / wrong; area-weighted share correct
  pixel_agreement  share of designer-mask pixels whose category matches
  iou              mask IoU of the aligned crops
  points_ok        detected points count == designer points count
  tails_ok         detected tails count == designer tails count
  flagged          the UI asks the user to check something (pipeline.is_flagged)
  success          color_agreement >= SUCCESS_AGREEMENT and no validation issues
  uni / uok        reported only, not the metric or the bar: color_agreement and
                   success with the generated crop scaled by the height ratio alone
                   (aspect kept, centered) instead of stretched per axis; shows where
                   a miss comes from the designer drawing other proportions

A stage that crashes on a photo is recorded as an error (not success, not flagged)
and listed separately. --sheet DIR saves photo | generated | designer PNGs;
--debug prints each scored element's category, generated winner and winning share.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # run as a script
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from rosette.assign import part_area  # noqa: E402
from rosette.pipeline import analyze, is_flagged, render  # noqa: E402
from rosette.roles import (category, classify_components, classify_masks, drawn_frame,  # noqa: E402
                           element_masks, render_categories)
from tools.rasterize import DEFAULT_PALETTE, rasterize  # noqa: E402
from tools.truth import truth_from_svg  # noqa: E402

REF = ROOT / "reference"
# Rasterization height shared by both SVGs.
HEIGHT = 850
# Category agreement at or above which a rosette counts as correct.
SUCCESS_AGREEMENT = 0.9
# Bar: share of IDs that must succeed.
TARGET = 0.8
DEFAULT_COLORS = dict(DEFAULT_PALETTE)
# Elements smaller than this share of the rosette area are not scored.
MIN_ELEMENT_SHARE = 0.002
_WHITE = (255, 255, 255)
# Shape: whole-silhouette IoU at or above which the outline counts as right. Over the 76
# reference photos (HEAD 476d07e) whole IoU runs 0.762-0.980 with no gap (steps <= 0.024),
# median 0.909. The outlines the mismatch analysis (docs/analysis) calls grossly wrong
# (blobs or squares for loops, missing layers, tails spread apart: S53-19 0.762, S18-15
# 0.786, S2-15 0.798, S57-17 0.811, S36-15B 0.828) all sit below 0.85; the best rows
# reach 0.97-0.98 (S49-17, S28-15). 0.90 sits at the median, so shape_ok is a coarse
# "at least as good as the typical output", not a fine gate: read the per-ID IoU and
# its change.
SHAPE_OK_IOU = 0.90
# Head roles scored per designer element in the shape layer score.
SHAPE_HEAD_ROLES = ("head", "ring", "points", "center")


def score_ids() -> list[str]:
    lines = (REF / "canonical.txt").read_text(encoding="utf-8").splitlines()
    return [s.strip() for s in lines if s.strip() and (REF / "photos" / f"{s.strip()}.png").exists()]


def _rgb(hex_color: str) -> tuple[int, int, int]:
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))


def categories(img: np.ndarray) -> np.ndarray:
    """Per pixel category 1-4 (nearest palette color), 0 for background."""
    cats = [0] + sorted(DEFAULT_COLORS)
    pal = np.array([_WHITE] + [_rgb(DEFAULT_COLORS[c]) for c in cats[1:]], float)
    d = ((img[:, :, None, :].astype(float) - pal[None, None]) ** 2).sum(axis=3)
    return np.array(cats, np.uint8)[d.argmin(axis=2)]


def _bbox(cat: np.ndarray) -> tuple[slice, slice]:
    ys, xs = np.nonzero(cat)
    if not len(ys):
        return slice(0, 1), slice(0, 1)
    return slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)


def _align(dcat: np.ndarray, gcat: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice]]:
    """Designer crop, generated crop resized onto it (nearest neighbor), designer bbox."""
    box = _bbox(dcat)
    d = dcat[box]
    g = cv2.resize(gcat[_bbox(gcat)], (d.shape[1], d.shape[0]), interpolation=cv2.INTER_NEAREST)
    return d, g, box


def _align_uniform(dcat: np.ndarray, gcat: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice]]:
    """Reported-only alignment: the generated crop scaled by the height ratio alone
    (aspect kept) and centered on the designer crop's middle column (the axis: both
    rosettes are mirror-symmetric); columns falling outside are cut, missing ones white."""
    box = _bbox(dcat)
    d = dcat[box]
    g = gcat[_bbox(gcat)]
    gw = max(1, int(round(g.shape[1] * d.shape[0] / g.shape[0])))
    g = cv2.resize(g, (gw, d.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = np.zeros_like(d)
    off = (d.shape[1] - gw) // 2
    x0, x1 = max(0, off), min(d.shape[1], off + gw)
    out[:, x0:x1] = g[:, x0 - off:x1 - off]
    return d, out, box


def _pixel(d: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    dm, gm = d > 0, g > 0
    agreement = float((d[dm] == g[dm]).mean()) if dm.any() else 0.0
    union = (dm | gm).sum()
    return agreement, float((dm & gm).sum() / union) if union else 0.0


def align_agreement(designer: np.ndarray, generated: np.ndarray) -> tuple[float, float]:
    """(pixel_agreement, iou) of two category-colored rasters aligned by mask bbox."""
    d, g, _ = _align(categories(designer), categories(generated))
    return _pixel(d, g)


def compare(designer_svg: str, generated_svg: str) -> dict:
    """color_agreement (per part region), pixel_agreement and iou of a generated SVG
    against the designer SVG."""
    dcat = categories(render_categories(designer_svg, HEIGHT, DEFAULT_COLORS))
    gcat = categories(rasterize(generated_svg, HEIGHT, DEFAULT_COLORS))
    d, g, box = _align(dcat, gcat)
    _, g_uniform, _ = _align_uniform(dcat, gcat)
    pixel, iou = _pixel(d, g)
    min_px = MIN_ELEMENT_SHARE * (d > 0).sum()
    masks = [(el, mask[box]) for el, mask in element_masks(designer_svg, HEIGHT)]
    # visible region: an element's own mask minus everything painted after it
    # (document order is paint order: Layer_1, then the Layer_2_1 groups)
    later = np.zeros(d.shape, bool)
    visible = [None] * len(masks)
    for i in range(len(masks) - 1, -1, -1):
        visible[i] = masks[i][1] & ~later
        later |= masks[i][1]
    right = right_uniform = total = 0.0
    elements = []
    for (el, _), region in zip(masks, visible):
        area = float(region.sum())
        if area < min_px:
            continue
        counts = np.bincount(g[region], minlength=5)
        winner = int(counts.argmax())
        ok = winner == category(el)
        total += area
        right += area if ok else 0.0
        if int(np.bincount(g_uniform[region], minlength=5).argmax()) == category(el):
            right_uniform += area
        elements.append(dict(category=category(el), winner=winner, share=float(counts[winner] / area),
                             area=area, ok=ok))
    return dict(color_agreement=right / total if total else 0.0, pixel_agreement=pixel, iou=iou,
                uniform_agreement=right_uniform / total if total else 0.0, elements=elements)


def _role_masks(svg: str, box: tuple[slice, slice], shape: tuple[int, int] | None, drawn: bool = False):
    """(frame, [(role, mask)]) per classified piece of every solid-configuration color
    element, cropped to `box` of the HEIGHT render and, when `shape` is given, resized
    onto it. drawn: classify in `drawn_frame` (our files: the head center we drew)
    instead of the frame `prepare` measures on the render."""
    if drawn:
        f = drawn_frame(svg, HEIGHT)
        items = classify_masks(element_masks(svg, HEIGHT), f)
    else:
        f, items = classify_components(svg, HEIGHT)
    out = []
    for _, role, piece in items:
        crop = piece[box]
        if shape is not None:
            crop = cv2.resize(crop.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        out.append((role, crop))
    return f, out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def shape_compare(designer_svg: str, generated_svg: str) -> dict:
    """Outline agreement, colors ignored: both SVGs rendered as silhouettes (every
    color class one fill, the shop's hidden pattern layer hidden) and aligned like
    compare() (bbox crop, generated resized onto the designer crop).

      shape_iou  whole-rosette silhouette IoU
      head_iou   IoU over the rows above the designer's tail top (first row below the head)
      tail_iou   IoU over the rows from the tail top down
      layers     per designer head element piece (head, ring, points, center): IoU of
                 its region with the union of our pieces of the same role class
                 (rosette.roles classifier on both files)
      layer_iou  area-weighted mean of those IoUs (None without head pieces)
      shape_ok   shape_iou >= SHAPE_OK_IOU
    """
    colors = {k: "#000000" for k in DEFAULT_COLORS}
    dm = render_categories(designer_svg, HEIGHT, colors).min(axis=2) < 128
    gm = render_categories(generated_svg, HEIGHT, colors).min(axis=2) < 128
    box, gbox = _bbox(dm), _bbox(gm)
    d = dm[box]
    g = cv2.resize(gm[gbox].astype(np.uint8), (d.shape[1], d.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    f, dparts = _role_masks(designer_svg, box, None)
    _, gparts = _role_masks(generated_svg, gbox, d.shape, drawn=True)
    split = int(np.clip(round(f.tails_top_y) - box[0].start, 0, d.shape[0]))
    layers, total, acc = [], 0.0, 0.0
    for role, m in dparts:
        if role not in SHAPE_HEAD_ROLES:
            continue
        ours = np.zeros(d.shape, bool)
        for r, q in gparts:
            if r == role:
                ours |= q
        iou, area = _iou(m, ours), float(m.sum())
        layers.append(dict(role=role, iou=iou, area=area))
        total += area
        acc += iou * area
    whole = _iou(d, g)
    return dict(shape_iou=whole, head_iou=_iou(d[:split], g[:split]), tail_iou=_iou(d[split:], g[split:]),
                layers=layers, layer_iou=acc / total if total else None, shape_ok=whole >= SHAPE_OK_IOU)


def _detected_counts(model) -> tuple[int, int]:
    pts = [q for q in model.parts if q.role == "points"]
    points = int(max(pts, key=lambda q: part_area(q, model.parts)).params["count"]) if pts else 0
    tails = [q for q in model.parts if q.role == "tail_stripe"]
    return points, int(tails[0].params["tails_count"]) if tails else 0


def _generate(sid: str):
    model = analyze((REF / "photos" / f"{sid}.png").read_bytes())
    return model, render(model)


def _designer(sid: str) -> np.ndarray:
    return rasterize((REF / "svg" / f"{sid}.svg").read_text(encoding="utf-8"), HEIGHT, DEFAULT_COLORS)


# Keys of every scoreboard row (error rows carry None for what was not measured).
ROW_KEYS = ("id", "color_agreement", "pixel_agreement", "iou", "uniform_agreement", "success_uniform",
            "elements", "points_ok", "tails_ok",
            "points", "points_truth", "tails", "tails_truth", "flagged", "issues", "success",
            "seconds", "error", "shape_iou", "head_iou", "tail_iou", "layers", "layer_iou", "shape_ok")


def score_one(sid: str) -> dict:
    t = time.perf_counter()
    model, result = _generate(sid)
    seconds = time.perf_counter() - t
    designer = (REF / "svg" / f"{sid}.svg").read_text(encoding="utf-8")
    scores = {**compare(designer, result.svg), **shape_compare(designer, result.svg)}
    truth = truth_from_svg(sid)
    points, tails = _detected_counts(model)
    return dict(id=sid, **scores,
                points_ok=points == truth["points"], tails_ok=tails == truth["tails"],
                points=points, points_truth=truth["points"], tails=tails, tails_truth=truth["tails"],
                flagged=is_flagged(model, result.issues), issues=len(result.issues),
                success=scores["color_agreement"] >= SUCCESS_AGREEMENT and not result.issues,
                success_uniform=scores["uniform_agreement"] >= SUCCESS_AGREEMENT and not result.issues,
                seconds=seconds, error=None)


def score_all(ids: list[str] | None = None) -> list[dict]:
    rows = []
    for sid in ids or score_ids():
        try:
            rows.append(score_one(sid))
        except Exception as e:  # a crashing stage is recorded, not fatal
            row = dict.fromkeys(ROW_KEYS)
            row.update(id=sid, flagged=False, success=False, error=f"{type(e).__name__}: {e}")
            rows.append(row)
    return rows


def summary(rows: list[dict]) -> dict:
    done = [r for r in rows if not r["error"]]
    return dict(n=len(rows),
                success=sum(r["success"] for r in done),
                flagged=sum(not r["success"] and r["flagged"] for r in done),
                silent=sum(not r["success"] and not r["flagged"] for r in done),
                errors=len(rows) - len(done))


def side_by_side(sid: str) -> np.ndarray:
    """photo | generated | designer, each HEIGHT px high, category colors for the SVGs."""
    photo = Image.open(REF / "photos" / f"{sid}.png").convert("RGB")
    photo = np.asarray(photo.resize((round(photo.width * HEIGHT / photo.height), HEIGHT)))
    _, result = _generate(sid)
    gen = rasterize(result.svg, HEIGHT, DEFAULT_COLORS)
    gap = np.full((HEIGHT, 10, 3), 128, np.uint8)
    return np.concatenate([photo, gap, gen, gap, _designer(sid)], axis=1)


def _yn(v: bool) -> str:
    return "yes" if v else "no"


def main(argv: list[str]) -> int:
    sheet = None
    debug = "--debug" in argv
    argv = [a for a in argv if a != "--debug"]
    if "--sheet" in argv:
        i = argv.index("--sheet")
        sheet = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    rows = score_all(argv or None)
    print(f"{'ID':10} {'agree':>6} {'pixel':>6} {'iou':>5} {'flag':>4} {'ok':>4} {'points':>7} {'tails':>6} "
          f"{'iss':>3} {'sec':>5} {'uni':>6} {'uok':>4} {'shape':>5} {'headS':>5} {'tailS':>5} {'layer':>5} {'sok':>3}")
    for r in rows:
        if r["error"]:
            print(f"{r['id']:10} ERROR {r['error']}")
            continue
        pts = f"{r['points']}/{r['points_truth']}"
        tl = f"{r['tails']}/{r['tails_truth']}"
        print(f"{r['id']:10} {r['color_agreement']:6.3f} {r['pixel_agreement']:6.3f} {r['iou']:5.2f} "
              f"{_yn(r['flagged']):>4} "
              f"{_yn(r['success']):>4} {pts:>7} {tl:>6} {r['issues']:3d} {r['seconds']:5.1f} "
              f"{r['uniform_agreement']:6.3f} {_yn(r['success_uniform']):>4} "
              f"{r['shape_iou']:5.3f} {r['head_iou']:5.3f} {r['tail_iou']:5.3f} "
              f"{r['layer_iou'] if r['layer_iou'] is None else round(r['layer_iou'], 3)!s:>5} "
              f"{_yn(r['shape_ok']):>3}")
        if debug:  # per element: designer category, generated winner, winning share, area
            for e in r["elements"]:
                print(f"    cat {e['category']} -> {e['winner']}  share {e['share']:.2f}  "
                      f"area {e['area']:.0f}  {'ok' if e['ok'] else 'WRONG'}")
    s = summary(rows)
    print(f"success {s['success']}  flagged-not-success {s['flagged']}  "
          f"silently-wrong {s['silent']}  errors {s['errors']}")
    print(f"PASS {s['success']}/{s['n']} (target >= {TARGET:.0%})")
    done = [r for r in rows if not r["error"]]
    if done:
        print(f"shape: mean IoU {np.mean([r['shape_iou'] for r in done]):.3f}  head "
              f"{np.mean([r['head_iou'] for r in done]):.3f}  tail {np.mean([r['tail_iou'] for r in done]):.3f}  "
              f"shape_ok {sum(r['shape_ok'] for r in done)}/{len(done)} (IoU >= {SHAPE_OK_IOU})")
    uni = sum(bool(r["success_uniform"]) for r in rows if not r["error"])
    print(f"reported only (not the bar): success under uniform-scale alignment {uni}/{s['n']}")
    if sheet:
        sheet.mkdir(parents=True, exist_ok=True)
        for r in rows:
            if not r["error"]:
                Image.fromarray(side_by_side(r["id"])).save(sheet / f"{r['id']}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
