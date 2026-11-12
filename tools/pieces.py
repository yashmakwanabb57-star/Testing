"""Piece overlays and the best-IoU table of the SAM 2.1 cloth pieces.

Usage: .venv/Scripts/python.exe tools/pieces.py [--ids A,B] [--out DIR] [--model small|tiny]
                                                [--subset FILE]

Per photo: DIR/<id>.png (the prepared photo with each piece's outline in its own color and its
index at the centroid; head pieces solid, tail pieces dashed) and a row of DIR/pieces.csv:
id, pieces, head_pieces, seconds, cached, layers, covered_070, mean_best_iou.

For each head element of the reference SVG (tools.score._role_masks, roles in SHAPE_HEAD_ROLES),
cropped to the designer silhouette bbox of the HEIGHT render and resized (nearest) onto the
photo's rosette-mask bbox crop, best_iou is the maximum IoU with any head piece cropped to that
same photo bbox (the brief's alignment: it stretches the reference by the photo's tail length, so
the head is misaligned whenever the tails differ); the circle_* columns repeat the comparison with
both head circles mapped to the same square instead. covered_070 counts layers with best_iou >= COVERED_IOU. Reference SVGs are
AI-made: read the table as evidence, not truth. --subset FILE: also summarize those IDs.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # run as a script
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from rosette.perception import pieces as pcs  # noqa: E402
from rosette.perception import settings as psettings  # noqa: E402
from rosette.roles import render_categories  # noqa: E402
from rosette.vision.prepare import prepare  # noqa: E402
from tools.score import (DEFAULT_COLORS, HEIGHT, REF, SHAPE_HEAD_ROLES, _bbox, _iou,  # noqa: E402
                         _role_masks, score_ids)

COVERED_IOU = 0.70
FIELDS = ("id", "pieces", "head_pieces", "seconds", "cached", "layers", "covered_070", "mean_best_iou",
          "circle_covered_070", "circle_mean_best_iou")
# Side of the head-circle-aligned comparison (see circle_layers).
CIRCLE_SIDE = 256
_PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
            (70, 240, 240), (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40),
            (128, 0, 0), (0, 0, 128)]


def best_iou_table(ref_layers, piece_masks) -> dict:
    """Best IoU of each reference layer mask against any piece mask (all the same shape)."""
    best = [max((_iou(r, m) if (r | m).any() else 0.0 for m in piece_masks), default=0.0)
            if r.any() else 0.0 for r in ref_layers]
    return dict(layers=len(best), best=best,
                covered_070=sum(b >= COVERED_IOU for b in best),
                mean_best_iou=float(np.mean(best)) if best else 0.0)


def _crop(mask: np.ndarray, box: tuple[slice, slice]) -> np.ndarray:
    return mask[box]


def reference_layers(sid: str, shape: tuple[int, int]) -> list[np.ndarray]:
    """Head element masks of the reference SVG, bbox-cropped and resized onto `shape`."""
    svg = (REF / "svg" / f"{sid}.svg").read_text(encoding="utf-8")
    colors = {k: "#000000" for k in DEFAULT_COLORS}
    box = _bbox(render_categories(svg, HEIGHT, colors).min(axis=2) < 128)
    _, parts = _role_masks(svg, box, shape)
    return [m for role, m in parts if role in SHAPE_HEAD_ROLES]


def _square(mask: np.ndarray, head: tuple[float, float, float]) -> np.ndarray:
    """The square around the head circle (cx, cy, r) resampled (nearest) to CIRCLE_SIDE."""
    cx, cy, r = head
    k = CIRCLE_SIDE / (2.0 * r)
    m = np.array([[k, 0, -(cx - r) * k], [0, k, -(cy - r) * k]], np.float32)
    return cv2.warpAffine(mask.astype(np.uint8), m, (CIRCLE_SIDE, CIRCLE_SIDE), flags=cv2.INTER_NEAREST) > 0


def circle_layers(sid: str) -> list[np.ndarray]:
    """Reference head element masks in the square around the reference head circle (the
    frame `rosette.roles` measures), for the comparison that ignores how long the tails are."""
    svg = (REF / "svg" / f"{sid}.svg").read_text(encoding="utf-8")
    f, parts = _role_masks(svg, (slice(None), slice(None)), None)
    return [_square(m, f.head) for role, m in parts if role in SHAPE_HEAD_ROLES]


def _dashed(img: np.ndarray, contour: np.ndarray, color, dash: int = 8) -> None:
    pts = contour.reshape(-1, 2)
    for i in range(0, len(pts) - 1, 2 * dash):
        seg = pts[i:i + dash + 1].reshape(-1, 1, 2)
        if len(seg) > 1:
            cv2.polylines(img, [seg], False, color, 2)


def overlay(p, result) -> np.ndarray:
    img = p.rgb.copy()
    for i, pc in enumerate(result.pieces):
        color = _PALETTE[i % len(_PALETTE)]
        cnts, _ = cv2.findContours(pc.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            if pc.in_head:
                cv2.polylines(img, [c], True, color, 2)
            else:
                _dashed(img, c, color)
        ys, xs = np.nonzero(pc.mask)
        pos = (int(xs.mean()) - 8, int(ys.mean()) + 5)
        cv2.putText(img, str(i), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, str(i), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return img


def evaluate(sid: str, settings, out: Path | None = None) -> dict:
    p = prepare((REF / "photos" / f"{sid}.png").read_bytes())
    result = pcs.segment(p, settings)
    box = _bbox(p.mask)
    shape = p.mask[box].shape
    heads = [_crop(pc.mask, box) for pc in result.pieces if pc.in_head]
    table = best_iou_table(reference_layers(sid, shape), heads)
    circle = best_iou_table(circle_layers(sid), [_square(pc.mask, p.head) for pc in result.pieces if pc.in_head])
    if out is not None:
        Image.fromarray(overlay(p, result)).save(out / f"{sid}.png")
    return dict(id=sid, pieces=len(result.pieces), head_pieces=len(heads),
                seconds=round(result.seconds, 1), cached=int(result.cached), layers=table["layers"],
                covered_070=table["covered_070"], mean_best_iou=round(table["mean_best_iou"], 4),
                circle_covered_070=circle["covered_070"],
                circle_mean_best_iou=round(circle["mean_best_iou"], 4))


def summary_line(label: str, rows: list[dict]) -> str:
    layers = sum(r["layers"] for r in rows)
    covered = sum(r["covered_070"] for r in rows)
    mean = float(np.mean([r["mean_best_iou"] for r in rows])) if rows else 0.0
    ccov = sum(r["circle_covered_070"] for r in rows)
    cmean = float(np.mean([r["circle_mean_best_iou"] for r in rows])) if rows else 0.0
    pieces = float(np.mean([r["pieces"] for r in rows])) if rows else 0.0
    return (f"[{label}] {len(rows)} photos  layers {layers}  covered >= {COVERED_IOU:.2f}: {covered}/{layers}"
            f"  per-photo mean of mean_best_iou {mean:.3f}  mean pieces {pieces:.1f}"
            f"  | head-circle alignment: covered {ccov}/{layers}  mean {cmean:.3f}")


def main(argv: list[str]) -> int:
    def opt(name: str, default=None):
        if name in argv:
            return argv[argv.index(name) + 1]
        return default

    ids = opt("--ids")
    ids = [s for s in ids.split(",") if s] if ids else sorted(q.stem for q in (REF / "photos").glob("*.png"))
    out = Path(opt("--out", ROOT / "output" / "pieces"))
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if opt("--model"):
        env["ROSETTE_SAM2"] = opt("--model")
    settings = psettings.load(env)
    if settings.model == "off":
        print("ROSETTE_SAM2=off: nothing to evaluate")
        return 1
    rows = []
    for sid in ids:
        try:
            rows.append(evaluate(sid, settings, out))
        except Exception as e:  # recorded, not fatal
            print(f"{sid}: ERROR {type(e).__name__}: {e}", flush=True)
            continue
        r = rows[-1]
        print(f"{sid:10} pieces {r['pieces']:3d} head {r['head_pieces']:3d} {r['seconds']:6.1f}s "
              f"cached {r['cached']} layers {r['layers']:2d} cov {r['covered_070']:2d} "
              f"mean_best_iou {r['mean_best_iou']:.3f} circle {r['circle_covered_070']}/{r['layers']} "
              f"{r['circle_mean_best_iou']:.3f}", flush=True)
        with open(out / "pieces.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            w.writerows(rows)
    print(summary_line("all", rows))
    subset = opt("--subset")
    if subset:
        want = {s.strip() for s in Path(subset).read_text(encoding="utf-8").splitlines() if s.strip()}
        print(summary_line("subset", [r for r in rows if r["id"] in want]))
    for key in ("mean_best_iou", "circle_mean_best_iou"):
        low = sorted(rows, key=lambda r: r[key])[:10]
        print(f"lowest {key}: " + ", ".join(f"{r['id']} {r[key]:.3f}" for r in low))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
