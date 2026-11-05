"""Part counts of a designer SVG (truth for the scoreboard and the detector tests)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from rosette.roles import category, circular_runs, classify_components, polar_mask

REF = Path(__file__).resolve().parent.parent / "reference"
# Rim depth (canvas units, 5-95 % spread of the outline radius around a smooth trend)
# at or above which a head is pinked: designer scallops measure 2.0-3.3, pixel steps
# on a smooth disc stay well under 1.
PINKED_DEPTH = 1.2


def svg_text(sid: str) -> str:
    return (REF / "svg" / f"{sid}.svg").read_text(encoding="utf-8")


def _lobes(mask: np.ndarray, cx: float, cy: float) -> int:
    """Lobe count of a points layer: most common number of runs on circles where the
    layer covers part (15-85 %) of the circle."""
    ys, xs = np.nonzero(mask)
    d = np.hypot(xs - cx, ys - cy)
    radii = np.linspace(d.min(), d.max(), 60)[1:-1]
    pol = polar_mask(mask, cx, cy, radii)
    counts = [circular_runs(row) for row, c in zip(pol, pol.mean(axis=1)) if 0.15 <= c <= 0.85]
    return int(np.bincount(counts).argmax()) if counts else 0


def _rim_depth(mask: np.ndarray) -> float:
    """5-95 % spread (px) of the outer outline's radius after removing its slow trend,
    radius taken from a least-squares circle through the outline."""
    filled = ndimage.binary_fill_holes(mask).astype(np.uint8)
    cs, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
    x, y = c[:, 0], c[:, 1]
    (a, b, k), *_ = np.linalg.lstsq(np.c_[2 * x, 2 * y, np.ones(len(x))], x * x + y * y, rcond=None)
    th, rr = np.arctan2(y - b, x - a), np.hypot(x - a, y - b)
    o = np.argsort(th)
    grid = np.linspace(-np.pi, np.pi, 2048, endpoint=False)
    rg = np.interp(grid, th[o], rr[o], period=2 * np.pi)
    hp = rg - ndimage.median_filter(rg, size=101, mode="wrap")
    return float(np.percentile(hp, 95) - np.percentile(hp, 5))


@lru_cache(maxsize=None)
def truth_from_svg(sid: str) -> dict:
    """Part counts of a designer SVG, from the role classifier.

    points: lobes of the largest points layer (points pieces united per color category,
            thin gaps such as trim lines closed);
    tails:  separate ribbons across the tails at 85 % of their height;
    rings:  distinct radial bands among ring pieces;
    pinked: the head disc's rim is zigzag (outline much longer than its convex hull).
    """
    f, items = classify_components(svg_text(sid))
    cx, cy, r = f.head
    shape = items[0][2].shape
    close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * round(3 * f.scale) + 1,) * 2)

    layers: dict[int, np.ndarray] = {}
    tails = np.zeros(shape, bool)
    head = np.zeros(shape, bool)
    bands: list[tuple[float, float]] = []
    for el, role, m in items:
        if role == "points":
            k = category(el)
            layers[k] = layers.get(k, np.zeros(shape, bool)) | m
        elif role == "tail_stripe":
            tails |= m
        elif role == "head":
            head |= m
        elif role == "ring":
            ys, xs = np.nonzero(m)
            d = np.hypot(xs - cx, ys - cy)
            bands.append((float(d.min()), float(d.max())))

    points = 0
    if layers:
        k = max(layers, key=lambda c: layers[c].sum())
        closed = cv2.morphologyEx(layers[k].astype(np.uint8), cv2.MORPH_CLOSE, close) > 0
        points = _lobes(closed, cx, cy)

    rings = 0
    end = -1.0
    for lo, hi in sorted(bands):
        if lo >= end:
            rings += 1
        end = max(end, hi)

    y = int(round(f.tails_top_y + 0.85 * (f.bottom_y - f.tails_top_y)))
    gap = max(1, int(round(2 * f.scale)))
    row = ndimage.binary_closing(tails[y], np.ones(2 * gap + 1, bool))
    n_tails = int(np.count_nonzero(row & ~np.concatenate([[False], row[:-1]])))

    pinked = False
    if head.any():
        pinked = _rim_depth(head) / f.scale >= PINKED_DEPTH
    return dict(points=points, tails=n_tails, rings=rings, pinked=bool(pinked))
