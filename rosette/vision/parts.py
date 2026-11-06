"""Detect the rosette's parts from the prepared photo and its color-group labels.

detect_parts(p, labels) -> list[Part] with params in canvas units (600x850 frame,
x=300 on the axis, rosette top y=25, bottom y=825), in the keys `build` consumes.
The work is split by part type:
- parts_head: head disc, rings, points rings, center (polar label profile);
- parts_tails: tails and their stripes (horizontal label profiles).
Mask regions neither part type explains (outside the head circle, above the tails)
become 'unknown' parts, so nothing detected is silently dropped.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from rosette.model import ColorGroup, Part
from rosette.vision.parts_common import Frame, majority, r3
from rosette.vision.parts_head import detect_head, refine_center
from rosette.vision.parts_tails import detect_tails
from rosette.vision.prepare import Prepared

# Leftover pieces are first opened by this many canvas units (drops rim slivers left
# by a slightly small head circle) and kept from this area (canvas units^2).
LEFTOVER_OPEN_UNITS = 3.0
LEFTOVER_MIN_UNITS2 = 150.0
# A leftover reaching to within this many px of the tails' top row is a tail top.
TAIL_TOUCH_PX = 2
# Head circle grown by this share before leftovers are taken.
HEAD_MARGIN = 1.03
# Confidence given to unexplained regions.
UNKNOWN_CONFIDENCE = 0.2


def _leftovers(p: Prepared, labels: np.ndarray, fr: Frame) -> list[Part]:
    cx, cy, r = p.head
    h, w = p.mask.shape
    yy, xx = np.ogrid[:h, :w]
    rest = p.mask & ((xx - cx) ** 2 + (yy - cy) ** 2 > (HEAD_MARGIN * r) ** 2) & (yy < p.tails_top_y)
    k = 2 * max(1, int(round(LEFTOVER_OPEN_UNITS * fr.scale))) + 1
    rest = cv2.morphologyEx(rest.astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    comp, n = ndimage.label(rest)
    parts = []
    for i in range(1, n + 1):
        region = comp == i
        area = int(region.sum())
        if area < LEFTOVER_MIN_UNITS2 * fr.scale ** 2:
            continue
        ys, xs = np.nonzero(region)
        if ys.max() >= p.tails_top_y - TAIL_TOUCH_PX:
            continue  # the top of a tail showing beside the head: drawn by the tails
        g, _ = majority(labels[region])
        if g < 0:
            continue
        parts.append(Part(f"unknown-{len(parts)}", "unknown",
                          {"cx": r3(fr.x(xs.mean())), "cy": r3(fr.y(ys.mean())),
                           "r": r3(fr.d(np.sqrt(area / np.pi)))}, g, 60, UNKNOWN_CONFIDENCE))
    return parts


def detect_parts(p: Prepared, labels: np.ndarray, groups: list[ColorGroup] | None = None) -> list[Part]:
    """Parts of the rosette: head-side parts, tail stripes, then unexplained regions.
    With the color groups given, a glitter group found on the tails (parts_tails, H6) is
    added to them once (detect_tails gives it the same id on every call)."""
    fr = Frame.of(p)
    head = detect_head(p, labels, fr)
    pts = [q for q in head if q.role == "points"]
    if pts:  # re-center on the points' symmetry and measure again
        p = refine_center(p, labels, fr, int(pts[0].params["count"]))
        head = detect_head(p, labels, fr)
    tails, glitter = detect_tails(p, labels, fr, groups)
    if glitter is not None and groups is not None and all(g.id != glitter.id for g in groups):
        groups.append(glitter)
    parts = head + tails + _leftovers(p, labels, fr)
    seen: dict[str, int] = {}
    for q in parts:  # unique ids
        n = seen.get(q.id, 0)
        seen[q.id] = n + 1
        if n:
            q.id = f"{q.id}-{n}"
    return parts
