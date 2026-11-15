"""R29: the rose of a one-color rosette.

A rosette made in one ribbon color (A13-15: a red pleated head, a red rose, red ribbons)
reads as a single color group with no center: the rose is the pleats' color, so the color
profile shows no disc. The rose still has to be drawn (the rose outline and the shop's rose
toggle sit on the center), so `rose_center` finds it by its folds and adds it as a center
disc in the same color, drawn above the head with its own outline:

- the middle of the head is busy with the rose's folds (median radial edge strength
  >= ROSE_EDGE on the circles inside ROSE_MIN head radii) and the pleats outside are calm;
  the rose ends at the first radius from ROSE_MIN out where the strength falls under
  ROSE_END (A13-15: 0.39 head radii; the designer's circle 0.44).

Only for one-color models with a head disc and no center; over the 76 photos A13-15 alone.
"""
from __future__ import annotations

import cv2
import numpy as np

from rosette.model import Part
from rosette.vision.parts_common import Frame, r3
from rosette.vision.prepare import Prepared

ROSE_EDGE, ROSE_END, ROSE_MIN = 10.0, 8.0, 0.2
# Confidence of the found rose (below the flag bar: the user confirms it).
ROSE_CONF = 0.5
ONE_COLOR_INFO = "One ribbon color: the rose is drawn as a center circle in the same color - check its size."


def rose_radius(p: Prepared) -> float | None:
    """Radius (px) of a rose in the head's middle by its fold edges, or None."""
    g = cv2.GaussianBlur(cv2.cvtColor(p.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 2)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    cx, cy, r = p.head
    th = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    vis = ~((th > np.radians(55)) & (th < np.radians(125)))  # the tails show below
    h, w = g.shape
    fracs = np.arange(0.05, 0.8, 0.0125)
    prof = []
    for f in fracs:
        xs = np.clip(np.rint(cx + f * r * np.cos(th)).astype(int), 0, w - 1)
        ys = np.clip(np.rint(cy + f * r * np.sin(th)).astype(int), 0, h - 1)
        prof.append(float(np.median(np.abs(gx[ys, xs] * np.cos(th) + gy[ys, xs] * np.sin(th))[vis])))
    prof = np.array(prof)
    inner = fracs < ROSE_MIN
    if not inner.any() or np.median(prof[inner]) < ROSE_EDGE:
        return None
    end = next((f for f, v in zip(fracs, prof) if f >= ROSE_MIN and v < ROSE_END), None)
    return None if end is None else float(end * r)


def rose_center(p: Prepared, parts: list[Part]) -> list[Part] | None:
    """R29: the parts with the rose of a one-color rosette added as a same-color center, or
    None when the reading does not apply (module doc)."""
    if len({q.color_group for q in parts}) != 1 or any(q.role == "center" for q in parts):
        return None
    head = next((q for q in parts if q.role == "head"), None)
    if head is None:
        return None
    rr = rose_radius(p)
    if rr is None:
        return None
    fr = Frame.of(p)
    c = Part("center", "center", {"cx": head.params["cx"], "cy": head.params["cy"], "r": r3(fr.d(rr))},
             head.color_group, 50, ROSE_CONF)
    return list(parts) + [c]
