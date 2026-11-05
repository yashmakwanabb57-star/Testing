"""Shared helpers for part detection: pixel -> canvas frame, polar sampling, majority."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rosette.vision.prepare import Prepared

# Canvas axis x and rosette top y (600x850 frame).
CANVAS_AXIS_X, CANVAS_TOP_Y = 300.0, 25.0


@dataclass
class Frame:
    """Working pixels -> canvas units (x=300 on the axis, mask top at y=25)."""
    axis_x: float
    top: float
    scale: float  # px per canvas unit

    @classmethod
    def of(cls, p: Prepared) -> "Frame":
        top = float(np.flatnonzero(p.mask.any(axis=1))[0])
        return cls(axis_x=p.axis_x, top=top, scale=p.scale)

    def x(self, px: float) -> float:
        return CANVAS_AXIS_X + (px - self.axis_x) / self.scale

    def y(self, py: float) -> float:
        return CANVAS_TOP_Y + (py - self.top) / self.scale

    def py(self, y: float) -> float:
        return self.top + (y - CANVAS_TOP_Y) * self.scale

    def d(self, length_px: float) -> float:
        return length_px / self.scale


def polar_sample(img: np.ndarray, cx: float, cy: float, radii: np.ndarray, n_theta: int,
                 fill=-1) -> np.ndarray:
    """(len(radii), n_theta) nearest samples of img on circles around (cx, cy).
    theta_j = 2*pi*j/n_theta from +x, clockwise on screen; outside the image -> fill."""
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    xs = np.rint(cx + radii[:, None] * np.cos(th)[None]).astype(int)
    ys = np.rint(cy + radii[:, None] * np.sin(th)[None]).astype(int)
    h, w = img.shape[:2]
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    out = np.full(xs.shape, fill, dtype=img.dtype)
    out[ok] = img[ys[ok], xs[ok]]
    return out


def majority(values: np.ndarray) -> tuple[int, float]:
    """Most common non-negative label and its share among the non-negative values."""
    v = np.asarray(values).ravel()
    v = v[v >= 0]
    if v.size == 0:
        return -1, 0.0
    counts = np.bincount(v)
    g = int(counts.argmax())
    return g, float(counts[g] / v.size)


def r3(v: float) -> float:
    return round(float(v), 3)
