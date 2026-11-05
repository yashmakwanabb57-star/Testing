"""Shared helpers for part detection: pixel -> canvas frame, polar sampling, majority."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from rosette.vision.prepare import Prepared

# Canvas axis x and rosette top y (600x850 frame).
CANVAS_AXIS_X, CANVAS_TOP_Y = 300.0, 25.0
# Star tips are as small as motif blobs (S40-15 black tips 740-1250 units^2, S35-19
# white tips ~840, A15-17 ~350) but are not absorbed when they form a ring: at least
# RING_MIN_BLOBS same-label blobs of RING_MIN_UNITS2 .. RING_MAX_UNITS2 inside the head
# circle, within RING_RADIUS_TOL head radii of each other's distance from the head
# center, areas within RING_AREA_RATIO of their median, evenly spread around it
# (|mean exp(i N theta)| >= RING_EVEN for N = their count or up to two more, hidden
# behind the tails). Measured over the 76 photos (constants as here): rings found on
# 33 photos, motif-sized blobs kept on 10 (A15-17, S17-15, S33-15, S35-19,
# S40-15, S45-15, S52-19, S53-19, S70-19, s44-15), all in even rings inside the head
# circle (not checked one by one against the photos; tail motifs lie below the head,
# head motifs scatter at many radii). The evenness margin is thin on S40-15 (0.85; its head center
# is off by a few units): RING_EVEN sits below it at 0.8.
RING_MIN_UNITS2, RING_MAX_UNITS2 = 250.0, 3600.0
RING_MIN_BLOBS, RING_RADIUS_TOL, RING_AREA_RATIO, RING_EVEN = 4, 0.15, 2.5, 0.8
# |mean exp(i N theta)| is also high for a few blobs bunched on one side (4 blobs within
# 24 degrees: 0.82), so the blobs must also go round: the largest angular gap between
# neighbours is at most RING_MAX_GAP steps of 360/N (3 = two blobs hidden side by side
# behind the tails, plus 10 % slack for hand-made spacing).
RING_MAX_GAP = 3.3


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


@dataclass
class BlobRing:
    """Blobs evenly spread around a center: their pixels, the visible count, the
    order (count including up to two blobs hidden behind the tails), evenness
    |mean exp(i order theta)| and the rotation (deg, clockwise from +x on screen) of
    the first blob's direction."""
    pixels: np.ndarray
    visible: int
    order: int
    evenness: float
    rotation_deg: float


def _max_gap_deg(theta: np.ndarray) -> float:
    """Largest angle (deg) between neighbouring directions around the circle."""
    t = np.sort(np.degrees(theta) % 360.0)
    return float(np.max(np.diff(np.concatenate([t, [t[0] + 360.0]]))))


def blob_rings(binary: np.ndarray, head: tuple[float, float, float], scale: float) -> list[BlobRing]:
    """Rings of blobs of `binary` around the head center (see RING_MIN_BLOBS): blobs
    of RING_MIN_UNITS2..RING_MAX_UNITS2 inside the head circle, grouped from the
    innermost out by distance from the center, kept when their areas agree and they
    are evenly spread around it."""
    cx, cy, r = head
    n, cc, stats, cen = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    a = stats[1:, cv2.CC_STAT_AREA] / scale ** 2
    d = np.hypot(cen[1:, 0] - cx, cen[1:, 1] - cy)
    th = np.arctan2(cen[1:, 1] - cy, cen[1:, 0] - cx)
    idx = np.flatnonzero((a >= RING_MIN_UNITS2) & (a < RING_MAX_UNITS2) & (d < r))
    idx = idx[np.argsort(d[idx], kind="stable")]
    used: set[int] = set()
    out = []
    for i in idx:
        if i in used:
            continue
        grp = [k for k in idx if k not in used and abs(d[k] - d[i]) <= RING_RADIUS_TOL * r]
        med = float(np.median(a[grp]))
        grp = [k for k in grp if med / RING_AREA_RATIO <= a[k] <= med * RING_AREA_RATIO]
        if len(grp) < RING_MIN_BLOBS:
            continue
        order, even = max(((m, float(abs(np.exp(1j * m * th[grp]).mean())))
                           for m in range(len(grp), len(grp) + 3)), key=lambda t: t[1])
        if even < RING_EVEN or _max_gap_deg(th[grp]) > RING_MAX_GAP * 360.0 / order:
            continue
        used.update(int(k) for k in grp)
        rot = float(np.degrees(np.angle(np.exp(1j * order * th[grp]).mean())) / order) % (360.0 / order)
        out.append(BlobRing(np.isin(cc, np.array(grp) + 1), len(grp), order, even, rot))
    return out
