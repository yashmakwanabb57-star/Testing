"""Load any supported image, normalize it and localize the rosette.

prepare(bytes) -> Prepared: working-size RGB, foreground mask, symmetry axis,
head circle, where the tails start, the bottom and the px-per-canvas-unit scale.
All coordinates are in working pixels (image resized to WORK_HEIGHT rows).
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

# Every image is resized to this many rows before analysis (aspect kept).
WORK_HEIGHT = 1200
# Canvas rows the rosette's vertical extent maps to (y 25..825 of 600x850).
CANVAS_SPAN = 800.0
# Alpha value (0-255) at or above which a pixel counts as foreground.
ALPHA_CUTOFF = 128
# Share of clearly transparent pixels needed before alpha is trusted as the mask.
MIN_TRANSPARENT_FRAC = 0.01
# Width in px of the image border sampled to estimate the background color.
BORDER_PX = 4
# Minimum Lab distance from the background color that counts as foreground.
MIN_DELTA_E = 3.0
# Foreground threshold as a multiple of the border's own color spread.
NOISE_FACTOR = 3.0
# Border color spread (95th pct Lab distance) above which there is no clean background.
MAX_BG_NOISE = 12.0
# Side of the square kernel that removes isolated speckles from the raw mask.
OPEN_PX = 3
# Diameter of the elliptic kernel that closes small gaps along the outline.
CLOSE_PX = 7
# Smallest mask area, as a share of the image, accepted as a rosette.
MIN_AREA_FRAC = 0.03
# Largest mask area, as a share of the image, accepted as a rosette.
MAX_AREA_FRAC = 0.97
# Smallest mask height, as a share of the image height, accepted as a rosette.
MIN_HEIGHT_FRAC = 0.2
# Axis refinement searches this share of the image width around the centroid.
AXIS_SEARCH_FRAC = 0.05
# Step in px between axis candidates tried by mirror correlation.
AXIS_STEP = 0.5
# Downsampling factor for the coarse head-center search.
HEAD_COARSE_DS = 4
# Share of the mask height (from the top) whose widest row bounds the head size.
HEAD_PROFILE_FRAC = 0.4
# Head radius search range, as multiples of that widest half-width (refs: 0.83-1.19).
HEAD_R_MIN_RATIO, HEAD_R_MAX_RATIO = 0.75, 1.3
# Symmetry is scored in a ring from (1 - m) to (1 + m) head radii, straddling the outline.
HEAD_RING_MARGIN = 0.15

NO_ROSETTE = "No rosette found in the image."
UNREADABLE = "Could not read the image file."


class InputError(Exception):
    """The upload cannot be turned into a rosette (unreadable, or no rosette)."""


@dataclass
class Prepared:
    rgb: np.ndarray  # (H, W, 3) uint8, H == WORK_HEIGHT
    mask: np.ndarray  # (H, W) bool, rosette foreground
    axis_x: float  # vertical symmetry axis
    head: tuple[float, float, float]  # (cx, cy, r) of the head circle
    tails_top_y: float  # first row below the head
    bottom_y: float  # last rosette row
    scale: float  # px per canvas unit


def _decode(data: bytes) -> tuple[np.ndarray, np.ndarray | None]:
    """Bytes -> (RGB uint8 composited on white, alpha uint8 or None), EXIF-upright."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        img = ImageOps.exif_transpose(img)
        has_alpha = img.mode in ("RGBA", "LA", "PA", "RGBa", "La") or (
            img.mode == "P" and "transparency" in img.info)
        if has_alpha:
            rgba = np.asarray(img.convert("RGBA"))
            alpha = rgba[..., 3]
            a = alpha[..., None].astype(np.float32) / 255.0
            rgb = (rgba[..., :3].astype(np.float32) * a + 255.0 * (1.0 - a)).round().astype(np.uint8)
            if (alpha < ALPHA_CUTOFF).mean() < MIN_TRANSPARENT_FRAC:
                alpha = None
            return rgb, alpha
        return np.asarray(img.convert("RGB")), None
    except InputError:
        raise
    except Exception as exc:  # Pillow raises many types for bad input
        raise InputError(UNREADABLE) from exc


def _resize(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    if h == WORK_HEIGHT:
        return arr
    new_w = max(1, round(w * WORK_HEIGHT / h))
    interp = cv2.INTER_AREA if h > WORK_HEIGHT else cv2.INTER_CUBIC
    return cv2.resize(arr, (new_w, WORK_HEIGHT), interpolation=interp)


def _color_mask(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    border = np.zeros(rgb.shape[:2], bool)
    b = BORDER_PX
    border[:b, :] = border[-b:, :] = border[:, :b] = border[:, -b:] = True
    bg = np.median(lab[border], axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    noise = float(np.percentile(dist[border], 95))
    if noise > MAX_BG_NOISE:
        raise InputError(NO_ROSETTE)
    return dist > max(MIN_DELTA_E, NOISE_FACTOR * noise)


def _clean(raw: np.ndarray) -> np.ndarray:
    m = raw.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((OPEN_PX, OPEN_PX), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_PX, CLOSE_PX)))
    m = ndimage.binary_fill_holes(m > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        raise InputError(NO_ROSETTE)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best


def _axis(mask: np.ndarray) -> float:
    c0 = float(np.nonzero(mask)[1].mean())
    w = mask.shape[1]
    sub = mask[::2]
    total = sub.sum()
    cols = np.arange(w)
    best_c, best_s = c0, -1.0
    span = AXIS_SEARCH_FRAC * w
    for c in np.arange(c0 - span, c0 + span + 1e-9, AXIS_STEP):
        idx = np.rint(2 * c - cols).astype(int)
        ok = (idx >= 0) & (idx < w)
        mirrored = np.zeros_like(sub)
        mirrored[:, ok] = sub[:, idx[ok]]
        s = (sub & mirrored).sum() / total
        if s > best_s:
            best_c, best_s = float(c), s
    return best_c


def _rot_score(mask: np.ndarray, ax: float, cy: float, top: float) -> float:
    """Agreement of the mask with its 90-degree rotation about (ax, cy) in the upper
    part of a ring around the candidate head outline (radius cy - top)."""
    h, w = mask.shape
    r = cy - top
    r_out, r_in = (1 + HEAD_RING_MARGIN) * r, (1 - HEAD_RING_MARGIN) * r
    n = int(np.ceil(r_out))
    dy, dx = np.mgrid[-n:n + 1, -n:n + 1]
    d2 = dx * dx + dy * dy
    # upper-left quadrant only: it and its rotation (upper-right) stay clear of the tails
    inside = (d2 <= r_out * r_out) & (d2 >= r_in * r_in) & (dx <= 0) & (dy <= 0)
    dx, dy = dx[inside], dy[inside]

    def sample(px, py):
        xi, yi = np.rint(px).astype(int), np.rint(py).astype(int)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        out = np.zeros(xi.shape, bool)
        out[ok] = mask[yi[ok], xi[ok]]
        return out

    here = sample(ax + dx, cy + dy)
    rotated = sample(ax - dy, cy + dx)
    # mean of foreground and background Jaccard: neither an all-inside nor an
    # all-outside ring scores well by default
    fg = (here & rotated).sum() / max(1, (here | rotated).sum())
    bg = (~here & ~rotated).sum() / max(1, (~here | ~rotated).sum())
    return float(fg + bg) / 2


def _head(mask: np.ndarray, ax: float, top: int, bottom: int) -> tuple[float, float, float]:
    """Head center = point on the axis about which the head is most rotation-symmetric.

    Rosette heads (pleated disks, stars, bows) are rotationally symmetric while the
    tails are not, so the score peaks at the head center; the head top is the mask
    top, giving r = cy - top. The search is bounded by the widest half-width of the
    upper mask so a pointed star tip cannot pass for a small head.
    """
    upper = mask[top:top + max(1, int(HEAD_PROFILE_FRAC * (bottom - top)))]
    half_width = float(np.abs(np.nonzero(upper)[1] - ax).max())
    ds = HEAD_COARSE_DS
    small = mask[::ds, ::ds]
    lo = top + HEAD_R_MIN_RATIO * half_width
    hi = top + HEAD_R_MAX_RATIO * half_width
    coarse = [(_rot_score(small, ax / ds, c / ds, top / ds), c)
              for c in np.arange(lo, hi + 1e-9, ds)]
    c0 = max(coarse)[1]
    fine = [(_rot_score(mask, ax, c, top), c)
            for c in np.arange(max(lo, c0 - 1.5 * ds), min(hi, c0 + 1.5 * ds) + 1e-9, 1.0)]
    cy = float(max(fine)[1])
    return ax, cy, cy - top


def prepare(data: bytes) -> Prepared:
    """Decode, normalize to WORK_HEIGHT rows and localize the rosette."""
    rgb, alpha = _decode(data)
    rgb = _resize(rgb)
    if alpha is not None:
        raw = _resize(alpha) >= ALPHA_CUTOFF
    else:
        raw = _color_mask(rgb)
    mask = _clean(raw)
    h, w = mask.shape
    rows = np.flatnonzero(mask.any(axis=1))
    top, bottom = int(rows[0]), int(rows[-1])
    area = mask.mean()
    if not (MIN_AREA_FRAC <= area <= MAX_AREA_FRAC) or (bottom - top + 1) < MIN_HEIGHT_FRAC * h:
        raise InputError(NO_ROSETTE)
    ax = _axis(mask)
    head = _head(mask, ax, top, bottom)
    tails_top = min(float(bottom), float(np.floor(head[1] + head[2]) + 1))
    return Prepared(rgb=np.ascontiguousarray(rgb, dtype=np.uint8), mask=mask, axis_x=ax,
                    head=head, tails_top_y=tails_top, bottom_y=float(bottom),
                    scale=(bottom - top) / CANVAS_SPAN)


def canvas_to_px(p: Prepared, x: float, y: float) -> tuple[float, float]:
    """Canvas point (600x850 frame: x=300 on the axis, y 25..825 = mask top..bottom)
    -> working-pixel (x, y)."""
    top = float(np.flatnonzero(p.mask.any(axis=1))[0])
    return p.axis_x + (x - 300.0) * p.scale, top + (y - 25.0) * p.scale
