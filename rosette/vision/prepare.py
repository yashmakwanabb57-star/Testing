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
# Tails start (H10): the tails' outer side edges are followed upward from this share of the
# mask height (below every head of the 76 photos, whose tails start at 0.27-0.44).
TAILS_TRACK_FROM = 0.7
# Each row's edges are predicted by lines fitted to TAILS_FIT_ROWS rows starting
# TAILS_GAP_ROWS below it. On the 76 photos' tails (rows 0.55-0.85 of the height) such a
# prediction misses by at most 2.2 px (median 1.0) and the edges slope at most 0.27 px/row.
TAILS_FIT_ROWS, TAILS_GAP_ROWS = 60, 40
# An edge off its prediction by more than this many px, on TAILS_RUN_ROWS consecutive rows,
# is no longer a tail edge (about 3x the worst tail miss).
TAILS_EDGE_TOL, TAILS_RUN_ROWS = 6.0, 3
# Head fit: hull vertices this share of the head height above the tails start are dropped
# (the cut through the tails is not head outline; about 12 px on the four takeover photos,
# which clears the cut's corner vertices and the pixel steps next to them).
FIT_CUT_MARGIN = 0.03
# A hull vertex weighs half the angle to its neighbours, each gap capped at this many
# degrees (a gap is a long hull chord: a loop or the tails, not outline). 15 is half the
# 30-degree tip spacing of a 12-point star, so its tips keep their full weight while a
# long loop's two chords add at most 15 degrees each.
FIT_GAP_CAP_DEG = 15.0
# A vertex supports a circle when its distance is within this share of the radius (on the
# four takeover photos 90 % of the hull vertices lie within 2.8-4.0 % of the fit).
FIT_TOL = 0.04
# The fit's coverage counts the angle between consecutive on-circle vertices up to this
# many degrees (the tips of a 6-point star are 60 apart).
FIT_COVER_GAP_DEG = 60.0
# The fitted circle replaces the symmetry circle when its radius is at least this
# multiple of the symmetry radius: the symmetry score has locked onto an inner structure
# (S29-15's star arms: 1.37). Heads whose two radii agree keep the symmetry circle.
FIT_TAKEOVER = 1.25
# ... and only when its coverage (`_coverage`) is at least this share of a full turn
# (the takeovers on the 76 photos and reference renders: 0.67-0.94; broken fits through
# a 6-point star's arms, s34-15 photo and S80-22 reference: 0.31 and 0.29).
FIT_MIN_COVERAGE = 0.5

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
    fit_ratio: float = 0.0  # fitted / symmetry head radius (0: no fit, no tails start found)
    fit_coverage: float = 0.0  # the fit's coverage (`_coverage`)


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


def _tails_start(mask: np.ndarray, top: int, bottom: int) -> int | None:
    """First row (scanning up from the bottom) of the tails: above it the silhouette's
    outer side edges stop running on straight lines.

    Tail edges are near-straight with slowly changing width; where the head starts,
    one edge jumps out (head or star arms wider than the tails) or in (tails wider than
    the head) within a few rows. Each row is tested against lines fitted to tail rows
    TAILS_GAP_ROWS below it, so a gradual curve cannot drag the lines along.
    None when no such row exists or the tails are too short to follow.
    """
    ys = np.arange(top, bottom + 1)
    sub = mask[top:bottom + 1]
    has = sub.any(axis=1)
    left = np.where(has, sub.argmax(axis=1), np.nan)
    right = np.where(has, sub.shape[1] - 1 - sub[:, ::-1].argmax(axis=1), np.nan)
    i0 = int(TAILS_TRACK_FROM * (bottom - top))
    if i0 + TAILS_GAP_ROWS + TAILS_FIT_ROWS > len(ys):
        return None
    run = 0
    for i in range(i0 - 1, -1, -1):
        w = slice(i + TAILS_GAP_ROWS, i + TAILS_GAP_ROWS + TAILS_FIT_ROWS)
        yy = ys[w].astype(float)
        if np.isnan(left[w]).any() or np.isnan(left[i]):
            return None
        a, b = np.polyfit(yy, left[w], 1)
        c, d = np.polyfit(yy, right[w], 1)
        off = (abs(left[i] - (a * ys[i] + b)) > TAILS_EDGE_TOL
               or abs(right[i] - (c * ys[i] + d)) > TAILS_EDGE_TOL)
        run = run + 1 if off else 0
        if run >= TAILS_RUN_ROWS:
            return int(ys[i + TAILS_RUN_ROWS - 1]) + 1
    return None


def _fit_head(mask: np.ndarray, ax: float, top: int, tails: int) -> tuple[float, float, float]:
    """(cy, r, coverage) of the circle (center on the axis) fitted to the head's outer
    boundary: the convex hull of the silhouette above the tails start.

    The hull bridges a star's concavities, so its vertices are the tips (the circle
    around the star). Each vertex votes with the angle it spans around the candidate
    center; a long loop is one vertex with capped gaps, an outlier the vote ignores. For
    each center row the best radius is the one whose FIT_TOL band collects the most
    angle. The third value is the fit's coverage (`_coverage`).
    """
    cnts, _ = cv2.findContours(mask[:tails].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(np.vstack([c[:, 0, :] for c in cnts]))[:, 0, :].astype(float)
    keep = hull[:, 1] < tails - FIT_CUT_MARGIN * (tails - top)
    x, y = hull[keep, 0] - ax, hull[keep, 1]
    if len(x) < 3:
        return float(tails), 0.0, 0.0
    cap = np.radians(FIT_GAP_CAP_DEG)

    def weights(cy: float) -> np.ndarray:
        th = np.arctan2(y - cy, x)
        o = np.argsort(th)
        gaps = np.minimum(np.diff(np.concatenate([th[o], th[o][:1] + 2 * np.pi])), cap)
        wt = np.empty_like(th)
        wt[o] = 0.5 * (gaps + np.roll(gaps, 1))
        return wt

    def vote(cy: float) -> tuple[float, float]:
        d = np.hypot(x, y - cy)
        rs = np.arange(max(1.0, d.min()), d.max() + 1.0, 1.0)
        inside = np.abs(d[None, :] - rs[:, None]) <= FIT_TOL * rs[:, None]
        score = inside.astype(float) @ weights(cy)
        j = int(np.argmax(score))
        return float(score[j]) / (2 * np.pi), float(rs[j])

    lo = top + 0.2 * (tails - top)
    (_, r), cy = max(((vote(c), c) for c in np.arange(lo, tails, 2.0)), key=lambda t: t[0][0])
    # the vote's band leaves the center a few px loose: least squares on its inliers
    # (x^2 + y^2 = 2 cy y + (r^2 - cy^2), weighted by angle), inliers re-chosen each pass
    for _ in range(3):
        wt = weights(cy)
        inl = np.abs(np.hypot(x, y - cy) - r) <= FIT_TOL * r
        if inl.sum() < 3:
            break
        sw = np.sqrt(wt[inl])
        a = np.stack([2 * y[inl], np.ones(int(inl.sum()))], 1) * sw[:, None]
        sol = np.linalg.lstsq(a, (x[inl] ** 2 + y[inl] ** 2) * sw, rcond=None)[0]
        if sol[1] + sol[0] ** 2 <= 0:
            break
        cy, r = float(sol[0]), float(np.sqrt(sol[1] + sol[0] ** 2))
    return float(cy), float(r), _coverage(x, y, cy, r)


def _coverage(x: np.ndarray, y: np.ndarray, cy: float, r: float) -> float:
    """Share of a full turn around (0, cy) spanned by consecutive hull vertices that both
    lie on the circle (within FIT_TOL), each gap counted up to FIT_COVER_GAP_DEG: a chord
    between two star tips counts, the gap over a loop or the tails does not."""
    th = np.arctan2(y - cy, x)
    o = np.argsort(th)
    th = th[o]
    on = (np.abs(np.hypot(x, y - cy) - r) <= FIT_TOL * r)[o]
    gaps = np.diff(np.concatenate([th, th[:1] + 2 * np.pi]))
    both = on & np.roll(on, -1)
    return float(np.minimum(gaps, np.radians(FIT_COVER_GAP_DEG))[both].sum() / (2 * np.pi))


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
    tails = _tails_start(mask, top, bottom)
    ratio = coverage = 0.0
    if tails is not None:
        cy, r, coverage = _fit_head(mask, ax, top, tails)
        ratio = r / head[2] if head[2] > 0 else 0.0
        if r >= FIT_TAKEOVER * head[2] and coverage >= FIT_MIN_COVERAGE:
            head = (ax, cy, r)
    tails_top = min(float(bottom), float(np.floor(head[1] + head[2]) + 1))
    return Prepared(rgb=np.ascontiguousarray(rgb, dtype=np.uint8), mask=mask, axis_x=ax,
                    head=head, tails_top_y=tails_top, bottom_y=float(bottom),
                    scale=(bottom - top) / CANVAS_SPAN, fit_ratio=ratio, fit_coverage=coverage)


def canvas_to_px(p: Prepared, x: float, y: float) -> tuple[float, float]:
    """Canvas point (600x850 frame: x=300 on the axis, y 25..825 = mask top..bottom)
    -> working-pixel (x, y)."""
    top = float(np.flatnonzero(p.mask.any(axis=1))[0])
    return p.axis_x + (x - 300.0) * p.scale, top + (y - 25.0) * p.scale
