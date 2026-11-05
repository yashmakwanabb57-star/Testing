"""Group the rosette's pixels into a few ribbon colors and flag patterned groups.

pixel_groups(Prepared) -> (labels, groups): labels is (H, W) int32, -1 outside the
mask and a group id (0 = largest) inside; each ColorGroup carries the median color
of its plain (non-absorbed) pixels, its share of the mask, a texture score and
whether that score marks it as a pattern (glitter, printed motifs).

Pipeline:
1. Features per pixel: Gaussian-smoothed CIELAB with L* down-weighted (satin shine
   and shadow move L* far more than hue) plus a "grain" channel (fine-detail energy
   that a median filter removes: glitter sparkle; ~0 on satin and ribbon borders).
2. k-means (seeded k-means++) on a sample of flat pixels, k in 2..max_groups by the
   cluster-averaged silhouette; every mask pixel takes its nearest center.
3. A group hiding two similar ribbon colors (navy beside navy-blue, dark beside
   light orange: the silhouette prefers the coarse split) is split in two when both
   halves are large, both run down the tails and their boundary there runs along
   the ribbon (stripe edges) rather than across it (folds and shading). A group
   mixing plain satin with same-hue glitter is split by grain.
4. Small same-label blobs (motifs such as hearts, glitter sparkle) are absorbed into
   the group around them; larger print blobs (text, logos) lying wholly on the
   tails and enclosed by one group are absorbed too.
5. texture = median local L* standard deviation inside the group, measured on the
   tails (flat ribbon, unlike the pleated head) and away from absorbed print;
   pattern = texture > PATTERN_T.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from rosette.model import ColorGroup
from rosette.vision.prepare import Prepared

# Weight of L* against a*/b* in the clustering features (shading robustness).
L_WEIGHT = 0.35
# Gaussian sigma (canvas units) smoothing Lab before clustering (averages glitter grain).
SMOOTH_UNITS = 2.0
# Percentile of feature-gradient below which a pixel is 'flat' and used to fit clusters.
FLAT_PCT = 60.0
# Weight of the fine-grain (glitter) energy feature, and the median size (canvas units) it uses.
GRAIN_WEIGHT = 3.0
GRAIN_MEDIAN_UNITS = 3.5
# Mask pixels sampled for k-means.
SAMPLE_N = 20000
# Of those, points used to score the silhouette (pairwise distances: O(n^2)).
SILHOUETTE_N = 2000
# k-means restarts per k and iterations per restart.
KMEANS_RESTARTS, KMEANS_ITERS = 3, 30
# Smallest distance (feature units) allowed between two cluster centers; a k whose
# centers come closer is rejected. Measured over the 76 reference photos with
# tools/color_report.py: one shaded red ribbon (A13-15) splits into centers 7.0 apart
# and a shaded single satin with RGB noise sigma 3 into 8.8, while the closest
# genuinely different ribbons (S58-17 cream vs white) are 14.9 apart.
MIN_CENTER_SEP = 11.0
# Seed for every random choice (sampling, k-means init): same input, same output.
SEED = 0
# A same-label connected blob smaller than this (canvas units^2) is a motif/text/sparkle
# blob and is absorbed into the group surrounding it.
MOTIF_MAX_UNITS2 = 900.0
# Glitter split: a cluster is split in two when at least SPLIT_MIN_FRAC of its pixels
# have grain below SATIN_GRAIN (satin) and as many above GLITTER_GRAIN (sparkle).
# The grain feature alone leaves same-hue satin + glitter in one cluster on two
# reference photos: S18-15B red (40% satin / 43% sparkle) and S5-15 gold (38% / 47%).
# The highest min(satin, sparkle) share of any other cluster over the 76 photos is
# 0.19 (A13-15B), hence 0.28. Satin grain is 0.2-0.8, glitter 10-13 (S46), so the
# split point between the two bounds (3.75) leaves no satin edge slivers in the
# glitter half on S18-15B / S5-15 (0 components thinner than 4 px).
SATIN_GRAIN, GLITTER_GRAIN, SPLIT_MIN_FRAC = 1.5, 6.0, 0.28
# Stripe split: a cluster is split by 2-means when the halves' centers are at least
# STRIPE_SPLIT_SEP apart, each half holds STRIPE_MIN_SHARE of the flat pixels and
# STRIPE_MIN_TAIL_SHARE of the cluster's tail pixels, and on the tails the halves
# change places STRIPE_MIN_HV times as often along rows as along columns (stripe
# edges run down the tails; shading and folds do not). Tuned on the tune split of
# the canonical photos (docs/pipeline.md): merged similar ribbons measure sep 10.6-20,
# smaller half >= 0.19, tail half >= 0.35, row/column ratio >= 3.0; the closest
# single ribbons fail at least one test (sep 9.7, half 0.18 / tail half 0.18, ratio 1.0).
STRIPE_SPLIT_SEP, STRIPE_MIN_SHARE, STRIPE_MIN_TAIL_SHARE, STRIPE_MIN_HV = 10.0, 0.17, 0.3, 2.5
# Print blobs on the tails (text, logos) up to this share of the mask are absorbed when
# at least PRINT_RING_DOM of a PRINT_RING_UNITS-wide ring around them is one other group
# and at most PRINT_RING_OUT of that ring lies outside the mask.
PRINT_MAX_FRAC = 0.05
PRINT_RING_UNITS, PRINT_RING_DOM, PRINT_RING_OUT = 3.0, 0.9, 0.1
# Side (canvas units) of the window over which local L* spread is measured.
TEXTURE_WIN_UNITS = 12.0
# Group pixels closer than this (canvas units) to another group are left out of texture.
TEXTURE_EDGE_UNITS = 4.0
# Tail-interior area (canvas units^2) a group needs before its texture is measured
# on the tails only; below it the whole group interior is used.
TEXTURE_MIN_TAIL_UNITS2 = 1500.0
# texture above this marks a pattern group. Tuned on S46-15 (glitter 20.1, satin <= 2.8)
# and S35-19 (hearts 11.4, satin <= 1.2); over all 76 reference photos plain ribbon
# groups measure <= 2.83 and patterned ones >= 8.47 (table: tools/color_report.py).
PATTERN_T = 5.0
# A plain group whose tail interior has local texture above PATTERN_T on at least
# MIXED_PATTERN_SHARE of its pixels (std measured over the group's own pixels only, so
# its borders add none) may hide a patterned ribbon merged with a plain one. Over the
# 76 reference photos: S2-15B (snowflake print merged with white satin) 0.45, S58-17
# (black tail with white text) 0.56; every other plain group <= 0.39 (S70-19).
# Set from the single merged case: the margin is thin.
MIXED_PATTERN_SHARE = 0.42
# Tail-interior pixels (canvas units^2) a group needs before it is checked.
MIXED_MIN_UNITS2 = 2000.0


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def _masked_blur(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that ignores pixels outside the mask (no background bleed)."""
    m = mask.astype(np.float32)
    den = np.maximum(cv2.GaussianBlur(m, (0, 0), sigma), 1e-6)
    if img.ndim == 2:
        return cv2.GaussianBlur(img * m, (0, 0), sigma) / den
    return cv2.GaussianBlur(img * m[..., None], (0, 0), sigma) / den[..., None]


def _grain(L: np.ndarray, mask: np.ndarray, scale: float) -> np.ndarray:
    """Fine-grain energy: mean |L* - median3x3-ish(L*)| around each pixel. A median
    keeps ribbon borders and large motifs, so this is high on glitter sparkle and
    thin print strokes but ~0 on satin and along straight ribbon edges."""
    k = max(3, int(round(GRAIN_MEDIAN_UNITS * scale)) | 1)
    L8 = np.clip(L * 2.55, 0, 255).astype(np.uint8)
    hp = np.abs(L8.astype(np.float32) - cv2.medianBlur(L8, k).astype(np.float32)) / 2.55
    return _masked_blur(hp, mask, max(0.5, SMOOTH_UNITS * scale))


def _kmeans(x: np.ndarray, k: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """k-means++ with restarts; returns (centers, inertia) of the best run, or
    (None, inf) when the points hold fewer than k distinct values."""
    best_c, best_in = None, np.inf
    for _restart in range(KMEANS_RESTARTS):
        c = [x[rng.integers(len(x))]]
        d2 = ((x - c[0]) ** 2).sum(1)
        for _center in range(1, k):
            if d2.max() <= 1e-9:  # fewer distinct points than k: k is not attainable
                return None, np.inf
            c.append(x[rng.choice(len(x), p=d2 / d2.sum())])
            d2 = np.minimum(d2, ((x - c[-1]) ** 2).sum(1))
        c = np.array(c)
        for _step in range(KMEANS_ITERS):
            lab = _nearest(x, c)
            new = np.array([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        inertia = ((x - c[_nearest(x, c)]) ** 2).sum()
        if inertia < best_in:
            best_c, best_in = c, inertia
    return best_c, best_in


def _nearest(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    return ((x[:, None, :] - c[None, :, :].astype(x.dtype)) ** 2).sum(2).argmin(1)


def _silhouette(x: np.ndarray, lab: np.ndarray) -> float:
    ks = np.unique(lab)
    if len(ks) < 2:
        return -1.0
    sq = (x * x).sum(1)
    d = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * (x @ x.T), 0.0))
    mean_to = np.stack([d[:, lab == j].mean(1) for j in ks], 1)  # (n, k)
    own = np.searchsorted(ks, lab)
    counts = np.array([(lab == j).sum() for j in ks])
    n_own = counts[own]
    a = mean_to[np.arange(len(x)), own] * n_own / np.maximum(n_own - 1, 1)
    mean_to[np.arange(len(x)), own] = np.inf
    b = mean_to.min(1)
    s = np.where(n_own > 1, (b - a) / np.maximum(np.maximum(a, b), 1e-9), 0.0)
    # macro average: every cluster counts equally, so a small ribbon color
    # (a thin glitter stripe) is not outvoted by the big tails
    return float(np.mean([s[lab == j].mean() for j in ks]))


def _min_sep(c: np.ndarray) -> float:
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(2))
    d[np.eye(len(c), dtype=bool)] = np.inf
    return float(d.min())


def _cluster(feat: np.ndarray, max_groups: int) -> np.ndarray:
    """Cluster centers for the feature rows, k in 2..max_groups by silhouette among
    the k whose centers are all MIN_CENTER_SEP apart; one center when none is."""
    rng = np.random.default_rng(SEED)
    x = feat[rng.choice(len(feat), min(SAMPLE_N, len(feat)), replace=False)].astype(np.float64)
    xs_idx = rng.choice(len(x), min(SILHOUETTE_N, len(x)), replace=False)
    best_c, best_s = None, -np.inf
    for k in range(2, max(2, max_groups) + 1):
        c, _ = _kmeans(x, k, rng)
        if c is None:
            break
        if _min_sep(c) < MIN_CENTER_SEP:  # one color split by shading or noise
            continue
        s = _silhouette(x[xs_idx], _nearest(x[xs_idx], c))
        if s > best_s:
            best_c, best_s = c, s
    return best_c if best_c is not None else x.mean(0, keepdims=True)  # one color only


def _absorb(labels: np.ndarray, mask: np.ndarray, max_area: float) -> np.ndarray:
    """Relabel small same-label blobs with the label most of their pixels are nearest to
    among large blobs. Returns the boolean map of absorbed (motif) pixels."""
    small = np.zeros(labels.shape, bool)
    comp = np.zeros(labels.shape, np.int32)
    next_id = 0
    for j in np.unique(labels[mask]):
        n, cc, stats, _ = cv2.connectedComponentsWithStats((labels == j).astype(np.uint8), connectivity=8)
        tiny = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] < max_area) + 1
        is_tiny = np.isin(cc, tiny)
        small |= is_tiny
        comp[is_tiny] = cc[is_tiny] + next_id
        next_id += n
    anchor = mask & ~small
    if not small.any() or not anchor.any():
        return np.zeros(labels.shape, bool)
    _, (iy, ix) = ndimage.distance_transform_edt(~anchor, return_indices=True)
    near = labels[iy, ix]
    ys, xs = np.nonzero(small)
    c, nl = comp[ys, xs], near[ys, xs]
    # majority nearest-anchor label per small component
    nlab = int(labels.max()) + 1
    votes = np.zeros((next_id + 1, nlab), np.int64)
    np.add.at(votes, (c, nl), 1)
    winner = votes.argmax(1)
    new = winner[c]
    absorbed = np.zeros(labels.shape, bool)
    changed = new != labels[ys, xs]
    absorbed[ys[changed], xs[changed]] = True
    labels[ys, xs] = new
    return absorbed


def _split_glitter(labels: np.ndarray, mask: np.ndarray, grain: np.ndarray, max_groups: int) -> None:
    """Split a group that mixes plain satin with glitter of the same hue (red satin
    beside red glitter): both a satin-smooth and a sparkling share of its pixels."""
    for j in np.unique(labels[mask]):
        if len(np.unique(labels[mask])) >= max_groups:
            return
        region = labels == j
        v = grain[region]
        if (v < SATIN_GRAIN).mean() >= SPLIT_MIN_FRAC and (v > GLITTER_GRAIN).mean() >= SPLIT_MIN_FRAC:
            labels[region & (grain > (SATIN_GRAIN + GLITTER_GRAIN) / 2)] = int(labels.max()) + 1


def _split_stripes(centers: np.ndarray, flat_feat: np.ndarray, feat_img: np.ndarray,
                   mask: np.ndarray, tails_top: float, max_groups: int) -> np.ndarray:
    """Centers with every cluster that hides two similar stripe colors split in two
    (see STRIPE_SPLIT_SEP); one pass over the original clusters."""
    rng = np.random.default_rng(SEED)
    x = flat_feat[rng.choice(len(flat_feat), min(SAMPLE_N, len(flat_feat)), replace=False)].astype(np.float64)
    near = _nearest(x, centers)
    labels = np.full(mask.shape, -1, np.int32)
    labels[mask] = _nearest_chunked(feat_img[mask], centers)
    tails = np.zeros(mask.shape, bool)
    tails[int(np.ceil(tails_top)):] = True
    out = list(centers)
    for j in range(len(centers)):
        if len(out) >= max_groups:
            break
        xj = x[near == j]
        if len(xj) < 2:
            continue
        sub, _ = _kmeans(xj, 2, np.random.default_rng(SEED))
        if sub is None or np.linalg.norm(sub[0] - sub[1]) < STRIPE_SPLIT_SEP:
            continue
        if np.bincount(_nearest(xj, sub), minlength=2).min() < STRIPE_MIN_SHARE * len(x):
            continue
        region = (labels == j) & tails
        if not region.any():
            continue
        half = np.full(mask.shape, -1, np.int8)
        half[region] = _nearest_chunked(feat_img[region], sub)
        if np.bincount(half[region], minlength=2).min() < STRIPE_MIN_TAIL_SHARE * region.sum():
            continue
        along_rows = ((half[:, :-1] >= 0) & (half[:, 1:] >= 0) & (half[:, :-1] != half[:, 1:])).sum()
        along_cols = ((half[:-1] >= 0) & (half[1:] >= 0) & (half[:-1] != half[1:])).sum()
        if along_rows < STRIPE_MIN_HV * max(along_cols, 1):
            continue
        out[j] = sub[0].astype(centers.dtype)
        out.append(sub[1].astype(centers.dtype))
    return np.array(out)


def _absorb_prints(labels: np.ndarray, mask: np.ndarray, tails_top: float, max_px: float,
                   ring_px: int) -> np.ndarray:
    """Relabel print blobs (text, logos, paw prints) on the tails with the stripe they
    sit on. Tail stripes run down to the pinked bottom, so a blob wholly inside the
    tails and enclosed by a single other group is print, not ribbon."""
    absorbed = np.zeros(labels.shape, bool)
    kernel = np.ones((2 * ring_px + 1, 2 * ring_px + 1), np.uint8)
    for j in np.unique(labels[mask]):
        n, cc, stats, _ = cv2.connectedComponentsWithStats((labels == j).astype(np.uint8), connectivity=8)
        for c in range(1, n):
            x, y, w, h, area = stats[c]
            if y < tails_top or area > max_px:
                continue
            y0, y1 = max(0, y - ring_px), min(labels.shape[0], y + h + ring_px)
            x0, x1 = max(0, x - ring_px), min(labels.shape[1], x + w + ring_px)
            comp = (cc[y0:y1, x0:x1] == c).astype(np.uint8)
            ring = (cv2.dilate(comp, kernel) > 0) & (comp == 0)
            around = labels[y0:y1, x0:x1][ring]
            if len(around) == 0 or (around < 0).mean() > PRINT_RING_OUT:
                continue
            around = around[around >= 0]
            dom = int(np.bincount(around).argmax())
            if (around == dom).mean() >= PRINT_RING_DOM:
                sel = np.zeros(labels.shape, bool)
                sel[y0:y1, x0:x1] = comp > 0
                labels[sel] = dom
                absorbed |= sel
    return absorbed


def _local_std(L: np.ndarray, win: int) -> np.ndarray:
    mean = cv2.blur(L, (win, win))
    sq = cv2.blur(L * L, (win, win))
    return np.sqrt(np.maximum(sq - mean * mean, 0.0))


def _texture(std: np.ndarray, region: np.ndarray, tails: np.ndarray, skip: np.ndarray,
             edge: int, min_px: int) -> float:
    """Median local L* spread over the region's interior, measured on the tails when
    the group has enough tail pixels there: tails are flat ribbon, while the head's
    pleats and folds add texture of their own to plain satin. Pixels in `skip`
    (near absorbed print) are left out whenever enough others remain."""
    inner = cv2.erode(region.astype(np.uint8), np.ones((2 * edge + 1, 2 * edge + 1), np.uint8)) > 0
    if inner.sum() < 50:
        inner = region
    for use in (inner & tails & ~skip, inner & ~skip, inner & tails, inner):
        if use.sum() >= min_px:
            return float(np.median(std[use]))
    return float(np.median(std[inner]))


def _features(lab: np.ndarray, mask: np.ndarray, scale: float) -> np.ndarray:
    """(H, W, 4) clustering features: smoothed L* (down-weighted), a*, b*, grain."""
    smooth = _masked_blur(lab, mask, max(0.5, SMOOTH_UNITS * scale))
    return np.concatenate([smooth * np.array([L_WEIGHT, 1.0, 1.0], np.float32),
                           GRAIN_WEIGHT * _grain(lab[..., 0], mask, scale)[..., None]], axis=2)


def _flat(feat_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mask pixels (in mask order) away from feature edges. Clusters are fit on these
    only: blend pixels along ribbon borders would otherwise form clusters of their own."""
    gx = cv2.Sobel(feat_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(feat_img, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt((gx * gx + gy * gy).sum(2))[mask]
    return grad <= np.percentile(grad, FLAT_PCT)


def pixel_groups(p: Prepared, max_groups: int = 6) -> tuple[np.ndarray, list[ColorGroup]]:
    """Label every mask pixel with a color group; see module docstring."""
    mask = p.mask
    lab = _lab(p.rgb)
    feat_img = _features(lab, mask, p.scale)
    feat = feat_img[mask]
    flat_feat = feat[_flat(feat_img, mask)]
    centers = _cluster(flat_feat, max_groups)
    centers = _split_stripes(centers, flat_feat, feat_img, mask, p.tails_top_y, max_groups)

    labels = np.full(mask.shape, -1, np.int32)
    labels[mask] = _nearest_chunked(feat, centers)
    _split_glitter(labels, mask, feat_img[..., 3] / GRAIN_WEIGHT, max_groups)
    absorbed = _absorb(labels, mask, MOTIF_MAX_UNITS2 * p.scale * p.scale)
    printed = _absorb_prints(labels, mask, p.tails_top_y, PRINT_MAX_FRAC * mask.sum(),
                             max(1, int(round(PRINT_RING_UNITS * p.scale))))
    absorbed |= printed

    # renumber by area, largest first
    ids, counts = np.unique(labels[mask], return_counts=True)
    order = ids[np.argsort(-counts, kind="stable")]
    remap = np.full(int(labels.max()) + 2, -1, np.int32)
    remap[order] = np.arange(len(order))
    labels = np.where(mask, remap[labels], -1)

    total = float(mask.sum())
    L = lab[..., 0]
    win = max(3, int(round(TEXTURE_WIN_UNITS * p.scale)) | 1)
    std = _local_std(L, win)
    # print (text) is sparse and must not make its ribbon a pattern: keep every
    # window that sees print out of the texture measurement
    near_print = cv2.dilate(printed.astype(np.uint8), np.ones((win, win), np.uint8)) > 0
    edge = max(1, int(round(TEXTURE_EDGE_UNITS * p.scale)))
    tails = np.zeros(mask.shape, bool)
    tails[int(np.ceil(p.tails_top_y)):] = True
    min_px = int(TEXTURE_MIN_TAIL_UNITS2 * p.scale * p.scale)
    groups = []
    for g in range(len(order)):
        region = labels == g
        plain = region & ~absorbed
        px = p.rgb[plain] if plain.any() else p.rgb[region]
        rgb = tuple(int(v) for v in np.median(px, axis=0).round())
        tex = _texture(std, region, tails, near_print, edge, min_px)
        groups.append(ColorGroup(id=g, rgb=rgb, pattern=bool(tex > PATTERN_T), texture=round(tex, 3),
                                 area=float(region.sum() / total), category=None, confidence=1.0))
    return labels, groups


def mixed_texture_groups(p: Prepared, labels: np.ndarray, groups: list[ColorGroup]) -> list[int]:
    """Ids of plain groups that look partly patterned on the tails (see
    MIXED_PATTERN_SHARE): a pattern ribbon may have been merged into them."""
    L = _lab(p.rgb)[..., 0]
    win = max(3, int(round(TEXTURE_WIN_UNITS * p.scale)) | 1)
    edge = max(1, int(round(TEXTURE_EDGE_UNITS * p.scale)))
    tails = np.zeros(p.mask.shape, bool)
    tails[int(np.ceil(p.tails_top_y)):] = True
    out = []
    for g in groups:
        if g.pattern:
            continue
        region = labels == g.id
        inner = cv2.erode(region.astype(np.uint8), np.ones((2 * edge + 1, 2 * edge + 1), np.uint8)) > 0
        use = inner & tails
        if use.sum() < MIXED_MIN_UNITS2 * p.scale * p.scale:
            continue
        m = region.astype(np.float32)
        den = np.maximum(cv2.blur(m, (win, win)), 1e-6)
        mean = cv2.blur(L * m, (win, win)) / den
        sq = cv2.blur(L * L * m, (win, win)) / den
        std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
        if (std[use] > PATTERN_T).mean() >= MIXED_PATTERN_SHARE:
            out.append(g.id)
    return out


def _nearest_chunked(x: np.ndarray, c: np.ndarray, chunk: int = 200000) -> np.ndarray:
    return np.concatenate([_nearest(x[i:i + chunk], c) for i in range(0, len(x), chunk)])
