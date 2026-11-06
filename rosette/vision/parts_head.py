"""Head-side part detection: head disc (pinked or smooth), rings, points rings, center.

Everything is read off a polar unwrap of the color-group labels around the head
center (radius step ~1 canvas unit, RING_THETA angles), leaving out the sector
straight below the center where the tails show:
- radial profile: a radius where one label covers >= FULL_COV of the circle is 'full';
  contiguous full radii of one label form a band. The band at the middle is the
  center disc, the outermost band of the head label is the head disc, the others
  are rings (inner ones drawn on top).
- the radii between bands are 'zones'; each label in a zone is a candidate points
  ring. Its count N is the symmetry order of its angular indicator: the largest N
  for which the indicator, turned by every multiple of 360/N, still correlates with
  itself (>= SYM_T). An FFT peak needs the full circle, which the tails sector breaks;
  the turned self-correlation only uses angles valid in both copies. Rotation is the
  phase of the N-th harmonic over the valid angles, kind comes from
  the lobe width profile across the zone (bow: wide at both ends, star: widest near
  its inner end and narrowing to the tip, diamond/petal: widest in the middle, petal
  the fuller of the two; a petal that keeps its width over most of its length and
  fills much of the spacing is the designer's loop block, a bow: LOOP_*). Geometry
  draws each kind with the designer's mined width profile (rosette.geometry.points);
  a bow running under a ring or the center starts where that cover ends.
  A same-label band just inside a layer whose lobes still cover most of the circle
  next to it is that layer's body: the layer becomes a star covering it.
  A layer without rotational symmetry whose label still covers the full circle over
  part of its zone (a band dropped as a bridge: hand-pleated edges make its coverage
  ramp up and down slowly) is a concentric pleated layer: a ring (the designer draws
  these as annuli), with a confidence from how well the ring explains the layer.
- the head disc exists when the outline of the upper half is a closed circle
  (radius spread < HEAD_ROUND); its label is the majority just inside the outline,
  and it is pinked when the outline, freed of its slow trend, still varies by
  PINKED_DEPTH units or more (designer scallops: 2-3.3 units).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage

from rosette.model import Part
from rosette.vision.parts_common import Frame, blob_rings, majority, polar_sample, r3
from rosette.vision.prepare import Prepared

# Radial sampling step (canvas units) and angular samples per circle.
RADIAL_STEP_UNITS = 1.0
RING_THETA = 720
# Coverage of one label at which a circle counts as full (a band radius).
FULL_COV = 0.8
# Bands thinner than this (canvas units) are noise or edge teeth, not parts.
MIN_BAND_UNITS = 3.0
# A band whose label covers >= BRIDGE_COV of the circles within BRIDGE_REACH_UNITS on
# both sides is where one points layer's lobes touch, not a ring.
BRIDGE_COV, BRIDGE_REACH_UNITS = 0.4, 3.0
# A same-label band just inside a layer is the layer's body (drawn as a star) when
# the layer still covers >= BODY_COV of the circles within BODY_UNITS outside it;
# a ring merely touched by its own color's trims falls far below that.
BODY_UNITS, BODY_COV = 5.0, 0.5
# A ring band is drawn out to where its label stops being the dominant one with at
# least RING_EXTEND_COV of the circle (a pleated ring's teeth and a center a few units
# off keep its outer part between 0.5 and FULL_COV), but not into another band nor
# past the inner radius of a points layer of the same label. Chosen from 0.5 / 0.6 / 0.7
# over the 34 scored IDs: S30-15 needs <= 0.6 (0.878 -> 0.972); at 0.5 S40-15's head ring
# grows to the rim and it drops 0.903 -> 0.900; at 0.6 A15-17 0.976 -> 0.949 and S43-15
# 1.00 -> 0.96 (rings now overlap the loops' inner ends, still correct).
RING_EXTEND_COV = 0.6
# Growth is at most RING_MAX_GROWTH_UNITS (a pleat's depth; far beyond it the ring
# would be a guess) and stops at another label's points layer once that label covers
# RING_STOP_COV of the circle (its lobes show there, the ring must not hide them).
RING_MAX_GROWTH_UNITS, RING_STOP_COV = 8.0, 0.4
# A pleated layer read as a ring (no rotational symmetry, a full band dropped as a
# bridge; see _pleated_ring) keeps its fit as confidence only when it looks like an
# annulus: the label covers the full circle (FULL_COV) on at least PLEATED_FULL_MIN of the
# ring's radii, and its fringe beyond the ring on either side (until its coverage falls
# under EXTENT_COV) reaches at most PLEATED_REACH_UNITS. Otherwise the confidence is
# capped at BLOB_UNCONFIRMED_CONF (flagged). Over the 76 photos the 16 layers this rule
# reads, checked against the designer files: the annuli (A7-15, A9-15, A16-17, S13-15,
# S15-15B, S18-15, S18-15B; plus the A1-15 badge rim, which the designer draws as the
# center disc: no star or loop) have full share 0.478-0.828 and
# fringe reach <= 12 units; the S55-15 loops have full share 0.338 (the loops overlap
# only near their base), the star layers of S11-15 and S40-15 reach 18 units.
PLEATED_FULL_MIN, PLEATED_REACH_UNITS = 0.4, 15.0
# Zones thinner than this (canvas units) are band edges (teeth, polygon corners).
MIN_ZONE_UNITS = 10.0
# Mean zone coverage a label needs to be a candidate points layer.
MIN_LAYER_COV = 0.01
# Coverage below which a radius does not belong to a layer's radial extent.
EXTENT_COV = 0.02
# Highest points count reported.
MAX_COUNT = 24
# Rotational self-correlation that fixes the symmetry order; below LOW_SYM a layer is unknown.
SYM_T, LOW_SYM = 0.6, 0.4
# Half-angle (deg) of the sector straight below the center left out of the head
# analysis: the tails show there between and below the head's parts.
TAIL_SECTOR_DEG = 35.0
# Lobe width profile (shares of the widest lobe width): a bow is wide at both ends; a
# star is widest in its inner STAR_PEAK of the radial extent and narrow at the tips.
BOW_ENDS = 0.5
STAR_PEAK, STAR_OUT = 0.35, 0.5
PETAL_FULLNESS = 0.6
# A petal-shaped lobe is the designer's loop block (H4) when it keeps >= LOOP_KEEP of
# its widest width over >= LOOP_PLATEAU of the 9 radii at 0.1..0.9 of its extent and
# covers >= LOOP_SHARE of the angular spacing at mid radius. Photo loops read as pointed
# ovals: the pleated ring hides their base and the folded loop end narrows the tip,
# but in between they keep their width (a sine lobe keeps 0.85 over 3 of 9 radii).
# Over the 76 photos (petal layers matched to the designer layer of the same Color
# and count): the 7 designer loops read as petals (S26-15, S31-15, S35-19, S38-15,
# S46-15, S56-17, S70-19) have plateau 5-6 of 9 and share 0.52-0.71; of the 21 other
# petal layers one passes (s39-15, a designer kite: 5 of 9, share 0.59).
LOOP_KEEP, LOOP_PLATEAU, LOOP_SHARE = 0.85, 5 / 9, 0.45
# A layer of label h read with low confidence (unknown, or points under EXPLAIN_MAX_CONF)
# may be another layer's trim (H4):
# - edge strips: a band of h right along both side edges of a loop (bow), at most
#   EDGE_MAX_SHARE of the loop's angular width wide and followed by a third label (h
#   filling the whole gap between loops is not a band), on >= EDGE_MIN_RADII of the
#   radii in the middle of the loop's extent, on both sides of >= EXPLAIN_MIN_LOBES of
#   the visible loops. Drawn as the designer does: two strips per loop, EDGE_SHARE of
#   the loop's widest width (median strip / loop width over the 14 strip layers on
#   loops in 13 designer files, quartiles 0.107-0.119);
# - backing tips: h shows on the axis of >= EXPLAIN_MIN_LOBES of another layer's
#   visible points (within BACK_TOL_DEG), continuing beyond their tips by
#   >= BACK_MIN_UNITS while covering under BACK_MAX_COV of the circle there (a band of h
#   outside the points is a ring, not tips). The designer draws backings as a frame or
#   tip behind each point (hollow frames in 9 of the 11 files with one; never a disc
#   under everything): drawn as a copy of the points reaching the measured tip.
# The layer is replaced by its trims when they cover >= EXPLAIN_MIN_SHARE of its pixels.
EXPLAIN_MAX_CONF = 0.6
EDGE_MAX_SHARE, EDGE_MIN_RADII, EDGE_SHARE = 0.35, 0.5, 0.108
BACK_TOL_DEG, BACK_MIN_UNITS, BACK_MAX_COV = 2.0, 3.0, 0.5
EXPLAIN_MIN_LOBES, EXPLAIN_MIN_SHARE = 2 / 3, 0.5
# An unknown head-zone layer in a seam - a zone between two bands at most SEAM_UNITS
# wide (two pleat depths, RING_MAX_GROWTH_UNITS: the center's pinked edge meets a ring's
# pleats) - whose label is one of those bands' labels and whose area is under
# SPECK_MAX_UNITS2 is that band's fringe (H4): it is folded into the band instead of
# being drawn as a disc, and a seam left empty is closed (the outer band, a ring,
# starts where the inner one ends). Over the 76 photos 14 seams (11-16 units wide) hold
# 22 such unknowns of 1525-5729 units^2 (S35-19: 1525 and 1943), all folded. The next
# wider zones (S17-15 and S1-15B 21 units, S52-19 and S57-17 28) hold real layers in
# the designer files (S17-15: a 4-point star) and are kept, as are unknowns of a third
# color (a badge's print, a gold rim) and zones open on one side (S5-15's loops).
SEAM_UNITS = 2 * RING_MAX_GROWTH_UNITS
SPECK_MAX_UNITS2 = 6000.0
# A points layer (not a loop or star) whose inner end lies within COVER_GAP_UNITS outside
# a ring or the center started under it (one pleat depth, RING_MAX_GROWTH_UNITS).
COVER_GAP_UNITS = 8.0
# Confidence of a points layer read from its blobs (below assign.FLAG_BELOW, so it is
# flagged) unless the layer's zone pixels are also rotationally symmetric at the blob
# count (score >= LOW_SYM) and no blob had to be assumed hidden behind the tails. The
# blob reading is only tried when that symmetry is weak, so in practice every
# blob-read layer is flagged.
BLOB_UNCONFIRMED_CONF = 0.5
# Head outline: angles sampled, search radius (head radii), inner start of the scan.
RIM_THETA = 1440
RIM_MAX, RIM_START = 1.2, 0.6
# Upper-half outline radius spread (5-95 %, head radii) below which the head is a disc.
HEAD_ROUND = 0.12
# Outline detrend window (canvas units of arc) and pinked threshold (canvas units).
RIM_TREND_UNITS = 40.0
PINKED_DEPTH = 1.2
# Center search: +- units along the axis, over radii RECENTER_FROM..RECENTER_TO head
# radii sampled every RECENTER_RADIAL_STEP units.
RECENTER_UNITS = 12.0
RECENTER_FROM, RECENTER_TO, RECENTER_RADIAL_STEP = 0.3, 0.95, 2.0
# Head label is sampled this far (canvas units) inside the outline.
RIM_LABEL_UNITS = 6.0
# A head band must reach this share of the head radius.
HEAD_BAND_EDGE = 0.8
# A band starting within this many canvas units of the middle is the center disc;
# one that also reaches HEAD_BAND_EDGE is a plain head disc.
CENTER_START_UNITS = 3.0


@dataclass
class _Band:
    i0: int
    i1: int  # exclusive
    label: int
    cov: float


def _rim(p: Prepared, fr: Frame) -> tuple[np.ndarray, np.ndarray]:
    """(theta, outline radius px) over the upper half of the head."""
    cx, cy, r = p.head
    step = max(0.5, fr.scale * RADIAL_STEP_UNITS / 2)
    radii = np.arange(RIM_START * r, RIM_MAX * r, step)
    pol = polar_sample(p.mask.astype(np.int8), cx, cy, radii, RIM_THETA, fill=0) > 0
    th = np.linspace(0, 2 * np.pi, RIM_THETA, endpoint=False)
    upper = th > np.pi  # y < cy on screen
    first_off = np.where(~pol, np.arange(len(radii))[:, None], len(radii)).min(axis=0)
    rim = np.where(first_off > 0, radii[np.maximum(first_off - 1, 0)], RIM_START * r)
    return th[upper], rim[upper]


def _rot_score(pol: np.ndarray, win: np.ndarray, n: int, n_groups: int) -> float:
    """Mean self-correlation of the label indicators turned by multiples of 360/n."""
    t = pol.shape[1]
    total, used = 0.0, 0
    for g in range(n_groups):
        x = (pol == g).astype(np.float32)
        if x[:, win].mean() < MIN_LAYER_COV:
            continue
        for m in range(1, n):
            sh = int(round(m * t / n))
            ok = win & np.roll(win, -sh)
            total += _pearson(x[:, ok].ravel(), np.roll(x, -sh, axis=1)[:, ok].ravel())
            used += 1
    return total / used if used else 0.0


def refine_center(p: Prepared, labels: np.ndarray, fr: Frame, n: int) -> Prepared:
    """Move the head center along the axis to where the points zone is most n-fold
    symmetric. The localization's center can sit a few units off (it scores the mask
    outline, not the inner parts), which blurs every angular profile."""
    cx, cy, r = p.head
    step = fr.scale * RECENTER_RADIAL_STEP
    radii = np.arange(RECENTER_FROM * r, RECENTER_TO * r, step)
    theta = np.degrees(np.linspace(0, 2 * np.pi, RING_THETA, endpoint=False))
    win = np.abs(theta - 90.0) > TAIL_SECTOR_DEG
    n_groups = int(labels.max()) + 1
    lab = labels.astype(np.int32)
    best = (-2.0, cy)
    for dy in np.arange(-RECENTER_UNITS, RECENTER_UNITS + 1e-9, 1.0):
        c = cy + dy * fr.scale
        sc = _rot_score(polar_sample(lab, cx, c, radii, RING_THETA, fill=-1), win, n, n_groups)
        if sc > best[0] + 1e-9 or (abs(sc - best[0]) <= 1e-9 and abs(dy) < abs(best[1] - cy) / fr.scale):
            best = (sc, c)
    c = best[1]
    top = cy - r
    return replace(p, head=(cx, float(c), float(c - top)))


def _head_disc(p: Prepared, labels: np.ndarray, fr: Frame) -> dict | None:
    cx, cy, r = p.head
    th, rim = _rim(p, fr)
    spread = (np.percentile(rim, 95) - np.percentile(rim, 5)) / r
    if spread >= HEAD_ROUND:
        return None
    arc = r * (th[1] - th[0])
    win = max(3, int(round(RIM_TREND_UNITS * fr.scale / arc)) | 1)
    hp = rim - ndimage.median_filter(rim, size=win, mode="nearest")
    depth = fr.d(np.percentile(hp, 95) - np.percentile(hp, 5))
    inside = rim - RIM_LABEL_UNITS * fr.scale
    xs = np.rint(cx + inside * np.cos(th)).astype(int)
    ys = np.rint(cy + inside * np.sin(th)).astype(int)
    ok = (xs >= 0) & (xs < labels.shape[1]) & (ys >= 0) & (ys < labels.shape[0])
    label, share = majority(labels[ys[ok], xs[ok]])
    if label < 0:
        return None
    pinked = depth >= PINKED_DEPTH
    pink_conf = min(1.0, 0.5 + abs(depth - PINKED_DEPTH) / PINKED_DEPTH / 2)
    return dict(label=label, r=float(np.percentile(rim, 90)), pinked=pinked,
                confidence=float(min(share, pink_conf, 1 - spread)))


def _drop_bridges(bands: list[_Band], cov: np.ndarray, reach: int, head_label: int) -> list[_Band]:
    """Leave out inner bands whose label also covers much of the circle just inside
    and just outside them: lobes of one points layer touching each other there.
    The head label is exempt (it shows between everyone's lobes)."""
    out = []
    for k, bd in enumerate(bands):
        if bd.label == head_label:
            out.append(bd)
            continue
        inner_side = cov[max(0, bd.i0 - reach):bd.i0, bd.label]
        outer_side = cov[bd.i1:bd.i1 + reach, bd.label]
        if (0 < k < len(bands) - 1 and inner_side.size and outer_side.size
                and inner_side.mean() >= BRIDGE_COV and outer_side.mean() >= BRIDGE_COV):
            continue
        out.append(bd)
    return out


def _bands(dom: np.ndarray, domc: np.ndarray, min_len: int) -> list[_Band]:
    bands, i, n = [], 0, len(dom)
    while i < n:
        if domc[i] < FULL_COV:
            i += 1
            continue
        j = i
        while j < n and domc[j] >= FULL_COV and dom[j] == dom[i]:
            j += 1
        if j - i >= min_len:
            bands.append(_Band(i, j, int(dom[i]), float(domc[i:j].mean())))
        i = j
    return bands


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def _score_at(x: np.ndarray, win: np.ndarray, n: int) -> float:
    """Mean correlation of indicator x with itself turned by m*360/n (m = 1..n-1)."""
    t = x.shape[1]
    cs = []
    for m in range(1, n):
        sh = int(round(m * t / n))
        ok = win & np.roll(win, -sh)
        cs.append(_pearson(x[:, ok].ravel(), np.roll(x, -sh, axis=1)[:, ok].ravel()))
    return float(np.mean(cs))


def _symmetry(ind: np.ndarray, win: np.ndarray) -> tuple[int, float, float]:
    """(order N, score, rotation deg) of a (radii, theta) indicator seen through the
    angular window `win`. Score of N = mean correlation of the indicator with itself
    turned by m*360/N (m = 1..N-1) over the angles valid in both; the order is the
    largest N scoring >= SYM_T (divisors of the true order score as high, multiples
    and unrelated N do not)."""
    t = ind.shape[1]
    x = ind.astype(np.float32)
    scores = {n: _score_at(x, win, n) for n in range(2, MAX_COUNT + 1)}
    strong = [n for n, sc in scores.items() if sc >= SYM_T]
    n = max(strong) if strong else max(scores, key=scores.get)
    th = np.linspace(0, 2 * np.pi, t, endpoint=False)
    c = (x[:, win] * np.exp(-1j * n * th[win])[None]).sum()
    rot = (-np.degrees(np.angle(c)) / n) % (360.0 / n)
    return n, max(0.0, scores[n]), float(rot)


def _kind(w: np.ndarray, spacing: np.ndarray | None = None) -> str:
    """Point kind from the lobe width profile w (inner -> outer radius). `spacing` is
    the arc between neighbouring lobes at the same radii (2 pi r / count); with it, a
    wide petal that keeps its width is a loop block (LOOP_*)."""
    wmax = w.max()
    q = max(1, len(w) // 4)
    w_in, w_out = w[:q].mean() / wmax, w[-q:].mean() / wmax
    peak = float(np.argmax(w)) / max(1, len(w) - 1)
    if w_in >= BOW_ENDS and w_out >= BOW_ENDS:
        return "bow"
    if peak <= STAR_PEAK and w_out <= STAR_OUT:
        return "star"
    if w.mean() / wmax < PETAL_FULLNESS:
        return "diamond"
    if spacing is not None:
        x = np.linspace(0, 1, len(w))
        keep = np.interp(np.linspace(0.1, 0.9, 9), x, w) >= LOOP_KEEP * wmax
        share = np.interp(0.5, x, w / spacing)
        if keep.mean() >= LOOP_PLATEAU - 1e-9 and share >= LOOP_SHARE:
            return "bow"
    return "petal"


def _ring_outer(bd: _Band, bands: list[_Band], dom: np.ndarray, cov: np.ndarray, radii: np.ndarray,
                fr: Frame, layers: list[tuple[float, int]]) -> int:
    """Outer radius index of a ring band grown while its label is dominant with
    >= RING_EXTEND_COV, by at most RING_MAX_GROWTH_UNITS, stopping at other bands, at
    the inner radius of a same-label points layer and at the inner radius of any
    other points layer whose label covers >= RING_STOP_COV of the circle there.
    `layers` are the points layers' (r_inner in canvas units, label)."""
    taken = {i for b in bands if b is not bd for i in range(b.i0, b.i1)}
    edge = fr.d(radii[bd.i1 - 1])

    def blocked(i: int) -> bool:
        r = fr.d(radii[i])
        return (r - edge > RING_MAX_GROWTH_UNITS
                or any(r >= r_in and (g == bd.label or cov[i, g] >= RING_STOP_COV) for r_in, g in layers))

    i1 = bd.i1
    while (i1 < len(radii) and i1 not in taken and dom[i1] == bd.label
           and cov[i1, bd.label] >= RING_EXTEND_COV and not blocked(i1)):
        i1 += 1
    return i1


def _blob_layer(p: Prepared, labels: np.ndarray, fr: Frame, g: int, r0: float, r1: float,
                step: float, ind: np.ndarray, win: np.ndarray) -> tuple[dict, float, int] | None:
    """Points layer of label g read from its blobs between radii r0..r1 (px) when they
    form an even ring (parts_common.blob_rings): star tips or loop ends whose pixel
    pattern is too irregular for the rotational self-correlation (hand-made layouts,
    a center a few units off). Returns (params without cx/cy, confidence, outer radius
    index) of the ring with the most blobs, or None. The confidence is the evenness
    only when an independent signal agrees - the layer's zone indicator `ind` (seen
    through `win`) is rotationally symmetric at the blob count (>= LOW_SYM) - and no
    blob is assumed hidden; otherwise BLOB_UNCONFIRMED_CONF (kind and count unverified).
    The blobs' own pixels are not used for that check: isolated blobs correlate with
    themselves whatever their shape."""
    cx, cy, _ = p.head
    h, w = labels.shape
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rings = blob_rings((labels == g) & (dist >= r0) & (dist <= r1), (cx, cy, r1), p.scale)
    if not rings:
        return None
    ring = max(rings, key=lambda q: (q.visible, q.evenness))
    d = dist[ring.pixels]
    ra, rb = float(np.percentile(d, 2)), float(np.percentile(d, 98))
    radii = np.arange(ra, rb + step / 2, step)
    if len(radii) < 2:
        return None
    pol = polar_sample(ring.pixels.astype(np.int8), cx, cy, radii, RING_THETA, fill=0) > 0
    width = pol.mean(axis=1) * 2 * np.pi * radii / ring.visible
    kind = _kind(width, 2 * np.pi * radii / ring.order)
    agree = _score_at(ind.astype(np.float32), win, ring.order) >= LOW_SYM
    conf = ring.evenness if agree and ring.order == ring.visible else BLOB_UNCONFIRMED_CONF
    r_in, r_out = fr.d(ra), fr.d(rb)
    if kind == "star":  # valleys sit where the lobes are widest
        r_in = fr.d(radii[int(np.argmax(width))])
    params = {"kind": kind, "count": int(ring.order), "r_inner": r3(r_in), "r_outer": r3(r_out),
              "rotation_deg": r3(ring.rotation_deg),
              "width_ratio": r3(fr.d(width.max()) / max(r_out - r_in, 1e-6))}
    return params, conf, int(np.searchsorted(np.arange(step / 2, p.head[2], step), rb))


def detect_head(p: Prepared, labels: np.ndarray, fr: Frame) -> list[Part]:
    """Head disc, rings, points rings and center from the polar label profile."""
    cx, cy, r = p.head
    step = fr.scale * RADIAL_STEP_UNITS
    radii = np.arange(step / 2, r, step)
    pol = polar_sample(labels.astype(np.int32), cx, cy, radii, RING_THETA, fill=-1)
    n_groups = int(labels.max()) + 1
    if n_groups <= 0 or len(radii) == 0:
        return []
    theta = np.degrees(np.linspace(0, 2 * np.pi, RING_THETA, endpoint=False))
    win = np.abs(theta - 90.0) > TAIL_SECTOR_DEG  # the tails show below the head
    cov = np.stack([(pol[:, win] == g).mean(axis=1) for g in range(n_groups)], axis=1)
    dom, domc = cov.argmax(axis=1), cov.max(axis=1)
    bands = _bands(dom, domc, max(1, int(round(MIN_BAND_UNITS / RADIAL_STEP_UNITS))))
    head = _head_disc(p, labels, fr)
    all_bands = bands
    bands = _drop_bridges(bands, cov, max(1, int(round(BRIDGE_REACH_UNITS / RADIAL_STEP_UNITS))),
                          head["label"] if head else -2)
    bridges = [bd for bd in all_bands if not any(bd is k for k in bands)]

    ccx, ccy = fr.x(cx), fr.y(cy)
    unit = lambda i: fr.d(radii[min(i, len(radii) - 1)])  # noqa: E731 - radius index -> canvas units
    parts: list[Part] = []
    head_band = None
    if head is not None:
        outer = [b for b in bands if b.label == head["label"] and radii[b.i1 - 1] >= HEAD_BAND_EDGE * r]
        head_band = outer[-1] if outer else None
        parts.append(Part("head", "head", {"cx": r3(ccx), "cy": r3(ccy), "r": r3(fr.d(head["r"])),
                                           "pinked": int(head["pinked"])},
                          head["label"], 10, r3(head["confidence"])))
    center_band = bands[0] if bands and fr.d(radii[bands[0].i0]) <= CENTER_START_UNITS else None
    if center_band is not None and center_band is head_band:
        center_band = None  # one plain disc from the middle to the rim

    # zones between bands -> points layers
    edges = [0] + [i for b in bands for i in (b.i0, b.i1)] + [len(radii)]
    zones = [(edges[k], edges[k + 1]) for k in range(0, len(edges), 2) if edges[k + 1] > edges[k]]
    layers = []
    meta: list[dict] = []  # per layer: zone, radius rows, the band it eats (a star's body)
    pleated: list[tuple[int, int, int, float]] = []  # (i0, i1, label, confidence) of plain layers
    for a, b in zones:
        if fr.d(radii[b - 1] - radii[a]) < MIN_ZONE_UNITS:
            continue
        zcov = cov[a:b]
        inner = next((bd for bd in bands if bd.i1 == a), None)
        for g in range(n_groups):
            if zcov[:, g].mean() < MIN_LAYER_COV or (head is not None and g == head["label"]):
                continue
            rows = np.flatnonzero(zcov[:, g] > EXTENT_COV)
            ra, rb = a + rows[0], a + rows[-1] + 1
            ind = pol[ra:rb] == g
            area = float((cov[ra:rb, g] * 2 * np.pi * radii[ra:rb]).sum() * step)
            n, share, rot = _symmetry(ind, win)
            if share < LOW_SYM:
                found = _blob_layer(p, labels, fr, g, radii[a], radii[b - 1] + step, step, ind, win)
                if found is not None:
                    params, conf, rb_blob = found
                    params.update(cx=r3(ccx), cy=r3(ccy))
                    layers.append(("points", g, conf, params, area, rb_blob))
                    meta.append(dict(zone=(a, b), rows=(ra, rb_blob), eats=None, inner=inner))
                    continue
                ring = _pleated_ring(g, a, b, bridges, cov, radii, fr)
                if ring is not None:
                    pleated.append(ring)
                    continue
                ys, xs = np.nonzero(ind[:, win])
                xs = np.flatnonzero(win)[xs]
                th = 2 * np.pi * xs / RING_THETA
                rr = radii[ra + ys]
                layers.append(("unknown", g, share, dict(
                    cx=r3(fr.x(cx + (rr * np.cos(th)).mean())), cy=r3(fr.y(cy + (rr * np.sin(th)).mean())),
                    r=r3(fr.d(np.sqrt(area / np.pi)))), area, rb))
                meta.append(dict(zone=(a, b), rows=(ra, rb), eats=None, inner=inner))
                continue
            body = max(1, int(round(BODY_UNITS / RADIAL_STEP_UNITS)))
            has_core = (inner is not None and inner.label == g and inner is not head_band
                        and cov[a:a + body, g].mean() >= BODY_COV)
            w = cov[ra:rb, g] * 2 * np.pi * radii[ra:rb] / n
            kind = "star" if has_core else _kind(w, 2 * np.pi * radii[ra:rb] / n)
            r_in, r_out = unit(ra), unit(rb - 1)
            if kind == "star" and not has_core:  # valleys sit where the lobes are widest
                r_in = unit(ra + int(np.argmax(w)))
            params = {"cx": r3(ccx), "cy": r3(ccy), "kind": kind, "count": int(n),
                      "r_inner": r3(r_in), "r_outer": r3(r_out), "rotation_deg": r3(rot),
                      "width_ratio": r3(fr.d(w.max()) / max(r_out - r_in, 1e-6))}
            layers.append(("points", g, share, params, area, rb))
            # a star with a core: the band is this layer's body, the star polygon covers it
            meta.append(dict(zone=(a, b), rows=(ra, rb), eats=inner if has_core else None, inner=inner))

    layers, meta = _explain_trims(layers, meta, pol, radii, win, fr)
    layers, meta = _fold_specks(layers, meta, bands, head_band, pleated, radii, fr)
    eaten = {id(m["eats"]) for m in meta if m["eats"] is not None}

    # points: largest layer first; outer layers drawn underneath (lower z)
    by_outer = sorted(range(len(layers)), key=lambda k: -layers[k][5])
    z_of = {k: min(29, 20 + rank) for rank, k in enumerate(by_outer)}
    order = sorted(range(len(layers)), key=lambda k: -layers[k][4])
    trims: set[int] = set()  # ids of trim parts (edge strips, backings)
    for k in order:
        role, g, share, params, _, _ = layers[k]
        z = z_of[k] if role == "points" else 20
        parts.append(Part(f"{role}-{len(parts)}", role, params, g, z, r3(min(1.0, share))))
        if meta[k].get("trim"):
            trims.add(id(parts[-1]))

    if center_band is not None and id(center_band) not in eaten:
        parts.append(Part("center", "center", {"cx": r3(ccx), "cy": r3(ccy), "r": r3(unit(center_band.i1))},
                          center_band.label, 50, r3(center_band.cov)))
    rings = [(bd.i0, bd) for bd in bands
             if bd is not head_band and bd is not center_band and id(bd) not in eaten]
    rings += [(i0, (i0, i1, g, conf)) for i0, i1, g, conf in pleated]
    rings.sort(key=lambda t: t[0])
    for depth, (_, bd) in enumerate(reversed(rings)):  # outermost first, inner rings on top
        if isinstance(bd, tuple):  # a pleated layer, already measured
            i0, i1, label, conf = bd
        else:
            i0, label, conf = bd.i0, bd.label, bd.cov
            i1 = _ring_outer(bd, bands, dom, cov, radii, fr, [
                (float(q.params["r_inner"]), q.color_group) for q in parts
                if q.role == "points" and id(q) not in trims])
        parts.append(Part(f"ring-{depth}", "ring", {"cx": r3(ccx), "cy": r3(ccy),
                                                    "r_outer": r3(unit(i1)), "r_inner": r3(unit(i0))},
                          label, min(49, 30 + depth), r3(conf)))
    _points_start_at_cover(parts)
    return parts


def _points_start_at_cover(parts: list[Part]) -> None:
    """Points layers start where the ring or center drawn above them ends (H4): a loop
    block (bow) running under the cover starts at its edge (its visible base is what the
    designer draws; the hidden part is guessed), and a layer of another kind (not a
    star) starting within COVER_GAP_UNITS outside the cover started under it (its thin
    inner tip vanishes in the cover's pleats; the designer's kites and petals start
    under the ring). The width stays."""
    covers = [(0.0 if q.role == "center" else float(q.params["r_inner"]),
               float(q.params["r" if q.role == "center" else "r_outer"]))
              for q in parts if q.role in ("ring", "center")]
    for q in parts:
        kind = q.params.get("kind")
        if q.role != "points" or kind == "star":
            continue
        ri, ro = float(q.params["r_inner"]), float(q.params["r_outer"])
        if kind == "bow":
            top = max((b for a, b in covers if a <= ri < b), default=ri)
        else:
            top = max((b for a, b in covers if b <= ri <= b + COVER_GAP_UNITS), default=ri)
        if top != ri and top < ro - MIN_ZONE_UNITS:
            w = float(q.params["width_ratio"]) * (ro - ri)
            q.params["r_inner"] = r3(top)
            q.params["width_ratio"] = r3(w / (ro - top))


def _pleated_ring(g: int, a: int, b: int, bridges: list[_Band], cov: np.ndarray,
                  radii: np.ndarray, fr: Frame) -> tuple[int, int, int, float] | None:
    """(i0, i1, label, confidence) of a ring for label g in the zone a..b when g covers
    the full circle there (a band of g dropped as a bridge lies in the zone), else None.
    The ring runs out from that band, both ways, while g still holds RING_EXTEND_COV
    of the circle (the pleats' teeth; the same share a ring band is grown by). Its
    confidence is its fit: the share of the zone's g area inside the ring times g's
    mean coverage between its radii, capped at BLOB_UNCONFIRMED_CONF unless the layer
    looks like an annulus (PLEATED_FULL_MIN, PLEATED_REACH_UNITS): loops and stars must
    stay flagged."""
    inside_bands = [bd for bd in bridges if bd.label == g and a <= bd.i0 and bd.i1 <= b]
    if not inside_bands:
        return None
    bd = max(inside_bands, key=lambda q: q.i1 - q.i0)
    i0, i1 = bd.i0, bd.i1
    while i0 > a and cov[i0 - 1, g] >= RING_EXTEND_COV:
        i0 -= 1
    while i1 < b and cov[i1, g] >= RING_EXTEND_COV:
        i1 += 1
    area = cov[a:b, g] * radii[a:b]
    inside = float(area[i0 - a:i1 - a].sum() / max(area.sum(), 1e-9))
    conf = inside * float(cov[i0:i1, g].mean())
    out = next((i for i in range(i1, b) if cov[i, g] < EXTENT_COV), b)
    inn = next((i for i in range(i0 - 1, a - 1, -1) if cov[i, g] < EXTENT_COV), a - 1)
    # clamped at the zone edges: tips crossing a zone edge read shorter than they are
    reach = max(fr.d(radii[min(out, len(radii) - 1)] - radii[i1 - 1]),
                fr.d(radii[i0] - radii[max(inn, 0)]))
    full_share = float((cov[i0:i1, g] >= FULL_COV).mean())
    if full_share < PLEATED_FULL_MIN or reach > PLEATED_REACH_UNITS:
        conf = min(conf, BLOB_UNCONFIRMED_CONF)
    return i0, i1, g, conf


def _axis_cols(params: dict) -> list[int]:
    """Polar column of each point's axis (RING_THETA columns, clockwise from +x)."""
    n, rot = int(params["count"]), float(params["rotation_deg"])
    return [int(round((rot + k * 360.0 / n) % 360 * RING_THETA / 360)) % RING_THETA for k in range(n)]


def _edge_bands(pol: np.ndarray, i0: int, i1: int, g: int, h: int, params: dict,
                win: np.ndarray) -> tuple[float, np.ndarray]:
    """(share of the visible loops with a band of h along both sides, polar mask of the
    band pixels found) for the loop layer of label g over radius rows i0..i1."""
    t = RING_THETA
    reach = t // int(params["count"])
    cut = (i1 - i0) // 5
    mid = range(i0 + cut, max(i0 + cut + 1, i1 - cut))
    mark = np.zeros(pol.shape, bool)
    lobes = []
    for c in _axis_cols(params):
        if not win[c]:
            continue
        sides = []
        for d in (1, -1):
            hits = seen = 0
            for i in range(i0, i1):
                row = pol[i]
                if row[c] != g:
                    continue
                steps = 0
                while row[(c + d * steps) % t] == g and steps < reach:
                    steps += 1
                m = 0
                while row[(c + d * (steps + m)) % t] == h and m < reach:
                    m += 1
                after = row[(c + d * (steps + m)) % t]
                band = 1 <= m <= EDGE_MAX_SHARE * 2 * steps and after not in (h, g)
                if band:  # with one sample of slack on each side (resampled edges)
                    mark[i, [(c + d * (steps + j)) % t for j in range(-1, m + 1)]] = True
                if i in mid:
                    seen += 1
                    hits += band
            sides.append(seen > 0 and hits >= EDGE_MIN_RADII * seen)
        lobes.append(all(sides))
    return (float(np.mean(lobes)) if lobes else 0.0), mark


def _backing_tips(pol: np.ndarray, radii: np.ndarray, i_out: int, h: int, params: dict,
                  win: np.ndarray, fr: Frame) -> tuple[float, int, np.ndarray]:
    """(share of the visible points with h on their axis beyond their tip, median
    outer row of h there, polar mask of the h pixels in those points' sectors) for a
    points layer ending at radius row i_out."""
    t = RING_THETA
    tol = max(1, int(round(BACK_TOL_DEG * t / 360)))
    cov_h = (pol == h)[:, win].mean(axis=1)
    mark = np.zeros(pol.shape, bool)
    half = fr.scale * float(params["width_ratio"]) * (float(params["r_outer"]) - float(params["r_inner"])) / 2
    ok, tips = [], []
    for c in _axis_cols(params):
        if not win[c]:
            continue
        cols = [(c + o) % t for o in range(-tol, tol + 1)]
        has = (pol[:, cols] == h).any(axis=1)
        start = next((i for i in range(max(0, i_out - 4), min(len(radii), i_out + 4)) if has[i]), None)
        if start is None:
            ok.append(False)
            continue
        i = start
        while i + 1 < len(radii) and (has[i + 1] or (i + 2 < len(radii) and has[i + 2])):
            i += 1
        beyond = fr.d(radii[i] - radii[i_out])
        good = bool(beyond >= BACK_MIN_UNITS and cov_h[i_out:i + 1].mean() < BACK_MAX_COV)
        ok.append(good)
        if good:
            tips.append(i)
            for j in range(i + 1):
                span = int(np.ceil(np.degrees(np.arctan2(half, max(radii[j], 1e-6))) * t / 360)) + tol
                mark[j, [(c + o) % t for o in range(-span, span + 1)]] = True
    tip = int(np.median(tips)) if tips else i_out
    return (float(np.mean(ok)) if ok else 0.0), tip, mark & (pol == h)


def _explain_trims(layers: list, meta: list[dict], pol: np.ndarray, radii: np.ndarray,
                   win: np.ndarray, fr: Frame) -> tuple[list, list[dict]]:
    """Replace low-confidence layers that are other layers' trims (edge strips along
    loops, backing tips behind points; EXPLAIN_*) by those trims."""
    out, out_meta, extra = [], [], []
    for k, (role, h, share, params, area, rb) in enumerate(layers):
        m = meta[k]
        if role == "points" and share >= EXPLAIN_MAX_CONF:
            out.append(layers[k])
            out_meta.append(m)
            continue
        ra_h, rb_h = m["rows"]
        mine = np.zeros(pol.shape, bool)
        mine[ra_h:rb_h] = pol[ra_h:rb_h] == h
        mine[:, ~win] = False
        explained = np.zeros(pol.shape, bool)
        inner = m["inner"]
        if inner is not None and inner.label == h:  # the fringe of h's own band (pleats)
            reach = int(round(RING_MAX_GROWTH_UNITS * fr.scale / (radii[1] - radii[0]))) if len(radii) > 1 else 0
            explained[inner.i1:inner.i1 + reach] = True
        trims = []
        for j, (role2, g, share2, p2, _, rb2) in enumerate(layers):
            if role2 != "points" or g == h or j == k or meta[j]["zone"] != m["zone"]:
                continue
            i0, i1 = meta[j]["rows"]
            if p2["kind"] == "bow":
                frac, mark = _edge_bands(pol, i0, i1, g, h, p2, win)
                if frac >= EXPLAIN_MIN_LOBES:
                    w = float(p2["width_ratio"]) * (float(p2["r_outer"]) - float(p2["r_inner"]))
                    q = dict(p2, edge_width=r3(EDGE_SHARE * w))
                    trims.append((("points", h, min(frac, share2), q, 0.0, rb2), dict(meta[j], eats=None, trim=True)))
                    explained |= mark
            frac, tip, mark = _backing_tips(pol, radii, i1 - 1, h, p2, win, fr)
            if frac >= EXPLAIN_MIN_LOBES and tip > i1 - 1:
                q = dict(p2, r_outer=r3(fr.d(radii[tip])))
                trims.append((("points", h, min(frac, share2), q, 0.0, tip + 1),
                              dict(meta[j], rows=(i0, tip + 1), eats=None, trim=True)))
                explained |= mark
        if trims and (explained & mine).sum() >= EXPLAIN_MIN_SHARE * max(int(mine.sum()), 1):
            extra += trims
            continue
        out.append(layers[k])
        out_meta.append(m)
    # trims come last: a strip ties its loop's outer radius and must sort after it (z)
    for layer, mm in extra:
        out.append(layer)
        out_meta.append(mm)
    return out, out_meta


def _fold_specks(layers: list, meta: list[dict], bands: list[_Band], head_band: _Band | None,
                 pleated: list, radii: np.ndarray, fr: Frame) -> tuple[list, list[dict]]:
    """Fold unknown layers that are a seam band's fringe (SEAM_UNITS, SPECK_MAX_UNITS2)
    into that band and close a seam they leave empty."""
    out, out_meta, folded = [], [], set()
    for k, (role, h, share, params, area, rb) in enumerate(layers):
        a, b = meta[k]["zone"]
        inner = next((bd for bd in bands if bd.i1 == a), None)
        outer = next((bd for bd in bands if bd.i0 == b), None)
        if (role == "unknown" and inner is not None and outer is not None
                and fr.d(radii[b - 1] - radii[a]) <= SEAM_UNITS
                and h in (inner.label, outer.label) and fr.d(fr.d(area)) < SPECK_MAX_UNITS2):
            folded.add((a, b))
            continue
        out.append(layers[k])
        out_meta.append(meta[k])
    for a, b in folded:
        if any(m["zone"] == (a, b) for m in out_meta) or any(a <= i0 < b for i0, *_ in pleated):
            continue
        outer = next(bd for bd in bands if bd.i0 == b)
        if outer is not head_band:
            outer.i0 = a
    return out, out_meta
