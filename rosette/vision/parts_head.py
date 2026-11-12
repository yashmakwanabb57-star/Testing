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
from shapely import contains_xy, unary_union

from rosette.geometry import disc, pinked_ring, points_ring, ring
from rosette.geometry.common import TOOTH
from rosette.geometry.clip import to_geom
from rosette.model import Part
from rosette.vision.parts_common import Frame, blob_rings, majority, polar_sample, r3
from rosette.vision.parts_trims import explain_trims, fold_specks, split_borders
from rosette.vision.prepare import Prepared

# ML-1 pleated rings: a ring band whose `pleat_texture` is at least PLEAT_TEXTURE is drawn
# pinked. Measured on the 38 detected rings the dataset labels (tools/build_layer_dataset.py):
# the 31 the reference files draw pinked read 0.053-0.68, the one smooth satin band (S55-15)
# 0.020; the 6 other "plain" labels are the head disc showing as a band and read 0.16-0.55
# (A10-15's inner red band is pleated on the photo). Circles: PLEAT_ROWS over the band less
# PLEAT_INSET of its width at each side, PLEAT_THETA samples, brightness against its running
# median over PLEAT_WINDOW_DEG, outside PLEAT_SECTOR_DEG of the tails, PLEAT_MIN_SAMPLES on
# the colour per circle.
PLEAT_TEXTURE = 0.035
# The bar rests on one smooth ring (0.020) and the lowest pleated one (0.053): a ring whose
# texture lies in PLEAT_UNSURE is drawn by the bar but flagged with PLEAT_WARNING
# (`pleat_warnings`); none of the 38 labelled rings falls inside it.
PLEAT_UNSURE = (0.03, 0.05)
PLEAT_WARNING = "The ring's pleating is uncertain - check the ring edge against the photo."
PLEAT_ROWS, PLEAT_INSET, PLEAT_THETA, PLEAT_WINDOW_DEG, PLEAT_SECTOR_DEG, PLEAT_MIN_SAMPLES = 8, 0.2, 1440, 10.0, 45.0, 100
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
# H11: a points layer whose lobes continue inward under layers painted above it starts at
# the cover (_cover_start): at its lobe axes the rows just inside its inner end hold the
# label of a part drawn above it on >= COVER_CUT_SHARE of the visible lobes, and the
# cover runs down to the outer edge of a ring or the center (at most COVER_MAX_UNITS
# further in), where the layer then starts.
COVER_CUT_SHARE, COVER_PROBE_UNITS = 0.6, 3.0
COVER_MAX_UNITS = 80.0
# H11: an unknown layer whose outline (outermost radius per angle) has an n-fold symmetry
# (self-correlation >= SYM_T), n >= STAR_MIN_ORDER equal to the count of a points layer of
# the head or half of it (two lobes per tip) and not reaching beyond its tips by more than STAR_INSIDE_TOL of its r_outer,
# is a points layer of n lobes (a star under another star; S29-15's dark inner star) read
# from its outline, flagged (BLOB_UNCONFIRMED_CONF).
STAR_MIN_ORDER, STAR_INSIDE_TOL = 3, 0.05
# H11: a star whose valleys lie at or beyond STAR_CONVEX of the convex limit
# (r_inner >= STAR_CONVEX * r_outer * cos(180 deg / count)) has no concave notches: it
# would be drawn as a square or polygon. Its valleys are hidden under the layer inside it
# (they were read where the visible lobes are widest), so they move to the start of its
# zone, or to STAR_HIDDEN_DEPTH of the convex limit when that is still hidden inside the
# zone's start; a star that cannot get notches that way is read as an annulus from its
# valleys to its tips instead (a ring, flagged).
STAR_CONVEX = 0.95
STAR_HIDDEN_DEPTH = 0.85
# H11 (Task 4): a points layer of PLEAT_MIN_COUNT or more lobes that fill
# >= PLEAT_FILL of their angular spacing where they are widest, in a zone closed by
# bands on both sides, is the pleated edge of a ring: drawn as a ring with a pinked edge.
PLEAT_MIN_COUNT, PLEAT_FILL = 16, 0.8
# Two such layers of one count and radii (ALT_RADIUS_UNITS) whose lobes are offset by half
# a spacing (within ALT_OFFSET of it) are one two-tone pleated edge: kept as points.
ALT_RADIUS_UNITS, ALT_OFFSET = 3.0, 0.25
# H12: a points layer whose symmetry changes with radius is two layers. Its radial
# extent is cut into SPLIT_BANDS bands; a band's count is the largest n whose turned
# self-correlation lies within SPLIT_MARGIN of the band's best (divisors of the true
# count score as high; n from STAR_MIN_ORDER: a half turn also maps the mirror-symmetric
# head onto itself), undecided when the best is under SPLIT_FLOOR. When the outermost
# decided bands agree on a count c2, none of the decided bands inside them holds c2, and
# at least SPLIT_MIN_INNER of those hold a divisor c1 of it (the largest), the layer
# splits where c2 starts: the inner part is a layer of c1 lobes, the outer part a
# flagged layer of c2 lobes. The split stands only when the trims rule reads the outer
# part as the edge strips of the loops it flanks (_split_and_explain). S29-15: a pink
# 6-point star whose rim shows the pink edge strips of the six loops, 12 lobes.
SPLIT_BANDS, SPLIT_FLOOR, SPLIT_MARGIN, SPLIT_MIN_INNER = 8, 0.2, 0.1, 2
# H12: a loop block's width is the width it shows through its axis over the middle
# BLOCK_MID of its radial extent (_block_width), unless that reads under BLOCK_MIN_SHARE
# of the mean-coverage width (the axis falls on another color: a center stripe, a print).
BLOCK_MID, BLOCK_MIN_SHARE = 0.6, 0.5
# H12: two drawn points layers of different colors that overlap are stacked as the photo
# shows them: over their drawn overlap (less the rings, the center and other layers of
# either color, and the tails sector), the layer whose label holds more than
# STACK_SHARE of the two labels' samples (1 canvas unit apart) is on top, when the two
# labels hold at least STACK_MIN_UNITS2 samples there (one sample per canvas unit^2);
# otherwise the H5 order (points_z) stands. The decided pairs, strongest first, are kept unless they close a cycle, and the
# layers are ordered by them, the H5 order breaking the ties; trims keep their place
# next to their host and the layers keep their set of z values.
STACK_SHARE, STACK_MIN_UNITS2 = 0.65, 200.0
# H12: where the photo does not decide, of two concentric points layers of one count and
# different colors (axes within NEST_AXIS_DEG) whose tips are nested (NEST_UNITS or more
# apart) the one with the shorter tips lies on top: a rosette is built from layers each
# sitting on a larger one. Only the z order changes. Mined on the 76 photos: 7
# such pairs; the photo decides 6, agreeing with the rule on 5 (S47-15, S58-17, S59-17,
# S61-19, S69-19; A15-17 disagrees and the photo wins); the rule decides S29-15.
NEST_AXIS_DEG, NEST_UNITS = 6.0, 3.0
# H13: a head without a round outline (no head disc, HEAD_ROUND: tips standing less than about
# 12 % of the head radius past a round disc keep the disc and get no base) whose silhouette above the
# tails has an n-fold lobed outline (n >= STAR_MIN_ORDER by the outline's turned
# self-correlation, SYM_T) stands on a star when the label just inside its tips (the
# BASE_TIP_UNITS under each visible tip) is one color group on >= BASE_TIP_SHARE of them:
# the base, a star of that color drawn below every other head part (BASE_Z; the head disc
# has z 10, points layers 20-29). Its tips are the outline's tips; its valleys are those of
# the straight-sided star that fits the outline's flanks outside the covering layers best
# (the outline lying more than BASE_FLANK_UNITS outside the largest ring or center and
# inside BASE_FLANK_TIP of the tips), confidence the outline's symmetry score (at most the
# tip share). No base when a points layer reaching within BASE_REACH of the tips already
# draws them: one of that color (a trim too: S80-22's backing), or one of another color on
# every tip axis (its count a multiple of n: S29-15's loops, whose edge strips hold the tip
# color). A layer of that color drawing only some of them (a proper divisor of n, its axes
# on tips, not a trim) is the base read in part (the other arms carry another color:
# S53-19's glitter triangles) and gives way to the base.
# H13 (Task 2b): rings of different colors overlapping by more than RING_OVERLAP_UNITS
# (beyond a pleat's teeth) are re-measured from the band where their color holds at least
# RING_MAJORITY of the circle (_rings_by_dominance).
RING_OVERLAP_UNITS, RING_MAJORITY = 8.0, 0.5
BASE_TIP_UNITS, BASE_TIP_SHARE = 6.0, 0.75
BASE_FLANK_UNITS, BASE_FLANK_TIP, BASE_REACH = 2.0, 0.95, 0.1
BASE_Z = 15
# The base's confidence is at most 1 - (median flank-fit error) / BASE_FIT_UNITS: an outline
# whose lobes are rounded (not a star's straight sides) is flagged. S53-19 fits to 1.7 units (0.72), a
# synthetic 8-lobed scalloped outline to 3.4 (0.43).
BASE_FIT_UNITS = 6.0
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
            wide = w.max()
            if kind == "bow":  # H12: a loop block is as wide as it shows at mid radius
                block = _block_width(ind, win, n, rot, radii[ra:rb])
                wide = block if block >= BLOCK_MIN_SHARE * wide else wide
            params = {"cx": r3(ccx), "cy": r3(ccy), "kind": kind, "count": int(n),
                      "r_inner": r3(r_in), "r_outer": r3(r_out), "rotation_deg": r3(rot),
                      "width_ratio": r3(fr.d(wide) / max(r_out - r_in, 1e-6))}
            layers.append(("points", g, share, params, area, rb))
            # a star with a core: the band is this layer's body, the star polygon covers it
            meta.append(dict(zone=(a, b), rows=(ra, rb), eats=inner if has_core else None, inner=inner))

    layers, meta = _split_and_explain(layers, meta, pol, radii, win, fr, ccx, ccy)
    layers, meta = split_borders(layers, meta, pol, radii, win, fr)
    layers, meta = fold_specks(layers, meta, bands, head_band, pleated, radii, fr, RING_MAX_GROWTH_UNITS)
    # H11: convex "stars" and pleated edges are rings (annuli), flagged / pinked
    keep = []
    for k, lay in enumerate(layers):
        role, g, share, params = lay[:4]
        if role == "points" and not meta[k].get("trim"):
            convex = _convex_star(params)
            if convex:  # its valleys are hidden under the layers inside its zone
                deeper = _hidden_valleys(params, unit(meta[k]["zone"][0]))
                if deeper is not None:
                    layers[k] = (role, g, share, deeper) + tuple(lay[4:])
                    keep.append(k)
                    continue
            if convex or (_pleat_layer(params, meta[k]["zone"], bands)
                          and not _alternating(k, layers, meta)):
                i0 = int(np.searchsorted(radii, float(params["r_inner"]) * fr.scale))
                i1 = min(len(radii), int(np.searchsorted(radii, float(params["r_outer"]) * fr.scale)) + 1)
                if i1 > i0:
                    conf = min(share, BLOB_UNCONFIRMED_CONF) if convex else share
                    pleated.append((i0, i1, g, conf, 0 if convex else int(params["count"])))
                    continue
        keep.append(k)
    layers, meta = [layers[k] for k in keep], [meta[k] for k in keep]
    eaten = {id(m["eats"]) for m in meta if m["eats"] is not None}

    z_of = dict(enumerate(points_z([(lay[0], lay[3].get("kind"), lay[5]) for lay in layers])))
    order = sorted(range(len(layers)), key=lambda k: -layers[k][4])
    trims: set[int] = set()  # ids of trim parts (edge strips, backings)
    for k in order:
        role, g, share, params, _, _ = layers[k]
        z = z_of[k]
        parts.append(Part(f"{role}-{len(parts)}", role, params, g, z, r3(min(1.0, share))))
        if meta[k].get("trim"):
            trims.add(id(parts[-1]))

    if center_band is not None and id(center_band) not in eaten:
        parts.append(Part("center", "center", {"cx": r3(ccx), "cy": r3(ccy), "r": r3(unit(center_band.i1))},
                          center_band.label, 50, r3(center_band.cov)))
    # by outer edge (H11: a pleated annulus can enclose a narrower ring, which lies on it)
    starts = [(float(q.params["r_inner"]), q.color_group) for q in parts if q.role == "points" and id(q) not in trims]
    found = []  # [sort key, i0, i1, label, confidence, pleats]
    for bd in bands:
        if bd is not head_band and bd is not center_band and id(bd) not in eaten:
            found.append([bd.i1, bd.i0, _ring_outer(bd, bands, dom, cov, radii, fr, starts), bd.label, bd.cov, 0])
    for ring in pleated:  # a pleated layer, already measured
        i0, i1, label, conf, pinked = tuple(ring) + (0,) * (5 - len(ring))
        found.append([i1, i0, i1, label, conf, pinked])
    _rings_by_dominance(found, dom, domc, fr, radii, center_band)
    found.sort(key=lambda t: t[0])
    for depth, (_, i0, i1, label, conf, pinked) in enumerate(reversed(found)):  # outermost first, inner on top
        rp = {"cx": r3(ccx), "cy": r3(ccy), "r_outer": r3(unit(i1)), "r_inner": r3(unit(i0))}
        if pinked:  # a pleated edge (H11): pinked, with its pleat count (re-centering)
            rp["pinked"], rp["pleats"] = 1, int(pinked)
        elif pleat_texture(p, labels == label, unit(i0) * fr.scale, unit(i1) * fr.scale) >= PLEAT_TEXTURE:
            rp["pinked"] = 1  # ML-1: a pleated ribbon band (the files draw these pinked)
        parts.append(Part(f"ring-{depth}", "ring", rp, label, min(49, 30 + depth), r3(conf)))
    _finish_points(parts, labels, p, fr, pol, radii, win, head["label"] if head else -2, trims)
    if head is None:
        _base_star(parts, p, labels, fr, win, trims)
    return parts


def pleat_texture(p: Prepared, member: np.ndarray, r0: float, r1: float,
                  center: tuple[float, float] | None = None) -> float:
    """Pleat texture of a ring band between radii r0..r1 (px) of the colour `member`: on
    PLEAT_ROWS circles over the middle of the band, the spread of the photo's brightness
    around its PLEAT_WINDOW_DEG running median along the circle (folds every few degrees),
    over its mean, on the colour's pixels outside PLEAT_SECTOR_DEG of the tails; the
    median over the circles (0 when too little shows). `center` (px): default the head's."""
    cx, cy = center if center is not None else p.head[:2]
    pad = PLEAT_INSET * (r1 - r0)
    rr = np.linspace(r0 + pad, r1 - pad, PLEAT_ROWS)
    gray = p.rgb.astype(np.float32).mean(axis=2)
    g = polar_sample(gray, cx, cy, rr, PLEAT_THETA, fill=np.nan)
    on = polar_sample(member.astype(np.int8), cx, cy, rr, PLEAT_THETA, fill=0) > 0
    th = np.degrees(np.linspace(0, 2 * np.pi, PLEAT_THETA, endpoint=False))
    vis = np.abs(th - 90.0) > PLEAT_SECTOR_DEG
    size = int(round(PLEAT_WINDOW_DEG / 360.0 * PLEAT_THETA)) | 1
    out = []
    for row, ok in zip(g, on):
        sel = vis & ok & np.isfinite(row)
        if sel.sum() < PLEAT_MIN_SAMPLES:
            continue
        x = np.where(np.isfinite(row), row, np.nanmean(row))
        hp = x - ndimage.median_filter(x, size=size, mode="wrap")
        out.append(float(np.std(hp[sel]) / max(float(np.mean(x[sel])), 1.0)))
    return float(np.median(out)) if out else 0.0


def pleat_warnings(parts: list[Part], p: Prepared, labels: np.ndarray) -> list[str]:
    """PLEAT_WARNING for every ring drawn by the pleat-texture rule whose texture lies in
    PLEAT_UNSURE (the bar rests on few rings; see PLEAT_TEXTURE). H11 pleated edges
    (rings with a pleat count) are read from their teeth and are not checked."""
    fr = Frame.of(p)
    out = []
    for q in parts:
        a = q.params
        if q.role != "ring" or a.get("pleats"):
            continue
        center = (fr.axis_x + (float(a["cx"]) - 300.0) * fr.scale, fr.py(float(a["cy"])))
        t = pleat_texture(p, labels == q.color_group, float(a["r_inner"]) * fr.scale,
                          float(a["r_outer"]) * fr.scale, center)
        if PLEAT_UNSURE[0] <= t < PLEAT_UNSURE[1]:
            out.append(PLEAT_WARNING)
    return out[:1]


def _rings_by_dominance(found: list[list], dom: np.ndarray, domc: np.ndarray, fr: Frame,
                        radii: np.ndarray, center_band) -> None:
    """H13: rings of different colors overlapping by more than RING_OVERLAP_UNITS (a pleat's
    depth: the pinked teeth) take their radii from the band where their color holds the
    majority of the circle (RING_MAJORITY), the run overlapping them most (outside the
    center's), extended over the rows no color holds to halfway to the next run. Pleated rings
    whose layers' extents all reach the same pleated edge (S53-19: blue, glitter and white
    rings all ending at 84.5) become the three concentric bands the photo shows. In place:
    [sort key, i0, i1, label, ...] rows; a moved ring's sort key becomes its new outer edge."""
    def over(a, b):
        return fr.d(radii[min(a[2], b[2]) - 1] - radii[max(a[1], b[1])]) if min(a[2], b[2]) > max(a[1], b[1]) else 0.0
    moved = {k for k, a in enumerate(found) for b in found
             if a is not b and a[3] != b[3] and over(a, b) > RING_OVERLAP_UNITS}
    if not moved:
        return
    n = len(dom)
    held = (domc >= RING_MAJORITY)
    lab = np.where(held, dom, -1)
    if center_band is not None:
        lab[:center_band.i1] = -2  # the center's own rows
    runs = []  # (i0, i1, label)
    i = 0
    while i < n:
        j = i
        while j < n and lab[j] == lab[i]:
            j += 1
        if lab[i] >= 0:
            runs.append((i, j, int(lab[i])))
        i = j
    min_len = max(1, int(round(MIN_BAND_UNITS / RADIAL_STEP_UNITS)))
    for k in sorted(moved):
        row = found[k]
        mine = [r for r in runs if r[2] == row[3] and r[1] - r[0] >= min_len]
        if not mine:
            continue
        a0, a1, _ = max(mine, key=lambda r: (min(r[1], row[2]) - max(r[0], row[1]),
                                            -abs((r[0] + r[1]) - (row[1] + row[2]))))
        if min(a1, row[2]) - max(a0, row[1]) <= 0:
            continue  # its color holds no band under it: as measured
        lo = a0  # over the unheld rows to the boundary with the next run (or the center)
        while lo > 0 and lab[lo - 1] == -1:
            lo -= 1
        hi = a1
        while hi < n and lab[hi] == -1:
            hi += 1
        # a gap bounded on both sides is split at one boundary row, the same for the rings on
        # either side of it (the inner one takes the odd row); an open gap is not taken
        lo = lo + (a0 - lo + 1) // 2 if lo > 0 else a0
        hi = a1 + (hi - a1 + 1) // 2 if hi < n else a1
        row[1], row[2], row[0] = lo, hi, hi
    moved_rows = [id(found[k]) for k in moved]
    seen, keep = set(), []
    for k, row in enumerate(found):  # one ring per color and radii
        key = (row[1], row[2], row[3])
        if k in moved and key in seen:
            continue
        seen.add(key)
        keep.append(row)
    found[:] = keep
    teeth = max(1, int(np.ceil(TOOTH / RADIAL_STEP_UNITS)))
    for row in found:  # a ring reaches under the pinked teeth of the ring inside it (drawn on top)
        if id(row) in moved_rows and any(r is not row and r[5] and r[3] != row[3] and r[2] == row[1] for r in found):
            row[1] = max(0, row[1] - teeth)


def _head_outline(p: Prepared, fr: Frame) -> np.ndarray:
    """Outline radius (px) per angle (RING_THETA angles, theta from +x clockwise on screen)
    of the head silhouette: the mask above the tails, followed out from RIM_START head
    radii to the first radius off it; NaN where the mask is off already at RIM_START or
    still on at RIM_MAX."""
    cx, cy, r = p.head
    step = max(0.5, fr.scale * RADIAL_STEP_UNITS / 2)
    radii = np.arange(RIM_START * r, RIM_MAX * r, step)
    sil = p.mask.copy()
    sil[int(np.ceil(p.tails_top_y)):] = False
    pol = polar_sample(sil.astype(np.int8), cx, cy, radii, RING_THETA, fill=0) > 0
    first_off = np.where(~pol, np.arange(len(radii))[:, None], len(radii)).min(axis=0)
    out = radii[np.clip(first_off - 1, 0, len(radii) - 1)].astype(np.float64)
    out[(first_off == 0) | (first_off == len(radii))] = np.nan
    return out


def _star_side(phi: np.ndarray, r_tip: float, r_valley: float, n: int) -> np.ndarray:
    """Radius at angle phi (radians from a tip axis, 0..pi/n) of the straight side of an
    n-point star with tips at r_tip and valleys at r_valley."""
    a = np.pi / n
    return r_tip * r_valley * np.sin(a) / (r_valley * np.sin(a - phi) + r_tip * np.sin(phi))


def _base_star(parts: list[Part], p: Prepared, labels: np.ndarray, fr: Frame, win: np.ndarray,
               trims: set[int]) -> None:
    """H13: the star the head stands on (BASE_*), appended to `parts` as part 'base'."""
    cx, cy, _ = p.head
    out = _head_outline(p, fr)
    valid = win & np.isfinite(out)
    if valid.sum() < 16:
        return
    x = np.where(valid, out, 0.0)
    n, score, rot = _outline_symmetry(x, valid)
    if n < STAR_MIN_ORDER:
        return
    t = len(out)
    deg = np.degrees(np.linspace(0, 2 * np.pi, t, endpoint=False))
    half = max(1, int(round(t / n / 4)))  # a tip is the outline's maximum within a quarter spacing
    tips, groups = [], []
    for k in range(n):
        c = int(round((rot + 360.0 * k / n) / 360.0 * t)) % t
        cols = [(c + d) % t for d in range(-half, half + 1)]
        if not win[c] or not all(valid[j] for j in cols):
            continue
        j = max(cols, key=lambda i: out[i])
        tip = float(out[j])
        rr = tip - fr.scale * np.arange(1.0, BASE_TIP_UNITS + 0.5, 1.0)
        a = np.radians(deg[j])
        xs = np.rint(cx + rr * np.cos(a)).astype(int)
        ys = np.rint(cy + rr * np.sin(a)).astype(int)
        ok = (xs >= 0) & (xs < labels.shape[1]) & (ys >= 0) & (ys < labels.shape[0])
        g, _ = majority(labels[ys[ok], xs[ok]])
        tips.append(tip)
        groups.append(g)
    if len(tips) < STAR_MIN_ORDER:
        return
    vals, counts = np.unique(np.array(groups), return_counts=True)
    g, share = int(vals[np.argmax(counts)]), float(counts.max() / len(groups))
    if g < 0 or share < BASE_TIP_SHARE - 1e-9:
        return
    r_tip = fr.d(float(np.median(tips)))
    cover = max([float(q.params["r" if q.role == "center" else "r_outer"])
                 for q in parts if q.role in ("ring", "center")], default=0.0)
    if r_tip - cover < MIN_ZONE_UNITS:
        return
    partial = []
    for k, q in enumerate(parts):
        if q.role != "points" or float(q.params["r_outer"]) < (1 - BASE_REACH) * r_tip:
            continue
        m = int(q.params["count"])
        d = (float(q.params["rotation_deg"]) - rot) % (360.0 / n)
        on_tips = min(d, 360.0 / n - d) <= NEST_AXIS_DEG
        if q.color_group == g and id(q) not in trims and m < n and n % m == 0 and on_tips:
            partial.append(k)
        elif q.color_group == g or (m % n == 0 and on_tips):
            return  # a layer (or a trim) of this color, or another color's on every tip, draws them
    # the valleys: the straight-sided star fitting the outline's flanks outside the cover
    phi = (np.radians(deg) - np.radians(rot)) % (2 * np.pi / n)
    phi = np.minimum(phi, 2 * np.pi / n - phi)
    xu = np.array([fr.d(v) for v in np.where(valid, out, 0.0)])
    flank = valid & (xu > cover + BASE_FLANK_UNITS) & (xu < BASE_FLANK_TIP * r_tip)
    limit = STAR_HIDDEN_DEPTH * r_tip * np.cos(np.pi / n)  # notched, never a polygon
    if flank.sum() < 8 or limit <= 1.0:
        return
    cands = np.arange(1.0, limit, 0.5)
    errs = [float(np.median(np.abs(xu[flank] - _star_side(phi[flank], r_tip, v, n)))) for v in cands]
    r_in = float(cands[int(np.argmin(errs))])
    fit = max(0.0, 1.0 - min(errs) / BASE_FIT_UNITS)  # straight sides: a star, rounded lobes: not
    for k in sorted(partial, reverse=True):
        del parts[k]
    chord = 2 * r_in * np.sin(np.pi / n)  # the widest lobe width, at the valleys
    parts.append(Part("base", "points", {
        "cx": r3(fr.x(cx)), "cy": r3(fr.y(cy)), "kind": "star", "count": int(n), "r_inner": r3(r_in),
        "r_outer": r3(r_tip), "rotation_deg": r3(rot),
        "width_ratio": r3(chord / max(r_tip - r_in, 1e-6))}, g, BASE_Z, r3(min(score, share, fit, 1.0))))


def _finish_points(parts: list[Part], labels: np.ndarray, p: Prepared, fr: Frame, pol: np.ndarray,
                   radii: np.ndarray, win: np.ndarray, head_label: int, trims: set[int]) -> None:
    """Points layers start at the cover over them (H4), are stacked as the photo shows
    them (H12) and then start under the layers drawn above them (H11 _cover_start, which
    reads z: it runs on the final order)."""
    _points_start_at_cover(parts)
    photo_stacking(parts, labels, p, fr, trims)
    _cover_start(parts, pol, radii, win, fr, head_label, trims)


def points_z(layers: list[tuple[str, str | None, float]]) -> list[int]:
    """z (20-29) of the head layers [(role, kind, outer radius)] (H5, mined from the
    reference files: tools/stacking.py). Outer layers are drawn underneath: of the
    points pieces that overlap or are cut along each other, the one ending further in
    lies on top in 92 % of the pairs with distinct ends (1292 of 1400). Loop over star is
    the exception to it that the references keep consistently (89 %, 63 of 71 pairs in
    10 files, whichever ends further out), so a star ranks as ending beyond every loop.
    Ties keep the layers' order (a trim after its host). Non-points layers get 20.
    The two rules can form a cycle (a kite ending inside the loops over a loop, a star
    ending inside the kite over the kite, the loop over the star): the radius rule loses
    for stars, which go under every loop and so under everything ranked above a loop."""
    loops = [rb for role, kind, rb in layers if role == "points" and kind == "bow"]
    stars = [rb for role, kind, rb in layers if role == "points" and kind == "star"]
    lift = max(0.0, max(loops) + 0.5 - min(stars)) if loops and stars else 0.0  # stars keep their order
    reach = [rb + lift if role == "points" and kind == "star" else rb for role, kind, rb in layers]
    by_outer = sorted(range(len(layers)), key=lambda k: -reach[k])
    rank = {k: i for i, k in enumerate(by_outer)}
    return [min(29, 20 + rank[k]) if layers[k][0] == "points" else 20 for k in range(len(layers))]


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


def _cover_start(parts: list[Part], pol: np.ndarray, radii: np.ndarray, win: np.ndarray, fr: Frame,
                 head_label: int, trims: set[int]) -> None:
    """H11: a points layer (not a star: its r_inner is its valleys) whose lobes are cut by
    layers painted above it starts at the cover, not at its first visible row (COVER_*).
    At the lobe axes (rotation + k 360/count, the visible ones) the rows just inside
    r_inner must hold, on >= COVER_CUT_SHARE of the lobes, the label of a part drawn
    above the layer (a ring, the center, or a points layer of higher z); r_inner then
    moves inward while each row is still covered that way (or shows the layer's own
    label) and must reach the outer edge of a ring or the center, where it stops. The
    width stays (width_ratio rescaled)."""
    t = pol.shape[1]
    edges = sorted({float(q.params["r" if q.role == "center" else "r_outer"])
                    for q in parts if q.role in ("ring", "center")})
    probe = max(1, int(round(COVER_PROBE_UNITS / RADIAL_STEP_UNITS)))
    top_z: dict[int, int] = {}
    for q in parts:
        if q.role in ("ring", "center", "points"):
            top_z[q.color_group] = max(top_z.get(q.color_group, -1), q.z)
    n_lab = int(max([pol.max(), head_label, *top_z]) + 2)
    for q in parts:
        if q.role != "points" or id(q) in trims or q.params.get("kind") == "star":
            continue
        n, rot = int(q.params["count"]), float(q.params["rotation_deg"])
        cols = {(int(round((rot + 360.0 * k / n) / 360.0 * t)) + d) % t for k in range(n) for d in (-2, 0, 2)}
        cols = np.array(sorted(c for c in cols if win[c]), int)
        if cols.size == 0:
            continue
        ri, ro = float(q.params["r_inner"]), float(q.params["r_outer"])
        i_in = int(np.searchsorted(radii, ri * fr.scale))
        if i_in <= probe:
            continue
        g = q.color_group
        above = np.zeros(n_lab, bool)
        for lab, z in top_z.items():
            above[lab] = lab != g and lab != head_label and z > q.z
        lut = lambda a: above[np.clip(a, 0, n_lab - 1)] & (a >= 0)  # noqa: E731
        if lut(pol[i_in - probe:i_in][:, cols]).mean() < COVER_CUT_SHARE:
            continue
        stop = max((e for e in edges if e <= ri + RADIAL_STEP_UNITS), default=None)
        if stop is None or stop >= ri - RADIAL_STEP_UNITS or ri - stop > COVER_MAX_UNITS:
            continue  # no ring or center inside, or the layer already starts at one
        i = i_in - 1
        while i > 0 and fr.d(radii[i]) > stop:
            row = pol[i, cols]
            if (lut(row) | (row == g)).mean() < COVER_CUT_SHARE:
                break
            i -= 1
        if fr.d(radii[i]) > stop + RADIAL_STEP_UNITS or stop >= ro - MIN_ZONE_UNITS:
            continue  # the cover does not reach down to a ring or the center
        w = float(q.params["width_ratio"]) * (ro - ri)
        q.params["r_inner"] = r3(stop)
        q.params["width_ratio"] = r3(w / (ro - stop))


def _outline_order(ind: np.ndarray, win: np.ndarray) -> tuple[int, float, float, np.ndarray]:
    """(order n, score, rotation deg, outline row per angle; -1 where empty) of a layer's
    outline: the outermost row of the (radii, theta) indicator per angle. The order is
    the largest n whose turned self-correlation over the visible angles reaches SYM_T."""
    rows = np.where(ind, np.arange(ind.shape[0])[:, None], -1).max(axis=0)
    n, sc, rot = _outline_symmetry(rows.astype(np.float64), win & (rows >= 0))
    return n, sc, rot, rows


def _outline_symmetry(x: np.ndarray, valid: np.ndarray) -> tuple[int, float, float]:
    """(order n, score, rotation deg) of an outline radius per angle `x` over the angles
    `valid`: the largest n whose turned self-correlation reaches SYM_T; the rotation is
    the direction of the first tip (the phase of the n-th harmonic)."""
    t = len(x)
    best: tuple[int, float] = (0, 0.0)
    for n in range(2, MAX_COUNT + 1):
        cs = []
        for m in range(1, n):
            sh = int(round(m * t / n))
            ok = valid & np.roll(valid, -sh)
            if ok.sum() < 8:
                continue
            cs.append(_pearson(x[ok], np.roll(x, -sh)[ok]))
        sc = float(np.mean(cs)) if cs else 0.0
        if sc >= SYM_T:
            best = (n, sc)
    n, sc = best
    if n == 0:
        return 0, 0.0, 0.0
    th = np.linspace(0, 2 * np.pi, t, endpoint=False)
    c = ((x[valid] - x[valid].mean()) * np.exp(-1j * n * th[valid])).sum()
    return n, sc, float((-np.degrees(np.angle(c)) / n) % (360.0 / n))


def _star_from_outline(layers: list, meta: list[dict], pol: np.ndarray, radii: np.ndarray, win: np.ndarray,
                       fr: Frame, ccx: float, ccy: float) -> None:
    """H11: an unknown layer whose outline has the symmetry of a points layer of the head
    and lies inside it (STAR_*) becomes a points layer of its outline's order, flagged."""
    counts = []
    for lay, mm in zip(layers, meta):
        if mm.get("outer"):  # a split counts once, as the layer it came from (H12)
            continue
        lay = mm["whole"][0] if "whole" in mm else lay
        if lay[0] == "points":
            counts.append((int(lay[3]["count"]), float(lay[3]["r_outer"])))
    for k, (role, g, share, params, area, rb) in enumerate(layers):
        if role != "unknown":
            continue
        ra, rb_ = meta[k]["rows"]
        ind = pol[ra:rb_] == g
        n, sc, rot, rows = _outline_order(ind, win)
        if n < STAR_MIN_ORDER:
            continue
        out = rows[win & (rows >= 0)] + ra
        r_tip = fr.d(radii[int(np.percentile(out, 95))])
        if not any(m in (n, 2 * n) and r_tip <= (1 + STAR_INSIDE_TOL) * ro for m, ro in counts):
            continue
        cov = ind[:, win].mean(axis=1)
        w = cov * 2 * np.pi * radii[ra:rb_] / n
        kind = _kind(w, 2 * np.pi * radii[ra:rb_] / n)
        r_in = fr.d(radii[int(np.percentile(out, 10))]) if kind == "star" else fr.d(radii[ra])
        if r_tip - r_in < MIN_ZONE_UNITS:
            continue
        layers[k] = ("points", g, min(sc, BLOB_UNCONFIRMED_CONF),
                     {"cx": r3(ccx), "cy": r3(ccy), "kind": kind, "count": int(n), "r_inner": r3(r_in),
                      "r_outer": r3(r_tip), "rotation_deg": r3(rot),
                      "width_ratio": r3(fr.d(w.max()) / max(r_tip - r_in, 1e-6))}, area, rb)


def band_orders(ind: np.ndarray, win: np.ndarray, nb: int = SPLIT_BANDS) -> tuple[np.ndarray, list[int]]:
    """H12: (band edges as row offsets, count per band; 0 = undecided) of a layer's
    (radii, theta) indicator cut into nb radial bands (SPLIT_*)."""
    edges = np.linspace(0, ind.shape[0], nb + 1).astype(int)
    orders = []
    for a, b in zip(edges[:-1], edges[1:]):
        x = ind[a:b].astype(np.float32)
        if b - a < 2 or x[:, win].mean() < MIN_LAYER_COV:
            orders.append(0)
            continue
        sc = {n: _score_at(x, win, n) for n in range(STAR_MIN_ORDER, MAX_COUNT + 1)}
        best = max(sc.values())
        orders.append(0 if best < SPLIT_FLOOR else max(n for n, s in sc.items() if s >= best - SPLIT_MARGIN))
    return edges, orders


def radial_split(orders: list[int]) -> tuple[int, int, int] | None:
    """H12: (c1, c2, first outer band) when the outermost decided bands agree on a count
    c2, none of the decided bands inside them holds c2 (or a multiple of it): the finer
    symmetry shows only outside, and SPLIT_MIN_INNER or more of them hold a divisor of
    c2 (c1: the largest such; the bands' counts are noisy, a divisor scores as high).
    Else None."""
    decided = [(k, o) for k, o in enumerate(orders) if o]
    if not decided:
        return None
    c2 = decided[-1][1]
    j = len(decided)
    while j > 0 and decided[j - 1][1] == c2:
        j -= 1
    inner = [o for _, o in decided[:j]]
    if not inner:
        return None
    if any(o % c2 == 0 for o in inner):
        return None
    divisors = [o for o in set(inner) if o >= STAR_MIN_ORDER and o < c2 and c2 % o == 0
                and inner.count(o) >= SPLIT_MIN_INNER]
    if not divisors:
        return None
    c1 = max(divisors)
    return c1, c2, decided[j][0]


def _axis_widths(ind: np.ndarray, win: np.ndarray, n: int, rot: float) -> np.ndarray:
    """H12: per row, the median (over the visible lobes) of the angular run of the
    indicator through each lobe's axis, in columns (0 where the axis is not covered):
    the width of the lobes themselves, leaving out other pieces of the label between
    them (a split layer's inner part still holds the outer part's strips)."""
    t = ind.shape[1]
    cols = [int(round((rot + k * 360.0 / n) % 360 * t / 360)) % t for k in range(n)]
    cols = [c for c in cols if win[c]]
    if not cols:
        return np.zeros(ind.shape[0])
    reach = t // (2 * n)
    out = np.zeros((ind.shape[0], len(cols)))
    for j, c in enumerate(cols):
        for i in range(ind.shape[0]):
            if not ind[i, c]:
                continue
            run = 1
            for d in (1, -1):
                s = 1
                while s <= reach and ind[i, (c + d * s) % t]:
                    s += 1
                run += s - 1
            out[i, j] = run
    return np.median(out, axis=1)


def _block_width(ind: np.ndarray, win: np.ndarray, n: int, rot: float, rr: np.ndarray) -> float:
    """H12: lateral width (px) of a loop block layer as it shows: per row of the middle
    BLOCK_MID share of its radial extent the median, over the visible lobes, of the run
    of its label through each lobe's axis; the widest of those rows (the loop profile's
    full width; the base under the cover and the folded end are left out). 0 when no
    lobe shows there. The mean coverage the other kinds use counts every piece of the
    label between the loops too."""
    m = len(rr)
    i0 = int(m * (0.5 - BLOCK_MID / 2))
    i1 = max(int(np.ceil(m * (0.5 + BLOCK_MID / 2))), i0 + 1)
    runs = _axis_widths(ind[i0:i1], win, n, rot) * 2 * np.pi * rr[i0:i1] / ind.shape[1]
    return float(runs.max()) if runs.size else 0.0


def _layer_params(ind: np.ndarray, win: np.ndarray, n: int, radii: np.ndarray, ra: int,
                  fr: Frame, ccx: float, ccy: float, star: bool = False) -> dict:
    """Points params of n lobes read off a (radii from row ra, theta) indicator: kind by
    the lobe width profile, rotation by the n-th harmonic, radial extent its rows (a star:
    from where its lobes are widest to the last row the label holds on most lobe axes;
    the rows end where the outer count starts, which lies beyond the tips). star: the
    layer is a star whatever its width profile (a star over its body band)."""
    x = ind.astype(np.float32)
    t = x.shape[1]
    th = np.linspace(0, 2 * np.pi, t, endpoint=False)
    c = (x[:, win] * np.exp(-1j * n * th[win])[None]).sum()
    rot = (-np.degrees(np.angle(c)) / n) % (360.0 / n)
    rr = radii[ra:ra + x.shape[0]]
    w = _axis_widths(ind, win, n, rot) * 2 * np.pi * rr / t
    kind = "star" if star else _kind(w, 2 * np.pi * rr / n) if w.max() > 0 else "petal"
    r_in, r_out = fr.d(rr[0]), fr.d(rr[-1])
    if kind == "star":  # valleys sit where the lobes are widest, tips where most axes end
        r_in = fr.d(rr[int(np.argmax(w))])
        if w.max() > 0:
            r_out = fr.d(rr[int(np.flatnonzero(w > 0).max())])
    return {"cx": r3(ccx), "cy": r3(ccy), "kind": kind, "count": int(n), "r_inner": r3(r_in),
            "r_outer": r3(r_out), "rotation_deg": r3(rot),
            "width_ratio": r3(fr.d(w.max()) / max(r_out - r_in, 1e-6))}


def _split_by_radius(layers: list, meta: list[dict], pol: np.ndarray, radii: np.ndarray, win: np.ndarray,
                     fr: Frame, ccx: float, ccy: float) -> tuple[list, list[dict]]:
    """H12: a points layer whose count changes with radius (radial_split) is replaced by
    its inner part (c1 lobes, its own confidence) and its outer part (c2 lobes, flagged)."""
    out, out_meta = [], []
    for lay, m in zip(layers, meta):
        role, g, share, params, area, rb = lay
        ra, rb_ = m["zone"][0], m["zone"][1]
        rows = np.flatnonzero((pol[ra:rb_] == g)[:, win].mean(axis=1) > EXTENT_COV)
        if role != "points" or rows.size == 0:
            out.append(lay)
            out_meta.append(m)
            continue
        ra, rb_ = ra + rows[0], ra + rows[-1] + 1
        ind = pol[ra:rb_] == g
        edges, orders = band_orders(ind, win)
        found = radial_split(orders)
        if found is None:
            out.append(lay)
            out_meta.append(m)
            continue
        c1, c2, k = found
        rs = ra + int(edges[k])
        if rs - ra < 2 or rb_ - rs < 2 or fr.d(radii[rs - 1] - radii[ra]) < MIN_ZONE_UNITS:
            out.append(lay)
            out_meta.append(m)
            continue
        inner = _layer_params(pol[ra:rs] == g, win, c1, radii, ra, fr, ccx, ccy, star=m["eats"] is not None)
        outer = _layer_params(pol[rs:rb_] == g, win, c2, radii, rs, fr, ccx, ccy)
        if m["eats"] is not None:  # a star over its body band keeps its valleys there
            inner["r_inner"] = params["r_inner"]
        out.append(("points", g, share, inner, area, rs))
        key = len(out)
        out_meta.append(dict(m, rows=(ra, rs), split=key, whole=(lay, m)))
        out.append(("points", g, min(share, BLOB_UNCONFIRMED_CONF), outer, 0.0, rb_))
        out_meta.append(dict(m, rows=(rs, rb_), eats=None, split=key, outer=True))
    return out, out_meta


def _restore(layers: list, meta: list[dict], undone: set) -> tuple[list, list[dict]]:
    """H12: the layers with the splits in `undone` put back whole (inner part -> the layer
    it came from, outer part dropped)."""
    out, out_meta = [], []
    for lay, mm in zip(layers, meta):
        key = mm.get("split")
        if key is None or key not in undone:
            out.append(lay)
            out_meta.append(mm)
        elif "whole" in mm:
            out.append(mm["whole"][0])
            out_meta.append(mm["whole"][1])
    return out, out_meta


def _split_and_explain(layers: list, meta: list[dict], pol: np.ndarray, radii: np.ndarray, win: np.ndarray,
                       fr: Frame, ccx: float, ccy: float) -> tuple[list, list[dict]]:
    """H12: the layers split by radius, stars read from outlines and trims explained, with
    each split standing only when its outer part became another layer's trims (edge
    strips); a split whose outer part stays a layer of its own is undone and the outline
    and trims steps run again on the layers as they were, so an undone split leaves no
    trace. Over the 76 photos the splits whose outer part stayed a layer of its own were
    loops' notched ends (S56-17B) and striped loop borders (S57-17), not new lobes."""
    split, split_meta = _split_by_radius(layers, meta, pol, radii, win, fr, ccx, ccy)
    undone: set = set()
    while True:
        cur, cur_meta = _restore(split, split_meta, undone)
        cur = list(cur)
        _star_from_outline(cur, cur_meta, pol, radii, win, fr, ccx, ccy)
        cur, cur_meta = explain_trims(cur, cur_meta, pol, radii, win, fr, RING_MAX_GROWTH_UNITS)
        stay = {mm["split"] for mm in cur_meta if mm.get("outer") and not mm.get("trim")} - undone
        if not stay:
            return cur, cur_meta
        undone |= stay


def _convex_star(params: dict) -> bool:
    """H11: a star without concave notches (STAR_CONVEX): drawn it would be a polygon."""
    n = int(params["count"])
    return (params.get("kind") == "star" and n >= 2
            and float(params["r_inner"]) >= STAR_CONVEX * float(params["r_outer"]) * np.cos(np.pi / n))


def _hidden_valleys(params: dict, zone_start: float) -> dict | None:
    """H11: the star with its valleys moved under the layers inside its zone (they were
    read where its visible lobes are widest): at the zone's start when that gives notches,
    else at STAR_HIDDEN_DEPTH of the convex limit when that still lies inside the zone's
    start (hidden); None when neither does."""
    deeper = dict(params, r_inner=r3(zone_start))
    if not _convex_star(deeper):
        return deeper
    n = int(params["count"])
    r_in = STAR_HIDDEN_DEPTH * float(params["r_outer"]) * float(np.cos(np.pi / n))
    return dict(params, r_inner=r3(r_in)) if r_in < zone_start else None


def _alternating(k: int, layers: list, meta: list[dict]) -> bool:
    """H11: layer k and another points layer of the same zone, count and radii (within
    ALT_RADIUS_UNITS) whose lobes sit between each other's: a two-tone pleated edge whose
    colors alternate, kept as points (as rings one color would hide the other)."""
    q = layers[k][3]
    n = int(q["count"])
    for j, lay in enumerate(layers):
        p2 = lay[3]
        if (j == k or lay[0] != "points" or meta[j]["zone"] != meta[k]["zone"] or int(p2.get("count", 0)) != n
                or abs(float(p2["r_inner"]) - float(q["r_inner"])) > ALT_RADIUS_UNITS
                or abs(float(p2["r_outer"]) - float(q["r_outer"])) > ALT_RADIUS_UNITS):
            continue
        off = (float(p2["rotation_deg"]) - float(q["rotation_deg"])) % (360.0 / n) / (360.0 / n)
        if abs(off - 0.5) <= ALT_OFFSET:
            return True
    return False


def _pleat_layer(params: dict, zone: tuple[int, int], bands: list[_Band]) -> bool:
    """H11 Task 4: a high-count points layer filling its spacing between two bands is a
    ring's pleated edge (PLEAT_*)."""
    n = int(params["count"])
    if n < PLEAT_MIN_COUNT or params.get("kind") == "star":
        return False
    a, b = zone
    if not (any(bd.i1 == a for bd in bands) and any(bd.i0 == b for bd in bands)):
        return False
    ri, ro = float(params["r_inner"]), float(params["r_outer"])
    w = float(params["width_ratio"]) * (ro - ri)
    peak = {"diamond": 0.4, "petal": 0.5, "bow": 0.7}.get(params.get("kind"), 0.5)
    return w >= PLEAT_FILL * 2 * np.pi * (ri + peak * (ro - ri)) / n


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


def _part_geom(q: Part):
    """Shapely geometry (canvas units) of a drawn head part, or None."""
    a = q.params
    if q.role == "points":
        shapes = points_ring(a["cx"], a["cy"], a["kind"], int(a["count"]), a["r_inner"], a["r_outer"],
                             a["rotation_deg"], a["width_ratio"],
                             None if a.get("inner_ratio") is None else float(a["inner_ratio"]),
                             None if a.get("edge_width") is None else float(a["edge_width"]))
    elif q.role == "ring":
        shapes = [(pinked_ring if int(a.get("pinked", 0)) else ring)(a["cx"], a["cy"], a["r_outer"], a["r_inner"])]
    elif q.role in ("center", "unknown"):
        shapes = [disc(a["cx"], a["cy"], a["r"])]
    else:
        return None
    return unary_union([to_geom(sh) for sh in shapes])


def stack_votes(parts: list[Part], labels: np.ndarray, p: Prepared, fr: Frame,
                trims: set[int]) -> list[tuple[int, int, int, int]]:
    """H12: (index a, index b, samples of a's label, samples of b's label) over the drawn
    overlap of every pair of non-trim points parts of different colors (STACK_*)."""
    geoms = {k: _part_geom(q) for k, q in enumerate(parts) if q.role in ("points", "ring", "center", "unknown")}
    pts = [k for k, q in enumerate(parts) if q.role == "points" and id(q) not in trims]
    cover = [geoms[k] for k, q in enumerate(parts) if q.role in ("ring", "center", "unknown") and geoms.get(k)]
    cover = unary_union(cover) if cover else None
    cx, cy, _ = p.head
    h, w = labels.shape
    out = []
    for i, a in enumerate(pts):
        for b in pts[i + 1:]:
            ga, gb = parts[a].color_group, parts[b].color_group
            if ga == gb:
                continue
            o = geoms[a].intersection(geoms[b])
            if o.is_empty:
                continue
            others = [geoms[k] for k, q in enumerate(parts)
                      if k not in (a, b) and q.role == "points" and q.color_group in (ga, gb)]
            cut = [g for g in [cover, *others] if g is not None]
            if cut:
                o = o.difference(unary_union(cut))
            if o.is_empty or o.area < 1.0:
                continue
            x0, y0, x1, y1 = o.bounds
            xs, ys = np.meshgrid(np.arange(np.floor(x0), x1 + 1), np.arange(np.floor(y0), y1 + 1))
            xs, ys = xs.ravel(), ys.ravel()
            inside = contains_xy(o, xs, ys)
            px = np.rint(fr.axis_x + (xs[inside] - 300.0) * fr.scale).astype(int)
            py = np.rint(fr.py(ys[inside])).astype(int)
            ang = np.degrees(np.arctan2(py - cy, px - cx))
            ok = (px >= 0) & (px < w) & (py >= 0) & (py < h) & (np.abs(ang - 90.0) > TAIL_SECTOR_DEG)
            lab = labels[py[ok], px[ok]]
            out.append((a, b, int((lab == ga).sum()), int((lab == gb).sum())))
    return out


def photo_stacking(parts: list[Part], labels: np.ndarray, p: Prepared, fr: Frame, trims: set[int]) -> None:
    """H12: points layers re-stacked as the photo shows them where they overlap (STACK_*);
    only their z values change."""
    pts = [k for k, q in enumerate(parts) if q.role == "points"]
    if len(pts) < 2:
        return
    edges = []  # (strength, lower, upper)
    decided = set()
    for a, b, na, nb in stack_votes(parts, labels, p, fr, trims):
        n = na + nb
        if n < STACK_MIN_UNITS2:
            continue
        share = na / n
        if share >= STACK_SHARE:
            edges.append((share, b, a))
        elif share <= 1 - STACK_SHARE:
            edges.append((1 - share, a, b))
        else:
            continue
        decided.add(frozenset((a, b)))
    hosts = [k for k in pts if id(parts[k]) not in trims]
    nested = [(b, a) for a, b in nested_pairs(parts, hosts) if frozenset((a, b)) not in decided]
    edges += [(STACK_SHARE / 2, lo, up) for lo, up in nested]  # after every photo decision
    if not edges:
        return
    prior = {k: i for i, k in enumerate(sorted(hosts, key=lambda k: (parts[k].z, k)))}
    above: dict[int, set[int]] = {k: set() for k in hosts}  # lower -> uppers

    def reaches(x: int, y: int) -> bool:
        seen, todo = set(), [x]
        while todo:
            u = todo.pop()
            if u == y:
                return True
            if u not in seen:
                seen.add(u)
                todo.extend(above[u])
        return False

    for _, lo, up in sorted(edges, key=lambda e: -e[0]):
        if not reaches(up, lo):
            above[lo].add(up)
    below = {k: sum(k in above[j] for j in hosts) for k in hosts}
    order, ready = [], [k for k in hosts if below[k] == 0]
    while ready:
        k = min(ready, key=prior.get)
        ready.remove(k)
        order.append(k)
        for u in above[k]:
            below[u] -= 1
            if below[u] == 0:
                ready.append(u)
    # trims next to their host (under it when they were under it)
    host_of = {}
    for t in pts:
        if id(parts[t]) in trims:
            q = parts[t]
            host_of[t] = min(hosts, key=lambda k: (parts[k].params.get("count") != q.params.get("count"),
                                                   parts[k].params.get("rotation_deg") != q.params.get("rotation_deg"),
                                                   abs(parts[k].z - q.z), k))
    full = []
    for k in order:
        full += [t for t in pts if host_of.get(t) == k and parts[t].z < parts[k].z]
        full.append(k)
        full += [t for t in pts if host_of.get(t) == k and parts[t].z >= parts[k].z]
    zs = sorted(parts[k].z for k in pts)
    for k, z in zip(full, zs):
        parts[k].z = z


def nested_pairs(parts: list[Part], hosts: list[int]) -> list[tuple[int, int]]:
    """H12: (front, back) indices of concentric points layers of one count and different
    colors whose tips are nested (NEST_*)."""
    out = []
    for a in hosts:
        for b in hosts:
            qa, qb = parts[a].params, parts[b].params
            if (a == b or parts[a].color_group == parts[b].color_group or int(qa["count"]) != int(qb["count"])
                    or abs(float(qa["cx"]) - float(qb["cx"])) > 1 or abs(float(qa["cy"]) - float(qb["cy"])) > 1):
                continue
            sp = 360.0 / int(qa["count"])
            d = (float(qa["rotation_deg"]) - float(qb["rotation_deg"])) % sp
            if min(d, sp - d) <= NEST_AXIS_DEG and float(qa["r_outer"]) + NEST_UNITS <= float(qb["r_outer"]):
                out.append((a, b))
    return out
