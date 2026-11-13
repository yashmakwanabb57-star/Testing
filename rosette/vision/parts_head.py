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

import logging
from dataclasses import dataclass, replace

import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from shapely import contains_xy, unary_union

from rosette.geometry import disc, pinked_ring, points_ring, ring
from rosette.geometry.common import TOOTH
from rosette.geometry.clip import to_geom
from rosette.model import Part
from rosette.vision.colors import RIM_TOUCH
from rosette.vision.parts_common import Frame, blob_rings, majority, polar_sample, r3
from rosette.vision.parts_trims import (BACK_GAP_UNITS, BACK_MIN_UNITS, BORDER_MIN_SIDES, _axis_cols,
                                       _backing_tips, _band_units, _border_bands, _rows, explain_trims,
                                       fold_specks, split_borders)
from rosette.vision.prepare import Prepared
from rosette.vision.templates import Piece as _Outline
from rosette.vision.templates import _piece_word

log = logging.getLogger(__name__)

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
# A16 (C6): information line of a two-tone pleated edge drawn as one ring (`alt_ring_info`); its prefix
# lets pipeline.use_head replace it for the drawn head.
ALT_INFO_PREFIX = "Two-tone pleated edge:"
ALT_INFO = ALT_INFO_PREFIX + " {part} is drawn in one color; the other color (group {group}) is left out."
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
# A16 (C1): runs of a layer's color outside its full rows (`_runs_beyond_band`): RUNS_MIN..RUNS_MAX
# per circle (more are a pleated edge's teeth) over at least RUNS_SPAN head radii.
RUNS_MIN, RUNS_MAX, RUNS_SPAN = 3, 12, 0.15
# A16 (C8): a layer whose runs (scaled from the visible arc to the circle) stay within RUNCOUNT_TOL
# of a multiple of its order on RUNCOUNT_ROWS consecutive circles has that many lobes (`_run_count`).
RUNCOUNT_TOL, RUNCOUNT_ROWS = 1.0, 3
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
# A16 (C4): stars of STAR_FEW or fewer points (S3-15, S18-15, S18-15B, S32-15: 4-point stars drawn as
# near-squares at 0.85-0.93 of the limit) use STAR_CONVEX_FEW and STAR_HIDDEN_FEW (under it, so a
# moved star has notches by the same bar).
STAR_FEW, STAR_CONVEX_FEW, STAR_HIDDEN_FEW = 4, 0.84, 0.8
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
# ML-2 (pieces, `_front_pieces`): a SAM 2 piece of the head disc's color that lies in the head and
# does not touch the rim (under RIM_TOUCH of its border pixels within RIM_LABEL_UNITS of the head
# circle or beyond it) is read over the full circle (no tail can hide or join a piece inside the
# head; the tails window is for color labels: A16-17's bottom tip lies in it). Its count n is the
# strongest harmonic (n in STAR_MIN_ORDER..PLEAT_MIN_COUNT - 1) of its outline's radial profile
# (A16-17: 12.1 units at 8 against 8.2 at 9); the turned self-correlation of its rows that are not
# full must reach SPLIT_FLOOR at n, and the outline must show n maxima of PIECE_PROMINENCE_UNITS or
# more (else the piece is refused, the reason logged). Then it is a front layer in the base color
# read from its own outline: tips its radial maxima, valleys its minima (medians over the axes),
# rotation the n-th harmonic of the outline. The pieces of one other color that lie inside its convex hull (>= PIECE_INSIDE of
# their pixels each), read together, are its inner layer when they have the same n and axes within
# BACK_TOL_DEG (one layer sewn on another on one set of axes: the nesting rule's tolerance,
# NEST_AXIS_DEG); the inner layer is drawn on the front's axes, above it. A front piece whose valleys
# lie within MIN_ZONE_UNITS of the middle is not a layer (S5-15: the center badge's print read as
# three lobes).
BACK_TOL_DEG = 6.0
PIECE_INSIDE = 0.95
# R18 (`_arm_sets`, `_front_pieces`): a star or petal layer in the disc's own color with no border is
# often cut by SAM 2 into one piece per arm (S42-15, S43-15: six turquoise arms, each refused alone,
# one lobe against a 3-fold harmonic). The disc-color head pieces refused alone that do not touch the
# rim and do not hold the middle (their ARM_SPAN_PCT percentile radius beyond MIN_ZONE_UNITS) are
# grouped by their radial span (single linkage: inner and outer radius, ARM_SPAN_PCT and
# 100 - ARM_SPAN_PCT percentiles, each within MIN_ZONE_UNITS of a member's, then each within
# MIN_ZONE_UNITS of the group's median); a group of
# STAR_MIN_ORDER or more is read together as one piece through the same bars. A base-color ring
# inside a piece front's span (from its first lobed row to its tips) is the front's own body read as
# a band and is not drawn (S42-15, S43-15: the turquoise band where the arms meet).
ARM_SPAN_PCT = 2.0
PIECE_PROMINENCE_UNITS = MIN_ZONE_UNITS / 2
# ML-4 (`harmonic_count`): an outline's second strongest harmonic is reported when its amplitude
# reaches this share of the first (a 2n or n/2 count the judge may try).
HARMONIC_SECOND = 0.6
# ML-2 (`stack_votes` with pieces): a piece belongs to the smallest drawn layer of its color whose
# geometry holds its centroid and at least PIECE_OWN_SHARE of its pixels (the head disc counts as a
# drawn layer, so the disc's own piece is not given to a star of its color drawn on it).
PIECE_OWN_SHARE = 0.5
# ML-2 switches for the two places pieces enter the detector (gates: docs/pipeline.md ML-2).
# PIECE_STACKING is off: on S29-15 the vote on visible pieces puts the loops over the stars (621 / 238
# piece pixels; labels 698 / 2624), against the photo, where the stars are whole over the loops: the
# stars' pieces cover a small part of what they show. On S28-15 it gives the photo's order (star over
# the loops, 10 / 1010), and the color score drops there through the star's wide body (valleys at
# 69.5 units), a geometry item. Off, the vote counts the labels as before and a front read from a
# piece casts no vote (H16). docs/pipeline.md ML-2.
PIECE_FRONT, PIECE_STACKING = True, False
# R19 (`_loose_votes`): with PIECE_STACKING off a front in the disc's color read from pieces with no
# backing (a loose front, `_loose_piece_fronts`) casts no label vote: its label cannot be told from
# the disc's. Its own pieces still decide its pair with a layer that has pieces when the piece vote is
# unanimous (every sample on one mask) and the winner's mask holds at least STACK_SHARE of the drawn
# overlap's samples: a vote from mask edges grazing the overlap is no vote. Mined on the 76 photos
# (docs/pipeline.md R19, fix round 1): of the 35 pairs voted with pieces on both sides, the votes
# whose masks only graze the overlap hold 0.007-0.018 of it (S18-15B 31 of 4729, S32-15 14 of 772,
# S3-15 9 of 732: 37 samples at most, so a bare sample count cannot tell them from S43-15's 13); the
# other 31 hold 0.18-1.00 on the winner's mask (S29-15 0.18 ... S47-15 0.93, S18-15 and S53-19 1.00);
# S43-15's turquoise arms hold 13 of the 13 (1.00, 12 of them 3 units or more inside the mask), the
# only unanimous vote under STACK_MIN_UNITS2. The bar STACK_SHARE stands 0.63 over the grazing votes.
# R17 (`_tip_lobes`, `_tips_to_stars`): a points layer read as blocks, kites or petals whose lobes
# are the TIPS of a star whose body is covered by other layers is that star. Its outermost lobe (the
# last run of rows its label holds through the lobe axes, the median over the visible axes) must
# taper to a point: the widest row within the inner TIP_PEAK of the lobe, the width beyond it
# falling along its chord (at most TIP_BULGE above it on average: a block keeps its width and then
# drops), the last row at most TIP_END of the widest; the lobe rows n-fold (SPLIT_FLOOR). And its
# inner end must be cut by a layer drawn above it, either (A, switched off: TIP_CUT_A) where the
# lobe is wide (its first row at least TIP_CUT_WIDE of the widest) by a ring, the center or a
# higher points layer on
# >= COVER_CUT_SHARE of the visible axes (walking in along each axis while the label holds, up to
# TIP_WALK_UNITS), or (B) by a higher points layer of another color, same count, on axes half a step
# off (within NEST_AXIS_DEG: the lobes sit in its valleys) that holds the labels beside the lobe
# where it narrows inward on >= TIP_SIDE of them. Mined on the 76 photos: docs/pipeline.md R17.
TIP_PEAK, TIP_BULGE, TIP_END = 0.55, 0.05, 0.5
TIP_CUT_WIDE, TIP_SIDE, TIP_WALK_UNITS = 0.6, 0.9, 20.0
# Cut (A) is off: it fires on S37-15, S42-15 and S45-15 (gold, white and light-blue tips beyond a
# ring), and on all three the lobes narrow inward beside labels that are no drawn cover (side 0.17,
# 0.33, 0.00: S37-15's and S42-15's tips are cut by a front layer in the disc's own color, which is
# not drawn), so the whole star shows its color where that layer lies. The new-image color mean fell
# 0.855 -> 0.854 (S37-15 0.949 -> 0.921, S45-15 0.935 -> 0.930) against its gate. docs/pipeline.md R17.
TIP_CUT_A = False
# R19 (`_axis_runs`, `_nested_runs`): stars nested on one set of axes (s44-15: gold, black, gold and
# black 4-point stars on the black disc, each sewn on the last; H15 reads the outer two as a front and
# its backing). Along the visible axes of every drawn star (count >= STAR_MIN_ORDER) the label held by
# the most axes on each row, in runs (shorter than MIN_BAND_UNITS joined to the run inside), is read
# from the middle out to the star's tips. A run is a star of its own color, the star's count and axes,
# when it spans >= MIN_ZONE_UNITS and ends >= BACK_MIN_UNITS inside the star's tips, the axes agree on
# it (>= BORDER_MIN_SIDES of them per row, mean), its rows are n-fold (SPLIT_FLOOR) and no full band
# (the color's mean coverage of those rows under FULL_COV: a pleated ring or a star's body, S80-22),
# it does not lie on a drawn ring or center of its color (>= RING_MAJORITY of its span), the next run
# out is of another color, n-fold, held by the axes and >= BACK_MIN_UNITS long (a run that meets the
# background or an unordered band is no tip: S15-15), a run of the star's own color has a run of that
# color outside it (else it is the star itself: A15-17, S58-17, s39-15), and no drawn layer of its
# color on those axes ends within MIN_ZONE_UNITS of it. Each is drawn from the star's r_inner to the
# run's end, straight-sided (`_straight_valley`, H16), on top of the drawn layers on those axes, the
# shorter above (H12 nested rule). Mined on the 76 photos: docs/pipeline.md R19.
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
# H15: id prefix of a front layer in the base color and its backing (`_front_in_base`); they do
# not set the count the head is re-centered on (parts.detect_parts), which is read before them.
FRONT_ID = "front-"
# H15: a front layer holds its own axes on >= FRONT_AXIS_SHARE of its rows (visible lobes): S70-19
# 0.45, S41-15 0.98, s44-15 0.63, S68-19 0.93; S46-15 0.14, where the "wedges" are its disc and loop
# edges showing between a gold star's arms and the loops.
FRONT_AXIS_SHARE = 0.3
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


def _harmonic_rotation(ind: np.ndarray, win: np.ndarray, n: int) -> float:
    """Rotation (deg) of an n-fold (radii, theta) indicator: the phase of its n-th harmonic."""
    x = ind.astype(np.float32)
    th = np.linspace(0, 2 * np.pi, x.shape[1], endpoint=False)
    c = (x[:, win] * np.exp(-1j * n * th[win])[None]).sum()
    return float((-np.degrees(np.angle(c)) / n) % (360.0 / n))


def _runs_beyond_band(ind: np.ndarray, win: np.ndarray, cov: np.ndarray, rr: np.ndarray, fr: Frame,
                      r_units: float) -> tuple[int, int, float, float] | None:
    """A16 (C1): a layer whose rows have no rotational symmetry as a whole (its lobes overlap into
    a full band near their base, S55-15's loops) is still a points layer when, outside its full
    rows, its color forms RUNS_MIN..RUNS_MAX separate runs on the visible arc of every circle over
    a stretch of at least RUNS_SPAN head radii (and 3 circles), whose turned self-correlation
    reaches SYM_T at an order n in that range. Returns (end row offset of the stretch, n, score,
    rotation deg) or None."""
    vis = np.flatnonzero(win)
    x = ind[:, vis].astype(np.int8)
    runs = (np.diff(x, axis=1) == 1).sum(axis=1) + x[:, 0]
    good = (runs >= RUNS_MIN) & (runs <= RUNS_MAX) & (cov < FULL_COV) & (cov > EXTENT_COV)
    full = np.flatnonzero(cov >= FULL_COV)
    if full.size:
        good[:full[-1] + 1] = False  # beyond the band: outside its last full row
    best, i = None, 0
    while i < len(good):
        if not good[i]:
            i += 1
            continue
        j = i
        while j < len(good) and good[j]:
            j += 1
        if best is None or j - i > best[1] - best[0]:
            best = (i, j)
        i = j
    if best is None:
        return None
    i, j = best
    if j - i < 3 or fr.d(rr[j - 1] - rr[i]) < RUNS_SPAN * r_units:
        return None
    n, sc, rot = _symmetry(ind[i:j], win)
    if not (RUNS_MIN <= n <= RUNS_MAX and sc >= SYM_T):
        return None
    return j, n, sc, rot


def _run_count(ind: np.ndarray, win: np.ndarray, n: int, rot: float) -> tuple[int, float]:
    """A16 (C8): (count, rotation) of a points layer of order n whose color forms, on at least
    RUNCOUNT_ROWS consecutive circles, a number of runs that stays within RUNCOUNT_TOL of k n lobes
    (k >= 2, under PLEAT_MIN_COUNT; the runs counted on the visible arc and scaled to the circle)
    and whose turned self-correlation at k n (over the whole layer) reaches SPLIT_FLOOR and lies
    within SPLIT_MARGIN of the one at n (a true k n-fold layer scores as high at its divisor n;
    loops' edges and stripes that double the runs do not): the lower star of layered stars
    shows all its points where the upper one's order (a divisor) won the symmetry (S49-17's 12-point
    star read as 4). The largest such multiple wins; else (n, rot)."""
    vis = np.flatnonzero(win)
    x = ind[:, vis].astype(np.int8)
    runs = (np.diff(x, axis=1) == 1).sum(axis=1) + x[:, 0]
    est = runs * ind.shape[1] / max(len(vis), 1)  # lobes on the whole circle
    x32 = ind.astype(np.float32)
    sn = _score_at(x32, win, n)
    for m in sorted((k * n for k in range(2, PLEAT_MIN_COUNT) if k * n < PLEAT_MIN_COUNT), reverse=True):
        near = np.abs(est - m) <= RUNCOUNT_TOL
        best, i = 0, 0
        while i < len(near):
            j = i
            while j < len(near) and near[j]:
                j += 1
            best = max(best, j - i)
            i = j + 1
        if best >= RUNCOUNT_ROWS:
            sm = _score_at(x32, win, m)
            if sm >= SPLIT_FLOOR and sm >= sn - SPLIT_MARGIN:
                return m, _harmonic_rotation(ind, win, m)
    return n, rot


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


def detect_head(p: Prepared, labels: np.ndarray, fr: Frame, pieces: list | None = None) -> list[Part]:
    """Head disc, rings, points rings and center from the polar label profile. With SAM 2 pieces
    (ML-2, `color_group` set) front layers in the base color are also read from piece outlines
    (`_front_pieces`) and the stacking vote counts visible pieces (`stack_votes`); without them
    the rules path runs alone."""
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
            if share < LOW_SYM:  # A16 (C1): lobes overlapping into a full band, symmetric beyond it
                lobed = _runs_beyond_band(ind, win, cov[ra:rb, g], radii[ra:rb], fr, fr.d(r))
                if lobed is not None:
                    end, n, share, rot = lobed
                    rb = ra + end
                    ind = pol[ra:rb] == g
                    area = float((cov[ra:rb, g] * 2 * np.pi * radii[ra:rb]).sum() * step)
            if share >= LOW_SYM:  # A16 (C8): layered stars: the runs show a multiple of the order
                n, rot = _run_count(ind, win, n, rot)
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
            if (has_core and _convex_star(params)
                    and pleat_texture(p, labels == g, radii[inner.i0], radii[inner.i1 - 1]) >= PLEAT_TEXTURE):
                has_core = False  # A16 (C4): a pleated band of its color over the hidden valleys: a ring on top
            layers.append(("points", g, share, params, area, rb))
            # a star with a core: the band is this layer's body, the star polygon covers it
            meta.append(dict(zone=(a, b), rows=(ra, rb), eats=inner if has_core else None, inner=inner))

    layers, meta = _split_and_explain(layers, meta, pol, radii, win, fr, ccx, ccy)
    if head is not None:
        _front_in_base(layers, meta, zones, pol, cov, radii, win, fr, head["label"], ccx, ccy)
    layers, meta = split_borders(layers, meta, pol, radii, win, fr)
    layers, meta = fold_specks(layers, meta, bands, head_band, pleated, radii, fr, RING_MAX_GROWTH_UNITS)
    if head is not None and pieces and PIECE_FRONT:  # ML-2: front layers from piece outlines
        _front_pieces(layers, meta, zones, radii, fr, head["label"], p.head, pieces)
    # H11: convex "stars" and pleated edges are rings (annuli), flagged / pinked
    keep, paired = [], set()
    for k, lay in enumerate(layers):
        role, g, share, params = lay[:4]
        if k in paired:
            continue  # A16 (C6): the other color of a two-tone pleated edge drawn as one ring
        if role == "points" and not meta[k].get("trim"):
            convex = _convex_star(params)
            if convex:  # its valleys are hidden under the layers inside its zone
                deeper = _hidden_valleys(params, unit(meta[k]["zone"][0]))
                if deeper is not None:
                    layers[k] = (role, g, share, deeper) + tuple(lay[4:])
                    for j, (lay2, m2) in enumerate(zip(layers, meta)):  # A16: its backing follows it
                        q2 = lay2[3]
                        if (m2.get("trim") and lay2[0] == "points" and int(q2.get("count", 0)) == int(params["count"])
                                and q2.get("r_inner") == params["r_inner"]
                                and abs(float(q2["rotation_deg"]) - float(params["rotation_deg"])) < 1e-6):
                            q2 = dict(q2, r_inner=deeper["r_inner"])  # the same width ratio as its host
                            layers[j] = (lay2[0], lay2[1], lay2[2], q2) + tuple(lay2[4:])
                    keep.append(k)
                    continue
            pleat = not convex and _pleat_layer(params, meta[k]["zone"], bands)
            j = _alternate_partner(k, layers, meta) if pleat else None
            alt = None
            if j is not None:  # A16 (C6, ruling: follow the reference look): one pinked ring in the
                paired.add(j)  # color holding more of the edge; the other is named on it (alt_group)
                if j in keep:
                    keep.remove(j)
                alt = layers[j][1]
                if layers[j][4] > lay[4]:
                    alt, (g, share) = g, (layers[j][1], layers[j][2])
            if convex or pleat:
                i0 = int(np.searchsorted(radii, float(params["r_inner"]) * fr.scale))
                i1 = min(len(radii), int(np.searchsorted(radii, float(params["r_outer"]) * fr.scale)) + 1)
                if i1 > i0:
                    conf = min(share, BLOB_UNCONFIRMED_CONF) if convex else share
                    pleated.append((i0, i1, g, conf, 0 if convex else int(params["count"]), alt))
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
        pid = f"{FRONT_ID}{len(parts)}" if meta[k].get("front") else f"{role}-{len(parts)}"
        parts.append(Part(pid, role, params, g, z, r3(min(1.0, share))))
        if meta[k].get("trim"):
            trims.add(id(parts[-1]))

    if center_band is not None and id(center_band) not in eaten:
        parts.append(Part("center", "center", {"cx": r3(ccx), "cy": r3(ccy), "r": r3(unit(center_band.i1))},
                          center_band.label, 50, r3(center_band.cov)))
    # by outer edge (H11: a pleated annulus can enclose a narrower ring, which lies on it)
    starts = [(float(q.params["r_inner"]), q.color_group) for q in parts if q.role == "points" and id(q) not in trims]
    found = []  # [sort key, i0, i1, label, confidence, pleats, dropped color of a two-tone edge]
    for bd in bands:
        if bd is not head_band and bd is not center_band and id(bd) not in eaten:
            found.append([bd.i1, bd.i0, _ring_outer(bd, bands, dom, cov, radii, fr, starts), bd.label, bd.cov, 0, None])
    for ring in pleated:  # a pleated layer, already measured
        i0, i1, label, conf, pinked, alt = tuple(ring) + (0, None)[len(ring) - 4:]
        found.append([i1, i0, i1, label, conf, pinked, alt])
    _rings_by_dominance(found, dom, domc, fr, radii, center_band)
    if head is not None:  # R18: a base-color band inside a piece front's span is the front's body
        bodies = [m["body"] for m in meta if m.get("body")]
        found = [t for t in found if t[3] != head["label"] or not any(
            lo - MIN_BAND_UNITS <= unit(t[1]) and unit(t[2]) <= hi for lo, hi in bodies)]
    found.sort(key=lambda t: t[0])
    for depth, (_, i0, i1, label, conf, pinked, alt) in enumerate(reversed(found)):  # outermost first, inner on top
        rp = {"cx": r3(ccx), "cy": r3(ccy), "r_outer": r3(unit(i1)), "r_inner": r3(unit(i0))}
        if pinked:  # a pleated edge (H11): pinked, with its pleat count (re-centering)
            rp["pinked"], rp["pleats"] = 1, int(pinked)
        if alt is not None:  # A16 (C6): the color the one-color ring leaves out (alt_ring_info)
            rp["alt_group"] = int(alt)
        elif pleat_texture(p, labels == label, unit(i0) * fr.scale, unit(i1) * fr.scale) >= PLEAT_TEXTURE:
            rp["pinked"] = 1  # ML-1: a pleated ribbon band (the files draw these pinked)
        parts.append(Part(f"ring-{depth}", "ring", rp, label, min(49, 30 + depth), r3(conf)))
    head_label = head["label"] if head else -2
    vote = pieces if PIECE_STACKING else None
    loose = pieces if PIECE_FRONT and not PIECE_STACKING else None  # R19: the loose fronts' pieces
    pre_cover = _finish_points(parts, labels, p, fr, pol, radii, win, head_label, trims, vote, loose)
    if _tips_to_stars(parts, pol, radii, win, fr, trims, pieces, p):  # R17: tips of covered stars
        _revote(parts, labels, p, fr, pol, radii, win, head_label, trims, pre_cover, vote, loose)
    _nested_runs(parts, pol, radii, win, fr)  # R19: stars nested along a drawn star's axes
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


def alt_ring_info(parts: list[Part]) -> list[str]:
    """A16 (C6): an information line (not a flag) for each ring drawn in one color from a
    two-tone pleated edge, naming the color left out (its `alt_group`)."""
    return [ALT_INFO.format(part=q.id, group=int(q.params["alt_group"]))
            for q in parts if q.role == "ring" and q.params.get("alt_group") is not None]


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


def _axis_halfwidth(row: np.ndarray, c: int, h: int, reach: int) -> int:
    """Columns of label h on either side of column c along one polar row, through c (0 when the
    axis holds another label): the lobe's angular width there, in columns."""
    t = row.shape[0]
    if row[c] != h:
        return 0
    n = 1
    for d in (1, -1):
        k = 1
        while k <= reach and row[(c + d * k) % t] == h:
            k += 1
        n += k - 1
    return n


def _straight_valley(pol: np.ndarray, radii: np.ndarray, rows: range, h: int, params: dict,
                     win: np.ndarray, limit: np.ndarray | None = None, fr: Frame | None = None) -> float | None:
    """H16: valley radius (px) of the straight-sided star of `params`' count and axes whose sides
    best fit the edge of h seen on `rows`: h's run through each visible axis gives the side's
    lateral offset y = r sin(phi) at x = r cos(phi) along the axis (`limit` caps phi per row,
    radians: the footprint h is seen in); the least-squares line y = p - m x (m > 0: the side
    closes outward) meets the valley ray (180 deg / count off the axis) at the valley. None
    without such a line. With `fr` (a backing's own tips, H16 fix round): one point per row,
    the median of h's run over the axes holding it there (a run jumping on one axis is dropped),
    and the points must come from >= BORDER_MIN_SIDES of the visible axes and span
    >= BACK_MIN_UNITS radially, else None."""
    t, n = pol.shape[1], int(params["count"])
    a = np.pi / n
    reach = t // (2 * n)
    cols_of = [c for c in _axis_cols(params, t) if win[c]]
    runs = np.array([[_axis_halfwidth(pol[i], c, h, reach) for c in cols_of] for i in rows],
                    dtype=float).reshape(len(rows), len(cols_of))
    if fr is not None:
        held = runs > 0
        used = np.flatnonzero(held.any(axis=1))
        if (not cols_of or used.size == 0 or held.any(axis=0).sum() < BORDER_MIN_SIDES * len(cols_of)
                or fr.d(radii[rows.start + used[-1]] - radii[rows.start + used[0]]) < BACK_MIN_UNITS):
            return None
        runs = np.array([[np.median(runs[j][held[j]])] if held[j].any() else [0.0] for j in range(len(rows))])
    xs, ys = [], []
    for j, i in enumerate(rows):
        for cols in runs[j]:
            if cols == 0:
                continue
            phi = min(cols / 2 * 2 * np.pi / t, a)
            if limit is not None:
                phi = min(phi, float(limit[j]))
            xs.append(radii[i] * np.cos(phi))
            ys.append(radii[i] * np.sin(phi))
    if len(xs) < 3 or np.ptp(xs) <= 0:
        return None
    slope, p = np.polyfit(np.asarray(xs), np.asarray(ys), 1)
    m = -float(slope)
    if m <= 0:
        return None
    x = p / (np.tan(a) + m)
    return float(x / np.cos(a))


def _star_halfangle(r: np.ndarray, r_tip: float, r_valley: float, n: int) -> np.ndarray:
    """Half-angle (radians) of a straight-sided n-point star (tips r_tip, valleys r_valley) at
    radii r: the angle off a tip's axis up to which its side reaches each radius. A convex star's
    side dips under its valley radius before the valley: the running minimum of the side radius
    is followed to its first lowest point, and a radius at or under it is reached over the whole
    half-angle."""
    a = np.pi / n
    phi = np.linspace(0.0, a, 361)
    rad = np.minimum.accumulate(_star_side(phi, r_tip, r_valley, n))
    k = int(np.argmin(rad))  # first lowest point: rad strictly falls up to it
    r = np.asarray(r, dtype=float)
    out = np.interp(np.clip(r, rad[k], r_tip), rad[k::-1], phi[k::-1])
    return np.where(r <= rad[k], a, out)


def _inner_star(pol: np.ndarray, radii: np.ndarray, a: int, tip: int, h: int, front: dict, win: np.ndarray,
                fr: Frame) -> tuple[dict, float, int] | None:
    """H16: an inner star of the backing color h on the front star (S70-19: white under red under
    white, nested chevrons at each tip). On the front star's axes h shows from the zone start (within
    a pleat's depth, RING_MAX_GROWTH_UNITS: the ring's pleated edge lies over its base) outward to
    a radius t_in at least BACK_MIN_UNITS inside the front's tips (the gaps of BACK_GAP_UNITS in it
    bridged), on >= BORDER_MIN_SIDES of the visible axes; h over the rows to t_in is n-fold
    (turned self-correlation >= SPLIT_FLOOR). Then (params, confidence, outer row): a star of
    the front's count and axes from its r_inner to t_in (median over those axes), its width that
    of the straight-sided star through h's edge inside the front's footprint; confidence
    min(axis share, symmetry)."""
    t, n = pol.shape[1], int(front["count"])
    slack, gap = _rows(RING_MAX_GROWTH_UNITS, radii, fr), _rows(BACK_GAP_UNITS, radii, fr)
    ends, seen = [], 0
    for c in _axis_cols(front, t):
        if not win[c]:
            continue
        seen += 1
        has = pol[a:tip, c] == h
        first = np.flatnonzero(has[:slack + 1])
        if first.size == 0:
            continue
        i = int(first[0])
        while i + 1 < len(has) and has[i + 1:i + 1 + gap].any():
            i += 1
        if fr.d(radii[tip - 1] - radii[a + i]) >= BACK_MIN_UNITS:
            ends.append(a + i)
    share = len(ends) / seen if seen else 0.0
    if share < BORDER_MIN_SIDES:
        return None
    end = int(np.median(ends))
    sym = _score_at((pol[a:end + 1] == h).astype(np.float32), win, n)
    if sym < SPLIT_FLOOR:
        return None
    r_in, r_out = float(front["r_inner"]), fr.d(radii[end])
    if r_out - r_in < MIN_ZONE_UNITS:
        return None
    rows = range(a, end + 1)
    cap = _star_halfangle(radii[a:end + 1], float(radii[tip - 1]), r_in * fr.scale, n)
    v = _straight_valley(pol, radii, rows, h, front, win, cap)
    v = r_in if v is None else max(r_in, fr.d(v))
    params = dict(front, r_outer=r3(r_out), width_ratio=r3(2 * np.pi * v / n / max(r_out - r_in, 1e-6)))
    return params, min(share, sym), end + 1


def _front_in_base(layers: list, meta: list[dict], zones: list[tuple[int, int]], pol: np.ndarray,
                   cov: np.ndarray, radii: np.ndarray, win: np.ndarray, fr: Frame, base: int,
                   ccx: float, ccy: float) -> None:
    """H15: in a zone between bands, the base disc's color forming n-fold wedges (3 <= n < 16,
    PLEAT_MIN_COUNT: more are a pleated edge's teeth; n the order of its rows that are not
    full, whose turned self-correlation must reach SPLIT_FLOOR: the disc's pleats and the gaps
    between loops blur it, S70-19 reads 0.28) whose sides carry a thin
    border of another color (the H13 border geometry, `_border_bands`, on >= BORDER_MIN_SIDES
    of the sides) is a points layer of the base color in front: a star of n lobes from the
    zone's start (the ring or center it meets) to the last row its border runs along, and the
    border color, when it shows at least BACK_MIN_UNITS beyond those tips, is its backing (a trim: the star wider by the border on each side, reaching
    the border's tips behind it). The border color must be read as a points layer of its own
    in the zone with n lobes or a multiple of n (S68-19's irregular lilac chevrons, read as
    unknown parts, are not a backing star). A layer of the border color on the same axes inside that
    reach (the border's tips read alone) gives way to the backing. S70-19: a red 6-point star
    on a white star over blue loops on the red pleated disc; the white showed as chevrons."""
    t = pol.shape[1]
    for a, b in zones:
        if fr.d(radii[b - 1] - radii[a]) < MIN_ZONE_UNITS:
            continue
        rows = np.flatnonzero((cov[a:b, base] > EXTENT_COV) & (cov[a:b, base] < FULL_COV))
        if rows.size < 2:
            continue
        ra, rb = a + rows[0], a + rows[-1] + 1
        n, share, rot = _symmetry(pol[ra:rb] == base, win)
        if not STAR_MIN_ORDER <= n < PLEAT_MIN_COUNT or share < SPLIT_FLOOR:  # 16+ teeth: a pleated edge
            continue
        probe = {"count": n, "rotation_deg": rot}
        best = None
        for h in sorted({int(g) for g in np.unique(pol[a:b]) if g >= 0 and g != base}):
            frac, mark = _border_bands(pol, a, b, base, h, probe, win, radii, fr, span=True, front=True)
            if frac >= BORDER_MIN_SIDES and (best is None or frac > best[0]):
                best = (frac, h, mark)
        if best is None:
            continue
        frac, h, mark = best
        if not any(lay[0] == "points" and lay[1] == h and meta[k]["zone"] == (a, b) and not meta[k].get("trim")
                   and int(lay[3]["count"]) % n == 0 for k, lay in enumerate(layers)):
            continue  # the border color is no n-fold layer of its own: not a backing star (S68-19)
        marked = np.flatnonzero(mark.any(axis=1))
        tip = int(marked[-1]) + 1
        r_in, r_out = fr.d(radii[a]), fr.d(radii[tip - 1])
        if r_out - r_in < MIN_ZONE_UNITS:
            continue
        axes = [c for c in (int(round((rot + 360.0 * k / n) % 360 * t / 360)) % t for k in range(n)) if win[c]]
        held = float((pol[a:tip][:, axes] == base).mean()) if axes else 0.0
        if held < FRONT_AXIS_SHARE:
            continue  # the wedges run along their axes: elsewhere the base shows between other layers
        rr = radii[a:tip]
        w = (pol[a:tip] == base)[:, win].mean(axis=1) * 2 * np.pi * rr / n
        front = {"cx": r3(ccx), "cy": r3(ccy), "kind": "star", "count": int(n), "r_inner": r3(r_in),
                 "r_outer": r3(r_out), "rotation_deg": r3(rot),
                 "width_ratio": r3(fr.d(float(w.max())) / max(r_out - r_in, 1e-6))}
        _, back_tip, _ = _backing_tips(pol, radii, tip - 1, h, front, win, fr)
        ro2 = fr.d(radii[max(back_tip, tip - 1)])
        if ro2 - r_out < BACK_MIN_UNITS:
            continue  # a backing shows beyond the front tips; a band ending with them is an edge
        lobes = max(1, sum(bool(win[int(round((rot + 360.0 * k / n) % 360 * t / 360)) % t]) for k in range(n)))
        side = _band_units(mark, radii, range(a, tip), lobes, fr)
        wide = float(front["width_ratio"]) * (r_out - r_in) + 2 * side
        # H16: the backing's width from its own tips beyond the front's (the white read over the
        # front's rows also holds an inner star and the border: S70-19 1.326), when the fit has
        # the evidence (`_straight_valley` with fr) and its valley lies outside the front's: else
        # the front's width plus the band on each side (before H16), never narrower than the front
        v = _straight_valley(pol, radii, range(tip, max(back_tip, tip - 1) + 1), h, front, win, fr=fr)
        if v is not None and 2 * np.pi * fr.d(v) / n >= float(front["width_ratio"]) * (r_out - r_in):
            wide = 2 * np.pi * max(r_in, fr.d(v)) / n
        back = dict(front, r_outer=r3(ro2), width_ratio=r3(wide / max(ro2 - r_in, 1e-6)))
        zone_meta = dict(zone=(a, b), rows=(a, tip), eats=None, inner=None)
        drop = []
        for k, lay in enumerate(layers):
            q = lay[3]
            if (lay[0] == "points" and lay[1] == h and meta[k]["zone"] == (a, b) and int(q["count"]) == n
                    and min((float(q["rotation_deg"]) - rot) % (360.0 / n),
                            (rot - float(q["rotation_deg"])) % (360.0 / n)) <= NEST_AXIS_DEG
                    and float(q["r_outer"]) <= ro2 + MIN_BAND_UNITS):
                drop.append(k)
        for k in reversed(drop):
            del layers[k]
            del meta[k]
        layers.append(("points", base, min(share, frac), front, 0.0, tip))
        meta.append(dict(zone_meta, front=True))
        layers.append(("points", h, min(share, frac), back, 0.0, max(back_tip, tip - 1) + 1))
        meta.append(dict(zone_meta, rows=(a, max(back_tip, tip - 1) + 1), trim=True, front=True))
        inner = _inner_star(pol, radii, a, tip, h, front, win, fr)
        if inner is not None:  # H16: an inner star of the backing's color on the front star
            q, conf, end = inner
            layers.append(("points", h, conf, q, 0.0, end))
            meta.append(dict(zone_meta, rows=(a, end), front=True))



def _touches_rim(mask: np.ndarray, cx: float, cy: float, r: float, fr: Frame) -> bool:
    """ML-2: RIM_TOUCH or more of the piece's border pixels lie within RIM_LABEL_UNITS of the head
    circle or beyond it."""
    border = mask & ~ndimage.binary_erosion(mask)
    ys, xs = np.nonzero(border)
    if len(ys) == 0:
        return True
    return float((np.hypot(xs - cx, ys - cy) >= r - RIM_LABEL_UNITS * fr.scale).mean()) >= RIM_TOUCH


def _bridged(out: np.ndarray) -> np.ndarray | None:
    """The outline with the angles it shows nothing at bridged (periodic); None when it shows < 3."""
    ok = np.isfinite(out)
    if ok.sum() < 3:
        return None
    return np.interp(np.arange(out.shape[0]), np.flatnonzero(ok), out[ok], period=out.shape[0])


def _harmonics(out: np.ndarray) -> dict[int, float]:
    """Amplitude (px) of the outline's n-th harmonic, n in STAR_MIN_ORDER..PLEAT_MIN_COUNT - 1."""
    th = np.linspace(0, 2 * np.pi, out.shape[0], endpoint=False)
    d = out - out.mean()
    return {n: float(2 * abs((d * np.exp(-1j * n * th)).mean())) for n in range(STAR_MIN_ORDER, PLEAT_MIN_COUNT)}


def harmonic_count(out: np.ndarray) -> tuple[int, float, int | None]:
    """ML-4: (n, amplitude, n2) of an outline in polar form: n the strongest harmonic (`_harmonics`,
    the lowest n on a tie), its amplitude (px), and the second strongest harmonic n2 when its
    amplitude is at least HARMONIC_SECOND of the first, else None."""
    amp = _harmonics(out)
    n = max(amp, key=amp.get)
    rest = {k: v for k, v in amp.items() if k != n}
    n2 = max(rest, key=rest.get) if rest else None
    if n2 is None or amp[n] <= 0 or rest[n2] < HARMONIC_SECOND * amp[n]:
        n2 = None
    return n, amp[n], n2


def _peaks(out: np.ndarray, prominence: float, n: int) -> np.ndarray:
    """Columns of the circular outline's maxima standing at least `prominence` (px) above both
    sides, at least half an n-step apart (a tip's ragged top is one maximum)."""
    t = out.shape[0]
    peaks, _ = find_peaks(np.concatenate([out, out, out]), prominence=prominence, distance=max(1, t // (2 * n)))
    return peaks[(peaks >= t) & (peaks < 2 * t)] - t


def _maxima(out: np.ndarray, prominence: float, n: int) -> int:
    """Count of the circular outline's maxima (`_peaks`)."""
    return int(_peaks(out, prominence, n).size)


def _piece_order(ind: np.ndarray, radii: np.ndarray, fr: Frame) -> tuple[int, float] | str:
    """ML-2: (n, score) of a piece's polar indicator read over the full circle: n the strongest
    harmonic of its outline, score the turned self-correlation of its rows that are not full at n.
    A reason (str) instead when no row is lobed (an annulus, a disc), the score is under SPLIT_FLOOR
    or the outline's maxima (PIECE_PROMINENCE_UNITS) are not n (R18: n - 1 maxima all on the
    harmonic's axes within BACK_TOL_DEG count as n: one tip cut by a block, the badge or the rim)."""
    full = np.ones(ind.shape[1], bool)
    cov = ind.mean(axis=1)
    rows = np.flatnonzero((cov > EXTENT_COV) & (cov < FULL_COV))
    if rows.size < 2:
        return "no lobed rows"
    out = _bridged(_outline_radii(ind, radii, full))
    if out is None:
        return "no outline"
    n = harmonic_count(out)[0]
    score = _score_at(ind[rows[0]:rows[-1] + 1].astype(np.float32), full, n)
    if score < SPLIT_FLOOR:
        return f"symmetry {score:.2f} at n={n} under {SPLIT_FLOOR}"
    peaks = _maxima(out, PIECE_PROMINENCE_UNITS * fr.scale, n)
    if peaks == n - 1 and _maxima_on_axes(out, PIECE_PROMINENCE_UNITS * fr.scale, n):
        peaks = n  # R18: one tip cut (by a block, the badge or the rim), the others on the axes
    if peaks != n:
        return f"{peaks} outline maxima against the strongest harmonic n={n}"
    return n, float(score)


def _maxima_on_axes(out: np.ndarray, prominence: float, n: int) -> bool:
    """R18: every maximum of the outline (`_peaks`) lies within BACK_TOL_DEG of an axis of its
    n-th harmonic (`_piece_rotation`)."""
    t = out.shape[0]
    peaks = _peaks(out, prominence, n)
    rot, step = _piece_rotation(out, n), 360.0 / n
    off = (peaks * 360.0 / t - rot) % step
    return bool(peaks.size) and bool((np.minimum(off, step - off) <= BACK_TOL_DEG).all())


def _outline_radii(ind: np.ndarray, radii: np.ndarray, win: np.ndarray) -> np.ndarray:
    """ML-2: the outermost radius (px) of a piece per angle (its outline in polar form); NaN where
    it shows nothing or in the tails sector."""
    on = ind.any(axis=0)
    last = ind.shape[0] - 1 - np.argmax(ind[::-1], axis=0)
    out = radii[last].astype(np.float64)
    out[~on | ~win] = np.nan
    return out


def _piece_rotation(out: np.ndarray, n: int) -> float:
    """ML-2: rotation (deg) of the n-fold outline: the phase of its n-th harmonic (the tips' axes)."""
    th = np.linspace(0, 2 * np.pi, out.shape[0], endpoint=False)
    ok = np.isfinite(out)
    c = ((out[ok] - out[ok].mean()) * np.exp(-1j * n * th[ok])).sum()
    return float((-np.degrees(np.angle(c)) / n) % (360.0 / n))


def _extremes(out: np.ndarray, n: int, rot: float, offset: float, fn) -> list[float]:
    """ML-2: fn (max: tips, min: valleys) of the outline within a quarter step of each axis
    rot + (k + offset) * 360 / n whose own column is visible."""
    t = out.shape[0]
    half = max(1, int(round(t / n / 4)))
    vals = []
    for k in range(n):
        c = int(round((rot + 360.0 * (k + offset) / n) % 360 * t / 360)) % t
        if not np.isfinite(out[c]):
            continue
        v = out[np.arange(c - half, c + half + 1) % t]
        vals.append(float(fn(v[np.isfinite(v)])))
    return vals


def _piece_layer(ind: np.ndarray, radii: np.ndarray, win: np.ndarray, fr: Frame,
                 head: tuple[float, float, float], n: int, rot: float) -> tuple[dict, int] | None:
    """ML-2: (params, outer row) of an n-fold layer on axes `rot` read from a piece's outline in polar
    form: r_outer the median tip radius, r_inner the median valley radius, kind by the head card's
    fitter on that outline (templates._piece_word: a star when it has deep lobes around the center),
    else the lobe-width fitter `_kind`; None when the tips do not stand MIN_ZONE_UNITS past the
    valleys."""
    cx, cy, r = head
    out = _outline_radii(ind, radii, win)
    tips, valleys = _extremes(out, n, rot, 0.0, np.max), _extremes(out, n, rot, 0.5, np.min)
    if not tips:
        return None
    r_tip = float(np.median(tips))
    r_val = float(np.median(valleys)) if valleys else float(np.nanmin(out))
    r_in, r_out = fr.d(r_val), fr.d(r_tip)
    if r_out - r_in < MIN_ZONE_UNITS:
        return None
    ra = int(np.searchsorted(radii, r_val))
    rb = min(len(radii), int(np.searchsorted(radii, r_tip)) + 1)
    if rb - ra < 2:
        return None
    w = ind[ra:rb][:, win].mean(axis=1) * 2 * np.pi * radii[ra:rb] / n
    t = out.shape[0]
    th = np.linspace(0, 2 * np.pi, t, endpoint=False)
    ok = np.isfinite(out)
    rr = np.interp(np.arange(t), np.flatnonzero(ok), out[ok], period=t) / r  # the hidden angles bridged
    word = _piece_word(_Outline(0, "", [np.stack([rr * np.cos(th), rr * np.sin(th)], axis=1)]))
    if word is not None and word.endswith("star"):
        kind, wide = "star", 2 * np.pi * r_in / n
    else:
        kind, wide = _kind(w, 2 * np.pi * radii[ra:rb] / n), fr.d(float(w.max()))
    params = {"cx": r3(fr.x(cx)), "cy": r3(fr.y(cy)), "kind": kind, "count": int(n), "r_inner": r3(r_in),
              "r_outer": r3(r_out), "rotation_deg": r3(rot),
              "width_ratio": r3(wide / max(r_out - r_in, 1e-6)), "source": "piece"}
    return params, rb


def _same_axes(q: dict, n: int, rot: float, tol: float) -> bool:
    """Count n and rotation within tol (deg, modulo the step) of rot."""
    if int(q.get("count", 0)) != n:
        return False
    d = (float(q["rotation_deg"]) - rot) % (360.0 / n)
    return min(d, 360.0 / n - d) <= tol


def _arm_sets(cand: list[tuple[np.ndarray, float, float]]) -> list[list[int]]:
    """R18: indices of the candidate arm pieces [(mask, inner radius, outer radius) in canvas units]
    grouped by single linkage on their radial span (inner and outer radius each within
    MIN_ZONE_UNITS of a member's), then capped: while a member's inner or outer radius lies more
    than MIN_ZONE_UNITS from the group's median, the farthest one leaves the group. Single linkage
    alone can chain arms through pieces of intermediate span to rim bits, and the union's outline
    (the outermost radius per angle) would then be the rim bits'. Complete linkage (every pair
    within MIN_ZONE_UNITS) would split S43-15's six arms (outer radii 94.8-105.9); the median cap
    keeps them and S42-15's. Groups of STAR_MIN_ORDER or more, in candidate order."""
    left, out = list(range(len(cand))), []
    while left:
        grp = [left.pop(0)]
        grown = True
        while grown:
            grown = False
            for k in list(left):
                if any(abs(cand[k][1] - cand[m][1]) <= MIN_ZONE_UNITS
                       and abs(cand[k][2] - cand[m][2]) <= MIN_ZONE_UNITS for m in grp):
                    grp.append(k)
                    left.remove(k)
                    grown = True
        while len(grp) >= STAR_MIN_ORDER:
            m0 = float(np.median([cand[k][1] for k in grp]))
            m1 = float(np.median([cand[k][2] for k in grp]))
            far = max(grp, key=lambda k: max(abs(cand[k][1] - m0), abs(cand[k][2] - m1)))
            if max(abs(cand[far][1] - m0), abs(cand[far][2] - m1)) <= MIN_ZONE_UNITS:
                break
            grp.remove(far)
        if len(grp) >= STAR_MIN_ORDER:
            out.append(sorted(grp))
    return out


def _front_pieces(layers: list, meta: list[dict], zones: list[tuple[int, int]], radii: np.ndarray,
                  fr: Frame, base: int, head: tuple[float, float, float], pieces: list) -> None:
    """ML-2: front layers in the base color read from SAM 2 pieces, and their inner layers (BACK_TOL_DEG
    and the constants with it). A rules layer the piece path reads again (a front layer of the same
    color on the same axes: count equal, rotation within NEST_AXIS_DEG, not a backing; for the inner
    layer also a layer of its color on those axes reaching no further than it, MIN_BAND_UNITS) gives
    way to the piece layer (params source "piece"); a rules layer without a piece counterpart stays.
    R18: the base-color pieces refused alone are then grouped into arm sets (`_arm_sets`,
    ARM_SPAN_PCT) and each set is read as one piece; a front's meta holds its span (`body`, canvas
    units: the first lobed row to the tips) for the ring check in `detect_head`. Runs after the H15
    rule and the border split."""
    cx, cy, r = head
    win = np.ones(RING_THETA, bool)  # pieces inside the head are read over the full circle
    head_pcs = [pc for pc in pieces if pc.in_head and pc.color_group is not None and pc.color_group >= 0]
    added: list[tuple[int, float]] = []

    def polar_of(mask: np.ndarray) -> np.ndarray:
        return polar_sample(mask.astype(np.uint8), cx, cy, radii, RING_THETA, fill=0) > 0

    def read(mask: np.ndarray, ind: np.ndarray, what: str) -> bool:
        """Read one piece (or an arm set's union) as a front layer; False when it is none."""
        order = _piece_order(ind, radii, fr)
        if isinstance(order, str):
            log.info("%s is no front layer: %s", what, order)
            return False
        n, score = order
        rot = _piece_rotation(_outline_radii(ind, radii, win), n)
        if any(_same_axes({"count": n2, "rotation_deg": rot2}, n, rot, NEST_AXIS_DEG) for n2, rot2 in added):
            return True  # a second piece of a front layer already read (M4)
        got = _piece_layer(ind, radii, win, fr, head, n, rot)
        if got is None:
            return False
        front, rb = got
        if float(front["r_inner"]) < MIN_ZONE_UNITS:
            return False  # lobes from the middle: a badge print (S5-15's letters), not a layer on the disc
        added.append((n, rot))
        ra = int(np.searchsorted(radii, float(front["r_inner"]) * fr.scale))
        hull = np.zeros(mask.shape, np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(mask.astype(np.uint8))), 1)
        hull = hull > 0
        by_color: dict[int, list] = {}
        for pc2 in head_pcs:
            if pc2.color_group != base and (pc2.mask & hull).sum() >= PIECE_INSIDE * max(1, pc2.area):
                by_color.setdefault(int(pc2.color_group), []).append(pc2)
        inner = None
        for h, group in sorted(by_color.items(), key=lambda kv: (-sum(q.area for q in kv[1]), kv[0])):
            mask2 = np.logical_or.reduce([q.mask for q in group])
            ind2 = polar_of(mask2)
            order2 = _piece_order(ind2, radii, fr)
            if isinstance(order2, str) or order2[0] != n:
                continue
            if not _same_axes({"count": n, "rotation_deg": _piece_rotation(_outline_radii(ind2, radii, win), n)},
                              n, rot, BACK_TOL_DEG):
                continue
            got2 = _piece_layer(ind2, radii, win, fr, head, n, rot)  # drawn on the front's axes
            if got2 is not None and float(got2[0]["r_outer"]) < float(front["r_outer"]):
                inner = (h, got2[0], got2[1], r3(min(1.0, order2[1])))
                break
        drop = {k for k, lay in enumerate(layers)
                if lay[0] == "points" and meta[k].get("front") and not meta[k].get("trim")
                and lay[1] == base and _same_axes(lay[3], n, rot, NEST_AXIS_DEG)}
        if inner is not None:
            h, q = inner[0], inner[1]
            drop |= {k for k, lay in enumerate(layers)
                     if lay[0] == "points" and not meta[k].get("trim") and lay[1] == h
                     and _same_axes(lay[3], n, rot, NEST_AXIS_DEG)
                     and float(lay[3]["r_outer"]) <= float(q["r_outer"]) + MIN_BAND_UNITS}
        for k in sorted(drop, reverse=True):
            del layers[k]
            del meta[k]
        zone = next(((a, b) for a, b in zones if a <= ra < b), (ra, rb))
        cov = ind.mean(axis=1)
        first = int(np.flatnonzero((cov > EXTENT_COV) & (cov < FULL_COV))[0])  # where the lobes start
        zone_meta = dict(zone=zone, rows=(ra, rb), eats=None, inner=None, front=True)
        layers.append(("points", base, r3(min(1.0, score)), front, 0.0, rb))
        meta.append(dict(zone_meta, body=(fr.d(radii[first]), float(front["r_outer"]))))
        if inner is not None:
            h, q, rb2, conf2 = inner
            layers.append(("points", h, conf2, q, 0.0, rb2))
            meta.append(dict(zone_meta, rows=(ra, rb2)))
        return True

    arms: list[tuple[np.ndarray, float, float]] = []
    arm_ind: list[np.ndarray] = []
    for i, pc in enumerate(head_pcs):
        if pc.color_group != base or _touches_rim(pc.mask, cx, cy, r, fr):
            continue
        ind = polar_of(pc.mask)
        if read(pc.mask, ind, f"piece {i} (area {pc.area})"):
            continue
        ys, xs = np.nonzero(pc.mask)
        if len(ys) == 0:
            continue
        rr = fr.d(np.hypot(xs - cx, ys - cy))
        r0, r1 = float(np.percentile(rr, ARM_SPAN_PCT)), float(np.percentile(rr, 100 - ARM_SPAN_PCT))
        if r0 > MIN_ZONE_UNITS:  # R18: a candidate arm (a piece holding the middle is no arm)
            arms.append((pc.mask, r0, r1))
            arm_ind.append(ind)
    for grp in _arm_sets(arms):  # R18: arms of one layer read together
        read(np.logical_or.reduce([arms[k][0] for k in grp]), np.logical_or.reduce([arm_ind[k] for k in grp]),
             f"arm set of {len(grp)} pieces")


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
    depth = STAR_HIDDEN_FEW if n <= STAR_FEW else STAR_HIDDEN_DEPTH  # A16: under the bar for its count
    limit = depth * r_tip * np.cos(np.pi / n)  # notched, never a polygon
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
                   radii: np.ndarray, win: np.ndarray, head_label: int, trims: set[int],
                   pieces: list | None = None, loose: list | None = None) -> dict[int, tuple]:
    """Points layers start at the cover over them (H4), are stacked as the photo shows
    them (H12) and then start under the layers drawn above them (H11 _cover_start, which
    reads z: it runs on the final order). Returns {id(points part): (r_inner, width_ratio)}
    before `_cover_start`, so a later re-vote that changes z (R17) can recompute the starts.
    `loose`: the pieces for the loose fronts' unanimous votes (R19, `photo_stacking`)."""
    _points_start_at_cover(parts)
    photo_stacking(parts, labels, p, fr, trims, pieces, loose)
    pre_cover = {id(q): (q.params["r_inner"], q.params["width_ratio"]) for q in parts if q.role == "points"}
    _cover_start(parts, pol, radii, win, fr, head_label, trims)
    return pre_cover


def _revote(parts: list[Part], labels: np.ndarray, p: Prepared, fr: Frame, pol: np.ndarray,
            radii: np.ndarray, win: np.ndarray, head_label: int, trims: set[int],
            pre_cover: dict[int, tuple], pieces: list | None = None, loose: list | None = None) -> None:
    """R17: the photo vote again after layers changed geometry (`_tips_to_stars`). The H11 cover
    starts read z, so when the vote changes any z every non-star points layer gets back its start
    from before `_cover_start` (`pre_cover`, from `_finish_points`) and the cover starts are read
    again on the new order."""
    z0 = [q.z for q in parts]
    photo_stacking(parts, labels, p, fr, trims, pieces, loose)
    if [q.z for q in parts] == z0:
        return
    for q in parts:
        if id(q) in pre_cover and q.params.get("kind") != "star":
            q.params["r_inner"], q.params["width_ratio"] = pre_cover[id(q)]
    _cover_start(parts, pol, radii, win, fr, head_label, trims)


def _axis_runs(pol: np.ndarray, cols: list[int], rows: int, minlen: int) -> list[tuple[int, int, int, float]]:
    """R19: runs [(label, first row, end row, mean agreement)] of the label held by the most of the
    polar columns `cols` on each of the first `rows` rows (agreement: the share of the columns holding
    it); a run shorter than `minlen` rows is joined to the run inside it, and runs of one label that
    meet are one run."""
    sub = pol[:rows][:, cols]
    mode = np.full(sub.shape[0], -1)
    agree = np.zeros(sub.shape[0])
    for i, row in enumerate(sub):
        v = row[row >= 0]
        if v.size:
            mode[i] = int(np.bincount(v).argmax())
            agree[i] = float((row == mode[i]).mean())
    runs: list[list[int]] = []
    i = 0
    while i < len(mode):
        j = i
        while j < len(mode) and mode[j] == mode[i]:
            j += 1
        if runs and (j - i < minlen or runs[-1][0] == mode[i]):
            runs[-1][2] = j
        else:
            runs.append([int(mode[i]), i, j])
        i = j
    return [(g, a, b, float(agree[a:b].mean())) for g, a, b in runs]


def _nested_runs(parts: list[Part], pol: np.ndarray, radii: np.ndarray, win: np.ndarray, fr: Frame) -> None:
    """R19: the stars nested on a drawn star's axes, each read from one color run along them (the
    conditions with the constants: R19 above). The new stars lie on top of the drawn layers on those
    axes, the shorter above, and the points layers above those move up with them. Their z is assumed
    (the photo vote does not see them), so they are flagged (confidence at most
    BLOB_UNCONFIRMED_CONF). None is added when the points layers would then reach the rings' z (30);
    the drawn layers on those axes are flagged then instead (fix round 1: not silent)."""
    t = pol.shape[1]
    n_groups = int(pol.max()) + 1
    if n_groups <= 0:
        return
    cov = np.stack([(pol[:, win] == g).mean(axis=1) for g in range(n_groups)], axis=1)
    minlen, gap = _rows(MIN_BAND_UNITS, radii, fr), _rows(BACK_MIN_UNITS, radii, fr)
    unit = lambda i: fr.d(radii[min(i, len(radii) - 1)])  # noqa: E731 - radius row -> canvas units
    pts = [q for q in parts if q.role == "points"]
    rings = [q for q in parts if q.role in ("ring", "center")]
    hosts = sorted((q for q in pts if q.params.get("kind") == "star" and int(q.params["count"]) >= STAR_MIN_ORDER),
                   key=lambda q: -float(q.params["r_outer"]))
    new: list[tuple[Part, Part]] = []  # (host, new star)
    for host in hosts:
        hp = host.params
        n, rot = int(hp["count"]), float(hp["rotation_deg"])
        cols = [c for c in _axis_cols(hp, t) if win[c]]
        if len(cols) < 2:
            continue
        tip = min(len(radii), int(np.searchsorted(radii, float(hp["r_outer"]) * fr.scale)) + 1)
        runs = [run for run in _axis_runs(pol, cols, min(len(radii), tip + gap), minlen) if run[0] >= 0]
        for k, (g, a, b, agr) in enumerate(runs):
            r_out = unit(b - 1)
            if (r_out - unit(a) < MIN_ZONE_UNITS or r_out > float(hp["r_outer"]) - BACK_MIN_UNITS
                    or agr < BORDER_MIN_SIDES or cov[a:b, g].mean() >= FULL_COV):
                continue
            nxt = runs[k + 1] if k + 1 < len(runs) else None
            if (nxt is None or nxt[0] == g or nxt[3] < BORDER_MIN_SIDES or nxt[2] - nxt[1] < gap
                    or _score_at((pol[nxt[1]:nxt[2]] == nxt[0]).astype(np.float32), win, n) < SPLIT_FLOOR):
                continue  # no tip: the run meets the background or a band
            if g == host.color_group and not any(run[0] == g for run in runs[k + 1:]):
                continue  # the star itself
            if any(q.color_group == g and min(r_out, float(q.params.get("r_outer", q.params.get("r", 0.0))))
                   - max(unit(a), float(q.params.get("r_inner", 0.0))) >= RING_MAJORITY * (r_out - unit(a))
                   for q in rings):
                continue  # a drawn ring or center of its color
            if any(q.color_group == g and _same_axes(q.params, n, rot, NEST_AXIS_DEG)
                   and abs(float(q.params["r_outer"]) - r_out) <= MIN_ZONE_UNITS
                   for q in pts + [x[1] for x in new]):
                continue  # drawn already
            sym = _score_at((pol[a:b] == g).astype(np.float32), win, n)
            r_in = float(hp["r_inner"])
            ra = max(a, int(np.searchsorted(radii, r_in * fr.scale)))
            if sym < SPLIT_FLOOR or r_out - r_in < MIN_ZONE_UNITS or b - ra < 2:
                continue
            # the host's footprint caps the side fit, as H16's inner star is capped by its front's
            cap = _star_halfangle(radii[ra:b], float(hp["r_outer"]) * fr.scale, r_in * fr.scale, n)
            v = _straight_valley(pol, radii, range(ra, b), g, hp, win, cap)
            v = r_in if v is None else max(r_in, fr.d(v))
            params = {"cx": hp["cx"], "cy": hp["cy"], "kind": "star", "count": n, "r_inner": r3(r_in),
                      "r_outer": r3(r_out), "rotation_deg": hp["rotation_deg"],
                      "width_ratio": r3(2 * np.pi * v / n / max(r_out - r_in, 1e-6))}
            conf = r3(min(agr, sym, BLOB_UNCONFIRMED_CONF))  # z assumed: flagged
            new.append((host, Part(f"points-{len(parts) + len(new)}", "points", params, g, 0, conf)))
    if not new:
        return
    under = [q for q in pts if any(_same_axes(q.params, int(h.params["count"]), float(h.params["rotation_deg"]),
                                              NEST_AXIS_DEG) for h, _ in new)]
    top = max(q.z for q in under)
    add = sorted((q for _, q in new), key=lambda q: -float(q.params["r_outer"]))
    if max(q.z for q in pts) + len(add) >= 30:
        log.warning("R19: no z left under the rings for %d nested stars; the layers on their axes are flagged",
                    len(add))
        for q in under:
            q.confidence = r3(min(float(q.confidence), BLOB_UNCONFIRMED_CONF))
        return
    for q in pts:
        if q.z > top:
            q.z += len(add)
    for k, q in enumerate(add):
        q.z = top + 1 + k
        parts.append(q)


def _covering(parts: list[Part], q: Part, lab: int, r: float) -> set[str]:
    """R17: what of the parts drawn above points layer q holds label `lab` at radius r (canvas
    units): "ring", "center" or "pts" (a points layer of higher z). The head disc lies under every
    points layer and covers nothing."""
    out = set()
    for o in parts:
        if o is q or o.color_group != lab:
            continue
        a = o.params
        if o.role == "ring" and float(a["r_inner"]) - MIN_BAND_UNITS / 2 <= r <= float(a["r_outer"]) + MIN_BAND_UNITS / 2:
            out.add("ring")
        elif o.role == "center" and r <= float(a["r"]) + MIN_BAND_UNITS / 2:
            out.add("center")
        elif (o.role == "points" and o.z > q.z
              and float(a["r_inner"]) - MIN_BAND_UNITS / 2 <= r <= float(a["r_outer"]) + MIN_BAND_UNITS / 2):
            out.add("pts")
    return out


def _tip_lobes(q: Part, parts: list[Part], pol: np.ndarray, radii: np.ndarray, win: np.ndarray,
               fr: Frame) -> dict | None:
    """R17: the measures of points layer q's outermost lobe (TIP_* and the constants with them):
    rows (b, e) of the lobe (absolute radius rows), peak (position of the widest row in the lobe,
    0..1), bulge, end and inner (last and first row's width over the widest), sym (turned
    self-correlation at the count over the lobe rows), cover (share of the visible axes whose label
    just inside the lobe is a ring, the center or a higher points layer), side (share of the labels
    beside the lobe, where it narrows inward, that are such a cover; 0 when no such label is seen:
    no evidence of a cover beside the lobe),
    interleaved (a higher points layer of another color, same count, axes half a step off). None
    when the lobe shows on fewer than 3 rows."""
    a = q.params
    t, n, g = pol.shape[1], int(a["count"]), q.color_group
    unit = radii / fr.scale
    i0 = int(np.searchsorted(unit, float(a["r_inner"])))
    i1 = min(len(radii), int(np.searchsorted(unit, float(a["r_outer"]))) + 2)
    cols = [c for c in _axis_cols(a, t) if win[c]]
    if not cols or i1 - i0 < 3:
        return None
    reach = t // (2 * n)
    runs = np.array([[_axis_halfwidth(pol[i], c, g, reach) for c in cols] for i in range(i0, i1)], float)
    w = np.median(runs, axis=1) * 2 * np.pi * unit[i0:i1] / t
    nz = np.flatnonzero(w > 0)
    if nz.size < 3:
        return None
    e = int(nz[-1])
    b = e
    while b - 1 >= 0 and (w[b - 1] > 0 or (b - 2 >= 0 and w[b - 2] > 0)):  # one-row gaps bridged
        b -= 1
    if w[b] == 0:
        b += 1
    seg = w[b:e + 1]
    if len(seg) < 3:
        return None
    wmax = float(seg.max())
    pk = int(np.argmax(seg))
    o = seg[pk:] / wmax
    bulge = float((o - np.linspace(o[0], o[-1], len(o))).mean()) if len(o) >= 3 else float("inf")
    sym = _score_at((pol[i0 + b:i0 + e + 1] == g).astype(np.float32), win, n)
    walk = max(1, int(round(TIP_WALK_UNITS / RADIAL_STEP_UNITS)))
    covered = 0
    for c in cols:  # the label just inside the lobe on each axis, past the lobe's own label
        j = i0 + b
        while j > 0 and pol[j, c] == g and i0 + b - j < walk:
            j -= 1
        if pol[j, c] == g:  # the walk limit ended on a lobe row: the next row inward
            j = max(0, j - 1)
        covered += bool(_covering(parts, q, int(pol[j, c]), float(unit[j])) & {"ring", "center", "pts"})
    seen = held = 0
    for ii in range(b, b + pk):  # beside the lobe where it is narrower than its widest
        if w[ii] >= 0.9 * wmax:
            continue
        i = i0 + ii
        for c in cols:
            if pol[i, c] != g:
                continue
            for d in (1, -1):
                k = 1
                while k <= reach and pol[i, (c + d * k) % t] == g:
                    k += 1
                if k > reach:
                    continue
                seen += 1
                held += bool(_covering(parts, q, int(pol[i, (c + d * k) % t]), float(unit[i]))
                             & {"ring", "center", "pts"})
    step = 360.0 / n
    interleaved = any(
        o is not q and o.role == "points" and o.color_group != g and o.z > q.z
        and int(o.params.get("count", 0)) == n
        and abs(((float(o.params["rotation_deg"]) - float(a["rotation_deg"])) % step) - step / 2) <= NEST_AXIS_DEG
        for o in parts)
    return dict(rows=(i0 + b, i0 + e), peak=pk / max(1, len(seg) - 1), bulge=bulge, end=float(seg[-1] / wmax),
                inner=float(seg[0] / wmax), sym=float(sym), cover=covered / len(cols),
                side=held / seen if seen else 0.0, interleaved=interleaved)


def _is_tip_layer(f: dict | None) -> str | None:
    """R17: "A" (cut where wide) or "B" (in an interleaved star's valleys) when the measures of
    `_tip_lobes` read the layer's lobes as the tips of a covered star, else None."""
    if f is None or f["sym"] < SPLIT_FLOOR:
        return None
    if not (f["peak"] <= TIP_PEAK and f["bulge"] <= TIP_BULGE and f["end"] <= TIP_END):
        return None
    if TIP_CUT_A and f["inner"] >= TIP_CUT_WIDE and f["cover"] >= COVER_CUT_SHARE:
        return "A"
    if f["interleaved"] and f["side"] >= TIP_SIDE:
        return "B"
    return None


def _tip_star(q: Part, f: dict, pol: np.ndarray, radii: np.ndarray, win: np.ndarray, fr: Frame,
              pieces: list | None, p: Prepared) -> dict:
    """R17: the star of q's count and axes whose tips are the lobes': tips at the lobe's last row,
    valleys where the straight sides through the lobe's edges (from its widest row out) meet the
    valley rays, never outside the lobe's first row (it is the star's own color there) nor within
    MIN_ZONE_UNITS of the middle. With SAM 2 pieces of q's color holding the lobe on
    >= BORDER_MIN_SIDES of the visible axes, the tips and sides are read from those pieces' union
    instead (params source "piece"), unless the pieces' tip lies more than PIECE_PROMINENCE_UNITS
    from the label's (the pieces hold more than the lobe): then the labels."""
    a = q.params
    g, n = q.color_group, int(a["count"])
    b, e = f["rows"]
    src, lab = None, pol
    if pieces:
        cx, cy, _ = p.head
        t = pol.shape[1]
        mid = (b + e) // 2
        cols = [c for c in _axis_cols(a, t) if win[c]]
        hold = []
        for pc in pieces:
            if not pc.in_head or pc.color_group != g:
                continue
            ind = polar_sample(pc.mask.astype(np.uint8), cx, cy, radii, t, fill=0) > 0
            if ind[mid, cols].any():
                hold.append(ind)
        if hold:
            ind = np.logical_or.reduce(hold)
            if ind[mid, cols].mean() >= BORDER_MIN_SIDES:
                ends = [int(np.flatnonzero(ind[:, c])[-1]) for c in cols if ind[:, c].any()]
                ep = max(b + 2, int(np.median(ends)))
                if abs(fr.d(radii[min(ep, len(radii) - 1)] - radii[e])) <= PIECE_PROMINENCE_UNITS:
                    e, lab, src = ep, np.where(ind, g, -1), "piece"
                # else the pieces hold more than the lobe (merged with another region): the labels
    pk = b + int(round(f["peak"] * (f["rows"][1] - f["rows"][0])))
    r_out = fr.d(radii[min(e, len(radii) - 1)])
    first = fr.d(radii[b])
    v = _straight_valley(lab, radii, range(pk, e + 1), g, a, win)
    r_in = first if v is None else min(first, fr.d(v))
    r_in = max(r_in, MIN_ZONE_UNITS)
    out = {k: v2 for k, v2 in a.items() if k not in ("inner_ratio", "edge_width")}
    out.update(kind="star", r_inner=r3(r_in), r_outer=r3(r_out),
               width_ratio=r3(2 * np.pi * r_in / n / max(r_out - r_in, 1e-6)))
    if src:
        out["source"] = src
    return out


def _tips_to_stars(parts: list[Part], pol: np.ndarray, radii: np.ndarray, win: np.ndarray, fr: Frame,
                   trims: set[int], pieces: list | None, p: Prepared) -> bool:
    """R17: points layers read as blocks, kites or petals whose lobes are the tips of a covered star
    (`_is_tip_layer`) become that star (`_tip_star`), keeping their color, count and axes (the
    caller re-runs the photo vote on the new geometry, so z is re-voted). A
    layer hosting trims (same count and axes), a front layer, or one whose color already has a
    star on its axes is left alone. True when any layer changed."""
    changed = False
    for q in parts:
        a = q.params
        if (q.role != "points" or id(q) in trims or q.id.startswith(FRONT_ID)
                or a.get("kind") not in ("bow", "diamond", "petal")
                or not STAR_MIN_ORDER <= int(a["count"]) < PLEAT_MIN_COUNT):
            continue
        n, rot = int(a["count"]), float(a["rotation_deg"])
        if any(o is not q and o.role == "points" and _same_axes(o.params, n, rot, NEST_AXIS_DEG)
               and (id(o) in trims or o.color_group == q.color_group) for o in parts):
            continue
        f = _tip_lobes(q, parts, pol, radii, win, fr)
        if _is_tip_layer(f) is None:
            continue
        q.params = _tip_star(q, f, pol, radii, win, fr, pieces, p)
        changed = True
    return changed


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
    bar = STAR_CONVEX_FEW if n <= STAR_FEW else STAR_CONVEX
    return (params.get("kind") == "star" and n >= 2
            and float(params["r_inner"]) >= bar * float(params["r_outer"]) * np.cos(np.pi / n))


def _hidden_valleys(params: dict, zone_start: float) -> dict | None:
    """H11: the star with its valleys moved under the layers inside its zone (they were
    read where its visible lobes are widest): at the zone's start when that gives notches,
    else at STAR_HIDDEN_DEPTH of the convex limit when that still lies inside the zone's
    start (hidden); None when neither does."""
    deeper = dict(params, r_inner=r3(zone_start))
    if not _convex_star(deeper):
        return deeper
    n = int(params["count"])
    depth = STAR_HIDDEN_FEW if n <= STAR_FEW else STAR_HIDDEN_DEPTH
    r_in = depth * float(params["r_outer"]) * float(np.cos(np.pi / n))
    return dict(params, r_inner=r3(r_in)) if r_in < zone_start else None


def _alternating(k: int, layers: list, meta: list[dict]) -> bool:
    """H11: layer k and another points layer of the same zone, count and radii (within
    ALT_RADIUS_UNITS) whose lobes sit between each other's: a two-tone pleated edge whose
    colors alternate. A16 (C6, ruling): drawn as one pinked ring in the color holding more of
    it (`_alternate_partner`); a one-color pleated ring's highlights or shadows read as a
    second color (S6-15, S69-19)."""
    return _alternate_partner(k, layers, meta) is not None


def _alternate_partner(k: int, layers: list, meta: list[dict]) -> int | None:
    """Index of the points layer alternating with layer k (`_alternating`), or None."""
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
            return j
    return None


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


def _front_roles(parts: list[Part], trims: set[int],
                 pieces: list | None = None) -> tuple[dict[int, int], set[int]]:
    """H16: ({front layer in the head disc's color: its backing}, indices of the backings). A
    front layer cannot be told from the disc behind it, so it casts no stacking vote; its backing
    (a trim of another color, a star of its own under it, same count and axes) votes in its
    place. A front without such a backing (dropped or rewritten after `_front_in_base`) is no
    key: it votes as a host as before H16. ML-2: with pieces in the vote, a front layer read
    from a piece (params source "piece") is no key either: it votes with its own piece."""
    base = next((q.color_group for q in parts if q.role == "head"), None)
    front = {k for k, q in enumerate(parts) if q.role == "points" and q.id.startswith(FRONT_ID)}
    backs = {k for k in front if id(parts[k]) in trims}
    front_of = {}
    for f in sorted(front):
        qf = parts[f].params
        if parts[f].color_group != base or id(parts[f]) in trims:
            continue
        if pieces and qf.get("source") == "piece":
            continue
        same = [b for b in backs if parts[b].params.get("count") == qf.get("count")
                and parts[b].params.get("rotation_deg") == qf.get("rotation_deg")]
        if same:
            front_of[f] = min(same, key=lambda b: (abs(parts[b].z - parts[f].z), b))
    return front_of, backs


def _own_pieces(parts: list[Part], geoms: dict, pieces: list, fr: Frame) -> dict[int, np.ndarray]:
    """ML-2: the visible mask of each drawn layer: the union of the pieces that belong to it. A piece
    (with its color group set) belongs to the SMALLEST drawn geometry of its color group (points,
    rings, center, unknown parts and the head disc) that holds the piece's centroid and at least
    PIECE_OWN_SHARE of its pixels (S70-19: the inner white arms lie inside both the white backing
    and the inner white star; they belong to the inner star, the smaller one). Layers no piece
    belongs to are missing from the result."""
    cand = dict(geoms)
    for k, q in enumerate(parts):
        if q.role == "head":
            cand[k] = to_geom(disc(q.params["cx"], q.params["cy"], q.params["r"]))
    own: dict[int, np.ndarray] = {}
    for pc in pieces:
        if pc.color_group is None or pc.color_group < 0 or not pc.area:
            continue
        ys, xs = np.nonzero(pc.mask)
        ux, uy = fr.x(xs.mean()), fr.y(ys.mean())
        step = max(1, len(xs) // 2000)
        sx, sy = fr.x(xs[::step].astype(np.float64)), fr.y(ys[::step].astype(np.float64))
        best = None
        for k, g in cand.items():
            if g is None or g.is_empty or parts[k].color_group != pc.color_group:
                continue
            if not contains_xy(g, ux, uy) or contains_xy(g, sx, sy).mean() < PIECE_OWN_SHARE:
                continue
            if best is None or g.area < cand[best].area:
                best = k
        if best is not None:
            own[best] = pc.mask if best not in own else (own[best] | pc.mask)
    return own


def stack_votes(parts: list[Part], labels: np.ndarray, p: Prepared, fr: Frame,
                trims: set[int], pieces: list | None = None, totals: bool = False) -> list[tuple]:
    """H12: (index a, index b, samples of a's label, samples of b's label) over the drawn
    overlap of every pair of non-trim points parts of different colors (STACK_*). A front layer
    in the disc's color does not vote; its backing does (H16, `_front_roles`). The other points
    layers of either voter's color are cut from the overlap (their samples are their own vote),
    except, when a voter is a front layer, a front layer that contains one of the voters (H16: the
    backing around the inner star; the inner star is still cut from the backing's overlap).
    ML-2: with SAM 2 pieces, a pair of layers that both have pieces counts the samples on each one's
    own visible mask (`_own_pieces`: the pieces whose centroid lies in the smallest drawn layer of
    their color) instead of its color label; when either has no piece, both count their labels as
    before (like with like: a partial piece set against a full label loses votes it should win).
    With `totals` (R19) each tuple also holds the overlap's sample count."""
    geoms = {k: _part_geom(q) for k, q in enumerate(parts) if q.role in ("points", "ring", "center", "unknown")}
    own = _own_pieces(parts, geoms, pieces, fr) if pieces else {}
    front_of, backs = _front_roles(parts, trims, pieces)
    loose = _loose_piece_fronts(parts, trims, pieces, front_of)
    pts = [k for k, q in enumerate(parts)
           if q.role == "points" and (id(q) not in trims or k in backs) and k not in front_of and k not in loose]
    front = {k for k, q in enumerate(parts) if q.role == "points" and q.id.startswith(FRONT_ID)}
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
                      if k not in (a, b) and q.role == "points" and q.color_group in (ga, gb)
                      and not (k in front and (a in front or b in front)
                               and (geoms[k].buffer(1.0).contains(geoms[a])
                                    or geoms[k].buffer(1.0).contains(geoms[b])))]
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
            if a in own and b in own:  # like with like: both visible masks, else both labels
                vote = (a, b, int(own[a][py[ok], px[ok]].sum()), int(own[b][py[ok], px[ok]].sum()))
            else:
                vote = (a, b, int((lab == ga).sum()), int((lab == gb).sum()))
            out.append(vote + (int(ok.sum()),) if totals else vote)
    return out


def photo_stacking(parts: list[Part], labels: np.ndarray, p: Prepared, fr: Frame, trims: set[int],
                   pieces: list | None = None, loose_pieces: list | None = None) -> None:
    """H12: points layers re-stacked as the photo shows them where they overlap (STACK_*);
    only their z values change. ML-2: with SAM 2 pieces the votes count visible pieces
    (`stack_votes`); a front read from a piece with a backing on its axes lies just on that
    backing and votes for it (`_carried_fronts`), and an inner layer read from pieces lies on its
    front. R19: with the vote on labels (`pieces` None), `loose_pieces` given, a loose front's
    unanimous piece votes (`_loose_votes`) decide its pairs, and such a front is a host."""
    pts = [k for k, q in enumerate(parts) if q.role == "points"]
    _clear_votes(parts)  # ML-4: this run's verdicts only (`_record_votes`)
    if len(pts) < 2:
        return
    edges = []  # (strength, lower, upper)
    decided = set()
    front_of, backs = _front_roles(parts, trims, pieces)
    carried = _carried_fronts(parts, trims, backs)
    tally: dict[tuple[int, int], list[int]] = {}  # ML-4: samples per voted pair, for the candidates
    for a, b, na, nb in stack_votes(parts, labels, p, fr, trims, pieces):
        a, b = carried.get(a, a), carried.get(b, b)  # ML-2: a piece front votes for its backing
        if a == b:
            continue
        t = tally.setdefault((min(a, b), max(a, b)), [0, 0])
        t[a > b] += na
        t[a < b] += nb
        n = na + nb
        if n < STACK_MIN_UNITS2:
            continue
        share = na / n
        if share >= STACK_SHARE:
            edges.append((share, b, a, n))
        elif share <= 1 - STACK_SHARE:
            edges.append((1 - share, a, b, n))
        else:
            continue
        decided.add(frozenset((a, b)))
    loose = _loose_piece_fronts(parts, trims, pieces, front_of)
    if loose and loose_pieces and not pieces:  # R19: a loose front's unanimous piece votes decide
        voters = {k for k in pts if (id(parts[k]) not in trims or k in backs)
                  and k not in front_of and k not in carried and k not in loose}
        free = loose - set(carried)  # a front carried by its backing (ML-2) is placed with it
        for a, b, na, nb in _loose_votes(parts, labels, p, fr, trims, loose_pieces, free):
            if a in voters or b in voters:  # at its share like any vote (unanimous: 1.0)
                edges.append((1.0, b, a, na + nb) if na else (1.0, a, b, na + nb))
                decided.add(frozenset((a, b)))
                loose -= {a, b}
    hosts = [k for k in pts if (id(parts[k]) not in trims or k in backs)
             and k not in front_of and k not in carried and k not in loose]
    nested = [(b, a) for a, b in nested_pairs(parts, hosts) if frozenset((a, b)) not in decided]
    edges += [(STACK_SHARE / 2, lo, up) for lo, up in nested]  # after every photo decision
    # H16: a front layer lies just on its backing (`full` below); a front layer of the backing's
    # color on the same axes (the inner star) lies on the front, so above the backing, first of all
    for f, b in front_of.items():
        qf = parts[f].params
        for k in hosts:
            qk = parts[k].params
            if (k not in backs and parts[k].id.startswith(FRONT_ID) and parts[k].color_group == parts[b].color_group
                    and qk.get("count") == qf.get("count") and qk.get("rotation_deg") == qf.get("rotation_deg")):
                edges.append((2.0, b, k))
    for f, b in carried.items():  # ML-2: a piece front lies just on its backing, its inner layer above
        front_of[f] = b
        qf = parts[f].params
        for k in hosts:
            qk = parts[k].params
            if (k not in backs and parts[k].id.startswith(FRONT_ID) and qk.get("count") == qf.get("count")
                    and qk.get("rotation_deg") == qf.get("rotation_deg")):
                edges.append((2.0, b, k))
    under: dict[int, int] = {}  # ML-2: inner layer -> its void piece front
    for f in [k for k in pts if k not in carried and (k in hosts or k in loose)]:
        # ML-2: an inner layer read from pieces lies on its front piece layer (no backing; the front
        # is void when the vote has no pieces, and is then placed with H16's rule for a void front)
        qf = parts[f].params
        if qf.get("source") != "piece" or not parts[f].id.startswith(FRONT_ID) or parts[f].color_group != base_of(parts):
            continue
        for k in hosts:
            qk = parts[k].params
            if (k != f and qk.get("source") == "piece" and parts[k].id.startswith(FRONT_ID)
                    and parts[k].color_group != parts[f].color_group and qk.get("count") == qf.get("count")
                    and qk.get("rotation_deg") == qf.get("rotation_deg")
                    and float(qk["r_outer"]) < float(qf["r_outer"])):
                if f in hosts:
                    edges.append((2.0, f, k))
                else:
                    under[k] = f  # a void front: drawn just under its inner layer
    if not edges and not front_of and not under:
        return
    # a backing keeps its front's place in the prior (before H16 it was a trim placed with it)
    zkey = {k: max([parts[k].z] + [parts[f].z for f, b in front_of.items() if b == k]) for k in hosts}
    prior = {k: i for i, k in enumerate(sorted(hosts, key=lambda k: (zkey[k], k)))}
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

    # strongest first; R19: ties by the vote's samples (a small unanimous piece vote after a large
    # unanimous label vote); the H16 / ML-2 placements (2.0) and the nested rule have no samples
    _record_votes(parts, tally)
    for e in sorted(edges, key=lambda e: (-e[0], -(e[3] if len(e) > 3 else 0))):
        lo, up = e[1], e[2]
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
        if id(parts[t]) in trims and t not in backs:
            q = parts[t]
            host_of[t] = min(hosts, key=lambda k: (parts[k].params.get("count") != q.params.get("count"),
                                                   parts[k].params.get("rotation_deg") != q.params.get("rotation_deg"),
                                                   abs(parts[k].z - q.z), k))
    full = []
    for k in order:
        full += [t for t in pts if host_of.get(t) == k and parts[t].z < parts[k].z]
        full += [f for i, f in under.items() if i == k and f not in full]  # ML-2: under its inner layer
        full.append(k)
        full += [f for f, b in front_of.items() if b == k]  # H16: the front just on its backing
        full += [t for t in pts if host_of.get(t) == k and parts[t].z >= parts[k].z]
    full += [k for k in pts if k not in full]  # none left: every points part is a host, trim or front
    zs = sorted(parts[k].z for k in pts)
    for k, z in zip(full, zs):
        parts[k].z = z


def _clear_votes(parts: list[Part]) -> None:
    """ML-4: drop the verdicts of an earlier `photo_stacking` run (R17 re-votes after geometry changed)."""
    for q in parts:
        for k in [k for k in q.params if k.startswith("vote_")]:
            del q.params[k]


def _record_votes(parts: list[Part], tally: dict[tuple[int, int], list[int]]) -> None:
    """ML-4: the H12 verdict of every voted pair, once, from its summed samples (a carried front and
    its backing can vote the same pair), on both parts' params (`vote_<other id>`: "over" when that
    part lies on top at STACK_SHARE, "under", or "undecided" below the share or STACK_MIN_UNITS2),
    read by `candidates.open_decisions`. The drawing never reads these keys."""
    for (a, b), (na, nb) in tally.items():
        n = na + nb
        v = ("undecided" if n < STACK_MIN_UNITS2 else "over" if na / n >= STACK_SHARE
             else "under" if na / n <= 1 - STACK_SHARE else "undecided")
        parts[a].params[f"vote_{parts[b].id}"] = v
        parts[b].params[f"vote_{parts[a].id}"] = {"over": "under", "under": "over"}.get(v, v)


def _carried_fronts(parts: list[Part], trims: set[int], backs: set[int]) -> dict[int, int]:
    """ML-2: {front read from a piece: its backing}. A front layer in the disc's color read from a piece
    whose backing (a trim of `backs`) lies on the same axes (NEST_AXIS_DEG) lies just on it, as H16
    places a void front: it is stacked with its backing, and its votes are its backing's."""
    base = base_of(parts)
    out = {}
    for f, q in enumerate(parts):
        if (q.role != "points" or q.params.get("source") != "piece" or not q.id.startswith(FRONT_ID)
                or q.color_group != base or id(q) in trims):
            continue
        same = [b for b in backs if _same_axes(parts[b].params, int(q.params["count"]),
                                               float(q.params["rotation_deg"]), NEST_AXIS_DEG)]
        if same:
            out[f] = min(same, key=lambda b: (abs(parts[b].z - q.z), b))
    return out


def _loose_piece_fronts(parts: list[Part], trims: set[int], pieces: list | None,
                        front_of: dict[int, int]) -> set[int]:
    """ML-2: front layers in the disc's color read from a piece, without a backing (`front_of`),
    when the vote has no pieces. Such a front cannot be told from the disc behind it: it casts
    no vote and is no host; an inner layer read from pieces carries it (`under` in
    `photo_stacking`). Empty with pieces in the vote, and for fronts not read from a piece
    (H16: those vote as hosts)."""
    if pieces:
        return set()
    base = base_of(parts)
    return {k for k, q in enumerate(parts)
            if q.role == "points" and q.id.startswith(FRONT_ID) and q.params.get("source") == "piece"
            and q.color_group == base and id(q) not in trims and k not in front_of}


def _loose_votes(parts: list[Part], labels: np.ndarray, p: Prepared, fr: Frame, trims: set[int],
                 pieces: list, loose: set[int]) -> list[tuple[int, int, int, int]]:
    """R19: the piece votes (`stack_votes` with pieces: a loose front votes there) of the pairs of
    a loose front and a layer that is neither loose nor its inner piece layer (placed on it by
    ML-2), both with pieces (`_own_pieces`), the vote unanimous and the winner's mask holding at
    least STACK_SHARE of the drawn overlap's samples (fix round 1: mask edges grazing an overlap
    give small unanimous votes too)."""
    geoms = {k: _part_geom(q) for k, q in enumerate(parts) if q.role in ("points", "ring", "center", "unknown")}
    own = _own_pieces(parts, geoms, pieces, fr)
    if not any(k in own for k in loose):
        return []

    def inner_of(f: int, k: int) -> bool:  # ML-2: an inner layer read from pieces lies on its front
        qf, qk = parts[f].params, parts[k].params
        return (qk.get("source") == "piece" and parts[k].id.startswith(FRONT_ID)
                and qk.get("count") == qf.get("count") and qk.get("rotation_deg") == qf.get("rotation_deg")
                and float(qk["r_outer"]) < float(qf["r_outer"]))

    return [(a, b, na, nb) for a, b, na, nb, n in stack_votes(parts, labels, p, fr, trims, pieces, totals=True)
            if ((a in loose and b not in loose and not inner_of(a, b))
                or (b in loose and a not in loose and not inner_of(b, a)))
            and a in own and b in own and min(na, nb) == 0 and na + nb >= STACK_SHARE * max(n, 1)]


def base_of(parts: list[Part]) -> int | None:
    """The head disc's color group, or None without a head disc."""
    return next((q.color_group for q in parts if q.role == "head"), None)


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
