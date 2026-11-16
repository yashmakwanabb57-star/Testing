"""Photo -> RosetteModel -> SVG, end to end.

analyze(data): prepare -> pixel_groups -> detect_parts -> tidy -> RosetteModel -> assign.
tidy (H6) leaves out unknown specks too small to draw and merges tail sliver stripes into
a neighbour, with warnings. Color groups no part uses are dropped before assignment (ids
kept as they are); the rose is on iff a center part was found.
A plain group that looks partly patterned on the tails (colors.mixed_texture_groups)
is warned: a patterned ribbon may have been merged into it. A small patterned color on
the center badge that is not clearly enclosed (colors.badge_print_unsure) is warned too.
Library head (H7): when a reference head from the library fits the photo's outline
better than the detected head (with_template, TEMPLATE_MIN_GAIN), it is drawn instead and
the detected head parts wait in the model's alternatives (use_rules_head switches back).
Head fit (head_fit): our head parts (every part but the tail stripes) are rendered and
compared with the photo's color labels inside the head circle. The measurement is
information for the UI (analyze_with_fit returns it; the API sends it as head_fit);
check_head flags the head only on a gross failure: layer agreement below HEAD_FIT_MIN or
silhouette IoU below HEAD_SILHOUETTE_MIN. Shape errors on a head whose layers overlap
the photo well (petals or loops drawn as blocks) are not caught; the UI always asks the
user to compare the head with the photo.

Pieces (ML-2): with ROSETTE_SAM2 not off, detect also runs SAM 2.1 (rosette.perception.pieces,
cached per photo) and hands the pieces, each colored by its majority label, to detect_parts.
Pieces off, a missing model or a failed run leave the rules path exactly as without pieces;
`perception` (Detection, PhotoContext; the API sends it) says which.

render(model): build -> write_svg -> validate. build may add warnings to the model
(unrecognised shapes drawn as placeholders).
"""
from __future__ import annotations

import atexit
import copy
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .assign import FLAG_BELOW, FLAGGED_CONF, assign, hex_rgb
from .build import build
from .model import Part, RosetteModel
from .perception import reader as reader_mod
from .perception import settings as perception_settings
from .perception.pieces import DEFAULT_CACHE_DIR, segment
from .perception.sam2 import MissingModel
from .roles import PAD_FRAC, render_categories
from .tidy import drop_specks, drop_two_point_layers, merge_tail_slivers
from .validate import Issue, validate
from .vision import candidates as vcandidates
from .vision import groups as vgroups
from .vision import judge as vjudge
from .vision.colors import badge_print_unsure, mixed_texture_groups, pixel_groups
from .vision.one_color import ONE_COLOR_INFO, badge_in_pleats, rose_center
from .vision.stacked_stars import edged_star_between_bows, stacked_stars, star_on_square
from .vision.shaded_rings import HUE_INFO, dark_loop_edges, shaded_rings
from .vision.parts_head import _badge_under_loops, _loops_on_rim, _petals_between_loops, _twin_points

# R32: the center of a hue-rebuilt head is checked for a badge from this share of the head radius.
HUE_BADGE_BIG = 0.3
from .vision.layer_classify import classify_head_layers
from .vision.parts import detect_parts
from .vision.parts_common import Frame, majority
from .vision.parts_head import ALT_INFO_PREFIX, alt_ring_info, pleat_warnings
from .vision.prepare import Prepared, prepare
from .vision.templates import (Pose, describe, load_library, photo_grid, rank, relabel_map, score_template,
                               template_by_id)
from .writer import write_svg

# Head fit gross-failure bars, from the 76 reference photos (docs/pipeline.md). Layer
# agreement (share of the head region's foreground union where the photo's color group
# equals the group we drew) of the heads the H1 analysis reads as right runs 0.60-0.96
# (s34-15 0.604 the lowest), and wrong heads overlap that range, so agreement only
# flags below the right-head range. A synthetic perfect head scores 0.95 (pinked rim)
# and a ring 5 units off 0.88, so values near 0.9 are boundary noise, not errors.
HEAD_FIT_MIN = 0.6
# Silhouette IoU of the right heads is 0.80-0.96 (S11a-15 0.803 the lowest); the bar sits
# in the gap below it (0.764 to 0.786 in the table): missing or extra layers, holes.
HEAD_SILHOUETTE_MIN = 0.78
# Compared region: a disc of HEAD_FIT_EXTENT head radii, without the rows more than
# HEAD_FIT_WINDOW radii below the head center (the tails cover the lower head),
# sampled on a HEAD_FIT_GRID x HEAD_FIT_GRID grid.
HEAD_FIT_EXTENT, HEAD_FIT_WINDOW, HEAD_FIT_GRID = 1.05, 0.55, 160
# Render height of the head-only SVG: 1 px per canvas unit (viewBox 600x850).
_RENDER_H = 850
_FIT_COLORS = {1: "#ff0000", 2: "#0000ff", 3: "#00c000", 4: "#ff00ff"}


# Library heads (H7, rosette/vision/templates.py). The best-matching reference head is
# drawn instead of the detected head parts when its outline fits the photo better: its
# head_fit silhouette IoU is at least TEMPLATE_MIN_GAIN above the rule-based head's. On
# the 76 photos (leave-one-out, docs/pipeline.md H7) the match score itself does not
# predict which head is better (low scores are the hard photos where the rules fail
# too), the silhouette difference does (correlation 0.63 with the head IoU gain), and
# 0.02 gives the highest LOO head IoU with one ID losing more than 0.02. No score floor
# (TEMPLATE_MIN_SCORE 0). The template part's confidence is its match score, so a match
# under FLAG_BELOW (0.6) is flagged for the user.
TEMPLATE_MIN_GAIN = 0.02
TEMPLATE_MIN_SCORE = 0.0
TEMPLATE_FLAG_SCORE = FLAG_BELOW
# When the library head and the rule-based head overlap less than this (silhouette IoU
# of the two drawn heads), the user is told to compare the head with the photo. Over the
# 10 photos drawn with a library head (full library, fix round 2) the overlap runs
# 0.64-0.87; 0.8 warns on 6, among them S40-15 (0.78), whose library head is otherwise
# unflagged while its color is not right (0.890).
TEMPLATE_DIFFERS_IOU = 0.8
TEMPLATE_DIFFERS_WARNING = ("The library head and the rule-based head differ a lot (outline overlap {iou:.0%}) - "
                            "compare the head with the photo and pick another head on the head card if one is closer.")
TEMPLATE_PART_ID = "template"
# Library heads offered on the head card (fix round 4): the matcher's TEMPLATE_CANDIDATES best
# templates; the first is the decision rule's, the others wait in alternatives (ids
# template_2, template_3) with the rule-based head. Own design among the top 3: 58 of 76.
TEMPLATE_CANDIDATES = 3
RULES_HEAD = "rules"
# Part id prefix of a library template chosen outside the candidates (add_library_head, H8).
LIBRARY_PART_ID = "library"
# A drawn library head whose pose moves it at least RECENTER_MIN (head radii, or scale
# change) from prepare's head circle leaves an information line (model.info, not a flag),
# so a later round on prepare has the data (H9).
RECENTER_MIN = 0.1
RECENTER_PREFIX = "Head re-centered:"
RECENTER_INFO = ("Head re-centered: the library head is drawn at an offset of {dx:+.2f}, {dy:+.2f} head radii "
                 "and scale {scale:.2f} from the photo's head circle.")
# Template colors follow the tails (H7 fix round 2): a reference category that is also one
# of the reference's tail Colors k is drawn in the group our tails carry as Color k, the
# head-only categories follow the overlap (templates.tail_consistent_mapping); the matcher
# scores the partition under that same mapping, and a template that cannot be mapped is
# not considered. TEMPLATE_MIN_PARTITION: the library head is drawn only when its
# tail-mapped partition agreement reaches this floor (docs/pipeline.md H7 fix round 2).
# Fix round 3 ruling (shape first): 0.54. Over the photos where the template's outline wins
# (full library 23) color is lost by more than 0.02 at partitions 0.282-0.697; from 0.54 up
# only S58-17 (0.697, 1.000 -> 0.972) loses. A rule-based head that fails the gross head
# check (HEAD_FIT_MIN, HEAD_SILHOUETTE_MIN) is replaced whatever the partition (gross_failure),
# and the library head is then flagged.
TEMPLATE_MIN_PARTITION = 0.54
# Fix round 4: the override needs the library head's outline to win by
# TEMPLATE_OVERRIDE_GAIN (head_fit silhouette over the rule-based head's) and its
# tail-mapped partition to reach TEMPLATE_OVERRIDE_PARTITION; otherwise the rule-based head
# is drawn with the colors warning and the library heads wait on the head card. On the 76
# photos (full library, docs/pipeline.md H7 fix round 4) this keeps S52-19 (partition
# 0.282) on its rule-based head, and with it S29-15 (0.231).
TEMPLATE_OVERRIDE_GAIN = 0.10
TEMPLATE_OVERRIDE_PARTITION = 0.30
# When the rule-based path drew no head layer at all (no head, ring or points part: only a
# center and/or unrecognised blobs, as on S29-15) the partition bar does not apply: a
# missing head is worse than any mappable library head, whose colors the user can fix on
# the head card. The gain bar still applies and the library head is flagged.
HEAD_LAYER_ROLES = ("head", "ring", "points")
# A library head chosen from the library whose tail Colors have no tail group on the photo is
# colored by the free overlap mapping (param free_colors 1): flagged, with this warning.
FREE_COLORS_WARNING = "Library head colors are not anchored to the tails - check every Color on the head card."
# H11 Task 2b: a library head whose head-fit layer agreement beats the rule-based head's
# by TEMPLATE_AGREE_GAIN is preferred too when its silhouette is at most
# TEMPLATE_AGREE_SIL_LOSS worse (a wrong rule-based head whose outline happens to fit).
# Margins 0.05-0.20 swept on the 76 photos (docs/pipeline.md H11): no margin changes the
# LOO head IoU or loses color; 0.10 is the smallest that adds no known-design head IoU
# loss over 0.01 (0.08 and below also draw S6-15's own head, -0.018). It switches S3-15
# (LOO and known) and S32-15 (known): color 0.844 -> 0.946 / 1.000 and 0.858 -> 1.000.
TEMPLATE_AGREE_GAIN, TEMPLATE_AGREE_SIL_LOSS = 0.10, 0.02
# B16 (diag52 C7): the outline path also needs the library head's layer agreement to be at
# most TEMPLATE_SIL_AGREE_LOSS below the rule-based head's (the mirror of
# TEMPLATE_AGREE_SIL_LOSS), so an outline that fits better does not replace layers that were
# right (S27-15, S31-15). The override (a grossly failed rule-based head, override_allowed)
# keeps its own conditions. On the 76 photos (new image, docs/pipeline.md B16) every bar
# from 0 to 0.04 sends S2-15B, S27-15, S31-15 and S56-17 to their rule-based heads and keeps
# the library heads that win on both measures (S11a-15, S40-15, S36-15B, S65-18); from
# 0.05 S31-15 keeps its library head. Layer IoU 0.2313 -> 0.2378, color 0.8444 -> 0.8461,
# head IoU 0.8511 -> 0.8494 (S27-15 and S31-15 lose outline), color OK 47 held.
TEMPLATE_SIL_AGREE_LOSS = 0.03
# H11 ruling: a rule-based head drawn while the closest library design agrees in colors but
# has another points count (POINTS_COUNT_MIN or more points, none of the rule-based points
# layers has that count) is flagged with this warning. On the 76 photos it fires on S10-19,
# S18-15, S53-19, S69-19 and S9-19 with the full library (all flagged already) and, leaving
# each photo's own design out, on S28-15, S45-15, S53-19, S63-17, S80-22 (a false alarm:
# its library neighbour's file has 4 points), S9-19 and s44-15.
POINTS_COUNT_MIN = 3
POINTS_COUNT_WARNING = ("The head's point count differs from the closest library design - "
                        "check the number of points against the photo.")
TEMPLATE_COLORS_WARNING = ("The library head {tid} fits the head better, but its colors do not agree "
                           "with the photo under the tail colors - the rule-based head is drawn; compare the head "
                           "with the photo and pick a library head on the head card if one is closer.")


@dataclass
class HeadFit:
    agreement: float  # share of the region's foreground union with the same color group
    silhouette: float  # foreground IoU


@dataclass
class RenderResult:
    svg: str
    issues: list[Issue] = field(default_factory=list)


def _part_groups(q) -> set[int]:
    if q.role == "template":
        return {q.color_group} | {int(v) for k, v in q.params.items() if k.startswith("group_")}
    return {q.color_group}


def _head_render(model: RosetteModel) -> np.ndarray | None:
    """Group id per canvas pixel (-1 none) of the model's head parts, or None."""
    head = [q for q in model.parts if q.role != "tail_stripe"]
    gids = sorted({g for q in head for g in _part_groups(q)})
    if not head or len(gids) > len(_FIT_COLORS):
        return None
    cat = {gid: i + 1 for i, gid in enumerate(gids)}
    groups = [copy.deepcopy(g) for g in model.groups if g.id in cat]
    for g in groups:
        g.category = cat[g.id]
    # whole stacked shapes, not visible pieces: abutting pieces leave antialiased seams (H14)
    svg = write_svg(build(RosetteModel(parts=copy.deepcopy(head), groups=groups), pieces=False))
    img = render_categories(svg, _RENDER_H, _FIT_COLORS)
    pad = int(round(PAD_FRAC * _RENDER_H))
    img = img[pad:img.shape[0] - pad, pad:img.shape[1] - pad].astype(np.int32)
    ref = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in _FIT_COLORS.values()])
    d = np.abs(img[:, :, None, :] - ref[None, None]).sum(axis=-1)
    lut = np.array([-1] + gids)
    return lut[np.where(d.min(axis=-1) < 90, d.argmin(axis=-1) + 1, 0)]


def head_fit(p: Prepared, labels: np.ndarray, model: RosetteModel,
             merged: dict[int, int] | None = None, render_of=None) -> HeadFit | None:
    """Our rendered head against the photo's color labels on a grid over the head
    circle (HEAD_FIT_*). merged maps photo label ids that assign merged away onto the
    group that absorbed them; other ids no model group has count as a mismatch. None
    when the model has no head parts."""
    ours_img = (render_of or _head_render)(model)
    if ours_img is None:
        return None
    cx, cy, r = p.head
    t = np.linspace(-HEAD_FIT_EXTENT, HEAD_FIT_EXTENT, HEAD_FIT_GRID)
    u, v = np.meshgrid(t, t)
    region = (u * u + v * v <= HEAD_FIT_EXTENT ** 2) & (v <= HEAD_FIT_WINDOW)
    xs, ys = cx + u * r, cy + v * r
    h, w = labels.shape
    xi, yi = np.rint(xs).astype(int), np.rint(ys).astype(int)
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    ph = np.full(xs.shape, -1, np.int64)
    ph[inside] = np.where(p.mask[yi[inside], xi[inside]], labels[yi[inside], xi[inside]], -1)
    for a, b in (merged or {}).items():
        ph[ph == a] = b
    # a photo color no model group has stays foreground and never equals our group id,
    # so it counts as a mismatch
    fr = Frame.of(p)
    cxs, cys = np.rint(fr.x(xs)).astype(int), np.rint(fr.y(ys)).astype(int)
    ok = (cxs >= 0) & (cxs < ours_img.shape[1]) & (cys >= 0) & (cys < ours_img.shape[0])
    ou = np.full(xs.shape, -1, np.int64)
    ou[ok] = ours_img[cys[ok], cxs[ok]]
    pf, of = (ph != -1) & region, (ou >= 0) & region
    union = pf | of
    n = int(union.sum())
    if n == 0:
        return None
    return HeadFit(agreement=float(((ph == ou) & union).sum() / n), silhouette=float((pf & of).sum() / n))


def check_head(p: Prepared, labels: np.ndarray, model: RosetteModel,
               merged: dict[int, int] | None = None) -> RosetteModel:
    """A copy of the model whose head part is flagged (confidence capped at FLAGGED_CONF,
    under FLAG_BELOW, and a warning added) on a gross head failure: head_fit agreement
    below HEAD_FIT_MIN or silhouette below HEAD_SILHOUETTE_MIN."""
    return _check_head(model, head_fit(p, labels, model, merged))


def head_overlap(a: RosetteModel, b: RosetteModel, render_of=None) -> float | None:
    """Silhouette IoU of two models' drawn heads (every part but the tail stripes).
    render_of: a (caching) replacement of _head_render."""
    ra, rb = (render_of or _head_render)(a), (render_of or _head_render)(b)
    if ra is None or rb is None:
        return None
    fa, fb = ra >= 0, rb >= 0
    union = (fa | fb).sum()
    return float((fa & fb).sum() / union) if union else None


def _check_head(model: RosetteModel, fit: HeadFit | None) -> RosetteModel:
    out = copy.deepcopy(model)
    if fit is None:
        return out
    bad_layers, bad_outline = fit.agreement < HEAD_FIT_MIN, fit.silhouette < HEAD_SILHOUETTE_MIN
    if not (bad_layers or bad_outline):
        return out
    for q in out.parts:
        if q.role in ("head", "template"):
            q.confidence = min(q.confidence, FLAGGED_CONF)
    what = []
    if bad_layers:
        what.append("the head layers (disc, rings, points, center: shapes, counts and sizes)")
    if bad_outline:
        what.append("the head outline (missing or extra layers, holes)")
    out.warnings.append(f"The drawn head is far from the photo (head match {fit.agreement:.0%}, outline "
                        f"{fit.silhouette:.0%}) - check {' and '.join(what)} against the photo.")
    return out


def prefer_template(rules: HeadFit | None, library: HeadFit | None) -> bool:
    """Draw the library head: its silhouette beats the detected head's by TEMPLATE_MIN_GAIN
    while its layer agreement is at most TEMPLATE_SIL_AGREE_LOSS lower (B16), or (H11) its
    layer agreement beats the detected head's by TEMPLATE_AGREE_GAIN while its silhouette is
    at most TEMPLATE_AGREE_SIL_LOSS worse (or the detected model has no head parts to
    measure)."""
    if library is None:
        return False
    if rules is None:
        return True
    if (library.silhouette >= rules.silhouette + TEMPLATE_MIN_GAIN - 1e-9
            and library.agreement >= rules.agreement - TEMPLATE_SIL_AGREE_LOSS - 1e-9):
        return True
    return (library.agreement >= rules.agreement + TEMPLATE_AGREE_GAIN - 1e-9
            and library.silhouette >= rules.silhouette - TEMPLATE_AGREE_SIL_LOSS - 1e-9)


def gross_failure(fit: HeadFit | None) -> bool:
    """The head-fit check's gross failure (layer agreement under HEAD_FIT_MIN or silhouette
    under HEAD_SILHOUETTE_MIN); False without a measurement."""
    return fit is not None and (fit.agreement < HEAD_FIT_MIN or fit.silhouette < HEAD_SILHOUETTE_MIN)


def has_head_layer(model: RosetteModel) -> bool:
    """The model draws a head layer (a head, ring or points part), not only a center and
    unrecognised shapes."""
    return any(q.role in HEAD_LAYER_ROLES for q in model.parts)


def override_allowed(rules: HeadFit | None, library: HeadFit | None, partition: float,
                     head_layer: bool = True) -> bool:
    """A grossly failed rule-based head (gross_failure) is replaced by the library head
    below the color floor only when the library outline wins by TEMPLATE_OVERRIDE_GAIN and
    its tail-mapped partition reaches TEMPLATE_OVERRIDE_PARTITION; the partition bar is
    dropped when the rule-based path drew no head layer (head_layer False). A rule-based
    head without a head fit (rules None: nothing drawn in the head) counts as failed."""
    if library is None or not (rules is None or gross_failure(rules)):
        return False
    gain_ok = rules is None or library.silhouette >= rules.silhouette + TEMPLATE_OVERRIDE_GAIN - 1e-9
    return gain_ok and (not head_layer or partition >= TEMPLATE_OVERRIDE_PARTITION - 1e-9)


def library_head_path(rules: HeadFit | None, library: HeadFit | None, partition: float,
                      head_layer: bool = True) -> str | None:
    """How the best library head is used: None (it does not fit better, prefer_template),
    "floor" (its colors agree, colors_agree), "override" (the rule-based head failed,
    override_allowed: drawn flagged) or "colors" (the rule-based head is drawn with the
    colors warning, the library head waits on the head card). The override keeps its own
    conditions (B16): when override_allowed holds, the agreement bar of prefer_template's
    outline path does not apply (a failed rule-based head has no layers worth keeping)."""
    if not prefer_template(rules, library) and not override_allowed(rules, library, partition, head_layer):
        return None
    if colors_agree(partition):
        return "floor"
    return "override" if override_allowed(rules, library, partition, head_layer) else "colors"


def colors_agree(partition: float) -> bool:
    """The library head's tail-mapped partition agreement reaches TEMPLATE_MIN_PARTITION."""
    return partition >= TEMPLATE_MIN_PARTITION - 1e-9


def template_confidence(score: float) -> float:
    return round(float(min(1.0, max(0.0, FLAG_BELOW + score - TEMPLATE_FLAG_SCORE))), 3)


def with_template(model: RosetteModel, mask: np.ndarray, labels: np.ndarray, head: tuple[float, float, float],
                  frame: Frame, groups: list, merged: dict[int, int] | None = None, lib=None,
                  exclude_key: str | None = None, min_score: float | None = None) -> RosetteModel:
    """A copy of the model whose head is the best library template (one `template` part,
    the detected head parts kept in `alternatives`) when its score is at least min_score
    (TEMPLATE_MIN_SCORE), else an unchanged copy. The next best templates (up to
    TEMPLATE_CANDIDATES in all, also at least min_score) wait in `alternatives` too, for
    the head card. head: the photo's head circle (px); exclude_key skips a design
    (leave-one-out evaluation only)."""
    out = copy.deepcopy(model)
    lib = load_library() if lib is None else lib
    have = {g.id for g in out.groups}
    if not lib or not have:
        return out
    grid = photo_grid(mask, labels, head, relabel_map(out, groups, merged))
    floor = TEMPLATE_MIN_SCORE if min_score is None else min_score
    ranked = [tf for tf in rank(grid, lib, exclude_key, tail_groups=_tail_groups(out), n=TEMPLATE_CANDIDATES)
              if tf[1]["score"] >= floor]
    circle = (round(frame.x(head[0]), 3), round(frame.y(head[1]), 3), round(frame.d(head[2]), 3))
    cands = [q for q in (_template_part(t, f, have, circle) for t, f in ranked) if q is not None]
    if not cands:
        return out
    for i, q in enumerate(cands):
        q.id = TEMPLATE_PART_ID if i == 0 else f"{TEMPLATE_PART_ID}_{i + 1}"
        q.params["rank"] = i + 1
    out.alternatives = [q for q in out.parts if q.role != "tail_stripe"] + cands[1:]
    out.parts = [q for q in out.parts if q.role == "tail_stripe"] + cands[:1]
    _split_groups(out)
    return out


def _template_part(t, f: dict, have: set[int], circle: tuple[float, float, float]) -> Part | None:
    """The template part of a matched template on the photo head circle (canvas cx, cy, r),
    or None when none of its categories maps onto a model group."""
    mapping = {c: g for c, g in f["mapping"].items() if g in have}
    if not mapping:
        return None
    first = next(pc.category for pc in t.pieces if pc.category in mapping)
    pose = f.get("pose") or Pose()
    params: dict = {"template_id": t.sid, "cx": round(circle[0] + pose.dx * circle[2], 3),
                    "cy": round(circle[1] + pose.dy * circle[2], 3), "r": round(circle[2] * pose.scale, 3),
                    "rotation_deg": float(pose.rotation), "pose_dx": pose.dx, "pose_dy": pose.dy,
                    "pose_scale": pose.scale,
                    "score": round(f["score"], 3), "partition": round(f["partition"], 3), "points": int(t.points),
                    "description": describe(t)}
    for c in t.categories:
        params[f"group_{c}"] = int(mapping.get(c, mapping[first]))
    return Part(TEMPLATE_PART_ID, "template", params, int(mapping[first]), 10, template_confidence(f["score"]))


def _tail_groups(model: RosetteModel) -> dict[int, int]:
    """Color -> the group our tail stripes carry as that Color (the larger one by stripe
    count when two groups share a Color)."""
    cat = {g.id: g.category for g in model.groups}
    count: dict[int, int] = {}
    for q in model.parts:
        if q.role == "tail_stripe" and cat.get(q.color_group):
            count[q.color_group] = count.get(q.color_group, 0) + 1
    out: dict[int, int] = {}
    for gid, n in sorted(count.items(), key=lambda kv: -kv[1]):
        out.setdefault(int(cat[gid]), gid)
    return out


def _split_groups(model: RosetteModel) -> None:
    """Groups the drawn parts use stay in model.groups; the others (used only by the
    head parts waiting in alternatives) move to alternative_groups, and back."""
    pool = sorted(model.groups + model.alternative_groups, key=lambda g: g.id)
    used = {g for q in model.parts for g in _part_groups(q)}
    model.groups = [g for g in pool if g.id in used]
    model.alternative_groups = [g for g in pool if g.id not in used]


def _head_pool(model: RosetteModel) -> tuple[list[Part], list[Part]]:
    """(detected head parts, library candidates by rank) over drawn parts and alternatives."""
    pool = [q for q in model.parts if q.role != "tail_stripe"] + list(model.alternatives)
    cands = sorted((q for q in pool if q.role == "template"), key=lambda q: int(q.params.get("rank", 1)))
    return [q for q in pool if q.role != "template"], cands


def use_head(model: RosetteModel, choice: str) -> RosetteModel:
    """The model drawing the head the user picked on the head card: RULES_HEAD (the
    detected head parts) or the template_id of one of the library candidates; every other
    head waits in alternatives. Color groups follow the parts that use them. Raises
    ValueError for a choice the model does not hold."""
    out = copy.deepcopy(model)
    rules, cands = _head_pool(out)
    current = drawn_head(out)
    if choice == RULES_HEAD:
        drawn = rules
    else:
        drawn = [q for q in cands if q.params.get("template_id") == choice][:1]
        if not drawn:
            raise ValueError(f"no head {choice!r} in the model")
    # each library head keeps its own pose (from its search, or the user's edit of it)
    keep = {id(q) for q in drawn}
    out.parts = [q for q in out.parts if q.role == "tail_stripe"] + drawn
    out.alternatives = [q for q in rules + cands if id(q) not in keep]
    if out.head_states:  # head-dependent warnings follow the head
        old = set(out.head_states.get(current, {}).get("warnings", []))
        out.warnings = [w for w in out.warnings if w not in old] + [
            w for w in out.head_states.get(choice, {}).get("warnings", [])]
    _split_groups(out)
    out.info = ([x for x in out.info if not x.startswith((RECENTER_PREFIX, ALT_INFO_PREFIX))]
                + _recenter_info(out) + alt_ring_info(out.parts))  # A16: a one-color two-tone edge
    return out


def _recenter_info(model: RosetteModel) -> list[str]:
    """The information line of a drawn library head moved RECENTER_MIN or more from the
    photo's head circle by its pose search (H9), else none."""
    t = next((q for q in model.parts if q.role == "template"), None)
    if t is None:
        return []
    dx, dy = float(t.params.get("pose_dx", 0.0)), float(t.params.get("pose_dy", 0.0))
    sc = float(t.params.get("pose_scale", 1.0))
    if max(abs(dx), abs(dy), abs(sc - 1.0)) < RECENTER_MIN - 1e-9:
        return []
    return [RECENTER_INFO.format(dx=dx, dy=dy, scale=sc)]


def drawn_head(model: RosetteModel) -> str:
    """The head the model draws: a template_id, or RULES_HEAD."""
    t = next((q for q in model.parts if q.role == "template"), None)
    return str(t.params["template_id"]) if t is not None else RULES_HEAD


def use_rules_head(model: RosetteModel) -> RosetteModel:
    """The toggle: the detected head parts when the model draws a library head, else the
    first-ranked library head (the decision rule's choice)."""
    if any(q.role == "template" for q in model.parts):
        return use_head(model, RULES_HEAD)
    _, cands = _head_pool(model)
    if not cands:
        return copy.deepcopy(model)
    return use_head(model, str(cands[0].params["template_id"]))


def _with_head_state(model: RosetteModel, fit: HeadFit | None, cap: bool,
                     warnings: list[str]) -> tuple[RosetteModel, list[str]]:
    """The model with its drawn head flagged as analyze flags it (a library head drawn
    without color agreement capped at FLAGGED_CONF, then the gross head check), and the
    head-dependent warnings: `warnings` plus those of the check."""
    m = copy.deepcopy(model)
    m.warnings = []
    if cap:
        for q in m.parts:
            if q.role == "template":
                q.confidence = min(q.confidence, FLAGGED_CONF)
    m = _check_head(m, fit)
    return m, list(warnings) + m.warnings


def choose_head(rules: RosetteModel, lib: RosetteModel | None,
                fit_of: Callable[[RosetteModel], HeadFit | None],
                overlap_of: Callable[[RosetteModel, RosetteModel], float | None]) -> tuple[RosetteModel, HeadFit | None]:
    """Which head analyze draws, and every head's state for the head card. rules: the
    detected model; lib: with_template's model (or None); fit_of: the head fit of a model
    (head_fit on the photo); overlap_of: head_overlap. The best library head is drawn on
    the "floor" and "override" paths (library_head_path), else the rule-based head; on
    every path the library candidates stay in alternatives. Each head's flagged confidence
    is stored on its parts and its head-dependent warnings and head fit in
    model.head_states (keyed by template_id or RULES_HEAD), so use_head can switch with
    the same flags and checks. Returns the model and the drawn head's fit."""
    rules_fit = fit_of(rules)
    if lib is None or not any(q.role == "template" for q in lib.parts):
        return _check_head(rules, rules_fit), rules_fit
    base = list(lib.warnings)
    _, cands = _head_pool(lib)
    first = cands[0]
    states: dict[str, dict] = {}
    conf: dict[str, float] = {}
    fits: dict[str, HeadFit | None] = {}
    path = None
    for q in cands:
        tid = str(q.params["template_id"])
        m = use_head(lib, tid)
        fits[tid] = fit_of(m)
        if q is first:
            path = library_head_path(rules_fit, fits[tid], float(q.params.get("partition", 0.0)),
                                     has_head_layer(rules))
        c, states[tid] = _library_head_state(rules, m, q, fits[tid], overlap_of)
        conf.update(c)
    colors = ([TEMPLATE_COLORS_WARNING.format(tid=first.params["template_id"])]
              if path in ("colors", "override") else [])
    colors += points_count_warnings(rules, first)
    checked, warns = _with_head_state(use_head(lib, RULES_HEAD), rules_fit, False, colors)
    conf.update({x.id: x.confidence for x in checked.parts if x.role != "tail_stripe"})
    states[RULES_HEAD] = {"warnings": warns, "fit": _fit_json(rules_fit)}
    out = copy.deepcopy(lib)
    for q in out.parts + out.alternatives:
        if q.id in conf:
            q.confidence = conf[q.id]
    choice = str(first.params["template_id"]) if path in ("floor", "override") else RULES_HEAD
    out.warnings = base
    out.head_states = {}
    out = use_head(out, choice)
    out.head_states = states
    out.warnings = base + states[choice]["warnings"]

    return out, (fits[choice] if choice != RULES_HEAD else rules_fit)


def points_count_warnings(rules: RosetteModel, first: Part) -> list[str]:
    """H11: [POINTS_COUNT_WARNING] for the rule-based head when the closest library design
    (the first candidate) has colors that agree with the photo (colors_agree, anchored to
    the tails), POINTS_COUNT_MIN or more points, and no rule-based points layer has its
    count; else []. Part of the rule-based head's state (shown while it is drawn)."""
    n = int(first.params.get("points", 0))
    if (n < POINTS_COUNT_MIN or first.params.get("free_colors")
            or not colors_agree(float(first.params.get("partition", 0.0)))):
        return []
    counts = {int(q.params["count"]) for q in rules.parts if q.role == "points"}
    return [POINTS_COUNT_WARNING] if counts and n not in counts else []


def _library_head_state(rules: RosetteModel, m: RosetteModel, q: Part, fit: HeadFit | None,
                        overlap_of) -> tuple[dict[str, float], dict]:
    """A library head's flags as if drawn (m draws it): the FLAGGED_CONF cap without color
    agreement, the heads-differ warning (overlap with the rule-based head under
    TEMPLATE_DIFFERS_IOU, or no color agreement), the gross head check. Returns the
    drawn parts' confidences by id and the head state (warnings, fit)."""
    free = bool(q.params.get("free_colors"))
    agree = colors_agree(float(q.params.get("partition", 0.0))) and not free
    overlap = overlap_of(rules, m)
    differs = ([TEMPLATE_DIFFERS_WARNING.format(iou=overlap)]
               if overlap is not None and (overlap < TEMPLATE_DIFFERS_IOU or not agree) else [])
    if free:
        differs = [FREE_COLORS_WARNING] + differs
    checked, warns = _with_head_state(m, fit, not agree, differs)
    return ({x.id: x.confidence for x in checked.parts if x.role != "tail_stripe"},
            {"warnings": warns, "fit": _fit_json(fit)})


class UnknownHead(ValueError):
    """No library template has this id."""


class UncoloredHead(ValueError):
    """A library template none of whose Colors maps onto a model group."""


@dataclass
class PhotoContext:
    """What a library head chosen after analyze needs of the photo (the server keeps it
    under the model's photo_token): the prepared photo, its color labels and groups, the
    label ids assign merged away and the detected (rule-based) model."""
    prepared: Prepared
    labels: np.ndarray
    groups: list
    merged: dict[int, int]
    rules: RosetteModel
    # The rule-based head before the learned classifier (ML-1): the layers a confirmed
    # download is logged against (rosette.confirmed).
    detected: RosetteModel | None = None
    # ML-2: the SAM 2 pieces' status (`perception`), sent by the API.
    perception: dict | None = None
    # ML-4: the piece reader's status (`reader`, see reader_step), sent by the API.
    reader: dict | None = None


def add_library_head(model: RosetteModel, tid: str, ctx: PhotoContext) -> RosetteModel:
    """The model drawing library template tid, chosen on the head card outside the
    candidates: placed on the photo head circle at its own searched pose (each library
    head keeps its own pose; use_head never copies placement), colored under the tail-bound mapping (or the free one), and
    checked like the candidates (_library_head_state). Raises ValueError for an unknown
    id or a template none of whose Colors maps onto a model group."""
    t = template_by_id(tid)
    if t is None:
        raise UnknownHead(f"no library template {tid!r}")
    _, cands = _head_pool(model)
    if any(q.params.get("template_id") == tid for q in cands):
        return use_head(model, tid)
    p, rules = ctx.prepared, ctx.rules
    grid = photo_grid(p.mask, ctx.labels, p.head, relabel_map(rules, ctx.groups, ctx.merged))
    f = score_template(grid, t, _tail_groups(rules))
    fr = Frame.of(p)
    circle = (round(fr.x(p.head[0]), 3), round(fr.y(p.head[1]), 3), round(fr.d(p.head[2]), 3))
    part = _template_part(t, f, {g.id for g in rules.groups}, circle)
    if part is None:
        raise UncoloredHead(f"library template {tid!r} cannot be colored on this photo")
    if not f.get("anchored", True):
        part.params["free_colors"] = 1
    part.id = f"{LIBRARY_PART_ID}_{tid}"
    part.params["rank"] = max([int(q.params.get("rank", 1)) for q in cands] + [TEMPLATE_CANDIDATES]) + 1
    out = copy.deepcopy(model)
    out.alternatives = out.alternatives + [part]
    m = use_head(out, tid)
    conf, state = _library_head_state(rules, m, part, head_fit(p, ctx.labels, m, ctx.merged), head_overlap)
    out.head_states = {**out.head_states, tid: state}
    for q in out.alternatives:
        if q.id in conf:
            q.confidence = conf[q.id]
    return use_head(out, tid)


def _fit_json(fit: HeadFit | None) -> dict | None:
    return None if fit is None else {"agreement": round(fit.agreement, 4), "silhouette": round(fit.silhouette, 4)}


CIRCLE_UNCERTAIN = "The head circle is uncertain - check the head size and position against the photo."
# Uncertain head circle (H10): the silhouette fit and the symmetry circle disagree by at
# least this ratio without a takeover, or the takeover was decided by a coverage within
# CIRCLE_COVERAGE_MARGIN of FIT_MIN_COVERAGE.
CIRCLE_RATIO_MIN, CIRCLE_COVERAGE_MARGIN = 1.1, 0.1


def circle_warnings(p) -> list[str]:
    """[CIRCLE_UNCERTAIN] when prepare's head circle is uncertain, else [] (p: Prepared)."""
    from .vision.prepare import FIT_MIN_COVERAGE, FIT_TAKEOVER
    ratio, cov = float(p.fit_ratio), float(p.fit_coverage)
    disagree = CIRCLE_RATIO_MIN - 1e-9 <= ratio < FIT_TAKEOVER
    hinged = ratio >= FIT_TAKEOVER and abs(cov - FIT_MIN_COVERAGE) < CIRCLE_COVERAGE_MARGIN
    return [CIRCLE_UNCERTAIN] if disagree or hinged else []


def analyze(data: bytes) -> RosetteModel:
    return analyze_with_fit(data)[0]


@dataclass
class Detection:
    """What analyze sees of one photo before the head checks: the prepared photo, its
    color labels and groups, the assigned model and the label ids assign merged away."""
    prepared: Prepared
    labels: np.ndarray
    groups: list
    model: RosetteModel
    merged: dict[int, int]
    merged_print: list
    rules_model: RosetteModel | None = None  # before the learned classifier (ML-1)
    perception: dict | None = None  # ML-2: the SAM 2 pieces' status (`perception`)
    reader: dict | None = None  # ML-4: the piece reader's status (reader_step)


PIECES_OFF = "SAM 2 pieces are off (ROSETTE_SAM2=off): the rules alone read the head."
# Where the pieces of each photo are cached (rosette.perception.pieces; git-ignored).
PIECES_CACHE_DIR = DEFAULT_CACHE_DIR


def perception(p: Prepared, labels: np.ndarray, env=None) -> tuple[list | None, dict]:
    """ML-2: (pieces or None, status) of the prepared photo. The pieces come from
    rosette.perception.pieces.segment (cached per photo), each with its color_group set to the
    majority label inside its mask (label -1 ignored; None when the mask holds no label).
    status: {"device", "model", "pieces", "seconds", "cached", "status", "message"} with status "on",
    "off" (ROSETTE_SAM2=off), "missing" (no weights; the message holds the fetch command) or
    "failed" (the error text). Never raises: without pieces the rules path runs alone."""
    out = {"device": None, "model": None, "pieces": 0, "seconds": 0.0, "cached": False, "status": "off",
           "message": PIECES_OFF}
    try:
        settings = perception_settings.load(env)
        if settings.model == "off":
            return None, out
        result = segment(p, settings, cache_dir=PIECES_CACHE_DIR)
    except MissingModel as exc:
        return None, dict(out, status="missing", message=str(exc))
    except Exception as exc:  # noqa: BLE001 - pieces are optional: the rules path runs without them
        return None, dict(out, status="failed", message=f"SAM 2 pieces failed - {type(exc).__name__}: {exc}")
    for pc in result.pieces:
        g, _ = majority(labels[pc.mask])
        pc.color_group = g if g >= 0 else None
    message = "cached pieces" if result.cached else "pieces computed"
    return result.pieces, {"device": result.device, "model": result.model_id, "pieces": len(result.pieces),
                           "seconds": round(float(result.seconds), 2), "cached": bool(result.cached), "status": "on",
                           "message": message}


READER_OFF = "ML-4 reader is off (ROSETTE_READER=off): the rules head stands."
READER_COLD = "no cached reading (ROSETTE_READER=cached)"
# One llama-server per process (ML-4): started on the first cold read, stopped at exit. Batch tools
# (tools/score.py, tools/head_templates.py) default to ROSETTE_READER=cached and never start it.
_READER_SERVER = None
_READER_ATEXIT = False
_READER_LOCK = threading.Lock()


def _resolve_reader(settings) -> "reader_mod.Resolved":
    """The executable and weights the reader would use; MissingReader when absent."""
    return reader_mod.ReaderServer(settings).resolve()


def _stop_reader_server() -> None:
    global _READER_SERVER
    with _READER_LOCK:
        srv, _READER_SERVER = _READER_SERVER, None
    if srv is not None:
        srv.stop()


def _reader_post(settings) -> Callable[[str, dict], dict]:
    """reader.read's `post`: the process's llama-server, started lazily on the first request (a warm
    cache never starts it) and kept for the next photos; restarted when it exited or the settings
    changed; stopped at exit (atexit)."""
    def post(url: str, body: dict) -> dict:
        global _READER_SERVER, _READER_ATEXIT
        with _READER_LOCK:
            srv = _READER_SERVER
            alive = srv is not None and srv.proc is not None and srv.proc.poll() is None
            if not alive or srv.settings != settings:
                _READER_SERVER = None
                if srv is not None:
                    srv.stop()
                srv = reader_mod.ReaderServer(settings).start()
                _READER_SERVER = srv
                if not _READER_ATEXIT:
                    atexit.register(_stop_reader_server)
                    _READER_ATEXIT = True
        return reader_mod._http_post(srv.url + "/v1/chat/completions", body)
    return post


class _ColdReading(Exception):
    """ROSETTE_READER=cached and the photo has no cached reading: the model is not asked."""


def _cold_post(url: str, body: dict) -> dict:
    raise _ColdReading(READER_COLD)


def _candidate_labels(decisions: list, choices: list[tuple[int, ...]]) -> list[str]:
    """One description per candidate: "rules head" for candidate 0, else the options it takes."""
    out = []
    for combo in choices:
        taken = [f"{d.kind} {'/'.join(d.part_ids)} {d.options[i]}" for d, i in zip(decisions, combo) if i]
        out.append(", ".join(taken) or "rules head")
    return out


def reader_step(p: Prepared, labels: np.ndarray, pieces: list | None, model: RosetteModel, env=None,
                merged: dict[int, int] | None = None) -> tuple[RosetteModel, dict]:
    """ML-4: read the head pieces (rosette.perception.reader), group them (vision.groups), open the
    head decisions and span the candidates (vision.candidates; candidate 0 is `model`), render and
    judge them against the photo (vision.judge) and write the verdict on the chosen head
    (judge.flag). ROSETTE_READER: "off" returns at once; "on", "auto" and "cached" need the
    executable and the weights to resolve (else status "missing" with the install line), then
    `pieces` that are not None (else "skipped"); "on" and "auto" ask the model on a cold cache
    (one llama-server per process, _reader_post), "cached" never does: a photo without a
    cached reading is "skipped" (READER_COLD). Returns (model, status); when the reader does
    not run, reads nothing, opens no decision or fails, `model` itself comes back untouched.
    status: {"status", "message", "seconds", "cached", "images", "decisions", "winner", "margin",
    "note", "tie", "removed"} with status "on", "off", "missing", "skipped" (no pieces, no head
    piece to read, or no cached reading) or "failed" (the error text); message is the judge's
    line when it judged. Never raises. The candidates change counts, z, kinds and the centre
    state only, never a colour group, so the chosen head needs no second assign."""
    t0 = time.perf_counter()
    out = {"status": "off", "message": READER_OFF, "seconds": 0.0, "cached": False, "images": 0, "decisions": 0,
           "winner": None, "margin": None, "note": "", "tie": [], "removed": []}

    def done(**kw) -> dict:
        return {**out, "seconds": round(time.perf_counter() - t0, 2), **kw}
    try:
        settings = perception_settings.load(env)
        if settings.reader == "off":
            return model, out
        try:
            resolved = _resolve_reader(settings)
        except reader_mod.MissingReader as exc:
            return model, done(status="missing", message=str(exc))
        if pieces is None:
            return model, done(status="skipped", message="no pieces")
        post = _cold_post if settings.reader == "cached" else _reader_post(settings)
        try:
            reading = reader_mod.read(p, pieces, settings, cache_dir=reader_mod.DEFAULT_READER_CACHE, post=post)
        except _ColdReading:
            return model, done(status="skipped", message=READER_COLD)
        note = "; ".join(x for x in (resolved.note, reading.note) if x)
        seen = {"cached": bool(reading.cached), "images": int(reading.images), "note": note}
        if not reading.marks:
            return model, done(status="skipped", message="no head pieces to read", **seen)
        found = vgroups.piece_groups(reading, pieces, p.head, Frame.of(p).scale)
        decisions = vcandidates.open_decisions(model, found, reading)
        if not decisions:
            return model, done(status="on", message="no open decisions: the rules head stands", winner=0, **seen)
        choices = vcandidates.candidate_choices(model, decisions)
        heads = vcandidates.candidate_heads(model, decisions)
        verdict = vjudge.judge(heads, p, labels, pieces, decisions=_candidate_labels(decisions, choices),
                               merged=merged)
        chosen = vjudge.flag(heads[verdict.winner], verdict)
        base = heads[verdict.winner]
        lines = chosen.warnings[len(base.warnings):] + chosen.info[len(base.info):]
        if verdict.note:
            seen["note"] = "; ".join(x for x in (note, verdict.note) if x)
        return chosen, done(status="on", message=lines[0] if lines else "the rules head stands",
                            decisions=len(decisions), winner=int(verdict.winner),
                            margin=round(float(verdict.margin), 4), tie=list(verdict.tie),
                            removed=list(verdict.removed), **seen)
    except reader_mod.MissingReader as exc:
        return model, done(status="missing", message=str(exc))
    except Exception as exc:  # noqa: BLE001 - the reader is optional: the rules head stands
        return model, done(status="failed", message=f"ML-4 reader failed - {type(exc).__name__}: {exc}")


# R32: the rebuilt pleat rings are widened outward by HUE_TUCK of the head radius (each ring's
# pleats run on under the ring outside it: A3-15's designer rings end at 0.58 and 0.78 head radii,
# the photo shows them to 0.40 and 0.60), and each ring starts where the ring inside it starts,
# so no head color shows between them (the inner ring lies on top).
HUE_TUCK = 0.18


# R33: a ring starting at most SEAM_GAP_UNITS outside the center or the ring inside it, over a
# gap that shows the head disc in the ring's own color, runs in to meet it: the gap is the same
# ribbon, not a part of its own (A9-15: a 9-unit red band between the badge and the red pleat
# ring drew an extra circle line and a pleated edge inside the red).
SEAM_GAP_UNITS = 12.0
SEAM_INFO = "A narrow band of the head's color between rings was joined to the ring around it."


# R44: a head of only bows over a star, no center found, the bows a light tint of the star's
# hue (S2-15: pink loops over a wine star) is the designer's rose head: a rose of the star's
# color over the center, the star's points and the loops running in under it, the loops edged
# in the star's color. The rose stands where the loops start in the photo.
ROSE_BOW_HUE_DEG, ROSE_BOW_DL = 20.0, 25.0
ROSE_BOW_SHARE = 0.8  # rose radius / where the loops start in the photo (designer rose 0.36 of the head)
ROSE_BOW_SPAN = 34.0  # a loop's span at its end, edges included (designer S2-15)
ROSE_BOW_BASE = 0.4  # a loop's width at the rose / its widest (designer S2-15)
ROSE_BOW_EDGE = 0.22  # each edge strip's share of the loop's width (designer S2-15)
ROSE_BOW_TUCK = 0.5  # the loops start at this share of the rose radius


def _hue_l(rgb) -> tuple[float, float]:
    import cv2
    lab = cv2.cvtColor(np.array([[rgb]], np.float32) / 255.0, cv2.COLOR_RGB2Lab)[0, 0]
    return float(np.degrees(np.arctan2(lab[2], lab[1]))), float(lab[0])


def _rose_over_bows(parts: list, groups: list) -> bool:
    head = [q for q in parts if q.role not in ("tail_stripe",)]
    bows = [q for q in head if q.role == "points" and q.params.get("kind") == "bow"]
    stars = [q for q in head if q.role == "points" and q.params.get("kind") == "star"]
    if (len(bows) != 1 or len(stars) != 1 or any(q.role in ("center", "ring", "head") for q in head)
            or bows[0].color_group == stars[0].color_group):
        return False
    bow, star = bows[0], stars[0]
    rgb = {g.id: g.rgb for g in groups}
    if bow.color_group not in rgb or star.color_group not in rgb:
        return False
    hb, lb = _hue_l(rgb[bow.color_group])
    hs, ls = _hue_l(rgb[star.color_group])
    if abs((hb - hs + 180.0) % 360.0 - 180.0) > ROSE_BOW_HUE_DEG or lb - ls < ROSE_BOW_DL:
        return False
    rr = round(ROSE_BOW_SHARE * float(bow.params["r_inner"]), 3)
    star.params["r_inner"] = rr
    ri, ro = ROSE_BOW_TUCK * rr, float(bow.params["r_outer"])
    w = float(bow.params["width_ratio"]) * (float(bow.params["r_outer"]) - float(bow.params["r_inner"]))
    # the photo's loops spread flat; the designer's span ROSE_BOW_SPAN degrees at their ends
    w = min(w, 2.0 * ro * np.sin(np.radians(ROSE_BOW_SPAN / 2.0)))
    bow.params["r_inner"] = round(ri, 3)
    bow.params["width_ratio"] = round(w / (ro - ri), 4)  # the loops keep their width
    bow.params["inner_ratio"] = ROSE_BOW_BASE  # and narrow in toward the rose
    parts.append(Part(f"edge-{bow.id}", "points", dict(bow.params, edge_width=round(ROSE_BOW_EDGE * w, 3)),
                      star.color_group, bow.z + 1, bow.confidence))
    parts.append(Part("center", "center", {"cx": star.params["cx"], "cy": star.params["cy"], "r": rr},
                      star.color_group, 50, star.confidence))
    return True


# R47: a star whose valleys lie past half its reach, between loops of its count half a step
# away (S27-15: purple corners between white loops), is a second set of loops: the photo's loop
# ends read as blunt star points. Both sets take the loops' width and run in under the ring
# around the center, the star's loops on top (designer S27-15: the purple loops over the white).
STAR_LOOP_VALLEY = 0.5


def _star_as_loops(parts: list) -> bool:
    ring = min((q for q in parts if q.role == "ring"), key=lambda q: float(q.params["r_inner"]), default=None)
    bows = [q for q in parts if q.role == "points" and q.params.get("kind") == "bow"
            and q.params.get("edge_width") is None]
    if sum(1 for q in parts if q.role == "points") != 2:  # just the star and the loops (not S29-15's layers)
        return False
    done = False
    for a in parts:
        pa = a.params
        if (a.role != "points" or pa.get("kind") != "star" or pa.get("tips") or pa.get("leaf")
                or float(pa["r_inner"]) <= STAR_LOOP_VALLEY * float(pa["r_outer"])):
            continue
        n = int(pa["count"])
        step = 360.0 / n
        mates = [b for b in bows if int(b.params["count"]) == n and b.color_group != a.color_group
                 and abs(((float(pa["rotation_deg"]) - float(b.params["rotation_deg"])) % step) - step / 2) <= 8.0]
        if not mates or ring is None:
            continue
        b = mates[0]
        ri = float(ring.params["r_inner"])
        w = float(b.params["width_ratio"]) * (float(b.params["r_outer"]) - float(b.params["r_inner"]))
        for q in (a, b):
            ro = float(q.params["r_outer"])
            q.params.update(kind="bow", r_inner=round(ri, 3), width_ratio=round(w / max(ro - ri, 1e-6), 4))
        if a.z <= b.z:  # the read star's loops lie on top (designer S27-15: purple over white)
            a.z, b.z = b.z, a.z
            if a.z == b.z:
                a.z += 1
        done = True
    return done


# R50: diamonds half a step from edged loops of their count, with a ring around the center
# (S30-15: lilac points between heart-print loops edged in pink), are a star behind the loops:
# its body runs in under them to the ring, and the loops run in under the ring, as the designer
# draws them. Read from the photo's tips, the library head does not replace it.
def _diamonds_behind_loops(parts: list) -> bool:
    ring = max((q for q in parts if q.role == "ring"), key=lambda q: float(q.params["r_outer"]), default=None)
    if ring is None:
        return False
    loops = [q for q in parts if q.role == "points" and q.params.get("kind") == "bow" and q.params.get("edge_width") is None]
    edged = {(int(q.params["count"]), round(float(q.params["rotation_deg"]), 1)) for q in parts
             if q.role == "points" and q.params.get("kind") == "bow" and q.params.get("edge_width") is not None}
    done = False
    for a in parts:
        pa = a.params
        if a.role != "points" or pa.get("kind") != "diamond":
            continue
        n = int(pa["count"])
        step = 360.0 / n
        mates = [b for b in loops if int(b.params["count"]) == n
                 and (n, round(float(b.params["rotation_deg"]), 1)) in edged
                 and abs(((float(pa["rotation_deg"]) - float(b.params["rotation_deg"])) % step) - step / 2) <= 8.0]
        if not mates:
            continue
        ri = float(ring.params["r_outer"])
        pa.update(kind="star", r_inner=round(ri, 3), tips=1)
        a.z = min(b.z for b in mates) - 1
        for q in parts:  # the loops and their edges run in under the ring
            if q.role == "points" and q.params.get("kind") == "bow" and int(q.params["count"]) == n:
                ro = float(q.params["r_outer"])
                w = float(q.params["width_ratio"]) * (ro - float(q.params["r_inner"]))
                q.params.update(r_inner=round(float(ring.params["r_inner"]), 3), width_ratio=round(w / (ro - float(ring.params["r_inner"])), 4))
        done = True
    return done


# R51: one loop set on a star of its count half a step away, around a center with no ring or
# head disc (S31-15: glitter loops edged in lilac on a blue star, a lilac rose, a blue flower over
# the loops), is the designer's head: the star and the loops run in under the center, the loops
# edged in the center's color, and a 6-point flower of the star's color lies over the loops'
# bases. Odd point sets of the center's color (read from the loop edges) go. Proportions are the
# designer's (S31-15, share of the star's tips).
FLOWER_LOOPS = (0.43, 0.92, 0.42, 0.15)  # loops: inner, outer, width, edge share
FLOWER_TIP = 0.62  # the flower over the loops: its tips
FLOWER_VALLEY = 0.42  # and its valleys
FLOWER_BACK_VALLEY = 0.55  # the star behind the loops: its valleys
FLOWER_MIN_LOOP_WIDTH = 0.6  # the read loops' width / length at least


def _loops_on_star_flower(parts: list) -> bool:
    c = next((q for q in parts if q.role == "center"), None)
    if c is None or any(q.role in ("ring", "head") for q in parts):
        return False
    bows = [q for q in parts if q.role == "points" and q.params.get("kind") == "bow"]
    stars = [q for q in parts if q.role == "points" and q.params.get("kind") == "star"]
    if len(bows) != 1 or len(stars) != 1 or bows[0].params.get("edge_width") is not None:
        return False
    bow, star = bows[0], stars[0]
    n = int(bow.params["count"])
    if n != int(star.params["count"]) or n < 5 or star.params.get("tips"):
        return False
    if float(bow.params["width_ratio"]) < FLOWER_MIN_LOOP_WIDTH:  # thin strips (S56-17B's loop edges)
        return False
    step = 360.0 / n
    if abs(((float(star.params["rotation_deg"]) - float(bow.params["rotation_deg"])) % step) - step / 2) > 8.0:
        return False
    if len({c.color_group, bow.color_group, star.color_group}) != 3:
        return False
    tip, rc = float(star.params["r_outer"]), float(c.params["r"])
    ri, ro = FLOWER_LOOPS[0] * tip, FLOWER_LOOPS[1] * tip
    w = FLOWER_LOOPS[2] * tip
    bow.params.update(r_inner=round(ri, 3), r_outer=round(ro, 3), width_ratio=round(w / (ro - ri), 4))
    star.params.update(r_inner=round(FLOWER_BACK_VALLEY * tip, 3), tips=1)
    star.z, bow.z = 20, 21
    parts[:] = [q for q in parts if not (q.role in ("points", "unknown") and q.color_group == c.color_group)]
    parts.append(Part(f"edge-{bow.id}", "points", dict(bow.params, edge_width=round(FLOWER_LOOPS[3] * w, 3)),
                      c.color_group, 22, bow.confidence))
    parts.append(Part("points-flower", "points", dict(star.params, r_inner=round(max(rc, FLOWER_VALLEY * tip), 3),
                                                      r_outer=round(FLOWER_TIP * tip, 3)),
                      star.color_group, 23, star.confidence))
    return True


# R53: no center, and a small blunt star in the middle of the head (S47-15: the white rose's
# petals read as a 6-point star inside the pink loops) is the rose: a center of its color at its
# reach, the loops running in under it.
ROSE_STAR_REACH = 0.45  # the star's tips / the head's reach at most
ROSE_STAR_BLUNT = 0.6  # its valleys / its tips at least


def _rose_from_small_star(parts: list) -> bool:
    if any(q.role == "center" for q in parts):
        return False
    pts = [q for q in parts if q.role == "points"]
    reach = max([float(q.params["r_outer"]) for q in pts]
                + [float(q.params["r"]) for q in parts if q.role == "head"], default=0.0)
    for a in pts:
        pa = a.params
        if (pa.get("kind") == "star" and float(pa["r_outer"]) <= ROSE_STAR_REACH * reach
                and float(pa["r_inner"]) >= ROSE_STAR_BLUNT * float(pa["r_outer"])):
            rr = float(pa["r_outer"])
            parts.remove(a)
            parts.append(Part("center", "center", {"cx": pa["cx"], "cy": pa["cy"], "r": round(rr, 3)},
                              a.color_group, 50, a.confidence))
            for q in parts:  # the loops run in under the rose
                if q.role == "points" and q.params.get("kind") == "bow" and float(q.params["r_inner"]) > 0.6 * rr:
                    ro, ri = float(q.params["r_outer"]), float(q.params["r_inner"])
                    w = float(q.params["width_ratio"]) * (ro - ri)
                    q.params.update(r_inner=round(0.6 * rr, 3), width_ratio=round(w / (ro - 0.6 * rr), 4))
            return True
    return False


def _taper_edged_bows(parts: list) -> bool:
    """R45: a loop set laid on a wider loop set of another color, one count and angle (S2-15B:
    blue loops on white), is the designer's edged loops: both narrow toward the center to
    ROSE_BOW_BASE of their widest, as in R44, and start together under the center."""
    if not any(q.role == "center" for q in parts):
        return False
    bows = [q for q in parts if q.role == "points" and q.params.get("kind") == "bow"
            and q.params.get("edge_width") is None and q.params.get("inner_ratio") is None]
    done = False
    for a in bows:
        for b in bows:
            if a is b or a.color_group == b.color_group or a.z <= b.z:
                continue
            if int(a.params["count"]) != int(b.params["count"]):
                continue
            step = 360.0 / int(a.params["count"])
            d = (float(a.params["rotation_deg"]) - float(b.params["rotation_deg"])) % step
            if min(d, step - d) > 3.0:
                continue
            wa = float(a.params["width_ratio"]) * (float(a.params["r_outer"]) - float(a.params["r_inner"]))
            wb = float(b.params["width_ratio"]) * (float(b.params["r_outer"]) - float(b.params["r_inner"]))
            if wb <= wa:
                continue
            ri = min(float(a.params["r_inner"]), float(b.params["r_inner"]))
            for q, w in ((a, wa), (b, wb)):
                q.params["r_inner"] = round(ri, 3)
                q.params["width_ratio"] = round(w / (float(q.params["r_outer"]) - ri), 4)
                q.params["inner_ratio"] = ROSE_BOW_BASE
            done = True
    return done


def close_seams(parts: list) -> bool:
    head = next((q for q in parts if q.role == "head"), None)
    center = next((q for q in parts if q.role == "center"), None)
    if head is None and center is None:
        return False
    edges = ([float(center.params["r"])] if center is not None else [])
    changed = False
    prev = None
    for q in sorted((q for q in parts if q.role == "ring"), key=lambda q: float(q.params["r_inner"])):
        ri = float(q.params["r_inner"])
        inner = max((e for e in edges if e <= ri), default=None)
        if (prev is not None and 0 <= ri - float(prev.params["r_outer"]) <= SEAM_GAP_UNITS and prev.z > q.z):
            # the ring inside lies on top: this ring runs in under it, so one edge shows
            q.params["r_inner"] = prev.params["r_inner"]
            changed = True
        elif (inner is not None and 0 < ri - inner <= SEAM_GAP_UNITS
                and ((head is not None and q.color_group == head.color_group)
                     or (center is not None and inner == float(center.params["r"])))):
            # a gap of the head's color, or any gap at the center (it lies on top of every ring)
            q.params["r_inner"] = round(inner, 3)
            changed = True
        edges.append(float(q.params["r_outer"]))
        prev = q
    return changed


# R49: a center far smaller than the ring of its own color right around it (S29-15: a 7-unit
# center inside a wine ring at 40-52, the badge's print read apart) is that badge: the center
# grows to the ring's outer edge and the ring goes.
BADGE_SPECK = 0.25  # center radius / the ring's inner radius at most


def _fill_badge(parts: list) -> bool:
    c = next((q for q in parts if q.role == "center"), None)
    if c is None:
        return False
    rc = float(c.params["r"])
    rings = sorted((q for q in parts if q.role == "ring"), key=lambda q: float(q.params["r_inner"]))
    if not rings or rings[0].color_group != c.color_group or rc > BADGE_SPECK * float(rings[0].params["r_inner"]):
        return False
    c.params["r"] = rings[0].params["r_outer"]
    parts[:] = [q for q in parts if q is not rings[0] and not (q.role == "unknown" and float(q.params.get("r", 0)) <= float(c.params["r"]))]
    return True


# R54: a plain badge (the outer star's color) inside two nested stars, with the band just outside
# it mixed light and shade of the inner star's color and the badge's (S29-15: the pink pleats
# round the wine badge read as pink and wine shade, half and half, so no ring was found). The
# band is a pleated ring of the inner star's color from the badge out, on top of the stars; the
# designer's scalloped ring there reaches 1.44 of the badge (S29-15), at most to the outer
# star's valleys.
BADGE_PLEATS = 1.44  # ring outer radius / badge radius (designer S29-15)
PLEAT_BAND = 0.15  # band sampled outside the badge, in badge radii
PLEAT_MIX = (0.3, 0.7)  # the inner star's share of that band: light and shade, not a flat body


def _pleats_round_badge(parts: list, p, labels: np.ndarray) -> bool:
    c = next((q for q in parts if q.role == "center"), None)
    stars = [q for q in parts if q.role == "points" and q.params.get("kind") == "star"]
    if c is None or len(stars) != 2 or any(q.role == "ring" for q in parts):
        return False
    if len([q for q in parts if q.role == "points"]) - len(stars) > 2:
        return False
    outer, inner = sorted(stars, key=lambda q: -float(q.params["r_outer"]))
    if int(outer.params["count"]) != int(inner.params["count"]):
        return False
    if c.color_group != outer.color_group or inner.color_group == c.color_group:
        return False
    rc = float(c.params["r"])
    r1 = min((1.0 + PLEAT_BAND) * rc, float(inner.params["r_inner"]))
    if r1 - rc < 2.0:
        return False
    fr = Frame.of(p)
    cx = p.axis_x + (float(c.params["cx"]) - 300.0) * fr.scale
    cy = fr.py(float(c.params["cy"]))
    radii = np.arange(rc + 1.0, r1, 1.0) * fr.scale
    from .vision.parts_common import polar_sample
    v = polar_sample(labels, cx, cy, radii, 360).ravel()
    v = v[(v == inner.color_group) | (v == c.color_group)]
    if v.size == 0:
        return False
    share = float((v == inner.color_group).mean())
    if not PLEAT_MIX[0] <= share <= PLEAT_MIX[1]:
        return False
    ro = min(BADGE_PLEATS * rc, float(outer.params["r_inner"]))
    parts.append(Part("ring-pleats", "ring", {"cx": c.params["cx"], "cy": c.params["cy"], "r_inner": round(rc, 3),
                                              "r_outer": round(ro, 3), "pinked": 1},
                      inner.color_group, 30, 0.5))
    return True


# R55: a star laid on a longer star of its count and axes in another color (its edge) with a
# third points layer of that count half a step off on top of it (s34-15: black glitter kites
# over the white star edged lavender). The third layer lies between the edge and the star:
# only its tips show between the star's points (the owner's check of s34-15, as the designer),
# and the star is the edge star scaled down (its valleys in by the same share as its tips), so
# the edge shows as an even border (the owner's check of s34-15). Kites only, and no layer
# read from tips (S17-15's stacked stars, S26-15's loops keep their order).
EDGE_STAR_AXES_DEG = 3.0  # star and edge star on the same axes
EDGE_STAR_INNER = 0.05  # and the same valleys, within this share of the radius
HALF_STEP_DEG = 8.0  # the third layer half a step off, within this


def _points_under_edged_star(parts: list) -> bool:
    pts = [q for q in parts if q.role == "points"]
    stars = [q for q in pts if q.params.get("kind") == "star"]
    for star in stars:
        n = int(star.params["count"])
        step = 360.0 / n
        rot = float(star.params["rotation_deg"])
        edge = next((e for e in stars if e is not star and int(e.params["count"]) == n
                     and e.color_group != star.color_group
                     and float(e.params["r_outer"]) > float(star.params["r_outer"])
                     and abs(((float(e.params["rotation_deg"]) - rot + step / 2) % step) - step / 2) <= EDGE_STAR_AXES_DEG
                     and abs(float(e.params["r_inner"]) - float(star.params["r_inner"]))
                     <= EDGE_STAR_INNER * float(star.params["r_outer"])), None)
        if edge is None:
            continue
        if star.params.get("tips") or edge.params.get("tips"):
            continue
        mid = [q for q in pts if q is not star and q is not edge and int(q.params["count"]) == n
               and q.params.get("kind") == "diamond" and not q.params.get("tips")
               and q.color_group not in (star.color_group, edge.color_group)
               and abs(((float(q.params["rotation_deg"]) - rot) % step) - step / 2) <= HALF_STEP_DEG]
        if len(mid) != 1 or edge.z < mid[0].z < star.z:
            continue
        lo, md, hi = sorted((edge.z, mid[0].z, star.z))
        if md == lo or md == hi:  # make room: three distinct levels
            md, hi = lo + 1, max(hi, lo + 2)
        edge.z, mid[0].z, star.z = lo, md, hi
        star.params["r_inner"] = round(float(edge.params["r_inner"]) * float(star.params["r_outer"])
                                       / float(edge.params["r_outer"]), 3)
        return True
    return False


# R56: no points layer read, and a head of one color whose outline has a clear count of lobes
# around a rose of another color (S36-15: six pointed red satin petals round a white rose on a
# black ground; the rules read rings and blobs and a library head of square loops won). The head
# is a petal flower: petals of the head's color with the outline's count, phase and tip reach
# around the rose's center, starting under the rose; the rose is the center. The designer's
# petals (S36-15, the owner's check: as the reference) run from under the rose's edge to the tip, widest at a third of that length, 1.05 times as wide as long (geometry
# FLOWER_PROFILE), and the rose's radius is the tip radius / 2.55 (the photo's white reads
# smaller, 1 / 3.0, its folds shaded). The head circle is not used: it takes in the
# top of the tails (S36-15: 45 px below the rose).
FLOWER_COUNTS = range(4, 10)
FLOWER_PEAK = 2.0  # the best lobe count's strength over the next one's at least
FLOWER_DEPTH = 0.3  # (deepest valley to highest tip) / tip radius at least
FLOWER_TAILS_DEG = 45.0  # the sector round straight down left out (the tails)
FLOWER_ROSE_MIN = 0.02  # rose area / head circle area at least
FLOWER_WIDTH = 1.05  # petal width / length (designer S36-15)
FLOWER_START = 0.85  # petals start at this share of the rose radius: their bases tuck under it (designer S36-15)
FLOWER_ROSE = 2.55  # tip radius / rose radius (designer S36-15)


# R57: as R56, but the petals alternate two colors and the rose has the color of one set, joined
# to those petals (S36-15B: red and white petals round a red rose). The flower's middle is the
# center of the largest circle inside the head's outline (above the tails); each lobe's color is
# read along its axis; two sets of n / 2 petals half a step apart, the set of the rose's color
# underneath (designer S36-15B: its second color's petals on top).
ROSE_COMPACT = 0.6  # rose area / circle through its farthest pixel at least (a rose, not petals)
LOBE_COLOR = (0.45, 0.75)  # a lobe's color read along its axis between these shares of the reach
LOBE_SHARE = 0.6  # that color's share there at least
ROSE_SAMPLE = 0.25  # the rose's color read within this share of the reach


def _flower_lobes(p, mx: float, my: float, r: float):
    """(count, tip angle, reach in px, profile window ok) of the outline round (mx, my), or None."""
    from .vision.parts_common import polar_sample
    radii = np.arange(0.0, 1.6 * r, 1.0)
    m = polar_sample(p.mask.astype(np.int32), mx, my, radii, 360, fill=0)
    prof = np.array([radii[np.flatnonzero(m[:, j])].max() if m[:, j].any() else 0.0 for j in range(360)])
    th = np.radians(np.arange(360.0))
    win = np.abs(np.degrees(th) - 90.0) > FLOWER_TAILS_DEG
    pw, tw = prof[win], th[win]
    if pw.max() <= 0 or (pw.max() - pw.min()) / pw.max() < FLOWER_DEPTH:
        return None
    x = pw - pw.mean()
    spec = {n: np.sum(x * np.exp(-1j * n * tw)) for n in FLOWER_COUNTS}
    mags = sorted((abs(v), n) for n, v in spec.items())
    if mags[-1][0] < FLOWER_PEAK * mags[-2][0]:
        return None
    n = mags[-1][1]
    return n, (float(np.degrees(np.angle(spec[n]))) / n) % (360.0 / n), float(np.percentile(pw, 95))


def _petal_flower(p, labels: np.ndarray, parts: list) -> list | None:
    if any(q.role in ("points", "template") for q in parts):
        return None
    from scipy import ndimage
    cx, cy, r = (float(v) for v in p.head)
    h, w = labels.shape
    yy, xx = np.mgrid[:h, :w]
    inside = (np.hypot(xx - cx, yy - cy) <= r) & p.mask & (labels >= 0)
    if not inside.any():
        return None
    head_g = int(np.bincount(labels[inside]).argmax())
    rose = None
    for g in {int(v) for v in np.unique(labels[inside])} - {head_g}:
        lab, n = ndimage.label((labels == g) & inside)
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            if xs.size < FLOWER_ROSE_MIN * np.pi * r * r:
                continue
            mx, my = xs.mean(), ys.mean()
            far = float(np.hypot(xs - mx, ys - my).max())
            if xs.size < ROSE_COMPACT * np.pi * far * far:
                continue  # petals joined to a rose (R57), not a rose
            d = np.hypot(mx - cx, my - cy)
            if rose is None or d < rose[0]:
                rose = (d, mx, my, g)
    lobes = None
    if rose is not None:  # R56: one color round a rose of its own
        _, mx, my, rose_g = rose
        lobes = _flower_lobes(p, mx, my, r)
    if lobes is not None:
        n, rot, reach = lobes
        sets = [(head_g, n, rot)]
    else:  # R57: two colors alternating, the rose joined to one set
        dt = ndimage.distance_transform_edt(p.mask[: int(cy + 0.2 * r)])
        my, mx = (float(v) for v in np.unravel_index(np.argmax(dt), dt.shape))
        lobes = _flower_lobes(p, mx, my, r)
        if lobes is None or lobes[0] % 2:
            return None
        n, rot, reach = lobes
        from .vision.parts_common import polar_sample
        radii = np.arange(LOBE_COLOR[0] * reach, LOBE_COLOR[1] * reach, 1.0)
        cols = []
        for k in range(n):
            a = np.radians(rot + 360.0 * k / n)
            v = labels[np.clip(np.rint(my + radii * np.sin(a)).astype(int), 0, h - 1),
                       np.clip(np.rint(mx + radii * np.cos(a)).astype(int), 0, w - 1)]
            v = v[v >= 0]
            if v.size == 0:
                return None
            g = int(np.bincount(v).argmax())
            if (v == g).mean() < LOBE_SHARE:
                return None
            cols.append(g)
        a_g, b_g = cols[0], cols[1]
        if a_g == b_g or any(c != (a_g, b_g)[k % 2] for k, c in enumerate(cols)):
            return None
        v = polar_sample(labels, mx, my, np.arange(1.0, ROSE_SAMPLE * reach, 1.0), 72).ravel()
        v = v[v >= 0]
        if v.size == 0:
            return None
        rose_g = int(np.bincount(v).argmax())
        step = 360.0 / n
        sets = [(a_g, n // 2, rot), (b_g, n // 2, (rot + step) % (2 * step))]
        sets.sort(key=lambda t: t[0] != rose_g)  # the rose's color underneath
    from .vision.layer_classify import KEEP_PARAM
    fr = Frame.of(p)
    ccx, ccy = round(float(fr.x(mx)), 3), round(float(fr.y(my)), 3)
    ro = reach / fr.scale
    ri = FLOWER_START * ro / FLOWER_ROSE
    out = [q for q in parts if q.role not in ("unknown", "ring", "center", "head")]
    for k, (g, cnt, rt) in enumerate(sets):
        pid = "points-petals" if len(sets) == 1 else f"points-petals-{k}"
        out.append(Part(pid, "points", {"cx": ccx, "cy": ccy, "kind": "petal", "count": cnt,
                                        "r_inner": round(ri, 3), "r_outer": round(ro, 3),
                                        "rotation_deg": round(rt, 3), "width_ratio": FLOWER_WIDTH,
                                        "flower": 1, "tips": 1, KEEP_PARAM: 1}, g, 20 + k, 0.6))
    out.append(Part("center", "center", {"cx": ccx, "cy": ccy, "r": round(ro / FLOWER_ROSE, 3)}, rose_g, 50, 0.6))
    return out


# R58: one points layer of a third color, eight or more, standing on the head just outside a ring
# of the head's color (S37-15: gold glitter points between red satin points on red pleats, the
# red points the pleats' color so never read; the review run read the gold as square loops). The
# gold is a star; a star of the head's color lies on it half a step off, under the ring. Designer
# S37-15 (the red star's sides cut the gold pieces): red tips at 0.88 and valleys at 0.53 of the
# gold tips' radius. A thin rim of the gold's color hugging the center (S37-15: the badge's gold rim)
# makes the center one rose of that color out to the rim (designer S37-15: a Color 2 rose, 0.3 of
# the head; the owner's check).
SATIN_TIP = 0.88
SATIN_RIM = 0.1  # rim width / head radius at most
# the pleated ring inside ends at this share of the head radius, the red points rising from its edge
# (designer S37-15: scallops 0.56-0.66; the read ring, 0.65, takes in the red points' bases; the
# owner's check)
SATIN_RING = 0.6
SATIN_RIM_GAP = 0.05  # the rim starts within this share of the head radius outside the center
SATIN_VALLEY = 0.53
SATIN_RING_GAP = 0.1  # the ring's outer edge within this share of the points' inner radius
SATIN_MIN_COUNT = 8
# the gold tips reach at least this share of the head radius (S37-15's photo 0.95, designer 1.0;
# the glitter thins toward a sharp tip, so the read stops short, 0.87)
SATIN_GOLD_TIP = 0.95


def _star_under_satin_star(parts: list) -> bool:
    from .vision.layer_classify import KEEP_PARAM
    head = next((q for q in parts if q.role == "head"), None)
    pts = [q for q in parts if q.role == "points"]
    if head is None or len(pts) != 1:
        return False
    gold = pts[0]
    n = int(gold.params["count"])
    if n < SATIN_MIN_COUNT or gold.color_group == head.color_group or gold.params.get("kind") not in ("bow", "star", "diamond"):
        return False
    ri = float(gold.params["r_inner"])
    ring = next((q for q in parts if q.role == "ring" and q.color_group == head.color_group
                 and abs(float(q.params["r_outer"]) - ri) <= SATIN_RING_GAP * ri), None)
    if ring is None:
        return False
    ring.params["r_outer"] = round(min(float(ring.params["r_outer"]), SATIN_RING * float(head.params["r"])), 3)
    ro = max(float(gold.params["r_outer"]), SATIN_GOLD_TIP * float(head.params["r"]))
    gold.params.update(kind="star", r_inner=round(SATIN_VALLEY * ro, 3), r_outer=round(ro, 3), tips=1, **{KEEP_PARAM: 1})
    gold.params.pop("edge_width", None)
    parts.append(Part("points-satin", "points", {
        "cx": gold.params["cx"], "cy": gold.params["cy"], "kind": "star", "count": n,
        "r_inner": round(SATIN_VALLEY * ro, 3), "r_outer": round(SATIN_TIP * ro, 3),
        "rotation_deg": round((float(gold.params["rotation_deg"]) + 180.0 / n) % (360.0 / n), 3),
        "width_ratio": float(gold.params.get("width_ratio", 0.5)), "tips": 1, KEEP_PARAM: 1},
        head.color_group, gold.z + 1, 0.5))
    c = next((q for q in parts if q.role == "center"), None)
    if c is not None:
        rc, rim_max = float(c.params["r"]), SATIN_RIM * float(head.params["r"])
        rim = next((q for q in parts if q.role == "ring" and q.color_group == gold.color_group
                    and abs(float(q.params["r_inner"]) - rc) <= SATIN_RIM_GAP * float(head.params["r"])
                    and float(q.params["r_outer"]) - rc <= rim_max), None)
        if rim is not None:
            c.color_group, c.params["r"] = rim.color_group, rim.params["r_outer"]
            parts.remove(rim)
    return True


# R59: the head's color group holds a satin star of a darker (redder) hue lying on pleats of a
# lighter hue, and another plain group has the lighter hue (S38-15: a dark orange satin star on
# orange pleats over glitter loops; one group took the star and the shaded pleats, the head was
# read as the star's color and the star as a ring). The pleats are the lighter group, the star is
# read from the dark-hue pixels: the run outward from just outside the rose along each ray, its
# count and phase from the runs' lobes, its tips the median run on the tip axes. Designer S38-15:
# leaf-shaped arms from under the rose, a rose of the pleats' color at 0.33 of the head.
SATIN_SPLIT_GAP = 5.0  # the group's two hue clusters (Otsu split in the head) at least this far apart (degrees)
SATIN_HUE_GAP = 8.0  # the light group's median hue at least this far above the head group's (degrees)
SATIN_ANCHOR = 0.4  # runs start at this share of the head radius (outside the rose)
SATIN_LOBE_PEAK = 1.8
SATIN_TIP_MIN = 0.55  # star tips at least this share of the head radius
SATIN_ROSE = 0.33  # rose radius / head radius (designer S38-15)
SATIN_VALLEY_SHARE = 0.36  # star valleys / head radius, under and just outside the rose (photo S38-15)
# the loops run in under the star to the rose, edged in the star's color (the owner's check of S38-15;
# designer S38-15: edge strips 0.06 of the head radius wide), ending at 0.84 of the head radius on
# their axes (their corners reach further) so a band of pleats shows round them (designer S38-15:
# 0.92 at the corners; the owner's check asked for more band)
SATIN_LOOP_EDGE = 0.06
SATIN_LOOP_END = 0.84


def _hue(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[..., k].astype(float) for k in range(3))
    return np.degrees(np.arctan2(np.sqrt(3.0) * (g - b), 2.0 * r - g - b)) % 360.0


def _satin_star_on_pleats(p, labels: np.ndarray, parts: list) -> list | None:
    from .vision.layer_classify import KEEP_PARAM
    from .vision.parts_common import polar_sample
    head = next((q for q in parts if q.role == "head"), None)
    pts = [q for q in parts if q.role == "points"]
    if head is None or len(pts) != 1 or pts[0].params.get("kind") == "star" or getattr(p, "rgb", None) is None:
        return None
    g = head.color_group
    hue = _hue(p.rgb)
    inside = p.mask & (labels >= 0)
    hv = hue[inside & (labels == g)]
    hv = hv[(hv < 90.0)]
    if hv.size < 100:
        return None
    hist, edges = np.histogram(hv, bins=90, range=(0.0, 90.0))
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    mids = (edges[:-1] + edges[1:]) / 2
    m0 = np.cumsum(hist * mids) / np.maximum(w0, 1)
    m1 = ((hist * mids).sum() - np.cumsum(hist * mids)) / np.maximum(w1, 1)
    k = int(np.argmax(w0 * w1 * (m0 - m1) ** 2))
    split, hg = float(edges[k + 1]), float(m0[k])
    if m1[k] - m0[k] < SATIN_SPLIT_GAP:
        return None
    others = [(float(np.median(hue[inside & (labels == k)])), k) for k in {int(v) for v in np.unique(labels[inside])}
              if k not in (g, pts[0].color_group)]
    light = [(hk, k) for hk, k in others if hk - hg >= SATIN_HUE_GAP]
    if not light:
        return None
    hl, lg = min(light)
    dark = ((labels == g) & (hue < split)).astype(np.int32)
    fr = Frame.of(p)
    hr = float(head.params["r"])
    cx, cy, r = p.axis_x + (float(head.params["cx"]) - 300.0) * fr.scale, fr.py(float(head.params["cy"])), hr * fr.scale
    radii = np.arange(0.0, r, 1.0)
    m = polar_sample(dark, cx, cy, radii, 720, fill=0).astype(float)
    sm = np.apply_along_axis(lambda c: np.convolve(c, np.ones(9) / 9, mode="same"), 0, m) > 0.5
    a0 = int(SATIN_ANCHOR * r)
    ends = np.zeros(720)
    for j in range(720):
        col = sm[:, j]
        if col[a0]:
            e = a0
            while e + 1 < len(col) and col[e + 1]:
                e += 1
            ends[j] = radii[e]
    th = np.radians(np.arange(720) / 2.0)
    win = np.abs(np.degrees(th) - 90.0) > 20.0
    x = ends[win] - ends[win].mean()
    spec = {n: np.sum(x * np.exp(-1j * n * th[win])) for n in FLOWER_COUNTS}
    mags = sorted((abs(v), n) for n, v in spec.items())
    if mags[-1][0] < SATIN_LOBE_PEAK * mags[-2][0]:
        return None
    n = mags[-1][1]
    rot = (float(np.degrees(np.angle(spec[n]))) / n) % (360.0 / n)
    tips = [float(np.median([ends[int(round((rot + 360.0 * k / n + d) * 2)) % 720] for d in (-1.5, -1, -0.5, 0, 0.5, 1, 1.5)]))
            for k in range(n)]
    tip = float(np.median(tips)) / fr.scale
    if tip < SATIN_TIP_MIN * hr:
        return None
    head.color_group = lg
    out = [q for q in parts if not (q.role in ("ring", "unknown"))]
    c = next((q for q in out if q.role == "center"), None)
    if c is None:
        c = Part("center", "center", {"cx": head.params["cx"], "cy": head.params["cy"]}, lg, 50, 0.5)
        out.append(c)
    c.color_group, c.params["r"] = lg, round(SATIN_ROSE * hr, 3)
    out.append(Part("points-star", "points", {"cx": head.params["cx"], "cy": head.params["cy"], "kind": "star", "count": n,
                                              "r_inner": round(SATIN_VALLEY_SHARE * hr, 3), "r_outer": round(tip, 3),
                                              "rotation_deg": round(rot, 3), "width_ratio": 0.5, "leaf": 1,
                                              "tips": 1, KEEP_PARAM: 1}, g, pts[0].z + 2, 0.5))
    bow = pts[0]
    if bow.params.get("kind") == "bow":
        ri_old, ro_old = float(bow.params["r_inner"]), float(bow.params["r_outer"])
        ro = min(ro_old, SATIN_LOOP_END * hr)
        bow.params["width_ratio"] = round(float(bow.params["width_ratio"]) * (ro_old - ri_old) / (ro - ri_old), 4)
        bow.params["r_outer"] = round(ro, 3)
        ri = SATIN_ROSE * hr
        if ri < ri_old:  # same width, run in under the star
            bow.params["width_ratio"] = round(float(bow.params["width_ratio"]) * (ro - ri_old) / (ro - ri), 4)
            bow.params["r_inner"] = round(ri, 3)
        out.append(Part(f"edge-{bow.id}", "points", dict(bow.params, edge_width=round(SATIN_LOOP_EDGE * hr, 3)),
                        g, bow.z + 1, 0.5))
        out[-1].params[KEEP_PARAM] = 1
    bow.params[KEEP_PARAM] = 1
    return out


# R60: a star under a shorter star of its count, half a step off, both of colors other than the
# head's (s39-15: the white satin star under the gold glitter star; its tips read at 0.79 of the
# head, the photo's white runs reach 0.92, designer 0.91). Its tips are the median end of its
# color's run within UNDER_TIP_SEARCH degrees of its tip axes (the photo's tips sit a few degrees off a
# regular star), from just outside the top star's valleys, when longer. Both stars are hexagrams
# for six points (designer s39-15: two triangles, valleys at 1 / sqrt(3) of the tips; the owner's
# check), and are read whole from the photo (no library head).
UNDER_TIP_STEP_DEG = 8.0  # the two stars half a step apart within this
UNDER_TIP_MAX = 0.97  # at most this share of the head radius
UNDER_TIP_SEARCH = 6.0
UNDER_VALLEY = 0.577  # valleys / tips: a hexagram's 1 / sqrt(3)
# on them a smaller star of each, in order: the under star's color on its axes, then the top star's
# (designer s39-15: tips 0.73 and 0.58 against 0.91 and 0.75; the owner's check)
UNDER_INNER = 0.80  # the under star's inner copy / its tips
TOP_INNER = 0.77  # the top star's inner copy / its tips


def _under_star_tips(parts: list, p, labels: np.ndarray) -> bool:
    head = next((q for q in parts if q.role == "head"), None)
    stars = [q for q in parts if q.role == "points" and q.params.get("kind") == "star"]
    if head is None or len(stars) != 2:
        return False
    under, top = sorted(stars, key=lambda q: q.z)
    n = int(under.params["count"])
    if n != int(top.params["count"]) or head.color_group in (under.color_group, top.color_group):
        return False
    step = 360.0 / n
    if abs(((float(under.params["rotation_deg"]) - float(top.params["rotation_deg"])) % step) - step / 2) > UNDER_TIP_STEP_DEG:
        return False
    if float(top.params["r_outer"]) > float(under.params["r_outer"]):
        return False
    fr = Frame.of(p)
    hr = float(head.params["r"])
    cx, cy = p.axis_x + (float(under.params["cx"]) - 300.0) * fr.scale, fr.py(float(under.params["cy"]))
    h, w = labels.shape
    rr = np.arange(0.0, UNDER_TIP_MAX * hr * fr.scale, 1.0)
    start = int(float(top.params["r_inner"]) * fr.scale)
    ends = []
    for k in range(n):
        best = None
        for d in np.arange(-UNDER_TIP_SEARCH, UNDER_TIP_SEARCH + 0.1, 1.0):
            a = np.radians(float(under.params["rotation_deg"]) + step * k + d)
            v = labels[np.clip(np.rint(cy + rr * np.sin(a)).astype(int), 0, h - 1),
                       np.clip(np.rint(cx + rr * np.cos(a)).astype(int), 0, w - 1)] == under.color_group
            on = np.flatnonzero(v[start:])
            if on.size == 0:
                continue
            e = start + on[0]
            while e + 1 < v.size and (v[e + 1] or (e + 3 < v.size and v[e + 3])):  # bridge 1-2 px of grain
                e += 1
            best = max(best or 0.0, rr[e] / fr.scale)
        if best is not None:
            ends.append(best)
    if len(ends) < n // 2:
        return False
    tip = max(float(np.median(ends)), float(under.params["r_outer"]))
    under.params["r_outer"] = round(tip, 3)
    from .vision.layer_classify import KEEP_PARAM
    for q in (under, top):  # read whole from the photo: no library head, the classifier keeps them
        q.params.update({"r_inner": round(UNDER_VALLEY * float(q.params["r_outer"]), 3), "tips": 1, KEEP_PARAM: 1})
    for k, (q, share) in enumerate(((under, UNDER_INNER), (top, TOP_INNER))):
        ro = share * float(q.params["r_outer"])
        parts.append(Part(f"{q.id}-inner", "points", dict(q.params, r_outer=round(ro, 3), r_inner=round(UNDER_VALLEY * ro, 3)),
                          q.color_group, top.z + 1 + k, 0.5))
    return True


# R61: glitter points between satin points of a third color, no head disc read (S40-15: black
# glitter points between light orange satin points on orange pleats; the pleats read as an outer
# ring, the satin points mostly lost into the pleats' group, the white badge into the black). The
# outer ring of the pleats' color is the head disc; the glitter points are a star lying on the satin
# star (the owner's check of S40-15), which is half a step off, its color the group whose hue is nearest the hue between the glitter
# points; the pleated ring inside lies on both; the badge takes the satin color. Designer S40-15:
# satin tips / glitter tips 0.93 / 0.78, center 0.34 of the head, six-point stars as hexagrams.
THIRD_SATIN_TIP = 0.93 / 0.78
THIRD_GLITTER_TIP = 0.78  # glitter tips at least this share of the head (the read stops short at a thin tip, 0.70)
THIRD_CENTER = 0.34
THIRD_HUE_GAP = 6.0  # the satin hue at least this far from the pleats' (degrees)
THIRD_RIM = 0.9  # the outer ring reaches at least this share of the head circle


def _satin_star_third_color(p, labels: np.ndarray, parts: list) -> list | None:
    from .vision.layer_classify import KEEP_PARAM
    if any(q.role in ("head", "template") for q in parts) or getattr(p, "rgb", None) is None:
        return None
    pts = [q for q in parts if q.role == "points"]
    rings = sorted((q for q in parts if q.role == "ring"), key=lambda q: float(q.params["r_outer"]))
    c = next((q for q in parts if q.role == "center"), None)
    if len(pts) != 1 or len(rings) != 2 or c is None:
        return None
    glit, (inner, outer) = pts[0], rings
    fr = Frame.of(p)
    H = float(outer.params["r_outer"])
    if inner.color_group != outer.color_group or glit.color_group == outer.color_group             or H < THIRD_RIM * float(p.head[2]) / fr.scale:
        return None
    n = int(glit.params["count"])
    step = 360.0 / n
    hue = _hue(p.rgb)
    g0 = outer.color_group
    h0 = float(np.median(hue[(labels == g0) & p.mask]))
    cx, cy = p.axis_x + (float(glit.params["cx"]) - 300.0) * fr.scale, fr.py(float(glit.params["cy"]))
    hh, ww = labels.shape
    rr = np.arange(float(inner.params["r_outer"]), 0.9 * H, 1.0) * fr.scale
    vals = []
    for k in range(n):
        a = np.radians(float(glit.params["rotation_deg"]) + step * (k + 0.5))
        ys = np.clip(np.rint(cy + rr * np.sin(a)).astype(int), 0, hh - 1)
        xs = np.clip(np.rint(cx + rr * np.cos(a)).astype(int), 0, ww - 1)
        vals.append(hue[ys, xs][p.mask[ys, xs]])
    v = np.concatenate(vals) if vals else np.array([])
    if v.size == 0:
        return None
    hs = float(np.median(v))
    if abs(hs - h0) < THIRD_HUE_GAP:
        return None
    cands = [(abs(float(np.median(hue[(labels == k) & p.mask])) - hs), k)
             for k in {int(x) for x in np.unique(labels[p.mask])} if k >= 0 and k not in (g0, glit.color_group)]
    if not cands:
        return None
    sg = min(cands)[1]
    geo = {"cx": glit.params["cx"], "cy": glit.params["cy"]}
    ro = max(float(glit.params["r_outer"]), THIRD_GLITTER_TIP * H)
    glit.params.update(kind="star", r_inner=round(UNDER_VALLEY * ro, 3), r_outer=round(ro, 3), tips=1, **{KEEP_PARAM: 1})
    rs = THIRD_SATIN_TIP * ro
    out = [q for q in parts if q.role not in ("unknown",) and q is not outer]
    out.append(Part("head", "head", dict(geo, r=round(H, 3)), g0, 10, 0.6))
    out.append(Part("points-satin", "points", dict(geo, kind="star", count=n, r_inner=round(UNDER_VALLEY * rs, 3),
                                                   r_outer=round(rs, 3),
                                                   rotation_deg=round((float(glit.params["rotation_deg"]) + step / 2) % step, 3),
                                                   width_ratio=0.5, tips=1, **{KEEP_PARAM: 1}), sg, glit.z, 0.5))
    glit.z += 1  # the glitter star on the satin star
    inner.z = max(inner.z, glit.z + 1)
    c.color_group, c.params["r"] = sg, round(max(float(c.params["r"]), THIRD_CENTER * H), 3)
    inner.params["r_inner"] = c.params["r"]
    return out


# R62: three stars of one count: two on one axis, one of the head's color and one of another, and a
# third color half a step off (S41-15: a white star, pink glitter points between its points, a wine
# star of the pleats' color round the badge; the glitter read under the white, the wine star's body
# as a pleated ring). Designer S41-15, bottom to top: the other color's star (tips 0.91 of the
# head), the half-step star (0.80), the head color's star (0.67), all hexagrams, no ring, the
# center 0.34 (the owner's check).
NESTED_TIPS = (0.91, 0.80, 0.67)
NESTED_CENTER = 0.34
NESTED_AXIS_DEG = 8.0
NESTED_SNAP_DEG = 6.0  # the stars stand upright: angles within this of the half-step grid snap to it


def _three_nested_stars(parts: list) -> bool:
    from .vision.layer_classify import KEEP_PARAM
    head = next((q for q in parts if q.role == "head"), None)
    pts = [q for q in parts if q.role == "points"]
    if head is None or len(pts) != 3 or any(q.params.get("kind") not in ("star", "diamond", "petal") for q in pts):
        return False
    n = int(pts[0].params["count"])
    if any(int(q.params["count"]) != n for q in pts) or len({q.color_group for q in pts}) != 3:
        return False
    step = 360.0 / n

    def off(a, b):
        d = (float(a.params["rotation_deg"]) - float(b.params["rotation_deg"])) % step
        return min(d, step - d)
    inner = next((q for q in pts if q.color_group == head.color_group), None)
    if inner is None:
        return False
    rest = [q for q in pts if q is not inner]
    same = [q for q in rest if off(q, inner) <= NESTED_AXIS_DEG]
    half = [q for q in rest if abs(off(q, inner) - step / 2) <= NESTED_AXIS_DEG]
    if len(same) != 1 or len(half) != 1:
        return False
    outer, mid = same[0], half[0]
    hr = float(head.params["r"])
    z0 = min(q.z for q in pts)
    half_step = step / 2
    for q in pts:  # the photo's slight tilt (S41-15: 25.3 for 30): upright as the designer draws them
        rot = float(q.params["rotation_deg"])
        snap = round(rot / half_step) * half_step
        if abs(rot - snap) <= NESTED_SNAP_DEG:
            q.params["rotation_deg"] = round(snap % step, 3)
    for k, (q, tip) in enumerate(zip((outer, mid, inner), NESTED_TIPS)):
        ro = tip * hr
        q.params.update({"kind": "star", "r_outer": round(ro, 3), "r_inner": round(UNDER_VALLEY * ro, 3),
                         "tips": 1, KEEP_PARAM: 1})
        q.params.pop("edge_width", None)
        q.z = z0 + k
    rim = NESTED_TIPS[2] * hr
    parts[:] = [q for q in parts if not (q.role == "ring" and q.color_group == head.color_group
                                         and float(q.params["r_outer"]) <= rim)]
    c = next((q for q in parts if q.role == "center"), None)
    if c is not None:
        c.params["r"] = round(max(float(c.params["r"]), NESTED_CENTER * hr), 3)
    return True


# R63: only a plain disc read (no points, no rings), two color groups, and the rim's darker pixels
# (lightness below the midpoint of the two groups') form a clear count of lobes (S4-15: dark violet
# loops showing at the rim under wide lavender loops; the two purples close, the satin highlights
# put most of the dark into the light group). Dark loops of the darker group at the lobes, light
# loops of the head's group half a step off on them, a rose of the darker group. Designer S4-15
# (head radii): dark loops to 1.01, 0.74 wide (short wide blocks at the rim, their corners past the
# light loops' ends; the owner's check); light loops to 0.95, 0.66 wide (nearly touching);
# rose 0.39; no disc. The light color is one piece from the rose to 0.66 (a ring under the rose, the
# light loops starting under it), so the dark loops show only past it (the owner's check of S4-15).
TONE_BAND = (0.85, 0.97)  # the rim band read, head radii
TONE_PEAK = 2.5
TONE_DARK_SHARE = 0.15  # the band's dark share at least
TONE_ROSE = 0.39
TONE_DARK = (1.01, 0.74)
TONE_LIGHT = (0.95, 0.66)
TONE_LIGHT_RING = 0.66  # the light ring's outer radius, head radii (designer S4-15: valleys 0.66)
TONE_UNDER = 0.8  # the light ring and loops start at this share of the rose radius (under it)


def _two_tone_loops(p, labels: np.ndarray, parts: list) -> list | None:
    from .vision.layer_classify import KEEP_PARAM
    from .vision.parts_common import polar_sample
    head = next((q for q in parts if q.role == "head"), None)
    if head is None or getattr(p, "rgb", None) is None or any(q.role in ("points", "ring", "template") for q in parts):
        return None
    gs = [int(v) for v in np.unique(labels[p.mask]) if v >= 0]
    if len(gs) != 2:
        return None
    light = head.color_group
    dark_g = next(g for g in gs if g != light)
    lum = p.rgb.astype(float).mean(-1)
    l_light, l_dark = float(np.median(lum[p.mask & (labels == light)])), float(np.median(lum[p.mask & (labels == dark_g)]))
    if l_dark >= l_light:
        return None
    dark = ((lum < (l_light + l_dark) / 2) & p.mask).astype(np.int32)
    fr = Frame.of(p)
    hr = float(head.params["r"])
    cx, cy = p.axis_x + (float(head.params["cx"]) - 300.0) * fr.scale, fr.py(float(head.params["cy"]))
    radii = np.arange(TONE_BAND[0], TONE_BAND[1], 0.01) * hr * fr.scale
    m = polar_sample(dark, cx, cy, radii, 360, fill=0).mean(0)
    th = np.radians(np.arange(360.0))
    win = np.abs(np.degrees(th) - 90.0) > 40.0
    if m[win].mean() < TONE_DARK_SHARE:
        return None
    x = m[win] - m[win].mean()
    spec = {n: np.sum(x * np.exp(-1j * n * th[win])) for n in FLOWER_COUNTS}
    mags = sorted((abs(v), n) for n, v in spec.items())
    if mags[-1][0] <= 0 or mags[-1][0] < TONE_PEAK * mags[-2][0]:
        return None
    n = mags[-1][1]
    step = 360.0 / n
    rot = (float(np.degrees(np.angle(spec[n]))) / n) % step
    geo = {"cx": head.params["cx"], "cy": head.params["cy"]}
    ri = TONE_ROSE * hr
    out = [q for q in parts if q.role not in ("head", "unknown", "center")]
    for pid, g, (ro, wd), r_ in (("points-dark", dark_g, TONE_DARK, rot), ("points-light", light, TONE_LIGHT, (rot + step / 2) % step)):
        out.append(Part(pid, "points", dict(geo, kind="bow", count=n, r_inner=round(ri, 3), r_outer=round(ro * hr, 3),
                                            rotation_deg=round(r_, 3), width_ratio=round(wd / (ro - TONE_ROSE), 4),
                                            tips=1, **{KEEP_PARAM: 1}), g, 20 if pid == "points-dark" else 21, 0.5))
    light_loops = out[-1]
    light_loops.params["r_inner"] = round(TONE_UNDER * ri, 3)
    light_loops.params["width_ratio"] = round(TONE_LIGHT[1] / (TONE_LIGHT[0] - TONE_UNDER * TONE_ROSE), 4)
    out.append(Part("ring-light", "ring", dict(geo, r_inner=round(TONE_UNDER * ri, 3), r_outer=round(TONE_LIGHT_RING * hr, 3)),
                    light, 21, 0.5))
    light_loops.z = 22
    out.append(Part("center", "center", dict(geo, r=round(ri, 3)), dark_g, 50, 0.5))
    return out


# R64: short points of a second color standing just outside a ring of the head's color, round a
# center of that second color (S42-15: a white star's tips past light blue satin petals on light
# blue pleats round a white rose; the tips read as short loops, the petal flower as a ring). The
# points are a star (a hexagram for six), petals of the head's color lie on it half a step off,
# the rose on top, no ring. Designer S42-15 (head radii): star tips 0.85, petal tips 0.71, rose
# 0.35; the petals as R56's (FLOWER_PROFILE), tucked under the rose; upright as R62.
PETAL_STAR_TIP = 0.85
PETAL_TIP = 0.71
PETAL_ROSE = 0.35
PETAL_RING_GAP = 0.1  # the ring's outer edge within this share of the points' inner radius
PETAL_SPAN = 0.635  # petal width / tip radius (designer S42-15)
PETAL_START = 0.3  # petals start at this share of their tip radius, deep under the rose, so they are at
# full width where they leave it and neighbours meet with no gap (the owner's check of S42-15)


def _star_under_petals(parts: list) -> bool:
    from .vision.layer_classify import KEEP_PARAM
    head = next((q for q in parts if q.role == "head"), None)
    pts = [q for q in parts if q.role == "points"]
    c = next((q for q in parts if q.role == "center"), None)
    if head is None or c is None or any(q.params.get(KEEP_PARAM) for q in pts):
        return False  # layers an earlier rule built whole (R58 on S37-15, S8-15) stand
    g = head.color_group
    others = [q for q in pts if q.color_group != g]
    if len(others) != 1:
        return False
    star = others[0]
    w = star.color_group
    if c.color_group != w or star.params.get("kind") not in ("bow", "star", "diamond"):
        return False
    n = int(star.params["count"])
    step = 360.0 / n
    ri = float(star.params["r_inner"])
    ring = next((q for q in parts if q.role == "ring" and q.color_group == g
                 and abs(float(q.params["r_outer"]) - ri) <= PETAL_RING_GAP * ri), None)
    # the app's reading (the perception models on): the petal flower as a star of the head's color
    # half a step off instead of a ring
    flower = [q for q in pts if q.color_group == g and int(q.params["count"]) == n
              and abs(((float(q.params["rotation_deg"]) - float(star.params["rotation_deg"])) % step) - step / 2) <= NESTED_AXIS_DEG]
    if (ring is None) == (len(flower) != 1) or len(pts) != 1 + len(flower):
        return False
    hr = float(head.params["r"])
    rot = float(star.params["rotation_deg"])
    snap = round(rot / (step / 2)) * (step / 2)
    if abs(rot - snap) <= NESTED_SNAP_DEG:
        rot = snap % step
    ro = max(float(star.params["r_outer"]), PETAL_STAR_TIP * hr)
    star.params.update({"kind": "star", "r_outer": round(ro, 3), "r_inner": round(UNDER_VALLEY * ro, 3),
                        "rotation_deg": round(rot, 3), "tips": 1, KEEP_PARAM: 1})
    star.params.pop("edge_width", None)
    rr = max(float(c.params["r"]), PETAL_ROSE * hr)
    c.params["r"] = round(rr, 3)
    pr = PETAL_TIP * hr
    pi = PETAL_START * pr
    for q in [ring] + flower:
        if q is not None:
            parts.remove(q)
    parts.append(Part("points-petals", "points", {"cx": star.params["cx"], "cy": star.params["cy"], "kind": "petal",
                                                  "count": n, "r_inner": round(pi, 3), "r_outer": round(pr, 3),
                                                  "rotation_deg": round((rot + step / 2) % step, 3),
                                                  "width_ratio": round(PETAL_SPAN / (1.0 - PETAL_START), 4),
                                                  "flower": 2, "tips": 1, KEEP_PARAM: 1},
                      g, star.z + 1, 0.5))
    return True


def _tuck_rings(parts: list) -> None:
    head = next((q for q in parts if q.role == "head"), None)
    rings = sorted((q for q in parts if q.role == "ring"), key=lambda q: float(q.params["r_outer"]))
    if head is None or not rings:
        return
    rim, grow = float(head.params["r"]), HUE_TUCK * float(head.params["r"])
    for i, q in enumerate(rings):
        q.params["r_outer"] = round(min(float(q.params["r_outer"]) + grow, 0.95 * rim), 3)
        if i:
            q.params["r_inner"] = rings[i - 1].params["r_inner"]
        q.z = 30 + len(rings) - i  # inner rings on top


# R42: two plain color groups of one hue (within SHADE_HUE_DEG; lightness and chroma within
# SHADE_DL / SHADE_DC) are one ribbon in light and shade when merging them makes the tail
# stripes mirror-symmetric (S15-15: the light and dark red read red | blue | yellow | blue |
# yellow). The tails are always symmetric in the designer files with a single block; A14-17's
# two oranges (as close) already are, so they stay apart. The larger group keeps its color.
SHADE_HUE_DEG, SHADE_DL, SHADE_DC, SHADE_MIN_CHROMA = 12.0, 12.0, 12.0, 25.0
SHADE_INFO = "Two shades of one ribbon were read as one color (light and shadow) - check the colors."


def _short_star_on_top(parts: list, p=None, labels=None) -> None:
    """R42: of two interleaved stars of one count (half a step apart), the shorter lies on top,
    as the designer stacks them (S15-15: blue star over the red; the photo vote reads the red
    over it once the red's two shades are one group)."""
    stars = [q for q in parts if q.role == "points" and q.params.get("kind") == "star"]
    for a in stars:
        for b in stars:
            if a is b or int(a.params["count"]) != int(b.params["count"]):
                continue
            step = 360.0 / int(a.params["count"])
            if abs(((float(a.params["rotation_deg"]) - float(b.params["rotation_deg"])) % step) - step / 2) > 8.0:
                continue
            if float(a.params["r_outer"]) == float(b.params["r_outer"]):
                continue  # one length (S18-15): the photo's order stands
            short, long_ = (a, b) if float(a.params["r_outer"]) < float(b.params["r_outer"]) else (b, a)
            if short.z <= long_.z:
                short.z, long_.z = long_.z, short.z
                if short.z == long_.z:
                    short.z += 1
            if p is not None and labels is not None:
                _body_ring(parts, long_, short, p, labels)
            return


def _long_star_under(parts: list, long_) -> None:
    """R42: the long star whose body shows as the pleated ring lies under every other points
    layer (S33-15, the owner's check: the violet star under the glitter star and its edge), and
    keeps its kind and count in the app: the classifier sees only its tips past the ring and
    reads them as kites (S33-15, S15-15B)."""
    from .vision.layer_classify import KEEP_PARAM
    others = [q.z for q in parts if q.role == "points" and q is not long_]
    if others and long_.z >= min(others):
        long_.z = min(others) - 1
    long_.params[KEEP_PARAM] = 1


def _body_ring(parts: list, long_, short, p, labels) -> None:
    """R42: the long star's body showing as a full band inside the short star (S15-15: the red
    pleated ring between the blue ring and the blue star) is a pinked ring on top of the stars."""
    fr = Frame.of(p)
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    vis = np.abs(np.degrees(th) - 90.0) > 35.0
    h, w = labels.shape
    inner = max([float(q.params["r_outer"]) for q in parts if q.role == "ring"]
                + [float(q.params["r"]) for q in parts if q.role == "center"], default=0.0)
    full = []
    for ru in np.arange(inner, float(short.params["r_inner"]), 1.0):
        rp = ru * fr.scale
        xs = np.clip(np.rint(p.head[0] + rp * np.cos(th)).astype(int), 0, w - 1)
        ys = np.clip(np.rint(p.head[1] + rp * np.sin(th)).astype(int), 0, h - 1)
        if float((labels[ys, xs][vis] == long_.color_group).mean()) >= 0.8:
            full.append(ru)
    if len(full) < 5:
        return
    from rosette.model import Part
    # the pleated ring runs out under the short star's valleys (no head color between them)
    parts.append(Part("ring-body", "ring", {"cx": long_.params["cx"], "cy": long_.params["cy"],
                                            "r_inner": round(inner, 3),
                                            "r_outer": round(float(short.params["r_inner"]) + 2.0, 3),
                                            "pinked": 1}, long_.color_group, 29, 0.5))
    _long_star_under(parts, long_)
    c = next((q for q in parts if q.role == "center"), None)
    if c is not None:  # rings start at the center: no band of other color, no extra edge lines
        rc = float(c.params["r"])
        for q in parts:
            if q.role == "ring" and q.id != "ring-body" and 0 < float(q.params["r_inner"]) - rc <= 12.0:
                q.params["r_inner"] = round(rc, 3)
            if q.id == "ring-body":  # under the ring inside it: one pinked edge where they meet
                q.params["r_inner"] = round(rc, 3)
        parts[:] = [q for q in parts if not (q.role == "unknown" and float(q.params.get("r", 0)) <= rc)]


def shade_pair(parts: list, groups: list) -> tuple[int, int] | None:
    """R42: (group kept, group merged into it) or None."""
    import itertools
    import cv2
    st = sorted((q for q in parts if q.role == "tail_stripe"),
                key=lambda q: (int(q.params["tail_index"]), int(q.params["stripe_index"])))
    def runs(x: list) -> list:  # consecutive repeats collapsed (slivers between stripes)
        return [v for i, v in enumerate(x) if i == 0 or v != x[i - 1]]
    seq = runs([q.color_group for q in st])
    if len(seq) < 3 or seq == seq[::-1] or len({int(q.params["tail_index"]) for q in st}) != 1:
        return None

    def lab(g):
        return cv2.cvtColor(np.array([[g.rgb]], np.float32) / 255.0, cv2.COLOR_RGB2Lab)[0, 0]
    for a, b in itertools.combinations([g for g in groups if not g.pattern], 2):
        la, lb = lab(a), lab(b)
        ca, cb = float(np.hypot(*la[1:])), float(np.hypot(*lb[1:]))
        if min(ca, cb) < SHADE_MIN_CHROMA:
            continue
        dh = abs((np.degrees(np.arctan2(la[2], la[1])) - np.degrees(np.arctan2(lb[2], lb[1])) + 180) % 360 - 180)
        if dh > SHADE_HUE_DEG or abs(la[0] - lb[0]) > SHADE_DL or abs(ca - cb) > SHADE_DC:
            continue
        merged = runs([a.id if g == b.id else g for g in seq])
        if merged == merged[::-1]:
            return (a.id, b.id) if a.area >= b.area else (b.id, a.id)
    return None


def detect(data: bytes, classify: bool = True, exclude_key: str | None = None) -> Detection:
    """prepare -> pixel_groups -> detect_parts -> tidy -> RosetteModel -> assign -> the
    learned layer classifier (ML-1) -> the piece reader (ML-4, reader_step). classify=False:
    the rules alone, neither classifier nor reader (Detection.reader None), as the
    classifier's training data needs; exclude_key: the classifier trained without that
    design, for leave-one-out evaluation only."""
    p = prepare(data)
    merged_print: list = []
    labels, groups = pixel_groups(p, merged=merged_print)
    pieces, seen = perception(p, labels)
    try:
        found = detect_parts(p, labels, groups, pieces)
    except Exception as exc:  # noqa: BLE001 - a fault in the piece path: the rules read the head alone
        if pieces is None:
            raise
        seen = dict(seen, status="failed", message=f"SAM 2 pieces failed in the detector - {type(exc).__name__}: {exc}")
        pieces = None  # rejected: the reader never reads them either
        found = detect_parts(p, labels, groups)
    pair = shade_pair(found, groups)  # R42: one ribbon in light and shade read as two colors
    if pair is not None:
        keep, drop = pair
        labels = labels.copy()
        labels[labels == drop] = keep
        groups = [g for g in groups if g.id != drop]
        try:
            found = detect_parts(p, labels, groups, pieces)
        except Exception:  # noqa: BLE001
            found = detect_parts(p, labels, groups)
    tidy = merge_tail_slivers(drop_two_point_layers(drop_specks(
        RosetteModel(parts=found), head_r=Frame.of(p).d(p.head[2]))))
    if pair is not None:
        tidy.info.append(SHADE_INFO)
    _short_star_on_top(tidy.parts, p, labels)
    stack = stacked_stars(p, labels, tidy.parts)  # R43: a head of stacked stars read along rays
    if stack is not None:
        tidy.parts = stack
    edged = edged_star_between_bows(p, labels, tidy.parts)  # R46: a star edged in a third color
    if edged is not None:
        tidy.parts = edged
    _star_under_satin_star(tidy.parts)  # R58: glitter points between satin points of the head's color
    satin = _satin_star_on_pleats(p, labels, tidy.parts)  # R59: a darker satin star on pleats of one group
    if satin is not None:
        tidy.parts = satin
    _under_star_tips(tidy.parts, p, labels)  # R60: a star under a shorter star, its tips from its color runs
    third = _satin_star_third_color(p, labels, tidy.parts)  # R61: glitter points between satin points of a third color
    if third is not None:
        tidy.parts = third
    _three_nested_stars(tidy.parts)  # R62: three stars of one count, nested as the designer stacks them
    tone = _two_tone_loops(p, labels, tidy.parts)  # R63: dark loops at the rim under light loops, two close colors
    if tone is not None:
        tidy.parts = tone
    _star_under_petals(tidy.parts)  # R64: star tips round a ring of the head's color are a star under petals
    flower = _petal_flower(p, labels, tidy.parts)  # R56: a one-color lobed head round a rose
    if flower is not None:
        tidy.parts = flower
    square = star_on_square(p, labels, tidy.parts)  # R48: a star on four loops, a square over its base
    if square is not None:
        tidy.parts = square
    parts = tidy.parts
    _fill_badge(parts)  # R49: a speck of center inside a ring of its color is the badge
    _pleats_round_badge(parts, p, labels)  # R54: pleats round a badge read as light and shade
    badge = badge_in_pleats(p, parts)  # R31: a badge inside pleats of its own color
    if badge is not None:
        parts = badge
    one = rose_center(p, parts)  # R29: the rose of a one-color rosette, same color
    if one is not None:
        parts = one
        tidy.info.append(ONE_COLOR_INFO)
    tidy.warnings.extend(pleat_warnings(parts, p, labels))  # ML-1: rings near the pleat bar
    used = {q.color_group for q in parts}
    model = RosetteModel(parts=parts, groups=[g for g in groups if g.id in used],
                         rose=any(q.role == "center" for q in parts), warnings=tidy.warnings,
                         info=list(tidy.info) + alt_ring_info(parts))  # A16: a one-color two-tone edge
    out = assign(model)
    # R32: shaded pleat rings read from hue, after the Colors are numbered on the labels' reading
    # (the rebuilt head reuses the same groups, so their Colors stand)
    # assign keeps the parts (same order and ids) and only relabels merged groups
    assert [q.id for q in model.parts] == [q.id for q in out.parts]
    merged = {a.color_group: b.color_group for a, b in zip(model.parts, out.parts)
              if a.color_group != b.color_group}
    edged = dark_loop_edges(p, labels, out.parts, out.groups)  # R34: dark loop edges (on the photo's loops)
    if edged is not None:
        out.parts = edged
    if _rose_from_small_star(out.parts):  # R53: a small blunt star with no center is the rose
        out.rose = True
    if _rose_over_bows(out.parts, out.groups):  # R44: bows and a star of one hue under a rose
        out.rose = True
    _loops_on_star_flower(out.parts)  # R51: loops on a star around a bare center
    _diamonds_behind_loops(out.parts)  # R50: points between edged loops are a star behind them
    _star_as_loops(out.parts)  # R47: a blunt star between loops of its count is loops too
    _taper_edged_bows(out.parts)  # R45: loops edged by a wider loop narrow toward the center
    _petals_between_loops(out.parts)  # R36: petals on top of loops (final stacking known here)
    _points_under_edged_star(out.parts)  # R55: points half a step off an edged star go under it
    _badge_under_loops(out.parts, p, labels)  # R39: a badge inside loops, read as print
    if _loops_on_rim(out.parts, p, labels, out.groups):  # R41: loops on a dark rim, star between
        out.rose = True
    if _twin_points(out.parts, p, labels):  # R40: two sets of points around the center
        out.rose = True
    if close_seams(out.parts):
        out.info.append(SEAM_INFO)
    hue = shaded_rings(p, out.parts, out.groups)
    if hue is not None:
        badged = badge_in_pleats(p, hue, big=HUE_BADGE_BIG, outer=True)  # the badge in the light pleats
        if badged is not None:
            # its print is no ribbon color: it takes the next ring's out (A3-15's designer badge is
            # the mid ring's Color 3); flagged with the rebuilt rings
            mid = sorted((q for q in badged if q.role == "ring" and q.id != "ring-badge"),
                         key=lambda q: float(q.params["r_inner"]))
            c = next(q for q in badged if q.role == "center")
            if mid:
                c.color_group, c.confidence = mid[0].color_group, min(c.confidence, 0.5)
        out.parts = badged or hue
        _tuck_rings(out.parts)
        out.info.append(HUE_INFO)
    # ML-1: the learned classifier's count, and its kind where it is sure (layer_classify)
    rules_model = copy.deepcopy(out)
    read_status = None
    if classify:
        out.warnings.extend(classify_head_layers(p, labels, out, merged, exclude_key=exclude_key))
        # ML-4: the reader's candidates against this head (candidate 0); off: `out` itself, untouched
        out, read_status = reader_step(p, labels, pieces, out, merged=merged)
    return Detection(p, labels, groups, out, merged, merged_print, rules_model, seen, read_status)


def analyze_with_fit(data: bytes, template: bool = True,
                     exclude_key: str | None = None) -> tuple[RosetteModel, HeadFit | None]:
    """analyze, plus the head fit measurement (None without head parts). template: try
    a library head (with_template); exclude_key: leave-one-out evaluation only."""
    model, fit, _ = analyze_with_context(data, template, exclude_key)
    return model, fit


def analyze_with_context(data: bytes, template: bool = True,
                         exclude_key: str | None = None) -> tuple[RosetteModel, HeadFit | None, PhotoContext]:
    """analyze_with_fit, plus the PhotoContext add_library_head needs later."""
    d = detect(data, exclude_key=exclude_key)
    p, labels, groups, out, merged, merged_print = d.prepared, d.labels, d.groups, d.model, d.merged, d.merged_print
    ctx = PhotoContext(p, labels, groups, merged, copy.deepcopy(out), d.rules_model, d.perception, d.reader)
    used = {q.color_group for q in out.parts} | set(merged)
    lib = None
    tip_star = any(q.params.get("tips") for q in out.parts)  # R37: a star read from its tips
    if template and ONE_COLOR_INFO not in out.info and HUE_INFO not in out.info and not tip_star:  # the rules head stands
        lib = with_template(out, p.mask, labels, p.head, Frame.of(p), groups, merged, exclude_key=exclude_key)
    cache: dict[int, tuple] = {}  # one render per head model (fit and overlap share it)

    def render_of(m: RosetteModel):
        if id(m) not in cache:  # the model is kept alive in the cache, so ids stay unique
            cache[id(m)] = (m, _head_render(m))
        return cache[id(m)][1]

    out, fit = choose_head(out, lib, lambda m: head_fit(p, labels, m, merged, render_of),
                           lambda a, b: head_overlap(a, b, render_of))
    by_id = {g.id: g for g in groups}
    for gid in mixed_texture_groups(p, labels, groups):
        if gid in used:
            out.warnings.append(f"Color {hex_rgb(by_id[gid].rgb)} is partly patterned - a patterned ribbon "
                                "may have been merged into it; check the colors.")
    out.warnings.extend(circle_warnings(p))
    for rgb in merged_print:
        out.warnings.append(f"Center badge print was merged into the badge color (print {hex_rgb(rgb)}) - "
                            "check the center against the photo.")
    for gid in badge_print_unsure(p, labels, groups):
        if gid in used:
            out.warnings.append(f"Color {hex_rgb(by_id[gid].rgb)} may be print on the center badge (a logo) "
                                "rather than a ribbon color - check the colors and the center.")
    return out, fit, ctx


def render(model: RosetteModel) -> RenderResult:
    svg = write_svg(build(model))
    return RenderResult(svg=svg, issues=validate(svg))


def is_flagged(model: RosetteModel, issues: list[Issue]) -> bool:
    """True when the user is asked to check something: a low-confidence color, a part
    of any role read with low confidence (head, ring, points, center, tail stripe), an
    unrecognised (placeholder) part, any warning, or a validation issue. app/static/logic.js
    flagItems applies the same rules."""
    return (any(g.confidence < FLAG_BELOW for g in model.groups)
            or any(q.confidence < FLAG_BELOW for q in model.parts)
            or any(q.role == "unknown" for q in model.parts)
            or bool(model.warnings)
            or bool(issues))
