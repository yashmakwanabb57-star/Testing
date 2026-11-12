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

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .assign import FLAG_BELOW, FLAGGED_CONF, assign, hex_rgb
from .build import build
from .model import Part, RosetteModel
from .perception import settings as perception_settings
from .perception.pieces import DEFAULT_CACHE_DIR, segment
from .perception.sam2 import MissingModel
from .roles import PAD_FRAC, render_categories
from .tidy import drop_specks, drop_two_point_layers, merge_tail_slivers
from .validate import Issue, validate
from .vision.colors import badge_print_unsure, mixed_texture_groups, pixel_groups
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


def detect(data: bytes, classify: bool = True, exclude_key: str | None = None) -> Detection:
    """prepare -> pixel_groups -> detect_parts -> tidy -> RosetteModel -> assign -> the
    learned layer classifier (ML-1; classify=False: the rules alone, as the classifier's
    training data needs; exclude_key: the classifier trained without that design, for
    leave-one-out evaluation only)."""
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
        found = detect_parts(p, labels, groups)
    tidy = merge_tail_slivers(drop_two_point_layers(drop_specks(
        RosetteModel(parts=found), head_r=Frame.of(p).d(p.head[2]))))
    parts = tidy.parts
    tidy.warnings.extend(pleat_warnings(parts, p, labels))  # ML-1: rings near the pleat bar
    used = {q.color_group for q in parts}
    model = RosetteModel(parts=parts, groups=[g for g in groups if g.id in used],
                         rose=any(q.role == "center" for q in parts), warnings=tidy.warnings,
                         info=list(tidy.info) + alt_ring_info(parts))  # A16: a one-color two-tone edge
    out = assign(model)
    # assign keeps the parts (same order and ids) and only relabels merged groups
    assert [q.id for q in model.parts] == [q.id for q in out.parts]
    merged = {a.color_group: b.color_group for a, b in zip(model.parts, out.parts)
              if a.color_group != b.color_group}
    # ML-1: the learned classifier's count, and its kind where it is sure (layer_classify)
    rules_model = copy.deepcopy(out)
    if classify:
        out.warnings.extend(classify_head_layers(p, labels, out, merged, exclude_key=exclude_key))
    return Detection(p, labels, groups, out, merged, merged_print, rules_model, seen)


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
    ctx = PhotoContext(p, labels, groups, merged, copy.deepcopy(out), d.rules_model, d.perception)
    used = {q.color_group for q in out.parts} | set(merged)
    lib = None
    if template:
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
