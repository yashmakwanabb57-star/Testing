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

render(model): build -> write_svg -> validate. build may add warnings to the model
(unrecognised shapes drawn as placeholders).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .assign import FLAG_BELOW, FLAGGED_CONF, assign, hex_rgb
from .build import build
from .model import Part, RosetteModel
from .roles import PAD_FRAC, render_categories
from .tidy import drop_specks, merge_tail_slivers
from .validate import Issue, validate
from .vision.colors import badge_print_unsure, mixed_texture_groups, pixel_groups
from .vision.parts import detect_parts
from .vision.parts_common import Frame
from .vision.prepare import Prepared, prepare
from .vision.templates import describe, load_library, photo_grid, rank, relabel_map
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
TEMPLATE_COLORS_WARNING = ("The library head {tid} fits the head outline better, but its colors do not agree "
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
    svg = write_svg(build(RosetteModel(parts=copy.deepcopy(head), groups=groups)))
    img = render_categories(svg, _RENDER_H, _FIT_COLORS)
    pad = int(round(PAD_FRAC * _RENDER_H))
    img = img[pad:img.shape[0] - pad, pad:img.shape[1] - pad].astype(np.int32)
    ref = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in _FIT_COLORS.values()])
    d = np.abs(img[:, :, None, :] - ref[None, None]).sum(axis=-1)
    lut = np.array([-1] + gids)
    return lut[np.where(d.min(axis=-1) < 90, d.argmin(axis=-1) + 1, 0)]


def head_fit(p: Prepared, labels: np.ndarray, model: RosetteModel,
             merged: dict[int, int] | None = None) -> HeadFit | None:
    """Our rendered head against the photo's color labels on a grid over the head
    circle (HEAD_FIT_*). merged maps photo label ids that assign merged away onto the
    group that absorbed them; other ids no model group has count as a mismatch. None
    when the model has no head parts."""
    ours_img = _head_render(model)
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


def head_overlap(a: RosetteModel, b: RosetteModel) -> float | None:
    """Silhouette IoU of two models' drawn heads (every part but the tail stripes)."""
    ra, rb = _head_render(a), _head_render(b)
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
    (or the detected model has no head parts to measure)."""
    if library is None:
        return False
    return rules is None or library.silhouette >= rules.silhouette + TEMPLATE_MIN_GAIN - 1e-9


def gross_failure(fit: HeadFit | None) -> bool:
    """The head-fit check's gross failure (layer agreement under HEAD_FIT_MIN or silhouette
    under HEAD_SILHOUETTE_MIN); False without a measurement."""
    return fit is not None and (fit.agreement < HEAD_FIT_MIN or fit.silhouette < HEAD_SILHOUETTE_MIN)


def override_allowed(rules: HeadFit | None, library: HeadFit | None, partition: float) -> bool:
    """A grossly failed rule-based head (gross_failure) is replaced by the library head
    below the color floor only when the library outline wins by TEMPLATE_OVERRIDE_GAIN and
    its tail-mapped partition reaches TEMPLATE_OVERRIDE_PARTITION."""
    return (gross_failure(rules) and library is not None
            and library.silhouette >= rules.silhouette + TEMPLATE_OVERRIDE_GAIN - 1e-9
            and partition >= TEMPLATE_OVERRIDE_PARTITION - 1e-9)


def library_head_path(rules: HeadFit | None, library: HeadFit | None, partition: float) -> str | None:
    """How the best library head is used: None (its outline does not win, prefer_template),
    "floor" (its colors agree, colors_agree), "override" (the rule-based head failed,
    override_allowed: drawn flagged) or "colors" (the rule-based head is drawn with the
    colors warning, the library head waits on the head card)."""
    if not prefer_template(rules, library):
        return None
    if colors_agree(partition):
        return "floor"
    return "override" if override_allowed(rules, library, partition) else "colors"


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
    params: dict = {"template_id": t.sid, "cx": circle[0], "cy": circle[1], "r": circle[2], "rotation_deg": 0.0,
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
    if choice == RULES_HEAD:
        drawn = rules
    else:
        drawn = [q for q in cands if q.params.get("template_id") == choice][:1]
        if not drawn:
            raise ValueError(f"no head {choice!r} in the model")
    keep = {id(q) for q in drawn}
    out.parts = [q for q in out.parts if q.role == "tail_stripe"] + drawn
    out.alternatives = [q for q in rules + cands if id(q) not in keep]
    _split_groups(out)
    return out


def use_rules_head(model: RosetteModel) -> RosetteModel:
    """The toggle: the detected head parts when the model draws a library head, else the
    first-ranked library head (the decision rule's choice)."""
    if any(q.role == "template" for q in model.parts):
        return use_head(model, RULES_HEAD)
    _, cands = _head_pool(model)
    if not cands:
        return copy.deepcopy(model)
    return use_head(model, str(cands[0].params["template_id"]))


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


def detect(data: bytes) -> Detection:
    """prepare -> pixel_groups -> detect_parts -> tidy -> RosetteModel -> assign."""
    p = prepare(data)
    merged_print: list = []
    labels, groups = pixel_groups(p, merged=merged_print)
    tidy = merge_tail_slivers(drop_specks(RosetteModel(parts=detect_parts(p, labels, groups)),
                                          head_r=Frame.of(p).d(p.head[2])))
    parts = tidy.parts
    used = {q.color_group for q in parts}
    model = RosetteModel(parts=parts, groups=[g for g in groups if g.id in used],
                         rose=any(q.role == "center" for q in parts), warnings=tidy.warnings)
    out = assign(model)
    # assign keeps the parts (same order and ids) and only relabels merged groups
    assert [q.id for q in model.parts] == [q.id for q in out.parts]
    merged = {a.color_group: b.color_group for a, b in zip(model.parts, out.parts)
              if a.color_group != b.color_group}
    return Detection(p, labels, groups, out, merged, merged_print)


def analyze_with_fit(data: bytes, template: bool = True,
                     exclude_key: str | None = None) -> tuple[RosetteModel, HeadFit | None]:
    """analyze, plus the head fit measurement (None without head parts). template: try
    a library head (with_template); exclude_key: leave-one-out evaluation only."""
    d = detect(data)
    p, labels, groups, out, merged, merged_print = d.prepared, d.labels, d.groups, d.model, d.merged, d.merged_print
    used = {q.color_group for q in out.parts} | set(merged)
    fit = head_fit(p, labels, out, merged)
    if template:
        lib = with_template(out, p.mask, labels, p.head, Frame.of(p), groups, merged, exclude_key=exclude_key)
        if lib.alternatives:
            lib_fit = head_fit(p, labels, lib, merged)
            tmpl = next(q for q in lib.parts if q.role == "template")
            path = library_head_path(fit, lib_fit, float(tmpl.params.get("partition", 0.0)))
            if path is not None:
                agree = path == "floor"
                if path != "colors":
                    overlap = head_overlap(out, lib)
                    out, fit = lib, lib_fit
                    if not agree:  # drawn because the rule-based head failed: flagged
                        for q in out.parts:
                            if q.role == "template":
                                q.confidence = min(q.confidence, FLAGGED_CONF)
                    if overlap is not None and (overlap < TEMPLATE_DIFFERS_IOU or not agree):
                        out.warnings.append(TEMPLATE_DIFFERS_WARNING.format(iou=overlap))
                else:  # the rule-based head, with the library head ready on the toggle
                    out = use_rules_head(lib)
                    out.warnings.append(TEMPLATE_COLORS_WARNING.format(tid=tmpl.params["template_id"]))
    out = _check_head(out, fit)
    by_id = {g.id: g for g in groups}
    for gid in mixed_texture_groups(p, labels, groups):
        if gid in used:
            out.warnings.append(f"Color {hex_rgb(by_id[gid].rgb)} is partly patterned - a patterned ribbon "
                                "may have been merged into it; check the colors.")
    for rgb in merged_print:
        out.warnings.append(f"Center badge print was merged into the badge color (print {hex_rgb(rgb)}) - "
                            "check the center against the photo.")
    for gid in badge_print_unsure(p, labels, groups):
        if gid in used:
            out.warnings.append(f"Color {hex_rgb(by_id[gid].rgb)} may be print on the center badge (a logo) "
                                "rather than a ribbon color - check the colors and the center.")
    return out, fit


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
