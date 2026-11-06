"""Photo -> RosetteModel -> SVG, end to end.

analyze(data): prepare -> pixel_groups -> detect_parts -> RosetteModel -> assign.
Color groups no part uses are dropped before assignment (ids kept as they are);
the rose is on iff a center part was found.
A plain group that looks partly patterned on the tails (colors.mixed_texture_groups)
is warned: a patterned ribbon may have been merged into it. A small patterned color on
the center badge that is not clearly enclosed (colors.badge_print_unsure) is warned too.
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
from .model import RosetteModel
from .roles import PAD_FRAC, render_categories
from .validate import Issue, validate
from .vision.colors import badge_print_unsure, mixed_texture_groups, pixel_groups
from .vision.parts import detect_parts
from .vision.parts_common import Frame
from .vision.prepare import Prepared, prepare
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


@dataclass
class HeadFit:
    agreement: float  # share of the region's foreground union with the same color group
    silhouette: float  # foreground IoU


@dataclass
class RenderResult:
    svg: str
    issues: list[Issue] = field(default_factory=list)


def _head_render(model: RosetteModel) -> np.ndarray | None:
    """Group id per canvas pixel (-1 none) of the model's head parts, or None."""
    head = [q for q in model.parts if q.role != "tail_stripe"]
    gids = sorted({q.color_group for q in head})
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


def _check_head(model: RosetteModel, fit: HeadFit | None) -> RosetteModel:
    out = copy.deepcopy(model)
    if fit is None:
        return out
    bad_layers, bad_outline = fit.agreement < HEAD_FIT_MIN, fit.silhouette < HEAD_SILHOUETTE_MIN
    if not (bad_layers or bad_outline):
        return out
    for q in out.parts:
        if q.role == "head":
            q.confidence = min(q.confidence, FLAGGED_CONF)
    what = []
    if bad_layers:
        what.append("the head layers (disc, rings, points, center: shapes, counts and sizes)")
    if bad_outline:
        what.append("the head outline (missing or extra layers, holes)")
    out.warnings.append(f"The drawn head is far from the photo (head match {fit.agreement:.0%}, outline "
                        f"{fit.silhouette:.0%}) - check {' and '.join(what)} against the photo.")
    return out


def analyze(data: bytes) -> RosetteModel:
    return analyze_with_fit(data)[0]


def analyze_with_fit(data: bytes) -> tuple[RosetteModel, HeadFit | None]:
    """analyze, plus the head fit measurement (None without head parts)."""
    p = prepare(data)
    merged_print: list = []
    labels, groups = pixel_groups(p, merged=merged_print)
    parts = detect_parts(p, labels)
    used = {q.color_group for q in parts}
    model = RosetteModel(parts=parts, groups=[g for g in groups if g.id in used],
                         rose=any(q.role == "center" for q in parts))
    out = assign(model)
    # assign keeps the parts (same order and ids) and only relabels merged groups
    assert [q.id for q in model.parts] == [q.id for q in out.parts]
    merged = {a.color_group: b.color_group for a, b in zip(model.parts, out.parts)
              if a.color_group != b.color_group}
    fit = head_fit(p, labels, out, merged)
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
