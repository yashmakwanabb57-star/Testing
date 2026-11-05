"""Photo -> RosetteModel -> SVG, end to end.

analyze(data): prepare -> pixel_groups -> detect_parts -> RosetteModel -> assign.
Color groups no part uses are dropped before assignment (ids kept as they are);
the rose is on iff a center part was found.
A plain group that looks partly patterned on the tails (colors.mixed_texture_groups)
is warned: a patterned ribbon may have been merged into it.

render(model): build -> write_svg -> validate. build may add warnings to the model
(unrecognised shapes drawn as placeholders).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .assign import FLAG_BELOW, assign, hex_rgb
from .build import build
from .model import RosetteModel
from .validate import Issue, validate
from .vision.colors import mixed_texture_groups, pixel_groups
from .vision.parts import detect_parts
from .vision.prepare import prepare
from .writer import write_svg


@dataclass
class RenderResult:
    svg: str
    issues: list[Issue] = field(default_factory=list)


def analyze(data: bytes) -> RosetteModel:
    p = prepare(data)
    labels, groups = pixel_groups(p)
    parts = detect_parts(p, labels)
    used = {q.color_group for q in parts}
    model = RosetteModel(parts=parts, groups=[g for g in groups if g.id in used],
                         rose=any(q.role == "center" for q in parts))
    out = assign(model)
    by_id = {g.id: g for g in groups}
    for gid in mixed_texture_groups(p, labels, groups):
        if gid in used:
            out.warnings.append(f"Color {hex_rgb(by_id[gid].rgb)} is partly patterned - a patterned ribbon "
                                "may have been merged into it; check the colors.")
    return out


def render(model: RosetteModel) -> RenderResult:
    svg = write_svg(build(model))
    return RenderResult(svg=svg, issues=validate(svg))


def is_flagged(model: RosetteModel, issues: list[Issue]) -> bool:
    """True when the user is asked to check something: a low-confidence color, a
    points ring read with low confidence (weak rotational symmetry), an unrecognised
    (placeholder) part, any warning, or a validation issue."""
    return (any(g.confidence < FLAG_BELOW for g in model.groups)
            or any(q.role == "points" and q.confidence < FLAG_BELOW for q in model.parts)
            or any(q.role == "unknown" for q in model.parts)
            or bool(model.warnings)
            or bool(issues))
