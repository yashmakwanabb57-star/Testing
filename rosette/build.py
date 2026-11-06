"""RosetteModel -> list[Shape] for the SVG writer."""
from __future__ import annotations

from .geometry import disc, outlines, pinked_circle, place_rose, points_ring, ring, tails
from .geometry.clip import clip_occluded
from .model import Part, RosetteModel
from .shapes import Shape

_TAIL_KEYS = ("tails_count", "top_y", "bottom_y", "top_width", "bottom_width",
              "spread_deg", "stripes", "stripes_bottom")


def _parse_stripes(s) -> list[float]:
    return [float(x) for x in str(s).split(",") if x.strip()]


def build(model: RosetteModel) -> list[Shape]:
    groups = {g.id: g for g in model.groups}
    colors: list[Shape] = []

    def finish(part: Part, shapes: list[Shape]) -> None:
        g = groups.get(part.color_group)
        if g is None:
            raise ValueError(f"part {part.id}: unknown color group {part.color_group}")
        if g.category is None:
            raise ValueError(f"part {part.id}: color group {g.id} has no category; run assignment first")
        for s in shapes:
            s.category = g.category
            s.pattern = g.pattern
            s.role = part.role
            s.z = part.z
            s.kind = "color"
            colors.append(s)

    tail_groups: dict[tuple, list[Part]] = {}
    center: Part | None = None
    for p in model.parts:
        q = p.params
        if p.role == "head":
            s = pinked_circle(q["cx"], q["cy"], q["r"]) if int(q.get("pinked", 1)) else disc(q["cx"], q["cy"], q["r"])
            finish(p, [s])
        elif p.role == "ring":
            finish(p, [ring(q["cx"], q["cy"], q["r_outer"], q["r_inner"])])
        elif p.role == "points":
            ir, ew = q.get("inner_ratio"), q.get("edge_width")
            finish(p, points_ring(q["cx"], q["cy"], q["kind"], int(q["count"]), q["r_inner"],
                                  q["r_outer"], q["rotation_deg"], q["width_ratio"],
                                  None if ir is None else float(ir), None if ew is None else float(ew)))
        elif p.role == "center":
            center = center or p
            finish(p, [disc(q["cx"], q["cy"], q["r"])])
        elif p.role == "tail_stripe":
            tail_groups.setdefault(tuple(q.get(k, "") for k in _TAIL_KEYS), []).append(p)
        else:
            msg = f"part {p.id}: unrecognised shape, drawn as a plain disc"
            if msg not in model.warnings:
                model.warnings.append(msg)
            finish(p, [disc(q["cx"], q["cy"], q["r"])])

    for key, parts in tail_groups.items():
        top_y, bottom_y, tw, bw, spread = key[1], key[2], key[3], key[4], key[5]
        polys = tails(top_y, bottom_y, int(key[0]), tw, bw, spread, _parse_stripes(key[6]),
                      stripes_bottom=_parse_stripes(key[7]) or None)
        for p in parts:
            ti, si = int(p.params["tail_index"]), int(p.params["stripe_index"])
            try:
                shape = polys[ti][si]
            except IndexError:
                raise ValueError(f"part {p.id}: tail {ti} stripe {si} out of range") from None
            finish(p, [shape])

    colors.sort(key=lambda s: s.z)
    colors = clip_occluded(colors)  # Color 1 is painted first whatever the stacking
    result = list(colors)
    result += outlines(colors)

    if model.rose and center is not None:
        c = center.params
        rose = place_rose(c["cx"], c["cy"], 0.8 * c["r"])
        result += rose
        circel = disc(c["cx"], c["cy"], c["r"])
        circel.kind, circel.role, circel.z = "circel", "center", center.z
        result.append(circel)
    return result
