"""Model clean-up between detection and assignment (H6).

drop_specks(model): an `unknown` part whose disc is under SPECK_MAX_HEAD_SHARE of the
head disc is not drawn (one warning names it), wherever it lies. Larger unknowns stay
placeholders; seam specks inside the head are folded by parts_trims before this.

drop_two_point_layers(model): a points layer of fewer than 3 lobes is not drawn (H13), with a
warning: two lobes are what a mirror-symmetric head's half turn reads, never a points layer.

merge_tail_slivers(model): a tail stripe under SLIVER_SHARE of the median stripe width
(assign.tail_slivers) is merged into a neighbouring stripe, with a warning.
"""
from __future__ import annotations

import copy
import math

from .assign import SLIVER_SHARE, tail_slivers
from .build import _TAIL_KEYS
from .model import Part, RosetteModel

# An unknown part under this share of the head circle's area is a speck and is not drawn.
# Over the 76 reference photos (H6, at db5d68f; 67 unknowns, area pi r^2 over the photo's
# head circle) the specks on the head rim (0.96-1.09 head radii out) measure 0.0022-0.0039
# (S1-15, S13-15, S55-15, S51-19, A13-15B twice); the next unknowns are real structure
# drawn as placeholders: S9-19's bow loop tips beyond the head circle 0.0059 and 0.0111,
# S58-17's badge print 0.0087, S29-15's loop tips 0.0100. The bar sits between them, a thin
# margin (x1.28 above the largest speck, x0.85 of the smallest kept piece). Real parts can
# be this small (A1-15 center 0.0024), but only unknown parts are dropped.
SPECK_MAX_HEAD_SHARE = 0.005
# Id prefix of the placeholder parts_tails makes for a tail stripe without a color group:
# part of the tail, never a speck (leaving it out would leave a hole in the tail).
TAIL_PLACEHOLDER = "unknown-tail-"


def drop_specks(model: RosetteModel, head_r: float | None = None) -> RosetteModel:
    """A copy of the model without unknown parts under SPECK_MAX_HEAD_SHARE of the head
    circle (radius head_r, canvas units: the photo's head circle; default the head part's
    radius), one warning per dropped part. Without either nothing is dropped."""
    out = copy.deepcopy(model)
    if head_r is None:
        head = next((q for q in out.parts if q.role == "head"), None)
        if head is None:
            return out
        head_r = float(head.params["r"])
    head_area = math.pi * head_r ** 2
    kept = []
    for q in out.parts:
        speck = q.role == "unknown" and not q.id.startswith(TAIL_PLACEHOLDER)
        share = math.pi * float(q.params.get("r", 0.0)) ** 2 / head_area if speck else 1.0
        if share < SPECK_MAX_HEAD_SHARE:
            out.warnings.append(f"Part {q.id} ({share:.1%} of the head) is too small to draw and was "
                                "left out - check that spot against the photo.")
            continue
        kept.append(q)
    out.parts = kept
    return out


# Points layers need at least this many lobes to be drawn (H13).
MIN_POINTS = 3


def drop_two_point_layers(model: RosetteModel) -> RosetteModel:
    """A copy of the model without points layers of fewer than MIN_POINTS lobes, one
    warning per dropped layer (S53-19: white pleat teeth read as a 2-point "star")."""
    out = copy.deepcopy(model)
    kept = []
    for q in out.parts:
        if q.role == "points" and int(q.params.get("count", MIN_POINTS)) < MIN_POINTS:
            out.warnings.append(f"Part {q.id} read as a {int(q.params['count'])}-point layer was left out - "
                                "check the head against the photo.")
            continue
        kept.append(q)
    out.parts = kept
    return out


def _fmt(fr: list[float]) -> str:
    return ",".join(f"{v:.4f}" for v in fr)


def merge_tail_slivers(model: RosetteModel) -> RosetteModel:
    """A copy of the model whose tail stripes under SLIVER_SHARE of the median stripe
    width (assign.tail_slivers) are merged into a neighbour: one of the sliver's own
    group, else both neighbours when they share a group (they join), else the wider
    one (by mean share). Stripes are renumbered (ids tail-k-i); one warning names the
    merged stripes. Tails with a stripe missing (an unknown placeholder) are left as
    they are."""
    out = copy.deepcopy(model)
    slivers = {q.id for q in tail_slivers(out.parts)}
    if not slivers:
        return out
    tails: dict[tuple, list[Part]] = {}
    for q in out.parts:
        if q.role == "tail_stripe":
            key = (q.params["tail_index"],) + tuple(q.params.get(k, "") for k in _TAIL_KEYS)
            tails.setdefault(key, []).append(q)
    merged, drop = [], set()
    for stripes in tails.values():
        stripes.sort(key=lambda q: int(q.params["stripe_index"]))
        top = [float(v) for v in str(stripes[0].params["stripes"]).split(",")]
        bot = [float(v) for v in str(stripes[0].params.get("stripes_bottom") or "").split(",") if v] or list(top)
        if ([int(q.params["stripe_index"]) for q in stripes] != list(range(len(top)))
                or len(bot) != len(top) or not any(q.id in slivers for q in stripes)):
            continue
        st, sb = sum(top), sum(bot)
        rows = [[q, t / st, b / sb] for q, t, b in zip(stripes, top, bot)]
        while True:
            cand = [i for i, r in enumerate(rows) if r[0].id in slivers and len(rows) > 1]
            if not cand:
                break
            i = min(cand, key=lambda j: rows[j][1] + rows[j][2])
            nb = [j for j in (i - 1, i + 1) if 0 <= j < len(rows)]
            own = [j for j in nb if rows[j][0].color_group == rows[i][0].color_group]
            if own:
                j = own[0]
            elif len(nb) == 2 and rows[nb[0]][0].color_group == rows[nb[1]][0].color_group:
                j = nb[0]  # both neighbours of one group: they join below
            else:
                j = max(nb, key=lambda n: rows[n][1] + rows[n][2])
            rows[j][1] += rows[i][1]
            rows[j][2] += rows[i][2]
            merged.append(rows[i][0].id)
            drop.add(id(rows[i][0]))
            del rows[i]
            k = 0
            while k < len(rows) - 1:  # neighbours of one group join
                if rows[k][0].color_group == rows[k + 1][0].color_group:
                    rows[k][1] += rows[k + 1][1]
                    rows[k][2] += rows[k + 1][2]
                    drop.add(id(rows[k + 1][0]))
                    del rows[k + 1]
                else:
                    k += 1
        for i, (q, _, _) in enumerate(rows):
            q.params["stripe_index"] = i
            q.params["stripes"] = _fmt([r[1] for r in rows])
            q.params["stripes_bottom"] = _fmt([r[2] for r in rows])
            q.id = f"tail-{q.params['tail_index']}-{i}"
    out.parts = [q for q in out.parts if id(q) not in drop]
    if merged:
        out.warnings.append(f"Tail stripe(s) {', '.join(merged)} narrower than {SLIVER_SHARE:.0%} of the median "
                            "stripe width were merged into a neighbouring stripe - check the tail stripes "
                            "against the photo.")
    return out
