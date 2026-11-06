"""Model clean-up between detection and assignment (H6).

drop_specks(model): an `unknown` part whose disc is under SPECK_MAX_HEAD_SHARE of the
head disc is not drawn (one warning names it), wherever it lies. Larger unknowns stay
placeholders; seam specks inside the head are folded by parts_trims before this.
"""
from __future__ import annotations

import copy
import math

from .model import RosetteModel

# An unknown part under this share of the head circle's area is a speck and is not drawn.
# Over the 76 reference photos (H6, at db5d68f; 67 unknowns, area pi r^2 over the photo's
# head circle) the specks on the head rim (0.96-1.09 head radii out) measure 0.0022-0.0039
# (S1-15, S13-15, S55-15, S51-19, A13-15B twice); the next unknowns are real structure
# drawn as placeholders: S9-19's bow loop tips beyond the head circle 0.0059 and 0.0111,
# S58-17's badge print 0.0087, S29-15's loop tips 0.0100. The bar sits between them, a thin
# margin (x1.28 above the largest speck, x0.85 of the smallest kept piece). Real parts can
# be this small (A1-15 center 0.0024), but only unknown parts are dropped.
SPECK_MAX_HEAD_SHARE = 0.005


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
        share = math.pi * float(q.params.get("r", 0.0)) ** 2 / head_area if q.role == "unknown" else 1.0
        if share < SPECK_MAX_HEAD_SHARE:
            out.warnings.append(f"Part {q.id} ({share:.1%} of the head) is too small to draw and was "
                                "left out - check that spot against the photo.")
            continue
        kept.append(q)
    out.parts = kept
    return out
