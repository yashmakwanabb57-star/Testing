"""Model clean-up between detection and assignment (H6).

drop_specks(model): an `unknown` part whose disc is under SPECK_MAX_HEAD_SHARE of the
head disc is not drawn (one warning names it), wherever it lies. Larger unknowns stay
placeholders; seam specks inside the head are folded by parts_trims before this.
"""
from __future__ import annotations

import copy
import math

from .model import RosetteModel

# An unknown part under this share of the head disc's area is a speck and is not drawn.
# Measured over the 76 reference photos (H6, at db5d68f): see docs/pipeline.md (H6).
SPECK_MAX_HEAD_SHARE = 0.01


def drop_specks(model: RosetteModel) -> RosetteModel:
    """A copy of the model without unknown parts under SPECK_MAX_HEAD_SHARE of the head
    disc, one warning per dropped part. Without a head part nothing is dropped."""
    out = copy.deepcopy(model)
    head = next((q for q in out.parts if q.role == "head"), None)
    if head is None:
        return out
    head_area = math.pi * float(head.params["r"]) ** 2
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
