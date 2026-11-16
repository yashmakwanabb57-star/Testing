from __future__ import annotations

import math

from ..shapes import Shape
from .common import TOOTH, points_attr, polar, scallop_edge


def pinked_circle(cx: float, cy: float, r: float, tooth: float = TOOTH) -> Shape:
    """Circle with a pleated rim: scallops reaching r (scallop_edge, as the designer draws it)."""
    return Shape("polygon", {"points": points_attr(scallop_edge(cx, cy, r))}, None)
