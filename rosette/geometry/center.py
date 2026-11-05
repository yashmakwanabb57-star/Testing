from __future__ import annotations

from ..shapes import Shape, fmt


def disc(cx: float, cy: float, r: float) -> Shape:
    return Shape("circle", {"cx": fmt(cx), "cy": fmt(cy), "r": fmt(r)}, None)
