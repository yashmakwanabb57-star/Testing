"""Cut color shapes where a part meant to lie above them would be hidden by the
shop's fixed layer order.

The SVG contract paints Color 1 first (Layer_1), then Colors 2, 3 and 4 (Layer_2_1_2/3/4),
whatever the parts' stacking. A Color 1 head over the tails' tops, or a Color 1
center over a Color 3 star, would therefore disappear under the lower part. The
designer files avoid this by drawing the lower part already cut (tail tops follow
the head's edge); `clip_occluded` does the same: every color shape is cut by the
union of the shapes meant to lie above it (higher z) that the layer order would
paint below it (a lower Color).
"""
from __future__ import annotations

import re

from shapely import Point, Polygon, unary_union
from shapely.geometry.base import BaseGeometry

from ..shapes import Shape, fmt

# Segments per quarter circle when circles become polygons.
QUAD_SEGS = 32
# Cut shapes are simplified to this tolerance (canvas units).
SIMPLIFY = 0.05
# Pieces smaller than this (canvas units^2) are dropped.
MIN_PIECE = 1.0

_NUM = r"-?\d*\.?\d+(?:e-?\d+)?"


def _paint_rank(category: int | None) -> int:
    return int(category or 0)


def _ring_circles(d: str) -> list[tuple[float, float, float]]:
    """(cx, cy, r) of each circle subpath written by geometry.rings."""
    out = []
    for m in re.finditer(rf"M({_NUM}),({_NUM})a({_NUM}),", d):
        x, y, r = (float(v) for v in m.groups())
        out.append((x - r, y, r))
    return out


def to_geom(s: Shape) -> BaseGeometry | None:
    """Shapely geometry of a color shape (None when the shape kind is not known)."""
    a = s.attrs
    if s.tag == "polygon":
        nums = [float(v) for v in re.findall(_NUM, a["points"])]
        pts = list(zip(nums[0::2], nums[1::2]))
        return Polygon(pts).buffer(0) if len(pts) >= 3 else None
    if s.tag == "circle":
        return Point(float(a["cx"]), float(a["cy"])).buffer(float(a["r"]), quad_segs=QUAD_SEGS)
    if s.tag == "path" and "a" in a.get("d", ""):
        circles = _ring_circles(a["d"])
        if not circles:
            return None
        geom = Point(circles[0][0], circles[0][1]).buffer(circles[0][2], quad_segs=QUAD_SEGS)
        for cx, cy, r in circles[1:]:
            geom = geom.symmetric_difference(Point(cx, cy).buffer(r, quad_segs=QUAD_SEGS))
        return geom
    return None


def _ring_d(coords) -> str:
    pts = list(coords)[:-1]  # shapely repeats the first point
    return "M" + " ".join(f"{fmt(x)},{fmt(y)}" for x, y in pts) + "z"


def from_geom(geom: BaseGeometry, like: Shape) -> Shape | None:
    """A shape with `like`'s color attributes drawing `geom` (polygon when it is one
    simple piece, otherwise an even-odd path); None when nothing is left."""
    polys = [g for g in getattr(geom, "geoms", [geom]) if isinstance(g, Polygon) and g.area >= MIN_PIECE]
    if not polys:
        return None
    if len(polys) == 1 and not polys[0].interiors:
        pts = list(polys[0].exterior.coords)[:-1]
        attrs = {"points": " ".join(f"{fmt(x)},{fmt(y)}" for x, y in pts)}
        tag = "polygon"
    else:
        d = "".join(_ring_d(r.coords) for p in polys for r in (p.exterior, *p.interiors))
        attrs = {"d": d, "fill-rule": "evenodd"}
        tag = "path"
    return Shape(tag, attrs, like.category, pattern=like.pattern, role=like.role, z=like.z, kind=like.kind)


def clip_occluded(colors: list[Shape]) -> list[Shape]:
    """`colors` in intended paint order (low z first); each shape cut by the later
    shapes the layer order paints underneath it. Shapes left empty are dropped."""
    geoms = [to_geom(s) for s in colors]
    out = []
    for i, s in enumerate(colors):
        above = [g for t, g in zip(colors[i + 1:], geoms[i + 1:])
                 if g is not None and _paint_rank(t.category) < _paint_rank(s.category)]
        g = geoms[i]
        if g is None or not above:
            out.append(s)
            continue
        cover = unary_union(above)
        if not g.intersects(cover) or g.intersection(cover).area < MIN_PIECE:
            out.append(s)
            continue
        shape = from_geom(g.difference(cover).simplify(SIMPLIFY), s)
        if shape is not None:
            out.append(shape)
    return out
