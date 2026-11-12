"""Cut every color shape to its visible piece (H14).

The SVG contract paints Color 1 first (Layer_1), then Colors 2, 3 and 4 (Layer_2_1_2/3/4),
whatever the parts' stacking, and the reference files draw each element as the piece of it
that shows: a loop stops at the star lying on it, a tail top follows the head's edge, so no
outline crosses another (79 references: no two head elements overlap in 72, apart from
elements written twice; H14 mining). `clip_visible` does the same: every color shape is
cut by the union of all shapes stacked above it - higher z, or equal z and painted later
by the shop (higher Color, then later in the file) - whatever their Colors or parts. The
shop shows the same colors as with whole shapes (the cut parts were hidden), the outline
view shows no hidden edge, and the pattern twin (Layer_3) is written from the same piece.
Before H14 (`clip_occluded`, H3) a shape was cut only where the layer order would have
painted it over a shape meant to lie above it.
"""
from __future__ import annotations

import re

from shapely import Point, Polygon, set_precision, unary_union
from shapely.geometry.base import BaseGeometry

from ..shapes import Shape, fmt

# Grid of the written coordinates (`shapes.fmt`: 3 decimals). Cut pieces are snapped to it
# before they are written, so rounding cannot make a ring cross or touch itself.
GRID = 1e-3
# Segments per quarter circle when circles become polygons.
QUAD_SEGS = 32
# Cut shapes are simplified to this tolerance (canvas units).
SIMPLIFY = 0.05
# Pieces smaller than this (canvas units^2) are dropped; a shape overlapped by less than
# this is kept whole.
MIN_PIECE = 1.0
# Visible pieces whose mean width (2 * area / perimeter, canvas units) is under this are
# slivers left between near-coincident edges (a strip along the cutter's side), dropped.
SLIVER = 0.5

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


def xor_all(geoms: list[BaseGeometry]) -> BaseGeometry | None:
    """Even-odd fill of the given rings' areas (symmetric difference of all), reduced
    pairwise: a long path of many rings takes O(n log n) instead of O(n^2) (S61-19)."""
    geoms = list(geoms)
    while len(geoms) > 1:
        geoms = [geoms[i].symmetric_difference(geoms[i + 1]) if i + 1 < len(geoms) else geoms[i]
                 for i in range(0, len(geoms), 2)]
    return geoms[0] if geoms else None


def to_geom(s: Shape) -> BaseGeometry:
    """Shapely geometry of a color shape; ValueError for a shape this module cannot
    read (only the geometry package's polygons, circles, ring paths and the even-odd
    paths of cut shapes are known)."""
    a = s.attrs
    if s.tag == "polygon":
        nums = [float(v) for v in re.findall(_NUM, a["points"])]
        pts = list(zip(nums[0::2], nums[1::2]))
        if len(pts) >= 3:
            return Polygon(pts).buffer(0)
    if s.tag == "circle":
        return Point(float(a["cx"]), float(a["cy"])).buffer(float(a["r"]), quad_segs=QUAD_SEGS)
    if s.tag == "path" and re.fullmatch(r"(M[-\d.,e ]+z)+", a.get("d", "")):
        rings = []  # an even-odd path of straight rings (from_geom)
        for sub in a["d"].split("M")[1:]:
            nums = [float(v) for v in re.findall(_NUM, sub)]
            if len(nums) >= 6:
                rings.append(Polygon(list(zip(nums[0::2], nums[1::2]))).buffer(0))
        geom = xor_all(rings)
        if geom is not None:
            return geom
    if s.tag == "path" and "a" in a.get("d", ""):
        circles = _ring_circles(a["d"])
        if not circles:
            raise ValueError(f"cannot clip {s.role or 'color'} path: {a['d'][:40]}")
        return xor_all([Point(cx, cy).buffer(r, quad_segs=QUAD_SEGS) for cx, cy, r in circles])
    raise ValueError(f"cannot clip {s.role or 'color'} shape <{s.tag}>")


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


def stack_key(s: Shape, index: int) -> tuple[int, int, int]:
    """Stacking of a color shape among the shapes of one build (higher shows): z, then the
    shop's paint order (Color, then the order in the file)."""
    return s.z, _paint_rank(s.category), index


def _piece(geom: BaseGeometry, like: Shape) -> Shape | None:
    """`like` cut to `geom`: simplified to SIMPLIFY, snapped to GRID (valid there), slivers
    and specks dropped."""
    geom = set_precision(geom.simplify(SIMPLIFY), GRID)
    keep = [g for g in getattr(geom, "geoms", [geom]) if isinstance(g, Polygon)
            and g.area >= MIN_PIECE and 2 * g.area / g.length >= SLIVER]
    return from_geom(unary_union(keep), like) if keep else None


def clip_visible(colors: list[Shape]) -> list[Shape]:
    """`colors` (in the order the writer gets them), each cut to the part no shape stacked
    above it (`stack_key`) covers. A shape nothing covers (by MIN_PIECE or more) is kept
    as it is, circles and ring paths included; shapes left empty are dropped."""
    geoms = [to_geom(s) for s in colors]
    out: list[Shape | None] = [None] * len(colors)
    cover = None
    for i in sorted(range(len(colors)), key=lambda k: stack_key(colors[k], k), reverse=True):
        s, g = colors[i], geoms[i]
        if cover is None or not g.intersects(cover) or g.intersection(cover).area < MIN_PIECE:
            out[i] = s
        else:
            out[i] = _piece(g.difference(cover), s)
        cover = g if cover is None else unary_union([cover, g])
    return [s for s in out if s is not None]


def clip_occluded(colors: list[Shape]) -> list[Shape]:
    """The pre-H14 cut (H3), kept for fill-only renders: each shape cut only by the shapes
    of strictly higher z that the shop's layer order paints underneath it, so the colors
    show as stacked but overlapping shapes stay whole. Such a render has no seams where
    visible pieces abut (antialiased seam pixels blend two colors), which the head fit's
    per-pixel color reading needs (`pipeline._head_render`); written files use
    `clip_visible`."""
    geoms = [to_geom(s) for s in colors]
    out = []
    for i, s in enumerate(colors):
        above = [g for t, g in zip(colors[i + 1:], geoms[i + 1:])
                 if t.z > s.z and _paint_rank(t.category) < _paint_rank(s.category)]
        g = geoms[i]
        cover = unary_union(above) if above else None
        if cover is None or not g.intersects(cover) or g.intersection(cover).area < MIN_PIECE:
            out.append(s)
            continue
        shape = from_geom(g.difference(cover).simplify(SIMPLIFY), s)
        if shape is not None:
            out.append(shape)
    return out
