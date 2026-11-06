"""st6 line art: the visible edges of the color shapes (H5).

The shop draws Layer_6 over everything, so an outline traced around a whole shape shows
edges that a shape painted later hides (A7-15: the tops of Color 1 tail stripes under the
Color 1 head disc). Each outline is the shape's boundary minus every shape painted after
it in the written file: Layer_1 (Color 1) first, then the Layer_2_1 groups (Colors 2, 3,
4; Layer_3 is their pattern twin), each in the shapes' order. Circles and ring paths that
nothing covers keep their own outline element."""
from __future__ import annotations

from shapely import LineString, MultiLineString, line_merge, unary_union

from ..shapes import Shape, fmt

# Edges of a lower shape within this distance (canvas units) of a later shape are left
# to the later shape's outline (they coincide: cut shapes follow the cutter's edge).
COVER_TOL = 0.05
# Visible fragments shorter than this (canvas units) are corner crumbs of the cut, dropped.
MIN_FRAGMENT = 2 * COVER_TOL


def _element(s: Shape) -> Shape:
    if s.tag == "polygon":
        nums = s.attrs["points"].split()
        return Shape("polyline", {"points": " ".join(nums + nums[:1])}, None, role=s.role, z=s.z, kind="outline")
    attrs = {k: v for k, v in s.attrs.items() if k != "fill-rule"}
    return Shape(s.tag, attrs, None, role=s.role, z=s.z, kind="outline")


def _lines(geom) -> list[LineString]:
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        merged = line_merge(geom)
        return list(getattr(merged, "geoms", [merged]))
    return [g for part in getattr(geom, "geoms", []) for g in _lines(part)]


def outlines(shapes: list[Shape]) -> list[Shape]:
    """st6 outlines (kind outline) of the color `shapes`, given in the order the writer
    emits them within each Color, tracing only what no later-painted shape hides."""
    from .clip import to_geom
    order = sorted(range(len(shapes)), key=lambda i: (int(shapes[i].category or 0), i))
    geoms = {i: to_geom(shapes[i]) for i in order}
    out: list[Shape] = []
    for rank, i in enumerate(order):
        s = shapes[i]
        later = [geoms[j] for j in order[rank + 1:] if geoms[j].intersects(geoms[i])]
        if not later:
            out.append(_element(s))
            continue
        edge = geoms[i].boundary.difference(unary_union(later).buffer(COVER_TOL))
        if edge.equals(geoms[i].boundary):
            out.append(_element(s))
            continue
        for line in _lines(edge):
            pts = list(line.coords)
            if len(pts) >= 2 and line.length >= MIN_FRAGMENT:
                out.append(Shape("polyline", {"points": " ".join(f"{fmt(x)},{fmt(y)}" for x, y in pts)},
                                 None, role=s.role, z=s.z, kind="outline"))
    return out
