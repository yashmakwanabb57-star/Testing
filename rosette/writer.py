"""Shapes -> SVG string in the exact structure the brroset configurator expects."""
from __future__ import annotations

from lxml import etree

from .shapes import Shape

SVG = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"
XML = "http://www.w3.org/XML/1998/namespace"
NUMBERS = (0, 1, 2, 3, 5)
STROKE_RULES = {
    "st4": "fill:none;stroke:#000000;stroke-linecap:round;stroke-miterlimit:10;",
    "st6": "fill:none;stroke:#000000;stroke-miterlimit:10;",
    "st_circel": "fill:none;stroke:#000000;stroke-miterlimit:10;",
}
KIND_CLASS = {"outline": "st6", "rose": "st4", "circel": "st_circel"}


def _q(name: str) -> str:
    return f"{{{SVG}}}{name}"


def _el(parent, tag, attrib=None):
    e = etree.SubElement(parent, _q(tag))
    for k, v in (attrib or {}).items():
        e.set(k, v)
    return e


def _assign_numbers(shapes: list[Shape]) -> dict[tuple[int, str], int]:
    """Per category, N cycles 0,1,2,3,5 by first appearance of role."""
    seen: dict[int, dict[str, int]] = {}
    for s in shapes:
        if s.kind != "color" or s.category is None:
            continue
        roles = seen.setdefault(s.category, {})
        if s.role not in roles:
            roles[s.role] = NUMBERS[len(roles) % len(NUMBERS)]
    return {(c, r): n for c, roles in seen.items() for r, n in roles.items()}


def write_svg(shapes: list[Shape]) -> str:
    numbers = _assign_numbers(shapes)
    use4 = any(s.kind == "color" and s.category == 4 for s in shapes)

    root = etree.Element(
        _q("svg"),
        nsmap={None: SVG, "xlink": XLINK},
    )
    root.set("version", "1.1")
    root.set("x", "0px")
    root.set("y", "0px")
    root.set("viewBox", "0 0 600 850")
    root.set("style", "enable-background:new 0 0 600 850;")
    root.set(f"{{{XML}}}space", "preserve")
    style = _el(root, "style", {"type": "text/css"})

    g1 = _el(root, "g", {"id": "Layer_1"})
    g3 = _el(root, "g", {"id": "Layer_3"})
    g32 = _el(g3, "g", {"id": "Layer_3_2"})
    defs = _el(g32, "defs")
    g33 = _el(g3, "g", {"id": "Layer_3_3"})
    g34 = _el(g3, "g", {"id": "Layer_3_4"}) if use4 else None
    g21 = _el(root, "g", {"id": "Layer_2_1"})
    g212 = _el(g21, "g", {"id": "Layer_2_1_2"})
    g213 = _el(g21, "g", {"id": "Layer_2_1_3"})
    g214 = _el(g21, "g", {"id": "Layer_2_1_4"}) if use4 else None
    g6 = _el(root, "g", {"id": "Layer_6"})
    grose = _el(root, "g", {"id": "Layer_rose"})

    used: list[str] = []
    pattern_ids: list[str] = []

    def emit(group, s: Shape, cls: str):
        if cls not in used:
            used.append(cls)
        attrs = {"class": cls}
        attrs.update(s.attrs)
        _el(group, s.tag, attrs)

    for s in sorted(shapes, key=lambda s: s.z):  # stable: ties keep input order
        if s.kind in KIND_CLASS:
            emit(g6 if s.kind == "outline" else grose, s, KIND_CLASS[s.kind])
            continue
        if s.category is None:
            raise ValueError("color shape needs a category")
        n = numbers[(s.category, s.role)]
        c = s.category
        if c == 1:
            emit(g1, s, f"st{n}_1")
        elif c == 2:
            pat = f"st{n}_2_pattern"
            emit(g32, s, pat)
            emit(g212, s, f"st{n}_2")
            if pat not in pattern_ids:
                pattern_ids.append(pat)
        elif c == 3:
            emit(g33, s, f"st{n}_3")
            emit(g213, s, f"st{n}_3")
        elif c == 4:
            emit(g34, s, f"st{n}_4")
            emit(g214, s, f"st{n}_4")
        else:
            raise ValueError(f"bad category {c}")

    for pat in pattern_ids:
        p = _el(defs, "pattern", {"id": f"image_{pat}", "patternUnits": "userSpaceOnUse",
                                  "width": "100", "height": "100"})
        _el(p, "image", {f"{{{XLINK}}}href": "", "x": "0", "y": "0",
                         "width": "100", "height": "100"})
    if not pattern_ids:
        g32.remove(defs)

    rules = [f".{c}{{{STROKE_RULES.get(c, 'fill:none;')}}}" for c in used]
    style.text = "\n" + "\n".join(rules) + "\n"
    return etree.tostring(root, encoding="unicode")
