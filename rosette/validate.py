"""Structural validator: blocks download of any SVG the shop configurator would mishandle."""
from __future__ import annotations

import importlib
import re
from collections import Counter
from dataclasses import dataclass

from lxml import etree

SVG = "http://www.w3.org/2000/svg"
SHAPE_TAGS = {"path", "polygon", "polyline", "circle", "ellipse"}
ALLOWED_N = {0, 1, 2, 3, 5}
VIEWBOX = "0 0 600 850"

# group id -> required parent id (None = root)
PARENTS = {
    "Layer_1": None, "Layer_3": None, "Layer_2_1": None, "Layer_6": None, "Layer_rose": None,
    "Layer_3_2": "Layer_3", "Layer_3_3": "Layer_3", "Layer_3_4": "Layer_3",
    "Layer_2_1_2": "Layer_2_1", "Layer_2_1_3": "Layer_2_1", "Layer_2_1_4": "Layer_2_1",
}
# spec 3.2: the top-level layer groups, in paint order
TOP_ORDER = ["Layer_1", "Layer_3", "Layer_2_1", "Layer_6", "Layer_rose"]
REQUIRED = ["Layer_1", "Layer_3", "Layer_3_2", "Layer_3_3", "Layer_2_1",
            "Layer_2_1_2", "Layer_2_1_3", "Layer_6", "Layer_rose"]
# color class suffix -> allowed layer groups
SUFFIX_LAYERS = {
    "1": {"Layer_1"},
    "2_pattern": {"Layer_3_2"},
    "2": {"Layer_2_1_2"},
    "3": {"Layer_3_3", "Layer_2_1_3"},
    "4": {"Layer_3_4", "Layer_2_1_4"},
}
SPECIAL_LAYERS = {"st4": {"Layer_rose"}, "st6": {"Layer_6"}, "st_circel": {"Layer_rose"}}
STROKE_ONLY = {"Layer_6", "Layer_rose"}
NEUTRAL_TAGS = {"svg", "g", "defs", "style", "pattern"}
COLOR_RE = re.compile(r"^st(\d+)_(1|2_pattern|2|3|4)$")
COLOR_GROUPS = {g for gs in SUFFIX_LAYERS.values() for g in gs}
TOGGLE_PAIRS = [("Layer_3_2", "Layer_2_1_2"), ("Layer_3_3", "Layer_2_1_3"),
                ("Layer_3_4", "Layer_2_1_4")]


@dataclass
class Issue:
    code: str
    message: str
    element: str | None = None


def _local(e) -> str:
    return etree.QName(e).localname


def _desc(e) -> str:
    bits = [_local(e)]
    for k in ("id", "class"):
        if e.get(k):
            bits.append(f'{k}="{e.get(k)}"')
    return "<" + " ".join(bits) + ">"


def _geom(e):
    attrs = tuple(sorted((k, " ".join(v.split())) for k, v in e.attrib.items() if k != "class"))
    return (_local(e), attrs)


def _layer_of(e, known: set[str]) -> str | None:
    p = e.getparent()
    while p is not None:
        if p.get("id") in known:
            return p.get("id")
        p = p.getparent()
    return None


def _is_closed(e) -> bool:
    tag = _local(e)
    if tag in ("polygon", "circle", "ellipse"):
        return True
    if tag == "polyline":
        pts = re.findall(r"-?\d*\.?\d+(?:e-?\d+)?", e.get("points", ""))
        return len(pts) >= 4 and pts[:2] == pts[-2:]
    if tag == "path":
        d = e.get("d", "")
        subs = [s for s in re.split(r"[Mm]", d) if s.strip()]
        return bool(subs) and all(re.search(r"[zZ]\s*$", s) for s in subs)
    return True


def validate(svg: str, simulate_check: bool = True) -> list[Issue]:
    issues: list[Issue] = []

    def add(code, msg, el=None):
        issues.append(Issue(code, msg, el if isinstance(el, str) or el is None else _desc(el)))

    try:
        root = etree.fromstring(svg.encode("utf-8"))
    except etree.XMLSyntaxError as exc:
        return [Issue("xml", f"not well-formed XML: {exc}", None)]

    # root
    if root.tag != f"{{{SVG}}}svg":
        add("root", "root element is not <svg> in the SVG namespace", root)
    if root.get("viewBox") != VIEWBOX:
        add("root", f'viewBox must be "{VIEWBOX}", got {root.get("viewBox")!r}', root)
    if root.get("id") is not None:
        add("root", "root <svg> must not have an id", root)

    # unique ids
    ids = Counter(e.get("id") for e in root.iter() if isinstance(e.tag, str) and e.get("id"))
    for i, n in ids.items():
        if n > 1:
            add("dup_id", f'id "{i}" appears {n} times', i)

    # groups
    groups: dict[str, list] = {}
    for g in root.iter(f"{{{SVG}}}g"):
        if g.get("id") in PARENTS:
            groups.setdefault(g.get("id"), []).append(g)
    for gid in REQUIRED:
        if gid not in groups:
            add("missing_group", f"required group {gid} is missing", gid)
    for a, b in (("Layer_3_4", "Layer_2_1_4"), ("Layer_2_1_4", "Layer_3_4")):
        if a in groups and b not in groups:
            add("missing_group", f"{a} present but {b} is missing", b)
    for gid, gs in groups.items():
        want = PARENTS[gid]
        for g in gs:
            p = g.getparent()
            ok = (p is root) if want is None else (p is not None and p.get("id") == want)
            if not ok:
                add("bad_parent", f"{gid} must be a child of {want or 'the root svg'}", g)
    top = [c.get("id") for c in root if isinstance(c.tag, str) and _local(c) == "g"
           and c.get("id") in TOP_ORDER]
    if top != sorted(top, key=TOP_ORDER.index):
        add("group_order", f"top-level groups are in the order {', '.join(top)}; "
            f"the shop expects {', '.join(TOP_ORDER)}", "svg")
    known = set(PARENTS)

    # only the five shape tags are drawable; <image> is allowed only inside <pattern>
    for e in root.iter():
        if not isinstance(e.tag, str) or _local(e) in SHAPE_TAGS | NEUTRAL_TAGS:
            continue
        if _local(e) == "image" and e.getparent() is not None                 and _local(e.getparent()) == "pattern":
            continue
        if _local(e) == "line" and _layer_of(e, known) in STROKE_ONLY:
            continue  # live shop SVGs use <line> in stroke-only layers; never filled
        add("bad_element", f"element <{_local(e)}> is not allowed", e)

    # shape elements: class vs layer
    shapes = [e for e in root.iter() if isinstance(e.tag, str) and _local(e) in SHAPE_TAGS]
    used_classes: set[str] = set()
    for e in shapes:
        cls = (e.get("class") or "").split()
        layer = _layer_of(e, known)
        if not cls:
            add("class_layer", "shape has no class", e)
            continue
        for c in cls:
            used_classes.add(c)
            m = COLOR_RE.match(c)
            if m:
                if int(m.group(1)) not in ALLOWED_N:
                    add("class_number", f"class {c}: N must be one of 0,1,2,3,5", e)
                elif layer not in SUFFIX_LAYERS[m.group(2)]:
                    add("class_layer", f"class {c} not allowed in {layer}", e)
            elif c in SPECIAL_LAYERS:
                if layer not in SPECIAL_LAYERS[c]:
                    add("class_layer", f"class {c} not allowed in {layer}", e)
            else:
                add("class_layer", f"unrecognised class {c}", e)

    # pattern defs
    needed = {f"image_{c}" for c in used_classes if c.endswith("_2_pattern") and COLOR_RE.match(c)}
    defined = {p.get("id"): p for p in root.iter(f"{{{SVG}}}pattern")}
    for pid in sorted(needed):
        p = defined.get(pid)
        if p is None or not p.findall(f"{{{SVG}}}image"):
            add("pattern_def_missing", f"pattern {pid} (with <image> child) is missing", pid)
    for pid in sorted(set(defined) - needed):
        add("pattern_def_unused", f"pattern {pid} is defined but never used", pid)

    # toggle sets and duplicates
    def shapes_of(gid):
        return [c for g in groups.get(gid, []) for c in g.iter()
                if isinstance(c.tag, str) and _local(c) in SHAPE_TAGS
                and _layer_of(c, known) == gid]

    for a, b in TOGGLE_PAIRS:
        if a in groups and b in groups:
            ca = Counter(_geom(e) for e in shapes_of(a))
            cb = Counter(_geom(e) for e in shapes_of(b))
            if ca != cb:
                add("toggle_mismatch", f"{a} and {b} do not contain the same geometry "
                    f"({sum(ca.values())} vs {sum(cb.values())} elements)", f"{a}/{b}")
    for gid in groups:
        for geo, n in Counter(_geom(e) for e in shapes_of(gid)).items():
            if n > 1:
                add("dup_geometry", f"{gid} has {n} identical {geo[0]} elements", gid)

    # closed color shapes
    for e in shapes:
        if _layer_of(e, known) in COLOR_GROUPS and not _is_closed(e):
            add("not_closed", "color shape is not closed", e)

    # style block
    declared: set[str] = set()
    for st in root.iter(f"{{{SVG}}}style"):
        declared.update(re.findall(r"\.([A-Za-z_][\w-]*)\s*\{", st.text or ""))
    for c in sorted(used_classes - declared):
        add("style_mismatch", f"class {c} is used but not declared in <style>", c)
    for c in sorted(declared - used_classes):
        add("style_mismatch", f"class {c} is declared in <style> but never used", c)

    # simulation: Task 4 provides rosette.simulate.sim_issues(svg) -> list[str].
    # Skip only when the module itself does not exist yet; any other failure propagates.
    if simulate_check:
        try:
            sim = importlib.import_module("rosette.simulate")
        except ModuleNotFoundError as exc:
            if exc.name != "rosette.simulate":
                raise
            sim = None
        if sim is not None:
            for desc in sim.sim_issues(svg):  # AttributeError if missing: fail closed
                add("sim_unfilled", f"element left unfilled by shop logic: {desc}", str(desc))

    return issues
