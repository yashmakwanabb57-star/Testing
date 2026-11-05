import re

from lxml import etree

from rosette.shapes import Shape, fmt
from rosette.writer import write_svg

NS = {"s": "http://www.w3.org/2000/svg"}
BASE = ["Layer_1", "Layer_3", "Layer_3_2", "Layer_3_3", "Layer_2_1",
        "Layer_2_1_2", "Layer_2_1_3", "Layer_6", "Layer_rose"]
ORDER = ["Layer_1", "Layer_3", "Layer_3_2", "Layer_3_3", "Layer_3_4", "Layer_2_1",
         "Layer_2_1_2", "Layer_2_1_3", "Layer_2_1_4", "Layer_6", "Layer_rose"]


def parse(shapes):
    return etree.fromstring(write_svg(shapes).encode())


def gid(root, i):
    return root.xpath(f"//s:g[@id='{i}']", namespaces=NS)[0]


def sh(cat, d="M0,0L1,1z", **kw):
    return Shape(tag="path", attrs={"d": d}, category=cat, **kw)


def classes(g):
    return [e.get("class") for e in g.xpath("./s:*[@class]", namespaces=NS)]


def test_structure_always_has_required_groups():
    root = parse([])
    ids = [g.get("id") for g in root.xpath("//s:g", namespaces=NS)]
    assert ids == BASE
    assert root.get("viewBox") == "0 0 600 850"
    assert root.get("id") is None
    assert root.get("version") == "1.1"
    assert root.get("x") == "0px"


def test_color2_pattern_duplicated():
    root = parse([sh(2, d="M1,2L3,4z", pattern=True, role="a")])
    pat = classes(gid(root, "Layer_3_2"))
    assert re.fullmatch(r"st\d_2_pattern", pat[0]) and len(pat) == 1
    n = pat[0][2]
    solid = gid(root, "Layer_2_1_2").xpath("./s:path", namespaces=NS)
    assert [p.get("class") for p in solid] == [f"st{n}_2"]
    assert solid[0].get("d") == "M1,2L3,4z"
    p = gid(root, "Layer_3_2").xpath("./s:defs/s:pattern", namespaces=NS)
    assert [x.get("id") for x in p] == [f"image_st{n}_2_pattern"]
    assert gid(root, "Layer_3_2")[0].tag.endswith("defs")


def test_color2_solid_only_still_duplicated():
    root = parse([sh(2, pattern=False, role="a")])
    assert len(classes(gid(root, "Layer_3_2"))) == 1
    assert len(classes(gid(root, "Layer_2_1_2"))) == 1
    assert len(root.xpath("//s:pattern", namespaces=NS)) == 1


def test_color3_in_both_toggles():
    root = parse([sh(3, role="a")])
    assert len(classes(gid(root, "Layer_3_3"))) == 1
    assert len(classes(gid(root, "Layer_2_1_3"))) == 1
    assert classes(gid(root, "Layer_3_3"))[0].endswith("_3")


def test_color4_groups_only_when_used():
    root = parse([sh(1), sh(2), sh(3)])
    ids = [g.get("id") for g in root.xpath("//s:g", namespaces=NS)]
    assert "Layer_3_4" not in ids and "Layer_2_1_4" not in ids
    root = parse([sh(4, role="a")])
    ids = [g.get("id") for g in root.xpath("//s:g", namespaces=NS)]
    assert ids == ORDER
    assert classes(gid(root, "Layer_3_4"))[0].endswith("_4")
    assert classes(gid(root, "Layer_2_1_4"))[0].endswith("_4")


def test_class_numbers_allowed():
    root = parse([sh(1, role=f"r{i}") for i in range(10)])
    ns = {re.fullmatch(r"st(\d)_1", c).group(1) for c in classes(gid(root, "Layer_1"))}
    assert ns <= set("01235") and len(ns) == 5
    # same role -> same N
    root = parse([sh(1, role="x"), sh(1, role="y"), sh(1, role="x")])
    c = classes(gid(root, "Layer_1"))
    assert c[0] == c[2] and c[0] != c[1]


def test_z_order_preserved():
    shapes = [sh(1, d=f"M{z}", z=z, role="a") for z in (5, 1, 3)]
    root = parse(shapes)
    assert [p.get("d") for p in gid(root, "Layer_1")] == ["M1", "M3", "M5"]


def test_outline_and_rose_kinds():
    root = parse([
        Shape("path", {"d": "M0"}, None, kind="outline"),
        Shape("path", {"d": "M1"}, None, kind="rose"),
        Shape("circle", {"r": "1"}, None, kind="circel"),
    ])
    assert classes(gid(root, "Layer_6")) == ["st6"]
    assert classes(gid(root, "Layer_rose")) == ["st4", "st_circel"]
    css = root.xpath("//s:style", namespaces=NS)[0].text
    assert ".st4{fill:none;stroke:#000000;stroke-linecap:round;stroke-miterlimit:10;}" in css
    assert ".st6{fill:none;stroke:#000000;stroke-miterlimit:10;}" in css
    assert ".st_circel{fill:none;stroke:#000000;stroke-miterlimit:10;}" in css


def test_style_block_matches_used_classes():
    root = parse([sh(1, role="a"), sh(2, role="b"), sh(3, role="c"), sh(4, role="d")])
    css = root.xpath("//s:style", namespaces=NS)[0].text
    declared = set(re.findall(r"\.([A-Za-z0-9_]+)\{", css))
    used = {c for e in root.xpath("//*[@class]") for c in e.get("class").split()}
    assert declared == used
    assert all(".%s{fill:none;}" % c in css for c in used)
    assert root.xpath("//s:style", namespaces=NS)[0].get("type") == "text/css"


def test_pattern_def_exact_and_unused_absent():
    root = parse([sh(1, role="a")])
    assert not root.xpath("//s:pattern", namespaces=NS)
    root = parse([sh(2, role="a")])
    p = root.xpath("//s:pattern", namespaces=NS)[0]
    assert dict(p.attrib) == {"id": p.get("id"), "patternUnits": "userSpaceOnUse",
                              "width": "100", "height": "100"}
    img = p[0]
    assert img.get("{http://www.w3.org/1999/xlink}href") == ""
    assert (img.get("x"), img.get("y"), img.get("width")) == ("0", "0", "100")


def test_no_metadata():
    out = write_svg([sh(1), sh(2), Shape("path", {"d": "M0"}, None, kind="rose")])
    assert "<!--" not in out and "<metadata" not in out and "data-" not in out


def test_fmt():
    assert fmt(1.0) == "1"
    assert fmt(1.23456) == "1.235"
    assert fmt(0.5) == "0.5"
    assert fmt(-0.0001) == "0"
    assert fmt(10.100) == "10.1"
