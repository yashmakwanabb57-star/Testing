import copy
from pathlib import Path

import pytest
from lxml import etree

from rosette.shapes import Shape
from rosette.validate import Issue, validate
from rosette.writer import write_svg

NS = {"s": "http://www.w3.org/2000/svg"}
REF = Path(__file__).resolve().parent.parent / "reference" / "svg" / "S46-15.svg"


def sh(cat, d="M0,0L1,0L1,1z", **kw):
    return Shape(tag="path", attrs={"d": d}, category=cat, **kw)


def mixed():
    return [
        sh(1, role="a", z=0),
        sh(2, "M10,10L20,10L20,20z", role="b", z=1),
        sh(3, "M30,30L40,30L40,40z", role="c", z=2),
        sh(4, "M50,50L60,50L60,60z", role="d", z=3),
        Shape(tag="path", attrs={"d": "M0,0L5,5"}, category=None, kind="outline", z=4),
        Shape(tag="path", attrs={"d": "M0,0L9,9"}, category=None, kind="rose", z=5),
        Shape(tag="circle", attrs={"cx": "1", "cy": "1", "r": "2"}, category=None,
              kind="circel", z=6),
    ]


def good_svg():
    return write_svg(mixed())


def root_of(svg=None):
    return etree.fromstring((svg or good_svg()).encode())


def gid(root, i):
    return root.xpath(f"//s:g[@id='{i}']", namespaces=NS)[0]


def codes(root):
    return {i.code for i in validate(etree.tostring(root, encoding="unicode"))}


def test_writer_output_passes():
    assert validate(good_svg()) == []


def test_issue_fields():
    issues = validate("<svg")
    assert isinstance(issues[0], Issue)
    assert issues[0].code == "xml"


def test_xml_malformed():
    assert [i.code for i in validate("<svg><g></svg>")] == ["xml"]


def test_root_wrong_viewbox():
    r = root_of()
    r.set("viewBox", "0 0 100 100")
    assert "root" in codes(r)


def test_root_has_id():
    r = root_of()
    r.set("id", "Layer_1")
    assert "root" in codes(r)


def test_missing_group():
    r = root_of()
    g = gid(r, "Layer_6")
    g.getparent().remove(g)
    assert "missing_group" in codes(r)


def test_missing_group_unpaired_4():
    r = root_of()
    g = gid(r, "Layer_2_1_4")
    g.getparent().remove(g)
    assert "missing_group" in codes(r)


def test_bad_parent():
    r = root_of()
    g = gid(r, "Layer_2_1_3")
    g.getparent().remove(g)
    gid(r, "Layer_3").append(g)
    assert "bad_parent" in codes(r)


def test_dup_id():
    r = root_of()
    gid(r, "Layer_6").set("id", "Layer_1")
    assert "dup_id" in codes(r)


def test_class_layer():
    r = root_of()
    el = gid(r, "Layer_1")[0]
    gid(r, "Layer_2_1_3").append(el)
    assert "class_layer" in codes(r)


def test_class_layer_st4_outside_rose():
    r = root_of()
    el = gid(r, "Layer_rose")[0]
    gid(r, "Layer_6").append(el)
    assert "class_layer" in codes(r)


def test_class_layer_circel_outside_rose():
    r = root_of()
    el = gid(r, "Layer_rose").xpath("./s:circle", namespaces=NS)[0]
    gid(r, "Layer_6").append(el)
    assert "class_layer" in codes(r)


def test_class_layer_st6_outside_outline():
    r = root_of()
    el = gid(r, "Layer_6")[0]
    gid(r, "Layer_rose").append(el)
    assert "class_layer" in codes(r)


def test_class_number():
    r = root_of()
    gid(r, "Layer_1")[0].set("class", "st4_1")
    assert "class_number" in codes(r)


def test_pattern_def_missing():
    r = root_of()
    for p in r.xpath("//s:pattern", namespaces=NS):
        p.getparent().remove(p)
    assert "pattern_def_missing" in codes(r)


def test_pattern_def_unused():
    r = root_of()
    p = r.xpath("//s:pattern", namespaces=NS)[0]
    extra = copy.deepcopy(p)
    extra.set("id", "image_st5_2_pattern")
    p.getparent().append(extra)
    assert "pattern_def_unused" in codes(r)


def test_toggle_mismatch():
    r = root_of()
    g = gid(r, "Layer_2_1_3")
    g.remove(g[0])
    assert "toggle_mismatch" in codes(r)


def test_toggle_mismatch_geometry_differs():
    r = root_of()
    gid(r, "Layer_2_1_2")[0].set("d", "M0,0L99,0L99,99z")
    assert "toggle_mismatch" in codes(r)


def test_dup_geometry():
    r = root_of()
    g = gid(r, "Layer_1")
    g.append(copy.deepcopy(g[0]))
    assert "dup_geometry" in codes(r)


def test_not_closed_path():
    r = root_of()
    for gname in ("Layer_3_3", "Layer_2_1_3"):
        gid(r, gname)[0].set("d", "M30,30L40,30L40,40")
    assert "not_closed" in codes(r)


def test_not_closed_polyline():
    r = root_of()
    el = gid(r, "Layer_1")[0]
    el.tag = "{http://www.w3.org/2000/svg}polyline"
    el.attrib.pop("d")
    el.set("points", "0,0 1,0 1,1")
    assert "not_closed" in codes(r)


def test_polygon_and_circle_are_closed():
    r = root_of()
    el = gid(r, "Layer_1")[0]
    el.tag = "{http://www.w3.org/2000/svg}polygon"
    el.attrib.pop("d")
    el.set("points", "0,0 1,0 1,1")
    assert "not_closed" not in codes(r)


def test_open_outline_is_fine():
    assert validate(good_svg()) == []  # outline "M0,0L5,5" is open by design


def test_style_mismatch_undeclared():
    r = root_of()
    r.xpath("//s:style", namespaces=NS)[0].text = ".st0_1{fill:none;}"
    assert "style_mismatch" in codes(r)


def test_style_mismatch_unused():
    r = root_of()
    st = r.xpath("//s:style", namespaces=NS)[0]
    st.text += "\n.st5_3{fill:none;}\n"
    assert "style_mismatch" in codes(r)


def test_simulate_flag_off_and_missing_module_ok():
    assert validate(good_svg(), simulate_check=False) == []
    assert validate(good_svg(), simulate_check=True) == []


@pytest.mark.skipif(not REF.exists(), reason="reference not present")
def test_s46_reference_structure():
    issues = validate(REF.read_text(encoding="utf-8"))
    extra = [i for i in issues if i.code not in {"root", "style_mismatch", "not_closed"}]
    assert extra == [], [(i.code, i.element, i.message) for i in extra]
