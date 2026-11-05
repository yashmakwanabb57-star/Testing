from lxml import etree

from rosette.simulate import SimConfig, simulate, sim_issues
from rosette.validate import validate
from tests.test_validate import good_svg

COLORS = {1: "#111111", 2: "#222222", 3: "#333333", 4: "#444444"}
PAT_COLORS = {**COLORS, 2: "url(x.png)"}
NS = {"s": "http://www.w3.org/2000/svg"}


def disp(res, gid):
    r = etree.fromstring(res.svg.encode())
    return r.xpath(f"//s:g[@id='{gid}']/@style", namespaces=NS)


def test_solid_mode_fills_all_visible():
    res = simulate(good_svg(), SimConfig(colors=COLORS, pattern_index=None, rose=True))
    assert res.unfilled == []
    assert res.visible_color_elements
    assert "display:none" in disp(res, "Layer_3")[0].replace(" ", "")
    assert "display:inline" in disp(res, "Layer_2_1")[0].replace(" ", "")


def test_pattern_mode_fills_all_visible():
    res = simulate(good_svg(), SimConfig(colors=PAT_COLORS, pattern_index=2, rose=True))
    assert res.unfilled == []
    assert "url(#image_st0_2_pattern)" in res.svg
    assert "display:none" in disp(res, "Layer_2_1")[0].replace(" ", "")
    r = etree.fromstring(res.svg.encode())
    vis = {r.getroottree().xpath(x)[0].get("class") for x in res.visible_color_elements}
    assert "st0_2_pattern" in vis and "st0_2" not in vis


def test_rose_off_hides_st4():
    svg = good_svg()
    off = simulate(svg, SimConfig(colors=COLORS, pattern_index=None, rose=False))
    on = simulate(svg, SimConfig(colors=COLORS, pattern_index=None, rose=True))
    r = etree.fromstring(off.svg.encode())
    st4 = r.xpath("//s:path[@class='st4']", namespaces=NS)[0]
    circ = r.xpath("//s:circle[@class='st_circel']", namespaces=NS)[0]
    assert "none" in st4.get("style") and "inline" in circ.get("style")
    r2 = etree.fromstring(on.svg.encode())
    assert "inline" in r2.xpath("//s:path[@class='st4']/@style", namespaces=NS)[0]


def test_st4_color_class_never_filled():
    svg = good_svg().replace('class="st0_1"', 'class="st4_1"')
    res = simulate(svg, SimConfig(colors=COLORS, pattern_index=None, rose=True))
    assert len(res.unfilled) == 1
    assert len(res.visible_color_elements) > 1


def test_missing_pattern_image_leaves_unfilled():
    svg = good_svg().replace('<image xlink:href="" x="0" y="0" width="100" height="100"/>', "")
    res = simulate(svg, SimConfig(colors=PAT_COLORS, pattern_index=2, rose=True))
    assert len(res.unfilled) == 1


def test_pattern_mode_does_not_fill_hidden_solid_2():
    res = simulate(good_svg(), SimConfig(colors=PAT_COLORS, pattern_index=2, rose=True))
    r = etree.fromstring(res.svg.encode())
    assert "st0_2" not in {r.getroottree().xpath(x)[0].get("class")
                           for x in res.visible_color_elements}


def test_sim_issues_clean_and_dirty():
    assert sim_issues(good_svg()) == []
    bad = good_svg().replace('class="st0_1"', 'class="st4_1"')
    assert sim_issues(bad)


def test_validate_valid_writer_output_still_clean():
    assert validate(good_svg()) == []


def test_validate_reports_sim_unfilled():
    bad = good_svg().replace('class="st0_1"', 'class="st4_1"')
    assert "sim_unfilled" in {i.code for i in validate(bad)}
