from collections import Counter

from lxml import etree

from rosette.roles import category, classify_roles
from tests.helpers import svg_text, truth_from_svg

ROLES = {"head", "ring", "points", "center", "tail_stripe", "unknown"}


def _by(svg_id):
    return classify_roles(svg_text(svg_id))


def test_every_solid_color_element_classified_once():
    items = _by("S35-19")
    # S35-19: 21 st0_1 + 8 st1_2 + 8 st2_3 outside Layer_3 (Layer_3 twins excluded)
    assert len(items) == 37
    assert len({id(el) for el, _ in items}) == 37
    for el, role in items:
        assert isinstance(el, etree._Element)
        assert role in ROLES
        assert not any(a.get("id") == "Layer_3" for a in el.iterancestors())


def test_s35_roles():
    items = _by("S35-19")
    roles = Counter(role for _, role in items)
    # 5 tail polygons (2 red, 2 blue, 1 yellow), one red pinked ring, one yellow
    # center circle, 6 blue bows and 6 yellow diamonds
    assert roles["tail_stripe"] == 5
    assert roles["ring"] == 1 and roles["center"] == 1
    assert roles["head"] == 0
    circle = [role for el, role in items if etree.QName(el).localname == "circle"]
    assert circle == ["center"]
    bows = [role for el, role in items if category(el) == 2 and role != "tail_stripe"]
    assert bows == ["points"] * 6


def test_s46_head_and_center():
    items = _by("S46-15")
    heads = [el for el, role in items if role == "head"]
    centers = [el for el, role in items if role == "center"]
    assert len(heads) == 1 and category(heads[0]) == 1
    assert len(centers) == 1 and etree.QName(centers[0]).localname == "ellipse"


def test_head_merged_with_tail_path_is_split():
    # S13-15 draws the head rim and the outer tails as one path: the rim still counts
    t = truth_from_svg("S13-15")
    assert t["pinked"] is True


def test_truth_shape():
    t = truth_from_svg("S35-19")
    assert set(t) == {"points", "tails", "rings", "pinked"}
    assert t["pinked"] is False  # no head disc: the head is made of bows


def test_drawn_frame_takes_the_center_of_the_files_rings_not_of_loose_discs():
    from rosette.roles import PAD_FRAC, drawn_frame
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 850"><g id="Layer_1">'
           '<polygon class="st0_1" points="260,300 340,300 400,825 200,825"/>'
           '<circle class="st1_1" cx="300" cy="190" r="130"/>'
           '<path class="st3_2" fill-rule="evenodd" d="M390,190a90,90 0 1 0 -180,0a90,90 0 1 0 180,0z"/>'
           '<circle class="st2_2" cx="300" cy="190" r="40"/>'
           '<circle class="st1_1" cx="420" cy="90" r="20"/></g></svg>')  # an off-center unknown disc
    f = drawn_frame(svg, 850)
    pad = round(PAD_FRAC * 850)
    assert abs(f.head[0] - (pad + 300)) < 1e-6 and abs(f.head[1] - (pad + 190)) < 1e-6
