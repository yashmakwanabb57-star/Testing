import re

import numpy as np
import pytest
from lxml import etree

from rosette.roles import element_masks
from tools import score
from tools.score import DEFAULT_COLORS, HEIGHT, align_agreement, compare, score_all
from tools.truth import svg_text


def _img(h, w, boxes):
    img = np.full((h, w, 3), 255, np.uint8)
    for (y0, y1, x0, x1), cat in boxes:
        img[y0:y1, x0:x1] = [int(DEFAULT_COLORS[cat][i:i + 2], 16) for i in (1, 3, 5)]
    return img


def test_align_agreement_ignores_offset_and_scale():
    designer = _img(100, 80, [((10, 60, 10, 50), 1), ((60, 90, 10, 50), 2)])
    # same picture, shifted and twice the size
    generated = _img(200, 200, [((40, 140, 60, 140), 1), ((140, 200, 60, 140), 2)])
    agreement, iou = align_agreement(designer, generated)
    assert agreement == pytest.approx(1.0)
    assert iou == pytest.approx(1.0)


def test_align_agreement_counts_wrong_category():
    designer = _img(100, 80, [((10, 60, 10, 50), 1), ((60, 90, 10, 50), 2)])
    generated = _img(100, 80, [((10, 60, 10, 50), 1), ((60, 90, 10, 50), 3)])
    agreement, _ = align_agreement(designer, generated)
    assert agreement == pytest.approx(50 / 80)


def test_compare_identical_svg_agrees_fully():
    svg = svg_text("S35-19")
    r = compare(svg, svg)
    assert r["color_agreement"] == pytest.approx(1.0)
    assert r["pixel_agreement"] == pytest.approx(1.0)


def test_compare_swapped_element_drops_by_its_area_share():
    svg = svg_text("S35-19")
    masks = element_masks(svg, HEIGHT)
    total = sum(m.sum() for _, m in masks)
    # the tail stripe reaching lowest, largest first: unoccluded, so its region is its own
    i = max(range(len(masks)), key=lambda k: (np.nonzero(masks[k][1])[0].max(), masks[k][1].sum()))
    el, m = masks[i]
    num, cat = re.match(r"st(\d+)_([1-4])$", el.get("class")).groups()
    other = 3 if cat != "3" else 1
    root = etree.fromstring(svg.encode("utf-8"))
    # same element order as element_masks: color elements outside Layer_3
    target = [e for e in root.iter() if isinstance(e.tag, str) and etree.QName(e).localname != "g"
              and re.match(r"st\d+_[1-4]$", e.get("class") or "")
              and not any(a.get("id") == "Layer_3" for a in e.iterancestors())][i]
    assert dict(target.attrib) == dict(el.attrib)
    target.set("class", f"st{num}_{other}")
    swapped = etree.tostring(root, encoding="unicode")
    r = compare(svg, swapped)
    assert r["color_agreement"] == pytest.approx(1 - m.sum() / total, abs=0.01)
    assert r["color_agreement"] < 0.99


def _discs(under: int, over: int) -> str:
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 850"><g id="Layer_1">'
            f'<circle class="st0_{under}" cx="300" cy="425" r="200"/>'
            f'<circle class="st1_{over}" cx="300" cy="425" r="180"/></g></svg>')


def test_compare_judges_covered_element_on_its_visible_part():
    # the disc is mostly covered by an overlay of its own category; only its visible
    # rim is miscolored in the generated file, and that must count as wrong
    r = compare(_discs(1, 1), _discs(3, 1))
    rim = 1 - (180 / 200) ** 2
    assert r["color_agreement"] == pytest.approx(1 - rim, abs=0.03)
    disc = r["elements"][0]
    assert (disc["category"], disc["winner"]) == (1, 3)


def test_crash_is_an_error_row_with_all_keys(monkeypatch):
    def boom(sid):
        raise ValueError("no head")
    monkeypatch.setattr(score, "score_one", boom)
    row, = score_all(["X"])
    assert set(row) == set(score.ROW_KEYS)
    assert row["error"] == "ValueError: no head" and not row["success"] and not row["flagged"]
    assert score.summary([row])["errors"] == 1


@pytest.mark.slow
def test_bar():
    rows = score_all()
    n = len(rows)
    ok = [r for r in rows if r["success"]]
    failing = [r for r in rows if not r["success"]]
    # spec s6: at least 8 of 10 succeed AND every remaining case is flagged
    silent = [r["id"] for r in failing if not r["flagged"]]
    assert len(ok) / n >= 0.8, f"{len(ok)}/{n} succeed"
    assert not silent, silent


def test_uniform_alignment_keeps_the_generated_aspect():
    # Reported only: the generated crop is scaled by the height ratio and centered,
    # not stretched to the designer's width. A generated picture half as wide keeps
    # half the width; per-axis alignment stretches it to full agreement.
    designer = _img(100, 80, [((0, 100, 0, 80), 1)])
    generated = _img(100, 40, [((0, 100, 0, 40), 1)])
    d, g, _ = score._align_uniform(score.categories(designer), score.categories(generated))
    assert d.shape == g.shape
    assert (g[:, 20:60] == 1).all() and (g[:, :20] == 0).all() and (g[:, 60:] == 0).all()
    assert align_agreement(designer, generated)[0] == pytest.approx(1.0)


def test_compare_reports_uniform_agreement_too():
    svg = svg_text("S35-19")
    r = compare(svg, svg)
    assert r["uniform_agreement"] == pytest.approx(1.0)


@pytest.mark.parametrize("sid", ["S1-15", "A2-15", "S9-19", "S18-15", "A10-15"])
def test_compare_identical_old_layout_svg_agrees_fully(sid):
    # old-layout designer files (no Layer_2_1 toggle, or color parts outside
    # Layer_1 / Layer_2_1_N) must score themselves as a perfect match, not 0
    svg = svg_text(sid)
    r = compare(svg, svg)
    assert r["color_agreement"] == pytest.approx(1.0)
    used = {int(m) for m in re.findall(r'class="st\d+_([1-4])"', svg)}
    assert {e["category"] for e in r["elements"]} == used


def _rosette(head_r: int = 130, center: str = "disc", tail_w: int = 200, cat: int = 1) -> str:
    """Synthetic rosette: one tail under a head disc with a center (disc or ring)."""
    c = ('<circle class="st2_2" cx="300" cy="190" r="40"/>' if center == "disc" else
         '<path class="st2_2" fill-rule="evenodd" d="M300 130a60 60 0 1 0 0.01 0zM300 160a30 30 0 1 0 0.01 0z"/>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 850"><g id="Layer_1">'
            f'<polygon class="st0_{cat}" points="260,300 340,300 {300 + tail_w / 2},825 {300 - tail_w / 2},825"/>'
            f'<circle class="st1_{cat}" cx="300" cy="190" r="{head_r}"/>{c}</g></svg>')


def test_shape_identical_is_one_and_ignores_color():
    s = score.shape_compare(_rosette(), _rosette(cat=3))
    assert s["shape_iou"] == pytest.approx(1.0)
    assert s["head_iou"] == pytest.approx(1.0)
    assert s["tail_iou"] == pytest.approx(1.0)
    assert s["layer_iou"] == pytest.approx(1.0)
    assert compare(_rosette(), _rosette(cat=3))["color_agreement"] < 0.5


def test_shape_wider_tails_hurt_the_tail_region_only():
    # still narrower than the head, so the bbox alignment is unchanged
    s = score.shape_compare(_rosette(), _rosette(tail_w=255))
    assert s["tail_iou"] < 0.95
    assert s["head_iou"] > 0.97
    assert s["tail_iou"] < s["shape_iou"] < 1.0


def test_shape_layer_scores_each_designer_head_element_by_role():
    # the designer's center disc is drawn as a ring: same color, other role class
    s = score.shape_compare(_rosette(), _rosette(center="ring"))
    by_role = {e["role"]: e["iou"] for e in s["layers"]}
    assert by_role["head"] == pytest.approx(1.0, abs=0.01)
    assert by_role["center"] == 0.0
    assert s["layer_iou"] < 1.0
    assert s["shape_iou"] == pytest.approx(1.0)  # same silhouette


def test_shape_ok_uses_the_whole_iou_bar():
    assert score.shape_compare(_rosette(), _rosette())["shape_ok"]
    assert not score.shape_compare(_rosette(), _rosette(head_r=60, tail_w=60))["shape_ok"]


def _ringed(cat: int = 1) -> str:
    """Synthetic rosette: tail, head disc, a ring and a center disc."""
    ring = ('<path class="st3_2" fill-rule="evenodd" '
            'd="M300 100a90 90 0 1 0 0.01 0zM300 130a60 60 0 1 0 0.01 0z"/>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 850"><g id="Layer_1">'
            f'<polygon class="st0_{cat}" points="260,300 340,300 400,825 200,825"/>'
            f'<circle class="st1_{cat}" cx="300" cy="190" r="130"/>{ring}'
            '<circle class="st2_2" cx="300" cy="190" r="40"/></g></svg>')


def test_layer_classifies_our_pieces_around_the_center_we_drew(monkeypatch):
    # The head circle `prepare` finds on our render can be off (S35-19: 9 px high, so our
    # ring and center were classed as points and scored 0). Our pieces are judged around
    # the head center our own file draws (its circles and rings), not the measured one.
    from dataclasses import replace
    from rosette import roles
    real = roles.frame_for

    def shifted(svg, height=roles.RENDER_H):
        f = real(svg, height)
        if 'class="st1_3"' in svg:  # our file
            cx, cy, r = f.head
            f = replace(f, head=(cx, cy - 9, r - 9))
        return f
    monkeypatch.setattr(roles, "frame_for", shifted)
    s = score.shape_compare(_ringed(), _ringed(cat=3))
    by_role = {e["role"]: e["iou"] for e in s["layers"]}
    assert by_role["ring"] == pytest.approx(1.0, abs=0.02)
    assert by_role["center"] == pytest.approx(1.0, abs=0.02)
