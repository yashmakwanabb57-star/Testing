import re

import numpy as np
import pytest
from lxml import etree

from rosette.roles import element_masks
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


@pytest.mark.slow
def test_bar():
    rows = score_all()
    n = len(rows)
    ok = [r for r in rows if r["success"]]
    failing = [r for r in rows if not r["success"]]
    assert len(ok) / n >= 0.8 or all(r["flagged"] for r in failing), \
        [r["id"] for r in failing if not r["flagged"]]
