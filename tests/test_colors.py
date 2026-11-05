from pathlib import Path

import cv2
import numpy as np
import pytest

from rosette.model import ColorGroup
from rosette.vision.colors import pixel_groups
from rosette.vision.prepare import Prepared, canvas_to_px, prepare

PHOTOS = Path(__file__).resolve().parent.parent / "reference" / "photos"


def _synthetic(lab: np.ndarray) -> Prepared:
    """Prepared around a Lab image (float32, L 0-100) with a full-frame mask."""
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB)
    rgb = (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)
    h, w = rgb.shape[:2]
    mask = np.ones((h, w), bool)
    return Prepared(rgb=rgb, mask=mask, axis_x=w / 2, head=(w / 2, h / 4, w / 3),
                    tails_top_y=h / 2, bottom_y=float(h - 1), scale=h / 800.0)


def _label_at(labels: np.ndarray, x: float, y: float, r: int = 3) -> int:
    """Majority label in a (2r+1)^2 window (robust to landing on a motif edge)."""
    xi, yi = int(round(x)), int(round(y))
    win = labels[yi - r:yi + r + 1, xi - r:xi + r + 1].ravel()
    win = win[win >= 0]
    return int(np.bincount(win).argmax())


def test_shading_merges():
    h, w = 600, 400
    lab = np.zeros((h, w, 3), np.float32)
    # left: one muted red, L* ramps 30 -> 80 top to bottom (satin shine/shadow)
    lab[:, :200, 0] = np.linspace(30, 80, h)[:, None]
    lab[:, :200, 1], lab[:, :200, 2] = 35, 20
    # right: a second hue at constant lightness
    lab[:, 200:] = (55, -10, -40)
    labels, groups = pixel_groups(_synthetic(lab))
    assert len(groups) == 2
    assert all(isinstance(g, ColorGroup) and g.category is None for g in groups)
    assert labels.shape == (h, w)
    assert len(np.unique(labels[:, :200])) == 1
    assert len(np.unique(labels[:, 200:])) == 1
    assert labels[0, 0] != labels[0, -1]
    assert sum(g.area for g in groups) == pytest.approx(1.0)
    assert not any(g.pattern for g in groups)


def test_four_plus_colors_reported():
    h, w = 600, 500
    hues = [(50, 60, 40), (60, -50, 40), (45, 20, -60), (80, 0, 80), (70, 50, -20)]
    lab = np.zeros((h, w, 3), np.float32)
    for i, c in enumerate(hues):
        lab[:, i * 100:(i + 1) * 100] = c
    labels, groups = pixel_groups(_synthetic(lab))
    assert len(groups) == 5
    for i in range(5):
        assert len(np.unique(labels[:, i * 100 + 5:(i + 1) * 100 - 5])) == 1
    assert not any(g.pattern for g in groups)


@pytest.fixture(scope="module")
def s46():
    p = prepare((PHOTOS / "S46-15.png").read_bytes())
    return p, *pixel_groups(p)


@pytest.fixture(scope="module")
def s35():
    p = prepare((PHOTOS / "S35-19.png").read_bytes())
    return p, *pixel_groups(p)


# S46-15 working-pixel points chosen by inspecting the photo (row 900 crosses
# all tail stripes): left/right glitter stripe, green tail, gold tail.
S46_GLITTER = [(160, 900), (415, 900)]
S46_GREEN = [(100, 900), (460, 900)]
S46_GOLD = [(220, 900), (370, 900)]


def test_glitter_is_pattern(s46):
    p, labels, groups = s46
    glitter = {_label_at(labels, x, y) for x, y in S46_GLITTER}
    green = {_label_at(labels, x, y) for x, y in S46_GREEN}
    gold = {_label_at(labels, x, y) for x, y in S46_GOLD}
    assert len(glitter) == len(green) == len(gold) == 1
    assert len(glitter | green | gold) == 3
    assert groups[glitter.pop()].pattern is True
    assert groups[green.pop()].pattern is False
    assert groups[gold.pop()].pattern is False


# S35-19 canvas points (600x850 frame): dark purple heart stripe, lavender
# outer tail, white center tail - all at canvas row 560 in the tails.
S35_HEARTS = [(240, 560), (360, 560)]
S35_LAVENDER = [(207, 560), (393, 560)]
S35_WHITE = [(292, 560)]


def test_hearts_motif_is_pattern(s35):
    p, labels, groups = s35
    pick = lambda pts: {_label_at(labels, *canvas_to_px(p, x, y)) for x, y in pts}
    hearts, lavender, white = pick(S35_HEARTS), pick(S35_LAVENDER), pick(S35_WHITE)
    assert len(hearts) == len(lavender) == 1
    assert len(hearts | lavender | white) == 3
    heart = groups[hearts.pop()]
    assert heart.pattern is True
    assert groups[lavender.pop()].pattern is False
    assert groups[white.pop()].pattern is False
    # gold hearts are absorbed: the heart group's color stays dark purple
    r, g, b = heart.rgb
    assert b > r + 30 and b > g + 30 and max(r, g, b) < 160
    # and the gold hearts do not survive as a group of their own
    assert not any(gr.rgb[0] > gr.rgb[2] + 40 for gr in groups)


def test_deterministic(s35):
    p, labels, groups = s35
    labels2, groups2 = pixel_groups(p)
    assert np.array_equal(labels, labels2)
    assert groups == groups2


def test_fast_enough(s46):
    import time
    p = s46[0]
    t = time.perf_counter()
    pixel_groups(p)
    assert time.perf_counter() - t < 3.0


def test_single_color_is_one_group():
    lab = np.zeros((400, 300, 3), np.float32)
    lab[:] = (50, 40, 30)
    labels, groups = pixel_groups(_synthetic(lab))
    assert len(groups) == 1 and groups[0].area == pytest.approx(1.0)
    assert (labels == 0).all()
