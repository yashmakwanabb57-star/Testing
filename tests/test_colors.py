from pathlib import Path

import cv2
import numpy as np
import pytest

from rosette.model import ColorGroup
from rosette.vision.colors import mixed_texture_groups, pixel_groups
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


# S46-15 canvas points (600x850 frame), chosen by inspecting the photo at working
# pixel row 900, which crosses all tail stripes (px x 160/415 glitter, 100/460
# green, 220/370 gold), mapped back to canvas once.
S46_GLITTER = [(212, 633), (388, 633)]
S46_GREEN = [(171, 633), (419, 633)]
S46_GOLD = [(254, 633), (357, 633)]


def test_glitter_is_pattern(s46):
    p, labels, groups = s46
    pick = lambda pts: {_label_at(labels, *canvas_to_px(p, x, y)) for x, y in pts}
    glitter, green, gold = pick(S46_GLITTER), pick(S46_GREEN), pick(S46_GOLD)
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


def _noisy(p: Prepared, sigma: float, seed: int = 1) -> Prepared:
    rng = np.random.default_rng(seed)
    rgb = np.clip(p.rgb.astype(np.float64) + rng.normal(0, sigma, p.rgb.shape), 0, 255).astype(np.uint8)
    return Prepared(rgb=rgb, mask=p.mask, axis_x=p.axis_x, head=p.head, tails_top_y=p.tails_top_y,
                    bottom_y=p.bottom_y, scale=p.scale)


def test_noisy_shaded_single_color_is_one_group():
    lab = np.zeros((600, 300, 3), np.float32)
    lab[..., 0] = np.linspace(30, 80, 600)[:, None]  # satin shine/shadow
    lab[..., 1], lab[..., 2] = 35, 20
    labels, groups = pixel_groups(_noisy(_synthetic(lab), sigma=3.0))
    assert len(groups) == 1
    assert (labels == 0).all()
    assert groups[0].pattern is False


def test_noisy_two_colors_still_two_groups():
    lab = np.zeros((600, 400, 3), np.float32)
    lab[:, :200, 0] = np.linspace(30, 80, 600)[:, None]
    lab[:, :200, 1], lab[:, :200, 2] = 35, 20
    lab[:, 200:] = (55, -10, -40)
    _, groups = pixel_groups(_noisy(_synthetic(lab), sigma=3.0))
    assert len(groups) == 2


# S58-17: block of white print ("Bomholt Intercoiffure ...") on the black centre
# tail, canvas x 268..327, y 460..683 (working px x 260..345, y 640..960).
S58_TEXT_BOX = ((268, 460), (327, 683))


def test_tail_text_absorbed_and_not_pattern():
    p = prepare((PHOTOS / "S58-17.png").read_bytes())
    labels, groups = pixel_groups(p)
    (x0, y0), (x1, y1) = (canvas_to_px(p, *c) for c in S58_TEXT_BOX)
    box = labels[int(y0):int(y1), int(x0):int(x1)]
    lum = p.rgb[int(y0):int(y1), int(x0):int(x1)].mean(axis=2)
    assert (lum > 150).mean() > 0.05  # the box really holds light print
    black = int(np.bincount(box.ravel()).argmax())
    assert max(groups[black].rgb) < 60
    assert (box == black).mean() > 0.97  # print absorbed into the black ribbon
    assert groups[black].pattern is False


def test_max_groups_bound(s46):
    p = s46[0]
    labels, groups = pixel_groups(p, max_groups=2)
    assert 1 <= len(groups) <= 2
    assert set(np.unique(labels[p.mask])) == set(range(len(groups)))


def test_tiny_mask_does_not_crash():
    rgb = np.full((120, 60, 3), 255, np.uint8)
    mask = np.zeros((120, 60), bool)
    mask[50:58, 20:28] = True
    rgb[50:58, 20:28] = (200, 30, 40)
    rgb[53:55, 23:25] = (20, 40, 200)  # enclosed blob: exercises print/motif absorption
    p = Prepared(rgb=rgb, mask=mask, axis_x=24.0, head=(24.0, 52.0, 3.0), tails_top_y=0.0,
                 bottom_y=57.0, scale=0.01)
    labels, groups = pixel_groups(p)
    assert (labels[~mask] == -1).all() and (labels[mask] >= 0).all()
    assert 1 <= len(groups) <= 2
    assert sum(g.area for g in groups) == pytest.approx(1.0)


def _row_labels(sid: str, fracs: tuple[float, ...]) -> list[int]:
    """Labels across the tails at 80 % of their height, at shares of the tail width."""
    p = prepare((PHOTOS / f"{sid}.png").read_bytes())
    labels, _ = pixel_groups(p)
    y = int(p.tails_top_y + 0.8 * (p.bottom_y - p.tails_top_y))
    xs = np.flatnonzero(p.mask[y])
    s, e = xs.min(), xs.max()
    return [_label_at(labels, s + f * (e - s), y) for f in fracs]


@pytest.mark.parametrize("sid", ["S10-19", "A15-17"])
def test_similar_hue_stripes_stay_apart(sid):
    # S10-19: navy outer stripe beside a lighter navy-blue stripe, then white;
    # A15-17: black outer, dark orange, light orange middle. k-means' silhouette
    # prefers the coarse split and merged the two similar hues into one group.
    outer, second, middle = _row_labels(sid, (0.06, 0.24, 0.5))
    assert len({outer, second, middle}) == 3


def test_partly_patterned_plain_group_reported():
    # S2-15B: white satin and a white-on-blue snowflake print merged into one plain
    # group; on the tails almost half of it has patterned texture
    p = prepare((PHOTOS / "S2-15B.png").read_bytes())
    labels, groups = pixel_groups(p)
    mixed = mixed_texture_groups(p, labels, groups)
    assert len(mixed) == 1 and not groups[mixed[0]].pattern
    assert min(groups[mixed[0]].rgb) > 150  # the light (white + snowflake) group


def test_clean_groups_not_reported_as_partly_patterned(s35):
    p, labels, groups = s35
    assert mixed_texture_groups(p, labels, groups) == []


@pytest.mark.parametrize("sid", ["S80-22", "S56-17B"])
def test_similar_stripes_apart_across_a_third_stripe(sid):
    # S80-22: light pink outer stripes and the burgundy middle, with pink glitter
    # stripes between them; S56-17B likewise. The two satin colors never touch, so
    # comparing only neighbouring pixels saw no stripe edges and kept them merged.
    outer, middle = _row_labels(sid, (0.04, 0.5))
    assert outer != middle


def test_star_tips_in_a_ring_are_not_absorbed_as_motifs():
    # S40-15: six black glitter star tips (740-1250 units^2 each) between the gold
    # points; about half are smaller than a motif blob and were absorbed into the
    # orange pleats, so the points layer could not be found.
    p = prepare((PHOTOS / "S40-15.png").read_bytes())
    labels, groups = pixel_groups(p)
    dark = min(groups, key=lambda g: sum(g.rgb)).id
    cx, cy, r = p.head
    n, cc, stats, cen = cv2.connectedComponentsWithStats((labels == dark).astype(np.uint8), connectivity=8)
    d = np.hypot(cen[1:, 0] - cx, cen[1:, 1] - cy)
    a = stats[1:, cv2.CC_STAT_AREA] / p.scale ** 2
    tips = ((d > 0.4 * r) & (d < r) & (a > 500)).sum()
    assert tips >= 6


def test_stripe_changes_do_not_cross_the_background_between_tails():
    # Two tails of one satin, the left lit and the right shaded, split by 2-means into
    # halves 0 and 1: a change seen across the background gap between the tails is
    # not a stripe edge (else shading alone could pass STRIPE_MIN_HV). A change across
    # another group's stripe (glitter, still inside the mask) is.
    from rosette.vision.colors import _changes
    half = np.full((4, 30), -1, np.int8)
    half[:, 0:10] = 0
    half[:, 20:30] = 1
    gap = np.ones((4, 30), bool)
    gap[:, 10:20] = False  # background between two tails
    assert _changes(half, gap) == 0
    glitter = np.ones((4, 30), bool)  # a glitter stripe of another group in between
    assert _changes(half, glitter) == 4
