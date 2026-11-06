import numpy as np

from rosette.model import ColorGroup
from rosette.vision.parts import detect_parts
from rosette.vision.prepare import Prepared

DARK, LIGHT = (30, 50, 160), (160, 188, 205)


def tail_photo(sparkle: bool, band_px: int = 10, glitter_label=None):
    """Head disc (label 0) and one tail: light (1) | thin alternating 0/1 bands | dark (0).
    With sparkle the middle zone is navy glitter with light sparkles in the photo."""
    rng = np.random.default_rng(1)
    h, w = 1200, 800
    yy, xx = np.mgrid[:h, :w]
    labels = np.full((h, w), -1, np.int32)
    rgb = np.full((h, w, 3), 255, np.uint8)
    head = (xx - 400) ** 2 + (yy - 300) ** 2 <= 250 ** 2
    tail = (yy > 550) & (yy < 1150) & (xx >= 250) & (xx < 550)
    labels[head] = 0
    rgb[head] = DARK
    left, mid, right = tail & (xx < 350), tail & (xx >= 350) & (xx < 430), tail & (xx >= 430)
    labels[left], rgb[left] = 1, LIGHT
    labels[right], rgb[right] = 0, DARK
    band = ((xx - 350 + (yy // 40) % 7) // band_px) % 2  # bands wander a little row to row
    labels[mid] = band[mid] if glitter_label is None else glitter_label
    if sparkle:
        rgb[mid] = (25, 35, 110)
        spark = mid & (rng.random((h, w)) < 0.3)
        rgb[spark] = LIGHT
    else:
        rgb[mid] = np.where(band[mid][:, None] == 1, LIGHT, DARK)
    p = Prepared(rgb=rgb, mask=labels >= 0, axis_x=400.0, head=(400.0, 300.0, 250.0),
                 tails_top_y=551.0, bottom_y=1149.0, scale=(1149 - 50) / 800)
    groups = [ColorGroup(0, DARK, False, 1.0, 0.6, None, 0.0), ColorGroup(1, LIGHT, False, 1.0, 0.3, None, 0.0)]
    if glitter_label is not None:
        groups.append(ColorGroup(glitter_label, (25, 35, 110), True, 12.0, 0.1, None, 0.0))
    return p, labels, groups


def stripes(parts):
    t = sorted((q for q in parts if q.role == "tail_stripe"), key=lambda q: int(q.params["stripe_index"]))
    fr = [float(v) for v in str(t[0].params["stripes"]).split(",")] if t else []
    return [q.color_group for q in t], fr


def test_textured_run_of_two_plain_tones_is_one_new_pattern_stripe():
    # A13-15B: navy glitter with light sparkles reads as thin bands of the light tail's
    # group and the royal tail's group; it is one glitter ribbon, a color of its own
    p, labels, groups = tail_photo(sparkle=True)
    parts = detect_parts(p, labels, groups)
    cg, fr = stripes(parts)
    assert len(groups) == 3 and groups[2].pattern and groups[2].id == 2
    assert cg == [1, 2, 0]
    assert fr == [__import__("pytest").approx(v, abs=0.03) for v in (100 / 300, 80 / 300, 120 / 300)]
    assert sum(groups[2].rgb) < sum(LIGHT)  # the glitter's own median color


def test_smooth_thin_bands_stay_separate_stripes():
    # the same bands in plain satin (no texture) are real stripes: nothing is merged
    p, labels, groups = tail_photo(sparkle=False, band_px=20)
    parts = detect_parts(p, labels, groups)
    cg, _ = stripes(parts)
    assert len(groups) == 2
    assert len(cg) >= 4


def test_pattern_ribbon_keeps_its_group():
    # a glitter ribbon already read as one pattern group is left as it is
    p, labels, groups = tail_photo(sparkle=True, glitter_label=2)
    parts = detect_parts(p, labels, groups)
    cg, _ = stripes(parts)
    assert len(groups) == 3 and cg == [1, 2, 0]


def test_without_groups_detection_is_unchanged():
    p, labels, groups = tail_photo(sparkle=True)
    cg, _ = stripes(detect_parts(p, labels))
    assert 2 not in cg


def test_detect_tails_returns_the_glitter_group_and_leaves_groups_alone():
    from rosette.vision.parts_common import Frame
    from rosette.vision.parts_tails import detect_tails
    p, labels, groups = tail_photo(sparkle=True)
    parts, glitter = detect_tails(p, labels, Frame.of(p), groups)
    assert len(groups) == 2  # not an output channel
    assert glitter is not None and glitter.pattern and glitter.id == 2
    assert any(q.color_group == 2 for q in parts)
    assert detect_tails(p, labels, Frame.of(p))[1] is None  # no groups, no glitter rule


def test_detecting_twice_adds_the_glitter_group_once():
    p, labels, groups = tail_photo(sparkle=True)
    a = detect_parts(p, labels, groups)
    b = detect_parts(p, labels, groups)
    assert [g.id for g in groups] == [0, 1, 2]
    assert stripes(a)[0] == stripes(b)[0] == [1, 2, 0]
