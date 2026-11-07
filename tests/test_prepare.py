import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rosette.vision.prepare import InputError, Prepared, canvas_to_px, prepare
from tools.rasterize import rasterize

REF = Path(__file__).resolve().parent.parent / "reference"
S46 = REF / "photos" / "S46-15.png"
WORK_H = 1200


def _png(img: Image.Image, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", **kw)
    return buf.getvalue()


def _jpeg(img: Image.Image, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92, **kw)
    return buf.getvalue()


@pytest.fixture(scope="module")
def s46_img():
    return Image.open(S46).convert("RGB")


@pytest.fixture(scope="module")
def s46(s46_img):
    return prepare(S46.read_bytes())


def test_mask_excludes_white_background(s46):
    assert isinstance(s46, Prepared)
    h, w = s46.mask.shape
    assert h == WORK_H and s46.rgb.shape == (h, w, 3) and s46.rgb.dtype == np.uint8
    assert s46.mask.dtype == bool
    assert 0.25 <= s46.mask.mean() <= 0.60
    for y, x in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        assert not s46.mask[y, x]


def test_localization_fields_on_photo(s46):
    h, w = s46.mask.shape
    assert abs(s46.axis_x - w / 2) < 0.05 * w
    cx, cy, r = s46.head
    assert abs(cx - s46.axis_x) < 1e-6
    rows = np.flatnonzero(s46.mask.any(axis=1))
    top, bottom = rows[0], rows[-1]
    assert s46.bottom_y == pytest.approx(bottom, abs=1)
    assert cy - r == pytest.approx(top, abs=0.02 * h)
    assert 0.15 * h < 2 * r < 0.6 * h
    assert cy + r < s46.tails_top_y <= cy + r + 2
    assert s46.scale == pytest.approx((s46.bottom_y - top) / 800, rel=1e-6)


def test_transparent_png_uses_alpha(s46_img):
    rgb = np.asarray(s46_img).copy()
    alpha = np.where(rgb.min(axis=2) >= 254, 0, 255).astype(np.uint8)
    rgb[alpha == 0] = (0, 0, 0)  # transparent pixels are black: only alpha separates them
    data = _png(Image.fromarray(np.dstack([rgb, alpha]), "RGBA"))
    p = prepare(data)
    h, w = p.mask.shape
    assert h == WORK_H
    ref = np.asarray(Image.fromarray(alpha).resize((w, h), Image.Resampling.BILINEAR)) >= 128
    iou = (p.mask & ref).sum() / (p.mask | ref).sum()
    assert iou > 0.97
    assert not p.mask[0, 0] and not p.mask[-1, -1]
    # transparent background is composited on white in the working rgb
    assert (p.rgb[0, 0] == 255).all()


def test_normalizes_size(s46_img, s46):
    w, h = s46_img.size
    big = prepare(_png(s46_img.resize((w * 4, h * 4), Image.Resampling.BICUBIC), compress_level=1))
    small = prepare(_png(s46_img.resize((300, 600), Image.Resampling.LANCZOS)))
    for p in (big, small):
        assert p.rgb.shape[0] == WORK_H and p.mask.shape[0] == WORK_H
        for got, want in zip(p.head, s46.head):
            assert got == pytest.approx(want, rel=0.03)
        assert p.scale == pytest.approx(s46.scale, rel=0.03)


def test_exif_and_modes(s46_img):
    display = s46_img.resize((500, 1000), Image.Resampling.LANCZOS)
    stored = display.transpose(Image.Transpose.ROTATE_90)  # landscape on disk
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90 CW to display
    rotated = prepare(_jpeg(stored, exif=exif.tobytes()))
    assert rotated.rgb.shape == (WORK_H, 600, 3)
    assert rotated.head[1] < WORK_H / 2  # head on top, i.e. auto-rotated upright
    gray = prepare(_png(display.convert("L")))
    assert gray.rgb.shape == (WORK_H, 600, 3) and gray.rgb.dtype == np.uint8
    cmyk = prepare(_jpeg(display.convert("CMYK")))
    assert cmyk.rgb.shape == (WORK_H, 600, 3) and cmyk.rgb.dtype == np.uint8
    palette = prepare(_png(display.convert("P")))
    assert palette.rgb.shape == (WORK_H, 600, 3)


def test_head_on_synthetic():
    svg = (REF / "svg" / "S35-19.svg").read_text(encoding="utf-8")
    raster = rasterize(svg, WORK_H, None)
    p = prepare(_png(Image.fromarray(raster)))
    k = WORK_H / 850  # viewBox 0 0 600 850 rendered at height 1200
    want_cx, want_cy = 300 * k, 167.372 * k
    cx, cy, r = p.head
    assert abs(cx - want_cx) <= 0.02 * WORK_H
    assert abs(cy - want_cy) <= 0.02 * WORK_H
    # head top is the star's top point (y=25), so r is about 142.4 canvas units
    assert r == pytest.approx(142.372 * k, rel=0.05)


def test_no_rosette_raises():
    blank = Image.new("RGB", (600, 1200), "white")
    with pytest.raises(InputError, match="^No rosette found in the image.$"):
        prepare(_png(blank))
    rng = np.random.default_rng(0)
    noise = Image.fromarray(rng.integers(0, 256, (1200, 600, 3), dtype=np.uint8))
    with pytest.raises(InputError, match="^No rosette found in the image.$"):
        prepare(_png(noise))


def test_corrupt_bytes_raises():
    for data in (b"", b"definitely not an image", _png(Image.new("RGB", (50, 100), "red"))[:40]):
        with pytest.raises(InputError, match="^Could not read the image file.$"):
            prepare(data)


def test_canvas_to_px(s46):
    top = float(np.flatnonzero(s46.mask.any(axis=1))[0])
    assert canvas_to_px(s46, 300, 25) == pytest.approx((s46.axis_x, top))
    assert canvas_to_px(s46, 300, 825)[1] == pytest.approx(s46.bottom_y)
    x, y = canvas_to_px(s46, 400, 425)
    assert x == pytest.approx(s46.axis_x + 100 * s46.scale)
    assert y == pytest.approx(top + 400 * s46.scale)


# ---------------------------------------------------------------- H10: head circle fit

def _synthetic(head: str, *, cy: float = 230.0, r: float = 180.0, tails_top: float = 330.0,
               tails_half_top: float = 110.0, tails_half_bottom: float = 170.0,
               k: int = 12, inner: float = 0.55, loop_to: float = 0.0) -> np.ndarray:
    """RGB 600x1200: straight tails (a trapezoid from tails_top to the bottom) under a
    head centered on x=300: 'disc', 'star' (k tips at r, valleys at inner * r) or
    'loop' (a disc with one narrow loop reaching loop_to * r at -60 degrees)."""
    import cv2
    img = np.full((1200, 600, 3), 255, np.uint8)
    tails = np.array([[300 - tails_half_top, tails_top], [300 + tails_half_top, tails_top],
                      [300 + tails_half_bottom, 1150], [300 - tails_half_bottom, 1150]], np.int32)
    cv2.fillPoly(img, [tails], (40, 60, 170))
    if head == "star":
        a = np.radians(-90 + np.arange(2 * k) * 180.0 / k)
        rad = np.where(np.arange(2 * k) % 2 == 0, r, inner * r)
        pts = np.stack([300 + rad * np.cos(a), cy + rad * np.sin(a)], 1)
        cv2.fillPoly(img, [np.rint(pts).astype(np.int32)], (200, 40, 50))
        cv2.circle(img, (300, int(cy)), int(0.3 * r), (240, 200, 40), -1)
    else:
        cv2.circle(img, (300, int(cy)), int(r), (200, 40, 50), -1)
        cv2.circle(img, (300, int(cy)), int(0.3 * r), (240, 200, 40), -1)
        if head == "loop":
            t = np.radians(-60)
            mid = (300 + 0.5 * (1 + loop_to) * r * np.cos(t), cy + 0.5 * (1 + loop_to) * r * np.sin(t))
            cv2.ellipse(img, (int(mid[0]), int(mid[1])), (int(0.5 * (loop_to - 0.6) * r), int(0.12 * r)),
                        -60, 0, 360, (200, 40, 50), -1)
    return img


def _prep(img: np.ndarray):
    return prepare(_png(Image.fromarray(img)))


def _mask_bounds(p):
    rows = np.flatnonzero(p.mask.any(axis=1))
    return int(rows[0]), int(rows[-1])


def test_tails_start_on_disc_over_straight_tails():
    from rosette.vision.prepare import _tails_start
    p = _prep(_synthetic("disc"))
    top, bottom = _mask_bounds(p)
    t = _tails_start(p.mask, top, bottom)
    # the tails' side edges leave the head outline where the circle crosses x = 300 +- tails half-width
    want = 230 + np.sqrt(180 ** 2 - 110 ** 2)
    assert t == pytest.approx(want, abs=12)


def test_tails_start_is_under_star_arms_that_stick_out_over_the_tails():
    from rosette.vision.prepare import _tails_start
    p = _prep(_synthetic("star", tails_half_top=60))
    top, bottom = _mask_bounds(p)
    t = _tails_start(p.mask, top, bottom)
    # the lowest arms beside the tails (tips at 165 and 195 degrees from the top) end near
    # cy + r cos(30); the tails start below them, not in the arms
    assert t >= 230 + 180 * np.cos(np.radians(30)) - 12


def test_fit_disc_head_over_straight_tails():
    from rosette.vision.prepare import _fit_head, _tails_start
    p = _prep(_synthetic("disc"))
    top, bottom = _mask_bounds(p)
    cy, r, _ = _fit_head(p.mask, p.axis_x, top, _tails_start(p.mask, top, bottom))
    assert cy == pytest.approx(230, abs=4) and r == pytest.approx(180, abs=4)


def test_fit_star_with_deep_concavities_is_the_circle_around_the_star():
    from rosette.vision.prepare import _fit_head, _tails_start
    for k in (6, 12):
        p = _prep(_synthetic("star", k=k, inner=0.5, tails_half_top=60))
        top, bottom = _mask_bounds(p)
        cy, r, _ = _fit_head(p.mask, p.axis_x, top, _tails_start(p.mask, top, bottom))
        assert cy == pytest.approx(230, abs=6), k
        assert r == pytest.approx(180, abs=6), k


def test_fit_radius_is_not_inflated_by_one_long_loop():
    from rosette.vision.prepare import _fit_head, _tails_start
    p = _prep(_synthetic("loop", r=170, loop_to=1.5))
    top, bottom = _mask_bounds(p)
    cy, r, _ = _fit_head(p.mask, p.axis_x, top, _tails_start(p.mask, top, bottom))
    assert r == pytest.approx(170, abs=5)
    assert cy == pytest.approx(230, abs=5)


def test_prepare_takes_the_fit_on_a_star_whose_arms_pass_for_tails():
    # a deep 12-point star with long arms over narrow tails: the head circle is the
    # circle around the star (center on the badge), and the tails start under it
    p = _prep(_synthetic("star", k=12, inner=0.45, tails_half_top=50, tails_half_bottom=120))
    cx, cy, r = p.head
    assert cx == pytest.approx(300, abs=3)
    assert cy == pytest.approx(230, abs=10) and r == pytest.approx(180, abs=10)
    assert p.tails_top_y >= cy + r


def test_prepare_keeps_the_symmetry_circle_when_the_fit_agrees():
    from rosette.vision.prepare import _head
    p = _prep(_synthetic("disc"))
    top, bottom = _mask_bounds(p)
    assert p.head == pytest.approx(_head(p.mask, p.axis_x, top, bottom))


def test_prepare_takes_the_fit_when_the_symmetry_circle_is_far_too_small(monkeypatch):
    # the symmetry score locked onto an inner structure (S29-15: the star's lower arms
    # taken for the tails top): a circle well inside the head is replaced by the fit
    import rosette.vision.prepare as prep
    img = _synthetic("star", k=12, inner=0.5, tails_half_top=60)
    monkeypatch.setattr(prep, "_head", lambda m, ax, top, bottom: (ax, top + 120.0, 120.0))
    p = _prep(img)
    cx, cy, r = p.head
    assert cy == pytest.approx(230, abs=6) and r == pytest.approx(180, abs=6)
    assert p.tails_top_y == pytest.approx(np.floor(cy + r) + 1)
    # a symmetry circle the fit agrees with (radius within FIT_TAKEOVER) is kept
    monkeypatch.setattr(prep, "_head", lambda m, ax, top, bottom: (ax, top + 160.0, 160.0))
    assert _prep(img).head == pytest.approx((p.axis_x, _mask_bounds(p)[0] + 160.0, 160.0))


def test_prepare_does_not_take_a_fit_with_little_coverage(monkeypatch):
    # a vote that collects little of the outline (sparse tips, a cut through a star's
    # arms) is not trusted over the symmetry circle, however large its radius
    import rosette.vision.prepare as prep
    img = _synthetic("disc")
    monkeypatch.setattr(prep, "_head", lambda m, ax, top, bottom: (ax, top + 120.0, 120.0))
    monkeypatch.setattr(prep, "_fit_head", lambda m, ax, top, tails: (300.0, 260.0, prep.FIT_MIN_COVERAGE - 0.01))
    p = _prep(img)
    assert p.head[2] == pytest.approx(120.0)
    monkeypatch.setattr(prep, "_fit_head", lambda m, ax, top, tails: (300.0, 260.0, prep.FIT_MIN_COVERAGE))
    assert _prep(img).head[2] == pytest.approx(260.0)


def test_fit_coverage_of_a_disc_cut_by_the_tails():
    from rosette.vision.prepare import _fit_head, _tails_start
    p = _prep(_synthetic("disc"))
    top, bottom = _mask_bounds(p)
    assert _fit_head(p.mask, p.axis_x, top, _tails_start(p.mask, top, bottom))[2] > 0.6


def test_prepared_exposes_the_fit_ratio_and_coverage(monkeypatch):
    import rosette.vision.prepare as prep
    img = _synthetic("disc")
    p = _prep(img)
    assert p.fit_ratio == pytest.approx(1.0, abs=0.05) and p.fit_coverage > 0.6
    monkeypatch.setattr(prep, "_head", lambda m, ax, top, bottom: (ax, top + 160.0, 160.0))
    q = _prep(img)  # fit 180 against symmetry 160: 1.125, kept (no takeover)
    assert q.fit_ratio == pytest.approx(180 / 160, abs=0.03) and q.head[2] == pytest.approx(160.0)
