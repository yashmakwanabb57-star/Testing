import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rosette.vision.prepare import InputError, Prepared, prepare
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
