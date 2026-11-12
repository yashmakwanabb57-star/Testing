from pathlib import Path

import numpy as np
import pytest

from rosette.perception import pieces
from rosette.perception.pieces import (Piece, Pieces, grid, in_head, nms, read_cache, segment,
                                       stability, write_cache)
from rosette.perception.sam2 import local_dir
from rosette.perception.settings import DEFAULT_MODELS_DIR, Settings
from rosette.vision.prepare import Prepared

PHOTOS = Path(__file__).resolve().parents[1] / "reference" / "photos"


def _prepared(h: int, w: int, head: tuple[float, float, float], tails_top_y: float) -> Prepared:
    yy, xx = np.mgrid[:h, :w]
    cx, cy, r = head
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    mask |= (yy >= tails_top_y) & (yy < h - 5) & (np.abs(xx - cx) < r * 0.6)
    rgb = np.full((h, w, 3), 255, np.uint8)
    rgb[mask] = (200, 30, 40)
    return Prepared(rgb=rgb, mask=mask, axis_x=cx, head=head, tails_top_y=tails_top_y,
                    bottom_y=h - 6, scale=h / 800.0)


@pytest.fixture
def small_photo() -> Prepared:
    return _prepared(200, 150, (75.0, 55.0, 45.0), 100.0)


def test_grid_points_lie_inside_mask_and_circle(small_photo):  # 200 px synthetic Prepared
    pts = grid(small_photo)
    assert len(pts) > 50 and all(small_photo.mask[y, x] for x, y in pts)
    cx, cy, r = small_photo.head
    head_pts = [(x, y) for x, y in pts if y < small_photo.tails_top_y]
    assert head_pts and all((x - cx) ** 2 + (y - cy) ** 2 <= r * r for x, y in head_pts)
    assert any(y >= small_photo.tails_top_y for _x, y in pts)


def test_grid_stays_inside_the_image_when_the_head_circle_does_not():
    # a failed head fit: the circle runs past the image edges
    p = _prepared(200, 150, (10.0, 20.0, 90.0), 100.0)
    pts = grid(p)
    h, w = p.mask.shape
    assert len(pts) > 0
    assert all(0 <= x < w and 0 <= y < h and p.mask[y, x] for x, y in pts)


def test_nms_drops_near_duplicates():
    a = np.zeros((20, 20), bool); a[2:12, 2:12] = True
    b = a.copy(); b[2, 2] = False
    c = np.zeros((20, 20), bool); c[14:19, 14:19] = True
    assert nms(np.stack([a, b, c]), np.array([0.9, 0.8, 0.7])) == [0, 2]


def test_stability_of_a_sharp_mask_is_one():
    logits = np.where(np.arange(100).reshape(10, 10) < 50, 5.0, -5.0)
    assert stability(logits) == 1.0


def test_stability_of_a_soft_mask_is_below_one():
    logits = np.linspace(-3.0, 3.0, 100).reshape(10, 10)
    assert stability(logits) < 0.9


def test_a_piece_spanning_head_and_tails_is_not_in_head(small_photo):
    tails_top = int(small_photo.tails_top_y)
    stripe = np.zeros_like(small_photo.mask); stripe[60:190, 70:80] = True  # mostly below
    head_only = np.zeros_like(small_photo.mask); head_only[30:80, 50:100] = True
    assert not in_head(stripe, tails_top)
    assert in_head(head_only, tails_top)


def test_cache_round_trip(tmp_path, small_photo):
    m = small_photo.mask.copy()
    fake = Pieces([Piece(mask=m, score=0.9, stability=0.95, area=int(m.sum()),
                         bbox=(1, 2, 3, 4), in_head=True)], "fake", "cpu", 1.5, False)
    write_cache(tmp_path / "k.npz", fake)
    out = read_cache(tmp_path / "k.npz", small_photo)
    assert out.cached and out.model_id == "fake" and len(out.pieces) == 1
    assert np.array_equal(out.pieces[0].mask, m)
    assert out.pieces[0].score == pytest.approx(0.9) and out.pieces[0].stability == pytest.approx(0.95)


def test_cache_hit_never_loads_the_model(tmp_path, small_photo, monkeypatch):
    fake = Pieces([Piece(mask=small_photo.mask.copy(), score=0.9, stability=1.0, area=int(small_photo.mask.sum()),
                         bbox=(0, 0, 10, 10), in_head=True)], "fake", "cpu", 0.0, False)
    write_cache(tmp_path / "k.npz", fake)
    monkeypatch.setattr(pieces, "_run_model", lambda *a, **k: pytest.fail("model called"))
    monkeypatch.setattr(pieces.sam2, "load", lambda *a, **k: pytest.fail("model loaded"))
    for _ in range(2):
        out = segment(small_photo, Settings("cpu", "small", tmp_path), cache_dir=tmp_path, key="k")
        assert out.cached and len(out.pieces) == 1


def test_default_key_folds_the_model_and_the_grid(tmp_path, small_photo, monkeypatch):
    import hashlib
    s = Settings("cpu", "small", tmp_path)
    key = hashlib.sha256(small_photo.rgb.tobytes() + b"|facebook/sam2.1-hiera-small|24").hexdigest()
    assert pieces.default_key(small_photo, s) == key
    assert pieces.default_key(small_photo, Settings("cpu", "tiny", tmp_path)) != key
    assert pieces.default_key(small_photo, s, head_n=32) != key
    write_cache(tmp_path / f"{key}.npz", Pieces([], "fake", "cpu", 0.0, False))
    monkeypatch.setattr(pieces.sam2, "load", lambda *a, **k: pytest.fail("model loaded"))
    assert segment(small_photo, s, cache_dir=tmp_path).cached


def test_a_legacy_cache_file_of_the_same_model_is_adopted_once(tmp_path, small_photo, monkeypatch):
    import hashlib
    s = Settings("cpu", "small", tmp_path)
    legacy = tmp_path / f"{hashlib.sha256(small_photo.rgb.tobytes()).hexdigest()}.npz"
    write_cache(legacy, Pieces([], "facebook/sam2.1-hiera-small", "cpu", 0.0, False))
    monkeypatch.setattr(pieces.sam2, "load", lambda *a, **k: pytest.fail("model loaded"))
    assert segment(small_photo, s, cache_dir=tmp_path).cached
    assert not legacy.exists() and (tmp_path / f"{pieces.default_key(small_photo, s)}.npz").exists()
    # another model's legacy file is left alone and not served
    write_cache(legacy, Pieces([], "facebook/sam2.1-hiera-tiny", "cpu", 0.0, False))
    class WouldLoad(Exception):
        pass

    def would_load(*a, **k):
        raise WouldLoad

    monkeypatch.setattr(pieces.sam2, "load", would_load)
    with pytest.raises(WouldLoad):
        segment(small_photo, s, cache_dir=tmp_path, head_n=32)  # another grid: no hit, no adoption
    assert legacy.exists()


def _star_like(piece: Piece) -> bool:
    import cv2
    m = piece.mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    return hull > 0 and piece.area / hull < 0.85


@pytest.mark.slow
def test_s70_19_pieces_include_the_blocks_and_stars(tmp_path):
    if not local_dir(DEFAULT_MODELS_DIR, "small").joinpath("model.safetensors").exists():
        pytest.skip("weights not fetched: run tools/fetch_models.py --model small")
    from rosette.vision.prepare import prepare
    p = prepare((PHOTOS / "S70-19.png").read_bytes())
    out = segment(p, Settings("cpu", "small", DEFAULT_MODELS_DIR), cache_dir=tmp_path)
    assert not out.cached and out.device == "cpu"
    rgb = p.rgb.astype(int)
    blue, white_or_red = [], []
    for pc in out.pieces:
        if not pc.in_head:
            continue
        r, g, b = np.median(rgb[pc.mask], axis=0)
        if b > r + 40 and b > g + 40:
            blue.append(pc)
        elif (r > 180 and g > 180 and b > 180) or (r > 150 and g < 100 and b < 100):
            white_or_red.append(pc)
    assert len(blue) >= 6, f"{len(blue)} blue head pieces"
    assert sum(_star_like(pc) for pc in white_or_red) >= 2
    again = segment(p, Settings("cpu", "small", DEFAULT_MODELS_DIR), cache_dir=tmp_path)
    assert again.cached and len(again.pieces) == len(out.pieces)
