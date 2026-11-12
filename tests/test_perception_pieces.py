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
    assert nms([a, b, c], np.array([0.9, 0.8, 0.7])) == [0, 2]  # a plain list, as _filtered passes


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
    tag = f"|facebook/sam2.1-hiera-small|24|v{pieces.PIECES_VERSION}".encode()
    key = hashlib.sha256(small_photo.rgb.tobytes() + tag).hexdigest()
    assert pieces.default_key(small_photo, s) == key
    assert pieces.default_key(small_photo, Settings("cpu", "tiny", tmp_path)) != key
    assert pieces.default_key(small_photo, s, head_n=32) != key
    write_cache(tmp_path / f"{key}.npz", Pieces([], "fake", "cpu", 0.0, False))
    monkeypatch.setattr(pieces.sam2, "load", lambda *a, **k: pytest.fail("model loaded"))
    assert segment(small_photo, s, cache_dir=tmp_path).cached


def test_a_legacy_cache_file_is_not_served(tmp_path, small_photo, monkeypatch):
    # a file named by the photo alone predates the pieces version: it holds first-pass pieces
    import hashlib
    s = Settings("cpu", "small", tmp_path)
    legacy = tmp_path / f"{hashlib.sha256(small_photo.rgb.tobytes()).hexdigest()}.npz"
    write_cache(legacy, Pieces([], "facebook/sam2.1-hiera-small", "cpu", 0.0, False))

    class WouldLoad(Exception):
        pass

    def would_load(*a, **k):
        raise WouldLoad

    monkeypatch.setattr(pieces.sam2, "load", would_load)
    with pytest.raises(WouldLoad):
        segment(small_photo, s, cache_dir=tmp_path)
    assert legacy.exists()


def test_default_key_changes_with_the_pieces_version(tmp_path, small_photo, monkeypatch):
    s = Settings("cpu", "small", tmp_path)
    before = pieces.default_key(small_photo, s)
    monkeypatch.setattr(pieces, "PIECES_VERSION", pieces.PIECES_VERSION + 1)
    assert pieces.default_key(small_photo, s) != before


# ---- piece completion (Task 3b) ----

def _blob_piece(shape, cx: float, cy: float, half_w: float, half_h: float, tails_top_y: float = 1e9) -> Piece:
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    m = (np.abs(xx - cx) <= half_w) & (np.abs(yy - cy) <= half_h)
    return pieces._piece(m, 0.95, 0.97, tails_top_y)


def _disc_piece(shape, head) -> Piece:
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    cx, cy, r = head
    return pieces._piece((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r, 0.97, 0.98, 1e9)


def test_symmetric_prompts_fill_the_missing_steps():
    head = (200.0, 200.0, 150.0)
    shape = (400, 400)
    rad = 0.7 * head[2]
    found = [_disc_piece(shape, head)]
    for a in (0, 60, 120):
        t = np.radians(a)
        found.append(_blob_piece(shape, head[0] + rad * np.cos(t), head[1] + rad * np.sin(t), 8, 8))
    pts = pieces.symmetric_prompts(found, head)
    assert pts.shape == (3, 2)
    want = [(head[0] + rad * np.cos(np.radians(a)), head[1] + rad * np.sin(np.radians(a)))
            for a in (180, 240, 300)]
    for wx, wy in want:
        assert min(np.hypot(pts[:, 0] - wx, pts[:, 1] - wy)) <= 2.0


def test_symmetric_prompts_need_two_pieces_at_equal_steps():
    head = (200.0, 200.0, 150.0)
    shape = (400, 400)
    one = [_disc_piece(shape, head), _blob_piece(shape, 305.0, 200.0, 8, 8)]
    assert len(pieces.symmetric_prompts(one, head)) == 0


def test_edge_prompts_lie_outside_the_block_and_other_pieces():
    import cv2
    head = (200.0, 200.0, 150.0)
    shape = (400, 400)
    unit = head[2] / 150.0
    block = _blob_piece(shape, 300.0, 200.0, 30, 15)  # radial along x, long sides along y
    arm = _blob_piece(shape, 230.0, 200.0, 15, 6)  # touches nothing, lies inward of the block
    found = [_disc_piece(shape, head), block, arm]
    pts = pieces.edge_prompts([found[0], block], head)
    assert pts.shape == (4, 2)
    outside = cv2.distanceTransform((~block.mask).astype(np.uint8), cv2.DIST_L2, 5)
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        assert not block.mask[yi, xi]
        assert abs(outside[yi, xi] - pieces.EDGE_UNITS * unit) <= 1.5
    # a point that falls inside another (not enclosing) piece is not offered
    arm_big = _blob_piece(shape, 262.0, 200.0, 4, 4)
    pts2 = pieces.edge_prompts([found[0], block, arm_big], head)
    assert all(not arm_big.mask[int(round(y)), int(round(x))] for x, y in pts2)
    near_block = [(x, y) for x, y in pts2 if outside[int(round(y)), int(round(x))] <= 5.5]
    assert len(near_block) == 3  # the inward end lies inside the small piece: not offered


class _FakeProc:
    def post_process_masks(self, masks, original_sizes, binarize=False):
        import torch
        return [masks[0].float()]  # the fake logits are already frame size


class _FakeLoaded:
    processor = _FakeProc()


def test_filtered_keeps_a_large_hypothesis_beside_a_better_small_one():
    import torch
    h, w = 120, 120
    p = _prepared(h, w, (60.0, 60.0, 55.0), 1e9)
    small = np.full((h, w), -8.0, np.float32); small[57:60, 57:60] = 8.0  # area 9: below the floor
    large = np.full((h, w), -8.0, np.float32); large[30:90, 30:90] = 8.0
    third = np.full((h, w), -8.0, np.float32); third[40:80, 40:80] = 0.2  # unstable
    logits = torch.from_numpy(np.stack([small, large, third]))[None]  # (1, 3, h, w)
    ious = torch.tensor([[0.99, 0.85, 0.90]])
    out = pieces._filtered(_FakeLoaded(), p, logits, ious, None)
    assert len(out) == 1 and out[0].area == 3600 and out[0].score == pytest.approx(0.85)


def test_a_thin_edge_piece_without_partners_is_dropped():
    head = (200.0, 200.0, 150.0)
    shape = (400, 400)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    p = _prepared(shape[0], shape[1], head, 1e9)
    disc = _disc_piece(shape, head)
    block = _blob_piece(shape, 300.0, 200.0, 30, 15)
    frame = (np.abs(xx - 300) <= 34) & (np.abs(yy - 200) <= 19) & ~block.mask  # a thin border
    border = pieces._piece(frame, 0.9, 0.95, 1e9)
    assert pieces._thin(border, p)
    assert pieces._kept_edge_pieces([border], p, head) == []
    # six such borders at equal steps repeat: kept
    borders = []
    for a in range(0, 360, 60):
        t = np.radians(a)
        bx, by = head[0] + 100 * np.cos(t), head[1] + 100 * np.sin(t)
        ring = ((np.abs(xx - bx) <= 19) & (np.abs(yy - by) <= 19)
                & ~((np.abs(xx - bx) <= 15) & (np.abs(yy - by) <= 15)))
        borders.append(pieces._piece(ring, 0.9, 0.95, 1e9))
    assert all(pieces._thin(b, p) for b in borders)
    assert len(pieces._kept_edge_pieces(borders, p, head)) == 6


def _v_piece(shape, head, ang_deg: float, rad: float) -> Piece:
    import cv2
    cx, cy, _r = head
    t = np.radians(ang_deg)
    ur, ut = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
    tip = np.array([cx, cy]) + (rad + 20) * ur
    poly = [tip, tip - 40 * ur + 30 * ut, tip - 40 * ur + 22 * ut, tip - 9 * ur,
            tip - 40 * ur - 22 * ut, tip - 40 * ur - 30 * ut]  # a V pointing outwards
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(np.array(poly)).astype(np.int32)], 1)
    return pieces._piece(m.astype(bool), 0.95, 0.97, 1e9)


def test_symmetric_prompts_of_a_v_piece_lie_on_the_cloth():
    # the centroid of a V lies in its notch, off the cloth: the prompt is a point of the piece
    head = (200.0, 200.0, 150.0)
    shape = (400, 400)
    found = [_disc_piece(shape, head)] + [_v_piece(shape, head, a, 100.0) for a in (0, 60, 120)]
    gx, gy = pieces._centroid(found[1])
    assert not found[1].mask[int(round(gy)), int(round(gx))]
    pts = pieces.symmetric_prompts(found, head)
    assert pts.shape == (3, 2)
    for a in (180, 240, 300):
        v = _v_piece(shape, head, a, 100.0)
        assert any(v.mask[int(round(y)), int(round(x))] for x, y in pts), a


def test_filtered_takes_a_per_point_area_floor():
    import torch
    h, w = 120, 120
    p = _prepared(h, w, (60.0, 60.0, 55.0), 1e9)
    lg = np.full((h, w), -8.0, np.float32); lg[57:60, 57:60] = 8.0  # area 9: below the floor
    logits = torch.from_numpy(np.stack([lg, lg, lg]))[None]
    ious = torch.tensor([[0.95, 0.10, 0.10]])
    assert pieces._filtered(_FakeLoaded(), p, logits, ious, None) == []
    out = pieces._filtered(_FakeLoaded(), p, logits, ious, None, min_area=np.array([5.0]))
    assert len(out) == 1 and out[0].area == 9


def test_registered_symmetric_points_find_a_repeat_off_the_ideal_step():
    # the photo's steps are not exact: the V at 180 degrees sits 5 degrees on (10 px at the tip)
    head = (200.0, 200.0, 150.0)
    shape = (400, 400)
    p = _prepared(shape[0], shape[1], head, 1e9)  # a red disc
    found = [_disc_piece(shape, head)]
    for a in (0, 60, 120):
        v = _v_piece(shape, head, a, 100.0)
        p.rgb[v.mask] = (250, 250, 250)
        found.append(v)
    targets = [_v_piece(shape, head, 185.0, 100.0), _v_piece(shape, head, 240.0, 100.0),
               _v_piece(shape, head, 300.0, 100.0)]
    for v in targets:
        p.rgb[v.mask] = (250, 250, 250)
    geometric = pieces.symmetric_prompts(found, head)
    assert not any(targets[0].mask[int(round(y)), int(round(x))] for x, y in geometric)
    pts, floors = pieces._symmetric_points(p, found)
    assert len(pts) == len(floors) >= 3
    for v in targets:
        assert any(v.mask[int(round(y)), int(round(x))] for x, y in pts)
    assert all(not f.mask[int(round(y)), int(round(x))] for f in found[1:] for x, y in pts)


# ---- fix round 1 ----

def _ring_of(shape, head, angles, rad: float, half: float = 8.0):
    cx, cy, _r = head
    out = []
    for a in angles:
        t = np.radians(a)
        out.append(_blob_piece(shape, cx + rad * np.cos(t), cy + rad * np.sin(t), half, half))
    return out


def test_two_pieces_never_fit_steps():
    head = (200.0, 200.0, 150.0)
    shape = (400, 400)
    disc = _disc_piece(shape, head)
    assert len(pieces.symmetric_prompts([disc] + _ring_of(shape, head, (0, 60), 100.0), head)) == 0
    assert len(pieces.symmetric_prompts([disc] + _ring_of(shape, head, (0, 180), 100.0), head)) == 0
    assert pieces._fit_steps([0.0, 60.0]) is None


def test_an_outlier_is_dropped_from_the_fit():
    head = (200.0, 200.0, 150.0)
    shape = (400, 400)
    angles = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 25.0]
    n, _off, inliers = pieces._fit_steps(angles)
    assert n == 6 and list(inliers) == [True] * 6 + [False]
    found = [_disc_piece(shape, head)] + _ring_of(shape, head, angles, 100.0)
    assert len(pieces.symmetric_prompts(found, head)) == 0


def test_five_of_six_fit_six_steps():
    head = (200.0, 200.0, 150.0)
    shape = (400, 400)
    n, _off, inliers = pieces._fit_steps([0.0, 60.0, 120.0, 180.0, 240.0])
    assert n == 6 and inliers.all()
    found = [_disc_piece(shape, head)] + _ring_of(shape, head, (0, 60, 120, 180, 240), 100.0)
    pts = pieces.symmetric_prompts(found, head)
    assert len(pts) == 1 and np.hypot(pts[0, 0] - 250.0, pts[0, 1] - (200.0 - 86.6)) <= 2.0


def test_thin_is_by_width_not_area():
    head = (200.0, 200.0, 150.0)
    shape = (400, 400)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    p = _prepared(shape[0], shape[1], head, 1e9)
    block = _blob_piece(shape, 300.0, 200.0, 25, 11)  # 51 x 23: 1.7 % of the rosette, 23 px wide
    assert block.area / p.mask.sum() < 0.02
    assert not pieces._thin(block, p)
    assert pieces._kept_edge_pieces([block], p, head) == [block]
    rr = np.hypot(xx - 200.0, yy - 200.0)
    ring = pieces._piece((rr >= 40) & (rr <= 44), 0.9, 0.95, 1e9)  # thin, centred, all around
    assert pieces._kept_edge_pieces([ring], p, head) == [ring]
    arc = pieces._piece((rr >= 40) & (rr <= 44) & (xx > 200), 0.9, 0.95, 1e9)  # half a ring
    assert pieces._thin(arc, p) and pieces._kept_edge_pieces([arc], p, head) == []


def test_a_thin_edge_piece_on_an_occupied_step_needs_new_partners():
    head = (200.0, 200.0, 150.0)
    shape = (400, 400)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    p = _prepared(shape[0], shape[1], head, 1e9)

    def border(a):
        t = np.radians(a)
        bx, by = head[0] + 100 * np.cos(t), head[1] + 100 * np.sin(t)
        m = ((np.abs(xx - bx) <= 19) & (np.abs(yy - by) <= 19)
             & ~((np.abs(xx - bx) <= 15) & (np.abs(yy - by) <= 15)))
        return pieces._piece(m, 0.9, 0.95, 1e9)

    one = border(0)
    assert pieces._kept_edge_pieces([one], p, head) == []
    assert len(pieces._kept_edge_pieces([border(a) for a in (0, 120, 240)], p, head)) == 3


def test_registration_falls_back_when_the_piece_has_the_disc_color():
    head = (200.0, 200.0, 150.0)
    shape = (400, 400)
    p = _prepared(shape[0], shape[1], head, 1e9)  # a red disc; the V pieces are red too
    found = [_disc_piece(shape, head)] + [_v_piece(shape, head, a, 100.0) for a in (0, 60, 120)]
    pts, _floors = pieces._symmetric_points(p, found)
    assert np.allclose(np.sort(pts, axis=0), np.sort(pieces.symmetric_prompts(found, head), axis=0))


def _soft(h, w, y0, y1, x0, x1):
    lg = np.full((h, w), -8.0, np.float32)
    lg[y0 - 2:y1 + 2, x0 - 2:x1 + 2] = 0.5  # between -1 and 1: lowers the stability
    lg[y0:y1, x0:x1] = 8.0
    return lg


def test_filtered_keeps_weak_candidates_apart_for_the_completion_pass():
    import torch
    h, w = 120, 120
    p = _prepared(h, w, (60.0, 60.0, 55.0), 1e9)
    lg = _soft(h, w, 40, 80, 40, 80)  # stability 1600 / 1936 = 0.83
    logits = torch.from_numpy(lg)[None, None]
    ious = torch.tensor([[0.75]])
    assert pieces._filtered(_FakeLoaded(), p, logits, ious, None) == []
    weak: list = []
    assert pieces._filtered(_FakeLoaded(), p, logits, ious, None, weak=weak) == []
    assert len(weak) == 1 and weak[0].area == 1936
    weak2: list = []
    pieces._filtered(_FakeLoaded(), p, logits, torch.tensor([[0.65]]), None, weak=weak2)
    assert weak2 == []


def test_a_weak_set_is_kept_only_when_it_repeats_in_one_color():
    head = (200.0, 200.0, 150.0)
    shape = (400, 400)
    p = _prepared(shape[0], shape[1], head, 1e9)
    three = _ring_of(shape, head, (0, 120, 240), 100.0)
    assert len(pieces._repeated_weak(three, p)) == 3
    assert pieces._repeated_weak(three[:2], p) == []
    p.rgb[three[1].mask] = (30, 40, 200)  # one of them is blue: no longer a repeat in one color
    assert pieces._repeated_weak(three, p) == []


# ---- S70-19 with the weights (slow) ----

def _models_dir() -> Path:
    import os
    return Path(os.environ.get("ROSETTE_MODELS_DIR") or DEFAULT_MODELS_DIR)


def _solidity(piece: Piece) -> float:
    import cv2
    m = piece.mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.contourArea(cv2.convexHull(max(contours, key=cv2.contourArea)))
    return piece.area / hull if hull > 0 else 1.0


@pytest.fixture(scope="module")
def s70_19(tmp_path_factory):
    models = _models_dir()
    if not local_dir(models, "small").joinpath("model.safetensors").exists():
        pytest.skip(f"weights not found in {models}: run tools/fetch_models.py --model small "
                    "or set ROSETTE_MODELS_DIR")
    from rosette.vision.prepare import prepare
    p = prepare((PHOTOS / "S70-19.png").read_bytes())
    cache = tmp_path_factory.mktemp("pieces")
    out = segment(p, Settings("cpu", "small", models), cache_dir=cache)
    again = segment(p, Settings("cpu", "small", models), cache_dir=cache)
    return p, out, again


def _kinds(p, out) -> dict[str, list[Piece]]:
    """S70-19 head pieces by kind: blue blocks, outer white chevrons (V, solidity < 0.85, beyond
    0.6 R), inner white star arms (0.3 R - 0.6 R), red pieces touching a block (borders), a
    centred red star (solidity < 0.85, larger than the ring, smaller than the disc)."""
    import cv2
    cx, cy, r = p.head
    rgb = p.rgb.astype(int)
    kinds: dict[str, list[Piece]] = {k: [] for k in ("block", "chevron", "arm", "border", "star")}
    blocks_near = np.zeros(p.mask.shape, bool)
    for pc in out.pieces:
        if not pc.in_head:
            continue
        rr, g, b = np.median(rgb[pc.mask], axis=0)
        if b > rr + 40 and b > g + 40 and pc.area < 0.02 * p.mask.sum():
            kinds["block"].append(pc)
            blocks_near |= cv2.dilate(pc.mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    for pc in out.pieces:
        if not pc.in_head:
            continue
        rr, g, b = np.median(rgb[pc.mask], axis=0)
        gx, gy = pieces._centroid(pc)
        rad = np.hypot(gx - cx, gy - cy) / r
        if min(rr, g, b) > 170:
            if rad > 0.6 and _solidity(pc) < 0.85:
                kinds["chevron"].append(pc)
            elif 0.3 <= rad <= 0.6:
                kinds["arm"].append(pc)
        elif rr > 150 and g < 110 and b < 110:
            if rad > 0.3 and (pc.mask & blocks_near).any() and pc.area < 0.05 * p.mask.sum():
                kinds["border"].append(pc)
            elif rad < 0.15 and _solidity(pc) < 0.85 and 0.08 < pc.area / (np.pi * r * r) < 0.8:
                kinds["star"].append(pc)
    return kinds


@pytest.mark.slow
def test_s70_19_pieces_include_the_blocks_chevrons_and_arms(s70_19):
    p, out, again = s70_19
    assert not out.cached and out.device == "cpu" and len(out.passes) == 2
    assert again.cached and len(again.pieces) == len(out.pieces)
    kinds = _kinds(p, out)
    counts = {k: len(v) for k, v in kinds.items()}
    print("S70-19 pieces:", counts, "passes (s):", out.passes)
    assert counts["block"] >= 6 and counts["chevron"] >= 6 and counts["arm"] >= 6, counts


@pytest.mark.slow
@pytest.mark.xfail(reason="SAM 2.1 small is not confident about red on red on S70-19: even the "
                          "repeat bars (0.70 / 0.75) give at most two red pieces, not a repeating set "
                          "(Task 3b report, fix round 1)", strict=False)
def test_s70_19_pieces_include_the_red_borders_or_the_red_star(s70_19):
    p, out, _again = s70_19
    kinds = _kinds(p, out)
    assert len(kinds["border"]) >= 6 or len(kinds["star"]) >= 1, {k: len(v) for k, v in kinds.items()}


def test_filtered_drops_a_mask_of_two_significant_parts():
    import torch
    h, w = 120, 120
    p = _prepared(h, w, (60.0, 60.0, 55.0), 1e9)
    two = np.full((h, w), -8.0, np.float32); two[30:50, 30:50] = 8.0; two[70:90, 70:90] = 8.0
    speck = np.full((h, w), -8.0, np.float32); speck[30:60, 30:60] = 8.0; speck[80:82, 80:82] = 8.0
    out = pieces._filtered(_FakeLoaded(), p, torch.from_numpy(np.stack([two, speck]))[:, None],
                           torch.tensor([[0.95], [0.9]]), None)
    assert len(out) == 1 and out[0].area == 904
