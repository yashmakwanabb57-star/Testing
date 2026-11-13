"""ML-4 Task 2: piece groups and signal counts (synthetic pieces on a 600 px frame)."""
import cv2
import numpy as np

from rosette.perception.pieces import Piece
from rosette.perception.reader import MarkAnswer, Reading
from rosette.vision import groups
from rosette.vision.parts_head import HARMONIC_SECOND, harmonic_count

SIZE = 600
HEAD = (300.0, 300.0, 250.0)
SCALE = 250.0 / 150.0  # px per canvas unit: the head radius is 150 units


def _piece(mask: np.ndarray, color_group: int | None = 1) -> Piece:
    ys, xs = np.nonzero(mask)
    return Piece(mask=mask, score=0.95, stability=0.95, area=int(mask.sum()),
                 bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), in_head=True,
                 color_group=color_group)


def _arm(deg: float, r0: float = 80.0, r1: float = 200.0, half_deg: float = 9.0) -> Piece:
    """A triangular arm: base at r0 spanning +-half_deg, tip at r1 on the axis `deg`."""
    cx, cy, _ = HEAD
    a, h = np.radians(deg), np.radians(half_deg)
    pts = np.array([[cx + r0 * np.cos(a - h), cy + r0 * np.sin(a - h)],
                    [cx + r1 * np.cos(a), cy + r1 * np.sin(a)],
                    [cx + r0 * np.cos(a + h), cy + r0 * np.sin(a + h)]], np.int32)
    m = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillConvexPoly(m, pts, 1)
    return _piece(m > 0)


def _sector(deg: float, r0: float, r1: float, half_deg: float) -> Piece:
    """An annular sector (canvas units) on the axis `deg`."""
    a = np.radians(np.linspace(deg - half_deg, deg + half_deg, 40))
    out = np.c_[300 + r1 * SCALE * np.cos(a), 300 + r1 * SCALE * np.sin(a)]
    inn = np.c_[300 + r0 * SCALE * np.cos(a[::-1]), 300 + r0 * SCALE * np.sin(a[::-1])]
    m = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillPoly(m, [np.concatenate([out, inn]).astype(np.int32)], 1)
    return _piece(m > 0)


def _wedge(deg: float, r0: float, r1: float, half_deg: float) -> Piece:
    """A triangle (canvas units) pointing inward: its point at r0, its base at r1 (mass outward)."""
    a, h = np.radians(deg), np.radians(half_deg)
    pts = np.array([[300 + r0 * SCALE * np.cos(a), 300 + r0 * SCALE * np.sin(a)],
                    [300 + r1 * SCALE * np.cos(a - h), 300 + r1 * SCALE * np.sin(a - h)],
                    [300 + r1 * SCALE * np.cos(a + h), 300 + r1 * SCALE * np.sin(a + h)]], np.int32)
    m = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillConvexPoly(m, pts, 1)
    return _piece(m > 0)


def _disc(r: float, r_in: float = 0.0) -> Piece:
    m = np.zeros((SIZE, SIZE), np.uint8)
    cv2.circle(m, (300, 300), int(r), 1, -1)
    if r_in > 0:
        cv2.circle(m, (300, 300), int(r_in), 0, -1)
    return _piece(m > 0)


def _reading(kinds: list[str | None], repeats: list[int] | None = None) -> Reading:
    marks = list(range(1, len(kinds) + 1))
    answers = {m: MarkAnswer(mark=m, kind=k, texture="plain", repeats=(repeats or [1] * len(kinds))[m - 1])
               for m, k in zip(marks, kinds) if k is not None}
    return Reading(marks=marks, piece_index={m: m - 1 for m in marks}, answers=answers, pairs=[],
                   seconds=0.0, cached=False, images=1)


def test_six_visible_arm_pieces_count_six_from_their_steps():
    pcs = [_arm(k * 60.0) for k in range(6)]
    out = groups.piece_groups(_reading(["star"] * 6, [6] * 6), pcs, HEAD, SCALE)
    assert len(out) == 1
    g = out[0]
    assert (g.kind, g.count, g.count_source, g.alternatives) == ("star", 6, "pieces", [])
    assert sorted(g.pieces) == list(range(6))
    assert g.confidence == 1.0
    assert abs(g.r_inner - 80 / SCALE) < 4  # the 2nd percentile radius: the arms' base
    assert 100 < g.r_outer <= 200 / SCALE  # the 98th percentile radius: short of the thin tips
    assert min(g.rotation_deg, 60 - g.rotation_deg) < 3  # arms on 0, 60, ...: rotation 0 mod 60


def test_eight_arms_with_two_hidden_count_eight_from_the_harmonic():
    pcs = [_arm(k * 45.0) for k in range(6)]  # steps 6 and 7 hidden
    g = groups.piece_groups(_reading(["star"] * 6, [8] * 6), pcs, HEAD, SCALE)[0]
    assert (g.count, g.count_source, g.alternatives) == (8, "harmonic", [])
    assert 0 < g.confidence <= 0.9
    assert groups.group_count(g, pcs, HEAD) == (8, "harmonic", [])


def test_the_models_repeats_become_an_alternative_when_they_differ():
    pcs = [_arm(k * 45.0) for k in range(6)]
    g = groups.piece_groups(_reading(["star"] * 6, [6] * 6), pcs, HEAD, SCALE)[0]
    assert (g.count, g.count_source, g.alternatives) == (8, "harmonic", [6])
    assert groups.group_count(g, pcs, HEAD, repeats=6) == (8, "harmonic", [6])
    assert groups.group_count(g, pcs, HEAD, repeats=8)[2] == []
    assert groups.group_count(g, pcs, HEAD, repeats=20)[2] == []  # outside 3..16


def test_a_piece_joins_one_group_only():
    inner = [_sector(k * 60.0, 20, 38, 12) for k in range(6)]  # span about 20-38 units
    outer = [_sector(k * 60.0 + 30.0, 32, 56, 12) for k in range(6)]  # about 32-56 units
    straddle = _wedge(15.0, 24, 47, 6)  # about 26-47: within MIN_ZONE_UNITS of both seeds
    pcs = outer + inner + [straddle]  # the outer sectors are larger: they seed first
    gs = groups.piece_groups(_reading(["block"] * 13), pcs, HEAD, SCALE)
    assert len(gs) == 2 and sum(len(g.pieces) for g in gs) == 13
    holding = [g for g in gs if 12 in g.pieces]
    assert len(holding) == 1
    # its span is nearer the inner seed's, but its centroid radius (about 38.6) lies only in the outer span
    s0, s1 = groups._span(straddle, HEAD, SCALE)
    si, so = groups._span(inner[0], HEAD, SCALE), groups._span(outer[0], HEAD, SCALE)
    assert abs(s0 - si[0]) + abs(s1 - si[1]) < abs(s0 - so[0]) + abs(s1 - so[1])
    assert set(holding[0].pieces) == set(range(6)) | {12}
    other = next(g for g in gs if g is not holding[0])
    assert set(other.pieces) == set(range(6, 12)) and other.count == 6 and other.count_source == "pieces"


def test_disc_ring_badge_rose_are_singleton_groups_with_count_one():
    pcs = [_disc(200), _disc(240, 210), _disc(40), _disc(30)]
    gs = groups.piece_groups(_reading(["pleated_disc", "ring", "badge", "rose"]), pcs, HEAD, SCALE)
    assert sorted(g.kind for g in gs) == ["badge", "pleated_disc", "ring", "rose"]
    for g in gs:
        assert len(g.pieces) == 1 and g.count == 1 and g.alternatives == []
    disc = next(g for g in gs if g.kind == "pleated_disc")
    assert disc.r_inner == 0.0 and disc.pieces == [0]


def test_harmonic_count_reports_a_second_peak_at_twice_n():
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    n, amp, second = harmonic_count(100 + 10 * np.cos(4 * th) + 7 * np.cos(8 * th))
    assert (n, second) == (4, 8) and abs(amp - 10) < 1e-6
    n, _, second = harmonic_count(100 + 10 * np.cos(4 * th) + (HARMONIC_SECOND * 10 - 1) * np.cos(8 * th))
    assert (n, second) == (4, None)


def test_unanswered_pieces_group_by_span_as_other_with_low_confidence():
    pcs = [_arm(k * 60.0) for k in range(6)]
    g = groups.piece_groups(_reading([None] * 6), pcs, HEAD, SCALE)
    assert len(g) == 1 and g[0].kind == "other" and g[0].count == 6 and g[0].confidence == 0.3
