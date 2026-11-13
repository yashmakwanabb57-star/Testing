"""ML-4 Task 4: the judge. The photo is a render of a known head (the renderer is the photo),
so the right candidate is known; no model run, no SAM."""
import copy
import re

import cv2
import numpy as np
import pytest

from rosette.model import ColorGroup, Part, RosetteModel
from rosette.perception.pieces import Piece
from rosette.vision import judge as J
from rosette.vision.prepare import Prepared

# Photo frame: 2 px per canvas unit, axis at x=350, the head's top row at y=50 (canvas y=25),
# so the canvas head circle (300, 175, 150) lands on (350, 350, 300) in pixels.
H, W, S = 760, 700, 2.0
CX, CY, R = 350.0, 350.0, 300.0
TAILS_TOP = int(CY + R)  # 650
COLORS = {0: (200, 40, 40), 1: (40, 60, 190), 2: (240, 200, 40)}


def _model(count: int, head_z: int = 0) -> RosetteModel:
    groups = [ColorGroup(g, COLORS[g], False, 1.0, 0.5, g + 1, 0.9) for g in COLORS]
    parts = [Part("head", "head", {"cx": 300.0, "cy": 175.0, "r": 150.0}, 0, head_z, 0.9),
             Part("p1", "points", {"cx": 300.0, "cy": 175.0, "kind": "star", "count": count, "r_inner": 45.0,
                                   "r_outer": 120.0, "rotation_deg": 0.0, "width_ratio": 0.5}, 1, 20, 0.9),
             Part("center", "center", {"cx": 300.0, "cy": 175.0, "r": 60.0}, 2, 40, 0.9)]
    return RosetteModel(parts=parts, groups=groups)


def _frame() -> Prepared:
    yy, xx = np.mgrid[:H, :W]
    mask = (xx - CX) ** 2 + (yy - CY) ** 2 <= R * R
    mask |= (yy >= TAILS_TOP) & (np.abs(xx - CX) < 120)  # the tails sector below the head
    return Prepared(rgb=np.zeros((H, W, 3), np.uint8), mask=mask, axis_x=CX, head=(CX, CY, R),
                    tails_top_y=float(TAILS_TOP), bottom_y=float(H - 1), scale=S)


def _photo(model: RosetteModel):
    """(Prepared, labels, pieces): labels and rgb painted from the model's render; pieces are
    the star arms (connected parts of the star's class), as SAM would cut them."""
    p = _frame()
    cls = J.render_classes(model, p)
    labels = np.where(p.mask, cls - 1, -1).astype(np.int32)
    labels[TAILS_TOP:][p.mask[TAILS_TOP:]] = 1  # tails: one stripe colour
    rgb = np.full((H, W, 3), 255, np.uint8)
    for g, c in COLORS.items():
        rgb[labels == g] = c
    p.rgb = rgb
    n, comp = cv2.connectedComponents((labels == 1).astype(np.uint8) * (np.arange(H)[:, None] < TAILS_TOP)
                                      .astype(np.uint8), connectivity=8)
    pieces = []
    for i in range(1, n):
        m = comp == i
        ys, xs = np.nonzero(m)
        pieces.append(Piece(mask=m, score=0.9, stability=0.9, area=int(m.sum()),
                            bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), in_head=True))
    return p, labels, pieces


def test_the_candidate_matching_the_photo_wins():
    p, labels, pieces = _photo(_model(6))
    assert len(pieces) == 6
    v = J.judge([_model(8), _model(6)], p, labels, pieces)
    assert v.winner == 1
    assert v.scores[1] > v.scores[0]
    assert v.margin >= J.JUDGE_MARGIN
    assert v.changed == ["p1"]
    # and the other way round: the rules head is right and stays
    v = J.judge([_model(6), _model(8)], p, labels, pieces)
    assert v.winner == 0 and v.margin >= J.JUDGE_MARGIN and v.changed == []


def test_identical_renders_keep_candidate_zero():
    p, labels, pieces = _photo(_model(6))
    rules, other = _model(8), _model(8, head_z=5)  # a z change under every other layer
    v = J.judge([rules, other], p, labels, pieces)
    assert v.scores[0] == pytest.approx(v.scores[1])
    assert v.margin == pytest.approx(0.0)
    assert v.winner == 0
    out = J.flag(rules, v)
    assert "judge undecided: candidate 0 or candidate 1 kept as the rules read it" in out.warnings
    assert [q.confidence for q in out.parts] == [q.confidence for q in rules.parts]


def test_terms_are_in_unit_range_and_ignore_the_tails():
    p, labels, pieces = _photo(_model(6))
    for m in (_model(6), _model(8)):
        s, terms = J.score(m, p, labels, pieces)
        assert set(terms) >= {"color", "edge", "piece"}
        for k in ("color", "edge", "piece"):
            assert 0.0 <= terms[k] <= 1.0, (k, terms[k])
        assert 0.0 <= s <= 1.0
        # another tails sector: other labels, other colours, a stripe edge -> same score
        labels2, p2 = labels.copy(), copy.copy(p)
        p2.rgb = p.rgb.copy()
        below = np.zeros(labels.shape, bool)
        below[TAILS_TOP:] = p.mask[TAILS_TOP:]
        labels2[below & (np.arange(W)[None, :] < CX)] = 2
        p2.rgb[below & (np.arange(W)[None, :] < CX)] = COLORS[2]
        s2, terms2 = J.score(m, p2, labels2, pieces)
        assert s2 == pytest.approx(s) and terms2 == pytest.approx(terms)
    # perfect candidate: colour and pieces agree fully
    _, terms = J.score(_model(6), p, labels, pieces)
    assert terms["color"] > 0.99 and terms["piece"] > 0.95 and terms["edge"] > 0.9
    # no pieces: the piece term is dropped and the weights renormalised
    s, terms = J.score(_model(6), p, labels, [])
    assert "piece" not in terms
    assert s == pytest.approx((J.W_COLOR * terms["color"] + J.W_EDGE * terms["edge"]) / (J.W_COLOR + J.W_EDGE))


def test_flag_caps_confidence_by_the_margin_and_writes_the_info_line():
    p, labels, pieces = _photo(_model(6))
    cands = [_model(8), _model(6)]
    v = J.judge(cands, p, labels, pieces, decisions=["8 points", "6 points"])
    assert v.winner == 1
    # a small decided margin caps the changed part only
    small = J.Verdict(winner=1, scores=[0.70, 0.73], terms=[{}, {}], margin=0.03, labels=v.labels,
                      changed=["p1"])
    out = J.flag(cands[1], small)
    conf = {q.id: q.confidence for q in out.parts}
    assert conf["p1"] == pytest.approx(min(0.9, 0.5 + 10 * 0.03))
    assert conf["head"] == 0.9 and conf["center"] == 0.9
    assert "judge: 8 points -> 6 points (margin 0.030)" in out.info
    assert cands[1].parts[1].confidence == 0.9 and cands[1].info == []  # the input is not changed
    # the real verdict: line format and cap by its own margin
    out = J.flag(cands[1], v)
    line = [x for x in out.info if x.startswith("judge:")]
    assert len(line) == 1 and re.fullmatch(r"judge: 8 points -> 6 points \(margin \d\.\d{3}\)", line[0])
    assert next(q for q in out.parts if q.id == "p1").confidence == pytest.approx(min(0.9, 0.5 + 10 * v.margin))
    assert not any(w.startswith("judge undecided") for w in out.warnings)


def test_merged_labels_count_as_their_group():
    p, labels, pieces = _photo(_model(6))
    m = _model(6)
    _, ref = J.score(m, p, labels, pieces)
    split = labels.copy()
    split[(labels == 2) & (np.arange(W)[None, :] < CX)] = 5  # assign merged label 5 into the centre's group 2
    _, raw = J.score(m, p, split, pieces)
    assert raw["color"] < ref["color"] - 0.01
    _, fixed = J.score(m, p, split, pieces, merged={5: 2})
    assert fixed["color"] == pytest.approx(ref["color"])
    v = J.judge([m, _model(8)], p, split, pieces, merged={5: 2})
    assert v.terms[0]["color"] == pytest.approx(ref["color"])


def test_colour_term_tracks_head_fit():
    from rosette import pipeline as P
    p, labels, pieces = _photo(_model(6))
    for m in (_model(6), _model(8)):
        _, terms = J.score(m, p, labels, pieces)
        assert terms["color"] == pytest.approx(P.head_fit(p, labels, m).agreement, abs=0.05)


def test_single_candidate():
    p, labels, pieces = _photo(_model(6))
    v = J.judge([_model(6)], p, labels, pieces)
    assert v.winner == 0 and v.margin == 0.0 and v.changed == []
    out = J.flag(_model(6), v)
    assert out.warnings == [] and out.info == []


def test_decisions_length_mismatch_falls_back_to_default_labels():
    p, labels, pieces = _photo(_model(6))
    v = J.judge([_model(8), _model(6)], p, labels, pieces, decisions=["only one"])
    assert v.labels == ["candidate 0", "candidate 1"]


def test_more_than_four_colours_skips_the_judge():
    p, labels, pieces = _photo(_model(6))
    big = _model(6)
    big.groups += [ColorGroup(g, (10 * g, 10, 10), False, 1.0, 0.5, 1, 0.9) for g in (3, 4)]
    big.parts += [Part(f"ring-{g}", "ring", {"cx": 300.0, "cy": 175.0, "r_outer": 30.0 - 5 * g,
                                            "r_inner": 25.0 - 5 * g}, g, 50 + g, 0.9) for g in (3, 4)]
    v = J.judge([big, _model(8)], p, labels, pieces)
    assert v.winner == 0 and v.note == "judge skipped: more than 4 colours"
    out = J.flag(big, v)
    assert "judge skipped: more than 4 colours" in out.info
    assert not any(w.startswith("judge undecided") for w in out.warnings)


def test_a_tie_among_alternatives_takes_the_fewest_changes(monkeypatch):
    rules = _model(8)
    one = _model(6)  # changes p1
    two = _model(6)
    two.parts[2].params["r"] = 50.0  # changes p1 and center
    fake = iter([0.50, 0.795, 0.80])
    monkeypatch.setattr(J, "score", lambda *a, **k: (next(fake), {}))
    p, labels, pieces = _photo(_model(6))
    v = J.judge([rules, one, two], p, labels, pieces)
    assert v.winner == 1 and v.margin == pytest.approx(0.295) and v.tie == [1, 2]  # the winner's own lead
    out = J.flag(one, v)
    assert any(x.startswith("judge: candidate 0 -> candidate 1") and "tie" in x for x in out.info)


def test_margin_is_measured_against_candidate_zero(monkeypatch):
    fake = iter([0.50, 0.80, 0.70])
    monkeypatch.setattr(J, "score", lambda *a, **k: (next(fake), {}))
    p, labels, pieces = _photo(_model(6))
    v = J.judge([_model(8), _model(6), _model(7)], p, labels, pieces)
    assert v.winner == 1 and v.margin == pytest.approx(0.30) and v.tie == []


def test_removed_parts_are_changed_and_named():
    p, labels, pieces = _photo(_model(6))
    rules = _model(6)
    rules.parts.append(Part("dot", "center", {"cx": 300.0, "cy": 175.0, "r": 140.0}, 1, 60, 0.9))
    v = J.judge([rules, _model(6)], p, labels, pieces)
    assert v.winner == 1 and "dot" in v.changed
    out = J.flag(_model(6), v)
    assert any("removed dot" in x for x in out.info)


def test_a_decided_rules_head_is_kept():
    p, labels, pieces = _photo(_model(6))
    v = J.judge([_model(6), _model(8)], p, labels, pieces)
    out = J.flag(_model(6), v)
    assert re.fullmatch(r"judge: candidate 0 kept \(margin \d\.\d{3}\)", out.info[0])


def test_edge_term_penalises_render_boundaries_without_photo_edges():
    p, labels, pieces = _photo(_model(6))
    busy = _model(6)
    busy.parts.append(Part("ring-x", "ring", {"cx": 300.0, "cy": 175.0, "r_outer": 140.0, "r_inner": 130.0},
                           2, 60, 0.9))
    plain = J.score(_model(6), p, labels, [])[1]["edge"]
    assert J.score(busy, p, labels, [])[1]["edge"] < plain - 0.05
