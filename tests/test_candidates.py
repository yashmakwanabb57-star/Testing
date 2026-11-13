"""ML-4 Task 3: open head decisions and candidate heads (hand-built models, no photo, no model run)."""
import copy

from rosette.model import ColorGroup, Part, RosetteModel
from rosette.perception.reader import PairAnswer, Reading
from rosette.vision import candidates as cd
from rosette.vision import parts_head
from rosette.vision.groups import Group


def _points(pid: str, kind: str, count: int, r0: float, r1: float, color: int, z: int,
            rot: float = 0.0) -> Part:
    return Part(pid, "points", {"cx": 300.0, "cy": 200.0, "kind": kind, "count": count, "r_inner": r0,
                                "r_outer": r1, "rotation_deg": rot, "width_ratio": 0.5}, color, z, 0.9)


def _model(*points: Part, center: bool = True, rose: bool = False) -> RosetteModel:
    groups = [ColorGroup(g, (40 * g, 30, 30), False, 0.1, 0.2, g, 0.9) for g in range(1, 9)]
    parts = [Part("head", "head", {"cx": 300.0, "cy": 200.0, "r": 150.0}, 8, 0, 0.9), *points]
    if center:
        parts.append(Part("center", "center", {"cx": 300.0, "cy": 200.0, "r": 20.0}, 7, 40, 0.9))
    return RosetteModel(parts=parts, groups=groups, rose=rose)


def _group(kind: str, r0: float, r1: float, color: int | None, count: int, pieces=(0,), alts=()) -> Group:
    return Group(kind=kind, texture="plain", color_group=color, pieces=list(pieces), r_inner=r0, r_outer=r1,
                 count=count, count_source="single" if kind in ("badge", "rose") else "pieces",
                 alternatives=list(alts), rotation_deg=0.0, confidence=0.9)


def _reading(pairs=(), marks=None) -> Reading:
    marks = marks or {}
    return Reading(marks=sorted(marks), piece_index=dict(marks), answers={}, pairs=list(pairs),
                   seconds=0.0, cached=False, images=1)


def _two_layers():
    """A star (colour 1, z 10) under blocks (colour 2, z 20); groups that agree on kind and count."""
    star, blocks = _points("star-a", "star", 6, 40, 120, 1, 10), _points("blocks", "diamond", 6, 60, 100, 2, 20)
    grps = [_group("star", 40, 120, 1, 6, pieces=(0,)), _group("block", 60, 100, 2, 6, pieces=(1,))]
    return star, blocks, grps


def test_candidate_zero_is_the_rules_head_unchanged():
    rules = _model(_points("blocks", "diamond", 6, 60, 100, 2, 20))
    before = copy.deepcopy(rules.to_json())
    grps = [_group("block", 58, 102, 2, 8, alts=(6,))]
    decisions = cd.open_decisions(rules, grps, _reading())
    heads = cd.candidate_heads(rules, decisions)
    assert heads[0] is not rules and heads[0].to_json() == before
    heads[1].parts[1].params["count"] = 99
    heads[0].parts[1].params["count"] = 98
    assert rules.to_json() == before
    # no decisions: the rules head alone
    only = cd.candidate_heads(rules, [])
    assert len(only) == 1 and only[0] is not rules and only[0].to_json() == before


def test_a_count_disagreement_makes_count_candidates_only():
    rules = _model(_points("blocks", "diamond", 6, 60, 100, 2, 20, rot=7.5))
    grps = [_group("block", 58, 102, 2, 8, alts=(6,))]
    decisions = cd.open_decisions(rules, grps, _reading())
    assert [(d.kind, d.part_ids, d.options) for d in decisions] == [("count", ("blocks",), (6, 8))]
    heads = cd.candidate_heads(rules, decisions)
    assert [h.parts[1].params["count"] for h in heads] == [6, 8]
    for h in heads:
        q = h.parts[1].params
        assert (q["r_inner"], q["r_outer"], q["rotation_deg"], q["kind"]) == (60, 100, 7.5, "diamond")
    # a group count equal to the rules count, no alternatives: nothing open
    assert cd.open_decisions(rules, [_group("block", 58, 102, 2, 6)], _reading()) == []


def test_an_unclear_or_contradicted_order_makes_an_order_decision():
    star, blocks, grps = _two_layers()
    rules = _model(star, blocks)
    # the vote put the blocks over the star; the model says the star (a) is on top: contradicted
    blocks.params["vote_star-a"], star.params["vote_blocks"] = "over", "under"
    reading = _reading([PairAnswer(1, 2, "a")], marks={1: 0, 2: 1})
    decisions = cd.open_decisions(rules, grps, reading)
    assert [(d.kind, d.part_ids, d.options) for d in decisions] == [("order", ("blocks", "star-a"),
                                                                      ("blocks", "star-a"))]
    heads = cd.candidate_heads(rules, decisions)
    assert len(heads) == 2
    z = lambda h, pid: next(q.z for q in h.parts if q.id == pid)  # noqa: E731
    assert z(heads[0], "blocks") > z(heads[0], "star-a")
    assert z(heads[1], "star-a") > z(heads[1], "blocks")
    # an undecided vote with a confident model answer is a decision too (the judge confirms it)
    blocks.params["vote_star-a"], star.params["vote_blocks"] = "undecided", "undecided"
    reading = _reading([PairAnswer(2, 1, "b")], marks={1: 0, 2: 1})
    assert [d.kind for d in cd.open_decisions(rules, grps, reading)] == ["order"]


def test_agreeing_order_makes_no_decision():
    star, blocks, grps = _two_layers()
    rules = _model(star, blocks)
    blocks.params["vote_star-a"], star.params["vote_blocks"] = "over", "under"
    assert cd.open_decisions(rules, grps, _reading([PairAnswer(1, 2, "b")], marks={1: 0, 2: 1})) == []
    # unclear model answer: nothing to decide
    assert cd.open_decisions(rules, grps, _reading([PairAnswer(1, 2, "unclear")], marks={1: 0, 2: 1})) == []
    # no vote on the pair (the vote saw no drawn overlap): a swap would not change the drawing
    del blocks.params["vote_star-a"], star.params["vote_blocks"]
    assert cd.open_decisions(rules, grps, _reading([PairAnswer(1, 2, "a")], marks={1: 0, 2: 1})) == []


def test_kind_disagreement_alone_is_one_candidate_not_a_silent_change():
    rules = _model(_points("blocks", "diamond", 6, 60, 100, 2, 20))
    grps = [_group("loop", 60, 100, 2, 6)]
    decisions = cd.open_decisions(rules, grps, _reading())
    assert [(d.kind, d.part_ids, d.options) for d in decisions] == [("layer_kind", ("blocks",), ("diamond", "bow"))]
    heads = cd.candidate_heads(rules, decisions)
    assert [h.parts[1].params["kind"] for h in heads] == ["diamond", "bow"]  # candidate 0 keeps the rules kind
    assert heads[1].parts[1].params["count"] == 6
    # every model kind maps onto today's points kinds
    for kind, mapped in (("star", "star"), ("petal", "petal"), ("loop", "bow")):
        d = cd.open_decisions(rules, [_group(kind, 60, 100, 2, 6)], _reading())
        assert [x.options for x in d] == [("diamond", mapped)]
    assert cd.open_decisions(rules, [_group("block", 60, 100, 2, 6)], _reading()) == []


def test_decisions_are_capped_at_four_and_candidates_at_twenty_four():
    layers = [_points(f"l{k}", "diamond", 6, 10 + 25 * k, 30 + 25 * k, k + 1, 10 + k) for k in range(5)]
    rules = _model(*layers)
    # five count decisions of three options each, and a kind disagreement after them
    grps = [_group("block", 10 + 25 * k, 30 + 25 * k, k + 1, 8, alts=(12,)) for k in range(5)]
    grps[4].kind = "loop"
    decisions = cd.open_decisions(rules, grps, _reading())
    assert len(decisions) == 4 and all(d.kind == "count" for d in decisions)
    assert [d.part_ids for d in decisions] == [("l0",), ("l1",), ("l2",), ("l3",)]
    assert all(d.options == (6, 8, 12) for d in decisions)
    heads = cd.candidate_heads(rules, decisions)
    assert len(heads) == 24
    assert heads[0].to_json() == rules.to_json()
    keys = [tuple(h.parts[1 + k].params["count"] for k in range(4)) for h in heads]
    assert len(set(keys)) == 24
    # the candidates closest to the rules head come first: one change before two
    changed = [sum(c != 6 for c in key) for key in keys]
    assert changed == sorted(changed) and changed[:9] == [0] + [1] * 8
    assert len(cd.candidate_heads(rules, decisions, limit=5)) == 5


def test_a_nested_stack_the_pieces_contradict_makes_a_nested_decision():
    host, nest = _points("host", "star", 6, 20, 120, 1, 10), _points("nest", "star", 6, 20, 80, 2, 11)
    rules = _model(host, nest)
    grps = [_group("block", 30, 80, 2, 6)]  # the model reads the nested star's band as blocks
    decisions = cd.open_decisions(rules, grps, _reading())
    assert [(d.kind, d.part_ids, d.options) for d in decisions] == [("nested", ("nest", "host"),
                                                                       ("with", "without"))]
    heads = cd.candidate_heads(rules, decisions)
    assert [[q.id for q in h.parts if q.role == "points"] for h in heads] == [["host", "nest"], ["host"]]
    # chevrons there support the stack; an unanswered piece says nothing
    assert cd.open_decisions(rules, [_group("chevron", 30, 80, 2, 6)], _reading()) == []
    assert cd.open_decisions(rules, [_group("other", 30, 80, 2, 6)], _reading()) == []


def test_a_centre_the_model_reads_otherwise_makes_a_center_decision():
    rules = _model(_points("blocks", "diamond", 6, 60, 100, 2, 20), rose=False)
    decisions = cd.open_decisions(rules, [_group("rose", 0, 20, 7, 1)], _reading())
    assert [(d.kind, d.part_ids, d.options) for d in decisions] == [("center", ("center",), ("badge", "rose"))]
    assert [h.rose for h in cd.candidate_heads(rules, decisions)] == [False, True]
    assert cd.open_decisions(rules, [_group("badge", 0, 20, 7, 1)], _reading()) == []
    # no centre drawn: nothing to flip
    assert cd.open_decisions(_model(center=False), [_group("rose", 0, 20, 7, 1)], _reading()) == []


def test_photo_stacking_records_each_voted_pair_on_both_parts(monkeypatch):
    star, blocks = _points("star-a", "star", 6, 40, 120, 1, 10), _points("blocks", "diamond", 6, 60, 100, 2, 20)
    third = _points("third", "petal", 6, 30, 50, 3, 30)
    parts = [star, blocks, third]
    big = int(parts_head.STACK_MIN_UNITS2)
    votes = [(0, 1, big, 10), (0, 2, big // 2, big // 2), (1, 2, 1, 2)]  # star over blocks, even, too few
    monkeypatch.setattr(parts_head, "stack_votes", lambda *a, **k: votes)
    parts_head.photo_stacking(parts, None, None, None, set())
    assert (star.params["vote_blocks"], blocks.params["vote_star-a"]) == ("over", "under")
    assert (star.params["vote_third"], third.params["vote_star-a"]) == ("undecided", "undecided")
    assert (blocks.params["vote_third"], third.params["vote_blocks"]) == ("undecided", "undecided")
    assert star.z > blocks.z
