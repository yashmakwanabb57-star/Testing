"""ML-4 candidates: the open head decisions the reader's groups and answers raise against the rules
head, and the candidate heads they span (spec "Candidates", rulings 1, 2, 5 and 6).

A rules points layer is *explained* by a repeat group when its radial span overlaps the group's span
by at least EXPLAIN_SHARE of the group's span and its colour group is the group's (or the group has
none); the best overlap wins. Decisions, in this order (at most MAX_DECISIONS; each lists the rules'
choice first, so option 0 everywhere is the rules head):

- count: an explained layer whose count differs from the group's, or a group with alternatives;
  options the rules count, the group count and its alternatives (deduplicated, at most MAX_OPTIONS).
- order: a touching pair the reader answered (a or b) whose pieces lie in groups explaining two
  points layers of different colours, where the H12 vote (`params["vote_<other id>"]`, written by
  `parts_head.photo_stacking`) contradicts the answer or is undecided; options the id on top (the
  rules' first). A pair the vote never measured (no drawn overlap) is no decision: a z swap there does
  not change the drawing. An agreeing vote is none either (ruling 5).
- nested: an R19 stack (a star nested on another star's axes, `parts_head.nested_pairs`) whose band a
  group explains under a kind other than chevron or star; options "with" (the rules) and "without".
  The R19 code path (`parts_head._nested_runs`) reads the photo's polar labels and cannot build a
  stack from a model, so chevrons on a star's axes where the rules drew none open no decision.
- center: the reader's badge or rose group against the rules' centre state (`model.rose`).
- layer_kind: an explained layer whose kind differs from the group's mapped kind (KIND_MAP; ruling 2).
  It never changes candidate 0 (Review Focus 2).

`candidate_heads` spans the cross product, the candidates with the fewest changes first, at most
`limit`; index 0 is a deep copy of the rules head.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass

from rosette.model import Part, RosetteModel
from rosette.perception.reader import Reading
from rosette.vision.groups import Group
from rosette.vision.parts_head import nested_pairs

KINDS = ("count", "order", "nested", "center", "layer_kind")
MAX_DECISIONS = 4
MAX_OPTIONS = 3
MAX_CANDIDATES = 24
EXPLAIN_SHARE = 0.5  # a layer explains a group overlapping at least this share of the group's span
KIND_MAP = {"star": "star", "petal": "petal", "loop": "bow", "block": "diamond"}  # reader -> points kind
NESTED_KINDS = ("chevron", "star")  # reader kinds that support an R19 stack
CENTER_KINDS = ("badge", "rose")
VOTE_KEY = "vote_{}"  # params key of the H12 verdict against another part: over | under | undecided


@dataclass
class Decision:
    kind: str  # one of KINDS
    part_ids: tuple[str, ...]
    options: tuple  # the rules' choice first
    source: str


def _span_share(group: Group, q: Part) -> float:
    lo, hi = float(group.r_inner), float(group.r_outer)
    width = hi - lo
    if width <= 0:
        return 0.0
    over = min(hi, float(q.params["r_outer"])) - max(lo, float(q.params["r_inner"]))
    return max(0.0, over) / width


def _explained(rules: RosetteModel, group: Group) -> Part | None:
    """The rules points layer a repeat group explains, or None."""
    if group.count_source == "single":
        return None
    best, share = None, 0.0
    for q in rules.parts:
        if q.role != "points" or (group.color_group is not None and q.color_group != group.color_group):
            continue
        s = _span_share(group, q)
        if s >= EXPLAIN_SHARE and s > share:
            best, share = q, s
    return best


def _stacks(rules: RosetteModel) -> dict[str, str]:
    """{front id: host id} of the R19 stacks: stars nested on another star's axes."""
    idx = [k for k, q in enumerate(rules.parts) if q.role == "points" and q.params.get("kind") == "star"]
    out: dict[str, str] = {}
    for a, b in nested_pairs(rules.parts, idx):
        out.setdefault(rules.parts[a].id, rules.parts[b].id)
    return out


def _count_decisions(pairs: list[tuple[Group, Part]]) -> list[Decision]:
    out: dict[str, Decision] = {}
    for g, q in pairs:
        if g.kind == "chevron":
            continue  # a chevron band is a nested question, not a count
        rules_n = int(q.params["count"])
        if g.count == rules_n and not g.alternatives:
            continue
        prev = out[q.id].options if q.id in out else (rules_n,)
        opts = tuple(dict.fromkeys([*prev, int(g.count), *map(int, g.alternatives)]))[:MAX_OPTIONS]
        if len(opts) > 1:
            out[q.id] = Decision("count", (q.id,), opts, g.count_source)
    return list(out.values())


def _vote(a: Part, b: Part) -> str | None:
    """The H12 verdict of a against b: "over", "under", "undecided", or None when not voted."""
    return a.params.get(VOTE_KEY.format(b.id))


def _order_decisions(rules: RosetteModel, groups: list[Group], reading: Reading,
                     layer_of: dict[int, Part]) -> list[Decision]:
    group_of = {i: k for k, g in enumerate(groups) for i in g.pieces}  # piece index -> group
    order = {id(q): k for k, q in enumerate(rules.parts)}
    out: dict[frozenset, Decision] = {}
    for pr in reading.pairs:
        if pr.on_top not in ("a", "b"):
            continue
        ga = group_of.get(reading.piece_index.get(pr.a, -1))
        gb = group_of.get(reading.piece_index.get(pr.b, -1))
        qa, qb = layer_of.get(ga), layer_of.get(gb)
        if qa is None or qb is None or qa is qb or qa.color_group == qb.color_group:
            continue
        top, low = (qa, qb) if pr.on_top == "a" else (qb, qa)
        verdict = _vote(top, low)
        if verdict is None or verdict == "over":
            continue
        key = frozenset((qa.id, qb.id))
        if key in out:
            continue
        rt, rl = sorted((qa, qb), key=lambda q: (q.z, order[id(q)]), reverse=True)  # the rules' top first
        out[key] = Decision("order", (rt.id, rl.id), (rt.id, rl.id),
                            "vote contradicted" if verdict == "under" else "vote undecided")
    return list(out.values())


def _nested_decisions(rules: RosetteModel, pairs: list[tuple[Group, Part]]) -> list[Decision]:
    stacks = _stacks(rules)
    out: dict[str, Decision] = {}
    for g, q in pairs:
        if q.id in stacks and g.kind in KIND_MAP and g.kind not in NESTED_KINDS and q.id not in out:
            out[q.id] = Decision("nested", (q.id, stacks[q.id]), ("with", "without"), "reader")
    return list(out.values())


def _center_decisions(rules: RosetteModel, groups: list[Group]) -> list[Decision]:
    center = next((q for q in rules.parts if q.role == "center"), None)
    read = next((g.kind for g in groups if g.kind in CENTER_KINDS), None)
    state = "rose" if rules.rose else "badge"
    if center is None or read is None or read == state:
        return []
    return [Decision("center", (center.id,), (state, read), "reader")]


def _kind_decisions(pairs: list[tuple[Group, Part]], skip: set[str]) -> list[Decision]:
    out: dict[str, Decision] = {}
    for g, q in pairs:
        mapped = KIND_MAP.get(g.kind)
        if mapped is None or q.id in skip or q.id in out or mapped == q.params.get("kind"):
            continue
        out[q.id] = Decision("layer_kind", (q.id,), (q.params["kind"], mapped), "reader")
    return list(out.values())


def open_decisions(rules: RosetteModel, groups: list[Group], reading: Reading) -> list[Decision]:
    """The open head decisions (module docstring), at most MAX_DECISIONS, ordered count, order,
    nested, center, layer_kind."""
    layer_of: dict[int, Part] = {}
    for k, g in enumerate(groups):
        q = _explained(rules, g)
        if q is not None:
            layer_of[k] = q
    pairs = [(groups[k], q) for k, q in layer_of.items()]
    nested = _nested_decisions(rules, pairs)
    out = (_count_decisions(pairs) + _order_decisions(rules, groups, reading, layer_of) + nested
           + _center_decisions(rules, groups) + _kind_decisions(pairs, {d.part_ids[0] for d in nested}))
    return out[:MAX_DECISIONS]


def candidate_choices(decisions: list[Decision], limit: int = MAX_CANDIDATES) -> list[tuple[int, ...]]:
    """The option index per decision of each candidate: the cross product, the fewest changes from
    the rules head first (stable), at most `limit`; the first is all zeros (the rules head)."""
    combos = list(itertools.product(*(range(len(d.options)) for d in decisions)))
    combos.sort(key=lambda c: sum(i != 0 for i in c))  # stable: product order within a tier
    return combos[:max(1, limit)]


def _part(model: RosetteModel, pid: str) -> Part | None:
    return next((q for q in model.parts if q.id == pid), None)


def _apply(model: RosetteModel, d: Decision, option) -> None:
    if d.kind == "count":
        q = _part(model, d.part_ids[0])
        if q is not None:
            q.params["count"] = int(option)  # radii and rotation kept
    elif d.kind == "order":
        top, low = _part(model, option), _part(model, next(p for p in d.part_ids if p != option))
        if top is None or low is None or top.z > low.z:
            return
        if top.z == low.z:
            top.z += 1
        else:
            top.z, low.z = low.z, top.z
    elif d.kind == "nested":
        if option == "without":
            model.parts = [q for q in model.parts if q.id != d.part_ids[0]]
    elif d.kind == "center":
        model.rose = option == "rose"
    elif d.kind == "layer_kind":
        q = _part(model, d.part_ids[0])
        if q is not None:
            q.params["kind"] = option
    else:
        raise ValueError(f"unknown decision kind {d.kind}")


def candidate_heads(rules: RosetteModel, decisions: list[Decision], limit: int = MAX_CANDIDATES) -> list[RosetteModel]:
    """Candidate heads (`candidate_choices` order): index 0 a deep copy of `rules`, each other one
    option per decision applied to a deep copy (a nested removal last, so the other decisions still
    find their parts)."""
    out = []
    for combo in candidate_choices(decisions, limit):
        m = copy.deepcopy(rules)
        steps = sorted(zip(decisions, combo), key=lambda t: t[0].kind == "nested")
        for d, i in steps:
            if i:
                _apply(m, d, d.options[i])
        out.append(m)
    return out
