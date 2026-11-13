"""ML-4 candidates: the open head decisions the reader's groups and answers raise against the rules
head, and the candidate heads they span (spec "Candidates", rulings 1, 2, 5 and 6).

A rules points layer is *explained* by a repeat group when its radial span overlaps the group's span
by at least EXPLAIN_SHARE of the group's span and its colour group is the group's (or the group has
none); the best overlap wins. Decisions, in this order (at most MAX_DECISIONS; each lists the rules'
choice first, so option 0 everywhere is the rules head):

- count: an explained layer whose count differs from the group's, or a group with alternatives;
  options the rules count, the group count and its alternatives within COUNT_MIN..COUNT_MAX
  (deduplicated, at most MAX_OPTIONS). A count candidate changes the layer as
  `layer_classify._set_count` does (width_ratio scaled by old / new, rotation and radii kept) and
  gives the same count to the points layers on its axes with its old count (edge strips, a backing,
  a nested front star: rotation within NEST_AXIS_DEG modulo the step, same centre).
- order: a touching pair the reader answered (a or b) whose pieces lie in groups explaining two
  points layers of different colours, where the H12 vote (`params["vote_<other id>"]`, written by
  `parts_head.photo_stacking`) contradicts the answer or is undecided; options the id on top (the
  rules' first). A pair the vote never measured (no drawn overlap) is no decision: a z swap there does
  not change the drawing. An agreeing vote is none either (ruling 5). The candidate swaps the two z
  values, so the layers between them change sides too (the brief's "swaps z"); on equal z the part
  put on top moves just after the other in the part list (the shop's layer order), keeping every z.
- nested: a stack the rules drew as one (`drawn_stacks`: an R19 nested star, or an H16 inner star on
  a front layer's axes) whose band a group names chevron, block, loop or petal; options "with" (the
  rules) and "without" (the front removed). A star or unanswered reading opens none. The R19 code path
  (`parts_head._nested_runs`) reads the photo's polar labels and cannot build a stack from a model, so
  chevrons on a star's axes where the rules drew none open no decision.
- center: the reader's badge or rose group (the larger one by its span's area when both were read)
  against the rules' centre state (`model.rose`).
- layer_kind: an explained layer whose kind differs from the group's mapped kind (KIND_MAP; ruling 2),
  also on a nested front (the reader's kind stays an option beside removing the stack). It never
  changes candidate 0 (Review Focus 2).

`candidate_heads` spans the cross product, the candidates with the fewest changes first, identical
heads dropped, at most `limit`; index 0 is a deep copy of the rules head.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
from dataclasses import dataclass

from rosette.model import Part, RosetteModel
from rosette.perception.reader import Reading
from rosette.vision.groups import Group
from rosette.vision.layer_classify import _set_count
from rosette.vision.parts_head import (BLOB_UNCONFIRMED_CONF, FRONT_ID, MAX_COUNT, NEST_AXIS_DEG, NEST_UNITS,
                                       _same_axes)

KINDS = ("count", "order", "nested", "center", "layer_kind")
MAX_DECISIONS = 4
MAX_OPTIONS = 3
MAX_CANDIDATES = 24
EXPLAIN_SHARE = 0.5  # a layer explains a group overlapping at least this share of the group's span
KIND_MAP = {"star": "star", "petal": "petal", "loop": "bow", "block": "diamond"}  # reader -> points kind
NESTED_OPENERS = ("chevron", "block", "loop", "petal")  # reader kinds that question a drawn stack
CENTER_KINDS = ("badge", "rose")
COUNT_MIN, COUNT_MAX = 2, MAX_COUNT  # the counts the head detector draws
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


def _concentric(a: Part, b: Part) -> bool:
    qa, qb = a.params, b.params
    return abs(float(qa["cx"]) - float(qb["cx"])) <= 1 and abs(float(qa["cy"]) - float(qb["cy"])) <= 1


def _on_axes(q: Part, of: Part) -> bool:
    """q has of's count and axes (rotation within NEST_AXIS_DEG modulo the step) and centre."""
    n = int(of.params["count"])
    return _concentric(q, of) and _same_axes(q.params, n, float(of.params["rotation_deg"]), NEST_AXIS_DEG)


def drawn_stacks(rules: RosetteModel) -> dict[str, str]:
    """{front id: host id} of the stacks the rules drew as stacks: a star on a larger star's axes
    (another colour, its tips NEST_UNITS inside, above it) that is either R19's nested star (the
    host's r_inner, confidence at most BLOB_UNCONFIRMED_CONF: its z is assumed) or H16's inner star
    (both front layers, FRONT_ID). Any other concentric star pair is two layers, not a stack. The
    host is the smallest such star."""
    stars = [q for q in rules.parts if q.role == "points" and q.params.get("kind") == "star"]
    out: dict[str, str] = {}
    for f in stars:
        hosts = []
        for h in stars:
            if (h is f or h.color_group == f.color_group or not _on_axes(f, h) or f.z <= h.z
                    or float(f.params["r_outer"]) + NEST_UNITS > float(h.params["r_outer"])):
                continue
            r19 = (float(f.params["r_inner"]) == float(h.params["r_inner"])
                   and float(f.confidence) <= BLOB_UNCONFIRMED_CONF)
            h16 = f.id.startswith(FRONT_ID) and h.id.startswith(FRONT_ID)
            if r19 or h16:
                hosts.append(h)
        if hosts:
            out[f.id] = min(hosts, key=lambda h: float(h.params["r_outer"])).id
    return out


def _count_decisions(pairs: list[tuple[Group, Part]]) -> list[Decision]:
    out: dict[str, Decision] = {}
    for g, q in pairs:
        if g.kind == "chevron":
            continue  # a chevron band is a nested question, not a count
        rules_n = int(q.params["count"])
        read = [int(g.count), *map(int, g.alternatives)]
        read = [n for n in read if COUNT_MIN <= n <= COUNT_MAX]
        prev = out[q.id].options if q.id in out else (rules_n,)
        opts = tuple(dict.fromkeys([*prev, *read]))[:MAX_OPTIONS]
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
    stacks = drawn_stacks(rules)
    out: dict[str, Decision] = {}
    for g, q in pairs:
        if q.id in stacks and g.kind in NESTED_OPENERS and q.id not in out:
            out[q.id] = Decision("nested", (q.id, stacks[q.id]), ("with", "without"), "reader")
    return list(out.values())


def _center_decisions(rules: RosetteModel, groups: list[Group]) -> list[Decision]:
    center = next((q for q in rules.parts if q.role == "center"), None)
    read = [g for g in groups if g.kind in CENTER_KINDS]
    if center is None or not read:
        return []
    # the larger centre group by its span's area (the first on a tie)
    best = max(read, key=lambda g: math.pi * (float(g.r_outer) ** 2 - float(g.r_inner) ** 2))
    state = "rose" if rules.rose else "badge"
    if best.kind == state:
        return []
    return [Decision("center", (center.id,), (state, best.kind), "reader")]


def _kind_decisions(pairs: list[tuple[Group, Part]]) -> list[Decision]:
    out: dict[str, Decision] = {}
    for g, q in pairs:
        mapped = KIND_MAP.get(g.kind)
        if mapped is None or q.id in out or mapped == q.params.get("kind"):
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
    out = (_count_decisions(pairs) + _order_decisions(rules, groups, reading, layer_of)
           + _nested_decisions(rules, pairs) + _center_decisions(rules, groups) + _kind_decisions(pairs))
    return out[:MAX_DECISIONS]


def _part(model: RosetteModel, pid: str) -> Part | None:
    return next((q for q in model.parts if q.id == pid), None)


def _apply(model: RosetteModel, d: Decision, option) -> None:
    if d.kind == "count":
        q = _part(model, d.part_ids[0])
        if q is None:
            return
        old = int(q.params["count"])
        followers = [p for p in model.parts if p is not q and p.role == "points"
                     and int(p.params.get("count", 0)) == old and _on_axes(p, q)]
        for p in [q, *followers]:
            _set_count(p, int(option), float(p.params["rotation_deg"]))  # rotation and radii kept
    elif d.kind == "order":
        top, low = _part(model, option), _part(model, next(p for p in d.part_ids if p != option))
        if top is None or low is None or top.z > low.z:
            return
        if top.z == low.z:  # the shop's layer order decides: just after the other, no z changes
            model.parts.remove(top)
            model.parts.insert(model.parts.index(low) + 1, top)
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


def _spanned(rules: RosetteModel, decisions: list[Decision], limit: int) -> list[tuple[tuple[int, ...], RosetteModel]]:
    """(option index per decision, head) of the cross product, the fewest changes first (stable),
    heads identical to an earlier one dropped, at most `limit`. A nested removal is applied last,
    so the other decisions still find their parts."""
    combos = list(itertools.product(*(range(len(d.options)) for d in decisions)))
    combos.sort(key=lambda c: sum(i != 0 for i in c))  # stable: product order within a tier
    out, seen = [], set()
    for combo in combos:
        if len(out) >= max(1, limit):
            break
        m = copy.deepcopy(rules)
        for d, i in sorted(zip(decisions, combo), key=lambda t: t[0].kind == "nested"):
            if i:
                _apply(m, d, d.options[i])
        key = json.dumps(m.to_json(), sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append((combo, m))
    return out


def candidate_choices(rules: RosetteModel, decisions: list[Decision],
                      limit: int = MAX_CANDIDATES) -> list[tuple[int, ...]]:
    """The option index per decision of each candidate of `candidate_heads` (same order); the first
    is all zeros (the rules head)."""
    return [c for c, _ in _spanned(rules, decisions, limit)]


def candidate_heads(rules: RosetteModel, decisions: list[Decision], limit: int = MAX_CANDIDATES) -> list[RosetteModel]:
    """Candidate heads: index 0 a deep copy of `rules`, each other one option per decision applied to
    a deep copy (`_spanned`: fewest changes first, identical heads dropped, at most `limit`)."""
    return [m for _, m in _spanned(rules, decisions, limit)]
