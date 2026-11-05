"""Mine Color 1-4 assignment rules from the designer's canonical SVGs.

Usage:
  .venv/Scripts/python.exe -m tools.mine_rules          write rosette/data/assign_rules.json, print the table
  .venv/Scripts/python.exe -m tools.mine_rules --loo    also leave-one-out validation of assign()

Per canonical ref, every solid-configuration color piece gets its role from
rosette.roles.classify_components (the per-piece classifier classify_roles is
built on, so a path merging head rim and tails counts as both) and its category
from its class suffix. Only the visible part of a piece counts (document order is
paint order). Tail stripes get a sub-role (outer/middle/inner) from their distance
to the head's axis, ranked as in rosette.assign.tail_positions. A Color 2 piece is
'pattern' when the Layer_3 `_2_pattern` twins cover at least half of it.

Each ref casts one vote per role it contains, split over the categories by area
share (so a design drawn in many small pieces does not outvote others), then
P(category | role) = (votes + ALPHA) / (refs + 4 ALPHA), and
pattern_role_prob[role] = (pattern votes + ALPHA) / (refs + 2 ALPHA); the latter
is informational (assign's pattern rule uses its RIBBON_ROLES).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from lxml import etree

from rosette import assign as A
from rosette.model import ColorGroup, Part, RosetteModel
from rosette.roles import RENDER_H, _render, _style, category, classify_components

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "reference"
ALPHA = 1.0
CATS = (1, 2, 3, 4)
ROLE_KEYS = ("head", "ring", "points", "center", "tail_stripe",
             "tail_outer", "tail_middle", "tail_inner", "unknown")
# Share of a Color 2 piece the pattern twins must cover for it to be pattern.
PATTERN_COVER = 0.5
_PATTERN = re.compile(r"^st\d+_2_pattern$")


def canonical() -> list[str]:
    return (REF / "canonical.txt").read_text(encoding="utf-8").split()


def pattern_mask(svg: str, height: int = RENDER_H) -> np.ndarray:
    """Union of the `_2_pattern` elements (Layer_3), in the padded render frame."""
    root = etree.fromstring(svg.encode("utf-8"), etree.XMLParser(remove_comments=True, huge_tree=True))
    for el in root.iter():
        if not isinstance(el.tag, str) or etree.QName(el).localname == "g":
            continue
        cls = (el.get("class") or "").split()
        if any(_PATTERN.match(c) for c in cls):
            _style(el, "display:inline;fill:#000000;fill-opacity:1;stroke:none")
        elif cls:
            _style(el, "display:none")
    return _render(root, height).min(axis=2) < 128


def ref_records(sid: str) -> list[dict]:
    """[{role, key, category, pattern, area, dx}] per visible piece of one ref."""
    svg = (REF / "svg" / f"{sid}.svg").read_text(encoding="utf-8")
    f, items = classify_components(svg)
    pat = pattern_mask(svg)
    cx = f.head[0]
    covered = np.zeros_like(items[0][2]) if items else None
    recs = []
    for el, role, m in reversed(items):  # later pieces paint over earlier ones
        vis = m & ~covered
        covered |= m
        n = int(vis.sum())
        if n == 0:
            continue
        cat = category(el)
        ys, xs = np.nonzero(vis)
        recs.append(dict(role=role, category=cat,
                         pattern=bool(cat == 2 and (pat & vis).sum() >= PATTERN_COVER * n),
                         area=n / f.scale ** 2, dx=float(xs.mean() - cx) / f.scale))
    recs.reverse()
    tails = [r for r in recs if r["role"] == "tail_stripe"]
    for r, pos in zip(tails, A.tail_positions([r["dx"] for r in tails])):
        r["key"] = f"tail_{pos}"
    for r in recs:
        r.setdefault("key", r["role"])
    return recs


def mine(records: dict[str, list[dict]]) -> dict:
    votes = {k: {c: 0.0 for c in CATS} for k in ROLE_KEYS}
    pvotes = {k: 0.0 for k in ROLE_KEYS}
    refs = {k: 0 for k in ROLE_KEYS}
    for sid in sorted(records):
        by_key: dict[str, list[dict]] = {}
        for r in records[sid]:
            by_key.setdefault(r["key"], []).append(r)
            if r["role"] == "tail_stripe":
                by_key.setdefault("tail_stripe", []).append(r)
        for key, rs in by_key.items():
            tot = sum(r["area"] for r in rs)
            if tot <= 0:
                continue
            refs[key] += 1
            for r in rs:
                votes[key][r["category"]] += r["area"] / tot
                if r["pattern"]:
                    pvotes[key] += r["area"] / tot
    roles = {k: {str(c): round((votes[k][c] + ALPHA) / (refs[k] + len(CATS) * ALPHA), 6) for c in CATS}
             for k in ROLE_KEYS}
    for k, dist in roles.items():  # rounding: keep each distribution summing to 1
        dist["1"] = round(1.0 - sum(v for c, v in dist.items() if c != "1"), 6)
    return {
        "alpha": ALPHA,
        "pattern_role_prob": {k: round((pvotes[k] + ALPHA) / (refs[k] + 2 * ALPHA), 6) for k in ROLE_KEYS},
        "refs": dict(refs),
        "roles": roles,
        "source": f"tools/mine_rules.py over {len(records)} canonical refs",
    }


def dump(rules: dict) -> str:
    return json.dumps(rules, indent=2, sort_keys=True) + "\n"


def table(rules: dict) -> str:
    lines = [f"{'role':<12} {'refs':>4}   {'P(1)':>6} {'P(2)':>6} {'P(3)':>6} {'P(4)':>6}   {'P(pat)':>6}"]
    for k in ROLE_KEYS:
        d = rules["roles"][k]
        lines.append(f"{k:<12} {rules['refs'][k]:>4}   " + " ".join(f"{d[str(c)]:>6.3f}" for c in CATS)
                     + f"   {rules['pattern_role_prob'][k]:>6.3f}")
    return "\n".join(lines)


# Distinct, far apart colors for the leave-one-out groups (never merged).
_LOO_RGB = [(220, 30, 30), (30, 60, 220), (240, 210, 40), (30, 160, 70), (200, 40, 200)]


def model_from_records(recs: list[dict], pattern_hint: bool = True) -> tuple[RosetteModel, dict[int, int]]:
    """Model with one group per (category, pattern) present (per category only when
    pattern_hint is False); returns it and each group's designer category."""
    keys = sorted({(r["category"], r["pattern"] and pattern_hint) for r in recs})
    gid = {k: i for i, k in enumerate(keys)}
    tot = sum(r["area"] for r in recs)
    groups = [ColorGroup(i, _LOO_RGB[i], bool(p), 0.0, sum(r["area"] for r in recs
                         if (r["category"], r["pattern"] and pattern_hint) == (c, p)) / tot, None, 0.0)
              for (c, p), i in gid.items()]
    parts = [Part(f"p{j}", r["role"], {"area": r["area"], "dx": r["dx"]},
                  gid[(r["category"], r["pattern"] and pattern_hint)], 0, 1.0) for j, r in enumerate(recs)]
    return RosetteModel(parts=parts, groups=groups), {i: c for (c, _), i in gid.items()}


def loo(records: dict[str, list[dict]], pattern_hint: bool = True) -> list[tuple[str, float, int, int, int]]:
    """Per ref: (id, area accuracy, wrong groups, flagged wrong groups, flagged right groups)."""
    out = []
    for sid in sorted(records):
        rules = mine({s: r for s, r in records.items() if s != sid})
        model, truth = model_from_records(records[sid], pattern_hint)
        got = A.assign(model, rules=rules)
        cat = {g.id: g.category for g in got.groups}
        confs = {g.id: g.confidence for g in got.groups}
        ok = sum(A.part_area(q) for q in got.parts if cat[q.color_group] == truth[q.color_group])
        acc = ok / sum(A.part_area(q) for q in got.parts)
        wrong = [g for g in cat if cat[g] != truth[g]]
        right = [g for g in cat if cat[g] == truth[g]]
        out.append((sid, acc, len(wrong), sum(confs[g] < A.FLAG_BELOW for g in wrong),
                    sum(confs[g] < A.FLAG_BELOW for g in right)))
    return out


def main(argv: list[str]) -> None:
    records = {sid: ref_records(sid) for sid in canonical()}
    used = {sid: sorted({r["category"] for r in rs}) for sid, rs in records.items()}
    skipped = [s for s, c in used.items() if c != list(range(1, len(c) + 1))]
    print(f"refs whose colors are not 1..k: {skipped or 'none'}")
    rules = mine(records)
    A.RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(A.RULES_PATH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(dump(rules))
    print(table(rules))
    if "--loo" in argv:
        for hint in (True, False):
            res = loo(records, hint)
            print(f"\nleave-one-out, pattern hint {'on' if hint else 'off'} (SCALE={A.SCALE}):")
            for sid, acc, nw, fw, fr in res:
                print(f"  {sid:<8} acc={acc:.3f} wrong_groups={nw} flagged_wrong={fw} flagged_right={fr}")
            accs = [r[1] for r in res]
            perfect = sum(a > 0.999 for a in accs)
            nw = sum(r[2] for r in res)
            print(f"  mean={np.mean(accs):.3f} all-right refs={perfect}/{len(res)} "
                  f"wrong groups={nw} flagged={sum(r[3] for r in res)} "
                  f"right groups flagged={sum(r[4] for r in res)}")


if __name__ == "__main__":
    main(sys.argv[1:])
