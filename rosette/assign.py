"""Assign the shop's Color 1-4 to the detected color groups.

assign(model) -> RosetteModel (a new model; the input is left untouched) with each
used group's `category` (1-4) and `confidence` set. There is no documented house
rule, so the role -> category statistics are mined from the designer's own SVGs by
tools/mine_rules.py into rosette/data/assign_rules.json.

1. Groups no part uses get no category (warned).
2. More than 4 used groups: the two nearest in CIELAB are merged (the smaller into
   the larger) until 4 remain; each merge is warned with both colors as hex.
3. Pattern rule: a pattern group must be Color 2 - but only if it covers a real
   ribbon role (tail stripe, points, ring) over PATTERN_MIN_RIBBON_SHARE of the
   rosette; a 'pattern' group on head print/logos (unknown, center, head pieces)
   is treated as plain and warned. Of several such groups only the one with the
   most ribbon area is Color 2; the others are flagged.
4. With k used groups, every one-to-one mapping of groups onto Colors 1..k is
   scored: sum over parts of area share * log P(category | role). The designer
   files never skip a color (k colors are always 1..k), so neither does this.
5. A group's confidence is the two-way softmax between the best mapping and the
   best mapping giving that group another color, at evidence scale SCALE; a tie
   gives 0.5. Below FLAG_BELOW the group is flagged for confirmation.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from rosette.model import ColorGroup, Part, RosetteModel

RULES_PATH = Path(__file__).resolve().parent / "data" / "assign_rules.json"
# Group confidence below this is flagged for the user to confirm.
FLAG_BELOW = 0.6
# Evidence scale of the softmax margin (score difference per unit area share).
# Chosen by a leave-one-out sweep over the 35 canonical refs (tools/mine_rules.py
# --loo; 1/2/3/4/6/8/12 tried). Wrong-but-unflagged groups with / without the
# pattern hint: 2/2 at 1, 4/3 at 2, 5/4 at 4, 5/15 at 12; correct groups flagged:
# 15/30 at 1, 13/22 at 2. 2 is the knee: near the fewest silent errors without
# flagging every other correct group.
SCALE = 2.0
# Most groups the shop can color.
MAX_COLORS = 4
# Roles in which a pattern texture is a real patterned ribbon.
RIBBON_ROLES = ("tail_stripe", "points", "ring")
# A pattern group needs at least this share of the rosette's part area in ribbon
# roles to force Color 2 (printed head text on A1-15 / A7-15 covers none).
PATTERN_MIN_RIBBON_SHARE = 0.03
# Tail stripes whose distances from the axis differ by at most this (canvas units)
# share a position level.
TAIL_POS_TOL = 3.0
# Confidence cap of a pattern group that lost Color 2 to a larger one.
LOST_PATTERN_CONF = 0.5


@lru_cache(maxsize=None)
def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_rules(path: Path = RULES_PATH) -> dict:
    return _load(str(path))


def hex_rgb(rgb) -> str:
    return "#" + "".join(f"{int(v):02x}" for v in rgb)


def _lab(rgb) -> np.ndarray:
    px = np.array([[rgb]], np.float32) / 255.0
    return cv2.cvtColor(px, cv2.COLOR_RGB2Lab)[0, 0]


def tail_positions(dxs: list[float]) -> list[str]:
    """outer / middle / inner per tail stripe from its distance to the axis: distinct
    distance levels (TAIL_POS_TOL apart) ranked from the axis out; the outermost third
    of the levels is outer, the innermost third inner. A single level is outer."""
    d = [abs(float(x)) for x in dxs]
    levels: list[float] = []
    for v in sorted(d):
        if not levels or v - levels[-1] > TAIL_POS_TOL:
            levels.append(v)
        else:
            levels[-1] = v  # chain: compare with the level's latest member
    out = []
    for v in d:
        lv = next(i for i, top in enumerate(levels) if v <= top + 1e-9)
        f = lv / (len(levels) - 1) if len(levels) > 1 else 1.0
        out.append("outer" if f > 2 / 3 else "inner" if f < 1 / 3 else "middle")
    return out


def _fracs(params: dict) -> list[float]:
    s = str(params.get("stripes", "") or "")
    try:
        f = [float(x) for x in s.split(",") if x.strip()]
    except ValueError:
        f = []
    return f or [1.0]


def stripe_dx(params: dict) -> float:
    """Signed x offset (canvas units) of a tail stripe's middle from the axis."""
    if "dx" in params:
        return float(params["dx"])
    n = int(params.get("tails_count", 1) or 1)
    k = int(params.get("tail_index", 0))
    i = int(params.get("stripe_index", 0))
    spread = float(params.get("spread_deg", 0.0) or 0.0)
    ang = math.radians(-spread / 2 + k * spread / (n - 1)) if n > 1 else 0.0
    ty, by = float(params.get("top_y", 0.0)), float(params.get("bottom_y", 0.0))
    w = (float(params.get("top_width", 0.0)) + float(params.get("bottom_width", 0.0))) / 2
    f = _fracs(params)
    i = min(max(i, 0), len(f) - 1)
    off = sum(f[:i]) / sum(f) + f[i] / sum(f) / 2 - 0.5
    return math.tan(ang) * (by - ty) / 2 + off * w


def part_area(part: Part, parts: list[Part] | None = None) -> float:
    """Visible area (canvas units^2) of a part: params['area'] when given, otherwise
    estimated from its geometry (the head minus the head parts drawn over it)."""
    p = part.params
    if "area" in p:
        return float(p["area"])
    pi = math.pi
    if part.role == "center" or part.role == "unknown":
        return pi * float(p.get("r", 0.0)) ** 2
    if part.role == "ring":
        return pi * max(0.0, float(p["r_outer"]) ** 2 - float(p["r_inner"]) ** 2)
    if part.role == "points":
        ro, ri = float(p["r_outer"]), float(p["r_inner"])
        annulus = pi * max(0.0, ro * ro - ri * ri)
        if "width_ratio" in p and "count" in p:
            return min(annulus, int(p["count"]) * 0.5 * float(p["width_ratio"]) * (ro - ri) ** 2)
        return 0.5 * annulus
    if part.role == "tail_stripe":
        f = _fracs(p)
        i = min(max(int(p.get("stripe_index", 0)), 0), len(f) - 1)
        w = (float(p.get("top_width", 0.0)) + float(p.get("bottom_width", 0.0))) / 2
        return f[i] / sum(f) * w * max(0.0, float(p.get("bottom_y", 0.0)) - float(p.get("top_y", 0.0)))
    if part.role == "head":
        disc = pi * float(p.get("r", 0.0)) ** 2
        over = sum(part_area(q) for q in parts or [] if q.role in ("ring", "points", "center"))
        return max(0.1 * disc, disc - over)
    return 0.0


def role_keys(parts: list[Part]) -> list[str]:
    """Rule key per part: its role, tail stripes refined to tail_outer/middle/inner."""
    keys = [q.role for q in parts]
    idx = [i for i, q in enumerate(parts) if q.role == "tail_stripe"]
    for i, pos in zip(idx, tail_positions([stripe_dx(parts[i].params) for i in idx])):
        keys[i] = f"tail_{pos}"
    return keys


def _prob(rules: dict, key: str, cat: int) -> float:
    roles = rules["roles"]
    dist = roles.get(key) or roles.get("tail_stripe" if key.startswith("tail_") else "unknown")
    return float(dist[str(cat)])


def _merge(out: RosetteModel, used: list[int]) -> list[int]:
    """Merge nearest used groups (CIELAB) until MAX_COLORS remain; returns used ids."""
    by_id = {g.id: g for g in out.groups}
    used = list(used)
    while len(used) > MAX_COLORS:
        best = None
        for a, b in itertools.combinations(used, 2):
            d = float(np.linalg.norm(_lab(by_id[a].rgb) - _lab(by_id[b].rgb)))
            if best is None or d < best[0]:
                best = (d, a, b)
        _, a, b = best
        ga, gb = by_id[a], by_id[b]
        keep, gone = (ga, gb) if (ga.area, -ga.id) >= (gb.area, -gb.id) else (gb, ga)
        total = keep.area + gone.area
        if total > 0:
            keep.texture = (keep.texture * keep.area + gone.texture * gone.area) / total
        keep.area = total
        for q in out.parts:
            if q.color_group == gone.id:
                q.color_group = keep.id
        out.groups = [g for g in out.groups if g.id != gone.id]
        used.remove(gone.id)
        out.warnings.append(f"Merged color {hex_rgb(gone.rgb)} into {hex_rgb(keep.rgb)}: the photo "
                            f"has more than {MAX_COLORS} color groups - check the colors.")
    return used


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0


def assign(model: RosetteModel, rules: dict | None = None) -> RosetteModel:
    rules = rules if rules is not None else load_rules()
    out = copy.deepcopy(model)
    by_id = {g.id: g for g in out.groups}
    used = sorted({q.color_group for q in out.parts if q.color_group in by_id})
    for g in out.groups:
        g.category, g.confidence = None, 0.0
        if g.id not in used:
            out.warnings.append(f"Color {hex_rgb(g.rgb)} is not used by any part - it gets no Color.")
    if not used:
        return out
    used = _merge(out, used)
    by_id = {g.id: g for g in out.groups}

    areas = [part_area(q, out.parts) for q in out.parts]
    total = sum(areas) or 1.0
    keys = role_keys(out.parts)
    k = len(used)
    col = {gid: i for i, gid in enumerate(used)}
    # per group, per color: summed area share * log P(color | role)
    gs = np.zeros((k, k))
    ribbon = np.zeros(k)
    for q, a, key in zip(out.parts, areas, keys):
        if q.color_group not in col:
            continue
        i = col[q.color_group]
        share = a / total
        for c in range(1, k + 1):
            gs[i, c - 1] += share * math.log(_prob(rules, key, c))
        if q.role in RIBBON_ROLES:
            ribbon[i] += share

    pattern = [i for i, gid in enumerate(used) if by_id[gid].pattern]
    real = [i for i in pattern if ribbon[i] >= PATTERN_MIN_RIBBON_SHARE]
    for i in pattern:
        if i not in real:
            out.warnings.append(f"Color {hex_rgb(by_id[used[i]].rgb)} looks like a pattern but covers no "
                                "ribbon (head print or small regions) - not forced to Color 2.")
    forced = max(real, key=lambda i: (ribbon[i], -used[i])) if real else None
    losers = [i for i in real if i != forced]
    for i in losers:
        out.warnings.append(f"Color {hex_rgb(by_id[used[i]].rgb)} also looks like a pattern, but only one "
                            "color can be the Color 2 pattern - check its Color.")
    if forced is not None and k < 2:
        out.warnings.append(f"Color {hex_rgb(by_id[used[forced]].rgb)} looks like a pattern, but the "
                            "rosette has a single color - it stays Color 1.")
        forced = None

    maps = []
    for perm in itertools.permutations(range(1, k + 1)):
        if forced is not None and perm[forced] != 2:
            continue
        maps.append((float(sum(gs[i, perm[i] - 1] for i in range(k))), perm))
    best_s, best = max(maps, key=lambda m: m[0])  # first of equal scores: stable order
    for i, gid in enumerate(used):
        g = by_id[gid]
        g.category = int(best[i])
        alts = [s for s, perm in maps if perm[i] != best[i]]
        c = _sigmoid(SCALE * (best_s - max(alts))) if alts else 1.0
        if i in losers:
            c = min(c, LOST_PATTERN_CONF)
        g.confidence = round(float(c), 3)
    return out
