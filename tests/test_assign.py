import json
import math

import pytest

from rosette import assign as A
from rosette.model import ColorGroup, Part, RosetteModel

ROLE_KEYS = {"head", "ring", "points", "center", "tail_stripe",
             "tail_outer", "tail_middle", "tail_inner", "unknown"}


def group(i, rgb, pattern=False, area=0.25):
    return ColorGroup(i, rgb, pattern, 10.0 if pattern else 1.0, area, None, 0.0)


def head(g, r=120.0):
    return Part("head", "head", {"cx": 300, "cy": 175, "r": r, "pinked": 1}, g, 10, 0.9)


def ring(pid, g, ro, ri):
    return Part(pid, "ring", {"cx": 300, "cy": 175, "r_outer": ro, "r_inner": ri}, g, 30, 0.9)


def points(pid, g):
    return Part(pid, "points", {"cx": 300, "cy": 175, "kind": "star", "count": 6, "r_inner": 50,
                                "r_outer": 100, "rotation_deg": 0, "width_ratio": 0.8}, g, 20, 0.9)


def center(g, r=30.0):
    return Part("center", "center", {"cx": 300, "cy": 175, "r": r}, g, 50, 0.9)


def unknown(pid, g, r=15.0):
    return Part(pid, "unknown", {"cx": 300, "cy": 120, "r": r}, g, 60, 0.2)


def tails(groups_per_stripe, fracs=None):
    """One tail (tails_count=1) split into stripes left to right. By the mining
    convention the stripe nearest the axis is 'inner': tails([a, b, c]) gives
    outer, inner, outer."""
    n = len(groups_per_stripe)
    fracs = fracs or [1.0 / n] * n
    stripes = ",".join(f"{f:.4f}" for f in fracs)
    common = {"tails_count": 1, "top_y": 300, "bottom_y": 825, "top_width": 140,
              "bottom_width": 300, "spread_deg": 0, "stripes": stripes}
    return [Part(f"tail-0-{i}", "tail_stripe", {"tail_index": 0, "stripe_index": i, **common}, g, 5, 0.9)
            for i, g in enumerate(groups_per_stripe)]


def cats(model):
    return {g.id: g.category for g in model.groups}


def conf(model):
    return {g.id: g.confidence for g in model.groups}


def test_flag_below_constant():
    assert A.FLAG_BELOW == 0.6


def test_rules_file_covers_roles():
    text = A.RULES_PATH.read_text(encoding="utf-8")
    rules = json.loads(text)
    # deterministic: the committed file is exactly the sorted-keys dump
    assert text == json.dumps(rules, indent=2, sort_keys=True) + "\n"
    assert "\r\n" not in text
    assert set(rules["roles"]) == ROLE_KEYS
    assert set(rules["pattern_role_prob"]) == ROLE_KEYS
    for role, dist in rules["roles"].items():
        assert set(dist) == {"1", "2", "3", "4"}
        assert all(p > 0 for p in dist.values())  # Laplace smoothed
        assert math.isclose(sum(dist.values()), 1.0, abs_tol=1e-6)
        assert 0 < rules["pattern_role_prob"][role] < 1


def test_pattern_group_is_color2():
    # tails dominate; without the pattern rule the tails group would not be Color 2
    groups = [group(0, (200, 30, 30)), group(1, (30, 30, 200), pattern=True), group(2, (240, 200, 40))]
    parts = [head(0), points("points-1", 2), *tails([1, 1, 1])]
    out = A.assign(RosetteModel(parts=parts, groups=groups))
    assert cats(out)[1] == 2
    assert sorted(cats(out).values()) == [1, 2, 3]


def test_pattern_rule_is_hard_even_against_rules():
    rules = {"roles": {k: {"1": 0.25, "2": 0.25, "3": 0.25, "4": 0.25} for k in ROLE_KEYS},
             "pattern_role_prob": {k: 0.5 for k in ROLE_KEYS}}
    for k in ("tail_outer", "tail_middle", "tail_inner", "tail_stripe"):
        rules["roles"][k] = {"1": 0.97, "2": 0.01, "3": 0.01, "4": 0.01}
    rules["roles"]["head"] = {"1": 0.01, "2": 0.97, "3": 0.01, "4": 0.01}
    groups = [group(0, (200, 30, 30)), group(1, (30, 30, 200), pattern=True)]
    parts = [head(0, r=150), *tails([1])]
    out = A.assign(RosetteModel(parts=parts, groups=groups), rules=rules)
    assert cats(out) == {0: 1, 1: 2}
    assert conf(out)[1] == 1.0  # a hard rule leaves no alternative for the pattern group


def test_shared_group_shares_category():
    groups = [group(0, (200, 30, 30)), group(1, (30, 30, 200)), group(2, (240, 200, 40))]
    parts = [head(0), ring("ring-0", 1, 60, 45), center(2), *tails([0, 1, 0])]
    out = A.assign(RosetteModel(parts=parts, groups=groups))
    c = cats(out)
    assert sorted(c.values()) == [1, 2, 3]  # injective: distinct groups, distinct colors
    by_part = {p.id: c[p.color_group] for p in out.parts}
    assert by_part["head"] == by_part["tail-0-0"] == by_part["tail-0-2"]
    assert by_part["ring-0"] == by_part["tail-0-1"]


def test_merge_over_four():
    groups = [group(0, (200, 30, 30), area=0.3), group(1, (30, 30, 200), area=0.2),
              group(2, (240, 200, 40), area=0.2), group(3, (20, 140, 60), area=0.2),
              group(4, (205, 35, 32), area=0.1)]  # 4 is nearly 0's red
    parts = [head(0), ring("ring-0", 1, 60, 45), center(2), points("points-1", 3), *tails([4, 0, 4])]
    out = A.assign(RosetteModel(parts=parts, groups=groups))
    assert len(out.groups) == 4
    assert sorted(cats(out).values()) == [1, 2, 3, 4]
    assert {p.color_group for p in out.parts} == {0, 1, 2, 3}
    merged = next(g for g in out.groups if g.id == 0)
    assert merged.area == pytest.approx(0.4)
    assert merged.confidence < A.FLAG_BELOW  # a merge is always confirmed
    w = [x for x in out.warnings if "#cd2320" in x and "#c81e1e" in x]
    assert w, out.warnings
    assert all("—" not in x and "–" not in x for x in out.warnings)


def test_merge_never_hides_pattern_group():
    # glitter red on every tail, Lab-near a plain red head: the pattern group must
    # not be swallowed by the plain one (other pairs merge first) and stays Color 2
    groups = [group(0, (200, 30, 30), area=0.3), group(1, (205, 35, 32), pattern=True, area=0.3),
              group(2, (240, 200, 40), area=0.1), group(3, (20, 140, 60), area=0.1),
              group(4, (30, 30, 200), area=0.1), group(5, (40, 40, 210), area=0.1)]
    parts = [head(0), ring("ring-0", 2, 60, 45), center(3), points("points-1", 4),
             points("points-2", 5), *tails([1, 1, 1])]
    out = A.assign(RosetteModel(parts=parts, groups=groups))
    assert len(out.groups) == 4
    tails_g = next(g for g in out.groups if g.id == out.parts[-1].color_group)
    assert tails_g.pattern
    assert tails_g.category == 2
    assert {0, 1} <= {g.id for g in out.groups}  # red head and glitter red both kept


def test_merge_keeps_input_untouched():
    groups = [group(i, rgb) for i, rgb in enumerate([(200, 30, 30), (30, 30, 200), (240, 200, 40),
                                                     (20, 140, 60), (205, 35, 32)])]
    parts = [head(0), ring("ring-0", 1, 60, 45), center(2), points("points-1", 3), *tails([4])]
    model = RosetteModel(parts=parts, groups=groups)
    A.assign(model)
    assert len(model.groups) == 5 and model.parts[-1].color_group == 4


def test_low_margin_flags():
    # two rings of identical role and area in different groups: swapping them scores
    # exactly the same, so neither may be confident
    groups = [group(0, (200, 30, 30)), group(1, (30, 30, 200)), group(2, (240, 200, 40))]
    parts = [head(0), ring("ring-0", 1, 80, 65), ring("ring-1", 2, 80, 65), *tails([0])]
    out = A.assign(RosetteModel(parts=parts, groups=groups))
    c = conf(out)
    assert c[1] < A.FLAG_BELOW and c[2] < A.FLAG_BELOW
    assert sorted(cats(out).values()) == [1, 2, 3]


def test_clear_case_is_confident():
    # one group on everything big, a second on a small center: no real alternative
    groups = [group(0, (200, 30, 30)), group(1, (30, 30, 200))]
    out = A.assign(RosetteModel(parts=[head(0), center(1, r=20), *tails([0, 0, 0])], groups=groups))
    assert all(0.0 <= v <= 1.0 for v in conf(out).values())
    assert sorted(cats(out).values()) == [1, 2]


def test_head_print_pattern_not_forced():
    # printed text in the head comes out as a small 'pattern' group on unknown/center
    # parts only: it must not force Color 2 and must be warned about
    groups = [group(0, (228, 179, 188)), group(1, (176, 137, 124), pattern=True, area=0.07)]
    parts = [head(0, r=157), unknown("unknown-1", 1, 57), unknown("unknown-2", 1, 33),
             center(1, r=7.5), ring("ring-0", 0, 55.5, 49.5), *tails([0])]
    rules = {"roles": {k: {"1": 0.25, "2": 0.25, "3": 0.25, "4": 0.25} for k in ROLE_KEYS},
             "pattern_role_prob": {k: 0.5 for k in ROLE_KEYS}}
    rules["roles"]["unknown"] = {"1": 0.7, "2": 0.1, "3": 0.1, "4": 0.1}
    rules["roles"]["center"] = {"1": 0.7, "2": 0.1, "3": 0.1, "4": 0.1}
    rules["roles"]["head"] = {"1": 0.1, "2": 0.7, "3": 0.1, "4": 0.1}
    out = A.assign(RosetteModel(parts=parts, groups=groups), rules=rules)
    assert cats(out) == {0: 2, 1: 1}  # rules decide, the head print is not forced to 2
    assert any("#b0897c" in w and "pattern" in w for w in out.warnings), out.warnings
    assert conf(out)[1] < A.FLAG_BELOW  # demoted pattern group is always confirmed


def test_head_print_pattern_ending_color2_is_flagged():
    # two groups: the head-print 'pattern' group ends Color 2 by elimination, which
    # must not look settled (A1-15)
    groups = [group(0, (228, 179, 188)), group(1, (176, 137, 124), pattern=True, area=0.07)]
    parts = [head(0, r=157), unknown("unknown-1", 1, 57), center(1, r=7.5), *tails([0])]
    out = A.assign(RosetteModel(parts=parts, groups=groups))
    assert cats(out)[1] == 2
    assert conf(out)[1] < A.FLAG_BELOW
    assert any("#b0897c" in w and "ignored" in w and "check its Color" in w for w in out.warnings)


def test_two_pattern_groups_only_one_color2():
    groups = [group(0, (200, 30, 30)), group(1, (30, 30, 200), pattern=True),
              group(2, (240, 200, 40), pattern=True)]
    parts = [head(0), points("points-1", 2), *tails([1, 0, 1], fracs=[0.35, 0.3, 0.35])]
    out = A.assign(RosetteModel(parts=parts, groups=groups))
    c = cats(out)
    assert c[1] == 2  # the larger ribbon pattern group
    assert c[2] != 2
    assert conf(out)[2] < A.FLAG_BELOW
    assert any("#f0c828" in w for w in out.warnings), out.warnings


def test_unused_group_gets_no_category():
    groups = [group(0, (200, 30, 30)), group(1, (30, 30, 200)), group(2, (240, 200, 40))]
    out = A.assign(RosetteModel(parts=[head(0), *tails([1])], groups=groups))
    assert cats(out)[2] is None
    assert sorted(v for v in cats(out).values() if v is not None) == [1, 2]
    assert any("#f0c828" in w for w in out.warnings)


def test_deterministic():
    groups = [group(0, (200, 30, 30)), group(1, (30, 30, 200)), group(2, (240, 200, 40))]
    parts = [head(0), ring("ring-0", 1, 80, 65), ring("ring-1", 2, 80, 65), *tails([0])]
    runs = [A.assign(RosetteModel(parts=parts, groups=groups)).to_json() for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_tail_positions():
    assert A.tail_positions([-50.0, 50.0]) == ["outer", "outer"]
    assert A.tail_positions([-60.0, -20.0, 0.5, 20.0, 60.0]) == \
        ["outer", "middle", "inner", "middle", "outer"]
    assert A.tail_positions([-30.0, 30.0, -1.0]) == ["outer", "outer", "inner"]


def test_stripe_keys_use_position():
    parts = tails([0, 1, 2, 1, 0])
    keys = A.role_keys(parts)
    assert keys == ["tail_outer", "tail_middle", "tail_inner", "tail_middle", "tail_outer"]


def test_part_area_estimates():
    assert A.part_area(center(0, r=10)) == pytest.approx(math.pi * 100)
    assert A.part_area(ring("r", 0, 20, 10)) == pytest.approx(math.pi * 300)
    t = tails([0, 1], fracs=[0.25, 0.75])
    assert A.part_area(t[1]) == pytest.approx(0.75 * 220 * 525)
    assert A.part_area(Part("x", "head", {"area": 123.0}, 0, 1, 1.0)) == 123.0
