import math

import pytest

from rosette import tidy
from rosette.model import ColorGroup, Part, RosetteModel


def group(i, pattern=False):
    return ColorGroup(i, (10 * i, 20, 30), pattern, 10.0 if pattern else 1.0, 0.3, None, 0.0)


def head(r=160.0):
    return Part("head", "head", {"cx": 300, "cy": 190, "r": r, "pinked": 1}, 0, 10, 0.9)


def unknown(pid, r, cx=430.0, cy=290.0, g=0):
    return Part(pid, "unknown", {"cx": cx, "cy": cy, "r": r}, g, 60, 0.2)


def test_tiny_unknown_is_dropped_with_a_warning_naming_it():
    # a speck of 0.4 % of the head on the rim (A13-15B: r 9.6 and 10.2 on a head of r 163.6)
    m = RosetteModel(parts=[head(), unknown("unknown-0", 10.0)], groups=[group(0)])
    out = tidy.drop_specks(m)
    assert [q.id for q in out.parts] == ["head"]
    assert len(out.warnings) == 1 and "unknown-0" in out.warnings[0]
    assert [q.id for q in m.parts] == ["head", "unknown-0"]  # input untouched


def test_tiny_unknown_is_dropped_wherever_it_lies():
    inside = unknown("unknown-0", 8.0, cx=300.0, cy=150.0)  # inside the head disc
    below = unknown("unknown-1", 8.0, cx=300.0, cy=600.0)  # down on the tails
    out = tidy.drop_specks(RosetteModel(parts=[head(), inside, below], groups=[group(0)]))
    assert [q.id for q in out.parts] == ["head"]
    assert len(out.warnings) == 2


def test_larger_unknown_is_kept():
    r = 160.0 * math.sqrt(2 * tidy.SPECK_MAX_HEAD_SHARE)  # twice the bar
    m = RosetteModel(parts=[head(), unknown("unknown-0", r)], groups=[group(0)])
    out = tidy.drop_specks(m)
    assert [q.id for q in out.parts] == ["head", "unknown-0"]
    assert out.warnings == []


def test_tail_placeholder_is_never_dropped():
    # a narrow tail stripe without a color group is a placeholder of the tail, not a speck:
    # leaving it out would leave a hole in the tail
    tail = unknown("unknown-tail-0-2", 3.0, cx=300.0, cy=600.0)
    out = tidy.drop_specks(RosetteModel(parts=[head(), tail], groups=[group(0)]))
    assert [q.id for q in out.parts] == ["head", "unknown-tail-0-2"]
    assert out.warnings == []


def test_without_a_head_nothing_is_dropped():
    m = RosetteModel(parts=[unknown("unknown-0", 2.0)], groups=[group(0)])
    assert [q.id for q in tidy.drop_specks(m).parts] == ["unknown-0"]


def test_real_small_parts_are_never_dropped():
    c = Part("center", "center", {"cx": 300, "cy": 190, "r": 5.0}, 0, 50, 0.9)
    out = tidy.drop_specks(RosetteModel(parts=[head(), c], groups=[group(0)]))
    assert [q.id for q in out.parts] == ["head", "center"]


def test_photo_head_circle_is_used_without_a_head_part():
    # headless models (loops or stars over a badge) still measure against the head circle
    m = RosetteModel(parts=[unknown("unknown-0", 5.0)], groups=[group(0)])
    assert tidy.drop_specks(m, head_r=150.0).parts == []
    assert len(tidy.drop_specks(m, head_r=50.0).parts) == 1


def tail(groups, fracs, bottom=None, k=0, n=1):
    common = {"tails_count": n, "top_y": 320.0, "bottom_y": 825.0, "top_width": 140.0,
              "bottom_width": 300.0, "spread_deg": 0.0, "stripes": ",".join(f"{f:.4f}" for f in fracs),
              "stripes_bottom": ",".join(f"{f:.4f}" for f in (bottom or fracs))}
    return [Part(f"tail-{k}-{i}", "tail_stripe", {"tail_index": k, "stripe_index": i, **common}, g, 9, 0.9)
            for i, g in enumerate(groups)]


def stripe_list(model, k=0):
    t = sorted((q for q in model.parts if q.role == "tail_stripe" and q.params["tail_index"] == k),
               key=lambda q: q.params["stripe_index"])
    return ([q.color_group for q in t], [float(v) for v in t[0].params["stripes"].split(",")],
            [q.id for q in t], [q.params["stripe_index"] for q in t])


def test_sliver_between_two_stripes_of_one_group_joins_them():
    m = RosetteModel(parts=[head()] + tail([1, 0, 1, 0], [0.3, 0.03, 0.3, 0.37]), groups=[group(0), group(1)])
    out = tidy.merge_tail_slivers(m)
    g, f, ids, idx = stripe_list(out)
    assert g == [1, 0] and f == [pytest.approx(0.63, abs=1e-3), pytest.approx(0.37, abs=1e-3)]
    assert ids == ["tail-0-0", "tail-0-1"] and idx == [0, 1]
    assert len(out.warnings) == 1 and "tail-0-1" in out.warnings[0]
    assert len(m.parts) == 5  # input untouched


def test_sliver_between_different_groups_goes_to_the_wider_neighbour():
    m = RosetteModel(parts=tail([1, 2, 0], [0.30, 0.03, 0.67], bottom=[0.2, 0.03, 0.77]),
                     groups=[group(0), group(1), group(2)])
    out = tidy.merge_tail_slivers(m)
    g, f, _, _ = stripe_list(out)
    assert g == [1, 0] and f == [pytest.approx(0.30, abs=1e-3), pytest.approx(0.70, abs=1e-3)]
    bottom = [float(v) for v in out.parts[0].params["stripes_bottom"].split(",")]
    assert bottom == [pytest.approx(0.2, abs=1e-3), pytest.approx(0.8, abs=1e-3)]
    assert len(out.warnings) == 1


def test_sliver_of_a_neighbours_group_merges_into_that_neighbour():
    # after an edit a sliver can sit next to a stripe of its own group: that one takes it
    m = RosetteModel(parts=tail([1, 0, 0, 2], [0.30, 0.25, 0.03, 0.42]), groups=[group(0), group(1), group(2)])
    g, f, _, _ = stripe_list(tidy.merge_tail_slivers(m))
    assert g == [1, 0, 2] and f[1] == pytest.approx(0.28, abs=1e-3)


def test_mirrored_tails_merge_alike_and_wide_stripes_stay():
    parts = tail([1, 0, 1, 0], [0.3, 0.03, 0.3, 0.37], k=0, n=2) + tail([0, 1, 0, 1], [0.37, 0.3, 0.03, 0.3], k=1, n=2)
    out = tidy.merge_tail_slivers(RosetteModel(parts=parts, groups=[group(0), group(1)]))
    assert stripe_list(out, 0)[0] == [1, 0] and stripe_list(out, 1)[0] == [0, 1]
    same = RosetteModel(parts=tail([1, 0, 1], [0.3, 0.2, 0.5]), groups=[group(0), group(1)])
    assert tidy.merge_tail_slivers(same).parts == same.parts


def test_a_two_point_layer_is_not_drawn_and_is_warned():
    # S53-19 (H13): white pleat teeth read as a 2-point "star" (r 0-101.5, width ratio 39)
    two = Part("points-2", "points", {"cx": 300, "cy": 190, "kind": "star", "count": 2, "r_inner": 1.0,
                                      "r_outer": 101.5, "rotation_deg": 114.6, "width_ratio": 39.0}, 1, 22, 0.6)
    three = Part("points-3", "points", {"cx": 300, "cy": 190, "kind": "star", "count": 3, "r_inner": 50.0,
                                        "r_outer": 101.5, "rotation_deg": 0.0, "width_ratio": 1.0}, 1, 21, 0.6)
    m = RosetteModel(parts=[head(), two, three], groups=[group(0), group(1)])
    out = tidy.drop_two_point_layers(m)
    assert [q.id for q in out.parts] == ["head", "points-3"]
    assert len(out.warnings) == 1 and "points-2" in out.warnings[0]
    assert [q.id for q in m.parts] == ["head", "points-2", "points-3"]  # the input is not changed
