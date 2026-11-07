"""Library heads (H7): a `template` part drawn from reference pieces, and the switch
between the template and the rule-based head."""
import numpy as np
import pytest

import rosette.build as build_mod
from rosette.build import build, color_shapes
from rosette.model import ColorGroup, Part, RosetteModel
from rosette.pipeline import FLAG_BELOW, is_flagged, render, template_confidence, use_rules_head, with_template
from rosette.vision.parts_common import Frame
from rosette.vision.templates import Piece, Template
from tests.test_build import _layered_cat_at


def _circle(r, n=96):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _star(n, r_in, r_out):
    t = np.linspace(-np.pi / 2, 1.5 * np.pi, 2 * n, endpoint=False)
    rr = np.where(np.arange(2 * n) % 2 == 0, r_out, r_in)
    return np.stack([rr * np.cos(t), rr * np.sin(t)], axis=1)


# reference paint order: a Color 2 star, then a Color 1 disc over its middle, then a
# Color 3 center (the ring hole shows the star underneath)
LIB = {"T1-15": Template("T1-15", "T1", [Piece(2, "points", [_star(6, 0.6, 1.0)]),
                                         Piece(1, "ring", [_circle(0.55), _circle(0.4)]),
                                         Piece(3, "center", [_circle(0.25)])])}


@pytest.fixture(autouse=True)
def library(monkeypatch):
    monkeypatch.setattr(build_mod, "template_by_id", lambda tid: LIB.get(tid))


def _groups():
    return [ColorGroup(1, (200, 30, 30), False, 0.1, 0.5, 1, 0.9),
            ColorGroup(2, (30, 30, 200), False, 0.1, 0.2, 3, 0.9),
            ColorGroup(3, (30, 200, 30), False, 0.1, 0.3, 2, 0.9)]


def _tail():
    return Part("t0", "tail_stripe", {"tail_index": 0, "stripe_index": 0, "tails_count": 1, "top_y": 250,
                                      "bottom_y": 800, "top_width": 80, "bottom_width": 120, "spread_deg": 0,
                                      "stripes": "1"}, 2, 0, 1.0)


def _model(**over):
    # reference Color 2 -> group 2 (our Color 3), Color 1 -> group 1 (Color 1), 3 -> group 3 (Color 2)
    params = {"template_id": "T1-15", "cx": 300, "cy": 200, "r": 150, "rotation_deg": 0, "score": 0.9,
              "group_1": 1, "group_2": 2, "group_3": 3, **over}
    return RosetteModel([_tail(), Part("template", "template", params, 1, 10, 0.9)], _groups())


def test_template_pieces_take_their_mapped_colors_in_reference_paint_order():
    pairs = [(s, p) for s, p in color_shapes(_model()) if p.role == "template"]
    assert [s.category for s, _ in pairs] == [3, 1, 2]
    assert [s.role for s, _ in pairs] == ["points", "ring", "center"]
    assert [s.z for s, _ in pairs] == sorted(s.z for s, _ in pairs)


def test_template_stacks_like_the_reference_whatever_the_colors():
    # the Color 1 ring is painted after the Color 3 star in the reference: it must show
    # although the shop paints Color 1 first; the hole shows the star; the center on top
    cats = _layered_cat_at(_model(), [(300 + 72, 200), (300 + 70, 200 - 0), (300 + 50, 200), (300, 200),
                                      (300, 200 - 140)])
    assert cats[0] == 1  # 0.48 R: the ring over the star
    assert cats[2] == 3  # 0.33 R: the ring's hole, the star shows
    assert cats[3] == 2  # the center
    assert cats[4] == 3  # a star tip


def test_template_head_is_placed_on_its_circle_and_validates():
    svg = render(_model(cx=320, cy=210, r=100)).svg
    assert render(_model()).issues == []
    cats = _layered_cat_at(_model(cx=320, cy=210, r=100), [(320, 210), (320, 210 - 95), (320, 210 - 120)])
    assert cats[0] == 2 and cats[1] == 3
    assert "Layer_2_1" in svg


def test_tail_is_cut_to_the_outside_of_the_template_head():
    shapes = [s for s, p in color_shapes(_model()) if p.role == "tail_stripe"]
    from rosette.geometry.clip import to_geom
    from shapely import Point
    assert not to_geom(shapes[0]).intersects(Point(300, 255).buffer(1))  # under the ring
    assert to_geom(shapes[0]).intersects(Point(300, 700).buffer(1))


def test_rose_sits_on_the_template_center_piece():
    m = _model()
    m.rose = True
    circel = [s for s in build(m) if s.kind == "circel"]
    assert len(circel) == 1
    assert float(circel[0].attrs["r"]) == pytest.approx(0.25 * 150, abs=0.5)


def test_unknown_template_id_raises():
    with pytest.raises(ValueError):
        build(_model(template_id="nope"))


def _labels_like(t, shape=(400, 400), head=(200, 200, 100), mapping=None):
    """A photo label image painted from a template (category c -> label mapping[c])."""
    from rosette.vision.templates import raster
    lut = np.zeros(5, np.uint8)
    for c, g in (mapping or {}).items():
        lut[c] = g + 1
    img = raster(t.pieces, shape, *head, lut=lut).astype(np.int32) - 1
    return img >= 0, img


def _detected():
    parts = [_tail(), Part("head", "head", {"cx": 300, "cy": 200, "r": 150, "pinked": 1}, 1, 10, 0.4),
             Part("u", "unknown", {"cx": 280, "cy": 190, "r": 30}, 3, 60, 0.1)]
    return RosetteModel(parts, _groups(), rose=False)


def test_with_template_replaces_the_head_parts_and_keeps_them_as_alternatives():
    t = LIB["T1-15"]
    mask, labels = _labels_like(t, mapping={1: 1, 2: 2, 3: 3})
    fr = Frame(axis_x=200, top=100, scale=1.0)  # px -> canvas: x + 100, y - 75
    out = with_template(_detected(), mask, labels, (200, 200, 100), fr, _groups(), lib=[t], min_score=0.5)
    tmpl = [q for q in out.parts if q.role == "template"]
    assert len(tmpl) == 1 and [q.role for q in out.parts] == ["tail_stripe", "template"]
    q = tmpl[0].params
    assert q["template_id"] == "T1-15" and (q["cx"], q["cy"], q["r"]) == (300, 125, 100)
    assert (q["group_1"], q["group_2"], q["group_3"]) == (1, 2, 3)
    assert q["score"] > 0.9 and tmpl[0].confidence >= FLAG_BELOW
    assert [p.id for p in out.alternatives] == ["head", "u"]
    assert not is_flagged(out, [])  # the unknown is no longer drawn
    # below the bar the rules head stays
    same = with_template(_detected(), mask, labels, (200, 200, 100), fr, _groups(), lib=[t], min_score=1.5)
    assert [p.id for p in same.parts] == ["t0", "head", "u"] and same.alternatives == []


def test_use_rules_head_switches_both_ways():
    m = _model()
    m.alternatives = [Part("head", "head", {"cx": 300, "cy": 200, "r": 150, "pinked": 1}, 1, 10, 0.4)]
    rules = use_rules_head(m)
    assert [p.role for p in rules.parts] == ["tail_stripe", "head"]
    assert [p.role for p in rules.alternatives] == ["template"]
    back = use_rules_head(rules)
    assert [p.role for p in back.parts] == ["tail_stripe", "template"]
    assert render(rules).issues == [] and render(back).issues == []


def test_model_json_round_trip_keeps_alternatives():
    m = _model()
    m.alternatives = [Part("head", "head", {"cx": 300, "cy": 200, "r": 150, "pinked": 1}, 1, 10, 0.4)]
    back = RosetteModel.from_json(m.to_json())
    assert back.alternatives == m.alternatives and back.parts == m.parts


def test_template_confidence_flags_a_weak_match():
    from rosette.pipeline import TEMPLATE_FLAG_SCORE
    assert template_confidence(TEMPLATE_FLAG_SCORE) == pytest.approx(FLAG_BELOW)
    assert template_confidence(TEMPLATE_FLAG_SCORE - 0.05) < FLAG_BELOW
    assert template_confidence(TEMPLATE_FLAG_SCORE + 0.05) > FLAG_BELOW


def test_a_piece_repeated_later_is_drawn_once(monkeypatch):
    t = Template("T2-15", "T2", [Piece(1, "points", [_star(6, 0.6, 1.0)]), Piece(1, "points", [_star(6, 0.6, 1.0)]),
                                 Piece(3, "center", [_circle(0.25)])])
    monkeypatch.setattr(build_mod, "template_by_id", lambda tid: t)
    assert render(_model(template_id="T2-15")).issues == []
    assert len([1 for _, p in color_shapes(_model(template_id="T2-15")) if p.role == "template"]) == 2


def test_the_library_head_is_drawn_only_when_its_outline_fits_better():
    from rosette.pipeline import TEMPLATE_MIN_GAIN, HeadFit, prefer_template
    rules = HeadFit(agreement=0.8, silhouette=0.80)
    assert prefer_template(rules, HeadFit(0.5, 0.80 + TEMPLATE_MIN_GAIN))
    assert not prefer_template(rules, HeadFit(0.99, 0.80 + TEMPLATE_MIN_GAIN / 2))
    assert prefer_template(None, HeadFit(0.5, 0.5))
    assert not prefer_template(rules, None)


def _detected_with_head_only_group():
    m = _detected()
    m.groups.append(ColorGroup(4, (250, 250, 250), False, 0.1, 0.05, 4, 0.5))
    m.parts.append(Part("ring", "ring", {"cx": 300, "cy": 200, "r_outer": 60, "r_inner": 50}, 4, 30, 0.5))
    return m


def test_groups_only_the_hidden_head_uses_leave_the_drawn_list_and_come_back():
    t = LIB["T1-15"]
    mask, labels = _labels_like(t, mapping={1: 1, 2: 2, 3: 3})
    fr = Frame(axis_x=200, top=100, scale=1.0)
    out = with_template(_detected_with_head_only_group(), mask, labels, (200, 200, 100), fr, _groups(), lib=[t],
                        min_score=0.5)
    assert [g.id for g in out.groups] == [1, 2, 3] and [g.id for g in out.alternative_groups] == [4]
    rules = use_rules_head(out)
    assert 4 in [g.id for g in rules.groups]
    assert [g.id for g in use_rules_head(rules).alternative_groups] == [4]
    assert RosetteModel.from_json(out.to_json()).alternative_groups == out.alternative_groups


def test_a_template_that_cannot_be_colored_under_the_tails_is_not_considered():
    t = Template("T3-15", "T3", LIB["T1-15"].pieces, tail_categories=(1, 2, 4))
    t4 = Template("T3-15", "T3", t.pieces + [Piece(4, "points", [_star(6, 0.3, 0.38)])], tail_categories=(1, 2, 4))
    mask, labels = _labels_like(t4, mapping={1: 1, 2: 2, 3: 3, 4: 3})
    fr = Frame(axis_x=200, top=100, scale=1.0)
    # our tails carry only Color 3 (group 2): the reference's tail Colors 1, 2 and 4 have no tail group
    out = with_template(_detected(), mask, labels, (200, 200, 100), fr, _groups(), lib=[t4], min_score=0.0)
    # the matcher does not consider a template it cannot color under the tails
    assert not any(q.role == "template" for q in out.parts) and out.alternatives == []


def test_a_template_whose_colors_agree_with_the_tails_is_used():
    # the reference's tail Color 3 is its center; our tails carry Color 3 as group 2
    t = Template("T4-15", "T4", LIB["T1-15"].pieces, tail_categories=(3,))
    mask, labels = _labels_like(t, mapping={1: 1, 2: 3, 3: 2})
    fr = Frame(axis_x=200, top=100, scale=1.0)
    out = with_template(_detected(), mask, labels, (200, 200, 100), fr, _groups(), lib=[t], min_score=0.0)
    tmpl = next(q for q in out.parts if q.role == "template")
    assert tmpl.params["group_3"] == 2


def test_head_overlap_of_two_models():
    from rosette.pipeline import head_overlap
    a = _model()
    assert head_overlap(a, a) == pytest.approx(1.0)
    b = _model(r=60)
    assert head_overlap(a, b) < 0.3


def test_partition_is_scored_under_the_tail_bound_mapping():
    from rosette.vision.templates import choose
    # same shape, two color layouts: the reference whose tail Color 1 is the ring wins
    # when our tails carry Color 1 as the ring's group (1)
    ring_tail = Template("T5-15", "T5", LIB["T1-15"].pieces, tail_categories=(1,))
    star_tail = Template("T6-15", "T6", LIB["T1-15"].pieces, tail_categories=(2,))
    mask, labels = _labels_like(ring_tail, mapping={1: 1, 2: 2, 3: 3})
    from rosette.vision.templates import photo_grid
    grid = photo_grid(mask, labels, (200, 200, 100))
    t, f = choose(grid, [star_tail, ring_tail], tail_groups={1: 1, 2: 3})
    assert t.sid == "T5-15" and f["partition"] > 0.9


def test_the_library_head_needs_its_tail_mapped_colors_to_agree():
    from rosette.pipeline import TEMPLATE_MIN_PARTITION, colors_agree
    assert colors_agree(TEMPLATE_MIN_PARTITION) and colors_agree(0.95)
    assert not colors_agree(TEMPLATE_MIN_PARTITION - 0.01)


def test_a_grossly_failed_rule_based_head_is_replaced_whatever_the_colors():
    from rosette.pipeline import HeadFit, gross_failure
    assert gross_failure(HeadFit(agreement=0.42, silhouette=0.60))
    assert gross_failure(HeadFit(agreement=0.9, silhouette=0.70))
    assert not gross_failure(HeadFit(agreement=0.62, silhouette=0.79))
    assert not gross_failure(None)


def test_the_override_needs_a_large_outline_gain_and_some_color_agreement():
    from rosette.pipeline import TEMPLATE_OVERRIDE_GAIN, TEMPLATE_OVERRIDE_PARTITION, HeadFit, override_allowed
    failed = HeadFit(agreement=0.42, silhouette=0.60)
    big = HeadFit(agreement=0.5, silhouette=0.60 + TEMPLATE_OVERRIDE_GAIN)
    assert override_allowed(failed, big, TEMPLATE_OVERRIDE_PARTITION)
    # a small outline gain or a library head whose colors hardly agree: the rules head stays
    assert not override_allowed(failed, HeadFit(0.5, 0.60 + TEMPLATE_OVERRIDE_GAIN - 0.01), 0.9)
    assert not override_allowed(failed, big, TEMPLATE_OVERRIDE_PARTITION - 0.01)
    # only a grossly failed rule-based head is overridden
    assert not override_allowed(HeadFit(0.9, 0.85), HeadFit(0.9, 1.0), 0.9)
    assert not override_allowed(failed, None, 0.9)


def test_library_head_path_floor_override_or_rules_with_the_warning():
    from rosette.pipeline import TEMPLATE_MIN_PARTITION, HeadFit, library_head_path
    failed, fine = HeadFit(0.42, 0.60), HeadFit(0.9, 0.85)
    assert library_head_path(fine, HeadFit(0.9, 0.88), TEMPLATE_MIN_PARTITION) == "floor"
    assert library_head_path(fine, HeadFit(0.9, 0.86), 0.9) is None  # outline gain under 0.02
    assert library_head_path(fine, HeadFit(0.9, 0.95), 0.40) == "colors"  # no gross failure, colors disagree
    assert library_head_path(failed, HeadFit(0.5, 0.88), 0.40) == "override"
    assert library_head_path(failed, HeadFit(0.5, 0.65), 0.40) == "colors"  # gain 0.05: rules + warning
    assert library_head_path(failed, HeadFit(0.5, 0.88), 0.23) == "colors"  # partition under 0.30
    assert library_head_path(None, HeadFit(0.5, 0.88), 0.9) == "floor"
