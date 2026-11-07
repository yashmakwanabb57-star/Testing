import json

import numpy as np
import pytest

from rosette.model import ColorGroup, RosetteModel
from rosette.vision.templates import (GRID, Piece, Template, choose, describe, design_key, extract_template,
                                      features, from_json, partition, photo_grid, rank, raster, relabel_map,
                                      tail_consistent_mapping, to_json)


def test_design_key_groups_variants_and_sizes():
    assert design_key("S36-15B") == design_key("S36-15") == "S36"
    assert design_key("S11a-15") == "S11"
    assert design_key("s34-15") == "S34"
    assert design_key("A13-15B") == "A13"


def _disc(r_cells, value, base=None):
    g = np.zeros((GRID, GRID), np.int16) if base is None else base.copy()
    yy, xx = np.mgrid[:GRID, :GRID]
    c = (GRID - 1) / 2
    g[(yy - c) ** 2 + (xx - c) ** 2 <= r_cells ** 2] = value
    return g


def test_partition_maps_template_categories_one_to_one():
    # template: Color 1 disc with a Color 2 center; photo: group 7 disc, group 3 center
    tmpl = _disc(10, 2, _disc(30, 1))
    photo = _disc(10, 3, _disc(30, 7, np.full((GRID, GRID), -1, np.int16)))
    share, mapping = partition(photo, tmpl.astype(np.uint8))
    assert mapping == {1: 7, 2: 3}
    assert share > 0.99


def test_features_prefer_the_matching_silhouette():
    photo = _disc(30, 0, np.full((GRID, GRID), -1, np.int16))
    same = _disc(30, 1).astype(np.uint8)
    small = _disc(15, 1).astype(np.uint8)
    assert features(photo, same)["score"] > features(photo, small)["score"]


def _circle(r, n=64):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _star(n, r_in, r_out):
    t = np.linspace(-np.pi / 2, 1.5 * np.pi, 2 * n, endpoint=False)
    rr = np.where(np.arange(2 * n) % 2 == 0, r_out, r_in)
    return np.stack([rr * np.cos(t), rr * np.sin(t)], axis=1)


def _template(sid, pieces):
    return Template(sid, design_key(sid), [Piece(c, role, rings) for c, role, rings in pieces])


def test_raster_paints_in_order_with_holes():
    t = _template("X1-15", [(1, "head", [_circle(1.0), _circle(0.5)]),  # an annulus
                            (2, "center", [_circle(0.3)])])
    img = raster(t.pieces, (201, 201), 100, 100, 80)
    assert img[100, 100] == 2  # center on top
    assert img[100, 100 + 32] == 0  # the annulus hole (0.4 R) shows the background
    assert img[100, 100 + 60] == 1  # 0.75 R: the annulus
    assert img[100, 100 + 90] == 0  # outside
    assert t.categories == [1, 2]


def test_json_round_trip_keeps_pieces_and_grid():
    t = _template("X1-15", [(1, "head", [_circle(1.0)]), (3, "points", [_star(6, 0.5, 1.2)])])
    back = from_json(json.loads(json.dumps(to_json([t]))))[0]
    assert back.sid == "X1-15" and back.key == "X1"
    assert [(p.category, p.role) for p in back.pieces] == [(1, "head"), (3, "points")]
    assert np.array_equal(back.grid, t.grid)


def test_choose_picks_the_matching_head_and_leave_one_out_skips_its_design():
    disc = _template("A1-15", [(1, "head", [_circle(1.0)]), (2, "center", [_circle(0.35)])])
    star = _template("B2-15", [(1, "points", [_star(6, 0.55, 1.0)]), (2, "center", [_circle(0.35)])])
    # the photo: a 6-point star of group 4 around a group 9 center
    photo = np.where(star.grid == 1, 4, np.where(star.grid == 2, 9, -1)).astype(np.int16)
    t, f = choose(photo, [disc, star])
    assert t.sid == "B2-15" and f["mapping"] == {1: 4, 2: 9} and f["score"] > 0.9
    t2, _ = choose(photo, [disc, star], exclude_key="B2", gate=None)
    assert t2.sid == "A1-15"
    # the lobe gate: a disc is never chosen for a star photo
    assert choose(photo, [disc, star], exclude_key="B2") is None


def test_head_color_count_feeds_the_score():
    one = _template("C1-15", [(1, "head", [_circle(1.0)])])
    two = _template("C2-15", [(1, "head", [_circle(1.0)]), (2, "center", [_circle(0.5)])])
    photo = np.where(two.grid == 1, 4, np.where(two.grid == 2, 9, -1)).astype(np.int16)
    f1, f2 = features(photo, one.grid, (0, 0, 0, 0, 1)), features(photo, two.grid, (0, 0, 0, 0, 1))
    assert (f1["colors_photo"], f1["colors_template"], f2["colors_template"]) == (2, 1, 2)
    assert f2["score"] > f1["score"]


def test_tail_categories_map_to_the_tail_groups_and_head_only_ones_by_overlap():
    t = Template("D1-15", "D1", [Piece(1, "head", [_circle(1.0)]), Piece(3, "center", [_circle(0.4)])],
                 tail_categories=(1, 2))
    # photo: group 5 disc, group 8 center; our tails carry Color 1 as group 7
    photo = np.where(t.grid == 1, 5, np.where(t.grid == 3, 8, -1)).astype(np.int16)
    assert tail_consistent_mapping(photo, t, {1: 7, 2: 6}) == {1: 7, 3: 8}
    # the reference's tail Color 1 has no tail group of Color 1 here: inconsistent
    assert tail_consistent_mapping(photo, t, {2: 6, 3: 8}) is None


SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 850">
<style>.st0_1{fill:#d0312d}.st0_2{fill:#2a3fa0}.st0_3{fill:#f2c12e}</style>
<g id="Layer_1"><polygon class="st0_1" points="240,300 360,300 420,825 180,825"/>
<circle class="st0_1" cx="300" cy="175" r="150"/></g>
<g id="Layer_2_1"><g id="Layer_2_1_2"><circle class="st0_2" cx="300" cy="175" r="45"/></g>
<g id="Layer_2_1_3"><polygon class="st0_3" points="{star}"/></g></g>
</svg>"""


@pytest.mark.slow
def test_extract_template_keeps_head_pieces_in_paint_order_normalized_to_the_head():
    pts = _star(8, 70, 110) + np.array([300, 175])
    svg = SVG.replace("{star}", " ".join(f"{x:.2f},{y:.2f}" for x, y in pts))
    t = extract_template("Z9-15", svg, height=900)
    cats = [p.category for p in t.pieces]
    assert cats == [1, 2, 3]  # head disc, center, star: document order, the tail left out
    reach = [float(np.hypot(*p.rings[0].T).max()) for p in t.pieces]
    assert reach[0] == pytest.approx(1.0, abs=0.05)  # the head circle is the unit
    assert reach[1] == pytest.approx(45 / 150, abs=0.05)
    assert reach[2] == pytest.approx(110 / 150, abs=0.05)


def test_relabel_map_sends_merged_and_unused_labels_to_model_groups():
    g = lambda i, rgb: ColorGroup(i, rgb, False, 0.0, 0.1, 1, 1.0)  # noqa: E731
    model = RosetteModel(groups=[g(0, (200, 0, 0)), g(2, (0, 0, 200))])
    groups = [g(0, (200, 0, 0)), g(1, (190, 10, 0)), g(2, (0, 0, 200)), g(3, (10, 0, 180))]
    assert relabel_map(model, groups, {3: 2}) == {3: 2, 1: 0}
    labels = np.array([[0, 1], [3, 2]])
    grid = photo_grid(np.ones((2, 2), bool), labels, (0.5, 0.5, 0.01), relabel_map(model, groups, {3: 2}))
    assert set(np.unique(grid)) <= {-1, 0, 2}


def test_rank_lists_the_best_templates_first_and_choose_is_its_head():
    disc = _template("A1-15", [(1, "head", [_circle(1.0)]), (2, "center", [_circle(0.35)])])
    star = _template("B2-15", [(1, "points", [_star(6, 0.55, 1.0)]), (2, "center", [_circle(0.35)])])
    star5 = _template("C3-15", [(1, "points", [_star(6, 0.65, 0.9)]), (2, "center", [_circle(0.35)])])
    photo = np.where(star.grid == 1, 4, np.where(star.grid == 2, 9, -1)).astype(np.int16)
    ranked = rank(photo, [disc, star5, star], gate=None, n=3)
    assert [t.sid for t, _ in ranked] == ["B2-15", "C3-15", "A1-15"]
    assert [f["score"] for _, f in ranked] == sorted((f["score"] for _, f in ranked), reverse=True)
    assert [t.sid for t, _ in rank(photo, [disc, star5, star], gate=None, n=2)] == ["B2-15", "C3-15"]
    assert choose(photo, [disc, star5, star], gate=None)[0].sid == "B2-15"
    assert rank(photo, [disc], n=3) == []  # the lobe gate leaves nothing


def _kite(angle, r_in, r_out, half=0.12):
    a = np.array([[r_in, 0], [(r_in + r_out) / 2, half], [r_out, 0], [(r_in + r_out) / 2, -half]])
    c, s = np.cos(angle), np.sin(angle)
    return a @ np.array([[c, s], [-s, c]])


def _loop(angle, r_in, r_out, half=0.2):
    # a block whose outer end is as wide as its inner end
    t = np.linspace(-half, half, 9)
    a = np.concatenate([[[r_in, -half]], np.stack([np.full(9, r_out), t], 1), [[r_in, half]]])
    c, s = np.cos(angle), np.sin(angle)
    return a @ np.array([[c, s], [-s, c]])


def test_describe_counts_the_template_pieces_by_shape():
    six = [i * np.pi / 3 for i in range(6)]
    pieces = ([(2, "points", [_loop(a, 0.5, 0.95)]) for a in six]
              + [(3, "points", [_kite(a + np.pi / 6, 0.5, 0.95)]) for a in six]
              + [(1, "ring", [_circle(0.55), _circle(0.4)]), (3, "center", [_circle(0.3)])])
    assert describe(_template("D1-15", pieces)) == "6 loops, 6 kites, ring, center"
    star = _template("E1-15", [(1, "points", [_star(6, 0.55, 1.0)]), (2, "center", [_circle(0.35)])])
    assert describe(star) == "6-point star, center"
    disc = _template("F1-15", [(1, "head", [_circle(1.0)]), (2, "center", [_circle(0.35)]),
                                (2, "center", [_circle(0.35)])])
    assert describe(disc) == "disc, center"  # a repeated piece counts once
    one = _template("G1-15", [(1, "points", [_kite(0.0, 0.5, 0.95)]), (2, "center", [_circle(0.3)])])
    assert describe(one) == "1 kite, center"


# ---------------------------------------------------------------- H9: pose search

def _posed_photo(t, dx=0.0, dy=0.0, s=1.0, rot=0.0, value=4):
    from rosette.vision.templates import EXTENT
    c, k = (GRID - 1) / 2, (GRID - 1) / (2 * EXTENT)
    g = raster(t.pieces, (GRID, GRID), c + dx * k, c + dy * k, k * s, rot)
    return np.where(g > 0, value + g.astype(np.int16), -1).astype(np.int16)


def test_pose_grid_matches_the_raster_at_that_pose():
    from rosette.vision.templates import Pose, pose_grid
    t = _template("B2-15", [(1, "points", [_star(6, 0.55, 1.0)]), (2, "center", [_circle(0.35)])])
    want = _posed_photo(t, 0.1, 0.3, 1.2, 15.0) >= 0
    got = pose_grid(t, Pose(0.1, 0.3, 1.2, 15.0)) > 0
    assert (want & got).sum() / (want | got).sum() > 0.93
    assert (pose_grid(t, Pose()) == t.grid).all()


def test_rank_finds_the_head_lower_and_larger_than_the_photo_circle():
    t = _template("B2-15", [(1, "points", [_star(6, 0.55, 1.0)]), (2, "center", [_circle(0.35)])])
    photo = _posed_photo(t, 0.0, 0.3, 1.2)
    (best, f), = rank(photo, [t], gate=None, n=1)
    p = f["pose"]
    assert abs(p.dy - 0.3) <= 0.051 and abs(p.scale - 1.2) <= 0.06 and abs(p.dx) <= 0.051
    still = features(photo, t.grid)
    assert f["score"] > still["score"] + 0.05


def test_rank_turns_a_star_within_its_symmetry():
    t = _template("B2-15", [(1, "points", [_star(6, 0.55, 1.0)]), (2, "center", [_circle(0.35)])])
    (best, f), = rank(_posed_photo(t, rot=15.0), [t], gate=None, n=1)
    assert f["pose"].rotation in (15.0, -45.0)  # 15 degrees, or the same star 60 degrees on
    assert f["iou"] > 0.9



def test_best_pose_breaks_ties_without_comparing_poses():
    from rosette.vision.templates import Pose, best_pose
    t = _template("B2-15", [(1, "points", [_star(6, 0.55, 1.0)]), (2, "center", [_circle(0.35)])])
    empty = np.full((GRID, GRID), -1, np.int16)  # every pose scores 0: all tie
    assert best_pose(empty, t) == Pose()


def test_an_unmoved_template_that_maps_under_the_tails_is_kept_when_the_pose_does_not(monkeypatch):
    from rosette.vision import templates as T
    t = _template("B2-15", [(1, "points", [_star(6, 0.55, 1.0)]), (2, "center", [_circle(0.35)])])
    photo = _posed_photo(t, 0.0, 0.3, 1.2)
    real = T.tail_consistent_mapping
    monkeypatch.setattr(T, "tail_consistent_mapping",
                        lambda labels, tp, tg: real(labels, tp, tg) if tp is t else None)
    f = T.posed_features(photo, t, tail_groups={1: 5})
    assert f is not None and f["pose"] == T.Pose() and f["anchored"]
    # with free_ok the posed free mapping does not beat the unmoved anchored one
    g = T.posed_features(photo, t, tail_groups={1: 5}, free_ok=True)
    assert g["pose"] == T.Pose() and g["anchored"]


def test_symmetry_is_cached_per_template_shape_not_per_id():
    from rosette.vision.templates import symmetry
    star = _template("Z1-15", [(1, "points", [_star(6, 0.55, 1.0)])])
    disc = _template("Z1-15", [(1, "head", [_circle(1.0)])])
    assert symmetry(star) >= 5 and symmetry(disc) < 3


def test_library_cache_is_not_stale_for_a_file_whose_head_circle_prepare_fits():
    # the cache holds templates normalized on prepare's head circle of each reference
    # render; S57-17's is the H10 silhouette fit (its symmetry circle sits at the top of
    # the rose), so a cache built before that change differs here
    from pathlib import Path

    from rosette.vision.templates import LIBRARY_PATH, template_by_id
    svg = (Path(__file__).resolve().parent.parent / "reference" / "svg" / "S57-17.svg").read_text(encoding="utf-8")
    fresh = extract_template("S57-17", svg)
    cached = template_by_id("S57-17", LIBRARY_PATH)
    assert cached is not None
    assert [(p.category, p.role) for p in cached.pieces] == [(p.category, p.role) for p in fresh.pieces]
    for a, b in zip(cached.pieces, fresh.pieces):
        assert len(a.rings) == len(b.rings)
        for ra, rb in zip(a.rings, b.rings):
            assert ra.shape == rb.shape and np.allclose(ra, rb, atol=1e-3)
