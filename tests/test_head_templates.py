import json

import numpy as np
import pytest

from rosette.model import ColorGroup, RosetteModel
from rosette.vision.templates import (GRID, Piece, Template, choose, design_key, extract_template, features,
                                      from_json, partition, photo_grid, raster, relabel_map, to_json)


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
    t2, _ = choose(photo, [disc, star], exclude_key="B2")
    assert t2.sid == "A1-15"


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
