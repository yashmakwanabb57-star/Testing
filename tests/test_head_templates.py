import numpy as np

from tools.head_templates import GRID, design_key, features, partition


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
