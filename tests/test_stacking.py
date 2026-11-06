import numpy as np

from tools import stacking as st


def _disc(shape, cx, cy, r):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _rect(shape, y0, y1, x0, x1):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def test_overlapping_masks_are_ordered_by_paint_order():
    sh = (100, 100)
    a, b = _rect(sh, 20, 60, 20, 60), _disc(sh, 60, 60, 15)
    assert st.stacking_pairs([(1, "ring", a), (2, "kite", b)], 1.0) == [("ring", "kite", False, "overlap")]
    assert st.stacking_pairs([(2, "kite", b), (1, "ring", a)], 1.0) == [("kite", "ring", False, "overlap")]


def test_a_piece_cut_along_another_lies_under_it():
    # the rectangle was drawn cut around the disc: the disc is on top, whatever the paint order
    sh = (100, 100)
    disc = _disc(sh, 60, 40, 15)
    rect = _rect(sh, 20, 60, 10, 60) & ~_disc(sh, 60, 40, 16)
    for items in ([(1, "loop", rect), (1, "center", disc)], [(1, "center", disc), (1, "loop", rect)]):
        assert st.stacking_pairs(items, 1.0) == [("loop", "center", True, "cut")]


def test_apart_masks_state_nothing():
    sh = (100, 100)
    assert st.stacking_pairs([(1, "loop", _rect(sh, 0, 10, 0, 10)), (2, "kite", _rect(sh, 50, 60, 50, 60))], 1.0) == []


def test_table_consensus_and_disagreements():
    ref = [("ring", "kite")] * 9 + [("kite", "ring")]
    tab = st.table(ref)
    assert tab == {("kite", "ring"): (9, 1)}
    assert st.consensus(tab, 0.8) == {("kite", "ring"): 0.9}
    assert st.disagreements(ref, [("kite", "ring")]) == [("kite", "ring")]
    assert st.disagreements(ref, [("ring", "kite")]) == []


def test_model_pairs_follow_z_then_layer_order():
    from rosette.model import ColorGroup, Part, RosetteModel
    groups = [ColorGroup(1, (200, 30, 30), False, 0.1, 0.3, 1, 0.9),
              ColorGroup(2, (30, 30, 200), False, 0.9, 0.3, 2, 0.9)]
    kites = Part("pts", "points", {"cx": 300, "cy": 200, "kind": "diamond", "count": 6, "r_inner": 40,
                                   "r_outer": 120, "rotation_deg": 0, "width_ratio": 0.4}, 2, 20, 1.0)
    ring = Part("ring", "ring", {"cx": 300, "cy": 200, "r_outer": 70, "r_inner": 30}, 1, 30, 1.0)
    pairs = st.model_pairs(RosetteModel([kites, ring], groups))
    assert set(pairs) == {("kite", "ring")}  # ring z 30 over the kites although Color 1
    ring.z = 20  # equal z: the shop's layer order, Color 2 later
    assert set(st.model_pairs(RosetteModel([kites, ring], groups))) == {("ring", "kite")}


def test_summary_lines_of_mined_pairs():
    # main's table from (below, above, same Color, how) pairs per file, no rendering
    per_file = {"A": [("star", "loop", False, "cut"), ("star", "loop", True, "overlap")],
                "B": [("loop", "star", False, "cut"), ("star", "loop", False, "cut")]}
    lines = st.summary_lines(per_file)
    row = next(line for line in lines if line.split()[:2] == ["loop", "star"])
    above, below, n, share, files, same_n, same_share = row.split()
    assert (n, share, files, same_n, same_share) == ("4", "0.75", "2", "1", "1.00")
