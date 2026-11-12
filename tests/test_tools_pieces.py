import numpy as np

from tools.pieces import best_iou_table


def _sq(y0, y1, x0, x1, shape=(40, 40)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def test_one_covered_one_missed():
    a, b = _sq(0, 10, 0, 10), _sq(20, 30, 20, 30)
    t = best_iou_table([a, b], [a.copy()])
    assert t["layers"] == 2 and t["covered_070"] == 1 and t["mean_best_iou"] == 0.5
    assert t["best"] == [1.0, 0.0]


def test_best_piece_is_the_max_and_threshold_is_inclusive():
    ref = _sq(0, 10, 0, 10)
    t = best_iou_table([ref], [_sq(0, 10, 0, 5), _sq(0, 10, 0, 7)])  # IoU 0.5 and 0.7
    assert abs(t["best"][0] - 0.7) < 1e-9 and t["covered_070"] == 1


def test_no_layers_or_no_pieces():
    assert best_iou_table([], [_sq(0, 5, 0, 5)])["mean_best_iou"] == 0.0
    t = best_iou_table([_sq(0, 5, 0, 5)], [])
    assert t["covered_070"] == 0 and t["mean_best_iou"] == 0.0
