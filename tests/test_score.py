import numpy as np
import pytest

from tools.score import DEFAULT_COLORS, align_agreement, score_all


def _img(h, w, boxes):
    img = np.full((h, w, 3), 255, np.uint8)
    for (y0, y1, x0, x1), cat in boxes:
        img[y0:y1, x0:x1] = [int(DEFAULT_COLORS[cat][i:i + 2], 16) for i in (1, 3, 5)]
    return img


def test_align_agreement_ignores_offset_and_scale():
    designer = _img(100, 80, [((10, 60, 10, 50), 1), ((60, 90, 10, 50), 2)])
    # same picture, shifted and twice the size
    generated = _img(200, 200, [((40, 140, 60, 140), 1), ((140, 200, 60, 140), 2)])
    agreement, iou = align_agreement(designer, generated)
    assert agreement == pytest.approx(1.0)
    assert iou == pytest.approx(1.0)


def test_align_agreement_counts_wrong_category():
    designer = _img(100, 80, [((10, 60, 10, 50), 1), ((60, 90, 10, 50), 2)])
    generated = _img(100, 80, [((10, 60, 10, 50), 1), ((60, 90, 10, 50), 3)])
    agreement, _ = align_agreement(designer, generated)
    assert agreement == pytest.approx(50 / 80)


@pytest.mark.slow
def test_bar():
    rows = score_all()
    n = len(rows)
    ok = [r for r in rows if r["success"]]
    failing = [r for r in rows if not r["success"]]
    assert len(ok) / n >= 0.8 or all(r["flagged"] for r in failing), \
        [r["id"] for r in failing if not r["flagged"]]
