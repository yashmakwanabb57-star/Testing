import numpy as np
import pytest

from rosette.vision.stacked_stars import _clean, _runs


def test_runs_and_clean_merge_short_grain():
    row = np.array([1, 1, 1, 1, 0, 0, 0, 2, 0, 0, 0, -1, -1])
    assert _runs(row)[0] == (1, 0, 4)
    assert _clean(_runs(row)) == [(1, 0, 4), (0, 4, 11), (-1, 11, 13)]


@pytest.mark.slow
def test_s17_15_reads_stacked_stars(monkeypatch):
    monkeypatch.setenv("ROSETTE_READER", "off")
    monkeypatch.setenv("ROSETTE_SAM2", "off")
    from rosette.pipeline import detect
    d = detect(open("reference/photos/S17-15.png", "rb").read(), classify=False)
    ids = {q.id: q for q in d.model.parts}
    assert {"points-back", "points-inset", "points-between", "points-top", "center"} <= set(ids)
    assert ids["points-top"].params["count"] == 8
    assert ids["points-top"].z > ids["points-between"].z > ids["points-inset"].z > ids["points-back"].z
    assert ids["points-top"].color_group == ids["points-back"].color_group
