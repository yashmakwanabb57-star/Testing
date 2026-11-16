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


def test_r48_square_sits_under_the_star():
    # S28-15, the owner's check: head, edged loops, star, square - the loops run in under the star
    from rosette.model import Part
    from rosette.vision.stacked_stars import star_on_square
    geo = {"cx": 300.0, "cy": 300.0}
    parts = [
        Part("head", "head", dict(geo, r=100.0), 1, 0, 0.9),
        Part("center", "center", dict(geo, r=30.0), 1, 50, 0.9),
        Part("points-1", "points", dict(geo, kind="bow", count=4, rotation_deg=45.0, r_inner=50.0, r_outer=90.0), 2, 20, 0.9),
        Part("points-2", "points", dict(geo, kind="star", count=4, rotation_deg=0.0, r_inner=30.0, r_outer=85.0), 3, 22, 0.9),
    ]
    out = star_on_square(None, np.zeros((1, 1), dtype=int), parts)
    assert out is not None
    ids = {q.id: q for q in out}
    assert ids["points-square"].color_group == 2
    assert ids["points-1"].z < ids["edge-points-1"].z < ids["points-2"].z < ids["points-square"].z < ids["center"].z
    assert ids["points-1"].params["r_inner"] == 30.0  # the loops run in to the center, under the star
