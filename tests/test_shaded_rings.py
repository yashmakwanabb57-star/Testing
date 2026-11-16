from pathlib import Path

import numpy as np
import pytest

from rosette.vision.shaded_rings import _nearest

PHOTOS = Path(__file__).resolve().parent.parent / "reference" / "photos"


def test_shaded_dark_blue_matches_dark_blue_by_hue():
    # A3-15: shaded dark blue pleats (a 16, b -48) lie nearer the mid blue in a*b* distance,
    # but their hue angle is the dark blue's
    cents = np.array([[4.5, -40.3], [-7.0, -7.4], [25.3, -61.0]])  # mid, light, dark
    assert _nearest(np.array([15.9, -48.4]), cents) == 2
    assert _nearest(np.array([-2.3, -30.8]), cents) == 0
    assert _nearest(np.array([-3.0, -9.0]), cents) == 1  # unsaturated: a*b* distance


@pytest.mark.slow
def test_a3_15_three_blue_pleat_rings(monkeypatch):
    monkeypatch.setenv("ROSETTE_READER", "off")
    monkeypatch.setenv("ROSETTE_SAM2", "off")
    from rosette.pipeline import analyze_with_context
    from rosette.vision.templates import design_key
    m, _, _ = analyze_with_context((PHOTOS / "A3-15.png").read_bytes(), exclude_key=design_key("A3-15"))
    cat = {g.id: g.category for g in m.groups}
    head = next(q for q in m.parts if q.role == "head")
    rings = sorted((q for q in m.parts if q.role == "ring"), key=lambda q: -float(q.params["r_outer"]))
    center = next(q for q in m.parts if q.role == "center")
    # the designer: outer pleats Color 2, the mid ring Color 3, the light ring Color 1, the badge Color 3
    assert cat[head.color_group] == 2 and [cat[q.color_group] for q in rings] == [3, 1]
    assert cat[center.color_group] == 3 and 30 <= float(center.params["r"]) <= 45
