from pathlib import Path

import pytest

from rosette.model import Part
from rosette.vision.one_color import rose_center

PHOTOS = Path(__file__).resolve().parent.parent / "reference" / "photos"


def test_a_two_color_model_or_one_with_a_center_is_left_alone():
    head = Part("head", "head", {"cx": 300.0, "cy": 180.0, "r": 150.0}, 0, 10, 0.9)
    two = [head, Part("tail-0-0", "tail_stripe", {"tails_count": 1}, 1, 9, 0.9)]
    assert rose_center(None, two) is None
    with_center = [head, Part("center", "center", {"cx": 300.0, "cy": 180.0, "r": 40.0}, 0, 50, 0.9)]
    assert rose_center(None, with_center) is None


@pytest.mark.slow
def test_a13_15_one_red_rosette_gets_its_rose_in_the_same_color(monkeypatch):
    monkeypatch.setenv("ROSETTE_READER", "off")
    monkeypatch.setenv("ROSETTE_SAM2", "off")
    from rosette.pipeline import analyze_with_context
    from rosette.vision.templates import design_key
    model, _, _ = analyze_with_context((PHOTOS / "A13-15.png").read_bytes(), exclude_key=design_key("A13-15"))
    head = next(q for q in model.parts if q.role == "head")
    center = next(q for q in model.parts if q.role == "center")
    assert center.color_group == head.color_group  # one color, as the photo
    assert 0.3 <= float(center.params["r"]) / float(head.params["r"]) <= 0.5  # designer 0.44
    assert len({q.color_group for q in model.parts}) == 1 and model.rose
