from pathlib import Path

from rosette.model import ColorGroup, Part, RosetteModel
from rosette.pipeline import RenderResult, analyze, is_flagged, render

PHOTOS = Path(__file__).resolve().parent.parent / "reference" / "photos"


def _photo(sid: str) -> bytes:
    return (PHOTOS / f"{sid}.png").read_bytes()


def test_end_to_end_s46_valid():
    result = render(analyze(_photo("S46-15")))
    assert isinstance(result, RenderResult)
    assert result.issues == []
    assert result.svg.startswith("<svg")


def test_end_to_end_s35_valid():
    assert render(analyze(_photo("S35-19"))).issues == []


def test_analyze_assigns_every_used_group_and_drops_unused():
    model = analyze(_photo("S46-15"))
    used = {q.color_group for q in model.parts}
    assert {g.id for g in model.groups} == used
    assert all(g.category in (1, 2, 3, 4) for g in model.groups)
    assert model.rose == any(q.role == "center" for q in model.parts)


def test_render_is_deterministic():
    data = _photo("S35-19")
    a = render(analyze(data))
    b = render(analyze(data))
    assert a.svg == b.svg


def _model(role="head", warnings=()):
    g = ColorGroup(0, (200, 0, 0), False, 0.0, 1.0, 1, 0.9)
    part = Part("p", role, {"cx": 300.0, "cy": 175.0, "r": 150.0}, 0, 0, 0.9)
    return RosetteModel(parts=[part], groups=[g], warnings=list(warnings))


def test_flagged_rules():
    assert not is_flagged(_model(), [])
    assert is_flagged(_model(role="unknown"), [])
    assert is_flagged(_model(warnings=["part p: unrecognised shape, drawn as a plain disc"]), [])
    low = _model()
    low.groups[0].confidence = 0.5
    assert is_flagged(low, [])
