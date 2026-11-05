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


def test_low_confidence_points_ring_is_flagged():
    # S41-15: a points ring whose rotational symmetry is weak (detector confidence
    # below FLAG_BELOW) is a misread layer as often as not - ask the user to check
    m = _model()
    pts = Part("points-1", "points", {"cx": 300.0, "cy": 175.0, "kind": "star", "count": 6, "r_inner": 50.0,
                                      "r_outer": 90.0, "rotation_deg": 0.0, "width_ratio": 0.5}, 0, 20, 0.9)
    m.parts.append(pts)
    assert not is_flagged(m, [])
    pts.confidence = 0.55
    assert is_flagged(m, [])


def test_merged_print_and_satin_is_flagged():
    # S2-15B: two colors found for a three-color ribbon (print merged with satin)
    model = analyze(_photo("S2-15B"))
    assert any("partly patterned" in w for w in model.warnings)
    assert is_flagged(model, render(model).issues)
