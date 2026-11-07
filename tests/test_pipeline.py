from pathlib import Path

import pytest

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


def test_low_confidence_part_of_any_role_is_flagged():
    # head, ring, center and tail stripe confidences count, not just points
    for role in ("head", "ring", "center", "tail_stripe"):
        m = _model(role=role)
        assert not is_flagged(m, []), role
        m.parts[0].confidence = 0.55
        assert is_flagged(m, []), role


def test_s31_low_confidence_tails_are_flagged():
    # S31-15: every tail stripe at 0.545 used to come back unflagged
    model = analyze(_photo("S31-15"))
    stripes = [q for q in model.parts if q.role == "tail_stripe"]
    assert stripes and all(q.confidence < 0.6 for q in stripes)
    assert is_flagged(model, render(model).issues)


@pytest.mark.slow
def test_every_reference_analysis_passes_check_params():
    # the pipeline must never produce a model that /api/render rejects
    for f in sorted(PHOTOS.iterdir()):
        model = analyze(f.read_bytes())
        RosetteModel.from_json(model.to_json())  # runs check_params on every part


def test_unsure_badge_print_is_warned(monkeypatch):
    # a group that may be print on the center badge but is not clearly enclosed is
    # kept as a color and the user is asked to check it
    import rosette.pipeline as pl
    monkeypatch.setattr(pl, "badge_print_unsure", lambda p, labels, groups: [groups[0].id])
    model = analyze(_photo("S46-15"))
    assert any("center badge" in w for w in model.warnings)
    assert is_flagged(model, render(model).issues)


def test_merged_badge_print_is_warned(monkeypatch):
    # merging print into the badge color changes the center: never silently
    import rosette.pipeline as pl
    real = pl.pixel_groups

    def fake(p, merged=None):
        labels, groups = real(p, merged=merged)
        merged.append((217, 201, 155))
        return labels, groups
    monkeypatch.setattr(pl, "pixel_groups", fake)
    model = analyze(_photo("S46-15"))
    assert any("Center badge print was merged" in w for w in model.warnings)
    assert is_flagged(model, render(model).issues)


def test_unknown_speck_is_left_out_and_warned(monkeypatch):
    # an unknown speck the detector reports is not drawn; the model says so
    import rosette.pipeline as pl
    real = pl.detect_parts

    def fake(p, labels, *args, **kwargs):
        parts = real(p, labels, *args, **kwargs)
        g = parts[0].color_group
        return parts + [Part("unknown-9", "unknown", {"cx": 430.0, "cy": 290.0, "r": 3.0}, g, 60, 0.2)]
    monkeypatch.setattr(pl, "detect_parts", fake)
    model = analyze(_photo("S46-15"))
    assert not any(q.id == "unknown-9" for q in model.parts)
    assert any("unknown-9" in w for w in model.warnings)
    assert render(model).issues == []


def test_uncertain_head_circle_bands():
    # the fit and the symmetry circle disagree without a takeover (ratio 1.1-1.25), or the
    # takeover hung on a coverage within 0.1 of its bar: the head circle is uncertain
    from types import SimpleNamespace

    from rosette.pipeline import CIRCLE_UNCERTAIN, circle_warnings
    from rosette.vision.prepare import FIT_MIN_COVERAGE, FIT_TAKEOVER

    def p(ratio, coverage):
        return SimpleNamespace(fit_ratio=ratio, fit_coverage=coverage)
    assert circle_warnings(p(1.0, 0.9)) == []
    assert circle_warnings(p(1.0, FIT_MIN_COVERAGE)) == []  # the fit agrees: coverage is moot
    assert circle_warnings(p(1.1, 0.9)) == [CIRCLE_UNCERTAIN]
    assert circle_warnings(p(FIT_TAKEOVER - 0.01, 0.2)) == [CIRCLE_UNCERTAIN]
    assert circle_warnings(p(FIT_TAKEOVER, FIT_MIN_COVERAGE + 0.15)) == []
    assert circle_warnings(p(FIT_TAKEOVER, FIT_MIN_COVERAGE + 0.05)) == [CIRCLE_UNCERTAIN]
    assert circle_warnings(p(1.4, FIT_MIN_COVERAGE - 0.05)) == [CIRCLE_UNCERTAIN]
    assert circle_warnings(p(1.4, FIT_MIN_COVERAGE - 0.2)) == []
    assert "–" not in CIRCLE_UNCERTAIN and "—" not in CIRCLE_UNCERTAIN


def test_uncertain_head_circle_is_warned_and_flagged(monkeypatch):
    import rosette.pipeline as pl
    monkeypatch.setattr(pl, "circle_warnings", lambda p: [pl.CIRCLE_UNCERTAIN])
    model = analyze(_photo("S46-15"))
    assert pl.CIRCLE_UNCERTAIN in model.warnings
    assert is_flagged(model, render(model).issues)
