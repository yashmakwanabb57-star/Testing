from pathlib import Path

from rosette.pipeline import RenderResult, analyze, render

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
