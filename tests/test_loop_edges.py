from pathlib import Path

import numpy as np
import pytest

from rosette.vision.parts_trims import edge_loops

PHOTOS = Path(__file__).resolve().parent.parent / "reference" / "photos"


class _Fr:
    """Frame stand-in: 1 px per canvas unit."""
    scale = 1.0

    def d(self, v):
        return v


def _polar(n=4, loop_deg=40.0, edge_deg=8.0, t=720, rows=60, loop=1, edge=2, disc=0):
    """(rows, t) labels: n loops of `loop` with an `edge` band of edge_deg on both sides, on `disc`."""
    pol = np.full((rows, t), disc, np.int32)
    th = np.arange(t) * 360.0 / t
    for k in range(n):
        d = np.abs((th - 360.0 * k / n + 180.0) % 360.0 - 180.0)
        pol[:, d <= loop_deg / 2 + edge_deg] = edge
        pol[:, d <= loop_deg / 2] = loop
    return pol


def _bow(count=4, **kw):
    return dict({"kind": "bow", "count": count, "rotation_deg": 0.0, "r_inner": 50.0, "r_outer": 109.0,
                 "width_ratio": 0.8, "cx": 300.0, "cy": 200.0}, **kw)


def _run(pol, extra=(), count=4):
    layers = [("points", 1, 0.8, _bow(count), 1000.0, 60)] + list(extra)
    meta = [dict(zone=(0, 60), rows=(0, 60), eats=None, inner=None) for _ in layers]
    return edge_loops(layers, meta, pol, np.arange(50.0, 110.0), np.ones(pol.shape[1], bool), _Fr(), base=0)


def test_edged_loops_get_a_wider_backing_of_the_edge_color():
    stray = ("points", 2, 0.63, _bow(2, width_ratio=0.7), 500.0, 60)  # the edging read as a 2-lobed bow
    out, out_meta = _run(_polar(), [stray])
    backs = [lay for lay, m in zip(out, out_meta) if m.get("trim")]
    assert len(backs) == 1
    role, g, _, q, _, rb = backs[0]
    assert g == 2 and q["kind"] == "bow" and q["count"] == 4 and rb == 61
    assert float(q["width_ratio"]) > 0.8 and float(q["r_outer"]) > 109.0
    assert not any(lay[1] == 2 and not m.get("trim") for lay, m in zip(out, out_meta))  # the misread layer


def test_loops_lying_on_the_disc_get_no_edging():
    out, out_meta = _run(_polar(edge_deg=0.0))  # the label past each loop's edge is the disc
    assert not any(m.get("trim") for m in out_meta) and len(out) == 1


def test_an_edge_wider_than_a_ribbon_trim_is_no_edging():
    out, out_meta = _run(_polar(loop_deg=30.0, edge_deg=30.0))  # 26-58 units at radius 50-110
    assert not any(m.get("trim") for m in out_meta)


def test_a_pleated_edge_is_no_loops_to_edge():
    # 17 pleats with highlights between them (S6-15) read as a 17-loop bow: teeth, not loops
    out, out_meta = _run(_polar(n=17, loop_deg=14.0, edge_deg=3.0), count=17)
    assert not any(m.get("trim") for m in out_meta)


def test_a_star_between_the_loops_is_not_their_edging():
    star = ("points", 2, 0.9, dict(_bow(4), kind="star", rotation_deg=45.0), 800.0, 60)
    out, out_meta = _run(_polar(), [star])
    assert not any(m.get("trim") for m in out_meta)
    assert any(lay[1] == 2 and lay[3]["kind"] == "star" for lay in out)


def test_loops_with_edge_strips_already_are_left_alone():
    strips = ("points", 2, 0.8, dict(_bow(4), edge_width=4.0), 0.0, 60)
    layers = [("points", 1, 0.8, _bow(4), 1000.0, 60), strips]
    meta = [dict(zone=(0, 60), rows=(0, 60), eats=None, inner=None), dict(zone=(0, 60), rows=(0, 60), trim=True)]
    out, out_meta = edge_loops(layers, meta, _polar(), np.arange(50.0, 110.0), np.ones(720, bool), _Fr(), base=0)
    assert len(out) == 2


@pytest.mark.slow
def test_s2_15b_snowflake_loops_have_white_edges(monkeypatch):
    monkeypatch.setenv("ROSETTE_READER", "off")
    monkeypatch.setenv("ROSETTE_SAM2", "off")
    from rosette.pipeline import analyze_with_context
    from rosette.vision.templates import design_key
    # a new photo: its own design out of the head library, so the rules head is drawn
    model, _, _ = analyze_with_context((PHOTOS / "S2-15B.png").read_bytes(), exclude_key=design_key("S2-15B"))
    white = max(model.groups, key=lambda g: sum(g.rgb))
    loops = [q for q in model.parts if q.role == "points" and q.params.get("kind") == "bow"]
    assert any(q.color_group == white.id for q in loops)  # the edging, behind the snowflake loops


def _cover_parts(ri, ring=None):
    from rosette.model import Part
    loop = Part("points-0", "points", _bow(4, r_inner=ri, r_outer=148.5, width_ratio=0.5), 1, 21, 0.9)
    parts = [loop, Part("center", "center", {"cx": 300.0, "cy": 200.0, "r": 59.5}, 0, 50, 0.9)]
    if ring:
        parts.append(Part("ring-0", "ring", {"cx": 300.0, "cy": 200.0, "r_inner": ring[0], "r_outer": ring[1]}, 2, 30, 0.9))
    return loop, parts


def test_a_loop_meeting_the_center_tucks_under_it_corners_on_its_edge():
    from rosette.vision.parts_head import _points_start_at_cover
    loop, parts = _cover_parts(59.5)
    w = 0.5 * (148.5 - 59.5)
    _points_start_at_cover(parts)
    q = loop.params
    assert q["r_inner"] < 59.5
    assert abs(np.hypot(q["r_inner"], w / 2) - 59.5) < 0.01  # corners on the circle
    assert abs(q["width_ratio"] * (q["r_outer"] - q["r_inner"]) - w) < 0.05  # same width (ratio stored to 3 decimals)


def test_a_loop_under_a_ring_still_starts_at_the_ring_edge():
    from rosette.vision.parts_head import _points_start_at_cover
    loop, parts = _cover_parts(64.0, ring=(60.0, 78.0))
    _points_start_at_cover(parts)
    assert loop.params["r_inner"] == 78.0
