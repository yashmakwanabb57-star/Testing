from rosette.model import ColorGroup, Part, RosetteModel


def sample():
    return RosetteModel(
        parts=[Part("h", "head", {"cx": 300, "cy": 300, "r": 100, "pinked": 1}, 0, 10, 0.9)],
        groups=[ColorGroup(0, (10, 20, 30), False, 0.1, 0.5, 1, 0.8)],
        rose=True, warnings=["w"])


def test_json_roundtrip():
    import json
    m = sample()
    d = json.loads(json.dumps(m.to_json()))
    assert RosetteModel.from_json(d) == m


def _rejects(params):
    import pytest
    from rosette.model import check_params
    with pytest.raises(ValueError):
        check_params(params)


def test_check_params_rejects_degenerate_geometry():
    from rosette.model import check_params
    ok = {"cx": 300, "cy": 150, "r_inner": 50, "r_outer": 100, "width_ratio": 1, "count": 6}
    check_params(ok)
    for k in ("r", "r_inner", "r_outer"):
        _rejects({"cx": 300, "cy": 150, k: 0})
        _rejects({"cx": 300, "cy": 150, k: -5})
    _rejects({**ok, "r_inner": 100})  # r_inner must be below r_outer
    _rejects({**ok, "r_inner": 120})
    _rejects({**ok, "width_ratio": 0})
    _rejects({**ok, "width_ratio": -1})
    check_params({**ok, "edge_width": 2.5})  # a loop's edge strips (H4)
    _rejects({**ok, "edge_width": 0})
    _rejects({**ok, "edge_width": -2})
    _rejects({**ok, "cx": -1})
    _rejects({**ok, "cx": 601})
    _rejects({**ok, "cy": -0.5})
    _rejects({**ok, "cy": 851})
    _rejects({**ok, "r": "x"})
    check_params({**ok, "cx": 0, "cy": 850})  # canvas edges are on the canvas


def test_check_params_rejects_degenerate_tails():
    from rosette.model import check_params
    ok = {"tails_count": 2, "top_y": 300, "bottom_y": 825, "top_width": 80, "bottom_width": 120,
          "spread_deg": 10, "tail_index": 0, "stripe_index": 0}
    check_params(ok)
    _rejects({**ok, "top_y": 825})
    _rejects({**ok, "top_y": 900})
    _rejects({**ok, "top_width": 0})
    _rejects({**ok, "bottom_width": -3})
