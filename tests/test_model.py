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
