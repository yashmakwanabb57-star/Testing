from types import SimpleNamespace

import numpy as np

from rosette.model import Part


def _flower(n: int = 6, lobes: bool = True):
    """A 600x600 photo, 1 px per canvas unit: a red head (0) with n pointed lobes reaching 210
    around a white rose (1) of radius 70 at (300, 230), tails (0) running down from the head."""
    h = w = 600
    yy, xx = np.mgrid[:h, :w]
    cx, cy = 300.0, 230.0
    r = np.hypot(xx - cx, yy - cy)
    th = np.arctan2(yy - cy, xx - cx)
    reach = 130 + 80 * np.abs(np.cos(n * th / 2)) ** 3 if lobes else np.full_like(r, 170.0)
    mask = r <= reach
    mask |= (np.abs(xx - cx) <= 90) & (yy >= cy) & (yy < 590)  # the tails
    labels = np.full((h, w), -1, dtype=int)
    labels[mask] = 0
    labels[r <= 70] = 1
    p = SimpleNamespace(axis_x=cx, scale=1.0, mask=mask, head=(cx, cy + 40.0, 220.0))
    return p, labels


def _rules_parts():
    return [Part("unknown-0", "unknown", {"cx": 300.0, "cy": 250.0, "r": 78.0}, 0, 20, 0.3),
            Part("center", "center", {"cx": 300.0, "cy": 260.0, "r": 30.0}, 1, 50, 0.5),
            Part("ring-0", "ring", {"cx": 300.0, "cy": 260.0, "r_inner": 74.0, "r_outer": 110.0}, 0, 30, 0.5),
            Part("tail-0", "tail_stripe", {"x0": 210.0}, 0, 1, 0.9)]


def test_r56_a_one_color_lobed_head_round_a_rose_is_a_petal_flower():
    from rosette.pipeline import _petal_flower
    from rosette.vision.layer_classify import KEEP_PARAM
    p, labels = _flower(6)
    out = _petal_flower(p, labels, _rules_parts())
    assert out is not None
    roles = sorted(q.role for q in out)
    assert roles == ["center", "points", "tail_stripe"]
    pet = next(q for q in out if q.role == "points")
    c = next(q for q in out if q.role == "center")
    assert pet.params["kind"] == "petal" and pet.params["count"] == 6 and pet.color_group == 0
    assert pet.params["tips"] == 1 and pet.params[KEEP_PARAM] == 1
    assert abs(pet.params["r_outer"] - 210) < 8
    assert abs(((pet.params["rotation_deg"] + 30) % 60) - 30) < 4  # a tip on +x, every 60 degrees
    from rosette.vision.parts_common import Frame
    assert c.color_group == 1 and abs(c.params["cy"] - Frame.of(p).y(230)) < 3 and abs(c.params["r"] - pet.params["r_outer"] / 2.55) < 1
    assert pet.params["r_inner"] < c.params["r"]  # the petals tuck under the rose
    assert pet.params["cy"] == c.params["cy"] and isinstance(c.params["cy"], float)
    assert pet.z < c.z


def test_r56_a_round_head_or_a_head_with_layers_stays():
    from rosette.pipeline import _petal_flower
    p, labels = _flower(lobes=False)
    assert _petal_flower(p, labels, _rules_parts()) is None
    p, labels = _flower(6)
    parts = _rules_parts() + [Part("points-0", "points", {"cx": 300.0, "cy": 230.0, "kind": "bow", "count": 6,
                                                         "r_inner": 70.0, "r_outer": 200.0, "rotation_deg": 0.0,
                                                         "width_ratio": 0.5}, 0, 20, 0.9)]
    assert _petal_flower(p, labels, parts) is None
