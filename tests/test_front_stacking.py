"""H16: a front layer in the head disc's color casts no stacking vote (its color cannot be told
from the disc behind it); its backing and the inner star on it vote against the other layers, and
the front lies just on its backing, under the inner star (S70-19: white under red under white over
the blue blocks). Synthetic labels only."""
import numpy as np
from shapely import contains_xy

from rosette.model import Part
from rosette.vision import parts_head as H
from rosette.vision.parts_common import Frame
from rosette.vision.prepare import Prepared

R = 300.0
CX, CY = 450.0, 330.0
FR = Frame(axis_x=CX, top=float(CY - R), scale=2.0)
CYU = 25.0 + R / 2  # the head center in canvas units


def _photo(labels):
    return Prepared(rgb=np.zeros(labels.shape + (3,), np.uint8), mask=labels >= 0, axis_x=CX, head=(CX, CY, R),
                    tails_top_y=CY + 0.5 * R, bottom_y=float(labels.shape[0] - 51), scale=2.0)


def _pts(pid, g, kind, ri, ro, rot, z, wr=0.7):
    return Part(pid, "points", {"cx": 300.0, "cy": CYU, "kind": kind, "count": 6, "r_inner": ri, "r_outer": ro,
                                "rotation_deg": rot, "width_ratio": wr}, g, z, 0.8)


def _painted(parts_bottom_up, shape=(1200, 900)):
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(float)
    x = 300.0 + (xx - FR.axis_x) / FR.scale
    y = 25.0 + (yy - FR.top) / FR.scale
    labels = np.full(shape, -1, np.int32)
    labels[np.hypot(x - 300.0, y - CYU) <= 140.0] = 0  # the red disc
    for q in parts_bottom_up:
        labels[contains_xy(H._part_geom(q), x, y)] = q.color_group
    return labels


def _head():
    return Part("head", "head", {"cx": 300.0, "cy": CYU, "r": 140.0, "pinked": 0}, 0, 10, 0.9)


def test_a_front_layer_in_the_disc_color_casts_no_vote():
    loops = _pts("points-1", 2, "bow", 45.0, 100.0, 0.0, 23)
    front = _pts("front-2", 0, "star", 45.0, 100.0, 30.0, 21)
    parts = [_head(), loops, front]
    labels = _painted([front, loops])  # the loops over the red star: blue in the whole overlap
    votes = H.stack_votes(parts, labels, _photo(labels), FR, set())
    assert not [v for v in votes if 2 in v[:2]]


def test_the_backing_and_the_inner_star_carry_the_front_layers_vote():
    # the photo: blocks under a white star, a red star (here hidden under the inner star) and an
    # inner white star; the H5 prior puts the loops over the stars
    loops = _pts("points-1", 2, "bow", 45.0, 100.0, 0.0, 23)
    back = _pts("front-3", 1, "star", 45.0, 112.0, 30.0, 20)
    front = _pts("front-2", 0, "star", 45.0, 100.0, 30.0, 21)
    inner = _pts("front-4", 1, "star", 45.0, 100.0, 30.0, 22)
    parts = [_head(), loops, front, back, inner]
    labels = _painted([loops, back, inner])
    H.photo_stacking(parts, labels, _photo(labels), FR, {id(back)})
    assert loops.z < back.z < front.z < inner.z
    assert sorted(q.z for q in parts[1:]) == [20, 21, 22, 23]
