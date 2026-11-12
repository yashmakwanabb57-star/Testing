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


def _vote(votes, a, b):
    """(samples of a, samples of b) of the vote between parts a and b, or None."""
    for x, y, nx, ny in votes:
        if (x, y) == (a, b):
            return nx, ny
        if (x, y) == (b, a):
            return ny, nx
    return None


def test_a_front_layer_in_the_disc_color_casts_no_vote():
    loops = _pts("points-1", 2, "bow", 45.0, 100.0, 0.0, 23)
    front = _pts("front-2", 0, "star", 45.0, 100.0, 30.0, 21)
    back = _pts("front-3", 1, "star", 45.0, 112.0, 30.0, 20)
    parts = [_head(), loops, front, back]
    labels = _painted([back, front, loops])  # the loops over both stars: blue in the whole overlap
    votes = H.stack_votes(parts, labels, _photo(labels), FR, {id(back)})
    assert not [v for v in votes if 2 in v[:2]]
    loops_n, back_n = _vote(votes, 1, 3)  # the backing still votes against the loops
    assert loops_n > 10 * max(back_n, 1) and loops_n + back_n >= H.STACK_MIN_UNITS2


def test_a_front_layer_without_its_backing_votes_as_a_host():
    # the backing dropped or rewritten after `_front_in_base`: the front is no void layer
    loops = _pts("points-1", 2, "bow", 45.0, 100.0, 0.0, 23)
    front = _pts("front-2", 0, "star", 45.0, 100.0, 30.0, 21)
    parts = [_head(), loops, front]
    labels = _painted([front, loops])
    votes = H.stack_votes(parts, labels, _photo(labels), FR, set())
    assert _vote(votes, 1, 2) is not None


def test_the_inner_star_is_cut_from_the_backings_vote():
    # inner star over the loops, backing under them: the white inside the inner star is the inner
    # star's evidence, not the backing's (I1: S70-19's blocks vs backing 360 / 583 -> 282 / 308)
    loops = _pts("points-1", 2, "star", 45.0, 100.0, 30.0, 21)
    back = _pts("front-3", 1, "star", 45.0, 112.0, 30.0, 20)
    front = _pts("front-2", 0, "star", 45.0, 100.0, 30.0, 22)
    inner = _pts("front-4", 1, "star", 45.0, 80.0, 30.0, 23)
    parts = [_head(), loops, front, back, inner]
    labels = _painted([back, front, loops, inner])
    votes = H.stack_votes(parts, labels, _photo(labels), FR, {id(back)})
    loops_n, back_n = _vote(votes, 1, 3)
    assert loops_n + back_n >= H.STACK_MIN_UNITS2 and back_n < 0.05 * loops_n
    loops_n, inner_n = _vote(votes, 1, 4)  # the backing is not cut from the inner star's vote
    assert inner_n > 10 * max(loops_n, 1)


def test_the_backing_and_the_inner_star_carry_the_front_layers_vote():
    # the photo: blocks under a white star, a red star (left unpainted: the backing's white shows
    # over the loops from the inner star's tips out) and an inner white star; the H5 prior puts
    # the loops over the stars
    loops = _pts("points-1", 2, "bow", 45.0, 100.0, 0.0, 23)
    back = _pts("front-3", 1, "star", 45.0, 112.0, 30.0, 20)
    front = _pts("front-2", 0, "star", 45.0, 100.0, 30.0, 21)
    inner = _pts("front-4", 1, "star", 45.0, 75.0, 30.0, 22)
    parts = [_head(), loops, front, back, inner]
    labels = _painted([loops, back, inner])
    H.photo_stacking(parts, labels, _photo(labels), FR, {id(back)})
    assert loops.z < back.z < front.z < inner.z
    assert sorted(q.z for q in parts[1:]) == [20, 21, 22, 23]


def test_the_inner_star_lies_above_the_backing_without_a_vote_between_them():
    # same color: no vote between them; the input z puts the inner star under the backing
    back = _pts("front-3", 1, "star", 45.0, 112.0, 30.0, 22)
    front = _pts("front-2", 0, "star", 45.0, 100.0, 30.0, 23)
    inner = _pts("front-4", 1, "star", 45.0, 75.0, 30.0, 20)
    parts = [_head(), front, back, inner]
    labels = _painted([])
    H.photo_stacking(parts, labels, _photo(labels), FR, {id(back)})
    assert back.z < front.z < inner.z
    assert sorted(q.z for q in parts[1:]) == [20, 22, 23]


def test_the_backing_keeps_its_fronts_place_in_the_prior():
    # the backing's own z is under the loops', its front's over them; no vote decides the loops
    # against the backing (the photo shows the disc's color in their overlap)
    loops = _pts("points-1", 2, "bow", 45.0, 100.0, 0.0, 21)
    back = _pts("front-3", 1, "star", 45.0, 112.0, 30.0, 20)
    front = _pts("front-2", 0, "star", 45.0, 100.0, 30.0, 22)
    parts = [_head(), loops, front, back]
    labels = _painted([])
    H.photo_stacking(parts, labels, _photo(labels), FR, {id(back)})
    assert loops.z < back.z < front.z
    assert sorted(q.z for q in parts[1:]) == [20, 21, 22]
