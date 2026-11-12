"""ML-2: with SAM 2 pieces the stacking vote counts, in the drawn overlap of two layers, the pixels of
each layer's own visible pieces instead of its color label; a piece belongs to the smallest drawn
layer of its color holding its centroid; a layer without a piece votes with its label. Synthetic."""
import numpy as np
from shapely import contains_xy

from rosette.model import Part
from rosette.perception.pieces import Piece
from rosette.vision import parts_head as H
from rosette.vision.parts_common import Frame
from rosette.vision.prepare import Prepared

R = 300.0
CX, CY = 450.0, 330.0
FR = Frame(axis_x=CX, top=float(CY - R), scale=2.0)
CYU = 25.0 + R / 2
SHAPE = (1200, 900)


def _photo(labels):
    return Prepared(rgb=np.zeros(labels.shape + (3,), np.uint8), mask=labels >= 0, axis_x=CX, head=(CX, CY, R),
                    tails_top_y=CY + 0.5 * R, bottom_y=float(labels.shape[0] - 51), scale=2.0)


def _pts(pid, g, kind, ri, ro, rot, z, wr=0.7):
    return Part(pid, "points", {"cx": 300.0, "cy": CYU, "kind": kind, "count": 6, "r_inner": ri, "r_outer": ro,
                                "rotation_deg": rot, "width_ratio": wr}, g, z, 0.8)


def _canvas():
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(float)
    return 300.0 + (xx - FR.axis_x) / FR.scale, 25.0 + (yy - FR.top) / FR.scale


def _mask(q):
    x, y = _canvas()
    return contains_xy(H._part_geom(q), x, y)


def _painted(parts_bottom_up):
    x, y = _canvas()
    labels = np.full(SHAPE, -1, np.int32)
    labels[np.hypot(x - 300.0, y - CYU) <= 140.0] = 0
    for q in parts_bottom_up:
        labels[_mask(q)] = q.color_group
    return labels


def _piece(mask, g):
    ys, xs = np.nonzero(mask)
    return Piece(mask=mask, score=0.95, stability=0.95, area=int(mask.sum()),
                 bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), in_head=True, color_group=g)


def _lobes(mask, g, rot=0.0):
    """One piece per lobe of a 6-fold mask (SAM 2 keeps one-part masks)."""
    x, y = _canvas()
    d = np.degrees(np.arctan2(y - CYU, x - 300.0)) % 360
    return [_piece(mask & (np.abs((d - rot - 60 * k + 180) % 360 - 180) <= 30), g) for k in range(6)]


def _head():
    return Part("head", "head", {"cx": 300.0, "cy": CYU, "r": 140.0, "pinked": 0}, 0, 10, 0.9)


def _counts(votes, parts, first):
    """(samples of `first`, samples of the other) of the single vote."""
    (a, _b, na, nb), = votes
    return (na, nb) if parts[a] is first else (nb, na)


def test_the_vote_counts_visible_pieces_not_labels():
    # the labels show the loops (A) over the star; the pieces show the star (B) whole and the loops
    # cut by it: B wins
    loops = _pts("points-1", 2, "bow", 45.0, 100.0, 0.0, 20)
    star = _pts("points-2", 1, "star", 45.0, 110.0, 30.0, 21)
    parts = [_head(), loops, star]
    labels = _painted([star, loops])
    pieces = [_piece(_mask(star), 1)] + _lobes(_mask(loops) & ~_mask(star), 2)
    n_loops, n_star = _counts(H.stack_votes(parts, labels, _photo(labels), FR, set()), parts, loops)
    assert n_loops > n_star
    n_loops, n_star = _counts(H.stack_votes(parts, labels, _photo(labels), FR, set(), pieces), parts, loops)
    assert n_star > n_loops
    H.photo_stacking(parts, labels, _photo(labels), FR, set(), pieces)
    assert loops.z < star.z
    H.photo_stacking(parts, labels, _photo(labels), FR, set())
    assert star.z < loops.z


def test_a_piece_belongs_to_the_smallest_layer_of_its_color_holding_its_centroid():
    # the inner white arms lie inside the white backing and the inner white star: the inner star's
    back = _pts("front-3", 1, "star", 45.0, 112.0, 30.0, 20)
    inner = _pts("front-4", 1, "star", 45.0, 80.0, 30.0, 22)
    parts = [_head(), back, inner]
    geoms = {k: H._part_geom(q) for k, q in enumerate(parts) if q.role == "points"}
    own = H._own_pieces(parts, geoms, [_piece(_mask(inner), 1)], FR)
    assert list(own) == [2]


def test_the_disc_piece_is_not_given_to_a_star_of_its_color():
    star = _pts("front-2", 0, "star", 45.0, 100.0, 30.0, 21)
    parts = [_head(), star]
    x, y = _canvas()
    disc = np.hypot(x - 300.0, y - CYU) <= 140.0
    own = H._own_pieces(parts, {1: H._part_geom(star)}, [_piece(disc, 0)], FR)
    assert list(own) == [0]


def test_a_pair_votes_like_with_like():
    # the labels show the star over the loops; only the star has a piece (cut by the loops), the
    # loops have none: both layers count their labels, never a partial piece set against a full label
    loops = _pts("points-1", 2, "bow", 45.0, 100.0, 0.0, 20)
    star = _pts("points-2", 1, "star", 45.0, 110.0, 30.0, 21)
    parts = [_head(), loops, star]
    labels = _painted([loops, star])
    pieces = [_piece(_mask(star) & ~_mask(loops), 1)]
    plain = _counts(H.stack_votes(parts, labels, _photo(labels), FR, set()), parts, loops)
    with_pieces = _counts(H.stack_votes(parts, labels, _photo(labels), FR, set(), pieces), parts, loops)
    assert with_pieces == plain and plain[1] > 0
    # with a piece on each side the masks count: the star's cut piece shows less than its label
    both = _counts(H.stack_votes(parts, labels, _photo(labels), FR, set(), pieces + _lobes(_mask(loops), 2)),
                   parts, loops)
    assert both[1] < plain[1] and both[0] > plain[0]
