"""R19 Part B: a front in the disc's color read from SAM 2 pieces with no backing (S43-15: six
turquoise arms over glitter blocks) casts no label vote (H16, ML-2); with PIECE_STACKING off its
pieces still decide its pair with a layer that has pieces when the vote is unanimous and holds
PIECE_UNANIMOUS_MIN samples. Synthetic labels only."""
import numpy as np

from rosette.vision import parts_head as H
from rosette.vision.parts import detect_parts
from tests.test_front_in_base import R, _arms, _block_pieces, _blocks, _grid, _photo, _red_star_over_blocks


def _parts(labels, pieces):
    return [q for q in detect_parts(_photo(labels), labels, pieces=pieces) if q.role != "tail_stripe"]


def _front_and_blocks(parts):
    f = [q for q in parts if q.role == "points" and q.color_group == 0]
    b = [q for q in parts if q.role == "points" and q.color_group == 2]
    assert len(f) == 1 and f[0].params.get("source") == "piece" and len(b) == 1
    return f[0], b[0]


def test_a_unanimous_piece_vote_puts_a_loose_front_over_the_blocks():
    # the arm pieces are whole, the block pieces are cut by them: the front lies on the blocks,
    # where the H5 prior (loops over a star) put it under them
    assert not H.PIECE_STACKING
    labels, star, blocks, d = _red_star_over_blocks()
    f, b = _front_and_blocks(_parts(labels, _arms(star, d) + _block_pieces(blocks, d)))
    assert f.z > b.z


def test_a_unanimous_piece_vote_for_the_blocks_keeps_them_on_top():
    labels, star, blocks, d = _red_star_over_blocks()
    yy, xx, dx, dy, r, th = _grid()
    whole = _blocks(dx, dy, 6, 0, 0.3 * R, 0.75 * R, 0.16 * R) & (r > 0.3 * R)
    labels[whole] = 2
    f, b = _front_and_blocks(_parts(labels, _arms(star & ~whole, d) + _block_pieces(whole, d)))
    assert b.z > f.z


def test_a_split_piece_vote_leaves_the_prior(monkeypatch):
    # every overlap sample on both masks: no unanimous vote, the front stays void under the blocks
    labels, star, blocks, d = _red_star_over_blocks()
    yy, xx, dx, dy, r, th = _grid()
    whole = _blocks(dx, dy, 6, 0, 0.3 * R, 0.75 * R, 0.16 * R) & (r > 0.3 * R)
    f, b = _front_and_blocks(_parts(labels, _arms(star, d) + _block_pieces(whole, d)))
    assert b.z > f.z


def test_without_pieces_nothing_changes():
    labels, star, blocks, d = _red_star_over_blocks()
    a = [(q.id, q.params, q.z) for q in detect_parts(_photo(labels), labels) if q.role != "tail_stripe"]
    assert not any(p.get("source") for _, p, _ in a)
