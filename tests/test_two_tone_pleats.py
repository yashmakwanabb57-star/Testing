"""A16 (C6, ruling): a two-tone pleated edge (H11 ALT pair) is one pinked ring in the color
holding more of it (S6-15, S69-19). Synthetic labels only."""
import numpy as np

from rosette.vision import parts_head as H
from rosette.vision.parts import detect_parts
from rosette.vision.prepare import Prepared

R = 300.0
CX, CY = 450.0, 330.0


def _grid(h=1200, w=900):
    yy, xx = np.mgrid[:h, :w].astype(float)
    dx, dy = xx - CX, yy - CY
    return yy, xx, dx, dy, np.hypot(dx, dy), np.degrees(np.arctan2(dy, dx)) % 360


def _base():
    yy, xx, dx, dy, r, th = _grid()
    labels = np.full(r.shape, -1, np.int32)
    labels[(yy > CY + 0.5 * R) & (yy < 1150) & (np.abs(xx - CX) < 110)] = 9
    return labels, dx, dy, r, th


def _photo(labels, rgb=None):
    return Prepared(rgb=np.zeros(labels.shape + (3,), np.uint8) if rgb is None else rgb, mask=labels >= 0,
                    axis_x=CX, head=(CX, CY, R), tails_top_y=CY + 0.5 * R, bottom_y=1149.0, scale=2.0)


def _head(parts):
    return [q for q in parts if q.role != "tail_stripe"]


def test_a_two_tone_pleated_edge_is_one_pinked_ring_in_its_larger_color():
    """S6-15: a wine pleated ring whose highlights read as a pink group: 17 + 17 alternating
    pleats between two bands. One pinked ring of 17 pleats in wine (the larger share)."""
    labels, dx, dy, r, th = _base()
    labels[r <= R] = 0
    n = 17
    ph = ((th % (360 / n)) - 180 / n) / (180 / n)  # -1..1 across one pleat
    t = np.clip((r - 0.45 * R) / (0.35 * R), 0, 1)
    edge = (r > 0.45 * R) & (r < 0.8 * R)
    labels[edge] = 5  # pink highlights between the pleats
    labels[edge & (np.abs(ph) <= 0.9 - 0.2 * np.abs(2 * t - 1))] = 1  # wine pleats, the larger share
    labels[r <= 0.45 * R] = 2
    labels[(r >= 0.8 * R) & (r <= 0.9 * R)] = 3
    labels[r <= 0.25 * R] = 4
    parts = _head(detect_parts(_photo(labels), labels))
    assert not any(q.role == "points" and q.color_group in (1, 5) for q in parts)
    rings = [q for q in parts if q.role == "ring" and q.color_group in (1, 5)]
    assert len(rings) == 1 and rings[0].color_group == 1 and rings[0].params.get("pleats") == n
