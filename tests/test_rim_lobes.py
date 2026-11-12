"""A16 (C1): lobes overlapping into a full band are a points layer when their color is
n-fold beyond it (S55-15). Synthetic labels only."""
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


def test_runs_beyond_a_full_band_make_a_points_layer():
    t = 720
    win = np.ones(t, bool)
    rows = [np.ones(t, bool)] * 10  # the full band where the lobes overlap
    for k in range(20):  # beyond it five lobes, narrowing
        row = np.zeros(t, bool)
        half = int(60 - 2 * k)
        for c in range(5):
            row[(np.arange(-half, half) + c * t // 5) % t] = True
        rows.append(row)
    ind = np.array(rows)
    cov = ind.mean(axis=1)

    class F:
        @staticmethod
        def d(v):
            return v
    rr = np.arange(len(rows), dtype=float) + 40.0
    end, n, sc, rot = H._runs_beyond_band(ind, win, cov, rr, F, 100.0)
    assert n == 5 and sc >= H.SYM_T and end == len(rows)
    assert H._runs_beyond_band(ind[:12], win, cov[:12], rr[:12], F, 100.0) is None  # too short a stretch
