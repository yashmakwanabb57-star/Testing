"""A16 (C8): the lower star of layered stars is counted by its runs when they are a stable
multiple of the order the symmetry chose (S49-17, S63-17). Synthetic indicators only."""
import numpy as np

from rosette.vision import parts_head as H


def test_run_count_reads_the_lower_star_of_layered_stars():
    t = 720
    th = np.arange(t) * 0.5
    win = np.abs(th - 90.0) > H.TAIL_SECTOR_DEG
    rows = []
    for k in range(20):  # 12 tips, every third one hidden near its base by a 4-point star above
        row = np.zeros(t, bool)
        for c in range(12):
            half = 8 - k // 4
            if c % 3 == 0 and k < 3:
                continue
            row[(np.arange(-half, half + 1) + c * 60) % t] = True
        rows.append(row)
    ind = np.array(rows)
    assert H._run_count(ind, win, 4, 0.0)[0] == 12
    # loops whose runs double (S61-19, S57-17: a split loop end or a stripe reads two runs per
    # lobe) but whose score at 12 lies more than SPLIT_MARGIN under the one at 6: kept at 6
    loops = np.zeros((20, t), bool)
    for c in range(6):  # six wide loops on 30 + 60 k (one in the tails sector), each split near its axis
        mid = 60 + c * 120
        loops[:, (np.arange(-45, 46) + mid) % t] = True
        cut = (mid - 2 + (c % 3) * 6) % t
        loops[:, cut:cut + 4] = False
    xl = loops.astype(np.float32)
    s6, s12 = H._score_at(xl, win, 6), H._score_at(xl, win, 12)
    assert H.SPLIT_FLOOR <= s12 < s6 - H.SPLIT_MARGIN  # refused by the margin alone, not the floor
    vis = np.flatnonzero(win)
    runs = (np.diff(loops[:, vis].astype(np.int8), axis=1) == 1).sum(axis=1) + loops[:, vis[0]]
    assert abs(float(np.median(runs)) * t / len(vis) - 12) <= H.RUNCOUNT_TOL  # the runs read 12
    assert H._run_count(loops, win, 6, 30.0)[0] == 6
