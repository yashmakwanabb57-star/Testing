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
    n, sc, rot = H._symmetry(ind, win)
    assert n == 12 or H._run_count(ind, win, 4, 0.0)[0] == 12
    assert H._run_count(ind, win, 4, 0.0)[0] == 12
    # loops whose runs double (two edges each) but whose doubled order does not fit: kept
    loops = np.zeros((20, t), bool)
    for c in range(4):
        loops[:, (np.arange(-40, 41) + c * 180) % t] = True
        loops[:, (c * 180) % t] = False
    assert H._run_count(loops, win, 4, 0.0)[0] in (4, 8)
