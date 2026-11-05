"""Tail detection: count, fan, widths and stripes from horizontal label profiles.

Rows at ROW_FRACS of the tails' height are cut through the mask: each run is a tail
(tails touching each other read as one ribbon, as the designer draws them), each
label run inside it a stripe. Centers and widths over the rows give a straight-line
fit (or, where fanned tails part only near the bottom, the tails' outer extent
and the head fix the fan); tails are then made mirror-symmetric about the axis (tail k and n-1-k share
mirrored offsets, widths and stripe fractions) and expressed the way
`geometry.tails` draws them: axes fanning out from (300, top_y), all bottoms on
y=bottom_y, one top/bottom width for all tails. top_y is where the outer axes meet
the center line when that is measurable; otherwise (and for a single tail) it sits
inside the head so the tails' top corners stay hidden behind it.
"""
from __future__ import annotations

import math

import numpy as np

from rosette.model import Part
from rosette.vision.parts_common import CANVAS_AXIS_X, Frame, majority, r3
from rosette.vision.prepare import Prepared

# Heights (share of the tails' span below the head) of the measured rows: the lower
# half densely, since fanned tails overlap near the head and part only further down.
ROW_FRACS = (0.25, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
# The tail count is read on the lowest rows (mode, ties to the larger count).
COUNT_ROWS = 3
# Rows with all tails apart spanning this share of the tails' height allow direct
# width and fan fits; otherwise the outer extent and the head fix the fan.
DENSE_SPAN = 0.15
# Gaps narrower than this (canvas units) inside a row are closed (antialiasing seams).
GAP_UNITS = 2.0
# Runs narrower than this (canvas units) are not tails / not stripes.
MIN_TAIL_UNITS, MIN_STRIPE_UNITS = 8.0, 3.0
# Rows a few px apart are averaged into one measurement (label noise).
ROW_BAND_PX = 2
# The tails' top corners are kept this far inside the head circle (share of the way).
TOP_INSET = 0.9


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) runs of True."""
    d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _close(row: np.ndarray, gap: int) -> np.ndarray:
    out = row.copy()
    runs = _runs(row)
    for (_, e0), (s1, _) in zip(runs, runs[1:]):
        if s1 - e0 <= gap:
            out[e0:s1] = True
    return out


def _stripes(lab: np.ndarray, min_px: int) -> list[tuple[int, int]]:
    """[(label, length)] along a tail cut, slivers merged into their larger neighbour."""
    seq: list[list[int]] = []
    for v in lab:
        if seq and seq[-1][0] == v:
            seq[-1][1] += 1
        else:
            seq.append([int(v), 1])
    while len(seq) > 1:
        k = min(range(len(seq)), key=lambda i: seq[i][1] if seq[i][0] >= 0 else -1)
        if seq[k][1] >= min_px and seq[k][0] >= 0:
            break
        nb = [j for j in (k - 1, k + 1) if 0 <= j < len(seq)]
        j = max(nb, key=lambda i: seq[i][1])
        seq[j][1] += seq[k][1]
        del seq[k]
        # re-join equal neighbours
        i = 0
        while i < len(seq) - 1:
            if seq[i][0] == seq[i + 1][0]:
                seq[i][1] += seq[i + 1][1]
                del seq[i + 1]
            else:
                i += 1
    return [(a, b) for a, b in seq]


def _cut(p: Prepared, labels: np.ndarray, y: int, fr: Frame):
    rows = slice(max(0, y - ROW_BAND_PX), y + ROW_BAND_PX + 1)
    m = p.mask[rows].mean(axis=0) >= 0.5
    m = _close(m, max(1, int(round(GAP_UNITS * fr.scale))))
    runs = [(s, e) for s, e in _runs(m) if e - s >= MIN_TAIL_UNITS * fr.scale]
    lab_row = labels[y]
    return [(s, e, _stripes(lab_row[s:e], max(1, int(round(MIN_STRIPE_UNITS * fr.scale)))))
            for s, e in runs]


def _line(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """(slope, intercept) of ys over xs; constant when fewer than two points."""
    if len(xs) >= 2 and np.ptp(xs) > 0:
        b, a = np.polyfit(xs, ys, 1)
        return float(b), float(a)
    return 0.0, float(np.mean(ys))


def _head_top_y(p: Prepared, fr: Frame, w_at) -> float:
    """Highest tail top whose corners (300 +- w/2, top_y) stay inside the head."""
    _, cy, r = p.head
    hy, hr = fr.y(cy), fr.d(r)
    top_y = hy
    for _ in range(8):
        half = w_at(top_y) / 2
        top_y = hy + TOP_INSET * math.sqrt(max(hr * hr - half * half, 0.0))
    return top_y


def detect_tails(p: Prepared, labels: np.ndarray, fr: Frame) -> list[Part]:
    top, bot = p.tails_top_y, p.bottom_y
    cuts = []
    for f in ROW_FRACS:
        y = int(round(top + f * (bot - top)))
        cuts.append((fr.y(y), _cut(p, labels, y, fr)))
    low = [len(c) for _, c in cuts[-COUNT_ROWS:]]
    n = max(set(low), key=lambda v: (low.count(v), v))  # tails part only toward the bottom
    if n == 0:
        return []
    ok = [(y, c) for y, c in cuts if len(c) == n]
    count_conf = 1.0 if len(set(low)) == 1 else 0.6

    # per tail, per row: canvas y, offset from the axis, width
    ys = np.array([y for y, _ in ok])
    dx = np.array([[fr.x((s + e) / 2) - CANVAS_AXIS_X for s, e, _ in c] for _, c in ok])  # (rows, n)
    wd = np.array([[fr.d(e - s) for s, e, _ in c] for _, c in ok])
    dx = (dx - dx[:, ::-1]) / 2  # mirror symmetry about the axis
    width = wd.mean(axis=1)  # one width per row, shared by all tails
    # outer extent of all tails together: seen on every row, overlapping or not
    lower = [(y, c) for y, c in cuts if c and y >= cuts[0][0]]
    ext_y = np.array([y for y, _ in lower])
    ext = np.array([fr.d(c[-1][1] - c[0][0]) for _, c in lower])
    ext_slope, _ = _line(ext_y, ext)

    span = fr.d(bot - top)
    if n > 1 and len(ys) >= 2 and np.ptp(ys) >= DENSE_SPAN * span:
        # tails apart over a good height: straight fits of width and outer offset;
        # the outer axes meet the center line at the fan's top
        b, a = _line(ys, width)
        slope, icpt = _line(ys, dx[:, -1])
        top_y = -icpt / slope if slope > 1e-6 else None
    elif n > 1:
        # tails apart only near the bottom: the outer extent's slope (2*tan(spread/2)
        # + width slope) and one measured row; the fan top is taken under the head
        y1, w1, d1 = float(ys.mean()), float(width.mean()), float(dx[:, -1].mean())
        b, a = 0.0, w1
        top_y = None
        for _ in range(4):
            ty = _head_top_y(p, fr, lambda y: max(1.0, a + b * y))
            slope = d1 / max(y1 - ty, 1e-6)
            b = ext_slope - 2 * slope
            a = w1 - b * y1
        top_y = ty
    else:
        b, a = _line(ys, width)
        top_y = None

    def w_at(y: float) -> float:
        return max(1.0, a + b * y)

    head_top = fr.y(p.head[1] - p.head[2])
    if top_y is None or not (head_top <= top_y < fr.y(top)):
        top_y = _head_top_y(p, fr, w_at)
    bottom_y = fr.y(bot)
    dy = ys - top_y
    angles = [math.atan2(float((dx[:, k] * dy).sum()), float((dy * dy).sum())) for k in range(n)]
    spread = math.degrees(angles[-1] - angles[0]) if n > 1 else 0.0

    # stripes per tail: rows agreeing on the most common stripe count
    fracs, groups, confs = [], [], []
    for k in range(n):
        seqs = [c[k][2] for _, c in ok]
        lens = [len(s) for s in seqs]
        m = max(set(lens), key=lens.count)
        use = [s for s in seqs if len(s) == m]
        fr_k = np.mean([[ln / sum(x[1] for x in s) for _, ln in s] for s in use], axis=0)
        lab_k = [majority(np.array([s[i][0] for s in use]))[0] for i in range(m)]
        fracs.append(fr_k)
        groups.append(lab_k)
        confs.append(count_conf * len(use) / len(seqs))
    for k in range(n):  # mirror-symmetric stripes where the colors mirror
        j = n - 1 - k
        if k <= j and groups[k] == groups[j][::-1]:
            f = (fracs[k] + fracs[j][::-1]) / 2
            fracs[k], fracs[j] = f, f[::-1]

    parts = []
    for k in range(n):
        z = 9 - min(9, int(round(abs(k - (n - 1) / 2) * 2)))  # middle tail on top
        f = fracs[k] / fracs[k].sum()
        stripes = ",".join(f"{v:.4f}" for v in f)
        params_common = {"tails_count": n, "top_y": r3(top_y), "bottom_y": r3(bottom_y),
                         "top_width": r3(w_at(top_y)), "bottom_width": r3(w_at(bottom_y)),
                         "spread_deg": r3(spread), "stripes": stripes}
        for i, g in enumerate(groups[k]):
            if g < 0:  # no color group along this stripe: keep it, as a flagged placeholder
                y_mid = float(ys.mean())
                x_mid = CANVAS_AXIS_X + float(dx[:, k].mean()) + (float(f[:i].sum()) + f[i] / 2 - 0.5) * w_at(y_mid)
                tail_g, _ = majority(np.array([x for x in groups[k] if x >= 0] or [-1]))
                parts.append(Part(f"unknown-tail-{k}-{i}", "unknown",
                                  {"cx": r3(x_mid), "cy": r3(y_mid), "r": r3(f[i] * w_at(y_mid) / 2)},
                                  max(tail_g, 0), 60, 0.1))
                continue
            parts.append(Part(f"tail-{k}-{i}", "tail_stripe",
                              {"tail_index": k, "stripe_index": i, **params_common},
                              g, z, r3(confs[k])))
    return parts
