"""Tail detection: count, fan, widths and stripes from horizontal label profiles.

Rows at ROW_FRACS of the tails' height are cut through the mask: each run is a tail
(tails touching each other read as one ribbon, as the designer draws them), each
label run inside it a stripe. Centers and widths over the rows give a straight-line
fit; tails are then made mirror-symmetric about the axis (tail k and n-1-k share
mirrored offsets, widths and stripe fractions) and expressed the way
`geometry.tails` draws them: axes fanning out from (300, top_y), all bottoms on
y=bottom_y, one top/bottom width for all tails. top_y sits inside the head so the
tails' top corners stay hidden behind it.
"""
from __future__ import annotations

import math

import numpy as np

from rosette.model import Part
from rosette.vision.parts_common import CANVAS_AXIS_X, Frame, majority, r3
from rosette.vision.prepare import Prepared

# Heights (share of the tails' span below the head) of the measured rows.
ROW_FRACS = (0.25, 0.5, 0.8)
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


def detect_tails(p: Prepared, labels: np.ndarray, fr: Frame) -> list[Part]:
    top, bot = p.tails_top_y, p.bottom_y
    cuts = []
    for f in ROW_FRACS:
        y = int(round(top + f * (bot - top)))
        cuts.append((y, _cut(p, labels, y, fr)))
    n = len(cuts[-1][1])
    if n == 0:
        return []
    ok = [(y, c) for y, c in cuts if len(c) == n]
    count_conf = len(ok) / len(cuts)

    # per tail, per row: canvas y, offset from the axis, width
    ys = np.array([fr.y(y) for y, _ in ok])
    dx = np.array([[fr.x((s + e) / 2) - CANVAS_AXIS_X for s, e, _ in c] for _, c in ok])  # (rows, n)
    wd = np.array([[fr.d(e - s) for s, e, _ in c] for _, c in ok])
    dx = (dx - dx[:, ::-1]) / 2  # mirror symmetry about the axis
    width = wd.mean(axis=1)  # one width per row, shared by all tails
    if len(ys) >= 2:
        b, a = np.polyfit(ys, width, 1)
    else:
        b, a = 0.0, float(width[0])

    def w_at(y: float) -> float:
        return max(1.0, a + b * y)

    cx, cy, r = p.head
    hy, hr = fr.y(cy), fr.d(r)
    top_y = hy
    for _ in range(8):  # top corners (300 +- w/2, top_y) inside the head circle
        half = w_at(top_y) / 2
        top_y = hy + TOP_INSET * math.sqrt(max(hr * hr - half * half, 0.0))
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
            parts.append(Part(f"tail-{k}-{i}", "tail_stripe",
                              {"tail_index": k, "stripe_index": i, **params_common},
                              max(g, 0), z, r3(confs[k])))
    return parts
