"""Trims of points layers (H4): a low-confidence layer that is another layer's edge
strips or backing tips is replaced by them, and unknown specks in a seam between two
bands are folded into the band. Called by parts_head.detect_head on its layers (role,
label, confidence, params, area, outer radius row) and their meta (zone, rows, eats,
inner band)."""
from __future__ import annotations

import numpy as np

from rosette.vision.parts_common import Frame, r3

# A layer of label h read with low confidence (unknown, or points under EXPLAIN_MAX_CONF)
# may be another layer's trim (H4):
# - edge strips: a band of h right along both side edges of a loop (bow), at most
#   EDGE_MAX_SHARE of the loop's angular width wide and followed by a third label (h
#   filling the whole gap between loops is not a band), on >= EDGE_MIN_RADII of the
#   radii in the middle of the loop's extent, on both sides of >= EXPLAIN_MIN_LOBES of
#   the visible loops. Drawn as the designer does: two strips per loop, EDGE_SHARE of
#   the loop's widest width (median strip / loop width over the 14 strip layers on
#   loops in 13 designer files, quartiles 0.107-0.119);
# - backing tips: h shows on the axis of >= EXPLAIN_MIN_LOBES of another layer's
#   visible points (within BACK_TOL_DEG), continuing beyond their tips by
#   >= BACK_MIN_UNITS while covering under BACK_MAX_COV of the circle there (a band of h
#   outside the points is a ring, not tips). The designer draws backings as a frame or
#   tip behind each point (hollow frames in 9 of the 11 files with one; never a disc
#   under everything): drawn as a copy of the points reaching the measured tip.
# The layer is replaced by its trims when they cover >= EXPLAIN_MIN_SHARE of its pixels.
EXPLAIN_MAX_CONF = 0.6
EDGE_MAX_SHARE, EDGE_MIN_RADII, EDGE_SHARE = 0.35, 0.5, 0.108
BACK_TOL_DEG, BACK_MIN_UNITS, BACK_MAX_COV = 2.0, 3.0, 0.5
# The backing must appear on the axis within BACK_START_UNITS of the points' tip, and
# gaps of up to BACK_GAP_UNITS in it along the axis (texture, a stray pixel) are bridged.
BACK_START_UNITS, BACK_GAP_UNITS = 4.0, 2.0
EXPLAIN_MIN_LOBES, EXPLAIN_MIN_SHARE = 2 / 3, 0.5
# An unknown head-zone layer in a seam - a zone between two bands at most SEAM_PLEATS
# pleat depths wide (parts_head.RING_MAX_GROWTH_UNITS: the center's pinked edge meets a
# ring's pleats) - whose label is one of those bands' labels and whose area is under
# SPECK_MAX_UNITS2 is that band's fringe (H4): it is folded into the band instead of
# being drawn as a disc, and a seam left empty is closed (the outer band, a ring,
# starts where the inner one ends). Over the 76 photos 14 seams (11-16 units wide) hold
# 22 such unknowns of 1525-5729 units^2 (S35-19: 1525 and 1943), all folded. The next
# wider zones (S17-15 and S1-15B 21 units, S52-19 and S57-17 28) hold real layers in
# the designer files (S17-15: a 4-point star) and are kept, as are unknowns of a third
# color (a badge's print, a gold rim) and zones open on one side (S5-15's loops).
SEAM_PLEATS = 2
SPECK_MAX_UNITS2 = 6000.0


def _axis_cols(params: dict, t: int) -> list[int]:
    """Polar column of each point's axis (t columns, clockwise from +x)."""
    n, rot = int(params["count"]), float(params["rotation_deg"])
    return [int(round((rot + k * 360.0 / n) % 360 * t / 360)) % t for k in range(n)]


def _rows(units: float, radii: np.ndarray, fr: Frame) -> int:
    """Radius rows spanning `units` canvas units."""
    step = radii[1] - radii[0] if len(radii) > 1 else fr.scale
    return max(1, int(round(units * fr.scale / step)))


def _edge_bands(pol: np.ndarray, i0: int, i1: int, g: int, h: int, params: dict,
                win: np.ndarray) -> tuple[float, np.ndarray]:
    """(share of the visible loops with a band of h along both sides, polar mask of the
    band pixels found) for the loop layer of label g over radius rows i0..i1."""
    t = pol.shape[1]
    reach = t // int(params["count"])
    cut = (i1 - i0) // 5
    mid = range(i0 + cut, max(i0 + cut + 1, i1 - cut))
    mark = np.zeros(pol.shape, bool)
    lobes = []
    for c in _axis_cols(params, t):
        if not win[c]:
            continue
        sides = []
        for d in (1, -1):
            hits = seen = 0
            for i in range(i0, i1):
                row = pol[i]
                if row[c] != g:
                    continue
                steps = 0
                while row[(c + d * steps) % t] == g and steps < reach:
                    steps += 1
                m = 0
                while row[(c + d * (steps + m)) % t] == h and m < reach:
                    m += 1
                after = row[(c + d * (steps + m)) % t]
                band = 1 <= m <= EDGE_MAX_SHARE * 2 * steps and after not in (h, g)
                if band:  # with one sample of slack on each side (resampled edges)
                    mark[i, [(c + d * (steps + j)) % t for j in range(-1, m + 1)]] = True
                if i in mid:
                    seen += 1
                    hits += band
            sides.append(seen > 0 and hits >= EDGE_MIN_RADII * seen)
        lobes.append(all(sides))
    return (float(np.mean(lobes)) if lobes else 0.0), mark


def _backing_tips(pol: np.ndarray, radii: np.ndarray, i_out: int, h: int, params: dict,
                  win: np.ndarray, fr: Frame) -> tuple[float, int, np.ndarray]:
    """(share of the visible points with h on their axis beyond their tip, median
    outer row of h there, polar mask of the h pixels in those points' sectors) for a
    points layer ending at radius row i_out."""
    t = pol.shape[1]
    tol = max(1, int(round(BACK_TOL_DEG * t / 360)))
    cov_h = (pol == h)[:, win].mean(axis=1)
    mark = np.zeros(pol.shape, bool)
    half = fr.scale * float(params["width_ratio"]) * (float(params["r_outer"]) - float(params["r_inner"])) / 2
    near, gap = _rows(BACK_START_UNITS, radii, fr), _rows(BACK_GAP_UNITS, radii, fr)
    ok, tips = [], []
    for c in _axis_cols(params, t):
        if not win[c]:
            continue
        cols = [(c + o) % t for o in range(-tol, tol + 1)]
        has = (pol[:, cols] == h).any(axis=1)
        start = next((i for i in range(max(0, i_out - near), min(len(radii), i_out + near)) if has[i]), None)
        if start is None:
            ok.append(False)
            continue
        i = start
        while i + 1 < len(radii) and has[i + 1:i + 1 + gap].any():
            i += 1
        beyond = fr.d(radii[i] - radii[i_out])
        good = bool(beyond >= BACK_MIN_UNITS and cov_h[i_out:i + 1].mean() < BACK_MAX_COV)
        ok.append(good)
        if good:
            tips.append(i)
            for j in range(i + 1):
                span = int(np.ceil(np.degrees(np.arctan2(half, max(radii[j], 1e-6))) * t / 360)) + tol
                mark[j, [(c + o) % t for o in range(-span, span + 1)]] = True
    tip = int(np.median(tips)) if tips else i_out
    return (float(np.mean(ok)) if ok else 0.0), tip, mark & (pol == h)


def explain_trims(layers: list, meta: list[dict], pol: np.ndarray, radii: np.ndarray,
                  win: np.ndarray, fr: Frame, pleat_units: float) -> tuple[list, list[dict]]:
    """Replace low-confidence layers that are other layers' trims (edge strips along
    loops, backing tips behind points; EXPLAIN_*) by those trims. A layer that hosts
    a replacement's trims in the same pass is kept, so no trim outlives its host."""
    found = {}  # k -> trims replacing layer k, each (host index, layer, meta)
    for k, (role, h, share, params, area, rb) in enumerate(layers):
        m = meta[k]
        if role == "points" and share >= EXPLAIN_MAX_CONF:
            continue
        ra_h, rb_h = m["rows"]
        mine = np.zeros(pol.shape, bool)
        mine[ra_h:rb_h] = pol[ra_h:rb_h] == h
        mine[:, ~win] = False
        explained = np.zeros(pol.shape, bool)
        inner = m["inner"]
        if inner is not None and inner.label == h:  # the fringe of h's own band (pleats)
            explained[inner.i1:inner.i1 + _rows(pleat_units, radii, fr)] = True
        trims = []
        for j, (role2, g, share2, p2, _, rb2) in enumerate(layers):
            if role2 != "points" or g == h or j == k or meta[j]["zone"] != m["zone"]:
                continue
            i0, i1 = meta[j]["rows"]
            if p2["kind"] == "bow":
                frac, mark = _edge_bands(pol, i0, i1, g, h, p2, win)
                if frac >= EXPLAIN_MIN_LOBES:
                    w = float(p2["width_ratio"]) * (float(p2["r_outer"]) - float(p2["r_inner"]))
                    q = dict(p2, edge_width=r3(EDGE_SHARE * w))
                    trims.append((j, ("points", h, min(frac, share2), q, 0.0, rb2),
                                  dict(meta[j], eats=None, trim=True)))
                    explained |= mark
            frac, tip, mark = _backing_tips(pol, radii, i1 - 1, h, p2, win, fr)
            if frac >= EXPLAIN_MIN_LOBES and tip > i1 - 1:
                q = dict(p2, r_outer=r3(fr.d(radii[tip])))
                trims.append((j, ("points", h, min(frac, share2), q, 0.0, tip + 1),
                              dict(meta[j], rows=(i0, tip + 1), eats=None, trim=True)))
                explained |= mark
        if trims and (explained & mine).sum() >= EXPLAIN_MIN_SHARE * max(int(mine.sum()), 1):
            found[k] = trims
    hosts = {j for trims in found.values() for j, _, _ in trims}
    replaced = {k: trims for k, trims in found.items() if k not in hosts}
    out = [layers[k] for k in range(len(layers)) if k not in replaced]
    out_meta = [meta[k] for k in range(len(layers)) if k not in replaced]
    # trims come last: a strip ties its loop's outer radius and must sort after it (z)
    for trims in replaced.values():
        for _, layer, mm in trims:
            out.append(layer)
            out_meta.append(mm)
    return out, out_meta


def fold_specks(layers: list, meta: list[dict], bands: list, head_band, pleated: list,
                radii: np.ndarray, fr: Frame, pleat_units: float) -> tuple[list, list[dict]]:
    """Fold unknown layers that are a seam band's fringe (SEAM_PLEATS, SPECK_MAX_UNITS2)
    into that band and close a seam they leave empty."""
    out, out_meta, folded = [], [], set()
    for k, (role, h, share, params, area, rb) in enumerate(layers):
        a, b = meta[k]["zone"]
        inner = next((bd for bd in bands if bd.i1 == a), None)
        outer = next((bd for bd in bands if bd.i0 == b), None)
        if (role == "unknown" and inner is not None and outer is not None
                and fr.d(radii[b - 1] - radii[a]) <= SEAM_PLEATS * pleat_units
                and h in (inner.label, outer.label) and area / fr.scale ** 2 < SPECK_MAX_UNITS2):
            folded.add((a, b))
            continue
        out.append(layers[k])
        out_meta.append(meta[k])
    for a, b in folded:
        if any(m["zone"] == (a, b) for m in out_meta) or any(a <= i0 < b for i0, *_ in pleated):
            continue
        outer = next(bd for bd in bands if bd.i0 == b)
        if outer is not head_band:
            outer.i0 = a
    return out, out_meta
