"""Measured features of one head layer for the learned layer classifier (ML-1).

`layer_features(pol, radii, win, members, r0, r1, head_r, ...)` reads one color's
indicator off the polar label unwrap the head detector uses (`parts_head.detect_head`:
radii in px, RING_THETA angles, `win` leaving out the tails sector) between the layer's
radial extent r0..r1 (px) and returns a flat dict of numbers:

- extent: r_in, r_out, thickness in head radii; whether it reaches the rim;
- coverage: the share of the visible circle the color holds at 9 positions across the
  extent (`cov_*`), the fill share of the annulus, the share of full rows;
- lobe width profile (coverage x radius, normalised to its widest row) at 9 positions and
  its shape numbers (ends, peak position, fullness): what the H4 kind rules threshold;
- angular profile at 5 radii, resampled to 72 bins: its harmonic magnitudes 1..24
  relative to its mean (rotation free), and the high-frequency share (pleat texture);
- circular runs (lobes seen on a circle) at 5 radii and their median;
- turned self-correlation for every n in 2..24 (the rules' symmetry score) and its peaks;
- surroundings: the color just inside and just outside (full band of another color =
  enclosed, the same color inside = a core), the group's category / pattern / texture.

Nothing here decides anything: the classifier (`rosette.layer_model`) does.
"""
from __future__ import annotations

import numpy as np

# Candidate lobe counts (as parts_head.MAX_COUNT).
COUNTS = tuple(range(2, 25))
# Positions across the extent for the coverage and width profiles, and the radii of the
# angular profiles (shares of the extent).
PROFILE_POS = np.linspace(0.0, 1.0, 9)
ANGULAR_POS = (0.1, 0.3, 0.5, 0.7, 0.9)
ANGULAR_BINS = 72
HARMONICS = 24
# Rows (canvas units) looked at just inside / outside the extent for the surroundings.
SIDE_UNITS = 4.0
# A row is full when the color holds this share of the visible circle (parts_head.FULL_COV).
FULL = 0.8
# Reaches the rim: outer edge at or beyond this many head radii.
RIM_R = 0.93


def _score_at(x: np.ndarray, win: np.ndarray, n: int) -> float:
    """Mean Pearson correlation of x (rows, theta) with itself turned by m*T/n, m=1..n-1,
    over the angles valid in both (parts_head._score_at)."""
    t = x.shape[1]
    cs = []
    for m in range(1, n):
        sh = int(round(m * t / n))
        ok = win & np.roll(win, -sh)
        a = x[:, ok].ravel()
        b = np.roll(x, -sh, axis=1)[:, ok].ravel()
        sa, sb = a.std(), b.std()
        cs.append(0.0 if sa == 0 or sb == 0 else float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)))
    return float(np.mean(cs))


def symmetry_scores(ind: np.ndarray, win: np.ndarray) -> np.ndarray:
    """Turned self-correlation of the indicator for n in COUNTS."""
    x = ind.astype(np.float32)
    return np.array([_score_at(x, win, n) for n in COUNTS])


def _runs(row: np.ndarray, win: np.ndarray) -> int:
    """Runs of True on the visible arc (the tails sector cuts the circle open)."""
    if not win.all():
        k = int(np.flatnonzero(~win)[-1]) + 1  # start the arc right after the sector
        row, win = np.roll(row, -k), np.roll(win, -k)
        r = row[win]
        return int(np.count_nonzero(r & ~np.concatenate([[False], r[:-1]])))
    if row.all():
        return 1
    return int(np.count_nonzero(row & ~np.roll(row, 1)))


def _angular(row: np.ndarray, win: np.ndarray) -> np.ndarray:
    """72-bin angular profile (share of each bin's visible samples), sector bins filled
    with the visible mean."""
    t = len(row)
    b = np.arange(t) * ANGULAR_BINS // t
    vis = np.bincount(b[win], minlength=ANGULAR_BINS).astype(float)
    on = np.bincount(b[win], weights=row[win].astype(float), minlength=ANGULAR_BINS)
    mean = row[win].mean() if win.any() else 0.0
    return np.where(vis > 0, on / np.maximum(vis, 1), mean)


def _side(pol: np.ndarray, win: np.ndarray, rows: range, members: np.ndarray) -> tuple[float, float]:
    """(share of the same color, share of the strongest other color) over `rows`."""
    if len(rows) == 0:
        return 0.0, 0.0
    v = pol[rows.start:rows.stop][:, win]
    same = float(np.isin(v, members).mean())
    other = v[~np.isin(v, members) & (v >= 0)]
    if other.size == 0:
        return same, 0.0
    return same, float(np.bincount(other).max() / v.size)


def layer_features(pol: np.ndarray, radii: np.ndarray, win: np.ndarray, members, r0: float, r1: float,
                   head_r: float, scale: float, category: int | None = None, pattern: bool = False,
                   texture: float = 0.0) -> dict[str, float]:
    """Feature dict of the layer of color(s) `members` between radii r0..r1 (px).
    `pol` is the (len(radii), T) label unwrap, `win` the visible angles, `scale` px per
    canvas unit."""
    members = np.atleast_1d(np.asarray(members))
    i0 = int(np.clip(np.searchsorted(radii, r0), 0, len(radii) - 1))
    i1 = int(np.clip(np.searchsorted(radii, r1, side="right"), i0 + 1, len(radii)))
    if i1 - i0 < 3:  # at least three rows
        i1 = min(len(radii), i0 + 3)
        i0 = max(0, i1 - 3)
    ind = np.isin(pol[i0:i1], members)
    rr = radii[i0:i1]
    cov = ind[:, win].mean(axis=1)
    f: dict[str, float] = {}
    f["r_in"], f["r_out"] = float(r0 / head_r), float(r1 / head_r)
    f["thick"] = f["r_out"] - f["r_in"]
    f["rim"] = float(f["r_out"] >= RIM_R)
    x = np.linspace(0, 1, len(cov))
    for k, t in enumerate(PROFILE_POS):
        f[f"cov_{k}"] = float(np.interp(t, x, cov))
    f["fill"] = float((cov * rr).sum() / max(rr.sum(), 1e-9))
    f["full_rows"] = float((cov >= FULL).mean())
    f["cov_max"], f["cov_min"] = float(cov.max()), float(cov.min())
    w = cov * rr
    wn = w / max(w.max(), 1e-9)
    for k, t in enumerate(PROFILE_POS):
        f[f"w_{k}"] = float(np.interp(t, x, wn))
    q = max(1, len(wn) // 4)
    f["w_in"], f["w_out"] = float(wn[:q].mean()), float(wn[-q:].mean())
    f["w_peak"] = float(np.argmax(wn) / max(1, len(wn) - 1))
    f["w_full"] = float(wn.mean())
    # lobe width over length in units of the extent (needs a count: the best symmetry)
    sym = symmetry_scores(ind, win)
    nbest = COUNTS[int(np.argmax(sym))]
    f["width_ratio"] = float(w.max() * 2 * np.pi / nbest / max(rr[-1] - rr[0] + scale, 1e-9))
    for k, n in enumerate(COUNTS):
        f[f"sym_{n}"] = float(sym[k])
    f["sym_max"] = float(sym.max())
    f["sym_best_n"] = float(nbest)
    strong = [n for n, s in zip(COUNTS, sym) if s >= 0.6]
    f["sym_strong_n"] = float(max(strong)) if strong else 0.0
    # angular profiles
    harm = np.zeros(HARMONICS)
    runs = []
    hf = []
    for k, t in enumerate(ANGULAR_POS):
        i = int(round(t * (len(rr) - 1)))
        prof = _angular(ind[i], win)
        spec = np.abs(np.fft.rfft(prof))
        harm += spec[1:HARMONICS + 1] / max(spec[0], 1e-9)
        hf.append(float((spec[HARMONICS + 1:] ** 2).sum() / max((spec[1:] ** 2).sum(), 1e-9)))
        runs.append(_runs(ind[i], win))
        f[f"runs_{k}"] = float(runs[-1])
        f[f"acov_{k}"] = float(prof.mean())
    harm /= len(ANGULAR_POS)
    for k in range(HARMONICS):
        f[f"harm_{k + 1}"] = float(harm[k])
    f["runs_med"] = float(np.median(runs))
    f["runs_max"] = float(max(runs))
    f["hf_share"] = float(np.mean(hf))
    # surroundings
    side = max(1, int(round(SIDE_UNITS * scale / max(radii[1] - radii[0], 1e-9)))) if len(radii) > 1 else 1
    same_in, other_in = _side(pol, win, range(max(0, i0 - side), i0), members)
    same_out, other_out = _side(pol, win, range(i1, min(len(radii), i1 + side)), members)
    f["same_in"], f["other_in"] = same_in, other_in
    f["same_out"], f["other_out"] = same_out, other_out
    f["enclosed"] = float(other_in >= FULL and other_out >= FULL)
    f["center_start"] = float(r0 <= 3 * scale)
    for c in (1, 2, 3, 4):
        f[f"cat_{c}"] = float(category == c)
    f["pattern"] = float(bool(pattern))
    f["texture"] = float(texture)
    return f


def feature_names() -> list[str]:
    """Names in the order `layer_features` returns them (from a synthetic layer)."""
    t = 720
    radii = np.arange(0.5, 100, 1.0)
    pol = np.zeros((len(radii), t), np.int32)
    win = np.ones(t, bool)
    return list(layer_features(pol, radii, win, [0], 20, 60, 100, 1.0).keys())
