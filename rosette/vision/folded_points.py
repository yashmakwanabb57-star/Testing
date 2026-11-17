"""R74: a head of folded points in four colors (S68-19: purple pleats, purple points on the diagonals,
silver glitter points in pairs round them, lavender points left and right, light lavender points up and
down, a lavender badge; read as a plain disc with two blobs). Read along rays round the head: the head's
color comes and goes four times (the diagonals), a second color twice (one axis), a third eight times
half a step off the diagonals pairs (the glitter), a fourth color is the other axis. The folds cannot be
counted in the photo: each point is a stack of nested folds of its color (Vs with parallel edges, as the
designer draws them), tips and fold sizes measured on designer S68-19 (head radii)."""
from __future__ import annotations

import numpy as np

from ..model import Part
from .layer_classify import KEEP_PARAM

FOLD_BANDS = ((0.3, 0.7), (0.5, 0.7))  # where the diagonals and the axis pair show; where the glitter pairs show
FOLD_DIAG_SWING = 0.35  # the head color's four-fold swing
FOLD_AXIS_SWING = 0.2  # the axis color's two-fold swing
FOLD_PAIR_SWING = 0.25  # the glitter's eight-fold swing
FOLD_PHASE_TOL = 10.0  # degrees
# (count, tips, fold length along the axis, fold half-width), bottom to top: the glitter pairs, the diagonals, the axes.
# Designer S68-19 (head radii, measured along rays and on a close render): the folds of a point are one V moved
# along the axis a step at a time, its arms cut square, its inner edge parallel to the outer from a notch on the
# next fold's tip, so all the edges of a point run parallel; the innermost fold is the same V left whole.
FOLD_DIAG = (4, (0.86, 0.76, 0.66, 0.56, 0.36), 0.29, 0.14)
FOLD_PAIRS = (8, (0.74, 0.65, 0.57), 0.2, 0.1)
FOLD_AXIS = (2, (0.87, 0.77, 0.67, 0.57, 0.47), 0.31, 0.17)
FOLD_OTHER_AXIS = (2, (0.89, 0.79, 0.69, 0.59, 0.50), 0.3, 0.165)
FOLD_BADGE = 0.25


def _fourier(ind: np.ndarray, th: np.ndarray, n: int) -> tuple[float, float]:
    """The n-fold swing of an indicator round the head and its peak (degrees from straight up, clockwise)."""
    z = np.mean(ind * np.exp(-1j * n * th))
    return 2 * float(abs(z)), float(np.degrees(np.angle(z)) / n % (360.0 / n))


def folded_points(p, labels: np.ndarray, parts: list) -> list | None:
    head = next((q for q in parts if q.role == "head"), None)
    c = next((q for q in parts if q.role == "center"), None)
    if head is None or c is None or any(q.role in ("points", "ring") for q in parts):
        return None
    g = head.color_group
    h, w = labels.shape
    th = np.radians(np.arange(0.0, 360.0, 2.0))

    def band(lo, hi):
        radii = np.linspace(lo, hi, 20) * p.head[2]
        X = np.clip(np.rint(p.head[0] + np.outer(np.sin(th), radii)).astype(int), 0, w - 1)
        Y = np.clip(np.rint(p.head[1] - np.outer(np.cos(th), radii)).astype(int), 0, h - 1)
        return labels[Y, X]

    inner, mid = band(*FOLD_BANDS[0]), band(*FOLD_BANDS[1])
    amp4, diag = _fourier((inner == g).mean(axis=1), th, 4)
    if amp4 < FOLD_DIAG_SWING:
        return None
    cols = [int(x) for x in np.unique(inner) if x >= 0 and x != g]
    axis = max(((_fourier((inner == x).mean(axis=1), th, 2), x) for x in cols), default=None)
    if axis is None or axis[0][0] < FOLD_AXIS_SWING:
        return None
    (_, ax_peak), a = axis
    if abs((ax_peak - diag) % 90.0 - 45.0) > FOLD_PHASE_TOL:
        return None
    pair = max(((_fourier((mid == x).mean(axis=1), th, 8), x) for x in cols if x != a), default=None)
    if pair is None or pair[0][0] < FOLD_PAIR_SWING:
        return None
    (_, pair_peak), s = pair
    if abs((pair_peak - diag) % 45.0 - 22.5) > FOLD_PHASE_TOL:
        return None
    rest = [x for x in cols if x not in (a, s)]
    if len(rest) != 1:
        return None
    o = rest[0]
    hr = float(head.params["r"])
    geo = {"cx": head.params["cx"], "cy": head.params["cy"]}
    out = [q for q in parts if q.role not in ("unknown", "center")]
    z = 20
    # parts angles run from +x clockwise: a peak at d degrees from up is at d - 90
    # the glitter pairs under the diagonals (the owner's check of S68-19: the purple points come before them)
    for col, (n, tips, length, half), up in ((s, FOLD_PAIRS, pair_peak), (g, FOLD_DIAG, diag),
                                             (a, FOLD_AXIS, ax_peak), (o, FOLD_OTHER_AXIS, ax_peak + 90.0)):
        rot = round((up - 90.0) % (360.0 / n), 3)
        for k, tip in enumerate(tips):
            fold = {"kind": "chevron", "count": n, "r_outer": round(tip * hr, 3), "r_inner": round((tip - length) * hr, 3),
                    "fold_half": round(half * hr, 3), "fold_notch": round(tips[k + 1] * hr, 3) if k + 1 < len(tips) else 0.0,
                    "rotation_deg": rot, "width_ratio": round(2 * half / length, 4), "tips": 1, KEEP_PARAM: 1}
            out.append(Part(f"points-fold-{col}-{k}", "points", dict(geo, **fold), col, z, 0.5))
            z += 1
    c.params.update(geo, r=round(FOLD_BADGE * hr, 3))
    out.append(c)
    return out
