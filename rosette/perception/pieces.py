"""Cloth pieces from SAM 2.1 by grid prompting, with a per-photo cache.

segment(p, settings) prompts SAM 2.1 with one positive point per grid cell (dense over the
head circle, coarse over the tails), takes each of the three masks per point as a candidate,
filters by predicted IoU, stability and area, removes duplicates by NMS and returns the pieces
in the `prepare` work frame. A second pass (`complete`) prompts the missing steps of a repeated
head piece and the borders just outside each head piece; a thin border piece is kept only when
it repeats. The image embedding is computed once.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import sam2
from .settings import Settings

# A mask is kept when the model's own IoU estimate is at least this.
PRED_IOU_MIN = 0.80
# ... and its stability (IoU of the mask at logits +1 and at logits -1) is at least this.
STABILITY_MIN = 0.90
# Two masks overlapping by more than this IoU are duplicates (the lower score is dropped).
NMS_IOU = 0.70
# Masks smaller / larger than these shares of the rosette mask's area are dropped.
MIN_AREA_FRAC = 0.002
MAX_AREA_FRAC = 0.95
# Point prompts per mask-decoder call.
POINTS_PER_BATCH = 64
# Head grid side used by segment(): the brief's fallback from 32 (the 32 x 32 grid took 132 s
# on S70-19 with the small model on this CPU, 24 x 24 took 84 s; the mask decoder dominates,
# about 0.12 s per point, so the tiny model is no faster). grid() keeps 32 as its default.
HEAD_N = 24
# A piece is in the head when at least this share of its pixels lies above the tails start
# (and its centroid does too).
IN_HEAD_FRAC = 0.8
# Logit offset used by the stability measure.
STABILITY_DELTA = 1.0
# Masks upsampled to the frame per post-processing chunk (bounds memory).
UPSAMPLE_CHUNK = 16

# Version of the pieces a cache file holds (folded into the cache key): 2 = all three
# hypotheses per point and the completion pass.
PIECES_VERSION = 2
# Symmetric re-prompting: head pieces of similar area (within SYM_AREA_TOL of each other) and
# centroid radius (within SYM_RADIUS_TOL of R) form a group; a group whose centroid angles all lie
# within SYM_TOL_DEG of n equal steps (the smallest n in SYM_N) is prompted at its missing steps.
SYM_TOL_DEG = 6.0
SYM_AREA_TOL = 0.35
SYM_RADIUS_TOL = 0.15
SYM_N = range(3, 17)
# A thin piece from an edge prompt is kept only when its group fills at least this many steps.
SYM_MIN_GROUP = 3
# A piece whose centroid lies within this share of R of the head centre is centred (the disc,
# the ring, the rose): it has no angle, so it is neither grouped nor edge-prompted.
CENTRE_FRAC = 0.15
# Edge prompts lie this many units (R / UNITS_PER_R) outside a head piece's outline.
EDGE_UNITS = 4
UNITS_PER_R = 150.0
# The repeat at a missing step is searched within this many units of the rotated piece, among
# pixels within the piece's own color spread (75th percentile of its Lab distances to its median,
# at least SYM_COLOR_MIN).
SYM_SHIFT_UNITS = 10
SYM_COLOR_MIN = 12.0
# At most this many points in the completion pass (symmetric prompts first).
MAX_EXTRA_POINTS = 160
# A piece is thin when its bbox fill ratio is below THIN_FILL or its area below THIN_AREA_FRAC
# of the rosette mask.
THIN_FILL = 0.35
THIN_AREA_FRAC = 0.02
# Another piece encloses a piece when it covers at least this share of it.
ENCLOSE_FRAC = 0.5

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "pieces"


@dataclass
class Piece:
    mask: np.ndarray  # (H, W) bool in the Prepared frame
    score: float  # predicted IoU
    stability: float
    area: int
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1), inclusive
    in_head: bool
    color_group: int | None = None  # majority labels value, set by the pipeline


@dataclass
class Pieces:
    pieces: list[Piece]
    model_id: str
    device: str
    seconds: float
    cached: bool
    passes: tuple[float, ...] = ()  # seconds of the grid pass and of the completion pass


def grid(p, head_n: int = 32, tail_cols: int = 8, tail_rows: int = 16) -> np.ndarray:
    """(N, 2) int (x, y) prompt points inside p.mask: head_n x head_n cell centres over the head
    circle's bounding square (kept inside the circle), plus tail_cols x tail_rows over the mask's
    rows below tails_top_y. Every point lies inside the image and the mask."""
    h, w = p.mask.shape
    cx, cy, r = p.head
    pts: list[tuple[int, int]] = []
    if r > 0 and head_n > 0:
        offs = (np.arange(head_n) + 0.5) / head_n * 2.0 - 1.0  # cell centres in [-1, 1]
        for v in offs:
            for u in offs:
                if u * u + v * v > 1.0:
                    continue
                x, y = int(round(cx + u * r)), int(round(cy + v * r))
                if 0 <= x < w and 0 <= y < h and p.mask[y, x]:
                    pts.append((x, y))
    top = min(max(int(np.ceil(p.tails_top_y)), 0), h)
    rows = np.nonzero(p.mask[top:].any(axis=1))[0]
    if len(rows) and tail_cols > 0 and tail_rows > 0:
        y0, y1 = top + rows[0], top + rows[-1]
        cols = np.nonzero(p.mask[y0:y1 + 1].any(axis=0))[0]
        x0, x1 = cols[0], cols[-1]
        for j in range(tail_rows):
            y = int(round(y0 + (j + 0.5) / tail_rows * (y1 - y0)))
            for i in range(tail_cols):
                x = int(round(x0 + (i + 0.5) / tail_cols * (x1 - x0)))
                if p.mask[y, x]:
                    pts.append((x, y))
    if not pts:
        return np.zeros((0, 2), int)
    return np.array(list(dict.fromkeys(pts)), int)


def nms(masks, scores: np.ndarray, iou: float = NMS_IOU) -> list[int]:
    """Greedy mask NMS: indices kept, highest score first."""
    n = len(masks)
    if n == 0:
        return []
    masks = [np.asarray(m) for m in masks]  # a list or a stacked array
    areas = [int(m.sum()) for m in masks]
    boxes = [_bbox(m) for m in masks]
    kept: list[int] = []
    for i in np.argsort(-np.asarray(scores), kind="stable"):
        if not _duplicate(masks[i], boxes[i], areas[i],
                          [(masks[k], boxes[k], areas[k]) for k in kept], iou):
            kept.append(int(i))
    return kept


def _duplicate(mask, box, area, kept, iou: float = NMS_IOU) -> bool:
    """True when mask overlaps one of kept [(mask, bbox, area)] by more than iou. The
    intersection is counted inside the overlap of the two boxes only."""
    for m2, b, a2 in kept:
        x0, y0 = max(box[0], b[0]), max(box[1], b[1])
        x1, y1 = min(box[2], b[2]), min(box[3], b[3])
        if x0 > x1 or y0 > y1:
            continue  # disjoint boxes: IoU 0
        inter = int(np.count_nonzero(mask[y0:y1 + 1, x0:x1 + 1] & m2[y0:y1 + 1, x0:x1 + 1]))
        union = area + a2 - inter
        if union and inter / union > iou:
            return True
    return False


def stability(logits: np.ndarray, delta: float = STABILITY_DELTA) -> float:
    """IoU of the mask thresholded at +delta and at -delta (1.0: a sharp mask)."""
    union = np.count_nonzero(logits > -delta)
    if union == 0:
        return 0.0
    return float(np.count_nonzero(logits > delta) / union)


def in_head(mask: np.ndarray, tails_top_y: float) -> bool:
    """Centroid above the tails start and at least IN_HEAD_FRAC of the pixels above it."""
    ys = np.nonzero(mask)[0]
    if len(ys) == 0:
        return False
    return bool(ys.mean() < tails_top_y and np.mean(ys < tails_top_y) >= IN_HEAD_FRAC)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _piece(mask: np.ndarray, score: float, stab: float, tails_top_y: float) -> Piece:
    return Piece(mask=mask, score=float(score), stability=float(stab), area=int(mask.sum()),
                 bbox=_bbox(mask), in_head=in_head(mask, tails_top_y))


def write_cache(path: Path, result: Pieces) -> None:
    """masks packed by np.packbits over the flattened (n, h, w), plus shape, scores,
    stability, model_id (and device, seconds, passes, version). Written atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if result.pieces:
        masks = np.stack([pc.mask for pc in result.pieces]).astype(bool)
    else:
        masks = np.zeros((0, 0, 0), bool)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        tmp,
        masks=np.packbits(masks.reshape(-1)),
        shape=np.array(masks.shape, np.int64),
        scores=np.array([pc.score for pc in result.pieces], np.float32),
        stability=np.array([pc.stability for pc in result.pieces], np.float32),
        model_id=np.array(result.model_id),
        device=np.array(result.device),
        seconds=np.array(result.seconds, np.float64),
        passes=np.array(result.passes, np.float64),
        version=np.array(PIECES_VERSION, np.int64),
    )
    tmp.replace(path)


def read_cache(path: Path, p) -> Pieces:
    with np.load(Path(path)) as z:
        shape = tuple(int(v) for v in z["shape"])
        masks = np.unpackbits(z["masks"], count=int(np.prod(shape))).astype(bool).reshape(shape)
        scores, stab = z["scores"], z["stability"]
        out = [_piece(masks[i], scores[i], stab[i], p.tails_top_y) for i in range(shape[0])]
        passes = tuple(float(v) for v in z["passes"]) if "passes" in z.files else ()
        return Pieces(out, str(z["model_id"]), str(z["device"]), float(z["seconds"]), True, passes)


def _embed(loaded, rgb: np.ndarray):
    """(image embeddings, original_sizes) of the photo: computed once per photo."""
    import torch

    inputs = loaded.processor(images=rgb, return_tensors="pt")
    with torch.inference_mode():
        emb = loaded.model.get_image_embeddings(inputs["pixel_values"].to(loaded.device))
    return emb, inputs["original_sizes"]


def _run_model(loaded, rgb: np.ndarray, points: np.ndarray, embedded=None):
    """(low-res logits (N, 3, h', w') of the three masks per point, float16 to bound memory,
    their predicted IoU (N, 3), original_sizes). One image embedding (`embedded` from `_embed`,
    or computed here); POINTS_PER_BATCH points per decoder call, each point its own object."""
    import torch

    model, proc, dev = loaded.model, loaded.processor, loaded.device
    emb, original_sizes = embedded if embedded is not None else _embed(loaded, rgb)
    logits, ious = [], []
    with torch.inference_mode():
        for s in range(0, len(points), POINTS_PER_BATCH):
            chunk = points[s:s + POINTS_PER_BATCH]
            for lg, io in _decode(model, proc, dev, emb, original_sizes, chunk):
                logits.append(lg)
                ious.append(io)
    if not logits:
        return (torch.zeros((0, 3, 256, 256), dtype=torch.float16), torch.zeros((0, 3)),
                original_sizes)
    return torch.cat(logits), torch.cat(ious), original_sizes


def _decode(model, proc, dev, emb, original_sizes, chunk):
    """The decoder on one chunk of points: [(logits (n, 3, h', w'), predicted IoU (n, 3))], all
    three hypotheses per point. The decoder copies the image embedding per point; when that runs
    out of memory the chunk is retried in halves (one point that still fails is skipped)."""
    import torch

    enc = proc(original_sizes=original_sizes,
               input_points=[[[[float(x), float(y)]] for x, y in chunk]],
               input_labels=[[[1] for _ in chunk]], return_tensors="pt")
    try:
        out = model(image_embeddings=emb, input_points=enc["input_points"].to(dev),
                    input_labels=enc["input_labels"].to(dev), multimask_output=True)
    except (RuntimeError, MemoryError) as exc:
        if "memory" not in str(exc).lower() and not isinstance(exc, MemoryError):
            raise
        if len(chunk) <= 1:
            return []
        half = len(chunk) // 2
        return (_decode(model, proc, dev, emb, original_sizes, chunk[:half])
                + _decode(model, proc, dev, emb, original_sizes, chunk[half:]))
    return [(out.pred_masks[0].to(torch.float16).cpu(), out.iou_scores[0].float().cpu())]


def _filtered(loaded, p, logits, ious, original_sizes, kept: list[Piece] | None = None,
              sources: bool = False, min_area: np.ndarray | None = None):
    """Every hypothesis (logits (N, 3, h', w'), ious (N, 3)) is a candidate: predicted-IoU,
    stability and area filters on the frame-size masks, then greedy NMS by predicted IoU (the
    candidates are taken best first, so only the kept masks are held). The `kept` pieces count
    as already kept: a candidate duplicating one is dropped. min_area: a per-point area floor in
    pixels in place of MIN_AREA_FRAC (the repeat of a found piece may be as small as that piece).
    Returns the new pieces only (with sources=True: (pieces, the prompt point index of each))."""
    rosette_area = int(p.mask.sum())
    lo, hi = MIN_AREA_FRAC * rosette_area, MAX_AREA_FRAC * rosette_area
    per_point = logits.shape[1] if logits.dim() == 4 else 1
    flat = logits.reshape(-1, *logits.shape[-2:])
    flat_iou = ious.reshape(-1).numpy().astype(np.float64)
    cand = np.nonzero(flat_iou >= PRED_IOU_MIN)[0]
    cand = cand[np.argsort(-flat_iou[cand], kind="stable")]
    have = [(pc.mask, pc.bbox, pc.area) for pc in kept or []]
    out: list[Piece] = []
    src: list[int] = []
    for s in range(0, len(cand), UPSAMPLE_CHUNK):
        sel = cand[s:s + UPSAMPLE_CHUNK]
        up = loaded.processor.post_process_masks(
            [flat[sel][:, None].float()], original_sizes, binarize=False)[0][:, 0].numpy()
        for j, i in enumerate(sel):
            st = stability(up[j])
            if st < STABILITY_MIN:
                continue
            m = up[j] > 0.0
            area = int(m.sum())
            floor = lo if min_area is None else min(lo, float(min_area[int(i) // per_point]))
            if not floor <= area <= hi:
                continue
            box = _bbox(m)
            if _duplicate(m, box, area, have):
                continue
            have.append((m, box, area))
            out.append(_piece(m, flat_iou[i], st, p.tails_top_y))
            src.append(int(i) // per_point)
    return (out, src) if sources else out


def _centroid(pc: Piece) -> tuple[float, float]:
    ys, xs = np.nonzero(pc.mask)
    return float(xs.mean()), float(ys.mean())


def _angular(pieces: list[Piece], head) -> list[tuple[int, float, float, float]]:
    """(index, area, centroid radius, centroid angle in degrees) of the head pieces that are
    neither the largest head piece (the disc) nor centred."""
    cx, cy, r = head
    heads = [i for i, pc in enumerate(pieces) if pc.in_head and pc.area > 0]
    if not heads or r <= 0:
        return []
    largest = max(heads, key=lambda i: pieces[i].area)
    out = []
    for i in heads:
        if i == largest:
            continue
        gx, gy = _centroid(pieces[i])
        rad = float(np.hypot(gx - cx, gy - cy))
        if rad < CENTRE_FRAC * r:
            continue
        out.append((i, float(pieces[i].area), rad, float(np.degrees(np.arctan2(gy - cy, gx - cx)))))
    return out


def _fit_steps(angles: list[float]) -> tuple[int, float] | None:
    """(n, offset in degrees) for the smallest n in SYM_N whose equal steps hold every angle
    within SYM_TOL_DEG, or None."""
    a = np.asarray(angles, float)
    for n in SYM_N:
        step = 360.0 / n
        res = (a - a[0] + step / 2.0) % step - step / 2.0
        m = float(res.mean())
        if np.all(np.abs(res - m) <= SYM_TOL_DEG):
            return n, float(a[0] + m)
    return None


def _groups(pieces: list[Piece], head):
    """Head pieces grouped by area and centroid radius: [(indices, (n, offset) | None, mean
    radius, occupied steps)]. Greedy, largest area first: a piece joins the first group whose
    first member is within SYM_AREA_TOL in area and SYM_RADIUS_TOL * R in radius."""
    r = head[2]
    items = sorted(_angular(pieces, head), key=lambda t: -t[1])
    groups: list[list[tuple[int, float, float, float]]] = []
    for it in items:
        for g in groups:
            seed = g[0]
            if (abs(it[1] - seed[1]) <= SYM_AREA_TOL * max(it[1], seed[1])
                    and abs(it[2] - seed[2]) <= SYM_RADIUS_TOL * r):
                g.append(it)
                break
        else:
            groups.append([it])
    out = []
    for g in groups:
        fit = _fit_steps([t[3] for t in g]) if len(g) >= 2 else None
        steps: set[int] = set()
        if fit is not None:
            n, off = fit
            steps = {int(round((t[3] - off) / (360.0 / n))) % n for t in g}
        out.append(([t[0] for t in g], fit, float(np.mean([t[2] for t in g])), steps))
    return out


def _interior_point(pc: Piece) -> tuple[float, float]:
    """The piece's centroid when it lies on the piece, else its pixel farthest from the outline
    (the centroid of a V lies in its notch, off the cloth)."""
    gx, gy = _centroid(pc)
    if pc.mask[int(round(gy)), int(round(gx))]:
        return gx, gy
    import cv2

    x0, y0, x1, y1 = pc.bbox
    sub = np.pad(pc.mask[y0:y1 + 1, x0:x1 + 1], 1).astype(np.uint8)
    dist = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
    yy, xx = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return float(x0 + xx - 1), float(y0 + yy - 1)


def _symmetric(pieces: list[Piece], head):
    """[(point (x, y), area floor, [(member index, rotation in degrees)])] for each missing step
    of each group (see `symmetric_prompts`). The point is the group's centroid at that step
    (mean radius, the fitted step); when the best member's centroid lies off its own cloth (a V)
    it is that member's interior point rotated by the steps between the two. The floor is
    (1 - SYM_AREA_TOL) of the group's largest area; the rotations take each member to the step."""
    cx, cy, _r = head
    out = []
    for idx, fit, rad, steps in _groups(pieces, head):
        if fit is None:
            continue
        n, off = fit
        step = 360.0 / n

        def k_of(i: int) -> int:
            gx, gy = _centroid(pieces[i])
            return int(round((np.degrees(np.arctan2(gy - cy, gx - cx)) - off) / step)) % n

        ks = {i: k_of(i) for i in idx}
        best = max(idx, key=lambda i: pieces[i].score)
        gx, gy = _centroid(pieces[best])
        on_cloth = bool(pieces[best].mask[int(round(gy)), int(round(gx))])
        bx, by = _interior_point(pieces[best])
        seed_area = max(pieces[i].area for i in idx)
        for k in range(n):
            if k in steps:
                continue
            if on_cloth:
                t = np.radians(off + k * step)
                pt = (cx + rad * np.cos(t), cy + rad * np.sin(t))
            else:
                pt = _rotated((bx, by), head, (k - ks[best]) * step)
            out.append((pt, (1.0 - SYM_AREA_TOL) * seed_area,
                        [(i, (k - ks[i]) * step) for i in idx]))
    return out


def _rotated(pt, head, deg: float) -> tuple[float, float]:
    cx, cy, _r = head
    t = np.radians(deg)
    dx, dy = pt[0] - cx, pt[1] - cy
    return (cx + dx * np.cos(t) - dy * np.sin(t), cy + dx * np.sin(t) + dy * np.cos(t))


def _registered(lab: np.ndarray, pc: Piece, deg: float, head) -> tuple[float, float] | None:
    """A prompt for the repeat of pc at deg degrees about the head centre: pc's mask is rotated
    there, shifted by up to SYM_SHIFT_UNITS units to cover the most pixels of pc's own color
    (the photo's steps are not exactly equal), and the point is the covered pixel farthest from
    the edge of that cover. None when the rotated mask covers no pixel of pc's color."""
    import cv2

    h, w = pc.mask.shape
    cx, cy, r = head
    shift = int(round(SYM_SHIFT_UNITS * r / UNITS_PER_R))
    x0, y0, x1, y1 = pc.bbox
    ref = lab[y0:y1 + 1, x0:x1 + 1][pc.mask[y0:y1 + 1, x0:x1 + 1]]
    med = np.median(ref, axis=0)
    tol = max(SYM_COLOR_MIN, float(np.percentile(np.linalg.norm(ref - med, axis=1), 75)))
    rot = cv2.getRotationMatrix2D((float(cx), float(cy)), -float(deg), 1.0)
    turned = cv2.warpAffine(pc.mask.astype(np.uint8), rot, (w, h), flags=cv2.INTER_NEAREST) > 0
    tb = _bbox(turned)
    if not turned.any():
        return None
    same = np.linalg.norm(lab - med, axis=2) <= tol
    tx0, ty0, tx1, ty1 = tb
    tmpl = turned[ty0:ty1 + 1, tx0:tx1 + 1].astype(np.float32)
    pad = np.pad(same, shift).astype(np.float32)
    win = pad[ty0:ty1 + 1 + 2 * shift, tx0:tx1 + 1 + 2 * shift]
    score = cv2.matchTemplate(win, tmpl, cv2.TM_CCORR)
    dy, dx = np.unravel_index(int(np.argmax(score)), score.shape)
    if score[dy, dx] <= 0:
        return None
    sx, sy = int(dx) - shift, int(dy) - shift
    cover = np.zeros((h, w), bool)
    ys, xs = np.nonzero(turned)
    ys, xs = ys + sy, xs + sx
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    cover[ys[ok], xs[ok]] = True
    cover &= same
    bx0, by0, bx1, by1 = _bbox(cover)
    sub = np.pad(cover[by0:by1 + 1, bx0:bx1 + 1], 1).astype(np.uint8)
    dist = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
    yy, xx = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return float(bx0 + xx - 1), float(by0 + yy - 1)


def _symmetric_points(p, pieces: list[Piece]) -> tuple[np.ndarray, np.ndarray]:
    """(points (k, 2), area floors (k,)) of the completion pass for the missing steps: per
    missing step, each member's registered repeat (`_registered`), else the geometric point."""
    import cv2

    recs = _symmetric(pieces, p.head)
    if not recs:
        return np.zeros((0, 2)), np.zeros(0)
    lab = cv2.cvtColor(p.rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    pts, floors = [], []
    for pt, floor, turns in recs:
        got = []
        for i, deg in turns:
            q = _registered(lab, pieces[i], deg, p.head)
            if q is not None and all(np.hypot(q[0] - a, q[1] - b) > 2.0 for a, b in got):
                got.append(q)
        for q in got or [pt]:
            pts.append(q)
            floors.append(floor)
    return np.array(pts, float).reshape(-1, 2), np.array(floors, float)


def symmetric_prompts(pieces: list[Piece], head: tuple[float, float, float]) -> np.ndarray:
    """(k, 2) float (x, y): for each group of two or more head pieces (similar area and centroid
    radius) whose centroid angles fit n equal steps, a point at each step no piece of the group
    holds: the group's centroid rotated there about the head centre at the group's mean radius
    (for a piece whose centroid is off its cloth, a V, a point of the piece rotated there)."""
    return np.array([pt for pt, _f, _t in _symmetric(pieces, head)], float).reshape(-1, 2)


def _covers(o: Piece, pc: Piece) -> float:
    """Share of pc's pixels inside o."""
    if pc.area == 0:
        return 0.0
    x0, y0, x1, y1 = pc.bbox
    inter = np.count_nonzero(o.mask[y0:y1 + 1, x0:x1 + 1] & pc.mask[y0:y1 + 1, x0:x1 + 1])
    return float(inter) / pc.area


def edge_prompts(pieces: list[Piece], head: tuple[float, float, float]) -> np.ndarray:
    """(k, 2) float (x, y): for each head piece that is neither the disc nor centred, a point
    EDGE_UNITS units (R / UNITS_PER_R) beyond the piece's farthest pixel on the ray from its
    centroid towards the head centre, away from it, and along both tangents (the two ends and
    the two long sides of a block). A point inside another piece that does not enclose this one
    is not offered: that piece is already found."""
    cx, cy, r = head
    off = EDGE_UNITS * r / UNITS_PER_R
    pts = []
    for i, _area, rad, ang in _angular(pieces, head):
        pc = pieces[i]
        h, w = pc.mask.shape
        gx, gy = _centroid(pc)
        t = np.radians(ang)
        ur = np.array([np.cos(t), np.sin(t)])
        ut = np.array([-ur[1], ur[0]])
        x0, y0, x1, y1 = pc.bbox
        diag = float(np.hypot(x1 - x0 + 1, y1 - y0 + 1))
        for d, limit in ((-ur, rad), (ur, diag), (ut, diag), (-ut, diag)):
            ts = np.arange(0.0, limit, 0.5)
            xs = np.round(gx + ts * d[0]).astype(int)
            ys = np.round(gy + ts * d[1]).astype(int)
            ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            hit = np.zeros(len(ts), bool)
            hit[ok] = pc.mask[ys[ok], xs[ok]]
            if not hit.any():
                continue
            far = ts[np.nonzero(hit)[0][-1]]
            qx, qy = gx + (far + off) * d[0], gy + (far + off) * d[1]
            xi, yi = int(round(qx)), int(round(qy))
            if not (0 <= xi < w and 0 <= yi < h) or pc.mask[yi, xi]:
                continue
            if any(j != i and o.mask[yi, xi] and _covers(o, pc) < ENCLOSE_FRAC
                   for j, o in enumerate(pieces)):
                continue
            pts.append((qx, qy))
    return np.array(pts, float).reshape(-1, 2)


def _thin(pc: Piece, p) -> bool:
    """Bbox fill ratio below THIN_FILL or area below THIN_AREA_FRAC of the rosette."""
    x0, y0, x1, y1 = pc.bbox
    fill = pc.area / max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
    return fill < THIN_FILL or pc.area < THIN_AREA_FRAC * int(p.mask.sum())


def _kept_edge_pieces(existing: list[Piece], new: list[Piece], p, head) -> list[Piece]:
    """The pieces from edge prompts that stay: a thin one only when its group (formed with the
    existing and the other new pieces) holds at least SYM_MIN_GROUP of its equal steps."""
    if not any(_thin(pc, p) for pc in new):
        return list(new)
    ok: set[int] = set()
    for idx, fit, _rad, steps in _groups(list(existing) + list(new), head):
        if fit is not None and len(steps) >= SYM_MIN_GROUP:
            ok.update(idx)
    base = len(existing)
    return [pc for j, pc in enumerate(new) if not _thin(pc, p) or base + j in ok]


def complete(p, loaded, pieces: list[Piece], original_sizes, emb=None) -> list[Piece]:
    """One extra decoder pass over `symmetric_prompts` and `edge_prompts` (inside the image and
    the mask, at most MAX_EXTRA_POINTS, symmetric first). The new pieces pass the same filters
    (a symmetric repeat's area floor is its group's, see `_symmetric`) and are NMS-merged into
    `pieces`, which stay as they are; a thin piece from an edge prompt stays only when it repeats.
    emb: the image embedding of the first pass (else recomputed)."""
    h, w = p.mask.shape

    def inside_mask(pts: np.ndarray) -> np.ndarray:
        xi, yi = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        ok[ok] = p.mask[yi[ok], xi[ok]]
        return ok

    sym_all, floors = _symmetric_points(p, pieces)
    ok = inside_mask(sym_all)
    sym, floors = sym_all[ok][:MAX_EXTRA_POINTS], floors[ok][:MAX_EXTRA_POINTS]
    edge_all = edge_prompts(pieces, p.head)
    edge = edge_all[inside_mask(edge_all)][:MAX_EXTRA_POINTS - len(sym)]
    pts = np.concatenate([sym, edge])
    if not len(pts):
        return list(pieces)
    embedded = (emb, original_sizes) if emb is not None else None
    logits, ious, sizes = _run_model(loaded, p.rgb, pts, embedded=embedded)
    # a symmetric prompt's repeat may be as small as its group allows; edge prompts keep the floor
    min_area = np.concatenate([floors, np.zeros(len(edge))])
    min_area[len(sym):] = np.inf
    new, src = _filtered(loaded, p, logits, ious, sizes, kept=pieces, sources=True,
                         min_area=min_area)
    from_sym = [pc for pc, k in zip(new, src) if k < len(sym)]
    from_edge = [pc for pc, k in zip(new, src) if k >= len(sym)]
    return (list(pieces) + from_sym
            + _kept_edge_pieces(list(pieces) + from_sym, from_edge, p, p.head))


def default_key(p, settings: Settings, head_n: int = HEAD_N) -> str:
    """sha256 of the photo's pixels, the model repo, the head grid size and PIECES_VERSION:
    pieces of another model, grid or version are never served for this one."""
    tag = f"|{sam2.REPOS.get(settings.model, settings.model)}|{head_n}|v{PIECES_VERSION}".encode()
    return hashlib.sha256(p.rgb.tobytes() + tag).hexdigest()


def segment(p, settings: Settings, cache_dir: Path | None = DEFAULT_CACHE_DIR,
            key: str | None = None, head_n: int = HEAD_N) -> Pieces:
    """Pieces of the prepared photo. With a cache_dir, `<cache_dir>/<key>.npz` (key: `default_key`,
    the photo, model, grid and pieces version, by default) is read when present, and the model
    is then never loaded; otherwise the pieces are computed and written there. seconds: model
    time of both passes (embedding, decoding, filters); passes: (grid pass, completion pass)."""
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"{key or default_key(p, settings, head_n)}.npz"
        if path.exists():
            return read_cache(path, p)
    loaded = sam2.load(settings)
    t0 = time.perf_counter()
    emb, sizes = _embed(loaded, p.rgb)
    logits, ious, _ = _run_model(loaded, p.rgb, grid(p, head_n=head_n), embedded=(emb, sizes))
    found = _filtered(loaded, p, logits, ious, sizes)
    del logits
    t1 = time.perf_counter()
    found = complete(p, loaded, found, sizes, emb=emb)
    t2 = time.perf_counter()
    result = Pieces(found, loaded.model_id, loaded.device, t2 - t0, False, (t1 - t0, t2 - t1))
    if path is not None:
        write_cache(path, result)
    return result
