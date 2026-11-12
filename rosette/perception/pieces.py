"""Cloth pieces from SAM 2.1 by grid prompting, with a per-photo cache.

segment(p, settings) prompts SAM 2.1 with one positive point per grid cell (dense over the
head circle, coarse over the tails), keeps the best of the three masks per point by
predicted IoU, filters by predicted IoU, stability and area, removes duplicates by NMS and
returns the pieces in the `prepare` work frame. The image embedding is computed once.
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
    flat = masks.reshape(n, -1)
    areas = flat.sum(axis=1)
    boxes = [_bbox(m) for m in masks]
    kept: list[int] = []
    for i in np.argsort(-np.asarray(scores), kind="stable"):
        dup = False
        for k in kept:
            a, b = boxes[i], boxes[k]
            if a[0] > b[2] or b[0] > a[2] or a[1] > b[3] or b[1] > a[3]:
                continue  # disjoint boxes: IoU 0
            inter = np.logical_and(flat[i], flat[k]).sum()
            union = areas[i] + areas[k] - inter
            if union and inter / union > iou:
                dup = True
                break
        if not dup:
            kept.append(int(i))
    return kept


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
    stability, model_id (and device, seconds). Written atomically."""
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
    )
    tmp.replace(path)


def read_cache(path: Path, p) -> Pieces:
    with np.load(Path(path)) as z:
        shape = tuple(int(v) for v in z["shape"])
        masks = np.unpackbits(z["masks"], count=int(np.prod(shape))).astype(bool).reshape(shape)
        scores, stab = z["scores"], z["stability"]
        out = [_piece(masks[i], scores[i], stab[i], p.tails_top_y) for i in range(shape[0])]
        return Pieces(out, str(z["model_id"]), str(z["device"]), float(z["seconds"]), True)


def _run_model(loaded, rgb: np.ndarray, points: np.ndarray):
    """(low-res logits (N, h', w') of the best of the three masks per point by predicted IoU,
    that predicted IoU (N,), original_sizes). One image embedding; POINTS_PER_BATCH points
    per decoder call, each point its own object."""
    import torch

    model, proc, dev = loaded.model, loaded.processor, loaded.device
    inputs = proc(images=rgb, return_tensors="pt")
    original_sizes = inputs["original_sizes"]
    logits, ious = [], []
    with torch.inference_mode():
        emb = model.get_image_embeddings(inputs["pixel_values"].to(dev))
        for s in range(0, len(points), POINTS_PER_BATCH):
            chunk = points[s:s + POINTS_PER_BATCH]
            for lg, io in _decode(model, proc, dev, emb, original_sizes, chunk):
                logits.append(lg)
                ious.append(io)
    if not logits:
        return torch.zeros((0, 256, 256)), torch.zeros((0,)), original_sizes
    return torch.cat(logits), torch.cat(ious), original_sizes


def _decode(model, proc, dev, emb, original_sizes, chunk):
    """The decoder on one chunk of points: [(logits (n, h', w'), predicted IoU (n,))]. The
    decoder copies the image embedding per point; when that runs out of memory the chunk is
    retried in halves (one point that still fails is skipped)."""
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
    iou = out.iou_scores[0]  # (n, 3)
    best = iou.argmax(dim=1)
    idx = torch.arange(len(chunk), device=iou.device)
    return [(out.pred_masks[0][idx, best].float().cpu(), iou[idx, best].float().cpu())]


def _filtered(loaded, p, logits, ious, original_sizes) -> list[Piece]:
    """Predicted-IoU, stability and area filters on the frame-size masks, then NMS."""
    rosette_area = int(p.mask.sum())
    lo, hi = MIN_AREA_FRAC * rosette_area, MAX_AREA_FRAC * rosette_area
    keep = np.nonzero(ious.numpy() >= PRED_IOU_MIN)[0]
    masks, scores, stabs = [], [], []
    for s in range(0, len(keep), UPSAMPLE_CHUNK):
        sel = keep[s:s + UPSAMPLE_CHUNK]
        up = loaded.processor.post_process_masks(
            [logits[sel][:, None]], original_sizes, binarize=False)[0][:, 0].numpy()
        for j, i in enumerate(sel):
            st = stability(up[j])
            if st < STABILITY_MIN:
                continue
            m = up[j] > 0.0
            area = int(m.sum())
            if not lo <= area <= hi:
                continue
            masks.append(m)
            scores.append(float(ious[i]))
            stabs.append(st)
    if not masks:
        return []
    order = nms(masks, np.array(scores))
    return [_piece(masks[i], scores[i], stabs[i], p.tails_top_y) for i in order]


def default_key(p, settings: Settings, head_n: int = HEAD_N) -> str:
    """sha256 of the photo's pixels, the model repo and the head grid size: pieces of another
    model or grid are never served for this one."""
    tag = f"|{sam2.REPOS.get(settings.model, settings.model)}|{head_n}".encode()
    return hashlib.sha256(p.rgb.tobytes() + tag).hexdigest()


def segment(p, settings: Settings, cache_dir: Path | None = DEFAULT_CACHE_DIR,
            key: str | None = None, head_n: int = HEAD_N) -> Pieces:
    """Pieces of the prepared photo. With a cache_dir, `<cache_dir>/<key>.npz` (key: `default_key`,
    the photo, model and grid, by default) is read when present, and the model is then never loaded; otherwise the
    pieces are computed and written there. seconds: model time (embedding, decoding, filters)."""
    path = None
    if cache_dir is not None:
        if key is None:
            key = default_key(p, settings, head_n)
            legacy = Path(cache_dir) / f"{hashlib.sha256(p.rgb.tobytes()).hexdigest()}.npz"
        else:
            legacy = None
        path = Path(cache_dir) / f"{key}.npz"
        if not path.exists() and legacy is not None and legacy.exists():
            with np.load(legacy) as z:  # a file from before the key carried the model and grid
                same = str(z["model_id"]) == sam2.REPOS.get(settings.model)
            if same:
                legacy.replace(path)
        if path.exists():
            return read_cache(path, p)
    loaded = sam2.load(settings)
    t0 = time.perf_counter()
    logits, ious, sizes = _run_model(loaded, p.rgb, grid(p, head_n=head_n))
    found = _filtered(loaded, p, logits, ious, sizes)
    result = Pieces(found, loaded.model_id, loaded.device, time.perf_counter() - t0, False)
    if path is not None:
        write_cache(path, result)
    return result
