"""Head template library: evaluation on the 76 photos (the library and the matcher live in
rosette/vision/templates.py, H7).

For each photo the pipeline's detection (rosette.pipeline.detect) gives the head circle,
the color labels and the assigned model; the template the matcher picks, leave-one-out
(the photo's design and its variants removed: design_key) and from the full library, is
placed on the photo head circle over our tails.

Head agreement is measured the way tools/score.py measures color_agreement, but only over
the reference head pieces: each piece's visible region (minus pieces painted after it,
under MIN_ELEMENT_SHARE of the rosette skipped) votes correct when the generated category
holding most of its pixels equals the reference category; area-weighted share correct.
The generated raster is aligned to the reference raster by matching head circles. 'pipe'
uses the pipeline's Color numbering (template categories reach a Color through the photo
group they are mapped to), 'geo' takes the best relabeling of generated categories
(geometry only), 'pix' is the per-pixel share under the best relabeling.

Usage:
  .venv/Scripts/python.exe -m tools.head_templates [--cache DIR] [--weights] [IDs...]
--weights evaluates the WEIGHT_SETS (LOO and full library, geo and pix means) instead of
the per-ID listing.
"""
from __future__ import annotations

import itertools
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # run as a script
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from rosette.build import build  # noqa: E402
from rosette.model import RosetteModel  # noqa: E402
from rosette.pipeline import detect, render, with_template  # noqa: E402
from rosette.roles import RENDER_H, category, classify_components, frame_for, render_categories  # noqa: E402
from rosette.vision.parts_common import Frame  # noqa: E402
from rosette.vision.templates import (HEAD_ROLES, choose, design_key, load_library, photo_grid,  # noqa: E402
                                      raster, relabel_map)
from rosette.writer import write_svg  # noqa: E402
from tools.rasterize import DEFAULT_PALETTE, rasterize  # noqa: E402
from tools.score import HEIGHT, MIN_ELEMENT_SHARE, _bbox, categories  # noqa: E402

REF = ROOT / "reference"
ALL_ROLES = HEAD_ROLES + ("tail_stripe",)
_COLORS = dict(DEFAULT_PALETTE)
# Weight sets (partition, iou, profile, lobes) compared by --weights.
WEIGHT_SETS = ((0.4, 0.3, 0.2, 0.1), (1.0, 0.0, 0.0, 0.0), (0.6, 0.2, 0.1, 0.1), (0.7, 0.1, 0.2, 0.0),
               (0.5, 0.0, 0.3, 0.2))


@dataclass
class Truth:
    """A reference file's head pieces for scoring (RENDER_H padded frame)."""
    sid: str
    head: tuple[float, float, float]
    full: np.ndarray  # categories 0..4 of the whole file
    pieces: list = field(default_factory=list)  # [(category, role, visible bool mask)]


def truth(sid: str, cache: Path | None = None) -> Truth:
    cp = cache / "truth" / f"{sid}.pkl" if cache else None
    if cp and cp.exists():
        return pickle.loads(cp.read_bytes())
    svg = (REF / "svg" / f"{sid}.svg").read_text(encoding="utf-8")
    f, items = classify_components(svg)
    shape = items[0][2].shape if items else (1, 1)
    later = np.zeros(shape, bool)
    pieces = []
    for el, role, m in reversed(items):  # document order is paint order
        pieces.append((category(el), role, m & ~later))
        later |= m
    pieces.reverse()
    t = Truth(sid, f.head, categories(render_categories(svg, RENDER_H, _COLORS)), pieces)
    if cp:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_bytes(pickle.dumps(t))
    return t


def photo_record(sid: str, cache: Path | None = None) -> dict:
    """What the pipeline sees of one photo (rosette.pipeline.detect), the rules SVG."""
    cp = cache / "photos" / f"{sid}.pkl" if cache else None
    if cp and cp.exists():
        rec = pickle.loads(cp.read_bytes())
        for k in ("alternatives", "alternative_groups"):  # records cached before these fields
            if not hasattr(rec["model"], k):
                setattr(rec["model"], k, [])
        return rec
    d = detect((REF / "photos" / f"{sid}.png").read_bytes())
    p = d.prepared
    fr = Frame.of(p)
    rec = dict(sid=sid, head=p.head, mask=np.packbits(p.mask), shape=p.mask.shape,
               labels=d.labels.astype(np.int16), model=d.model, merged=d.merged, groups=d.groups,
               frame=(fr.axis_x, fr.top, fr.scale), svg=write_svg(build(d.model)))
    if cp:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_bytes(pickle.dumps(rec))
    return rec


def record_grid(rec: dict) -> np.ndarray:
    mask = np.unpackbits(rec["mask"])[: rec["shape"][0] * rec["shape"][1]].reshape(rec["shape"]).astype(bool)
    return photo_grid(mask, rec["labels"], rec["head"], relabel_map(rec["model"], rec["groups"], rec["merged"]))


# ---------------------------------------------------------------- placement and scoring


def _render(svg: str) -> np.ndarray:
    return categories(rasterize(svg, RENDER_H, _COLORS))


def tails_only_svg(model: RosetteModel) -> str:
    m = RosetteModel(parts=[q for q in model.parts if q.role == "tail_stripe"], groups=model.groups,
                     rose=False, warnings=[])
    return write_svg(build(m))


def generated_head(rec: dict) -> tuple[float, float, float]:
    """The photo's head circle in generated-raster pixels (RENDER_H rows, 600x850 canvas)."""
    ax, top, sc = rec["frame"]
    k = RENDER_H / 850.0
    cx, cy, r = rec["head"]
    return (300 + (cx - ax) / sc) * k, (25 + (cy - top) / sc) * k, r / sc * k


def placed(rec: dict, t, mapping: dict[int, int]) -> np.ndarray:
    """Generated category raster (RENDER_H rows): our tails, the template head on top,
    template categories turned into pipeline Colors through the mapped photo groups."""
    model = rec["model"]
    cat_of = {g.id: g.category for g in model.groups}
    default = min((c for c in cat_of.values() if c), default=1)
    lut = np.zeros(5, np.uint8)
    for c in range(1, 5):
        lut[c] = cat_of.get(mapping.get(c), None) or default
    img = _render(tails_only_svg(model))
    hx, hy, hr = generated_head(rec)
    head = raster(t.pieces, img.shape, hx, hy, hr, lut=lut)
    out = img.copy()
    out[head > 0] = head[head > 0]
    return out


def aligned_head(tr: Truth, gen: np.ndarray, head: tuple[float, float, float]) -> np.ndarray:
    """Generated raster moved and scaled so its head circle lands on the reference head
    circle: judges the head's shape apart from the rosette's overall proportions."""
    hx, hy, hr = head
    tx, ty, trr = tr.head
    s = hr / trr  # generated px per reference px
    m = np.float32([[s, 0, hx - s * tx], [0, s, hy - s * ty]])
    return cv2.warpAffine(gen, m, (tr.full.shape[1], tr.full.shape[0]),
                          flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0)


def agreement(tr: Truth, gen_aligned: np.ndarray, roles=HEAD_ROLES) -> tuple[float, float, float]:
    """(pipe, geo, pix) head agreement (module doc)."""
    min_px = MIN_ELEMENT_SHARE * (tr.full > 0).sum()
    rows = []
    for cat, role, vis in tr.pieces:
        a = float(vis.sum())
        if role not in roles or a < min_px:
            continue
        rows.append((cat, a, np.bincount(gen_aligned[vis], minlength=5)))
    if not rows:
        return 0.0, 0.0, 0.0
    total = sum(a for _, a, _ in rows)
    pipe = sum(a for cat, a, cnt in rows if int(cnt.argmax()) == cat) / total
    geo = pix = 0.0
    for perm in itertools.permutations(range(1, 5)):
        lut = np.array([0, *perm])  # generated label g -> lut[g]
        right = px = 0.0
        for cat, a, cnt in rows:
            relab = np.bincount(lut, weights=cnt, minlength=5)
            right += a if int(relab.argmax()) == cat else 0.0
            px += float(relab[cat])
        geo, pix = max(geo, right / total), max(pix, px / total)
    return pipe, geo, pix


# ---------------------------------------------------------------- the pipeline path (H7 decision)


def head_iou(designer_svg: str, generated_svg: str, tails_top_y: float | None = None) -> float:
    """tools/score.py shape_compare head_iou (silhouette IoU above the designer's tail top,
    bounding-box alignment) without the per-layer classification."""
    colors = {k: "#000000" for k in _COLORS}
    dm = render_categories(designer_svg, HEIGHT, colors).min(axis=2) < 128
    gm = render_categories(generated_svg, HEIGHT, colors).min(axis=2) < 128
    box, gbox = _bbox(dm), _bbox(gm)
    d = dm[box]
    g = cv2.resize(gm[gbox].astype(np.uint8), (d.shape[1], d.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    top = frame_for(designer_svg, HEIGHT).tails_top_y if tails_top_y is None else tails_top_y
    split = int(np.clip(round(top) - box[0].start, 0, d.shape[0]))
    a, b = d[:split], g[:split]
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def decision_row(sid: str, cache: Path | None = None, scores: bool = False) -> dict:
    """The rules head, the LOO template and the full-library template, all through the
    pipeline's own with_template, build and writer: head IoU (score.py head_iou), head_fit
    (agreement, silhouette) on the photo, the overlap of template and rules heads, the pick
    and its score (None when no template passes the gates). scores: also color agreement,
    whole shape and layer IoU (tools/score.py compare and shape_compare)."""
    from types import SimpleNamespace

    from rosette.pipeline import head_fit, head_overlap
    from tools.score import compare, shape_compare
    rec = photo_record(sid, cache)
    mask = np.unpackbits(rec["mask"])[: rec["shape"][0] * rec["shape"][1]].reshape(rec["shape"]).astype(bool)
    fr = Frame(*rec["frame"])
    p = SimpleNamespace(mask=mask, head=rec["head"], axis_x=rec["frame"][0], scale=rec["frame"][2])
    designer = (REF / "svg" / f"{sid}.svg").read_text(encoding="utf-8")
    top = frame_for(designer, HEIGHT).tails_top_y

    def measure(model, svg):
        f = head_fit(p, rec["labels"], model, rec["merged"])
        out = dict(head=head_iou(designer, svg, top), fit=(f.agreement, f.silhouette) if f else (0.0, 0.0))
        if scores:
            c, sh = compare(designer, svg), shape_compare(designer, svg)
            out.update(color=c["color_agreement"], shape=sh["shape_iou"], layer=sh["layer_iou"] or 0.0)
        return out

    row = dict(id=sid, rules=measure(rec["model"], rec["svg"]))
    for mode, excl in (("loo", design_key(sid)), ("full", None)):
        m = with_template(rec["model"], mask, rec["labels"], rec["head"], fr, rec["groups"], rec["merged"],
                          exclude_key=excl, min_score=-1.0)
        t = next((q for q in m.parts if q.role == "template"), None)
        if t is None:
            row[mode] = dict(row["rules"], pick=None, score=None, overlap=None, issues=0)
            continue
        res = render(m)
        row[mode] = dict(measure(m, res.svg), pick=t.params["template_id"], score=float(t.params["score"]),
                         partition=float(t.params.get("partition", 0.0)),
                         overlap=head_overlap(rec["model"], m), issues=len(res.issues))
    return row


# ---------------------------------------------------------------- main


def photo_ids() -> list[str]:
    return sorted(p.stem for p in (REF / "photos").glob("*.png") if (REF / "svg" / f"{p.stem}.svg").exists())


def evaluate(ids: list[str], cache: Path | None, weight_sets=(None,), verbose=True) -> dict:
    """{weights: [row per ID]}: rules, LOO and full-library head agreement."""
    lib = load_library()
    out = {w: [] for w in weight_sets}
    for sid in ids:
        tr = truth(sid, cache)
        rec = photo_record(sid, cache)
        labels = record_grid(rec)
        head = generated_head(rec)
        cur = agreement(tr, aligned_head(tr, _render(rec["svg"]), head))
        for w in weight_sets:
            row = dict(id=sid, cur=cur)
            for mode, excl in (("loo", design_key(sid)), ("full", None)):
                t, f = choose(labels, lib, excl, w)
                row[mode] = agreement(tr, aligned_head(tr, placed(rec, t, f["mapping"]), head))
                row[mode + "_pick"], row[mode + "_score"] = t.sid, f["score"]
            out[w].append(row)
            if verbose:
                fmt = lambda v: "/".join(f"{x:.3f}" for x in v)  # noqa: E731
                print(f"{sid:9} {w} cur {fmt(row['cur'])}  loo {fmt(row['loo'])} [{row['loo_pick']} "
                      f"{row['loo_score']:.3f}]  full {fmt(row['full'])} [{row['full_pick']} "
                      f"{row['full_score']:.3f}]", flush=True)
    return out


def main(argv: list[str]) -> int:
    cache = None
    if "--cache" in argv:
        i = argv.index("--cache")
        cache = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    sets = (None,)
    if "--weights" in argv:
        argv = [a for a in argv if a != "--weights"]
        sets = WEIGHT_SETS
    ids = argv or photo_ids()
    res = evaluate(ids, cache, sets, verbose=sets == (None,))
    for w, rows in res.items():
        n = len(rows)
        means = {k: [np.mean([r[k][i] for r in rows]) for i in range(3)] for k in ("cur", "loo", "full")}
        print(f"weights {w}: n={n}  " + "  ".join(f"{k} pipe/geo/pix {m[0]:.3f}/{m[1]:.3f}/{m[2]:.3f}"
                                                  for k, m in means.items())
              + f"  full picks own design {sum(design_key(r['full_pick']) == design_key(r['id']) for r in rows)}/{n}"
              + f"  LOO geo better/worse than rules {sum(r['loo'][1] > r['cur'][1] + 1e-9 for r in rows)}/"
                f"{sum(r['loo'][1] < r['cur'][1] - 1e-9 for r in rows)}")
    if cache:
        (cache / "head_templates_rows.pkl").write_bytes(pickle.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
