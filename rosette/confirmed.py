"""Learning from confirmed conversions (ML-1): a downloaded model's head layers as training rows.

When the user downloads an SVG (every flag confirmed) and the "Learn from my downloads"
setting is on (the default; app/static/app.js), the app posts the final model to
/api/confirm. For each head layer of the rule-based head as detected
(`PhotoContext.detected`, before the learned classifier) this writes its measured features
(`layer_classify.head_layer_rows`, the same as `rosette/data/layers.csv`) with the class and
count of the part with the same id in the downloaded model as the label (a layer the user
deleted is `none`), to `rosette/data/confirmed_layers.csv`. `tools/train_layers.py` trains
on them together with the reference dataset. Nothing leaves the computer.

Not logged: a model whose drawn head is a library head (no rule-based layers to label),
or a photo no longer in the server's memory. A photo confirmed again replaces its rows.
"""
from __future__ import annotations

import csv
import hashlib
import threading
from pathlib import Path

from rosette.layer_model import POINTS_CLASSES, rules_features
from rosette.vision.layer_classify import head_layer_rows, rules_class

CONFIRMED_PATH = Path(__file__).resolve().parent / "data" / "confirmed_layers.csv"
HEAD_ROLES = ("head", "ring", "points", "center", "unknown")
_RULES_KEYS = set(rules_features("none", "points", 0))


def photo_id(ctx) -> str:
    """A stable id of the analyzed photo (its prepared pixels)."""
    return "confirmed-" + hashlib.sha256(ctx.prepared.rgb.tobytes()).hexdigest()[:12]


def design_of(filename: str) -> str:
    """'s46-15.svg' -> 'S46'; '' when the name does not start like a shop id (a letter and
    digits)."""
    import re

    from rosette.vision.templates import design_key
    stem = filename.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0].strip()
    return design_key(stem) if re.match(r"^[A-Za-z]\d+", stem) else ""


def confirmed_records(ctx, final, filename: str = "") -> list[dict]:
    """Rows (layers.csv columns) of the photo's detected head layers labelled by the
    downloaded model `final`; [] when the drawn head is a library head or nothing was
    detected. `design`: the design key of the download name (`filename`, e.g.
    's46-15.svg' -> 'S46') when it reads as a shop id, else the photo id, so held-out
    evaluations can leave a design's confirmed rows out."""
    detected = getattr(ctx, "detected", None) or ctx.rules
    if any(q.role == "template" for q in final.parts):
        return []
    rows, _ = head_layer_rows(ctx.prepared, ctx.labels, detected, ctx.merged)
    by_id = {q.id: q for q in final.parts if q.role in HEAD_ROLES}
    pid = photo_id(ctx)
    design = design_of(filename) or pid
    out = []
    for r in rows:
        q = by_id.get(r["part"].id)
        label = rules_class(q, final.parts) if q is not None else "none"
        count = int(q.params.get("count", 0)) if q is not None and label in POINTS_CLASSES else 0
        rec = dict(photo=pid, design=design, part=r["part"].id, role=r["part"].role, rules=r["rules"],
                   rules_count=r["rules_count"], label=label, count=count, iou=1.0, source="confirmed")
        rec.update({k: round(float(v), 6) for k, v in r["features"].items() if k not in _RULES_KEYS})
        out.append(rec)
    return out


_WRITE = threading.Lock()


def append_confirmed(records: list[dict], path: Path | None = None) -> int:
    """Writes the records, replacing earlier rows of the same photo; returns their number.
    Writes are serialised (the server runs requests in threads)."""
    if not records:
        return 0
    with _WRITE:
        return _write(records, path or CONFIRMED_PATH)


def _write(records: list[dict], path: Path) -> int:
    old = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            old = [r for r in csv.DictReader(fh) if r["photo"] != records[0]["photo"]]
    keys = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows([r for r in old if set(keys) <= set(r)] + records)
    return len(records)
