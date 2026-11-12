"""ML-1 Task 4: confirmed downloads become local training rows (synthetic photo and model)."""
import copy
import csv

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from rosette import confirmed
from rosette.model import ColorGroup, Part, RosetteModel
from rosette.pipeline import PhotoContext
from rosette.vision.prepare import Prepared
from tools import train_layers as tl


def _ctx():
    h = w = 400
    c = 200.0
    yy, xx = np.mgrid[:h, :w]
    d, t = np.hypot(xx - c, yy - c), np.degrees(np.arctan2(yy - c, xx - c)) % 360
    labels = np.full((h, w), 0, np.int32)
    labels[(d >= 30) & (d <= 60)] = 1
    labels[(d >= 70) & (d <= 140) & (np.mod(t, 60) < 20)] = 2
    rgb = np.full((h, w, 3), 200, np.uint8)
    p = Prepared(rgb=rgb, mask=d <= 150, axis_x=c, head=(c, c, 150.0), tails_top_y=350, bottom_y=399, scale=1.0)
    cy = 25 + (c - 50)  # canvas y of the head center (mask top at row 50)
    parts = [Part("ring-0", "ring", dict(cx=300, cy=cy, r_inner=30, r_outer=60), 1, 30, 0.9),
             Part("points-0", "points", dict(cx=300, cy=cy, kind="diamond", count=6, r_inner=70, r_outer=140,
                                             rotation_deg=10, width_ratio=0.5), 2, 20, 0.9)]
    groups = [ColorGroup(i, (10 * i, 0, 0), False, 0.0, 0.1, i, 1.0) for i in (0, 1, 2)]
    model = RosetteModel(parts=parts, groups=groups)
    return PhotoContext(p, labels, groups, {}, copy.deepcopy(model), copy.deepcopy(model)), model


def test_downloaded_model_labels_the_detected_layers(tmp_path):
    ctx, model = _ctx()
    final = copy.deepcopy(model)
    final.parts = [q for q in final.parts if q.id != "ring-0"]  # the user deleted the ring
    final.parts[0].params["kind"] = "bow"  # and drew the points as loops
    recs = confirmed.confirmed_records(ctx, final)
    got = {r["part"]: (r["rules"], r["label"], r["count"]) for r in recs}
    assert got == {"ring-0": ("ring", "none", 0), "points-0": ("kite", "loop", 6)}
    path = tmp_path / "confirmed.csv"
    assert confirmed.append_confirmed(recs, path) == 2
    assert confirmed.append_confirmed(recs, path) == 2  # the same photo again replaces its rows
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert len(rows) == 2 and rows[0]["source"] == "confirmed"


def test_confirmed_rows_have_the_dataset_columns():
    ctx, model = _ctx()
    rec = confirmed.confirmed_records(ctx, model)[0]
    header = next(csv.reader(open(tl.DATASET, encoding="utf-8")))
    assert list(rec.keys()) == header


def test_library_heads_are_not_logged():
    ctx, model = _ctx()
    final = copy.deepcopy(model)
    final.parts.append(Part("template", "template", dict(cx=300, cy=175, r=150, template_id="x"), 1, 10, 0.9))
    assert confirmed.confirmed_records(ctx, final) == []


def test_api_confirm_without_the_photo_logs_nothing():
    client = TestClient(app)
    r = client.post("/api/confirm", json={"model": RosetteModel().to_json()})
    assert r.status_code == 200 and r.json() == {"logged": 0}
    assert client.post("/api/confirm", json={"model": {"parts": [{"bad": 1}]}}).status_code == 422
