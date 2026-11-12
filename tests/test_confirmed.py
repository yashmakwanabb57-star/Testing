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
    recs = confirmed.confirmed_records(ctx, final, "S46-15B kopi.svg")
    assert {r["design"] for r in recs} == {"S46"}
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


def test_design_of_the_download_name():
    assert confirmed.design_of("s46-15.svg") == "S46"
    assert confirmed.design_of("C:/x/S11a-15 kopi.svg") == "S11"
    assert confirmed.design_of("rosette.svg") == ""
    ctx, model = _ctx()
    recs = confirmed.confirmed_records(ctx, model)
    assert recs[0]["design"] == recs[0]["photo"]  # unknown design: the photo id


def test_api_confirm_rejects_malformed_parts_without_a_crash(tmp_path, monkeypatch):
    from app.main import photos
    monkeypatch.setattr(confirmed, "CONFIRMED_PATH", tmp_path / "confirmed.csv")
    ctx, model = _ctx()
    bad = copy.deepcopy(model)
    twin = copy.deepcopy(bad.parts[1])
    twin.id, twin.color_group = "points-1", 1
    bad.parts[1].params["rotation_deg"] = "ten"  # not a number, and another layer to compare it with
    bad.parts.append(twin)
    bad.photo_token = photos.put(ctx)
    r = TestClient(app).post("/api/confirm", json={"model": bad.to_json(), "filename": "s1-15.svg"})
    assert r.status_code == 422


def test_held_out_models_never_see_confirmed_rows_or_the_design(tmp_path, monkeypatch):
    from rosette import layer_model as lm
    header = next(csv.reader(open(tl.DATASET, encoding="utf-8")))
    feats = [k for k in header if k not in lm.META]
    rows = []
    for photo, design, source in (("S1-15", "S1", "ref"), ("S2-15", "S2", "ref"), ("confirmed-ab", "S2", "confirmed"),
                                  ("S2-15B", "S2", "ref")):
        r = dict.fromkeys(header, "0")
        r.update(photo=photo, design=design, part="points-0", role="points", rules="loop", rules_count="6",
                 label="loop", count="6", iou="1", source=source)
        rows.append(r)
    data = tmp_path / "layers.csv"
    with open(data, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, lineterminator=chr(10))
        w.writeheader()
        w.writerows(rows)
    seen = []

    class Base:
        features = feats
        gate, count_gate = 0.6, 0.8

    def fake_train(fs, labels, counts, cols):
        seen.append(len(labels))
        return lm.LayerModel(kind=None, count=None, features=cols)

    monkeypatch.setattr(lm, "DATASET", data)
    monkeypatch.setattr(lm, "load", lambda path=None: Base())
    monkeypatch.setattr(lm, "train", fake_train)
    monkeypatch.setattr(lm, "_WITHOUT", {})
    assert lm.model_without("S2").train_hash == "without S2"
    assert seen == [1]  # S1-15 only: S2-15, S2-15B (the design) and the confirmed row left out


def test_api_confirm_writes_rows_for_a_known_photo(tmp_path, monkeypatch):
    from app.main import photos
    monkeypatch.setattr(confirmed, "CONFIRMED_PATH", tmp_path / "confirmed.csv")
    ctx, model = _ctx()
    model.photo_token = photos.put(ctx)
    r = TestClient(app).post("/api/confirm", json={"model": model.to_json(), "filename": "s1-15.svg"})
    assert r.status_code == 200 and r.json() == {"logged": 2}
    assert (tmp_path / "confirmed.csv").exists()
