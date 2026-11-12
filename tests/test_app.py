"""ML-2: /api/analyze reports the SAM 2 pieces' status (`perception`): off, missing (with the fetch
command), failed (with the error) or on; the analysis never fails because of pieces."""
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

import rosette.pipeline as pl
from app.main import app
from rosette.perception.pieces import Piece, Pieces

PHOTO = Path(__file__).resolve().parent.parent / "reference" / "photos" / "S46-15.png"
KEYS = {"device", "model", "pieces", "seconds", "cached", "status", "message"}
client = TestClient(app)


def _analyze():
    r = client.post("/api/analyze", files={"file": ("S46-15.png", PHOTO.read_bytes(), "image/png")})
    assert r.status_code == 200
    return r.json()


def test_pieces_off_is_reported():
    body = _analyze()  # tests/conftest.py: ROSETTE_SAM2=off
    assert set(body["perception"]) == KEYS
    assert body["perception"]["status"] == "off" and body["perception"]["pieces"] == 0
    assert body["svg"].startswith("<svg")


def test_missing_weights_are_reported_with_the_fetch_command(monkeypatch, tmp_path):
    monkeypatch.setenv("ROSETTE_SAM2", "small")
    monkeypatch.setenv("ROSETTE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(pl, "PIECES_CACHE_DIR", tmp_path / "cache")  # no cached pieces for the photo
    body = _analyze()
    assert body["perception"]["status"] == "missing"
    assert "tools/fetch_models.py --model small" in body["perception"]["message"]
    assert body["svg"].startswith("<svg") and body["issues"] == []


def test_a_failed_run_is_reported_and_the_rules_read_the_head(monkeypatch):
    monkeypatch.setenv("ROSETTE_SAM2", "small")

    def boom(*args, **kwargs):
        raise RuntimeError("out of memory")
    monkeypatch.setattr(pl, "segment", boom)
    body = _analyze()
    assert body["perception"]["status"] == "failed" and "out of memory" in body["perception"]["message"]
    off = pl.analyze_with_fit(PHOTO.read_bytes())[0]
    assert body["model"]["parts"] == off.to_json()["parts"]


def test_a_fault_in_the_piece_path_falls_back_to_the_rules(monkeypatch):
    import rosette.vision.parts_head as H
    monkeypatch.setenv("ROSETTE_SAM2", "small")

    def fake(p, settings, cache_dir=None):
        mask = np.zeros(p.mask.shape, bool)
        mask[:10, :10] = True
        return Pieces([Piece(mask=mask, score=0.9, stability=0.95, area=100, bbox=(0, 0, 9, 9), in_head=True)],
                      "m", "cpu", 1.0, True)

    def broken(*args, **kwargs):
        raise ValueError("bad piece")
    monkeypatch.setattr(pl, "segment", fake)
    monkeypatch.setattr(H, "_own_pieces", broken)
    monkeypatch.setattr(H, "PIECE_STACKING", True)
    body = _analyze()
    assert body["perception"]["status"] == "failed" and "bad piece" in body["perception"]["message"]
    monkeypatch.setenv("ROSETTE_SAM2", "off")
    assert body["model"]["parts"] == pl.analyze_with_fit(PHOTO.read_bytes())[0].to_json()["parts"]


def test_pieces_on_are_counted_and_colored(monkeypatch):
    monkeypatch.setenv("ROSETTE_SAM2", "small")
    seen = []

    def fake(p, settings, cache_dir=None):
        mask = np.zeros(p.mask.shape, bool)
        mask[:10, :10] = True
        pc = Piece(mask=mask, score=0.9, stability=0.95, area=100, bbox=(0, 0, 9, 9), in_head=False)
        seen.append(pc)
        return Pieces([pc], "facebook/sam2.1-hiera-small", "cpu", 1.234, True, (1.0, 0.2))
    monkeypatch.setattr(pl, "segment", fake)
    body = _analyze()
    assert body["perception"] == {"device": "cpu", "model": "facebook/sam2.1-hiera-small", "pieces": 1,
                                  "seconds": 1.23, "cached": True, "status": "on", "message": "cached pieces"}
    assert seen[0].color_group is None  # the photo's corner is background (label -1)
