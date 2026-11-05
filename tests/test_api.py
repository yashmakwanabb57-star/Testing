import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import MAX_UPLOAD, app, download_name

PHOTOS = Path(__file__).resolve().parent.parent / "reference" / "photos"
UNSUPPORTED = "Unsupported file type. Use PNG, WEBP, JPG or JPEG."
client = TestClient(app)


def _png() -> bytes:
    return (PHOTOS / "S46-15.png").read_bytes()


def _webp() -> bytes:
    buf = io.BytesIO()
    Image.open(PHOTOS / "S46-15.png").convert("RGB").save(buf, "WEBP")
    return buf.getvalue()


def _post(name, data, ctype="application/octet-stream"):
    return client.post("/api/analyze", files={"file": (name, data, ctype)})


def test_analyze_png():
    r = _post("S46-15.png", _png(), "image/png")
    assert r.status_code == 200
    body = r.json()
    assert body["svg"].startswith("<svg")
    assert body["issues"] == []
    assert body["filename"] == "s46-15.svg"
    assert isinstance(body["flagged"], bool)
    assert body["model"]["parts"]


def test_analyze_webp():
    r = _post("S46-15.WEBP", _webp(), "image/webp")
    assert r.status_code == 200
    assert r.json()["svg"].startswith("<svg")


def test_reject_unsupported():
    for name in ("a.gif", "a.txt", "noext"):
        r = _post(name, b"GIF89a....")
        assert r.status_code == 422
        assert r.json()["detail"] == UNSUPPORTED


def test_valid_extension_unreadable():
    r = _post("a.png", b"not an image")
    assert r.status_code == 422
    assert r.json()["detail"] == "Could not read the image file."


def test_content_must_be_supported_format():
    buf = io.BytesIO()
    Image.new("RGB", (50, 50), "white").save(buf, "GIF")
    r = _post("a.png", buf.getvalue())
    assert r.status_code == 422
    assert r.json()["detail"] == UNSUPPORTED


def test_no_rosette_422():
    buf = io.BytesIO()
    Image.new("RGB", (400, 400), "white").save(buf, "PNG")
    r = _post("blank.png", buf.getvalue())
    assert r.status_code == 422
    assert r.json()["detail"] == "No rosette found in the image."


def test_too_large_413():
    r = _post("big.png", b"\0" * (MAX_UPLOAD + 1))
    assert r.status_code == 413


def test_render_roundtrip():
    first = _post("S46-15.png", _png()).json()
    model = first["model"]
    pts = next(p for p in model["parts"] if p["role"] == "points")
    pts["params"]["count"] = int(pts["params"]["count"]) + 2
    r = client.post("/api/render", json={"model": model})
    assert r.status_code == 200
    body = r.json()
    assert body["svg"] != first["svg"]
    assert body["issues"] == []
    assert isinstance(body["flagged"], bool)


def test_render_invalid_model_422():
    for payload in ({"model": {"parts": [{"bogus": 1}]}}, {"model": 5}, {}, {"model": {"groups": [{"id": 1}]}}, {"model": None}, []):
        r = client.post("/api/render", json=payload)
        assert r.status_code == 422, payload
    r = client.post("/api/render", content=b"{bad", headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_download_name():
    assert download_name("S46-15 kopi.WEBP") == "s46-15 kopi.svg"
    assert download_name("\u00d8.jpg") == "\u00f8.svg"


def test_index_served():
    assert client.get("/").status_code == 200
    assert client.get("/static/index.html").status_code == 200


def _points_model():
    model = _post("S46-15.png", _png()).json()["model"]
    return model, next(p for p in model["parts"] if p["role"] == "points")


def test_render_rejects_bad_counts_quickly():
    for value in (10**9, -1, 0, 201, 2.5, "x", None, True):
        model, pts = _points_model()
        pts["params"]["count"] = value
        r = client.post("/api/render", json={"model": model})
        assert r.status_code == 422, value


def test_render_rejects_non_finite_numbers():
    for token in ("NaN", "Infinity"):
        model, pts = _points_model()
        pts["params"]["cx"] = 0.123456
        raw = json.dumps({"model": model}).replace("0.123456", token)
        r = client.post("/api/render", content=raw, headers={"content-type": "application/json"})
        assert r.status_code == 422, token
