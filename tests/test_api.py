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


def test_render_rejects_degenerate_geometry():
    for key, value in (("r_outer", -10), ("r_inner", 0), ("r_inner", 500), ("width_ratio", 0),
                       ("cx", 900), ("cy", -20)):
        model, pts = _points_model()
        pts["params"][key] = value
        r = client.post("/api/render", json={"model": model})
        assert r.status_code == 422, (key, value)
        assert r.json()["detail"] == "The model data is invalid."
    model = _post("S46-15.png", _png()).json()["model"]
    tail = next(p for p in model["parts"] if p["role"] == "tail_stripe")
    tail["params"]["top_y"] = tail["params"]["bottom_y"]
    assert client.post("/api/render", json={"model": model}).status_code == 422


def test_analyze_sends_head_fit_as_information():
    body = _post("S46-15.png", _png(), "image/png").json()
    fit = body["head_fit"]
    assert set(fit) == {"agreement", "silhouette"}
    assert 0 < fit["agreement"] <= fit["silhouette"] <= 1


def _picker_model() -> dict:
    """Tails, the drawn library head A1-15 and, waiting on the head card, the rule-based
    head and the second candidate A6-15 (both in the real library)."""
    from rosette.model import ColorGroup, Part, RosetteModel
    tail = Part("t0", "tail_stripe", {"tail_index": 0, "stripe_index": 0, "tails_count": 1, "top_y": 250,
                                      "bottom_y": 800, "top_width": 80, "bottom_width": 120, "spread_deg": 0,
                                      "stripes": "1"}, 1, 0, 1.0)

    def tmpl(pid, tid, rank):
        return Part(pid, "template", {"template_id": tid, "cx": 300, "cy": 200, "r": 150, "rotation_deg": 0,
                                      "score": 0.7, "rank": rank, "group_1": 1, "group_2": 2, "group_3": 2}, 1, 10, 0.7)

    groups = [ColorGroup(1, (200, 30, 30), False, 0.1, 0.5, 1, 0.9), ColorGroup(2, (30, 30, 200), False, 0.1, 0.3, 2, 0.9)]
    head = Part("head", "head", {"cx": 300, "cy": 200, "r": 150, "pinked": 1}, 2, 10, 0.4)
    return RosetteModel([tail, tmpl("template", "A1-15", 1)], groups,
                        alternatives=[head, tmpl("template_2", "A6-15", 2)]).to_json()


def test_render_draws_the_head_picked_on_the_head_card():
    model = _picker_model()
    plain = client.post("/api/render", json={"model": model}).json()
    assert "model" not in plain
    r = client.post("/api/render", json={"model": model, "head": "A6-15"})
    assert r.status_code == 200
    body = r.json()
    drawn = [p for p in body["model"]["parts"] if p["role"] != "tail_stripe"]
    assert [p["params"]["template_id"] for p in drawn] == ["A6-15"]
    assert sorted(p["id"] for p in body["model"]["alternatives"]) == ["head", "template"]
    assert body["issues"] == [] and body["svg"] != plain["svg"]
    rules = client.post("/api/render", json={"model": body["model"], "head": "rules"}).json()
    assert [p["id"] for p in rules["model"]["parts"]] == ["t0", "head"]
    assert rules["issues"] == []
    assert client.post("/api/render", json={"model": model, "head": "S99-15"}).status_code == 422


def test_render_takes_a_library_head_outside_the_top_3():
    model = _picker_model()
    model["library_heads"] = [{"id": "library_S36-15", "role": "template", "params": {
        "template_id": "S36-15", "cx": 300, "cy": 200, "r": 150, "rotation_deg": 0, "score": 0.4, "rank": 7,
        "description": "6 points, center", "group_1": 1, "group_2": 2}, "color_group": 1, "z": 10,
        "confidence": 0.5}]
    model["warnings"] = ["general"]
    model["head_states"] = {"A1-15": {"warnings": [], "fit": None},
                            "S36-15": {"warnings": ["far: S36-15"], "fit": {"agreement": 0.5, "silhouette": 0.7}}}
    model["parts"][1]["params"]["cx"] = 305
    body = client.post("/api/render", json={"model": model, "head": "S36-15"}).json()
    drawn = [p for p in body["model"]["parts"] if p["role"] == "template"]
    assert [p["params"]["template_id"] for p in drawn] == ["S36-15"]
    assert drawn[0]["params"]["cx"] == 305 and drawn[0]["confidence"] == 0.5
    assert body["model"]["warnings"] == ["general", "far: S36-15"]
    assert body["model"]["library_heads"] == [] and body["issues"] == [] and body["flagged"]
