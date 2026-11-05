"""The workbench page and its static assets are served, and nothing fetches off-machine."""
import re
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

STATIC = Path(__file__).resolve().parent.parent / "app" / "static"
ASSETS = ["index.html", "app.js", "logic.js", "simulator.js", "styles.css"]
client = TestClient(app)


def test_workbench_html():
    r = client.get("/")
    assert r.status_code == 200
    html = r.text
    for needle in ('id="file-input"', 'id="preview"', 'id="parts"', 'id="groups"',
                   'id="checks"', 'id="download"', 'src="/static/app.js"', 'href="/static/styles.css"',
                   'src="/static/logo.webp" alt="BR Roset"'):
        assert needle in html, needle
    assert re.search(r'<button[^>]*id="download"[^>]*disabled', html)


def test_static_assets_served():
    for name in ASSETS:
        r = client.get(f"/static/{name}")
        assert r.status_code == 200, name
        assert r.text


def test_no_external_urls():
    allowed = {"http://www.w3.org/"}
    for name in ASSETS:
        text = (STATIC / name).read_text(encoding="utf-8")
        for url in re.findall(r"(?:https?:)?//[\w.-]+\.[a-z]{2,}[^\s'\"<>)]*", text):
            assert any(url.startswith(a) for a in allowed), f"{name}: {url}"
        assert "@import" not in text, name


def test_no_fancy_dashes_in_ui():
    for name in ASSETS:
        text = (STATIC / name).read_text(encoding="utf-8")
        assert chr(0x2014) not in text and chr(0x2013) not in text, name


def test_logo_served():
    r = client.get("/static/logo.webp")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/webp"
    assert r.content[:4] == b"RIFF" and r.content[8:12] == b"WEBP"
