"""The workbench page and its static assets are served, and nothing fetches off-machine."""
import re
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

STATIC = Path(__file__).resolve().parent.parent / "app" / "static"
ASSETS = ["index.html", "app.js", "logic.js", "guided.js", "simulator.js", "styles.css"]
client = TestClient(app)


def test_workbench_html():
    r = client.get("/")
    assert r.status_code == 200
    html = r.text
    for needle in ('id="file-input"', 'id="preview"', 'id="parts"', 'id="groups"',
                   'id="checks"', 'id="download"', 'src="/static/app.js"', 'href="/static/styles.css"',
                   'src="/static/logo.webp" alt="BR Roset"', 'src="/static/guided.js"',
                   'id="mode-workbench"', 'id="mode-guided"', 'id="guided"', 'id="g-next"', 'id="g-crumbs"'):
        assert needle in html, needle
    assert re.search(r'<button[^>]*id="download"[^>]*disabled', html)
    # Workbench is the default view; the guided view starts hidden.
    assert re.search(r'<main[^>]*id="guided"[^>]*hidden', html)
    assert re.search(r'id="mode-workbench"[^>]*aria-pressed="true"', html)


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


def test_svg_source_has_not_ready_banner():
    html = client.get("/").text
    assert re.search(r'<div[^>]*id="svg-source"[^>]*hidden', html)
    assert 'id="svg-source-note"' in html and 'id="svg-source-text"' in html
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "sourceNotice" in js


def test_convert_button_starts_disabled():
    html = client.get("/").text
    assert re.search(r'<button[^>]*id="convert"[^>]*disabled[^>]*>Convert to SVG</button>', html)
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    # choosing a file only previews it; the Convert action calls /api/analyze
    assert "chooseFile" in js and "loadFile" not in js
    assert "'Converting...'" in js
