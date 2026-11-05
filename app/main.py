"""Local web API over the rosette pipeline. Binds to 127.0.0.1 only (see run.bat)."""
from __future__ import annotations

import io
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from rosette.model import RosetteModel
from rosette.pipeline import analyze, is_flagged, render
from rosette.vision.prepare import UNREADABLE, InputError

MAX_UPLOAD = 25 * 1024 * 1024
CHUNK = 1024 * 1024
UNSUPPORTED = "Unsupported file type. Use PNG, WEBP, JPG or JPEG."
INVALID_MODEL = "The model data is invalid."
EXTENSIONS = {".png", ".webp", ".jpg", ".jpeg"}
FORMATS = {"PNG", "WEBP", "JPEG"}
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Rosette SVG generator")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class RenderRequest(BaseModel):
    model: Any


def download_name(upload_name: str) -> str:
    """'S46-15 kopi.WEBP' -> 's46-15 kopi.svg' (lower-cased stem, .svg extension)."""
    name = upload_name.replace("\\", "/").rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0] if "." in name else name
    return (stem.strip() or "rosette").lower() + ".svg"


def _issues(issues) -> list[dict]:
    return [asdict(i) for i in issues]


def _read_limited(file: UploadFile) -> bytes:
    buf = bytearray()
    while chunk := file.file.read(CHUNK):
        buf += chunk
        if len(buf) > MAX_UPLOAD:
            raise HTTPException(413, "File is too large. The limit is 25 MB.")
    return bytes(buf)


def _check_image(name: str, data: bytes) -> None:
    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in EXTENSIONS:
        raise HTTPException(422, UNSUPPORTED)
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            im.load()
    except Exception:
        raise HTTPException(422, UNREADABLE) from None
    if fmt not in FORMATS:
        raise HTTPException(422, UNSUPPORTED)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.post("/api/analyze")
def api_analyze(file: UploadFile) -> dict:
    data = _read_limited(file)
    name = file.filename or ""
    _check_image(name, data)
    try:
        model = analyze(data)
    except InputError as exc:
        raise HTTPException(422, str(exc)) from None
    result = render(model)
    return {"model": model.to_json(), "svg": result.svg, "issues": _issues(result.issues),
            "flagged": is_flagged(model, result.issues), "filename": download_name(name)}


@app.post("/api/render")
def api_render(req: RenderRequest) -> dict:
    try:
        model = RosetteModel.from_json(req.model)
        result = render(model)
    except Exception:
        raise HTTPException(422, INVALID_MODEL) from None
    return {"svg": result.svg, "issues": _issues(result.issues),
            "flagged": is_flagged(model, result.issues)}
