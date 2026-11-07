"""Local web API over the rosette pipeline. Binds to 127.0.0.1 only (see run.bat)."""
from __future__ import annotations

import io
import logging
import secrets
import threading
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from rosette.model import RosetteModel
from rosette.pipeline import (RULES_HEAD, UncoloredHead, add_library_head, analyze_with_context, is_flagged, render,
                              use_head)
from rosette.vision.templates import library_index, template_by_id
from rosette.vision.prepare import UNREADABLE, InputError

MAX_UPLOAD = 25 * 1024 * 1024
CHUNK = 1024 * 1024
UNSUPPORTED = "Unsupported file type. Use PNG, WEBP, JPG or JPEG."
INVALID_MODEL = "The model data is invalid."
PHOTO_EXPIRED = ("The photo for this model is no longer in memory - upload the photo again to choose a "
                 "head from the library.")
UNKNOWN_HEAD = "There is no library head with this id - pick a head from the list."
UNCOLORED_HEAD = "This library head cannot be colored with the photo's colors - pick another head."
# Analyzed photos kept for library heads chosen later (PhotoCache): the last few uploads.
PHOTO_CACHE_SIZE = 8
EXTENSIONS = {".png", ".webp", ".jpg", ".jpeg"}
FORMATS = {"PNG", "WEBP", "JPEG"}
STATIC = Path(__file__).resolve().parent / "static"

log = logging.getLogger("rosette.api")
app = FastAPI(title="Rosette SVG generator")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class RenderRequest(BaseModel):
    model: Any
    # The head picked on the head card: "rules" or a library candidate's template_id
    # (rosette.pipeline.use_head); the response then carries the switched model.
    head: str | None = None


class PhotoCache:
    """The last `size` analyzed photos (pipeline.PhotoContext) by token; the least
    recently used one goes first. Thread-safe (FastAPI runs sync routes in threads)."""

    def __init__(self, size: int = PHOTO_CACHE_SIZE):
        self.size = size
        self._items: OrderedDict[str, object] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, ctx) -> str:
        token = secrets.token_hex(12)
        with self._lock:
            self._items[token] = ctx
            while len(self._items) > self.size:
                self._items.popitem(last=False)
        return token

    def get(self, token: str):
        with self._lock:
            ctx = self._items.get(token) if token else None
            if ctx is not None:
                self._items.move_to_end(token)
            return ctx


photos = PhotoCache()


def download_name(upload_name: str) -> str:
    """'S46-15 kopi.WEBP' -> 's46-15 kopi.svg' (lower-cased stem, .svg extension)."""
    name = upload_name.replace("\\", "/").rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0] if "." in name else name
    return (stem.strip() or "rosette").lower() + ".svg"


def _issues(issues) -> list[dict]:
    return [asdict(i) for i in issues]


def _read_limited(file: UploadFile) -> bytes:
    # Starlette has already spooled the upload to disk, so this caps memory, not disk.
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
        model, fit, ctx = analyze_with_context(data)
    except InputError as exc:
        raise HTTPException(422, str(exc)) from None
    model.photo_token = photos.put(ctx)
    result = render(model)
    return {"model": model.to_json(), "svg": result.svg, "issues": _issues(result.issues),
            "flagged": is_flagged(model, result.issues), "filename": download_name(name),
            "head_fit": asdict(fit) if fit else None}


def _holds(model: RosetteModel, tid: str) -> bool:
    return any(q.role == "template" and q.params.get("template_id") == tid for q in model.parts + model.alternatives)


@app.get("/api/library")
def api_library() -> dict:
    """Every library template by id with its description (the head card's chooser)."""
    return {"templates": [{"id": i, "description": d} for i, d in library_index()]}


@app.post("/api/render")
def api_render(req: RenderRequest) -> dict:
    try:
        model = RosetteModel.from_json(req.model)
    except Exception:
        log.exception("render failed for submitted model")
        raise HTTPException(422, INVALID_MODEL) from None
    if req.head is not None and req.head != RULES_HEAD and not _holds(model, req.head):
        if template_by_id(req.head) is None:  # a library head outside the candidates
            raise HTTPException(422, UNKNOWN_HEAD)
        ctx = photos.get(model.photo_token)
        if ctx is None:
            raise HTTPException(422, PHOTO_EXPIRED)
        try:
            model = add_library_head(model, req.head, ctx)
        except UncoloredHead:
            raise HTTPException(422, UNCOLORED_HEAD) from None
        except Exception:
            log.exception("library head failed for submitted model")
            raise HTTPException(422, INVALID_MODEL) from None
    try:
        if req.head is not None:
            model = use_head(model, req.head)
        result = render(model)
    except Exception:
        log.exception("render failed for submitted model")
        raise HTTPException(422, INVALID_MODEL) from None
    out = {"svg": result.svg, "issues": _issues(result.issues), "flagged": is_flagged(model, result.issues)}
    if req.head is not None:
        out["model"] = model.to_json()
    return out
