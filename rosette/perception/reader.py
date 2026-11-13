"""ML-4 piece reader: the head pieces (SAM 2.1, ML-2) are numbered on the head crop and a local
vision-language model (Qwen3.5-4B GGUF, llama.cpp's llama-server on 127.0.0.1) answers closed,
schema-constrained questions: per mark its kind, texture and repeats, per touching pair which lies on top.
The answers are evidence for later steps, never a decision here; they are cached per photo.
Never imports torch."""
from __future__ import annotations

import base64
import io
import json
import subprocess
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

from rosette.perception.pieces import Piece, _interior_point, default_key
from rosette.perception.settings import Settings

PROMPT_VERSION = "ml4-1"
DEFAULT_READER_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "reader"
INSTALL_LINE = ("ML-4 reader: install llama.cpp release b12342 (Windows CPU x64) into models/llama-cpp/ and "
                "Qwen3.5-4B-Q8_0.gguf (or Qwen3.5-4B-Q4_K_M.gguf) with mmproj-Qwen3.5-4B-Q8_0.gguf from "
                "ggml-org/Qwen3.5-4B-GGUF into models/Qwen__Qwen3.5-4B-GGUF/ (README, 'ML-4 reader (optional)')")

KINDS = ("pleated_disc", "pleated_ring", "star", "petal", "loop", "block", "chevron", "ring", "badge", "rose", "other")
TEXTURES = ("plain", "glitter", "patterned")
FLOOR_SHARE = 0.004  # a piece must cover this share of the head area to be asked about
MAX_MARKS = 12  # marks per image
MAX_PIECES = 24  # two images: the time budget
MAX_PAIRS = 12
MAX_REPEATS = 16
PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
           (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40), (128, 0, 0), (0, 0, 128)]
HEALTH_WAIT_S = 240

Q1 = """This is the round head of a fabric rosette (award ribbon). Numbered coloured marks sit on pieces of ribbon found by a segmenter; each mark's outline is drawn in the same colour as its number.
For EVERY mark number from {first} to {n} answer: kind (pleated_disc = the big pleated circle at the back; pleated_ring = a pleated band; star = a layer with straight pointed arms; petal = one rounded petal; loop = one folded ribbon loop; block = one rectangular piece; chevron = nested V stripes at a star tip; ring = a plain flat ring; badge = the round printed button; rose = the ribbon rose; other), texture (plain, glitter, patterned), and repeats = how many identical pieces like this one are arranged around the centre to form a complete ring of repeats, counting this one (1 if it is a single central piece or a full disc or ring). Count the repeats by going once around the whole circle.
Do not skip any mark. Answer as compact JSON on one line, no spaces or newlines."""
Q2 = """Same image. For each pair of marks listed, say which piece lies ON TOP of the other where they touch (the piece whose edge is visible over the other), or unclear.
Pairs: {pairs}
Answer as compact JSON on one line, no spaces or newlines."""


def _s1(marks: list[int]) -> dict:
    return {"type": "array", "minItems": len(marks), "maxItems": len(marks), "items": {
        "type": "object", "properties": {
            "mark": {"type": "integer", "minimum": min(marks), "maximum": max(marks)},
            "kind": {"type": "string", "enum": list(KINDS)},
            "texture": {"type": "string", "enum": list(TEXTURES)},
            "repeats": {"type": "integer", "minimum": 1, "maximum": MAX_REPEATS}},
        "required": ["mark", "kind", "texture", "repeats"]}}


def _s2(n_pairs: int, lo: int, hi: int) -> dict:
    return {"type": "array", "minItems": n_pairs, "maxItems": n_pairs, "items": {
        "type": "object", "properties": {
            "a": {"type": "integer", "minimum": lo, "maximum": hi}, "b": {"type": "integer", "minimum": lo, "maximum": hi},
            "on_top": {"type": "string", "enum": ["a", "b", "unclear"]}},
        "required": ["a", "b", "on_top"]}}


class MissingReader(Exception):
    """The llama-server executable or the model files are absent; the message is the install line."""


@dataclass
class MarkAnswer:
    mark: int
    kind: str
    texture: str
    repeats: int


@dataclass
class PairAnswer:
    a: int
    b: int
    on_top: str  # "a" | "b" | "unclear"


@dataclass
class Reading:
    marks: list[int]  # global mark ids 1..n
    piece_index: dict[int, int]  # mark -> index into the pieces list
    answers: dict[int, MarkAnswer]  # unanswered marks are absent
    pairs: list[PairAnswer]
    seconds: float
    cached: bool
    images: int
    note: str = ""  # status line: the weights used when it is the Q4 fallback


def choose_pieces(p, pieces: list[Piece]) -> list[int]:
    """Indices into `pieces` of the head pieces of at least 0.4 % of the head area, largest first
    (ties keep the list order), at most MAX_PIECES."""
    area = float(np.pi * p.head[2] ** 2)
    idx = [i for i, pc in enumerate(pieces) if pc.in_head and pc.area >= FLOOR_SHARE * area]
    idx.sort(key=lambda i: -pieces[i].area)
    return idx[:MAX_PIECES]


def marks_image(p, pieces: list[Piece], chosen: list[int], side: int = 640,
                first: int = 1) -> tuple[Image.Image, list[dict]]:
    """The head crop (1.08 radii around the centre) at side x side with each chosen piece's outline in its own
    colour and its number (first, first + 1, ...) in a filled circle at its interior point.
    meta: one dict per mark: mark, area_share, r_centroid (of the interior point, in head radii), score."""
    cx, cy, R = p.head
    m = 1.08 * R
    box = (int(cx - m), int(cy - m), int(cx + m), int(cy + m))
    im = Image.fromarray(p.rgb).crop(box).resize((side, side), Image.LANCZOS).convert("RGB")
    s = side / (2 * m)
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
    head_area = float(np.pi * R * R)
    meta = []
    for k, i in enumerate(chosen, first):
        pc = pieces[i]
        col = PALETTE[(k - 1) % len(PALETTE)]
        edge = pc.mask ^ ndi.binary_erosion(pc.mask, iterations=2)
        ys, xs = np.nonzero(edge)
        for x, y in zip(xs[::2], ys[::2]):
            draw.point(((x - box[0]) * s, (y - box[1]) * s), fill=col)
        gx, gy = _interior_point(pc)
        u, v = (gx - box[0]) * s, (gy - box[1]) * s
        draw.ellipse((u - 17, v - 17, u + 17, v + 17), fill=col, outline=(255, 255, 255), width=2)
        draw.text((u, v), str(k), fill=(255, 255, 255), font=font, anchor="mm")
        meta.append({"mark": k, "area_share": round(pc.area / head_area, 4),
                     "r_centroid": round(float(np.hypot(gx - cx, gy - cy) / R), 3), "score": round(float(pc.score), 3)})
    return im, meta


def touching_pairs(pieces: list[Piece], chosen: list[int], dilate: int = 3,
                   limit: int = MAX_PAIRS) -> list[tuple[int, int]]:
    """(i, j) with i < j, 1-based positions in `chosen`, of the pieces whose masks overlap after a `dilate`-pixel
    dilation of each; largest contact first, at most `limit`."""
    dil = [ndi.binary_dilation(pieces[i].mask, iterations=dilate) for i in chosen]
    out = []
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            ov = int((dil[i] & dil[j]).sum())
            if ov > 0:
                out.append((i + 1, j + 1, ov))
    out.sort(key=lambda t: -t[2])
    return [(a, b) for a, b, _ in out[:limit]]


class Resolved(NamedTuple):
    exe: Path
    model: Path
    mmproj: Path
    note: str


class ReaderServer:
    """llama-server on 127.0.0.1, started and stopped by the tool (a context manager)."""

    def __init__(self, settings: Settings, log_path: Path | None = None):
        self.settings = settings
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.note = ""

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.settings.reader_port}"

    def resolve(self) -> Resolved:
        root = Path(self.settings.models_dir)
        exe = root / "llama-cpp" / "llama-server.exe"
        gg = root / "Qwen__Qwen3.5-4B-GGUF"
        mmproj = gg / "mmproj-Qwen3.5-4B-Q8_0.gguf"
        q8, q4 = gg / "Qwen3.5-4B-Q8_0.gguf", gg / "Qwen3.5-4B-Q4_K_M.gguf"
        if not exe.is_file():
            raise MissingReader(f"llama-server.exe not found at {exe}. {INSTALL_LINE}")
        if not mmproj.is_file():
            raise MissingReader(f"{mmproj.name} not found in {gg}. {INSTALL_LINE}")
        if q8.is_file():
            return Resolved(exe, q8, mmproj, "")
        if q4.is_file():
            return Resolved(exe, q4, mmproj, "Qwen3.5-4B-Q4_K_M weights used: Q8_0 is absent")
        raise MissingReader(f"neither {q8.name} nor {q4.name} found in {gg}. {INSTALL_LINE}")

    def command(self) -> list[str]:
        r = self.resolve()
        s = self.settings
        return [str(r.exe), "-m", str(r.model), "--mmproj", str(r.mmproj), "--host", "127.0.0.1",
                "--port", str(s.reader_port), "-c", "4096", "-t", str(s.reader_threads), "-np", "1",
                "--no-mmproj-offload", "--jinja", "--reasoning-budget", "0", "--cache-reuse", "256",
                "--image-max-tokens", "768"]

    def _healthy(self) -> bool:
        try:
            return urllib.request.urlopen(self.url + "/health", timeout=5).status == 200
        except Exception:
            return False

    def start(self) -> "ReaderServer":
        cmd = self.command()
        self.note = self.resolve().note
        if self._healthy():
            raise RuntimeError(f"port {self.settings.reader_port} already answers; only one model process at a time")
        log = self.log_path or (DEFAULT_READER_CACHE / "server.err")
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "ab") as err:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err)
        deadline = time.time() + HEALTH_WAIT_S
        while time.time() < deadline:
            if self._healthy():
                return self
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.proc.returncode}; see {log}")
            time.sleep(2)
        self.stop()
        raise RuntimeError(f"llama-server was not healthy after {HEALTH_WAIT_S} s; see {log}")

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)

    def __enter__(self) -> "ReaderServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def _http_post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def _body(b64: str, text: str, schema: dict, max_tokens: int) -> dict:
    return {"model": "qwen", "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                {"type": "text", "text": text}]}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}},
            "chat_template_kwargs": {"enable_thinking": False}}


def _content(resp: dict) -> list:
    try:
        out = json.loads(resp["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError, ValueError):
        return []
    return out if isinstance(out, list) else []


def _int(v) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _parse_marks(items: list, lo: int, hi: int) -> dict[int, MarkAnswer]:
    out: dict[int, MarkAnswer] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        k, rep = _int(it.get("mark")), _int(it.get("repeats"))
        if k is None or not lo <= k <= hi or k in out or rep is None:
            continue
        if it.get("kind") not in KINDS or it.get("texture") not in TEXTURES:
            continue
        out[k] = MarkAnswer(k, it["kind"], it["texture"], max(1, min(MAX_REPEATS, rep)))
    return out


def _parse_pairs(items: list, asked: list[tuple[int, int]]) -> list[PairAnswer]:
    out: list[PairAnswer] = []
    seen: set[tuple[int, int]] = set()
    for it in items:
        if not isinstance(it, dict) or it.get("on_top") not in ("a", "b", "unclear"):
            continue
        a, b = _int(it.get("a")), _int(it.get("b"))
        if (a, b) in asked and (a, b) not in seen:
            seen.add((a, b))
            out.append(PairAnswer(a, b, it["on_top"]))
        elif (b, a) in asked and (b, a) not in seen:  # the model swapped the pair: flip its answer
            seen.add((b, a))
            out.append(PairAnswer(b, a, {"a": "b", "b": "a"}.get(it["on_top"], "unclear")))
    return out


def _to_json(r: Reading) -> dict:
    return {"marks": r.marks, "piece_index": {str(k): v for k, v in r.piece_index.items()},
            "answers": {str(k): asdict(v) for k, v in r.answers.items()}, "pairs": [asdict(x) for x in r.pairs],
            "seconds": r.seconds, "images": r.images, "note": r.note}


def _from_json(d: dict) -> Reading:
    return Reading(marks=list(d["marks"]), piece_index={int(k): v for k, v in d["piece_index"].items()},
                   answers={int(k): MarkAnswer(**v) for k, v in d["answers"].items()},
                   pairs=[PairAnswer(**x) for x in d["pairs"]], seconds=float(d["seconds"]), cached=True,
                   images=int(d["images"]), note=d.get("note", ""))


def read(p, pieces: list[Piece], settings: Settings, cache_dir: Path | None = DEFAULT_READER_CACHE,
         post: Callable[[str, dict], dict] | None = None) -> Reading:
    """Ask the model about the head pieces. Per image of at most 12 marks (global ids): one question on the marks
    (kind, texture, repeats) and, when pieces touch, one on the touching pairs (which is on top); pairs whose pieces
    sit in different images are not asked. A mark the model skipped is absent from `answers` (ids never shift).
    No head pieces: an empty Reading, no model. `post(url, body) -> dict` is the HTTP call (default: urllib to the
    local server, which is started and stopped here when `post` is None and the cache is cold: MissingReader when
    the executable or weights are absent). Cached at <cache_dir>/<pieces key>-<PROMPT_VERSION>.json."""
    chosen = choose_pieces(p, pieces)
    if not chosen:
        return Reading([], {}, {}, [], 0.0, False, 0)
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"{default_key(p, settings)}-{PROMPT_VERSION}.json"
        if path.exists():
            try:
                return _from_json(json.loads(path.read_text(encoding="utf-8")))
            except (ValueError, KeyError, TypeError):
                pass  # a damaged cache file is asked again
    if post is None:
        with ReaderServer(settings) as srv:
            reading = _ask(p, pieces, chosen, lambda body: _http_post(srv.url + "/v1/chat/completions", body))
            reading.note = srv.note
    else:
        reading = _ask(p, pieces, chosen, lambda body: post(f"http://127.0.0.1:{settings.reader_port}/v1/chat/completions", body))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_to_json(reading), indent=1), encoding="utf-8")
        tmp.replace(path)
    return reading


def _ask(p, pieces: list[Piece], chosen: list[int], send: Callable[[dict], dict]) -> Reading:
    t0 = time.perf_counter()
    answers: dict[int, MarkAnswer] = {}
    pairs: list[PairAnswer] = []
    images = 0
    for start in range(0, len(chosen), MAX_MARKS):
        chunk = chosen[start:start + MAX_MARKS]
        lo, hi = start + 1, start + len(chunk)
        im, _meta = marks_image(p, pieces, chunk, first=lo)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
        b64 = base64.b64encode(buf.getvalue()).decode()
        images += 1
        marks = list(range(lo, hi + 1))
        answers.update(_parse_marks(_content(send(_body(b64, Q1.format(first=lo, n=hi), _s1(marks), 700))), lo, hi))
        local = touching_pairs(pieces, chunk)
        asked = [(a + start, b + start) for a, b in local]
        if asked:
            text = Q2.format(pairs=", ".join(f"({a},{b})" for a, b in asked))
            pairs.extend(_parse_pairs(_content(send(_body(b64, text, _s2(len(asked), lo, hi), 600))), asked))
    return Reading(marks=list(range(1, len(chosen) + 1)), piece_index={k: i for k, i in enumerate(chosen, 1)},
                   answers=answers, pairs=pairs, seconds=time.perf_counter() - t0, cached=False, images=images)
