# ML-2: cloth pieces from SAM 2 - implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Segment each rosette photo into visible cloth pieces with SAM 2.1 offline on CPU (GPU when usable), show and measure the pieces on all 76 photos, and feed them to the detector for front layers in the base color and the stacking vote.

**Architecture:** A new package `rosette/perception/` (settings, device, offline model loading, grid-prompt segmentation with a per-photo cache) that the pipeline calls once per photo and passes to `detect_parts(..., pieces=...)`. Pieces are evidence for the existing fitters; with `ROSETTE_SAM2=off` nothing changes.

**Tech Stack:** Python 3.14 in `.venv`; torch 2.15.0 (CPU wheel), torchvision 0.29.1, transformers 5.22.2 (`Sam2Model`, `Sam2Processor`), huggingface_hub (for the one-time fetch), numpy, shapely (existing).

**Spec:** `docs/superpowers/specs/2026-11-12-ml2-sam2-pieces-design.md`

## Global Constraints

- Python only via `.venv/Scripts/python.exe`; never the global interpreter or global pip. Every new package pinned exactly in `requirements.txt` (run `pip freeze` after install and pin every newly added distribution, not only the three named).
- No network call at run time: weights load with `local_files_only=True` and `HF_HUB_OFFLINE=1` set in-process before any transformers import; the only download is `tools/fetch_models.py`.
- `ROSETTE_SAM2=off` must reproduce today's output byte for byte (verified on the 76 photos).
- No ID-specific logic. Plain hyphens in user-facing strings and commit messages; no AI attribution in commits.
- Do not use git stash. Do not start a server. `git add` only your own files.
- Tests: fast tests must not need the weights (mark weight-dependent tests `slow` and skip them with a clear reason when `models/` lacks the checkpoint).

## Review Focus

1. The pipeline on a machine without the weights: analysis must still succeed, with `perception.status == "missing"` and the fetch command in the message (Task 2 tests `MissingModel`; Task 5 tests the endpoint field).
2. A photo whose rosette is tiny or whose head circle fit failed: the grid must still lie inside the image and inside the mask (Task 3 test with a 200 px photo).
3. CPU time: the small model with the 32 x 32 grid must stay under 60 s per photo here; if not, the builder lowers the grid to 24 x 24 and records the timing (Task 3 report).
4. A piece that spans head and tails (a tail stripe running under the head): `in_head` must be False and the piece must not be offered as a head layer (Task 4 test).
5. Pieces on, cache hit: the analysis time must not include model time (Task 3 test reads the cache twice and asserts the second call never touches the model).

---

### Task 1: settings, device and dependencies

**Files:**
- Create: `rosette/perception/__init__.py`, `rosette/perception/settings.py`, `rosette/perception/device.py`
- Modify: `requirements.txt`, `.gitignore` (add `models/`, `.cache/`)
- Test: `tests/test_perception_settings.py`

**Interfaces:**
- Produces: `settings.Settings(device: str, model: str, models_dir: Path)` and `settings.load(env: Mapping[str, str] | None = None) -> Settings` reading `ROSETTE_DEVICE` (`auto`|`cpu`|`cuda`, default `auto`), `ROSETTE_SAM2` (`small`|`tiny`|`off`, default `small`), `ROSETTE_MODELS_DIR` (default `<repo>/models`); invalid values raise `ValueError` naming the variable and the allowed values.
- Produces: `device.pick(setting: str) -> tuple[str, str]` returning `("cuda" | "cpu", reason)`: `cpu` -> cpu; `cuda` -> cuda if `torch.cuda.is_available()` else `ValueError`; `auto` -> cuda when available and the device's compute capability is supported by this torch build (`torch.cuda.get_device_capability()` in `torch.cuda.get_arch_list()`), else cpu with the reason ("no CUDA device", "driver too old", "GT 730 compute capability 3.5 not in this build").

- [ ] **Step 1: Install and pin.** In the venv: `pip install torch==2.15.0 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu` then `pip install transformers==5.22.2 huggingface_hub`. Run `pip freeze`, and add every new distribution with its exact version to `requirements.txt` (keep the file sorted as it is). Note the torch CPU index in a comment line at the top of `requirements.txt` (`# torch/torchvision: CPU wheels from https://download.pytorch.org/whl/cpu`). Verify: `.venv/Scripts/python.exe -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"` prints `2.15.0+cpu 5.22.2 False`.
- [ ] **Step 2: Write the failing tests** in `tests/test_perception_settings.py`:

```python
def test_defaults():
    s = load({})
    assert (s.device, s.model) == ("auto", "small") and s.models_dir.name == "models"

def test_invalid_value_names_the_variable():
    with pytest.raises(ValueError, match="ROSETTE_SAM2"):
        load({"ROSETTE_SAM2": "huge"})

def test_cpu_setting_is_cpu():
    assert device.pick("cpu") == ("cpu", "forced by setting")

def test_auto_without_cuda_is_cpu(monkeypatch):
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: False)
    assert device.pick("auto")[0] == "cpu"

def test_auto_with_unsupported_capability_is_cpu(monkeypatch):
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device.torch.cuda, "get_device_capability", lambda: (3, 5))
    monkeypatch.setattr(device.torch.cuda, "get_arch_list", lambda: ["sm_75", "sm_80"])
    d, reason = device.pick("auto")
    assert d == "cpu" and "3.5" in reason
```

- [ ] **Step 3: Run them to see them fail** (`ImportError`).
- [ ] **Step 4: Implement** `settings.py` and `device.py` as in Interfaces. `device.py` imports torch lazily inside `pick` (so the settings module never imports torch).
- [ ] **Step 5: Run the tests; all pass.** Then the whole fast suite still passes.
- [ ] **Step 6: Commit** `feat: ML-2 - perception settings (ROSETTE_DEVICE, ROSETTE_SAM2, ROSETTE_MODELS_DIR) and device pick with CPU fallback; torch 2.15.0 CPU, torchvision 0.29.1, transformers 5.22.2 pinned`.

### Task 2: one-time model fetch and offline loading

**Files:**
- Create: `tools/fetch_models.py`, `rosette/perception/sam2.py`
- Test: `tests/test_perception_sam2.py`

**Interfaces:**
- Consumes: `settings.Settings`.
- Produces: `sam2.REPOS = {"small": "facebook/sam2.1-hiera-small", "tiny": "facebook/sam2.1-hiera-tiny"}`; `sam2.local_dir(models_dir, size) -> Path` (= `models_dir / repo.replace("/", "__")`); `sam2.MissingModel(Exception)` whose message contains `tools/fetch_models.py --model <size>`; `sam2.load(settings) -> Loaded(model, processor, device, model_id)` that sets `os.environ["HF_HUB_OFFLINE"] = "1"` before importing transformers, loads `Sam2Model.from_pretrained(local_dir, local_files_only=True)` and `Sam2Processor.from_pretrained(local_dir, local_files_only=True)`, moves the model to the device from `device.pick`, `eval()`; cached per process (`functools.lru_cache` on (models_dir, size, device)).
- `tools/fetch_models.py --model small|tiny|all [--dir models]` downloads with `huggingface_hub.snapshot_download(repo, local_dir=..., allow_patterns=["*.json", "*.safetensors", "README.md"])` and prints the folder and size; the only file in the repo allowed to use the network.

- [ ] **Step 1: Failing tests:**

```python
def test_missing_model_names_the_fetch_command(tmp_path):
    with pytest.raises(MissingModel, match=r"tools/fetch_models.py --model small"):
        load(Settings(device="cpu", model="small", models_dir=tmp_path))

@pytest.mark.slow
def test_loads_small_on_cpu():
    s = Settings(device="cpu", model="small", models_dir=MODELS)
    if not local_dir(MODELS, "small").joinpath("model.safetensors").exists():
        pytest.skip("weights not fetched: run tools/fetch_models.py --model small")
    ld = load(s)
    assert ld.device == "cpu" and ld.model_id == "facebook/sam2.1-hiera-small"
```

- [ ] **Step 2: Run; fail on import.**
- [ ] **Step 3: Implement** `sam2.py` and `tools/fetch_models.py`. Run the fetch for `small` and `tiny` into `models/` (about 200 MB each; report sizes). Confirm `models/` is ignored by git.
- [ ] **Step 4: Run the tests** including `-m slow -k loads_small`; pass. Then verify offline: with the network blocked (`HF_HUB_OFFLINE=1` is set by the loader; additionally run with `HTTP_PROXY=http://127.0.0.1:9` to prove no request is made) `load` still succeeds.
- [ ] **Step 5: Commit** `feat: ML-2 - SAM 2.1 weights fetched once by tools/fetch_models.py into models/ and loaded offline (local files only); MissingModel names the fetch command`.

### Task 3: pieces by grid prompting, with a cache

**Files:**
- Create: `rosette/perception/pieces.py`
- Test: `tests/test_perception_pieces.py`

**Interfaces:**
- Consumes: `sam2.load`, `rosette.vision.prepare.Prepared` (`rgb`, `mask`, `head=(cx, cy, r)`, `tails_top_y`).
- Produces: `Piece` dataclass (`mask: np.ndarray[bool]` in the `Prepared` frame, `score: float`, `stability: float`, `area: int`, `bbox: tuple[int, int, int, int]`, `in_head: bool`, `color_group: int | None = None`); `segment(p: Prepared, settings: Settings, cache_dir: Path | None = ".cache/pieces", key: str | None = None) -> Pieces(pieces: list[Piece], model_id: str, device: str, seconds: float, cached: bool)`; `grid(p: Prepared, head_n: int = 32, tail_cols: int = 8, tail_rows: int = 16) -> np.ndarray[(N, 2)]` of (x, y) prompt points inside `p.mask`: `head_n x head_n` over the head circle's bounding square keeping points inside the circle and the mask, plus the tail grid over the rows below `tails_top_y`; `nms(masks, scores, iou=0.70) -> indices`; `stability(logits, delta=1.0) -> float` = IoU of `logits > delta` and `logits > -delta`.
- Constants: `PRED_IOU_MIN = 0.80`, `STABILITY_MIN = 0.90`, `NMS_IOU = 0.70`, `MIN_AREA_FRAC = 0.002`, `MAX_AREA_FRAC = 0.95` (of `p.mask.sum()`), `POINTS_PER_BATCH = 64`.
- Cache: `key` defaults to the sha256 of `p.rgb.tobytes()`; file `<cache_dir>/<key>.npz` with `masks` packed by `np.packbits` over the flattened (n, h, w), `shape`, `scores`, `stability`, `model_id`; a cache hit never calls `sam2.load`.

- [ ] **Step 1: Failing tests** (no weights needed except the slow one):

```python
def test_grid_points_lie_inside_mask_and_circle(small_photo):  # 200 px synthetic Prepared
    pts = grid(small_photo)
    assert len(pts) > 50 and all(small_photo.mask[y, x] for x, y in pts)

def test_nms_drops_near_duplicates():
    a = np.zeros((20, 20), bool); a[2:12, 2:12] = True
    b = a.copy(); b[2, 2] = False
    c = np.zeros((20, 20), bool); c[14:19, 14:19] = True
    assert nms(np.stack([a, b, c]), np.array([0.9, 0.8, 0.7])) == [0, 2]

def test_stability_of_a_sharp_mask_is_one():
    logits = np.where(np.arange(100).reshape(10, 10) < 50, 5.0, -5.0)
    assert stability(logits) == 1.0

def test_cache_hit_never_loads_the_model(tmp_path, small_photo, monkeypatch):
    fake = Pieces([Piece(mask=small_photo.mask.copy(), score=0.9, stability=1.0, area=int(small_photo.mask.sum()),
                         bbox=(0, 0, 10, 10), in_head=True)], "fake", "cpu", 0.0, False)
    write_cache(tmp_path / "k.npz", fake)
    monkeypatch.setattr(pieces, "_run_model", lambda *a, **k: pytest.fail("model called"))
    out = segment(small_photo, Settings("cpu", "small", tmp_path), cache_dir=tmp_path, key="k")
    assert out.cached and len(out.pieces) == 1

@pytest.mark.slow
def test_s70_19_pieces_include_the_blocks_and_stars():
    # weights required (skip otherwise): at least 6 blue block pieces and >= 2 star-shaped white/red pieces in the head
    ...
```

- [ ] **Step 2: Run; fail.**
- [ ] **Step 3: Implement** `pieces.py`: `_run_model(loaded, rgb, points) -> (masks_logits, iou_scores)` computes the image embedding once (`model.get_image_embeddings` / `image_embeddings=` as the transformers docs show) and loops over point batches of `POINTS_PER_BATCH` with `multimask_output=True`, keeping the best of the three masks per point by predicted IoU; `post_process_masks` to the frame size; then the filters, `nms`, `in_head` (mask centroid above `tails_top_y` and >= 80 % of the mask's pixels above it), the cache. Time it on S70-19 on CPU (small); if over 60 s, lower `head_n` to 24 and record both timings in the report.
- [ ] **Step 4: Run tests** (fast, then `-m slow -k s70`); pass.
- [ ] **Step 5: Commit** `feat: ML-2 - pieces: SAM 2.1 grid prompting over the head and tails with one image embedding, predicted-IoU and stability filters, NMS, and a per-photo cache`.

### Task 4: evaluation tool and overlays on the 76 photos

**Files:**
- Create: `tools/pieces.py`
- Test: `tests/test_tools_pieces.py` (the table math on synthetic masks only)

**Interfaces:**
- Consumes: `pieces.segment`, `tools.score._role_masks`, `tools.score._designer` (reference raster), `tools.score._bbox`, `rosette.vision.prepare.prepare`.
- Produces: `python tools/pieces.py [--ids A,B] [--out DIR] [--model small|tiny]` writing `DIR/<id>.png` (photo with each piece's outline in a distinct color and its index at the centroid; head pieces solid, tail pieces dashed) and `DIR/pieces.csv` with columns `id, pieces, head_pieces, seconds, cached, layers, covered_070, mean_best_iou` where for each reference head element (`_role_masks` of the reference SVG, cropped to the designer bbox and resized to the photo's mask bbox crop) `best_iou` is the maximum IoU with any head piece cropped the same way; `covered_070` counts layers with best IoU >= 0.70. Prints a summary over all IDs and over the IDs in `--subset FILE` (one ID per line; the 52 not-working IDs are in the controller's scratchpad `not_working_ids.txt`, copy it to the plan workspace).

- [ ] **Step 1: Failing test** for `best_iou_table(ref_layers, piece_masks)` on two synthetic squares: one covered at 1.0, one at 0.0 -> `covered_070 == 1`, `mean_best_iou == 0.5`.
- [ ] **Step 2: Run; fail. Step 3: Implement. Step 4: Run; pass.**
- [ ] **Step 5: Run the tool on all 76** (`--out` the plan workspace folder `pieces/`), paste the summary lines and the 10 lowest `mean_best_iou` IDs into the report; look at S70-19, S29-15, S53-19, A13-15, S46-15 overlays and describe in the report what the pieces got right and wrong (which cloth pieces are one mask, which are split or merged).
- [ ] **Step 6: Commit** `feat: ML-2 - tools/pieces.py: piece overlays and the best-IoU table against the reference head layers`.

### Task 5: pieces into the detector and the API (after the H16 detector round has merged)

**Files:**
- Modify: `rosette/vision/parts.py` (`detect_parts(..., pieces: list[Piece] | None = None)`), `rosette/vision/parts_head.py` (`_front_in_base`, `stack_votes`), `rosette/pipeline.py` (`detect`: run `segment` when `settings.model != "off"`, assign `color_group` per piece by majority `labels`, pass pieces; `Detection.perception`), `app/main.py` (`perception` field in `/api/analyze`), `app/static` workbench line, `README.md`, `docs/pipeline.md`, `CLAUDE.md`
- Test: `tests/test_front_in_base.py` (piece path), `tests/test_stacking_pieces.py`, `tests/test_app.py` (perception field)

**Interfaces:**
- Consumes: `Pieces`, `Piece.color_group`.
- Produces: in `_front_in_base(..., pieces=None)`: with pieces, a candidate front layer is any head piece of the base color group that does not touch the rim (no pixel within `RIM_TOUCH` of the head circle) whose turned self-correlation at some n in 3..15 reaches `SPLIT_FLOOR`; count n, rotation from the piece's radial maxima, `r_outer` = median tip radius, `r_inner` = median valley radius, kind by the existing `describe` fitters on the piece polygon, `params["source"] = "piece"`; an inner layer = a head piece of another color whose mask lies inside the front piece's convex hull with the same n and axes (tolerance `BACK_TOL_DEG`), placed above it. `stack_votes(..., pieces=None)`: with pieces, for each layer the "own" mask is the union of pieces with the layer's color group whose centroid lies in the layer's drawn geometry; the vote counts `own_a` versus `own_b` pixels in the overlap (fallback to labels when a layer has no piece).
- `perception` in the API: `{"device", "model", "pieces", "seconds", "status": "on"|"off"|"missing"|"failed", "message"}`.

- [ ] **Step 1: Failing tests:** (a) `base_head()` synthetic image from `tests/test_front_in_base.py` with synthetic pieces (the red star mask, the white star mask, the inner white star mask, six block masks) -> three stars white < red < white on the same axes and all three above the blocks; (b) `stack_votes` with pieces: two overlapping layers where the labels say A but the pieces say B -> B wins; (c) `/api/analyze` with `ROSETTE_SAM2=off` returns `perception.status == "off"`; with a missing models dir, `"missing"` and the fetch command in `message`.
- [ ] **Step 2: Run; fail. Step 3: Implement. Step 4: Run; pass; full fast suite passes.**
- [ ] **Step 5: Verify** `ROSETTE_SAM2=off` reproduces the 76 SVGs byte for byte against a clean export of the base commit; with pieces on, run `tools/score.py` (new-image) and the controller's outline comparison for S70-19 and the changed IDs; gates in the spec.
- [ ] **Step 6: Docs** (README fetch step and settings; `docs/pipeline.md` ML-2 section with the pieces table, per-ID changes, constants; CLAUDE.md rule line) and **commit** `feat: ML-2 - pieces feed the detector (front layers in the base color and their inner layers from piece outlines; the stacking vote on visible pieces) and the API reports perception status`.
