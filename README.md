# Rosette SVG generator

A local tool that turns a product photo of a ribbon rosette into a configurator SVG for brroset.dk. It uses classical computer vision plus free local models: a small scikit-learn classifier for the kind and count of each head layer (trained on the reference photos) and, optionally, SAM 2.1 (Meta's segmentation model, Apache 2.0) to cut the photo into cloth pieces. No paid services or cloud APIs, and no network calls at run time. The SVG follows the shop contract in section 3 of [the design spec](docs/superpowers/specs/2026-11-05-rosette-svg-generator-design.md): fixed layer order, the Color 1-4 categories the shop script recolors, and a pattern-capable Color 2.

The pipeline, its tuning constants, the scoreboard and the known limitations are in [docs/pipeline.md](docs/pipeline.md).

## Requirements

- Python 3.14, in a virtualenv at `.venv`.
- Node.js is optional: it is only used by the JavaScript tests of the browser UI (they are skipped without it).

## Install

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

All dependencies are pinned in `requirements.txt` (torch and torchvision are the CPU wheels from the PyTorch index named in it).

### SAM 2 weights (once, optional)

The head reader also uses the cloth pieces SAM 2.1 finds in the photo (docs/pipeline.md, ML-2). Fetch the weights once into `models/` (git-ignored; the small model is about 180 MB). This is the only step that uses the network:

```
.venv\Scripts\python.exe tools/fetch_models.py --model small
```

Without the weights the tool runs as before (the rules alone) and the line under the photo says the model is missing, with this command.

Three settings, environment variables read at each analysis (set them before starting the server):

- `ROSETTE_SAM2`: `small` (default), `tiny` (fetch it with `--model tiny`) or `off` (no pieces: the output is exactly what the rules alone draw).
- `ROSETTE_DEVICE`: `auto` (default: a usable CUDA GPU, else the CPU), `cpu` or `cuda`. Current PyTorch cannot use the GeForce GT 730 on this machine, so it runs on the CPU.
- `ROSETTE_MODELS_DIR`: the weights folder (default `models/`).

A photo's pieces take about a minute on this CPU the first time; they are cached in `.cache/pieces/` (git-ignored), so the same photo is fast afterwards.

## Run

Double-click `run.bat`, or run:

```
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

then open http://127.0.0.1:8765. The server only listens on the local machine.

## The two UI modes

Choose or drop a photo (it is shown as a preview), press **Convert to SVG**, then work in either mode (switch at the top; both share the same state):

- **Workbench** - everything on one screen: the photo, the detected model (parts, color groups, Color 1-4 assignment, rose on/off), a live preview of the SVG in the shop's solid and pattern configurations, and edit controls for every part and color. Best for fixing a result.
- **Guided** - the same controls one step at a time, asking about each color and each part. Best for a quick check.

Heads the per-kind rules cannot build well are drawn from a **library of the reference heads** (H7): the closest reference head is placed on the photo's head circle when its outline fits the photo better than the detected head and its colors agree with the photo under the tail colors (or the detected head fails grossly and the library outline wins clearly; the library head is then flagged); otherwise the rule-based head is drawn. Each template is placed at its best pose on the photo (center offset, scale, turn). The head card shows the automatic head, its match and its colors; there is no head picker. A new image is converted automatically: scored with each photo's own design left out of the library (the honest estimate for a new design), the 34-photo scoreboard reaches PASS 23/34 with head IoU 0.844 (known design, full library: PASS 23/34, head 0.847).

A result is **flagged** when the tool is unsure (a low-confidence color, any part read with low confidence - head, ring, points, center or tail stripes - a part drawn as a placeholder disc, a warning, or a validation issue). Confirm every flagged item before downloading the SVG.

## Output

Download the SVG from the UI (the download is enabled once the flags are confirmed) and keep it in `output/` (the tool does not write there by itself). With **Learn from my downloads** on (the default, under the Download button), the head layers of each downloaded SVG are also kept on this computer in `rosette/data/confirmed_layers.csv` as training examples for the layer classifier; retrain it with `.venv\Scripts\python.exe tools/train_layers.py` (the reference dataset is rebuilt with `tools/build_layer_dataset.py`). Nothing is sent anywhere. Samples: `output/s35-19.svg`, `output/s46-15.svg`. Every file is checked by `rosette.validate.validate` against the contract before it is offered.

## Tests

```
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m pytest -q -m slow
```

The first command runs the fast suite: `pyproject.toml` already sets `-m "not slow"`, so the slow tests are skipped by default. The tests run with the SAM 2 pieces off (`tests/conftest.py`) unless a test turns them on. The second runs only the tests marked `slow` (the scoreboard bar and a full pass over the reference photos). The slow bar test currently fails because the success bar is not met (see docs/pipeline.md).

## Scoring

```
.venv\Scripts\python.exe tools/score.py [IDs...] [--sheet DIR] [--debug]
```

Runs the pipeline on every photo in `reference/photos` that has a designer SVG listed in `reference/canonical.txt` (34) and compares the result against the designer's file, region by region. It takes a few minutes. `--sheet DIR` saves photo | generated | designer images; `--debug` prints each scored element.

Current scoreboard: success 23 of 34 (the bar is 28), flagged-not-success 10, silently wrong 1 (S46-15), errors 0. Each row also reports a shape score (silhouette IoU, whole / head / tail, and a per-layer score); mean whole IoU 0.922 (head 0.851), 27 of 34 at or above 0.90. The UI always asks you to compare the head with the photo (a checklist item, not a flag): the head flag only catches gross head failures.

## How to test on the shop

After converting a photo and confirming the flags, upload the SVG to a test product and check:

1. Color 1, Color 2 and Color 3: change each one independently and confirm only the parts you expect change.
2. Color 2 in solid color mode: the ribbons take the chosen solid color.
3. Color 2 in pattern mode: the ribbons show the chosen pattern.
4. Color 3 in both modes: it must stay correct in solid and in pattern mode.
5. Color 4 (only when the design uses it): change it and confirm only its parts change, in solid and in pattern mode.
6. Look over the whole rosette for missing elements, duplicated elements, or parts with the wrong color.
7. If something is off, fix it in the UI (or note it for the pipeline) and re-test.
