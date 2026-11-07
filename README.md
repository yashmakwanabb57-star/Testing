# Rosette SVG generator

A local tool that turns a product photo of a ribbon rosette into a configurator SVG for brroset.dk. It uses classical computer vision only (no AI/ML models, no paid APIs, no network calls at run time). The SVG follows the shop contract in section 3 of [the design spec](docs/superpowers/specs/2026-11-05-rosette-svg-generator-design.md): fixed layer order, the Color 1-4 categories the shop script recolors, and a pattern-capable Color 2.

The pipeline, its tuning constants, the scoreboard and the known limitations are in [docs/pipeline.md](docs/pipeline.md).

## Requirements

- Python 3.14, in a virtualenv at `.venv`.
- Node.js is optional: it is only used by the JavaScript tests of the browser UI (they are skipped without it).

## Install

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

All dependencies are pinned in `requirements.txt`.

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

Heads the per-kind rules cannot build well are drawn from a **library of the reference heads** (H7): the closest reference head is placed on the photo's head circle when its outline fits the photo better than the detected head and its colors agree with the photo under the tail colors (or the detected head fails grossly and the library outline wins clearly; the library head is then flagged); otherwise it waits on the head card. The head card lists the rule-based head and the three best library templates (id, match, points count and a short description such as "6 loops, 6 kites, ring, center"), and the user can pick any of them, or choose any other library head by id ("Choose from the library", filtered by typing part of an id).

A result is **flagged** when the tool is unsure (a low-confidence color, any part read with low confidence - head, ring, points, center or tail stripes - a part drawn as a placeholder disc, a warning, or a validation issue). Confirm every flagged item before downloading the SVG.

## Output

Download the SVG from the UI (the download is enabled once the flags are confirmed) and keep it in `output/` (the tool does not write there by itself). Samples: `output/s35-19.svg`, `output/s46-15.svg`. Every file is checked by `rosette.validate.validate` against the contract before it is offered.

## Tests

```
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m pytest -q -m slow
```

The first command runs the fast suite: `pyproject.toml` already sets `-m "not slow"`, so the slow tests are skipped by default. The second runs only the tests marked `slow` (the scoreboard bar and a full pass over the reference photos). The slow bar test currently fails because the success bar is not met (see docs/pipeline.md).

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
