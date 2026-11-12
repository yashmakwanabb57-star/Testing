# ML-2: cloth pieces from SAM 2 (design)

Date: 2026-11-12. Status: approved by the user ("start ML-2 with SAM 2 now").

## Intent

The rule detector finds layers from color runs. It fails wherever two pieces of cloth share a
color (a red star on a red disc), where a piece's edge is lost in a texture, and wherever a new
construction appears that no rule has been mined for: 52 of the 76 photos still miss the shape
(`.superpowers/sdd/.../diag52-report.md`). A segmentation model returns every visible piece of
cloth as its own mask with its true edges, regardless of color. ML-2 adds that evidence
("pieces") to the pipeline as an optional, local, free perception step, and uses it first where
the detector is weakest: layers in the disc's color and the stacking order.

## Goals

1. Segment each rosette photo into visible pieces with SAM 2.1 (Meta, Apache 2.0), fully offline
   at run time, on CPU by default, with a usable GPU picked up automatically when one exists.
2. Measure the pieces against the reference layers on all 76 photos and show them (overlays), so
   the user sees what the model finds before any of it reaches the SVG.
3. Feed the pieces into the detector as evidence for (a) front layers in the base color and their
   nested inner layers (S70-19's three stars) and (b) the stacking vote, replacing color-run
   guesses with visible masks. No model output reaches the SVG silently: every piece-derived
   decision carries the usual flags and stays editable in the UI.

## Non-goals (this round)

- Replacing the detector's shape fitting with a learned model (that is the synthetic-data round,
  ML-3). Pieces are evidence for the existing fitters, not a new fitter.
- The vision-language model for naming (Qwen2.5-VL): off until the pieces pay off.
- Video, tails text, badge recognition.

## Constraints (binding)

- No paid service, no cloud API, no network call at run time. Weights are fetched once by a tool
  into a git-ignored `models/` folder and loaded with local files only.
- Python 3.14 in `.venv` only; every package pinned exactly in `requirements.txt`.
- Hardware on the user's machine: GeForce GT 730, 2 GB, driver 456.71. Current PyTorch CUDA builds
  cannot use it. The tool must run on CPU here and detect a usable CUDA device elsewhere.
  A setting forces CPU. The analysis must stay usable on CPU: the small model, one image
  embedding per photo reused across all prompts, prompts in batches.
- The existing scoreboards may not go down (new-image PASS, color OK, silently-wrong) when the
  piece evidence is on; pieces off must reproduce today's output exactly.
- Licenses: everything Apache 2.0 or BSD. Verified 2026-11-12 against the official sources:
  `facebook/sam2.1-hiera-tiny` and `facebook/sam2.1-hiera-small` model cards carry
  `license: apache-2.0` and ship Transformers weights (`model.safetensors`, `config.json`,
  `preprocessor_config.json`, `processor_config.json`); `transformers` 5.22.2 is Apache 2.0
  (requires Python >= 3.11); PyTorch is BSD-3; CPU wheels for cp314 win_amd64 exist for
  torch 2.15.0 and torchvision 0.29.1 (download.pytorch.org/whl/cpu and PyPI).

## Decisions (made here)

- **Runtime stack:** `torch` 2.15.0 (CPU wheel), `torchvision` 0.29.1, `transformers` 5.22.2 with
  `Sam2Model` and `Sam2Processor` (the Transformers port of SAM 2.1; no git install of Meta's
  repo, which recommends WSL on Windows). Default model `facebook/sam2.1-hiera-small`; a `tiny`
  option for speed. Model size and device come from `rosette/perception/settings.py`, read from
  environment variables `ROSETTE_DEVICE` (`auto` | `cpu` | `cuda`, default `auto`),
  `ROSETTE_SAM2` (`small` | `tiny` | `off`, default `small`), `ROSETTE_MODELS_DIR` (default
  `models/`).
- **Offline loading:** `tools/fetch_models.py` downloads the chosen checkpoints with
  `huggingface_hub.snapshot_download` into `models/<repo>`. The loader sets `HF_HUB_OFFLINE=1`
  in-process and loads with `local_files_only=True`. A missing model raises `MissingModel` with
  the exact fetch command; the pipeline then runs with pieces off and an information line.
- **Prompting:** automatic mask generation by a point grid: the image embedding is computed once
  (`image_embeddings` reused across calls), prompts go in batches of 64 points with
  `multimask_output=True`; a mask is kept when its predicted IoU >= 0.80 and its stability
  (IoU between the mask thresholded at logits 0 and at +-1) >= 0.90; duplicates removed by
  non-maximum suppression at mask IoU 0.70; masks smaller than 0.2 % of the rosette or larger
  than 95 % of it dropped. The grid covers the head disc densely (32 x 32 inside the head circle
  from `prepare`) and the tails coarsely (8 columns x 16 rows). Masks live in the `prepare` work
  frame (same shape as `labels`).
- **Cache:** pieces per photo in `.cache/pieces/<sha256 of the file bytes>.npz` (masks packed as
  bits, scores, model id); the cache folder is git-ignored. The comparison pages and the scorer
  never re-run the model for an unchanged photo.
- **Completion pass and the repeat bars (ruling, Task 3b fix round 1):** after the grid pass, one
  extra decoder pass prompts the missing steps of repeated head pieces and the borders just
  outside each head piece. Its candidates pass the 0.80 / 0.90 bars as above. A candidate of this
  pass only (never the grid pass) that fails those bars but passes `REPEAT_IOU_MIN` 0.70 and
  `REPEAT_STABILITY_MIN` 0.75 is "weak". Weak candidates are kept only as a set that shows the
  repetition by itself:
  - NMS-deduplicated against the kept pieces and each other
  - at least 3 distinct steps of a robust equal-step fit (n <= 2 m, outliers dropped)
  - area within 35 %, centroid radius within 0.15 R and Lab median colors within 15 of each other
  Membership in a group of high-bar pieces does not count. Every inlier of such a set is kept, or
  none. A mask with more than one connected part over 10 % of its area is dropped in both passes.
- **Piece record:** `Piece(mask: np.ndarray[bool], score: float, stability: float, area: int,
  bbox: (x0, y0, x1, y1), color_group: int | None, in_head: bool)`. `color_group` is the majority
  `labels` value inside the mask (set by the pipeline, not the segmenter).
- **Where pieces enter the detector:** `detect_parts(p, labels, groups, pieces=None)`. With
  pieces:
  - *Front layers in the base color* (`_front_in_base`): a piece of the base color that lies
    inside the head, does not touch the rim, and has n-fold symmetry (the existing turned
    self-correlation, SPLIT_FLOOR) is a front layer; its radii, count and rotation come from the
    piece's own outline (tips = the piece's radial maxima, valleys = minima), not from mined
    border bands. A further piece of another color nested inside it on the same axes is the
    inner layer (S70-19's inner white star). The H15 rule stays as the fallback with pieces off.
  - *Stacking* (`stack_votes`): with pieces, the vote for a pair of layers counts, inside the
    drawn overlap, the pixels that belong to each layer's piece (the visible mask), not the
    color labels; a front layer in the base color therefore votes with its own piece instead of
    being void.
  - Every part whose geometry came from a piece gets `source: "piece"` in its params and the
    same confidence flags as today (symmetry below the bar -> flagged).
- **Evaluation:** `tools/pieces.py` writes an overlay per photo (photo with piece outlines and
  ids) and a table: pieces found, time, and per reference layer the best piece IoU (reference
  layers rasterized by `tools/rasterize.py` per color class, as the scorer does), with the mean
  over layers and the share of layers covered at IoU >= 0.70; totals over all 76 and over the
  52 "not working" IDs. The references are AI-made and unfaithful in places; the overlays are
  the primary check, the table the trend.
- **UI:** `/api/analyze` returns `perception: {device, model, pieces, seconds, status}` where
  status is `on`, `off` (setting), `missing` (no weights; the fetch command) or `failed` (the
  error text); the workbench shows it as one line under the photo. No new controls.
- **Docs:** README install step for the weights; `docs/pipeline.md` ML-2 section; CLAUDE.md rule
  text: "no paid services or cloud APIs; free local models allowed; no runtime network calls".

## Acceptance

- `ROSETTE_SAM2=off` reproduces the current SVGs byte for byte on the 76 photos.
- With pieces on (small, CPU): per photo under 60 s on this machine for segmentation (cached
  afterwards); the whole analysis with a warm cache under the current timing test.
- Pieces table recorded in the docs; overlays for all 76 in the scratchpad and the 52 IDs
  inspected by the controller with the user's reference-outline comparison.
- S70-19 with pieces: three nested stars (white, red, white) on the same axes, blocks under the
  stars, in both the SVG and the outline comparison. New-image scoreboard not down; no new
  silently-wrong ID.

## Left to the builder

The grid density and batch size if the timing bound fails on CPU; the exact packing of the cache;
how the overlay draws ids. Ask by report, not by stopping, unless a constraint cannot be met.
