# Pipeline

Photo in, configurator SVG out (`rosette/pipeline.py`):

1. `prepare` (rosette/vision/prepare.py) - decode, normalize to 1200 px, foreground mask, symmetry axis, head circle, head/tails split.
2. `pixel_groups` (rosette/vision/colors.py) - per-pixel color groups with pattern texture.
3. `detect_parts` (rosette/vision/parts.py) - head, rings, points, center, tail stripes, leftovers as `unknown`.
4. Model - groups no part uses are dropped (ids kept); rose on iff a center part exists.
5. `assign` (rosette/assign.py) - Color 1-4 per group from mined designer rules, confidence, warnings.
6. `build` -> `write_svg` -> `validate` - shapes, SVG string, contract issues (`render` returns `RenderResult(svg, issues)`).

A result is **flagged** (`pipeline.is_flagged`) when a color group has confidence below `FLAG_BELOW` (0.6), a part is `unknown` (drawn as a placeholder), the model has any warning, or validation reports an issue.

## Scoreboard

`.venv/Scripts/python.exe tools/score.py [IDs...] [--sheet DIR]` scores every ID in `reference/canonical.txt` that has a photo (34; `TEST` has none). Generated and designer SVGs are rasterized at 850 px with category colors (solid configuration, outlines hidden), cropped to their non-white bounding boxes, the generated crop resized onto the designer crop (nearest neighbor).

- **agree** (primary, `color_agreement`) - per part region: each designer color element in `Layer_1` / `Layer_2_1_2/3/4` (under 0.2 % of the rosette area skipped) is rendered alone in the same frame; the generated category with the most pixels in that region decides correct or wrong; area-weighted share of correct elements.
- **pixel** (secondary) - share of designer-mask pixels with the same category (geometry mismatches count against it).
- **success** - agree >= 0.9 and no validation issues. **silently wrong** - not success and not flagged.
- **points / tails** - detected / designer count (`tools/truth.truth_from_svg`).
- Bar test: `pytest -m slow` (`tests/test_score.py::test_bar`); slow tests are skipped by default.

## Baseline (Task 11, before tuning)

| ID | agree (region) | pixel | IoU | flagged | success | points | tails | points ok | tails ok |
|---|---|---|---|---|---|---|---|---|---|
| A10-15B | 0.852 | 0.771 | 0.87 | yes | no | 0/1 | 1/1 | no | yes |
| A13-15B | 0.219 | 0.173 | 0.92 | yes | no | 0/4 | 1/1 | no | yes |
| A14-17 | 0.312 | 0.404 | 0.96 | no | no | 6/6 | 1/1 | yes | yes |
| A15-17 | 0.082 | 0.119 | 0.96 | yes | no | 6/6 | 1/1 | yes | yes |
| A16-17 | 0.537 | 0.328 | 0.90 | yes | no | 0/8 | 1/1 | no | yes |
| A9-15B | 1.000 | 0.828 | 0.94 | no | yes | 19/1 | 1/1 | no | yes |
| S10-19 | 0.169 | 0.239 | 0.86 | yes | no | 0/4 | 1/1 | no | yes |
| S13-15 | 0.817 | 0.676 | 0.96 | no | no | 6/6 | 1/1 | yes | yes |
| S17-15 | 0.594 | 0.497 | 0.83 | yes | no | 0/8 | 1/1 | no | yes |
| S2-15B | 0.000 | 0.047 | 0.90 | no | no | 4/4 | 1/1 | yes | yes |
| S26-15 | 0.841 | 0.751 | 0.93 | yes | no | 6/6 | 1/1 | yes | yes |
| S28-15 | 0.886 | 0.755 | 0.98 | no | no | 4/4 | 1/1 | yes | yes |
| S29-15 | 0.152 | 0.263 | 0.88 | yes | no | 0/6 | 1/1 | no | yes |
| S30-15 | 0.870 | 0.693 | 0.84 | yes | no | 6/6 | 1/1 | yes | yes |
| S31-15 | 0.947 | 0.759 | 0.93 | no | yes | 6/6 | 1/1 | yes | yes |
| S33-15 | 0.818 | 0.736 | 0.92 | no | no | 6/6 | 1/1 | yes | yes |
| S35-19 | 0.933 | 0.811 | 0.87 | yes | yes | 6/6 | 1/1 | yes | yes |
| S37-15 | 1.000 | 0.692 | 0.91 | yes | yes | 8/8 | 1/1 | yes | yes |
| S40-15 | 0.441 | 0.454 | 0.86 | yes | no | 0/6 | 1/1 | no | yes |
| S41-15 | 0.701 | 0.590 | 0.91 | no | no | 6/6 | 1/1 | yes | yes |
| S43-15 | 0.915 | 0.785 | 0.95 | no | yes | 6/6 | 1/1 | yes | yes |
| S45-15 | 0.872 | 0.606 | 0.93 | yes | no | 7/8 | 1/1 | no | yes |
| S46-15 | 0.716 | 0.508 | 0.83 | no | no | 6/6 | 1/1 | yes | yes |
| S5-15 | 0.774 | 0.555 | 0.85 | yes | no | 4/4 | 1/1 | yes | yes |
| S53-19 | 0.535 | 0.398 | 0.77 | yes | no | 22/12 | 3/1 | no | no |
| S56-17B | 0.698 | 0.602 | 0.88 | no | no | 6/6 | 1/1 | yes | yes |
| S58-17 | 1.000 | 0.762 | 0.90 | yes | yes | 6/6 | 1/1 | yes | yes |
| S67-19 | 0.942 | 0.797 | 0.95 | no | yes | 6/6 | 1/1 | yes | yes |
| S68-19 | 0.711 | 0.556 | 0.88 | yes | no | 0/8 | 1/1 | no | yes |
| S69-19 | 0.968 | 0.781 | 0.90 | no | yes | 6/8 | 1/1 | no | yes |
| S70-19 | 0.967 | 0.790 | 0.96 | yes | yes | 6/6 | 1/1 | yes | yes |
| S8-15 | 0.618 | 0.592 | 0.93 | yes | no | 0/8 | 1/1 | no | yes |
| S80-22 | 0.601 | 0.547 | 0.93 | no | no | 6/6 | 1/1 | yes | yes |
| s34-15 | 0.905 | 0.746 | 0.94 | yes | yes | 6/6 | 1/1 | yes | yes |

**Summary:** success 10, flagged-not-success 15, silently-wrong 9, errors 0 - `PASS 10/34 (target >= 80%)`.

Runtime: 1.4-2.4 s per photo for analyze + render; the scoreboard over all 34 IDs (with per-element designer renders) takes a few minutes.
