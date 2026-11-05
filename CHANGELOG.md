# Changelog

## [Unreleased]

### Added
- Photo to configurator SVG: upload a rosette photo and get an SVG that follows the shop contract (fixed layer order, Color 1-4 categories, pattern-capable Color 2), validated before download.
- Workbench UI: detected model, live solid and pattern previews, edit controls for every part and color, and a flag on results the tool is unsure about.
- Guided UI: the same controls one step at a time.
- `run.bat` to start the local server on 127.0.0.1:8765.
- `tools/score.py` scoreboard against the designer SVGs (23 of 34 reference rosettes reach the 0.9 agreement bar; the 80 % bar is not met, see docs/pipeline.md).
- Sample outputs `output/s35-19.svg` and `output/s46-15.svg`, README and pipeline documentation.

### Changed
- Bow points are drawn as ribbon-loop blocks, from `inner_ratio` x the width at the inner end to the full width at the outer end. Models saved before this with bow points and no `inner_ratio` now draw as constant-width blocks instead of pinched bow-ties.

## [0.1.0.0] - 2026-11-05

### Added
- Project bootstrap: Python venv with pinned dependencies, pytest configuration.
- Reference data: 79 designer SVGs, 76 product photos, canonical structure list, shop coloring script and master rose.
