# Changelog

## [Unreleased]

### Changed
- Bow points are drawn as ribbon-loop blocks, from `inner_ratio` x the width at the inner end to the full width at the outer end. Models saved before this with bow points and no `inner_ratio` now draw as constant-width blocks instead of pinched bow-ties.

## [0.1.0.0] - 2026-11-05

### Added
- Project bootstrap: Python venv with pinned dependencies, pytest configuration.
- Reference data: 79 designer SVGs, 76 product photos, canonical structure list, shop coloring script and master rose.
