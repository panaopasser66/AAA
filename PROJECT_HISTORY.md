# Project History

## Background
The workspace contained three parallel development tracks:
- `kossel_line_test/`: multi-frame classical pipeline and stage-3 manual review/line-removal flow
- `project/stage1` + `project/stage2`: line-mask labeling/training plus background-sampling & contrast-suppression experiments
- `project2/`: self-supervised inpainting pipeline with training/inference variants

Over time, code and docs diverged into multiple folders with duplicated intent and mixed outputs.

## Reorganization Actions (2026-05-12)

1. Recursively indexed all `.py/.md/.yaml/.json` files and archived the inventory.
2. Preserved original folders by moving them into `legacy_workspace/` (no deletion).
3. Rebuilt a GitHub-ready top-level structure.
4. Classified valuable code into four technical routes (A/B/C/D).
5. Copied route-relevant scripts/modules into curated `src/` and `scripts/` paths.
6. Centralized task documents into `docs/tasks/`.
7. Added root-level governance files: `README.md`, `ROADMAP.md`, `.gitignore`, `requirements.txt`.

## Current Principle
- Keep legacy evidence immutable in `legacy_workspace/`.
- Continue new development in curated route directories.
- Keep outputs, raw data, and model artifacts outside Git tracking.
