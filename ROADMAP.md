# Roadmap

## R0: Repository Governance (Done)
- [x] Preserve legacy workspace
- [x] Route-based code classification
- [x] Git ignore policy for data/outputs/weights/images
- [x] Syntax compile check baseline

## R1: Route A Hardening
- [ ] Add smoke tests for route-A script chain (`00` -> `05` / `run_stage1.py`)
- [ ] Add deterministic config profile for quick CI subset run
- [ ] Normalize encoding issues in historical task docs

## R2: Route B/C/D Unification
- [ ] Extract shared utility module (I/O normalization, overlays, mask utilities)
- [ ] Align CLI argument naming across routes
- [ ] Add per-route README with input/output contracts

## R3: Reproducibility & CI
- [ ] Add lightweight GitHub Actions for `compileall` + lint on curated code
- [ ] Add optional dependency groups (`requirements-*.txt`) and lock files
- [ ] Publish minimal sample data protocol (without committing raw proprietary TIFF)

## R4: Research Delivery
- [ ] Add experiment registry template (run metadata, hyperparameters, artifacts)
- [ ] Standardize manual-review dataset export format
- [ ] Plan transition from classical candidates to trainable defect detector benchmark
