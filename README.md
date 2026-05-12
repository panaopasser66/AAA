# Kossel/SiC Defect Research Repository

This repository is a GitHub-ready reorganization of a previously fragmented Kossel/SiC X-ray defect project workspace.

## Scope

- Preserve all original workspaces without deletion (`legacy_workspace/`)
- Curate and classify valuable code by technical route
- Keep reproducible research code (`src/`, `scripts/`, `configs/`, `docs/`)
- Exclude heavy/raw/generated artifacts from Git

## Technical Routes

### Route A: Classical Multi-frame Registration + Fusion + Candidate Mining
- Code package: `src/route_a_multiframe_classical/`
- Entry scripts: `scripts/route_a_multiframe_classical/`
- Config: `configs/route_a_multiframe_config.yaml`
- Purpose: registration, fusion, line risk, blob candidates, manual review, line-removal baseline

### Route B: Line Label Preparation + U-Net Line Masking
- Entry scripts: `scripts/route_b_line_mask_unet/`
- Purpose: pseudo-label construction, patch regeneration, line-mask U-Net training/inference

### Route C: Local Background Sampling + Line-band Contrast Suppression
- Entry scripts: `scripts/route_c_bg_sampling_band_suppress/`
- Purpose: non-DL line attenuation with local texture/background replacement strategies

### Route D: Self-supervised Inpainting
- Entry scripts: `scripts/route_d_selfsupervised_inpainting/`
- Purpose: data preparation, inpainting U-Net training, inference variants

## Repository Layout

```text
.
├─ src/
├─ scripts/
├─ configs/
├─ docs/
│  ├─ tasks/
│  └─ inventory/
└─ legacy_workspace/
```

## Install

```bash
pip install -r requirements.txt
```

## Notes

- Original raw/generated experiment trees are preserved under `legacy_workspace/`.
- They are intentionally ignored by Git to keep the research repo lightweight.
- Full discovered file inventory is at `docs/inventory/file_inventory.txt`.
