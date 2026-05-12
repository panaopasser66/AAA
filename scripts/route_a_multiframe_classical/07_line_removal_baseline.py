"""Stage 3B — narrow-line masks + multi-method inpainting baseline.

Spec: task_stage3.md §7–11.

Outputs:
  outputs/line_removal/masks/line_mask_radius{r}.png/.npy
  outputs/line_removal/masks/defect_protect_mask.png/.npy
  outputs/line_removal/masks/line_removal_mask_coverage.csv
  outputs/line_removal/cleaned_images/<variant>.png/.npy
  outputs/line_removal/comparison_sheets/<variant>.png

Variant naming (per spec §11):
  cleaned_<input_name>_r<radius>_<method>_<param>_<protect|no_protect>
where input_name in {0000, fused_v1, fused_v2}.
"""

import _bootstrap  # noqa: F401

import argparse
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import cv2
import numpy as np
import pandas as pd

from src.groups import load_config
from src.io_utils import ensure_dir, load_tif_float32, robust_rescale_for_preview, save_png
from src.line_inpainting import (
    inpaint_local_median,
    inpaint_navier_stokes,
    inpaint_telea,
    residual,
)
from src.line_removal_masks import (
    aggregate_protect_candidates,
    coverage_fraction,
    defect_protect_mask,
    line_mask_coverage_table,
    narrow_line_mask,
)
from src.normalize import percentile_normalize


INPUT_NAME_MAP = {
    "0000": "0000",
    "combined_fused_delined": "fused_v1",
    "combined_fused_delined_v2": "fused_v2",
}


def _opt_npy(path: str):
    if os.path.isfile(path):
        return np.load(path).astype(np.float32)
    return None


def _load_input_image(out_dir: str, raw_dir: str, key: str, config: dict) -> Optional[np.ndarray]:
    """Load and (if needed) percentile-normalize the input image to [0,1]."""
    if key == "0000":
        ref_norm = _opt_npy(os.path.join(out_dir, "normalized", "0000.npy"))
        if ref_norm is not None:
            return ref_norm.astype(np.float32)
        # Fall back: load raw & normalize
        ref_raw_path = os.path.join(raw_dir, config.get("reference_image", "0000.tif"))
        if not os.path.isfile(ref_raw_path):
            return None
        ref_raw = load_tif_float32(ref_raw_path)
        norm_cfg = config.get("normalization", {}) or {}
        return percentile_normalize(
            ref_raw,
            float(norm_cfg.get("percentile_low", 0.5)),
            float(norm_cfg.get("percentile_high", 99.5)),
        )
    # fused images already in [0,1]
    return _opt_npy(os.path.join(out_dir, "fusion", f"{key}.npy"))


def _make_comparison_sheet(
    original: np.ndarray,
    line_mask: np.ndarray,
    cleaned: np.ndarray,
    residual_img: np.ndarray,
    zoom_boxes: List[List[int]],
    title: str,
) -> np.ndarray:
    """Build a comparison sheet: original, line_mask, cleaned, residual + zoom crops."""
    H, W = original.shape
    target_w = 480
    scale = target_w / W if W > target_w else 1.0
    new_w = int(round(W * scale))
    new_h = int(round(H * scale))

    def _resize_gray(img):
        if scale >= 1.0:
            return img
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    panels = [
        ("original", _resize_gray(original)),
        ("line_mask", _resize_gray(line_mask.astype(np.float32))),
        ("cleaned", _resize_gray(cleaned)),
        ("residual = original - cleaned", _resize_gray(residual_img)),
    ]
    panel_imgs = []
    for name, img in panels:
        prev = robust_rescale_for_preview(img)
        bgr = cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR)
        panel_imgs.append((name, bgr))

    # Zoom crops
    for box in zoom_boxes or []:
        x0, y0, x1, y1 = [int(v) for v in box]
        x0 = max(0, x0); y0 = max(0, y0); x1 = min(W, x1); y1 = min(H, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        zoom_orig = original[y0:y1, x0:x1]
        zoom_clean = cleaned[y0:y1, x0:x1]
        zoom_resid = residual_img[y0:y1, x0:x1]
        zh, zw = zoom_orig.shape
        if zw < 240:
            f = 240 / zw
            zh2 = int(round(zh * f)); zw2 = int(round(zw * f))
            zoom_orig = cv2.resize(zoom_orig, (zw2, zh2), interpolation=cv2.INTER_NEAREST)
            zoom_clean = cv2.resize(zoom_clean, (zw2, zh2), interpolation=cv2.INTER_NEAREST)
            zoom_resid = cv2.resize(zoom_resid, (zw2, zh2), interpolation=cv2.INTER_NEAREST)
        for tag, img in (
            (f"zoom[{x0},{y0}]:orig", zoom_orig),
            (f"zoom[{x0},{y0}]:cleaned", zoom_clean),
            (f"zoom[{x0},{y0}]:residual", zoom_resid),
        ):
            prev = robust_rescale_for_preview(img)
            bgr = cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR)
            panel_imgs.append((tag, bgr))

    # Tile in a 4-col grid; pad smaller tiles to row max height.
    cols = 4
    rows = (len(panel_imgs) + cols - 1) // cols
    pad = 6
    title_h = 18

    # Compute per-row max sizes
    row_h = [0] * rows
    col_w = [0] * cols
    for i, (_, img) in enumerate(panel_imgs):
        r, c = divmod(i, cols)
        row_h[r] = max(row_h[r], img.shape[0])
        col_w[c] = max(col_w[c], img.shape[1])
    grid_h = sum(row_h) + (rows + 1) * pad + rows * title_h + 24
    grid_w = sum(col_w) + (cols + 1) * pad
    sheet = np.full((grid_h, grid_w, 3), 16, dtype=np.uint8)

    cv2.putText(sheet, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)

    y_cursor = 24 + pad
    for r in range(rows):
        x_cursor = pad
        for c in range(cols):
            i = r * cols + c
            if i >= len(panel_imgs):
                continue
            name, img = panel_imgs[i]
            cv2.putText(sheet, name, (x_cursor, y_cursor + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            sheet[
                y_cursor + title_h: y_cursor + title_h + img.shape[0],
                x_cursor: x_cursor + img.shape[1],
                :,
            ] = img
            x_cursor += col_w[c] + pad
        y_cursor += row_h[r] + title_h + pad
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="raw")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    config = load_config(args.config)
    lr_cfg = config.get("line_removal", {}) or {}
    if not bool(lr_cfg.get("enabled", True)):
        print("[07] line_removal.enabled=false; skipping.")
        return

    input_keys: List[str] = list(lr_cfg.get(
        "input_images", ["0000", "combined_fused_delined", "combined_fused_delined_v2"],
    ))
    radii: List[int] = list(lr_cfg.get("mask_radii", [2, 4, 6]))
    too_aggr = float(lr_cfg.get("too_aggressive_for_inpainting_threshold", 0.35))
    protect_enabled = bool(lr_cfg.get("protect_candidates", True))
    protect_extra = int(lr_cfg.get("protect_radius_extra_px", 2))
    protect_min = int(lr_cfg.get("protect_min_radius_px", 4))
    zoom_boxes = lr_cfg.get("comparison_zoom_boxes", []) or []
    methods_cfg = lr_cfg.get("methods", {}) or {}

    masks_dir = os.path.join(args.out_dir, "line_removal", "masks")
    cleaned_dir = os.path.join(args.out_dir, "line_removal", "cleaned_images")
    cmp_dir = os.path.join(args.out_dir, "line_removal", "comparison_sheets")
    ensure_dir(masks_dir); ensure_dir(cleaned_dir); ensure_dir(cmp_dir)

    # Load skeletons
    refined_static_skel = _opt_npy(os.path.join(args.out_dir, "line_risk", "refined_static_line_skeleton.npy"))
    dynamic_line_skel = _opt_npy(os.path.join(args.out_dir, "line_risk", "combined_line_skeleton.npy"))
    if refined_static_skel is None or dynamic_line_skel is None:
        raise FileNotFoundError(
            "Need refined_static_line_skeleton.npy and combined_line_skeleton.npy. "
            "Run scripts/02_fuse_delined.py first."
        )

    # ---- Build narrow line masks at each radius ----
    masks_by_r: Dict[int, np.ndarray] = {}
    for r in radii:
        m = narrow_line_mask(refined_static_skel, dynamic_line_skel.astype(np.uint8), int(r))
        masks_by_r[int(r)] = m
        save_png(os.path.join(masks_dir, f"line_mask_radius{r}.png"), (m * 255).astype(np.uint8))
        np.save(os.path.join(masks_dir, f"line_mask_radius{r}.npy"), m)
        print(f"[07] line_mask_radius{r} coverage = {coverage_fraction(m)*100:.2f}%")

    cov_df = line_mask_coverage_table(masks_by_r, too_aggressive_threshold=too_aggr)
    cov_df.to_csv(os.path.join(masks_dir, "line_removal_mask_coverage.csv"), index=False)

    # ---- Defect protect mask ----
    H, W = refined_static_skel.shape
    protect_sources_paths = [
        os.path.join(args.out_dir, "manual_review", "manual_review_top100.csv"),
        os.path.join(args.out_dir, "candidates", "candidates_line_adjacent_uncertain_v2.csv"),
        os.path.join(args.out_dir, "candidates", "candidates_texture_uncertain_v2.csv"),
    ]
    protect_df = aggregate_protect_candidates(args.out_dir, protect_sources_paths)
    print(f"[07] Aggregated {len(protect_df)} protect candidates from {sum(os.path.isfile(p) for p in protect_sources_paths)}/{len(protect_sources_paths)} sources")
    protect_mask = defect_protect_mask(
        (H, W), protect_df, extra_radius_px=protect_extra, min_radius_px=protect_min,
    )
    save_png(os.path.join(masks_dir, "defect_protect_mask.png"), (protect_mask * 255).astype(np.uint8))
    np.save(os.path.join(masks_dir, "defect_protect_mask.npy"), protect_mask)
    print(f"[07] defect_protect_mask coverage = {coverage_fraction(protect_mask)*100:.2f}%")

    # ---- Run all variant combinations ----
    n_written = 0
    for input_key in input_keys:
        input_short = INPUT_NAME_MAP.get(input_key, input_key)
        img = _load_input_image(args.out_dir, args.raw_dir, input_key, config)
        if img is None:
            print(f"[07] WARN: input image {input_key} unavailable; skipping")
            continue
        for r in radii:
            mask = masks_by_r[r]
            for protect_label in (("protect" if protect_enabled else None), "no_protect"):
                if protect_label == "protect" and not protect_enabled:
                    continue
                pm = protect_mask if protect_label == "protect" else None

                # local_median
                lm = methods_cfg.get("local_median", {}) or {}
                if lm.get("enabled", True):
                    for w in lm.get("window_sizes", [15, 25, 41]):
                        cleaned = inpaint_local_median(img, mask, int(w), pm)
                        name = f"cleaned_{input_short}_r{r}_localmedian_w{int(w)}_{protect_label}"
                        np.save(os.path.join(cleaned_dir, f"{name}.npy"), cleaned.astype(np.float32))
                        save_png(os.path.join(cleaned_dir, f"{name}.png"),
                                 robust_rescale_for_preview(cleaned))
                        if w == lm.get("window_sizes", [15, 25, 41])[1]:  # middle preset
                            sheet = _make_comparison_sheet(
                                img, mask, cleaned, residual(img, cleaned),
                                zoom_boxes, name,
                            )
                            save_png(os.path.join(cmp_dir, f"{name}.png"), sheet)
                        n_written += 1

                # telea
                tl = methods_cfg.get("telea", {}) or {}
                if tl.get("enabled", True):
                    for rad in tl.get("radii", [3, 5, 7]):
                        cleaned = inpaint_telea(img, mask, int(rad), pm)
                        name = f"cleaned_{input_short}_r{r}_telea_rad{int(rad)}_{protect_label}"
                        np.save(os.path.join(cleaned_dir, f"{name}.npy"), cleaned.astype(np.float32))
                        save_png(os.path.join(cleaned_dir, f"{name}.png"),
                                 robust_rescale_for_preview(cleaned))
                        if rad == tl.get("radii", [3, 5, 7])[1]:
                            sheet = _make_comparison_sheet(
                                img, mask, cleaned, residual(img, cleaned),
                                zoom_boxes, name,
                            )
                            save_png(os.path.join(cmp_dir, f"{name}.png"), sheet)
                        n_written += 1

                # navier_stokes
                ns = methods_cfg.get("navier_stokes", {}) or {}
                if ns.get("enabled", True):
                    for rad in ns.get("radii", [3, 5, 7]):
                        cleaned = inpaint_navier_stokes(img, mask, int(rad), pm)
                        name = f"cleaned_{input_short}_r{r}_ns_rad{int(rad)}_{protect_label}"
                        np.save(os.path.join(cleaned_dir, f"{name}.npy"), cleaned.astype(np.float32))
                        save_png(os.path.join(cleaned_dir, f"{name}.png"),
                                 robust_rescale_for_preview(cleaned))
                        n_written += 1

    print(f"[07] Wrote {n_written} cleaned variants under {cleaned_dir}")


if __name__ == "__main__":
    main()
