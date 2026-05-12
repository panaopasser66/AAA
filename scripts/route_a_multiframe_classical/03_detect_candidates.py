import _bootstrap  # noqa: F401

import argparse
import os
from collections import OrderedDict

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.blob_detection import detect_blob_candidates
from src.candidate_scoring import apply_sweep, categorize, score_candidates
from src.groups import get_enabled_groups, load_config
from src.invalid_region import warp_valid_mask
from src.io_utils import ensure_dir, robust_rescale_for_preview


CATEGORY_COLORS = {
    "high_confidence_clean":   (0, 255, 80),
    "line_adjacent_uncertain": (0, 200, 255),
    "texture_uncertain":       (220, 220, 80),
    "rejected_line_like":      (60, 60, 255),
    "rejected_tiny_response":  (180, 180, 180),
}


def _load_aligned_images(out_dir, enabled_groups, reference):
    aligned_dir = os.path.join(out_dir, "aligned")
    images = OrderedDict()
    ref_added = False
    for name, spec in enabled_groups.items():
        if reference in spec.get("files", []):
            ref_path = os.path.join(aligned_dir, name, reference.replace(".tif", ".npy"))
            if os.path.isfile(ref_path) and not ref_added:
                images[reference] = np.load(ref_path).astype(np.float32)
                ref_added = True
                break
    for name, spec in enabled_groups.items():
        for fname in spec.get("files", []):
            if fname == reference and ref_added:
                continue
            label = f"{name}/{fname}"
            p = os.path.join(aligned_dir, name, fname.replace(".tif", ".npy"))
            if os.path.isfile(p):
                images[label] = np.load(p).astype(np.float32)
    return images


def _load_valid_masks(out_dir, enabled_groups, reference, shape):
    aligned_dir = os.path.join(out_dir, "aligned")
    qc_dir = os.path.join(out_dir, "registration_qc")
    valid = OrderedDict()
    H, W = shape
    ref_added = False
    for name, spec in enabled_groups.items():
        if reference in spec.get("files", []):
            stem = os.path.splitext(reference)[0]
            vp = os.path.join(aligned_dir, name, f"{stem}_valid.npy")
            if os.path.isfile(vp):
                valid[reference] = np.load(vp).astype(np.uint8)
            else:
                valid[reference] = np.ones((H, W), dtype=np.uint8)
            ref_added = True
            break
    for name, spec in enabled_groups.items():
        csv = os.path.join(qc_dir, f"{name}_transforms.csv")
        if not os.path.isfile(csv):
            continue
        df = pd.read_csv(csv)
        for _, row in df.iterrows():
            fname = row["filename"]
            stem = os.path.splitext(fname)[0]
            if fname == reference and ref_added:
                continue
            label = f"{name}/{fname}"
            vp = os.path.join(aligned_dir, name, f"{stem}_valid.npy")
            if os.path.isfile(vp):
                valid[label] = np.load(vp).astype(np.uint8)
            else:
                valid[label] = warp_valid_mask((H, W), float(row["dx"]), float(row["dy"]))
    return valid


def _draw_overlay(img, df, out_path, color_by_category=True, max_label=200, color=None):
    prev = robust_rescale_for_preview(img)
    bgr = cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR)
    for i, row in df.iterrows():
        x, y = int(round(row["x"])), int(round(row["y"]))
        r = max(2, int(round(float(row.get("radius_px", 3.0)) + 2)))
        if color is not None:
            c = color
        elif color_by_category:
            c = CATEGORY_COLORS.get(row.get("category", "uncertain"), (200, 200, 200))
        else:
            c = (0, 255, 255) if row.get("polarity", "") == "bright" else (255, 80, 255)
        cv2.circle(bgr, (x, y), r, c, 1, cv2.LINE_AA)
        if i < max_label:
            cv2.putText(bgr, str(int(row["candidate_id"])),
                        (x + r + 1, y - r - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1, cv2.LINE_AA)
    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, bgr)


def _df_to_csv(df: pd.DataFrame, path: str):
    d = df.copy()
    for col in ["per_frame_anisotropy", "per_frame_contrast", "per_frame_valid"]:
        if col in d.columns:
            d[col] = d[col].apply(
                lambda v: ";".join(
                    (f"{float(x):.4f}" if (isinstance(x, (int, float)) and np.isfinite(x)) else "nan")
                    for x in (v if isinstance(v, list) else [])
                )
            )
    d.to_csv(path, index=False)


def _detect_for_version(
    version: str,
    fused: np.ndarray,
    line_risk: np.ndarray,
    dyn_dist: np.ndarray,
    stat_dist: np.ndarray,
    comb_dist: np.ndarray,
    refined_stat_dist: np.ndarray,
    refined_comb_dist: np.ndarray,
    invalid_region: np.ndarray,
    aligned_images,
    valid_masks,
    config,
    cand_dir: str,
):
    """Run blob+score+categorize+sweep for one combined-fusion version."""
    suffix = "" if version == "v1" else f"_{version}"

    candidates = detect_blob_candidates(fused, config)
    print(f"[03] [{version}] Detected {len(candidates)} raw candidates after blob+NMS")
    df = score_candidates(
        candidates, fused, line_risk, dyn_dist, stat_dist, comb_dist,
        invalid_region, aligned_images, valid_masks, config,
        refined_static_line_distance=refined_stat_dist,
        refined_combined_exclusion_distance=refined_comb_dist,
    )
    df = categorize(df, config)
    print(f"[03] [{version}] Category counts: {df['category'].value_counts().to_dict()}")

    csv_paths = {
        "all": os.path.join(cand_dir, f"candidates_all{suffix}.csv"),
        "high_confidence_clean": os.path.join(cand_dir, f"candidates_high_confidence_clean{suffix}.csv"),
        "line_adjacent_uncertain": os.path.join(cand_dir, f"candidates_line_adjacent_uncertain{suffix}.csv"),
        "texture_uncertain": os.path.join(cand_dir, f"candidates_texture_uncertain{suffix}.csv"),
        "rejected_line_like": os.path.join(cand_dir, f"candidates_rejected_line_like{suffix}.csv"),
        "rejected_tiny_response": os.path.join(cand_dir, f"candidates_rejected_tiny_response{suffix}.csv"),
    }
    _df_to_csv(df, csv_paths["all"])
    for cat in ("high_confidence_clean", "line_adjacent_uncertain", "texture_uncertain",
                "rejected_line_like", "rejected_tiny_response"):
        _df_to_csv(df[df.category == cat], csv_paths[cat])

    # v1 legacy aliases (kept for Stage 1.2 compatibility)
    if version == "v1":
        _df_to_csv(df, os.path.join(cand_dir, "candidates.csv"))
        _df_to_csv(df[df.category == "high_confidence_clean"], os.path.join(cand_dir, "candidates_high_confidence.csv"))
        _df_to_csv(df[df.category == "line_adjacent_uncertain"], os.path.join(cand_dir, "candidates_line_risk.csv"))
        _df_to_csv(
            df[df.category.isin(["rejected_line_like", "rejected_tiny_response"])],
            os.path.join(cand_dir, "candidates_rejected.csv"),
        )

    # Overlays
    _draw_overlay(fused, df, os.path.join(cand_dir, f"candidates_overlay_all{suffix}.png"))
    for cat in ("high_confidence_clean", "line_adjacent_uncertain", "texture_uncertain", "rejected_line_like"):
        sub = df[df.category == cat].reset_index(drop=True)
        _draw_overlay(
            fused, sub,
            os.path.join(cand_dir, f"candidates_overlay_{cat}{suffix}.png"),
        )

    if version == "v1":
        _draw_overlay(
            fused, df[df.category == "high_confidence_clean"].reset_index(drop=True),
            os.path.join(cand_dir, "candidates_overlay_high_confidence.png"),
        )
        _draw_overlay(
            fused, df[df.category == "line_adjacent_uncertain"].reset_index(drop=True),
            os.path.join(cand_dir, "candidates_overlay_line_risk.png"),
        )
        _draw_overlay(
            fused, df[df.category.isin(["rejected_line_like", "rejected_tiny_response"])].reset_index(drop=True),
            os.path.join(cand_dir, "candidates_overlay_rejected.png"),
        )
        _draw_overlay(fused, df, os.path.join(cand_dir, "candidates_overlay.png"))

    # Parameter sweep
    hc_cfg = (config.get("candidate_classification") or {}).get("high_confidence_clean", {}) or {}
    sweep_cfg = config.get("parameter_sweep", {}) or {}
    sweep_dir = os.path.join(cand_dir, "sweeps")
    ensure_dir(sweep_dir)
    sweep_counts = {}
    for preset_name, overrides in sweep_cfg.items():
        sub = apply_sweep(df, hc_cfg, overrides)
        sub = sub.sort_values("final_score", ascending=False).reset_index(drop=True)
        out_csv = os.path.join(sweep_dir, f"{preset_name}_high_confidence{suffix}.csv")
        out_png = os.path.join(sweep_dir, f"{preset_name}_overlay{suffix}.png")
        _df_to_csv(sub, out_csv)
        _draw_overlay(
            fused, sub, out_png, color=CATEGORY_COLORS["high_confidence_clean"],
        )
        sweep_counts[preset_name] = len(sub)
    print(f"[03] [{version}] sweep counts: {sweep_counts}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    config = load_config(args.config)
    enabled = get_enabled_groups(config)
    reference = config.get("reference_image", "0000.tif")
    versions_cfg = config.get("combined_versions", {}) or {}
    if not versions_cfg:
        versions_cfg = {"v1": {"groups": list(enabled.keys())}}

    line_risk_path = os.path.join(args.out_dir, "line_risk", "combined_line_risk.npy")
    dyn_dist_path = os.path.join(args.out_dir, "line_risk", "combined_line_distance.npy")
    stat_dist_path = os.path.join(args.out_dir, "line_risk", "static_line_distance.npy")
    comb_dist_path = os.path.join(args.out_dir, "line_risk", "combined_line_exclusion_distance.npy")
    refined_stat_dist_path = os.path.join(args.out_dir, "line_risk", "refined_static_line_distance.npy")
    refined_comb_dist_path = os.path.join(args.out_dir, "line_risk", "refined_combined_line_exclusion_distance.npy")
    invalid_path = os.path.join(args.out_dir, "invalid_region", "invalid_region_mask.npy")
    for p in (line_risk_path, dyn_dist_path, stat_dist_path, comb_dist_path,
              refined_stat_dist_path, refined_comb_dist_path, invalid_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing artefact: {p}. Run 02_fuse_delined first.")

    line_risk = np.load(line_risk_path).astype(np.float32)
    dyn_dist = np.load(dyn_dist_path).astype(np.float32)
    stat_dist = np.load(stat_dist_path).astype(np.float32)
    comb_dist = np.load(comb_dist_path).astype(np.float32)
    refined_stat_dist = np.load(refined_stat_dist_path).astype(np.float32)
    refined_comb_dist = np.load(refined_comb_dist_path).astype(np.float32)
    invalid_region = np.load(invalid_path).astype(np.uint8)

    aligned_images = _load_aligned_images(args.out_dir, enabled, reference)

    cand_dir = os.path.join(args.out_dir, "candidates")
    ensure_dir(cand_dir)

    for version in versions_cfg.keys():
        if version == "v1":
            fused_path = os.path.join(args.out_dir, "fusion", "combined_fused_delined.npy")
        else:
            fused_path = os.path.join(args.out_dir, "fusion", f"combined_fused_delined_{version}.npy")
            if not os.path.isfile(fused_path):
                print(f"[03] Skipping version {version}: {fused_path} missing")
                continue
        fused = np.load(fused_path).astype(np.float32)
        valid_masks = _load_valid_masks(args.out_dir, enabled, reference, fused.shape)
        _detect_for_version(
            version, fused, line_risk, dyn_dist, stat_dist, comb_dist,
            refined_stat_dist, refined_comb_dist, invalid_region,
            aligned_images, valid_masks, config, cand_dir,
        )


if __name__ == "__main__":
    main()
