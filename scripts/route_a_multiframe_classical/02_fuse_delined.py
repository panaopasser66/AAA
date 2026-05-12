import _bootstrap  # noqa: F401

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import cv2
import numpy as np
import pandas as pd

from src.fusion import combine_group_fusions, fuse_stack, load_aligned_stack
from src.groups import get_enabled_groups, load_config
from src.invalid_region import build_invalid_region_mask, warp_valid_mask
from src.io_utils import ensure_dir, load_tif_float32, robust_rescale_for_preview, save_png
from src.line_mask import build_line_masks
from src.line_risk import combine_line_risks, compute_line_risk_from_mad
from src.mask_coverage import collect_coverage
from src.refined_static_line_mask import (
    distance_from_skeleton,
    dilate,
    refine_static_line_mask,
    union_skeleton,
)
from src.static_line_mask import (
    build_combined_exclusion,
    build_static_line_masks,
    static_line_response,
)


def _load_per_frame_valid_masks(out_dir: str, enabled: dict, reference: str):
    qc_dir = os.path.join(out_dir, "registration_qc")
    aligned_dir = os.path.join(out_dir, "aligned")
    valid_dict = {}
    H = W = None
    for name, spec in enabled.items():
        csv = os.path.join(qc_dir, f"{name}_transforms.csv")
        if not os.path.isfile(csv):
            continue
        df = pd.read_csv(csv)
        for _, row in df.iterrows():
            fname = row["filename"]
            stem = os.path.splitext(fname)[0]
            label = reference if fname == reference else f"{name}/{fname}"
            valid_npy = os.path.join(aligned_dir, name, f"{stem}_valid.npy")
            if os.path.isfile(valid_npy):
                vm = np.load(valid_npy).astype(np.uint8)
            else:
                if H is None:
                    ref_npy = os.path.join(aligned_dir, name, f"{stem}.npy")
                    arr = np.load(ref_npy)
                    H, W = arr.shape
                vm = warp_valid_mask((H, W), float(row["dx"]), float(row["dy"]))
            valid_dict[label] = vm
    return valid_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="raw")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    config = load_config(args.config)
    enabled = get_enabled_groups(config)
    if not enabled:
        print("[02] No enabled groups; nothing to do.")
        return

    fusion_cfg = config.get("fusion", {}) or {}
    lr_cfg = config.get("line_risk", {}) or {}
    lm_cfg = config.get("line_mask", {}) or {}
    slm_cfg = config.get("static_line_mask", {}) or {}
    rslm_cfg = config.get("refined_static_line_mask", {}) or {}
    cle_cfg = config.get("combined_line_exclusion", {}) or {}
    rcle_cfg = config.get("refined_combined_line_exclusion", {}) or {}
    combined_method = fusion_cfg.get("combined_method", "median")
    lr_p_high = float(lr_cfg.get("normalize_percentile_high", 99.5))

    aligned_dir = os.path.join(args.out_dir, "aligned")
    fusion_dir = os.path.join(args.out_dir, "fusion")
    line_risk_dir = os.path.join(args.out_dir, "line_risk")
    invalid_dir = os.path.join(args.out_dir, "invalid_region")
    reports_dir = os.path.join(args.out_dir, "reports")
    ensure_dir(fusion_dir)
    ensure_dir(line_risk_dir)
    ensure_dir(invalid_dir)
    ensure_dir(reports_dir)

    # ----- Per-group fusion + MAD -> dynamic line risk -----
    per_group_median = {}
    per_group_risk = {}
    per_group_mad = {}
    for name, spec in enabled.items():
        files = spec.get("files", [])
        print(f"[02] Fusing group '{name}' ({len(files)} files)")
        stack, used = load_aligned_stack(aligned_dir, name, files)
        out = fuse_stack(stack)
        median = out["median"]; mean = out["mean"]; std = out["std"]; mad = out["mad"]
        np.save(os.path.join(fusion_dir, f"{name}_fused_median.npy"), median)
        save_png(os.path.join(fusion_dir, f"{name}_fused_median.png"), robust_rescale_for_preview(median))
        save_png(os.path.join(fusion_dir, f"{name}_fused_mean.png"), robust_rescale_for_preview(mean))
        np.save(os.path.join(fusion_dir, f"{name}_fused_std.npy"), std)
        np.save(os.path.join(fusion_dir, f"{name}_fused_mad.npy"), mad)
        np.save(os.path.join(line_risk_dir, f"{name}_mad.npy"), mad)
        save_png(os.path.join(line_risk_dir, f"{name}_mad.png"), robust_rescale_for_preview(mad))
        risk = compute_line_risk_from_mad(mad, percentile_high=lr_p_high)
        np.save(os.path.join(line_risk_dir, f"{name}_line_risk.npy"), risk)
        save_png(os.path.join(line_risk_dir, f"{name}_line_risk.png"), robust_rescale_for_preview(risk, 0, 100))
        # Per-group dynamic line mask (for axis1_angle visualisation per task E)
        dyn_p = int(lm_cfg.get("primary_percentile", 97))
        dyn_thr_grp = float(np.percentile(risk, dyn_p))
        dyn_mask_grp = (risk >= dyn_thr_grp).astype(np.uint8)
        save_png(
            os.path.join(line_risk_dir, f"{name}_dynamic_line_mask.png"),
            (dyn_mask_grp * 255).astype(np.uint8),
        )
        np.save(os.path.join(line_risk_dir, f"{name}_dynamic_line_mask.npy"), dyn_mask_grp)
        per_group_median[name] = median
        per_group_risk[name] = risk
        per_group_mad[name] = mad

    # ----- combined_versions: build v1 (axis2+axis4) and v2 (+ axis1) etc -----
    versions_cfg = config.get("combined_versions", {}) or {}
    if not versions_cfg:
        versions_cfg = {"v1": {"groups": list(enabled.keys())}}

    combined_fused = {}
    combined_risk = {}
    for v_name, v_spec in versions_cfg.items():
        sel_groups = [g for g in v_spec.get("groups", []) if g in per_group_median]
        if not sel_groups:
            print(f"[02] Skipping version {v_name}: no enabled groups in {v_spec.get('groups')}")
            continue
        meds = [per_group_median[g] for g in sel_groups]
        risks = [per_group_risk[g] for g in sel_groups]
        cf = combine_group_fusions(meds, method=combined_method)
        cr = combine_line_risks(risks)
        if v_name == "v1":
            cf_name = "combined_fused_delined"
            cr_name = "combined_line_risk"
        else:
            cf_name = f"combined_fused_delined_{v_name}"
            cr_name = f"combined_line_risk_{v_name}"
        np.save(os.path.join(fusion_dir, f"{cf_name}.npy"), cf)
        save_png(os.path.join(fusion_dir, f"{cf_name}.png"), robust_rescale_for_preview(cf))
        np.save(os.path.join(line_risk_dir, f"{cr_name}.npy"), cr)
        save_png(os.path.join(line_risk_dir, f"{cr_name}.png"), robust_rescale_for_preview(cr, 0, 100))
        combined_fused[v_name] = cf
        combined_risk[v_name] = cr
        print(
            f"[02] version {v_name}: groups={sel_groups} -> {cf_name}.png, {cr_name}.png"
        )

    # ----- Dynamic line mask / skeleton / distance (uses combined v1 line risk) -----
    primary_risk = combined_risk.get("v1") if "v1" in combined_risk else next(iter(combined_risk.values()))
    percentiles = list(lm_cfg.get("percentiles", [90, 95, 97]))
    primary = int(lm_cfg.get("primary_percentile", 97))
    morph_close = int(lm_cfg.get("morphology_close_px", 1))
    distance_clip = int(lm_cfg.get("distance_clip_px", 60))
    rm_small = int(lm_cfg.get("remove_small_objects_px", 0))
    dyn_masks, dyn_skel, dyn_distance, dyn_thr = build_line_masks(
        primary_risk, percentiles=percentiles, primary_percentile=primary,
        morph_close_px=morph_close, remove_small_objects_px=rm_small,
        distance_clip_px=distance_clip,
    )
    for p, m in dyn_masks.items():
        save_png(os.path.join(line_risk_dir, f"combined_line_mask_p{p}.png"), (m * 255).astype(np.uint8))
        np.save(os.path.join(line_risk_dir, f"combined_line_mask_p{p}.npy"), m.astype(np.uint8))
    save_png(os.path.join(line_risk_dir, "combined_line_mask.png"), (dyn_masks[primary] * 255).astype(np.uint8))
    np.save(os.path.join(line_risk_dir, "combined_line_mask.npy"), dyn_masks[primary].astype(np.uint8))
    save_png(os.path.join(line_risk_dir, "combined_line_skeleton.png"), (dyn_skel * 255).astype(np.uint8))
    np.save(os.path.join(line_risk_dir, "combined_line_skeleton.npy"), dyn_skel.astype(np.uint8))
    np.save(os.path.join(line_risk_dir, "combined_line_distance.npy"), dyn_distance)
    save_png(os.path.join(line_risk_dir, "combined_line_distance.png"), robust_rescale_for_preview(dyn_distance, 0, 100))
    print(f"[02] dynamic line_mask p{primary} threshold={dyn_thr:.4f}")

    # ----- Static line response (legacy Stage 1.2) -----
    ref_name = config.get("reference_image", "0000.tif")
    ref_raw_path = os.path.join(args.raw_dir, ref_name)
    if not os.path.isfile(ref_raw_path):
        raise FileNotFoundError(f"Raw reference TIFF missing: {ref_raw_path}")
    ref_raw = load_tif_float32(ref_raw_path)
    sigmas = list(slm_cfg.get("sigmas", [1.0, 2.0, 3.5, 5.0, 8.0]))
    detect_bright = bool(slm_cfg.get("detect_bright", True))
    detect_dark = bool(slm_cfg.get("detect_dark", True))
    slm_percentiles = list(slm_cfg.get("percentiles", [90, 95, 97]))
    slm_primary = int(slm_cfg.get("primary_percentile", 95))
    slm_close = int(slm_cfg.get("morphology_close_px", 1))
    slm_distance_clip = int(slm_cfg.get("distance_clip_px", 60))
    source_for_mask = str(slm_cfg.get("source_for_mask", "fused"))

    fused_v1 = combined_fused.get("v1") if "v1" in combined_fused else next(iter(combined_fused.values()))
    print(f"[02] Computing static line response (sato sigmas={sigmas})")
    resp_raw = static_line_response(ref_raw, sigmas, detect_bright, detect_dark)
    resp_fused = static_line_response(fused_v1, sigmas, detect_bright, detect_dark)
    np.save(os.path.join(line_risk_dir, "static_line_response_0000.npy"), resp_raw)
    np.save(os.path.join(line_risk_dir, "static_line_response_fused.npy"), resp_fused)
    save_png(os.path.join(line_risk_dir, "static_line_response_0000.png"), robust_rescale_for_preview(resp_raw, 0, 99))
    save_png(os.path.join(line_risk_dir, "static_line_response_fused.png"), robust_rescale_for_preview(resp_fused, 0, 99))

    chosen = {"raw": resp_raw, "max_both": np.maximum(resp_raw, resp_fused)}.get(source_for_mask, resp_fused)
    stat_masks, stat_skel, stat_distance, stat_thr = build_static_line_masks(
        chosen, slm_percentiles, slm_primary, slm_close, slm_distance_clip
    )
    for p, m in stat_masks.items():
        save_png(os.path.join(line_risk_dir, f"static_line_mask_p{p}.png"), (m * 255).astype(np.uint8))
        np.save(os.path.join(line_risk_dir, f"static_line_mask_p{p}.npy"), m.astype(np.uint8))
    save_png(os.path.join(line_risk_dir, "static_line_mask.png"), (stat_masks[slm_primary] * 255).astype(np.uint8))
    np.save(os.path.join(line_risk_dir, "static_line_mask.npy"), stat_masks[slm_primary].astype(np.uint8))
    save_png(os.path.join(line_risk_dir, "static_line_skeleton.png"), (stat_skel * 255).astype(np.uint8))
    np.save(os.path.join(line_risk_dir, "static_line_skeleton.npy"), stat_skel.astype(np.uint8))
    np.save(os.path.join(line_risk_dir, "static_line_distance.npy"), stat_distance)
    save_png(os.path.join(line_risk_dir, "static_line_distance.png"), robust_rescale_for_preview(stat_distance, 0, 100))
    print(f"[02] static line_mask source={source_for_mask}, p{slm_primary} threshold={stat_thr:.4f}")

    # ----- REFINED static line mask: shape-filtered components -----
    refined_response = chosen
    save_png(
        os.path.join(line_risk_dir, "refined_static_line_response.png"),
        robust_rescale_for_preview(refined_response, 0, 99),
    )
    np.save(os.path.join(line_risk_dir, "refined_static_line_response.npy"), refined_response)
    refined_mask, refined_components, refined_skel, refined_kept = refine_static_line_mask(
        refined_response,
        source_percentile=float(rslm_cfg.get("source_percentile", 90.0)),
        morphology_close_px=int(rslm_cfg.get("morphology_close_px", 3)),
        min_area_px=int(rslm_cfg.get("min_area_px", 30)),
        min_major_axis_length=float(rslm_cfg.get("min_major_axis_length", 30.0)),
        min_aspect_ratio=float(rslm_cfg.get("min_aspect_ratio", 4.0)),
        min_eccentricity=float(rslm_cfg.get("min_eccentricity", 0.90)),
        min_skeleton_length_px=int(rslm_cfg.get("min_skeleton_length_px", 25)),
        max_circularity=float(rslm_cfg.get("max_circularity", 0.6)),
    )
    rslm_dist_clip = int(rslm_cfg.get("distance_clip_px", 60))
    refined_distance = distance_from_skeleton(refined_skel, distance_clip_px=rslm_dist_clip)
    np.save(os.path.join(line_risk_dir, "refined_static_line_mask.npy"), refined_mask)
    np.save(os.path.join(line_risk_dir, "refined_static_line_skeleton.npy"), refined_skel)
    np.save(os.path.join(line_risk_dir, "refined_static_line_distance.npy"), refined_distance)
    save_png(os.path.join(line_risk_dir, "refined_static_line_mask.png"), (refined_mask * 255).astype(np.uint8))
    save_png(os.path.join(line_risk_dir, "refined_static_line_skeleton.png"), (refined_skel * 255).astype(np.uint8))
    save_png(os.path.join(line_risk_dir, "refined_static_line_distance.png"), robust_rescale_for_preview(refined_distance, 0, 100))
    save_png(os.path.join(line_risk_dir, "refined_static_line_components.png"), refined_components)
    pd.DataFrame.from_dict(refined_kept, orient="index").to_csv(
        os.path.join(line_risk_dir, "refined_static_line_components.csv")
    )
    print(
        f"[02] refined_static_line_mask kept {int(refined_mask.sum())} px "
        f"({100.0 * refined_mask.mean():.2f}% of frame) from {len(refined_kept)} components"
    )

    # ----- Combined exclusion v1 (legacy, raw static skeleton + dynamic) -----
    dilate_list_v1 = list(cle_cfg.get("dilate_px_list", [8, 12, 16]))
    primary_dilate = int(cle_cfg.get("primary_dilate_px", 12))
    cle_distance_clip = int(cle_cfg.get("distance_clip_px", 60))
    cle = build_combined_exclusion(
        dyn_masks[primary],
        stat_masks[slm_primary],
        dilate_list_v1,
        primary_dilate,
        cle_distance_clip,
    )
    save_png(os.path.join(line_risk_dir, "combined_line_exclusion_mask.png"), (cle["mask"] * 255).astype(np.uint8))
    np.save(os.path.join(line_risk_dir, "combined_line_exclusion_mask.npy"), cle["mask"].astype(np.uint8))
    save_png(os.path.join(line_risk_dir, "combined_line_exclusion_skeleton.png"), (cle["skeleton"] * 255).astype(np.uint8))
    np.save(os.path.join(line_risk_dir, "combined_line_exclusion_distance.npy"), cle["distance"])
    save_png(os.path.join(line_risk_dir, "combined_line_exclusion_distance.png"), robust_rescale_for_preview(cle["distance"], 0, 100))
    for n in dilate_list_v1:
        m = cle.get(f"dilate_{n}")
        if m is None: continue
        save_png(os.path.join(line_risk_dir, f"combined_line_exclusion_mask_dilate{n}.png"), (m * 255).astype(np.uint8))
        np.save(os.path.join(line_risk_dir, f"combined_line_exclusion_mask_dilate{n}.npy"), m.astype(np.uint8))
    print(f"[02] combined exclusion (v1) covers {float(cle['mask'].mean())*100:.1f}% of frame")

    # ----- REFINED combined exclusion: dynamic_skeleton ∪ refined_static_skeleton -----
    refined_dilate_list = list(rcle_cfg.get("dilate_px_list", [6, 8, 10]))
    refined_primary_dilate = int(rcle_cfg.get("primary_dilate_px", 8))
    refined_dist_clip = int(rcle_cfg.get("distance_clip_px", 60))
    refined_skel_union = union_skeleton(dyn_skel, refined_skel)
    np.save(os.path.join(line_risk_dir, "refined_combined_line_exclusion_skeleton.npy"), refined_skel_union)
    save_png(os.path.join(line_risk_dir, "refined_combined_line_exclusion_skeleton.png"),
             (refined_skel_union * 255).astype(np.uint8))
    refined_dist_union = distance_from_skeleton(refined_skel_union, distance_clip_px=refined_dist_clip)
    np.save(os.path.join(line_risk_dir, "refined_combined_line_exclusion_distance.npy"), refined_dist_union)
    save_png(
        os.path.join(line_risk_dir, "refined_combined_line_exclusion_distance.png"),
        robust_rescale_for_preview(refined_dist_union, 0, 100),
    )
    refined_dilations = {}
    for n in refined_dilate_list:
        d = dilate(refined_skel_union, int(n))
        refined_dilations[n] = d
        save_png(
            os.path.join(line_risk_dir, f"refined_combined_line_exclusion_dilate{n}.png"),
            (d * 255).astype(np.uint8),
        )
        np.save(os.path.join(line_risk_dir, f"refined_combined_line_exclusion_dilate{n}.npy"), d.astype(np.uint8))
    primary_refined_dilate = refined_dilations.get(
        refined_primary_dilate, dilate(refined_skel_union, refined_primary_dilate)
    )
    print(f"[02] refined combined exclusion (primary dilate {refined_primary_dilate}) covers "
          f"{float(primary_refined_dilate.mean())*100:.1f}% of frame")

    # ----- Invalid region & legacy safe_search_mask (Stage 1.2) -----
    ref_norm_path = os.path.join(args.out_dir, "normalized", ref_name.replace(".tif", ".npy"))
    if not os.path.isfile(ref_norm_path):
        raise FileNotFoundError(f"Reference normalized npy missing: {ref_norm_path}")
    ref_norm = np.load(ref_norm_path).astype(np.float32)
    valid_dict = _load_per_frame_valid_masks(args.out_dir, enabled, ref_name)
    inv_components = build_invalid_region_mask(ref_norm, list(valid_dict.values()), config)
    np.save(os.path.join(invalid_dir, "invalid_region_mask.npy"), inv_components["invalid"].astype(np.uint8))
    save_png(os.path.join(invalid_dir, "invalid_region_mask.png"), (inv_components["invalid"] * 255).astype(np.uint8))
    save_png(os.path.join(invalid_dir, "valid_region_mask.png"),
             (inv_components["warp_valid_combined"] * 255).astype(np.uint8))
    np.save(os.path.join(invalid_dir, "warp_valid_combined.npy"), inv_components["warp_valid_combined"])
    save_png(os.path.join(invalid_dir, "dark_vignette_mask.png"),
             (inv_components["dark_vignette"] * 255).astype(np.uint8))
    save_png(os.path.join(invalid_dir, "bright_saturation_mask.png"),
             (inv_components["bright_saturation"] * 255).astype(np.uint8))
    invalid_frac = float(inv_components["invalid"].mean())
    print(f"[02] invalid_region covers {invalid_frac*100:.1f}% of frame")

    # Legacy safe_search_mask = ~(cle.primary_dilate | invalid)
    safe_v1 = ((cle["primary_dilate"] == 0) & (inv_components["invalid"] == 0)).astype(np.uint8)
    np.save(os.path.join(invalid_dir, "safe_search_mask.npy"), safe_v1)
    save_png(os.path.join(invalid_dir, "safe_search_mask.png"), (safe_v1 * 255).astype(np.uint8))

    # ----- Refined safe_search_mask family -----
    for n, dmask in refined_dilations.items():
        safe = ((dmask == 0) & (inv_components["invalid"] == 0)).astype(np.uint8)
        np.save(os.path.join(invalid_dir, f"refined_safe_search_mask_dilate{n}.npy"), safe)
        save_png(
            os.path.join(invalid_dir, f"refined_safe_search_mask_dilate{n}.png"),
            (safe * 255).astype(np.uint8),
        )
        print(f"[02] refined_safe_search_mask_dilate{n} covers {float(safe.mean())*100:.1f}%")

    # ----- Mask coverage report -----
    mc_cfg = config.get("mask_coverage", {}) or {}
    too_aggr = float(mc_cfg.get("too_aggressive_threshold", 0.50))
    cov_df = collect_coverage(args.out_dir, too_aggressive_threshold=too_aggr)
    cov_path = os.path.join(reports_dir, "mask_coverage.csv")
    cov_df.to_csv(cov_path, index=False)
    print(f"[02] Wrote mask coverage report: {cov_path}")


if __name__ == "__main__":
    main()
