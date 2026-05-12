import _bootstrap  # noqa: F401

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from src.groups import get_enabled_groups, load_config
from src.report import build_report


CATEGORIES = (
    "high_confidence_clean",
    "line_adjacent_uncertain",
    "texture_uncertain",
    "rejected_line_like",
    "rejected_tiny_response",
)


def _version_label(v: str) -> str:
    if v == "v1":
        return "Combined v1 (axis2_x + axis4_y)"
    if v == "v2":
        return "Combined v2 (axis2_x + axis4_y + axis1_angle)"
    return f"Combined {v}"


def _build_version_block(out_dir: str, config: dict, version: str, max_caps):
    suffix = "" if version == "v1" else f"_{version}"
    cand_dir = os.path.join(out_dir, "candidates")
    csv_all = os.path.join(cand_dir, f"candidates_all{suffix}.csv")
    all_df = pd.read_csv(csv_all) if os.path.isfile(csv_all) else pd.DataFrame()
    if "category" not in all_df.columns:
        all_df["category"] = "uncertain"

    counts = {"total": int(len(all_df))}
    for c in CATEGORIES:
        counts[c] = int((all_df["category"] == c).sum())

    overlays = {
        "all": os.path.join(cand_dir, f"candidates_overlay_all{suffix}.png"),
        "high_confidence_clean": os.path.join(cand_dir, f"candidates_overlay_high_confidence_clean{suffix}.png"),
        "line_adjacent_uncertain": os.path.join(cand_dir, f"candidates_overlay_line_adjacent_uncertain{suffix}.png"),
        "texture_uncertain": os.path.join(cand_dir, f"candidates_overlay_texture_uncertain{suffix}.png"),
        "rejected_line_like": os.path.join(cand_dir, f"candidates_overlay_rejected_line_like{suffix}.png"),
    }
    csvs = {
        "all": csv_all,
        "high_confidence_clean": os.path.join(cand_dir, f"candidates_high_confidence_clean{suffix}.csv"),
        "line_adjacent_uncertain": os.path.join(cand_dir, f"candidates_line_adjacent_uncertain{suffix}.csv"),
        "texture_uncertain": os.path.join(cand_dir, f"candidates_texture_uncertain{suffix}.csv"),
        "rejected_line_like": os.path.join(cand_dir, f"candidates_rejected_line_like{suffix}.csv"),
        "rejected_tiny_response": os.path.join(cand_dir, f"candidates_rejected_tiny_response{suffix}.csv"),
    }

    def crops_for(cat, n):
        sub = all_df[all_df.category == cat]
        if "final_score" in sub.columns:
            sub = sub.sort_values("final_score", ascending=False)
        out = []
        for _, r in sub.head(n).iterrows():
            out.append({
                "candidate_id": int(r["candidate_id"]),
                "polarity": str(r.get("polarity", "")),
                "x": float(r["x"]), "y": float(r["y"]),
                "diameter_px": float(r.get("diameter_px", 0.0)),
                "area_px": float(r.get("area_px", 0.0)) if pd.notna(r.get("area_px", 0.0)) else 0.0,
                "circularity": float(r.get("circularity", 0.0)) if pd.notna(r.get("circularity", 0.0)) else 0.0,
                "aspect_ratio": float(r.get("aspect_ratio", 0.0)) if pd.notna(r.get("aspect_ratio", 0.0)) else 0.0,
                "final_score": float(r.get("final_score", 0.0)),
                "anisotropy": float(r.get("anisotropy", 0.0)) if pd.notna(r.get("anisotropy", 0.0)) else 0.0,
                "distance_to_refined_static_line_skeleton": float(r.get("distance_to_refined_static_line_skeleton", 0.0)) if pd.notna(r.get("distance_to_refined_static_line_skeleton", 0.0)) else 0.0,
                "distance_to_refined_combined_line_exclusion": float(r.get("distance_to_refined_combined_line_exclusion", 0.0)) if pd.notna(r.get("distance_to_refined_combined_line_exclusion", 0.0)) else 0.0,
                "mean_line_risk_in_patch": float(r.get("mean_line_risk_in_patch", 0.0)) if pd.notna(r.get("mean_line_risk_in_patch", 0.0)) else 0.0,
                "valid_frame_count": int(r.get("valid_frame_count", 0)),
                "uncovered_frame_count": int(r.get("uncovered_frame_count", 0)),
                "spot_present_when_uncovered": int(r.get("spot_present_when_uncovered", 0)),
                "in_invalid_region": int(r.get("in_invalid_region", 0)),
                "crop_path": str(r.get("crop_path", "")) if pd.notna(r.get("crop_path", "")) else "",
            })
        return out

    crops_by_category = {c: crops_for(c, max_caps[c]) for c in CATEGORIES}

    show_cols = [c for c in [
        "candidate_id", "x", "y", "diameter_px", "polarity",
        "area_px", "circularity", "aspect_ratio", "anisotropy",
        "distance_to_refined_static_line_skeleton",
        "distance_to_refined_combined_line_exclusion",
        "mean_line_risk_in_patch",
        "valid_frame_count", "uncovered_frame_count", "spot_present_when_uncovered",
        "border_distance", "final_score",
    ] if c in all_df.columns]
    hc_df = all_df[all_df.category == "high_confidence_clean"].sort_values(
        "final_score", ascending=False
    ).head(max_caps["high_confidence_clean"])
    table_html = {
        "high_confidence_clean": (
            hc_df[show_cols].to_html(index=False, float_format="%.3f")
            if not hc_df.empty
            else "<i>No high-confidence-clean candidates passed all gates (allowed per task spec).</i>"
        ),
    }

    sweep_cfg = config.get("parameter_sweep", {}) or {}
    sweep_dir = os.path.join(cand_dir, "sweeps")
    sweeps = []
    for preset, overrides in sweep_cfg.items():
        csv_p = os.path.join(sweep_dir, f"{preset}_high_confidence{suffix}.csv")
        ov_p = os.path.join(sweep_dir, f"{preset}_overlay{suffix}.png")
        n = 0
        if os.path.isfile(csv_p):
            try:
                n = len(pd.read_csv(csv_p))
            except Exception:
                n = 0
        sweeps.append({"name": preset, "count": n, "csv": csv_p, "overlay": ov_p})

    return {
        "label": _version_label(version),
        "counts": counts,
        "overlays": overlays,
        "csvs": csvs,
        "crops": crops_by_category,
        "tables": table_html,
        "sweeps": sweeps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    config = load_config(args.config)
    enabled = get_enabled_groups(config)
    report_cfg = config.get("report", {}) or {}
    max_caps = {
        "high_confidence_clean": int(report_cfg.get("max_high_confidence_in_html", 200)),
        "line_adjacent_uncertain": int(report_cfg.get("max_line_adjacent_in_html", 80)),
        "texture_uncertain": int(report_cfg.get("max_texture_uncertain_in_html", 60)),
        "rejected_line_like": int(report_cfg.get("max_rejected_in_html", 30)),
        "rejected_tiny_response": 20,
    }

    out_dir = args.out_dir
    metadata_csv = os.path.join(out_dir, "inspect", "metadata.csv")
    inspect_sheet = os.path.join(out_dir, "inspect", "contact_sheet.png")
    mask_coverage_csv = os.path.join(out_dir, "reports", "mask_coverage.csv")

    enabled_meta = []
    for name, spec in enabled.items():
        enabled_meta.append({
            "name": name,
            "transform_model": spec.get("transform_model", ""),
            "files": spec.get("files", []),
            "transforms_csv": os.path.join(out_dir, "registration_qc", f"{name}_transforms.csv"),
            "aligned_sheet": os.path.join(out_dir, "registration_qc", f"{name}_aligned_contact_sheet.png"),
            "diff_before": os.path.join(out_dir, "registration_qc", f"{name}_diff_before.png"),
            "diff_after": os.path.join(out_dir, "registration_qc", f"{name}_diff_after.png"),
            "aligned_std": os.path.join(out_dir, "registration_qc", f"{name}_aligned_std.png"),
            "valid_region": os.path.join(out_dir, "registration_qc", f"{name}_valid_region.png"),
            "dynamic_line_mask": os.path.join(out_dir, "line_risk", f"{name}_dynamic_line_mask.png"),
        })

    fused_compare = [
        {"title": "combined_fused_delined (v1)", "path": os.path.join(out_dir, "fusion", "combined_fused_delined.png")},
    ]
    v2_path = os.path.join(out_dir, "fusion", "combined_fused_delined_v2.png")
    if os.path.isfile(v2_path):
        fused_compare.append({"title": "combined_fused_delined (v2)", "path": v2_path})
    per_group_fused = [
        {"title": f"{name}_fused_median", "path": os.path.join(out_dir, "fusion", f"{name}_fused_median.png")}
        for name in enabled.keys()
    ]

    line_risk_images = [
        {"title": "combined_line_risk (v1)", "path": os.path.join(out_dir, "line_risk", "combined_line_risk.png")},
    ]
    v2_lr = os.path.join(out_dir, "line_risk", "combined_line_risk_v2.png")
    if os.path.isfile(v2_lr):
        line_risk_images.append({"title": "combined_line_risk (v2)", "path": v2_lr})
    for name in enabled.keys():
        line_risk_images.append({
            "title": f"{name}_line_risk",
            "path": os.path.join(out_dir, "line_risk", f"{name}_line_risk.png"),
        })

    static_compare = [
        {"title": "raw static line mask (Sato p95)", "path": os.path.join(out_dir, "line_risk", "static_line_mask.png")},
        {"title": "refined static line components (kept=white)", "path": os.path.join(out_dir, "line_risk", "refined_static_line_components.png")},
        {"title": "refined static line mask", "path": os.path.join(out_dir, "line_risk", "refined_static_line_mask.png")},
        {"title": "refined static line skeleton", "path": os.path.join(out_dir, "line_risk", "refined_static_line_skeleton.png")},
        {"title": "refined static line distance", "path": os.path.join(out_dir, "line_risk", "refined_static_line_distance.png")},
    ]

    exclusion_compare = [
        {"title": "v1 combined_exclusion_mask_dilate12 (coarse)",
         "path": os.path.join(out_dir, "line_risk", "combined_line_exclusion_mask_dilate12.png")},
        {"title": "refined_combined_exclusion_dilate6",
         "path": os.path.join(out_dir, "line_risk", "refined_combined_line_exclusion_dilate6.png")},
        {"title": "refined_combined_exclusion_dilate8 (primary)",
         "path": os.path.join(out_dir, "line_risk", "refined_combined_line_exclusion_dilate8.png")},
        {"title": "refined_combined_exclusion_dilate10",
         "path": os.path.join(out_dir, "line_risk", "refined_combined_line_exclusion_dilate10.png")},
        {"title": "refined_combined_exclusion_distance",
         "path": os.path.join(out_dir, "line_risk", "refined_combined_line_exclusion_distance.png")},
    ]

    safe_images = [
        {"title": "safe_search_mask (v1, legacy)",
         "path": os.path.join(out_dir, "invalid_region", "safe_search_mask.png")},
        {"title": "refined_safe_search_mask_dilate6",
         "path": os.path.join(out_dir, "invalid_region", "refined_safe_search_mask_dilate6.png")},
        {"title": "refined_safe_search_mask_dilate8 (primary)",
         "path": os.path.join(out_dir, "invalid_region", "refined_safe_search_mask_dilate8.png")},
        {"title": "refined_safe_search_mask_dilate10",
         "path": os.path.join(out_dir, "invalid_region", "refined_safe_search_mask_dilate10.png")},
        {"title": "invalid_region_mask",
         "path": os.path.join(out_dir, "invalid_region", "invalid_region_mask.png")},
    ]

    versions = list((config.get("combined_versions", {}) or {"v1": {}}).keys())
    version_blocks = [_build_version_block(out_dir, config, v, max_caps) for v in versions]

    # Stage 3 artefacts (manual review + cleaned variants)
    mr_cfg = config.get("manual_review", {}) or {}
    lr_cfg = config.get("line_removal", {}) or {}
    cd_cfg = config.get("cleaned_candidate_detection", {}) or {}
    review_dir = os.path.join(out_dir, "manual_review")
    line_removal_dir = os.path.join(out_dir, "line_removal")
    masks_dir = os.path.join(line_removal_dir, "masks")
    cleaned_dir = os.path.join(line_removal_dir, "cleaned_images")
    cmp_dir = os.path.join(line_removal_dir, "comparison_sheets")
    cleaned_cand_dir = os.path.join(line_removal_dir, "candidates_on_cleaned")

    review_top_counts = list(mr_cfg.get("top_k", mr_cfg.get("top_counts", [50, 100])))
    review_csvs = {n: os.path.join(review_dir, f"manual_review_top{n}.csv") for n in review_top_counts}
    review_overlays = {n: os.path.join(review_dir, f"manual_review_overlay_top{n}.png") for n in review_top_counts}
    review_indices = {n: os.path.join(review_dir, f"review_index_top{n}.html") for n in review_top_counts}
    review_labels_template = os.path.join(review_dir, "manual_labels_template.csv")

    narrow_masks = []
    for r in lr_cfg.get("mask_radii", [2, 4, 6]):
        narrow_masks.append({
            "title": f"line_mask_radius{r}",
            "path": os.path.join(masks_dir, f"line_mask_radius{r}.png"),
        })
    narrow_masks.append({
        "title": "defect_protect_mask",
        "path": os.path.join(masks_dir, "defect_protect_mask.png"),
    })

    cleaned_images = []
    if os.path.isdir(cleaned_dir):
        import glob as _glob
        for png in sorted(_glob.glob(os.path.join(cleaned_dir, "cleaned_*.png")))[:24]:
            name = os.path.splitext(os.path.basename(png))[0]
            cleaned_images.append({"title": name, "path": png})

    comparison_sheets = []
    if os.path.isdir(cmp_dir):
        import glob as _glob
        for png in sorted(_glob.glob(os.path.join(cmp_dir, "*.png")))[:12]:
            comparison_sheets.append({
                "title": os.path.splitext(os.path.basename(png))[0],
                "path": png,
            })

    cleaned_summary_rows = []
    summary_csv = os.path.join(cleaned_cand_dir, "summary.csv")
    if os.path.isfile(summary_csv):
        sdf = pd.read_csv(summary_csv)
        for _, r in sdf.iterrows():
            v = r.get("variant", "")
            cleaned_summary_rows.append({
                "variant": v,
                "status": r.get("status", ""),
                "total": int(r.get("total", 0)),
                "passes_shape_filter": int(r.get("passes_shape_filter", 0)),
                "outside_invalid": int(r.get("outside_invalid", 0)),
                "csv": os.path.join(cleaned_cand_dir, v, "candidates.csv"),
                "overlay": os.path.join(cleaned_cand_dir, v, "overlay.png"),
            })
    cleaned_overlay_preview = cleaned_summary_rows[0].get("overlay", "") if cleaned_summary_rows else ""

    line_mask_coverage_csv = os.path.join(masks_dir, "line_removal_mask_coverage.csv")

    stage3 = {
        "review_source": mr_cfg.get("source_version", "v2"),
        "review_top_counts": review_top_counts,
        "review_csvs": review_csvs,
        "review_overlays": review_overlays,
        "review_indices": review_indices,
        "review_labels_template": review_labels_template,
        "narrow_masks": narrow_masks,
        "narrow_mask_coverage_csv": line_mask_coverage_csv,
        "cleaned_images": cleaned_images,
        "comparison_sheets": comparison_sheets,
        "cleaned_summary_rows": cleaned_summary_rows,
        "cleaned_overlay_preview": cleaned_overlay_preview,
        "selected_variants": list(cd_cfg.get("selected_variants", [])),
    }

    report_path = os.path.join(out_dir, "reports", "report.html")
    build_report(
        out_dir=out_dir,
        metadata_csv=metadata_csv,
        inspect_sheet=inspect_sheet,
        enabled_groups_meta=enabled_meta,
        mask_coverage_csv=mask_coverage_csv,
        fused_compare=fused_compare,
        per_group_fused=per_group_fused,
        line_risk_images=line_risk_images,
        static_compare=static_compare,
        exclusion_compare=exclusion_compare,
        safe_images=safe_images,
        version_blocks=version_blocks,
        stage3=stage3,
        report_path=report_path,
    )
    print(f"[05] Wrote report: {report_path}")
    for vb in version_blocks:
        print(f"[05] {vb['label']} counts: {vb['counts']}")


if __name__ == "__main__":
    main()
