import _bootstrap  # noqa: F401

import argparse
import os
import subprocess
import sys
import time


STEPS = [
    ("00_inspect_dataset.py", ["--raw_dir", "{raw_dir}", "--config", "{config}", "--out_dir", "{out_dir}"]),
    ("01_register_groups.py", ["--raw_dir", "{raw_dir}", "--config", "{config}", "--out_dir", "{out_dir}"]),
    ("02_fuse_delined.py", ["--raw_dir", "{raw_dir}", "--config", "{config}", "--out_dir", "{out_dir}"]),
    ("03_detect_candidates.py", ["--config", "{config}", "--out_dir", "{out_dir}"]),
    ("04_export_candidate_crops.py", ["--config", "{config}", "--out_dir", "{out_dir}"]),
    ("06_rank_manual_review_candidates.py", ["--config", "{config}", "--out_dir", "{out_dir}"]),
    ("07_line_removal_baseline.py", ["--raw_dir", "{raw_dir}", "--config", "{config}", "--out_dir", "{out_dir}"]),
    ("08_detect_candidates_on_cleaned.py", ["--config", "{config}", "--out_dir", "{out_dir}"]),
    ("05_make_report.py", ["--config", "{config}", "--out_dir", "{out_dir}"]),
]

KEY_OUTPUTS = {
    "00_inspect_dataset.py": [
        "{out_dir}/inspect/metadata.csv",
        "{out_dir}/inspect/contact_sheet.png",
    ],
    "01_register_groups.py": [
        "{out_dir}/registration_qc/axis2_x_transforms.csv",
        "{out_dir}/registration_qc/axis4_y_transforms.csv",
        "{out_dir}/registration_qc/axis1_angle_transforms.csv",
        "{out_dir}/registration_qc/axis1_angle_aligned_contact_sheet.png",
        "{out_dir}/registration_qc/axis1_angle_diff_before.png",
        "{out_dir}/registration_qc/axis1_angle_diff_after.png",
        "{out_dir}/registration_qc/axis2_x_aligned_std.png",
        "{out_dir}/registration_qc/axis1_angle_aligned_std.png",
    ],
    "02_fuse_delined.py": [
        "{out_dir}/fusion/combined_fused_delined.png",
        "{out_dir}/fusion/combined_fused_delined_v2.png",
        "{out_dir}/fusion/axis1_angle_fused_median.png",
        "{out_dir}/line_risk/combined_line_risk.png",
        "{out_dir}/line_risk/combined_line_risk_v2.png",
        "{out_dir}/line_risk/axis1_angle_dynamic_line_mask.png",
        "{out_dir}/line_risk/axis1_angle_mad.png",
        "{out_dir}/line_risk/refined_static_line_response.png",
        "{out_dir}/line_risk/refined_static_line_components.png",
        "{out_dir}/line_risk/refined_static_line_mask.png",
        "{out_dir}/line_risk/refined_static_line_skeleton.png",
        "{out_dir}/line_risk/refined_static_line_distance.png",
        "{out_dir}/line_risk/refined_combined_line_exclusion_dilate6.png",
        "{out_dir}/line_risk/refined_combined_line_exclusion_dilate8.png",
        "{out_dir}/line_risk/refined_combined_line_exclusion_dilate10.png",
        "{out_dir}/line_risk/refined_combined_line_exclusion_distance.png",
        "{out_dir}/line_risk/refined_combined_line_exclusion_distance.npy",
        "{out_dir}/invalid_region/refined_safe_search_mask_dilate6.png",
        "{out_dir}/invalid_region/refined_safe_search_mask_dilate8.png",
        "{out_dir}/invalid_region/refined_safe_search_mask_dilate10.png",
        "{out_dir}/reports/mask_coverage.csv",
    ],
    "03_detect_candidates.py": [
        "{out_dir}/candidates/candidates_all.csv",
        "{out_dir}/candidates/candidates_all_v2.csv",
        "{out_dir}/candidates/candidates_high_confidence_clean.csv",
        "{out_dir}/candidates/candidates_high_confidence_clean_v2.csv",
        "{out_dir}/candidates/candidates_overlay_high_confidence_clean.png",
        "{out_dir}/candidates/candidates_overlay_high_confidence_clean_v2.png",
        "{out_dir}/candidates/sweeps/conservative_high_confidence.csv",
        "{out_dir}/candidates/sweeps/medium_high_confidence.csv",
        "{out_dir}/candidates/sweeps/loose_high_confidence.csv",
        "{out_dir}/candidates/sweeps/ultra_loose_sanity_check_high_confidence.csv",
        "{out_dir}/candidates/sweeps/ultra_loose_sanity_check_overlay.png",
        "{out_dir}/candidates/sweeps/conservative_high_confidence_v2.csv",
        "{out_dir}/candidates/sweeps/medium_high_confidence_v2.csv",
        "{out_dir}/candidates/sweeps/loose_high_confidence_v2.csv",
        "{out_dir}/candidates/sweeps/ultra_loose_sanity_check_high_confidence_v2.csv",
    ],
    "04_export_candidate_crops.py": [
        "{out_dir}/candidate_crops/high_confidence_clean",
        "{out_dir}/candidate_crops/line_adjacent_uncertain",
        "{out_dir}/candidate_crops/texture_uncertain",
        "{out_dir}/candidate_crops/high_confidence_clean_v2",
        "{out_dir}/candidate_crops/line_adjacent_uncertain_v2",
        "{out_dir}/candidate_crops/texture_uncertain_v2",
    ],
    "06_rank_manual_review_candidates.py": [
        "{out_dir}/manual_review/manual_review_candidates_all.csv",
        "{out_dir}/manual_review/manual_review_top50.csv",
        "{out_dir}/manual_review/manual_review_top100.csv",
        "{out_dir}/manual_review/manual_review_overlay_top50.png",
        "{out_dir}/manual_review/manual_review_overlay_top100.png",
        "{out_dir}/manual_review/review_index.html",
        "{out_dir}/manual_review/manual_labels_template.csv",
        "{out_dir}/manual_review/crops_top50",
        "{out_dir}/manual_review/crops_top100",
    ],
    "07_line_removal_baseline.py": [
        "{out_dir}/line_removal/masks/line_mask_radius2.png",
        "{out_dir}/line_removal/masks/line_mask_radius4.png",
        "{out_dir}/line_removal/masks/line_mask_radius6.png",
        "{out_dir}/line_removal/masks/defect_protect_mask.png",
        "{out_dir}/line_removal/masks/line_removal_mask_coverage.csv",
        "{out_dir}/line_removal/cleaned_images",
        "{out_dir}/line_removal/comparison_sheets",
    ],
    "08_detect_candidates_on_cleaned.py": [
        "{out_dir}/line_removal/candidates_on_cleaned/summary.csv",
    ],
    "05_make_report.py": ["{out_dir}/reports/report.html"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="raw")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    subs = {"raw_dir": args.raw_dir, "config": args.config, "out_dir": args.out_dir}

    for script, argv_template in STEPS:
        script_path = os.path.join(scripts_dir, script)
        cmd = [sys.executable, script_path] + [a.format(**subs) for a in argv_template]
        print(f"\n=== Running {script} ===")
        print("  $", " ".join(cmd))
        t0 = time.time()
        result = subprocess.run(cmd, cwd=os.path.dirname(scripts_dir))
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"[run_stage1] FAILED at {script} (rc={result.returncode}, {elapsed:.1f}s)")
            sys.exit(result.returncode)
        print(f"[run_stage1] {script} OK ({elapsed:.1f}s)")
        for key in KEY_OUTPUTS.get(script, []):
            p = key.format(**subs)
            mark = "OK" if os.path.exists(p) else "MISSING"
            print(f"   - {mark}: {p}")

    print("\n[run_stage1] Stage 1.3 + Stage 2 finished.")


if __name__ == "__main__":
    main()
