import _bootstrap  # noqa: F401

import argparse
import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.crop_export import export_candidate_crop_sheet
from src.groups import get_enabled_groups, load_config
from src.io_utils import ensure_dir


CATEGORIES = (
    "high_confidence_clean",
    "line_adjacent_uncertain",
    "texture_uncertain",
    "rejected_line_like",
    "rejected_tiny_response",
)


def _load_aligned(out_dir, enabled, reference):
    aligned_dir = os.path.join(out_dir, "aligned")
    images = OrderedDict()
    ref_added = False
    for name, spec in enabled.items():
        if reference in spec.get("files", []):
            p = os.path.join(aligned_dir, name, reference.replace(".tif", ".npy"))
            if os.path.isfile(p) and not ref_added:
                images[reference] = np.load(p).astype(np.float32)
                ref_added = True
                break
    for name, spec in enabled.items():
        for fname in spec.get("files", []):
            if fname == reference and ref_added:
                continue
            label = f"{name}/{fname}"
            p = os.path.join(aligned_dir, name, fname.replace(".tif", ".npy"))
            if os.path.isfile(p):
                images[label] = np.load(p).astype(np.float32)
    return images


def _opt_npy(path):
    return np.load(path).astype(np.float32) if os.path.isfile(path) else None


def _export_version(out_dir, config, version: str):
    suffix = "" if version == "v1" else f"_{version}"
    ce = config.get("crop_export", {}) or {}
    include_fusion = bool(ce.get("include_fusion", True))
    include_line_risk = bool(ce.get("include_line_risk", True))
    include_static = bool(ce.get("include_static_line_mask", True))
    include_excl = bool(ce.get("include_combined_exclusion_mask", True))
    include_safe = bool(ce.get("include_safe_search_mask", True))
    caps = ce.get("per_category_caps", {}) or {}
    cap_map = {c: int(caps.get(c, 200)) for c in CATEGORIES}

    enabled = get_enabled_groups(config)
    reference = config.get("reference_image", "0000.tif")

    csv_path = os.path.join(out_dir, "candidates", f"candidates_all{suffix}.csv")
    if not os.path.isfile(csv_path):
        print(f"[04] {csv_path} missing — skipping version {version}")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[04] candidates_all{suffix}.csv empty — nothing to export.")
        return
    if "category" not in df.columns:
        df["category"] = "uncertain"
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)

    aligned = _load_aligned(out_dir, enabled, reference)

    extras = OrderedDict()
    if include_fusion:
        fused_name = "combined_fused_delined" if version == "v1" else f"combined_fused_delined_{version}"
        v = _opt_npy(os.path.join(out_dir, "fusion", f"{fused_name}.npy"))
        if v is not None:
            extras[f"combined_fused{suffix}"] = v
    if include_line_risk:
        risk_name = "combined_line_risk" if version == "v1" else f"combined_line_risk_{version}"
        v = _opt_npy(os.path.join(out_dir, "line_risk", f"{risk_name}.npy"))
        if v is not None:
            extras[f"dynamic_line_risk{suffix}"] = v
    if include_static:
        v = _opt_npy(os.path.join(out_dir, "line_risk", "refined_static_line_mask.npy"))
        if v is not None:
            extras["refined_static_line_mask"] = v
        v = _opt_npy(os.path.join(out_dir, "line_risk", "static_line_mask.npy"))
        if v is not None:
            extras["raw_static_line_mask"] = v
    if include_excl:
        v = _opt_npy(os.path.join(out_dir, "line_risk", "refined_combined_line_exclusion_dilate8.npy"))
        if v is not None:
            extras["refined_combined_exclusion_dilate8"] = v
    if include_safe:
        v = _opt_npy(os.path.join(out_dir, "invalid_region", "refined_safe_search_mask_dilate8.npy"))
        if v is not None:
            extras["refined_safe_search_dilate8"] = v
        v = _opt_npy(os.path.join(out_dir, "invalid_region", "safe_search_mask.npy"))
        if v is not None:
            extras["safe_search_mask_v1"] = v

    images_dict_template = OrderedDict()
    for k, v in aligned.items():
        images_dict_template[k] = v
    for k, v in extras.items():
        images_dict_template[k] = v

    base_out = os.path.join(out_dir, "candidate_crops")
    ensure_dir(base_out)

    df["crop_path"] = ""
    for category in CATEGORIES:
        sub_df = df[df.category == category].head(cap_map[category])
        sub_dir = os.path.join(base_out, f"{category}{suffix}")
        ensure_dir(sub_dir)
        for _, row in sub_df.iterrows():
            cid = int(row["candidate_id"])
            score = float(row.get("final_score", 0.0))
            x = int(round(float(row["x"])))
            y = int(round(float(row["y"])))
            fname = f"candidate_{cid:04d}_x{x:04d}_y{y:04d}_score{score:.3f}.png"
            out_path = os.path.join(sub_dir, fname)
            export_candidate_crop_sheet(row, images_dict_template, out_path, config)
            df.loc[df["candidate_id"] == cid, "crop_path"] = out_path
        print(f"[04] [{version}] Exported {len(sub_df)} crops -> {sub_dir}")

    # Refresh CSV crop_paths for this version
    df_full = pd.read_csv(csv_path)
    df_full["crop_path"] = ""
    for cid, p in zip(df["candidate_id"], df["crop_path"]):
        if p:
            df_full.loc[df_full["candidate_id"] == cid, "crop_path"] = p
    df_full.to_csv(csv_path, index=False)

    cand_dir = os.path.join(out_dir, "candidates")
    for cat in CATEGORIES:
        out_csv = os.path.join(cand_dir, f"candidates_{cat}{suffix}.csv")
        df_full[df_full.category == cat].to_csv(out_csv, index=False)

    if version == "v1":
        df_full[df_full.category == "high_confidence_clean"].to_csv(
            os.path.join(cand_dir, "candidates_high_confidence.csv"), index=False
        )
        df_full[df_full.category == "line_adjacent_uncertain"].to_csv(
            os.path.join(cand_dir, "candidates_line_risk.csv"), index=False
        )
        df_full[df_full.category.isin(["rejected_line_like", "rejected_tiny_response"])].to_csv(
            os.path.join(cand_dir, "candidates_rejected.csv"), index=False
        )
        df_full.to_csv(os.path.join(cand_dir, "candidates.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    config = load_config(args.config)
    versions = list((config.get("combined_versions", {}) or {"v1": {}}).keys())
    if not versions:
        versions = ["v1"]
    for v in versions:
        _export_version(args.out_dir, config, v)


if __name__ == "__main__":
    main()
