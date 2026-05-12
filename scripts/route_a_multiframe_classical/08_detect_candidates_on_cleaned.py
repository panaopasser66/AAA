"""Stage 3B — blob detection on selected cleaned variants.

Spec: task_stage3.md §12.

For each variant in `cleaned_candidate_detection.selected_variants`, runs LoG
detection at 3–7 px, applies the soft shape filters (no big exclusion gate),
and exports candidates.csv + overlay.png + candidate_crops/.
"""

import _bootstrap  # noqa: F401

import argparse
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import cv2
import numpy as np
import pandas as pd
from skimage.feature import blob_log

from src.candidate_scoring import (
    circularity_and_aspect_from_patch,
    local_contrast_score,
)
from src.groups import get_enabled_groups, load_config
from src.io_utils import ensure_dir, robust_rescale_for_preview
from src.structure_tensor import anisotropy_at


@dataclass
class CleanedCand:
    candidate_id: int
    x: float
    y: float
    radius: float
    diameter_px: float
    polarity: str
    log_response: Optional[float]


# Reuse Stage 3A tile order (axis1 frames included)
TILE_ORDER = [
    "0000.tif",
    "axis2_x/0-100.tif", "axis2_x/0100.tif",
    "axis4_y/000-1.tif", "axis4_y/0001.tif",
    "axis1_angle/-2000.tif", "axis1_angle/-1000.tif",
    "axis1_angle/1000.tif", "axis1_angle/2000.tif",
    "combined_fused_delined",
    "combined_fused_delined_v2",
    "combined_line_risk_v2",
    "refined_static_line_mask",
    "refined_combined_line_exclusion",
    "defect_protect_mask",
]


def _to01(img):
    a = np.asarray(img, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-12:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)


def _detect(img: np.ndarray, cfg: dict) -> List[CleanedCand]:
    min_d = float(cfg.get("min_diameter_px", 3))
    max_d = float(cfg.get("max_diameter_px", 7))
    min_sigma = float(cfg.get("log_sigma_min", 1.0))
    max_sigma = float(cfg.get("log_sigma_max", 2.6))
    num_sigma = int(cfg.get("num_sigma", 8))
    threshold_abs = float(cfg.get("threshold_abs", 0.04))
    overlap = float(cfg.get("overlap", 0.5))
    exclude_border = int(cfg.get("exclude_border_px", 32))
    nms_d = float(cfg.get("nms_distance_px", 4))

    H, W = img.shape[:2]
    arr = _to01(img)
    rows: List[CleanedCand] = []
    for polarity, src in (("bright", arr), ("dark", 1.0 - arr)):
        blobs = blob_log(src, min_sigma=min_sigma, max_sigma=max_sigma,
                         num_sigma=num_sigma, threshold=threshold_abs, overlap=overlap)
        for b in blobs:
            y, x, sigma = float(b[0]), float(b[1]), float(b[2])
            radius = sigma * np.sqrt(2.0)
            d = 2.0 * radius
            if d < min_d or d > max_d:
                continue
            if (x < exclude_border or y < exclude_border
                    or x >= W - exclude_border or y >= H - exclude_border):
                continue
            ix, iy = int(round(x)), int(round(y))
            ix = max(0, min(W - 1, ix)); iy = max(0, min(H - 1, iy))
            rows.append(CleanedCand(
                candidate_id=-1, x=x, y=y, radius=radius, diameter_px=d,
                polarity=polarity, log_response=float(src[iy, ix]),
            ))
    rows.sort(key=lambda c: -(c.log_response or 0.0))
    kept: List[CleanedCand] = []
    d2 = nms_d * nms_d
    for r in rows:
        ok = True
        for k in kept:
            if (r.x - k.x) ** 2 + (r.y - k.y) ** 2 < d2:
                ok = False
                break
        if ok:
            kept.append(r)
    for i, r in enumerate(kept):
        r.candidate_id = i + 1
    return kept


def _score(
    cands: List[CleanedCand], cleaned: np.ndarray, base: np.ndarray,
    invalid: np.ndarray, refined_static_dist: np.ndarray,
    refined_combined_dist: np.ndarray, protect_mask: Optional[np.ndarray],
    cfg: dict,
) -> pd.DataFrame:
    r_center = 4.0
    r_in = 7.0
    r_out = 12.0
    H, W = cleaned.shape

    rows = []
    for c in cands:
        ix = int(round(c.x)); iy = int(round(c.y))
        ix = max(0, min(W - 1, ix)); iy = max(0, min(H - 1, iy))
        in_invalid = bool(invalid[iy, ix])
        contrast_cleaned = local_contrast_score(cleaned, c.x, c.y, r_center, r_in, r_out, c.polarity)
        contrast_base = local_contrast_score(base, c.x, c.y, r_center, r_in, r_out, c.polarity)
        shape = circularity_and_aspect_from_patch(cleaned, c.x, c.y, c.radius, c.polarity)
        aniso = anisotropy_at(cleaned, c.x, c.y, 8, 1.5)
        d_stat = float(refined_static_dist[iy, ix]) if refined_static_dist is not None else float("nan")
        d_comb = float(refined_combined_dist[iy, ix]) if refined_combined_dist is not None else float("nan")
        in_protect = bool(protect_mask[iy, ix]) if protect_mask is not None else False

        rows.append({
            "candidate_id": c.candidate_id,
            "x": c.x, "y": c.y,
            "radius_px": c.radius, "diameter_px": c.diameter_px,
            "polarity": c.polarity, "log_response": c.log_response,
            "local_contrast_cleaned": float(contrast_cleaned),
            "local_contrast_base": float(contrast_base),
            "contrast_gain_vs_base": float(contrast_cleaned - contrast_base),
            "area_px": shape["area_px"],
            "circularity": shape["circularity"],
            "aspect_ratio": shape["aspect_ratio"],
            "anisotropy": float(aniso) if np.isfinite(aniso) else float("nan"),
            "distance_to_refined_static_line_skeleton": d_stat,
            "distance_to_refined_combined_line_exclusion": d_comb,
            "in_invalid_region": int(in_invalid),
            "in_or_near_protect_mask": int(in_protect),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        df["passes_shape_filter"] = []
        df["cleaned_score"] = []
        return df

    # Spec §12: shape filter (do not use big exclusion gate)
    min_a = float(cfg.get("min_area_px", 10))
    max_a = float(cfg.get("max_area_px", 80))
    min_c = float(cfg.get("min_circularity", 0.5))
    max_c = float(cfg.get("max_circularity", 1.4))
    max_aspect = float(cfg.get("max_aspect_ratio", 2.0))
    max_aniso = float(cfg.get("max_anisotropy", 3.0))

    passes = (
        (df["area_px"].between(min_a, max_a))
        & (df["circularity"].between(min_c, max_c))
        & (df["aspect_ratio"].fillna(99) <= max_aspect)
        & (df["anisotropy"].fillna(99) <= max_aniso)
        & (df["in_invalid_region"] == 0)
    )
    df["passes_shape_filter"] = passes.astype(int)

    # Score: contrast + circularity bonus + protect-mask bonus - aniso/aspect penalties
    contrast = df["local_contrast_cleaned"].clip(lower=0).to_numpy()
    contrast_norm = (
        np.clip(contrast / max(np.percentile(contrast, 95), 1e-9), 0, 1)
        if contrast.size > 5
        else contrast
    )
    circ = df["circularity"].fillna(0.0).clip(0.0, 1.0).to_numpy()
    aniso = df["anisotropy"].fillna(99.0).to_numpy()
    aniso_pen = np.clip((aniso - 1.0) / 9.0, 0.0, 1.0)
    aspect = df["aspect_ratio"].fillna(99.0).to_numpy()
    aspect_pen = np.clip((aspect - 1.0) / 4.0, 0.0, 1.0)
    invalid_arr = df["in_invalid_region"].fillna(0).astype(float).to_numpy()
    protect = df["in_or_near_protect_mask"].fillna(0).astype(float).to_numpy()

    df["cleaned_score"] = (
        + 1.0 * contrast_norm
        + 0.5 * circ
        + 0.4 * protect
        - 0.4 * aniso_pen
        - 0.4 * aspect_pen
        - 1.5 * invalid_arr
    ).astype(np.float32)

    df = df.sort_values(["passes_shape_filter", "cleaned_score"], ascending=[False, False])
    df = df.reset_index(drop=True)

    max_n = int(cfg.get("max_candidates_per_variant", 300))
    if len(df) > max_n:
        df = df.head(max_n)
    return df


def _draw_overlay(img: np.ndarray, df: pd.DataFrame, out_path: str, max_label: int = 200):
    prev = robust_rescale_for_preview(img)
    bgr = cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR)
    for i, row in df.iterrows():
        x, y = int(round(row["x"])), int(round(row["y"]))
        r = max(2, int(round(float(row.get("radius_px", 3.0)) + 2)))
        passes = int(row.get("passes_shape_filter", 0)) == 1
        color = (0, 255, 80) if passes else (60, 60, 255)
        cv2.circle(bgr, (x, y), r, color, 1, cv2.LINE_AA)
        if i < max_label:
            cv2.putText(bgr, str(int(row["candidate_id"])),
                        (x + r + 1, y - r - 1), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, color, 1, cv2.LINE_AA)
    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, bgr)


def _crop_around(img: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    H, W = img.shape[:2]
    half = size // 2
    x0 = int(round(cx - half)); y0 = int(round(cy - half))
    x1 = x0 + size; y1 = y0 + size
    pad_left = max(0, -x0); pad_top = max(0, -y0)
    pad_right = max(0, x1 - W); pad_bottom = max(0, y1 - H)
    x0c = max(0, x0); y0c = max(0, y0)
    x1c = min(W, x1); y1c = min(H, y1)
    patch = img[y0c:y1c, x0c:x1c]
    if pad_left or pad_top or pad_right or pad_bottom:
        patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
    return patch


def _label(img_bgr, title):
    h, w = img_bgr.shape[:2]
    out = np.full((h + 16, w, 3), 32, dtype=np.uint8)
    src = img_bgr if img_bgr.ndim == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    out[16:, :, :] = src
    cv2.putText(out, title, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _crop_sheet(row: dict, images: Dict[str, np.ndarray], size: int, out_path: str):
    x = float(row["x"]); y = float(row["y"])
    radius = float(row.get("radius_px", 3.0))
    tiles = []
    for key in TILE_ORDER:
        img = images.get(key)
        if img is None:
            continue
        crop = _crop_around(img, x, y, size)
        prev = robust_rescale_for_preview(crop)
        bgr = cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR)
        cx, cy = bgr.shape[1] // 2, bgr.shape[0] // 2
        cv2.circle(bgr, (cx, cy), max(2, int(round(radius))), (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(_label(bgr, key))
    if not tiles:
        return
    cols = 5
    rows = (len(tiles) + cols - 1) // cols
    cell_h, cell_w = tiles[0].shape[:2]
    pad = 4
    sheet = np.full((rows * cell_h + pad * (rows + 1),
                     cols * cell_w + pad * (cols + 1), 3), 16, dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y0 = pad + r * (cell_h + pad)
        x0 = pad + c * (cell_w + pad)
        sheet[y0:y0 + cell_h, x0:x0 + cell_w, :] = t
    cv2.imwrite(out_path, sheet)


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


def _opt_npy(p):
    return np.load(p).astype(np.float32) if os.path.isfile(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    config = load_config(args.config)
    cfg = config.get("cleaned_candidate_detection", {}) or {}
    if not bool(cfg.get("enabled", True)):
        print("[08] cleaned_candidate_detection.enabled=false; skipping.")
        return
    selected = list(cfg.get("selected_variants", []))
    crop_size = int(cfg.get("crop_size", 96))
    crops_top_n = int(cfg.get("crops_top_n", 100))

    cleaned_dir = os.path.join(args.out_dir, "line_removal", "cleaned_images")
    if not os.path.isdir(cleaned_dir):
        raise FileNotFoundError(f"{cleaned_dir} missing — run 07_line_removal_baseline.py first.")

    out_root = os.path.join(args.out_dir, "line_removal", "candidates_on_cleaned")
    ensure_dir(out_root)

    # base reference image for contrast_gain comparison
    base_img = _opt_npy(os.path.join(args.out_dir, "fusion", "combined_fused_delined_v2.npy"))
    if base_img is None:
        base_img = _opt_npy(os.path.join(args.out_dir, "fusion", "combined_fused_delined.npy"))
    if base_img is None:
        raise FileNotFoundError("Need a fused base image (v1 or v2).")

    invalid = np.load(os.path.join(args.out_dir, "invalid_region", "invalid_region_mask.npy")).astype(np.uint8)
    refined_static_dist = _opt_npy(os.path.join(args.out_dir, "line_risk", "refined_static_line_distance.npy"))
    refined_combined_dist = _opt_npy(os.path.join(args.out_dir, "line_risk", "refined_combined_line_exclusion_distance.npy"))
    protect_mask = _opt_npy(os.path.join(args.out_dir, "line_removal", "masks", "defect_protect_mask.npy"))

    enabled = get_enabled_groups(config)
    reference = config.get("reference_image", "0000.tif")
    aligned = _load_aligned(args.out_dir, enabled, reference)

    extras = OrderedDict()
    for name, p in [
        ("combined_fused_delined", os.path.join(args.out_dir, "fusion", "combined_fused_delined.npy")),
        ("combined_fused_delined_v2", os.path.join(args.out_dir, "fusion", "combined_fused_delined_v2.npy")),
        ("combined_line_risk_v2", os.path.join(args.out_dir, "line_risk", "combined_line_risk_v2.npy")),
        ("refined_static_line_mask", os.path.join(args.out_dir, "line_risk", "refined_static_line_mask.npy")),
        ("refined_combined_line_exclusion", os.path.join(args.out_dir, "line_risk", "refined_combined_line_exclusion_dilate8.npy")),
        ("defect_protect_mask", os.path.join(args.out_dir, "line_removal", "masks", "defect_protect_mask.npy")),
    ]:
        v = _opt_npy(p)
        if v is not None:
            extras[name] = v

    summary_rows = []
    for variant_name in selected:
        var_npy = os.path.join(cleaned_dir, f"{variant_name}.npy")
        if not os.path.isfile(var_npy):
            print(f"[08] WARN: variant {variant_name} npy missing — skipping")
            summary_rows.append({
                "variant": variant_name, "status": "missing",
                "total": 0, "passes_shape_filter": 0, "outside_invalid": 0,
            })
            continue
        cleaned = np.load(var_npy).astype(np.float32)
        print(f"[08] Detecting on {variant_name} ...")
        cands = _detect(cleaned, cfg)
        df = _score(cands, cleaned, base_img, invalid,
                    refined_static_dist, refined_combined_dist, protect_mask, cfg)

        var_dir = os.path.join(out_root, variant_name)
        ensure_dir(var_dir)
        df.to_csv(os.path.join(var_dir, "candidates.csv"), index=False)
        _draw_overlay(cleaned, df, os.path.join(var_dir, "overlay.png"))

        n_total = len(df)
        n_pass = int((df["passes_shape_filter"] == 1).sum()) if not df.empty else 0
        n_outside = int((df["in_invalid_region"] == 0).sum()) if not df.empty else 0
        summary_rows.append({
            "variant": variant_name, "status": "ok",
            "total": n_total,
            "passes_shape_filter": n_pass,
            "outside_invalid": n_outside,
        })
        print(f"   - {n_total} candidates ({n_pass} pass shape filter, {n_outside} outside invalid)")

        # crop sheets for top-N
        if df.empty:
            continue
        images_template = OrderedDict()
        for k, v in aligned.items():
            images_template[k] = v
        images_template["combined_fused_delined"] = extras.get("combined_fused_delined")
        images_template["combined_fused_delined_v2"] = extras.get("combined_fused_delined_v2")
        images_template["combined_line_risk_v2"] = extras.get("combined_line_risk_v2")
        images_template["refined_static_line_mask"] = extras.get("refined_static_line_mask")
        images_template["refined_combined_line_exclusion"] = extras.get("refined_combined_line_exclusion")
        images_template["defect_protect_mask"] = extras.get("defect_protect_mask")
        images_template[variant_name] = cleaned

        crops_dir = os.path.join(var_dir, "candidate_crops")
        ensure_dir(crops_dir)
        for _, row in df.head(crops_top_n).iterrows():
            cid = int(row["candidate_id"])
            score = float(row.get("cleaned_score", 0.0))
            x = int(round(float(row["x"])))
            y = int(round(float(row["y"])))
            fname = f"candidate_{cid:04d}_x{x:04d}_y{y:04d}_score{score:.3f}.png"
            _crop_sheet(row.to_dict(), images_template, crop_size, os.path.join(crops_dir, fname))

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(out_root, "summary.csv"), index=False,
    )
    print(f"[08] Wrote summary.csv with {len(summary_rows)} variants")


if __name__ == "__main__":
    main()
