from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

from .blob_detection import Candidate
from .structure_tensor import anisotropy_at, mean_value_in_disk


# ---------------------------------------------------------------------------
# Local feature helpers
# ---------------------------------------------------------------------------

def _circular_mask(h: int, w: int, cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _annulus_mask(h: int, w: int, cx: float, cy: float, r_in: float, r_out: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return (d2 >= r_in * r_in) & (d2 <= r_out * r_out)


def local_contrast_score(
    img: np.ndarray, x: float, y: float, r_center: float, r_inner: float,
    r_outer: float, polarity: str = "bright",
) -> float:
    H, W = img.shape[:2]
    pad = int(np.ceil(r_outer)) + 2
    x0 = max(0, int(x - pad)); x1 = min(W, int(x + pad + 1))
    y0 = max(0, int(y - pad)); y1 = min(H, int(y + pad + 1))
    patch = img[y0:y1, x0:x1]
    cx, cy = x - x0, y - y0
    h, w = patch.shape
    if h <= 0 or w <= 0:
        return 0.0
    cm = _circular_mask(h, w, cx, cy, r_center)
    am = _annulus_mask(h, w, cx, cy, r_inner, r_outer)
    if cm.sum() == 0 or am.sum() == 0:
        return 0.0
    c_med = float(np.median(patch[cm]))
    a_med = float(np.median(patch[am]))
    return (a_med - c_med) if polarity == "dark" else (c_med - a_med)


def circularity_and_aspect_from_patch(
    img: np.ndarray, x: float, y: float, radius: float, polarity: str
) -> Dict[str, float]:
    H, W = img.shape[:2]
    pad = int(max(8, np.ceil(radius * 4)))
    x0 = max(0, int(x - pad)); x1 = min(W, int(x + pad + 1))
    y0 = max(0, int(y - pad)); y1 = min(H, int(y + pad + 1))
    patch = img[y0:y1, x0:x1]
    if patch.size == 0:
        return {"area_px": float("nan"), "circularity": float("nan"), "aspect_ratio": float("nan")}
    p = patch.astype(np.float32)
    med = float(np.median(p))
    mad = float(np.median(np.abs(p - med))) + 1e-6
    if polarity == "bright":
        binary = (p > med + 1.5 * mad).astype(np.uint8)
    else:
        binary = (p < med - 1.5 * mad).astype(np.uint8)
    if binary.sum() == 0:
        return {"area_px": 0.0, "circularity": float("nan"), "aspect_ratio": float("nan")}
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return {"area_px": 0.0, "circularity": float("nan"), "aspect_ratio": float("nan")}
    cx, cy = x - x0, y - y0
    best_idx = -1; best_d = 1e18
    for i in range(1, num):
        sx, sy, sw, sh = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                          stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        ccx = sx + sw / 2.0; ccy = sy + sh / 2.0
        d = (ccx - cx) ** 2 + (ccy - cy) ** 2
        if d < best_d:
            best_d = d; best_idx = i
    if best_idx < 0:
        return {"area_px": 0.0, "circularity": float("nan"), "aspect_ratio": float("nan")}
    mask = (labels == best_idx).astype(np.uint8)
    area = float(mask.sum())
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"area_px": area, "circularity": float("nan"), "aspect_ratio": float("nan")}
    cnt = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(cnt, True))
    circ = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 1e-6 else float("nan")
    if len(cnt) >= 5:
        (_, (w_, h_), _) = cv2.fitEllipse(cnt)
        aspect = max(w_, h_) / max(1e-6, min(w_, h_))
    else:
        sw = stats[best_idx, cv2.CC_STAT_WIDTH]
        sh = stats[best_idx, cv2.CC_STAT_HEIGHT]
        aspect = max(sw, sh) / max(1.0, min(sw, sh))
    return {"area_px": area, "circularity": circ, "aspect_ratio": float(aspect)}


def _safe_index(img: np.ndarray, x: float, y: float) -> float:
    H, W = img.shape[:2]
    ix = int(round(x)); iy = int(round(y))
    ix = max(0, min(W - 1, ix)); iy = max(0, min(H - 1, iy))
    return float(img[iy, ix])


# ---------------------------------------------------------------------------
# Score candidates
# ---------------------------------------------------------------------------

def score_candidates(
    candidates: List[Candidate],
    fused_img: np.ndarray,
    line_risk: np.ndarray,
    dynamic_line_distance: np.ndarray,
    static_line_distance: np.ndarray,
    combined_exclusion_distance: np.ndarray,
    invalid_region: np.ndarray,
    aligned_images: Dict[str, np.ndarray],
    aligned_valid_masks: Dict[str, np.ndarray],
    config: dict,
    refined_static_line_distance: np.ndarray = None,
    refined_combined_exclusion_distance: np.ndarray = None,
) -> pd.DataFrame:
    """Compute Stage-1.2 feature set per candidate. Returns sorted DataFrame."""
    feat_cfg = config.get("candidate_features", {}) or {}
    st_cfg = config.get("structure_tensor", {}) or {}

    r_center = float(feat_cfg.get("crop_radius_for_contrast_px", 5))
    r_in = float(feat_cfg.get("annulus_inner_radius_px", 8))
    r_out = float(feat_cfg.get("annulus_outer_radius_px", 14))
    lr_patch_r = float(feat_cfg.get("patch_radius_for_line_risk_px", 8))
    per_frame_aniso_thr = float(feat_cfg.get("per_frame_anisotropy_threshold", 6.0))
    per_frame_spot_thr = float(feat_cfg.get("per_frame_spot_contrast_threshold", 0.015))

    st_sigma = float(st_cfg.get("sigma", 1.5))
    st_patch_r = int(st_cfg.get("patch_radius_px", 8))

    H, W = fused_img.shape[:2]

    rows = []
    for c in candidates:
        iy = int(round(min(H - 1, max(0, c.y))))
        ix = int(round(min(W - 1, max(0, c.x))))
        in_invalid = bool(invalid_region[iy, ix])

        lc = local_contrast_score(fused_img, c.x, c.y, r_center, r_in, r_out, c.polarity)
        shape = circularity_and_aspect_from_patch(fused_img, c.x, c.y, c.radius, c.polarity)
        aniso = anisotropy_at(fused_img, c.x, c.y, st_patch_r, st_sigma)

        line_risk_at_center = _safe_index(line_risk, c.x, c.y)
        mean_lr = mean_value_in_disk(line_risk, c.x, c.y, lr_patch_r)
        dist_dyn = _safe_index(dynamic_line_distance, c.x, c.y)
        dist_stat = _safe_index(static_line_distance, c.x, c.y)
        dist_comb = _safe_index(combined_exclusion_distance, c.x, c.y)
        dist_refined_stat = (
            _safe_index(refined_static_line_distance, c.x, c.y)
            if refined_static_line_distance is not None else float("nan")
        )
        dist_refined_comb = (
            _safe_index(refined_combined_exclusion_distance, c.x, c.y)
            if refined_combined_exclusion_distance is not None else float("nan")
        )

        per_frame_aniso: List[float] = []
        per_frame_contrast: List[float] = []
        per_frame_valid: List[int] = []
        for label, frame in aligned_images.items():
            vmask = aligned_valid_masks.get(label, None)
            is_valid = True
            if vmask is not None:
                iy2 = max(0, min(vmask.shape[0] - 1, int(round(c.y))))
                ix2 = max(0, min(vmask.shape[1] - 1, int(round(c.x))))
                is_valid = bool(vmask[iy2, ix2])
            per_frame_valid.append(int(is_valid))
            if is_valid:
                a = anisotropy_at(frame, c.x, c.y, st_patch_r, st_sigma)
                ct = local_contrast_score(frame, c.x, c.y, r_center, r_in, r_out, c.polarity)
            else:
                a = float("nan"); ct = float("nan")
            per_frame_aniso.append(a); per_frame_contrast.append(ct)

        valid_frame_count = int(sum(per_frame_valid))
        line_covered = 0
        uncovered_spot_present = 0
        for v, a, ct in zip(per_frame_valid, per_frame_aniso, per_frame_contrast):
            if not v:
                continue
            covered = np.isfinite(a) and a > per_frame_aniso_thr
            if covered:
                line_covered += 1
            elif np.isfinite(ct) and ct >= per_frame_spot_thr:
                uncovered_spot_present += 1
        uncovered_frame_count = valid_frame_count - line_covered

        border_distance = float(min(c.x, c.y, W - 1 - c.x, H - 1 - c.y))

        rows.append({
            "candidate_id": c.candidate_id,
            "x": c.x, "y": c.y,
            "radius_px": c.radius, "diameter_px": c.diameter_px,
            "polarity": c.polarity,
            "log_response": c.log_response,
            "area_px": shape["area_px"],
            "circularity": shape["circularity"],
            "aspect_ratio": shape["aspect_ratio"],
            "local_contrast": float(lc),
            "anisotropy": float(aniso) if np.isfinite(aniso) else float("nan"),
            "line_risk_at_center": float(line_risk_at_center),
            "mean_line_risk_in_patch": float(mean_lr) if np.isfinite(mean_lr) else float("nan"),
            "distance_to_dynamic_line_skeleton": float(dist_dyn),
            "distance_to_static_line_skeleton": float(dist_stat),
            "distance_to_combined_line_exclusion": float(dist_comb),
            "distance_to_refined_static_line_skeleton": float(dist_refined_stat),
            "distance_to_refined_combined_line_exclusion": float(dist_refined_comb),
            "valid_frame_count": valid_frame_count,
            "line_covered_frame_count": int(line_covered),
            "uncovered_frame_count": int(uncovered_frame_count),
            "spot_present_when_uncovered": int(uncovered_spot_present),
            "border_distance": border_distance,
            "in_invalid_region": int(in_invalid),
            "per_frame_anisotropy": [float(v) for v in per_frame_aniso],
            "per_frame_contrast": [float(v) for v in per_frame_contrast],
            "per_frame_valid": list(per_frame_valid),
            "crop_path": "",
        })

    df = pd.DataFrame(rows)
    if df.empty:
        df["final_score"] = []
        return df

    # ----- final_score with explicit rewards + penalties -----
    contrast_arr = df["local_contrast"].clip(lower=0).to_numpy()
    if contrast_arr.size and contrast_arr.max() > 1e-9:
        contrast_norm = np.clip(
            contrast_arr / np.percentile(contrast_arr, 95.0) if contrast_arr.size > 5 else contrast_arr / contrast_arr.max(),
            0, 1,
        )
    else:
        contrast_norm = np.zeros_like(contrast_arr)

    circ_raw = df["circularity"].to_numpy()
    circ_valid = np.isfinite(circ_raw)
    circ_clipped = np.where(circ_valid, np.clip(circ_raw, 0.0, 1.0), 0.0)
    circ_invalid_mask = (~circ_valid).astype(np.float32)

    area_arr = df["area_px"].to_numpy()
    area_too_small = (np.where(np.isnan(area_arr), 0.0, area_arr) < 10.0).astype(np.float32)

    aniso_arr = df["anisotropy"].to_numpy()
    aniso_norm = np.where(
        np.isfinite(aniso_arr),
        np.clip((aniso_arr - 1.0) / 9.0, 0.0, 1.0),
        0.5,
    )

    aspect_arr = df["aspect_ratio"].to_numpy()
    aspect_norm = np.where(
        np.isfinite(aspect_arr),
        np.clip((aspect_arr - 1.0) / 4.0, 0.0, 1.0),
        1.0,
    )

    mean_lr_arr = df["mean_line_risk_in_patch"].fillna(0.0).clip(lower=0.0, upper=1.0).to_numpy()
    dist_stat_arr = df["distance_to_static_line_skeleton"].to_numpy()
    dist_comb_arr = df["distance_to_combined_line_exclusion"].to_numpy()
    dist_stat_pen = np.exp(-dist_stat_arr / 8.0).astype(np.float32)
    dist_comb_pen = np.exp(-dist_comb_arr / 8.0).astype(np.float32)
    # Refined-distance penalties: missing → 0 (no bias) until features are present
    if "distance_to_refined_static_line_skeleton" in df.columns:
        rs = df["distance_to_refined_static_line_skeleton"].fillna(distance_clip := 60).to_numpy()
        dist_refined_stat_pen = np.exp(-rs / 8.0).astype(np.float32)
    else:
        dist_refined_stat_pen = np.zeros_like(dist_stat_pen)
    if "distance_to_refined_combined_line_exclusion" in df.columns:
        rc = df["distance_to_refined_combined_line_exclusion"].fillna(60).to_numpy()
        dist_refined_comb_pen = np.exp(-rc / 8.0).astype(np.float32)
    else:
        dist_refined_comb_pen = np.zeros_like(dist_comb_pen)

    invalid_arr = df["in_invalid_region"].to_numpy().astype(np.float32)

    valid_cnt = df["valid_frame_count"].to_numpy().astype(np.float32)
    uncov = df["uncovered_frame_count"].to_numpy().astype(np.float32)
    spot = df["spot_present_when_uncovered"].to_numpy().astype(np.float32)
    spot_ratio = np.where(uncov > 0, spot / np.maximum(uncov, 1.0), 0.0)
    uncov_ratio = np.where(valid_cnt > 0, uncov / np.maximum(valid_cnt, 1.0), 0.0)

    final = (
        + 1.0 * spot_ratio
        + 0.5 * uncov_ratio
        + 0.4 * contrast_norm
        + 0.4 * circ_clipped
        - 0.5 * aniso_norm
        - 0.5 * aspect_norm
        - 0.5 * dist_stat_pen
        - 0.5 * dist_comb_pen
        - 1.0 * dist_refined_stat_pen
        - 1.0 * dist_refined_comb_pen
        - 0.6 * area_too_small
        - 1.0 * circ_invalid_mask
        - 0.8 * mean_lr_arr
        - 1.5 * invalid_arr
    )
    df["final_score"] = final.astype(np.float32)
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Categorisation — Stage 1.2 (five categories)
# ---------------------------------------------------------------------------

def _check_high_confidence_clean(r, hc) -> bool:
    """Strict HC-clean test. Stage 1.3 uses refined-distance fields when
    present; legacy keys are still honoured if they appear in the config.
    """
    d = float(r.get("diameter_px", 0))
    area = float(r.get("area_px", 0)) if pd.notna(r.get("area_px", 0)) else 0
    circ = r.get("circularity", float("nan"))
    aspect = r.get("aspect_ratio", float("nan"))
    aniso = r.get("anisotropy", float("nan"))
    dist_dyn = float(r.get("distance_to_dynamic_line_skeleton", 0))
    dist_stat = float(r.get("distance_to_static_line_skeleton", 0))
    dist_comb = float(r.get("distance_to_combined_line_exclusion", 0))
    dist_refined_stat = float(r.get("distance_to_refined_static_line_skeleton", 0)) \
        if pd.notna(r.get("distance_to_refined_static_line_skeleton", 0)) else 0
    dist_refined_comb = float(r.get("distance_to_refined_combined_line_exclusion", 0)) \
        if pd.notna(r.get("distance_to_refined_combined_line_exclusion", 0)) else 0
    mean_lr = float(r.get("mean_line_risk_in_patch", 0)) if pd.notna(r.get("mean_line_risk_in_patch", 0)) else 0
    uncov = int(r.get("uncovered_frame_count", 0))
    spot = int(r.get("spot_present_when_uncovered", 0))
    border = float(r.get("border_distance", 0))
    contrast = float(r.get("local_contrast", 0))
    in_invalid = bool(r.get("in_invalid_region", 0))

    if in_invalid:
        return False
    if not (hc["min_diameter_px"] <= d <= hc["max_diameter_px"]):
        return False
    if not (hc["min_area_px"] <= area <= hc["max_area_px"]):
        return False
    if not pd.notna(circ):
        return False
    if not (hc["min_circularity"] <= float(circ) <= hc["max_circularity"]):
        return False
    if pd.notna(aspect) and float(aspect) > hc["max_aspect_ratio"]:
        return False
    if pd.notna(aniso) and float(aniso) > hc["max_anisotropy"]:
        return False
    if "min_distance_to_dynamic_skeleton_px" in hc and dist_dyn < hc["min_distance_to_dynamic_skeleton_px"]:
        return False
    if "min_distance_to_static_skeleton_px" in hc and dist_stat < hc["min_distance_to_static_skeleton_px"]:
        return False
    if "min_distance_to_combined_exclusion_px" in hc and dist_comb < hc["min_distance_to_combined_exclusion_px"]:
        return False
    if "min_distance_to_refined_static_skeleton_px" in hc and dist_refined_stat < hc["min_distance_to_refined_static_skeleton_px"]:
        return False
    if "min_distance_to_refined_combined_exclusion_px" in hc and dist_refined_comb < hc["min_distance_to_refined_combined_exclusion_px"]:
        return False
    if mean_lr > hc["max_mean_line_risk_in_patch"]:
        return False
    if uncov < hc["min_uncovered_frame_count"]:
        return False
    if spot < hc["min_spot_present_when_uncovered"]:
        return False
    if contrast < hc.get("min_local_contrast", 0.0):
        return False
    if border < hc["min_border_distance_px"]:
        return False
    return True


def categorize(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Five-category split. Appends column `category` and `passes_high_confidence`."""
    cls = config.get("candidate_classification", {}) or {}
    hc = cls.get("high_confidence_clean", {}) or {}
    la = cls.get("line_adjacent_uncertain", {}) or {}
    tu = cls.get("texture_uncertain", {}) or {}
    rej = cls.get("rejected_line_like", {}) or {}
    tiny = cls.get("rejected_tiny_response", {}) or {}

    # Sensible defaults — Stage 1.3 prefers refined-distance fields.
    hc_defaults = dict(
        min_diameter_px=3, max_diameter_px=7,
        min_area_px=10, max_area_px=80,
        min_circularity=0.5, max_circularity=1.4,
        max_aspect_ratio=1.8, max_anisotropy=2.5,
        min_distance_to_dynamic_skeleton_px=6,
        min_distance_to_refined_static_skeleton_px=8,
        min_distance_to_refined_combined_exclusion_px=8,
        max_mean_line_risk_in_patch=0.30,
        min_uncovered_frame_count=2,
        min_spot_present_when_uncovered=2,
        min_local_contrast=0.015,
        min_border_distance_px=40,
    )
    for k, v in hc_defaults.items():
        hc.setdefault(k, v)

    la_defaults = dict(
        min_diameter_px=3, max_diameter_px=8,
        min_area_px=6, min_circularity=0.3, max_aspect_ratio=3.0,
        max_distance_to_refined_combined_exclusion_px=8,
        or_mean_line_risk_above=0.40,
    )
    for k, v in la_defaults.items():
        la.setdefault(k, v)

    tu_defaults = dict(
        min_distance_to_refined_combined_exclusion_px=6,
        min_anisotropy=4.0,
        or_aspect_ratio_above=2.0,
    )
    for k, v in tu_defaults.items():
        tu.setdefault(k, v)

    rej_defaults = dict(
        min_anisotropy=8.0,
        max_aspect_ratio_strict=4.0,
        drop_in_invalid_region=True,
    )
    for k, v in rej_defaults.items():
        rej.setdefault(k, v)

    tiny_defaults = dict(
        max_area_px=4, drop_if_circularity_nan=True, drop_if_circularity_below=0.10,
    )
    for k, v in tiny_defaults.items():
        tiny.setdefault(k, v)

    cats = []
    passes = []
    for _, r in df.iterrows():
        area = r.get("area_px", float("nan"))
        circ = r.get("circularity", float("nan"))
        aniso = r.get("anisotropy", float("nan"))
        aspect = r.get("aspect_ratio", float("nan"))
        d = float(r.get("diameter_px", 0))
        # Prefer refined distance if available
        if "distance_to_refined_combined_line_exclusion" in df.columns and pd.notna(
            r.get("distance_to_refined_combined_line_exclusion", float("nan"))
        ):
            dist_for_categ = float(r["distance_to_refined_combined_line_exclusion"])
        else:
            dist_for_categ = float(r.get("distance_to_combined_line_exclusion", 0))
        mean_lr = float(r.get("mean_line_risk_in_patch", 0)) if pd.notna(r.get("mean_line_risk_in_patch", 0)) else 0
        in_invalid = bool(r.get("in_invalid_region", 0))

        # 1. rejected_tiny_response
        tiny_area = (pd.isna(area) or float(area) <= float(tiny["max_area_px"]))
        circ_invalid = pd.isna(circ) or float(circ) < float(tiny["drop_if_circularity_below"])
        if tiny_area or (tiny["drop_if_circularity_nan"] and circ_invalid):
            cats.append("rejected_tiny_response"); passes.append(False); continue

        # 2. clearly line-like or in invalid region
        if rej["drop_in_invalid_region"] and in_invalid:
            cats.append("rejected_line_like"); passes.append(False); continue
        if pd.notna(aniso) and float(aniso) >= float(rej["min_anisotropy"]):
            cats.append("rejected_line_like"); passes.append(False); continue
        if pd.notna(aspect) and float(aspect) >= float(rej["max_aspect_ratio_strict"]):
            cats.append("rejected_line_like"); passes.append(False); continue

        # 3. high_confidence_clean
        if _check_high_confidence_clean(r, hc):
            cats.append("high_confidence_clean"); passes.append(True); continue

        # 4. line_adjacent_uncertain (uses refined distance preferentially)
        defect_shape = (
            la["min_diameter_px"] <= d <= la["max_diameter_px"]
            and pd.notna(area) and float(area) >= float(la["min_area_px"])
            and pd.notna(circ) and float(circ) >= float(la["min_circularity"])
            and (pd.isna(aspect) or float(aspect) <= float(la["max_aspect_ratio"]))
        )
        la_max_dist = float(
            la.get("max_distance_to_refined_combined_exclusion_px",
                   la.get("max_distance_to_combined_exclusion_px", 8))
        )
        near_line = (dist_for_categ < la_max_dist) or (mean_lr > float(la["or_mean_line_risk_above"]))
        if defect_shape and near_line:
            cats.append("line_adjacent_uncertain"); passes.append(False); continue

        # 5. texture_uncertain
        tu_min_dist = float(
            tu.get("min_distance_to_refined_combined_exclusion_px",
                   tu.get("min_distance_to_combined_exclusion_px", 6))
        )
        not_near = dist_for_categ >= tu_min_dist
        textured = (
            (pd.notna(aniso) and float(aniso) > float(tu["min_anisotropy"]))
            or (pd.notna(aspect) and float(aspect) > float(tu["or_aspect_ratio_above"]))
        )
        if not_near and textured:
            cats.append("texture_uncertain"); passes.append(False); continue

        cats.append("texture_uncertain"); passes.append(False)

    df = df.copy()
    df["category"] = cats
    df["passes_high_confidence"] = passes
    return df


# ---------------------------------------------------------------------------
# Parameter sweep — preset filter for HC-style gates
# ---------------------------------------------------------------------------

def apply_sweep(df: pd.DataFrame, hc_cfg: dict, preset_overrides: dict) -> pd.DataFrame:
    """Return rows that pass HC rules with two overrides:
    `min_distance_to_combined_exclusion_px` and `max_anisotropy`.
    """
    cfg = dict(hc_cfg) if hc_cfg else {}
    cfg.update(preset_overrides or {})
    keep_idx = []
    for i, r in df.iterrows():
        if _check_high_confidence_clean(r, cfg):
            keep_idx.append(i)
    return df.loc[keep_idx].copy()
