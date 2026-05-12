"""Stage 3 — manual-review reranking.

We take `line_adjacent_uncertain` and `texture_uncertain` from the v2 detection
and rebuild a single ordered list suitable for human review. The Stage 1/2
`final_score` was tuned for "is it clean defect" (it punishes anything near a
line). For human review we want the opposite emphasis: surface candidates that
look most defect-like *given* the line context, so reviewers can judge whether
the spot is real even when it sits near a line.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd


def _norm_pos(arr, p_high=95.0):
    a = np.asarray(arr, dtype=np.float32)
    a = np.where(np.isfinite(a), np.clip(a, 0, None), 0.0)
    if a.size == 0:
        return a
    hi = float(np.percentile(a, p_high)) if a.size > 5 else float(a.max())
    if hi <= 1e-9:
        return np.zeros_like(a)
    return np.clip(a / hi, 0.0, 1.0)


def _norm_dist(arr, scale=10.0):
    a = np.asarray(arr, dtype=np.float32)
    a = np.where(np.isfinite(a), np.clip(a, 0, None), 0.0)
    return np.clip(a / scale, 0.0, 1.0)


def rerank_for_review(
    df: pd.DataFrame, config: dict, persistence_field_priority: Sequence[str] = (
        "spot_present_when_uncovered", "uncovered_frame_count", "persistence_ratio",
    ),
) -> pd.DataFrame:
    """Return df sorted by review_score with shape filters applied."""
    cfg = config.get("manual_review", {}) or {}
    w = cfg.get("weights", {}) or {}
    min_d = float(cfg.get("min_diameter_px", 3))
    max_d = float(cfg.get("max_diameter_px", 7))
    min_a = float(cfg.get("min_area_px", 6))
    max_a = float(cfg.get("max_area_px", 120))
    exclude_invalid = bool(cfg.get("exclude_in_invalid_region", True))

    if df.empty:
        df = df.copy()
        df["review_score"] = []
        return df

    keep = df.copy()
    keep = keep[(keep["diameter_px"] >= min_d) & (keep["diameter_px"] <= max_d)]
    keep = keep[(keep["area_px"] >= min_a) & (keep["area_px"] <= max_a)]
    if exclude_invalid and "in_invalid_region" in keep.columns:
        keep = keep[keep["in_invalid_region"].astype(int) == 0]
    keep = keep.reset_index(drop=True)
    if keep.empty:
        keep["review_score"] = []
        return keep

    contrast = _norm_pos(keep["local_contrast"].to_numpy())
    circ = np.where(
        np.isfinite(keep["circularity"].to_numpy()),
        np.clip(keep["circularity"].to_numpy(), 0.0, 1.0),
        0.0,
    )
    aspect = keep["aspect_ratio"].fillna(99.0).to_numpy()
    aspect_pen = np.clip((aspect - 1.0) / 4.0, 0.0, 1.0)
    aniso = keep["anisotropy"].fillna(99.0).to_numpy()
    aniso_pen = np.clip((aniso - 1.0) / 9.0, 0.0, 1.0)
    mean_lr = keep["mean_line_risk_in_patch"].fillna(0.0).clip(0.0, 1.0).to_numpy()
    invalid = keep["in_invalid_region"].fillna(0).astype(int).to_numpy().astype(np.float32)

    dist_static = _norm_dist(
        keep.get("distance_to_refined_static_line_skeleton",
                 pd.Series([0.0] * len(keep))).to_numpy(),
        scale=12.0,
    )
    dist_combined = _norm_dist(
        keep.get("distance_to_refined_combined_line_exclusion",
                 pd.Series([0.0] * len(keep))).to_numpy(),
        scale=10.0,
    )

    persistence = None
    valid = keep.get("valid_frame_count", pd.Series([0] * len(keep))).astype(float).to_numpy()
    for field in persistence_field_priority:
        if field in keep.columns:
            arr = keep[field].fillna(0.0).astype(float).to_numpy()
            if field == "spot_present_when_uncovered":
                uncov = keep["uncovered_frame_count"].fillna(0).astype(float).to_numpy()
                persistence = np.where(uncov > 0, arr / np.maximum(uncov, 1.0), 0.0)
            elif field == "uncovered_frame_count":
                persistence = np.where(valid > 0, arr / np.maximum(valid, 1.0), 0.0)
            else:
                persistence = np.clip(arr, 0.0, 1.0)
            break
    if persistence is None:
        persistence = np.zeros(len(keep), dtype=np.float32)

    score = (
        + float(w.get("contrast", 0.6)) * contrast
        + float(w.get("circularity", 0.6)) * circ
        + float(w.get("distance_to_refined_combined_line_exclusion", 0.4)) * dist_combined
        + float(w.get("distance_to_refined_static_line_skeleton", 0.4)) * dist_static
        + float(w.get("persistence", 1.0)) * persistence
        - float(w.get("anisotropy_penalty", 0.5)) * aniso_pen
        - float(w.get("aspect_penalty", 0.4)) * aspect_pen
        - float(w.get("mean_line_risk_penalty", 0.6)) * mean_lr
        - float(w.get("invalid_penalty", 1.5)) * invalid
    )
    keep["review_score"] = score.astype(np.float32)
    keep = keep.sort_values("review_score", ascending=False).reset_index(drop=True)
    return keep


def nms_in_place(df: pd.DataFrame, min_distance_px: float = 4.0) -> pd.DataFrame:
    """Greedy NMS keeping highest-review_score points first."""
    if df.empty:
        return df
    df = df.sort_values("review_score", ascending=False).reset_index(drop=True)
    xs = df["x"].to_numpy()
    ys = df["y"].to_numpy()
    keep = []
    d2 = min_distance_px * min_distance_px
    for i in range(len(df)):
        ok = True
        for j in keep:
            if (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 < d2:
                ok = False
                break
        if ok:
            keep.append(i)
    return df.iloc[keep].reset_index(drop=True)


def labels_template(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Build a manual-labels CSV with empty rating columns."""
    out = df.copy()
    out.insert(0, "review_rank", np.arange(1, len(out) + 1))
    for c in columns:
        if c not in out.columns:
            out[c] = ""
    return out
