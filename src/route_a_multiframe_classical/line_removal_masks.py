"""Stage 3B narrow line masks + defect-protect mask (task_stage3.md §8–9)."""

from __future__ import annotations

from typing import Dict, Iterable, List

import cv2
import numpy as np
import pandas as pd


def narrow_line_mask(
    refined_static_skeleton: np.ndarray,
    dynamic_line_skeleton: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Union of static + dynamic skeleton dilated by ``radius`` px."""
    s = refined_static_skeleton.astype(np.uint8)
    d = dynamic_line_skeleton.astype(np.uint8)
    union = ((s > 0) | (d > 0)).astype(np.uint8)
    if radius <= 0:
        return union
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(union, k).astype(np.uint8)


def coverage_fraction(mask: np.ndarray) -> float:
    m = np.asarray(mask)
    if m.size == 0:
        return 0.0
    return float((m > 0).mean())


def line_mask_coverage_table(
    masks: Dict[int, np.ndarray], too_aggressive_threshold: float = 0.35
) -> pd.DataFrame:
    rows = []
    for r, m in masks.items():
        cov = coverage_fraction(m)
        flag = "too_aggressive_for_inpainting" if cov > too_aggressive_threshold else ""
        rows.append({
            "name": f"line_mask_radius{r}",
            "radius_px": int(r),
            "coverage_fraction": cov,
            "coverage_percent": cov * 100.0,
            "flag": flag,
        })
    return pd.DataFrame(rows)


def defect_protect_mask(
    shape,
    candidates_df: pd.DataFrame,
    extra_radius_px: int = 2,
    min_radius_px: int = 4,
) -> np.ndarray:
    """Disk mask around each candidate centre.

    radius = max(min_radius_px, diameter_px / 2 + extra_radius_px)

    Returns uint8 0/1.
    """
    H, W = shape
    out = np.zeros((H, W), dtype=np.uint8)
    if candidates_df is None or candidates_df.empty:
        return out
    for _, row in candidates_df.iterrows():
        x = float(row.get("x", 0.0))
        y = float(row.get("y", 0.0))
        d = float(row.get("diameter_px", 6.0))
        r = max(int(min_radius_px), int(round(d / 2.0 + extra_radius_px)))
        cx, cy = int(round(x)), int(round(y))
        if cx < -r or cy < -r or cx > W + r or cy > H + r:
            continue
        cv2.circle(out, (cx, cy), r, 1, thickness=-1, lineType=cv2.LINE_8)
    return out


def aggregate_protect_candidates(
    out_dir: str,
    sources: Iterable[str],
) -> pd.DataFrame:
    """Concatenate candidate CSVs from several Stage-3A outputs."""
    import os
    parts: List[pd.DataFrame] = []
    for path in sources:
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
