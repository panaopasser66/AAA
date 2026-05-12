"""Coverage statistics for binary mask outputs."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def coverage_fraction(mask: np.ndarray) -> float:
    """Return fraction of pixels equal to 1 (>0) in the mask."""
    m = np.asarray(mask)
    if m.size == 0:
        return 0.0
    return float((m > 0).mean())


def collect_coverage(out_dir: str, too_aggressive_threshold: float = 0.50) -> pd.DataFrame:
    """Walk a fixed list of expected mask npy files; report fraction + flag.

    The list is hand-curated rather than glob-based to keep the report stable
    even when a particular mask is missing (it shows up with status `missing`).
    """
    candidates: List[Tuple[str, str, str]] = [
        # (group, name, path)
        ("dynamic", "combined_line_mask_p90", os.path.join(out_dir, "line_risk", "combined_line_mask_p90.npy")),
        ("dynamic", "combined_line_mask_p95", os.path.join(out_dir, "line_risk", "combined_line_mask_p95.npy")),
        ("dynamic", "combined_line_mask_p97", os.path.join(out_dir, "line_risk", "combined_line_mask_p97.npy")),
        ("dynamic", "combined_line_mask_primary", os.path.join(out_dir, "line_risk", "combined_line_mask.npy")),
        ("static_raw", "static_line_mask_p90", os.path.join(out_dir, "line_risk", "static_line_mask_p90.npy")),
        ("static_raw", "static_line_mask_p95", os.path.join(out_dir, "line_risk", "static_line_mask_p95.npy")),
        ("static_raw", "static_line_mask_p97", os.path.join(out_dir, "line_risk", "static_line_mask_p97.npy")),
        ("static_raw", "static_line_mask_primary", os.path.join(out_dir, "line_risk", "static_line_mask.npy")),
        ("static_refined", "refined_static_line_mask", os.path.join(out_dir, "line_risk", "refined_static_line_mask.npy")),
        ("static_refined", "refined_static_line_skeleton", os.path.join(out_dir, "line_risk", "refined_static_line_skeleton.npy")),
        ("combined_v1", "combined_line_exclusion_mask", os.path.join(out_dir, "line_risk", "combined_line_exclusion_mask.npy")),
        ("combined_v1", "combined_line_exclusion_mask_dilate8", os.path.join(out_dir, "line_risk", "combined_line_exclusion_mask_dilate8.npy")),
        ("combined_v1", "combined_line_exclusion_mask_dilate12", os.path.join(out_dir, "line_risk", "combined_line_exclusion_mask_dilate12.npy")),
        ("combined_v1", "combined_line_exclusion_mask_dilate16", os.path.join(out_dir, "line_risk", "combined_line_exclusion_mask_dilate16.npy")),
        ("combined_refined", "refined_combined_line_exclusion_dilate6", os.path.join(out_dir, "line_risk", "refined_combined_line_exclusion_dilate6.npy")),
        ("combined_refined", "refined_combined_line_exclusion_dilate8", os.path.join(out_dir, "line_risk", "refined_combined_line_exclusion_dilate8.npy")),
        ("combined_refined", "refined_combined_line_exclusion_dilate10", os.path.join(out_dir, "line_risk", "refined_combined_line_exclusion_dilate10.npy")),
        ("safe_v1", "safe_search_mask", os.path.join(out_dir, "invalid_region", "safe_search_mask.npy")),
        ("safe_refined", "refined_safe_search_mask_dilate6", os.path.join(out_dir, "invalid_region", "refined_safe_search_mask_dilate6.npy")),
        ("safe_refined", "refined_safe_search_mask_dilate8", os.path.join(out_dir, "invalid_region", "refined_safe_search_mask_dilate8.npy")),
        ("safe_refined", "refined_safe_search_mask_dilate10", os.path.join(out_dir, "invalid_region", "refined_safe_search_mask_dilate10.npy")),
        ("invalid", "invalid_region_mask", os.path.join(out_dir, "invalid_region", "invalid_region_mask.npy")),
    ]
    rows = []
    for group, name, path in candidates:
        if not os.path.isfile(path):
            rows.append({
                "group": group,
                "name": name,
                "path": os.path.relpath(path, out_dir),
                "coverage_fraction": float("nan"),
                "coverage_percent": float("nan"),
                "flag": "missing",
            })
            continue
        try:
            arr = np.load(path)
        except Exception as e:
            rows.append({
                "group": group, "name": name,
                "path": os.path.relpath(path, out_dir),
                "coverage_fraction": float("nan"),
                "coverage_percent": float("nan"),
                "flag": f"load_error:{e}",
            })
            continue
        frac = coverage_fraction(arr)
        is_exclusion_like = (
            group in ("combined_v1", "combined_refined")
            and "dilate" in name
        )
        flag = ""
        if is_exclusion_like and frac > too_aggressive_threshold:
            flag = "too_aggressive"
        rows.append({
            "group": group,
            "name": name,
            "path": os.path.relpath(path, out_dir),
            "coverage_fraction": float(frac),
            "coverage_percent": float(frac * 100.0),
            "flag": flag,
        })

    return pd.DataFrame(rows)
