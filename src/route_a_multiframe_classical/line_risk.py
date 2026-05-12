from __future__ import annotations

from typing import List

import numpy as np


def compute_line_risk_from_mad(
    mad_img: np.ndarray, percentile_high: float = 99.5
) -> np.ndarray:
    """Normalize MAD/std image into [0,1] line-risk map."""
    arr = np.asarray(mad_img, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    hi = float(np.percentile(finite, percentile_high))
    if hi <= 1e-12:
        hi = float(finite.max())
        if hi <= 1e-12:
            return np.zeros_like(arr, dtype=np.float32)
    out = np.clip(arr / hi, 0.0, 1.0)
    return out.astype(np.float32, copy=False)


def combine_line_risks(risk_maps: List[np.ndarray]) -> np.ndarray:
    """Use elementwise max to combine per-group risk maps."""
    if not risk_maps:
        raise ValueError("combine_line_risks: empty list")
    stack = np.stack([np.asarray(m, dtype=np.float32) for m in risk_maps], axis=0)
    return stack.max(axis=0).astype(np.float32)
