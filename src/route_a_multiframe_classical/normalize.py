from __future__ import annotations

import numpy as np


def _sanitize(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def percentile_normalize(
    img: np.ndarray, low: float = 0.5, high: float = 99.5
) -> np.ndarray:
    """Robust percentile normalization to float32 in [0,1]."""
    if high <= low:
        raise ValueError(f"percentile_normalize: high ({high}) must be > low ({low})")
    arr = _sanitize(img)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(finite, low))
    hi = float(np.percentile(finite, high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        lo = float(finite.min())
        hi = float(finite.max())
        if hi - lo < 1e-12:
            return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32, copy=False)


def _median_mad(arr: np.ndarray) -> tuple:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    m = float(np.median(finite))
    mad = float(np.median(np.abs(finite - m)))
    if mad < 1e-6:
        mad = float(np.std(finite))
    if mad < 1e-6:
        mad = 1.0
    return m, mad


def match_median_mad_to_reference(
    img: np.ndarray, ref: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Match image median/MAD to reference. Inputs assumed already in [0,1]."""
    img_f = _sanitize(img)
    ref_f = _sanitize(ref)
    m_img, s_img = _median_mad(img_f)
    m_ref, s_ref = _median_mad(ref_f)
    s_img = max(s_img, eps)
    scale = s_ref / s_img
    out = (img_f - m_img) * scale + m_ref
    out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32, copy=False)
