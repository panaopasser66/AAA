from __future__ import annotations

import os
from typing import Union

import numpy as np
import tifffile
import cv2


def ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def load_tif_float32(path: str) -> np.ndarray:
    """Load TIFF as float32 2D image. Preserve raw values before normalization."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"TIFF not found: {path}")
    img = tifffile.imread(path)
    if img.ndim != 2:
        raise ValueError(
            f"Expected 2D TIFF, got shape={img.shape} for {path}. "
            "Multi-page / 3D TIFFs are not supported in Stage 1."
        )
    return img.astype(np.float32, copy=True)


def robust_rescale_for_preview(
    img: np.ndarray, p_low: float = 1.0, p_high: float = 99.0
) -> np.ndarray:
    """Return uint8 preview image in [0,255] using robust percentile clipping."""
    arr = np.asarray(img, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.percentile(finite, p_low))
    hi = float(np.percentile(finite, p_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        lo = float(finite.min())
        hi = float(finite.max())
        if hi - lo < 1e-12:
            return np.zeros(arr.shape, dtype=np.uint8)
    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def save_png(path: str, img: Union[np.ndarray]) -> None:
    """Save image (uint8 or float in [0,1]) to PNG via OpenCV."""
    ensure_dir(os.path.dirname(path))
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        a = np.asarray(arr, dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
        if a.max() <= 1.0 + 1e-6 and a.min() >= -1e-6:
            arr = (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        else:
            arr = robust_rescale_for_preview(a)
    # cv2 expects BGR or grayscale; grayscale arrays are written as-is
    ok = cv2.imwrite(path, arr)
    if not ok:
        raise IOError(f"Failed to write PNG: {path}")
