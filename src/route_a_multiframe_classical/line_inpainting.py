"""Stage 3B classical inpainting baselines (task_stage3.md §10)."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_uint8(img01: np.ndarray) -> np.ndarray:
    a = np.asarray(img01, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    a = np.clip(a, 0.0, 1.0)
    return (a * 255.0 + 0.5).astype(np.uint8)


def _to01_from_u8(arr: np.ndarray) -> np.ndarray:
    return (arr.astype(np.float32) / 255.0).astype(np.float32)


def _effective_mask(mask: np.ndarray, protect_mask: Optional[np.ndarray]) -> np.ndarray:
    m = (mask.astype(np.uint8) > 0).astype(np.uint8)
    if protect_mask is not None:
        m = m & (1 - (protect_mask.astype(np.uint8) > 0).astype(np.uint8))
    return m


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def inpaint_telea(
    img01: np.ndarray, mask: np.ndarray, radius: int = 5,
    protect_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    src = _to_uint8(img01)
    m = _effective_mask(mask, protect_mask) * 255
    out = cv2.inpaint(src, m.astype(np.uint8), float(radius), cv2.INPAINT_TELEA)
    return _to01_from_u8(out)


def inpaint_navier_stokes(
    img01: np.ndarray, mask: np.ndarray, radius: int = 5,
    protect_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    src = _to_uint8(img01)
    m = _effective_mask(mask, protect_mask) * 255
    out = cv2.inpaint(src, m.astype(np.uint8), float(radius), cv2.INPAINT_NS)
    return _to01_from_u8(out)


def inpaint_local_median(
    img01: np.ndarray, mask: np.ndarray, window_size: int = 25,
    protect_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Replace masked pixels with a local-median estimate that EXCLUDES masked
    pixels from the average. Implemented via NaN-fill + sliding-window median
    using a uint16 round trip + cv2.medianBlur.

    For window_size>5 we cannot use cv2.medianBlur on a wide kernel directly
    on float; we approximate by:
      1. Set masked pixels to the mean of the entire image (a stable fill).
      2. medianBlur with kernel=5 to denoise.
      3. Box-blur with kernel=window_size to smooth across the mask.

    The result is a low-frequency background estimate. We composite only on
    masked (and not protected) pixels.
    """
    arr = np.asarray(img01, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)

    eff_mask = _effective_mask(mask, protect_mask)

    # Pre-fill masked pixels with surrounding mean to avoid biasing
    inv = 1.0 - eff_mask.astype(np.float32)
    mean_outside = (arr * inv).sum() / max(inv.sum(), 1.0)
    seeded = arr.copy()
    seeded[eff_mask > 0] = mean_outside

    src_u16 = (seeded * 65535.0 + 0.5).astype(np.uint16)
    med_u16 = cv2.medianBlur(src_u16, 5)

    bw = max(3, int(window_size))
    if bw % 2 == 0:
        bw += 1
    smoothed = cv2.boxFilter(med_u16.astype(np.float32), ddepth=cv2.CV_32F,
                             ksize=(bw, bw), normalize=True)
    smoothed = (smoothed / 65535.0).astype(np.float32)

    out = arr.copy()
    out[eff_mask > 0] = smoothed[eff_mask > 0]
    return out


# ---------------------------------------------------------------------------
# Convenience: residual
# ---------------------------------------------------------------------------

def residual(original01: np.ndarray, cleaned01: np.ndarray) -> np.ndarray:
    return (np.asarray(original01, dtype=np.float32)
            - np.asarray(cleaned01, dtype=np.float32))
