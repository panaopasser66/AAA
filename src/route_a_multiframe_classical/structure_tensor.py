from __future__ import annotations

import cv2
import numpy as np


def anisotropy_at(
    img: np.ndarray,
    x: float,
    y: float,
    patch_radius: int = 8,
    sigma: float = 1.5,
) -> float:
    """Return local structure-tensor eigenvalue ratio (>=1) at (x,y).

    Computed on a small patch centered at (x,y). Uses Sobel gradients and
    Gaussian-smoothed second-moment matrix [[Jxx, Jxy],[Jxy, Jyy]]. Returns
    lambda_max / lambda_min. Value ~1 means isotropic (spot-like). Large value
    means anisotropic (edge / line). NaN if the patch is too small or flat.
    """
    H, W = img.shape[:2]
    pad = max(3, int(patch_radius) + 2)
    x0 = max(0, int(round(x - pad)))
    x1 = min(W, int(round(x + pad + 1)))
    y0 = max(0, int(round(y - pad)))
    y1 = min(H, int(round(y + pad + 1)))
    p = np.ascontiguousarray(img[y0:y1, x0:x1], dtype=np.float32)
    if min(p.shape) < 5:
        return float("nan")
    Ix = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    Jxx = cv2.GaussianBlur(Ix * Ix, (0, 0), sigma)
    Jxy = cv2.GaussianBlur(Ix * Iy, (0, 0), sigma)
    Jyy = cv2.GaussianBlur(Iy * Iy, (0, 0), sigma)
    cy = (y1 - y0) // 2
    cx = (x1 - x0) // 2
    cy = max(0, min(p.shape[0] - 1, cy))
    cx = max(0, min(p.shape[1] - 1, cx))
    a = float(Jxx[cy, cx])
    b = float(Jxy[cy, cx])
    c = float(Jyy[cy, cx])
    tr = a + c
    det = a * c - b * b
    disc = tr * tr / 4.0 - det
    if disc < 0:
        disc = 0.0
    sq = float(np.sqrt(disc))
    l1 = tr / 2.0 + sq
    l2 = tr / 2.0 - sq
    if l2 < 1e-10 or not np.isfinite(l2):
        return float("inf") if l1 > 1e-8 else float("nan")
    return float(l1 / l2)


def mean_value_in_disk(img: np.ndarray, x: float, y: float, radius: float) -> float:
    """Mean of img inside a circular disk centered at (x,y), radius `radius` px."""
    H, W = img.shape[:2]
    r = int(np.ceil(radius)) + 1
    x0 = max(0, int(round(x)) - r)
    x1 = min(W, int(round(x)) + r + 1)
    y0 = max(0, int(round(y)) - r)
    y1 = min(H, int(round(y)) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return float("nan")
    patch = img[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.ogrid[: patch.shape[0], : patch.shape[1]]
    cx = x - x0
    cy = y - y0
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
    if not mask.any():
        return float("nan")
    return float(patch[mask].mean())
