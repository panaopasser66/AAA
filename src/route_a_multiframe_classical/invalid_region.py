from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np


def warp_valid_mask(shape, dx: float, dy: float) -> np.ndarray:
    """Return uint8 (0/1) valid mask for cv2.warpAffine translation by (dx,dy).

    A destination pixel (u,v) maps to source (u+dx, v+dy). Valid iff source
    is in-bounds. Border mode used for actual warp is REPLICATE, so out-of-bound
    pixels are filled — we want to mark those as invalid for later analysis.
    """
    H, W = shape
    ones = np.ones((H, W), dtype=np.float32)
    M = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    out = cv2.warpAffine(
        ones, M, (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (out > 0.5).astype(np.uint8)


def _keep_large_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1
    return keep


def build_invalid_region_mask(
    ref_norm: np.ndarray,
    valid_masks: List[np.ndarray],
    config: dict,
) -> dict:
    """Build an invalid-region mask combining:
      1. image border (border_px)
      2. dark vignette / bottom arc (ref_norm < dark_vignette_norm_max)
      3. over-exposed center (ref_norm > bright_saturation_norm_min)
      4. union of warp-invalid regions (1 - AND(valid_masks))

    Returns a dict with each component as uint8 (0/1) plus the combined mask.
    """
    cfg = config.get("invalid_region", {}) or {}
    border = int(cfg.get("border_px", 32))
    dark_max = float(cfg.get("dark_vignette_norm_max", 0.02))
    bright_min = float(cfg.get("bright_saturation_norm_min", 0.98))
    vignette_min_area = int(cfg.get("vignette_min_area_px", 1500))
    sat_min_area = int(cfg.get("saturation_min_area_px", 800))
    close_px = int(cfg.get("morphology_close_px", 9))
    open_px = int(cfg.get("morphology_open_px", 5))
    use_warp = bool(cfg.get("use_warp_valid_mask", True))
    dilate_px = int(cfg.get("invalid_dilate_px", 3))

    H, W = ref_norm.shape[:2]

    # 1. Border
    border_mask = np.ones((H, W), dtype=np.uint8)
    if border > 0:
        border_mask[border:-border, border:-border] = 0

    # 2. Dark vignette
    dark = (ref_norm <= dark_max).astype(np.uint8)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k)
    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)
    dark = _keep_large_components(dark, vignette_min_area)

    # 3. Bright saturation
    bright = (ref_norm >= bright_min).astype(np.uint8)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k)
    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k)
    bright = _keep_large_components(bright, sat_min_area)

    # 4. Warp invalid (combined)
    if use_warp and valid_masks:
        valid_all = np.ones((H, W), dtype=np.uint8)
        for vm in valid_masks:
            valid_all &= vm.astype(np.uint8)
        warp_invalid = (1 - valid_all).astype(np.uint8)
    else:
        valid_all = np.ones((H, W), dtype=np.uint8)
        warp_invalid = np.zeros((H, W), dtype=np.uint8)

    invalid = border_mask | dark | bright | warp_invalid
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        invalid = cv2.dilate(invalid, k)

    return {
        "border": border_mask,
        "dark_vignette": dark,
        "bright_saturation": bright,
        "warp_invalid": warp_invalid,
        "warp_valid_combined": valid_all,
        "invalid": invalid.astype(np.uint8),
    }
