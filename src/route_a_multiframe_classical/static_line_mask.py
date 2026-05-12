from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np
from skimage.filters import sato
from skimage.morphology import skeletonize


def _to01(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn > 1e-12:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr)
    return arr.astype(np.float32, copy=False)


def static_line_response(
    img: np.ndarray,
    sigmas: List[float],
    detect_bright: bool = True,
    detect_dark: bool = True,
) -> np.ndarray:
    """Multi-scale Sato ridge response. Max across bright + dark ridges."""
    arr = _to01(img)
    out = np.zeros_like(arr, dtype=np.float32)
    if detect_bright:
        rb = sato(arr, sigmas=sigmas, black_ridges=False).astype(np.float32)
        out = np.maximum(out, rb)
    if detect_dark:
        rd = sato(arr, sigmas=sigmas, black_ridges=True).astype(np.float32)
        out = np.maximum(out, rd)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def build_static_line_masks(
    response: np.ndarray,
    percentiles: List[int],
    primary_percentile: int,
    morph_close_px: int = 1,
    distance_clip_px: int = 60,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray, float]:
    """From a ridge-response map, build percentile masks, skeleton, distance map."""
    arr = np.asarray(response, dtype=np.float32)
    masks: Dict[int, np.ndarray] = {}
    for p in percentiles:
        thr = float(np.percentile(arr, p))
        m = (arr >= thr).astype(np.uint8)
        if morph_close_px and morph_close_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_px, morph_close_px))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        masks[int(p)] = m
    primary = masks[int(primary_percentile)]
    primary_thr = float(np.percentile(arr, int(primary_percentile)))
    skeleton = skeletonize(primary > 0).astype(np.uint8)
    distance = cv2.distanceTransform((1 - skeleton).astype(np.uint8), cv2.DIST_L2, 5)
    distance = np.clip(distance, 0.0, float(distance_clip_px)).astype(np.float32)
    return masks, skeleton, distance, primary_thr


def build_combined_exclusion(
    dynamic_mask: np.ndarray,
    static_mask: np.ndarray,
    dilate_list: List[int],
    primary_dilate_px: int,
    distance_clip_px: int = 60,
) -> Dict[str, np.ndarray]:
    """Union dynamic + static line masks; build skeleton, distance, dilations.

    Returns dict with:
      mask: union mask (uint8)
      skeleton: skeleton (uint8)
      distance: distance to skeleton (float32, clipped)
      dilate_<n>: dilated union for each n in dilate_list (uint8)
      primary_dilate: convenience copy of dilate_<primary_dilate_px>
    """
    union = ((dynamic_mask > 0) | (static_mask > 0)).astype(np.uint8)
    skel = skeletonize(union > 0).astype(np.uint8)
    distance = cv2.distanceTransform((1 - skel).astype(np.uint8), cv2.DIST_L2, 5)
    distance = np.clip(distance, 0.0, float(distance_clip_px)).astype(np.float32)
    out = {"mask": union, "skeleton": skel, "distance": distance}
    for n in dilate_list:
        if n <= 0:
            out[f"dilate_{n}"] = union.copy()
            continue
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n * 2 + 1, n * 2 + 1))
        out[f"dilate_{n}"] = cv2.dilate(union, k).astype(np.uint8)
    if f"dilate_{primary_dilate_px}" in out:
        out["primary_dilate"] = out[f"dilate_{primary_dilate_px}"]
    else:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (primary_dilate_px * 2 + 1, primary_dilate_px * 2 + 1)
        )
        out["primary_dilate"] = cv2.dilate(union, k).astype(np.uint8)
    return out
