"""Refined static line mask: keep only long thin components from a Sato response.

The Stage 1.2 static mask thresholded the Sato/Frangi response by percentile,
which kept many small blobs and round bright spots. This module performs
connected-component shape analysis and keeps only components that look like
genuine Kossel lines (long, thin, eccentric).
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
from skimage.measure import regionprops
from skimage.morphology import skeletonize


def refine_static_line_mask(
    response: np.ndarray,
    source_percentile: float = 90.0,
    morphology_close_px: int = 3,
    min_area_px: int = 30,
    min_major_axis_length: float = 30.0,
    min_aspect_ratio: float = 4.0,
    min_eccentricity: float = 0.90,
    min_skeleton_length_px: int = 25,
    max_circularity: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, dict]]:
    """Run component-level filtering on a ridge response.

    Returns:
      keep_mask (uint8 0/1)
      component_visualisation (uint8 grayscale, kept=255, dropped=80)
      skeleton (uint8 0/1)
      stats_per_kept_component: dict {label_id: properties}
    """
    arr = np.asarray(response, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return arr.astype(np.uint8), arr.astype(np.uint8), arr.astype(np.uint8), {}

    thr = float(np.percentile(arr, source_percentile))
    binary = (arr >= thr).astype(np.uint8)
    if morphology_close_px and morphology_close_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morphology_close_px, morphology_close_px)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        z = np.zeros_like(binary, dtype=np.uint8)
        return z, z, z, {}

    # Use skimage regionprops for shape descriptors
    props = regionprops(labels)
    keep_mask = np.zeros_like(binary, dtype=np.uint8)
    component_vis = np.zeros_like(binary, dtype=np.uint8)
    kept_info: Dict[int, dict] = {}

    for p in props:
        if p.label == 0:
            continue
        area = float(p.area)
        # Major/minor axis (handle near-degenerate components)
        major = float(p.major_axis_length or 0.0)
        minor = float(p.minor_axis_length or 0.0)
        if minor < 1e-6:
            aspect = float(major) if major > 0 else 0.0
        else:
            aspect = major / minor
        ecc = float(p.eccentricity)
        perim = float(p.perimeter or 0.0)
        circ = (
            (4.0 * np.pi * area / (perim * perim))
            if perim > 1e-6
            else float("inf")
        )

        # Per-component skeleton length
        coords = p.coords
        ys = coords[:, 0]
        xs = coords[:, 1]
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        sub = (labels[y0 : y1 + 1, x0 : x1 + 1] == p.label).astype(np.uint8)
        sub_skel = skeletonize(sub > 0).astype(np.uint8)
        skel_len = int(sub_skel.sum())

        component_vis[labels == p.label] = 80  # mark dropped by default

        if area < min_area_px:
            continue
        if major < min_major_axis_length:
            continue
        if aspect < min_aspect_ratio:
            continue
        if ecc < min_eccentricity:
            continue
        if skel_len < min_skeleton_length_px:
            continue
        if circ > max_circularity:
            continue

        keep_mask[labels == p.label] = 1
        component_vis[labels == p.label] = 255
        kept_info[int(p.label)] = {
            "area_px": area,
            "major_axis_length": major,
            "minor_axis_length": minor,
            "aspect_ratio": float(aspect),
            "eccentricity": ecc,
            "circularity": float(circ) if np.isfinite(circ) else None,
            "skeleton_length_px": skel_len,
            "centroid_x": float(p.centroid[1]),
            "centroid_y": float(p.centroid[0]),
        }

    skeleton = skeletonize(keep_mask > 0).astype(np.uint8)
    return keep_mask, component_vis, skeleton, kept_info


def distance_from_skeleton(skeleton: np.ndarray, distance_clip_px: int = 60) -> np.ndarray:
    """Distance transform from skeleton pixels, clipped."""
    dt = cv2.distanceTransform((1 - skeleton).astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(dt, 0.0, float(distance_clip_px)).astype(np.float32)


def union_skeleton(*skeletons: np.ndarray) -> np.ndarray:
    """Element-wise OR of skeleton masks (uint8 0/1)."""
    if not skeletons:
        raise ValueError("union_skeleton: empty input")
    out = np.zeros_like(skeletons[0], dtype=np.uint8)
    for s in skeletons:
        out |= (s > 0).astype(np.uint8)
    return out


def dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask.astype(np.uint8)
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius_px * 2 + 1, radius_px * 2 + 1)
    )
    return cv2.dilate(mask.astype(np.uint8), k).astype(np.uint8)
