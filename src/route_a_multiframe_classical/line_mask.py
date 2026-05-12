from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize


def build_line_masks(
    combined_line_risk: np.ndarray,
    percentiles: List[int],
    primary_percentile: int,
    morph_close_px: int = 3,
    remove_small_objects_px: int = 0,
    distance_clip_px: int = 60,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray, float]:
    """From a [0,1] line-risk map, build:
      - dict {percentile -> binary mask (uint8 0/1)}
      - skeleton (uint8 0/1) of primary mask
      - distance-from-skeleton map (float32, clipped at distance_clip_px)
      - primary_threshold (float)
    """
    risk = np.asarray(combined_line_risk, dtype=np.float32)
    masks: Dict[int, np.ndarray] = {}
    for p in percentiles:
        thr = float(np.percentile(risk, p))
        m = (risk >= thr).astype(np.uint8)
        if morph_close_px and morph_close_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_px, morph_close_px))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        if remove_small_objects_px and remove_small_objects_px > 0:
            num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            keep = np.zeros_like(m)
            for i in range(1, num):
                if stats[i, cv2.CC_STAT_AREA] >= remove_small_objects_px:
                    keep[labels == i] = 1
            m = keep
        masks[int(p)] = m

    primary = masks[int(primary_percentile)]
    primary_thr = float(np.percentile(risk, int(primary_percentile)))
    skeleton = skeletonize(primary > 0).astype(np.uint8)
    # distance to nearest skeleton pixel
    distance = cv2.distanceTransform((1 - skeleton).astype(np.uint8), cv2.DIST_L2, 5)
    distance = np.clip(distance, 0.0, float(distance_clip_px)).astype(np.float32)
    return masks, skeleton, distance, primary_thr
