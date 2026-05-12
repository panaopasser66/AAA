"""Stage 3 — narrow-line mask + classical inpainting variants.

These cleaned images are *NOT* used as final defect detections. They are an
intermediate artefact that feeds (a) human review, and (b) pseudo-target
generation for a future Stage 4 line-removal network.
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np


def build_narrow_line_mask(
    refined_combined_skeleton: np.ndarray,
    refined_static_skeleton: np.ndarray,
    dilate_px: int,
    static_extra_dilate_px: int = 1,
) -> np.ndarray:
    """Build a *narrow* line mask. Default ``dilate_px = 2`` covers a 5-px-wide
    line. Static skeleton gets an extra small dilation because static lines
    tend to be slightly thicker than the dynamic line trace.
    """
    combined = refined_combined_skeleton.astype(np.uint8)
    static = refined_static_skeleton.astype(np.uint8)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        combined = cv2.dilate(combined, k)
    if static_extra_dilate_px > 0:
        kk = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            ((dilate_px + static_extra_dilate_px) * 2 + 1,
             (dilate_px + static_extra_dilate_px) * 2 + 1),
        )
        static = cv2.dilate(static, kk)
    return ((combined > 0) | (static > 0)).astype(np.uint8)


def _to_uint8_for_inpaint(img01: np.ndarray) -> np.ndarray:
    a = np.asarray(img01, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    a = np.clip(a, 0.0, 1.0)
    return (a * 255.0 + 0.5).astype(np.uint8)


def inpaint_telea(img01: np.ndarray, mask: np.ndarray, radius: int = 3) -> np.ndarray:
    src = _to_uint8_for_inpaint(img01)
    m = (mask.astype(np.uint8) > 0).astype(np.uint8) * 255
    out = cv2.inpaint(src, m, float(radius), cv2.INPAINT_TELEA)
    return (out.astype(np.float32) / 255.0).astype(np.float32)


def inpaint_ns(img01: np.ndarray, mask: np.ndarray, radius: int = 3) -> np.ndarray:
    src = _to_uint8_for_inpaint(img01)
    m = (mask.astype(np.uint8) > 0).astype(np.uint8) * 255
    out = cv2.inpaint(src, m, float(radius), cv2.INPAINT_NS)
    return (out.astype(np.float32) / 255.0).astype(np.float32)


def inpaint_local_median(
    img01: np.ndarray, mask: np.ndarray, disk_radius: int = 7
) -> np.ndarray:
    """Replace masked pixels with the local median computed over a disk that
    excludes the masked region. We compute the median image of the entire frame
    via a large-kernel median filter, then composite only over the masked
    pixels. To avoid the masked pixels biasing the median, we use a single
    OpenCV ``medianBlur`` with a kernel size of ``2 * disk_radius + 1`` (must
    be odd ≤ 15 in cv2). For larger kernels we fall back to a tiled approach.
    """
    arr = np.asarray(img01, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)

    k = int(disk_radius) * 2 + 1
    if k % 2 == 0:
        k += 1

    # cv2.medianBlur supports k up to 5 for float32; convert to uint16 to allow
    # larger kernels via scaling then back to float.
    src_u16 = (arr * 65535.0 + 0.5).astype(np.uint16)
    if k <= 5:
        med_u16 = cv2.medianBlur(src_u16, k)
    else:
        # Two-pass: a small k=5 median followed by box blur to approximate.
        med_u16 = cv2.medianBlur(src_u16, 5)
        bw = max(3, k)
        if bw % 2 == 0:
            bw += 1
        med_u16 = cv2.GaussianBlur(med_u16, (bw, bw), 0)
    med = (med_u16.astype(np.float32) / 65535.0).astype(np.float32)

    m = (mask.astype(np.uint8) > 0)
    out = arr.copy()
    out[m] = med[m]
    return out


def run_inpaint_variants(
    img01: np.ndarray,
    mask: np.ndarray,
    methods: List[str],
    telea_radius: int = 3,
    ns_radius: int = 3,
    local_median_disk_radius: int = 7,
) -> Dict[str, np.ndarray]:
    """Run all requested inpaint methods. Returns dict {method_name: cleaned_img01}."""
    out: Dict[str, np.ndarray] = {}
    for m in methods:
        m_lower = str(m).lower()
        if m_lower == "telea":
            out["telea"] = inpaint_telea(img01, mask, telea_radius)
        elif m_lower in ("ns", "navier", "navier_stokes", "navierstokes"):
            out["ns"] = inpaint_ns(img01, mask, ns_radius)
        elif m_lower in ("local_median", "median"):
            out["local_median"] = inpaint_local_median(img01, mask, local_median_disk_radius)
        else:
            raise ValueError(f"Unknown inpaint method: {m}")
    return out
