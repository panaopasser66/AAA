from __future__ import annotations

import os
from typing import Dict, List

import cv2
import numpy as np

from .io_utils import ensure_dir, robust_rescale_for_preview, save_png


def _crop_around(img: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    H, W = img.shape[:2]
    half = size // 2
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x1 = x0 + size
    y1 = y0 + size
    # Clamp and pad
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - W)
    pad_bottom = max(0, y1 - H)
    x0c = max(0, x0)
    y0c = max(0, y0)
    x1c = min(W, x1)
    y1c = min(H, y1)
    patch = img[y0c:y1c, x0c:x1c]
    if pad_left or pad_top or pad_right or pad_bottom:
        patch = np.pad(
            patch,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="edge",
        )
    return patch


def _to_preview(crop: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    return robust_rescale_for_preview(crop, p_low=p_low, p_high=p_high)


def _draw_circle(crop_u8: np.ndarray, radius_px: float, color=255) -> np.ndarray:
    out = crop_u8.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        color_bgr = (0, 255, 255)
    else:
        color_bgr = color
    cx = out.shape[1] // 2
    cy = out.shape[0] // 2
    r = max(2, int(round(radius_px)))
    cv2.circle(out, (cx, cy), r, color_bgr, 1, cv2.LINE_AA)
    return out


def _label_image(img_bgr: np.ndarray, title: str) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    out = np.full((h + 18, w, 3), 32, dtype=np.uint8)
    out[18:, :, :] = img_bgr if img_bgr.ndim == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        out,
        title,
        (2, 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def export_candidate_crop_sheet(
    candidate_row,
    images_dict: Dict[str, np.ndarray],
    out_path: str,
    config: dict,
) -> str:
    """Export one contact sheet for a candidate.

    images_dict ordering controls layout. Keys included as titles.
    """
    ce = config.get("crop_export", {}) or {}
    size = int(ce.get("crop_size", 80))
    draw_circle = bool(ce.get("draw_candidate_circle", True))

    x = float(candidate_row["x"])
    y = float(candidate_row["y"])
    radius = float(candidate_row.get("radius_px", 3.0))

    titles = list(images_dict.keys())
    crops_bgr: List[np.ndarray] = []
    for title in titles:
        img = images_dict[title]
        crop = _crop_around(img, x, y, size)
        prev = _to_preview(crop)
        if draw_circle:
            prev = _draw_circle(prev, radius)
        else:
            prev = cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR) if prev.ndim == 2 else prev
        crops_bgr.append(_label_image(prev, title))

    # Build grid
    cols = min(4, len(crops_bgr))
    rows = (len(crops_bgr) + cols - 1) // cols
    cell_h, cell_w = crops_bgr[0].shape[:2]
    pad = 4
    sheet = np.full(
        (rows * cell_h + pad * (rows + 1), cols * cell_w + pad * (cols + 1), 3),
        16,
        dtype=np.uint8,
    )
    for i, img in enumerate(crops_bgr):
        r, c = divmod(i, cols)
        y0 = pad + r * (cell_h + pad)
        x0 = pad + c * (cell_w + pad)
        sheet[y0 : y0 + cell_h, x0 : x0 + cell_w, :] = img

    header_h = 22
    header = np.full((header_h, sheet.shape[1], 3), 0, dtype=np.uint8)
    score = candidate_row.get("combined_score", 0.0)
    polarity = candidate_row.get("polarity", "?")
    cid = candidate_row.get("candidate_id", 0)
    cls = candidate_row.get("class_suggestion", "")
    text = (
        f"id={cid:04d} x={x:.0f} y={y:.0f} d={candidate_row.get('diameter_px',0):.1f}px "
        f"pol={polarity} class={cls} score={score:.3f}"
    )
    cv2.putText(
        header,
        text,
        (4, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    full = np.vstack([header, sheet])

    ensure_dir(os.path.dirname(out_path))
    ok = cv2.imwrite(out_path, full)
    if not ok:
        raise IOError(f"Failed to write crop sheet: {out_path}")
    return out_path
