from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
from skimage.feature import blob_log


@dataclass
class Candidate:
    candidate_id: int
    x: float
    y: float
    radius: float
    diameter_px: float
    polarity: str  # "bright" or "dark"
    log_response: Optional[float]


def _to_blob_input(img: np.ndarray) -> np.ndarray:
    """Ensure float32 in [0,1] for blob_log."""
    arr = np.asarray(img, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _run_blob_log(
    image: np.ndarray,
    min_sigma: float,
    max_sigma: float,
    num_sigma: int,
    threshold_rel: float,
    threshold_abs,
    overlap: float,
) -> np.ndarray:
    """Return blobs array with rows (y, x, sigma)."""
    kwargs = dict(
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        overlap=overlap,
    )
    if threshold_abs is not None:
        kwargs["threshold"] = float(threshold_abs)
    else:
        # threshold_rel is supported in skimage>=0.18
        kwargs["threshold_rel"] = float(threshold_rel)
    try:
        return blob_log(image, **kwargs)
    except TypeError:
        kwargs.pop("threshold_rel", None)
        kwargs.setdefault("threshold", 0.02)
        return blob_log(image, **kwargs)


def _nms(candidates: List[Candidate], nms_distance: float) -> List[Candidate]:
    if not candidates:
        return []
    # Sort by response desc (None last)
    cs = sorted(
        candidates,
        key=lambda c: (-(c.log_response if c.log_response is not None else -1e9)),
    )
    kept: List[Candidate] = []
    d2 = nms_distance * nms_distance
    for c in cs:
        ok = True
        for k in kept:
            if (c.x - k.x) ** 2 + (c.y - k.y) ** 2 < d2:
                ok = False
                break
        if ok:
            kept.append(c)
    return kept


def detect_blob_candidates(img: np.ndarray, config: dict) -> List[Candidate]:
    """Detect bright and dark blobs on combined_fused_delined image."""
    bd = config.get("blob_detection", {}) or {}
    cf = config.get("candidate_filter", {}) or {}

    detect_bright = bool(bd.get("detect_bright", True))
    detect_dark = bool(bd.get("detect_dark", True))
    min_d = float(bd.get("min_diameter_px", 3))
    max_d = float(bd.get("max_diameter_px", 10))
    log_sigma_min = float(bd.get("log_sigma_min", max(0.5, min_d / 2.0 / 1.414)))
    log_sigma_max = float(bd.get("log_sigma_max", max_d / 2.0 / 1.414))
    num_sigma = int(bd.get("num_sigma", 8))
    threshold_abs = bd.get("threshold_abs", None)
    threshold_rel = float(bd.get("threshold_rel", 0.04))
    overlap = float(bd.get("overlap", 0.5))
    exclude_border = int(bd.get("exclude_border_px", 32))

    H, W = img.shape[:2]
    image = _to_blob_input(img)

    raw_candidates: List[Candidate] = []
    if detect_bright:
        blobs = _run_blob_log(
            image, log_sigma_min, log_sigma_max, num_sigma, threshold_rel, threshold_abs, overlap
        )
        for b in blobs:
            y, x, sigma = float(b[0]), float(b[1]), float(b[2])
            radius = sigma * np.sqrt(2.0)
            diameter = 2.0 * radius
            if diameter < min_d or diameter > max_d:
                continue
            if (
                x < exclude_border
                or y < exclude_border
                or x >= W - exclude_border
                or y >= H - exclude_border
            ):
                continue
            # Use raw image intensity at center as a proxy response
            ix, iy = int(round(x)), int(round(y))
            ix = max(0, min(W - 1, ix))
            iy = max(0, min(H - 1, iy))
            raw_candidates.append(
                Candidate(
                    candidate_id=-1,
                    x=x,
                    y=y,
                    radius=radius,
                    diameter_px=diameter,
                    polarity="bright",
                    log_response=float(image[iy, ix]),
                )
            )

    if detect_dark:
        inv = 1.0 - image
        blobs = _run_blob_log(
            inv, log_sigma_min, log_sigma_max, num_sigma, threshold_rel, threshold_abs, overlap
        )
        for b in blobs:
            y, x, sigma = float(b[0]), float(b[1]), float(b[2])
            radius = sigma * np.sqrt(2.0)
            diameter = 2.0 * radius
            if diameter < min_d or diameter > max_d:
                continue
            if (
                x < exclude_border
                or y < exclude_border
                or x >= W - exclude_border
                or y >= H - exclude_border
            ):
                continue
            ix, iy = int(round(x)), int(round(y))
            ix = max(0, min(W - 1, ix))
            iy = max(0, min(H - 1, iy))
            raw_candidates.append(
                Candidate(
                    candidate_id=-1,
                    x=x,
                    y=y,
                    radius=radius,
                    diameter_px=diameter,
                    polarity="dark",
                    log_response=float(inv[iy, ix]),
                )
            )

    nms_d = float(cf.get("nms_distance_px", 5))
    kept = _nms(raw_candidates, nms_d)

    max_cand = int(cf.get("max_candidates", 500))
    if len(kept) > max_cand:
        kept = kept[:max_cand]

    # Assign ids
    for i, c in enumerate(kept):
        c.candidate_id = i + 1
    return kept


def candidates_to_dataframe(candidates: List[Candidate]):
    import pandas as pd

    return pd.DataFrame([asdict(c) for c in candidates])
