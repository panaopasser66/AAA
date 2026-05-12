"""Manual review ranking — Stage 3A.

Implements the exact scoring formula from task_stage3.md §4.

  review_priority_score =
      0.30 * diameter_closeness_score
    + 0.25 * shape_score
    + 0.30 * persistence_score
    - 0.10 * line_penalty_soft
    - 0.05 * texture_penalty_soft

Each sub-score is in [0,1]. Penalties are also in [0,1] (so soft, never zeroing
out a candidate). `review_reason` is a short human-readable tag of why this
score landed where it did.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Sub-scores
# ---------------------------------------------------------------------------

def diameter_closeness_score(diameter_px: float, target: float = 5.5) -> float:
    if not np.isfinite(diameter_px):
        return 0.0
    return float(np.exp(-abs(diameter_px - target) / 2.0))


def circularity_subscore(circ: float, low: float = 0.5, high: float = 1.4) -> float:
    """1.0 within [low, high], linearly decays to 0 outside; NaN → 0."""
    if circ is None or not np.isfinite(circ):
        return 0.0
    c = float(circ)
    if low <= c <= high:
        return 1.0
    # Decay over half the in-range width on each side
    band = max(0.1, (high - low) * 0.5)
    if c < low:
        return float(max(0.0, 1.0 - (low - c) / band))
    return float(max(0.0, 1.0 - (c - high) / band))


def aspect_ratio_subscore(aspect: float, good: float = 1.5, mid: float = 2.5) -> float:
    if aspect is None or not np.isfinite(aspect):
        return 0.3
    a = float(aspect)
    if a <= good:
        return 1.0
    if a <= mid:
        return float(1.0 - 0.5 * (a - good) / max(1e-6, mid - good))  # 1 → 0.5
    return float(max(0.0, 0.5 - 0.5 * (a - mid) / max(1e-6, mid)))  # 0.5 → 0 as a → 2*mid


def anisotropy_subscore(aniso: float, good: float = 2.0, mid: float = 3.5) -> float:
    if aniso is None or not np.isfinite(aniso):
        return 0.3
    a = float(aniso)
    if a <= good:
        return 1.0
    if a <= mid:
        return float(1.0 - 0.5 * (a - good) / max(1e-6, mid - good))
    return float(max(0.0, 0.5 - 0.5 * (a - mid) / max(1e-6, mid)))


def shape_score(circ: float, aspect: float, aniso: float, thresholds: Dict[str, float]) -> float:
    c = circularity_subscore(
        circ,
        thresholds.get("circularity_low", 0.5),
        thresholds.get("circularity_high", 1.4),
    )
    a = aspect_ratio_subscore(
        aspect,
        thresholds.get("aspect_ratio_good", 1.5),
        thresholds.get("aspect_ratio_mid", 2.5),
    )
    n = anisotropy_subscore(
        aniso,
        thresholds.get("anisotropy_good", 2.0),
        thresholds.get("anisotropy_mid", 3.5),
    )
    return float(c * a * n)


def persistence_score(
    spot_present: int, uncovered: int, weak_threshold: int = 2,
    weak_penalty: float = 0.5,
) -> float:
    sp = float(spot_present or 0)
    un = float(uncovered or 0)
    ratio = sp / max(un, 1.0)
    if sp < weak_threshold:
        ratio *= (1.0 - weak_penalty)
    return float(min(max(ratio, 0.0), 1.0))


def line_penalty_soft(
    dist_static: float, dist_combined: float, mean_lr: float,
    scale_static: float = 16.0, scale_combined: float = 12.0,
) -> float:
    """Soft penalty for line proximity. All three components in [0,1]; mean
    of the three. Far from line → low penalty.
    """
    ds = 0.0 if (dist_static is None or not np.isfinite(dist_static)) else float(dist_static)
    dc = 0.0 if (dist_combined is None or not np.isfinite(dist_combined)) else float(dist_combined)
    mlr = 0.0 if (mean_lr is None or not np.isfinite(mean_lr)) else float(mean_lr)
    pen_static = float(np.exp(-ds / scale_static))
    pen_comb = float(np.exp(-dc / scale_combined))
    pen_lr = float(np.clip(mlr, 0.0, 1.0))
    return float((pen_static + pen_comb + pen_lr) / 3.0)


def texture_penalty_soft(aniso: float, aspect: float) -> float:
    """Penalty for texture-like context."""
    a = 0.0 if (aniso is None or not np.isfinite(aniso)) else float(aniso)
    ar = 0.0 if (aspect is None or not np.isfinite(aspect)) else float(aspect)
    p1 = float(np.clip((a - 2.0) / 6.0, 0.0, 1.0))
    p2 = float(np.clip((ar - 1.5) / 3.0, 0.0, 1.0))
    return float((p1 + p2) / 2.0)


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------

REVIEW_COLUMNS = [
    "diameter_closeness_score",
    "shape_score",
    "persistence_score",
    "line_penalty_soft",
    "texture_penalty_soft",
    "review_priority_score",
    "review_reason",
]


def _reason(d_close, shape, persistence, line_pen, tex_pen) -> str:
    parts = []
    if d_close >= 0.8:
        parts.append("size_match")
    elif d_close < 0.4:
        parts.append("size_off")
    if shape >= 0.6:
        parts.append("shape_ok")
    elif shape < 0.2:
        parts.append("shape_weak")
    if persistence >= 0.5:
        parts.append("multi_frame_evidence")
    elif persistence < 0.2:
        parts.append("low_evidence")
    if line_pen >= 0.5:
        parts.append("near_line")
    if tex_pen >= 0.5:
        parts.append("textured")
    return "|".join(parts) if parts else "neutral"


def rank_candidates(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Return df with new score columns; sorted by review_priority_score desc."""
    mr = config.get("manual_review", {}) or {}
    target_d = float(mr.get("target_diameter_px", 5.5))
    thresholds = mr.get("shape_subscore_thresholds", {}) or {}
    w = mr.get("weights", {}) or {}
    w_d = float(w.get("diameter_closeness", 0.30))
    w_shape = float(w.get("shape", 0.25))
    w_pers = float(w.get("persistence", 0.30))
    w_line = float(w.get("line_penalty_soft", 0.10))
    w_tex = float(w.get("texture_penalty_soft", 0.05))
    pers_cfg = mr.get("persistence", {}) or {}
    weak_threshold = int(pers_cfg.get("min_spot_present_when_uncovered_for_full_credit", 2))
    weak_penalty = float(pers_cfg.get("weak_evidence_penalty", 0.5))

    if df.empty:
        out = df.copy()
        for c in REVIEW_COLUMNS:
            out[c] = []
        return out

    d_close = df["diameter_px"].apply(lambda x: diameter_closeness_score(x, target_d))
    sh = df.apply(
        lambda r: shape_score(
            r.get("circularity", float("nan")),
            r.get("aspect_ratio", float("nan")),
            r.get("anisotropy", float("nan")),
            thresholds,
        ),
        axis=1,
    )
    pers = df.apply(
        lambda r: persistence_score(
            int(r.get("spot_present_when_uncovered", 0) or 0),
            int(r.get("uncovered_frame_count", 0) or 0),
            weak_threshold, weak_penalty,
        ),
        axis=1,
    )
    line_pen = df.apply(
        lambda r: line_penalty_soft(
            r.get("distance_to_refined_static_line_skeleton", float("nan")),
            r.get("distance_to_refined_combined_line_exclusion", float("nan")),
            r.get("mean_line_risk_in_patch", float("nan")),
        ),
        axis=1,
    )
    tex_pen = df.apply(
        lambda r: texture_penalty_soft(
            r.get("anisotropy", float("nan")),
            r.get("aspect_ratio", float("nan")),
        ),
        axis=1,
    )

    score = w_d * d_close + w_shape * sh + w_pers * pers - w_line * line_pen - w_tex * tex_pen

    out = df.copy()
    out["diameter_closeness_score"] = d_close.astype(np.float32)
    out["shape_score"] = sh.astype(np.float32)
    out["persistence_score"] = pers.astype(np.float32)
    out["line_penalty_soft"] = line_pen.astype(np.float32)
    out["texture_penalty_soft"] = tex_pen.astype(np.float32)
    out["review_priority_score"] = score.astype(np.float32)
    out["review_reason"] = [
        _reason(a, b, c, d, e)
        for a, b, c, d, e in zip(d_close, sh, pers, line_pen, tex_pen)
    ]
    out = out.sort_values("review_priority_score", ascending=False).reset_index(drop=True)
    return out


def nms_in_place(df: pd.DataFrame, min_distance_px: float = 4.0,
                 score_col: str = "review_priority_score") -> pd.DataFrame:
    """Greedy NMS preserving highest-score points first."""
    if df.empty:
        return df
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    xs = df["x"].to_numpy()
    ys = df["y"].to_numpy()
    keep: List[int] = []
    d2 = min_distance_px * min_distance_px
    for i in range(len(df)):
        ok = True
        for j in keep:
            if (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 < d2:
                ok = False
                break
        if ok:
            keep.append(i)
    return df.iloc[keep].reset_index(drop=True)


def labels_template(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Build a manual-labels CSV with empty rating columns."""
    out = df.copy()
    out.insert(0, "review_rank", np.arange(1, len(out) + 1))
    for c in columns:
        if c not in out.columns:
            out[c] = ""
    return out
