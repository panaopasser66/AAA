"""Stage 2h - Visual Line Hide v5 (deep mask hybrid).

v5 swaps the rule-based sato + black_line_score mask of v3/v4 for a
generic *line probability* map and keeps the rest of the v4 pipeline
(orientation-aware bg, core+halo + residual cleanup).

Mask sources (priority order):
  1. --line-prob PATH  (a .tif probability map in [0, 1]; preferred input
     once Route B's U-Net is available)
  2. --line-mask PATH  (a binary .png; treated as line_prob = 0/1)
  3. fallback synthesis: line_prob = w_sato * sato_norm
                                   + w_score * sigmoid(black_score - τ)
     This lets the v5 pipeline run today without any trained model and
     produces output shapes a real U-Net would also produce — so the
     U-Net hook is a pure data swap.

Core / halo construction:
  core = directional_closing(line_prob > core_threshold)   # along ridge
  halo = (dilate(line_prob > halo_threshold, halo_dilate)) & ~core

Suppression (carried from v4):
  - core: hard replace with orientation-aware bg (median over ±normal
    samples at distances {wc+2, wc+4, wc+8, wc+12}); side-band fallback
  - halo: residual attenuation with keep_halo
  - feather + post-feather hard core overwrite
  - protect split: protect_true_blob preserved, protect_on_line allowed
  - optional v4-style residual cleanup pass (kept on by default)

Sweep (kept tight, 10 variants):
  5 (core_th, halo_th) pairs x 2 closing_len = 10.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage
from skimage import measure
from skimage.morphology import binary_closing, binary_dilation, disk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def load_tif(path: Path) -> np.ndarray:
    return tifffile.imread(str(path))


def load_mask(path: Path) -> np.ndarray:
    img = Image.open(str(path)).convert("L")
    return np.asarray(img) > 127


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(str(path))


def save_uint8(path: Path, arr8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr8).save(str(path))


def save_tif(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr, compression="zlib")


# --------------------------------------------------------------------------- #
# Mask sources for line probability
# --------------------------------------------------------------------------- #

def load_line_prob_from_tif(path: Path) -> np.ndarray:
    """Load a line probability map. Accepts floats in [0,1] or any range;
    everything is rescaled to [0,1] by min-max if needed."""
    arr = tifffile.imread(str(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.nanmin(arr[finite]))
    hi = float(np.nanmax(arr[finite]))
    if hi <= 1.0 and lo >= 0.0:
        return np.clip(arr, 0.0, 1.0).astype(np.float32)
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).clip(0.0, 1.0).astype(np.float32)


def load_line_prob_from_mask(path: Path) -> np.ndarray:
    return load_mask(path).astype(np.float32)


def synthesize_line_prob_from_sato(
    sato_response: np.ndarray, img_f: np.ndarray,
    logit_gain: float = 6.0, logit_bias: float = 2.0,
    sigma_bg: float = 20.0, sigma_noise: float = 8.0,
    smooth_sigma: float = 0.8,
) -> tuple[np.ndarray, dict]:
    """Fallback: build a line probability map from sato + black-line score.

    Designed so the OUTPUT SHAPE matches a Route B U-Net's output (a float32
    image in [0, 1] with high values on dark ridges, low on background).

    Composition::

        raw = sato_norm * max(black_score, 0)
        logit = logit_gain * raw - logit_bias
        prob = sigmoid(logit)

    Sigmoid centered at `raw = logit_bias / logit_gain` so background pixels
    (low sato * low score) get prob ~ sigmoid(-logit_bias) ≈ 0.1; strong
    dark ridges (sato_norm ≈ 1, score ≈ 2-3 σ) saturate near 1. Defaults
    (gain=6, bias=2) put the transition near raw=0.33, matching the user's
    spec where 0.5/0.6/0.7 selects core and 0.25/0.35/0.45 selects halo.
    """
    sato_p99 = float(np.percentile(sato_response, 99))
    sato_norm = np.clip(sato_response / max(sato_p99, 1e-6), 0.0, 1.0).astype(np.float32)

    # Black-line score (same definition as v2/v3/v4)
    bg_low = ndimage.gaussian_filter(img_f, sigma=sigma_bg).astype(np.float32)
    dark_residual = (bg_low - img_f).astype(np.float32)
    res = (img_f - bg_low).astype(np.float32)
    med = float(np.median(res))
    global_mad = float(np.median(np.abs(res - med)) * 1.4826) + 1e-6
    local_mean = ndimage.gaussian_filter(res, sigma=sigma_noise)
    centered = res - local_mean
    local_var = ndimage.gaussian_filter(centered * centered, sigma=sigma_noise)
    local_noise = np.sqrt(np.maximum(local_var, 1e-12)).astype(np.float32)
    noise = np.maximum(local_noise, global_mad)
    score = np.clip(dark_residual, 0.0, None) / np.maximum(noise, 1e-6)

    raw = sato_norm * score
    logit = logit_gain * raw - logit_bias
    prob = (1.0 / (1.0 + np.exp(-logit))).astype(np.float32)
    if smooth_sigma > 0:
        prob = ndimage.gaussian_filter(prob, sigma=smooth_sigma).astype(np.float32)
    prob = np.clip(prob, 0.0, 1.0)
    stats = {
        "source": "synthesized",
        "composition": "sigmoid(gain*sato_norm*score - bias)",
        "logit_gain": logit_gain, "logit_bias": logit_bias,
        "smooth_sigma": smooth_sigma,
        "sato_p99": sato_p99,
        "global_mad": global_mad,
        "prob_p50": float(np.percentile(prob, 50)),
        "prob_p90": float(np.percentile(prob, 90)),
        "prob_p99": float(np.percentile(prob, 99)),
        "prob_coverage_gt_0.5_pct": float((prob > 0.5).mean() * 100.0),
        "prob_coverage_gt_0.6_pct": float((prob > 0.6).mean() * 100.0),
        "prob_coverage_gt_0.7_pct": float((prob > 0.7).mean() * 100.0),
        "prob_coverage_gt_0.25_pct": float((prob > 0.25).mean() * 100.0),
        "prob_coverage_gt_0.35_pct": float((prob > 0.35).mean() * 100.0),
        "prob_coverage_gt_0.45_pct": float((prob > 0.45).mean() * 100.0),
    }
    return prob, stats


def load_or_synthesize_line_prob(
    *,
    line_prob_path: Path | None,
    line_mask_path: Path | None,
    sato_response_path: Path | None,
    img_f: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Resolve a line probability map from the three supported sources.

    Returns (line_prob, info_dict). The info_dict records which source was
    used and any parameters so the result is reproducible."""
    if line_prob_path is not None and Path(line_prob_path).exists():
        prob = load_line_prob_from_tif(Path(line_prob_path))
        info = {"source": "line_prob", "path": str(line_prob_path),
                "prob_p99": float(np.percentile(prob, 99))}
        return prob, info
    if line_mask_path is not None and Path(line_mask_path).exists():
        prob = load_line_prob_from_mask(Path(line_mask_path))
        info = {"source": "line_mask", "path": str(line_mask_path)}
        return prob, info
    if sato_response_path is not None and Path(sato_response_path).exists():
        sato = load_tif(Path(sato_response_path)).astype(np.float32)
        prob, info = synthesize_line_prob_from_sato(sato, img_f)
        info["sato_path"] = str(sato_response_path)
        return prob, info
    raise FileNotFoundError(
        "No line-mask source available. Pass --line-prob or --line-mask, "
        "or ensure outputs/prepare/responses/response_sato.tif exists for "
        "the fallback synthesis."
    )


# --------------------------------------------------------------------------- #
# Structure tensor / side-band / directional bg (carried from v3 / v4)
# --------------------------------------------------------------------------- #

def compute_normal_direction(
    img_f: np.ndarray, sigma_pre: float = 1.0, sigma_tensor: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img_s = ndimage.gaussian_filter(img_f, sigma=sigma_pre)
    gx = ndimage.sobel(img_s, axis=1)
    gy = ndimage.sobel(img_s, axis=0)
    Jxx = ndimage.gaussian_filter(gx * gx, sigma=sigma_tensor)
    Jxy = ndimage.gaussian_filter(gx * gy, sigma=sigma_tensor)
    Jyy = ndimage.gaussian_filter(gy * gy, sigma=sigma_tensor)
    theta_n = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)
    cos_n = np.cos(theta_n).astype(np.float32)
    sin_n = np.sin(theta_n).astype(np.float32)
    tmp = np.sqrt(np.maximum((Jxx - Jyy) ** 2 + 4.0 * Jxy ** 2, 0.0))
    lam1 = 0.5 * (Jxx + Jyy + tmp)
    lam2 = 0.5 * (Jxx + Jyy - tmp)
    coh = (lam1 - lam2) / (lam1 + lam2 + 1e-12)
    coh = np.clip(coh, 0.0, 1.0).astype(np.float32)
    return cos_n, sin_n, coh


def side_band_bg(
    img_f: np.ndarray, edit_mask: np.ndarray, protect_mask: np.ndarray,
    side_radius: int, sigma: float,
) -> np.ndarray:
    outer = binary_dilation(edit_mask, disk(side_radius))
    side_band = outer & ~edit_mask & ~protect_mask
    weight = side_band.astype(np.float32)
    num = ndimage.gaussian_filter(img_f * weight, sigma=sigma)
    den = ndimage.gaussian_filter(weight, sigma=sigma)
    bg = num / np.maximum(den, 1e-6)
    fb_sigma = max(sigma * 2.0, 30.0)
    fb_num = ndimage.gaussian_filter(img_f * weight, sigma=fb_sigma)
    fb_den = ndimage.gaussian_filter(weight, sigma=fb_sigma)
    fb = fb_num / np.maximum(fb_den, 1e-6)
    no_support = den < 1e-3
    if no_support.any():
        bg = np.where(no_support, fb, bg)
    return bg.astype(np.float32)


def directional_bg(
    img_f: np.ndarray, forbid_mask: np.ndarray, side_bg_fallback: np.ndarray,
    cos_n: np.ndarray, sin_n: np.ndarray, distances: list[float],
    reduction: str = "median",
) -> tuple[np.ndarray, np.ndarray]:
    H, W = img_f.shape
    y_idx, x_idx = np.mgrid[0:H, 0:W].astype(np.float32)
    forbid_f = forbid_mask.astype(np.float32)
    samples, valids = [], []
    for d in distances:
        for sign in (+1.0, -1.0):
            sy = y_idx + sign * d * sin_n
            sx = x_idx + sign * d * cos_n
            coords = np.stack([sy, sx], axis=0)
            s = ndimage.map_coordinates(img_f, coords, order=1, mode="reflect").astype(np.float32)
            f = ndimage.map_coordinates(forbid_f, coords, order=0, mode="constant", cval=1.0)
            v = (f < 0.5).astype(np.float32)
            samples.append(s); valids.append(v)
    samp = np.stack(samples, axis=0)
    val = np.stack(valids, axis=0)
    n_valid = val.sum(axis=0)
    has_dir = n_valid > 0
    if reduction == "median":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            masked = np.where(val.astype(bool), samp, np.nan)
            bg_dir = np.nanmedian(masked, axis=0)
            bg_dir = np.where(np.isnan(bg_dir), 0.0, bg_dir)
    else:
        total = (samp * val).sum(axis=0)
        bg_dir = total / np.maximum(n_valid, 1.0)
    bg = np.where(has_dir, bg_dir, side_bg_fallback).astype(np.float32)
    return bg, has_dir


# --------------------------------------------------------------------------- #
# Multi-angle line SE + directional closing (from v4)
# --------------------------------------------------------------------------- #

def make_line_se(length: int, width: int, angle_deg: float) -> np.ndarray:
    if length <= 1:
        return np.ones((max(1, width), max(1, width)), dtype=bool)
    size = int(np.ceil(np.hypot(length, max(width, 1)))) + 2
    if size % 2 == 0:
        size += 1
    se = np.zeros((size, size), dtype=bool)
    cy = cx = size // 2
    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    half_l = (length - 1) / 2.0
    half_w = (width - 1) / 2.0
    d_steps = max(length, 3) * 2
    w_steps = max(width, 1)
    for d in np.linspace(-half_l, half_l, d_steps):
        for w in (np.linspace(-half_w, half_w, w_steps) if w_steps > 1 else [0.0]):
            y = int(round(cy + d * sin_t + w * cos_t))
            x = int(round(cx + d * cos_t - w * sin_t))
            if 0 <= y < size and 0 <= x < size:
                se[y, x] = True
    return se


def directional_closing(
    mask: np.ndarray, length: int, width: int,
    angles_deg=(0, 30, 60, 90, 120, 150),
) -> np.ndarray:
    if length <= 0:
        return mask.copy()
    out = mask.copy()
    for ang in angles_deg:
        se = make_line_se(length=length, width=width, angle_deg=ang)
        out |= binary_closing(mask, footprint=se)
    return out


# --------------------------------------------------------------------------- #
# Build core / halo from line_prob
# --------------------------------------------------------------------------- #

def build_core_halo_from_prob(
    line_prob: np.ndarray, core_threshold: float, halo_threshold: float,
    protect_mask: np.ndarray, closing_len: int, closing_width: int,
    halo_dilate: int,
) -> tuple[np.ndarray, np.ndarray]:
    core = line_prob > core_threshold
    if closing_len > 0:
        core = directional_closing(core, closing_len, closing_width)
    halo_seed = line_prob > halo_threshold
    if halo_dilate > 0:
        halo_seed = binary_dilation(halo_seed, disk(halo_dilate))
    halo = halo_seed & ~core
    core = core & ~protect_mask
    halo = halo & ~protect_mask
    return core, halo


# --------------------------------------------------------------------------- #
# Protect mask split (from v4)
# --------------------------------------------------------------------------- #

def split_protect_mask(
    protect_mask: np.ndarray, broad_line_mask: np.ndarray,
    dilate_radius: int = 3, overlap_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, dict]:
    broad_dilated = binary_dilation(broad_line_mask, disk(dilate_radius))
    lbl = measure.label(protect_mask, connectivity=2)
    protect_true = np.zeros_like(protect_mask, dtype=bool)
    protect_on_line = np.zeros_like(protect_mask, dtype=bool)
    n_true = n_on_line = 0
    for region in measure.regionprops(lbl):
        coords = region.coords
        area = region.area
        on_line_count = int(broad_dilated[coords[:, 0], coords[:, 1]].sum())
        if on_line_count / max(area, 1) > overlap_threshold:
            protect_on_line[coords[:, 0], coords[:, 1]] = True
            n_on_line += 1
        else:
            protect_true[coords[:, 0], coords[:, 1]] = True
            n_true += 1
    return protect_true, protect_on_line, {
        "n_true_blob": n_true, "n_on_line": n_on_line,
        "true_coverage_pct": float(protect_true.mean() * 100.0),
        "on_line_coverage_pct": float(protect_on_line.mean() * 100.0),
    }


# --------------------------------------------------------------------------- #
# Suppression (v3-style)
# --------------------------------------------------------------------------- #

def feather_outer_alpha(edit_mask: np.ndarray, feather_radius: float) -> np.ndarray:
    if feather_radius <= 0:
        return edit_mask.astype(np.float32)
    dist_outside = ndimage.distance_transform_edt(~edit_mask)
    alpha = np.clip(1.0 - dist_outside / float(feather_radius), 0.0, 1.0)
    alpha[edit_mask] = 1.0
    return alpha.astype(np.float32)


def suppress(
    img_f: np.ndarray, bg: np.ndarray,
    core_mask: np.ndarray, halo_mask: np.ndarray,
    keep_core: float, keep_halo: float, feather_radius: float,
) -> np.ndarray:
    residual = img_f - bg
    out = img_f.astype(np.float32).copy()
    if halo_mask.any():
        out[halo_mask] = (bg[halo_mask] + keep_halo * residual[halo_mask]).astype(np.float32)
    if core_mask.any():
        out[core_mask] = (bg[core_mask] + keep_core * residual[core_mask]).astype(np.float32)
    edit = core_mask | halo_mask
    if feather_radius > 0 and edit.any():
        alpha = feather_outer_alpha(edit, feather_radius)
        out = alpha * out + (1.0 - alpha) * img_f
        if halo_mask.any():
            out[halo_mask] = (bg[halo_mask] + keep_halo * residual[halo_mask]).astype(np.float32)
        if core_mask.any():
            out[core_mask] = (bg[core_mask] + keep_core * residual[core_mask]).astype(np.float32)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Preview / panel helpers
# --------------------------------------------------------------------------- #

def percentile_range(img: np.ndarray, lo_p: float = 1.0,
                     hi_p: float = 99.0) -> tuple[float, float]:
    lo = float(np.percentile(img, lo_p))
    hi = float(np.percentile(img, hi_p))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    return lo, hi


def to_uint8(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    a = (arr.astype(np.float32) - lo) / (hi - lo)
    return (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def overlay_core_halo(
    gray8: np.ndarray, core: np.ndarray, halo: np.ndarray,
    core_color=(220, 30, 30), halo_color=(240, 170, 30), alpha: float = 0.55,
) -> np.ndarray:
    rgb = np.stack([gray8, gray8, gray8], axis=-1).astype(np.float32)
    for mask, color in [(halo, halo_color), (core, core_color)]:
        if not mask.any():
            continue
        for ch, val in enumerate(color):
            rgb[mask, ch] = alpha * val + (1.0 - alpha) * rgb[mask, ch]
    return rgb.astype(np.uint8)


def diff_heatmap(diff: np.ndarray, abs_scale: float) -> np.ndarray:
    base = 128.0
    s = float(max(abs_scale, 1e-6))
    a = np.clip(diff / s, -1.0, 1.0)
    rgb = np.full(diff.shape + (3,), base, dtype=np.float32)
    neg = a < 0
    pos = a > 0
    if neg.any():
        m = -a[neg] * base
        rgb[neg, 0] = base + m
        rgb[neg, 1] = base - 0.5 * m
        rgb[neg, 2] = base - 0.5 * m
    if pos.any():
        m = a[pos] * base
        rgb[pos, 2] = base + m
        rgb[pos, 1] = base - 0.5 * m
        rgb[pos, 0] = base - 0.5 * m
    return np.clip(rgb, 0, 255).astype(np.uint8)


def downscale_uint8(arr: np.ndarray, max_side: int = 520) -> np.ndarray:
    h, w = arr.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return arr
    scale = max_side / s
    img = Image.fromarray(arr)
    img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
    return np.asarray(img)


# --------------------------------------------------------------------------- #
# Zoom crops (auto-pick on sato or line_prob density)
# --------------------------------------------------------------------------- #

def select_zoom_crops(
    density: np.ndarray, n: int = 4, crop_size: int = 480, margin: int = 20,
) -> tuple[list[tuple[int, int]], int]:
    H, W = density.shape
    smoothed = ndimage.gaussian_filter(density.astype(np.float32), sigma=12.0)
    work = smoothed.copy()
    half = crop_size // 2
    work[:half + margin] = -np.inf
    work[-half - margin:] = -np.inf
    work[:, :half + margin] = -np.inf
    work[:, -half - margin:] = -np.inf
    crops: list[tuple[int, int]] = []
    for _ in range(n):
        idx = int(np.argmax(work))
        cy, cx = divmod(idx, W)
        if not np.isfinite(work[cy, cx]):
            break
        crops.append((int(cy), int(cx)))
        y0 = max(0, cy - crop_size); y1 = min(H, cy + crop_size)
        x0 = max(0, cx - crop_size); x1 = min(W, cx + crop_size)
        work[y0:y1, x0:x1] = -np.inf
    crops.sort(key=lambda c: (c[0], c[1]))
    return crops, half


def extract_crop(arr: np.ndarray, cy: int, cx: int, half: int) -> np.ndarray:
    y0 = cy - half; x0 = cx - half
    y1 = y0 + 2 * half; x1 = x0 + 2 * half
    return arr[y0:y1, x0:x1].copy()


# --------------------------------------------------------------------------- #
# Per-variant panel & sheets
# --------------------------------------------------------------------------- #

def save_summary_panel(
    out_path: Path, raw8: np.ndarray, sup8: np.ndarray,
    core: np.ndarray, halo: np.ndarray, removed: np.ndarray,
    abs_scale: float, title: str,
    zoom_centers: list[tuple[int, int]], zoom_half: int,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 9.0))
    axes[0, 0].imshow(raw8, cmap="gray"); axes[0, 0].set_title("raw")
    axes[0, 1].imshow(overlay_core_halo(raw8, core, halo))
    axes[0, 1].set_title("core (red) + halo (orange)")
    axes[0, 2].imshow(sup8, cmap="gray"); axes[0, 2].set_title("v5 suppressed")
    axes[0, 3].imshow(removed, cmap="seismic", vmin=-abs_scale, vmax=abs_scale)
    axes[0, 3].set_title("raw - v5")
    for j in range(4):
        axes[0, j].set_axis_off()
    for j in range(4):
        ax = axes[1, j]
        if j >= len(zoom_centers):
            ax.set_axis_off(); continue
        cy, cx = zoom_centers[j]
        ax.imshow(extract_crop(sup8, cy, cx, zoom_half), cmap="gray")
        ax.set_title(f"Z{j + 1} v5 (y={cy}, x={cx})", fontsize=8)
        ax.set_axis_off()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_contact_sheet(records: list[dict], sheet_path: Path,
                       cols: int = 5) -> None:
    n = len(records)
    if n == 0:
        return
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.7))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    for idx in range(rows * cols):
        ax = axes.flat[idx]
        if idx >= n:
            ax.set_axis_off(); continue
        rec = records[idx]
        ax.imshow(rec["thumbs"]["suppressed"], cmap="gray")
        ax.set_title(rec["name"], fontsize=6)
        ax.set_axis_off()
    fig.suptitle(f"contact_sheet — v5 suppressed previews ({n} variants)")
    fig.tight_layout()
    fig.savefig(sheet_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_compare_sheet(records: list[dict], sheet_path: Path,
                       title: str | None = None) -> None:
    n = len(records)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 3, figsize=(9.5, n * 2.0))
    if n == 1:
        axes = axes.reshape(1, -1)
    for r, rec in enumerate(records):
        axes[r, 0].imshow(rec["thumbs"]["overlay"])
        axes[r, 0].set_title(f"{rec['name']} | core+halo", fontsize=7)
        axes[r, 1].imshow(rec["thumbs"]["suppressed"], cmap="gray")
        axes[r, 1].set_title("v5 suppressed", fontsize=7)
        axes[r, 2].imshow(rec["thumbs"]["removed"])
        axes[r, 2].set_title("raw - v5", fontsize=7)
        for c in range(3):
            axes[r, c].set_axis_off()
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(sheet_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_zoom_compare_sheet(
    sheet_path: Path, raw8: np.ndarray,
    zoom_centers: list[tuple[int, int]], zoom_half: int,
    top_records: list[dict], suppressed8_lookup: dict[str, np.ndarray],
) -> None:
    n_zoom = len(zoom_centers)
    rows = 1 + len(top_records)
    cols = n_zoom
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    for j, (cy, cx) in enumerate(zoom_centers):
        axes[0, j].imshow(extract_crop(raw8, cy, cx, zoom_half), cmap="gray")
        axes[0, j].set_title(f"raw — Z{j + 1} (y={cy}, x={cx})", fontsize=8)
        axes[0, j].set_axis_off()
    for i, rec in enumerate(top_records):
        sup8 = suppressed8_lookup[rec["name"]]
        for j, (cy, cx) in enumerate(zoom_centers):
            axes[i + 1, j].imshow(extract_crop(sup8, cy, cx, zoom_half), cmap="gray")
            axes[i + 1, j].set_title(f"{rec['name']} — Z{j + 1}", fontsize=7)
            axes[i + 1, j].set_axis_off()
    fig.suptitle("zoom_compare_sheet — v5 top variants vs raw", fontsize=10)
    fig.tight_layout()
    fig.savefig(sheet_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #

VARIANT_FIELDS = [
    "variant_name", "core_threshold", "halo_threshold",
    "closing_len", "closing_width", "halo_dilate",
    "keep_core", "keep_halo",
    "core_coverage_percent", "halo_coverage_percent",
    "edit_coverage_percent",
    "dir_bg_coverage_core_percent",
    "mean_abs_removed_core", "mean_abs_removed_halo",
    "mean_abs_removed_outside",
    "core_residual_after_suppress",
]

# 5 (core_th, halo_th) pairs spanning loose -> tight
DEFAULT_PAIRS = [
    (0.5, 0.25),
    (0.5, 0.35),
    (0.6, 0.35),
    (0.6, 0.45),
    (0.7, 0.45),
]
DEFAULT_CLOSING_LENS = [0, 15]


def expand_sweep(pairs: list[tuple[float, float]],
                 closing_lens: list[int]) -> list[dict]:
    out: list[dict] = []
    for (ct, ht), cl in product(pairs, closing_lens):
        out.append({"core_th": ct, "halo_th": ht, "closing_len": cl})
    return out


def variant_name_of(core_th: float, halo_th: float, closing_len: int) -> str:
    return f"v5_ct{core_th:.2f}_ht{halo_th:.2f}_cl{closing_len}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/data.tif")
    ap.add_argument("--prepared", default="outputs/prepare")
    ap.add_argument("--out", default="outputs/visual_line_hide_v5_deepmask")

    # Line-prob source
    ap.add_argument("--line-prob", default=None,
                    help="Path to a line probability .tif. If given, used "
                         "directly. Otherwise --line-mask or fallback synthesis.")
    ap.add_argument("--line-mask", default=None,
                    help="Path to a binary line mask .png. Used if "
                         "--line-prob is not given.")
    ap.add_argument("--sato-response", default=None,
                    help="Path to response_sato.tif used by the synthesis "
                         "fallback. Defaults to prepared/responses/response_sato.tif.")

    # Synthesis params (only used if no --line-prob / --line-mask)
    ap.add_argument("--synth-score-center", type=float, default=1.0,
                    help="Sigmoid center for the black-line score component")
    ap.add_argument("--synth-smooth-sigma", type=float, default=0.8)

    # Sweep
    ap.add_argument("--core-thresholds", type=float, nargs="+", default=None,
                    help="If given, expand cross-product with --halo-thresholds. "
                         "Otherwise use the 5 default (core,halo) pairs.")
    ap.add_argument("--halo-thresholds", type=float, nargs="+", default=None)
    ap.add_argument("--closing-lens", type=int, nargs="+",
                    default=DEFAULT_CLOSING_LENS)
    ap.add_argument("--closing-width", type=int, default=1)
    ap.add_argument("--halo-dilate", type=int, default=2)
    ap.add_argument("--keep-core", type=float, default=0.0)
    ap.add_argument("--keep-halo", type=float, default=0.3)
    ap.add_argument("--feather-radius", type=float, default=2.0)
    ap.add_argument("--side-sigma", type=float, default=15.0)
    ap.add_argument("--side-radius", type=int, default=40)
    ap.add_argument("--tensor-sigma", type=float, default=4.0)
    ap.add_argument("--core-distances", type=int, nargs="+",
                    default=[9, 11, 15, 19])
    ap.add_argument("--zoom-crop-size", type=int, default=480)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    in_path = Path(args.input)
    prepared = Path(args.prepared)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "reports").mkdir(parents=True, exist_ok=True)
    (out_root / "variants").mkdir(parents=True, exist_ok=True)

    print(f"input        = {in_path}")
    print(f"prepared     = {prepared}")
    print(f"out          = {out_root}")

    # ---- Image ---- #
    img_raw = load_tif(in_path)
    img_dtype = img_raw.dtype
    img_f = img_raw.astype(np.float32)
    if np.issubdtype(img_dtype, np.integer):
        di = np.iinfo(img_dtype)
        img_clip = (float(di.min), float(di.max))
    else:
        img_clip = (float(img_f.min()), float(img_f.max()))
    preview_lo, preview_hi = percentile_range(img_f)
    raw8_full = to_uint8(img_f, preview_lo, preview_hi)

    # ---- Line probability ---- #
    print("[1/5] line probability source")
    sato_path = (
        Path(args.sato_response) if args.sato_response
        else prepared / "responses" / "response_sato.tif"
    )
    line_prob, prob_info = load_or_synthesize_line_prob(
        line_prob_path=Path(args.line_prob) if args.line_prob else None,
        line_mask_path=Path(args.line_mask) if args.line_mask else None,
        sato_response_path=sato_path,
        img_f=img_f,
    )
    print(f"  line_prob source: {prob_info.get('source')}")
    save_uint8(out_root / "line_prob_preview.png",
               to_uint8(line_prob, 0.0, 1.0))
    save_tif(out_root / "line_prob.tif", line_prob)
    with open(out_root / "reports" / "line_prob_info.json", "w") as f:
        json.dump(prob_info, f, indent=2)
    print(f"  prob p50={np.percentile(line_prob, 50):.3f} "
          f"p90={np.percentile(line_prob, 90):.3f} "
          f"p99={np.percentile(line_prob, 99):.3f}")
    print(f"  prob>0.5 cov={(line_prob > 0.5).mean()*100:.2f}%, "
          f">0.6 cov={(line_prob > 0.6).mean()*100:.2f}%, "
          f">0.7 cov={(line_prob > 0.7).mean()*100:.2f}%")

    # ---- Masks ---- #
    print("[2/5] protect split + structure tensor")
    protect_full = load_mask(prepared / "masks" / "protect_mask.png")
    broad_path = prepared / "masks" / "mask_F_sato75.png"
    if broad_path.exists():
        broad_line_mask = load_mask(broad_path)
        print(f"  broad_line_mask (sato75): {broad_line_mask.mean()*100:.2f}%")
    else:
        broad_line_mask = load_mask(prepared / "masks" / "mask_D_sato85.png")
        print(f"  broad_line_mask (sato85 fallback): {broad_line_mask.mean()*100:.2f}%")

    protect_true_blob, protect_on_line, prot_stats = split_protect_mask(
        protect_full, broad_line_mask, dilate_radius=3, overlap_threshold=0.3,
    )
    save_mask(out_root / "protect_true_blob.png", protect_true_blob)
    save_mask(out_root / "protect_on_line.png", protect_on_line)
    print(f"  protect split: {prot_stats}")

    cos_n, sin_n, coh = compute_normal_direction(
        img_f, sigma_pre=1.0, sigma_tensor=args.tensor_sigma,
    )
    print(f"  mean coherence={coh.mean():.3f}")

    # Compute bg_low ONCE (for core_residual metric)
    bg_low_global = ndimage.gaussian_filter(img_f, sigma=20.0).astype(np.float32)

    # ---- Zoom crops ---- #
    zoom_density = load_tif(prepared / "responses" / "response_sato.tif")
    zoom_centers, zoom_half = select_zoom_crops(
        zoom_density, n=4, crop_size=args.zoom_crop_size,
    )
    print(f"  zoom_crops: {zoom_centers} half={zoom_half}")
    with open(out_root / "reports" / "zoom_crops.json", "w") as f:
        json.dump({"centers": zoom_centers, "half": zoom_half,
                   "crop_size": args.zoom_crop_size}, f, indent=2)

    # ---- Sweep ---- #
    if args.core_thresholds is not None and args.halo_thresholds is not None:
        pairs = [(c, h) for c, h in product(args.core_thresholds, args.halo_thresholds)
                 if c > h]
    else:
        pairs = DEFAULT_PAIRS
    sweep = expand_sweep(pairs, args.closing_lens)
    print(f"[3/5] sweep: {len(sweep)} variants ({len(pairs)} pairs × "
          f"{len(args.closing_lens)} closing_lens)")

    # Group by (core_th, halo_th, closing_len) so we don't redo masks
    csv_rows: list[dict] = []
    records: list[dict] = []
    suppressed8_lookup: dict[str, np.ndarray] = {}
    t0 = time.time()
    done = 0

    for s in sweep:
        if args.limit is not None and done >= args.limit:
            break
        ct, ht, cl = s["core_th"], s["halo_th"], s["closing_len"]
        name = variant_name_of(ct, ht, cl)

        core, halo = build_core_halo_from_prob(
            line_prob, core_threshold=ct, halo_threshold=ht,
            protect_mask=protect_true_blob,
            closing_len=cl, closing_width=args.closing_width,
            halo_dilate=args.halo_dilate,
        )
        edit = core | halo
        if not edit.any():
            print(f"  [skip] {name}: empty edit mask")
            continue

        # Side-band fallback + directional bg
        side_bg = side_band_bg(
            img_f, edit, protect_full,
            side_radius=args.side_radius, sigma=args.side_sigma,
        )
        forbid = edit | broad_line_mask | protect_full
        bg, has_dir = directional_bg(
            img_f, forbid, side_bg, cos_n, sin_n,
            distances=args.core_distances, reduction="median",
        )
        suppressed = suppress(
            img_f, bg, core, halo,
            keep_core=args.keep_core, keep_halo=args.keep_halo,
            feather_radius=args.feather_radius,
        )

        # Metrics
        outside = ~edit
        mean_core = float(np.abs((img_f - suppressed)[core]).mean()) if core.any() else 0.0
        mean_halo = float(np.abs((img_f - suppressed)[halo]).mean()) if halo.any() else 0.0
        mean_out = float(np.abs((img_f - suppressed)[outside]).mean()) if outside.any() else 0.0
        dir_cov = (
            float((has_dir & core).sum() / max(core.sum(), 1) * 100.0)
            if core.any() else 0.0
        )
        if core.any():
            core_resid = float((bg_low_global[core] - suppressed[core]).mean())
        else:
            core_resid = 0.0

        metrics = {
            "variant_name": name,
            "core_threshold": ct, "halo_threshold": ht,
            "closing_len": cl, "closing_width": args.closing_width,
            "halo_dilate": args.halo_dilate,
            "keep_core": args.keep_core, "keep_halo": args.keep_halo,
            "core_coverage_percent": float(core.mean() * 100.0),
            "halo_coverage_percent": float(halo.mean() * 100.0),
            "edit_coverage_percent": float(edit.mean() * 100.0),
            "dir_bg_coverage_core_percent": dir_cov,
            "mean_abs_removed_core": mean_core,
            "mean_abs_removed_halo": mean_halo,
            "mean_abs_removed_outside": mean_out,
            "core_residual_after_suppress": core_resid,
        }
        csv_rows.append(metrics)

        # Per-variant saves
        variant_dir = out_root / "variants" / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        save_mask(variant_dir / "core_mask.png", core)
        save_mask(variant_dir / "halo_mask.png", halo)
        overlay_img = overlay_core_halo(raw8_full, core, halo)
        save_uint8(variant_dir / "overlay.png", overlay_img)
        sup8 = to_uint8(suppressed, preview_lo, preview_hi)
        save_uint8(variant_dir / "suppressed_preview.png", sup8)
        suppressed8_lookup[name] = sup8
        removed = img_f - suppressed
        abs_scale = max(1.0, float(np.percentile(np.abs(removed), 99.5)) * 1.5)
        removed_rgb = diff_heatmap(removed, abs_scale)
        save_uint8(variant_dir / "removed.png", removed_rgb)
        save_uint8(variant_dir / "cleanup_mask.png",
                   (edit.astype(np.uint8) * 255))
        save_tif(variant_dir / "suppressed.tif",
                 np.clip(suppressed, *img_clip).astype(img_dtype))
        save_summary_panel(
            variant_dir / "summary_panel.png",
            raw8_full, sup8, core, halo, removed, abs_scale,
            f"{name}   edit_cov={edit.mean()*100:.2f}%  "
            f"core_resid={core_resid:+.2f}",
            zoom_centers, zoom_half,
        )
        with open(variant_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        records.append({
            "name": name, "metrics": metrics,
            "thumbs": {
                "overlay": downscale_uint8(overlay_img, max_side=520),
                "suppressed": downscale_uint8(sup8, max_side=520),
                "removed": downscale_uint8(removed_rgb, max_side=520),
            },
        })

        done += 1
        elapsed = time.time() - t0
        print(f"  [{done}/{len(sweep)}] {name}  core_cov={metrics['core_coverage_percent']:.2f}%  "
              f"edit_cov={metrics['edit_coverage_percent']:.2f}%  "
              f"dir%={dir_cov:.1f}  core_resid={core_resid:+.2f}  "
              f"({elapsed:.1f}s)")

    print(f"finished {done} variants in {time.time() - t0:.1f}s")

    # ---- Reports ---- #
    print("[4/5] reports")
    reports = out_root / "reports"
    csv_path = reports / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VARIANT_FIELDS)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"  wrote {csv_path}")
    save_contact_sheet(records, reports / "contact_sheet.png")
    save_compare_sheet(records, reports / "compare_sheet.png",
                       title=f"v5 compare_sheet (rows={len(records)})")
    # Pick top by smallest |core_residual| then highest dir%
    def _key(rec):
        m = rec["metrics"]
        return (abs(m["core_residual_after_suppress"]),
                -m["dir_bg_coverage_core_percent"],
                m["edit_coverage_percent"])
    top = sorted(records, key=_key)[:5]
    save_zoom_compare_sheet(
        reports / "zoom_compare_sheet.png",
        raw8_full, zoom_centers, zoom_half, top, suppressed8_lookup,
    )
    print(f"  wrote contact_sheet.png, compare_sheet.png, zoom_compare_sheet.png")

    print("[5/5] done. To plug in a real Route B U-Net result, rerun with:")
    print("        --line-prob path/to/unet_line_prob.tif")


if __name__ == "__main__":
    main()
