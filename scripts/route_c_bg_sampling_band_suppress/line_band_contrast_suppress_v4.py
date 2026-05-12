"""Stage 2f - Visual Line Hide v4.

Builds on the v3 baseline `C90b_bt1.0_wc7_h6_kh0.3` and adds a *residual
dark cleanup* pass that targets the discontinuous dark dots / segments
the reviewer flagged on v3 (line crossings, original long-line locations).

Pipeline:

1. Run the v3 baseline ONCE with the fixed parameters above and obtain
   `suppressed_v3`, `v3_core`, `v3_halo`, the structure-tensor normal
   field and the side-band fallback bg.

2. Compute a *residual dark score* on `suppressed_v3`:

       bg_low_after = gaussian(suppressed_v3, sigma=20)
       dark_residual_after = bg_low_after - suppressed_v3
       residual_score = max(dark_residual_after, 0) / max(local_noise, MAD)

   Threshold sweep: 0.5σ / 0.8σ / 1.0σ.

3. Split the prepared `protect_mask` into:

   - `protect_true_blob`: compact blobs that do NOT sit on lines
     (overlap with `dilate(broad_line_mask, 3)` ≤ 30 %). Preserved.
   - `protect_on_line`: compact blobs that DO sit on lines
     (overlap > 30 %). Allowed to be cleaned.

4. Directional closing of `v3_core` along multiple line orientations
   ({0,30,60,90,120,150}°) at (length=closing_len, width=closing_width)
   to fill in along-the-line breaks. The new pixels are added to the
   cleanup mask only where they also fall inside `broad_line_mask`
   (sato75), so we never bridge unrelated structure.

5. Cleanup candidates = (residual_score > threshold)
   AND inside `dilate(line_region, margin=4)`
   AND component is either elongated OR overlaps `broad_line_mask`.
   The cleanup mask is the union of these candidates and the
   close-extension pixels, with `protect_true_blob` subtracted.

6. The cleanup mask is filled with the same orientation-aware bg as v3
   (sampled along ±normal, smaller distance set), falling back to the
   v2 side-band bg where directional sampling has no valid neighbour.
   Cleanup pixels are hard-replaced; no soft retention of raw.

7. Outputs per variant: summary_panel.png (full + 4 zoom crops),
   suppressed_preview.png, removed.png, cleanup_mask.png,
   residual_dark_mask.png; aggregate sheets and a dedicated
   baseline_v3_vs_v4_compare panel.

Sweep budget: 3 residual_threshold × {cl=0(1), cl=15(cw1,cw2), cl=31(cw1,cw2)}
= 3 × 5 = 15 variants.
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
# Black-line / residual dark score
# --------------------------------------------------------------------------- #

def compute_dark_score(
    img_f: np.ndarray, sigma_bg: float = 20.0, sigma_noise: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (score, bg_low, global_mad). Score is (bg_low - img)+ / noise."""
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
    return score.astype(np.float32), bg_low, global_mad


# --------------------------------------------------------------------------- #
# Structure tensor (carried from v3)
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


# --------------------------------------------------------------------------- #
# Core / halo (v3 carryover)
# --------------------------------------------------------------------------- #

def build_core_halo(
    sato_mask: np.ndarray, black_score: np.ndarray, protect_mask: np.ndarray,
    width_core: int, halo_width: int, black_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    line_core_thin = sato_mask & (black_score > black_threshold)
    core = binary_dilation(line_core_thin, disk(width_core))
    core = binary_closing(core, disk(2 if width_core >= 5 else 1))
    halo_outer = binary_dilation(core, disk(halo_width))
    halo = halo_outer & ~core
    core = core & ~protect_mask
    halo = halo & ~protect_mask
    return core, halo


# --------------------------------------------------------------------------- #
# Side-band / directional bg (v3 carryover)
# --------------------------------------------------------------------------- #

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
    cos_n: np.ndarray, sin_n: np.ndarray,
    distances: list[float], reduction: str = "median",
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
# v3 suppression (used to build the fixed baseline)
# --------------------------------------------------------------------------- #

def feather_outer_alpha(edit_mask: np.ndarray, feather_radius: float) -> np.ndarray:
    if feather_radius <= 0:
        return edit_mask.astype(np.float32)
    dist_outside = ndimage.distance_transform_edt(~edit_mask)
    alpha = np.clip(1.0 - dist_outside / float(feather_radius), 0.0, 1.0)
    alpha[edit_mask] = 1.0
    return alpha.astype(np.float32)


def suppress_v3(
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
    edit_mask = core_mask | halo_mask
    if feather_radius > 0 and edit_mask.any():
        alpha = feather_outer_alpha(edit_mask, feather_radius)
        out = alpha * out + (1.0 - alpha) * img_f
        if halo_mask.any():
            out[halo_mask] = (bg[halo_mask] + keep_halo * residual[halo_mask]).astype(np.float32)
        if core_mask.any():
            out[core_mask] = (bg[core_mask] + keep_core * residual[core_mask]).astype(np.float32)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Protect mask split
# --------------------------------------------------------------------------- #

def split_protect_mask(
    protect_mask: np.ndarray, broad_line_mask: np.ndarray,
    dilate_radius: int = 3, overlap_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, dict]:
    broad_dilated = binary_dilation(broad_line_mask, disk(dilate_radius))
    lbl = measure.label(protect_mask, connectivity=2)
    protect_true = np.zeros_like(protect_mask, dtype=bool)
    protect_on_line = np.zeros_like(protect_mask, dtype=bool)
    n_true = 0; n_on_line = 0
    for region in measure.regionprops(lbl):
        coords = region.coords
        area = region.area
        on_line_count = int(broad_dilated[coords[:, 0], coords[:, 1]].sum())
        ratio = on_line_count / max(area, 1)
        if ratio > overlap_threshold:
            protect_on_line[coords[:, 0], coords[:, 1]] = True
            n_on_line += 1
        else:
            protect_true[coords[:, 0], coords[:, 1]] = True
            n_true += 1
    stats = {
        "n_true_blob": n_true, "n_on_line": n_on_line,
        "true_coverage_pct": float(protect_true.mean() * 100.0),
        "on_line_coverage_pct": float(protect_on_line.mean() * 100.0),
    }
    return protect_true, protect_on_line, stats


# --------------------------------------------------------------------------- #
# Multi-angle line SE + directional closing
# --------------------------------------------------------------------------- #

def make_line_se(length: int, width: int, angle_deg: float) -> np.ndarray:
    """Approximate a thin line of given length and width at angle_deg.
    Returns a square binary SE big enough to contain the rotated line."""
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
    d_steps = max(length, 3) * 2  # oversample to avoid gaps
    w_steps = max(width, 1)
    for d in np.linspace(-half_l, half_l, d_steps):
        for w in np.linspace(-half_w, half_w, w_steps) if w_steps > 1 else [0.0]:
            y = int(round(cy + d * sin_t + w * cos_t))
            x = int(round(cx + d * cos_t - w * sin_t))
            if 0 <= y < size and 0 <= x < size:
                se[y, x] = True
    return se


def directional_closing(
    mask: np.ndarray, length: int, width: int,
    angles_deg: list[int] = (0, 30, 60, 90, 120, 150),
) -> np.ndarray:
    if length <= 0:
        return mask.copy()
    out = mask.copy()
    for ang in angles_deg:
        se = make_line_se(length=length, width=width, angle_deg=ang)
        closed = binary_closing(mask, footprint=se)
        out |= closed
    return out


# --------------------------------------------------------------------------- #
# Cleanup mask construction
# --------------------------------------------------------------------------- #

def linear_filter_candidates(
    candidate: np.ndarray, broad_line_mask: np.ndarray,
    min_eccentricity: float = 0.85, min_area_line: int = 4,
    min_area_isolated: int = 12,
) -> np.ndarray:
    """Keep candidate components that overlap broad_line_mask OR are
    eccentric (elongated). Drop small compact blobs not on lines."""
    if not candidate.any():
        return candidate
    lbl = measure.label(candidate, connectivity=2)
    out = np.zeros_like(candidate, dtype=bool)
    for region in measure.regionprops(lbl):
        coords = region.coords
        area = region.area
        on_line_count = int(broad_line_mask[coords[:, 0], coords[:, 1]].sum())
        on_line = on_line_count > 0
        if on_line and area >= min_area_line:
            out[coords[:, 0], coords[:, 1]] = True
            continue
        if not on_line and area >= min_area_isolated and region.eccentricity > min_eccentricity:
            out[coords[:, 0], coords[:, 1]] = True
    return out


def build_cleanup_mask(
    residual_score: np.ndarray, residual_threshold: float,
    v3_core: np.ndarray, v3_halo: np.ndarray, broad_line_mask: np.ndarray,
    protect_true_blob: np.ndarray,
    closing_len: int, closing_width: int, line_margin: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (cleanup_mask, residual_dark_mask, v3_core_extended_added)."""
    residual_dark_mask = residual_score > residual_threshold

    line_region = v3_core | v3_halo | broad_line_mask
    line_region_dilated = binary_dilation(line_region, disk(line_margin))

    # Residual-driven candidates within line region
    candidate = residual_dark_mask & line_region_dilated
    candidate = linear_filter_candidates(candidate, broad_line_mask)

    # Directional-closing-driven additions (forced clean inside broad_line_mask)
    if closing_len > 0:
        v3_core_ext = directional_closing(v3_core, closing_len, closing_width)
        added_from_closing = v3_core_ext & broad_line_mask & ~v3_core
    else:
        added_from_closing = np.zeros_like(v3_core, dtype=bool)

    cleanup = (candidate | added_from_closing) & ~protect_true_blob
    return cleanup, residual_dark_mask, added_from_closing


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


def overlay_mask(gray8: np.ndarray, mask: np.ndarray, color=(220, 30, 30),
                 alpha: float = 0.55) -> np.ndarray:
    rgb = np.stack([gray8, gray8, gray8], axis=-1).astype(np.float32)
    if mask.any():
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
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    img = Image.fromarray(arr)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    return np.asarray(img)


# --------------------------------------------------------------------------- #
# Zoom crops
# --------------------------------------------------------------------------- #

def select_zoom_crops(
    sato_response: np.ndarray, n: int = 4, crop_size: int = 480,
    margin: int = 20,
) -> tuple[list[tuple[int, int]], int]:
    H, W = sato_response.shape
    smoothed = ndimage.gaussian_filter(sato_response.astype(np.float32), sigma=12.0)
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
# Per-variant panel
# --------------------------------------------------------------------------- #

def save_summary_panel(
    out_path: Path,
    raw8: np.ndarray, v3_sup8: np.ndarray, v4_sup8: np.ndarray,
    cleanup_mask: np.ndarray, removed: np.ndarray, abs_scale: float,
    title: str, zoom_centers: list[tuple[int, int]], zoom_half: int,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 9.0))
    axes[0, 0].imshow(raw8, cmap="gray"); axes[0, 0].set_title("raw")
    axes[0, 1].imshow(v3_sup8, cmap="gray"); axes[0, 1].set_title("v3 baseline")
    axes[0, 2].imshow(v4_sup8, cmap="gray"); axes[0, 2].set_title("v4 (after cleanup)")
    axes[0, 3].imshow(overlay_mask(v3_sup8, cleanup_mask))
    axes[0, 3].set_title("cleanup_mask over v3")
    for j in range(4):
        axes[0, j].set_axis_off()
    for j in range(4):
        ax = axes[1, j]
        if j >= len(zoom_centers):
            ax.set_axis_off(); continue
        cy, cx = zoom_centers[j]
        ax.imshow(extract_crop(v4_sup8, cy, cx, zoom_half), cmap="gray")
        ax.set_title(f"Z{j + 1} v4 (y={cy}, x={cx})", fontsize=8)
        ax.set_axis_off()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Aggregate sheets
# --------------------------------------------------------------------------- #

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
    fig.suptitle(f"contact_sheet — v4 suppressed previews ({n} variants)")
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
        axes[r, 0].imshow(rec["thumbs"]["cleanup_overlay"])
        axes[r, 0].set_title(f"{rec['name']} | cleanup", fontsize=7)
        axes[r, 1].imshow(rec["thumbs"]["suppressed"], cmap="gray")
        axes[r, 1].set_title("v4 suppressed", fontsize=7)
        axes[r, 2].imshow(rec["thumbs"]["removed"])
        axes[r, 2].set_title("raw - v4", fontsize=7)
        for c in range(3):
            axes[r, c].set_axis_off()
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(sheet_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_zoom_compare_sheet(
    sheet_path: Path, raw8: np.ndarray, v3_sup8: np.ndarray,
    zoom_centers: list[tuple[int, int]], zoom_half: int,
    top_records: list[dict], suppressed8_lookup: dict[str, np.ndarray],
) -> None:
    n_zoom = len(zoom_centers)
    n_var = len(top_records)
    rows = 2 + n_var  # raw, v3 baseline, then v4 variants
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
        axes[1, j].imshow(extract_crop(v3_sup8, cy, cx, zoom_half), cmap="gray")
        axes[1, j].set_title(f"v3 baseline — Z{j + 1}", fontsize=8)
        axes[1, j].set_axis_off()
    for i, rec in enumerate(top_records):
        sup8 = suppressed8_lookup[rec["name"]]
        for j, (cy, cx) in enumerate(zoom_centers):
            axes[i + 2, j].imshow(extract_crop(sup8, cy, cx, zoom_half), cmap="gray")
            axes[i + 2, j].set_title(f"{rec['name']} — Z{j + 1}", fontsize=7)
            axes[i + 2, j].set_axis_off()
    fig.suptitle("zoom_compare_sheet — v4 top variants vs raw / v3 baseline",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(sheet_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_baseline_compare(
    sheet_path: Path, raw8: np.ndarray, v3_sup8: np.ndarray, v4_sup8: np.ndarray,
    zoom_centers: list[tuple[int, int]], zoom_half: int, v4_name: str,
) -> None:
    n_zoom = len(zoom_centers)
    fig, axes = plt.subplots(3, 1 + n_zoom, figsize=((1 + n_zoom) * 3.0, 9.0))
    titles = [(raw8, "raw"), (v3_sup8, "v3 baseline"), (v4_sup8, f"v4 {v4_name}")]
    for r, (img, lbl) in enumerate(titles):
        axes[r, 0].imshow(img, cmap="gray")
        axes[r, 0].set_title(lbl + " — full", fontsize=9)
        axes[r, 0].set_axis_off()
        for j, (cy, cx) in enumerate(zoom_centers):
            axes[r, 1 + j].imshow(extract_crop(img, cy, cx, zoom_half), cmap="gray")
            axes[r, 1 + j].set_title(f"{lbl} — Z{j + 1}", fontsize=8)
            axes[r, 1 + j].set_axis_off()
    fig.suptitle("baseline_v3_vs_v4_compare", fontsize=10)
    fig.tight_layout()
    fig.savefig(sheet_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #

VARIANT_FIELDS = [
    "variant_name", "residual_threshold", "closing_len", "closing_width",
    "keep_halo",
    "baseline_core_coverage_percent", "baseline_edit_coverage_percent",
    "cleanup_coverage_percent", "cleanup_on_line_fraction",
    "cleanup_dir_bg_coverage_percent",
    "mean_abs_removed_cleanup",
    "residual_p99_v3", "residual_p99_v4",
    "core_residual_v3", "core_residual_v4",
]


def expand_sweep(
    residual_thresholds: list[float], closing_lens: list[int],
    closing_widths: list[int],
) -> list[dict]:
    out: list[dict] = []
    for rt in residual_thresholds:
        for cl in closing_lens:
            if cl <= 0:
                out.append({"rt": rt, "cl": 0, "cw": 1})  # cw irrelevant
                continue
            for cw in closing_widths:
                out.append({"rt": rt, "cl": cl, "cw": cw})
    return out


def variant_name_of(rt: float, cl: int, cw: int) -> str:
    if cl <= 0:
        return f"v4_rt{rt:.1f}_cl0"
    return f"v4_rt{rt:.1f}_cl{cl}_cw{cw}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/data.tif")
    ap.add_argument("--prepared", default="outputs/prepare")
    ap.add_argument("--out", default="outputs/visual_line_hide_v4")

    # Fixed v3 baseline params (per task spec)
    ap.add_argument("--baseline-mask", default="C_sato90")
    ap.add_argument("--baseline-bt", type=float, default=1.0)
    ap.add_argument("--baseline-wc", type=int, default=7)
    ap.add_argument("--baseline-hw", type=int, default=6)
    ap.add_argument("--baseline-kc", type=float, default=0.0)
    ap.add_argument("--baseline-kh", type=float, default=0.3)
    ap.add_argument("--baseline-side-sigma", type=float, default=15.0)
    ap.add_argument("--baseline-side-radius", type=int, default=40)
    ap.add_argument("--baseline-feather", type=float, default=2.0)
    ap.add_argument("--bg-sigma", type=float, default=20.0)
    ap.add_argument("--noise-sigma", type=float, default=8.0)
    ap.add_argument("--tensor-sigma", type=float, default=4.0)

    # v4 cleanup sweep
    ap.add_argument("--residual-thresholds", type=float, nargs="+",
                    default=[0.5, 0.8, 1.0])
    ap.add_argument("--closing-lens", type=int, nargs="+",
                    default=[0, 15, 31])
    ap.add_argument("--closing-widths", type=int, nargs="+",
                    default=[1, 2])
    ap.add_argument("--keep-halo", type=float, default=0.3,
                    help="Fixed at 0.3 per task spec; cleanup itself is hard "
                         "replace, this only controls the v3 baseline halo")

    ap.add_argument("--line-margin", type=int, default=4)
    ap.add_argument("--cleanup-distances", type=int, nargs="+",
                    default=[4, 7, 10, 14])
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

    # ---- Load image & dtype ---- #
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

    # ---- Black-line score (for baseline mask building) ---- #
    print("[1/5] black-line score + structure tensor")
    score, bg_low_raw, mad = compute_dark_score(
        img_f, sigma_bg=args.bg_sigma, sigma_noise=args.noise_sigma,
    )
    cos_n, sin_n, coh = compute_normal_direction(
        img_f, sigma_pre=1.0, sigma_tensor=args.tensor_sigma,
    )
    print(f"  global_mad={mad:.4f}, mean coherence={coh.mean():.3f}")

    # ---- Masks ---- #
    protect_full = load_mask(prepared / "masks" / "protect_mask.png")
    broad_path = prepared / "masks" / "mask_F_sato75.png"
    if broad_path.exists():
        broad_line_mask = load_mask(broad_path)
        print(f"  broad_line_mask (sato75): {broad_line_mask.mean()*100:.2f}%")
    else:
        broad_line_mask = load_mask(prepared / "masks" / "mask_D_sato85.png")
        print(f"  broad_line_mask (sato85 fallback): {broad_line_mask.mean()*100:.2f}%")

    sato_baseline = load_mask(prepared / "masks" / "mask_C_sato90.png")

    # ---- Split protect mask ---- #
    protect_true_blob, protect_on_line, prot_stats = split_protect_mask(
        protect_full, broad_line_mask,
        dilate_radius=3, overlap_threshold=0.3,
    )
    print(f"  protect split: {prot_stats}")
    save_mask(out_root / "protect_true_blob.png", protect_true_blob)
    save_mask(out_root / "protect_on_line.png", protect_on_line)

    # ---- v3 baseline ---- #
    print("[2/5] v3 baseline suppression")
    v3_core, v3_halo = build_core_halo(
        sato_baseline, score, protect_full,
        width_core=args.baseline_wc, halo_width=args.baseline_hw,
        black_threshold=args.baseline_bt,
    )
    v3_edit = v3_core | v3_halo
    side_bg_baseline = side_band_bg(
        img_f, v3_edit, protect_full,
        side_radius=args.baseline_side_radius,
        sigma=args.baseline_side_sigma,
    )
    forbid_baseline = v3_edit | broad_line_mask | protect_full
    baseline_distances = [
        args.baseline_wc + 2, args.baseline_wc + 4,
        args.baseline_wc + 8, args.baseline_wc + 12,
    ]
    bg_baseline, _has_dir_baseline = directional_bg(
        img_f, forbid_baseline, side_bg_baseline, cos_n, sin_n,
        distances=baseline_distances, reduction="median",
    )
    suppressed_v3 = suppress_v3(
        img_f, bg_baseline, v3_core, v3_halo,
        keep_core=args.baseline_kc, keep_halo=args.baseline_kh,
        feather_radius=args.baseline_feather,
    )
    v3_sup8 = to_uint8(suppressed_v3, preview_lo, preview_hi)
    save_uint8(out_root / "v3_baseline_preview.png", v3_sup8)
    save_tif(out_root / "v3_baseline.tif",
             np.clip(suppressed_v3, *img_clip).astype(img_dtype))
    bcore_pct = float(v3_core.mean() * 100)
    bedit_pct = float(v3_edit.mean() * 100)
    print(f"  v3 baseline: core={bcore_pct:.2f}%, edit={bedit_pct:.2f}%")

    # ---- Residual dark score on suppressed_v3 ---- #
    print("[3/5] residual dark score on v3 baseline")
    residual_score, bg_low_after, mad_after = compute_dark_score(
        suppressed_v3, sigma_bg=args.bg_sigma, sigma_noise=args.noise_sigma,
    )
    res_p99_v3 = float(np.percentile(residual_score, 99))
    save_uint8(
        out_root / "residual_dark_score_preview.png",
        to_uint8(residual_score, 0.0, max(float(np.percentile(residual_score, 99.5)), 1e-6)),
    )
    # Core residual on baseline
    if v3_core.any():
        core_resid_v3 = float((bg_low_after[v3_core] - suppressed_v3[v3_core]).mean())
    else:
        core_resid_v3 = 0.0
    print(f"  residual_p99_v3={res_p99_v3:.3f}, core_resid_v3={core_resid_v3:+.2f}")

    # ---- Zoom crops ---- #
    sato_response = load_tif(prepared / "responses" / "response_sato.tif")
    zoom_centers, zoom_half = select_zoom_crops(
        sato_response, n=4, crop_size=args.zoom_crop_size,
    )
    print(f"  zoom_crops: {zoom_centers} half={zoom_half}")
    with open(out_root / "reports" / "zoom_crops.json", "w") as f:
        json.dump({
            "centers": zoom_centers, "half": zoom_half,
            "crop_size": args.zoom_crop_size,
        }, f, indent=2)

    # ---- Sweep ---- #
    sweep = expand_sweep(
        args.residual_thresholds, args.closing_lens, args.closing_widths,
    )
    print(f"[4/5] sweep: {len(sweep)} variants")

    # Cache directional-closing of v3_core by (cl, cw)
    closing_cache: dict[tuple[int, int], np.ndarray] = {}

    csv_rows: list[dict] = []
    records: list[dict] = []
    suppressed8_lookup: dict[str, np.ndarray] = {}
    t0 = time.time()
    done = 0

    for s in sweep:
        if args.limit is not None and done >= args.limit:
            break
        rt, cl, cw = s["rt"], s["cl"], s["cw"]
        name = variant_name_of(rt, cl, cw)

        # Build cleanup mask
        cleanup_mask, residual_dark_mask, added_from_closing = build_cleanup_mask(
            residual_score, residual_threshold=rt,
            v3_core=v3_core, v3_halo=v3_halo,
            broad_line_mask=broad_line_mask,
            protect_true_blob=protect_true_blob,
            closing_len=cl, closing_width=cw,
            line_margin=args.line_margin,
        )

        if cleanup_mask.any():
            # Cleanup bg: same orientation-aware approach with smaller distances
            forbid_cleanup = (broad_line_mask | cleanup_mask | protect_full | v3_edit)
            side_bg_cleanup = side_band_bg(
                img_f, cleanup_mask | v3_edit, protect_full,
                side_radius=args.baseline_side_radius,
                sigma=args.baseline_side_sigma,
            )
            cleanup_bg, has_dir_clean = directional_bg(
                img_f, forbid_cleanup, side_bg_cleanup, cos_n, sin_n,
                distances=args.cleanup_distances, reduction="median",
            )
            dir_cov_cleanup = float(
                (has_dir_clean & cleanup_mask).sum() / max(cleanup_mask.sum(), 1) * 100.0
            )
        else:
            cleanup_bg = suppressed_v3
            dir_cov_cleanup = 0.0

        # Apply cleanup (hard replace)
        suppressed_v4 = suppressed_v3.astype(np.float32).copy()
        if cleanup_mask.any():
            suppressed_v4[cleanup_mask] = cleanup_bg[cleanup_mask].astype(np.float32)

        # Metrics
        residual_score_v4, bg_low_v4, _ = compute_dark_score(
            suppressed_v4, sigma_bg=args.bg_sigma, sigma_noise=args.noise_sigma,
        )
        res_p99_v4 = float(np.percentile(residual_score_v4, 99))
        if v3_core.any():
            core_resid_v4 = float((bg_low_v4[v3_core] - suppressed_v4[v3_core]).mean())
        else:
            core_resid_v4 = 0.0

        cleanup_on_line_frac = float(
            (cleanup_mask & broad_line_mask).sum() / max(cleanup_mask.sum(), 1)
        )
        if cleanup_mask.any():
            mean_abs_removed = float(np.abs(
                suppressed_v3[cleanup_mask] - suppressed_v4[cleanup_mask]
            ).mean())
        else:
            mean_abs_removed = 0.0

        # Per-variant saves
        variant_dir = out_root / "variants" / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        save_mask(variant_dir / "cleanup_mask.png", cleanup_mask)
        save_mask(variant_dir / "residual_dark_mask.png", residual_dark_mask)
        save_mask(variant_dir / "added_from_closing.png", added_from_closing)

        v4_sup8 = to_uint8(suppressed_v4, preview_lo, preview_hi)
        suppressed8_lookup[name] = v4_sup8
        save_uint8(variant_dir / "suppressed_preview.png", v4_sup8)

        removed_total = img_f - suppressed_v4
        abs_scale = max(1.0, float(np.percentile(np.abs(removed_total), 99.5)) * 1.5)
        removed_rgb = diff_heatmap(removed_total, abs_scale)
        save_uint8(variant_dir / "removed.png", removed_rgb)

        cleanup_overlay = overlay_mask(v3_sup8, cleanup_mask)
        save_uint8(variant_dir / "cleanup_overlay.png", cleanup_overlay)

        save_tif(variant_dir / "suppressed.tif",
                 np.clip(suppressed_v4, *img_clip).astype(img_dtype))

        save_summary_panel(
            variant_dir / "summary_panel.png",
            raw8_full, v3_sup8, v4_sup8,
            cleanup_mask, removed_total, abs_scale,
            f"{name}   cleanup_cov={cleanup_mask.mean()*100:.2f}%  "
            f"on_line={cleanup_on_line_frac*100:.0f}%  "
            f"res_p99={res_p99_v4:.2f}",
            zoom_centers, zoom_half,
        )

        metrics = {
            "variant_name": name,
            "residual_threshold": rt, "closing_len": cl, "closing_width": cw,
            "keep_halo": args.keep_halo,
            "baseline_core_coverage_percent": bcore_pct,
            "baseline_edit_coverage_percent": bedit_pct,
            "cleanup_coverage_percent": float(cleanup_mask.mean() * 100.0),
            "cleanup_on_line_fraction": cleanup_on_line_frac,
            "cleanup_dir_bg_coverage_percent": dir_cov_cleanup,
            "mean_abs_removed_cleanup": mean_abs_removed,
            "residual_p99_v3": res_p99_v3,
            "residual_p99_v4": res_p99_v4,
            "core_residual_v3": core_resid_v3,
            "core_residual_v4": core_resid_v4,
        }
        with open(variant_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        csv_rows.append(metrics)

        thumbs = {
            "cleanup_overlay": downscale_uint8(cleanup_overlay, max_side=520),
            "suppressed": downscale_uint8(v4_sup8, max_side=520),
            "removed": downscale_uint8(removed_rgb, max_side=520),
        }
        records.append({"name": name, "metrics": metrics, "thumbs": thumbs})

        done += 1
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1e-6)
        print(f"  [{done}/{len(sweep)}] {name}  cleanup_cov={metrics['cleanup_coverage_percent']:.2f}%  "
              f"on_line={cleanup_on_line_frac:.2f}  res_p99={res_p99_v4:.2f}  "
              f"({elapsed:.1f}s, {rate:.2f}/s)")

    print(f"finished {done} variants in {time.time() - t0:.1f}s")

    # ---- Reports ---- #
    print("[5/5] reports")
    reports_dir = out_root / "reports"
    csv_path = reports_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VARIANT_FIELDS)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"  wrote {csv_path}")

    save_contact_sheet(records, reports_dir / "contact_sheet.png")
    print(f"  wrote {reports_dir / 'contact_sheet.png'}")

    save_compare_sheet(
        records, reports_dir / "compare_sheet.png",
        title=f"v4 compare_sheet (rows={len(records)})",
    )
    print(f"  wrote {reports_dir / 'compare_sheet.png'}")

    # Top picks: lowest residual_p99_v4 with reasonable cleanup_on_line_fraction
    def _key(rec):
        m = rec["metrics"]
        return (
            m["residual_p99_v4"],
            -m["cleanup_on_line_fraction"],
            m["cleanup_coverage_percent"],
        )
    top = sorted(records, key=_key)[:5]
    save_zoom_compare_sheet(
        reports_dir / "zoom_compare_sheet.png",
        raw8_full, v3_sup8, zoom_centers, zoom_half, top, suppressed8_lookup,
    )
    print(f"  wrote {reports_dir / 'zoom_compare_sheet.png'} "
          f"({len(top)} top × {len(zoom_centers)} zooms)")

    main_pick = top[0]
    save_baseline_compare(
        reports_dir / "baseline_v3_vs_v4_compare.png",
        raw8_full, v3_sup8, suppressed8_lookup[main_pick["name"]],
        zoom_centers, zoom_half, v4_name=main_pick["name"],
    )
    print(f"  wrote {reports_dir / 'baseline_v3_vs_v4_compare.png'} "
          f"(main pick = {main_pick['name']})")


if __name__ == "__main__":
    main()
