"""Stage 2d - Visual Line Hide v3.

Extends v2 with *orientation-aware background filling*:

For every core / halo pixel we estimate the local ridge normal from the
structure tensor of the smoothed raw image, then sample the image along
+/- normal at distances {wc+2, wc+4, wc+8, wc+12}. Samples that land on
known line pixels (broad sato75 mask) or on the current edit mask or on
the protect mask are dropped. The remaining valid samples are reduced
with a per-pixel median; pixels with zero valid samples fall back to the
v2 side-band gaussian bg.

This is the closest the non-DL pipeline gets to "Photoshop line repair":
the replacement at the core comes from along-the-line neighbors, not
from a Gaussian blur of the surrounding region — so cross-line texture
is preserved and the line core no longer fights back through residual
contrast.

Sweep budget: 32 variants total
- C90b: 3 (black_threshold) x 2 (core_width) x 2 (halo_width) x 2 (keep_halo) = 24
- D85b: 2 (black_threshold)  x 2 (core_width) x 1 (halo_width) x 2 (keep_halo) = 8

Per-variant summary panels embed 4 auto-picked zoom crops; an aggregate
zoom_compare_sheet shows top variants vs raw across all 4 crops.
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
# Black-line score (carried over from v2)
# --------------------------------------------------------------------------- #

def compute_black_line_score(
    img_f: np.ndarray, sigma_bg: float = 20.0, sigma_noise: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    bg_low = ndimage.gaussian_filter(img_f, sigma=sigma_bg).astype(np.float32)
    dark_residual = (bg_low - img_f).astype(np.float32)
    res = (img_f - bg_low).astype(np.float32)
    med = float(np.median(res))
    global_mad = float(np.median(np.abs(res - med)) * 1.4826) + 1e-6
    res_local_mean = ndimage.gaussian_filter(res, sigma=sigma_noise)
    centered = res - res_local_mean
    local_var = ndimage.gaussian_filter(centered * centered, sigma=sigma_noise)
    local_noise = np.sqrt(np.maximum(local_var, 1e-12)).astype(np.float32)
    noise = np.maximum(local_noise, global_mad)
    score = np.clip(dark_residual, 0.0, None) / np.maximum(noise, 1e-6)
    return score.astype(np.float32), bg_low, global_mad


# --------------------------------------------------------------------------- #
# Structure tensor -> ridge normal direction
# --------------------------------------------------------------------------- #

def compute_normal_direction(
    img_f: np.ndarray, sigma_pre: float = 1.0, sigma_tensor: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (cos_n, sin_n, coherence).

    (cos_n, sin_n) is a unit vector pointing along the dominant gradient
    (which equals the ridge *normal* for thin ridges). It is given in
    image coordinates so that sampling at offset d uses
        (y + d*sin_n, x + d*cos_n).
    coherence in [0, 1] is the eigenvalue ratio (lam1 - lam2) / (lam1 + lam2);
    high near real ridges, low in featureless regions.
    """
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
# Core + halo masks
# --------------------------------------------------------------------------- #

def build_core_halo(
    sato_mask: np.ndarray, black_score: np.ndarray, protect_mask: np.ndarray,
    width_core: int, halo_width: int, black_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    line_core_thin = sato_mask & (black_score > black_threshold)
    core = binary_dilation(line_core_thin, disk(width_core))
    core = binary_closing(core, disk(2 if width_core >= 5 else 1))
    halo_outer = binary_dilation(core, disk(halo_width))
    halo = halo_outer & ~core
    core = core & ~protect_mask
    halo = halo & ~protect_mask
    return core, halo, line_core_thin


# --------------------------------------------------------------------------- #
# Side-band bg (v2 carryover, used as fallback only)
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


# --------------------------------------------------------------------------- #
# Directional bg sampling
# --------------------------------------------------------------------------- #

def directional_bg(
    img_f: np.ndarray, forbid_mask: np.ndarray,
    side_bg_fallback: np.ndarray,
    cos_n: np.ndarray, sin_n: np.ndarray,
    distances: list[float], reduction: str = "median",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bg, has_dir, n_valid).

    For every pixel sample img along +/- normal at each distance; drop
    samples where forbid_mask is True (or outside image); reduce valid
    samples per-pixel by median (default) or mean. Pixels with no valid
    sample fall back to side_bg_fallback.
    """
    H, W = img_f.shape
    y_idx, x_idx = np.mgrid[0:H, 0:W].astype(np.float32)
    forbid_f = forbid_mask.astype(np.float32)

    samples = []
    valids = []
    for d in distances:
        for sign in (+1.0, -1.0):
            sy = y_idx + sign * d * sin_n
            sx = x_idx + sign * d * cos_n
            coords = np.stack([sy, sx], axis=0)
            s = ndimage.map_coordinates(
                img_f, coords, order=1, mode="reflect",
            ).astype(np.float32)
            f = ndimage.map_coordinates(
                forbid_f, coords, order=0, mode="constant", cval=1.0,
            )
            v = (f < 0.5).astype(np.float32)
            samples.append(s)
            valids.append(v)

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
        denom = np.maximum(n_valid, 1.0)
        bg_dir = total / denom

    bg = np.where(has_dir, bg_dir, side_bg_fallback).astype(np.float32)
    return bg, has_dir, n_valid.astype(np.int32)


# --------------------------------------------------------------------------- #
# Suppression
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
        y0 = max(0, cy - crop_size)
        y1 = min(H, cy + crop_size)
        x0 = max(0, cx - crop_size)
        x1 = min(W, cx + crop_size)
        work[y0:y1, x0:x1] = -np.inf
    crops.sort(key=lambda c: (c[0], c[1]))
    return crops, half


def extract_crop(arr: np.ndarray, cy: int, cx: int, half: int) -> np.ndarray:
    y0 = cy - half
    x0 = cx - half
    y1 = y0 + 2 * half
    x1 = x0 + 2 * half
    return arr[y0:y1, x0:x1].copy()


# --------------------------------------------------------------------------- #
# Preview utilities
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
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    img = Image.fromarray(arr)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    return np.asarray(img)


# --------------------------------------------------------------------------- #
# Per-variant summary panel (with zooms)
# --------------------------------------------------------------------------- #

def save_summary_panel(
    out_path: Path, raw8: np.ndarray, sup8: np.ndarray,
    core_mask: np.ndarray, halo_mask: np.ndarray,
    removed: np.ndarray, abs_scale: float, title: str,
    zoom_centers: list[tuple[int, int]], zoom_half: int,
) -> None:
    n_zoom = len(zoom_centers)
    fig, axes = plt.subplots(2, max(4, n_zoom), figsize=(4 * max(4, n_zoom), 9.0))
    if axes.ndim == 1:
        axes = axes.reshape(2, -1)

    # Row 0: full image panels
    axes[0, 0].imshow(raw8, cmap="gray"); axes[0, 0].set_title("raw")
    axes[0, 1].imshow(overlay_core_halo(raw8, core_mask, halo_mask))
    axes[0, 1].set_title("core (red) + halo (orange)")
    axes[0, 2].imshow(sup8, cmap="gray"); axes[0, 2].set_title("suppressed")
    axes[0, 3].imshow(removed, cmap="seismic", vmin=-abs_scale, vmax=abs_scale)
    axes[0, 3].set_title("raw - suppressed")
    for j in range(4, axes.shape[1]):
        axes[0, j].set_axis_off()
    for j in range(4):
        axes[0, j].set_axis_off()

    # Row 1: zoom crops of suppressed
    for j in range(axes.shape[1]):
        ax = axes[1, j]
        if j >= n_zoom:
            ax.set_axis_off(); continue
        cy, cx = zoom_centers[j]
        crop = extract_crop(sup8, cy, cx, zoom_half)
        ax.imshow(crop, cmap="gray")
        ax.set_title(f"Z{j + 1} suppressed  (y={cy}, x={cx})", fontsize=8)
        ax.set_axis_off()

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Aggregate sheets
# --------------------------------------------------------------------------- #

def save_contact_sheet(records: list[dict], sheet_path: Path, cols: int = 8) -> None:
    n = len(records)
    if n == 0:
        return
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.5))
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
        ax.set_title(rec["name"], fontsize=5)
        ax.set_axis_off()
    fig.suptitle(f"contact_sheet — v3 suppressed previews ({n} variants)")
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
        axes[r, 1].set_title("suppressed", fontsize=7)
        axes[r, 2].imshow(rec["thumbs"]["removed"])
        axes[r, 2].set_title("raw - suppressed", fontsize=7)
        for c in range(3):
            axes[r, c].set_axis_off()
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(sheet_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_zoom_compare_sheet(
    sheet_path: Path,
    raw8: np.ndarray,
    zoom_centers: list[tuple[int, int]], zoom_half: int,
    top_records: list[dict],
    suppressed8_lookup: dict[str, np.ndarray],
) -> None:
    n_zoom = len(zoom_centers)
    n_var = len(top_records)
    rows = 1 + n_var
    cols = n_zoom
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    # row 0: raw
    for j, (cy, cx) in enumerate(zoom_centers):
        axes[0, j].imshow(extract_crop(raw8, cy, cx, zoom_half), cmap="gray")
        axes[0, j].set_title(f"raw — Z{j + 1} (y={cy}, x={cx})", fontsize=8)
        axes[0, j].set_axis_off()
    # rows 1..n_var
    for i, rec in enumerate(top_records):
        sup8 = suppressed8_lookup[rec["name"]]
        for j, (cy, cx) in enumerate(zoom_centers):
            axes[i + 1, j].imshow(extract_crop(sup8, cy, cx, zoom_half), cmap="gray")
            axes[i + 1, j].set_title(f"{rec['name']} — Z{j + 1}", fontsize=7)
            axes[i + 1, j].set_axis_off()
    fig.suptitle("zoom_compare_sheet — v3 top variants at 4 zoom regions",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(sheet_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Variant runner
# --------------------------------------------------------------------------- #

VARIANT_FIELDS = [
    "variant_name", "mask_name", "black_threshold",
    "width_core", "halo_width", "keep_core", "keep_halo",
    "side_sigma", "feather_radius",
    "core_coverage_percent", "halo_coverage_percent",
    "edit_coverage_percent",
    "dir_bg_coverage_core_percent",
    "mean_abs_removed_core", "mean_abs_removed_halo",
    "mean_abs_removed_outside",
    "core_residual_after_suppress",
]


def variant_name_of(
    mask_tag: str, black_threshold: float,
    width_core: int, halo_width: int, keep_halo: float,
) -> str:
    return (
        f"{mask_tag}_bt{black_threshold:.1f}_wc{width_core}_h{halo_width}_"
        f"kh{keep_halo:.1f}"
    )


def run_variant(
    *,
    img_f: np.ndarray, img_dtype: np.dtype, bg_low: np.ndarray,
    core_mask: np.ndarray, halo_mask: np.ndarray, has_dir: np.ndarray,
    bg: np.ndarray,
    keep_core: float, keep_halo: float, feather_radius: float,
    variant_dir: Path, variant_name: str, mask_tag: str,
    black_threshold: float, width_core: int, halo_width: int,
    side_sigma: float,
    preview_lo: float, preview_hi: float,
    img_clip: tuple[float, float],
    raw8_full: np.ndarray,
    zoom_centers: list[tuple[int, int]], zoom_half: int,
) -> tuple[dict, dict[str, np.ndarray], np.ndarray]:
    suppressed = suppress_v3(
        img_f, bg, core_mask, halo_mask,
        keep_core=keep_core, keep_halo=keep_halo,
        feather_radius=feather_radius,
    )
    removed = img_f - suppressed
    variant_dir.mkdir(parents=True, exist_ok=True)

    save_mask(variant_dir / "core_mask.png", core_mask)
    save_mask(variant_dir / "halo_mask.png", halo_mask)
    overlay_img = overlay_core_halo(raw8_full, core_mask, halo_mask)
    save_uint8(variant_dir / "overlay.png", overlay_img)

    sup8 = to_uint8(suppressed, preview_lo, preview_hi)
    save_uint8(variant_dir / "suppressed_preview.png", sup8)

    abs_scale = max(1.0, float(np.percentile(np.abs(removed), 99.5)) * 1.5)
    removed_rgb = diff_heatmap(removed, abs_scale)
    save_uint8(variant_dir / "removed.png", removed_rgb)

    lo_c, hi_c = img_clip
    sup_int = np.clip(suppressed, lo_c, hi_c).astype(img_dtype)
    bg_int = np.clip(bg, lo_c, hi_c).astype(img_dtype)
    save_tif(variant_dir / "suppressed.tif", sup_int)
    save_tif(variant_dir / "background_estimate.tif", bg_int)

    save_summary_panel(
        variant_dir / "summary_panel.png",
        raw8_full, sup8, core_mask, halo_mask, removed, abs_scale, variant_name,
        zoom_centers, zoom_half,
    )

    edit_mask = core_mask | halo_mask
    outside = ~edit_mask
    mean_core = float(np.abs(removed[core_mask]).mean()) if core_mask.any() else 0.0
    mean_halo = float(np.abs(removed[halo_mask]).mean()) if halo_mask.any() else 0.0
    mean_out = float(np.abs(removed[outside]).mean()) if outside.any() else 0.0
    dir_cov = (
        float((has_dir & core_mask).sum() / max(core_mask.sum(), 1) * 100.0)
        if core_mask.any() else 0.0
    )
    # how dark is the core *after* suppression relative to bg_low?
    if core_mask.any():
        core_residual = float((bg_low[core_mask] - suppressed[core_mask]).mean())
    else:
        core_residual = 0.0

    metrics = {
        "variant_name": variant_name, "mask_name": mask_tag,
        "black_threshold": black_threshold,
        "width_core": width_core, "halo_width": halo_width,
        "keep_core": keep_core, "keep_halo": keep_halo,
        "side_sigma": side_sigma, "feather_radius": feather_radius,
        "core_coverage_percent": float(core_mask.mean() * 100.0),
        "halo_coverage_percent": float(halo_mask.mean() * 100.0),
        "edit_coverage_percent": float(edit_mask.mean() * 100.0),
        "dir_bg_coverage_core_percent": dir_cov,
        "mean_abs_removed_core": mean_core,
        "mean_abs_removed_halo": mean_halo,
        "mean_abs_removed_outside": mean_out,
        "core_residual_after_suppress": core_residual,
    }
    with open(variant_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    thumbs = {
        "overlay": downscale_uint8(overlay_img, max_side=520),
        "suppressed": downscale_uint8(sup8, max_side=520),
        "removed": downscale_uint8(removed_rgb, max_side=520),
    }
    return metrics, thumbs, sup8


# --------------------------------------------------------------------------- #
# Sweep definition
# --------------------------------------------------------------------------- #

MASK_FILES = {
    "C_sato90": ("mask_C_sato90.png", "C90b"),
    "D_sato85": ("mask_D_sato85.png", "D85b"),
}

DEFAULT_SWEEP = [
    # (mask_name, black_thresholds, core_widths, halo_widths, keep_halos)
    ("C_sato90", [0.5, 0.8, 1.0], [5, 7], [6, 8], [0.2, 0.3]),
    ("D_sato85", [0.5, 1.0],      [5, 7], [8],    [0.2, 0.3]),
]


def expand_sweep() -> list[dict]:
    out: list[dict] = []
    for mask_name, bts, wcs, hws, khs in DEFAULT_SWEEP:
        for bt, wc, hw, kh in product(bts, wcs, hws, khs):
            out.append({
                "mask_name": mask_name, "black_threshold": bt,
                "core_width": wc, "halo_width": hw, "keep_halo": kh,
            })
    return out


def priority_top_variants(records: list[dict], n: int = 5) -> list[dict]:
    """Pick top variants for the zoom_compare_sheet.

    Preference: keep_halo=0.3, core_width covering both 5 & 7, span
    black_threshold values, include one D85b.
    """
    by_name = {r["name"]: r for r in records}
    desired = [
        "C90b_bt0.5_wc5_h8_kh0.3",
        "C90b_bt0.8_wc5_h8_kh0.3",
        "C90b_bt1.0_wc5_h8_kh0.3",
        "C90b_bt0.5_wc7_h8_kh0.3",
        "D85b_bt0.5_wc7_h8_kh0.3",
    ]
    picked = [by_name[name] for name in desired if name in by_name]
    if len(picked) >= n:
        return picked[:n]
    for r in records:
        if r in picked:
            continue
        picked.append(r)
        if len(picked) >= n:
            break
    return picked[:n]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/data.tif")
    ap.add_argument("--prepared", default="outputs/prepare")
    ap.add_argument("--out", default="outputs/visual_line_hide_v3")
    ap.add_argument("--side-sigma", type=float, default=15.0)
    ap.add_argument("--side-radius", type=int, default=40)
    ap.add_argument("--feather-radius", type=float, default=2.0)
    ap.add_argument("--bg-sigma", type=float, default=20.0)
    ap.add_argument("--noise-sigma", type=float, default=8.0)
    ap.add_argument("--tensor-sigma", type=float, default=4.0)
    ap.add_argument("--keep-core", type=float, default=0.0)
    ap.add_argument("--zoom-crop-size", type=int, default=480)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dir-reduction", choices=["median", "mean"],
                    default="median")
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

    # --- Image / dtype --- #
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

    # --- Black-line score --- #
    print("[1/4] black-line score")
    score, bg_low, mad = compute_black_line_score(
        img_f, sigma_bg=args.bg_sigma, sigma_noise=args.noise_sigma,
    )
    sc_clip = float(np.percentile(score, 99.5))
    save_uint8(
        out_root / "black_line_score_preview.png",
        to_uint8(score, 0.0, max(sc_clip, 1e-6)),
    )
    with open(out_root / "reports" / "black_line_score_stats.json", "w") as f:
        json.dump({
            "bg_sigma": args.bg_sigma, "noise_sigma": args.noise_sigma,
            "global_mad": mad, "score_p99_5": sc_clip,
            "score_p99": float(np.percentile(score, 99)),
            "score_max": float(score.max()),
            "score_coverage_at_0.5_pct": float((score > 0.5).mean() * 100),
            "score_coverage_at_0.8_pct": float((score > 0.8).mean() * 100),
            "score_coverage_at_1.0_pct": float((score > 1.0).mean() * 100),
        }, f, indent=2)
    print(f"  global_mad={mad:.4f}, score p99.5={sc_clip:.2f}")

    # --- Structure tensor / normal direction --- #
    print("[2/4] structure tensor + ridge normal")
    cos_n, sin_n, coh = compute_normal_direction(
        img_f, sigma_pre=1.0, sigma_tensor=args.tensor_sigma,
    )
    save_uint8(
        out_root / "coherence_preview.png",
        to_uint8(coh, 0.0, 1.0),
    )
    print(f"  mean coherence={coh.mean():.3f}")

    # --- Masks --- #
    protect = load_mask(prepared / "masks" / "protect_mask.png")
    print(f"  protect_mask coverage={protect.mean()*100:.3f}%")
    broad_path = prepared / "masks" / "mask_F_sato75.png"
    if broad_path.exists():
        broad_line_mask = load_mask(broad_path)
        print(f"  broad_line_mask (sato75) coverage={broad_line_mask.mean()*100:.2f}%")
    else:
        broad_line_mask = load_mask(prepared / "masks" / "mask_D_sato85.png")
        print(f"  broad_line_mask (fallback sato85) coverage={broad_line_mask.mean()*100:.2f}%")

    sato_masks: dict[str, np.ndarray] = {}
    for name in MASK_FILES:
        path = prepared / "masks" / MASK_FILES[name][0]
        sato_masks[name] = load_mask(path)
        print(f"  {name}: coverage={sato_masks[name].mean()*100:.2f}%")

    # --- Zoom crops --- #
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

    # --- Sweep --- #
    sweep = expand_sweep()
    print(f"[3/4] sweep: {len(sweep)} variants")

    # Group by (mask_name, bt, wc, hw) so we share the directional bg
    groups: dict[tuple[str, float, int, int], list[dict]] = {}
    for s in sweep:
        key = (s["mask_name"], s["black_threshold"], s["core_width"], s["halo_width"])
        groups.setdefault(key, []).append(s)

    distances_for = lambda wc: [wc + 2, wc + 4, wc + 8, wc + 12]
    csv_rows: list[dict] = []
    records: list[dict] = []
    suppressed8_lookup: dict[str, np.ndarray] = {}
    t0 = time.time()
    done = 0

    for (mask_name, bt, wc, hw), variants in groups.items():
        sato_m = sato_masks[mask_name]
        mask_tag = MASK_FILES[mask_name][1]

        core, halo, _ = build_core_halo(
            sato_m, score, protect,
            width_core=wc, halo_width=hw, black_threshold=bt,
        )
        edit_mask = core | halo
        if not edit_mask.any():
            print(f"  [skip] {mask_tag}_bt{bt}_wc{wc}_h{hw}: empty edit_mask")
            continue

        # Side-band bg (fallback)
        side_bg = side_band_bg(
            img_f, edit_mask, protect,
            side_radius=args.side_radius, sigma=args.side_sigma,
        )

        # Directional bg
        forbid = edit_mask | broad_line_mask | protect
        dist = distances_for(wc)
        bg, has_dir, n_valid = directional_bg(
            img_f, forbid, side_bg, cos_n, sin_n,
            distances=dist, reduction=args.dir_reduction,
        )

        for s in variants:
            if args.limit is not None and done >= args.limit:
                break
            kh = s["keep_halo"]
            name = variant_name_of(mask_tag, bt, wc, hw, kh)
            metrics, thumbs, sup8 = run_variant(
                img_f=img_f, img_dtype=img_dtype, bg_low=bg_low,
                core_mask=core, halo_mask=halo, has_dir=has_dir, bg=bg,
                keep_core=args.keep_core, keep_halo=kh,
                feather_radius=args.feather_radius,
                variant_dir=out_root / "variants" / name,
                variant_name=name, mask_tag=mask_tag,
                black_threshold=bt, width_core=wc, halo_width=hw,
                side_sigma=args.side_sigma,
                preview_lo=preview_lo, preview_hi=preview_hi,
                img_clip=img_clip, raw8_full=raw8_full,
                zoom_centers=zoom_centers, zoom_half=zoom_half,
            )
            csv_rows.append(metrics)
            records.append({"name": name, "metrics": metrics, "thumbs": thumbs})
            suppressed8_lookup[name] = sup8
            done += 1
            if done % 4 == 0 or done == 1:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                print(f"  [{done}/{len(sweep)}] {name}  "
                      f"({elapsed:.1f}s, {rate:.2f}/s, "
                      f"dir_bg_cov_core={metrics['dir_bg_coverage_core_percent']:.1f}%)")
        if args.limit is not None and done >= args.limit:
            break

    print(f"finished {done} variants in {time.time() - t0:.1f}s")

    # --- Reports --- #
    print("[4/4] reports")
    reports_dir = out_root / "reports"
    csv_path = reports_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VARIANT_FIELDS)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"  wrote {csv_path}")

    # compare_sheet: keep_halo=0.3 subset
    compare_subset = [r for r in records
                      if abs(r["metrics"]["keep_halo"] - 0.3) < 1e-6]
    if not compare_subset:
        compare_subset = records[:16]
    save_compare_sheet(
        compare_subset, reports_dir / "compare_sheet.png",
        title=f"v3 compare_sheet — keep_halo=0.3 (rows={len(compare_subset)})",
    )
    print(f"  wrote {reports_dir / 'compare_sheet.png'} ({len(compare_subset)} rows)")

    if len(records) != len(compare_subset):
        save_compare_sheet(
            records, reports_dir / "compare_sheet_all.png",
            title=f"v3 compare_sheet_all (rows={len(records)})",
        )
        print(f"  wrote {reports_dir / 'compare_sheet_all.png'} ({len(records)} rows)")

    save_contact_sheet(records, reports_dir / "contact_sheet.png")
    print(f"  wrote {reports_dir / 'contact_sheet.png'}")

    # zoom_compare_sheet
    top = priority_top_variants(records, n=5)
    save_zoom_compare_sheet(
        reports_dir / "zoom_compare_sheet.png",
        raw8_full, zoom_centers, zoom_half, top, suppressed8_lookup,
    )
    print(f"  wrote {reports_dir / 'zoom_compare_sheet.png'} "
          f"({len(top)} top variants × {len(zoom_centers)} zooms)")


if __name__ == "__main__":
    main()
