"""Stage 2j — Visual Line Hide v5 UNet refinement.

v5_unet review verdict: ``v5_ct0.50_ht0.45_cl0`` is the strongest
line-removal so far but introduces visible scrub marks / bright bands /
over-modified background. The mask is *fine* — the problem is the
suppression being too aggressive.

This script keeps the v5_unet baseline mask exactly:

    line_prob    : Route B v1 UNet output
    core_th      : 0.50
    halo_th      : 0.45
    closing_len  : 0
    halo_dilate  : 2

…and only varies how the replacement is *applied*:

1. **Soft core replacement** — replace
   ``out_core = bg_dir`` (hard, v5_unet)
   with
   ``out_core = raw * (1 - core_mix) + bg_dir * core_mix``.
   `core_mix=1.0` reproduces v5_unet; lower values preserve some raw.

2. **Distance-based alpha (mode="distance_core")** — vary the core
   replacement strength from center to edge. At core center the gamma
   reaches ``core_mix``; at the core boundary it falls back to
   ``1 - keep_halo`` (matching halo treatment, so the transition is
   continuous). This is a clean monotone gradient via
   ``alpha = clip(distance_inside_core / max_dist, 0, 1)``.

3. **Halo attenuation knob** — sweep ``keep_halo`` ∈ {0.3, 0.5, 0.7}.
   Higher values keep more of the original halo raw, reducing brush
   marks at the line shoulders.

4. **Background texture preservation** — optionally add a low-pass of
   non-line residual back to ``bg_dir``:
   ``bg = bg_dir + texture_scale * local_texture``.

Sweep budget: 3 core_mix × 3 keep_halo × 2 mode × 1 texture_scale = 18
variants. ``texture_scale`` defaults to 0.0; can be re-run on a small
set with ``--texture-scales 0.0 0.15`` for a phase-2 comparison.
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
from skimage.morphology import binary_dilation, disk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# IO helpers (lifted from v5_deepmask)
# --------------------------------------------------------------------------- #

def load_tif(path: Path) -> np.ndarray:
    return tifffile.imread(str(path))


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(str(path)).convert("L")) > 127


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(str(path))


def save_uint8(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(str(path))


def save_tif(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr, compression="zlib")


def load_line_prob(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.max() > 1.0 or arr.min() < 0.0:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        arr = (arr - lo) / max(hi - lo, 1e-6)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Structure tensor / side-band / directional bg (lifted from v5_deepmask)
# --------------------------------------------------------------------------- #

def compute_normal_direction(
    img_f: np.ndarray, sigma_pre: float = 1.0, sigma_tensor: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    img_s = ndimage.gaussian_filter(img_f, sigma=sigma_pre)
    gx = ndimage.sobel(img_s, axis=1)
    gy = ndimage.sobel(img_s, axis=0)
    Jxx = ndimage.gaussian_filter(gx * gx, sigma=sigma_tensor)
    Jxy = ndimage.gaussian_filter(gx * gy, sigma=sigma_tensor)
    Jyy = ndimage.gaussian_filter(gy * gy, sigma=sigma_tensor)
    theta_n = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)
    return np.cos(theta_n).astype(np.float32), np.sin(theta_n).astype(np.float32)


def side_band_bg(
    img_f: np.ndarray, edit_mask: np.ndarray, protect_mask: np.ndarray,
    side_radius: int, sigma: float,
) -> np.ndarray:
    outer = binary_dilation(edit_mask, disk(side_radius))
    band = outer & ~edit_mask & ~protect_mask
    weight = band.astype(np.float32)
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
            samples.append(s); valids.append((f < 0.5).astype(np.float32))
    samp = np.stack(samples, axis=0)
    val = np.stack(valids, axis=0)
    has_dir = val.sum(axis=0) > 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        masked = np.where(val.astype(bool), samp, np.nan)
        bg_dir = np.nanmedian(masked, axis=0)
        bg_dir = np.where(np.isnan(bg_dir), 0.0, bg_dir)
    bg = np.where(has_dir, bg_dir, side_bg_fallback).astype(np.float32)
    return bg, has_dir


# --------------------------------------------------------------------------- #
# Protect split (lifted from v4)
# --------------------------------------------------------------------------- #

def split_protect_mask(
    protect_mask: np.ndarray, broad_line_mask: np.ndarray,
    dilate_radius: int = 3, overlap_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    broad_dilated = binary_dilation(broad_line_mask, disk(dilate_radius))
    lbl = measure.label(protect_mask, connectivity=2)
    pt = np.zeros_like(protect_mask, dtype=bool)
    pl = np.zeros_like(protect_mask, dtype=bool)
    for region in measure.regionprops(lbl):
        coords = region.coords
        on_line = int(broad_dilated[coords[:, 0], coords[:, 1]].sum()) / max(region.area, 1)
        if on_line > overlap_threshold:
            pl[coords[:, 0], coords[:, 1]] = True
        else:
            pt[coords[:, 0], coords[:, 1]] = True
    return pt, pl


# --------------------------------------------------------------------------- #
# Local texture estimation (NEW in refine)
# --------------------------------------------------------------------------- #

def compute_local_texture(
    img_f: np.ndarray, broad_line_mask: np.ndarray,
    sigma_bg: float = 20.0, sigma_smooth: float = 8.0,
) -> np.ndarray:
    """Smooth, line-free estimate of the fine-scale texture residual.

    Compute ``raw - gaussian(raw, sigma_bg)`` (everywhere positive variation
    above the slow trend), mask to non-line areas, then gaussian-blur with
    a smaller sigma so the estimate naturally extends into line regions.
    Returns a zero-mean, signed map in raw intensity units.
    """
    bg_low = ndimage.gaussian_filter(img_f, sigma=sigma_bg)
    residual = (img_f - bg_low).astype(np.float32)
    weight = (~broad_line_mask).astype(np.float32)
    num = ndimage.gaussian_filter(residual * weight, sigma=sigma_smooth)
    den = ndimage.gaussian_filter(weight, sigma=sigma_smooth)
    return (num / np.maximum(den, 1e-6)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Refined suppression (NEW)
# --------------------------------------------------------------------------- #

def suppress_refine(
    img_f: np.ndarray, bg_dir: np.ndarray,
    core_mask: np.ndarray, halo_mask: np.ndarray,
    core_mix: float, keep_halo: float, mode: str,
    distance_max: float = 5.0, feather_radius: float = 2.0,
    texture: np.ndarray | None = None, texture_scale: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply refined suppression.

    Per-pixel mixing:
        out = raw * (1 - gamma) + bg * gamma
    where bg = bg_dir + texture_scale * texture, and
    gamma is built per-region:
        outside edit:    gamma = 0
        halo:            gamma = 1 - keep_halo
        core (hard_core): gamma = core_mix
        core (distance_core): gamma = lerp(1 - keep_halo, core_mix, alpha)
            with alpha = clip(distance_inside_core / distance_max, 0, 1)

    The hard core boundary (mode=="hard_core") has a small feather applied
    *to gamma* across the outer boundary of (core|halo) to soften the join.
    """
    if texture is not None and texture_scale != 0.0:
        bg = bg_dir + texture_scale * texture
    else:
        bg = bg_dir

    gamma = np.zeros_like(img_f, dtype=np.float32)
    if halo_mask.any():
        gamma[halo_mask] = 1.0 - keep_halo
    if core_mask.any():
        if mode == "hard_core":
            gamma[core_mask] = float(core_mix)
        elif mode == "distance_core":
            d_inside = ndimage.distance_transform_edt(core_mask).astype(np.float32)
            alpha = np.clip(d_inside / max(distance_max, 1e-6), 0.0, 1.0)
            edge_gamma = 1.0 - keep_halo
            gamma_core = (1.0 - alpha) * edge_gamma + alpha * float(core_mix)
            gamma[core_mask] = gamma_core[core_mask]
        else:
            raise ValueError(f"Unknown mode {mode!r}")

    edit_mask = core_mask | halo_mask
    if feather_radius > 0 and edit_mask.any():
        dist_outside = ndimage.distance_transform_edt(~edit_mask)
        outer_alpha = np.clip(1.0 - dist_outside / float(feather_radius), 0.0, 1.0)
        outer_alpha[edit_mask] = 1.0
        gamma = gamma * outer_alpha  # smooth taper to 0 just outside

    out = img_f * (1.0 - gamma) + bg * gamma
    return out.astype(np.float32), gamma


# --------------------------------------------------------------------------- #
# Preview helpers
# --------------------------------------------------------------------------- #

def percentile_range(img: np.ndarray, lo_p: float = 1.0, hi_p: float = 99.0):
    lo = float(np.percentile(img, lo_p))
    hi = float(np.percentile(img, hi_p))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    return lo, hi


def to_uint8(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    a = (arr.astype(np.float32) - lo) / (hi - lo)
    return (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def overlay_core_halo(gray8: np.ndarray, core, halo,
                      core_color=(220, 30, 30), halo_color=(240, 170, 30),
                      alpha: float = 0.55) -> np.ndarray:
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
# Zoom crops
# --------------------------------------------------------------------------- #

def select_zoom_crops(density: np.ndarray, n: int = 4, crop_size: int = 480,
                      margin: int = 20):
    H, W = density.shape
    smoothed = ndimage.gaussian_filter(density.astype(np.float32), sigma=12.0)
    work = smoothed.copy()
    half = crop_size // 2
    work[:half + margin] = -np.inf
    work[-half - margin:] = -np.inf
    work[:, :half + margin] = -np.inf
    work[:, -half - margin:] = -np.inf
    crops = []
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
    return arr[y0:y0 + 2 * half, x0:x0 + 2 * half].copy()


# --------------------------------------------------------------------------- #
# Per-variant panel + sheets
# --------------------------------------------------------------------------- #

def save_summary_panel(out_path: Path, raw8, sup8, gamma_map, removed,
                       abs_scale, title, zoom_centers, zoom_half) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 9.0))
    axes[0, 0].imshow(raw8, cmap="gray"); axes[0, 0].set_title("raw")
    axes[0, 1].imshow(gamma_map, cmap="magma", vmin=0, vmax=1)
    axes[0, 1].set_title("gamma (bg replacement strength)")
    axes[0, 2].imshow(sup8, cmap="gray"); axes[0, 2].set_title("refine suppressed")
    axes[0, 3].imshow(removed, cmap="seismic", vmin=-abs_scale, vmax=abs_scale)
    axes[0, 3].set_title("raw - refine")
    for j in range(4):
        axes[0, j].set_axis_off()
    for j in range(4):
        ax = axes[1, j]
        if j >= len(zoom_centers):
            ax.set_axis_off(); continue
        cy, cx = zoom_centers[j]
        ax.imshow(extract_crop(sup8, cy, cx, zoom_half), cmap="gray")
        ax.set_title(f"Z{j+1} refine (y={cy}, x={cx})", fontsize=8)
        ax.set_axis_off()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_contact_sheet(records, path, cols: int = 6) -> None:
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
    fig.suptitle(f"contact_sheet — refine suppressed ({n} variants)")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_compare_sheet(records, path, title=None) -> None:
    n = len(records)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 3, figsize=(9.5, n * 2.0))
    if n == 1:
        axes = axes.reshape(1, -1)
    for r, rec in enumerate(records):
        axes[r, 0].imshow(rec["thumbs"]["gamma"], cmap="magma", vmin=0, vmax=1)
        axes[r, 0].set_title(f"{rec['name']} | gamma", fontsize=7)
        axes[r, 1].imshow(rec["thumbs"]["suppressed"], cmap="gray")
        axes[r, 1].set_title("refine suppressed", fontsize=7)
        axes[r, 2].imshow(rec["thumbs"]["removed"])
        axes[r, 2].set_title("raw - refine", fontsize=7)
        for c in range(3):
            axes[r, c].set_axis_off()
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_zoom_compare_sheet(path, raw8, v5unet_baseline_sup8,
                            zoom_centers, zoom_half, top_records,
                            suppressed8_lookup) -> None:
    n_zoom = len(zoom_centers)
    rows = 2 + len(top_records)
    cols = n_zoom
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    for j, (cy, cx) in enumerate(zoom_centers):
        axes[0, j].imshow(extract_crop(raw8, cy, cx, zoom_half), cmap="gray")
        axes[0, j].set_title(f"raw — Z{j+1} (y={cy}, x={cx})", fontsize=8)
        axes[0, j].set_axis_off()
        axes[1, j].imshow(extract_crop(v5unet_baseline_sup8, cy, cx, zoom_half), cmap="gray")
        axes[1, j].set_title(f"v5_unet baseline — Z{j+1}", fontsize=8)
        axes[1, j].set_axis_off()
    for i, rec in enumerate(top_records):
        sup8 = suppressed8_lookup[rec["name"]]
        for j, (cy, cx) in enumerate(zoom_centers):
            axes[i + 2, j].imshow(extract_crop(sup8, cy, cx, zoom_half), cmap="gray")
            axes[i + 2, j].set_title(f"{rec['name']} — Z{j+1}", fontsize=7)
            axes[i + 2, j].set_axis_off()
    fig.suptitle("zoom_compare_sheet — refine top variants vs raw / v5_unet baseline",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_baseline_compare(path, raw8, v5unet_baseline_sup8, refine_best_sup8,
                          refine_best_name, zoom_centers, zoom_half) -> None:
    n_zoom = len(zoom_centers)
    fig, axes = plt.subplots(3, 1 + n_zoom, figsize=((1 + n_zoom) * 3.0, 9.0))
    rows = [
        (raw8, "raw"),
        (v5unet_baseline_sup8, "v5_unet baseline"),
        (refine_best_sup8, f"refine best  {refine_best_name}"),
    ]
    for r, (img, lbl) in enumerate(rows):
        axes[r, 0].imshow(img, cmap="gray")
        axes[r, 0].set_title(lbl + " — full", fontsize=9)
        axes[r, 0].set_axis_off()
        for j, (cy, cx) in enumerate(zoom_centers):
            axes[r, 1 + j].imshow(extract_crop(img, cy, cx, zoom_half), cmap="gray")
            axes[r, 1 + j].set_title(f"{lbl} — Z{j+1}", fontsize=8)
            axes[r, 1 + j].set_axis_off()
    fig.suptitle("baseline_compare — raw / v5_unet baseline / refine best",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #

VARIANT_FIELDS = [
    "variant_name", "core_mix", "keep_halo", "mode", "texture_scale",
    "core_coverage_percent", "edit_coverage_percent",
    "mean_abs_removed_core", "mean_abs_removed_halo",
    "mean_abs_removed_outside",
    "gamma_mean_core", "gamma_mean_halo",
]


def variant_name_of(core_mix: float, keep_halo: float, mode: str,
                    texture_scale: float) -> str:
    mtag = {"hard_core": "hc", "distance_core": "dc"}[mode]
    return f"v5r_cm{core_mix:.2f}_kh{keep_halo:.1f}_{mtag}_ts{texture_scale:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/data.tif")
    ap.add_argument("--prepared", default="outputs/prepare")
    ap.add_argument("--out", default="outputs/visual_line_hide_v5_unet_refine")
    ap.add_argument(
        "--line-prob",
        default="/e/dapro/dapro-deep-mask-v1/outputs/route_b_line_mask_v1/line_prob.tif",
        help="Path to Route B UNet line_prob.tif (fixed for this refine script).",
    )
    # Fixed baseline (v5_unet best mask)
    ap.add_argument("--core-th", type=float, default=0.50)
    ap.add_argument("--halo-th", type=float, default=0.45)
    ap.add_argument("--halo-dilate", type=int, default=2)
    ap.add_argument("--closing-len", type=int, default=0,
                    help="Fixed at 0; v5_unet review showed closing introduces gray bands.")

    # Refine sweep
    ap.add_argument("--core-mixes", type=float, nargs="+",
                    default=[0.70, 0.85, 1.00])
    ap.add_argument("--keep-halos", type=float, nargs="+",
                    default=[0.3, 0.5, 0.7])
    ap.add_argument("--modes", nargs="+",
                    default=["hard_core", "distance_core"],
                    choices=["hard_core", "distance_core"])
    ap.add_argument("--texture-scales", type=float, nargs="+",
                    default=[0.0])

    # Suppression internals (carried over from v5_deepmask main pick)
    ap.add_argument("--feather-radius", type=float, default=2.0)
    ap.add_argument("--side-sigma", type=float, default=15.0)
    ap.add_argument("--side-radius", type=int, default=40)
    ap.add_argument("--tensor-sigma", type=float, default=4.0)
    ap.add_argument("--core-distances", type=int, nargs="+",
                    default=[9, 11, 15, 19])
    ap.add_argument("--distance-max", type=float, default=5.0,
                    help="Distance-transform normalisation for distance_core mode")
    ap.add_argument("--zoom-crop-size", type=int, default=480)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--v5unet-baseline-preview", default=None,
                    help="Optional path to v5_unet baseline suppressed_preview.png "
                         "used in zoom_compare and baseline_compare; otherwise "
                         "derived from the script's own hard_core/cm=1.0/kh=0.3 variant.")
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
    print(f"line-prob    = {args.line_prob}")

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

    # ---- line_prob ---- #
    line_prob = load_line_prob(Path(args.line_prob))
    print(f"line_prob p50={np.percentile(line_prob, 50):.3f} "
          f"p90={np.percentile(line_prob, 90):.3f} "
          f"p99={np.percentile(line_prob, 99):.3f}")

    # ---- Build fixed core / halo masks ---- #
    protect_full = load_mask(prepared / "masks" / "protect_mask.png")
    broad_path = prepared / "masks" / "mask_F_sato75.png"
    broad_line_mask = (
        load_mask(broad_path) if broad_path.exists()
        else load_mask(prepared / "masks" / "mask_D_sato85.png")
    )
    protect_true_blob, protect_on_line = split_protect_mask(
        protect_full, broad_line_mask, dilate_radius=3, overlap_threshold=0.3,
    )
    save_mask(out_root / "protect_true_blob.png", protect_true_blob)
    save_mask(out_root / "protect_on_line.png", protect_on_line)

    core = line_prob > args.core_th
    halo_seed = line_prob > args.halo_th
    if args.halo_dilate > 0:
        halo_seed = binary_dilation(halo_seed, disk(args.halo_dilate))
    halo = halo_seed & ~core
    core = core & ~protect_true_blob
    halo = halo & ~protect_true_blob

    print(f"  core_cov={core.mean()*100:.2f}%  halo_cov={halo.mean()*100:.2f}%  "
          f"edit_cov={(core|halo).mean()*100:.2f}%")
    save_mask(out_root / "core_mask.png", core)
    save_mask(out_root / "halo_mask.png", halo)

    # ---- Structure tensor + bg ---- #
    cos_n, sin_n = compute_normal_direction(
        img_f, sigma_pre=1.0, sigma_tensor=args.tensor_sigma,
    )
    edit = core | halo
    side_bg = side_band_bg(
        img_f, edit, protect_full,
        side_radius=args.side_radius, sigma=args.side_sigma,
    )
    forbid = edit | broad_line_mask | protect_full
    bg_dir, has_dir = directional_bg(
        img_f, forbid, side_bg, cos_n, sin_n, distances=args.core_distances,
    )
    print(f"  dir_bg coverage on core = "
          f"{(has_dir & core).sum() / max(core.sum(), 1)*100:.1f}%")

    # ---- Local texture ---- #
    texture = compute_local_texture(
        img_f, broad_line_mask, sigma_bg=20.0, sigma_smooth=8.0,
    )

    # ---- Zoom crops ---- #
    sato_response = load_tif(prepared / "responses" / "response_sato.tif")
    zoom_centers, zoom_half = select_zoom_crops(
        sato_response, n=4, crop_size=args.zoom_crop_size,
    )
    print(f"  zoom_crops: {zoom_centers} half={zoom_half}")
    with open(out_root / "reports" / "zoom_crops.json", "w") as f:
        json.dump({"centers": zoom_centers, "half": zoom_half,
                   "crop_size": args.zoom_crop_size}, f, indent=2)

    # ---- v5_unet baseline reference for compare images ---- #
    if args.v5unet_baseline_preview and Path(args.v5unet_baseline_preview).exists():
        v5_baseline_8 = np.asarray(
            Image.open(args.v5unet_baseline_preview).convert("L")
        )
    else:
        # Synthesize on the fly using hard_core + cm=1.0 + kh=0.3 (v5_unet equivalent)
        v5_sup, _ = suppress_refine(
            img_f, bg_dir, core, halo,
            core_mix=1.0, keep_halo=0.3, mode="hard_core",
            distance_max=args.distance_max, feather_radius=args.feather_radius,
            texture=None, texture_scale=0.0,
        )
        v5_baseline_8 = to_uint8(v5_sup, preview_lo, preview_hi)
    save_uint8(out_root / "v5_unet_baseline_preview.png", v5_baseline_8)

    # ---- Sweep ---- #
    combos = list(product(args.core_mixes, args.keep_halos,
                          args.modes, args.texture_scales))
    print(f"[sweep] {len(combos)} variants")
    csv_rows: list[dict] = []
    records: list[dict] = []
    suppressed8_lookup: dict[str, np.ndarray] = {}
    t0 = time.time()
    done = 0

    for core_mix, keep_halo, mode, texture_scale in combos:
        if args.limit is not None and done >= args.limit:
            break
        name = variant_name_of(core_mix, keep_halo, mode, texture_scale)
        suppressed, gamma_map = suppress_refine(
            img_f, bg_dir, core, halo,
            core_mix=core_mix, keep_halo=keep_halo, mode=mode,
            distance_max=args.distance_max, feather_radius=args.feather_radius,
            texture=texture, texture_scale=texture_scale,
        )

        variant_dir = out_root / "variants" / name
        variant_dir.mkdir(parents=True, exist_ok=True)

        sup8 = to_uint8(suppressed, preview_lo, preview_hi)
        save_uint8(variant_dir / "suppressed_preview.png", sup8)
        suppressed8_lookup[name] = sup8

        removed = img_f - suppressed
        abs_scale = max(1.0, float(np.percentile(np.abs(removed), 99.5)) * 1.5)
        removed_rgb = diff_heatmap(removed, abs_scale)
        save_uint8(variant_dir / "removed.png", removed_rgb)
        save_uint8(
            variant_dir / "gamma_map.png",
            (np.clip(gamma_map, 0, 1) * 255 + 0.5).astype(np.uint8),
        )
        save_tif(
            variant_dir / "suppressed.tif",
            np.clip(suppressed, *img_clip).astype(img_dtype),
        )
        save_summary_panel(
            variant_dir / "summary_panel.png",
            raw8_full, sup8, gamma_map, removed, abs_scale,
            f"{name}   cm={core_mix:.2f}  kh={keep_halo:.1f}  mode={mode}  "
            f"ts={texture_scale:.2f}",
            zoom_centers, zoom_half,
        )

        mean_core = float(np.abs(removed[core]).mean()) if core.any() else 0.0
        mean_halo = float(np.abs(removed[halo]).mean()) if halo.any() else 0.0
        outside = ~(core | halo)
        mean_out = float(np.abs(removed[outside]).mean()) if outside.any() else 0.0
        gm_core = float(gamma_map[core].mean()) if core.any() else 0.0
        gm_halo = float(gamma_map[halo].mean()) if halo.any() else 0.0
        metrics = {
            "variant_name": name,
            "core_mix": core_mix, "keep_halo": keep_halo, "mode": mode,
            "texture_scale": texture_scale,
            "core_coverage_percent": float(core.mean() * 100.0),
            "edit_coverage_percent": float((core | halo).mean() * 100.0),
            "mean_abs_removed_core": mean_core,
            "mean_abs_removed_halo": mean_halo,
            "mean_abs_removed_outside": mean_out,
            "gamma_mean_core": gm_core,
            "gamma_mean_halo": gm_halo,
        }
        with open(variant_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        csv_rows.append(metrics)
        records.append({
            "name": name,
            "metrics": metrics,
            "thumbs": {
                "gamma": downscale_uint8(
                    (np.clip(gamma_map, 0, 1) * 255 + 0.5).astype(np.uint8),
                    max_side=520),
                "suppressed": downscale_uint8(sup8, max_side=520),
                "removed": downscale_uint8(removed_rgb, max_side=520),
            },
        })

        done += 1
        elapsed = time.time() - t0
        print(f"  [{done}/{len(combos)}] {name}  "
              f"rm_core={mean_core:.1f} rm_halo={mean_halo:.1f} "
              f"gm_core={gm_core:.2f} gm_halo={gm_halo:.2f}  "
              f"({elapsed:.1f}s)")

    print(f"finished {done} variants in {time.time() - t0:.1f}s")

    # ---- Reports ---- #
    reports = out_root / "reports"
    csv_path = reports / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VARIANT_FIELDS)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"wrote {csv_path}")

    save_contact_sheet(records, reports / "contact_sheet.png")
    save_compare_sheet(records, reports / "compare_sheet.png",
                       title=f"refine compare_sheet (rows={len(records)})")

    # Top picks: prefer smaller |gamma_mean_core - gamma_mean_halo|
    # (smoother transition), then lower edit removed_outside.
    def _key(rec):
        m = rec["metrics"]
        return (
            abs(m["gamma_mean_core"] - m["gamma_mean_halo"]),
            -m["mean_abs_removed_core"],
            m["mean_abs_removed_halo"],
        )
    top = sorted(records, key=_key)[:5]
    save_zoom_compare_sheet(
        reports / "zoom_compare_sheet.png",
        raw8_full, v5_baseline_8, zoom_centers, zoom_half, top,
        suppressed8_lookup,
    )

    refine_best = top[0]
    save_baseline_compare(
        reports / "baseline_compare.png",
        raw8_full, v5_baseline_8, suppressed8_lookup[refine_best["name"]],
        refine_best["name"], zoom_centers, zoom_half,
    )
    print(f"refine main pick (auto): {refine_best['name']}")
    print("done.")


if __name__ == "__main__":
    main()
