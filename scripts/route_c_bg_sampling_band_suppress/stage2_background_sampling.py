"""Stage 2 - Kossel line local random background sampling test.

This stage does NOT train a model and does NOT detect defects.
It only tests whether labelled Kossel line regions can be made to visually
blend into the background by local random patch sampling + soft blending.

Run:
    python stage2_background_sampling.py \
        --input data.tif \
        --dataset line_stage1_dataset \
        --out stage2_bg_sampling_test

If model probability is available:
    python stage2_background_sampling.py \
        --input data.tif \
        --dataset line_stage1_dataset \
        --line-prob line_stage1_infer_v1/line_prob.tif \
        --out stage2_bg_sampling_test
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage
from skimage import measure, morphology
from skimage.morphology import disk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------- I/O helpers ------------------------------- #


def find_file(*candidates):
    for c in candidates:
        if c is None:
            continue
        p = Path(c)
        if p.exists():
            return p
    return None


def load_grayscale(path):
    p = Path(path)
    if p.suffix.lower() in (".tif", ".tiff"):
        img = tifffile.imread(str(p))
    else:
        img = np.array(Image.open(str(p)))
    if img.ndim == 3:
        if img.shape[-1] in (3, 4):
            img = img[..., :3].mean(axis=-1)
        elif img.shape[0] in (1, 3, 4):
            img = img[0] if img.shape[0] == 1 else img[:3].mean(axis=0)
        else:
            img = img[..., 0]
    return img


def to_preview_uint8(arr, lo=None, hi=None):
    a = np.asarray(arr, dtype=np.float32)
    if lo is None:
        lo = float(np.percentile(a, 1))
    if hi is None:
        hi = float(np.percentile(a, 99))
    if hi <= lo:
        hi = lo + 1.0
    a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return (a * 255).astype(np.uint8)


def removed_to_uint8(removed):
    a = np.asarray(removed, dtype=np.float32)
    m = float(np.percentile(np.abs(a), 99))
    if m < 1e-6:
        m = 1e-6
    a = np.clip(a / m, -1.0, 1.0)
    return ((a + 1.0) * 0.5 * 255).astype(np.uint8)


def uncertainty_to_uint8(unc):
    a = np.asarray(unc, dtype=np.float32)
    hi = float(np.percentile(a, 99))
    if hi < 1e-6:
        return np.zeros_like(a, dtype=np.uint8)
    return (np.clip(a / hi, 0.0, 1.0) * 255).astype(np.uint8)


def make_overlay(gray_uint8, mask, color=(255, 0, 0), alpha=0.5):
    rgb = np.stack([gray_uint8] * 3, axis=-1).astype(np.float32)
    m = mask.astype(bool)
    for c in range(3):
        rgb[..., c] = np.where(m, color[c] * alpha + rgb[..., c] * (1 - alpha), rgb[..., c])
    return np.clip(rgb, 0, 255).astype(np.uint8)


# ----------------------------- mask building ------------------------------ #


def build_raw_masks(label, sato_norm, line_prob=None):
    masks = {}
    masks["A_clear_only"] = (label == 1)
    p90 = float(np.percentile(sato_norm, 90))
    p85 = float(np.percentile(sato_norm, 85))
    p80 = float(np.percentile(sato_norm, 80))
    ignore = (label == 255)
    masks["B_clear_sato90"] = (label == 1) | (ignore & (sato_norm > p90))
    masks["C_clear_sato85"] = (label == 1) | (ignore & (sato_norm > p85))
    masks["D_clear_sato80"] = (label == 1) | (ignore & (sato_norm > p80))
    if line_prob is not None:
        masks["E_prob05"] = line_prob > 0.5
        masks["F_prob03"] = line_prob > 0.3
    return masks


def postprocess_mask(mask, area_min=8, dilate_radius=1):
    m = morphology.remove_small_objects(mask.astype(bool), min_size=area_min)
    if dilate_radius > 0:
        m = morphology.binary_dilation(m, disk(dilate_radius))
    return m


# ------------------------------ protect mask ------------------------------ #


def compute_protect_mask(img_f, sigma=15.0, percentile=99.3,
                         area_min=4, area_max=500,
                         ecc_max=0.9, aspect_max=4.0):
    local_bg = ndimage.gaussian_filter(img_f, sigma=sigma)
    residual = img_f - local_bg
    abs_res = np.abs(residual)
    thr = float(np.percentile(abs_res, percentile))
    cand = abs_res > thr
    labels = measure.label(cand, connectivity=2)
    protect = np.zeros_like(cand, dtype=bool)
    for region in measure.regionprops(labels):
        if not (area_min <= region.area <= area_max):
            continue
        if region.eccentricity > ecc_max:
            continue
        minr, minc, maxr, maxc = region.bbox
        h, w = max(maxr - minr, 1), max(maxc - minc, 1)
        ar = max(h, w) / min(h, w)
        if ar > aspect_max:
            continue
        coords = region.coords
        protect[coords[:, 0], coords[:, 1]] = True
    return protect


# ---------------------- random offset background sampling ----------------- #


def compute_fallback_bg(img_f, exclude_mask, sigma=20.0):
    """Smooth fallback background, ignoring excluded regions."""
    if not exclude_mask.any():
        return ndimage.gaussian_filter(img_f, sigma=sigma)
    weight = (~exclude_mask).astype(np.float32)
    num = ndimage.gaussian_filter(img_f.astype(np.float32) * weight, sigma=sigma)
    den = ndimage.gaussian_filter(weight, sigma=sigma)
    den = np.maximum(den, 1e-6)
    return num / den


def random_offset_sampling(img_f, inpaint_mask, source_exclude_mask, *,
                           num_samples=16,
                           offset_min=10,
                           offset_max=60,
                           max_retries=10,
                           rng=None):
    """For each masked pixel, draw num_samples random source pixels at
    distances in [offset_min, offset_max], avoiding source_exclude_mask.

    Returns a dict with:
        samples:              (K, N) sampled values (with fallback for failures)
        ys, xs:               (N,)   coordinates of masked pixels
        success_count:        (N,)   number of K samples that found a real source
        first_source_y/x:     (N,)   source coords from k=0 (-1 if fallback)
        first_used_fallback:  (N,)   True if k=0 fell back for that pixel
    """
    if rng is None:
        rng = np.random.default_rng()
    H, W = img_f.shape
    ys, xs = np.where(inpaint_mask)
    N = ys.size

    if N == 0:
        return {
            "samples": np.zeros((num_samples, 0), dtype=np.float32),
            "ys": ys, "xs": xs,
            "success_count": np.zeros(0, dtype=np.int32),
            "first_source_y": np.zeros(0, dtype=np.int32),
            "first_source_x": np.zeros(0, dtype=np.int32),
            "first_used_fallback": np.zeros(0, dtype=bool),
        }

    fallback = compute_fallback_bg(img_f, source_exclude_mask, sigma=20.0)
    samples = np.empty((num_samples, N), dtype=np.float32)
    success_count = np.zeros(N, dtype=np.int32)
    first_source_y = np.full(N, -1, dtype=np.int32)
    first_source_x = np.full(N, -1, dtype=np.int32)
    first_used_fallback = np.zeros(N, dtype=bool)

    for k in range(num_samples):
        success_this_k = np.zeros(N, dtype=bool)
        remaining = np.arange(N)
        for _ in range(max_retries):
            if remaining.size == 0:
                break
            r = remaining
            angles = rng.uniform(0.0, 2.0 * np.pi, size=r.size)
            radii = rng.uniform(offset_min, offset_max, size=r.size)
            dy = np.round(radii * np.sin(angles)).astype(np.int32)
            dx = np.round(radii * np.cos(angles)).astype(np.int32)
            sy = ys[r] + dy
            sx = xs[r] + dx
            in_bounds = (sy >= 0) & (sy < H) & (sx >= 0) & (sx < W)
            sy_c = np.clip(sy, 0, H - 1)
            sx_c = np.clip(sx, 0, W - 1)
            not_excluded = ~source_exclude_mask[sy_c, sx_c]
            valid = in_bounds & not_excluded
            ok_idx = r[valid]
            samples[k, ok_idx] = img_f[sy[valid], sx[valid]]
            success_this_k[ok_idx] = True
            if k == 0:
                first_source_y[ok_idx] = sy[valid]
                first_source_x[ok_idx] = sx[valid]
            remaining = r[~valid]
        if remaining.size > 0:
            samples[k, remaining] = fallback[ys[remaining], xs[remaining]]
            if k == 0:
                first_used_fallback[remaining] = True
        success_count += success_this_k.astype(np.int32)

    return {
        "samples": samples,
        "ys": ys, "xs": xs,
        "success_count": success_count,
        "first_source_y": first_source_y,
        "first_source_x": first_source_x,
        "first_used_fallback": first_used_fallback,
    }


def soft_blend(img_f, fill_image, inpaint_mask, soft_sigma=1.5):
    soft = ndimage.gaussian_filter(inpaint_mask.astype(np.float32), sigma=soft_sigma)
    soft = np.clip(soft, 0.0, 1.0)
    return img_f * (1.0 - soft) + fill_image * soft


# --------------------------------- panels --------------------------------- #


def find_zoom_crop(mask, size=128):
    H, W = mask.shape
    size = min(size, H, W)
    if mask.any():
        half = max(size // 2, 1)
        density = ndimage.uniform_filter(mask.astype(np.float32), size=half)
        cy, cx = np.unravel_index(int(np.argmax(density)), density.shape)
    else:
        cy, cx = H // 2, W // 2
    y0 = int(max(0, min(H - size, cy - size // 2)))
    x0 = int(max(0, min(W - size, cx - size // 2)))
    return y0, x0, size


def make_summary_panel(out_path, img_f, mask, protect_mask, inpaint_mask,
                       delined, removed, uncertainty, title=""):
    img_prev = to_preview_uint8(img_f)
    delined_prev = to_preview_uint8(delined)
    mask_overlay = make_overlay(img_prev, mask, color=(255, 0, 0), alpha=0.5)
    protect_overlay = make_overlay(img_prev, protect_mask, color=(0, 255, 0), alpha=0.7)
    inpaint_overlay = make_overlay(img_prev, inpaint_mask, color=(255, 128, 0), alpha=0.5)
    removed_img = removed_to_uint8(removed)
    unc_img = uncertainty_to_uint8(uncertainty)

    y0, x0, sz = find_zoom_crop(inpaint_mask, size=min(192, *img_f.shape))
    zoom_in = img_prev[y0:y0 + sz, x0:x0 + sz]
    zoom_out = delined_prev[y0:y0 + sz, x0:x0 + sz]
    zoom = np.concatenate([zoom_in, zoom_out], axis=1)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    panels = [
        (img_prev, "input"),
        (mask_overlay, "mask overlay"),
        (protect_overlay, "protect overlay"),
        (inpaint_overlay, "inpaint mask"),
        (delined_prev, "delined preview"),
        (removed_img, "removed (img - delined)"),
        (unc_img, "uncertainty"),
        (zoom, "zoom: input | delined"),
    ]
    for ax, (im, name) in zip(axes.ravel(), panels):
        if im.ndim == 2:
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(im)
        ax.set_title(name, fontsize=10)
        ax.set_axis_off()
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_debug_sampling_panel(out_path, img_f, mask,
                              fill_raw, fill_median, fallback_mask_img,
                              delined, removed, uncertainty,
                              title=""):
    img_prev = to_preview_uint8(img_f)
    fill_raw_prev = to_preview_uint8(fill_raw)
    fill_med_prev = to_preview_uint8(fill_median)
    delined_prev = to_preview_uint8(delined)
    mask_overlay = make_overlay(img_prev, mask, color=(255, 0, 0), alpha=0.5)
    fb_overlay = make_overlay(img_prev, fallback_mask_img, color=(255, 255, 0), alpha=0.85)
    removed_img = removed_to_uint8(removed)
    unc_img = uncertainty_to_uint8(uncertainty)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    panels = [
        (img_prev, "input"),
        (mask_overlay, "mask (red)"),
        (fill_raw_prev, "sampled_fill_raw_01 (k=0)"),
        (fill_med_prev, "sampled_fill_median"),
        (fb_overlay, "fallback_mask (yellow)"),
        (delined_prev, "delined (soft-blend)"),
        (removed_img, "removed = img - delined"),
        (unc_img, "uncertainty (std over K)"),
    ]
    for ax, (im, name) in zip(axes.ravel(), panels):
        if im.ndim == 2:
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(im)
        ax.set_title(name, fontsize=10)
        ax.set_axis_off()
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_contact_sheet(out_path, variants, ncols=4):
    n = len(variants)
    if n == 0:
        return
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)
    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        ax.set_axis_off()
        if i >= n:
            continue
        v = variants[i]
        im = np.array(Image.open(v["delined_preview_path"]))
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        m = v["metrics"]
        title = (
            f"{v['name']}\n"
            f"mask={m['mask_coverage_percent']:.2f}%  "
            f"|removed|={m['mean_abs_removed']:.3f}\n"
            f"unc_mean={m['mean_uncertainty']:.3f}"
        )
        ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_compare_sheet(out_path, variants):
    n = len(variants)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = np.atleast_2d(axes)
    for i, v in enumerate(variants):
        mask_overlay = np.array(Image.open(v["mask_overlay_path"]))
        delined = np.array(Image.open(v["delined_preview_path"]))
        removed = np.array(Image.open(v["removed_path"]))
        for ax, im, name in zip(
            axes[i],
            [mask_overlay, delined, removed],
            ["mask_overlay", "delined", "removed"],
        ):
            if im.ndim == 2:
                ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(im)
            ax.set_axis_off()
            ax.set_title(f"{v['name']} | {name}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================ Stage 2 - v2 ================================ #
#
# v2 introduces three fill modes that share a single random-source draw:
#   1. median_raw      - pixel-wise median of K random source values (= v1)
#   2. single_texture  - bg_low[target] + texture[k=0 source]
#   3. bestk_texture   - among K sources, pick the one whose bg_low best
#                        matches target's bg_low, then bg_low[target] +
#                        texture[best_source]
#
# bg_low = gaussian-blurred image with mask region weighted-out
# texture = img - bg_low
# Modes 2/3 do NOT take median; they preserve a single texture patch and
# are intended to keep high-frequency continuity instead of smearing it.


def random_offset_sampling_v2(img_f, inpaint_mask, source_exclude_mask, *,
                              num_samples=8,
                              offset_min=10,
                              offset_max=80,
                              max_retries=10,
                              rng=None):
    """Like v1 sampling but tracks source coords for ALL K samples."""
    if rng is None:
        rng = np.random.default_rng()
    H, W = img_f.shape
    ys, xs = np.where(inpaint_mask)
    N = ys.size

    if N == 0:
        return {
            "samples": np.zeros((num_samples, 0), dtype=np.float32),
            "ys": ys, "xs": xs,
            "source_ys": np.zeros((num_samples, 0), dtype=np.int32),
            "source_xs": np.zeros((num_samples, 0), dtype=np.int32),
            "success_mask": np.zeros((num_samples, 0), dtype=bool),
            "success_count": np.zeros(0, dtype=np.int32),
        }

    fallback = compute_fallback_bg(img_f, source_exclude_mask, sigma=20.0)
    samples = np.empty((num_samples, N), dtype=np.float32)
    source_ys_all = np.full((num_samples, N), -1, dtype=np.int32)
    source_xs_all = np.full((num_samples, N), -1, dtype=np.int32)
    success_mask = np.zeros((num_samples, N), dtype=bool)

    for k in range(num_samples):
        remaining = np.arange(N)
        for _ in range(max_retries):
            if remaining.size == 0:
                break
            r = remaining
            angles = rng.uniform(0.0, 2.0 * np.pi, size=r.size)
            radii = rng.uniform(offset_min, offset_max, size=r.size)
            dy = np.round(radii * np.sin(angles)).astype(np.int32)
            dx = np.round(radii * np.cos(angles)).astype(np.int32)
            sy = ys[r] + dy
            sx = xs[r] + dx
            in_bounds = (sy >= 0) & (sy < H) & (sx >= 0) & (sx < W)
            sy_c = np.clip(sy, 0, H - 1)
            sx_c = np.clip(sx, 0, W - 1)
            not_excluded = ~source_exclude_mask[sy_c, sx_c]
            valid = in_bounds & not_excluded
            ok_idx = r[valid]
            samples[k, ok_idx] = img_f[sy[valid], sx[valid]]
            source_ys_all[k, ok_idx] = sy[valid]
            source_xs_all[k, ok_idx] = sx[valid]
            success_mask[k, ok_idx] = True
            remaining = r[~valid]
        if remaining.size > 0:
            samples[k, remaining] = fallback[ys[remaining], xs[remaining]]

    success_count = success_mask.sum(axis=0).astype(np.int32)
    return {
        "samples": samples,
        "ys": ys, "xs": xs,
        "source_ys": source_ys_all,
        "source_xs": source_xs_all,
        "success_mask": success_mask,
        "success_count": success_count,
    }


def fill_median_raw(img_f, ys, xs, samples):
    bg = img_f.copy()
    if samples.shape[1] > 0:
        bg[ys, xs] = np.median(samples, axis=0)
    return bg


def fill_single_texture(img_f, bg_low, texture, ys, xs,
                        source_ys, source_xs, success_mask):
    bg = img_f.copy()
    N = ys.size
    if N == 0:
        return bg
    sy0 = source_ys[0]
    sx0 = source_xs[0]
    succ0 = success_mask[0]
    target_bg = bg_low[ys, xs]
    fill_vals = target_bg.copy().astype(np.float32)
    if succ0.any():
        fill_vals[succ0] = target_bg[succ0] + texture[sy0[succ0], sx0[succ0]]
    bg[ys, xs] = fill_vals
    return bg


def fill_bestk_texture(img_f, bg_low, texture, ys, xs,
                       source_ys, source_xs, success_mask):
    bg = img_f.copy()
    K, N = source_ys.shape
    if N == 0:
        return bg
    target_bg = bg_low[ys, xs]
    sy_c = np.clip(source_ys, 0, bg_low.shape[0] - 1)
    sx_c = np.clip(source_xs, 0, bg_low.shape[1] - 1)
    src_bg = bg_low[sy_c, sx_c]                       # (K, N)
    score = np.abs(src_bg - target_bg[None, :])
    score = np.where(success_mask, score, np.inf)
    best_k = np.argmin(score, axis=0)                 # (N,)
    no_valid = ~success_mask.any(axis=0)
    best_sy = np.take_along_axis(source_ys, best_k[None, :], axis=0)[0]
    best_sx = np.take_along_axis(source_xs, best_k[None, :], axis=0)[0]
    fill_vals = target_bg.copy().astype(np.float32)
    valid = ~no_valid
    if valid.any():
        fill_vals[valid] = target_bg[valid] + texture[best_sy[valid], best_sx[valid]]
    bg[ys, xs] = fill_vals
    return bg


def make_v2_debug_panel(out_path, img_f, mask, fill_image, fallback_mask_img,
                        delined, removed, uncertainty, mode_name, title=""):
    img_prev = to_preview_uint8(img_f)
    fill_prev = to_preview_uint8(fill_image)
    delined_prev = to_preview_uint8(delined)
    mask_overlay = make_overlay(img_prev, mask, color=(255, 0, 0), alpha=0.5)
    fb_overlay = make_overlay(img_prev, fallback_mask_img, color=(255, 255, 0), alpha=0.85)
    removed_img = removed_to_uint8(removed)
    unc_img = uncertainty_to_uint8(uncertainty)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    panels = [
        (img_prev, "input"),
        (mask_overlay, "mask (red)"),
        (fill_prev, f"fill ({mode_name})"),
        (fb_overlay, "fallback_mask (yellow)"),
        (delined_prev, "delined (soft-blend)"),
        (removed_img, "removed = img - delined"),
        (unc_img, "uncertainty (std over K)"),
        (None, ""),
    ]
    for ax, (im, name) in zip(axes.ravel(), panels):
        if im is None:
            ax.set_axis_off()
            continue
        if im.ndim == 2:
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(im)
        ax.set_title(name, fontsize=10)
        ax.set_axis_off()
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_v2_compare_sheet(out_path, mask_records):
    """One row per mask. Cols: mask_overlay | median_raw | single_texture |
    bestk_texture | removed_bestk."""
    n = len(mask_records)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))
    if n == 1:
        axes = np.atleast_2d(axes)
    for i, rec in enumerate(mask_records):
        col_specs = [
            (rec["mask_overlay_path"], f"{rec['mask_name']} | mask"),
            (rec["fills"]["median_raw"]["delined_preview_path"], "median_raw"),
            (rec["fills"]["single_texture"]["delined_preview_path"], "single_texture"),
            (rec["fills"]["bestk_texture"]["delined_preview_path"], "bestk_texture"),
            (rec["fills"]["bestk_texture"]["removed_path"], "removed (bestk)"),
        ]
        for ax, (path, title) in zip(axes[i], col_specs):
            im = np.array(Image.open(path))
            if im.ndim == 2:
                ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(im)
            ax.set_axis_off()
            ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_v2_pipeline(img_f, masks, protect_mask, out_dir, *, rng,
                    offset_min=10, offset_max=80, num_samples=8,
                    source_exclude_dilate=2, soft_sigma=1.0):
    print()
    print(f"[stage2-v2] modes=[median_raw, single_texture, bestk_texture]  "
          f"offset=[{offset_min},{offset_max}]  K={num_samples}  "
          f"src_excl_dil={source_exclude_dilate}  soft_sigma={soft_sigma}")

    v2_root = out_dir / "v2"
    (v2_root / "variants").mkdir(parents=True, exist_ok=True)
    (v2_root / "reports").mkdir(parents=True, exist_ok=True)

    fill_modes = ["median_raw", "single_texture", "bestk_texture"]
    summary_rows = []
    mask_records = []

    for mask_name, m in masks.items():
        print(f"[v2 mask] {mask_name}")
        mask_dir = v2_root / "variants" / mask_name
        mask_dir.mkdir(parents=True, exist_ok=True)

        inpaint_mask = m & (~protect_mask)
        source_exclude = morphology.binary_dilation(
            m | protect_mask, disk(source_exclude_dilate)
        )

        sample_out = random_offset_sampling_v2(
            img_f, inpaint_mask, source_exclude,
            num_samples=num_samples,
            offset_min=offset_min,
            offset_max=offset_max,
            rng=rng,
        )
        samples = sample_out["samples"]
        ys = sample_out["ys"]
        xs = sample_out["xs"]
        source_ys = sample_out["source_ys"]
        source_xs = sample_out["source_xs"]
        success_mask = sample_out["success_mask"]
        success_count = sample_out["success_count"]
        N_pix = ys.size
        K = num_samples

        if N_pix > 0:
            total_uses = K * N_pix
            total_fb_uses = total_uses - int(success_count.sum())
            fallback_ratio = total_fb_uses / total_uses
            any_fallback_pixel_ratio = float(np.mean(success_count < K))
            all_fallback_pixel_ratio = float(np.mean(success_count == 0))
            mean_success_count = float(success_count.mean())
        else:
            fallback_ratio = 0.0
            any_fallback_pixel_ratio = 0.0
            all_fallback_pixel_ratio = 0.0
            mean_success_count = 0.0

        warn = "[WARN] " if fallback_ratio > 0.05 else ""
        print(f"  {warn}fallback_ratio={fallback_ratio*100:.2f}%  "
              f"any={any_fallback_pixel_ratio*100:.2f}%  "
              f"all={all_fallback_pixel_ratio*100:.2f}%  "
              f"mean_success={mean_success_count:.2f}/{K}")

        # uncertainty (std over K) - shared across modes
        background_std = np.zeros_like(img_f, dtype=np.float32)
        if N_pix > 0:
            background_std[ys, xs] = np.std(samples, axis=0)

        # bg_low / texture for texture-based modes
        bg_low = compute_fallback_bg(img_f, source_exclude, sigma=20.0)
        texture = (img_f - bg_low).astype(np.float32)

        # debug rasters at mask level
        fallback_mask_img = np.zeros(img_f.shape, dtype=bool)
        sample_count_map = np.zeros(img_f.shape, dtype=np.int32)
        if N_pix > 0:
            ever = success_count == 0
            if ever.any():
                fallback_mask_img[ys[ever], xs[ever]] = True
            sample_count_map[ys, xs] = success_count
        Image.fromarray((m * 255).astype(np.uint8)).save(mask_dir / "mask.png")
        Image.fromarray((inpaint_mask * 255).astype(np.uint8)).save(mask_dir / "inpaint_mask.png")
        Image.fromarray((fallback_mask_img * 255).astype(np.uint8)).save(mask_dir / "fallback_mask.png")
        sc_scale = 255.0 / max(K, 1)
        Image.fromarray(np.clip(sample_count_map * sc_scale, 0, 255).astype(np.uint8))\
            .save(mask_dir / "sample_count_map.png")
        tifffile.imwrite(str(mask_dir / "bg_low.tif"), bg_low.astype(np.float32))
        tifffile.imwrite(str(mask_dir / "texture.tif"), texture.astype(np.float32))
        (mask_dir / "fallback_ratio.txt").write_text(
            f"fallback_ratio: {fallback_ratio:.6f}\n"
            f"any_fallback_pixel_ratio: {any_fallback_pixel_ratio:.6f}\n"
            f"all_fallback_pixel_ratio: {all_fallback_pixel_ratio:.6f}\n"
            f"mean_success_count: {mean_success_count:.4f}  (out of K={K})\n"
            f"inpaint_pixel_count: {N_pix}\n"
            f"warn_threshold: 0.05\n"
        )

        fills = {}
        for mode in fill_modes:
            mode_dir = mask_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)

            if mode == "median_raw":
                fill_image = fill_median_raw(img_f, ys, xs, samples)
            elif mode == "single_texture":
                fill_image = fill_single_texture(
                    img_f, bg_low, texture, ys, xs,
                    source_ys, source_xs, success_mask,
                )
            else:  # bestk_texture
                fill_image = fill_bestk_texture(
                    img_f, bg_low, texture, ys, xs,
                    source_ys, source_xs, success_mask,
                )

            delined = soft_blend(img_f, fill_image, inpaint_mask, soft_sigma=soft_sigma)
            removed = img_f - delined

            mean_abs_removed = float(np.mean(np.abs(removed)))
            std_removed = float(np.std(removed))
            mean_unc = float(np.mean(background_std[inpaint_mask])) if inpaint_mask.any() else 0.0
            p95_unc = float(np.percentile(background_std[inpaint_mask], 95)) if inpaint_mask.any() else 0.0

            tifffile.imwrite(str(mode_dir / "delined.tif"), delined.astype(np.float32))
            tifffile.imwrite(str(mode_dir / "fill.tif"), fill_image.astype(np.float32))
            Image.fromarray(to_preview_uint8(delined)).save(mode_dir / "delined_preview.png")
            Image.fromarray(removed_to_uint8(removed)).save(mode_dir / "removed.png")
            Image.fromarray(uncertainty_to_uint8(background_std)).save(mode_dir / "uncertainty.png")

            metrics = {
                "variant_name": f"{mask_name}__{mode}",
                "mask_name": mask_name,
                "mode": mode,
                "mask_coverage_percent": float(m.mean()) * 100.0,
                "inpaint_coverage_percent": float(inpaint_mask.mean()) * 100.0,
                "mean_abs_removed": mean_abs_removed,
                "std_removed": std_removed,
                "mean_uncertainty": mean_unc,
                "p95_uncertainty": p95_unc,
                "fallback_ratio": fallback_ratio,
                "any_fallback_pixel_ratio": any_fallback_pixel_ratio,
                "all_fallback_pixel_ratio": all_fallback_pixel_ratio,
                "mean_success_count": mean_success_count,
                "offset_min": offset_min,
                "offset_max": offset_max,
                "num_samples": num_samples,
                "soft_sigma": soft_sigma,
                "source_exclude_dilate": source_exclude_dilate,
            }
            with open(mode_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            make_v2_debug_panel(
                mode_dir / "debug_panel.png",
                img_f, m, fill_image, fallback_mask_img, delined, removed, background_std,
                mode_name=mode,
                title=(f"{mask_name} | {mode}  "
                       f"fallback={fallback_ratio*100:.2f}%  "
                       f"|removed|={mean_abs_removed:.2f}  "
                       f"unc_mean={mean_unc:.2f}"),
            )

            summary_rows.append(metrics)
            fills[mode] = {
                "delined_preview_path": mode_dir / "delined_preview.png",
                "removed_path": mode_dir / "removed.png",
                "uncertainty_path": mode_dir / "uncertainty.png",
                "debug_panel_path": mode_dir / "debug_panel.png",
            }
            print(f"    {mode:16s}  |removed|={mean_abs_removed:7.3f}  "
                  f"std_removed={std_removed:7.2f}  unc_mean={mean_unc:6.2f}")

        mask_records.append({
            "mask_name": mask_name,
            "mask_overlay_path": out_dir / "overlays" / f"{mask_name}_overlay.png",
            "fills": fills,
        })

    # csv
    csv_path = v2_root / "reports" / "stage2_v2_summary.csv"
    fieldnames = [
        "variant_name", "mask_name", "mode",
        "mask_coverage_percent", "inpaint_coverage_percent",
        "mean_abs_removed", "std_removed",
        "mean_uncertainty", "p95_uncertainty",
        "fallback_ratio", "any_fallback_pixel_ratio",
        "all_fallback_pixel_ratio", "mean_success_count",
        "offset_min", "offset_max", "num_samples", "soft_sigma",
        "source_exclude_dilate",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    print("[v2 reports] compare sheet ...")
    make_v2_compare_sheet(v2_root / "reports" / "stage2_v2_compare_sheet.png", mask_records)

    print()
    print("Stage 2 v2 done.")
    print("Please check:")
    print(f"  {v2_root / 'reports' / 'stage2_v2_compare_sheet.png'}")
    print(f"  {v2_root / 'reports' / 'stage2_v2_summary.csv'}")
    print(f"  {v2_root / 'variants' / '<mask>' / '<mode>' / 'debug_panel.png'}")


# ---------------------------------- main ---------------------------------- #


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to data.tif")
    ap.add_argument("--dataset", default="line_stage1_dataset",
                    help="Path to line_stage1_dataset/ (label + sato live here)")
    ap.add_argument("--line-prob", default=None, help="Optional line_prob.tif from model")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--offset-min", type=int, default=8)
    ap.add_argument("--offset-max", type=int, default=80)
    ap.add_argument("--source-exclude-dilate", type=int, default=4)
    ap.add_argument("--soft-sigma", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", choices=("v1", "v2", "both"), default="v2",
                    help="v1: pixel-wise median over 4 sampling configs; "
                         "v2: 3 fill modes (median_raw / single_texture / bestk_texture) "
                         "with simplified params; both: run both")
    # v2 params (used when --stage in {v2, both}); the v1 sampling configs
    # are still hard-coded as before.
    ap.add_argument("--v2-offset-min", type=int, default=10)
    ap.add_argument("--v2-offset-max", type=int, default=80)
    ap.add_argument("--v2-num-samples", type=int, default=8)
    ap.add_argument("--v2-source-exclude-dilate", type=int, default=2)
    ap.add_argument("--v2-soft-sigma", type=float, default=1.0)
    return ap.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    (out_dir / "variants").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)

    ds = Path(args.dataset)
    cwd = Path(".")

    print(f"[load] input image: {args.input}")
    img = load_grayscale(args.input)
    img_f = img.astype(np.float32)
    H, W = img_f.shape
    print(f"  shape={img_f.shape}  dtype={img.dtype}  "
          f"min={float(img_f.min()):.4f}  max={float(img_f.max()):.4f}")

    label_path = find_file(
        ds / "pseudo_labels" / "train_label_selected.png",
        ds / "train_label_selected.png",
        cwd / "train_label_selected.png",
    )
    if label_path is None:
        raise FileNotFoundError("train_label_selected.png not found in dataset/cwd")
    print(f"[load] label: {label_path}")
    label = np.array(Image.open(str(label_path)))
    if label.ndim == 3:
        label = label[..., 0]
    if label.shape != img_f.shape:
        raise ValueError(f"label shape {label.shape} != image shape {img_f.shape}")

    sato_path = find_file(
        ds / "responses" / "response_sato.tif",
        ds / "response_sato.tif",
        cwd / "response_sato.tif",
    )
    if sato_path is None:
        raise FileNotFoundError("response_sato.tif not found in dataset/cwd")
    print(f"[load] sato:  {sato_path}")
    sato = load_grayscale(sato_path).astype(np.float32)
    if sato.shape != img_f.shape:
        raise ValueError(f"sato shape {sato.shape} != image shape {img_f.shape}")
    sato_norm = (sato - float(sato.min())) / max(float(sato.max() - sato.min()), 1e-12)

    line_prob = None
    if args.line_prob:
        lp_path = Path(args.line_prob)
        if lp_path.exists():
            print(f"[load] line_prob: {lp_path}")
            line_prob = load_grayscale(lp_path).astype(np.float32)
            if line_prob.max() > 1.5:
                line_prob = line_prob / float(line_prob.max())
            if line_prob.shape != img_f.shape:
                print(f"  [warn] line_prob shape {line_prob.shape} != image; ignoring")
                line_prob = None
        else:
            print(f"  [warn] --line-prob not found: {lp_path}")

    # 1. mask variants
    print("[mask] building variants ...")
    raw_masks = build_raw_masks(label, sato_norm, line_prob=line_prob)

    img_preview = to_preview_uint8(img_f)
    masks = {}
    for name, raw in raw_masks.items():
        m = postprocess_mask(raw, area_min=8, dilate_radius=1)
        cov = float(m.mean()) * 100.0
        warn = "  [WARN coverage > 20%]" if cov > 20.0 else ""
        print(f"  {name}: coverage={cov:.2f}%{warn}")
        masks[name] = m
        Image.fromarray((m * 255).astype(np.uint8)).save(out_dir / "masks" / f"{name}.png")
        ov = make_overlay(img_preview, m, color=(255, 0, 0), alpha=0.5)
        Image.fromarray(ov).save(out_dir / "overlays" / f"{name}_overlay.png")

    # 2. protect mask
    print("[protect] computing protect mask ...")
    protect_mask = compute_protect_mask(img_f)
    print(f"  protect coverage: {float(protect_mask.mean()) * 100:.3f}%")
    Image.fromarray((protect_mask * 255).astype(np.uint8)).save(out_dir / "protect_mask.png")
    Image.fromarray(make_overlay(img_preview, protect_mask, color=(0, 255, 0), alpha=0.7))\
        .save(out_dir / "protect_overlay.png")

    # 3. sampling configs
    sampling_cfgs = [
        ("S1", dict(offset_min=8,  offset_max=40,  num_samples=8,  soft_sigma=1.0,
                    source_exclude_dilate=args.source_exclude_dilate)),
        ("S2", dict(offset_min=10, offset_max=60,  num_samples=16, soft_sigma=1.5,
                    source_exclude_dilate=args.source_exclude_dilate)),
        ("S3", dict(offset_min=20, offset_max=80,  num_samples=16, soft_sigma=2.0,
                    source_exclude_dilate=args.source_exclude_dilate)),
        ("S4", dict(offset_min=30, offset_max=120, num_samples=24, soft_sigma=2.0,
                    source_exclude_dilate=args.source_exclude_dilate)),
    ]

    rng = np.random.default_rng(args.seed)
    summary_rows = []
    variant_records = []

    run_v1 = args.stage in ("v1", "both")
    run_v2 = args.stage in ("v2", "both")

    if not run_v1:
        # skip v1 loop entirely; keep sampling_cfgs unused noise out of report
        masks_for_v1 = {}
    else:
        masks_for_v1 = masks

    for mask_name, m in masks_for_v1.items():
        for s_name, cfg in sampling_cfgs:
            variant = f"{mask_name}__{s_name}"
            print(f"[variant] {variant}")
            v_dir = out_dir / "variants" / variant
            v_dir.mkdir(parents=True, exist_ok=True)

            inpaint_mask = m & (~protect_mask)
            source_exclude = morphology.binary_dilation(
                m | protect_mask, disk(cfg["source_exclude_dilate"])
            )

            sample_out = random_offset_sampling(
                img_f, inpaint_mask, source_exclude,
                num_samples=cfg["num_samples"],
                offset_min=cfg["offset_min"],
                offset_max=cfg["offset_max"],
                rng=rng,
            )
            samples = sample_out["samples"]
            ys = sample_out["ys"]
            xs = sample_out["xs"]
            success_count = sample_out["success_count"]
            first_source_y = sample_out["first_source_y"]
            first_source_x = sample_out["first_source_x"]
            N_pix = ys.size
            K = cfg["num_samples"]

            background_median = img_f.copy()
            background_std = np.zeros_like(img_f, dtype=np.float32)
            sampled_fill_raw_01 = img_f.copy()
            if N_pix > 0:
                background_median[ys, xs] = np.median(samples, axis=0)
                background_std[ys, xs] = np.std(samples, axis=0)
                sampled_fill_raw_01[ys, xs] = samples[0]

            delined = soft_blend(img_f, background_median, inpaint_mask,
                                 soft_sigma=cfg["soft_sigma"])
            removed = img_f - delined

            # debug maps over full image
            sample_count_map = np.zeros(img_f.shape, dtype=np.int32)
            fallback_mask_img = np.zeros(img_f.shape, dtype=bool)
            source_y_map = np.full(img_f.shape, -1, dtype=np.int32)
            source_x_map = np.full(img_f.shape, -1, dtype=np.int32)
            if N_pix > 0:
                sample_count_map[ys, xs] = success_count
                ever_fallback = success_count == 0
                if ever_fallback.any():
                    fallback_mask_img[ys[ever_fallback], xs[ever_fallback]] = True
                source_y_map[ys, xs] = first_source_y
                source_x_map[ys, xs] = first_source_x

            # fallback metrics
            if N_pix > 0:
                total_uses = K * N_pix
                total_fallback_uses = total_uses - int(success_count.sum())
                fallback_ratio = total_fallback_uses / total_uses
                any_fallback_pixel_ratio = float(np.mean(success_count < K))
                all_fallback_pixel_ratio = float(np.mean(success_count == 0))
                mean_success_count = float(success_count.mean())
            else:
                fallback_ratio = 0.0
                any_fallback_pixel_ratio = 0.0
                all_fallback_pixel_ratio = 0.0
                mean_success_count = 0.0

            if fallback_ratio > 0.05:
                print(f"  [WARN] fallback_ratio={fallback_ratio*100:.2f}% (>5%)  "
                      f"any={any_fallback_pixel_ratio*100:.2f}%  "
                      f"all={all_fallback_pixel_ratio*100:.2f}%  "
                      f"-> consider lowering source_exclude_dilate / smaller mask / wider offset")
            else:
                print(f"  fallback_ratio={fallback_ratio*100:.2f}%  "
                      f"any={any_fallback_pixel_ratio*100:.2f}%  "
                      f"all={all_fallback_pixel_ratio*100:.2f}%")

            # save outputs
            tifffile.imwrite(str(v_dir / "delined.tif"), delined.astype(np.float32))
            tifffile.imwrite(str(v_dir / "background_median.tif"),
                             background_median.astype(np.float32))
            tifffile.imwrite(str(v_dir / "sampled_fill_raw_01.tif"),
                             sampled_fill_raw_01.astype(np.float32))
            tifffile.imwrite(str(v_dir / "sampled_fill_median.tif"),
                             background_median.astype(np.float32))
            tifffile.imwrite(str(v_dir / "source_y_map.tif"), source_y_map)
            tifffile.imwrite(str(v_dir / "source_x_map.tif"), source_x_map)

            Image.fromarray((m * 255).astype(np.uint8)).save(v_dir / "mask.png")
            Image.fromarray((inpaint_mask * 255).astype(np.uint8)).save(v_dir / "inpaint_mask.png")
            Image.fromarray(to_preview_uint8(delined)).save(v_dir / "delined_preview.png")
            Image.fromarray(removed_to_uint8(removed)).save(v_dir / "removed.png")
            Image.fromarray(uncertainty_to_uint8(background_std)).save(v_dir / "uncertainty.png")
            Image.fromarray(uncertainty_to_uint8(background_std)).save(
                v_dir / "background_uncertainty.png"
            )

            # sample_count_map: scale 0..K -> 0..255 (white = all K succeeded)
            sc_scale = 255.0 / max(K, 1)
            sample_count_vis = np.clip(sample_count_map * sc_scale, 0, 255).astype(np.uint8)
            Image.fromarray(sample_count_vis).save(v_dir / "sample_count_map.png")
            Image.fromarray((fallback_mask_img * 255).astype(np.uint8)).save(
                v_dir / "fallback_mask.png"
            )

            (v_dir / "fallback_ratio.txt").write_text(
                f"fallback_ratio: {fallback_ratio:.6f}\n"
                f"any_fallback_pixel_ratio: {any_fallback_pixel_ratio:.6f}\n"
                f"all_fallback_pixel_ratio: {all_fallback_pixel_ratio:.6f}\n"
                f"mean_success_count: {mean_success_count:.4f}  (out of K={K})\n"
                f"inpaint_pixel_count: {N_pix}\n"
                f"warn_threshold: 0.05\n"
            )

            # metrics
            mask_cov = float(m.mean()) * 100.0
            inpaint_cov = float(inpaint_mask.mean()) * 100.0
            mean_abs_removed = float(np.mean(np.abs(removed)))
            std_removed = float(np.std(removed))
            if inpaint_mask.any():
                mean_unc = float(np.mean(background_std[inpaint_mask]))
                p95_unc = float(np.percentile(background_std[inpaint_mask], 95))
            else:
                mean_unc = 0.0
                p95_unc = 0.0

            metrics = {
                "variant_name": variant,
                "mask_name": mask_name,
                "sampling_name": s_name,
                "mask_coverage_percent": mask_cov,
                "inpaint_coverage_percent": inpaint_cov,
                "mean_abs_removed": mean_abs_removed,
                "std_removed": std_removed,
                "mean_uncertainty": mean_unc,
                "p95_uncertainty": p95_unc,
                "fallback_ratio": fallback_ratio,
                "any_fallback_pixel_ratio": any_fallback_pixel_ratio,
                "all_fallback_pixel_ratio": all_fallback_pixel_ratio,
                "mean_success_count": mean_success_count,
                "offset_min": cfg["offset_min"],
                "offset_max": cfg["offset_max"],
                "num_samples": cfg["num_samples"],
                "soft_sigma": cfg["soft_sigma"],
                "source_exclude_dilate": cfg["source_exclude_dilate"],
            }
            with open(v_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            make_summary_panel(
                v_dir / "summary_panel.png",
                img_f, m, protect_mask, inpaint_mask,
                delined, removed, background_std,
                title=variant,
            )
            make_debug_sampling_panel(
                v_dir / "debug_sampling_panel.png",
                img_f, m, sampled_fill_raw_01, background_median,
                fallback_mask_img, delined, removed, background_std,
                title=f"{variant} | fallback_ratio={fallback_ratio*100:.2f}%  "
                      f"mean_success={mean_success_count:.2f}/{K}",
            )

            summary_rows.append(metrics)
            variant_records.append({
                "name": variant,
                "mask_name": mask_name,
                "delined_preview_path": v_dir / "delined_preview.png",
                "mask_overlay_path": out_dir / "overlays" / f"{mask_name}_overlay.png",
                "removed_path": v_dir / "removed.png",
                "metrics": metrics,
            })

    # 4. v1 summary csv & sheets
    if run_v1:
        csv_path = out_dir / "reports" / "stage2_summary.csv"
        fieldnames = [
            "variant_name", "mask_name", "sampling_name",
            "mask_coverage_percent", "inpaint_coverage_percent",
            "mean_abs_removed", "std_removed",
            "mean_uncertainty", "p95_uncertainty",
            "fallback_ratio", "any_fallback_pixel_ratio",
            "all_fallback_pixel_ratio", "mean_success_count",
            "offset_min", "offset_max", "num_samples", "soft_sigma",
            "source_exclude_dilate",
        ]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in summary_rows:
                w.writerow({k: row.get(k, "") for k in fieldnames})

        print("[reports] contact sheet & compare sheet ...")
        make_contact_sheet(out_dir / "reports" / "stage2_contact_sheet.png", variant_records)
        make_compare_sheet(out_dir / "reports" / "stage2_compare_sheet.png", variant_records)

        print()
        print("Stage 2 v1 done.")
        print("Please check:")
        print(f"  {out_dir / 'reports' / 'stage2_contact_sheet.png'}")
        print(f"  {out_dir / 'reports' / 'stage2_compare_sheet.png'}")
        print(f"  {out_dir / 'variants' / '<best_variant>' / 'summary_panel.png'}")

    # 5. v2 pipeline (3 fill modes per mask)
    if run_v2:
        run_v2_pipeline(
            img_f, masks, protect_mask, out_dir, rng=rng,
            offset_min=args.v2_offset_min,
            offset_max=args.v2_offset_max,
            num_samples=args.v2_num_samples,
            source_exclude_dilate=args.v2_source_exclude_dilate,
            soft_sigma=args.v2_soft_sigma,
        )


if __name__ == "__main__":
    main()
