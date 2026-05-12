"""Stage 2 - Line-band contrast suppression.

Reads labelled Kossel-line masks and rewrites each line band so the line is
pulled visually back into its local background. For every variant we run

    bg          = local background estimate
    residual    = img - bg
    out_core    = bg + keep_ratio * residual

then composite with a hard core + feathered boundary back into the raw image.

Sweeps mask (B / C / D), band width, background method (masked_gaussian /
side_band), sigma, and keep_ratio, then writes per-variant artifacts plus a
contact_sheet and a compare_sheet for hand picking.

Usage:
    python scripts/line_band_contrast_suppress.py \
        --input data.tif \
        --prepared outputs/prepare \
        --out outputs/line_band_suppress_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import time
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
# I/O helpers
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
# Prepared inputs (Kossel masks + protect mask)
# --------------------------------------------------------------------------- #

def find_response(prepared_dir: Path, fallback: Path | None) -> Path:
    candidates = [
        prepared_dir / "responses" / "response_sato.tif",
        prepared_dir / "response_sato.tif",
        fallback,
    ]
    for c in candidates:
        if c is not None and c.exists():
            return c
    raise FileNotFoundError(
        f"Cannot find response_sato.tif. Tried: {[str(c) for c in candidates]}"
    )


KOSSEL_MASK_SPEC = {
    "A_sato98": ("mask_A_clear_sato98.png", 98),
    "B_sato95": ("mask_B_sato95.png", 95),
    "C_sato90": ("mask_C_sato90.png", 90),
    "D_sato85": ("mask_D_sato85.png", 85),
}


def ensure_kossel_masks(prepared_dir: Path, response_path: Path,
                        wanted: list[str]) -> dict[str, np.ndarray]:
    masks_dir = prepared_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    response = None
    masks: dict[str, np.ndarray] = {}
    for name in wanted:
        if name not in KOSSEL_MASK_SPEC:
            raise ValueError(f"unknown mask name {name}")
        filename, pct = KOSSEL_MASK_SPEC[name]
        path = masks_dir / filename
        if path.exists():
            masks[name] = load_mask(path)
            print(f"  loaded {filename} (coverage={masks[name].mean()*100:.3f}%)")
            continue
        if response is None:
            print(f"  loading response_sato from {response_path}")
            response = load_tif(response_path).astype(np.float32)
        thr = float(np.percentile(response, pct))
        mask = response > thr
        save_mask(path, mask)
        masks[name] = mask
        print(
            f"  generated {filename} "
            f"(p{pct}={thr:.4f}, coverage={mask.mean()*100:.3f}%)"
        )
    return masks


def ensure_protect_mask(prepared_dir: Path, img_f: np.ndarray) -> np.ndarray:
    path = prepared_dir / "masks" / "protect_mask.png"
    if path.exists():
        m = load_mask(path)
        print(f"  loaded protect_mask (coverage={m.mean()*100:.4f}%)")
        return m
    print("  computing protect_mask from img - gaussian(img, 15) residual")
    local_bg = ndimage.gaussian_filter(img_f, sigma=15)
    abs_res = np.abs(img_f - local_bg)
    thr = float(np.percentile(abs_res, 99.3))
    candidate = abs_res > thr

    lbl = measure.label(candidate, connectivity=2)
    protect = np.zeros_like(candidate, dtype=bool)
    keep_labels: list[int] = []
    for region in measure.regionprops(lbl):
        a = region.area
        if a < 4 or a > 500:
            continue
        if region.eccentricity > 0.9:
            continue
        h = region.bbox[2] - region.bbox[0]
        w = region.bbox[3] - region.bbox[1]
        ar = max(h, w) / max(1, min(h, w))
        if ar >= 4:
            continue
        keep_labels.append(region.label)
    if keep_labels:
        protect = np.isin(lbl, keep_labels)
    save_mask(path, protect)
    print(f"  protect_mask coverage={protect.mean()*100:.4f}% "
          f"({len(keep_labels)} blobs)")
    return protect


# --------------------------------------------------------------------------- #
# Line band + background estimation
# --------------------------------------------------------------------------- #

def build_line_band(line_core: np.ndarray, width: int) -> np.ndarray:
    band = binary_dilation(line_core, disk(width))
    closing_radius = 2 if width >= 4 else 1
    band = binary_closing(band, disk(closing_radius))
    return band


def masked_gaussian_bg(img_f: np.ndarray, valid: np.ndarray, sigma: float,
                       fallback_sigma: float | None = None) -> np.ndarray:
    weight = valid.astype(np.float32)
    num = ndimage.gaussian_filter(img_f * weight, sigma=sigma)
    den = ndimage.gaussian_filter(weight, sigma=sigma)
    bg = num / np.maximum(den, 1e-6)
    if fallback_sigma is not None:
        fb_num = ndimage.gaussian_filter(img_f * weight, sigma=fallback_sigma)
        fb_den = ndimage.gaussian_filter(weight, sigma=fallback_sigma)
        fb = fb_num / np.maximum(fb_den, 1e-6)
        no_support = den < 1e-3
        if no_support.any():
            bg = np.where(no_support, fb, bg)
    return bg.astype(np.float32)


def side_band_bg(img_f: np.ndarray, edit_mask: np.ndarray,
                 protect_mask: np.ndarray, width: int,
                 side_radius: int, sigma: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    inner = binary_dilation(edit_mask, disk(width + 2))
    outer = binary_dilation(edit_mask, disk(width + side_radius))
    side_band = outer & ~inner & ~edit_mask & ~protect_mask
    bg = masked_gaussian_bg(
        img_f, side_band, sigma=sigma,
        fallback_sigma=max(sigma * 2.0, 25.0),
    )
    return bg, side_band


# --------------------------------------------------------------------------- #
# Suppression (hard core + feather boundary)
# --------------------------------------------------------------------------- #

def feather_alpha(edit_mask: np.ndarray, feather_radius: float) -> np.ndarray:
    if feather_radius <= 0:
        return edit_mask.astype(np.float32)
    dist_outside = ndimage.distance_transform_edt(~edit_mask)
    alpha = np.clip(1.0 - dist_outside / float(feather_radius), 0.0, 1.0)
    alpha[edit_mask] = 1.0
    return alpha.astype(np.float32)


def suppress(img_f: np.ndarray, bg: np.ndarray, edit_mask: np.ndarray,
             keep_ratio: float, alpha: np.ndarray) -> np.ndarray:
    residual = img_f - bg
    out_core = bg + keep_ratio * residual
    out = alpha * out_core + (1.0 - alpha) * img_f
    out[edit_mask] = out_core[edit_mask]
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Preview helpers
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


def overlay_red(gray8: np.ndarray, mask: np.ndarray,
                alpha: float = 0.55) -> np.ndarray:
    rgb = np.stack([gray8, gray8, gray8], axis=-1).astype(np.float32)
    if mask.any():
        rgb[mask, 0] = alpha * 255.0 + (1 - alpha) * rgb[mask, 0]
        rgb[mask, 1] *= (1 - alpha)
        rgb[mask, 2] *= (1 - alpha)
    return rgb.astype(np.uint8)


def diff_heatmap(diff: np.ndarray, abs_scale: float) -> np.ndarray:
    """Render signed diff as a divergent heatmap on a gray base.

    Negative pixels (suppressed > raw, i.e. line core brightened up) become red,
    positive pixels become blue. abs_scale is the magnitude that saturates."""
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


def downscale_uint8(arr: np.ndarray, max_side: int = 500) -> np.ndarray:
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
# Per-variant runner
# --------------------------------------------------------------------------- #

VARIANT_FIELDS = [
    "variant_name", "mask_name", "width", "method", "sigma", "keep_ratio",
    "feather_radius", "edit_coverage_percent",
    "mean_abs_removed_inside", "mean_abs_removed_outside",
    "removed_inside_outside_ratio",
]


def save_summary_panel(out_path: Path, raw8: np.ndarray, sup8: np.ndarray,
                       mask: np.ndarray, removed: np.ndarray,
                       abs_scale: float, title: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    axes[0].imshow(raw8, cmap="gray"); axes[0].set_title("raw")
    axes[1].imshow(overlay_red(raw8, mask)); axes[1].set_title("edit_mask")
    axes[2].imshow(sup8, cmap="gray"); axes[2].set_title("suppressed")
    axes[3].imshow(removed, cmap="seismic", vmin=-abs_scale, vmax=abs_scale)
    axes[3].set_title("raw - suppressed")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def run_variant(
    *,
    img_f: np.ndarray, img_dtype: np.dtype,
    edit_mask: np.ndarray, line_band: np.ndarray,
    bg: np.ndarray, alpha: np.ndarray,
    keep_ratio: float, feather_radius: float,
    variant_dir: Path, variant_name: str,
    mask_name: str, width: int, method: str, sigma: float,
    preview_lo: float, preview_hi: float,
    img_clip: tuple[float, float],
    raw8_full: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    suppressed = suppress(img_f, bg, edit_mask, keep_ratio, alpha)
    removed = img_f - suppressed

    variant_dir.mkdir(parents=True, exist_ok=True)

    save_mask(variant_dir / "line_band.png", line_band)
    save_mask(variant_dir / "edit_mask.png", edit_mask)

    overlay_img = overlay_red(raw8_full, edit_mask)
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
        raw8_full, sup8, edit_mask, removed, abs_scale, variant_name,
    )

    inside = edit_mask
    outside = ~edit_mask
    mean_in = float(np.abs(removed[inside]).mean()) if inside.any() else 0.0
    mean_out = float(np.abs(removed[outside]).mean()) if outside.any() else 0.0
    metrics = {
        "variant_name": variant_name, "mask_name": mask_name, "width": width,
        "method": method, "sigma": sigma, "keep_ratio": keep_ratio,
        "feather_radius": feather_radius,
        "edit_coverage_percent": float(edit_mask.mean() * 100.0),
        "mean_abs_removed_inside": mean_in,
        "mean_abs_removed_outside": mean_out,
        "removed_inside_outside_ratio": mean_in / max(mean_out, 1e-6),
    }
    with open(variant_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    thumbs = {
        "overlay": downscale_uint8(overlay_img, max_side=520),
        "suppressed": downscale_uint8(sup8, max_side=520),
        "removed": downscale_uint8(removed_rgb, max_side=520),
    }
    return metrics, thumbs


# --------------------------------------------------------------------------- #
# Sheets
# --------------------------------------------------------------------------- #

def save_contact_sheet(records: list[dict], sheet_path: Path,
                       cols: int = 9) -> None:
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
        thumb = rec["thumbs"]["suppressed"]
        ax.imshow(thumb, cmap="gray")
        ax.set_title(rec["name"], fontsize=5)
        ax.set_axis_off()
    fig.suptitle(f"contact_sheet — suppressed previews ({n} variants)")
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
        axes[r, 0].set_title(f"{rec['name']} | overlay", fontsize=7)
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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

PRIORITY_NAMES = [
    "C_sato90_width4_side_sigma15_keep0.1",
    "C_sato90_width6_side_sigma15_keep0.1",
    "D_sato85_width4_side_sigma15_keep0.1",
    "D_sato85_width6_side_sigma15_keep0.1",
    "B_sato95_width4_side_sigma15_keep0.1",
]


def method_tag(method: str) -> str:
    return {"masked_gaussian": "mg", "side_band": "side"}[method]


def variant_name_of(mask_name: str, width: int, method: str,
                    sigma: float, keep_ratio: float) -> str:
    return (
        f"{mask_name}_width{width}_{method_tag(method)}_"
        f"sigma{int(round(sigma))}_keep{keep_ratio:.1f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data.tif")
    ap.add_argument("--prepared", default="outputs/prepare")
    ap.add_argument("--response", default=None,
                    help="Path to response_sato.tif (fallback if --prepared "
                         "does not already contain it)")
    ap.add_argument("--out", default="outputs/line_band_suppress_v1")
    ap.add_argument("--masks", nargs="+",
                    default=["B_sato95", "C_sato90", "D_sato85"])
    ap.add_argument("--widths", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--methods", nargs="+",
                    default=["masked_gaussian", "side_band"])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[15.0, 25.0])
    ap.add_argument("--keep-ratios", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2])
    ap.add_argument("--feather-radius", type=float, default=2.0)
    ap.add_argument("--side-radius", type=int, default=40)
    ap.add_argument("--texture-scale", type=float, default=0.0,
                    help="Reserved for future use; 0 disables texture injection")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap on number of variants to actually run")
    ap.add_argument("--priority-only", action="store_true",
                    help="Only run the priority variant list")
    args = ap.parse_args()

    in_path = Path(args.input)
    prepared = Path(args.prepared)
    out_root = Path(args.out)

    fallback_response = Path(args.response) if args.response else Path("response_sato.tif")
    response_path = find_response(prepared, fallback_response)

    print(f"input         = {in_path}")
    print(f"prepared      = {prepared}")
    print(f"out           = {out_root}")
    print(f"response_sato = {response_path}")

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

    print("[prepare]")
    kossel_masks = ensure_kossel_masks(prepared, response_path, args.masks)
    protect_mask = ensure_protect_mask(prepared, img_f)

    out_root.mkdir(parents=True, exist_ok=True)
    variants_dir = out_root / "variants"
    reports_dir = out_root / "reports"
    variants_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    line_cache: dict[tuple[str, int], dict] = {}
    for mask_name in args.masks:
        line_core = kossel_masks[mask_name]
        for width in args.widths:
            line_band = build_line_band(line_core, width)
            edit_mask = line_band & ~protect_mask
            alpha = feather_alpha(edit_mask, args.feather_radius)
            line_cache[(mask_name, width)] = {
                "line_band": line_band,
                "edit_mask": edit_mask,
                "alpha": alpha,
            }

    priority_set = set(PRIORITY_NAMES) if args.priority_only else None

    combos = list(product(
        args.masks, args.widths, args.methods, args.sigmas, args.keep_ratios,
    ))
    print(f"variants planned: {len(combos)}"
          + (f" (priority-only ⇒ {len(priority_set)})" if priority_set else ""))

    csv_rows: list[dict] = []
    records: list[dict] = []
    t0 = time.time()
    done = 0

    grouped: dict[tuple[str, int, str, float], list[float]] = {}
    for mask_name, width, method, sigma, keep_ratio in combos:
        key = (mask_name, width, method, sigma)
        grouped.setdefault(key, []).append(keep_ratio)

    for (mask_name, width, method, sigma), keep_ratios in grouped.items():
        cache = line_cache[(mask_name, width)]
        edit_mask = cache["edit_mask"]
        line_band = cache["line_band"]
        alpha = cache["alpha"]

        if method == "masked_gaussian":
            bg = masked_gaussian_bg(
                img_f, ~edit_mask, sigma=sigma,
                fallback_sigma=max(sigma * 2.0, 25.0),
            )
        elif method == "side_band":
            bg, _ = side_band_bg(
                img_f, edit_mask, protect_mask,
                width=width, side_radius=args.side_radius, sigma=sigma,
            )
        else:
            raise ValueError(f"unknown method {method}")

        for keep_ratio in keep_ratios:
            name = variant_name_of(mask_name, width, method, sigma, keep_ratio)
            if priority_set is not None and name not in priority_set:
                continue
            if args.limit is not None and done >= args.limit:
                break

            variant_dir = variants_dir / name
            metrics, thumbs = run_variant(
                img_f=img_f, img_dtype=img_dtype,
                edit_mask=edit_mask, line_band=line_band,
                bg=bg, alpha=alpha,
                keep_ratio=keep_ratio,
                feather_radius=args.feather_radius,
                variant_dir=variant_dir, variant_name=name,
                mask_name=mask_name, width=width, method=method, sigma=sigma,
                preview_lo=preview_lo, preview_hi=preview_hi,
                img_clip=img_clip,
                raw8_full=raw8_full,
            )
            csv_rows.append(metrics)
            records.append({"name": name, "metrics": metrics, "thumbs": thumbs})

            done += 1
            if done % 6 == 0 or done == 1:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                print(f"  [{done}/{len(combos)}] {name}  "
                      f"({elapsed:.1f}s, {rate:.2f}/s)")
        if args.limit is not None and done >= args.limit:
            break

    print(f"finished {done} variants in {time.time() - t0:.1f}s")

    csv_path = reports_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VARIANT_FIELDS)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"wrote {csv_path}")

    compare_subset = [r for r in records
                      if "_side_sigma15_" in r["name"]]
    if compare_subset:
        save_compare_sheet(
            compare_subset,
            reports_dir / "compare_sheet.png",
            title="compare_sheet — side_band, sigma=15 "
                  f"(rows={len(compare_subset)})",
        )
        print(f"wrote {reports_dir / 'compare_sheet.png'} "
              f"({len(compare_subset)} rows)")
    if len(records) != len(compare_subset):
        save_compare_sheet(
            records,
            reports_dir / "compare_sheet_all.png",
            title=f"compare_sheet_all (rows={len(records)})",
        )
        print(f"wrote {reports_dir / 'compare_sheet_all.png'} "
              f"({len(records)} rows)")

    save_contact_sheet(records, reports_dir / "contact_sheet.png")
    print(f"wrote {reports_dir / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
