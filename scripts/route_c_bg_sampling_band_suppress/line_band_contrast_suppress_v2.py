"""Stage 2c - Visual Line Hide v2.

Improvements over `line_band_contrast_suppress.py` (v1):

* **Black-line score** instead of pure sato ridge: estimate a low-frequency
  background `bg_low` and form `dark_residual = bg_low - raw`. The score is
  the rectified residual divided by local noise (Gaussian std with a global
  MAD floor). Used to filter the sato mask down to *dark* ridges only — v1
  treated bright ridges the same as dark and that diluted the suppression.

* **Core + halo dual mask** with separate `keep` ratios:
  - `core_mask`: high-confidence dark ridge, narrow.
    Aggressive replacement (`keep_core` ∈ {0.0, 0.05, 0.1}). Core pixels are
    hard-overwritten by `bg + keep_core * (raw - bg)`; raw is NOT preserved
    here, which was the v1 failure mode.
  - `halo_mask`: ring around the core, wider.
    Gentle compression (`keep_halo` ∈ {0.2, 0.3}). Soft transition keeps the
    join with surrounding texture believable.

* **Side-band background only**: estimate `bg` from a ring outside the
  edit zone (`core | halo`), excluding the protect mask. We do not fall back
  to global Gaussian over the whole image.

Outputs:
  outputs/visual_line_hide_v2/black_line_score_preview.png
  outputs/visual_line_hide_v2/reports/{contact_sheet,compare_sheet,summary.csv}
  outputs/visual_line_hide_v2/variants/<name>/{summary_panel.png,
      suppressed_preview.png, removed.png, overlay.png, core_mask.png,
      halo_mask.png, suppressed.tif, background_estimate.tif, metrics.json}

Typical invocation:

    python scripts/route_c_bg_sampling_band_suppress/line_band_contrast_suppress_v2.py \\
        --input data/data.tif \\
        --prepared outputs/prepare \\
        --out outputs/visual_line_hide_v2
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
# IO helpers (kept compatible with v1 layout)
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
# Mask sources from prepare/
# --------------------------------------------------------------------------- #

SATO_MASK_FILES = {
    "C_sato90": "mask_C_sato90.png",
    "D_sato85": "mask_D_sato85.png",
}


def load_sato_mask(prepared: Path, name: str) -> np.ndarray:
    fname = SATO_MASK_FILES[name]
    return load_mask(prepared / "masks" / fname)


def load_protect_mask(prepared: Path) -> np.ndarray | None:
    p = prepared / "masks" / "protect_mask.png"
    if not p.exists():
        return None
    return load_mask(p)


# --------------------------------------------------------------------------- #
# Black-line score
# --------------------------------------------------------------------------- #

def compute_black_line_score(
    img_f: np.ndarray,
    sigma_bg: float = 20.0,
    sigma_noise: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (black_line_score, bg_low, global_mad).

    Dark Kossel lines sit below a slowly varying background. Subtracting a
    large-sigma Gaussian gives a signed residual; positive values mean
    "darker than local background". We normalize that by a local noise
    estimate (Gaussian-weighted std with a global-MAD floor) so the score is
    in units of "standard deviations above noise".
    """
    bg_low = ndimage.gaussian_filter(img_f, sigma=sigma_bg).astype(np.float32)
    dark_residual = (bg_low - img_f).astype(np.float32)

    # global MAD of the signed residual as a robust noise floor
    res = (img_f - bg_low).astype(np.float32)
    med = float(np.median(res))
    global_mad = float(np.median(np.abs(res - med)) * 1.4826) + 1e-6

    # local std (Gaussian-weighted)
    res_local_mean = ndimage.gaussian_filter(res, sigma=sigma_noise)
    centered = res - res_local_mean
    local_var = ndimage.gaussian_filter(centered * centered, sigma=sigma_noise)
    local_noise = np.sqrt(np.maximum(local_var, 1e-12)).astype(np.float32)
    noise = np.maximum(local_noise, global_mad)

    score = np.clip(dark_residual, 0.0, None) / np.maximum(noise, 1e-6)
    return score.astype(np.float32), bg_low, global_mad


# --------------------------------------------------------------------------- #
# Core + halo masks
# --------------------------------------------------------------------------- #

def build_core_halo(
    sato_mask: np.ndarray,
    black_score: np.ndarray,
    protect_mask: np.ndarray,
    width_core: int,
    halo_width: int,
    black_threshold: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (core_mask, halo_mask, line_core_thin)."""
    line_core_thin = sato_mask & (black_score > black_threshold)
    core = binary_dilation(line_core_thin, disk(width_core))
    core = binary_closing(core, disk(2 if width_core >= 5 else 1))
    halo_outer = binary_dilation(core, disk(halo_width))
    halo = halo_outer & ~core
    core = core & ~protect_mask
    halo = halo & ~protect_mask
    return core, halo, line_core_thin


# --------------------------------------------------------------------------- #
# Side-band background
# --------------------------------------------------------------------------- #

def side_band_bg(
    img_f: np.ndarray,
    edit_mask: np.ndarray,
    protect_mask: np.ndarray,
    side_radius: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate background from a ring outside the edit zone."""
    outer = binary_dilation(edit_mask, disk(side_radius))
    side_band = outer & ~edit_mask & ~protect_mask

    weight = side_band.astype(np.float32)
    num = ndimage.gaussian_filter(img_f * weight, sigma=sigma)
    den = ndimage.gaussian_filter(weight, sigma=sigma)
    bg = num / np.maximum(den, 1e-6)

    fallback_sigma = max(sigma * 2.0, 30.0)
    fb_num = ndimage.gaussian_filter(img_f * weight, sigma=fallback_sigma)
    fb_den = ndimage.gaussian_filter(weight, sigma=fallback_sigma)
    fb = fb_num / np.maximum(fb_den, 1e-6)
    no_support = den < 1e-3
    if no_support.any():
        bg = np.where(no_support, fb, bg)
    return bg.astype(np.float32), side_band


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


def suppress_v2(
    img_f: np.ndarray, bg: np.ndarray,
    core_mask: np.ndarray, halo_mask: np.ndarray,
    keep_core: float, keep_halo: float,
    feather_radius: float,
) -> np.ndarray:
    residual = img_f - bg
    out = img_f.astype(np.float32).copy()

    # halo first (so core overrides if they overlap; they should not)
    if halo_mask.any():
        out[halo_mask] = (bg[halo_mask] + keep_halo * residual[halo_mask]).astype(np.float32)
    if core_mask.any():
        out[core_mask] = (bg[core_mask] + keep_core * residual[core_mask]).astype(np.float32)

    edit_mask = core_mask | halo_mask
    if feather_radius > 0 and edit_mask.any():
        alpha = feather_outer_alpha(edit_mask, feather_radius)
        out = alpha * out + (1.0 - alpha) * img_f
        # re-enforce strict edits inside the masks
        if halo_mask.any():
            out[halo_mask] = (bg[halo_mask] + keep_halo * residual[halo_mask]).astype(np.float32)
        if core_mask.any():
            out[core_mask] = (bg[core_mask] + keep_core * residual[core_mask]).astype(np.float32)

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


def overlay_core_halo(
    gray8: np.ndarray,
    core: np.ndarray, halo: np.ndarray,
    core_color=(220, 30, 30), halo_color=(240, 170, 30),
    alpha: float = 0.55,
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


def save_summary_panel(
    out_path: Path, raw8: np.ndarray, sup8: np.ndarray,
    core_mask: np.ndarray, halo_mask: np.ndarray,
    removed: np.ndarray, abs_scale: float, title: str,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    axes[0].imshow(raw8, cmap="gray"); axes[0].set_title("raw")
    axes[1].imshow(overlay_core_halo(raw8, core_mask, halo_mask))
    axes[1].set_title("core (red) + halo (orange)")
    axes[2].imshow(sup8, cmap="gray"); axes[2].set_title("suppressed")
    axes[3].imshow(removed, cmap="seismic", vmin=-abs_scale, vmax=abs_scale)
    axes[3].set_title("raw - suppressed")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Variant runner
# --------------------------------------------------------------------------- #

VARIANT_FIELDS = [
    "variant_name", "mask_name", "width_core", "halo_width",
    "keep_core", "keep_halo", "side_sigma", "feather_radius",
    "core_coverage_percent", "halo_coverage_percent",
    "edit_coverage_percent",
    "mean_abs_removed_core", "mean_abs_removed_halo",
    "mean_abs_removed_outside", "core_outside_ratio",
]


def variant_name_of(mask_tag: str, width_core: int, halo_width: int,
                    keep_core: float, keep_halo: float, side_sigma: float) -> str:
    return (
        f"{mask_tag}_wc{width_core}_h{halo_width}_"
        f"kc{keep_core:.2f}_kh{keep_halo:.1f}_s{int(round(side_sigma))}"
    )


def run_variant(
    *,
    img_f: np.ndarray, img_dtype: np.dtype,
    core_mask: np.ndarray, halo_mask: np.ndarray,
    bg: np.ndarray,
    keep_core: float, keep_halo: float, feather_radius: float,
    variant_dir: Path, variant_name: str, mask_tag: str,
    width_core: int, halo_width: int, side_sigma: float,
    preview_lo: float, preview_hi: float,
    img_clip: tuple[float, float],
    raw8_full: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    suppressed = suppress_v2(
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
    )

    edit_mask = core_mask | halo_mask
    outside = ~edit_mask
    mean_core = float(np.abs(removed[core_mask]).mean()) if core_mask.any() else 0.0
    mean_halo = float(np.abs(removed[halo_mask]).mean()) if halo_mask.any() else 0.0
    mean_out = float(np.abs(removed[outside]).mean()) if outside.any() else 0.0
    metrics = {
        "variant_name": variant_name, "mask_name": mask_tag,
        "width_core": width_core, "halo_width": halo_width,
        "keep_core": keep_core, "keep_halo": keep_halo,
        "side_sigma": side_sigma, "feather_radius": feather_radius,
        "core_coverage_percent": float(core_mask.mean() * 100.0),
        "halo_coverage_percent": float(halo_mask.mean() * 100.0),
        "edit_coverage_percent": float(edit_mask.mean() * 100.0),
        "mean_abs_removed_core": mean_core,
        "mean_abs_removed_halo": mean_halo,
        "mean_abs_removed_outside": mean_out,
        "core_outside_ratio": mean_core / max(mean_out, 1e-6),
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

def save_contact_sheet(records: list[dict], sheet_path: Path, cols: int = 9) -> None:
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
    fig.suptitle(f"contact_sheet — v2 suppressed previews ({n} variants)")
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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

MASK_TAGS = {
    "C_sato90": "C90b",
    "D_sato85": "D85b",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/data.tif")
    ap.add_argument("--prepared", default="outputs/prepare")
    ap.add_argument("--out", default="outputs/visual_line_hide_v2")
    ap.add_argument("--masks", nargs="+",
                    default=["C_sato90", "D_sato85"],
                    choices=list(SATO_MASK_FILES.keys()))
    ap.add_argument("--width-cores", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--halo-widths", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--keep-cores", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1])
    ap.add_argument("--keep-halos", type=float, nargs="+",
                    default=[0.2, 0.3])
    ap.add_argument("--side-sigmas", type=float, nargs="+",
                    default=[10.0, 15.0])
    ap.add_argument("--feather-radius", type=float, default=2.0)
    ap.add_argument("--side-radius", type=int, default=40)
    ap.add_argument("--bg-sigma", type=float, default=20.0,
                    help="Gaussian sigma for bg_low used by black-line score")
    ap.add_argument("--noise-sigma", type=float, default=8.0,
                    help="Gaussian sigma for local-noise estimation")
    ap.add_argument("--black-threshold", type=float, default=1.0,
                    help="black_line_score threshold to filter sato mask")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    in_path = Path(args.input)
    prepared = Path(args.prepared)
    out_root = Path(args.out)

    print(f"input        = {in_path}")
    print(f"prepared     = {prepared}")
    print(f"out          = {out_root}")

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

    print("[1/3] computing black-line score")
    score, bg_low, mad = compute_black_line_score(
        img_f, sigma_bg=args.bg_sigma, sigma_noise=args.noise_sigma,
    )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "reports").mkdir(parents=True, exist_ok=True)
    (out_root / "variants").mkdir(parents=True, exist_ok=True)

    sc_clip = float(np.percentile(score, 99.5))
    save_uint8(
        out_root / "black_line_score_preview.png",
        to_uint8(score, 0.0, max(sc_clip, 1e-6)),
    )
    with open(out_root / "reports" / "black_line_score_stats.json", "w") as f:
        json.dump({
            "bg_sigma": args.bg_sigma,
            "noise_sigma": args.noise_sigma,
            "global_mad": mad,
            "score_p99_5": sc_clip,
            "score_p99": float(np.percentile(score, 99)),
            "score_max": float(score.max()),
        }, f, indent=2)
    print(f"  global_mad={mad:.4f}, score p99.5={sc_clip:.2f}, "
          f"score>1 coverage={(score>args.black_threshold).mean()*100:.2f}%")

    protect = load_protect_mask(prepared)
    if protect is None:
        print("[warn] protect_mask.png not found in prepared/masks — using zeros")
        protect = np.zeros_like(img_f, dtype=bool)
    print(f"  protect_mask coverage={protect.mean()*100:.3f}%")

    print("[2/3] enumerating variants")
    sato_masks = {n: load_sato_mask(prepared, n) for n in args.masks}
    for n, m in sato_masks.items():
        print(f"  {n}: coverage={m.mean()*100:.2f}%")

    # Group by (mask, width_core, halo_width, side_sigma) -> one bg + masks
    # then inner-loop over (keep_core, keep_halo).
    grouped: dict[tuple[str, int, int, float], dict] = {}
    for mask_name, wc, hw, ks in product(
        args.masks, args.width_cores, args.halo_widths, args.side_sigmas,
    ):
        key = (mask_name, wc, hw, ks)
        if key not in grouped:
            grouped[key] = {"keep_combos": []}
    for kc, kh in product(args.keep_cores, args.keep_halos):
        for key in grouped:
            grouped[key]["keep_combos"].append((kc, kh))

    total = sum(len(v["keep_combos"]) for v in grouped.values())
    print(f"  groups={len(grouped)}, variants={total}")

    csv_rows: list[dict] = []
    records: list[dict] = []
    t0 = time.time()
    done = 0

    for (mask_name, wc, hw, side_sigma), entry in grouped.items():
        mask_tag = MASK_TAGS[mask_name]
        sato_m = sato_masks[mask_name]
        core, halo, _thin = build_core_halo(
            sato_m, score, protect,
            width_core=wc, halo_width=hw,
            black_threshold=args.black_threshold,
        )
        edit_mask = core | halo
        if not edit_mask.any():
            print(f"  [skip] {mask_tag}_wc{wc}_h{hw}_s{int(side_sigma)} "
                  f"empty edit_mask")
            continue
        bg, _side = side_band_bg(
            img_f, edit_mask, protect,
            side_radius=args.side_radius, sigma=side_sigma,
        )
        for keep_core, keep_halo in entry["keep_combos"]:
            if args.limit is not None and done >= args.limit:
                break
            name = variant_name_of(
                mask_tag, wc, hw, keep_core, keep_halo, side_sigma,
            )
            variant_dir = out_root / "variants" / name
            metrics, thumbs = run_variant(
                img_f=img_f, img_dtype=img_dtype,
                core_mask=core, halo_mask=halo, bg=bg,
                keep_core=keep_core, keep_halo=keep_halo,
                feather_radius=args.feather_radius,
                variant_dir=variant_dir, variant_name=name,
                mask_tag=mask_tag, width_core=wc, halo_width=hw,
                side_sigma=side_sigma,
                preview_lo=preview_lo, preview_hi=preview_hi,
                img_clip=img_clip, raw8_full=raw8_full,
            )
            csv_rows.append(metrics)
            records.append({"name": name, "metrics": metrics, "thumbs": thumbs})

            done += 1
            if done % 8 == 0 or done == 1:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                print(f"  [{done}/{total}] {name}  "
                      f"({elapsed:.1f}s, {rate:.2f}/s)")
        if args.limit is not None and done >= args.limit:
            break

    print(f"finished {done} variants in {time.time() - t0:.1f}s")

    print("[3/3] writing reports")
    reports_dir = out_root / "reports"
    csv_path = reports_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VARIANT_FIELDS)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"  wrote {csv_path}")

    # compare_sheet: focus on (kc=0.05, kh=0.3) family — illustrates the
    # core/halo split well; if none matches, fall back to the first 24.
    compare_subset = [r for r in records
                      if r["metrics"]["keep_core"] == 0.05
                      and r["metrics"]["keep_halo"] == 0.3]
    if not compare_subset:
        compare_subset = records[:24]
    save_compare_sheet(
        compare_subset, reports_dir / "compare_sheet.png",
        title=f"v2 compare_sheet — keep_core=0.05, keep_halo=0.3 "
              f"(rows={len(compare_subset)})",
    )
    print(f"  wrote {reports_dir / 'compare_sheet.png'} "
          f"({len(compare_subset)} rows)")

    if len(records) != len(compare_subset):
        save_compare_sheet(
            records, reports_dir / "compare_sheet_all.png",
            title=f"v2 compare_sheet_all (rows={len(records)})",
        )
        print(f"  wrote {reports_dir / 'compare_sheet_all.png'} "
              f"({len(records)} rows)")

    save_contact_sheet(records, reports_dir / "contact_sheet.png")
    print(f"  wrote {reports_dir / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
