"""Stage 1A: pseudo-label variant sweep.

Reads pre-computed responses under <dataset>/responses/ and emits 8 candidate
pseudo-label sets for human review under <dataset>/label_sweep/.

This script does NOT train a model, does NOT do background inpainting, and
does NOT do defect detection. Its only output is overlays + summary so the
human can pick the best label variant for later U-Net training.

Run as standalone:
    python sweep_pseudo_labels.py --dataset line_stage1_dataset

Or through the existing prepare_line_dataset.py:
    python prepare_line_dataset.py --input data.tif --out line_stage1_dataset --label_sweep
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import frangi, sato
from skimage.measure import label as cc_label
from skimage.measure import regionprops
from skimage.morphology import binary_dilation, disk, remove_small_objects


# ----------------------------- CLI -------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1A pseudo-label variant sweep")
    p.add_argument("--dataset", required=True, help="Existing dataset root, e.g. line_stage1_dataset")
    p.add_argument("--input", default=None, help="Optional fallback path to the source .tif (used if responses missing).")
    p.add_argument("--patch-grid", type=int, default=384, help="Patch size for patch_compare contact sheet.")
    return p.parse_args()


# ----------------------------- IO helpers ------------------------------------


def normalize01(x: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    lo = float(np.percentile(x, lo_pct))
    hi = float(np.percentile(x, hi_pct))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _structure_coherence(img: np.ndarray, sigma_d: float = 2.0, sigma_s: float = 8.0) -> np.ndarray:
    gx = ndi.gaussian_filter(img, sigma_d, order=(0, 1))
    gy = ndi.gaussian_filter(img, sigma_d, order=(1, 0))
    Jxx = ndi.gaussian_filter(gx * gx, sigma_s)
    Jxy = ndi.gaussian_filter(gx * gy, sigma_s)
    Jyy = ndi.gaussian_filter(gy * gy, sigma_s)
    trace = Jxx + Jyy
    disc = np.sqrt(np.maximum(0.0, ((Jxx - Jyy) ** 2) / 4.0 + Jxy * Jxy))
    lam1 = trace / 2.0 + disc
    lam2 = trace / 2.0 - disc
    return ((lam1 - lam2) / (lam1 + lam2 + 1e-12)) ** 2


def load_or_compute_responses(dataset: Path, input_path: Optional[Path]):
    """Prefer cached .tif responses; otherwise recompute from data.tif."""
    rdir = dataset / "responses"
    sato_p = rdir / "response_sato.tif"
    frangi_p = rdir / "response_frangi.tif"
    coh_p = rdir / "response_coherence.tif"

    if sato_p.exists() and frangi_p.exists() and coh_p.exists():
        print(f"[INFO] using cached responses from {rdir}")
        return (
            tifffile.imread(str(sato_p)).astype(np.float32),
            tifffile.imread(str(frangi_p)).astype(np.float32),
            tifffile.imread(str(coh_p)).astype(np.float32),
        )

    if input_path is None or not Path(input_path).exists():
        candidates = [dataset.parent / "data.tif", Path("data.tif")]
        input_path = next((c for c in candidates if c.exists()), None)
    if input_path is None:
        raise FileNotFoundError(
            "Cached responses not found and no usable --input .tif. "
            "Either run prepare_line_dataset.py first or pass --input."
        )

    print(f"[INFO] cached responses missing; recomputing from {input_path}")
    img = tifffile.imread(str(input_path)).astype(np.float32)
    if img.ndim == 3:
        print(f"[WARN] {input_path} is 3D {img.shape}; using first frame.")
        img = img[0]
    abs_img = np.abs(img)
    smooth = ndi.gaussian_filter(abs_img, sigma=1.0)
    sigmas = [2.0, 3.5, 5.5, 8.0]
    sato_r = sato(smooth, sigmas=sigmas, black_ridges=False).astype(np.float32)
    frangi_r = frangi(smooth, sigmas=sigmas, black_ridges=False).astype(np.float32)
    coh_r = _structure_coherence(smooth, 2.0, 8.0).astype(np.float32)
    return sato_r, frangi_r, coh_r


def load_preview(dataset: Path, fallback: np.ndarray) -> np.ndarray:
    p = dataset / "previews" / "input_preview.png"
    if p.exists():
        arr = np.array(Image.open(p)).astype(np.float32) / 255.0
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr
    print(f"[WARN] {p} not found; falling back to sato_norm as preview base.")
    return fallback


# ----------------------------- Variant logic ---------------------------------


def post_clear(clear_bool: np.ndarray) -> np.ndarray:
    cleaned = remove_small_objects(clear_bool, min_size=8)
    return binary_dilation(cleaned, footprint=disk(1))


def post_ignore(loose_excluding_clear: np.ndarray, clear_dilated: np.ndarray) -> np.ndarray:
    ig = loose_excluding_clear & (~clear_dilated)
    ig = binary_dilation(ig, footprint=disk(2))
    edge_band = binary_dilation(clear_dilated, footprint=disk(2)) & (~clear_dilated)
    ig = ig | edge_band
    return ig & (~clear_dilated)


def compact_protect_mask(clear_raw: np.ndarray) -> np.ndarray:
    """Connected components in clear_raw that look more like compact blobs than lines."""
    lab = cc_label(clear_raw, connectivity=2)
    keep = np.zeros_like(clear_raw, dtype=bool)
    for r in regionprops(lab):
        if r.area < 4 or r.area > 500:
            continue
        ecc = float(r.eccentricity)
        if r.minor_axis_length < 1e-6:
            ar = float("inf")
        else:
            ar = float(r.major_axis_length / r.minor_axis_length)
        if ecc < 0.85 and ar < 4.0:
            keep[lab == r.label] = True
    return keep


def build_variants(sato_norm: np.ndarray, frangi_norm: np.ndarray, coh_norm: np.ndarray):
    pct = lambda v, q: float(np.percentile(v, q))

    s975 = pct(sato_norm, 97.5)
    s98 = pct(sato_norm, 98.0)
    s90 = pct(sato_norm, 90.0)
    s85 = pct(sato_norm, 85.0)
    s82 = pct(sato_norm, 82.0)
    s80 = pct(sato_norm, 80.0)
    f90 = pct(frangi_norm, 90.0)
    f88 = pct(frangi_norm, 88.0)
    c90 = pct(coh_norm, 90.0)
    c88 = pct(coh_norm, 88.0)

    print(
        f"[INFO] sato thresholds: p97.5={s975:.4f} p98={s98:.4f} "
        f"p90={s90:.4f} p85={s85:.4f} p82={s82:.4f} p80={s80:.4f}"
    )

    raw: list[tuple[str, np.ndarray, np.ndarray]] = []

    raw.append(("A_clear97p5_loose90", sato_norm > s975, sato_norm > s90))
    raw.append(("B_clear97p5_loose85", sato_norm > s975, sato_norm > s85))
    raw.append(("C_clear98_loose85", sato_norm > s98, sato_norm > s85))
    raw.append(("D_clear98_loose80", sato_norm > s98, sato_norm > s80))
    raw.append(
        (
            "E_sato85_frangi90_coh90",
            sato_norm > s975,
            (sato_norm > s85) | (frangi_norm > f90) | (coh_norm > c90),
        )
    )
    raw.append(
        (
            "F_strong_clear_wide_ignore",
            sato_norm > s98,
            (sato_norm > s82) | (frangi_norm > f88) | (coh_norm > c88),
        )
    )

    # G: compact-blob protection
    g_clear_raw = sato_norm > s975
    g_loose = sato_norm > s85
    print("[INFO] computing compact-blob protection (variant G) ...")
    g_compact = compact_protect_mask(g_clear_raw)
    g_clear = g_clear_raw & (~g_compact)
    g_loose_with_compact = g_loose | g_compact
    raw.append(("G_compact_protect", g_clear, g_loose_with_compact))

    # H: simplified junction ignore
    h_clear = sato_norm > s975
    h_loose = sato_norm > s85
    h_junction = binary_dilation(h_clear, footprint=disk(5)) & (sato_norm > s90)
    h_loose_with_junction = h_loose | h_junction
    raw.append(("H_junction_ignore", h_clear, h_loose_with_junction))

    out = []
    for name, clear, loose in raw:
        clear_d = post_clear(clear)
        ignore_only = post_ignore(loose & (~clear_d), clear_d)
        train_label = np.zeros(clear_d.shape, dtype=np.uint8)
        train_label[ignore_only] = 255
        train_label[clear_d] = 1  # clear has priority over ignore
        out.append(
            {
                "name": name,
                "clear": clear_d.astype(bool),
                "ignore": ignore_only.astype(bool),
                "train_label": train_label,
            }
        )
    return out


# ----------------------------- Rendering -------------------------------------


def _overlay_rgb(preview: np.ndarray, clear: np.ndarray, ignore: np.ndarray) -> np.ndarray:
    base = np.clip(preview, 0, 1).astype(np.float32)
    rgb = np.stack([base, base, base], axis=-1)
    if ignore.any():
        rgb[ignore] = 0.45 * rgb[ignore] + 0.55 * np.array([1.0, 1.0, 0.0], dtype=np.float32)
    if clear.any():
        rgb[clear] = 0.45 * rgb[clear] + 0.55 * np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return rgb


def _save_uint8(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path)


def make_full_contact_sheet(variants, preview, out_path: Path) -> None:
    n = len(variants)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 4.0))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i >= n:
            continue
        v = variants[i]
        rgb = _overlay_rgb(preview, v["clear"], v["ignore"])
        ax.imshow(np.clip(rgb, 0, 1))
        L = v["clear"].mean() * 100.0
        I = v["ignore"].mean() * 100.0
        ax.set_title(f"{v['name']}\nL={L:.2f}%  I={I:.2f}%", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _select_representative_patches(
    sato_norm: np.ndarray,
    base_clear: np.ndarray,
    patch_size: int,
    total: int = 16,
):
    H, W = sato_norm.shape
    ps = min(patch_size, H, W)
    ys = list(range(0, H - ps + 1, ps))
    xs = list(range(0, W - ps + 1, ps))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]

    metrics = []
    for y in ys:
        for x in xs:
            sub_sato = sato_norm[y : y + ps, x : x + ps]
            sub_clear = base_clear[y : y + ps, x : x + ps]
            metrics.append(
                {
                    "y": y,
                    "x": x,
                    "mean_sato": float(sub_sato.mean()),
                    "max_sato": float(sub_sato.max()),
                    "clear_density": float(sub_clear.mean()),
                }
            )

    median_mean = float(np.median([m["mean_sato"] for m in metrics]))
    sorted_strong = sorted(metrics, key=lambda m: m["mean_sato"], reverse=True)
    sorted_jn = sorted(metrics, key=lambda m: m["clear_density"], reverse=True)
    sorted_weak = sorted(metrics, key=lambda m: abs(m["mean_sato"] - median_mean))
    sorted_bg = sorted(metrics, key=lambda m: m["mean_sato"])

    chosen: list[tuple[str, dict]] = []
    used: set[tuple[int, int]] = set()

    def take(lst, k, tag):
        out = []
        for m in lst:
            key = (m["y"], m["x"])
            if key in used:
                continue
            used.add(key)
            out.append((tag, m))
            if len(out) == k:
                break
        return out

    chosen.extend(take(sorted_strong, 4, "strong_line"))
    chosen.extend(take(sorted_jn, 4, "junction"))
    chosen.extend(take(sorted_weak, 4, "weak_line"))
    chosen.extend(take(sorted_bg, 4, "background"))

    if len(chosen) < total:
        for m in metrics:
            if (m["y"], m["x"]) in used:
                continue
            used.add((m["y"], m["x"]))
            chosen.append(("extra", m))
            if len(chosen) >= total:
                break
    return chosen[:total], ps


def make_patch_compare_sheet(
    variants, preview: np.ndarray, sato_norm: np.ndarray, out_path: Path, patch_size: int = 384
) -> None:
    base_clear = variants[0]["clear"]
    chosen, ps = _select_representative_patches(sato_norm, base_clear, patch_size, total=16)
    n_rows = len(chosen)
    n_cols = len(variants)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.7, n_rows * 1.8))
    axes = np.atleast_2d(axes)
    for r, (cat, m) in enumerate(chosen):
        y, x = m["y"], m["x"]
        for c, v in enumerate(variants):
            ax = axes[r, c]
            ax.axis("off")
            sub_pre = preview[y : y + ps, x : x + ps]
            sub_clear = v["clear"][y : y + ps, x : x + ps]
            sub_ignore = v["ignore"][y : y + ps, x : x + ps]
            rgb = _overlay_rgb(sub_pre, sub_clear, sub_ignore)
            ax.imshow(np.clip(rgb, 0, 1))
            if r == 0:
                ax.set_title(v["name"].split("_")[0], fontsize=10)
            if c == 0:
                ax.text(
                    0.02,
                    0.98,
                    f"{cat}\n({y},{x})",
                    transform=ax.transAxes,
                    fontsize=7,
                    color="white",
                    ha="left",
                    va="top",
                    bbox=dict(facecolor="black", alpha=0.55, pad=2, edgecolor="none"),
                )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ----------------------------- Main entry ------------------------------------


def run_sweep(dataset_dir: Path, input_path: Optional[Path] = None, patch_grid: int = 384) -> Path:
    dataset_dir = Path(dataset_dir)
    out_root = dataset_dir / "label_sweep"
    for sub in ("labels", "masks", "overlays", "reports"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    sato_r, frangi_r, coh_r = load_or_compute_responses(dataset_dir, input_path)
    sato_norm = normalize01(sato_r)
    frangi_norm = normalize01(frangi_r)
    coh_norm = normalize01(coh_r)
    preview = load_preview(dataset_dir, sato_norm)

    variants = build_variants(sato_norm, frangi_norm, coh_norm)

    H, W = sato_norm.shape
    total_pixels = H * W

    csv_rows = []
    print("[INFO] writing labels / masks / overlays ...")
    for v in variants:
        name = v["name"]
        clear = v["clear"]
        ignore = v["ignore"]
        train_label = v["train_label"]

        clear_pix = int(clear.sum())
        ignore_pix = int(ignore.sum())
        bg_pix = total_pixels - clear_pix - ignore_pix
        clear_pct = clear_pix / total_pixels * 100.0
        ignore_pct = ignore_pix / total_pixels * 100.0
        bg_pct = bg_pix / total_pixels * 100.0

        _save_uint8(train_label, out_root / "labels" / f"{name}_train_label.png")
        _save_uint8((clear.astype(np.uint8) * 255), out_root / "masks" / f"{name}_clear.png")
        _save_uint8((ignore.astype(np.uint8) * 255), out_root / "masks" / f"{name}_ignore.png")

        rgb = _overlay_rgb(preview, clear, ignore)
        overlay_name = f"{name}_overlay_L{clear_pct:.1f}_I{ignore_pct:.1f}.png"
        _save_uint8((np.clip(rgb, 0, 1) * 255).astype(np.uint8), out_root / "overlays" / overlay_name)

        warns = []
        if ignore_pct > 30.0:
            warns.append(f"ignore > 30% ({ignore_pct:.1f}%), too much ignored area")
        if clear_pct > 12.0:
            warns.append(f"clear > 12% ({clear_pct:.1f}%), clear line may be too broad")
        if clear_pct < 1.0:
            warns.append(f"clear < 1% ({clear_pct:.1f}%), clear line may be too conservative")
        warning = "; ".join(warns)

        csv_rows.append(
            {
                "variant_name": name,
                "clear_pixels": clear_pix,
                "clear_percent": round(clear_pct, 4),
                "ignore_pixels": ignore_pix,
                "ignore_percent": round(ignore_pct, 4),
                "background_pixels": bg_pix,
                "background_percent": round(bg_pct, 4),
                "warning": warning,
            }
        )
        marker = "  WARN: " + warning if warning else ""
        print(f"  {name}: L={clear_pct:.2f}% I={ignore_pct:.2f}% B={bg_pct:.2f}%{marker}")

    print("[INFO] generating contact sheets ...")
    sheet_path = out_root / "reports" / "label_sweep_contact_sheet.png"
    make_full_contact_sheet(variants, preview, sheet_path)
    try:
        make_patch_compare_sheet(
            variants,
            preview,
            sato_norm,
            out_root / "reports" / "patch_compare_contact_sheet.png",
            patch_size=patch_grid,
        )
    except Exception as e:
        print(f"[WARN] patch_compare_contact_sheet failed: {e}")

    csv_path = out_root / "reports" / "label_sweep_summary.csv"
    fieldnames = [
        "variant_name",
        "clear_pixels",
        "clear_percent",
        "ignore_pixels",
        "ignore_percent",
        "background_pixels",
        "background_percent",
        "warning",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in csv_rows:
            w.writerow(row)

    print()
    print("Done label sweep.")
    print("Please check:")
    print(f"  {sheet_path.as_posix()}")
    print(f"  {csv_path.as_posix()}")
    return sheet_path


def main() -> None:
    args = parse_args()
    run_sweep(
        Path(args.dataset),
        Path(args.input) if args.input else None,
        patch_grid=args.patch_grid,
    )


if __name__ == "__main__":
    main()
