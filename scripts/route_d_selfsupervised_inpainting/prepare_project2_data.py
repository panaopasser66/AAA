"""Stage 1: prepare data, responses, line/protect masks, and self-supervised
inpainting training patches from a single Kossel-projection-like image.

Outputs:
  outputs/prepare/previews/{input_preview,abs_preview}.png
  outputs/prepare/reports/input_stats.json
  outputs/prepare/responses/response_*.{tif,png}
  outputs/prepare/masks/{mask_A_clear_sato98,mask_B_sato95,mask_C_sato90,mask_D_sato85,protect_mask}.png
  outputs/prepare/overlays/*_overlay.png
  outputs/prepare/reports/{mask_stats.csv,mask_contact_sheet.png}
  outputs/prepare/patches/images/img_NNNNNN.npz
  outputs/prepare/manifests/train_manifest.csv
  outputs/prepare/global/global_arrays.npz   (full-image normalized layers, used by inference)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import morphology
from skimage.filters import sato, frangi, meijering, gaussian
from skimage.measure import label, regionprops


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _read_tif(path: Path) -> np.ndarray:
    img = tifffile.imread(str(path))
    if img.ndim == 3:
        print(f"[warn] input is 3D {img.shape}; using first frame")
        img = img[0]
    if img.ndim != 2:
        raise ValueError(f"unexpected image ndim={img.ndim}, shape={img.shape}")
    return img.astype(np.float32)


def _save_png(path: Path, arr: np.ndarray) -> None:
    """Save a 2D float array as 8-bit png via robust percentile stretch."""
    import imageio.v2 as imageio  # type: ignore
    a = arr.astype(np.float32)
    finite = np.isfinite(a)
    if finite.sum() == 0:
        u = np.zeros_like(a, dtype=np.uint8)
    else:
        lo, hi = np.percentile(a[finite], [1.0, 99.0])
        if hi - lo < 1e-12:
            hi = float(a[finite].max())
            lo = float(a[finite].min())
        if hi - lo < 1e-12:
            u = np.zeros_like(a, dtype=np.uint8)
        else:
            u = np.clip((a - lo) / (hi - lo), 0, 1)
            u = (u * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), u)


def _save_overlay(path: Path, base: np.ndarray, mask: np.ndarray, color=(255, 60, 60)) -> None:
    import imageio.v2 as imageio
    a = base.astype(np.float32)
    finite = np.isfinite(a)
    lo, hi = np.percentile(a[finite], [1.0, 99.0]) if finite.any() else (0.0, 1.0)
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    u = np.clip((a - lo) / (hi - lo), 0, 1)
    rgb = np.stack([u, u, u], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)
    m = mask.astype(bool)
    rgb[m] = color
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), rgb)


def _save_mask_png(path: Path, mask: np.ndarray) -> None:
    import imageio.v2 as imageio
    u = (mask.astype(bool).astype(np.uint8)) * 255
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), u)


def _save_tif(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr.astype(np.float32))


def _normalize_minmax(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype=np.float32)
    lo = float(a[finite].min())
    hi = float(a[finite].max())
    if hi - lo < 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    out = (a - lo) / (hi - lo)
    return out.astype(np.float32)


def _normalize_percentile(arr: np.ndarray, p_lo=1.0, p_hi=99.0) -> np.ndarray:
    a = arr.astype(np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype=np.float32)
    lo, hi = np.percentile(a[finite], [p_lo, p_hi])
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    out = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

def compute_responses(abs_img: np.ndarray, sigmas, pre_sigma=1.0,
                       sigma_d=2.0, sigma_s=8.0):
    """Compute the 6 response maps used by downstream stages.

    Returns dict[str -> np.ndarray] with raw (unnormalized) responses.
    """
    img_s = gaussian(abs_img, sigma=pre_sigma, preserve_range=True)

    print("  [resp] sato")
    r_sato = sato(img_s, sigmas=sigmas, black_ridges=False)

    print("  [resp] frangi")
    r_frangi = frangi(img_s, sigmas=sigmas, black_ridges=False)

    print("  [resp] meijering")
    r_meij = meijering(img_s, sigmas=sigmas, black_ridges=False)

    print("  [resp] gradient")
    gx = ndi.sobel(img_s, axis=1)
    gy = ndi.sobel(img_s, axis=0)
    r_grad = np.hypot(gx, gy)

    print("  [resp] coherence")
    Jxx = ndi.gaussian_filter(gx * gx, sigma=sigma_s)
    Jxy = ndi.gaussian_filter(gx * gy, sigma=sigma_s)
    Jyy = ndi.gaussian_filter(gy * gy, sigma=sigma_s)
    tmp = np.sqrt(np.maximum((Jxx - Jyy) ** 2 + 4.0 * Jxy ** 2, 0.0))
    lam1 = 0.5 * (Jxx + Jyy + tmp)
    lam2 = 0.5 * (Jxx + Jyy - tmp)
    coh = (lam1 - lam2) / (lam1 + lam2 + 1e-12)
    r_coh = np.clip(coh, 0.0, 1.0).astype(np.float32)

    s_n = _normalize_percentile(r_sato)
    f_n = _normalize_percentile(r_frangi)
    g_n = _normalize_percentile(r_grad)
    c_n = r_coh.astype(np.float32)
    fused = 0.5 * s_n + 0.2 * f_n + 0.15 * g_n + 0.15 * c_n

    return {
        "sato": r_sato.astype(np.float32),
        "frangi": r_frangi.astype(np.float32),
        "meijering": r_meij.astype(np.float32),
        "gradient": r_grad.astype(np.float32),
        "coherence": r_coh.astype(np.float32),
        "fused": fused.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def _postprocess_mask(mask: np.ndarray, min_size=8, dilate_radius=1) -> np.ndarray:
    m = morphology.remove_small_objects(mask.astype(bool), min_size=min_size)
    if dilate_radius > 0:
        m = morphology.binary_dilation(m, footprint=morphology.disk(dilate_radius))
    return m.astype(bool)


def build_line_masks(sato_norm: np.ndarray):
    """Return ordered dict-like list of (name, mask_bool, percentile)."""
    levels = [
        ("mask_A_clear_sato98", 98.0),
        ("mask_B_sato95", 95.0),
        ("mask_C_sato90", 90.0),
        ("mask_D_sato85", 85.0),
        ("mask_E_sato80", 80.0),
        ("mask_F_sato75", 75.0),
    ]
    out = []
    for name, p in levels:
        thr = float(np.percentile(sato_norm, p))
        m = sato_norm > thr
        m = _postprocess_mask(m, min_size=8, dilate_radius=1)
        cov = float(m.mean())
        if cov > 0.30:
            print(f"  [warn] mask {name} coverage {cov*100:.2f}% > 30%")
        out.append((name, m, p, thr, cov))
    return out


def build_protect_mask(img: np.ndarray):
    """Detect small compact "blob" candidates we don't want inpainting to erase."""
    local_bg = ndi.gaussian_filter(img, sigma=15.0)
    residual = img - local_bg
    abs_res = np.abs(residual)
    thr = float(np.percentile(abs_res, 99.3))
    cand = abs_res > thr
    lbl = label(cand, connectivity=2)
    keep = np.zeros_like(cand, dtype=bool)
    n_kept = 0
    for region in regionprops(lbl):
        if region.area < 4 or region.area > 500:
            continue
        if region.eccentricity >= 0.9:
            continue
        minr, minc, maxr, maxc = region.bbox
        h = maxr - minr
        w = maxc - minc
        ar = max(h, w) / max(1, min(h, w))
        if ar >= 4:
            continue
        keep[lbl == region.label] = True
        n_kept += 1
    keep = morphology.binary_dilation(keep, footprint=morphology.disk(1))
    print(f"  [protect] kept {n_kept} compact components, coverage={keep.mean()*100:.3f}%")
    return keep, thr


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def make_patches(layers: dict, channel_names: list, patch_size: int, stride: int,
                  out_patch_dir: Path, manifest_path: Path):
    H, W = layers[channel_names[0]].shape
    out_patch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    ys = list(range(0, max(1, H - patch_size + 1), stride))
    xs = list(range(0, max(1, W - patch_size + 1), stride))
    if ys[-1] + patch_size < H:
        ys.append(H - patch_size)
    if xs[-1] + patch_size < W:
        xs.append(W - patch_size)

    rows = []
    idx = 0
    for y in ys:
        for x in xs:
            stack = np.stack(
                [layers[c][y:y + patch_size, x:x + patch_size] for c in channel_names],
                axis=0,
            ).astype(np.float32)

            # v2: forbidden uses line_mask_75 (strictest line cutoff) + protect_mask
            line_v2_idx = channel_names.index("line_mask_75")
            line_v1_idx = channel_names.index("line_mask_85")
            protect_idx = channel_names.index("protect_mask")
            forbidden_v2 = (stack[line_v2_idx] > 0.5) | (stack[protect_idx] > 0.5)
            forbidden_v1 = (stack[line_v1_idx] > 0.5) | (stack[protect_idx] > 0.5)
            valid_frac = float((~forbidden_v2).mean())
            valid_frac_v1 = float((~forbidden_v1).mean())

            idx += 1
            fname = f"img_{idx:06d}.npz"
            np.savez_compressed(
                out_patch_dir / fname,
                data=stack,
                channel_names=np.array(channel_names),
                y=y,
                x=x,
            )
            rows.append({
                "id": idx,
                "file": str((out_patch_dir / fname).as_posix()),
                "y": y,
                "x": x,
                "patch_size": patch_size,
                "valid_hole_frac": round(valid_frac, 4),
                "valid_hole_frac_v1": round(valid_frac_v1, 4),
            })

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"  [patches] wrote {len(rows)} patches; manifest -> {manifest_path}")
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _imshow_norm(ax, arr, title=None, cmap="gray"):
    a = arr.astype(np.float32)
    finite = np.isfinite(a)
    if finite.any():
        lo, hi = np.percentile(a[finite], [1, 99])
    else:
        lo, hi = 0.0, 1.0
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    ax.imshow(a, vmin=lo, vmax=hi, cmap=cmap)
    if title:
        ax.set_title(title, fontsize=8)
    ax.set_axis_off()


def make_mask_contact_sheet(path: Path, abs_img: np.ndarray, masks_named, protect_mask):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(masks_named) + 1  # +protect
    rows = 2
    cols = n
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 6.0))
    if rows == 1:
        axes = axes[None, :]

    # Row 0: mask alone (boolean)
    # Row 1: overlay on abs_img
    a = abs_img.astype(np.float32)
    finite = np.isfinite(a)
    lo, hi = np.percentile(a[finite], [1, 99]) if finite.any() else (0.0, 1.0)
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    base = np.clip((a - lo) / (hi - lo), 0, 1)

    def _overlay_rgb(mask, color=(1.0, 0.2, 0.2)):
        rgb = np.stack([base, base, base], axis=-1)
        m = mask.astype(bool)
        rgb[m] = color
        return rgb

    items = list(masks_named) + [("protect_mask", protect_mask, None, None, float(protect_mask.mean()))]
    for j, (name, m, _p, _thr, cov) in enumerate(items):
        axes[0, j].imshow(m.astype(np.uint8) * 255, cmap="gray", vmin=0, vmax=255)
        axes[0, j].set_title(f"{name}\ncov={cov*100:.2f}%", fontsize=8)
        axes[0, j].set_axis_off()
        axes[1, j].imshow(_overlay_rgb(m))
        axes[1, j].set_title(f"{name} overlay", fontsize=8)
        axes[1, j].set_axis_off()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patch-size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "previews").mkdir(parents=True, exist_ok=True)
    (out / "responses").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "patches" / "images").mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "global").mkdir(parents=True, exist_ok=True)

    print(f"[1/5] reading {args.input}")
    img = _read_tif(Path(args.input))
    abs_img = np.abs(img)

    stats = {
        "shape": list(img.shape),
        "dtype_in": str(np.asarray(tifffile.imread(args.input)).dtype),
        "raw_min": float(img.min()),
        "raw_max": float(img.max()),
        "raw_mean": float(img.mean()),
        "raw_std": float(img.std()),
        "abs_min": float(abs_img.min()),
        "abs_max": float(abs_img.max()),
        "abs_mean": float(abs_img.mean()),
    }
    with open(out / "reports" / "input_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    _save_png(out / "previews" / "input_preview.png", img)
    _save_png(out / "previews" / "abs_preview.png", abs_img)

    print("[2/5] computing responses (multi-scale)")
    sigmas = [2.0, 3.5, 5.5, 8.0]
    responses = compute_responses(abs_img, sigmas=sigmas, pre_sigma=1.0,
                                  sigma_d=2.0, sigma_s=8.0)
    for name, r in responses.items():
        _save_tif(out / "responses" / f"response_{name}.tif", r)
        _save_png(out / "responses" / f"response_{name}.png", r)

    sato_raw = responses["sato"]
    sato_norm = _normalize_percentile(sato_raw, 1, 99)

    print("[3/5] line masks")
    masks_named = build_line_masks(sato_norm)
    for name, m, p, thr, cov in masks_named:
        _save_mask_png(out / "masks" / f"{name}.png", m)
        _save_overlay(out / "overlays" / f"{name}_overlay.png", abs_img, m)

    print("[3b/5] protect mask")
    protect_mask, prot_thr = build_protect_mask(img)
    _save_mask_png(out / "masks" / "protect_mask.png", protect_mask)
    _save_overlay(out / "overlays" / "protect_overlay.png", abs_img, protect_mask, color=(80, 200, 60))

    # csv stats
    with open(out / "reports" / "mask_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "percentile", "threshold", "coverage_pct"])
        for name, m, p, thr, cov in masks_named:
            w.writerow([name, p, thr, round(cov * 100, 4)])
        w.writerow(["protect_mask", "", round(prot_thr, 6), round(protect_mask.mean() * 100, 4)])

    make_mask_contact_sheet(out / "reports" / "mask_contact_sheet.png", abs_img, masks_named, protect_mask)

    print("[4/5] preparing global normalized layers and patches")
    raw_norm = _normalize_percentile(img, 1, 99)
    abs_norm = _normalize_percentile(abs_img, 1, 99)
    sato_n = _normalize_percentile(responses["sato"])
    frangi_n = _normalize_percentile(responses["frangi"])
    coherence_n = responses["coherence"].astype(np.float32)  # already in [0,1]
    gradient_n = _normalize_percentile(responses["gradient"])

    line_75 = next(m for n, m, *_ in masks_named if n == "mask_F_sato75").astype(np.float32)
    line_80 = next(m for n, m, *_ in masks_named if n == "mask_E_sato80").astype(np.float32)
    line_85 = next(m for n, m, *_ in masks_named if n == "mask_D_sato85").astype(np.float32)
    line_90 = next(m for n, m, *_ in masks_named if n == "mask_C_sato90").astype(np.float32)
    line_95 = next(m for n, m, *_ in masks_named if n == "mask_B_sato95").astype(np.float32)
    line_98 = next(m for n, m, *_ in masks_named if n == "mask_A_clear_sato98").astype(np.float32)
    protect = protect_mask.astype(np.float32)

    layers = {
        "raw_norm": raw_norm,
        "abs_norm": abs_norm,
        "sato_norm": sato_n,
        "frangi_norm": frangi_n,
        "coherence_norm": coherence_n,
        "gradient_norm": gradient_n,
        "line_mask_98": line_98,
        "line_mask_95": line_95,
        "line_mask_90": line_90,
        "line_mask_85": line_85,
        "line_mask_80": line_80,
        "line_mask_75": line_75,
        "protect_mask": protect,
    }

    np.savez_compressed(
        out / "global" / "global_arrays.npz",
        **layers,
        channel_names=np.array(list(layers.keys())),
    )

    channel_names = list(layers.keys())
    rows = make_patches(
        layers=layers,
        channel_names=channel_names,
        patch_size=args.patch_size,
        stride=args.stride,
        out_patch_dir=out / "patches" / "images",
        manifest_path=out / "manifests" / "train_manifest.csv",
    )

    summary = {
        "patches": len(rows),
        "patch_size": args.patch_size,
        "stride": args.stride,
        "channel_names": channel_names,
        "image_shape": list(img.shape),
    }
    with open(out / "reports" / "prepare_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[5/5] done.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
