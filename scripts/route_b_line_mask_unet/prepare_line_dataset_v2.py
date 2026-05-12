"""Stage 1 v2 — Kossel line dataset preparation with black-line-aware labels.

Differences from `prepare_line_dataset.py` (v1):

1. Adds **black_line_score** channel = (bg_low - img)+ / max(local_noise, MAD).
   Dropped frangi/meijering — keeps the channel stack at 6 inputs:
       raw, abs, sato, black_score, coherence, gradient.

2. Wider, black-line-aware **pseudo labels**:
   - positive (1): `sato_norm > p87` **AND** `black_line_score > 1.0`,
     dilated by 1 px, small components removed.
     Target ≈ 7–12 % coverage (v1 default was ~3 %).
   - ignore (255):
       * loose ridge (`sato_norm > p75` **OR** `black_line_score > 0.5`)
         not in positive
       * edge buffer = `dilate(positive, 4) − positive`
   - background (0): everything else (clearly non-line + non-uncertain).

   Rationale: never treat a visibly-weak Kossel line as background; weak /
   in-between pixels go to `ignore` so the masked BCE / Dice loss skips them.

3. Same output layout as v1 so existing `train_line_mask_unet.py` consumes
   it directly:
       <out>/patches/{images,labels,ignore}/...
       <out>/manifests/train_manifest.csv
       <out>/pseudo_labels/{train_label.png,...}
       <out>/responses/response_{sato,coherence,gradient,black_score}.{tif,png}

Run:
    python prepare_line_dataset_v2.py --input data/data.tif \
        --out outputs/route_b_prepare_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import sato
from skimage.morphology import binary_dilation, disk, remove_small_objects


CHANNEL_NAMES = ["raw", "abs", "sato", "black_score", "coherence", "gradient"]


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def read_tif(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim == 3:
        print(f"[WARN] {path.name} is 3D with shape {arr.shape}; using first frame.")
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Unsupported tif shape {arr.shape}; expected 2D.")
    return arr.astype(np.float32, copy=False)


def save_preview_png(img: np.ndarray, out_path: Path, lo_pct: float, hi_pct: float) -> None:
    lo = float(np.percentile(img, lo_pct))
    hi = float(np.percentile(img, hi_pct))
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((norm * 255).astype(np.uint8)).save(out_path)


def save_response_pair(resp: np.ndarray, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(base.with_suffix(".tif")), resp.astype(np.float32))
    save_preview_png(resp, base.with_suffix(".png"), 0.5, 99.5)


def normalize01(x: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    lo = float(np.percentile(x, lo_pct))
    hi = float(np.percentile(x, hi_pct))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Response channels
# --------------------------------------------------------------------------- #

def structure_coherence(img: np.ndarray, sigma_d: float = 2.0, sigma_s: float = 8.0) -> np.ndarray:
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


def gradient_magnitude(smooth: np.ndarray) -> np.ndarray:
    gx = ndi.gaussian_filter(smooth, sigma=1.5, order=(0, 1))
    gy = ndi.gaussian_filter(smooth, sigma=1.5, order=(1, 0))
    return np.hypot(gx, gy).astype(np.float32)


def compute_black_line_score(
    img_f: np.ndarray, sigma_bg: float = 20.0, sigma_noise: float = 8.0,
) -> tuple[np.ndarray, float]:
    bg_low = ndi.gaussian_filter(img_f, sigma=sigma_bg).astype(np.float32)
    dark_residual = (bg_low - img_f).astype(np.float32)
    res = (img_f - bg_low).astype(np.float32)
    med = float(np.median(res))
    global_mad = float(np.median(np.abs(res - med)) * 1.4826) + 1e-6
    local_mean = ndi.gaussian_filter(res, sigma=sigma_noise)
    centered = res - local_mean
    local_var = ndi.gaussian_filter(centered * centered, sigma=sigma_noise)
    local_noise = np.sqrt(np.maximum(local_var, 1e-12)).astype(np.float32)
    noise = np.maximum(local_noise, global_mad)
    return (np.clip(dark_residual, 0.0, None) / np.maximum(noise, 1e-6)).astype(np.float32), global_mad


def compute_responses(img: np.ndarray, sigmas: list[float]) -> dict[str, np.ndarray]:
    """Compute the 4 response channels used as inputs and label sources.

    IMPORTANT — fix vs v1: v1 used ``sato(abs_img, black_ridges=False)``,
    which detects *bright* ridges. For an X-ray Kossel projection the
    Kossel lines are *dark*; an empirical probe showed
        on top-5% darkest pixels: sato(bright)=21, sato(dark)=197
        on top-5% sato(bright) pixels: dark_residual=-203 (BRIGHTER than bg!)
    so v1's sato hit bright structures near (not on) the dark lines, which
    is why v1's sato_mask + black_score AND coverage collapsed to ~0.1%
    and v1's pseudo labels needed huge dilation to compensate.

    v2 uses ``sato(raw, black_ridges=True)``: the response peaks exactly
    on the dark Kossel ridges, so sato_high ∩ black_high is now a clean
    positive prior.
    """
    abs_img = np.abs(img)
    smooth_abs = ndi.gaussian_filter(abs_img, sigma=1.0)
    smooth_raw = ndi.gaussian_filter(img, sigma=1.0)
    print(f"[INFO] Sato (raw, black_ridges=True, sigmas={sigmas}) ...")
    r_sato = sato(smooth_raw, sigmas=sigmas, black_ridges=True).astype(np.float32)
    print("[INFO] black_line_score ...")
    r_black, mad = compute_black_line_score(img, sigma_bg=20.0, sigma_noise=8.0)
    print(f"  global_mad={mad:.4f}, black_score>1.0 cov="
          f"{(r_black > 1.0).mean()*100:.2f}%")
    print("[INFO] structure tensor coherence (on smooth raw) ...")
    coh = structure_coherence(smooth_raw, sigma_d=2.0, sigma_s=8.0).astype(np.float32)
    print("[INFO] gradient magnitude (on smooth abs) ...")
    grad = gradient_magnitude(smooth_abs)
    return {
        "sato": r_sato, "black_score": r_black,
        "coherence": coh, "gradient": grad,
    }


# --------------------------------------------------------------------------- #
# Pseudo labels — wider, black-line-aware
# --------------------------------------------------------------------------- #

def make_pseudo_labels(
    sato_resp: np.ndarray, black_score: np.ndarray,
    sato_pos_pct: float = 87.0, sato_loose_pct: float = 75.0,
    black_pos_sigma: float = 1.0, black_loose_sigma: float = 0.5,
    pos_dilate: int = 1, edge_buffer: int = 4, min_blob_area: int = 15,
) -> dict[str, np.ndarray]:
    sato_norm = normalize01(sato_resp, 0.5, 99.5)
    thr_pos = float(np.percentile(sato_norm, sato_pos_pct))
    thr_loose = float(np.percentile(sato_norm, sato_loose_pct))
    print(f"[INFO] sato thresholds: positive={thr_pos:.4f}  loose={thr_loose:.4f}")
    print(f"[INFO] black thresholds: positive={black_pos_sigma}σ  loose={black_loose_sigma}σ")

    # Positive: ridge AND dark
    positive_core = (sato_norm > thr_pos) & (black_score > black_pos_sigma)
    if pos_dilate > 0:
        positive = binary_dilation(positive_core, footprint=disk(pos_dilate))
    else:
        positive = positive_core
    positive = remove_small_objects(positive, min_size=min_blob_area)

    # Loose ridge or weak dark — uncertain band
    loose_ridge = (sato_norm > thr_loose) | (black_score > black_loose_sigma)
    ignore_zone = loose_ridge & ~positive

    # Edge buffer around positive
    if edge_buffer > 0:
        edge = binary_dilation(positive, footprint=disk(edge_buffer)) & ~positive
        ignore_zone = ignore_zone | edge

    ignore_zone = ignore_zone & ~positive

    train_label = np.zeros(positive.shape, dtype=np.uint8)
    train_label[positive] = 1
    train_label[ignore_zone] = 255

    return {
        "positive": positive.astype(np.uint8),
        "ignore": ignore_zone.astype(np.uint8),
        "train_label": train_label,
        "stats": {
            "sato_pos_threshold": thr_pos,
            "sato_loose_threshold": thr_loose,
            "black_pos_sigma": black_pos_sigma,
            "black_loose_sigma": black_loose_sigma,
            "positive_coverage_pct": float(positive.mean() * 100.0),
            "ignore_coverage_pct": float(ignore_zone.mean() * 100.0),
            "background_coverage_pct": float((train_label == 0).mean() * 100.0),
        },
    }


# --------------------------------------------------------------------------- #
# Overlays
# --------------------------------------------------------------------------- #

def img_to_rgb(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    lo = float(np.percentile(img, lo_pct))
    hi = float(np.percentile(img, hi_pct))
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return np.stack([norm, norm, norm], axis=-1).astype(np.float32)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color, alpha: float = 0.55) -> np.ndarray:
    out = rgb.copy()
    m = mask.astype(bool)
    if m.any():
        out[m] = (1.0 - alpha) * rgb[m] + alpha * np.array(color, dtype=np.float32)
    return out


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)


# --------------------------------------------------------------------------- #
# Patches
# --------------------------------------------------------------------------- #

def slice_patches(
    stacked: np.ndarray, train_label: np.ndarray,
    patch_size: int, stride: int,
    min_line_ratio: float, max_ignore_ratio: float,
    root: Path, base_name: str,
) -> list[dict]:
    _, H, W = stacked.shape
    if patch_size > H or patch_size > W:
        print(f"[WARN] patch_size {patch_size} > image {H}x{W}; skip.")
        return []

    rows: list[dict] = []
    pid = 0
    channel_names = np.array(CHANNEL_NAMES)

    ys = list(range(0, H - patch_size + 1, stride))
    xs = list(range(0, W - patch_size + 1, stride))
    if ys[-1] != H - patch_size:
        ys.append(H - patch_size)
    if xs[-1] != W - patch_size:
        xs.append(W - patch_size)

    img_dir = root / "patches" / "images"
    lab_dir = root / "patches" / "labels"
    ign_dir = root / "patches" / "ignore"
    for d in (img_dir, lab_dir, ign_dir):
        d.mkdir(parents=True, exist_ok=True)

    for y in ys:
        for x in xs:
            patch = stacked[:, y:y + patch_size, x:x + patch_size]
            label = train_label[y:y + patch_size, x:x + patch_size]
            line_ratio = float((label == 1).mean())
            ignore_ratio = float((label == 255).mean())
            if line_ratio < min_line_ratio:
                continue
            if ignore_ratio > max_ignore_ratio:
                continue

            pid += 1
            patch_id = f"{base_name}_{pid:06d}"
            img_path = img_dir / f"img_{patch_id}.npz"
            label_path = lab_dir / f"label_{patch_id}.png"
            ignore_path = ign_dir / f"ignore_{patch_id}.png"

            np.savez_compressed(img_path, image=patch, channel_names=channel_names)
            Image.fromarray(label).save(label_path)
            Image.fromarray(((label == 255).astype(np.uint8) * 255)).save(ignore_path)

            rows.append({
                "patch_id": patch_id,
                "image_path": str(img_path.relative_to(root)).replace("\\", "/"),
                "label_path": str(label_path.relative_to(root)).replace("\\", "/"),
                "ignore_path": str(ignore_path.relative_to(root)).replace("\\", "/"),
                "x0": x, "y0": y, "w": patch_size, "h": patch_size,
                "line_ratio": line_ratio, "ignore_ratio": ignore_ratio,
            })
    return rows


def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["patch_id", "image_path", "label_path", "ignore_path",
                  "x0", "y0", "w", "h", "line_ratio", "ignore_ratio"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def patch_contact_sheet(rows: list[dict], root: Path, out_path: Path, max_n: int = 48) -> None:
    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda r: r["line_ratio"], reverse=True)[:max_n]
    n = len(rows_sorted)
    cols = 8
    nrows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 2, nrows * 2))
    axes = np.atleast_2d(axes)
    for i in range(nrows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i >= n:
            continue
        r = rows_sorted[i]
        with np.load(root / r["image_path"]) as d:
            patch = d["image"]
        raw = patch[0]
        label = np.array(Image.open(root / r["label_path"]))
        rgb = np.stack([raw, raw, raw], axis=-1).astype(np.float32)
        rgb = overlay_mask(rgb, label == 255, (1.0, 1.0, 0.0), alpha=0.45)
        rgb = overlay_mask(rgb, label == 1, (0.0, 1.0, 0.0), alpha=0.55)
        ax.imshow(np.clip(rgb, 0, 1))
        ax.set_title(f"{r['patch_id']}\nL={r['line_ratio']:.2%}  I={r['ignore_ratio']:.2%}",
                     fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1 v2 line-dataset preparation")
    p.add_argument("--input", required=True, help="Path to data.tif")
    p.add_argument("--out", required=True, help="Output dataset root")
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--sato-pos-pct", type=float, default=87.0)
    p.add_argument("--sato-loose-pct", type=float, default=75.0)
    p.add_argument("--black-pos-sigma", type=float, default=1.0)
    p.add_argument("--black-loose-sigma", type=float, default=0.5)
    p.add_argument("--pos-dilate", type=int, default=1)
    p.add_argument("--edge-buffer", type=int, default=4)
    p.add_argument("--min-blob-area", type=int, default=15)
    p.add_argument("--min-line-ratio", type=float, default=0.005)
    p.add_argument("--max-ignore-ratio", type=float, default=0.7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.out)
    for sub in ("previews", "responses", "pseudo_labels", "overlays",
                "patches/images", "patches/labels", "patches/ignore",
                "manifests", "reports"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] reading {args.input}")
    img = read_tif(Path(args.input))
    abs_img = np.abs(img)

    save_preview_png(img, root / "previews" / "input_preview.png", 1.0, 99.0)
    save_preview_png(abs_img, root / "previews" / "abs_preview.png", 0.0, 99.5)

    sigmas = [2.0, 3.5, 5.5, 8.0]
    resp = compute_responses(img, sigmas)
    for name in ("sato", "black_score", "coherence", "gradient"):
        save_response_pair(resp[name], root / "responses" / f"response_{name}")

    pseudo = make_pseudo_labels(
        resp["sato"], resp["black_score"],
        sato_pos_pct=args.sato_pos_pct, sato_loose_pct=args.sato_loose_pct,
        black_pos_sigma=args.black_pos_sigma,
        black_loose_sigma=args.black_loose_sigma,
        pos_dilate=args.pos_dilate, edge_buffer=args.edge_buffer,
        min_blob_area=args.min_blob_area,
    )
    print(f"[INFO] pseudo label stats: {pseudo['stats']}")
    Image.fromarray(pseudo["positive"] * 255).save(root / "pseudo_labels" / "positive_mask.png")
    Image.fromarray(pseudo["ignore"] * 255).save(root / "pseudo_labels" / "ignore_mask.png")
    Image.fromarray(pseudo["train_label"]).save(root / "pseudo_labels" / "train_label.png")
    # Compat name expected by infer.py
    Image.fromarray(pseudo["train_label"]).save(root / "pseudo_labels" / "train_label_selected.png")

    rgb_base = img_to_rgb(img)
    save_rgb(overlay_mask(rgb_base, pseudo["positive"] > 0, (0.0, 1.0, 0.0)),
             root / "overlays" / "overlay_positive.png")
    save_rgb(overlay_mask(rgb_base, pseudo["ignore"] > 0, (1.0, 0.65, 0.0)),
             root / "overlays" / "overlay_ignore.png")
    tl_rgb = overlay_mask(rgb_base, pseudo["train_label"] == 1, (0.0, 1.0, 0.0))
    tl_rgb = overlay_mask(tl_rgb, pseudo["train_label"] == 255, (1.0, 1.0, 0.0), alpha=0.40)
    save_rgb(tl_rgb, root / "overlays" / "overlay_train_label.png")

    # Build channel stack
    raw_n = normalize01(img, 1.0, 99.0)
    abs_n = normalize01(abs_img, 0.0, 99.5)
    sato_n = normalize01(resp["sato"], 0.5, 99.5)
    # black_score is already in σ units; we clip and rescale to [0, 1]
    black_p99 = float(np.percentile(resp["black_score"], 99.5))
    black_n = np.clip(resp["black_score"] / max(black_p99, 1e-6), 0.0, 1.0).astype(np.float32)
    coh_n = normalize01(resp["coherence"], 0.5, 99.5)
    grad_n = normalize01(resp["gradient"], 0.5, 99.5)
    stacked = np.stack([raw_n, abs_n, sato_n, black_n, coh_n, grad_n], axis=0).astype(np.float32)

    rows = slice_patches(
        stacked, pseudo["train_label"],
        patch_size=args.patch_size, stride=args.stride,
        min_line_ratio=args.min_line_ratio,
        max_ignore_ratio=args.max_ignore_ratio,
        root=root, base_name=Path(args.input).stem,
    )
    write_manifest(rows, root / "manifests" / "train_manifest.csv")
    patch_contact_sheet(rows, root, root / "reports" / "patch_contact_sheet.png", max_n=48)

    summary = {
        "input": str(args.input),
        "image_shape": list(img.shape),
        "patch_size": args.patch_size, "stride": args.stride,
        "num_patches": len(rows),
        "channel_names": CHANNEL_NAMES,
        "pseudo_label_stats": pseudo["stats"],
        "label_values": {"background": 0, "line": 1, "ignore": 255},
        "black_score_norm_scale_p99_5": black_p99,
    }
    with open(root / "reports" / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[DONE] patches written: {len(rows)}")
    print(f"[DONE] dataset root: {root.resolve()}")


if __name__ == "__main__":
    main()
