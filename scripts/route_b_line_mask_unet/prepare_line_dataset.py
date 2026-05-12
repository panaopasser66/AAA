"""Stage 1: prepare Kossel line dataset from X-ray TIF images.

Outputs into --out directory:
  previews/, responses/, pseudo_labels/, overlays/,
  patches/{images,labels,ignore}/, manifests/, reports/

This script does NOT train any model and does NOT run defect detection.
It only builds the data structures needed for later LineMaskNet training.

Run:
    python prepare_line_dataset.py --input data.tif --out line_stage1_dataset
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
from skimage.filters import frangi, meijering, sato
from skimage.morphology import binary_dilation, disk, remove_small_objects


# ----------------------------- CLI & directories -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1 line-dataset preparation")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", type=str, help="Path to a single .tif file.")
    g.add_argument("--input_dir", type=str, help="Directory containing .tif files.")
    p.add_argument("--out", type=str, required=True, help="Output dataset directory.")
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--min-line-ratio", type=float, default=0.002)
    p.add_argument("--max-ignore-ratio", type=float, default=0.7)
    p.add_argument("--clear-line-percentile", type=float, default=97.0)
    p.add_argument("--loose-line-percentile", type=float, default=92.0)
    p.add_argument("--min-blob-area", type=int, default=15)
    p.add_argument("--max-patches-contact", type=int, default=64)
    p.add_argument(
        "--label_sweep",
        action="store_true",
        help="Skip the v0 pipeline and run sweep_pseudo_labels.run_sweep against --out.",
    )
    return p.parse_args()


def make_dirs(root: Path) -> None:
    for sub in (
        "previews",
        "responses",
        "pseudo_labels",
        "overlays",
        "patches/images",
        "patches/labels",
        "patches/ignore",
        "manifests",
        "reports",
    ):
        (root / sub).mkdir(parents=True, exist_ok=True)


def collect_inputs(args: argparse.Namespace) -> list[Path]:
    if args.input:
        return [Path(args.input)]
    d = Path(args.input_dir)
    return sorted(p for p in d.iterdir() if p.suffix.lower() in {".tif", ".tiff"})


# ----------------------------- I/O helpers -----------------------------------


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
    Image.fromarray((norm * 255).astype(np.uint8)).save(out_path)


def save_response_pair(resp: np.ndarray, base: Path) -> None:
    """Save raw float .tif plus an 8-bit preview .png."""
    tifffile.imwrite(str(base.with_suffix(".tif")), resp.astype(np.float32))
    save_preview_png(resp, base.with_suffix(".png"), 0.5, 99.5)


def normalize01(x: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    lo = float(np.percentile(x, lo_pct))
    hi = float(np.percentile(x, hi_pct))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# ----------------------------- Response channels -----------------------------


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


def fuse_responses(*responses: np.ndarray) -> np.ndarray:
    weights = [0.4, 0.2, 0.2, 0.1, 0.1]
    fused = np.zeros_like(responses[0], dtype=np.float32)
    for w, r in zip(weights, responses):
        fused = fused + w * normalize01(r)
    return fused


def compute_responses(img: np.ndarray, sigmas: list[float]) -> dict[str, np.ndarray]:
    abs_img = np.abs(img)
    smooth = ndi.gaussian_filter(abs_img, sigma=1.0)

    print(f"[INFO] Sato (sigmas={sigmas}) ...")
    r_sato = sato(smooth, sigmas=sigmas, black_ridges=False).astype(np.float32)
    print(f"[INFO] Frangi (sigmas={sigmas}) ...")
    r_frangi = frangi(smooth, sigmas=sigmas, black_ridges=False).astype(np.float32)
    print(f"[INFO] Meijering (sigmas={sigmas}) ...")
    r_meijering = meijering(smooth, sigmas=sigmas, black_ridges=False).astype(np.float32)

    print("[INFO] structure tensor coherence ...")
    coh = structure_coherence(smooth, sigma_d=2.0, sigma_s=8.0).astype(np.float32)

    print("[INFO] gradient magnitude ...")
    gx = ndi.gaussian_filter(smooth, sigma=1.5, order=(0, 1))
    gy = ndi.gaussian_filter(smooth, sigma=1.5, order=(1, 0))
    grad = np.hypot(gx, gy).astype(np.float32)

    fused = fuse_responses(r_sato, r_frangi, r_meijering, coh, grad)

    return {
        "sato": r_sato,
        "frangi": r_frangi,
        "meijering": r_meijering,
        "coherence": coh,
        "gradient": grad,
        "fused": fused,
    }


# ----------------------------- Pseudo labels ---------------------------------


def make_pseudo_labels(sato_resp: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    sato_norm = normalize01(sato_resp)

    thr_clear = float(np.percentile(sato_norm, args.clear_line_percentile))
    thr_loose = float(np.percentile(sato_norm, args.loose_line_percentile))
    print(f"[INFO] sato thresholds: clear={thr_clear:.4f}, loose={thr_loose:.4f}")

    clear = sato_norm > thr_clear
    loose = sato_norm > thr_loose

    clear = remove_small_objects(clear, min_size=args.min_blob_area)
    loose = remove_small_objects(loose, min_size=max(args.min_blob_area // 2, 4))

    # tighten clear lines slightly so adjacent pixels aren't labeled background
    clear_d = binary_dilation(clear, footprint=disk(1))

    # ignore = uncertain band: loose-but-not-clear, plus a buffer around clear edges
    ignore = loose & (~clear_d)
    ignore = binary_dilation(ignore, footprint=disk(2))
    edge_band = binary_dilation(clear_d, footprint=disk(2)) & (~clear_d)
    ignore = ignore | edge_band
    ignore = ignore & (~clear_d)

    train_label = np.zeros(clear_d.shape, dtype=np.uint8)
    train_label[clear_d] = 1
    train_label[ignore & (~clear_d)] = 255

    return {
        "clear": clear_d.astype(np.uint8),
        "loose": loose.astype(np.uint8),
        "ignore": ignore.astype(np.uint8),
        "train_label": train_label,
    }


# ----------------------------- Overlays --------------------------------------


def img_to_rgb(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    lo = float(np.percentile(img, lo_pct))
    hi = float(np.percentile(img, hi_pct))
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return np.stack([norm, norm, norm], axis=-1).astype(np.float32)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], alpha: float = 0.55) -> np.ndarray:
    out = rgb.copy()
    m = mask.astype(bool)
    if m.any():
        out[m] = (1.0 - alpha) * rgb[m] + alpha * np.array(color, dtype=np.float32)
    return out


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)


# ----------------------------- Patch slicing ---------------------------------


def stack_channels(
    raw_n: np.ndarray,
    abs_n: np.ndarray,
    sato_n: np.ndarray,
    frangi_n: np.ndarray,
    coh_n: np.ndarray,
    grad_n: np.ndarray,
) -> np.ndarray:
    return np.stack([raw_n, abs_n, sato_n, frangi_n, coh_n, grad_n], axis=0).astype(np.float32)


def slice_patches(
    stacked: np.ndarray,
    train_label: np.ndarray,
    args: argparse.Namespace,
    root: Path,
    base_name: str,
) -> list[dict]:
    _, H, W = stacked.shape
    ps = args.patch_size
    st = args.stride
    if ps > H or ps > W:
        print(f"[WARN] patch size {ps} exceeds image {H}x{W}; no patches produced.")
        return []

    rows: list[dict] = []
    pid = 0
    channel_names = np.array(["raw", "abs", "sato", "frangi", "coherence", "gradient"])

    ys = list(range(0, H - ps + 1, st))
    xs = list(range(0, W - ps + 1, st))
    if ys[-1] != H - ps:
        ys.append(H - ps)
    if xs[-1] != W - ps:
        xs.append(W - ps)

    for y in ys:
        for x in xs:
            patch = stacked[:, y : y + ps, x : x + ps]
            label = train_label[y : y + ps, x : x + ps]
            line_ratio = float((label == 1).mean())
            ignore_ratio = float((label == 255).mean())
            if line_ratio < args.min_line_ratio:
                continue
            if ignore_ratio > args.max_ignore_ratio:
                continue

            pid += 1
            patch_id = f"{base_name}_{pid:06d}"
            img_path = root / "patches" / "images" / f"img_{patch_id}.npz"
            label_path = root / "patches" / "labels" / f"label_{patch_id}.png"
            ignore_path = root / "patches" / "ignore" / f"ignore_{patch_id}.png"

            np.savez_compressed(img_path, image=patch, channel_names=channel_names)
            Image.fromarray(label).save(label_path)
            Image.fromarray(((label == 255).astype(np.uint8) * 255)).save(ignore_path)

            rows.append(
                {
                    "patch_id": patch_id,
                    "image_path": str(img_path.relative_to(root)).replace("\\", "/"),
                    "label_path": str(label_path.relative_to(root)).replace("\\", "/"),
                    "ignore_path": str(ignore_path.relative_to(root)).replace("\\", "/"),
                    "x0": x,
                    "y0": y,
                    "w": ps,
                    "h": ps,
                    "line_ratio": line_ratio,
                    "ignore_ratio": ignore_ratio,
                }
            )
    return rows


def write_manifest(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "patch_id",
        "image_path",
        "label_path",
        "ignore_path",
        "x0",
        "y0",
        "w",
        "h",
        "line_ratio",
        "ignore_ratio",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def patch_contact_sheet(rows: list[dict], root: Path, out_path: Path, max_n: int = 64) -> None:
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
        rgb = overlay_mask(rgb, label == 1, (0.0, 1.0, 0.0), alpha=0.45)
        rgb = overlay_mask(rgb, label == 255, (1.0, 1.0, 0.0), alpha=0.45)
        ax.imshow(np.clip(rgb, 0, 1))
        ax.set_title(f"{r['patch_id']}\nL={r['line_ratio']:.2%}", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def hard_cases_contact_sheet(rows: list[dict], root: Path, out_path: Path, max_n: int = 64) -> None:
    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda r: r["ignore_ratio"], reverse=True)[:max_n]
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
        rgb = overlay_mask(rgb, label == 1, (0.0, 1.0, 0.0), alpha=0.45)
        rgb = overlay_mask(rgb, label == 255, (1.0, 1.0, 0.0), alpha=0.45)
        ax.imshow(np.clip(rgb, 0, 1))
        ax.set_title(f"{r['patch_id']}\nI={r['ignore_ratio']:.2%}", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ----------------------------- Per-image driver ------------------------------


def process_image(input_path: Path, root: Path, args: argparse.Namespace) -> dict:
    img = read_tif(input_path)
    stats = {
        "filename": input_path.name,
        "shape": list(img.shape),
        "dtype": str(img.dtype),
        "min": float(np.min(img)),
        "max": float(np.max(img)),
        "p1": float(np.percentile(img, 1)),
        "p50": float(np.percentile(img, 50)),
        "p99": float(np.percentile(img, 99)),
    }
    with open(root / "reports" / "input_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[INFO] input stats: {stats}")

    save_preview_png(img, root / "previews" / "input_preview.png", 1.0, 99.0)
    save_preview_png(np.abs(img), root / "previews" / "abs_preview.png", 0.0, 99.5)

    sigmas = [2.0, 3.5, 5.5, 8.0]
    resp = compute_responses(img, sigmas)
    for name in ("sato", "frangi", "meijering", "coherence", "gradient", "fused"):
        save_response_pair(resp[name], root / "responses" / f"response_{name}")

    pseudo = make_pseudo_labels(resp["sato"], args)
    Image.fromarray(pseudo["clear"] * 255).save(root / "pseudo_labels" / "line_mask_clear.png")
    Image.fromarray(pseudo["loose"] * 255).save(root / "pseudo_labels" / "line_mask_loose.png")
    Image.fromarray(pseudo["ignore"] * 255).save(root / "pseudo_labels" / "ignore_mask.png")
    Image.fromarray(pseudo["train_label"]).save(root / "pseudo_labels" / "train_label.png")

    rgb_base = img_to_rgb(img)
    save_rgb(
        overlay_mask(rgb_base, pseudo["clear"] > 0, (0.0, 1.0, 0.0)),
        root / "overlays" / "overlay_clear_line.png",
    )
    save_rgb(
        overlay_mask(rgb_base, pseudo["loose"] > 0, (0.0, 1.0, 0.0)),
        root / "overlays" / "overlay_loose_line.png",
    )
    save_rgb(
        overlay_mask(rgb_base, pseudo["ignore"] > 0, (1.0, 0.65, 0.0)),
        root / "overlays" / "overlay_ignore.png",
    )
    tl = overlay_mask(rgb_base, pseudo["train_label"] == 1, (0.0, 1.0, 0.0))
    tl = overlay_mask(tl, pseudo["train_label"] == 255, (1.0, 1.0, 0.0))
    save_rgb(tl, root / "overlays" / "overlay_train_label.png")

    raw_n = normalize01(img, 1.0, 99.0)
    abs_n = normalize01(np.abs(img), 0.0, 99.5)
    sato_n = normalize01(resp["sato"])
    frangi_n = normalize01(resp["frangi"])
    coh_n = normalize01(resp["coherence"])
    grad_n = normalize01(resp["gradient"])
    stacked = stack_channels(raw_n, abs_n, sato_n, frangi_n, coh_n, grad_n)

    rows = slice_patches(stacked, pseudo["train_label"], args, root, input_path.stem)
    return {"rows": rows, "train_label": pseudo["train_label"]}


# ----------------------------- Main ------------------------------------------


def main() -> None:
    args = parse_args()
    root = Path(args.out)

    if args.label_sweep:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from sweep_pseudo_labels import run_sweep

        input_path = Path(args.input) if args.input else None
        run_sweep(root, input_path)
        return

    make_dirs(root)

    inputs = collect_inputs(args)
    if not inputs:
        print("[ERROR] no input .tif files found.", file=sys.stderr)
        sys.exit(1)

    all_rows: list[dict] = []
    last_label: np.ndarray | None = None
    for inp in inputs:
        print(f"[INFO] processing {inp}")
        result = process_image(inp, root, args)
        all_rows.extend(result["rows"])
        last_label = result["train_label"]

    write_manifest(all_rows, root / "manifests" / "train_manifest.csv")
    patch_contact_sheet(
        all_rows,
        root,
        root / "reports" / "patch_contact_sheet.png",
        max_n=args.max_patches_contact,
    )
    hard_cases_contact_sheet(
        all_rows,
        root,
        root / "reports" / "hard_cases_contact_sheet.png",
        max_n=args.max_patches_contact,
    )

    if last_label is not None:
        line_pixel_ratio = float((last_label == 1).mean())
        ignore_pixel_ratio = float((last_label == 255).mean())
    else:
        line_pixel_ratio = ignore_pixel_ratio = 0.0

    summary = {
        "num_input_images": len(inputs),
        "patch_size": args.patch_size,
        "stride": args.stride,
        "num_patches": len(all_rows),
        "line_pixel_ratio": line_pixel_ratio,
        "ignore_pixel_ratio": ignore_pixel_ratio,
        "response_source": "sato",
        "label_values": {"background": 0, "line": 1, "ignore": 255},
    }
    with open(root / "reports" / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] patches written: {len(all_rows)}")
    print(f"[DONE] dataset root: {root.resolve()}")


if __name__ == "__main__":
    main()
