"""Stage 1B inference: produce a full-image Kossel-line probability map.

Reads a checkpoint trained by train_line_mask_unet.py, rebuilds the input
channel stack from data.tif (+ cached response tifs when available), runs
sliding-window inference with Hann-weighted blending, and writes overlays
and contact sheets for human inspection.

Run:
    python infer_line_mask_unet.py --model line_stage1_train_v1/checkpoints/best.pt \
        --dataset line_stage1_dataset --input data.tif --out line_stage1_infer_v1

This script does NOT remove lines, does NOT inpaint background, and does NOT
do defect detection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import frangi, sato as sato_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_line_mask_unet import UNet  # noqa: E402


# ------------------------------ CLI ------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1B U-Net inference")
    p.add_argument("--model", required=True, help="Path to checkpoint .pt")
    p.add_argument("--dataset", required=True, help="Stage 1 dataset root (for cached responses & label compare)")
    p.add_argument("--input", required=True, help="Source .tif image")
    p.add_argument("--out", required=True, help="Output directory for inference artifacts")
    p.add_argument("--config", default=None, help="Override config.json path (default: <model>/../../config.json)")
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--device", default=None)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    return p.parse_args()


# ------------------------------ Helpers --------------------------------------


def normalize01(x: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
    lo = float(np.percentile(x, lo_pct))
    hi = float(np.percentile(x, hi_pct))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


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


def load_response(rdir: Path, name: str, smooth: np.ndarray | None = None, sigmas=None) -> np.ndarray:
    p = rdir / f"response_{name}.tif"
    if p.exists():
        return tifffile.imread(str(p)).astype(np.float32)
    if smooth is None:
        raise FileNotFoundError(f"missing {p} and no fallback smooth image provided")
    print(f"[INFO] {p.name} missing → recomputing")
    if name == "sato":
        return sato_filter(smooth, sigmas=sigmas, black_ridges=False).astype(np.float32)
    if name == "frangi":
        return frangi(smooth, sigmas=sigmas, black_ridges=False).astype(np.float32)
    if name == "coherence":
        return structure_coherence(smooth, 2.0, 8.0).astype(np.float32)
    if name == "gradient":
        return gradient_magnitude(smooth).astype(np.float32)
    raise ValueError(f"unknown response name: {name}")


def build_channel_stack(input_path: Path, dataset: Path, channel_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    img = tifffile.imread(str(input_path)).astype(np.float32)
    if img.ndim == 3:
        print(f"[WARN] {input_path} is 3D {img.shape}; using first frame.")
        img = img[0]
    abs_img = np.abs(img)
    smooth = ndi.gaussian_filter(abs_img, sigma=1.0)
    sigmas = [2.0, 3.5, 5.5, 8.0]
    rdir = dataset / "responses"

    cache: dict[str, np.ndarray] = {}

    def get(name: str) -> np.ndarray:
        if name in cache:
            return cache[name]
        if name == "raw":
            v = normalize01(img, 1.0, 99.0)
        elif name == "abs":
            v = normalize01(abs_img, 0.0, 99.5)
        elif name in ("sato", "frangi", "coherence", "gradient"):
            r = load_response(rdir, name, smooth, sigmas)
            v = normalize01(r, 0.5, 99.5)
        else:
            raise ValueError(f"Unknown channel name '{name}'")
        cache[name] = v
        return v

    chans = [get(n) for n in channel_names]
    stacked = np.stack(chans, axis=0).astype(np.float32)
    return img, stacked


def hann_window(size: int) -> np.ndarray:
    w = np.hanning(size).astype(np.float32) + 0.05
    return np.outer(w, w)


def positions(L: int, ps: int, st: int) -> list[int]:
    if L <= ps:
        return [0]
    ys = list(range(0, L - ps + 1, st))
    if ys[-1] != L - ps:
        ys.append(L - ps)
    return ys


@torch.no_grad()
def sliding_window_predict(
    model: torch.nn.Module,
    image_chw: np.ndarray,
    patch_size: int,
    stride: int,
    device: torch.device,
) -> np.ndarray:
    C, H, W = image_chw.shape
    if H < patch_size or W < patch_size:
        pad_h = max(0, patch_size - H)
        pad_w = max(0, patch_size - W)
        image_chw = np.pad(image_chw, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        Hp, Wp = H + pad_h, W + pad_w
    else:
        Hp, Wp = H, W

    ys = positions(Hp, patch_size, stride)
    xs = positions(Wp, patch_size, stride)
    win = hann_window(patch_size)

    out_prob = np.zeros((Hp, Wp), dtype=np.float32)
    weight = np.zeros((Hp, Wp), dtype=np.float32)

    model.eval()
    n_total = len(ys) * len(xs)
    done = 0
    for y in ys:
        for x in xs:
            patch = image_chw[:, y : y + patch_size, x : x + patch_size]
            t = torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0).to(device)
            logits = model(t)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
            out_prob[y : y + patch_size, x : x + patch_size] += probs * win
            weight[y : y + patch_size, x : x + patch_size] += win
            done += 1
            if done % max(1, n_total // 10) == 0:
                print(f"[INFO] sliding window {done}/{n_total}")
    fused = out_prob / np.maximum(weight, 1e-6)
    return fused[:H, :W]


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color, alpha: float = 0.55) -> np.ndarray:
    if not mask.any():
        return rgb
    out = rgb.copy()
    out[mask] = (1.0 - alpha) * rgb[mask] + alpha * np.asarray(color, dtype=np.float32)
    return out


def to_rgb(gray: np.ndarray) -> np.ndarray:
    g = np.clip(gray, 0, 1).astype(np.float32)
    return np.stack([g, g, g], axis=-1)


def save_uint8(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path)


def save_prob_png(prob: np.ndarray, path: Path) -> None:
    save_uint8((np.clip(prob, 0, 1) * 255).astype(np.uint8), path)


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    save_uint8((np.clip(rgb, 0, 1) * 255).astype(np.uint8), path)


# ------------------------------ Main -----------------------------------------


def load_checkpoint_and_config(model_path: Path, override_cfg: Path | None):
    ckpt = torch.load(str(model_path), map_location="cpu")
    cfg_path = override_cfg or (model_path.parent.parent / "config.json")
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}
    in_channels = ckpt.get("in_channels", cfg.get("in_channels", 6))
    base_channels = ckpt.get("base_channels", cfg.get("base_channels", 32))
    channel_names = ckpt.get("channel_names", cfg.get("channel_names", ["raw", "abs", "sato", "frangi", "coherence", "gradient"]))
    return ckpt, cfg, in_channels, base_channels, list(channel_names)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model)
    cfg_override = Path(args.config) if args.config else None
    ckpt, cfg, in_channels, base_channels, channel_names = load_checkpoint_and_config(model_path, cfg_override)
    print(f"[INFO] checkpoint epoch={ckpt.get('epoch', '?')}  channels={channel_names}")

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"[INFO] device = {device}")

    model = UNet(in_channels=in_channels, base=base_channels).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = Path(args.dataset)
    input_path = Path(args.input)
    raw_img, stack = build_channel_stack(input_path, dataset, channel_names)
    H, W = raw_img.shape
    print(f"[INFO] image shape: {raw_img.shape}, channel stack: {stack.shape}")

    print(f"[INFO] sliding window: ps={args.patch_size} stride={args.stride}")
    prob = sliding_window_predict(model, stack, args.patch_size, args.stride, device)
    prob = np.clip(prob, 0.0, 1.0).astype(np.float32)

    tifffile.imwrite(str(out_root / "line_prob.tif"), prob)
    save_prob_png(prob, out_root / "line_prob.png")

    raw_norm = normalize01(raw_img, 1.0, 99.0)
    base_rgb = to_rgb(raw_norm)

    threshold_results = {}
    for thr in args.thresholds:
        tag = f"thr{int(round(thr * 10)):02d}"
        mask = prob > thr
        save_uint8((mask.astype(np.uint8) * 255), out_root / f"line_mask_{tag}.png")
        rgb = overlay_mask(base_rgb, mask, (0.0, 1.0, 0.0), alpha=0.55)
        save_rgb(rgb, out_root / f"overlay_{tag}.png")
        threshold_results[tag] = {
            "threshold": thr,
            "mask_pixel_ratio": float(mask.mean()),
        }

    sel_label_path = dataset / "pseudo_labels" / "train_label_selected.png"
    sel_label = np.array(Image.open(sel_label_path)) if sel_label_path.exists() else None

    if sel_label is not None and sel_label.shape == raw_img.shape:
        # side-by-side: selected label overlay vs predicted overlay (thr=0.5)
        clear_lab = sel_label == 1
        ign_lab = sel_label == 255
        rgb_lab = overlay_mask(base_rgb, ign_lab, (1.0, 1.0, 0.0), alpha=0.40)
        rgb_lab = overlay_mask(rgb_lab, clear_lab, (0.0, 1.0, 0.0), alpha=0.55)

        pred_mask_05 = prob > 0.5
        rgb_pred = overlay_mask(base_rgb, pred_mask_05, (0.0, 1.0, 0.0), alpha=0.55)

        # difference: TP green, FP red, FN blue (vs selected clear label, ignoring 255)
        valid = sel_label != 255
        tp = pred_mask_05 & clear_lab
        fp = pred_mask_05 & (~clear_lab) & valid
        fn = (~pred_mask_05) & clear_lab
        rgb_diff = base_rgb.copy()
        rgb_diff = overlay_mask(rgb_diff, fn, (0.0, 0.5, 1.0), alpha=0.6)
        rgb_diff = overlay_mask(rgb_diff, fp, (1.0, 0.2, 0.2), alpha=0.6)
        rgb_diff = overlay_mask(rgb_diff, tp, (0.0, 1.0, 0.0), alpha=0.6)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(np.clip(rgb_lab, 0, 1)); axes[0].set_title("selected label"); axes[0].axis("off")
        axes[1].imshow(np.clip(rgb_pred, 0, 1)); axes[1].set_title("prediction (thr=0.5)"); axes[1].axis("off")
        axes[2].imshow(np.clip(rgb_diff, 0, 1)); axes[2].set_title("diff: TP green / FP red / FN blue"); axes[2].axis("off")
        fig.tight_layout()
        fig.savefig(out_root / "compare_selected_label_vs_prediction.png", dpi=120)
        plt.close(fig)
    else:
        if sel_label is None:
            print(f"[WARN] selected label {sel_label_path} not found; skipping compare image.")
        else:
            print(f"[WARN] selected label shape {sel_label.shape} mismatches image {raw_img.shape}.")

    summary = {
        "model_checkpoint": str(model_path),
        "input_image": str(input_path),
        "image_shape": list(raw_img.shape),
        "channel_names": channel_names,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "thresholds": threshold_results,
        "prob_min": float(prob.min()),
        "prob_max": float(prob.max()),
        "prob_mean": float(prob.mean()),
    }
    with open(out_root / "inference_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # contact sheet
    sato_resp_path = dataset / "responses" / "response_sato.tif"
    sato_disp = None
    if sato_resp_path.exists():
        sato_disp = normalize01(tifffile.imread(str(sato_resp_path)).astype(np.float32), 0.5, 99.5)

    panels = []
    panels.append((to_rgb(raw_norm), "input_preview"))
    if sato_disp is not None:
        panels.append((to_rgb(sato_disp), "response_sato"))
    if sel_label is not None and sel_label.shape == raw_img.shape:
        rgb_lab = overlay_mask(base_rgb, sel_label == 255, (1.0, 1.0, 0.0), alpha=0.40)
        rgb_lab = overlay_mask(rgb_lab, sel_label == 1, (0.0, 1.0, 0.0), alpha=0.55)
        panels.append((rgb_lab, "selected_label_overlay"))
    panels.append((prob, "line_prob"))
    for thr in args.thresholds:
        tag = f"thr{int(round(thr * 10)):02d}"
        m = prob > thr
        panels.append((overlay_mask(base_rgb, m, (0.0, 1.0, 0.0), 0.55), f"overlay_{tag}"))
    if sel_label is not None and sel_label.shape == raw_img.shape:
        valid = sel_label != 255
        clear_lab = sel_label == 1
        pred_mask_05 = prob > 0.5
        tp = pred_mask_05 & clear_lab
        fp = pred_mask_05 & (~clear_lab) & valid
        fn = (~pred_mask_05) & clear_lab
        rgb_diff = base_rgb.copy()
        rgb_diff = overlay_mask(rgb_diff, fn, (0.0, 0.5, 1.0), 0.6)
        rgb_diff = overlay_mask(rgb_diff, fp, (1.0, 0.2, 0.2), 0.6)
        rgb_diff = overlay_mask(rgb_diff, tp, (0.0, 1.0, 0.0), 0.6)
        panels.append((rgb_diff, "diff TP/FP/FN"))

    n = len(panels)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i >= n:
            continue
        img_panel, title = panels[i]
        if img_panel.ndim == 2:
            ax.imshow(img_panel, cmap="magma" if title == "line_prob" else "gray", vmin=0, vmax=1)
        else:
            ax.imshow(np.clip(img_panel, 0, 1))
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_root / "inference_contact_sheet.png", dpi=110)
    plt.close(fig)

    print()
    print("Inference done.")
    print("Please check:")
    print(f"  {(out_root / 'inference_contact_sheet.png').as_posix()}")
    for thr in args.thresholds:
        tag = f"thr{int(round(thr * 10)):02d}"
        print(f"  {(out_root / f'overlay_{tag}.png').as_posix()}")


if __name__ == "__main__":
    main()
