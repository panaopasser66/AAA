"""Stage 1B: train a lightweight U-Net to predict Kossel-line probability.

Reads patches + labels emitted by prepare_line_dataset.py / regenerate_patches.py
and trains a binary segmentation head with masked BCE + masked Dice loss.

Run:
    python train_line_mask_unet.py --dataset line_stage1_dataset --out line_stage1_train_v1 \
        --epochs 120 --batch-size 4 --lr 1e-3

This script does NOT do background inpainting and does NOT do defect detection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ------------------------------ CLI ------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1B U-Net training")
    p.add_argument("--dataset", required=True, help="Stage 1 dataset root")
    p.add_argument("--out", required=True, help="Training output directory")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--preview-every", type=int, default=10)
    p.add_argument("--dice-weight", type=float, default=0.5)
    p.add_argument("--pos-weight-min", type=float, default=3.0)
    p.add_argument("--pos-weight-max", type=float, default=30.0)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--device", default=None, help="cuda / cpu (auto-detect if omitted)")
    return p.parse_args()


# ------------------------------ Dataset --------------------------------------


class PatchDataset(Dataset):
    """Loads (multi-channel image, label) pairs from a manifest subset."""

    def __init__(self, root: Path, rows: list[dict], augment: bool = False):
        self.root = Path(root)
        self.rows = rows
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, npz_path: Path) -> tuple[np.ndarray, list[str]]:
        with np.load(npz_path, allow_pickle=False) as d:
            keys = list(d.keys())
            if "image" in keys:
                arr = d["image"]
                names = list(d["channel_names"]) if "channel_names" in keys else []
            else:
                # Stack named channels in deterministic order
                preferred = ["raw", "abs", "sato", "frangi", "coherence", "gradient"]
                names = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]
                arr = np.stack([np.asarray(d[k]) for k in names], axis=0)
        arr = np.asarray(arr).astype(np.float32, copy=False)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(f"Unexpected image ndim {arr.ndim} in {npz_path}")
        if names:
            n = len(names)
            if arr.shape[0] != n and arr.shape[-1] == n:
                arr = np.transpose(arr, (2, 0, 1))
        else:
            # heuristic: small first dim → CHW
            if arr.shape[0] > 16 and arr.shape[-1] <= 16:
                arr = np.transpose(arr, (2, 0, 1))
            names = [f"ch{i}" for i in range(arr.shape[0])]
        return arr, [str(n) for n in names]

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image, names = self._load_image(self.root / row["image_path"])
        label = np.array(Image.open(self.root / row["label_path"]))
        if label.ndim != 2:
            raise ValueError(f"Unexpected label ndim {label.ndim} in {row['label_path']}")
        label = label.astype(np.uint8, copy=False)

        if self.augment:
            image, label = self._augment(image, label)

        image_t = torch.from_numpy(image.copy())
        label_t = torch.from_numpy(label.copy())
        target = (label_t == 1).float().unsqueeze(0)
        valid = (label_t != 255).float().unsqueeze(0)
        return {
            "image": image_t,
            "label": label_t,
            "target": target,
            "valid": valid,
            "patch_id": row["patch_id"],
            "channel_names": names,
        }

    @staticmethod
    def _augment(image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image = image[:, :, ::-1]
            label = label[:, ::-1]
        if random.random() < 0.5:
            image = image[:, ::-1, :]
            label = label[::-1, :]
        k = random.randint(0, 3)
        if k:
            image = np.rot90(image, k, axes=(1, 2))
            label = np.rot90(label, k)
        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)
        if random.random() < 0.5:
            image = image + np.random.randn(*image.shape).astype(np.float32) * 0.01
        if random.random() < 0.5:
            scale = 1.0 + (random.random() - 0.5) * 0.1
            shift = (random.random() - 0.5) * 0.05
            image = image * scale + shift
        return image, label


# ------------------------------ Model ----------------------------------------


def _gn(c: int) -> nn.GroupNorm:
    groups = max(1, min(8, c))
    while c % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, c)


class DoubleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            _gn(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            _gn(out_c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """4-level lightweight U-Net (32→64→128→256)."""

    def __init__(self, in_channels: int, base: int = 32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.enc1 = DoubleConv(in_channels, c[0])
        self.enc2 = DoubleConv(c[0], c[1])
        self.enc3 = DoubleConv(c[1], c[2])
        self.enc4 = DoubleConv(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, 2)
        self.dec3 = DoubleConv(c[3], c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, 2)
        self.dec2 = DoubleConv(c[2], c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, 2)
        self.dec1 = DoubleConv(c[1], c[0])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ------------------------------ Loss + metrics -------------------------------


def masked_bce_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    pos_weight: float,
    dice_weight: float = 0.5,
) -> tuple[torch.Tensor, float, float]:
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    bce_per_pixel = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none")
    n_valid = valid.sum().clamp(min=1.0)
    bce = (bce_per_pixel * valid).sum() / n_valid

    probs = torch.sigmoid(logits) * valid
    tgt = target * valid
    intersection = (probs * tgt).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1.0) / (union + 1.0)
    dice_loss = (1.0 - dice).mean()

    return bce + dice_weight * dice_loss, float(bce.item()), float(dice_loss.item())


@torch.no_grad()
def batch_metrics(probs: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, thr: float = 0.5):
    pred = (probs > thr).float() * valid
    tgt = target * valid
    tp = (pred * tgt).sum().item()
    fp = (pred * (1 - tgt)).sum().item() - (pred * (1 - valid)).sum().item()
    # safer:
    fp = (pred * (1 - tgt) * valid).sum().item()
    fn = ((1 - pred) * tgt * valid).sum().item()
    return tp, fp, fn


def precision_recall_f1_iou(tp: float, fp: float, fn: float):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return p, r, f1, iou


# ------------------------------ Helpers --------------------------------------


def read_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def split_train_val(rows: list[dict], val_fraction: float, seed: int):
    idxs = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    n_val = max(1, int(round(len(rows) * val_fraction)))
    val = sorted(idxs[:n_val])
    train = sorted(idxs[n_val:])
    return [rows[i] for i in train], [rows[i] for i in val]


def estimate_pos_weight(root: Path, rows: list[dict], lo: float, hi: float) -> float:
    pos = 0
    neg = 0
    for r in rows:
        l = np.array(Image.open(root / r["label_path"]))
        pos += int((l == 1).sum())
        neg += int((l == 0).sum())
    pw = neg / max(pos, 1)
    return float(np.clip(pw, lo, hi))


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color, alpha: float = 0.55) -> np.ndarray:
    if not mask.any():
        return rgb
    out = rgb.copy()
    out[mask] = (1.0 - alpha) * rgb[mask] + alpha * np.asarray(color, dtype=np.float32)
    return out


def to_rgb(gray: np.ndarray) -> np.ndarray:
    g = np.clip(gray, 0, 1).astype(np.float32)
    return np.stack([g, g, g], axis=-1)


def make_val_preview(
    model: nn.Module,
    val_dataset: PatchDataset,
    fixed_indices: list[int],
    channel_names: list[str],
    out_path: Path,
    device: torch.device,
) -> None:
    raw_idx = channel_names.index("raw") if "raw" in channel_names else 0
    n = len(fixed_indices)
    fig, axes = plt.subplots(n, 4, figsize=(4 * 3.0, n * 3.0))
    axes = np.atleast_2d(axes)
    model.eval()
    for r, idx in enumerate(fixed_indices):
        s = val_dataset[idx]
        img = s["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        raw = s["image"][raw_idx].cpu().numpy()
        label = s["label"].cpu().numpy()

        rgb_raw = to_rgb(raw)
        clear = label == 1
        ign = label == 255
        rgb_lab = overlay_mask(rgb_raw, ign, (1.0, 1.0, 0.0), alpha=0.45)
        rgb_lab = overlay_mask(rgb_lab, clear, (0.0, 1.0, 0.0), alpha=0.55)

        pred_mask = prob > 0.5
        rgb_pred = overlay_mask(rgb_raw, ign, (1.0, 1.0, 0.0), alpha=0.30)
        rgb_pred = overlay_mask(rgb_pred, pred_mask, (0.0, 1.0, 0.0), alpha=0.55)

        axes[r, 0].imshow(np.clip(rgb_raw, 0, 1))
        axes[r, 1].imshow(np.clip(rgb_lab, 0, 1))
        axes[r, 2].imshow(prob, cmap="magma", vmin=0, vmax=1)
        axes[r, 3].imshow(np.clip(rgb_pred, 0, 1))

        for c in range(4):
            axes[r, c].axis("off")
        if r == 0:
            axes[r, 0].set_title("raw", fontsize=10)
            axes[r, 1].set_title("selected label", fontsize=10)
            axes[r, 2].set_title("prob", fontsize=10)
            axes[r, 3].set_title("pred>0.5", fontsize=10)
        axes[r, 0].set_ylabel(s["patch_id"], fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_curves(metrics_rows: list[dict], curves_dir: Path) -> None:
    if not metrics_rows:
        return
    epochs = [int(r["epoch"]) for r in metrics_rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [float(r["train_loss"]) for r in metrics_rows], label="train_loss")
    ax.plot(epochs, [float(r["val_loss"]) for r in metrics_rows], label="val_loss")
    ax.plot(epochs, [float(r["val_bce"]) for r in metrics_rows], label="val_bce", alpha=0.6)
    ax.plot(epochs, [float(r["val_dice_loss"]) for r in metrics_rows], label="val_dice_loss", alpha=0.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(curves_dir / "loss_curve.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [float(r["val_precision"]) for r in metrics_rows], label="precision")
    ax.plot(epochs, [float(r["val_recall"]) for r in metrics_rows], label="recall")
    ax.plot(epochs, [float(r["val_f1"]) for r in metrics_rows], label="f1")
    ax.plot(epochs, [float(r["val_iou"]) for r in metrics_rows], label="iou")
    ax.set_xlabel("epoch")
    ax.set_ylabel("metric")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(curves_dir / "metric_curve.png", dpi=110)
    plt.close(fig)


# ------------------------------ Train loop -----------------------------------


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_root = Path(args.dataset)
    out_root = Path(args.out)
    for sub in ("checkpoints", "previews", "logs", "curves", "splits"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(dataset_root / "manifests" / "train_manifest.csv")
    if not manifest:
        raise SystemExit("[ERROR] empty manifest; run prepare_line_dataset.py first.")

    train_rows, val_rows = split_train_val(manifest, args.val_fraction, args.seed)
    print(f"[INFO] manifest: {len(manifest)} patches → train {len(train_rows)}, val {len(val_rows)}")

    (out_root / "splits" / "train_ids.txt").write_text(
        "\n".join(r["patch_id"] for r in train_rows), encoding="utf-8"
    )
    (out_root / "splits" / "val_ids.txt").write_text(
        "\n".join(r["patch_id"] for r in val_rows), encoding="utf-8"
    )

    train_ds = PatchDataset(dataset_root, train_rows, augment=True)
    val_ds = PatchDataset(dataset_root, val_rows, augment=False)

    sample = train_ds[0]
    in_channels = sample["image"].shape[0]
    channel_names = sample["channel_names"]
    print(f"[INFO] in_channels={in_channels}, channel_names={channel_names}")

    pos_weight = estimate_pos_weight(dataset_root, train_rows, args.pos_weight_min, args.pos_weight_max)
    print(f"[INFO] pos_weight (clamped) = {pos_weight:.3f}")

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"[INFO] device = {device}")

    model = UNet(in_channels=in_channels, base=args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] U-Net params = {n_params/1e6:.2f}M")

    config = {
        "in_channels": in_channels,
        "channel_names": channel_names,
        "base_channels": args.base_channels,
        "patch_size": int(sample["image"].shape[-1]),
        "pos_weight": pos_weight,
        "dice_weight": args.dice_weight,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "label_values": {"background": 0, "line": 1, "ignore": 255},
        "selected_label": "pseudo_labels/train_label_selected.png",
    }
    with open(out_root / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, min(args.batch_size, len(val_ds))),
        shuffle=False,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.01
    )

    fixed_val_indices = list(range(min(6, len(val_ds))))

    metrics_path = out_root / "metrics.csv"
    metric_fields = [
        "epoch", "train_loss", "val_loss", "val_bce", "val_dice_loss",
        "val_precision", "val_recall", "val_f1", "val_iou", "learning_rate",
    ]
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=metric_fields).writeheader()

    metrics_rows: list[dict] = []
    best_f1 = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            image = batch["image"].to(device)
            target = batch["target"].to(device)
            valid = batch["valid"].to(device)
            optimizer.zero_grad()
            logits = model(image)
            loss, _, _ = masked_bce_dice(logits, target, valid, pos_weight, args.dice_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            bs = image.shape[0]
            train_loss_sum += loss.item() * bs
            train_count += bs
        train_loss = train_loss_sum / max(train_count, 1)

        # validation
        model.eval()
        val_loss_sum = 0.0
        val_bce_sum = 0.0
        val_dice_sum = 0.0
        val_count = 0
        tp_total = fp_total = fn_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device)
                target = batch["target"].to(device)
                valid = batch["valid"].to(device)
                logits = model(image)
                loss, bce_v, dice_v = masked_bce_dice(logits, target, valid, pos_weight, args.dice_weight)
                bs = image.shape[0]
                val_loss_sum += loss.item() * bs
                val_bce_sum += bce_v * bs
                val_dice_sum += dice_v * bs
                val_count += bs
                probs = torch.sigmoid(logits)
                tp, fp, fn = batch_metrics(probs, target, valid, thr=0.5)
                tp_total += tp
                fp_total += fp
                fn_total += fn
        val_loss = val_loss_sum / max(val_count, 1)
        val_bce = val_bce_sum / max(val_count, 1)
        val_dice_loss = val_dice_sum / max(val_count, 1)
        prec, rec, f1, iou = precision_recall_f1_iou(tp_total, fp_total, fn_total)

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_bce": f"{val_bce:.6f}",
            "val_dice_loss": f"{val_dice_loss:.6f}",
            "val_precision": f"{prec:.4f}",
            "val_recall": f"{rec:.4f}",
            "val_f1": f"{f1:.4f}",
            "val_iou": f"{iou:.4f}",
            "learning_rate": f"{lr_now:.6f}",
        }
        metrics_rows.append(row)
        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=metric_fields).writerow(row)

        msg = (
            f"epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"P={prec:.3f} R={rec:.3f} F1={f1:.3f} IoU={iou:.3f}  lr={lr_now:.2e}"
        )
        print(msg)

        ckpt = {
            "model_state": model.state_dict(),
            "in_channels": in_channels,
            "base_channels": args.base_channels,
            "channel_names": channel_names,
            "pos_weight": pos_weight,
            "epoch": epoch,
        }
        torch.save(ckpt, out_root / "checkpoints" / "last.pt")
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            torch.save(ckpt, out_root / "checkpoints" / "best.pt")

        if (epoch % args.preview_every == 0 or epoch == args.epochs) and fixed_val_indices:
            preview_path = out_root / "previews" / f"epoch_{epoch:03d}_val_predictions.png"
            try:
                make_val_preview(model, val_ds, fixed_val_indices, channel_names, preview_path, device)
            except Exception as e:
                print(f"[WARN] preview failed at epoch {epoch}: {e}")

    plot_curves(metrics_rows, out_root / "curves")

    if fixed_val_indices:
        try:
            make_val_preview(
                model, val_ds, fixed_val_indices, channel_names,
                out_root / "previews" / "final_val_predictions.png", device,
            )
        except Exception as e:
            print(f"[WARN] final preview failed: {e}")

    print()
    print("Training done.")
    print(f"  best val_f1 = {best_f1:.4f} at epoch {best_epoch}")
    print("Please check:")
    print(f"  {(out_root / 'curves' / 'loss_curve.png').as_posix()}")
    print(f"  {(out_root / 'previews' / 'final_val_predictions.png').as_posix()}")
    print(f"  {(out_root / 'checkpoints' / 'best.pt').as_posix()}")


if __name__ == "__main__":
    main()
