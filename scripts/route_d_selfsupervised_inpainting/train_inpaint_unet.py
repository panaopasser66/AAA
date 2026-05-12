"""Stage 2: train a self-supervised inpainting U-Net.

Random holes are sampled only inside the valid region (NOT line_mask_F AND
NOT protect_mask). The loss is computed inside the holes (+ context band +
gradient term).

V2 changes:
  - holes include line-like patterns (long lines, crossings, stars, fine
    short lines) so the model learns to fill structures that look like
    Kossel lines;
  - forbidden region uses line_mask_75 by default (so weak lines stay out
    of the training target);
  - --config v2a (drop sato from inputs) or v2b (keep sato).

Run:
  python scripts/train_inpaint_unet.py \
      --prepared outputs/prepare \
      --out outputs/train_inpaint_v2b --config v2b \
      --epochs 200 --batch-size 4 --lr 1e-3
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Channel names exposed by prepare_project2_data.py.
# ---------------------------------------------------------------------------
PREPARE_CHANNELS = [
    "raw_norm",
    "abs_norm",
    "sato_norm",
    "frangi_norm",
    "coherence_norm",
    "gradient_norm",
    "line_mask_98",
    "line_mask_95",
    "line_mask_90",
    "line_mask_85",
    "line_mask_80",
    "line_mask_75",
    "protect_mask",
]

# Per-config network input feature lists. masked_raw_norm and hole_mask are
# always first; the rest comes from PREPARE_CHANNELS.
INPUT_FEATURES_BY_CONFIG = {
    # v1 (legacy): keep the same as Stage2B v1 model.
    "v1": [
        "masked_raw_norm", "hole_mask",
        "abs_norm", "sato_norm", "gradient_norm", "coherence_norm",
    ],
    # v2a: ablation without sato input
    "v2a": [
        "masked_raw_norm", "hole_mask",
        "abs_norm", "gradient_norm", "coherence_norm",
    ],
    # v2b: with sato input (recommended default for v2)
    "v2b": [
        "masked_raw_norm", "hole_mask",
        "abs_norm", "sato_norm", "gradient_norm", "coherence_norm",
    ],
}


# ---------------------------------------------------------------------------
# Random hole generators (numpy, work on shape (H, W) bool)
# ---------------------------------------------------------------------------

def _draw_thick_line(m: np.ndarray, y0, x0, y1, x1, thickness):
    """Stamp a thick segment onto a binary mask (no anti-alias)."""
    H, W = m.shape
    n = max(abs(int(y1) - int(y0)), abs(int(x1) - int(x0))) + 1
    n = max(2, n)
    ys = np.linspace(y0, y1, n * 3).astype(np.int32)
    xs = np.linspace(x0, x1, n * 3).astype(np.int32)
    valid = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    ys = ys[valid]; xs = xs[valid]
    if len(ys) == 0:
        return m
    tmp = np.zeros_like(m, dtype=bool)
    tmp[ys, xs] = True
    if thickness > 1:
        from skimage.morphology import binary_dilation, disk
        tmp = binary_dilation(tmp, footprint=disk(max(1, int(thickness // 2))))
    return m | tmp


def _sample_long_line(H, W, rng):
    """A single long thin line crossing the patch at a random angle."""
    m = np.zeros((H, W), dtype=bool)
    angle = rng.uniform(0, math.pi)
    length = rng.randint(int(min(H, W) * 0.4), int(max(H, W) * 1.5))
    thickness = rng.randint(2, 13)
    cy = rng.randint(int(H * 0.1), int(H * 0.9))
    cx = rng.randint(int(W * 0.1), int(W * 0.9))
    dx = math.cos(angle); dy = math.sin(angle)
    y0 = int(cy - 0.5 * length * dy); x0 = int(cx - 0.5 * length * dx)
    y1 = int(cy + 0.5 * length * dy); x1 = int(cx + 0.5 * length * dx)
    return _draw_thick_line(m, y0, x0, y1, x1, thickness)


def _sample_crossing_lines(H, W, rng):
    m = np.zeros((H, W), dtype=bool)
    n_lines = rng.randint(2, 6)
    cy0 = rng.randint(int(H * 0.2), int(H * 0.8))
    cx0 = rng.randint(int(W * 0.2), int(W * 0.8))
    for _ in range(n_lines):
        angle = rng.uniform(0, math.pi)
        length = rng.randint(int(min(H, W) * 0.6), int(max(H, W) * 1.6))
        thickness = rng.randint(2, 11)
        # Each line passes near (cy0, cx0) but with a small offset
        offy = rng.randint(-int(H * 0.15), int(H * 0.15) + 1)
        offx = rng.randint(-int(W * 0.15), int(W * 0.15) + 1)
        cy = cy0 + offy; cx = cx0 + offx
        dx = math.cos(angle); dy = math.sin(angle)
        y0 = int(cy - 0.5 * length * dy); x0 = int(cx - 0.5 * length * dx)
        y1 = int(cy + 0.5 * length * dy); x1 = int(cx + 0.5 * length * dx)
        m = _draw_thick_line(m, y0, x0, y1, x1, thickness)
    return m


def _sample_star_pattern(H, W, rng):
    """3-6 lines that all pass through a single junction point."""
    m = np.zeros((H, W), dtype=bool)
    cy = rng.randint(int(H * 0.2), int(H * 0.8))
    cx = rng.randint(int(W * 0.2), int(W * 0.8))
    n_arms = rng.randint(3, 7)
    base_angle = rng.uniform(0, math.pi)
    for k in range(n_arms):
        angle = (base_angle + k * (math.pi / n_arms)) + rng.uniform(-0.1, 0.1)
        length = rng.randint(int(min(H, W) * 0.6), int(max(H, W) * 1.5))
        thickness = rng.randint(2, 9)
        dx = math.cos(angle); dy = math.sin(angle)
        y0 = int(cy - 0.5 * length * dy); x0 = int(cx - 0.5 * length * dx)
        y1 = int(cy + 0.5 * length * dy); x1 = int(cx + 0.5 * length * dx)
        m = _draw_thick_line(m, y0, x0, y1, x1, thickness)
    return m


def _sample_short_thin_lines(H, W, rng):
    """Many short thin lines scattered around the patch."""
    m = np.zeros((H, W), dtype=bool)
    n = rng.randint(8, 30)
    for _ in range(n):
        angle = rng.uniform(0, math.pi)
        length = rng.randint(15, 80)
        thickness = rng.randint(1, 4)
        cy = rng.randint(0, H - 1)
        cx = rng.randint(0, W - 1)
        dx = math.cos(angle); dy = math.sin(angle)
        y0 = int(cy - 0.5 * length * dy); x0 = int(cx - 0.5 * length * dx)
        y1 = int(cy + 0.5 * length * dy); x1 = int(cx + 0.5 * length * dx)
        m = _draw_thick_line(m, y0, x0, y1, x1, thickness)
    return m


def _sample_rectangle(H, W, rng):
    h = rng.randint(int(H * 0.04), int(H * 0.18))
    w = rng.randint(int(W * 0.04), int(W * 0.18))
    y = rng.randint(0, H - h)
    x = rng.randint(0, W - w)
    m = np.zeros((H, W), dtype=bool)
    m[y:y + h, x:x + w] = True
    return m


def _sample_freeform(H, W, rng):
    m = np.zeros((H, W), dtype=bool)
    n_strokes = rng.randint(1, 4)
    for _ in range(n_strokes):
        n_pts = rng.randint(3, 9)
        ys = [rng.randint(0, H - 1) for _ in range(n_pts)]
        xs = [rng.randint(0, W - 1) for _ in range(n_pts)]
        thick = rng.randint(4, 14)
        for i in range(n_pts - 1):
            m = _draw_thick_line(m, ys[i], xs[i], ys[i+1], xs[i+1], thick)
    return m


def _sample_blobs(H, W, rng):
    m = np.zeros((H, W), dtype=bool)
    n = rng.randint(2, 8)
    for _ in range(n):
        r = rng.randint(3, 12)
        cy = rng.randint(r, H - r)
        cx = rng.randint(r, W - r)
        yy, xx = np.ogrid[:H, :W]
        m |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    return m


# Generators by config. v2 favors line-like generators heavily.
HOLE_GENERATORS_V1 = [
    _sample_rectangle,
    _sample_freeform,
    _sample_short_thin_lines,
    _sample_blobs,
]
HOLE_GENERATORS_V2 = [
    _sample_long_line,
    _sample_long_line,
    _sample_long_line,
    _sample_crossing_lines,
    _sample_crossing_lines,
    _sample_star_pattern,
    _sample_short_thin_lines,
    _sample_short_thin_lines,
    _sample_freeform,
    _sample_rectangle,
    _sample_blobs,
]


def make_random_hole(H, W, valid_region: np.ndarray, rng,
                     generators=None,
                     min_frac=0.005, max_frac=0.20, max_tries=10) -> np.ndarray:
    if generators is None:
        generators = HOLE_GENERATORS_V2
    for _ in range(max_tries):
        k = rng.randint(1, 4)
        m = np.zeros((H, W), dtype=bool)
        for _ in range(k):
            m |= generators[rng.randint(0, len(generators) - 1)](H, W, rng)
        m &= valid_region
        frac = float(m.mean())
        if min_frac <= frac <= max_frac:
            return m
    # fallback: small rectangle inside valid_region
    m = np.zeros((H, W), dtype=bool)
    ys, xs = np.where(valid_region)
    if len(ys) == 0:
        return m
    idx = rng.randint(0, len(ys) - 1)
    cy, cx = ys[idx], xs[idx]
    h = rng.randint(8, 24); w = rng.randint(8, 24)
    y0, y1 = max(0, cy - h // 2), min(H, cy + h // 2)
    x0, x1 = max(0, cx - w // 2), min(W, cx + w // 2)
    m[y0:y1, x0:x1] = valid_region[y0:y1, x0:x1]
    return m


# ---------------------------------------------------------------------------
# Build network input from per-patch arrays
# ---------------------------------------------------------------------------

def build_network_input(in_features, channel_index, arr, masked_raw, hole):
    """Stack the requested feature channels into a [C, H, W] float32 tensor."""
    chan_map = {
        "masked_raw_norm": masked_raw,
        "hole_mask": hole.astype(np.float32),
        "raw_norm": arr[channel_index["raw_norm"]],
        "abs_norm": arr[channel_index["abs_norm"]],
        "sato_norm": arr[channel_index["sato_norm"]],
        "frangi_norm": arr[channel_index["frangi_norm"]],
        "coherence_norm": arr[channel_index["coherence_norm"]],
        "gradient_norm": arr[channel_index["gradient_norm"]],
    }
    return np.stack([chan_map[c].astype(np.float32) for c in in_features], axis=0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InpaintPatchDataset(Dataset):
    def __init__(self, prepare_dir: Path, in_features, forbidden_key="line_mask_75",
                 crop: int = 256, augment: bool = True,
                 min_valid_frac: float = 0.20, seed: int = 0,
                 hole_generators=None):
        self.prepare_dir = prepare_dir
        self.in_features = list(in_features)
        self.forbidden_key = forbidden_key
        self.crop = crop
        self.augment = augment
        self.rng = random.Random(seed)
        self.hole_generators = hole_generators or HOLE_GENERATORS_V2
        manifest = prepare_dir / "manifests" / "train_manifest.csv"
        with open(manifest) as f:
            self.records = [row for row in csv.DictReader(f)]
        self.records = [r for r in self.records
                        if float(r.get("valid_hole_frac", 1)) >= min_valid_frac]
        if not self.records:
            raise RuntimeError("no patches with sufficient valid hole fraction")

        sample = np.load(self.records[0]["file"], allow_pickle=True)
        chans = list(sample["channel_names"])
        for c in PREPARE_CHANNELS:
            if c not in chans:
                raise RuntimeError(
                    f"channel {c} missing in {self.records[0]['file']} -- "
                    "run prepare_project2_data.py to regenerate patches.")
        self.channel_index = {c: chans.index(c) for c in PREPARE_CHANNELS}

    def __len__(self):
        return len(self.records)

    def _load_patch(self, idx):
        rec = self.records[idx]
        with np.load(rec["file"], allow_pickle=True) as data:
            arr = data["data"]
        return arr

    def __getitem__(self, idx):
        arr = self._load_patch(idx)
        C, H, W = arr.shape
        crop = self.crop
        y0 = self.rng.randint(0, H - crop) if H > crop else 0
        x0 = self.rng.randint(0, W - crop) if W > crop else 0
        arr = arr[:, y0:y0 + crop, x0:x0 + crop]

        if self.augment:
            if self.rng.random() < 0.5:
                arr = arr[:, :, ::-1].copy()
            if self.rng.random() < 0.5:
                arr = arr[:, ::-1, :].copy()
            k = self.rng.randint(0, 3)
            if k:
                arr = np.rot90(arr, k=k, axes=(1, 2)).copy()

        ci = self.channel_index
        raw = arr[ci["raw_norm"]]
        forbid = (arr[ci[self.forbidden_key]] > 0.5) | (arr[ci["protect_mask"]] > 0.5)
        valid = ~forbid

        if valid.mean() < 0.05:
            return self.__getitem__((idx + 1) % len(self))

        Hc, Wc = raw.shape
        rng = np.random.RandomState(self.rng.randint(0, 2**31 - 1))
        hole = make_random_hole(Hc, Wc, valid, rng,
                                generators=self.hole_generators,
                                min_frac=0.005, max_frac=0.25)
        if hole.sum() == 0:
            return self.__getitem__((idx + 1) % len(self))

        from scipy.ndimage import gaussian_filter as gf
        local_mean = gf(raw.astype(np.float32), sigma=4.0)
        masked_raw = np.where(hole, local_mean, raw).astype(np.float32)

        x = build_network_input(self.in_features, ci, arr, masked_raw, hole)
        target = raw.astype(np.float32)[None]
        hole_mask = hole.astype(np.float32)[None]
        return {
            "x": torch.from_numpy(x),
            "target": torch.from_numpy(target),
            "hole": torch.from_numpy(hole_mask),
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(min(8, cout), cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(min(8, cout), cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=6, base=32):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.enc1 = ConvBlock(in_ch, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = ConvBlock(c4, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = ConvBlock(c3, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = ConvBlock(c2, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def gradient_diff_loss(pred, target, hole):
    pdy = pred[..., 1:, :] - pred[..., :-1, :]
    tdy = target[..., 1:, :] - target[..., :-1, :]
    pdx = pred[..., :, 1:] - pred[..., :, :-1]
    tdx = target[..., :, 1:] - target[..., :, :-1]
    hy = hole[..., 1:, :] * hole[..., :-1, :]
    hx = hole[..., :, 1:] * hole[..., :, :-1]
    eps = 1e-6
    ly = (torch.abs(pdy - tdy) * hy).sum() / (hy.sum() + eps)
    lx = (torch.abs(pdx - tdx) * hx).sum() / (hx.sum() + eps)
    return 0.5 * (ly + lx)


def context_band(hole, k=5):
    pad = k // 2
    dil = F.max_pool2d(hole, kernel_size=k, stride=1, padding=pad)
    return torch.clamp(dil - hole, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Preview rendering
# ---------------------------------------------------------------------------

def save_preview(path: Path, samples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(samples)
    cols = 6
    fig, axes = plt.subplots(n, cols, figsize=(2.6 * cols, 2.6 * n))
    if n == 1:
        axes = axes[None, :]
    titles = ["raw", "hole", "masked", "pred", "target", "|err|"]
    for r, s in enumerate(samples):
        for c, key in enumerate(titles):
            arr = s[key]
            if key in ("hole",):
                axes[r, c].imshow(arr, cmap="gray", vmin=0, vmax=1)
            elif key == "|err|":
                axes[r, c].imshow(arr, cmap="magma")
            else:
                lo = float(np.percentile(arr, 1))
                hi = float(np.percentile(arr, 99))
                if hi - lo < 1e-6:
                    hi = lo + 1e-6
                axes[r, c].imshow(arr, cmap="gray", vmin=lo, vmax=hi)
            axes[r, c].set_title(key, fontsize=8)
            axes[r, c].set_axis_off()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--preview-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="v2b", choices=list(INPUT_FEATURES_BY_CONFIG.keys()))
    ap.add_argument("--forbidden-key", default="line_mask_75",
                    help="channel used as forbidden region for hole sampling")
    ap.add_argument("--hole-style", default="v2", choices=["v1", "v2"])
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "curves").mkdir(parents=True, exist_ok=True)
    (out / "previews").mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  config={args.config}  forbidden={args.forbidden_key}  hole_style={args.hole_style}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    in_features = INPUT_FEATURES_BY_CONFIG[args.config]
    print(f"[features] {in_features}")
    hole_gen = HOLE_GENERATORS_V1 if args.hole_style == "v1" else HOLE_GENERATORS_V2

    full = InpaintPatchDataset(
        Path(args.prepared), in_features=in_features,
        forbidden_key=args.forbidden_key,
        crop=args.crop, augment=True,
        seed=args.seed, hole_generators=hole_gen,
    )
    n_val = max(1, int(len(full) * args.val_frac))
    val_idx = list(range(0, len(full), max(1, len(full) // n_val)))[:n_val]
    train_idx = [i for i in range(len(full)) if i not in set(val_idx)]
    print(f"[data] total={len(full)} train={len(train_idx)} val={len(val_idx)}")

    class Subset(torch.utils.data.Dataset):
        def __init__(self, base, idxs):
            self.base = base; self.idxs = idxs
        def __len__(self): return len(self.idxs)
        def __getitem__(self, i): return self.base[self.idxs[i]]

    train_ds = Subset(full, train_idx)
    val_ds_full = InpaintPatchDataset(
        Path(args.prepared), in_features=in_features,
        forbidden_key=args.forbidden_key,
        crop=args.crop, augment=False,
        seed=args.seed + 999, hole_generators=hole_gen,
    )
    val_ds = Subset(val_ds_full, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size), shuffle=False,
                            num_workers=0)

    model = UNetSmall(in_ch=len(in_features), base=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    cfg = {
        "config": args.config,
        "in_features": in_features,
        "forbidden_key": args.forbidden_key,
        "hole_style": args.hole_style,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "crop": args.crop,
        "model": "UNetSmall(base=32)",
        "loss": "L_hole + 0.1*L_context + 0.1*L_grad",
    }
    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    metrics_path = out / "logs" / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_hole", "train_ctx", "train_grad",
                                "val_loss", "val_hole", "lr", "time_s"])

    best_val = float("inf")

    def run_epoch(loader, train=True):
        if train:
            model.train()
        else:
            model.eval()
        agg = dict(loss=0.0, hole=0.0, ctx=0.0, grad=0.0, n=0)
        ctx_iter = torch.enable_grad if train else torch.no_grad
        with ctx_iter():
            for batch in loader:
                x = batch["x"].to(device, non_blocking=True)
                tgt = batch["target"].to(device, non_blocking=True)
                hole = batch["hole"].to(device, non_blocking=True)
                pred = model(x)
                eps = 1e-6
                l_hole = (torch.abs(pred - tgt) * hole).sum() / (hole.sum() + eps)
                ctx_band = context_band(hole, k=5)
                l_ctx = (torch.abs(pred - tgt) * ctx_band).sum() / (ctx_band.sum() + eps)
                l_grad = gradient_diff_loss(pred, tgt, hole)
                loss = l_hole + 0.1 * l_ctx + 0.1 * l_grad
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                bs = x.size(0)
                agg["loss"] += float(loss) * bs
                agg["hole"] += float(l_hole) * bs
                agg["ctx"] += float(l_ctx) * bs
                agg["grad"] += float(l_grad) * bs
                agg["n"] += bs
        for k in ("loss", "hole", "ctx", "grad"):
            agg[k] /= max(1, agg["n"])
        return agg

    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(train_loader, train=True)
        vl = run_epoch(val_loader, train=False)
        sched.step()
        dt = time.time() - t0
        lr_now = opt.param_groups[0]["lr"]
        history.append({"epoch": epoch, **{f"tr_{k}": v for k, v in tr.items() if k != "n"},
                        **{f"vl_{k}": v for k, v in vl.items() if k != "n"}, "lr": lr_now})
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr['loss']:.6f}", f"{tr['hole']:.6f}",
                                     f"{tr['ctx']:.6f}", f"{tr['grad']:.6f}",
                                     f"{vl['loss']:.6f}", f"{vl['hole']:.6f}",
                                     f"{lr_now:.6e}", f"{dt:.2f}"])
        print(f"epoch {epoch:03d}/{args.epochs}  train={tr['loss']:.4f} hole={tr['hole']:.4f}"
              f"  val={vl['loss']:.4f} hole={vl['hole']:.4f}  lr={lr_now:.2e}  {dt:.1f}s")

        torch.save({
            "model": model.state_dict(),
            "in_features": in_features,
            "config": cfg,
        }, out / "checkpoints" / "last.pt")

        if vl["loss"] < best_val:
            best_val = vl["loss"]
            torch.save({
                "model": model.state_dict(),
                "in_features": in_features,
                "config": cfg,
                "epoch": epoch,
                "val_loss": best_val,
            }, out / "checkpoints" / "best.pt")

        if epoch % args.preview_every == 0 or epoch == args.epochs:
            samples = []
            model.eval()
            with torch.no_grad():
                vit = iter(val_loader)
                try:
                    b = next(vit)
                except StopIteration:
                    b = next(iter(train_loader))
                x = b["x"].to(device)
                tgt = b["target"].to(device)
                hole = b["hole"].to(device)
                pred = model(x)
                bs = min(3, x.size(0))
                for k in range(bs):
                    raw_np = tgt[k, 0].cpu().numpy()
                    h_np = hole[k, 0].cpu().numpy()
                    masked_np = x[k, 0].cpu().numpy()
                    pred_np = pred[k, 0].cpu().numpy()
                    err = np.abs(pred_np - raw_np) * h_np
                    samples.append({
                        "raw": raw_np,
                        "hole": h_np,
                        "masked": masked_np,
                        "pred": pred_np,
                        "target": raw_np,
                        "|err|": err,
                    })
            save_preview(out / "previews" / f"epoch_{epoch:03d}.png", samples)

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                arr_e = [h["epoch"] for h in history]
                arr_t = [h["tr_loss"] for h in history]
                arr_v = [h["vl_loss"] for h in history]
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(arr_e, arr_t, label="train")
                ax.plot(arr_e, arr_v, label="val")
                ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(); ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(out / "curves" / "loss_curve.png", dpi=120)
                plt.close(fig)
            except Exception as e:
                print("[warn] curve plot failed:", e)

    last_prev = sorted((out / "previews").glob("epoch_*.png"))
    if last_prev:
        import shutil
        shutil.copyfile(last_prev[-1], out / "previews" / "final_train_examples.png")
    print(f"[done] best val loss = {best_val:.6f}; checkpoints under {out / 'checkpoints'}")


if __name__ == "__main__":
    main()
