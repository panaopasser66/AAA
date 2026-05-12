"""Stage 3: sliding-window inference with line-mask variants.

V2 changes:
  - 6 base masks (A/B/C/D/E/F) × 5 dilations (0/1/2/3/4) = 30 variants;
  - binary closing on the line mask before dilation, to connect short gaps;
  - core-hard + boundary-feather blending: pixels in the core mask are fully
    replaced by the prediction; pixels in a feather ring around the core
    fade from pred to raw via a distance-transform-based alpha;
  - coverage warning at >35% but never auto-skipped;
  - per-variant zoom crop column in the compare sheet;
  - optional --compare-against <v1_dir> to emit a v1-vs-v2 compare sheet.

Run:
  python scripts/infer_inpaint_unet.py \
      --prepared outputs/prepare \
      --model outputs/train_inpaint_v2b/checkpoints/best.pt \
      --out outputs/infer_inpaint_v2b
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import tifffile
import torch
from scipy import ndimage as ndi
from skimage import morphology


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def _load_train_module(scripts_dir: Path):
    path = scripts_dir / "train_inpaint_unet.py"
    spec = importlib.util.spec_from_file_location("train_inpaint_unet", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_inpaint_unet"] = mod
    spec.loader.exec_module(mod)
    return mod


def _norm_for_show(arr):
    a = arr.astype(np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype=np.uint8)
    lo, hi = np.percentile(a[finite], [1, 99])
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    u = np.clip((a - lo) / (hi - lo), 0, 1)
    return (u * 255).astype(np.uint8)


def _save_png(path: Path, arr_u8):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), arr_u8)


def _save_overlay(path: Path, base, mask, color=(255, 60, 60)):
    base_u = _norm_for_show(base)
    rgb = np.stack([base_u, base_u, base_u], axis=-1)
    rgb[mask.astype(bool)] = color
    _save_png(path, rgb)


def _save_tif(path: Path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr.astype(np.float32))


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------

def _hann2d(h, w):
    wy = np.hanning(h); wx = np.hanning(w)
    win = np.outer(wy, wx).astype(np.float32)
    return np.clip(win, 1e-3, None)


def sliding_inpaint(model, device, in_features, channels, inpaint_mask,
                    patch=512, stride=256):
    """Run U-Net across the image with a Hanning-window blend.

    `channels` is a dict of full-image float32 arrays keyed by channel name
    (matches names used in train_inpaint_unet.build_network_input).
    """
    raw = channels["raw_norm"].astype(np.float32)
    H, W = raw.shape
    pad_h = (patch - H) if H < patch else (patch - H % patch) % patch
    pad_w = (patch - W) if W < patch else (patch - W % patch) % patch

    def _pad(a):
        return np.pad(a, ((0, pad_h), (0, pad_w)), mode="reflect") if (pad_h or pad_w) else a

    padded = {k: _pad(v.astype(np.float32)) for k, v in channels.items()}
    hole_p = _pad(inpaint_mask.astype(np.float32))

    # masked_raw = raw with hole filled by smoothed local estimate
    smoothed = ndi.gaussian_filter(padded["raw_norm"], sigma=4.0)
    masked_raw_p = np.where(hole_p > 0.5, smoothed, padded["raw_norm"]).astype(np.float32)

    feature_arrays = []
    for fname in in_features:
        if fname == "masked_raw_norm":
            feature_arrays.append(masked_raw_p)
        elif fname == "hole_mask":
            feature_arrays.append(hole_p)
        else:
            feature_arrays.append(padded[fname])
    feature_stack = np.stack(feature_arrays, axis=0)  # [C, Hp, Wp]

    Hp, Wp = masked_raw_p.shape
    pred_acc = np.zeros((Hp, Wp), dtype=np.float32)
    weight_acc = np.zeros((Hp, Wp), dtype=np.float32)
    win = _hann2d(patch, patch)

    ys = list(range(0, max(1, Hp - patch + 1), stride))
    xs = list(range(0, max(1, Wp - patch + 1), stride))
    if not ys or ys[-1] + patch < Hp:
        ys.append(max(0, Hp - patch))
    if not xs or xs[-1] + patch < Wp:
        xs.append(max(0, Wp - patch))

    model.eval()
    with torch.no_grad():
        for y in ys:
            for x in xs:
                tile = feature_stack[:, y:y + patch, x:x + patch][None]
                t = torch.from_numpy(tile.copy()).to(device)
                p = model(t)[0, 0].cpu().numpy()
                pred_acc[y:y + patch, x:x + patch] += p * win
                weight_acc[y:y + patch, x:x + patch] += win

    weight_acc = np.maximum(weight_acc, 1e-6)
    pred_p = pred_acc / weight_acc
    return pred_p[:H, :W]


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

VARIANT_DEFS = [
    ("A_clear_sato98", "line_mask_98"),
    ("B_sato95",        "line_mask_95"),
    ("C_sato90",        "line_mask_90"),
    ("D_sato85",        "line_mask_85"),
    ("E_sato80",        "line_mask_80"),
    ("F_sato75",        "line_mask_75"),
]
DILATIONS = [0, 1, 2, 3, 4]


def _build_inpaint_mask(line_mask, protect_mask, dilate_radius, closing_radius=2):
    m = line_mask.astype(bool)
    if closing_radius > 0:
        m = morphology.binary_closing(m, footprint=morphology.disk(closing_radius))
    if dilate_radius > 0:
        m = morphology.binary_dilation(m, footprint=morphology.disk(dilate_radius))
    m = m & (~protect_mask.astype(bool))
    return m


def core_feather_blend(raw, pred, core_mask, feather_radius=3):
    """Hard-core + distance-feather blend.

    Inside `core_mask` -> pred. Inside the feather ring around core ->
    alpha = 1 - dist_to_core / feather_radius. Outside -> raw.
    Returns (delined, alpha).
    """
    core = core_mask.astype(bool)
    if feather_radius <= 0:
        alpha = core.astype(np.float32)
    else:
        # distance from each pixel to nearest core pixel
        dist = ndi.distance_transform_edt(~core)
        dilated = morphology.binary_dilation(core, footprint=morphology.disk(feather_radius))
        feather_ring = dilated & (~core)
        alpha = np.zeros_like(core, dtype=np.float32)
        alpha[core] = 1.0
        a_band = 1.0 - np.clip(dist[feather_ring] / float(feather_radius), 0.0, 1.0)
        alpha[feather_ring] = a_band
    delined = raw * (1.0 - alpha) + pred * alpha
    return delined.astype(np.float32), alpha.astype(np.float32)


# ---------------------------------------------------------------------------
# Zoom crop selection
# ---------------------------------------------------------------------------

def pick_zoom_box(sato_norm, crop=512, margin=64):
    H, W = sato_norm.shape
    crop = min(crop, H - 2 * margin if H > 2 * margin else H, W - 2 * margin if W > 2 * margin else W)
    crop = max(crop, 256)
    density = ndi.gaussian_filter(sato_norm.astype(np.float32), sigma=30.0)
    valid = np.zeros_like(density, dtype=bool)
    y0v = min(margin, max(0, H - crop))
    x0v = min(margin, max(0, W - crop))
    valid[y0v: max(y0v + 1, H - crop - margin), x0v: max(x0v + 1, W - crop - margin)] = True
    if not valid.any():
        cy = H // 2; cx = W // 2
    else:
        masked = np.where(valid, density, -np.inf)
        cy, cx = np.unravel_index(np.argmax(masked), density.shape)
    y0 = max(0, min(H - crop, cy - crop // 2))
    x0 = max(0, min(W - crop, cx - crop // 2))
    return int(y0), int(x0), int(crop)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _summary_panel(path, raw, inpaint_mask, pred, delined, removed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    titles = ["raw", "inpaint_mask", "background_pred", "delined", "removed=raw-delined"]
    arrs = [raw, inpaint_mask.astype(np.uint8) * 255, pred, delined, removed]
    fig, axes = plt.subplots(1, 5, figsize=(3.0 * 5, 3.4))
    for ax, t, a in zip(axes, titles, arrs):
        if t == "inpaint_mask":
            ax.imshow(a, cmap="gray", vmin=0, vmax=255)
        elif t == "removed=raw-delined":
            v = float(np.nanmax(np.abs(a)))
            v = max(v, 1e-6)
            ax.imshow(a, cmap="seismic", vmin=-v, vmax=v)
        else:
            lo, hi = np.percentile(a, [1, 99])
            if hi - lo < 1e-6:
                hi = lo + 1e-6
            ax.imshow(a, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(t, fontsize=9)
        ax.set_axis_off()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _contact_sheet(path, items):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(items)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    if rows == 1:
        axes = axes[None, :]
    for i, (name, d) in enumerate(items):
        r, c = i // cols, i % cols
        a = d["delined"]
        lo, hi = np.percentile(a, [1, 99])
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        axes[r, c].imshow(a, cmap="gray", vmin=lo, vmax=hi)
        axes[r, c].set_title(f"{name}\ncov={d['coverage']*100:.1f}%", fontsize=8)
        axes[r, c].set_axis_off()
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].set_axis_off()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _compare_sheet(path, items, raw, zoom_box):
    """Per-row: inpaint_overlay | delined | removed | zoom crop of delined."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y0, x0, crop = zoom_box
    n = len(items)
    cols = 4
    fig, axes = plt.subplots(n, cols, figsize=(3.2 * cols, 3.0 * n))
    if n == 1:
        axes = axes[None, :]
    titles = ["inpaint_overlay", "delined", "removed", f"zoom delined [{y0}:{y0+crop},{x0}:{x0+crop}]"]
    for i, (name, d) in enumerate(items):
        for c, t in enumerate(titles):
            if t == "inpaint_overlay":
                arr = d["inpaint_overlay"]
                axes[i, c].imshow(arr)
            elif t == "removed":
                arr = d["removed"]
                v = float(np.nanmax(np.abs(arr)))
                v = max(v, 1e-6)
                axes[i, c].imshow(arr, cmap="seismic", vmin=-v, vmax=v)
            elif t.startswith("zoom"):
                arr = d["delined"][y0:y0+crop, x0:x0+crop]
                lo, hi = np.percentile(arr, [1, 99])
                if hi - lo < 1e-6:
                    hi = lo + 1e-6
                axes[i, c].imshow(arr, cmap="gray", vmin=lo, vmax=hi)
            else:
                arr = d["delined"]
                lo, hi = np.percentile(arr, [1, 99])
                if hi - lo < 1e-6:
                    hi = lo + 1e-6
                axes[i, c].imshow(arr, cmap="gray", vmin=lo, vmax=hi)
            axes[i, c].set_title(f"{name}::{t}" if c == 0 else t, fontsize=8)
            axes[i, c].set_axis_off()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _v1_vs_v2_sheet(path, raw, v2_items_by_name, v1_dir: Path, zoom_box):
    """For each common (variant_dilation) tag found under v1_dir/variants/<tag>/delined.tif,
    show: raw_zoom | v1_delined_zoom | v2_delined_zoom | abs(v2-v1)_zoom."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y0, x0, crop = zoom_box
    rows = []
    for name, v2d in v2_items_by_name.items():
        v1_tif = v1_dir / "variants" / name / "delined.tif"
        if not v1_tif.exists():
            continue
        v1_delined = tifffile.imread(str(v1_tif)).astype(np.float32)
        if v1_delined.shape != raw.shape:
            continue
        rows.append((name, v1_delined, v2d["delined"]))
    if not rows:
        print("[warn] no overlapping variants between v1 and v2; skipping v1-vs-v2 sheet")
        return
    n = len(rows)
    cols = 4
    fig, axes = plt.subplots(n, cols, figsize=(3.2 * cols, 3.0 * n))
    if n == 1:
        axes = axes[None, :]
    titles = ["raw_zoom", "v1_delined_zoom", "v2_delined_zoom", "|v2-v1|_zoom"]
    raw_z = raw[y0:y0+crop, x0:x0+crop]
    for i, (name, v1d, v2d) in enumerate(rows):
        v1z = v1d[y0:y0+crop, x0:x0+crop]
        v2z = v2d[y0:y0+crop, x0:x0+crop]
        diff = np.abs(v2z - v1z)
        for c, t in enumerate(titles):
            if t == "raw_zoom":
                arr = raw_z; cmap = "gray"; vmin, vmax = np.percentile(arr, [1, 99])
            elif t == "v1_delined_zoom":
                arr = v1z; cmap = "gray"; vmin, vmax = np.percentile(arr, [1, 99])
            elif t == "v2_delined_zoom":
                arr = v2z; cmap = "gray"; vmin, vmax = np.percentile(arr, [1, 99])
            else:
                arr = diff; cmap = "magma"; vmin, vmax = 0.0, max(1e-6, float(arr.max()))
            if vmax - vmin < 1e-6:
                vmax = vmin + 1e-6
            axes[i, c].imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[i, c].set_title(f"{name}::{t}" if c == 0 else t, fontsize=8)
            axes[i, c].set_axis_off()
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patch", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--feather-radius", type=int, default=3,
                    help="boundary feather ring width in pixels (alpha decays linearly)")
    ap.add_argument("--closing-radius", type=int, default=2,
                    help="binary closing radius applied to base line mask")
    ap.add_argument("--coverage-warn", type=float, default=35.0)
    ap.add_argument("--zoom-crop", type=int, default=512)
    ap.add_argument("--compare-against", default=None,
                    help="path to a previous infer dir (e.g. outputs/infer_inpaint_v1) "
                         "to emit a v1-vs-v2 compare sheet")
    ap.add_argument("--compare-variants", default=None,
                    help="comma-separated list of variant tags to use in compare_sheet "
                         "(default: dilate2 row from each base mask)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "variants").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    scripts_dir = Path(__file__).resolve().parent
    train_mod = _load_train_module(scripts_dir)

    ckpt = torch.load(args.model, map_location=device)
    in_features = ckpt.get("in_features")
    if in_features is None:
        in_features = train_mod.INPUT_FEATURES_BY_CONFIG.get("v2b")
    print(f"[ckpt] in_features = {in_features}")
    if "config" in ckpt:
        print(f"[ckpt] config = {ckpt['config'].get('config', '?')}")

    model = train_mod.UNetSmall(in_ch=len(in_features), base=32).to(device)
    model.load_state_dict(ckpt["model"])

    glob_path = Path(args.prepared) / "global" / "global_arrays.npz"
    print(f"[load] {glob_path}")
    with np.load(glob_path, allow_pickle=True) as data:
        layers = {k: data[k] for k in data.files if k != "channel_names"}

    raw = layers["raw_norm"].astype(np.float32)
    H, W = raw.shape
    protect_mask = layers["protect_mask"].astype(bool)
    sato = layers["sato_norm"].astype(np.float32)

    zoom_box = pick_zoom_box(sato, crop=args.zoom_crop)
    print(f"[zoom] y0={zoom_box[0]} x0={zoom_box[1]} crop={zoom_box[2]}")

    # Channels available to the model (full-image)
    channels = {
        "raw_norm": layers["raw_norm"],
        "abs_norm": layers["abs_norm"],
        "sato_norm": layers["sato_norm"],
        "frangi_norm": layers["frangi_norm"],
        "coherence_norm": layers["coherence_norm"],
        "gradient_norm": layers["gradient_norm"],
    }
    # Only keep channels that the model actually consumes
    needed = set(in_features) - {"masked_raw_norm", "hole_mask"}
    channels = {k: v for k, v in channels.items() if k in needed}

    items = []
    items_by_name = {}
    for vname, mask_key in VARIANT_DEFS:
        if mask_key not in layers:
            print(f"[warn] {mask_key} missing in global arrays; skipping {vname}")
            continue
        line_mask = layers[mask_key].astype(bool)
        for d in DILATIONS:
            tag = f"{vname}_dilate{d}"
            print(f"\n=== variant {tag} ===")
            inpaint_mask = _build_inpaint_mask(
                line_mask, protect_mask, d,
                closing_radius=args.closing_radius,
            )
            cov = float(inpaint_mask.mean())
            print(f"  inpaint coverage: {cov*100:.3f}% (closing={args.closing_radius}, dilate={d})")
            if cov * 100 > args.coverage_warn:
                print(f"  [warn] coverage exceeds {args.coverage_warn:.1f}%; not skipping")

            pred = sliding_inpaint(model, device, in_features, channels, inpaint_mask,
                                   patch=args.patch, stride=args.stride)
            delined, alpha = core_feather_blend(raw, pred, inpaint_mask,
                                                feather_radius=args.feather_radius)
            removed = raw - delined

            v_dir = out / "variants" / tag
            v_dir.mkdir(parents=True, exist_ok=True)
            _save_png(v_dir / "inpaint_mask.png", (inpaint_mask.astype(np.uint8) * 255))
            _save_overlay(v_dir / "inpaint_overlay.png", raw, inpaint_mask)
            _save_tif(v_dir / "background_pred.tif", pred)
            _save_tif(v_dir / "delined.tif", delined)
            _save_tif(v_dir / "alpha.tif", alpha)
            _save_png(v_dir / "delined_preview.png", _norm_for_show(delined))
            removed_abs = np.abs(removed)
            _save_png(v_dir / "removed.png", _norm_for_show(removed_abs))
            _summary_panel(v_dir / "summary_panel.png", raw, inpaint_mask, pred, delined, removed)

            base_u = _norm_for_show(raw)
            ovl_rgb = np.stack([base_u, base_u, base_u], axis=-1)
            ovl_rgb[inpaint_mask] = (255, 60, 60)

            entry = {
                "delined": delined,
                "pred": pred,
                "removed": removed,
                "inpaint_overlay": ovl_rgb,
                "coverage": cov,
            }
            items.append((tag, entry))
            items_by_name[tag] = entry

    # Contact sheet (all variants)
    _contact_sheet(out / "reports" / "inpaint_contact_sheet.png", items)

    # Compare sheet (default: dilate2 row from each base mask)
    if args.compare_variants:
        compare_names = [s.strip() for s in args.compare_variants.split(",") if s.strip()]
    else:
        compare_names = [f"{vn}_dilate2" for vn, _ in VARIANT_DEFS]
    compare_items = [(n, items_by_name[n]) for n in compare_names if n in items_by_name]
    if compare_items:
        _compare_sheet(out / "reports" / "inpaint_compare_sheet.png", compare_items, raw, zoom_box)

    # Optional v1-vs-v2 sheet
    if args.compare_against:
        v1_dir = Path(args.compare_against)
        _v1_vs_v2_sheet(out / "reports" / "v1_vs_v2_compare.png",
                        raw, items_by_name, v1_dir, zoom_box)

    summary = {
        "n_variants": len(items),
        "image_shape": [H, W],
        "patch": args.patch,
        "stride": args.stride,
        "feather_radius": args.feather_radius,
        "closing_radius": args.closing_radius,
        "coverage_warn": args.coverage_warn,
        "model": args.model,
        "ckpt_in_features": in_features,
        "zoom_box": list(zoom_box),
        "coverage_per_variant": {n: round(d["coverage"] * 100, 4) for n, d in items},
    }
    with open(out / "reports" / "infer_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[done]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
