"""Rebuild patches / manifest / contact sheets from a manually selected pseudo-label.

Inputs (already on disk after Stage 1 + label sweep):
  <dataset>/pseudo_labels/train_label_selected.png   (the chosen label, e.g. variant D)
  <dataset>/responses/response_{sato,frangi,coherence,gradient}.tif
  <input>/data.tif

Outputs (overwritten):
  <dataset>/patches/{images,labels,ignore}/...
  <dataset>/manifests/train_manifest.csv
  <dataset>/reports/patch_contact_sheet.png
  <dataset>/reports/hard_cases_contact_sheet.png
  <dataset>/reports/summary.json    (selected_label + ratio fields refreshed)

Run:
    python regenerate_patches.py --dataset line_stage1_dataset
    python regenerate_patches.py --dataset line_stage1_dataset --input data.tif

This script does NOT train a model and does NOT do defect detection.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_line_dataset import (  # noqa: E402
    hard_cases_contact_sheet,
    normalize01,
    patch_contact_sheet,
    read_tif,
    slice_patches,
    stack_channels,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate patches from a selected pseudo-label")
    p.add_argument("--dataset", required=True, help="Existing dataset root, e.g. line_stage1_dataset")
    p.add_argument("--input", default=None, help="Path to source .tif (defaults to ./data.tif or <dataset>/../data.tif).")
    p.add_argument(
        "--label",
        default=None,
        help="Override label path (default: <dataset>/pseudo_labels/train_label_selected.png).",
    )
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--min-line-ratio", type=float, default=0.002)
    p.add_argument("--max-ignore-ratio", type=float, default=0.7)
    p.add_argument("--max-patches-contact", type=int, default=64)
    p.add_argument("--keep-old-patches", action="store_true",
                   help="Skip wiping existing patch files before regenerating.")
    return p.parse_args()


def resolve_input(dataset: Path, explicit: Path | None) -> Path:
    if explicit and explicit.exists():
        return explicit
    for cand in (dataset.parent / "data.tif", Path("data.tif")):
        if cand.exists():
            return cand
    raise SystemExit("[ERROR] could not locate source .tif; pass --input")


def clear_patch_files(root: Path) -> None:
    for sub in ("patches/images", "patches/labels", "patches/ignore"):
        d = root / sub
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.is_file():
                f.unlink()


def load_response(rdir: Path, name: str) -> np.ndarray:
    p = rdir / f"response_{name}.tif"
    if not p.exists():
        raise SystemExit(f"[ERROR] missing cached response {p}; rerun prepare_line_dataset.py first")
    return tifffile.imread(str(p)).astype(np.float32)


def main() -> None:
    args = parse_args()
    root = Path(args.dataset)
    if not root.exists():
        raise SystemExit(f"[ERROR] dataset directory not found: {root}")

    label_path = Path(args.label) if args.label else (root / "pseudo_labels" / "train_label_selected.png")
    if not label_path.exists():
        raise SystemExit(f"[ERROR] selected label not found: {label_path}")

    input_path = resolve_input(root, Path(args.input) if args.input else None)

    train_label = np.array(Image.open(label_path))
    if train_label.ndim != 2:
        raise SystemExit(f"[ERROR] selected label must be 2D; got shape {train_label.shape}")
    train_label = train_label.astype(np.uint8, copy=False)
    print(f"[INFO] selected label: {label_path}  shape={train_label.shape}")

    img = read_tif(input_path)
    if img.shape != train_label.shape:
        raise SystemExit(f"[ERROR] image shape {img.shape} vs label shape {train_label.shape}")

    rdir = root / "responses"
    sato_r = load_response(rdir, "sato")
    frangi_r = load_response(rdir, "frangi")
    coh_r = load_response(rdir, "coherence")
    grad_r = load_response(rdir, "gradient")

    raw_n = normalize01(img, 1.0, 99.0)
    abs_n = normalize01(np.abs(img), 0.0, 99.5)
    sato_n = normalize01(sato_r)
    frangi_n = normalize01(frangi_r)
    coh_n = normalize01(coh_r)
    grad_n = normalize01(grad_r)
    stacked = stack_channels(raw_n, abs_n, sato_n, frangi_n, coh_n, grad_n)

    if not args.keep_old_patches:
        print("[INFO] wiping previous patches/{images,labels,ignore}")
        clear_patch_files(root)
    for sub in ("patches/images", "patches/labels", "patches/ignore",
                "manifests", "reports"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    slice_args = Namespace(
        patch_size=args.patch_size,
        stride=args.stride,
        min_line_ratio=args.min_line_ratio,
        max_ignore_ratio=args.max_ignore_ratio,
    )
    print(f"[INFO] slicing patches: ps={slice_args.patch_size} stride={slice_args.stride} "
          f"min_line_ratio={slice_args.min_line_ratio} max_ignore_ratio={slice_args.max_ignore_ratio}")
    rows = slice_patches(stacked, train_label, slice_args, root, input_path.stem)
    print(f"[INFO] kept {len(rows)} patches")

    manifest_path = root / "manifests" / "train_manifest.csv"
    write_manifest(rows, manifest_path)

    print("[INFO] generating contact sheets ...")
    patch_contact_sheet(
        rows, root, root / "reports" / "patch_contact_sheet.png", max_n=args.max_patches_contact
    )
    hard_cases_contact_sheet(
        rows, root, root / "reports" / "hard_cases_contact_sheet.png", max_n=args.max_patches_contact
    )

    line_ratio = float((train_label == 1).mean())
    ignore_ratio = float((train_label == 255).mean())

    summary_path = root / "reports" / "summary.json"
    summary: dict = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    summary.update(
        {
            "patch_size": args.patch_size,
            "stride": args.stride,
            "num_patches": len(rows),
            "line_pixel_ratio": line_ratio,
            "ignore_pixel_ratio": ignore_ratio,
            "response_source": "sato",
            "label_values": {"background": 0, "line": 1, "ignore": 255},
            "selected_label": label_path.name,
            "selected_label_source": str(label_path.relative_to(root)).replace("\\", "/"),
        }
    )
    summary.setdefault("num_input_images", 1)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("Done regenerate.")
    print(f"  selected label : {label_path}")
    print(f"  patches kept   : {len(rows)}")
    print(f"  line ratio     : {line_ratio:.4f}")
    print(f"  ignore ratio   : {ignore_ratio:.4f}")
    print(f"  manifest       : {manifest_path.as_posix()}")
    print(f"  contact sheet  : {(root / 'reports' / 'patch_contact_sheet.png').as_posix()}")


if __name__ == "__main__":
    main()
