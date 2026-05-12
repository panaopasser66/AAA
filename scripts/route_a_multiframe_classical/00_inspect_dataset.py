import _bootstrap  # noqa: F401

import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.filename_parser import scan_raw_dir
from src.io_utils import ensure_dir, load_tif_float32, robust_rescale_for_preview, save_png


def _build_contact_sheet(items, cols=4, target_w=320, pad=4, title_h=18):
    if not items:
        return np.zeros((1, 1), dtype=np.uint8)
    thumbs = []
    titles = []
    for title, img in items:
        prev = robust_rescale_for_preview(img)
        if prev.shape[1] > target_w:
            scale = target_w / prev.shape[1]
            prev = cv2.resize(
                prev,
                (int(prev.shape[1] * scale), int(prev.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        thumbs.append(prev)
        titles.append(title)
    H, W = thumbs[0].shape
    rows = (len(thumbs) + cols - 1) // cols
    cell_h = H + title_h + pad
    cell_w = W + pad
    sheet = np.full((rows * cell_h + pad, cols * cell_w + pad), 32, dtype=np.uint8)
    for i, (prev, title) in enumerate(zip(thumbs, titles)):
        r, c = divmod(i, cols)
        y0 = pad + r * cell_h + title_h
        x0 = pad + c * cell_w
        sheet[y0 : y0 + H, x0 : x0 + W] = prev
        cv2.putText(
            sheet,
            title,
            (x0, y0 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            255,
            1,
            cv2.LINE_AA,
        )
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="raw")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    inspect_dir = os.path.join(args.out_dir, "inspect")
    prev_dir = os.path.join(inspect_dir, "previews")
    ensure_dir(inspect_dir)
    ensure_dir(prev_dir)

    df = scan_raw_dir(args.raw_dir)
    print(f"[00] Found {len(df)} raw TIFFs in {args.raw_dir}")

    rows = []
    contact_items = []
    for _, row in df.iterrows():
        path = row["path"]
        img = load_tif_float32(path)
        finite = img[np.isfinite(img)]
        stats = {
            "filename": row["filename"],
            "shape": f"{img.shape[0]}x{img.shape[1]}",
            "dtype_raw": str(img.dtype),
            "min": float(finite.min()) if finite.size else float("nan"),
            "max": float(finite.max()) if finite.size else float("nan"),
            "p0.5": float(np.percentile(finite, 0.5)) if finite.size else float("nan"),
            "p1": float(np.percentile(finite, 1)) if finite.size else float("nan"),
            "p50": float(np.median(finite)) if finite.size else float("nan"),
            "p99": float(np.percentile(finite, 99)) if finite.size else float("nan"),
            "p99.5": float(np.percentile(finite, 99.5)) if finite.size else float("nan"),
            "axis1": int(row["axis1"]),
            "axis2": int(row["axis2"]),
            "axis3": int(row["axis3"]),
            "axis4": int(row["axis4"]),
            "path": path,
        }
        rows.append(stats)
        prev_path = os.path.join(prev_dir, row["filename"].replace(".tif", ".png"))
        save_png(prev_path, robust_rescale_for_preview(img))
        contact_items.append((row["filename"], img))

    out_meta = pd.DataFrame(rows)
    meta_csv = os.path.join(inspect_dir, "metadata.csv")
    out_meta.to_csv(meta_csv, index=False)
    print(f"[00] Wrote metadata: {meta_csv}")

    sheet = _build_contact_sheet(contact_items, cols=4, target_w=320)
    sheet_path = os.path.join(inspect_dir, "contact_sheet.png")
    save_png(sheet_path, sheet)
    print(f"[00] Wrote contact sheet: {sheet_path}")


if __name__ == "__main__":
    main()
