"""Stage 3A — manual review reranking, crop sheets, label template.

Spec: task_stage3.md §3–6.

Outputs (under outputs/manual_review/):
  manual_review_candidates_all.csv
  manual_review_top{N}.csv          for N in manual_review.top_k
  manual_review_overlay_top{N}.png
  crops_top{N}/<rank>_id<id>_x<x>_y<y>_score<s>.png
  review_index.html
  manual_labels_template.csv
"""

import _bootstrap  # noqa: F401

import argparse
import os
from collections import OrderedDict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import cv2
import numpy as np
import pandas as pd

from src.groups import get_enabled_groups, load_config
from src.io_utils import ensure_dir, robust_rescale_for_preview
from src.manual_review_ranking import (
    REVIEW_COLUMNS,
    labels_template,
    nms_in_place,
    rank_candidates,
)


# Spec-listed text-panel fields (task_stage3.md §6)
TEXT_FIELDS = [
    "candidate_id", "x", "y", "diameter_px", "polarity", "source_category",
    "review_priority_score", "diameter_closeness_score", "shape_score",
    "persistence_score", "spot_present_when_uncovered", "uncovered_frame_count",
    "distance_to_refined_static_line_skeleton",
    "distance_to_refined_combined_line_exclusion",
    "anisotropy", "aspect_ratio", "circularity",
]

# Spec-listed crop tile order (task_stage3.md §6)
TILE_ORDER = [
    "0000.tif",
    "axis2_x/0-100.tif", "axis2_x/0100.tif",
    "axis4_y/000-1.tif", "axis4_y/0001.tif",
    "axis1_angle/-2000.tif", "axis1_angle/-1000.tif",
    "axis1_angle/1000.tif", "axis1_angle/2000.tif",
    "combined_fused_delined",
    "combined_fused_delined_v2",
    "combined_line_risk_v2",
    "refined_static_line_mask",
    "refined_combined_line_exclusion",
    "refined_safe_search_mask",
]


def _opt_npy(path: str):
    return np.load(path).astype(np.float32) if os.path.isfile(path) else None


def _load_aligned(out_dir, enabled, reference):
    aligned_dir = os.path.join(out_dir, "aligned")
    images: Dict[str, np.ndarray] = OrderedDict()
    ref_added = False
    for name, spec in enabled.items():
        if reference in spec.get("files", []):
            p = os.path.join(aligned_dir, name, reference.replace(".tif", ".npy"))
            if os.path.isfile(p) and not ref_added:
                images[reference] = np.load(p).astype(np.float32)
                ref_added = True
                break
    for name, spec in enabled.items():
        for fname in spec.get("files", []):
            if fname == reference and ref_added:
                continue
            label = f"{name}/{fname}"
            p = os.path.join(aligned_dir, name, fname.replace(".tif", ".npy"))
            if os.path.isfile(p):
                images[label] = np.load(p).astype(np.float32)
    return images


def _crop_around(img: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    H, W = img.shape[:2]
    half = size // 2
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x1 = x0 + size
    y1 = y0 + size
    pad_left = max(0, -x0); pad_top = max(0, -y0)
    pad_right = max(0, x1 - W); pad_bottom = max(0, y1 - H)
    x0c = max(0, x0); y0c = max(0, y0)
    x1c = min(W, x1); y1c = min(H, y1)
    patch = img[y0c:y1c, x0c:x1c]
    if pad_left or pad_top or pad_right or pad_bottom:
        patch = np.pad(
            patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge",
        )
    return patch


def _label_image(img_bgr: np.ndarray, title: str) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    out = np.full((h + 18, w, 3), 32, dtype=np.uint8)
    src = img_bgr if img_bgr.ndim == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    out[18:, :, :] = src
    cv2.putText(out, title, (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _draw_circle(crop_u8: np.ndarray, radius_px: float) -> np.ndarray:
    if crop_u8.ndim == 2:
        out = cv2.cvtColor(crop_u8, cv2.COLOR_GRAY2BGR)
    else:
        out = crop_u8.copy()
    cx = out.shape[1] // 2
    cy = out.shape[0] // 2
    r = max(2, int(round(radius_px)))
    cv2.circle(out, (cx, cy), r, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def _build_text_panel(row: dict, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 16, dtype=np.uint8)
    y = 18
    line_h = 14
    for field in TEXT_FIELDS:
        v = row.get(field, "")
        if isinstance(v, float):
            txt = f"{field}: {v:.3f}"
        elif isinstance(v, (int, np.integer)):
            txt = f"{field}: {int(v)}"
        else:
            s = "" if v is None else str(v)
            if len(s) > 28:
                s = s[:25] + "..."
            txt = f"{field}: {s}"
        cv2.putText(panel, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (240, 240, 240), 1, cv2.LINE_AA)
        y += line_h
        if y > height - 4:
            break
    return panel


def _render_crop_sheet(
    row: dict, images_dict: Dict[str, np.ndarray], size: int, out_path: str,
):
    x = float(row["x"])
    y = float(row["y"])
    radius = float(row.get("radius_px", 3.0))

    # Build tiles in spec order; skip missing
    tiles: List[np.ndarray] = []
    titles: List[str] = []
    for key in TILE_ORDER:
        img = images_dict.get(key)
        if img is None:
            continue
        crop = _crop_around(img, x, y, size)
        prev = robust_rescale_for_preview(crop)
        prev = _draw_circle(prev, radius)
        tiles.append(_label_image(prev, key))
        titles.append(key)

    if not tiles:
        return

    cols = 5
    rows = (len(tiles) + cols - 1) // cols
    cell_h, cell_w = tiles[0].shape[:2]
    pad = 4
    grid_w = cols * cell_w + pad * (cols + 1)
    grid_h = rows * cell_h + pad * (rows + 1)
    sheet = np.full((grid_h, grid_w, 3), 16, dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y0 = pad + r * (cell_h + pad)
        x0 = pad + c * (cell_w + pad)
        sheet[y0:y0 + cell_h, x0:x0 + cell_w, :] = t

    text_w = 320
    text_panel = _build_text_panel(row, text_w, max(grid_h, 240))

    H = max(sheet.shape[0], text_panel.shape[0])
    full = np.full((H, sheet.shape[1] + text_panel.shape[1], 3), 16, dtype=np.uint8)
    full[:sheet.shape[0], :sheet.shape[1], :] = sheet
    full[:text_panel.shape[0], sheet.shape[1]:, :] = text_panel

    # Header
    cid = int(row.get("candidate_id", 0))
    sc = float(row.get("review_priority_score", 0.0))
    polarity = row.get("polarity", "")
    cls = row.get("source_category", "")
    reason = row.get("review_reason", "")
    header = np.full((22, full.shape[1], 3), 0, dtype=np.uint8)
    text = (
        f"id={cid:04d} x={x:.0f} y={y:.0f} d={row.get('diameter_px',0):.1f}px "
        f"pol={polarity} src={cls} score={sc:.3f} reason={reason}"
    )
    cv2.putText(header, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    full = np.vstack([header, full])

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, full)


def _draw_overlay(img: np.ndarray, df: pd.DataFrame, out_path: str,
                  max_label: int = 200, ranked: bool = True):
    prev = robust_rescale_for_preview(img)
    bgr = cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR)
    for i, row in df.iterrows():
        x = int(round(row["x"])); y = int(round(row["y"]))
        r = max(3, int(round(float(row.get("radius_px", 3.0)) + 2)))
        color = (0, 255, 255)
        cv2.circle(bgr, (x, y), r, color, 1, cv2.LINE_AA)
        if i < max_label:
            tag = f"{i+1}" if ranked else str(int(row.get("candidate_id", 0)))
            cv2.putText(bgr, tag, (x + r + 1, y - r - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, bgr)


def _write_review_index(html_path: str, rows: list, missing_files: list):
    review_dir = os.path.dirname(html_path)
    parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Manual Review</title>",
        "<style>",
        "body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:16px;color:#222}",
        "table{border-collapse:collapse;font-size:12px}",
        "th,td{border:1px solid #ccc;padding:2px 6px;vertical-align:top}",
        "th{background:#f5f5f5}",
        "img.crop{width:280px;border:1px solid #ddd;display:block}",
        ".note{color:#666;font-size:12px;margin-bottom:8px}",
        ".caveat{background:#fff5d6;border-left:4px solid #d4a300;padding:8px;margin:8px 0;font-size:13px}",
        "</style></head><body>",
        "<h1>Manual review</h1>",
        "<div class=caveat>"
        "<b>Caveats (task_stage3.md §16):</b>"
        "<ol>"
        "<li>No deep learning is being used at this stage.</li>"
        "<li>This page is Stage 3A — manual review ranking.</li>"
        "<li>Stage 3B (separate page) is a traditional inpainting / line-removal baseline.</li>"
        "<li>Candidates detected on cleaned images are still NOT final defects.</li>"
        "<li>Manual labels collected here will feed Stage 4 deep-learning training.</li>"
        "</ol></div>",
        "<p class=note>Scoring formula:<br><code>"
        "review_priority_score = 0.30 * diameter_closeness + 0.25 * shape "
        "+ 0.30 * persistence - 0.10 * line_penalty_soft - 0.05 * texture_penalty_soft"
        "</code></p>",
    ]
    if missing_files:
        parts.append("<div class=caveat><b>Missing input artefacts:</b><ul>")
        for m in missing_files:
            parts.append(f"<li><code>{m}</code></li>")
        parts.append("</ul></div>")
    parts.append("<table>")
    parts.append(
        "<tr><th>rank</th><th>id</th><th>src</th><th>x</th><th>y</th><th>d (px)</th>"
        "<th>polarity</th><th>priority</th><th>d_close</th><th>shape</th>"
        "<th>persist</th><th>line_pen</th><th>tex_pen</th><th>reason</th>"
        "<th>crop</th></tr>"
    )
    for i, r in enumerate(rows):
        cp = r.get("crop_path", "")
        cp_rel = ""
        if cp and os.path.isfile(cp):
            cp_rel = os.path.relpath(cp, review_dir).replace("\\", "/")
        parts.append("<tr>")
        parts.append(f"<td>{i+1}</td>")
        parts.append(f"<td>{r.get('candidate_id','')}</td>")
        parts.append(f"<td>{r.get('source_category','')}</td>")
        parts.append(f"<td>{int(r.get('x',0))}</td><td>{int(r.get('y',0))}</td>")
        parts.append(f"<td>{r.get('diameter_px',0):.1f}</td>")
        parts.append(f"<td>{r.get('polarity','')}</td>")
        parts.append(f"<td>{r.get('review_priority_score',0):.3f}</td>")
        parts.append(f"<td>{r.get('diameter_closeness_score',0):.2f}</td>")
        parts.append(f"<td>{r.get('shape_score',0):.2f}</td>")
        parts.append(f"<td>{r.get('persistence_score',0):.2f}</td>")
        parts.append(f"<td>{r.get('line_penalty_soft',0):.2f}</td>")
        parts.append(f"<td>{r.get('texture_penalty_soft',0):.2f}</td>")
        parts.append(f"<td>{r.get('review_reason','')}</td>")
        if cp_rel:
            parts.append(f"<td><img class=crop src='{cp_rel}'></td>")
        else:
            parts.append("<td></td>")
        parts.append("</tr>")
    parts.append("</table></body></html>")
    ensure_dir(review_dir)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    config = load_config(args.config)
    mr_cfg = config.get("manual_review", {}) or {}
    if not bool(mr_cfg.get("enabled", True)):
        print("[06] manual_review.enabled=false; skipping.")
        return

    source_categories = list(mr_cfg.get(
        "source_categories", ["line_adjacent_uncertain_v2", "texture_uncertain_v2"]
    ))
    top_k = list(mr_cfg.get("top_k", [50, 100]))
    crop_size = int(mr_cfg.get("crop_size", 96))
    dedupe_d = float(mr_cfg.get("dedupe_distance_px", 4))
    exclude_invalid = bool(mr_cfg.get("exclude_in_invalid_region", True))

    cand_dir = os.path.join(args.out_dir, "candidates")
    review_dir = os.path.join(args.out_dir, "manual_review")
    ensure_dir(review_dir)

    # Read source candidate CSVs. Spec uses ..._v2 suffix in the category name
    # itself; on disk we have candidates_<cat>_v2.csv.
    parts: List[pd.DataFrame] = []
    missing_files: List[str] = []
    for src_cat in source_categories:
        if src_cat.endswith("_v2"):
            base_cat = src_cat[:-3]
            suffix = "_v2"
        elif src_cat.endswith("_v1"):
            base_cat = src_cat[:-3]
            suffix = ""
        else:
            base_cat = src_cat
            suffix = ""
        p = os.path.join(cand_dir, f"candidates_{base_cat}{suffix}.csv")
        if not os.path.isfile(p):
            print(f"[06] WARN: missing source CSV {p}")
            missing_files.append(os.path.relpath(p, args.out_dir))
            continue
        sub = pd.read_csv(p)
        if sub.empty:
            continue
        sub["source_category"] = src_cat
        parts.append(sub)
    if not parts:
        print("[06] No source candidates found; nothing to do.")
        return
    df = pd.concat(parts, ignore_index=True)
    print(f"[06] Source pool: {len(df)} candidates from {source_categories}")

    if exclude_invalid and "in_invalid_region" in df.columns:
        df = df[df["in_invalid_region"].astype(int) == 0].reset_index(drop=True)
        print(f"[06] After in_invalid_region filter: {len(df)} candidates")

    df = rank_candidates(df, config)
    df = nms_in_place(df, min_distance_px=dedupe_d)
    print(f"[06] After rerank + NMS: {len(df)} candidates")

    # All ranked CSV
    all_csv = os.path.join(review_dir, "manual_review_candidates_all.csv")
    df.to_csv(all_csv, index=False)
    print(f"[06] Wrote {all_csv}")

    # Image dict for crop sheets (axis1 frames included via _load_aligned)
    enabled = get_enabled_groups(config)
    reference = config.get("reference_image", "0000.tif")
    aligned = _load_aligned(args.out_dir, enabled, reference)

    extras: Dict[str, np.ndarray] = OrderedDict()
    pairs = [
        ("combined_fused_delined", os.path.join(args.out_dir, "fusion", "combined_fused_delined.npy")),
        ("combined_fused_delined_v2", os.path.join(args.out_dir, "fusion", "combined_fused_delined_v2.npy")),
        ("combined_line_risk_v2", os.path.join(args.out_dir, "line_risk", "combined_line_risk_v2.npy")),
        ("refined_static_line_mask", os.path.join(args.out_dir, "line_risk", "refined_static_line_mask.npy")),
        ("refined_combined_line_exclusion", os.path.join(args.out_dir, "line_risk", "refined_combined_line_exclusion_dilate8.npy")),
        ("refined_safe_search_mask", os.path.join(args.out_dir, "invalid_region", "refined_safe_search_mask_dilate8.npy")),
    ]
    for name, p in pairs:
        v = _opt_npy(p)
        if v is not None:
            extras[name] = v
        else:
            missing_files.append(os.path.relpath(p, args.out_dir))
    images_template = OrderedDict()
    for k, v in aligned.items():
        images_template[k] = v
    for k, v in extras.items():
        images_template[k] = v

    # Overlays use combined_fused_delined_v2 if available
    overlay_base = extras.get("combined_fused_delined_v2", extras.get("combined_fused_delined"))

    # Render top-K crops + overlays. Largest K writes the full set; smaller K
    # reuses a slice.
    n_max = max(top_k)
    top_df = df.head(n_max).copy().reset_index(drop=True)

    # Build per-K crops once into crops_top{N} directories.
    rendered_paths_by_index = [None] * len(top_df)
    for k in sorted(top_k):
        crops_dir = os.path.join(review_dir, f"crops_top{k}")
        ensure_dir(crops_dir)
        for i in range(min(k, len(top_df))):
            row = top_df.iloc[i].to_dict()
            cid = int(row["candidate_id"])
            x = int(round(float(row["x"]))); y = int(round(float(row["y"])))
            sc = float(row.get("review_priority_score", 0.0))
            fname = f"rank{i+1:04d}_id{cid:04d}_x{x:04d}_y{y:04d}_score{sc:.3f}.png"
            out_path = os.path.join(crops_dir, fname)
            _render_crop_sheet(row, images_template, crop_size, out_path)
            # Track the largest-K path as the canonical crop_path
            if k == n_max:
                rendered_paths_by_index[i] = out_path

    top_df["crop_path"] = [
        rendered_paths_by_index[i] if rendered_paths_by_index[i] else ""
        for i in range(len(top_df))
    ]

    # Per-K CSV + overlay + index
    for k in top_k:
        sub = top_df.head(k).copy().reset_index(drop=True)
        sub.to_csv(os.path.join(review_dir, f"manual_review_top{k}.csv"), index=False)
        if overlay_base is not None:
            _draw_overlay(overlay_base, sub,
                          os.path.join(review_dir, f"manual_review_overlay_top{k}.png"))
        # review_index_top{k}.html — we always also emit review_index.html (top of largest K)
        rows = sub.to_dict(orient="records")
        _write_review_index(
            os.path.join(review_dir, f"review_index_top{k}.html"), rows, missing_files,
        )

    # review_index.html — single page covering top-N (largest K)
    rows_max = top_df.to_dict(orient="records")
    _write_review_index(os.path.join(review_dir, "review_index.html"), rows_max, missing_files)

    # Manual labels template — spec columns
    label_columns = ["manual_label", "manual_confidence", "notes"]
    keep_cols = [
        "candidate_id", "source_category", "x", "y", "diameter_px",
        "polarity", "review_priority_score", "crop_path",
    ]
    keep_cols = [c for c in keep_cols if c in top_df.columns]
    template_df = top_df[keep_cols].copy()
    template_df = labels_template(template_df, label_columns)
    template_df.to_csv(os.path.join(review_dir, "manual_labels_template.csv"), index=False)

    # ALSO write a README explaining manual_label allowed values
    readme = os.path.join(review_dir, "MANUAL_LABEL_VALUES.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            "manual_label allowed values:\n"
            "  true_defect\n"
            "  suspected_defect\n"
            "  kossel_line_artifact\n"
            "  texture_artifact\n"
            "  noise\n"
            "  edge_or_border\n"
            "  uncertain\n\n"
            "manual_confidence: 1..5 (5 = high confidence)\n"
        )
    print(f"[06] Wrote manual_labels_template.csv and MANUAL_LABEL_VALUES.txt")


if __name__ == "__main__":
    main()
