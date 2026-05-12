from __future__ import annotations

import os
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd

from .invalid_region import warp_valid_mask
from .io_utils import (
    ensure_dir,
    load_tif_float32,
    robust_rescale_for_preview,
    save_png,
)
from .normalize import match_median_mad_to_reference, percentile_normalize


def build_registration_image(img: np.ndarray, blur_sigma: float = 5.0) -> np.ndarray:
    """Smooth image for registration to reduce Kossel-line / noise influence."""
    arr = np.asarray(img, dtype=np.float32)
    if blur_sigma <= 0:
        return arr
    k = int(max(3, round(blur_sigma * 6)))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(arr, (k, k), blur_sigma, borderType=cv2.BORDER_REFLECT)


def estimate_translation_phase(
    ref_reg: np.ndarray, moving_reg: np.ndarray
) -> Tuple[float, float, float]:
    """Phase correlation. Returns (dx, dy, response) mapping moving -> ref."""
    ref = np.asarray(ref_reg, dtype=np.float32)
    mov = np.asarray(moving_reg, dtype=np.float32)
    (sx, sy), response = cv2.phaseCorrelate(ref, mov)
    return float(-sx), float(-sy), float(response)


def refine_translation_ecc(
    ref_reg: np.ndarray,
    moving_reg: np.ndarray,
    init_dx: float,
    init_dy: float,
    max_iterations: int,
    eps: float,
) -> Tuple[float, float, float, bool]:
    ref = np.asarray(ref_reg, dtype=np.float32)
    mov = np.asarray(moving_reg, dtype=np.float32)
    warp = np.array([[1.0, 0.0, float(init_dx)], [0.0, 1.0, float(init_dy)]], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(max_iterations), float(eps))
    try:
        cc, warp_out = cv2.findTransformECC(
            ref, mov, warp, cv2.MOTION_TRANSLATION, criteria, None, 5
        )
        return float(warp_out[0, 2]), float(warp_out[1, 2]), float(cc), True
    except cv2.error:
        return float(init_dx), float(init_dy), float("nan"), False


def warp_translation(
    img: np.ndarray, dx: float, dy: float, output_shape: Tuple[int, int]
) -> np.ndarray:
    H, W = output_shape
    M = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    return cv2.warpAffine(
        np.asarray(img, dtype=np.float32),
        M,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def refine_similarity_ecc(
    ref_reg: np.ndarray,
    moving_reg: np.ndarray,
    init_dx: float,
    init_dy: float,
    max_iterations: int,
    eps: float,
) -> Tuple[np.ndarray, float, bool, float, float]:
    """Refine to similarity via ECC. Returns (2x3 warp, cc, ok, rot_deg, scale)."""
    ref = np.asarray(ref_reg, dtype=np.float32)
    mov = np.asarray(moving_reg, dtype=np.float32)
    warp = np.array(
        [[1.0, 0.0, float(init_dx)], [0.0, 1.0, float(init_dy)]],
        dtype=np.float32,
    )
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(max_iterations), float(eps))
    try:
        cc, warp_out = cv2.findTransformECC(
            ref, mov, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5
        )
        a = float(warp_out[0, 0])
        b = float(warp_out[1, 0])
        rot_deg = float(np.degrees(np.arctan2(b, a)))
        scale = float(np.sqrt(a * a + b * b))
        return warp_out.astype(np.float32), float(cc), True, rot_deg, scale
    except cv2.error:
        return warp, float("nan"), False, 0.0, 1.0


def warp_affine(
    img: np.ndarray, M: np.ndarray, output_shape: Tuple[int, int]
) -> np.ndarray:
    H, W = output_shape
    return cv2.warpAffine(
        np.asarray(img, dtype=np.float32),
        np.asarray(M, dtype=np.float32),
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def warp_valid_mask_affine(shape, M: np.ndarray) -> np.ndarray:
    H, W = shape
    ones = np.ones((H, W), dtype=np.float32)
    out = cv2.warpAffine(
        ones, np.asarray(M, dtype=np.float32), (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (out > 0.5).astype(np.uint8)


def _downsample(im, target_w=480):
    if im.shape[1] <= target_w:
        return im
    scale = target_w / im.shape[1]
    return cv2.resize(im, (int(im.shape[1] * scale), int(im.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def _build_contact_sheet(
    images: List[np.ndarray], titles: List[str], cols: int = 3
) -> np.ndarray:
    if not images:
        return np.zeros((1, 1), dtype=np.uint8)
    rows = (len(images) + cols - 1) // cols
    H, W = images[0].shape[:2]
    pad = 4
    title_h = 18
    cell_h = H + title_h + pad
    cell_w = W + pad
    sheet = np.full((rows * cell_h + pad, cols * cell_w + pad), 32, dtype=np.uint8)
    for i, (img, title) in enumerate(zip(images, titles)):
        r, c = divmod(i, cols)
        y0 = pad + r * cell_h + title_h
        x0 = pad + c * cell_w
        prev = robust_rescale_for_preview(img) if img.dtype != np.uint8 else img
        sheet[y0 : y0 + H, x0 : x0 + W] = prev[:H, :W]
        cv2.putText(sheet, title, (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1, cv2.LINE_AA)
    return sheet


def _build_raw_vs_aligned_sheet(
    raw_crops: List[np.ndarray],
    aligned_crops: List[np.ndarray],
    titles: List[str],
) -> np.ndarray:
    """Two-column sheet: raw crop on left, aligned crop on right per row."""
    n = len(raw_crops)
    if n == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    H, W = raw_crops[0].shape[:2]
    pad = 6
    title_h = 18
    cell_h = H + title_h + pad
    cell_w = W + pad
    sheet = np.full((n * cell_h + pad, 2 * cell_w + pad), 32, dtype=np.uint8)
    for i, (r, a, t) in enumerate(zip(raw_crops, aligned_crops, titles)):
        y0 = pad + i * cell_h + title_h
        sheet[y0 : y0 + H, pad : pad + W] = robust_rescale_for_preview(r)
        sheet[y0 : y0 + H, pad + cell_w : pad + cell_w + W] = robust_rescale_for_preview(a)
        cv2.putText(sheet, f"{t}  raw", (pad, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1, cv2.LINE_AA)
        cv2.putText(sheet, f"{t}  aligned", (pad + cell_w, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1, cv2.LINE_AA)
    return sheet


def register_group(
    group_name: str,
    group_files: List[str],
    raw_dir: str,
    normalized_dir: str,
    out_dir: str,
    config: dict,
) -> pd.DataFrame:
    """Register all images in group to reference. Save aligned data + extended QC."""
    ref_name = config.get("reference_image", "0000.tif")
    if ref_name not in group_files:
        raise ValueError(
            f"Group {group_name} does not contain reference image {ref_name}. "
            f"Files: {group_files}"
        )

    norm_cfg = config.get("normalization", {}) or {}
    reg_cfg = dict(config.get("registration", {}) or {})
    overrides = (config.get("group_registration_overrides", {}) or {}).get(group_name, {}) or {}
    reg_cfg.update(overrides)

    # transform model from per-group spec; fall back to translation
    group_spec = (config.get("groups", {}) or {}).get(group_name, {}) or {}
    transform_model = str(group_spec.get("transform_model", "translation")).lower()

    p_low = float(norm_cfg.get("percentile_low", 0.5))
    p_high = float(norm_cfg.get("percentile_high", 99.5))
    match_ref = bool(norm_cfg.get("match_reference_median_mad", True))
    blur_sigma = float(reg_cfg.get("registration_blur_sigma", 5.0))
    use_phase = bool(reg_cfg.get("use_phase_correlation_init", True))
    use_ecc = bool(reg_cfg.get("use_ecc_refine", True))
    ecc_max_iter = int(reg_cfg.get("ecc_max_iterations", 300))
    ecc_eps = float(reg_cfg.get("ecc_eps", 1e-6))
    allow_fallback = bool(reg_cfg.get("allow_ecc_failure_fallback", True))
    max_shift = float(reg_cfg.get("max_abs_shift_px", 200))
    qc_box = reg_cfg.get("qc_crop_box", None)

    aligned_dir = os.path.join(out_dir, "aligned", group_name)
    qc_dir = os.path.join(out_dir, "registration_qc")
    norm_out_dir = normalized_dir
    ensure_dir(aligned_dir)
    ensure_dir(qc_dir)
    ensure_dir(norm_out_dir)

    ref_path = os.path.join(raw_dir, ref_name)
    ref_raw = load_tif_float32(ref_path)
    ref_norm = percentile_normalize(ref_raw, p_low, p_high)
    ref_reg = build_registration_image(ref_norm, blur_sigma)
    H, W = ref_norm.shape

    ref_npy = os.path.join(norm_out_dir, ref_name.replace(".tif", ".npy"))
    if not os.path.isfile(ref_npy):
        np.save(ref_npy, ref_norm.astype(np.float32))
        save_png(
            os.path.join(norm_out_dir, ref_name.replace(".tif", ".png")),
            robust_rescale_for_preview(ref_norm),
        )

    rows = []
    aligned_imgs: List[np.ndarray] = []
    raw_imgs: List[np.ndarray] = []
    before_imgs: List[np.ndarray] = []
    diff_imgs: List[np.ndarray] = []
    valid_masks: List[np.ndarray] = []
    titles: List[str] = []

    for fname in group_files:
        path = os.path.join(raw_dir, fname)
        raw = load_tif_float32(path)
        norm = percentile_normalize(raw, p_low, p_high)
        if match_ref and fname != ref_name:
            norm = match_median_mad_to_reference(norm, ref_norm)

        npy_path = os.path.join(norm_out_dir, fname.replace(".tif", ".npy"))
        png_path = os.path.join(norm_out_dir, fname.replace(".tif", ".png"))
        if not os.path.isfile(npy_path):
            np.save(npy_path, norm.astype(np.float32))
            save_png(png_path, robust_rescale_for_preview(norm))

        phase_score = float("nan")
        rotation_deg = 0.0
        scale_factor = 1.0
        if fname == ref_name:
            dx_p, dy_p = 0.0, 0.0
            dx, dy = 0.0, 0.0
            ecc_score = float("nan")
            ecc_ok = True
            method = "reference"
            M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        else:
            mov_reg = build_registration_image(norm, blur_sigma)
            if use_phase:
                dx_p, dy_p, phase_score = estimate_translation_phase(ref_reg, mov_reg)
            else:
                dx_p, dy_p = 0.0, 0.0

            if transform_model == "translation":
                if use_ecc:
                    dx, dy, ecc_score, ecc_ok = refine_translation_ecc(
                        ref_reg, mov_reg, dx_p, dy_p, ecc_max_iter, ecc_eps
                    )
                    if not ecc_ok:
                        if allow_fallback:
                            dx, dy = dx_p, dy_p
                            method = "phase_only_after_ecc_fail"
                        else:
                            raise RuntimeError(f"ECC failed for {fname}")
                    else:
                        method = "phase+ecc"
                else:
                    dx, dy = dx_p, dy_p
                    ecc_score = float("nan")
                    ecc_ok = True
                    method = "phase_only"
                M = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
            elif transform_model in ("similarity", "euclidean"):
                # ECC with MOTION_EUCLIDEAN — true similarity (rot+scale) is not
                # supported directly by findTransformECC; Euclidean is the
                # closest stable choice (rot + translation, unit scale). The
                # API permits scale via affine, but we prefer the smaller model
                # because we expect only small rotations between angles.
                M, ecc_score, ecc_ok, rotation_deg, scale_factor = refine_similarity_ecc(
                    ref_reg, mov_reg, dx_p, dy_p, ecc_max_iter, ecc_eps
                )
                dx = float(M[0, 2]); dy = float(M[1, 2])
                if not ecc_ok:
                    if allow_fallback:
                        dx, dy = dx_p, dy_p
                        rotation_deg = 0.0; scale_factor = 1.0
                        M = np.array(
                            [[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]],
                            dtype=np.float32,
                        )
                        method = "phase_only_after_ecc_fail(similarity)"
                    else:
                        raise RuntimeError(f"Similarity ECC failed for {fname}")
                else:
                    method = "phase+ecc(euclidean)"
            else:
                raise ValueError(f"Unknown transform_model: {transform_model!r}")

        warned = False
        if max(abs(dx), abs(dy)) > max_shift:
            print(f"[WARN] {group_name}/{fname}: shift ({dx:.2f},{dy:.2f}) exceeds max {max_shift}")
            warned = True

        aligned = warp_affine(norm, M, (H, W))
        valid_mask = warp_valid_mask_affine((H, W), M)
        aligned_path = os.path.join(aligned_dir, fname.replace(".tif", ".npy"))
        aligned_png = os.path.join(aligned_dir, fname.replace(".tif", ".png"))
        valid_path = os.path.join(aligned_dir, fname.replace(".tif", "_valid.npy"))
        np.save(aligned_path, aligned.astype(np.float32))
        np.save(valid_path, valid_mask.astype(np.uint8))
        save_png(aligned_png, robust_rescale_for_preview(aligned))

        diff_before = np.abs(norm - ref_norm)
        diff_after = np.abs(aligned - ref_norm)

        rows.append(
            {
                "group": group_name,
                "filename": fname,
                "transform_model": transform_model,
                "dx_phase": dx_p,
                "dy_phase": dy_p,
                "dx": dx,
                "dy": dy,
                "rotation_deg": float(rotation_deg),
                "scale": float(scale_factor),
                "phase_score": phase_score,
                "ecc_score": ecc_score,
                "ecc_ok": ecc_ok,
                "method": method,
                "shift_exceeds_max": warned,
                "diff_before_mean": float(diff_before.mean()),
                "diff_after_mean": float(diff_after.mean()),
                "aligned_npy": aligned_path,
                "aligned_png": aligned_png,
                "valid_npy": valid_path,
                "warp_M00": float(M[0, 0]), "warp_M01": float(M[0, 1]), "warp_M02": float(M[0, 2]),
                "warp_M10": float(M[1, 0]), "warp_M11": float(M[1, 1]), "warp_M12": float(M[1, 2]),
            }
        )

        aligned_imgs.append(aligned)
        raw_imgs.append(norm)
        before_imgs.append(diff_before)
        diff_imgs.append(diff_after)
        valid_masks.append(valid_mask)
        titles.append(fname)

    df = pd.DataFrame(rows)
    transforms_csv = os.path.join(qc_dir, f"{group_name}_transforms.csv")
    df.to_csv(transforms_csv, index=False)

    # 1) before / after diff contact sheets — separate files
    diff_before_small = [_downsample(im) for im in before_imgs]
    diff_after_small = [_downsample(im) for im in diff_imgs]
    save_png(
        os.path.join(qc_dir, f"{group_name}_diff_before.png"),
        _build_contact_sheet(diff_before_small, titles, cols=min(3, len(titles))),
    )
    save_png(
        os.path.join(qc_dir, f"{group_name}_diff_after.png"),
        _build_contact_sheet(diff_after_small, titles, cols=min(3, len(titles))),
    )

    # 2) Aligned contact sheet + interleaved before/after diff (legacy)
    aligned_small = [_downsample(im) for im in aligned_imgs]
    save_png(
        os.path.join(qc_dir, f"{group_name}_aligned_contact_sheet.png"),
        _build_contact_sheet(aligned_small, titles, cols=min(3, len(aligned_small))),
    )
    interleaved = []
    interleaved_titles = []
    for i, fname in enumerate(group_files):
        interleaved.append(diff_before_small[i])
        interleaved_titles.append(f"{fname} | before")
        interleaved.append(diff_after_small[i])
        interleaved_titles.append(f"{fname} | after")
    save_png(
        os.path.join(qc_dir, f"{group_name}_before_after_diff_contact.png"),
        _build_contact_sheet(interleaved, interleaved_titles, cols=2),
    )

    # 3) Aligned-stack std / MAD per group
    stack = np.stack(aligned_imgs, axis=0).astype(np.float32)
    aligned_std = stack.std(axis=0).astype(np.float32)
    aligned_median = np.median(stack, axis=0).astype(np.float32)
    aligned_mad = np.median(np.abs(stack - aligned_median[None, ...]), axis=0).astype(np.float32)
    np.save(os.path.join(qc_dir, f"{group_name}_aligned_std.npy"), aligned_std)
    np.save(os.path.join(qc_dir, f"{group_name}_aligned_mad.npy"), aligned_mad)
    save_png(
        os.path.join(qc_dir, f"{group_name}_aligned_std.png"),
        robust_rescale_for_preview(aligned_std, 0.0, 99.5),
    )
    save_png(
        os.path.join(qc_dir, f"{group_name}_aligned_mad.png"),
        robust_rescale_for_preview(aligned_mad, 0.0, 99.5),
    )

    # 4) Per-group valid_region mask (intersection of all frame warp masks)
    valid_combined = np.ones((H, W), dtype=np.uint8)
    for vm in valid_masks:
        valid_combined &= vm
    np.save(os.path.join(qc_dir, f"{group_name}_valid_region.npy"), valid_combined.astype(np.uint8))
    save_png(
        os.path.join(qc_dir, f"{group_name}_valid_region.png"),
        (valid_combined * 255).astype(np.uint8),
    )

    # 5) Raw vs aligned side-by-side QC crops (center 700×700 by default)
    if qc_box is None or len(qc_box) != 4:
        cx_c, cy_c = W // 2, H // 2
        half = min(350, W // 2 - 10, H // 2 - 10)
        x0, y0, x1, y1 = cx_c - half, cy_c - half, cx_c + half, cy_c + half
    else:
        x0, y0, x1, y1 = [int(v) for v in qc_box]
        x0 = max(0, x0); y0 = max(0, y0); x1 = min(W, x1); y1 = min(H, y1)
    raw_crops = [im[y0:y1, x0:x1] for im in raw_imgs]
    aligned_crops = [im[y0:y1, x0:x1] for im in aligned_imgs]
    save_png(
        os.path.join(qc_dir, f"{group_name}_raw_vs_aligned.png"),
        _build_raw_vs_aligned_sheet(raw_crops, aligned_crops, titles),
    )

    return df
