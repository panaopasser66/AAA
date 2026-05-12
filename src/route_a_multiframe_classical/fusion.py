from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np


def load_aligned_stack(
    aligned_dir: str, group_name: str, filenames: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Load aligned npy stack with shape [N,H,W]."""
    group_dir = os.path.join(aligned_dir, group_name)
    arrs = []
    used = []
    for fname in filenames:
        npy_path = os.path.join(group_dir, fname.replace(".tif", ".npy"))
        if not os.path.isfile(npy_path):
            raise FileNotFoundError(f"Aligned npy missing: {npy_path}")
        arrs.append(np.load(npy_path).astype(np.float32))
        used.append(fname)
    stack = np.stack(arrs, axis=0)
    return stack, used


def fuse_stack(stack: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute median/mean/std/MAD of stack along axis 0."""
    if stack.ndim != 3:
        raise ValueError(f"Expected [N,H,W], got {stack.shape}")
    median = np.median(stack, axis=0).astype(np.float32)
    mean = stack.mean(axis=0).astype(np.float32)
    std = stack.std(axis=0).astype(np.float32)
    mad = np.median(np.abs(stack - median[None, ...]), axis=0).astype(np.float32)
    return {"median": median, "mean": mean, "std": std, "mad": mad}


def combine_group_fusions(
    group_fused_images: List[np.ndarray], method: str = "median"
) -> np.ndarray:
    """Combine per-group fused images into a single fused image."""
    if not group_fused_images:
        raise ValueError("combine_group_fusions: empty list")
    stack = np.stack(group_fused_images, axis=0).astype(np.float32)
    if method == "median":
        return np.median(stack, axis=0).astype(np.float32)
    if method == "mean":
        return stack.mean(axis=0).astype(np.float32)
    if method == "min":
        return stack.min(axis=0).astype(np.float32)
    raise ValueError(f"Unknown combine method: {method}")
