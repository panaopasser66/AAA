from __future__ import annotations

import os
from typing import Dict

import yaml


def load_config(config_path: str) -> dict:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_enabled_groups(config: dict) -> Dict[str, dict]:
    groups = config.get("groups", {}) or {}
    return {
        name: spec
        for name, spec in groups.items()
        if spec and spec.get("enabled", False)
    }


def validate_group_files(raw_dir: str, config: dict) -> None:
    """Ensure all files in enabled groups exist in raw_dir."""
    enabled = get_enabled_groups(config)
    missing = []
    for name, spec in enabled.items():
        for fname in spec.get("files", []) or []:
            path = os.path.join(raw_dir, fname)
            if not os.path.isfile(path):
                missing.append(f"{name}: {path}")
    ref = config.get("reference_image")
    if ref:
        ref_path = os.path.join(raw_dir, ref)
        if not os.path.isfile(ref_path):
            missing.append(f"reference_image: {ref_path}")
    if missing:
        raise FileNotFoundError(
            "Missing required raw TIFF files:\n  " + "\n  ".join(missing)
        )
