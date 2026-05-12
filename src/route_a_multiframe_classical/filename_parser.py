from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from typing import List

import pandas as pd


_TOKEN_RE = re.compile(r"-?\d")


@dataclass
class MotionInfo:
    filename: str
    stem: str
    axis1: int
    axis2: int
    axis3: int
    axis4: int


def _parse_axes_from_stem(stem: str) -> List[int]:
    tokens = _TOKEN_RE.findall(stem)
    if len(tokens) != 4:
        raise ValueError(
            f"Filename stem '{stem}' produced {len(tokens)} axis tokens "
            f"({tokens}). Expected exactly 4. Use names like 0000, 0-100, 000-1, -2000."
        )
    return [int(t) for t in tokens]


def parse_motion_filename(filename: str) -> MotionInfo:
    """Parse names like 0000.tif, 0-100.tif, 000-1.tif, -2000.tif."""
    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]
    axes = _parse_axes_from_stem(stem)
    return MotionInfo(
        filename=base,
        stem=stem,
        axis1=axes[0],
        axis2=axes[1],
        axis3=axes[2],
        axis4=axes[3],
    )


def scan_raw_dir(raw_dir: str) -> pd.DataFrame:
    """Return metadata dataframe with filename, path, axis1..axis4."""
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")
    rows = []
    for name in sorted(os.listdir(raw_dir)):
        if not name.lower().endswith((".tif", ".tiff")):
            continue
        path = os.path.join(raw_dir, name)
        try:
            info = parse_motion_filename(name)
        except ValueError as e:
            raise ValueError(f"Cannot parse {name}: {e}") from e
        row = asdict(info)
        row["path"] = path
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No TIFF files found in {raw_dir}")
    df = pd.DataFrame(rows)
    df = df[["filename", "stem", "axis1", "axis2", "axis3", "axis4", "path"]]
    return df.reset_index(drop=True)


# Smoke tests
if __name__ == "__main__":
    cases = {
        "0000.tif": (0, 0, 0, 0),
        "0100.tif": (0, 1, 0, 0),
        "0-100.tif": (0, -1, 0, 0),
        "0001.tif": (0, 0, 0, 1),
        "000-1.tif": (0, 0, 0, -1),
        "1000.tif": (1, 0, 0, 0),
        "-1000.tif": (-1, 0, 0, 0),
        "2000.tif": (2, 0, 0, 0),
        "-2000.tif": (-2, 0, 0, 0),
        "0010.tif": (0, 0, 1, 0),
        "00-10.tif": (0, 0, -1, 0),
    }
    for name, expected in cases.items():
        info = parse_motion_filename(name)
        got = (info.axis1, info.axis2, info.axis3, info.axis4)
        assert got == expected, f"{name}: expected {expected}, got {got}"
    print("filename_parser self-test OK")
