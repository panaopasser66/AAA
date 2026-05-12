import _bootstrap  # noqa: F401

import argparse
import os

import matplotlib
matplotlib.use("Agg")

from src.groups import get_enabled_groups, load_config, validate_group_files
from src.io_utils import ensure_dir
from src.registration import register_group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="raw")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    config = load_config(args.config)
    validate_group_files(args.raw_dir, config)
    enabled = get_enabled_groups(config)
    if not enabled:
        print("[01] No enabled groups; nothing to do.")
        return

    normalized_dir = os.path.join(args.out_dir, "normalized")
    ensure_dir(normalized_dir)

    for name, spec in enabled.items():
        files = spec.get("files", [])
        print(f"[01] Registering group '{name}' ({len(files)} files)")
        df = register_group(name, files, args.raw_dir, normalized_dir, args.out_dir, config)
        for _, row in df.iterrows():
            print(
                f"    {row['filename']}: dx={row['dx']:.2f} dy={row['dy']:.2f} "
                f"method={row['method']} ecc={row['ecc_score']!s}"
            )


if __name__ == "__main__":
    main()
