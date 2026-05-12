"""run_stage3.py — Stage 3A + 3B only.

Re-uses outputs from Stage 1.3 / Stage 2 (must already exist under outputs/).
"""

import _bootstrap  # noqa: F401

import argparse
import os
import subprocess
import sys
import time


STEPS = [
    ("06_rank_manual_review_candidates.py",
     ["--config", "{config}", "--out_dir", "{out_dir}"]),
    ("07_line_removal_baseline.py",
     ["--raw_dir", "{raw_dir}", "--config", "{config}", "--out_dir", "{out_dir}"]),
    ("08_detect_candidates_on_cleaned.py",
     ["--config", "{config}", "--out_dir", "{out_dir}"]),
    ("05_make_report.py",
     ["--config", "{config}", "--out_dir", "{out_dir}"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="raw")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    subs = {"raw_dir": args.raw_dir, "config": args.config, "out_dir": args.out_dir}

    for script, argv_template in STEPS:
        script_path = os.path.join(scripts_dir, script)
        cmd = [sys.executable, script_path] + [a.format(**subs) for a in argv_template]
        print(f"\n=== Running {script} ===")
        print("  $", " ".join(cmd))
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=os.path.dirname(scripts_dir)).returncode
        elapsed = time.time() - t0
        if rc != 0:
            print(f"[run_stage3] FAILED at {script} (rc={rc}, {elapsed:.1f}s)")
            sys.exit(rc)
        print(f"[run_stage3] {script} OK ({elapsed:.1f}s)")
    print("\n[run_stage3] Stage 3 finished.")


if __name__ == "__main__":
    main()
