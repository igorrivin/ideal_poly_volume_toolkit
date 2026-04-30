#!/usr/bin/env python3
"""
Grader shim for the volume-record benchmark (blind or open variant).

Auto-selects the ground-truth file based on `--variant` (default
`blind`).  Forwards everything else to the unified checker
`check_volume_lb_response.py` whose grading logic is identical:
recompute volume from submitted vertices, compare to `our_record` in
the ground-truth file, classify as strong/pass/fail.

Usage:
    python scripts/check_volume_record_response.py response.json
    python scripts/check_volume_record_response.py response.json --variant open
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("response", type=Path)
    p.add_argument("--variant", choices=("blind", "open"), default="blind")
    p.add_argument("--verify", action="store_true",
                   help="passed through to the underlying checker")
    args = p.parse_args()

    suffix = "_open" if args.variant == "open" else ""
    gt = REPO / "results" / "data" / f"llm_challenges_volume_record{suffix}" / "ground_truth.json"
    if not gt.exists():
        sys.exit(f"ground-truth file not found: {gt}")

    checker = REPO / "scripts" / "check_volume_lb_response.py"
    cmd = [sys.executable, str(checker), str(args.response),
           "--ground-truth", str(gt)]
    if args.verify:
        cmd.append("--verify")
    os.execv(cmd[0], cmd)


if __name__ == "__main__":
    main()
