#!/usr/bin/env python3
"""
Train one per-dimension PPO polytope controller on the refactored env.

Saves to `results/rl_models/ppo_d{d}_v2_final.zip` plus a sidecar
`ppo_d{d}_v2_config.json` recording the exact training configuration.

Usage:
    python scripts/train_polytope_controller.py --d 3 --steps 500000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", type=int, required=True, choices=(3, 4, 5, 6))
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--horizon", type=int, default=40)
    # --n-low default needs N-1 >= d for non-degenerate stereographic Delaunay;
    # we add margin and match the old default of 7 for d >= 5.  For d=3,4 we
    # can start as low as 5 (paper uses N in {5..30}).
    p.add_argument("--n-low", type=int, default=None,
                   help="Low end of N range; default = max(7, d+1)")
    p.add_argument("--n-high", type=int, default=25)
    p.add_argument("--tag", type=str, default="v2",
                   help="Suffix added to the zip filename (default v2)")
    args = p.parse_args()
    if args.n_low is None:
        args.n_low = max(7, args.d + 1)

    from ideal_poly_volume_toolkit.rl.polytope import make_polytope_env
    from ideal_poly_volume_toolkit.rl.training import train_controller

    save_path = REPO / "results" / "rl_models" / f"ppo_d{args.d}_{args.tag}_final.zip"
    config_path = save_path.with_suffix("").with_suffix(".json")
    config_path = save_path.parent / f"ppo_d{args.d}_{args.tag}_config.json"

    config = {
        "d": args.d,
        "total_timesteps": args.steps,
        "seed": args.seed,
        "horizon": args.horizon,
        "N_range": [args.n_low, args.n_high],
        "d_range": [args.d, args.d],
        "n_steps": 2048,
        "batch_size": 256,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "device": "cpu",
        "env_factory": "make_polytope_env",
        "commit": "run git rev-parse HEAD to recover",
    }
    print(f"[d={args.d}] training config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    t0 = time.time()

    def factory():
        return make_polytope_env(
            d_range=(args.d, args.d),
            N_range=(args.n_low, args.n_high),
            horizon=args.horizon,
            seed=args.seed,
        )

    model = train_controller(
        factory,
        total_timesteps=args.steps,
        save_path=save_path,
        seed=args.seed,
        device="cpu",
        verbose=1,
    )
    elapsed = time.time() - t0

    config["elapsed_sec"] = round(elapsed, 1)
    config["saved_at"] = str(save_path.relative_to(REPO))
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"[d={args.d}] done in {elapsed/60:.1f} min; saved {save_path.name} + sidecar")


if __name__ == "__main__":
    main()
