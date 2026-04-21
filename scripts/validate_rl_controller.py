#!/usr/bin/env python3
"""
Validation harness for the RL polytope controller.

Runs the 20-task eval grid described in docs/plans/rl-controller-validation.md
and compares the trained PPO controller against two baseline styles at
matched evaluation budget:

  * solo        — one inner optimizer consumes the entire budget
  * round-robin — budget split equally across all inner optimizers

For each (d, N, seed) the runner records `best_value`, `total_evals`,
and elapsed wall time per method, dumps everything to a single JSON, and
prints an aggregate summary (geometric-mean ratio, win rate).

Usage:
    python scripts/validate_rl_controller.py \\
        --model-template 'results/rl_models/ppo_d{d}_v2_final.zip' \\
        --out results/rl_validation/results.json \\
        --seeds 10

If a `{d}` model is missing the RL arm for that dimension is skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


REPO = Path(__file__).resolve().parent.parent

# Default evaluation grid (see docs/plans/rl-controller-validation.md §3).
DEFAULT_GRID: Dict[int, List[int]] = {
    3: [5, 7, 10, 15, 20],
    4: [5, 7, 10, 16, 20],
    5: [6, 7, 10, 14, 20],
    6: [7, 10, 16, 20, 24],
}

HORIZON = 40


# ---------------------------------------------------------------------------
# One-task runners
# ---------------------------------------------------------------------------

def run_rl_episode(model_path: Path, d: int, N: int, seed: int) -> Dict[str, Any]:
    """Evaluate the RL controller for one episode; returns best_value + budget."""
    from stable_baselines3 import PPO
    from ideal_poly_volume_toolkit.rl.polytope import make_polytope_env

    env = make_polytope_env(d_range=(d, d), N_range=(N, N),
                            horizon=HORIZON, seed=seed)
    model = PPO.load(str(model_path), env=env)
    t0 = time.time()
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        done = bool(term or trunc)
    return {
        "method": "rl",
        "best_value": float(info["best_value"]),
        "total_evals": int(info["total_evals"]),
        "elapsed_sec": time.time() - t0,
    }


def _baseline_solo(opt_name: str, d: int, N: int, seed: int,
                   budget: int) -> Dict[str, Any]:
    """Run ONE inner optimizer with the full `budget` evaluations from a
    random start seeded reproducibly."""
    from ideal_poly_volume_toolkit.rl.polytope import make_polytope_env

    env = make_polytope_env(d_range=(d, d), N_range=(N, N),
                            horizon=HORIZON, seed=seed)
    env.reset(seed=seed)
    task = env.task
    sol = task.copy_solution(env.root.best_solution)
    opt = next(o for o in env.inner_optimizers if o.name == opt_name)

    rng = np.random.default_rng(seed + 1_000_003)
    t0 = time.time()
    remaining = budget
    best_sol = sol
    best_val = task.evaluate(best_sol)
    evals_used = 1
    chunk = max(4, opt.default_budget)
    while remaining > 0:
        this_budget = min(chunk, remaining)
        new_sol, values = opt.run(best_sol, task, this_budget, rng)
        evals_used += len(values)
        v = task.evaluate(new_sol)
        if v > best_val:
            best_val = v
            best_sol = new_sol
        remaining -= max(1, len(values))
    return {
        "method": f"solo_{opt_name}",
        "best_value": float(best_val),
        "total_evals": int(evals_used),
        "elapsed_sec": time.time() - t0,
    }


def _baseline_round_robin(d: int, N: int, seed: int, budget: int) -> Dict[str, Any]:
    """Split the budget equally across all env inner optimizers; keep the
    best solution seen across the mix."""
    from ideal_poly_volume_toolkit.rl.polytope import make_polytope_env

    env = make_polytope_env(d_range=(d, d), N_range=(N, N),
                            horizon=HORIZON, seed=seed)
    env.reset(seed=seed)
    task = env.task
    opts = list(env.inner_optimizers)

    rng = np.random.default_rng(seed + 1_000_007)
    t0 = time.time()
    best_sol = task.copy_solution(env.root.best_solution)
    best_val = task.evaluate(best_sol)
    evals_used = 1
    per_opt_budget = max(opts[0].default_budget, budget // len(opts))
    remaining = budget
    round_i = 0
    while remaining > 0:
        opt = opts[round_i % len(opts)]
        this_budget = min(per_opt_budget, remaining)
        new_sol, values = opt.run(best_sol, task, this_budget, rng)
        evals_used += len(values)
        v = task.evaluate(new_sol)
        if v > best_val:
            best_val = v
            best_sol = new_sol
        remaining -= max(1, len(values))
        round_i += 1
    return {
        "method": "round_robin",
        "best_value": float(best_val),
        "total_evals": int(evals_used),
        "elapsed_sec": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _model_path(template: str, d: int) -> Optional[Path]:
    path = Path(template.format(d=d))
    return path if path.exists() else None


def run_one_task(d: int, N: int, seed: int, model_path: Optional[Path]
                 ) -> List[Dict[str, Any]]:
    """Run RL + all baselines on one (d, N, seed).  Matches budgets to RL."""
    rows: List[Dict[str, Any]] = []
    common = {"d": d, "N": N, "seed": seed}

    if model_path is not None:
        rl_row = run_rl_episode(model_path, d, N, seed)
        rl_row.update(common)
        rows.append(rl_row)
        budget = rl_row["total_evals"]
    else:
        # No RL model -> use a fixed-ish budget based on horizon * avg step cost.
        budget = HORIZON * 12

    # Use an env instance just to enumerate the optimizer names.
    from ideal_poly_volume_toolkit.rl.polytope import make_polytope_env
    env = make_polytope_env(d_range=(d, d), N_range=(N, N),
                            horizon=HORIZON, seed=seed)
    env.reset(seed=seed)
    opt_names = [o.name for o in env.inner_optimizers]

    for name in opt_names:
        row = _baseline_solo(name, d, N, seed, budget)
        row.update(common)
        rows.append(row)

    rr = _baseline_round_robin(d, N, seed, budget)
    rr.update(common)
    rows.append(rr)

    return rows


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute geometric-mean ratio and win rate per method."""
    # Group by (d, N, seed)
    by_task: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        key = (r["d"], r["N"], r["seed"])
        by_task.setdefault(key, []).append(r)

    # Per-task best baseline (non-rl, non-round_robin) for the "best baseline"
    # reference.
    def is_baseline(r): return r["method"].startswith("solo_")

    method_names = sorted({r["method"] for r in rows})

    # Per-task ratios
    per_method_ratios: Dict[str, List[float]] = {m: [] for m in method_names}
    per_method_wins: Dict[str, int] = {m: 0 for m in method_names}

    for key, task_rows in by_task.items():
        ref = max((r["best_value"] for r in task_rows if is_baseline(r)),
                  default=1e-12)
        ref = max(ref, 1e-12)
        vals = {r["method"]: r["best_value"] for r in task_rows}
        winner = max(vals, key=lambda m: vals[m])
        for m, v in vals.items():
            per_method_ratios[m].append(v / ref)
            if m == winner:
                per_method_wins[m] += 1

    summary: Dict[str, Any] = {}
    n_tasks = len(by_task)
    for m in method_names:
        rs = per_method_ratios[m]
        gmean = math.exp(sum(math.log(max(r, 1e-12)) for r in rs) / max(len(rs), 1))
        mean = sum(rs) / max(len(rs), 1)
        summary[m] = {
            "geomean_ratio_vs_best_solo": gmean,
            "mean_ratio_vs_best_solo": mean,
            "win_rate": per_method_wins[m] / max(n_tasks, 1),
            "n_tasks": n_tasks,
        }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-template", type=str,
                   default="results/rl_models/ppo_d{d}_v2_final.zip")
    p.add_argument("--out", type=Path,
                   default=REPO / "results" / "rl_validation" / "results.json")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--dims", type=str, default="3,4,5,6")
    p.add_argument("--limit-n", type=int, default=None,
                   help="Keep only the first <limit_n> N values per dim (smoke test)")
    args = p.parse_args()

    dims = [int(x) for x in args.dims.split(",") if x.strip()]
    grid = {d: (DEFAULT_GRID[d][:args.limit_n] if args.limit_n else DEFAULT_GRID[d])
            for d in dims if d in DEFAULT_GRID}

    rows: List[Dict[str, Any]] = []
    total_tasks = sum(len(Ns) for Ns in grid.values()) * args.seeds
    task_i = 0
    for d, Ns in grid.items():
        model_path = _model_path(args.model_template, d)
        if model_path is None:
            print(f"[warn] no model found for d={d} (tried {args.model_template.format(d=d)}); "
                  f"running baselines only")
        for N in Ns:
            for seed in range(args.seeds):
                task_i += 1
                t0 = time.time()
                task_rows = run_one_task(d, N, seed, model_path)
                rows.extend(task_rows)
                best_rl = next((r["best_value"] for r in task_rows if r["method"] == "rl"), None)
                best_solo = max((r["best_value"] for r in task_rows
                                 if r["method"].startswith("solo_")), default=0)
                print(f"[{task_i:>3}/{total_tasks}] d={d} N={N} seed={seed} "
                      f"rl={best_rl if best_rl is None else f'{best_rl:.4f}'} "
                      f"best_solo={best_solo:.4f} "
                      f"({time.time()-t0:.1f}s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = aggregate(rows)
    args.out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    try:
        where = args.out.relative_to(REPO)
    except ValueError:
        where = args.out
    print(f"\nWrote {len(rows)} rows to {where}")

    print("\n--- Aggregate summary ---")
    print(f"{'method':>22} {'geomean':>10} {'mean':>10} {'win_rate':>10}")
    for m, s in sorted(summary.items(),
                       key=lambda kv: -kv[1]["geomean_ratio_vs_best_solo"]):
        print(f"{m:>22} {s['geomean_ratio_vs_best_solo']:>10.3f} "
              f"{s['mean_ratio_vs_best_solo']:>10.3f} {s['win_rate']:>10.3f}")


if __name__ == "__main__":
    main()
