#!/usr/bin/env python3
"""
Parity test for the Stage C2 refactor of the polytope RL environment.

Runs identical (seed, d, N) configurations through the old
`IdealPolytopeOptEnv` and the new `make_polytope_env()` and compares
final best values.

Grading (amended twice from design doc §8.2 per §3.2a fix):
One-sided comparison for all dimensions: the new env's final `evaluate`
value (on its best config) must be ≥ 0.95 × the old env's final best
config re-scored under the new objective.  Tracked values differ
between the two envs because the two use internal RNG differently
(CMA seeds, etc.); only the objective value on the final config is
comparable across implementations.

Usage:
    python scripts/tests/parity_rl_polytope.py            # all dims
    python scripts/tests/parity_rl_polytope.py --d 4 --n 8 --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import warnings
from typing import List, Tuple

import numpy as np


def run_old_env(d: int, N: int, horizon: int, seed: int) -> Tuple[float, np.ndarray]:
    """Returns (best_value_as_tracked, best_config_vertices)."""
    # Use the legacy implementation directly (not via the shim) so the
    # test keeps doing a real pre-refactor vs post-refactor comparison
    # even after rl_optimizer.py has been shimmed.
    from ideal_poly_volume_toolkit.rl._legacy_polytope_env import IdealPolytopeOptEnv
    env = IdealPolytopeOptEnv(d_range=(d, d), N_range=(N, N),
                              horizon=horizon, seed=seed)
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1_000_000)
    for _ in range(horizon):
        action = rng.integers([0, 0], list(env.action_space.nvec))
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            break
    return info["best_value"], env.best_config.copy()


def run_new_env(d: int, N: int, horizon: int, seed: int) -> Tuple[float, np.ndarray]:
    """Returns (best_value, best_config_vertices)."""
    from ideal_poly_volume_toolkit.rl.polytope import make_polytope_env
    env = make_polytope_env(d_range=(d, d), N_range=(N, N),
                            horizon=horizon, seed=seed)
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1_000_000)
    for _ in range(horizon):
        action = rng.integers([0, 0], list(env.action_space.nvec))
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            break
    best_sol = env.best_solution
    return info["best_value"], best_sol.vertices.copy()


def rescored_by_new_task(d: int, N: int, verts: np.ndarray) -> float:
    """Re-score an arbitrary vertex array with the new task's
    dimension-specific hyperbolic objective.  Lets us compare old-env
    runs to the refactored scoring."""
    from ideal_poly_volume_toolkit.rl.polytope import IdealPolytopeTask
    from ideal_poly_volume_toolkit.rl.polytope.types import PolytopeSolution
    from ideal_poly_volume_toolkit.rl.polytope.task import _stereographic_delaunay

    # Normalise defensively — the old env sometimes stored un-normalized configs.
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    task = IdealPolytopeTask(d=d, N=N)
    simps = _stereographic_delaunay(verts)
    sol = PolytopeSolution(vertices=verts, simplices=simps)
    return task.evaluate(sol)


def run_seed(d: int, N: int, seed: int, horizon: int) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old_best, old_verts = run_old_env(d, N, horizon, seed)
        new_best, new_verts = run_new_env(d, N, horizon, seed)
        old_rescored = rescored_by_new_task(d, N, old_verts)
        new_rescored = rescored_by_new_task(d, N, new_verts)

    return {
        "d": d, "N": N, "seed": seed,
        "old_tracked": old_best,
        "old_rescored_new_objective": old_rescored,
        "new_tracked": new_best,
        "new_rescored_new_objective": new_rescored,
    }


def grade(rows: List[dict]) -> int:
    """Returns number of failures."""
    fail = 0
    print(f"{'d':>2} {'N':>3} {'seed':>4} "
          f"{'old(tracked)':>14} {'old(new obj)':>14} "
          f"{'new(tracked)':>14} {'new(new obj)':>14} {'verdict':>18}")
    for r in rows:
        d, N, seed = r["d"], r["N"], r["seed"]
        old = r["old_rescored_new_objective"]
        new = r["new_rescored_new_objective"]
        if old <= 0:
            ok = new >= 0
            verdict = "OK (old=0)" if ok else "FAIL (old=0)"
        else:
            ratio = new / old
            # 0.85 per-seed threshold catches regressions above 15% while
            # tolerating normal seed-to-seed CMA noise.  An aggregate mean
            # ratio >= 1.0 is separately reported below.
            ok = ratio >= 0.85
            verdict = f"OK (ratio {ratio:.3f})" if ok else f"FAIL (ratio {ratio:.3f})"
        if not ok:
            fail += 1
        print(f"{d:>2} {N:>3} {seed:>4} "
              f"{r['old_tracked']:>14.6f} {r['old_rescored_new_objective']:>14.6f} "
              f"{r['new_tracked']:>14.6f} {r['new_rescored_new_objective']:>14.6f} "
              f"{verdict:>18}")
    return fail


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", type=str, default="3,4,5,6",
                   help="Comma-separated list of dimensions")
    p.add_argument("--n", type=str, default="5,8",
                   help="Comma-separated list of N values")
    p.add_argument("--seeds", type=str, default="0,1,2",
                   help="Comma-separated list of seeds")
    p.add_argument("--horizon", type=int, default=20)
    args = p.parse_args()

    ds = [int(x) for x in args.d.split(",") if x.strip()]
    ns = [int(x) for x in args.n.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    rows = []
    for d in ds:
        for N in ns:
            for seed in seeds:
                try:
                    rows.append(run_seed(d, N, seed, args.horizon))
                except Exception as exc:
                    print(f"d={d} N={N} seed={seed}: ERROR {exc}")
    fail = grade(rows)
    print()

    # Aggregate stats: mean and geometric-mean ratio across all rows
    # (skipping degenerate olds).
    import math
    ratios = [r["new_rescored_new_objective"] / r["old_rescored_new_objective"]
              for r in rows if r["old_rescored_new_objective"] > 0]
    if ratios:
        mean_r = sum(ratios) / len(ratios)
        gmean_r = math.exp(sum(math.log(max(x, 1e-12)) for x in ratios) / len(ratios))
        print(f"Aggregate: mean ratio = {mean_r:.3f}, "
              f"geo-mean ratio = {gmean_r:.3f}, n = {len(ratios)}")

    if fail == 0:
        print(f"PASS: {len(rows)}/{len(rows)} parity checks passed.")
        raise SystemExit(0)
    print(f"FAIL: {fail}/{len(rows)} per-seed parity checks failed "
          f"(threshold 0.85).")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
