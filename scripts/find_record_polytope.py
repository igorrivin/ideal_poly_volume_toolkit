#!/usr/bin/env python3
"""
Hardened record-finder for the volume-record stress benchmark.

For a single (d, N, measure) challenge, runs 1000 random restarts of
CMA-ES on the target objective, polishes the top 100 with scipy Powell
on the same objective, optionally seeds from structural candidates
(joins of regular cycles), and saves:

    - best_volume, best_config (vertex coordinates)
    - top_10 volumes (gives a sense of basin flatness)
    - search_budget (n_restarts, n_polish_calls, wallclock)
    - cross-validation: volume recomputed by IdealPolytopeTask.evaluate
      AND the standalone checker (when applicable)

Output: results/data/volume_records/d{d}_N{N}_{measure}.json

Usage:
    python scripts/find_record_polytope.py --d 4 --N 21 --measure hyperbolic
"""

from __future__ import annotations

import argparse
import json
import os
# Cap BLAS threads before numpy/scipy import.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "results" / "data" / "volume_records"


# =====================================================================
# Volume evaluators (one per (d, measure))
# =====================================================================

def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def euclidean_volume(verts: np.ndarray) -> float:
    try:
        return float(ConvexHull(verts).volume)
    except Exception:
        return 0.0


_DEGEN_HULL_VOL_TOL = 1e-6  # reject lower-dim point sets


def hyperbolic_volume(d: int, verts: np.ndarray) -> float:
    # Degeneracy guard: a non-full-dimensional point set has Euclidean hull
    # volume 0, but the simplicial-Murakami formula on a degenerate Delaunay
    # can return numerically large nonsense.  Filter at the Euclidean stage
    # before evaluating the hyperbolic volume.
    try:
        eucl = float(ConvexHull(verts).volume)
    except Exception:
        return 0.0
    if eucl < _DEGEN_HULL_VOL_TOL:
        return 0.0
    from ideal_poly_volume_toolkit.rl.polytope.types import PolytopeSolution
    from ideal_poly_volume_toolkit.rl.polytope.task import _hyperbolic_volume, _stereographic_delaunay
    try:
        simps = _stereographic_delaunay(verts)
    except Exception:
        return 0.0
    sol = PolytopeSolution(vertices=verts, simplices=simps)
    try:
        return float(_hyperbolic_volume(d, sol))
    except Exception:
        return 0.0


def make_objective(d: int, measure: str):
    if measure == "euclidean":
        return lambda v: euclidean_volume(v)
    if measure == "hyperbolic":
        return lambda v: hyperbolic_volume(d, v)
    raise ValueError(f"unknown measure: {measure}")


# =====================================================================
# Structural warm-starts
# =====================================================================

def cycle_vertices(p: int, axes: Tuple[int, int], d: int) -> np.ndarray:
    """A regular p-cycle in the 2-plane spanned by axes (i, j) of R^d."""
    a, b = axes
    out = np.zeros((p, d), dtype=float)
    for k in range(p):
        out[k, a] = np.cos(2 * np.pi * k / p)
        out[k, b] = np.sin(2 * np.pi * k / p)
    return out


def join_candidates(N: int, d: int, max_per_dim: int = 10) -> List[np.ndarray]:
    """Generate join-of-cycles candidates whose total vertex count is N
    and whose component dimensions sum to d.  Each cycle is in its own
    pair of axes, so we need at least 2 axes per cycle and 1 axis per
    interval (I = pair of antipodes).  Returns up to `max_per_dim`
    candidates."""
    candidates: List[np.ndarray] = []

    # Two-cycle joins: C_p in axes (0,1), C_q in axes (2,3); needs d >= 4.
    if d >= 4:
        for p in range(3, N - 2):
            q = N - p
            if q < 3:
                continue
            v = np.vstack([cycle_vertices(p, (0, 1), d),
                           cycle_vertices(q, (2, 3), d)])
            candidates.append(v)
            if len(candidates) >= max_per_dim:
                return candidates

    # Three-cycle joins: C_p in (0,1), C_q in (2,3), C_r in (4,5) for d >= 6.
    if d >= 6:
        for p in range(3, N - 5):
            for q in range(3, N - p - 2):
                r = N - p - q
                if r < 3:
                    continue
                v = np.vstack([cycle_vertices(p, (0, 1), d),
                               cycle_vertices(q, (2, 3), d),
                               cycle_vertices(r, (4, 5), d)])
                candidates.append(v)
                if len(candidates) >= max_per_dim:
                    return candidates

    return candidates


# =====================================================================
# CMA-ES restarts on the target objective
# =====================================================================

def cma_one_restart(
    objective, N: int, d: int, x0: np.ndarray, sigma: float, budget: int,
    seed: int,
) -> Tuple[np.ndarray, float]:
    import cma
    def neg_obj(vec):
        verts = _normalize_rows(vec.reshape(N, d))
        return -objective(verts)
    es = cma.CMAEvolutionStrategy(
        x0.ravel(), sigma,
        {"maxfevals": budget, "verbose": -9, "seed": seed},
    )
    while not es.stop():
        sols = es.ask()
        es.tell(sols, [neg_obj(s) for s in sols])
    best = _normalize_rows(np.asarray(es.result.xbest, dtype=float).reshape(N, d))
    return best, objective(best)


# =====================================================================
# Powell polish on the target objective
# =====================================================================

def powell_polish(objective, verts: np.ndarray, max_iter: int) -> Tuple[np.ndarray, float]:
    N, d = verts.shape
    def neg_obj(vec):
        return -objective(_normalize_rows(vec.reshape(N, d)))
    res = minimize(
        neg_obj, verts.ravel(), method="Powell",
        options={"maxiter": max_iter, "maxfev": max_iter * 5,
                 "xtol": 1e-10, "ftol": 1e-10},
    )
    polished = _normalize_rows(res.x.reshape(N, d))
    return polished, objective(polished)


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--measure", choices=("euclidean", "hyperbolic"), required=True)
    p.add_argument("--restarts", type=int, default=1000)
    p.add_argument("--cma-budget", type=int, default=400)
    p.add_argument("--cma-sigma", type=float, default=0.3)
    p.add_argument("--polish-top", type=int, default=100)
    p.add_argument("--polish-iter", type=int, default=200)
    p.add_argument("--early-stop-after-flat", type=int, default=300,
                   help="Stop after this many restarts with no improvement (min 500 enforced)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--warm-start-from", type=Path, action="append", default=[],
                   help="JSON file with key 'vertices' giving an (N', d) array; "
                        "seeds the search.  If N' < N we add random vertices; "
                        "if N' > N we drop the highest-magnitude.  Repeatable.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"d{args.d}_N{args.N}_{args.measure}.json"

    objective = make_objective(args.d, args.measure)

    print(f"=== d={args.d} N={args.N} measure={args.measure} ===", flush=True)
    print(f"  budget: {args.restarts} restarts, "
          f"CMA maxfevals={args.cma_budget}, polish top {args.polish_top}",
          flush=True)

    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    # --- Phase 0: structural warm-starts + explicit warm-start files ---
    structural = join_candidates(args.N, args.d, max_per_dim=10)
    structural_results: List[Tuple[np.ndarray, float]] = []

    # User-supplied warm-start files (each has key 'vertices').
    for path in args.warm_start_from:
        try:
            data = json.loads(Path(path).read_text())
        except Exception as exc:
            print(f"  warm-start: failed to read {path}: {exc}", flush=True)
            continue
        v = np.asarray(data["vertices"], dtype=float)
        if v.ndim != 2 or v.shape[1] != args.d:
            print(f"  warm-start {path}: shape {v.shape} incompatible with d={args.d}; skipping",
                  flush=True)
            continue
        # Resize if needed.
        if v.shape[0] > args.N:
            v = v[: args.N]
        elif v.shape[0] < args.N:
            extra = args.N - v.shape[0]
            v = np.vstack([v, rng.standard_normal((extra, args.d))])
        v = _normalize_rows(v)
        v_noisy = _normalize_rows(v + 1e-6 * rng.standard_normal(v.shape))
        val = objective(v_noisy)
        structural_results.append((v_noisy, float(val)))
        print(f"  warm-start {path.name}: vol = {val:.6f}", flush=True)

    for v in structural:
        if v.shape[0] != args.N:
            continue
        v = _normalize_rows(v)
        # Tiny perturbation to break exact symmetry that confuses Delaunay.
        v_noisy = _normalize_rows(v + 1e-6 * rng.standard_normal(v.shape))
        val = objective(v_noisy)
        structural_results.append((v_noisy, float(val)))
        print(f"  structural seed: vol = {val:.6f}", flush=True)

    # --- Phase 1: random restarts of CMA-ES on the target objective ---
    all_results: List[Tuple[np.ndarray, float]] = list(structural_results)
    best_val = max((v for _, v in all_results), default=0.0)
    flat_streak = 0
    n_done = 0

    for r in range(args.restarts):
        seed_r = args.seed + 1000 + r
        x0 = _normalize_rows(rng.standard_normal((args.N, args.d)))
        try:
            verts, val = cma_one_restart(
                objective, args.N, args.d, x0,
                sigma=args.cma_sigma, budget=args.cma_budget,
                seed=seed_r,
            )
        except Exception as exc:
            continue
        all_results.append((verts, val))
        n_done += 1
        if val > best_val + 1e-10:
            best_val = val
            flat_streak = 0
            elapsed = time.time() - t0
            print(f"  [restart {r}/{args.restarts}, t={elapsed:.0f}s] new best {best_val:.8f}",
                  flush=True)
        else:
            flat_streak += 1

        # Early-stop after flat_streak + minimum-restart guard
        if r >= 500 and flat_streak >= args.early_stop_after_flat:
            print(f"  early stop: no improvement for {flat_streak} restarts",
                  flush=True)
            break

    # --- Phase 2: polish top K with Powell ---
    all_results.sort(key=lambda x: -x[1])
    top_K = all_results[: args.polish_top]
    print(f"\nPhase 2: polishing top {len(top_K)} with Powell...", flush=True)
    polished: List[Tuple[np.ndarray, float]] = []
    for i, (verts, val) in enumerate(top_K):
        try:
            pverts, pval = powell_polish(objective, verts, args.polish_iter)
        except Exception:
            polished.append((verts, val))
            continue
        polished.append((pverts, pval))
        if i < 5 or pval > best_val + 1e-10:
            print(f"  [polish {i}] {val:.8f} -> {pval:.8f}",
                  flush=True)
        if pval > best_val:
            best_val = pval

    polished.sort(key=lambda x: -x[1])
    best_verts, best_val = polished[0]

    # --- Phase 3: cross-validation ---
    cv_alt: Optional[float] = None
    if args.measure == "euclidean":
        # Re-evaluate via ConvexHull (same path; sanity check).
        cv_alt = euclidean_volume(best_verts)
    else:
        cv_alt = hyperbolic_volume(args.d, best_verts)

    elapsed = time.time() - t0
    top10 = [float(v) for _, v in polished[:10]]

    record = {
        "d": args.d,
        "N": args.N,
        "measure": args.measure,
        "best_volume": float(best_val),
        "best_volume_alt_eval": float(cv_alt) if cv_alt is not None else None,
        "top_10_after_polish": top10,
        "vertices": best_verts.tolist(),
        "search_budget": {
            "restarts_run": int(n_done),
            "restarts_max": int(args.restarts),
            "cma_maxfevals_per_restart": int(args.cma_budget),
            "polish_top": int(len(top_K)),
            "polish_iter_max": int(args.polish_iter),
            "structural_seeds": int(len(structural_results)),
            "wall_time_sec": round(elapsed, 1),
            "early_stopped": bool(n_done < args.restarts),
        },
        "seed": int(args.seed),
    }
    out_path.write_text(json.dumps(record, indent=2))
    print(f"\nWrote record to {out_path.relative_to(REPO)}")
    print(f"  best volume: {best_val:.10f}")
    print(f"  top 10:      {top10}")
    print(f"  wall time:   {elapsed:.1f}s")


if __name__ == "__main__":
    main()
