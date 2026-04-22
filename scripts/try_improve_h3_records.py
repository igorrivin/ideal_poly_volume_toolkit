#!/usr/bin/env python3
"""
Attempt to improve the paper H^3 volume records with the refactored harness.

The paper used 100 angle-optimization restarts per N (for N <= 12).  We
run many more restarts of scipy Powell + random initialization and report
the best volume found per N alongside the paper value.

If we beat the paper anywhere we also save the winning vertex coordinates
(in both R^3 and complex) so the record can be updated.

Usage:
    python scripts/try_improve_h3_records.py --Ns 11,12 --restarts 500
"""

from __future__ import annotations

import argparse
import json
import os
# Cap BLAS threads before importing numpy/scipy.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


REPO = Path(__file__).resolve().parent.parent

# Paper Table 1 values (paper/maximal_ideal_polyhedra.tex lines 353-367).
PAPER_RECORDS: Dict[int, float] = {
    4: 1.014942, 5: 2.029883, 6: 3.663862, 7: 4.986773,
    8: 6.488469, 9: 8.162538, 10: 9.839315,
    11: 11.449290, 12: 13.529628,
    20: 27.699, 30: 46.486, 50: 84.971, 70: 123.893, 100: 182.671,
}


def try_one_N(N: int, restarts: int, seed_base: int) -> Tuple[float, np.ndarray]:
    """Run `restarts` CMA-ES / Powell attempts and return the best vol + verts."""
    from ideal_poly_volume_toolkit.rl.polytope.task import (
        _h3_volume, _random_on_sphere, _normalize_rows,
    )
    from ideal_poly_volume_toolkit.rl.polytope.optimizers import LBFGSHyperbolicD3
    from ideal_poly_volume_toolkit.rl.polytope import IdealPolytopeTask, PolytopeSolution
    from ideal_poly_volume_toolkit.rl.polytope.task import _stereographic_delaunay
    from ideal_poly_volume_toolkit.rl.inner_optimizers import CMAESOptimizer

    task = IdealPolytopeTask(d=3, N=N)
    lbfgs = LBFGSHyperbolicD3(steps=15)  # more steps than the default of 5
    cma = CMAESOptimizer(sigma=0.3, use_surrogate=False)  # hyperbolic objective

    best_vol = 0.0
    best_verts = None
    t0 = time.time()

    for r in range(restarts):
        rng = np.random.default_rng(seed_base + r)
        verts = _random_on_sphere(N, 3, rng)
        try:
            simps = _stereographic_delaunay(verts)
        except Exception:
            continue
        sol = PolytopeSolution(vertices=verts, simplices=simps)

        # Phase 1: short CMA-ES on hyperbolic objective
        try:
            sol2, _ = cma.run(sol, task, 200, rng)
        except Exception:
            sol2 = sol

        # Phase 2: polish with LBFGS (scipy Powell internally for d=3)
        try:
            sol3, _ = lbfgs.run(sol2, task, 20, rng)
        except Exception:
            sol3 = sol2

        v = _h3_volume(sol3.vertices)
        if v > best_vol:
            best_vol = v
            best_verts = sol3.vertices.copy()
            elapsed = time.time() - t0
            print(f"  [N={N} r={r}] new best: {best_vol:.6f} "
                  f"(paper: {PAPER_RECORDS[N]:.6f}, ratio {best_vol/PAPER_RECORDS[N]:.4f}) "
                  f"[{elapsed:.1f}s]", flush=True)

    return best_vol, best_verts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--Ns", type=str, default="11,12",
                   help="Comma-separated N values to attempt")
    p.add_argument("--restarts", type=int, default=200,
                   help="Restart count per N")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path,
                   default=REPO / "results" / "data" / "h3_record_improvements.json")
    args = p.parse_args()

    Ns = [int(x) for x in args.Ns.split(",")]

    summary = []
    for N in Ns:
        paper_v = PAPER_RECORDS.get(N)
        print(f"\n=== N={N} (paper: {paper_v}) ===", flush=True)
        best_v, best_verts = try_one_N(N, args.restarts, args.seed)
        improved = paper_v is not None and best_v > paper_v * (1 + 1e-6)
        row = {
            "N": N,
            "best_volume_found": best_v,
            "paper_record": paper_v,
            "ratio": best_v / paper_v if paper_v else None,
            "improved": improved,
            "restarts": args.restarts,
        }
        if best_verts is not None:
            row["best_vertices_S2"] = best_verts.tolist()
        summary.append(row)
        status = "*** IMPROVED ***" if improved else "did not improve"
        print(f"[N={N}] best {best_v:.6f} vs paper {paper_v}  {status}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    try:
        where = args.out.relative_to(REPO)
    except ValueError:
        where = args.out
    print(f"\nWrote {len(summary)} entries to {where}")

    n_improved = sum(1 for r in summary if r["improved"])
    print(f"\nImproved {n_improved}/{len(summary)} paper records.")
    sys.exit(0 if n_improved > 0 else 0)  # no-op failures are OK


if __name__ == "__main__":
    main()
