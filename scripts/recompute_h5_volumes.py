#!/usr/bin/env python3
"""
Recompute H^5 hyperbolic volumes of the volume-optimal d=5 ideal polytopes.

The optimization was done (PPO models for d=5 live in results/rl_models/ and
the paper's Table 5 reports the hyperbolic volumes for N in {7,8,9,10,12,14,
16,20}). Only the Euclidean-optimum vertex coordinates were persisted to
disk, in results/data/optimal_d5_d6_symmetry_analysis.json.

Per the paper (§7.7), for d=5 the CMA-ES Euclidean optimum coincides with
the hyperbolic optimum to full precision -- so the stored configurations
ARE the H^5-optimal ones. This script recomputes the hyperbolic volumes
from them via the Schläfli path integral and writes a proper
`ideal_H5_polytope_results.json` matching the H^6 schema, plus a
comparison against the paper's Table 5.

Usage
-----
    python scripts/recompute_h5_volumes.py            # all d=5 entries
    python scripts/recompute_h5_volumes.py --max-N 12 # stop at N=12
    python scripts/recompute_h5_volumes.py --only 6,7,8
    python scripts/recompute_h5_volumes.py --face-samples 4096 --steps 16   # faster, noisier
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from ideal_poly_volume_toolkit.h5_volume_torch import (
    schlafli_volume_ideal_polytope,
    triangulate_S4,
)

REPO = Path(__file__).resolve().parent.parent
INPUT_JSON = REPO / "results" / "data" / "optimal_d5_d6_symmetry_analysis.json"
OUTPUT_JSON = REPO / "results" / "data" / "ideal_H5_polytope_results.json"

# Paper Table 5 (maximal_ideal_polyhedra.pdf, pp. 25-26).  Values are quoted
# to the precision printed in the paper; we cross-check against these.
PAPER_TABLE_5 = {
    7:  0.13123,
    8:  0.24414,
    9:  0.41394,
    10: 0.69954,
    12: 1.13886,
    14: 1.64660,
    16: 2.23211,
    20: 3.75231,
}


def count_hull_facets(vertices_S4: np.ndarray) -> int:
    """Number of facets of the convex hull in R^5."""
    try:
        hull = ConvexHull(vertices_S4)
        return int(len(hull.simplices))
    except Exception:
        return -1


def recompute_one(N: int, vertices: np.ndarray, *, face_samples: int,
                  steps: int, seed: int) -> dict:
    """Run the Schläfli path integral for one configuration."""
    t0 = time.time()
    simplices = triangulate_S4(vertices)
    n_simplices = len(simplices)
    hyp_vol = schlafli_volume_ideal_polytope(
        vertices,
        simplices=simplices,
        face_samples=face_samples,
        steps=steps,
        seed=seed,
    )
    elapsed = time.time() - t0
    return {
        "N": int(N),
        "hyp_vol": float(hyp_vol),
        "best_hyp_vol": float(hyp_vol),
        "n_simplices": int(n_simplices),
        "n_facets": count_hull_facets(vertices),
        "elapsed_sec": round(elapsed, 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-N", type=int, default=None,
                   help="Skip entries with N larger than this")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated list of N values to process "
                        "(e.g. '6,7,8'); overrides --max-N")
    p.add_argument("--face-samples", type=int, default=1 << 13,
                   help="QMC samples per face volume (default 8192)")
    p.add_argument("--steps", type=int, default=32,
                   help="Schläfli path integration steps (default 32)")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed (default 0)")
    p.add_argument("--input", type=Path, default=INPUT_JSON)
    p.add_argument("--output", type=Path, default=OUTPUT_JSON)
    p.add_argument("--dry-run", action="store_true",
                   help="Load configs and print what would be run; do not compute")
    args = p.parse_args()

    if args.only is not None:
        only_Ns = {int(x) for x in args.only.split(",") if x.strip()}
    else:
        only_Ns = None

    all_data = json.loads(args.input.read_text())
    d5_keys = sorted(
        (k for k in all_data if k.startswith("d5_N")),
        key=lambda k: int(k.split("N")[1]),
    )
    print(f"Found {len(d5_keys)} d=5 entries in {args.input.relative_to(REPO)}")
    print(f"  Ns: {[all_data[k]['N'] for k in d5_keys]}")

    results = []
    for key in d5_keys:
        entry = all_data[key]
        N = int(entry["N"])
        if only_Ns is not None and N not in only_Ns:
            continue
        if args.max_N is not None and N > args.max_N:
            continue

        vertices = np.asarray(entry["vertices"], dtype=float)
        eucl_vol = float(entry["euclidean_volume"])

        print(f"\n--- d=5, N={N} ---")
        print(f"  vertices shape: {vertices.shape}")
        print(f"  stored Euclidean volume: {eucl_vol:.10f}")

        if args.dry_run:
            continue

        res = recompute_one(
            N, vertices,
            face_samples=args.face_samples,
            steps=args.steps,
            seed=args.seed,
        )
        res["eucl_vol"] = eucl_vol
        res["vertices"] = entry["vertices"]
        if N in PAPER_TABLE_5:
            res["paper_hyp_vol"] = PAPER_TABLE_5[N]
            res["paper_rel_err"] = abs(res["hyp_vol"] - PAPER_TABLE_5[N]) / PAPER_TABLE_5[N]
            # Two regimes. Paper Table 5 values are printed to 5 decimals; a
            # gap below ~0.5% is consistent with paper rounding on the SAME
            # configuration, so our recomputed value from our own coords is
            # the honest LB. A gap above 0.5% means the stored JSON config is
            # genuinely worse than what the paper reports, so the paper value
            # is a tighter (verified, but not in-coords) LB.
            PAPER_ROUNDING_TOL = 0.005
            if res["paper_rel_err"] <= PAPER_ROUNDING_TOL:
                res["best_lb"] = res["hyp_vol"]
                res["best_lb_source"] = "recomputed_from_json"
            else:
                res["best_lb"] = max(res["hyp_vol"], res["paper_hyp_vol"])
                res["best_lb_source"] = (
                    "paper_table_5" if res["paper_hyp_vol"] > res["hyp_vol"]
                    else "recomputed_from_json"
                )
        else:
            res["best_lb"] = res["hyp_vol"]
            res["best_lb_source"] = "recomputed_from_json"

        print(f"  H^5 volume: {res['hyp_vol']:.10f}"
              f"  ({res['n_simplices']} simplices, {res['elapsed_sec']:.1f}s)")
        if "paper_hyp_vol" in res:
            print(f"  paper Table 5: {res['paper_hyp_vol']:.5f}"
                  f"   rel err: {res['paper_rel_err']:.2e}"
                  f"   best LB source: {res['best_lb_source']}")

        results.append(res)

    if args.dry_run or not results:
        return

    results.sort(key=lambda r: r["N"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {len(results)} entries to {args.output.relative_to(REPO)}")

    print("\nCross-check against paper Table 5:")
    print(f"{'N':>4} {'computed':>14} {'paper':>10} {'rel err':>10}")
    for r in results:
        if "paper_hyp_vol" in r:
            print(f"{r['N']:>4} {r['hyp_vol']:>14.10f} "
                  f"{r['paper_hyp_vol']:>10.5f} {r['paper_rel_err']:>10.2e}")
        else:
            print(f"{r['N']:>4} {r['hyp_vol']:>14.10f}   (no paper value)")


if __name__ == "__main__":
    main()
