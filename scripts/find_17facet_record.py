#!/usr/bin/env python3
"""
Logged search for the 17-facet constrained surface-area record at N=8 on S^3.

Maximizes Euclidean surface 3-volume (sum of Gram-determinant 3-volumes of
boundary tetrahedral facets) over 8-vertex inscribed simplicial 4-polytopes
with exactly 17 facets.

Saves top-K configurations (vertices + value) to JSON for basin comparison
against Gemini's submitted 5.0359 polytope.

Usage:
    python scripts/find_17facet_record.py --restarts 1000 --workers 32 \
        --seed-strategy mixed --out results/data/sa_17facet_record.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ideal_poly_volume_toolkit.geometry_4d import ideal_tet_volume_3d


def gram_tet_vol(v4: np.ndarray) -> float:
    """Euclidean 3-volume of a tetrahedron (4 points in R^4) via Gram det."""
    edges = v4[1:] - v4[0]  # (3, 4)
    G = edges @ edges.T
    det = np.linalg.det(G)
    if det <= 0:
        return 0.0
    return float(np.sqrt(det)) / 6.0


# Module-level objective selector (set in main()) — multiprocessing needs
# a top-level value, not a closure.
_FACET_VOL_FN = gram_tet_vol


def surface_area_and_count(pts_4d: np.ndarray) -> tuple[float, int]:
    """Return (surface 3-volume, n_facets) for a point set on S^3.

    Uses _FACET_VOL_FN (Euclidean Gram det or hyperbolic ideal tet) per facet.
    """
    try:
        hull = ConvexHull(pts_4d)
    except Exception:
        return 0.0, 0
    total = 0.0
    for facet in hull.simplices:
        total += _FACET_VOL_FN(pts_4d[facet])
    return total, len(hull.simplices)


def project_S3(x: np.ndarray) -> np.ndarray:
    pts = x.reshape(-1, 4)
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return pts / n


def objective(x: np.ndarray, target_facets: int = 17) -> float:
    """Negative surface area with hard penalty for n_facets != target."""
    pts = project_S3(x)
    sa, nf = surface_area_and_count(pts)
    if nf != target_facets:
        # Penalty scales with how far off we are.
        return -sa + 100.0 * abs(nf - target_facets)
    return -sa


def cross_polytope_seed(rng: np.random.Generator) -> np.ndarray:
    """Random rotation of the 16-cell ±e_i."""
    base = np.vstack([np.eye(4), -np.eye(4)])  # (8, 4) — but we need 8, take all
    # 16-cell has 8 vertices: ±e_1, ±e_2, ±e_3, ±e_4 — that's 8 vertices.
    # Apply random rotation.
    A = rng.standard_normal((4, 4))
    Q, _ = np.linalg.qr(A)
    return base @ Q.T


def perturbed_16cell_seed(rng: np.random.Generator, sigma: float = 0.15) -> np.ndarray:
    pts = cross_polytope_seed(rng)
    pts = pts + sigma * rng.standard_normal(pts.shape)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def random_seed(rng: np.random.Generator) -> np.ndarray:
    pts = rng.standard_normal((8, 4))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def make_seed(strategy: str, rng: np.random.Generator) -> np.ndarray:
    if strategy == "random":
        return random_seed(rng)
    if strategy == "perturbed_16cell":
        return perturbed_16cell_seed(rng)
    if strategy == "mixed":
        return perturbed_16cell_seed(rng) if rng.random() < 0.5 else random_seed(rng)
    raise ValueError(f"unknown strategy: {strategy}")


def worker(args):
    seed, strategy = args
    rng = np.random.default_rng(seed)
    pts0 = make_seed(strategy, rng)

    # Powell with two stages: penalised search, then unconstrained polish if
    # the facet count is right.
    res1 = minimize(objective, pts0.flatten(), method="Powell",
                    options={"maxiter": 5000, "xtol": 1e-8, "ftol": 1e-10})
    pts1 = project_S3(res1.x)
    sa1, nf1 = surface_area_and_count(pts1)
    if nf1 != 17:
        return {"value": sa1, "n_facets": nf1, "vertices": pts1.tolist(),
                "seed": int(seed), "strategy": strategy, "stage": "penalised"}

    # Polish without penalty (still penalised below if it leaves 17, but
    # near a local max it usually stays).
    res2 = minimize(objective, res1.x, method="Powell",
                    options={"maxiter": 2000, "xtol": 1e-10, "ftol": 1e-12})
    pts2 = project_S3(res2.x)
    sa2, nf2 = surface_area_and_count(pts2)
    if nf2 == 17 and sa2 > sa1:
        return {"value": sa2, "n_facets": nf2, "vertices": pts2.tolist(),
                "seed": int(seed), "strategy": strategy, "stage": "polished"}
    return {"value": sa1, "n_facets": nf1, "vertices": pts1.tolist(),
            "seed": int(seed), "strategy": strategy, "stage": "penalised"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--restarts", type=int, default=1000)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--seed-strategy", choices=("random", "perturbed_16cell", "mixed"),
                   default="mixed")
    p.add_argument("--seed-base", type=int, default=20260430)
    p.add_argument("--out", type=Path,
                   default=REPO / "results" / "data" / "sa_17facet_record.json")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--target", type=float, default=None,
                   help="reference value for logging.")
    p.add_argument("--measure", choices=("euclidean", "hyperbolic"),
                   default="euclidean",
                   help="Per-facet 3-volume: Gram det (Euclidean) or "
                        "ideal-tet H^3 volume (hyperbolic, ideal polytope).")
    args = p.parse_args()

    global _FACET_VOL_FN
    if args.measure == "hyperbolic":
        _FACET_VOL_FN = ideal_tet_volume_3d
    else:
        _FACET_VOL_FN = gram_tet_vol

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[start] restarts={args.restarts} strategy={args.seed_strategy} "
          f"workers={args.workers}", flush=True)
    work = [(args.seed_base + i, args.seed_strategy) for i in range(args.restarts)]

    t0 = time.time()
    with Pool(args.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(worker, work, chunksize=4), 1):
            results.append(r)
            if i % 25 == 0 or i == args.restarts:
                vals = sorted([x["value"] for x in results
                               if x["n_facets"] == 17], reverse=True)
                best = vals[0] if vals else float("nan")
                hits = sum(1 for x in results if x["n_facets"] == 17)
                print(f"[{i:4d}/{args.restarts}] elapsed={time.time()-t0:6.1f}s "
                      f"hits-17={hits} best={best:.6f}", flush=True)

    valid = [r for r in results if r["n_facets"] == 17]
    valid.sort(key=lambda r: -r["value"])
    top = valid[: args.top_k]

    out = {
        "N": 8, "d": 4, "constraint": "n_facets=17",
        "objective": f"{args.measure}_surface_area",
        "measure": args.measure,
        "strategy": args.seed_strategy, "restarts": args.restarts,
        "target_reference": args.target,
        "n_valid": len(valid), "n_total": len(results),
        "best_value": top[0]["value"] if top else None,
        "top_k": top,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] best={out['best_value']} valid={len(valid)}/{len(results)} "
          f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
