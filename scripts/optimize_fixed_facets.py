#!/usr/bin/env python3
"""
Maximize ideal 4-polytope volume subject to a fixed number of facets.

Strategy:
  1. For each candidate vertex count N, run many random restarts of
     volume optimization (unconstrained).
  2. After each optimization, compute f3 (facet count) of the result.
  3. Keep only results matching the target f3.
  4. Polish the best hits with further optimization, checking that f3
     is preserved.
  5. Report the best volume for each (N, f3) pair.

Since f3 is discontinuous in the vertex positions, we cannot use
gradient-based methods to enforce f3 = target directly.  Instead we
rely on the fact that different random starts land on different
combinatorial types, and filter post hoc.

For simplicial 4-polytopes (generic), f3 = f1 - f0 by Dehn-Sommerville.
"""

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
from itertools import combinations
from collections import defaultdict
from multiprocessing import Pool
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ideal_poly_volume_toolkit.geometry_4d import (
    ideal_4simplex_volume_from_tet,
    stereo_project_S3,
    regular_ideal_4simplex_volume,
)


# ============================================================================
# Parameterization
# ============================================================================

def angles_to_S3(angles, n):
    """Convert 3*n angles to n points on S^3."""
    a = angles[:n]
    b = angles[n:2*n]
    c = angles[2*n:3*n]
    sa, ca = np.sin(a), np.cos(a)
    sb, cb = np.sin(b), np.cos(b)
    sc, cc = np.sin(c), np.cos(c)
    return np.column_stack([sa*sb*cc, sa*sb*sc, sa*cb, ca])


def pts_to_angles(pts):
    """Convert S^3 points back to hyperspherical angles."""
    a = np.arccos(np.clip(pts[:, 3], -1, 1))
    sin_a = np.sin(a) + 1e-30
    b = np.arccos(np.clip(pts[:, 2] / sin_a, -1, 1))
    c = np.arctan2(pts[:, 1], pts[:, 0])
    return np.concatenate([a, b, c])


def compute_volume_4d(pts_4d):
    """Compute ideal polytope volume from S^3 points."""
    pts_3d, _ = stereo_project_S3(pts_4d, pole_idx=0)
    try:
        tri = Delaunay(pts_3d)
    except Exception:
        return 0.0

    total = 0.0
    for tet_idx in tri.simplices:
        v = pts_3d[tet_idx]
        try:
            total += ideal_4simplex_volume_from_tet(v)
        except Exception:
            pass
    return total


def compute_f3(pts_4d):
    """Compute f3 (number of 3-facets) of the convex hull in R^4."""
    try:
        hull = ConvexHull(pts_4d)
        return len(hull.simplices)
    except Exception:
        return -1


def neg_volume_4d(angles):
    """Negative volume for minimization."""
    n = len(angles) // 3
    pts = angles_to_S3(angles, n)
    return -compute_volume_4d(pts)


# ============================================================================
# Optimization worker
# ============================================================================

def optimize_and_classify(args):
    """
    Single optimization run.  Returns (volume, f3, angles).
    """
    n_vertices, seed, maxiter1, maxiter2 = args
    rng = np.random.default_rng(seed)

    # Random initial points on S^3
    pts0 = rng.standard_normal((n_vertices, 4))
    pts0 /= np.linalg.norm(pts0, axis=1, keepdims=True)
    x0 = pts_to_angles(pts0)

    # Phase 1: quick optimization
    res = minimize(neg_volume_4d, x0, method='L-BFGS-B',
                   options={'maxiter': maxiter1, 'ftol': 1e-14})

    # Phase 2: polish
    res2 = minimize(neg_volume_4d, res.x, method='L-BFGS-B',
                    options={'maxiter': maxiter2, 'ftol': 1e-15, 'gtol': 1e-11})

    vol = -res2.fun
    pts_opt = angles_to_S3(res2.x, n_vertices)
    f3 = compute_f3(pts_opt)

    return vol, f3, res2.x


def polish_with_f3_check(angles, n_vertices, target_f3, n_polish=5):
    """
    Further polish an optimized configuration, checking that f3 is preserved.
    Uses small random perturbations + re-optimization.
    Returns best (volume, angles) that maintains target f3.
    """
    best_vol = -neg_volume_4d(angles)
    best_angles = angles.copy()

    for i in range(n_polish):
        # Small perturbation
        scale = 0.01 / (1 + i)
        perturbed = angles + scale * np.random.randn(len(angles))

        res = minimize(neg_volume_4d, perturbed, method='L-BFGS-B',
                       options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12})

        vol = -res.fun
        pts = angles_to_S3(res.x, n_vertices)
        f3 = compute_f3(pts)

        if f3 == target_f3 and vol > best_vol:
            best_vol = vol
            best_angles = res.x.copy()

    return best_vol, best_angles


# ============================================================================
# Main
# ============================================================================

def run_facet_constrained_optimization(
    target_f3,
    n_vertices_range=None,
    n_restarts=500,
    n_workers=32,
):
    """
    Find the maximum-volume ideal 4-polytope with exactly target_f3 facets.
    """
    if n_vertices_range is None:
        # Estimate: for random S^3 points, mean f3 ≈ (N/2)*(something)
        # From our data: f3/N ratio grows with N.
        # Rough inversion: try a range around target_f3 / 4.5
        n_center = max(5, int(target_f3 / 4.0))
        # Also, volume-optimized polytopes have fewer facets than random,
        # so also try larger N
        n_lo = max(5, n_center - 5)
        n_hi = n_center + 8
        n_vertices_range = range(n_lo, n_hi + 1)

    print(f"Target f3 = {target_f3}")
    print(f"Vertex counts to try: {list(n_vertices_range)}")
    print(f"Restarts per N: {n_restarts}")
    print(f"Workers: {n_workers}")
    print()

    # Collect results by (N, f3)
    all_hits = []  # (vol, f3, N, angles)
    best_by_f3 = defaultdict(lambda: (0.0, None, None, None))

    for n_v in n_vertices_range:
        t0 = time.time()

        work = [(n_v, 42000 + n_v * 10000 + i, 2000, 2000)
                for i in range(n_restarts)]

        with Pool(processes=n_workers) as pool:
            results = pool.map(optimize_and_classify, work)

        # Classify by f3
        f3_counts = defaultdict(list)
        for vol, f3, angles in results:
            f3_counts[f3].append((vol, angles))

        elapsed = time.time() - t0

        print(f"N = {n_v:3d}  ({elapsed:.1f}s)  f3 distribution of optimized polytopes:")
        for f3_val in sorted(f3_counts.keys()):
            vols = [v for v, _ in f3_counts[f3_val]]
            marker = " <-- TARGET" if f3_val == target_f3 else ""
            print(f"    f3={f3_val:4d}: {len(vols):4d} hits, "
                  f"vol range [{min(vols):.4f}, {max(vols):.4f}]{marker}")

            if f3_val == target_f3:
                for vol, angles in f3_counts[f3_val]:
                    all_hits.append((vol, f3_val, n_v, angles))

        # Track best for nearby f3 values too
        for f3_val, entries in f3_counts.items():
            best_vol_here = max(v for v, _ in entries)
            best_angles_here = max(entries, key=lambda x: x[0])[1]
            if best_vol_here > best_by_f3[f3_val][0]:
                best_by_f3[f3_val] = (best_vol_here, f3_val, n_v, best_angles_here)

        print()

    # Report hits for target f3
    print(f"\n{'='*70}")
    print(f"Results for target f3 = {target_f3}")
    print(f"{'='*70}")

    if not all_hits:
        print("No configurations found with target f3!")
        print("\nBest volumes by nearby f3 values:")
        for f3_val in sorted(best_by_f3.keys()):
            vol, _, n_v, _ = best_by_f3[f3_val]
            print(f"  f3={f3_val:4d}:  vol={vol:.6f}  (N={n_v})")
        return None

    # Sort by volume
    all_hits.sort(key=lambda x: x[0], reverse=True)

    print(f"\nFound {len(all_hits)} configurations with f3 = {target_f3}")
    print(f"\nTop 10 by volume:")
    for i, (vol, f3, n_v, angles) in enumerate(all_hits[:10]):
        print(f"  #{i+1}: vol = {vol:.8f}  (N = {n_v})")

    # Polish the top candidates
    print(f"\nPolishing top 5 candidates...")
    best_overall = all_hits[0]
    for i, (vol, f3, n_v, angles) in enumerate(all_hits[:5]):
        pol_vol, pol_angles = polish_with_f3_check(
            angles, n_v, target_f3, n_polish=10
        )
        print(f"  #{i+1}: {vol:.8f} -> {pol_vol:.8f}")
        if pol_vol > best_overall[0]:
            best_overall = (pol_vol, target_f3, n_v, pol_angles)

    print(f"\n{'='*70}")
    print(f"BEST: vol = {best_overall[0]:.10f}  "
          f"(N = {best_overall[2]}, f3 = {target_f3})")
    print(f"{'='*70}")

    # Print f-vector and analysis of best
    pts = angles_to_S3(best_overall[3], best_overall[2])
    hull = ConvexHull(pts)
    edges = set()
    triangles = set()
    for simplex in hull.simplices:
        for i, j in combinations(sorted(simplex), 2):
            edges.add((i, j))
        for tri in combinations(sorted(simplex), 3):
            triangles.add(tri)

    f0, f1, f2, f3 = len(pts), len(edges), len(triangles), len(hull.simplices)
    print(f"f-vector: ({f0}, {f1}, {f2}, {f3})")
    print(f"Euler: {f0 - f1 + f2 - f3} (should be 0)")

    # Volume decomposition
    pts_3d, _ = stereo_project_S3(pts, pole_idx=0)
    tri = Delaunay(pts_3d)
    tet_vols = []
    for tet_idx in tri.simplices:
        v = pts_3d[tet_idx]
        try:
            tet_vols.append(ideal_4simplex_volume_from_tet(v))
        except Exception:
            pass
    tet_vols = np.array(tet_vols)
    v_reg = regular_ideal_4simplex_volume()
    print(f"\nDelaunay: {len(tet_vols)} simplices")
    print(f"Tet volumes: min={tet_vols.min():.6f}, max={tet_vols.max():.6f}, "
          f"mean={tet_vols.mean():.6f}")
    print(f"Max tet / regular: {tet_vols.max()/v_reg:.6f}")

    # Also report best volumes for nearby f3 values
    print(f"\n{'='*70}")
    print(f"Best volumes by f3 (from all runs):")
    print(f"{'='*70}")
    for f3_val in sorted(best_by_f3.keys()):
        vol, _, n_v, _ = best_by_f3[f3_val]
        marker = " <-- TARGET" if f3_val == target_f3 else ""
        print(f"  f3={f3_val:4d}:  vol={vol:.6f}  (N={n_v}){marker}")

    return best_overall


def main():
    if len(sys.argv) < 2:
        print("Usage: python optimize_fixed_facets.py <target_f3> [n_lo] [n_hi] [n_restarts]")
        print()
        print("Example: python optimize_fixed_facets.py 124 26 35 500")
        sys.exit(1)

    target_f3 = int(sys.argv[1])

    if len(sys.argv) >= 4:
        n_lo = int(sys.argv[2])
        n_hi = int(sys.argv[3])
        n_range = range(n_lo, n_hi + 1)
    else:
        n_range = None

    n_restarts = int(sys.argv[4]) if len(sys.argv) >= 5 else 500

    result = run_facet_constrained_optimization(
        target_f3=target_f3,
        n_vertices_range=n_range,
        n_restarts=n_restarts,
    )


if __name__ == '__main__':
    main()
