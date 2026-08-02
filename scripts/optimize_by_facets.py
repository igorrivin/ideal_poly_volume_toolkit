#!/usr/bin/env python3
"""
Optimize ideal 4-polytope volume parameterized by FACET NORMALS.

The key idea (Minkowski's variational principle): to construct a polytope
with f prescribed facet normals u_1,...,u_f on S^3 and maximal Euclidean
volume, take equal support numbers:

    P = {x in R^4 : u_i . x <= 1  for all i}

This polytope is circumscribed about the unit sphere.  Its vertices are
found by polarity: Q = conv(u_1,...,u_f), and each facet of Q (a simplex
with vertex indices I) gives a vertex v of P via solving U_I . v = 1.

To get an ideal polytope, project the vertices of P onto S^3:
    v_hat = v / |v|

Since radial projection from an interior point preserves the face lattice,
the resulting ideal polytope conv(v_hat_1,...,v_hat_m) has exactly f facets.

ALGORITHM:
  1. Parameterize f points on S^3 (3f hyperspherical angles)
  2. Compute Q = conv(u_1,...,u_f)
  3. For each facet of Q, solve 4x4 system -> dual vertex
  4. Project dual vertices to S^3
  5. Compute H^4 volume via Delaunay + angle formula
  6. Optimize over the facet normals
"""

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
from itertools import combinations
from collections import Counter
from multiprocessing import Pool
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ideal_poly_volume_toolkit.geometry_4d import (
    ideal_4simplex_volume_from_tet,
    tet_dihedral_angles,
    circumsphere,
    face_distances_from_circumcenter,
    stereo_project_S3,
    regular_ideal_4simplex_volume,
)


# ============================================================================
# Parameterization: f facet normals on S^3
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
    """Convert S^3 points to hyperspherical angles."""
    a = np.arccos(np.clip(pts[:, 3], -1, 1))
    sin_a = np.sin(a) + 1e-30
    b = np.arccos(np.clip(pts[:, 2] / sin_a, -1, 1))
    c = np.arctan2(pts[:, 1], pts[:, 0])
    return np.concatenate([a, b, c])


# ============================================================================
# Core: facet normals -> ideal polytope vertices
# ============================================================================

def normals_to_ideal_vertices(normals):
    """
    Given f facet normals on S^3, construct the dual ideal polytope.

    P = {x : u_i . x <= 1}  (circumscribed about unit sphere)
    Vertices of P are found via polarity with Q = conv(u_1,...,u_f).

    Returns
    -------
    ideal_pts : (m, 4) array of points on S^3 (ideal vertices)
    hull_Q : ConvexHull of Q (for diagnostics)
    dual_vertices : (m, 4) array of vertices of P before projection
    """
    # Q = conv(normals)
    hull_Q = ConvexHull(normals)

    # Check that origin is inside Q
    # (needed for polar duality to produce a bounded P)
    # For points on S^3, this holds iff they don't lie in a hemisphere.
    # We check via the equations: all hull_Q.equations[:,3] < 0
    # (offset term should be negative if origin is inside)
    if np.any(hull_Q.equations[:, -1] > 0):
        return None, hull_Q, None

    # Each facet of Q -> a vertex of P
    dual_verts = []
    for simplex in hull_Q.simplices:
        U = normals[simplex]  # (4, 4) matrix
        try:
            v = np.linalg.solve(U, np.ones(4))
            dual_verts.append(v)
        except np.linalg.LinAlgError:
            continue

    if len(dual_verts) == 0:
        return None, hull_Q, None

    dual_verts = np.array(dual_verts)

    # Project onto S^3
    norms = np.linalg.norm(dual_verts, axis=1, keepdims=True)
    ideal_pts = dual_verts / norms

    return ideal_pts, hull_Q, dual_verts


# ============================================================================
# Volume computation
# ============================================================================

def compute_ideal_volume(ideal_pts):
    """Compute H^4 volume of ideal polytope given vertices on S^3."""
    if ideal_pts is None or len(ideal_pts) < 5:
        return 0.0

    # Stereographic project from first point
    pts_3d, _ = stereo_project_S3(ideal_pts, pole_idx=0)

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


# ============================================================================
# Objective function
# ============================================================================

def neg_volume_from_normals(angles):
    """
    Objective: negative H^4 volume as function of facet normal angles.
    """
    n = len(angles) // 3
    normals = angles_to_S3(angles, n)

    ideal_pts, hull_Q, _ = normals_to_ideal_vertices(normals)

    if ideal_pts is None:
        return 0.0  # penalty: infeasible

    vol = compute_ideal_volume(ideal_pts)
    return -vol


# ============================================================================
# Optimization
# ============================================================================

def optimize_worker(args):
    """Single optimization run from random facet normals."""
    n_facets, seed, maxiter1, maxiter2 = args
    rng = np.random.default_rng(seed)

    # Random initial facet normals on S^3
    normals0 = rng.standard_normal((n_facets, 4))
    normals0 /= np.linalg.norm(normals0, axis=1, keepdims=True)
    x0 = pts_to_angles(normals0)

    # Phase 1: quick
    res = minimize(neg_volume_from_normals, x0, method='L-BFGS-B',
                   options={'maxiter': maxiter1, 'ftol': 1e-13})

    # Phase 2: polish
    res2 = minimize(neg_volume_from_normals, res.x, method='L-BFGS-B',
                    options={'maxiter': maxiter2, 'ftol': 1e-14, 'gtol': 1e-10})

    vol = -res2.fun
    return vol, res2.x


def analyze_result(angles, n_facets):
    """Analyze an optimized configuration."""
    normals = angles_to_S3(angles, n_facets)
    ideal_pts, hull_Q, dual_verts = normals_to_ideal_vertices(normals)

    if ideal_pts is None:
        print("  INFEASIBLE (origin not inside convex hull of normals)")
        return

    n_verts = len(ideal_pts)
    vol = compute_ideal_volume(ideal_pts)

    # f-vector of the ideal polytope
    hull_P = ConvexHull(ideal_pts)
    edges = set()
    triangles = set()
    for simplex in hull_P.simplices:
        for i, j in combinations(sorted(simplex), 2):
            edges.add((i, j))
        for tri in combinations(sorted(simplex), 3):
            triangles.add(tri)

    f0 = n_verts
    f1 = len(edges)
    f2 = len(triangles)
    f3 = len(hull_P.simplices)

    print(f"  Volume: {vol:.10f}")
    print(f"  Vertices (from {n_facets} facet normals): {n_verts}")
    print(f"  f-vector: ({f0}, {f1}, {f2}, {f3})")
    print(f"  Euler: {f0 - f1 + f2 - f3}")

    # Degree sequence
    deg = Counter()
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
    deg_seq = sorted([deg[i] for i in range(f0)], reverse=True)
    print(f"  Degree seq: {deg_seq}")

    # Dual vertex norms (how far from S^3 before projection)
    norms = np.linalg.norm(dual_verts, axis=1)
    print(f"  Dual vertex norms: min={norms.min():.4f}, max={norms.max():.4f}, "
          f"mean={norms.mean():.4f}")

    # Delaunay decomposition
    pts_3d, _ = stereo_project_S3(ideal_pts, pole_idx=0)
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
    print(f"  Delaunay: {len(tet_vols)} simplices")
    print(f"  Tet volumes: min={tet_vols.min():.6f}, max={tet_vols.max():.6f}, "
          f"mean={tet_vols.mean():.6f}")
    print(f"  Max tet / regular: {tet_vols.max()/v_reg:.6f}")

    # Check: does the ideal polytope actually have n_facets facets?
    # The facets of the ideal polytope correspond to vertices of Q
    # (the convex hull of the normals). So f3 of ideal polytope should
    # relate to f0 of Q = n_facets.
    # But the ideal polytope's ConvexHull gives its own f3.
    print(f"\n  Facet check: input normals = {n_facets}, "
          f"ideal polytope f3 = {f3}")

    return vol, ideal_pts, f3


def run_optimization(n_facets, n_restarts=200, n_workers=32):
    """Run parallel optimization for given number of facets."""
    print(f"\n{'='*70}")
    print(f"Optimizing ideal 4-polytope volume with {n_facets} facet normals")
    print(f"{'='*70}")
    print(f"Restarts: {n_restarts}, Workers: {n_workers}")

    t0 = time.time()

    work = [(n_facets, 99999 + n_facets * 7777 + i, 300, 300)
            for i in range(n_restarts)]

    with Pool(processes=min(n_workers, n_restarts)) as pool:
        results = pool.map(optimize_worker, work)

    elapsed = time.time() - t0

    volumes = [r[0] for r in results]
    valid = [(v, x) for v, x in results if v > 0]
    valid.sort(key=lambda r: r[0], reverse=True)

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Valid results: {len(valid)}/{len(results)}")

    if not valid:
        print("No valid results!")
        return

    vols = np.array([v for v, _ in valid])
    print(f"\nVolume statistics:")
    print(f"  Best:   {vols[0]:.10f}")
    print(f"  Top 10: {', '.join(f'{v:.4f}' for v in vols[:10])}")
    print(f"  Mean:   {vols.mean():.6f}")
    print(f"  Std:    {vols.std():.6f}")

    # Analyze the best result
    print(f"\nBest configuration:")
    best_vol, best_angles = valid[0]
    result = analyze_result(best_angles, n_facets)

    # Also check: what f3 do we actually get?
    # The ideal polytope's facet count may differ from n_facets
    # because some facets of Q might be non-simplicial or degenerate.
    if result is not None:
        _, _, actual_f3 = result
        if actual_f3 != n_facets:
            print(f"\n  WARNING: ideal polytope has {actual_f3} facets, "
                  f"not {n_facets}!")
            print(f"  (Some facet normals may have become redundant)")

    # Distribution of actual f3 among top results
    print(f"\nActual f3 distribution (top 50 results):")
    f3_counts = Counter()
    for v, x in valid[:50]:
        normals = angles_to_S3(x, n_facets)
        ideal_pts, _, _ = normals_to_ideal_vertices(normals)
        if ideal_pts is not None:
            f3 = compute_f3_quick(ideal_pts)
            f3_counts[f3] += 1
    for f3_val in sorted(f3_counts.keys()):
        print(f"    f3={f3_val}: {f3_counts[f3_val]} results")

    return valid


def compute_f3_quick(pts):
    """Quick f3 computation."""
    try:
        hull = ConvexHull(pts)
        return len(hull.simplices)
    except Exception:
        return -1


def main():
    if len(sys.argv) < 2:
        print("Usage: python optimize_by_facets.py <n_facets> [n_restarts]")
        print()
        print("Examples:")
        print("  python optimize_by_facets.py 24       # 24-cell has 24 facets")
        print("  python optimize_by_facets.py 124      # match our N=30 result")
        print("  python optimize_by_facets.py 50 500   # 50 facets, 500 restarts")
        sys.exit(1)

    facet_counts = [int(sys.argv[1])]
    n_restarts = int(sys.argv[2]) if len(sys.argv) >= 3 else 200

    if not facet_counts:
        facet_counts = [24]

    v_reg = regular_ideal_4simplex_volume()
    print(f"Regular ideal 4-simplex volume: {v_reg:.8f}")

    all_results = {}
    for nf in facet_counts:
        valid = run_optimization(nf, n_restarts=n_restarts)
        if valid:
            all_results[nf] = valid[0][0]  # best volume

    if len(all_results) > 1:
        print(f"\n\n{'='*70}")
        print("Summary: best volume vs number of facets")
        print(f"{'='*70}")
        print(f"{'Facets':>8s}  {'Best Volume':>14s}  {'Vol/v_reg':>10s}")
        for nf in sorted(all_results.keys()):
            v = all_results[nf]
            print(f"{nf:8d}  {v:14.8f}  {v/v_reg:10.2f}")


if __name__ == '__main__':
    main()
