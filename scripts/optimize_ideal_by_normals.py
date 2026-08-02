#!/usr/bin/env python3
"""
Optimize ideal 4-polytope volume parameterized by facet normals.

Two-level optimization:
  Inner: Given f normals on S^3, find support numbers h such that
         P = {x : u_i . x <= h_i} has all vertices on S^3.
         This is a system of quadratic equations solved by least-squares.

  Outer: Optimize H^4 volume over the normals u_1,...,u_f on S^3.

The inner problem has f unknowns (h_1,...,h_f) and m equations
(one per vertex: |v_j|^2 = 1), where m = number of facets of
Q = conv(u_1/h_1,...,u_f/h_f).  Typically m > f (overdetermined),
but Levenberg-Marquardt finds solutions to machine precision.
"""

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize, least_squares
from itertools import combinations
from collections import Counter
from multiprocessing import Pool
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ideal_poly_volume_toolkit.geometry_4d import (
    ideal_4simplex_volume_from_tet,
    stereo_project_S3,
    regular_ideal_4simplex_volume,
)


# ============================================================================
# Inner problem: solve for support numbers
# ============================================================================

def get_combinatorial_type(normals, h_probe=0.3):
    """
    Determine the combinatorial type of P = {x : u_i.x <= h_i}
    at a probe value h = h_probe * ones.

    Returns list of index-tuples, one per vertex of P.
    Each tuple gives the 4 normal indices whose hyperplanes
    meet at that vertex.
    """
    f = len(normals)
    w = normals / h_probe
    try:
        hull = ConvexHull(w)
    except Exception:
        return None

    # Check origin inside
    if np.any(hull.equations[:, -1] > 1e-10):
        return None

    vertex_defs = [tuple(sorted(s)) for s in hull.simplices]
    return vertex_defs


def build_vertex_maps(normals, vertex_defs):
    """
    Build the linear maps v_j = A_j @ h.

    For vertex j defined by indices I_j, v_j = U_{I_j}^{-1} h_{I_j}.
    Expanding: A_j is a (4 x f) matrix with A_j[:, I_j[k]] = U_inv[:, k].
    """
    f = len(normals)
    A_matrices = []

    for indices in vertex_defs:
        U = normals[list(indices)]
        try:
            U_inv = np.linalg.inv(U)
        except np.linalg.LinAlgError:
            return None

        A = np.zeros((4, f))
        for k, idx in enumerate(indices):
            A[:, idx] = U_inv[:, k]
        A_matrices.append(A)

    return A_matrices


def solve_support_numbers(normals, vertex_defs, A_matrices, h0=None):
    """
    Solve for support numbers h such that all vertices lie on S^3.

    The system: |A_j @ h|^2 = 1 for all j.
    """
    f = len(normals)
    n_verts = len(vertex_defs)

    if h0 is None:
        h0 = np.full(f, 0.3)

    def residuals(h):
        res = np.zeros(n_verts)
        for j, A in enumerate(A_matrices):
            v = A @ h
            res[j] = np.dot(v, v) - 1.0
        return res

    def jacobian(h):
        J = np.zeros((n_verts, f))
        for j, A in enumerate(A_matrices):
            v = A @ h
            J[j, :] = 2 * (A.T @ v)
        return J

    result = least_squares(residuals, h0, jac=jacobian, method='lm',
                           ftol=1e-15, xtol=1e-15, gtol=1e-15,
                           max_nfev=10000)

    if result.cost < 1e-20:
        return result.x, result.cost
    else:
        return None, result.cost


# ============================================================================
# Volume computation
# ============================================================================

def compute_volume(vertices):
    """Compute H^4 volume from ideal vertices on S^3."""
    if vertices is None or len(vertices) < 5:
        return 0.0

    pts_3d, _ = stereo_project_S3(vertices, pole_idx=0)
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
# Full pipeline: normals -> volume
# ============================================================================

def normals_to_volume(angles, f, h_cache={}):
    """
    Given 3f angles parameterizing f normals on S^3,
    solve for ideal polytope and return its volume.
    """
    # Decode normals
    a = angles[:f]
    b = angles[f:2*f]
    c = angles[2*f:3*f]
    sa, ca = np.sin(a), np.cos(a)
    sb, cb = np.sin(b), np.cos(b)
    sc, cc = np.sin(c), np.cos(c)
    normals = np.column_stack([sa*sb*cc, sa*sb*sc, sa*cb, ca])

    # Get combinatorial type
    vertex_defs = get_combinatorial_type(normals)
    if vertex_defs is None:
        return 0.0

    # Build linear maps
    A_matrices = build_vertex_maps(normals, vertex_defs)
    if A_matrices is None:
        return 0.0

    # Solve inner problem
    # Use cached h as warm start if available
    cache_key = id(angles)  # rough caching
    h0 = h_cache.get('last_h', np.full(f, 0.3))

    h_opt, cost = solve_support_numbers(normals, vertex_defs, A_matrices, h0=h0)
    if h_opt is None:
        # Try with default h0
        h_opt, cost = solve_support_numbers(normals, vertex_defs, A_matrices)
        if h_opt is None:
            return 0.0

    h_cache['last_h'] = h_opt

    # Compute vertices
    vertices = np.array([A @ h_opt for A in A_matrices])

    # Check they're on S^3
    r = np.linalg.norm(vertices, axis=1)
    if np.max(np.abs(r - 1)) > 1e-6:
        return 0.0

    # Compute volume
    vol = compute_volume(vertices)
    return vol


def neg_volume_from_angles(angles, f):
    """Negative volume for minimization."""
    return -normals_to_volume(angles, f)


# ============================================================================
# Optimization worker
# ============================================================================

def optimize_worker(args):
    """Single optimization from random initial normals."""
    n_facets, seed = args
    rng = np.random.default_rng(seed)

    # Random initial normals on S^3
    normals0 = rng.standard_normal((n_facets, 4))
    normals0 /= np.linalg.norm(normals0, axis=1, keepdims=True)

    # Convert to angles
    a0 = np.arccos(np.clip(normals0[:, 3], -1, 1))
    sin_a = np.sin(a0) + 1e-30
    b0 = np.arccos(np.clip(normals0[:, 2] / sin_a, -1, 1))
    c0 = np.arctan2(normals0[:, 1], normals0[:, 0])
    x0 = np.concatenate([a0, b0, c0])

    # Optimize using Nelder-Mead (gradient-free, since inner solve
    # makes the objective non-smooth at combinatorial transitions)
    try:
        res = minimize(neg_volume_from_angles, x0, args=(n_facets,),
                       method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-8,
                                'fatol': 1e-10, 'adaptive': True})
        vol = -res.fun
    except Exception:
        vol = 0.0
        res = None

    # Also try Powell
    try:
        res2 = minimize(neg_volume_from_angles, x0, args=(n_facets,),
                        method='Powell',
                        options={'maxiter': 3000, 'ftol': 1e-12})
        vol2 = -res2.fun
        if vol2 > vol:
            vol = vol2
            res = res2
    except Exception:
        pass

    return vol, res.x if res is not None else x0


def analyze_result(angles, n_facets):
    """Full analysis of an optimized result."""
    f = n_facets
    a = angles[:f]
    b = angles[f:2*f]
    c = angles[2*f:3*f]
    sa, ca = np.sin(a), np.cos(a)
    sb, cb = np.sin(b), np.cos(b)
    sc, cc = np.sin(c), np.cos(c)
    normals = np.column_stack([sa*sb*cc, sa*sb*sc, sa*cb, ca])

    vertex_defs = get_combinatorial_type(normals)
    if vertex_defs is None:
        print("  INFEASIBLE")
        return

    A_matrices = build_vertex_maps(normals, vertex_defs)
    if A_matrices is None:
        print("  Singular normal matrix")
        return

    h_opt, cost = solve_support_numbers(normals, vertex_defs, A_matrices)
    if h_opt is None:
        print(f"  Failed to solve ideal constraint (cost={cost:.2e})")
        return

    vertices = np.array([A @ h_opt for A in A_matrices])
    r = np.linalg.norm(vertices, axis=1)
    vol = compute_volume(vertices)

    print(f"  Volume: {vol:.10f}")
    print(f"  Input normals: {n_facets}")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Max |r-1|: {np.max(np.abs(r-1)):.2e}")
    print(f"  Support numbers h: [{h_opt.min():.4f}, {h_opt.max():.4f}]")

    # f-vector of ideal polytope
    try:
        hull = ConvexHull(vertices, qhull_options='QJ')
        edges = set()
        triangles = set()
        for simplex in hull.simplices:
            for i, j in combinations(sorted(simplex), 2):
                edges.add((i, j))
            for tri in combinations(sorted(simplex), 3):
                triangles.add(tri)
        f0 = len(vertices)
        f1 = len(edges)
        f2 = len(triangles)
        f3 = len(hull.simplices)
        print(f"  f-vector: ({f0}, {f1}, {f2}, {f3})")
        print(f"  Euler: {f0 - f1 + f2 - f3}")
    except Exception as e:
        print(f"  Could not compute f-vector: {e}")

    # Delaunay decomposition
    pts_3d, _ = stereo_project_S3(vertices, pole_idx=0)
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


def run_optimization(n_facets, n_restarts=50, n_workers=32):
    """Run parallel optimization."""
    print(f"\n{'='*70}")
    print(f"Optimizing ideal 4-polytope volume with {n_facets} facet normals")
    print(f"Restarts: {n_restarts}, Workers: {n_workers}")
    print(f"{'='*70}")

    t0 = time.time()
    work = [(n_facets, 55555 + n_facets * 3333 + i)
            for i in range(n_restarts)]

    with Pool(processes=min(n_workers, n_restarts)) as pool:
        results = pool.map(optimize_worker, work)

    elapsed = time.time() - t0

    valid = [(v, x) for v, x in results if v > 0]
    valid.sort(key=lambda r: r[0], reverse=True)

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Valid: {len(valid)}/{len(results)}")

    if not valid:
        print("No valid results!")
        return None

    vols = np.array([v for v, _ in valid])
    print(f"\nVolume statistics:")
    print(f"  Best:   {vols[0]:.10f}")
    if len(vols) >= 10:
        print(f"  Top 10: {', '.join(f'{v:.4f}' for v in vols[:10])}")
    print(f"  Mean:   {vols.mean():.6f}")
    print(f"  Std:    {vols.std():.6f}")

    print(f"\nBest configuration:")
    analyze_result(valid[0][1], n_facets)

    return valid


def main():
    if len(sys.argv) < 2:
        print("Usage: python optimize_ideal_by_normals.py <n_facets> [n_restarts]")
        print()
        print("Examples:")
        print("  python optimize_ideal_by_normals.py 10 50")
        print("  python optimize_ideal_by_normals.py 24 100")
        sys.exit(1)

    n_facets = int(sys.argv[1])
    n_restarts = int(sys.argv[2]) if len(sys.argv) >= 3 else 50

    v_reg = regular_ideal_4simplex_volume()
    print(f"Regular ideal 4-simplex volume: {v_reg:.8f}")

    run_optimization(n_facets, n_restarts=n_restarts)


if __name__ == '__main__':
    main()
