#!/usr/bin/env python3
"""
Optimize ideal polytope volume parameterized by facet normals + support numbers.

Given f unit normals u_1,...,u_f on S^3, the polytope is
    P(h) = {x in R^4 : u_i . x <= h_i}

For small h_i, all vertices are inside B^4.  We project each vertex
onto S^3 via v_hat = v / |v|, then compute the H^4 volume of
conv(v_hat_1,...,v_hat_m) using the exact angle formula.

This "projected ideal volume" is a smooth surrogate that:
  - Equals the true H^4 volume when all vertices are on S^3
  - Increases as vertices approach S^3
  - Uses our fast, exact ideal simplex volume formula

The optimizer pushes h outward to grow the polytope until vertices
hit S^3. Non-inscribable types naturally have vertices coalesce.
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
    stereo_project_S3,
    regular_ideal_4simplex_volume,
)


# ============================================================================
# Polytope from normals + support numbers
# ============================================================================

def compute_vertices(normals, h):
    """
    Compute vertices of P = {x : u_i . x <= h_i}.

    Each vertex is the intersection of exactly 4 bounding hyperplanes.
    We enumerate all C(f,4) subsets, solve the 4x4 system, and check
    feasibility against all other constraints.

    Returns (vertices, vertex_defs) or (None, None).
    """
    f = len(normals)
    vertices = []
    vertex_defs = []

    for combo in combinations(range(f), 4):
        U = normals[list(combo)]
        h_sub = h[list(combo)]

        try:
            v = np.linalg.solve(U, h_sub)
        except np.linalg.LinAlgError:
            continue

        # Check feasibility: u_i . v <= h_i + eps for all i
        if np.all(normals @ v <= h + 1e-10):
            vertices.append(v)
            vertex_defs.append(combo)

    if len(vertices) < 5:
        return None, None

    vertices = np.array(vertices)

    # Remove near-duplicates
    unique = [0]
    for i in range(1, len(vertices)):
        is_dup = False
        for j in unique:
            if np.linalg.norm(vertices[i] - vertices[j]) < 1e-10:
                is_dup = True
                break
        if not is_dup:
            unique.append(i)

    vertices = vertices[unique]
    vertex_defs = [vertex_defs[i] for i in unique]

    return vertices, vertex_defs


def compute_vertices_fast(normals, h):
    """
    Fast vertex computation using polar duality.
    Only works when all h_i > 0.
    Falls back to enumeration otherwise.
    """
    if np.any(h <= 1e-10):
        return compute_vertices(normals, h)

    f = len(normals)
    w = normals / h[:, None]

    try:
        hull = ConvexHull(w)
    except Exception:
        return compute_vertices(normals, h)

    # Check origin is inside
    if np.any(hull.equations[:, -1] > 1e-10):
        return compute_vertices(normals, h)

    vertices = []
    vertex_defs = []
    for simplex in hull.simplices:
        W = w[simplex]
        try:
            v = np.linalg.solve(W, np.ones(4))
            vertices.append(v)
            vertex_defs.append(tuple(sorted(simplex)))
        except np.linalg.LinAlgError:
            continue

    if len(vertices) < 5:
        return compute_vertices(normals, h)

    return np.array(vertices), vertex_defs


# ============================================================================
# Projected ideal volume
# ============================================================================

def projected_ideal_volume(vertices):
    """
    Project vertices onto S^3, then compute ideal polytope volume.

    v_hat = v / |v|, then Vol = sum of ideal 4-simplex volumes
    in the Delaunay triangulation of the projected points.
    """
    if vertices is None or len(vertices) < 5:
        return 0.0

    norms = np.linalg.norm(vertices, axis=1)
    if np.any(norms < 1e-10):
        return 0.0

    # Project onto S^3
    ideal_pts = vertices / norms[:, None]

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
            vol = ideal_4simplex_volume_from_tet(v)
            if vol > 0:
                total += vol
        except Exception:
            pass

    return total


# ============================================================================
# Objective function
# ============================================================================

def neg_projected_volume(h, normals, use_fast=True):
    """
    Negative projected ideal volume as function of support numbers.
    """
    if use_fast:
        verts, _ = compute_vertices_fast(normals, h)
    else:
        verts, _ = compute_vertices(normals, h)

    if verts is None:
        return 1e6

    # Soft constraint: penalize vertices outside B^4
    r = np.linalg.norm(verts, axis=1)
    max_r = np.max(r)
    if max_r > 1.0:
        penalty = 1e4 * (max_r - 1.0)**2
        return penalty

    vol = projected_ideal_volume(verts)
    return -vol


# ============================================================================
# Optimization
# ============================================================================

def optimize_from_normals(normals, h0=None, verbose=True):
    """
    Optimize support numbers to maximize projected ideal volume.
    """
    f = len(normals)

    if h0 is None:
        h0 = np.full(f, 0.15)

    # Bounds
    bounds = [(-3.0, 3.0)] * f

    # Phase 1: Nelder-Mead
    res = minimize(neg_projected_volume, h0, args=(normals,),
                   method='Nelder-Mead',
                   options={'maxiter': 3000, 'xatol': 1e-6,
                            'fatol': 1e-8, 'adaptive': True})

    # Phase 2: Powell (polish)
    res2 = minimize(neg_projected_volume, res.x, args=(normals,),
                    method='Powell',
                    options={'maxiter': 1000, 'ftol': 1e-10})
    if res2.fun < res.fun:
        res = res2
    h_opt = res.x
    vol = -res.fun

    verts, vdefs = compute_vertices_fast(normals, h_opt)

    if verts is None:
        if verbose:
            print("  FAILED: degenerate polytope")
        return None

    r = np.linalg.norm(verts, axis=1)

    if verbose:
        print(f"  Volume: {vol:.8f}")
        print(f"  Vertices: {len(verts)}")
        print(f"  Vertex |v|: min={r.min():.6f}, max={r.max():.6f}")
        print(f"  # on S^3 (|v|>0.999): {np.sum(r > 0.999)}/{len(verts)}")
        print(f"  h range: [{h_opt.min():.4f}, {h_opt.max():.4f}]")
        print(f"  Optimizer: {res.message}")

    return {
        'volume': vol,
        'h': h_opt,
        'vertices': verts,
        'normals': normals,
        'n_facets': f,
        'n_vertices': len(verts),
        'max_r': r.max(),
        'n_ideal': int(np.sum(r > 0.999)),
    }


def optimize_worker(args):
    """Worker for parallel optimization."""
    n_facets, seed = args
    rng = np.random.default_rng(seed)
    normals = rng.standard_normal((n_facets, 4))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    result = optimize_from_normals(normals, verbose=False)
    if result is not None:
        return result
    return None


# ============================================================================
# Analysis
# ============================================================================

def analyze_result(result):
    """Full analysis of an optimization result."""
    verts = result['vertices']
    normals = result['normals']
    h = result['h']
    f = result['n_facets']

    r = np.linalg.norm(verts, axis=1)
    ideal_pts = verts / r[:, None]

    print(f"\n  f-vector analysis:")
    try:
        hull = ConvexHull(ideal_pts, qhull_options='QJ')
        edges = set()
        triangles = set()
        for simplex in hull.simplices:
            for i, j in combinations(sorted(simplex), 2):
                edges.add((i, j))
            for tri in combinations(sorted(simplex), 3):
                triangles.add(tri)
        f0, f1, f2, f3 = len(ideal_pts), len(edges), len(triangles), len(hull.simplices)
        print(f"  f-vector: ({f0}, {f1}, {f2}, {f3})")
        print(f"  Euler: {f0 - f1 + f2 - f3}")
    except Exception as e:
        print(f"  ConvexHull error: {e}")

    # Degree sequence
    deg = Counter()
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
    if deg:
        deg_seq = sorted([deg[i] for i in range(f0)], reverse=True)
        print(f"  Degree seq: {deg_seq[:15]}{'...' if len(deg_seq) > 15 else ''}")

    # Delaunay decomposition
    pts_3d, _ = stereo_project_S3(ideal_pts, pole_idx=0)
    tri = Delaunay(pts_3d)
    tet_vols = []
    for tet_idx in tri.simplices:
        v = pts_3d[tet_idx]
        try:
            tv = ideal_4simplex_volume_from_tet(v)
            if tv > 0:
                tet_vols.append(tv)
        except Exception:
            pass
    tet_vols = np.array(tet_vols)
    v_reg = regular_ideal_4simplex_volume()
    print(f"  Delaunay: {len(tet_vols)} simplices")
    print(f"  Tet volumes: min={tet_vols.min():.6f}, max={tet_vols.max():.6f}, "
          f"mean={tet_vols.mean():.6f}")
    print(f"  Max tet / regular: {tet_vols.max()/v_reg:.6f}")

    # Vertex radii histogram
    print(f"\n  Vertex radius distribution:")
    for thresh in [0.5, 0.9, 0.99, 0.999, 0.9999]:
        n_above = np.sum(r > thresh)
        print(f"    |v| > {thresh}: {n_above}/{len(r)}")


# ============================================================================
# Main
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python optimize_projected_volume.py <n_facets> [n_restarts]")
        sys.exit(1)

    n_facets = int(sys.argv[1])
    n_restarts = int(sys.argv[2]) if len(sys.argv) >= 3 else 20

    v_reg = regular_ideal_4simplex_volume()
    print(f"Projected ideal volume optimization")
    print(f"Facets: {n_facets}, Restarts: {n_restarts}")
    print(f"Regular ideal 4-simplex volume: {v_reg:.8f}")
    print()

    t0 = time.time()

    if n_restarts <= 3:
        # Sequential with verbose
        results = []
        for i in range(n_restarts):
            rng = np.random.default_rng(88888 + n_facets * 1000 + i)
            normals = rng.standard_normal((n_facets, 4))
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)

            print(f"Trial {i+1}/{n_restarts}:")
            result = optimize_from_normals(normals, verbose=True)
            if result is not None:
                results.append(result)
            print()
    else:
        # Parallel
        work = [(n_facets, 88888 + n_facets * 1000 + i)
                for i in range(n_restarts)]
        with Pool(processes=min(32, n_restarts)) as pool:
            raw = pool.map(optimize_worker, work)
        results = [r for r in raw if r is not None]

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")
    print(f"Valid: {len(results)}/{n_restarts}")

    if not results:
        print("No valid results!")
        return

    # Sort by volume
    results.sort(key=lambda r: r['volume'], reverse=True)

    vols = np.array([r['volume'] for r in results])
    n_verts_arr = np.array([r['n_vertices'] for r in results])
    n_ideal_arr = np.array([r['n_ideal'] for r in results])

    print(f"\nVolume statistics:")
    print(f"  Best:   {vols[0]:.10f}")
    print(f"  Top 10: {', '.join(f'{v:.4f}' for v in vols[:10])}")
    print(f"  Mean:   {vols.mean():.6f}")
    print(f"  Std:    {vols.std():.6f}")

    print(f"\nVertex counts: min={n_verts_arr.min()}, max={n_verts_arr.max()}, "
          f"mean={n_verts_arr.mean():.1f}")
    print(f"Ideal vertices: min={n_ideal_arr.min()}, max={n_ideal_arr.max()}, "
          f"mean={n_ideal_arr.mean():.1f}")

    # Analyze best
    print(f"\n{'='*60}")
    print(f"Best result:")
    print(f"{'='*60}")
    best = results[0]
    print(f"  Volume: {best['volume']:.10f}")
    print(f"  Vertices: {best['n_vertices']}")
    print(f"  Ideal vertices: {best['n_ideal']}")
    print(f"  Max |v|: {best['max_r']:.8f}")
    analyze_result(best)

    # Volume vs number of facets table
    if n_restarts > 5:
        print(f"\n{'='*60}")
        print(f"Summary for f = {n_facets} facet normals")
        print(f"{'='*60}")
        print(f"{'volume':>12s}  {'n_vert':>6s}  {'n_ideal':>7s}  {'max_r':>8s}")
        for r in results[:20]:
            print(f"{r['volume']:12.6f}  {r['n_vertices']:6d}  "
                  f"{r['n_ideal']:7d}  {r['max_r']:8.6f}")


if __name__ == '__main__':
    main()
