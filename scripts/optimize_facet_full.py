#!/usr/bin/env python3
"""
Full two-level optimization of ideal 4-polytope volume by facet normals.

Outer: vary f normals on S^3 (3f parameters) to maximize volume.
Inner: solve |v_j|^2 = 1 for support numbers h via Levenberg-Marquardt.

Pipeline:
  1. Scan random normals, keep those where inner solve succeeds.
  2. From the best starting points, run Powell on the normal angles.
  3. Report best volumes with full analysis.
"""

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial import ConvexHull, Delaunay
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
# Helpers
# ============================================================================

def angles_to_S3(angles, n):
    """Convert 3n angles to n points on S^3."""
    a, b, c = angles[:n], angles[n:2*n], angles[2*n:3*n]
    sa, ca = np.sin(a), np.cos(a)
    sb, cb = np.sin(b), np.cos(b)
    sc, cc = np.sin(c), np.cos(c)
    return np.column_stack([sa*sb*cc, sa*sb*sc, sa*cb, ca])


def normals_to_angles(normals):
    """Convert S^3 points to 3n angles."""
    a = np.arccos(np.clip(normals[:, 3], -1, 1))
    sin_a = np.sin(a) + 1e-30
    b = np.arccos(np.clip(normals[:, 2] / sin_a, -1, 1))
    c = np.arctan2(normals[:, 1], normals[:, 0])
    return np.concatenate([a, b, c])


# ============================================================================
# Inner solver
# ============================================================================

def solve_inner(normals, h_probe=None):
    """
    Given normals, find support numbers h such that all vertices
    of P = {x : u_i.x <= h_i} lie on S^3.

    Returns (h, vertices, A_matrices) or (None, None, None).
    """
    f = len(normals)

    probes = [0.3, 0.1, 0.5, 0.2, 0.7, 1.0]
    if h_probe is not None:
        probes = [h_probe] + probes

    for hp in probes:
        w = normals / hp
        try:
            hull = ConvexHull(w)
            if np.any(hull.equations[:, -1] > 1e-10):
                continue
        except Exception:
            continue

        # Build vertex linear maps
        Am = []
        ok = True
        for s in hull.simplices:
            U = normals[list(s)]
            try:
                Ui = np.linalg.inv(U)
                A = np.zeros((4, f))
                for k, idx in enumerate(s):
                    A[:, idx] = Ui[:, k]
                Am.append(A)
            except np.linalg.LinAlgError:
                ok = False
                break
        if not ok:
            continue

        nv = len(Am)

        def residuals(h):
            return np.array([np.dot(A @ h, A @ h) - 1.0 for A in Am])

        def jacobian(h):
            return np.array([2.0 * (A.T @ (A @ h)) for A in Am])

        # Try multiple initial h
        for h_init in [hp, 0.3, 0.5]:
            result = least_squares(
                residuals, np.full(f, h_init), jac=jacobian,
                method='lm', ftol=1e-15, xtol=1e-15, gtol=1e-15,
                max_nfev=3000
            )
            if result.cost < 1e-20:
                h_opt = result.x
                verts = np.array([A @ h_opt for A in Am])
                r = np.linalg.norm(verts, axis=1)
                if np.max(np.abs(r - 1)) < 1e-6:
                    return h_opt, verts, Am

    return None, None, None


def compute_volume_from_verts(verts):
    """Compute H^4 volume from ideal vertices on S^3."""
    if verts is None or len(verts) < 5:
        return 0.0
    pts_3d, _ = stereo_project_S3(verts, pole_idx=0)
    try:
        tri = Delaunay(pts_3d)
        return sum(
            ideal_4simplex_volume_from_tet(pts_3d[t])
            for t in tri.simplices
        )
    except Exception:
        return 0.0


# ============================================================================
# Outer objective
# ============================================================================

# Global cache for warm-starting inner solver
_cache = {'h_prev': None}


def outer_objective(normal_angles, f):
    """
    Negative volume as function of normal angles.
    Solves inner problem at each evaluation.
    """
    normals = angles_to_S3(normal_angles, f)

    # Use cached h as probe
    h_probe = None
    if _cache['h_prev'] is not None and len(_cache['h_prev']) == f:
        h_probe = np.mean(np.abs(_cache['h_prev']))

    h_opt, verts, Am = solve_inner(normals, h_probe=h_probe)

    if h_opt is None:
        return 0.0  # infeasible; return 0 not penalty (so optimizer avoids)

    _cache['h_prev'] = h_opt
    vol = compute_volume_from_verts(verts)
    return -vol


# ============================================================================
# Workers
# ============================================================================

def scan_worker(args):
    """Try random normals, return (vol, angles, n_verts) or None."""
    f, seed = args
    rng = np.random.default_rng(seed)
    normals = rng.standard_normal((f, 4))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    h_opt, verts, Am = solve_inner(normals)
    if h_opt is None:
        return None

    vol = compute_volume_from_verts(verts)
    angles = normals_to_angles(normals)
    return (vol, angles, len(verts))


def optimize_worker(args):
    """Outer optimization from given starting angles."""
    f, start_angles, trial_id = args
    global _cache
    _cache = {'h_prev': None}

    # Powell optimization
    res = minimize(
        outer_objective, start_angles, args=(f,),
        method='Powell',
        options={'maxiter': 1000, 'ftol': 1e-10}
    )

    vol = -res.fun if res.fun < 0 else 0.0

    # Get final vertex info
    normals = angles_to_S3(res.x, f)
    h_opt, verts, _ = solve_inner(normals)
    nv = len(verts) if verts is not None else 0

    return (vol, res.x, nv, trial_id)


# ============================================================================
# Main
# ============================================================================

def analyze_best(angles, f):
    """Full analysis of the best configuration."""
    normals = angles_to_S3(angles, f)
    h_opt, verts, Am = solve_inner(normals)

    if h_opt is None:
        print("  Failed to solve inner problem!")
        return

    r = np.linalg.norm(verts, axis=1)
    vol = compute_volume_from_verts(verts)

    print(f"  Volume: {vol:.10f}")
    print(f"  Normals (facets): {f}")
    print(f"  Vertices: {len(verts)}")
    print(f"  Vertex |v|: min={r.min():.10f}, max={r.max():.10f}")
    print(f"  Max |r-1|: {np.max(np.abs(r-1)):.2e}")
    print(f"  h range: [{h_opt.min():.6f}, {h_opt.max():.6f}]")

    # f-vector
    try:
        hull = ConvexHull(verts, qhull_options='QJ')
        edges = set()
        triangles = set()
        for simplex in hull.simplices:
            for i, j in combinations(sorted(simplex), 2):
                edges.add((i, j))
            for tri in combinations(sorted(simplex), 3):
                triangles.add(tri)
        f0, f1, f2, f3 = len(verts), len(edges), len(triangles), len(hull.simplices)
        print(f"  f-vector: ({f0}, {f1}, {f2}, {f3})")
        print(f"  Euler: {f0 - f1 + f2 - f3}")
    except Exception:
        pass

    # Degree sequence
    if edges:
        deg = Counter()
        for i, j in edges:
            deg[i] += 1
            deg[j] += 1
        deg_seq = sorted([deg[i] for i in range(f0)], reverse=True)
        print(f"  Degree seq: {deg_seq[:20]}{'...' if len(deg_seq) > 20 else ''}")

    # Delaunay
    pts_3d, _ = stereo_project_S3(verts, pole_idx=0)
    tri = Delaunay(pts_3d)
    tet_vols = []
    for t in tri.simplices:
        try:
            tv = ideal_4simplex_volume_from_tet(pts_3d[t])
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


def main():
    f = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    n_scan = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    n_optimize = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    v_reg = regular_ideal_4simplex_volume()
    print(f"{'='*70}")
    print(f"Two-level optimization: f = {f} facet normals")
    print(f"Regular ideal 4-simplex volume: {v_reg:.8f}")
    print(f"Phase 1: scan {n_scan} random normals")
    print(f"Phase 2: optimize top {n_optimize} starting points")
    print(f"{'='*70}")

    # Phase 1: Scan
    print(f"\nPhase 1: Scanning...")
    t0 = time.time()
    work = [(f, 42000 + f * 1000 + i) for i in range(n_scan)]

    with Pool(processes=32) as pool:
        scan_results = pool.map(scan_worker, work)

    valid_scans = [r for r in scan_results if r is not None]
    valid_scans.sort(key=lambda r: r[0], reverse=True)
    scan_time = time.time() - t0

    print(f"  Found {len(valid_scans)}/{n_scan} valid configs in {scan_time:.1f}s")
    if not valid_scans:
        print("  No valid configurations found!")
        return

    scan_vols = [r[0] for r in valid_scans]
    print(f"  Scan volumes: best={scan_vols[0]:.4f}, "
          f"mean={np.mean(scan_vols):.4f}, "
          f"top5={[f'{v:.2f}' for v in scan_vols[:5]]}")

    # Phase 2: Optimize top starts
    n_opt = min(n_optimize, len(valid_scans))
    print(f"\nPhase 2: Optimizing top {n_opt} starting points...")
    t1 = time.time()

    opt_work = [(f, valid_scans[i][1], i) for i in range(n_opt)]

    # Run in parallel (each worker does sequential Powell)
    with Pool(processes=min(32, n_opt)) as pool:
        opt_results = pool.map(optimize_worker, opt_work)

    opt_results = [r for r in opt_results if r[0] > 0]
    opt_results.sort(key=lambda r: r[0], reverse=True)
    opt_time = time.time() - t1

    print(f"  Optimized {len(opt_results)} configs in {opt_time:.1f}s")

    if not opt_results:
        print("  Optimization produced no valid results!")
        print("  Using best scan result instead.")
        best_angles = valid_scans[0][1]
        best_vol = valid_scans[0][0]
    else:
        opt_vols = [r[0] for r in opt_results]
        print(f"  Optimized volumes: best={opt_vols[0]:.4f}, "
              f"top5={[f'{v:.2f}' for v in opt_vols[:5]]}")

        best_vol = opt_results[0][0]
        best_angles = opt_results[0][1]

        # Compare scan vs optimized
        print(f"\n  Improvement: {scan_vols[0]:.4f} -> {opt_vols[0]:.4f} "
              f"({100*(opt_vols[0]/scan_vols[0]-1):.1f}%)")

    # Phase 3: Analyze best
    print(f"\n{'='*70}")
    print(f"Best result for f = {f} facet normals")
    print(f"{'='*70}")
    analyze_best(best_angles, f)

    print(f"\n{'='*70}")
    print(f"Total time: {time.time()-t0:.1f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
