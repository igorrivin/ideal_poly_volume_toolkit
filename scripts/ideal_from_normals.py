#!/usr/bin/env python3
"""
Construct ideal polytopes in H^4 from prescribed facet normals.

Given f unit vectors u_1,...,u_f on S^3 (facet normals), find support
numbers h_1,...,h_f such that

    P = {x in R^4 : u_i . x <= h_i  for all i}

has ALL vertices on S^3 (i.e., P is an ideal polytope in the Klein model).

Algorithm:
  1. Fix normals u_i on S^3.
  2. Set w_i = u_i / h_i (scaled normals).
  3. Q = conv(w_1,...,w_f).  Each facet of Q gives a vertex v of P
     by solving W_I . v = 1.
  4. Minimize L(h) = sum_v (|v|^2 - 1)^2 over h.
  5. When L = 0, all vertices lie on S^3.

The optimization is f-dimensional (one variable per facet).
For fixed combinatorial type, vertices depend linearly on h,
so the objective is quartic — smooth and well-behaved.

Once the ideal constraint is solved, we can compute H^4 volume
and then optimize over the normals themselves.
"""

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize, differential_evolution
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
# Core: normals + support numbers -> vertices
# ============================================================================

def compute_vertices(normals, h):
    """
    Compute vertices of P = {x : u_i . x <= h_i}.

    Uses polar duality: set w_i = u_i / h_i, compute Q = conv(w),
    each facet of Q gives a vertex of P.

    Returns
    -------
    vertices : (m, 4) array, or None if infeasible
    facet_indices : list of tuples, which normals define each vertex
    """
    f = len(normals)

    # Scaled normals
    # h_i must be positive for this to work
    if np.any(h <= 0):
        return None, None

    w = normals / h[:, None]

    try:
        hull = ConvexHull(w)
    except Exception:
        return None, None

    # Check origin is inside Q (needed for bounded P)
    # hull.equations: [n, d] where n.x + d <= 0 for interior
    if np.any(hull.equations[:, -1] > 1e-10):
        return None, None

    vertices = []
    facet_indices = []
    for simplex in hull.simplices:
        W = w[simplex]  # (4, 4) matrix
        try:
            v = np.linalg.solve(W, np.ones(4))
            vertices.append(v)
            facet_indices.append(tuple(simplex))
        except np.linalg.LinAlgError:
            continue

    if len(vertices) == 0:
        return None, None

    return np.array(vertices), facet_indices


def ideal_loss(h, normals):
    """
    Loss function: sum_v (|v|^2 - 1)^2.
    Zero iff all vertices lie on S^3.
    """
    vertices, _ = compute_vertices(normals, h)
    if vertices is None:
        return 1e10

    r2 = np.sum(vertices**2, axis=1)
    return np.sum((r2 - 1.0)**2)


def ideal_loss_with_grad(h, normals):
    """
    Loss and gradient.

    For vertex v defined by W_I . v = 1 (where W_I = U_I / h_I),
    changing h_k (for k in I) changes w_k = u_k / h_k, hence v.

    Let v = W_I^{-1} . 1.  Then dv/dh_k = -W_I^{-1} . (dW_I/dh_k) . v
    where dW_I/dh_k has only one nonzero row: d(u_k/h_k)/dh_k = -u_k/h_k^2.

    So dv/dh_k = (1/h_k^2) * W_I^{-1} . e_j . (u_k . v)
    where j is the position of k in I... actually let me be more careful.

    W_I . v = 1, where row j of W_I is u_{I_j} / h_{I_j}.
    If k = I_j, then d(row j)/dh_k = -u_k / h_k^2.
    So d(W_I . v)/dh_k = -e_j (u_k . v) / h_k^2 + W_I . dv/dh_k = 0
    Hence W_I . dv/dh_k = e_j (u_k . v) / h_k^2
    So dv/dh_k = (u_k . v / h_k^2) * W_I^{-1} . e_j
             = (u_k . v / h_k^2) * (column j of W_I^{-1})

    Loss L = sum_v (|v|^2 - 1)^2
    dL/dh_k = sum_{v containing k} 2(|v|^2 - 1) * 2 v . dv/dh_k
    """
    vertices, facet_indices = compute_vertices(normals, h)
    if vertices is None:
        return 1e10, np.zeros_like(h)

    f = len(h)
    r2 = np.sum(vertices**2, axis=1)
    residuals = r2 - 1.0
    loss = np.sum(residuals**2)

    grad = np.zeros(f)

    for vi, (v, indices) in enumerate(zip(vertices, facet_indices)):
        if abs(residuals[vi]) < 1e-15:
            continue

        W = normals[list(indices)] / h[list(indices), None]
        try:
            W_inv = np.linalg.inv(W)
        except np.linalg.LinAlgError:
            continue

        for j, k in enumerate(indices):
            uk_dot_v = normals[k] @ v
            dv_dhk = (uk_dot_v / h[k]**2) * W_inv[:, j]
            v_dot_dv = v @ dv_dhk
            grad[k] += 4.0 * residuals[vi] * v_dot_dv

    return loss, grad


# ============================================================================
# Solve for ideal polytope
# ============================================================================

def solve_ideal(normals, h0=None, method='L-BFGS-B', verbose=True):
    """
    Find support numbers h such that P(normals, h) is an ideal polytope.

    Parameters
    ----------
    normals : (f, 4) array of unit vectors on S^3
    h0 : initial guess for support numbers (default: all 0.5)

    Returns
    -------
    h_opt : optimal support numbers, or None if failed
    vertices : ideal vertices on S^3, or None
    loss : final loss value
    """
    f = len(normals)

    if h0 is None:
        h0 = np.full(f, 0.5)

    # Bounds: h_i > 0
    bounds = [(0.01, 5.0)] * f

    if method == 'L-BFGS-B':
        res = minimize(
            lambda h: ideal_loss_with_grad(h, normals),
            h0, method='L-BFGS-B', jac=True,
            bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-30, 'gtol': 1e-15}
        )
    else:
        res = minimize(
            ideal_loss, h0, args=(normals,),
            method=method, bounds=bounds,
            options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-30}
        )

    h_opt = res.x
    loss = res.fun

    if verbose:
        print(f"  Optimization: {res.message}")
        print(f"  Final loss: {loss:.2e}")
        print(f"  Iterations: {res.nit}")

    vertices, facet_indices = compute_vertices(normals, h_opt)

    if vertices is not None and loss < 1e-10:
        # Check actual residuals
        r = np.linalg.norm(vertices, axis=1)
        if verbose:
            print(f"  Vertex radii: min={r.min():.10f}, max={r.max():.10f}")
            print(f"  Max |r - 1|: {np.max(np.abs(r - 1)):.2e}")
        return h_opt, vertices, loss
    else:
        if verbose:
            print(f"  FAILED to find ideal polytope (loss = {loss:.2e})")
        return None, None, loss


# ============================================================================
# Volume computation
# ============================================================================

def compute_ideal_volume(ideal_pts):
    """Compute H^4 volume of ideal polytope given vertices on S^3."""
    if ideal_pts is None or len(ideal_pts) < 5:
        return 0.0

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
# Full pipeline: random normals -> ideal polytope -> volume
# ============================================================================

def random_ideal_polytope(n_facets, seed=None, verbose=False):
    """
    Generate a random ideal polytope with n_facets facets.

    1. Pick n_facets random unit vectors on S^3
    2. Solve for support numbers making all vertices ideal
    3. Return vertices, volume, diagnostics
    """
    rng = np.random.default_rng(seed)
    normals = rng.standard_normal((n_facets, 4))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    # Try multiple initial h values
    best_result = None
    best_loss = np.inf

    for h_init in [0.5, 0.3, 0.7, 1.0, 0.1]:
        h0 = np.full(n_facets, h_init)
        # Also try random perturbation
        h0 *= (1 + 0.1 * rng.standard_normal(n_facets))
        h0 = np.clip(h0, 0.01, 5.0)

        h_opt, verts, loss = solve_ideal(normals, h0, verbose=False)
        if loss < best_loss:
            best_loss = loss
            best_result = (h_opt, verts, loss, normals)

    h_opt, verts, loss, normals = best_result

    if verbose:
        if verts is not None and loss < 1e-10:
            vol = compute_ideal_volume(verts)
            r = np.linalg.norm(verts, axis=1)
            print(f"  Success: {len(verts)} vertices, vol={vol:.6f}, "
                  f"max|r-1|={np.max(np.abs(r-1)):.2e}, loss={loss:.2e}")
        else:
            print(f"  Failed: loss={loss:.2e}")

    if verts is not None and loss < 1e-10:
        vol = compute_ideal_volume(verts)
        return {
            'normals': normals,
            'h': h_opt,
            'vertices': verts,
            'volume': vol,
            'loss': loss,
            'n_facets': n_facets,
            'n_vertices': len(verts),
        }
    return None


def worker_random_ideal(args):
    """Worker for parallel random ideal polytope generation."""
    n_facets, seed = args
    result = random_ideal_polytope(n_facets, seed=seed, verbose=False)
    if result is not None:
        return (result['volume'], result['n_vertices'], result['loss'],
                result['normals'], result['h'])
    return None


# ============================================================================
# Main: study distribution and optimize
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python ideal_from_normals.py <n_facets> [n_samples] [mode]")
        print()
        print("Modes:")
        print("  sample  - generate random ideal polytopes, report distribution")
        print("  test    - single test with verbose output")
        sys.exit(1)

    n_facets = int(sys.argv[1])
    n_samples = int(sys.argv[2]) if len(sys.argv) >= 3 else 50
    mode = sys.argv[3] if len(sys.argv) >= 4 else 'sample'

    v_reg = regular_ideal_4simplex_volume()
    print(f"Regular ideal 4-simplex volume: {v_reg:.8f}")

    if mode == 'test':
        print(f"\nSingle test: {n_facets} facet normals")
        print("=" * 60)
        result = random_ideal_polytope(n_facets, seed=42, verbose=True)
        if result:
            print(f"\n  Volume: {result['volume']:.10f}")
            print(f"  Vertices: {result['n_vertices']}")
            print(f"  Facets (input): {result['n_facets']}")

            # Check actual facet count of ideal polytope
            hull = ConvexHull(result['vertices'])
            actual_f3 = len(hull.simplices)
            print(f"  Facets (actual): {actual_f3}")

            # f-vector
            edges = set()
            for simplex in hull.simplices:
                for i, j in combinations(sorted(simplex), 2):
                    edges.add((i, j))
            print(f"  f-vector: ({result['n_vertices']}, {len(edges)}, "
                  f"?, {actual_f3})")
        return

    # Sample mode
    print(f"\nSampling {n_samples} random ideal polytopes with {n_facets} facets")
    print("=" * 60)

    t0 = time.time()
    work = [(n_facets, 12345 + i) for i in range(n_samples)]

    with Pool(processes=32) as pool:
        results = pool.map(worker_random_ideal, work)

    elapsed = time.time() - t0
    valid = [r for r in results if r is not None]

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Success rate: {len(valid)}/{n_samples} "
          f"({100*len(valid)/n_samples:.1f}%)")

    if not valid:
        print("No valid results!")
        return

    volumes = np.array([r[0] for r in valid])
    n_verts = np.array([r[1] for r in valid])
    losses = np.array([r[2] for r in valid])

    print(f"\nVolume distribution:")
    print(f"  min:    {volumes.min():.6f}")
    print(f"  max:    {volumes.max():.6f}")
    print(f"  mean:   {volumes.mean():.6f}")
    print(f"  std:    {volumes.std():.6f}")
    print(f"  median: {np.median(volumes):.6f}")

    print(f"\nVertex count distribution:")
    print(f"  min:    {n_verts.min()}")
    print(f"  max:    {n_verts.max()}")
    print(f"  mean:   {n_verts.mean():.1f}")
    vert_counts = Counter(n_verts)
    print(f"  distribution:")
    for nv in sorted(vert_counts.keys()):
        pct = 100 * vert_counts[nv] / len(valid)
        bar = '#' * int(pct)
        print(f"    {nv:4d}: {vert_counts[nv]:4d}  ({pct:5.1f}%)  {bar}")

    print(f"\nLoss (should be ~0):")
    print(f"  max loss: {losses.max():.2e}")
    print(f"  mean loss: {losses.mean():.2e}")

    # Top 10 by volume
    idx = np.argsort(volumes)[::-1]
    print(f"\nTop 10 by volume:")
    for i in range(min(10, len(valid))):
        j = idx[i]
        print(f"  #{i+1}: vol={valid[j][0]:.6f}, "
              f"n_verts={valid[j][1]}, loss={valid[j][2]:.2e}")

    # Check actual facet counts for top results
    print(f"\nActual facet counts (top 10):")
    for i in range(min(10, len(valid))):
        j = idx[i]
        normals, h = valid[j][3], valid[j][4]
        verts, _ = compute_vertices(normals, h)
        if verts is not None:
            try:
                hull = ConvexHull(verts)
                print(f"  #{i+1}: input f={n_facets}, "
                      f"actual f3={len(hull.simplices)}, "
                      f"n_verts={len(verts)}")
            except Exception:
                print(f"  #{i+1}: hull computation failed")


if __name__ == '__main__':
    main()
