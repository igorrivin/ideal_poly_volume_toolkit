#!/usr/bin/env python3
"""
Optimize hyperbolic 4-volume of polytopes in the Klein model,
parameterized by facet normals and support numbers.

Given f unit normals u_1,...,u_f on S^3, the polytope is
    P(h) = {x in R^4 : u_i . x <= h_i}

For small h_i, all vertices are well inside B^4.  We maximize
the hyperbolic volume of P(h), subject to |v| <= 1 for all vertices.

The volume element in the Klein model of H^4 is:
    dV_hyp = d^4x / (1 - |x|^2)^{5/2}

For each 4-simplex in a triangulation of P, we integrate this
numerically using Gauss-Jacobi quadrature on the Duffy-transformed
standard simplex.

As h_i grow, vertices approach S^3 and volume increases.
At the optimum, the polytope is (approximately) ideal.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations
from collections import Counter
from multiprocessing import Pool
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Quadrature for the Klein model volume of a 4-simplex
# ============================================================================

def _build_quadrature(n_per_dim=6):
    """
    Build quadrature points and weights for the standard 4-simplex
    using Duffy transform + tensor-product Gauss-Jacobi rules.

    The standard 4-simplex integral:
        int_{lambda_i >= 0, sum <= 1} f(lambda) d^4 lambda

    is transformed via:
        lambda_1 = t_1
        lambda_2 = (1-t_1) t_2
        lambda_3 = (1-t_1)(1-t_2) t_3
        lambda_4 = (1-t_1)(1-t_2)(1-t_3) t_4

    with Jacobian (1-t_1)^3 (1-t_2)^2 (1-t_3), each t_i in [0,1].

    We use Gauss-Jacobi quadrature for each dimension:
        t_1: weight (1-t)^3 -> Jacobi(alpha=0, beta=3)
        t_2: weight (1-t)^2 -> Jacobi(alpha=0, beta=2)
        t_3: weight (1-t)^1 -> Jacobi(alpha=0, beta=1)
        t_4: weight 1        -> Gauss-Legendre

    Returns (points, weights) where points is (N, 4) in barycentric
    coords (lambda_0,...,lambda_4 with lambda_0 = 1 - sum lambda_i)
    mapped to physical simplex, and weights include the Jacobian.
    """
    from numpy.polynomial.legendre import leggauss

    # Gauss-Jacobi nodes and weights on [-1, 1]
    # For Jacobi(alpha, beta), scipy has roots_jacobi
    from scipy.special import roots_jacobi

    # Get nodes/weights for each dimension, mapped to [0, 1]
    def jacobi_01(n, alpha, beta):
        nodes, weights = roots_jacobi(n, alpha, beta)
        # Map from [-1,1] to [0,1]: t = (x+1)/2, dt = dx/2
        # Weight function (1-x)^beta (1+x)^alpha -> (1-2t+1)^beta (2t)^alpha * 2
        # But roots_jacobi already includes the weight, so we just transform
        t = (nodes + 1) / 2
        w = weights / (2**(alpha + beta + 1))  # Adjust for [0,1]
        return t, w

    def legendre_01(n):
        nodes, weights = leggauss(n)
        t = (nodes + 1) / 2
        w = weights / 2
        return t, w

    n = n_per_dim
    t1, w1 = jacobi_01(n, 0, 3)  # weight (1-t)^3
    t2, w2 = jacobi_01(n, 0, 2)  # weight (1-t)^2
    t3, w3 = jacobi_01(n, 0, 1)  # weight (1-t)^1
    t4, w4 = legendre_01(n)       # weight 1

    # Build tensor product
    all_lambdas = []
    all_weights = []

    for i1 in range(n):
        for i2 in range(n):
            for i3 in range(n):
                for i4 in range(n):
                    lam1 = t1[i1]
                    lam2 = (1 - t1[i1]) * t2[i2]
                    lam3 = (1 - t1[i1]) * (1 - t2[i2]) * t3[i3]
                    lam4 = (1 - t1[i1]) * (1 - t2[i2]) * (1 - t3[i3]) * t4[i4]
                    lam0 = 1 - lam1 - lam2 - lam3 - lam4

                    wt = w1[i1] * w2[i2] * w3[i3] * w4[i4]

                    all_lambdas.append([lam0, lam1, lam2, lam3, lam4])
                    all_weights.append(wt)

    return np.array(all_lambdas), np.array(all_weights)


# Precompute quadrature rules (module-level)
# Fast rule for optimization
_QUAD_LAMBDAS_FAST, _QUAD_WEIGHTS_FAST = _build_quadrature(n_per_dim=4)
# Accurate rule for final evaluation
_QUAD_LAMBDAS, _QUAD_WEIGHTS = _build_quadrature(n_per_dim=6)


def klein_simplex_volume(vertices, quad_lambdas=None, quad_weights=None):
    """
    Compute the hyperbolic 4-volume of a 4-simplex in the Klein model.

    Parameters
    ----------
    vertices : (5, 4) array
        Vertices of the 4-simplex in the Klein ball B^4.
        All vertices must satisfy |v| < 1.

    Returns
    -------
    volume : float
        Hyperbolic 4-volume.
    """
    if quad_lambdas is None:
        quad_lambdas = _QUAD_LAMBDAS
        quad_weights = _QUAD_WEIGHTS

    V = vertices  # (5, 4)

    # Map quadrature points to physical simplex
    # x = sum_i lambda_i * v_i  (barycentric)
    x = quad_lambdas @ V  # (N_quad, 4)

    # |x|^2
    x2 = np.sum(x**2, axis=1)

    # Check all points inside B^4
    if np.any(x2 >= 1.0):
        # Some quadrature points outside the ball -- bad
        return 0.0

    # Integrand: 1 / (1 - |x|^2)^{5/2}
    integrand = 1.0 / (1.0 - x2)**2.5

    # Jacobian: |det(M)| where M = [v1-v0, v2-v0, v3-v0, v4-v0]
    M = V[1:] - V[0]
    det_M = np.abs(np.linalg.det(M))

    # Volume = |det(M)| * sum(weights * integrand)
    volume = det_M * np.sum(quad_weights * integrand)

    return volume


# ============================================================================
# Polytope from normals + support numbers
# ============================================================================

def compute_polytope(normals, h):
    """
    Compute vertices of P = {x : u_i . x <= h_i}.

    Returns vertices (m, 4) array, or None if degenerate.
    Uses polar duality with Q = conv(u_i / h_i).
    Handles mixed signs of h by direct vertex enumeration for
    the subset with h_i > 0.
    """
    f = len(normals)

    # Build halfspace representation: u_i . x <= h_i
    # Direct vertex enumeration: each vertex is intersection of 4 hyperplanes
    # For moderate f, enumerate all C(f,4) subsets and check feasibility

    if f > 40:
        return _compute_polytope_hull(normals, h)

    vertices = []
    vertex_defs = []

    for combo in combinations(range(f), 4):
        U = normals[list(combo)]
        h_sub = h[list(combo)]

        try:
            v = np.linalg.solve(U, h_sub)
        except np.linalg.LinAlgError:
            continue

        # Check feasibility: u_i . v <= h_i for all i
        dots = normals @ v
        if np.all(dots <= h + 1e-10):
            vertices.append(v)
            vertex_defs.append(combo)

    if len(vertices) < 5:
        return None

    vertices = np.array(vertices)

    # Remove near-duplicates
    unique_mask = np.ones(len(vertices), dtype=bool)
    for i in range(len(vertices)):
        if not unique_mask[i]:
            continue
        for j in range(i+1, len(vertices)):
            if unique_mask[j] and np.linalg.norm(vertices[i] - vertices[j]) < 1e-10:
                unique_mask[j] = False
    vertices = vertices[unique_mask]

    return vertices


def _compute_polytope_hull(normals, h):
    """Compute vertices using ConvexHull for large f."""
    f = len(normals)

    # Need all h_i to have same sign for simple polar dual
    # For mixed signs, use scipy HalfspaceIntersection
    from scipy.spatial import HalfspaceIntersection

    # Halfspaces: u_i . x - h_i <= 0, i.e., [u_i, -h_i]
    halfspaces = np.column_stack([normals, -h])

    # Interior point: origin should work if all h_i > 0
    # Otherwise find a feasible point via LP
    feasible_point = np.zeros(4)
    dots = normals @ feasible_point
    if not np.all(dots < h - 1e-10):
        # Origin not feasible, try to find one
        from scipy.optimize import linprog
        # Maximize slack: min -s subject to u_i . x + s <= h_i
        c = np.zeros(5)
        c[4] = -1  # maximize s
        A_ub = np.column_stack([normals, np.ones(f)])
        b_ub = h
        res = linprog(c, A_ub=A_ub, b_ub=b_ub)
        if res.success and res.x[4] > 1e-10:
            feasible_point = res.x[:4]
        else:
            return None

    try:
        hs = HalfspaceIntersection(halfspaces, feasible_point)
    except Exception:
        return None

    # hs.intersections gives the vertices
    vertices = hs.intersections

    # Filter to unique vertices
    if len(vertices) < 5:
        return None

    return vertices


# ============================================================================
# Volume of a polytope in the Klein model
# ============================================================================

def klein_polytope_volume(vertices, fast=False):
    """
    Compute hyperbolic volume of a convex polytope in the Klein model.

    Triangulates from the centroid into 4-simplices (one per boundary
    3-simplex), then sums their volumes.
    """
    if vertices is None or len(vertices) < 5:
        return 0.0

    # Check all vertices inside B^4
    r = np.linalg.norm(vertices, axis=1)
    if np.any(r >= 1.0):
        return 0.0

    # Centroid as apex
    centroid = np.mean(vertices, axis=0)
    if np.linalg.norm(centroid) >= 1.0:
        return 0.0

    # Convex hull gives boundary 3-simplices
    try:
        hull = ConvexHull(vertices)
    except Exception:
        return 0.0

    ql = _QUAD_LAMBDAS_FAST if fast else _QUAD_LAMBDAS
    qw = _QUAD_WEIGHTS_FAST if fast else _QUAD_WEIGHTS

    total_vol = 0.0
    for simplex in hull.simplices:
        # 4-simplex: centroid + 4 boundary vertices
        tet_verts = np.vstack([centroid, vertices[simplex]])
        vol = klein_simplex_volume(tet_verts, ql, qw)
        total_vol += vol

    return total_vol


# ============================================================================
# Optimization
# ============================================================================

def neg_volume_objective(h, normals, r_max=0.999):
    """
    Negative volume + penalty for vertices outside B^4.
    """
    vertices = compute_polytope(normals, h)
    if vertices is None:
        return 1e10

    r = np.linalg.norm(vertices, axis=1)
    max_r = np.max(r)

    # Hard constraint: vertices must be inside B^4
    if max_r >= 1.0:
        # Barrier: penalize heavily
        excess = max_r - r_max
        return 1e6 * excess**2

    vol = klein_polytope_volume(vertices, fast=True)
    return -vol


def optimize_from_normals(normals, h0=None, verbose=True):
    """
    Optimize support numbers to maximize Klein model volume.
    """
    f = len(normals)

    if h0 is None:
        h0 = np.full(f, 0.1)  # Start small (vertices well inside B^4)

    # Check initial polytope
    verts0 = compute_polytope(normals, h0)
    if verts0 is None:
        if verbose:
            print("  Initial h0 gives degenerate polytope")
        return None

    r0 = np.linalg.norm(verts0, axis=1)
    if verbose:
        vol0 = klein_polytope_volume(verts0)
        print(f"  Initial: {len(verts0)} vertices, max|v|={r0.max():.4f}, "
              f"vol={vol0:.6f}")

    # Bounds: keep h_i in a reasonable range
    # Upper bound: don't let vertices escape B^4
    bounds = [(-2.0, 2.0)] * f

    # Phase 1: Nelder-Mead (derivative-free, handles non-smoothness)
    res = minimize(neg_volume_objective, h0, args=(normals,),
                   method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-8,
                            'fatol': 1e-10, 'adaptive': True})

    # Phase 2: Powell (derivative-free, good for refining)
    res = minimize(neg_volume_objective, res.x, args=(normals,),
                   method='Powell',
                   options={'maxiter': 5000, 'ftol': 1e-12})

    h_opt = res.x
    verts_opt = compute_polytope(normals, h_opt)

    if verts_opt is None:
        if verbose:
            print("  Optimization produced degenerate polytope")
        return None

    r_opt = np.linalg.norm(verts_opt, axis=1)
    vol_opt = klein_polytope_volume(verts_opt)

    if verbose:
        print(f"  Optimized: {len(verts_opt)} vertices, max|v|={r_opt.max():.6f}, "
              f"vol={vol_opt:.6f}")
        print(f"  h range: [{h_opt.min():.4f}, {h_opt.max():.4f}]")
        print(f"  Vertex radii: min={r_opt.min():.6f}, max={r_opt.max():.6f}")
        print(f"  Iterations: {res.nit}, Status: {res.message}")

    return {
        'volume': vol_opt,
        'h': h_opt,
        'vertices': verts_opt,
        'normals': normals,
        'n_facets': f,
        'n_vertices': len(verts_opt),
        'max_r': r_opt.max(),
    }


def optimize_worker(args):
    """Worker for parallel optimization."""
    n_facets, seed = args
    rng = np.random.default_rng(seed)
    normals = rng.standard_normal((n_facets, 4))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    result = optimize_from_normals(normals, verbose=False)
    if result is not None:
        return (result['volume'], result['n_vertices'],
                result['max_r'], result['h'], result['normals'])
    return None


# ============================================================================
# Main
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python optimize_klein_volume.py <n_facets> [n_restarts]")
        sys.exit(1)

    n_facets = int(sys.argv[1])
    n_restarts = int(sys.argv[2]) if len(sys.argv) >= 3 else 20

    print(f"Klein model volume optimization")
    print(f"Facets: {n_facets}, Restarts: {n_restarts}")
    print(f"Quadrature: {len(_QUAD_WEIGHTS)} points per 4-simplex")
    print()

    # Quick validation: volume of a small regular simplex
    print("Validation: small regular 4-simplex at origin")
    eps = 0.1
    v0 = np.array([eps, 0, 0, 0])
    v1 = np.array([0, eps, 0, 0])
    v2 = np.array([0, 0, eps, 0])
    v3 = np.array([0, 0, 0, eps])
    v4 = np.array([0, 0, 0, 0])
    verts_test = np.array([v0, v1, v2, v3, v4])
    vol_klein = klein_simplex_volume(verts_test)
    # Euclidean volume of this simplex
    M_test = verts_test[1:] - verts_test[0]
    vol_eucl = np.abs(np.linalg.det(M_test)) / 24
    print(f"  Klein vol: {vol_klein:.8f}")
    print(f"  Euclidean vol: {vol_eucl:.8f}")
    print(f"  Ratio (should be ~1 for small simplex): {vol_klein/vol_eucl:.6f}")
    print()

    t0 = time.time()

    if n_restarts <= 5:
        # Sequential with verbose output
        results = []
        for i in range(n_restarts):
            rng = np.random.default_rng(77777 + n_facets * 1000 + i)
            normals = rng.standard_normal((n_facets, 4))
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)

            print(f"Trial {i+1}/{n_restarts}:")
            result = optimize_from_normals(normals, verbose=True)
            if result is not None:
                results.append(result)
            print()
    else:
        # Parallel
        work = [(n_facets, 77777 + n_facets * 1000 + i)
                for i in range(n_restarts)]
        with Pool(processes=32) as pool:
            raw = pool.map(optimize_worker, work)
        results = [r for r in raw if r is not None]

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")
    print(f"Valid: {len(results)}/{n_restarts}")

    if not results:
        print("No valid results!")
        return

    if n_restarts <= 5:
        vols = [r['volume'] for r in results]
    else:
        vols = [r[0] for r in results]

    vols = np.array(vols)
    idx = np.argsort(vols)[::-1]

    print(f"\nVolume statistics:")
    print(f"  Best:   {vols[idx[0]]:.10f}")
    print(f"  Top 5:  {', '.join(f'{vols[idx[i]]:.4f}' for i in range(min(5, len(idx))))}")
    print(f"  Mean:   {vols.mean():.6f}")

    if n_restarts > 5:
        best = results[idx[0]]
        print(f"\nBest result:")
        print(f"  Volume: {best[0]:.10f}")
        print(f"  Vertices: {best[1]}")
        print(f"  Max |v|: {best[2]:.8f}")


if __name__ == '__main__':
    main()
