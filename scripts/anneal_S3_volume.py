#!/usr/bin/env python3
"""
Simulated annealing + greedy hill-climb for max-volume polytopes on S^3.

Uses tangent-space perturbations on the sphere, adaptive step size,
and an annealing phase to hop between combinatorial basins.

Can optimize Euclidean or hyperbolic volume.
"""

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from multiprocessing import Pool
import time
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ideal_poly_volume_toolkit.geometry_4d import (
    ideal_4simplex_volume_from_tet,
    stereo_project_S3,
)


# ============================================================================
# Volume computation
# ============================================================================

def euclidean_volume(pts):
    """Euclidean 4-volume of convex hull."""
    try:
        return ConvexHull(pts).volume
    except:
        return 0.0


def hyperbolic_volume(pts):
    """Hyperbolic 4-volume of ideal polytope with vertices on S^3."""
    pts_3d, _ = stereo_project_S3(pts, pole_idx=0)
    try:
        tri = Delaunay(pts_3d)
        return sum(ideal_4simplex_volume_from_tet(pts_3d[idx])
                   for idx in tri.simplices)
    except:
        return 0.0


def all_on_hull(pts):
    """Check that all points are vertices of the convex hull."""
    try:
        hull = ConvexHull(pts)
        hull_verts = set()
        for simplex in hull.simplices:
            hull_verts.update(simplex)
        return len(hull_verts) == len(pts)
    except:
        return False


# ============================================================================
# Sphere geometry
# ============================================================================

def random_S3(n, rng):
    """Sample n uniform points on S^3."""
    pts = rng.standard_normal((n, 4))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def tangent_perturb(pts, indices, sigma, rng):
    """
    Perturb selected vertices by tangent-space Gaussian on S^3.

    For each index i:
      1. Sample g ~ N(0, I) in R^4
      2. Project to tangent space: z = g - (g . v_i) v_i
      3. Move: v_i' = normalize(v_i + sigma * z / |z|)
    """
    new_pts = pts.copy()
    for i in indices:
        v = pts[i]
        g = rng.standard_normal(4)
        g -= np.dot(g, v) * v  # project to tangent space
        norm_g = np.linalg.norm(g)
        if norm_g < 1e-15:
            continue
        z = g / norm_g
        new_v = v + sigma * z
        new_pts[i] = new_v / np.linalg.norm(new_v)
    return new_pts


def coulomb_repulsion(pts, n_steps=200, dt=0.01):
    """
    Push points apart on S^3 by Coulomb-like repulsion.
    Good for generating well-spread initial configurations.
    """
    pts = pts.copy()
    n = len(pts)
    for step in range(n_steps):
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    diff = np.random.randn(4) * 0.01
                    dist = np.linalg.norm(diff)
                f = diff / (dist ** 3)
                forces[i] += f
                forces[j] -= f
        # Project forces to tangent space and step
        for i in range(n):
            f = forces[i]
            f -= np.dot(f, pts[i]) * pts[i]  # tangent projection
            pts[i] += dt * f
            pts[i] /= np.linalg.norm(pts[i])
    return pts


# ============================================================================
# Annealing optimizer
# ============================================================================

def anneal_optimize(N, volume_fn, seed=0,
                    T0=0.004, sigma0=0.18,
                    n_anneal=15000, n_greedy=40000,
                    init_method='random',
                    target_accept_rate=0.3,
                    cool_rate=0.9997,
                    sigma_adapt_interval=200,
                    verbose=False):
    """
    Simulated annealing + greedy polish for max volume on S^3.

    Parameters
    ----------
    N : int
        Number of vertices.
    volume_fn : callable
        pts -> volume (to maximize).
    seed : int
        Random seed.
    T0 : float
        Initial temperature.
    sigma0 : float
        Initial step size.
    n_anneal : int
        Number of annealing proposals.
    n_greedy : int
        Number of greedy proposals.
    init_method : str
        'random', 'coulomb', or 'random+coulomb'.
    target_accept_rate : float
        Target acceptance rate for sigma adaptation.
    cool_rate : float
        Geometric cooling factor per step.
    sigma_adapt_interval : int
        Adapt sigma every this many steps.
    verbose : bool

    Returns
    -------
    best_vol : float
    best_pts : (N, 4) array
    stats : dict
    """
    rng = np.random.default_rng(seed)

    # Initialize
    if init_method == 'coulomb':
        pts = coulomb_repulsion(random_S3(N, rng), n_steps=300, dt=0.01)
    elif init_method == 'random+coulomb':
        pts = random_S3(N, rng)
        pts = coulomb_repulsion(pts, n_steps=200, dt=0.01)
    else:
        pts = random_S3(N, rng)

    # Ensure all on hull
    if not all_on_hull(pts):
        pts = coulomb_repulsion(pts, n_steps=100, dt=0.01)

    current_vol = volume_fn(pts)
    best_vol = current_vol
    best_pts = pts.copy()

    sigma = sigma0
    T = T0
    accepts = 0
    total = 0
    improvements = 0
    basin_hops = 0

    # k-selection probabilities: mostly 1-2 vertices, occasionally more
    k_probs = np.array([0.0, 0.45, 0.30, 0.15, 0.05, 0.05])
    k_probs = k_probs / k_probs.sum()
    k_values = np.arange(len(k_probs))

    for phase, n_steps in [('anneal', n_anneal), ('greedy', n_greedy)]:
        phase_accepts = 0

        for step in range(n_steps):
            # Choose how many vertices to perturb
            k = rng.choice(k_values, p=k_probs)
            if k == 0:
                k = 1
            indices = rng.choice(N, size=k, replace=False)

            # Propose
            new_pts = tangent_perturb(pts, indices, sigma, rng)

            # Reject if any vertex falls off hull
            if not all_on_hull(new_pts):
                total += 1
                continue

            new_vol = volume_fn(new_pts)
            delta = new_vol - current_vol

            # Accept/reject
            accept = False
            if delta > 0:
                accept = True
                improvements += 1
            elif phase == 'anneal' and T > 1e-15:
                if rng.random() < np.exp(delta / T):
                    accept = True
                    basin_hops += 1

            if accept:
                pts = new_pts
                current_vol = new_vol
                accepts += 1
                phase_accepts += 1
                if current_vol > best_vol:
                    best_vol = current_vol
                    best_pts = pts.copy()

            total += 1

            # Cool temperature (annealing phase only)
            if phase == 'anneal':
                T *= cool_rate

            # Adapt step size
            if (step + 1) % sigma_adapt_interval == 0 and phase_accepts > 0:
                recent_rate = phase_accepts / min(step + 1, sigma_adapt_interval)
                if recent_rate > target_accept_rate * 1.2:
                    sigma *= 1.1  # too many accepts, explore more
                elif recent_rate < target_accept_rate * 0.8:
                    sigma *= 0.9  # too few accepts, be more conservative
                sigma = np.clip(sigma, 0.001, 1.0)

        if verbose:
            print(f"  {phase}: {phase_accepts}/{n_steps} accepted, "
                  f"best_vol={best_vol:.10f}, sigma={sigma:.4f}" +
                  (f", T={T:.2e}" if phase == 'anneal' else ""))

    stats = {
        'total_proposals': total,
        'total_accepts': accepts,
        'improvements': improvements,
        'basin_hops': basin_hops,
        'final_sigma': sigma,
        'final_T': T,
        'accept_rate': accepts / max(total, 1),
    }

    return best_vol, best_pts, stats


def worker(args):
    """Worker for parallel multi-start."""
    N, seed, volume_fn_name, kwargs = args

    if volume_fn_name == 'euclidean':
        vol_fn = euclidean_volume
    else:
        vol_fn = hyperbolic_volume

    # Vary initialization method
    init_methods = ['random', 'coulomb', 'random+coulomb']
    init_method = init_methods[seed % 3]

    # Slight variation in parameters
    rng = np.random.default_rng(seed + 1000000)
    T0 = kwargs.get('T0', 0.004) * (0.5 + rng.random())
    sigma0 = kwargs.get('sigma0', 0.18) * (0.5 + rng.random())

    best_vol, best_pts, stats = anneal_optimize(
        N, vol_fn, seed=seed,
        T0=T0, sigma0=sigma0,
        n_anneal=kwargs.get('n_anneal', 15000),
        n_greedy=kwargs.get('n_greedy', 40000),
        init_method=init_method,
    )

    return best_vol, best_pts, stats, seed


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Anneal S^3 volume optimizer')
    parser.add_argument('N', type=int, help='Number of vertices')
    parser.add_argument('--metric', choices=['euclidean', 'hyperbolic'], default='euclidean')
    parser.add_argument('--restarts', type=int, default=64)
    parser.add_argument('--n-anneal', type=int, default=15000)
    parser.add_argument('--n-greedy', type=int, default=40000)
    parser.add_argument('--T0', type=float, default=0.004)
    parser.add_argument('--sigma0', type=float, default=0.18)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    N = args.N
    n_restarts = args.restarts
    metric = args.metric

    print(f"Annealing optimizer: N={N}, metric={metric}, {n_restarts} restarts")
    print(f"  n_anneal={args.n_anneal}, n_greedy={args.n_greedy}")
    print(f"  T0={args.T0}, sigma0={args.sigma0}")
    sys.stdout.flush()

    kwargs = {
        'n_anneal': args.n_anneal,
        'n_greedy': args.n_greedy,
        'T0': args.T0,
        'sigma0': args.sigma0,
    }

    t0 = time.time()
    work = [(N, 77777 + i, metric, kwargs) for i in range(n_restarts)]

    with Pool(min(64, n_restarts)) as pool:
        results = pool.map(worker, work)

    # Find best
    best_idx = max(range(len(results)), key=lambda i: results[i][0])
    best_vol, best_pts, best_stats, best_seed = results[best_idx]

    # Also get top 5
    sorted_results = sorted(results, key=lambda r: r[0], reverse=True)
    top5 = [r[0] for r in sorted_results[:5]]

    dt = time.time() - t0

    # Compute complementary metric
    if metric == 'euclidean':
        other_vol = hyperbolic_volume(best_pts)
        other_name = 'hyperbolic'
    else:
        other_vol = euclidean_volume(best_pts)
        other_name = 'euclidean'

    hull = ConvexHull(best_pts)

    print(f"\nRESULT ({dt:.1f}s):")
    print(f"  Best {metric} volume: {best_vol:.12f}")
    print(f"  {other_name} volume:  {other_vol:.10f}")
    print(f"  Facets: {len(hull.simplices)}")
    print(f"  Seed: {best_seed}")
    print(f"  Accept rate: {best_stats['accept_rate']:.3f}")
    print(f"  Basin hops: {best_stats['basin_hops']}")
    print(f"  Top 5: {[f'{v:.8f}' for v in top5]}")

    # Save
    outpath = args.output or f"results/data/anneal_N{N}_{metric}.json"
    result = {
        'n_vertices': N,
        'metric': metric,
        f'{metric}_volume': float(best_vol),
        f'{other_name}_volume': float(other_vol),
        'n_facets': len(hull.simplices),
        'vertices': best_pts.tolist(),
        'stats': best_stats,
        'top5': top5,
        'n_restarts': n_restarts,
        'time_seconds': dt,
    }
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {outpath}")


if __name__ == '__main__':
    main()
