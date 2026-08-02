#!/usr/bin/env python3
"""
Find maximum-volume inscribed polytopes in S^{d-1} for d=3,4.

For each N, optimize vertex positions on the sphere to maximize
the convex hull volume. Compare optimal deficit to random deficit.

Uses scipy.optimize.minimize with L-BFGS-B, multiple random restarts.
"""

import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from multiprocessing import Pool
import time


def pts_from_angles_3d(angles):
    """Convert 2N angles to N points on S^2."""
    n = len(angles) // 2
    theta = angles[:n]
    phi = angles[n:]
    return np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])


def neg_vol_3d(angles):
    pts = pts_from_angles_3d(angles)
    try:
        return -ConvexHull(pts).volume
    except:
        return 0.0


def pts_from_angles_4d(angles):
    """Convert 3N angles to N points on S^3.
    Parameterize via: (sin(a)sin(b)cos(c), sin(a)sin(b)sin(c), sin(a)cos(b), cos(a))
    """
    n = len(angles) // 3
    a = angles[:n]
    b = angles[n:2*n]
    c = angles[2*n:]
    sa, ca = np.sin(a), np.cos(a)
    sb, cb = np.sin(b), np.cos(b)
    sc, cc = np.sin(c), np.cos(c)
    return np.column_stack([sa*sb*cc, sa*sb*sc, sa*cb, ca])


def neg_vol_4d(angles):
    pts = pts_from_angles_4d(angles)
    try:
        return -ConvexHull(pts).volume
    except:
        return 0.0


def optimize_worker_3d(args):
    n_vertices, seed = args
    rng = np.random.default_rng(seed)
    pts0 = rng.standard_normal((n_vertices, 3))
    pts0 /= np.linalg.norm(pts0, axis=1, keepdims=True)
    theta0 = np.arccos(np.clip(pts0[:, 2], -1, 1))
    phi0 = np.arctan2(pts0[:, 1], pts0[:, 0])
    x0 = np.concatenate([theta0, phi0])
    res = minimize(neg_vol_3d, x0, method='L-BFGS-B',
                   options={'maxiter': 2000, 'ftol': 1e-15})
    return -res.fun


def optimize_worker_4d(args):
    n_vertices, seed = args
    rng = np.random.default_rng(seed)
    pts0 = rng.standard_normal((n_vertices, 4))
    pts0 /= np.linalg.norm(pts0, axis=1, keepdims=True)
    # Convert to hyperspherical coordinates
    a0 = np.arccos(np.clip(pts0[:, 3], -1, 1))
    b0 = np.arccos(np.clip(pts0[:, 2] / (np.sin(a0) + 1e-30), -1, 1))
    c0 = np.arctan2(pts0[:, 1], pts0[:, 0])
    x0 = np.concatenate([a0, b0, c0])
    res = minimize(neg_vol_4d, x0, method='L-BFGS-B',
                   options={'maxiter': 2000, 'ftol': 1e-15})
    return -res.fun


def run_optimizations(dim, n_vertices, n_restarts):
    worker = optimize_worker_3d if dim == 3 else optimize_worker_4d
    work = [(n_vertices, 54321 + n_vertices * 100 + i) for i in range(n_restarts)]
    with Pool(processes=min(64, n_restarts)) as pool:
        volumes = pool.map(worker, work)
    return max(volumes)


def main():
    import os
    import sys

    outdir = os.path.join(os.path.dirname(__file__), '..', 'results', 'data')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, 'optimal_inscribed_polytopes.txt')

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    outfile = open(outpath, 'w')
    tee = Tee(sys.stdout, outfile)

    def tprint(*args, **kwargs):
        print(*args, file=tee, **kwargs)

    vol_b3 = 4 * np.pi / 3
    vol_b4 = np.pi**2 / 2

    # --- d = 3 ---
    tprint("=" * 70)
    tprint(f"d=3: Optimal inscribed polytopes (vol(B^3) = {vol_b3:.6f})")
    tprint(f"{'N':>5} {'opt_vol':>12} {'deficit':>12} {'N*deficit':>12} "
           f"{'E[N*def]':>10} {'ratio':>8} {'time':>7}")
    tprint("-" * 70)

    cases_3d = [4, 6, 8, 10, 12, 16, 20, 30, 40, 50, 60, 80, 100]
    n_restarts = 50

    for n in cases_3d:
        t0 = time.time()
        v = run_optimizations(3, n, n_restarts)
        dt = time.time() - t0
        deficit = vol_b3 - v

        # Exact expected deficit
        ratio_exact = 1.0
        for k in range(1, 4):
            ratio_exact *= (n - k) / (n + k)
        E_deficit = vol_b3 * (1 - ratio_exact)

        ratio = deficit / E_deficit if E_deficit > 0 else 0
        tprint(f"{n:5d} {v:12.6f} {deficit:12.6f} {n*deficit:12.4f} "
               f"{n*E_deficit:10.4f} {ratio:8.4f} {dt:6.1f}s")

    # --- d = 4 ---
    tprint()
    tprint("=" * 70)
    tprint(f"d=4: Optimal inscribed polytopes (vol(B^4) = {vol_b4:.6f})")
    tprint(f"{'N':>5} {'opt_vol':>12} {'deficit':>12} {'N^(2/3)*def':>12} {'time':>7}")
    tprint("-" * 70)

    cases_4d = [5, 8, 10, 16, 20, 30, 50, 80, 100]
    n_restarts = 50

    for n in cases_4d:
        t0 = time.time()
        v = run_optimizations(4, n, n_restarts)
        dt = time.time() - t0
        deficit = vol_b4 - v
        tprint(f"{n:5d} {v:12.6f} {deficit:12.6f} {n**(2/3)*deficit:12.4f} {dt:6.1f}s")

    tprint()
    tprint("Done.")
    outfile.close()


if __name__ == '__main__':
    main()
