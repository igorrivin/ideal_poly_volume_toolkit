#!/usr/bin/env python3
"""
Larger-scale search for maximum-volume inscribed polytopes.

More restarts, larger N, and a two-phase approach:
1. Many quick L-BFGS-B runs from random starts
2. Take the best and polish with tighter tolerances

For d=3 and d=4.
"""

import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from multiprocessing import Pool
import time


# ============================================================
# d = 3
# ============================================================

def neg_vol_3d(angles):
    n = len(angles) // 2
    theta = angles[:n]
    phi = angles[n:]
    pts = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    try:
        return -ConvexHull(pts).volume
    except:
        return 0.0


def optimize_3d_worker(args):
    n_vertices, seed = args
    rng = np.random.default_rng(seed)
    pts0 = rng.standard_normal((n_vertices, 3))
    pts0 /= np.linalg.norm(pts0, axis=1, keepdims=True)
    theta0 = np.arccos(np.clip(pts0[:, 2], -1, 1))
    phi0 = np.arctan2(pts0[:, 1], pts0[:, 0])
    x0 = np.concatenate([theta0, phi0])

    # Phase 1: quick optimization
    res = minimize(neg_vol_3d, x0, method='L-BFGS-B',
                   options={'maxiter': 5000, 'ftol': 1e-15})

    # Phase 2: polish
    res2 = minimize(neg_vol_3d, res.x, method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12})

    return -res2.fun, res2.x


# ============================================================
# d = 4
# ============================================================

def neg_vol_4d(angles):
    n = len(angles) // 3
    a = angles[:n]
    b = angles[n:2*n]
    c = angles[2*n:]
    sa, ca = np.sin(a), np.cos(a)
    sb, cb = np.sin(b), np.cos(b)
    sc, cc = np.sin(c), np.cos(c)
    pts = np.column_stack([sa*sb*cc, sa*sb*sc, sa*cb, ca])
    try:
        return -ConvexHull(pts).volume
    except:
        return 0.0


def optimize_4d_worker(args):
    n_vertices, seed = args
    rng = np.random.default_rng(seed)
    pts0 = rng.standard_normal((n_vertices, 4))
    pts0 /= np.linalg.norm(pts0, axis=1, keepdims=True)
    a0 = np.arccos(np.clip(pts0[:, 3], -1, 1))
    sin_a = np.sin(a0) + 1e-30
    b0 = np.arccos(np.clip(pts0[:, 2] / sin_a, -1, 1))
    c0 = np.arctan2(pts0[:, 1], pts0[:, 0])
    x0 = np.concatenate([a0, b0, c0])

    res = minimize(neg_vol_4d, x0, method='L-BFGS-B',
                   options={'maxiter': 5000, 'ftol': 1e-15})
    res2 = minimize(neg_vol_4d, res.x, method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12})

    return -res2.fun, res2.x


def run_opt(dim, n_vertices, n_restarts):
    worker = optimize_3d_worker if dim == 3 else optimize_4d_worker
    work = [(n_vertices, 12345 + n_vertices * 137 + i) for i in range(n_restarts)]
    with Pool(processes=min(64, n_restarts)) as pool:
        results = pool.map(worker, work)
    volumes = [r[0] for r in results]
    best_idx = np.argmax(volumes)
    return volumes[best_idx], results[best_idx][1], volumes


def main():
    import os
    import sys

    outdir = os.path.join(os.path.dirname(__file__), '..', 'results', 'data')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, 'optimal_inscribed_large.txt')

    # Tee output to both stdout and file
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
    tprint("=" * 80)
    tprint(f"d=3: Max-volume inscribed polytopes (vol(B^3) = {vol_b3:.6f})")
    tprint(f"  200 restarts per N, two-phase L-BFGS-B")
    tprint(f"{'N':>5} {'best_vol':>12} {'deficit':>12} {'N*def':>10} "
           f"{'E[N*def]':>10} {'opt/rnd':>8} {'spread':>8} {'time':>7}")
    tprint("-" * 80)

    cases_3d = [4, 6, 8, 10, 12, 16, 20, 30, 50, 80, 100, 150, 200, 300, 500]
    n_restarts_3d = 200

    results_3d = []
    for n in cases_3d:
        t0 = time.time()
        best_vol, best_x, all_vols = run_opt(3, n, n_restarts_3d)
        dt = time.time() - t0
        deficit = vol_b3 - best_vol

        ratio_exact = 1.0
        for k in range(1, 4):
            ratio_exact *= (n - k) / (n + k)
        E_deficit = vol_b3 * (1 - ratio_exact)
        opt_over_rnd = deficit / E_deficit

        spread = (max(all_vols) - np.median(all_vols)) / E_deficit

        results_3d.append((n, best_vol, deficit, opt_over_rnd))
        tprint(f"{n:5d} {best_vol:12.6f} {deficit:12.6f} {n*deficit:10.4f} "
               f"{n*E_deficit:10.4f} {opt_over_rnd:8.4f} {spread:8.5f} {dt:6.1f}s")

    # --- d = 4 ---
    tprint()
    tprint("=" * 80)
    tprint(f"d=4: Max-volume inscribed polytopes (vol(B^4) = {vol_b4:.6f})")
    tprint(f"  200 restarts per N, two-phase L-BFGS-B")
    tprint(f"{'N':>5} {'best_vol':>12} {'deficit':>12} {'N^(2/3)*def':>12} "
           f"{'rnd N^(2/3)*def':>15} {'opt/rnd':>8} {'time':>7}")
    tprint("-" * 80)

    cases_4d = [5, 8, 10, 16, 20, 30, 50, 80, 100, 150, 200]
    n_restarts_4d = 200

    # Approximate random deficit from our earlier run (N^{2/3} * E[deficit])
    rnd_4d = {100: 39.96, 200: 42.67, 500: 44.89, 1000: 45.89}

    for n in cases_4d:
        t0 = time.time()
        best_vol, best_x, all_vols = run_opt(4, n, n_restarts_4d)
        dt = time.time() - t0
        deficit = vol_b4 - best_vol
        rescaled = n**(2/3) * deficit

        # Interpolate random for ratio
        rnd_val = rnd_4d.get(n, None)
        if rnd_val:
            ratio = rescaled / rnd_val
            tprint(f"{n:5d} {best_vol:12.6f} {deficit:12.6f} {rescaled:12.4f} "
                   f"{rnd_val:15.4f} {ratio:8.4f} {dt:6.1f}s")
        else:
            tprint(f"{n:5d} {best_vol:12.6f} {deficit:12.6f} {rescaled:12.4f} "
                   f"{'---':>15} {'---':>8} {dt:6.1f}s")

    tprint()
    tprint("Done.")
    outfile.close()


if __name__ == '__main__':
    main()
