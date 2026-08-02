#!/usr/bin/env python3
"""Push to larger N to pin down the variance constant more precisely."""

import numpy as np
from scipy.spatial import ConvexHull
from multiprocessing import Pool, cpu_count
import time

VOL_BALL = 4.0 * np.pi / 3.0
N_JOBS = 64


def worker(args):
    n_vertices, chunk_size, seed = args
    rng = np.random.default_rng(seed)
    deficits = np.empty(chunk_size, dtype=np.float64)
    for i in range(chunk_size):
        pts = rng.standard_normal((n_vertices, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        hull = ConvexHull(pts)
        deficits[i] = VOL_BALL - hull.volume
    return deficits


def run(n_vertices, n_samples):
    n_jobs = min(N_JOBS, cpu_count())
    per_worker = n_samples // n_jobs
    remainder = n_samples % n_jobs
    work = [(n_vertices, per_worker + (1 if i < remainder else 0),
             99999 + n_vertices * 100 + i) for i in range(n_jobs)]
    with Pool(processes=n_jobs) as pool:
        results = pool.map(worker, work)
    return np.concatenate(results)


def main():
    # Larger N values, fewer samples (still enough for good stats)
    cases = [
        (10000, 200_000),
        (20000, 100_000),
        (30000, 100_000),
        (50000, 50_000),
    ]

    print(f"{'N':>7} {'E[N*dV]':>12} {'N^(3/2)*std':>14} {'skew':>8} {'kurt':>8} {'time':>7}")
    print("-" * 62)

    for nv, ns in cases:
        t0 = time.time()
        deficits = run(nv, ns)
        dt = time.time() - t0

        rescaled = nv * deficits
        mu = np.mean(rescaled)
        sigma_scaled = np.sqrt(nv) * np.std(rescaled)  # = N^(3/2) * std(deficit)
        skew = ((rescaled - mu)**3).mean() / np.std(rescaled)**3
        kurt = ((rescaled - mu)**4).mean() / np.std(rescaled)**4 - 3

        print(f"{nv:7d} {mu:12.6f} {sigma_scaled:14.6f} {skew:8.4f} {kurt:8.4f} {dt:6.1f}s")

    # Also check: what are some candidate constants near 42.6?
    print(f"\nReference constants:")
    print(f"  24*sqrt(pi)         = {24*np.sqrt(np.pi):.6f}")
    print(f"  16*pi*sqrt(2/3)     = {16*np.pi*np.sqrt(2/3):.6f}")
    print(f"  4*pi*sqrt(8*pi/3)   = {4*np.pi*np.sqrt(8*np.pi/3):.6f}")
    print(f"  8*sqrt(2*pi)        = {8*np.sqrt(2*np.pi):.6f}")
    print(f"  sqrt(576*pi)        = {np.sqrt(576*np.pi):.6f}")


if __name__ == '__main__':
    main()
