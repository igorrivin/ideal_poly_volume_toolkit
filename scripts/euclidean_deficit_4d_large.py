#!/usr/bin/env python3
"""
Large-N run for d=4 Euclidean volume deficit.

Focus: (1) pin down the mean constant in E[deficit] ~ C * N^{-2/3}
       (2) determine the variance exponent: is it really 7/3 (RVW)?

For the variance exponent, we compute log-log slopes between consecutive
N values to see local convergence.
"""

import numpy as np
from scipy.spatial import ConvexHull
from multiprocessing import Pool, cpu_count
import time

VOL_B4 = np.pi**2 / 2.0
N_JOBS = 64
MEAN_EXP = 2.0 / 3.0


def worker(args):
    n_vertices, chunk_size, seed = args
    rng = np.random.default_rng(seed)
    deficits = np.empty(chunk_size, dtype=np.float64)
    for i in range(chunk_size):
        pts = rng.standard_normal((n_vertices, 4))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        hull = ConvexHull(pts)
        deficits[i] = VOL_B4 - hull.volume
    return deficits


def run(n_vertices, n_samples):
    n_jobs = min(N_JOBS, cpu_count())
    per_worker = n_samples // n_jobs
    remainder = n_samples % n_jobs
    work = [(n_vertices, per_worker + (1 if i < remainder else 0),
             77777 + n_vertices * 137 + i) for i in range(n_jobs)]
    with Pool(processes=n_jobs) as pool:
        results = pool.map(worker, work)
    return np.concatenate(results)


def main():
    # Dense grid at large N to resolve the exponent
    cases = [
        (100,    500_000),
        (200,    500_000),
        (500,    500_000),
        (1000,   500_000),
        (2000,   200_000),
        (5000,   100_000),
        (10000,   50_000),
        (20000,   20_000),
        (50000,   10_000),
        (100000,   5_000),
    ]

    print(f"d=4 volume deficit: vol(B^4) = {VOL_B4:.10f}")
    print(f"RVW prediction: Var ~ N^(-7/3), mean ~ N^(-2/3)")
    print()

    header = (f"{'N':>7} {'mean_def':>12} {'N^(2/3)*def':>12} "
              f"{'var':>14} {'N^(7/3)*var':>14} {'samples':>8} {'time':>7}")
    print(header)
    print("-" * len(header))

    all_data = []

    for nv, ns in cases:
        t0 = time.time()
        deficits = run(nv, ns)
        dt = time.time() - t0

        mu = np.mean(deficits)
        var = np.var(deficits)
        sd = np.std(deficits)

        mean_resc = nv**MEAN_EXP * mu
        var_resc_73 = nv**(7.0/3.0) * var

        all_data.append({
            'N': nv, 'mean': mu, 'var': var, 'sd': sd,
            'mean_resc': mean_resc, 'var_resc': var_resc_73,
            'n_samples': ns, 'time': dt,
        })

        print(f"{nv:7d} {mu:12.8f} {mean_resc:12.6f} "
              f"{var:14.4e} {var_resc_73:14.6f} {ns:8d} {dt:6.1f}s")

    # Log-log slope analysis for variance exponent
    print(f"\n--- Variance exponent: local log-log slopes ---")
    print(f"{'N1':>7} {'N2':>7} {'slope':>10}  (RVW predicts 7/3 ≈ 2.333)")
    for i in range(1, len(all_data)):
        d1, d2 = all_data[i-1], all_data[i]
        if d1['var'] > 0 and d2['var'] > 0:
            slope = -(np.log(d2['var']) - np.log(d1['var'])) / \
                     (np.log(d2['N']) - np.log(d1['N']))
            print(f"{d1['N']:7d} {d2['N']:7d} {slope:10.4f}")

    # Mean constant candidates
    print(f"\n--- Mean constant N^(2/3)*deficit ---")
    for d in all_data[-5:]:
        print(f"  N={d['N']:6d}: {d['mean_resc']:.6f}")
    print(f"  15*pi = {15*np.pi:.6f}")
    print(f"  48    = 48.000000")

    # Try different variance rescalings
    print(f"\n--- Variance rescalings ---")
    print(f"{'N':>7} {'N^(7/3)*var':>14} {'N^2*var':>14} {'N^(5/2)*var':>14}")
    for d in all_data:
        n = d['N']
        v = d['var']
        print(f"{n:7d} {n**(7/3)*v:14.6f} {n**2*v:14.6f} {n**2.5*v:14.6f}")


if __name__ == '__main__':
    main()
