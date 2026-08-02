#!/usr/bin/env python3
"""
Euclidean volume deficit for random inscribed polytopes in d=4.

For N iid uniform points on S^3 ⊂ R^4, the convex hull P_N satisfies:
    E[vol(B^4) - vol(P_N)] ~ C * N^{-2/3}

(NOT 1/N as in d=3; the exponent 2/(d-1) = 2/3 comes from the curvature
of the sphere for d >= 4.)

We estimate C and the variance scaling.
"""

import numpy as np
from scipy.spatial import ConvexHull
from multiprocessing import Pool, cpu_count
import time

VOL_B4 = np.pi**2 / 2.0  # vol(B^4) = pi^2/2
N_JOBS = 64
DEFICIT_EXP = 2.0 / 3.0  # E[deficit] ~ C * N^{-2/3}


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
             88888 + n_vertices * 100 + i) for i in range(n_jobs)]
    with Pool(processes=n_jobs) as pool:
        results = pool.map(worker, work)
    return np.concatenate(results)


def main():
    cases = [
        (10,    500_000),
        (20,    500_000),
        (50,    500_000),
        (100,   500_000),
        (200,   500_000),
        (500,   200_000),
        (1000,  200_000),
        (2000,  100_000),
        (5000,   50_000),
        (10000,  20_000),
        (20000,  10_000),
        (50000,   5_000),
    ]

    print(f"d=4 Euclidean volume deficit: vol(B^4) = pi^2/2 = {VOL_B4:.10f}")
    print(f"Deficit scaling: N^(-2/3)")
    print()
    print(f"{'N':>7} {'N^(2/3)*def':>14} {'std*N^?':>14} "
          f"{'skew':>8} {'kurt':>8} {'time':>7}")
    print("-" * 65)

    all_data = []

    for nv, ns in cases:
        t0 = time.time()
        deficits = run(nv, ns)
        dt = time.time() - t0

        mu = np.mean(deficits)
        sd = np.std(deficits)
        mean_rescaled = nv**DEFICIT_EXP * mu

        # Try different variance exponents to see which converges
        # RVW: Var ~ N^{-(d+3)/(d-1)} = N^{-7/3} for d=4
        var_exp_rvw = 7.0 / 3.0
        std_rescaled = nv**(var_exp_rvw / 2) * sd

        rescaled_deficit = nv**DEFICIT_EXP * deficits
        skew = float(((rescaled_deficit - rescaled_deficit.mean())**3).mean()
                      / np.std(rescaled_deficit)**3)
        kurt = float(((rescaled_deficit - rescaled_deficit.mean())**4).mean()
                      / np.std(rescaled_deficit)**4 - 3)

        all_data.append((nv, mu, sd, mean_rescaled, std_rescaled, skew, kurt))

        print(f"{nv:7d} {mean_rescaled:14.6f} {std_rescaled:14.6f} "
              f"{skew:8.4f} {kurt:8.4f} {dt:6.1f}s")

    # Print variance exponent analysis
    print(f"\n--- Variance exponent analysis ---")
    print(f"If Var(deficit) ~ C * N^(-alpha), then log(Var)/log(N) -> -alpha")
    print(f"{'N':>7} {'log(var)/log(N)':>16} {'N^(7/6)*std':>14} {'N^(4/3)*std':>14}")
    for nv, mu, sd, _, _, _, _ in all_data:
        if nv >= 50:
            var_ratio = -np.log(sd**2) / np.log(nv)
            s1 = nv**(7.0/6.0) * sd
            s2 = nv**(4.0/3.0) * sd
            print(f"{nv:7d} {var_ratio:16.4f} {s1:14.6f} {s2:14.6f}")

    print(f"\nReference constants near {all_data[-1][3]:.2f}:")
    print(f"  15*pi           = {15*np.pi:.6f}")
    print(f"  48              = 48.000000")
    print(f"  16*pi           = {16*np.pi:.6f}")


if __name__ == '__main__':
    main()
