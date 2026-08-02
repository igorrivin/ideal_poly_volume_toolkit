#!/usr/bin/env python3
"""
Estimate the constant in the volume deficit scaling for random inscribed polytopes.

For N iid uniform points on S^2 ⊂ R^3, the convex hull P_N satisfies:
    vol(B^3) - vol(P_N) ~ C / N

We estimate C = lim N * (vol(B^3) - E[vol(P_N)]) and its variance,
for a range of N values, to see convergence and test the CLT.
"""

import numpy as np
from scipy.spatial import ConvexHull
from multiprocessing import Pool, cpu_count
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VOL_BALL = 4.0 * np.pi / 3.0  # vol(B^3)
N_JOBS = 64


def sample_sphere(n, rng):
    """Sample n uniform points on S^2 via normalized Gaussians."""
    pts = rng.standard_normal((n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def worker(args):
    """Compute volume deficits for a chunk of samples at a given N."""
    n_vertices, chunk_size, seed = args
    rng = np.random.default_rng(seed)
    deficits = np.empty(chunk_size, dtype=np.float64)
    for i in range(chunk_size):
        pts = sample_sphere(n_vertices, rng)
        hull = ConvexHull(pts)
        deficits[i] = VOL_BALL - hull.volume
    return deficits


def run_one_N(n_vertices, n_samples):
    """Run n_samples trials for a given N, return rescaled deficit statistics."""
    n_jobs = min(N_JOBS, cpu_count())
    per_worker = n_samples // n_jobs
    remainder = n_samples % n_jobs

    work = []
    for i in range(n_jobs):
        chunk = per_worker + (1 if i < remainder else 0)
        work.append((n_vertices, chunk, 77777 + n_vertices * 1000 + i))

    with Pool(processes=n_jobs) as pool:
        results = pool.map(worker, work)

    deficits = np.concatenate(results)
    rescaled = n_vertices * deficits  # N * (vol(B^3) - vol(P_N))

    return {
        'N': n_vertices,
        'deficit_mean': float(np.mean(deficits)),
        'deficit_std': float(np.std(deficits)),
        'rescaled_mean': float(np.mean(rescaled)),
        'rescaled_std': float(np.std(rescaled)),
        'rescaled_skew': float(((rescaled - np.mean(rescaled))**3).mean() / np.std(rescaled)**3),
        'rescaled_kurtosis': float(((rescaled - np.mean(rescaled))**4).mean() / np.std(rescaled)**4 - 3),
    }


def main():
    # Range of N values: dense at small N, sparser at large N
    N_values = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750,
                1000, 1500, 2000, 3000, 5000, 7500, 10000]
    n_samples = 500_000  # plenty for good statistics

    print(f"Euclidean volume deficit: N * (vol(B^3) - vol(P_N))")
    print(f"vol(B^3) = {VOL_BALL:.10f}")
    print(f"{n_samples} samples per N, {N_JOBS} workers")
    print()

    header = (f"{'N':>7} {'deficit_mean':>14} {'N*deficit':>14} "
              f"{'N*deficit_std':>14} {'skew':>8} {'kurt':>8}")
    print(header)
    print("-" * len(header))

    all_results = []
    t0 = time.time()

    for nv in N_values:
        t1 = time.time()
        r = run_one_N(nv, n_samples)
        dt = time.time() - t1
        all_results.append(r)
        print(f"{r['N']:7d} {r['deficit_mean']:14.8f} {r['rescaled_mean']:14.6f} "
              f"{r['rescaled_std']:14.6f} {r['rescaled_skew']:8.4f} "
              f"{r['rescaled_kurtosis']:8.4f}  ({dt:.1f}s)")

    total = time.time() - t0
    print(f"\nTotal time: {total:.1f}s")

    # --- Plot ---
    Ns = [r['N'] for r in all_results]
    means = [r['rescaled_mean'] for r in all_results]
    stds = [r['rescaled_std'] for r in all_results]
    skews = [r['rescaled_skew'] for r in all_results]
    kurts = [r['rescaled_kurtosis'] for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(r'Volume Deficit: $N \cdot (\mathrm{vol}(B^3) - \mathrm{vol}(P_N))$ '
                 'for uniform points on $S^2$',
                 fontsize=14, fontweight='bold')

    # Rescaled mean (should converge to constant C)
    ax = axes[0, 0]
    ax.semilogx(Ns, means, 'o-', color='steelblue', linewidth=2, markersize=5)
    # Show apparent limit
    if len(means) >= 3:
        C_est = np.mean(means[-3:])
        ax.axhline(C_est, color='red', linestyle='--', alpha=0.7,
                    label=f'C ≈ {C_est:.4f}')
    ax.set_xlabel('N (number of points)')
    ax.set_ylabel(r'$\mu = E[N \cdot \Delta V]$')
    ax.set_title(r'Rescaled Mean Deficit $\to C$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rescaled std (should converge if CLT holds for the rescaled quantity)
    ax = axes[0, 1]
    ax.semilogx(Ns, stds, 'o-', color='darkorange', linewidth=2, markersize=5)
    if len(stds) >= 3:
        sigma_est = np.mean(stds[-3:])
        ax.axhline(sigma_est, color='red', linestyle='--', alpha=0.7,
                    label=f'σ ≈ {sigma_est:.4f}')
    ax.set_xlabel('N (number of points)')
    ax.set_ylabel(r'$\sigma$ of $N \cdot \Delta V$')
    ax.set_title('Rescaled Std Dev')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Skewness (should → 0 for CLT)
    ax = axes[1, 0]
    ax.semilogx(Ns, skews, 'o-', color='seagreen', linewidth=2, markersize=5)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('N (number of points)')
    ax.set_ylabel('Skewness')
    ax.set_title('Skewness (→ 0 for CLT)')
    ax.grid(True, alpha=0.3)

    # Excess kurtosis (should → 0 for CLT)
    ax = axes[1, 1]
    ax.semilogx(Ns, kurts, 'o-', color='purple', linewidth=2, markersize=5)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('N (number of points)')
    ax.set_ylabel('Excess Kurtosis')
    ax.set_title('Excess Kurtosis (→ 0 for CLT)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = 'results/plots/euclidean_deficit_scaling.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")

    # Print summary
    print(f"\nAsymptotic estimates (average of last 3 data points):")
    print(f"  C = E[N * deficit] → {np.mean(means[-3:]):.6f}")
    print(f"  σ(N * deficit)    → {np.mean(stds[-3:]):.6f}")


if __name__ == '__main__':
    main()
