#!/usr/bin/env python3
"""
Sweep n_vertices from 10 to 100 (step 5) and compute:
  (mean, alpha, beta, KS_statistic)
for the volume distribution of random ideal polyhedra.

Produces a table and a 4-panel plot.
"""

import numpy as np
from scipy import stats
from multiprocessing import Pool, cpu_count
import time
import sys
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay

N_SAMPLES = 200_000
N_JOBS = 64
SERIES_TERMS = 96
VERTEX_COUNTS = list(range(10, 101, 5))  # 10, 15, 20, ..., 100


def generate_one_sample(rng, n_vertices):
    n_random = n_vertices - 3
    fixed = [complex(0, 0), complex(1, 0)]

    vecs = rng.standard_normal((n_random, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    x, y, z = vecs[:, 0], vecs[:, 1], vecs[:, 2]
    w = (x + 1j * y) / (1.0 - z)

    vertices = np.array(fixed + list(w), dtype=np.complex128)
    return ideal_poly_volume_via_delaunay(
        vertices, mode='fast', series_terms=SERIES_TERMS
    )


def worker(args):
    chunk_size, seed, n_vertices = args
    rng = np.random.default_rng(seed)
    volumes = np.empty(chunk_size, dtype=np.float64)
    for i in range(chunk_size):
        volumes[i] = generate_one_sample(rng, n_vertices)
    return volumes


def run_one_vertex_count(n_vertices):
    n_jobs = min(N_JOBS, cpu_count())
    samples_per_worker = N_SAMPLES // n_jobs
    remainder = N_SAMPLES % n_jobs

    work = []
    base_seed = 12345 + n_vertices * 1000
    for i in range(n_jobs):
        chunk = samples_per_worker + (1 if i < remainder else 0)
        work.append((chunk, base_seed + i, n_vertices))

    t0 = time.time()
    with Pool(processes=n_jobs) as pool:
        results = pool.map(worker, work)

    volumes = np.concatenate(results)
    elapsed = time.time() - t0

    mean_vol = float(np.mean(volumes))
    std_vol = float(np.std(volumes))

    a, b, loc, scale = stats.beta.fit(volumes, floc=0)
    ks_stat, ks_pval = stats.kstest(volumes, 'beta', args=(a, b, loc, scale))

    return {
        'n_vertices': n_vertices,
        'mean': mean_vol,
        'std': std_vol,
        'min': float(np.min(volumes)),
        'max': float(np.max(volumes)),
        'alpha': float(a),
        'beta': float(b),
        'scale': float(scale),
        'ks_stat': float(ks_stat),
        'ks_pval': float(ks_pval),
        'elapsed': elapsed,
    }


def main():
    print(f"Sweeping n_vertices = {VERTEX_COUNTS[0]} to {VERTEX_COUNTS[-1]}, "
          f"step {VERTEX_COUNTS[1]-VERTEX_COUNTS[0]}")
    print(f"{N_SAMPLES} samples per vertex count, {N_JOBS} workers")
    print()

    all_results = []
    header = f"{'n':>5} {'mean':>12} {'alpha':>12} {'beta':>12} {'KS':>10} {'KS_p':>10} {'time':>7}"
    print(header)
    print("-" * len(header))

    for nv in VERTEX_COUNTS:
        r = run_one_vertex_count(nv)
        all_results.append(r)
        print(f"{r['n_vertices']:5d} {r['mean']:12.6f} {r['alpha']:12.6f} "
              f"{r['beta']:12.6f} {r['ks_stat']:10.6f} {r['ks_pval']:10.6f} "
              f"{r['elapsed']:6.1f}s")

    # Save raw results
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'results', 'data')
    os.makedirs(outdir, exist_ok=True)
    json_path = os.path.join(outdir, 'vertex_sweep_10_100.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nData saved to {json_path}")

    # --- Plot ---
    ns = [r['n_vertices'] for r in all_results]
    means = [r['mean'] for r in all_results]
    alphas = [r['alpha'] for r in all_results]
    betas = [r['beta'] for r in all_results]
    ks_stats = [r['ks_stat'] for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Ideal Polyhedra Volume Distribution vs. Number of Vertices',
                 fontsize=14, fontweight='bold')

    # GPT's claimed values for n=40
    gpt_n = 40
    gpt_mean, gpt_alpha, gpt_beta, gpt_ks = 49.509, 704.19, 87.18, 0.0014

    # Mean volume
    ax = axes[0, 0]
    ax.plot(ns, means, 'o-', color='steelblue', linewidth=2, markersize=5)
    ax.plot(gpt_n, gpt_mean, 's', color='red', markersize=10, zorder=5,
            label=f'GPT (n=40): {gpt_mean}')
    ax.set_xlabel('Number of vertices')
    ax.set_ylabel('Mean volume')
    ax.set_title('Mean Volume')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Alpha
    ax = axes[0, 1]
    ax.plot(ns, alphas, 'o-', color='darkorange', linewidth=2, markersize=5)
    ax.plot(gpt_n, gpt_alpha, 's', color='red', markersize=10, zorder=5,
            label=f'GPT (n=40): {gpt_alpha}')
    ax.set_xlabel('Number of vertices')
    ax.set_ylabel('Alpha (shape parameter)')
    ax.set_title('Beta Distribution: Alpha')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Beta
    ax = axes[1, 0]
    ax.plot(ns, betas, 'o-', color='seagreen', linewidth=2, markersize=5)
    ax.plot(gpt_n, gpt_beta, 's', color='red', markersize=10, zorder=5,
            label=f'GPT (n=40): {gpt_beta}')
    ax.set_xlabel('Number of vertices')
    ax.set_ylabel('Beta (shape parameter)')
    ax.set_title('Beta Distribution: Beta')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # KS statistic
    ax = axes[1, 1]
    ax.plot(ns, ks_stats, 'o-', color='purple', linewidth=2, markersize=5)
    ax.axhline(0.0015, color='gray', linestyle='--', alpha=0.7, label='KS = 0.0015 threshold')
    ax.plot(gpt_n, gpt_ks, 's', color='red', markersize=10, zorder=5,
            label=f'GPT (n=40): {gpt_ks}')
    ax.set_xlabel('Number of vertices')
    ax.set_ylabel('KS statistic')
    ax.set_title('Kolmogorov-Smirnov Goodness of Fit')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plotdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'results', 'plots')
    os.makedirs(plotdir, exist_ok=True)
    plot_path = os.path.join(plotdir, 'vertex_sweep_10_100.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")


if __name__ == '__main__':
    main()
