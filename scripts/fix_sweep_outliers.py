#!/usr/bin/env python3
"""
Rerun n=55 and n=100 with method-of-moments initial guesses to fix scipy
beta fit convergence failures, then regenerate the plot.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from multiprocessing import Pool
import json
import sys
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay

N_SAMPLES = 200_000
N_JOBS = 64
SERIES_TERMS = 96


def generate_one_sample(rng, n_vertices):
    n_random = n_vertices - 3
    fixed = [complex(0, 0), complex(1, 0)]
    vecs = rng.standard_normal((n_random, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    x, y, z = vecs[:, 0], vecs[:, 1], vecs[:, 2]
    w = (x + 1j * y) / (1.0 - z)
    vertices = np.array(fixed + list(w), dtype=np.complex128)
    return ideal_poly_volume_via_delaunay(vertices, mode='fast', series_terms=SERIES_TERMS)


def worker(args):
    chunk_size, seed, n_vertices = args
    rng = np.random.default_rng(seed)
    volumes = np.empty(chunk_size, dtype=np.float64)
    for i in range(chunk_size):
        volumes[i] = generate_one_sample(rng, n_vertices)
    return volumes


def robust_beta_fit(volumes):
    """Fit beta distribution with method-of-moments fallback."""
    # First try scipy's default MLE
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, b, loc, scale = stats.beta.fit(volumes, floc=0)

    ks_stat, ks_pval = stats.kstest(volumes, 'beta', args=(a, b, loc, scale))

    if ks_stat < 0.01:
        return a, b, loc, scale, ks_stat, ks_pval

    # MLE failed; use method of moments with loc=0
    vmin, vmax = np.min(volumes), np.max(volumes)
    scale_mom = vmax * 1.02
    # Normalize to (0,1)
    y = volumes / scale_mom
    mu = np.mean(y)
    var = np.var(y)
    # Method of moments for beta(a,b) on (0,1):
    # mu = a/(a+b), var = ab/((a+b)^2(a+b+1))
    common = mu * (1 - mu) / var - 1
    a_mom = mu * common
    b_mom = (1 - mu) * common

    # Now refine by MLE with these initial guesses
    def neg_ll(params):
        a_, b_, s_ = params
        if a_ <= 0 or b_ <= 0 or s_ <= 0:
            return 1e20
        if np.any(volumes >= s_) or np.any(volumes <= 0):
            return 1e20
        try:
            return -np.sum(stats.beta.logpdf(volumes, a_, b_, loc=0, scale=s_))
        except:
            return 1e20

    from scipy.optimize import minimize
    res = minimize(neg_ll, [a_mom, b_mom, scale_mom], method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
    a, b, scale = res.x
    loc = 0.0
    ks_stat, ks_pval = stats.kstest(volumes, 'beta', args=(a, b, loc, scale))
    return a, b, loc, scale, ks_stat, ks_pval


def sample_and_fit(n_vertices):
    n_jobs = min(N_JOBS, 64)
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
    a, b, loc, scale, ks_stat, ks_pval = robust_beta_fit(volumes)

    print(f"  n={n_vertices}: mean={mean_vol:.6f} alpha={a:.6f} beta={b:.6f} "
          f"KS={ks_stat:.6f} p={ks_pval:.6f} ({elapsed:.1f}s)")

    return {
        'n_vertices': n_vertices,
        'mean': mean_vol,
        'std': float(np.std(volumes)),
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
    # Load existing results
    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'results', 'data', 'vertex_sweep_10_100.json')
    with open(json_path) as f:
        all_results = json.load(f)

    # Identify and rerun failed fits (KS > 0.01)
    rerun_indices = []
    for i, r in enumerate(all_results):
        if r['ks_stat'] > 0.01:
            rerun_indices.append(i)
            print(f"Rerunning n={r['n_vertices']} (KS was {r['ks_stat']:.6f})")

    for i in rerun_indices:
        nv = all_results[i]['n_vertices']
        all_results[i] = sample_and_fit(nv)

    # Save updated results
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nUpdated {json_path}")

    # --- Regenerate plot ---
    ns = [r['n_vertices'] for r in all_results]
    means = [r['mean'] for r in all_results]
    alphas = [r['alpha'] for r in all_results]
    betas = [r['beta'] for r in all_results]
    ks_stats = [r['ks_stat'] for r in all_results]

    gpt_n = 40
    gpt_mean, gpt_alpha, gpt_beta, gpt_ks = 49.509, 704.19, 87.18, 0.0014

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Ideal Polyhedra Volume Distribution vs. Number of Vertices\n'
                 '(200k samples per point, no vertex rejection)',
                 fontsize=14, fontweight='bold')

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
    plot_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'results', 'plots', 'vertex_sweep_10_100.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")


if __name__ == '__main__':
    main()
