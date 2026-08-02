#!/usr/bin/env python3
"""
Statistical study of ideal 4-polytope volumes in H^4.

For each N, generate N random points on S^3, compute the total hyperbolic
4-volume of the ideal polytope (via Delaunay in R^3 after stereographic
projection), and fit the distribution.

Mirrors sweep_vertex_distribution.py (the 3D study) but for dimension 4.
"""

import numpy as np
from scipy import stats
from scipy.spatial import Delaunay
from multiprocessing import Pool, cpu_count
import time
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ideal_poly_volume_toolkit.geometry_4d import (
    ideal_4simplex_volume_from_tet,
    stereo_project_S3,
)

# --- Configuration ---
N_SAMPLES = 100_000
N_JOBS = 64
# Vertex counts: start small, go up to 100
VERTEX_COUNTS = [6, 8, 10, 15, 20, 30, 50, 75, 100]


def compute_volume_4d(pts_4d):
    """
    Compute ideal polytope volume for points on S^3.

    Stereographic project from vertex 0 to R^3, Delaunay triangulate,
    sum ideal 4-simplex volumes.
    """
    pts_3d, _ = stereo_project_S3(pts_4d, pole_idx=0)
    tri = Delaunay(pts_3d)
    total = 0.0
    for tet_idx in tri.simplices:
        total += ideal_4simplex_volume_from_tet(pts_3d[tet_idx])
    return total


def worker(args):
    chunk_size, seed, n_vertices = args
    rng = np.random.default_rng(seed)
    volumes = np.empty(chunk_size, dtype=np.float64)
    for i in range(chunk_size):
        pts = rng.standard_normal((n_vertices, 4))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        volumes[i] = compute_volume_4d(pts)
    return volumes


def run_one_vertex_count(n_vertices, tprint):
    n_jobs = min(N_JOBS, cpu_count())
    samples_per_worker = N_SAMPLES // n_jobs
    remainder = N_SAMPLES % n_jobs

    work = []
    base_seed = 77777 + n_vertices * 1000
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
    skew_vol = float(stats.skew(volumes))
    kurt_vol = float(stats.kurtosis(volumes))

    # Expected volume: for N uniform points on S^3, the ideal polytope volume.
    # In 3D, E[Vol] = vol(S^3)/2 * prod_{k=1}^{3} (N-k)/(N+k).
    # In 4D this is unknown; we measure it.

    # Fit beta distribution (support [0, scale])
    try:
        a, b, loc, scale = stats.beta.fit(volumes, floc=0)
        ks_stat, ks_pval = stats.kstest(volumes, 'beta', args=(a, b, loc, scale))
    except Exception as e:
        # Fallback: method of moments
        tprint(f"  [beta.fit failed for N={n_vertices}: {e}, using MoM]")
        v_min, v_max = volumes.min(), volumes.max()
        scale = v_max * 1.01
        loc = 0.0
        y = volumes / scale
        m, s2 = y.mean(), y.var()
        common = m * (1 - m) / s2 - 1
        a = float(m * common)
        b = float((1 - m) * common)
        ks_stat, ks_pval = stats.kstest(volumes, 'beta', args=(a, b, loc, scale))

    # Also try normal fit for comparison
    norm_ks, norm_pval = stats.kstest(volumes, 'norm', args=(mean_vol, std_vol))

    result = {
        'n_vertices': n_vertices,
        'n_samples': N_SAMPLES,
        'mean': mean_vol,
        'std': std_vol,
        'skew': skew_vol,
        'kurtosis': kurt_vol,
        'min': float(np.min(volumes)),
        'max': float(np.max(volumes)),
        'median': float(np.median(volumes)),
        'alpha': float(a),
        'beta_param': float(b),
        'scale': float(scale),
        'ks_stat': float(ks_stat),
        'ks_pval': float(ks_pval),
        'normal_ks': float(norm_ks),
        'normal_ks_pval': float(norm_pval),
        'elapsed': elapsed,
    }
    return result, volumes


def main():
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'results', 'data')
    os.makedirs(outdir, exist_ok=True)
    logdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'results', 'logs')
    os.makedirs(logdir, exist_ok=True)

    outpath_json = os.path.join(outdir, 'volume_sweep_4d.json')
    outpath_txt = os.path.join(outdir, 'volume_sweep_4d.txt')

    txtfile = open(outpath_txt, 'w')

    def tprint(*args, **kwargs):
        msg = ' '.join(str(a) for a in args)
        print(msg, **kwargs)
        txtfile.write(msg + '\n')
        txtfile.flush()

    tprint(f"4D Ideal Polytope Volume Distribution Study")
    tprint(f"N_SAMPLES = {N_SAMPLES}, N_JOBS = {N_JOBS}")
    tprint(f"Vertex counts: {VERTEX_COUNTS}")
    tprint()

    header = (f"{'N':>5} {'mean':>12} {'std':>10} {'skew':>8} {'kurt':>8} "
              f"{'alpha':>10} {'beta':>10} {'scale':>10} "
              f"{'KS_beta':>10} {'KS_norm':>10} {'time':>7}")
    tprint(header)
    tprint("-" * len(header))

    all_results = []

    for nv in VERTEX_COUNTS:
        r, volumes = run_one_vertex_count(nv, tprint)
        all_results.append(r)

        tprint(f"{r['n_vertices']:5d} {r['mean']:12.4f} {r['std']:10.4f} "
               f"{r['skew']:8.4f} {r['kurtosis']:8.4f} "
               f"{r['alpha']:10.4f} {r['beta_param']:10.4f} {r['scale']:10.4f} "
               f"{r['ks_stat']:10.6f} {r['normal_ks']:10.6f} "
               f"{r['elapsed']:6.1f}s")

        # Save percentiles for detailed analysis
        r['percentiles'] = {
            str(p): float(np.percentile(volumes, p))
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }

    # Save JSON
    with open(outpath_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    tprint(f"\nJSON saved to {outpath_json}")
    tprint(f"Table saved to {outpath_txt}")

    # Summary analysis
    tprint("\n" + "=" * 70)
    tprint("Summary: Does volume follow a beta distribution in 4D?")
    tprint("=" * 70)
    for r in all_results:
        verdict = "GOOD" if r['ks_stat'] < 0.01 else "MARGINAL" if r['ks_stat'] < 0.02 else "POOR"
        norm_verdict = "GOOD" if r['normal_ks'] < 0.01 else "MARGINAL" if r['normal_ks'] < 0.02 else "POOR"
        tprint(f"N={r['n_vertices']:3d}: Beta KS={r['ks_stat']:.6f} [{verdict:8s}]  "
               f"Normal KS={r['normal_ks']:.6f} [{norm_verdict:8s}]  "
               f"alpha={r['alpha']:.2f}, beta={r['beta_param']:.2f}")

    # Scaling analysis
    tprint("\n" + "=" * 70)
    tprint("Scaling analysis")
    tprint("=" * 70)
    tprint(f"{'N':>5} {'E[Vol]':>12} {'N*std':>10} {'N^2*var':>12}")
    tprint("-" * 45)
    for r in all_results:
        n = r['n_vertices']
        tprint(f"{n:5d} {r['mean']:12.4f} {n*r['std']:10.4f} "
               f"{n**2 * r['std']**2:12.4f}")

    tprint(f"\nDone.")
    txtfile.close()


if __name__ == '__main__':
    main()
