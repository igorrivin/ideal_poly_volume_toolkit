#!/usr/bin/env python3
"""
Compute the volume distribution of random ideal polyhedra with 60 vertices.

Three vertices fixed at 0, 1, infinity; remaining 57 sampled uniformly on
the Riemann sphere (stereographic projection from north pole). Volumes
computed via Delaunay triangulation + Lobachevsky function.

Reports: (mean, alpha, beta, KS_statistic) to 6 decimal places.
"""

import numpy as np
from scipy import stats
from multiprocessing import Pool, cpu_count
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay

N_VERTICES = 60
N_RANDOM = N_VERTICES - 3  # 57 random vertices (0, 1, inf are fixed)
N_SAMPLES = 200_000
N_JOBS = 64
SERIES_TERMS = 96


def generate_one_sample(rng):
    """Generate one random 60-vertex ideal polyhedron volume.
    No rejection: every uniform sphere sample is used."""
    # Fixed finite vertices: 0 and 1 (infinity is implicit)
    fixed = [complex(0, 0), complex(1, 0)]

    # Sample N_RANDOM uniform points on the unit sphere via normalized Gaussians
    vecs = rng.standard_normal((N_RANDOM, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    # Stereographic projection from north pole: (x,y,z) -> x/(1-z) + i*y/(1-z)
    x, y, z = vecs[:, 0], vecs[:, 1], vecs[:, 2]
    w = (x + 1j * y) / (1.0 - z)

    vertices = np.array(fixed + list(w), dtype=np.complex128)

    vol = ideal_poly_volume_via_delaunay(
        vertices, mode='fast', series_terms=SERIES_TERMS
    )
    return vol


def worker(args):
    """Worker: generate a chunk of volume samples."""
    chunk_size, seed = args
    rng = np.random.default_rng(seed)
    volumes = np.empty(chunk_size, dtype=np.float64)
    for i in range(chunk_size):
        volumes[i] = generate_one_sample(rng)
    return volumes


def main():
    n_jobs = min(N_JOBS, cpu_count())
    samples_per_worker = N_SAMPLES // n_jobs
    remainder = N_SAMPLES % n_jobs

    work = []
    base_seed = 12345
    for i in range(n_jobs):
        chunk = samples_per_worker + (1 if i < remainder else 0)
        work.append((chunk, base_seed + i))

    print(f"Sampling {N_SAMPLES} random {N_VERTICES}-vertex ideal polyhedra")
    print(f"Using {n_jobs} parallel workers")
    t0 = time.time()

    with Pool(processes=n_jobs) as pool:
        results = pool.map(worker, work)

    volumes = np.concatenate(results)
    elapsed = time.time() - t0

    print(f"Completed: {len(volumes)} samples in {elapsed:.1f}s")
    print()

    # --- Statistics ---
    mean_vol = np.mean(volumes)
    print(f"Mean volume:   {mean_vol:.6f}")
    print(f"Std deviation: {np.std(volumes):.6f}")
    print(f"Min volume:    {np.min(volumes):.6f}")
    print(f"Max volume:    {np.max(volumes):.6f}")
    print()

    # --- Beta distribution fit ---
    # Volumes are positive; fix loc=0, let scale float.
    a, b, loc, scale = stats.beta.fit(volumes, floc=0)
    ks_stat, ks_pval = stats.kstest(volumes, 'beta', args=(a, b, loc, scale))

    print(f"Beta fit (floc=0):")
    print(f"  alpha (a): {a:.6f}")
    print(f"  beta  (b): {b:.6f}")
    print(f"  scale:     {scale:.6f}")
    print(f"  KS stat:   {ks_stat:.6f}")
    print(f"  KS p-val:  {ks_pval:.6f}")
    print()

    print("=" * 60)
    print("ANSWER (mean, alpha, beta, KS_statistic):")
    print(f"  ({mean_vol:.6f}, {a:.6f}, {b:.6f}, {ks_stat:.6f})")
    print("=" * 60)

    # If KS > 0.0015, try with scale pinned near observed max
    if ks_stat > 0.0015:
        print("\nKS > 0.0015; trying fscale = max * 1.005 ...")
        fscale = np.max(volumes) * 1.005
        a2, b2, _, scale2 = stats.beta.fit(volumes, floc=0, fscale=fscale)
        ks2, pv2 = stats.kstest(volumes, 'beta', args=(a2, b2, 0, scale2))
        print(f"  alpha: {a2:.6f}, beta: {b2:.6f}, scale: {scale2:.6f}")
        print(f"  KS: {ks2:.6f}, p-value: {pv2:.6f}")
        print(f"  ({mean_vol:.6f}, {a2:.6f}, {b2:.6f}, {ks2:.6f})")


if __name__ == '__main__':
    main()
