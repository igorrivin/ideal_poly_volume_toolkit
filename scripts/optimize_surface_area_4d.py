#!/usr/bin/env python3
"""
Optimize ideal 4-polytope surface area and isoperimetric ratio.

For N vertices on S^3, find configurations maximizing:
  (a) Total 3-surface area (sum of H^3-volumes of boundary facets)
  (b) Isoperimetric ratio Vol_4 / Area_3  (most efficient polytope)

Also records the volume-maximizer's surface area for comparison.
"""

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
from multiprocessing import Pool
import time
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ideal_poly_volume_toolkit.geometry_4d import (
    ideal_4simplex_volume_from_tet,
    ideal_tet_volume_3d,
    stereo_project_S3,
    ideal_polytope_surface_area_4d,
    ideal_polytope_volume_4d,
)


def angles_to_S3(angles, n):
    a = angles[:n]
    b = angles[n:2*n]
    c = angles[2*n:3*n]
    sa, ca = np.sin(a), np.cos(a)
    sb, cb = np.sin(b), np.cos(b)
    sc, cc = np.sin(c), np.cos(c)
    return np.column_stack([sa*sb*cc, sa*sb*sc, sa*cb, ca])


def compute_both(pts_4d):
    """Compute both 4-volume and 3-surface-area."""
    pts_4d = pts_4d / np.linalg.norm(pts_4d, axis=1, keepdims=True)

    # Volume via Delaunay
    pts_3d, _ = stereo_project_S3(pts_4d, pole_idx=0)
    try:
        tri = Delaunay(pts_3d)
        vol4 = sum(ideal_4simplex_volume_from_tet(pts_3d[idx])
                    for idx in tri.simplices)
    except Exception:
        vol4 = 0.0

    # Surface area via ConvexHull
    try:
        sa, _, _ = ideal_polytope_surface_area_4d(pts_4d)
    except Exception:
        sa = 0.0

    return vol4, sa


def neg_surface_area(angles):
    n = len(angles) // 3
    pts = angles_to_S3(angles, n)
    _, sa = compute_both(pts)
    return -sa


def neg_isoperimetric(angles):
    """Negative Vol/Area for maximizing isoperimetric ratio."""
    n = len(angles) // 3
    pts = angles_to_S3(angles, n)
    vol, sa = compute_both(pts)
    if sa < 1e-10:
        return 0.0
    return -(vol / sa)


def neg_volume(angles):
    n = len(angles) // 3
    pts = angles_to_S3(angles, n)
    pts_3d, _ = stereo_project_S3(pts, pole_idx=0)
    try:
        tri = Delaunay(pts_3d)
        return -sum(ideal_4simplex_volume_from_tet(pts_3d[idx])
                    for idx in tri.simplices)
    except Exception:
        return 0.0


def optimize_worker(args):
    n_vertices, seed, objective_fn = args
    rng = np.random.default_rng(seed)
    pts0 = rng.standard_normal((n_vertices, 4))
    pts0 /= np.linalg.norm(pts0, axis=1, keepdims=True)
    a0 = np.arccos(np.clip(pts0[:, 3], -1, 1))
    sin_a = np.sin(a0) + 1e-30
    b0 = np.arccos(np.clip(pts0[:, 2] / sin_a, -1, 1))
    c0 = np.arctan2(pts0[:, 1], pts0[:, 0])
    x0 = np.concatenate([a0, b0, c0])

    res = minimize(objective_fn, x0, method='L-BFGS-B',
                   options={'maxiter': 3000, 'ftol': 1e-14})
    res2 = minimize(objective_fn, res.x, method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-11})
    return -res2.fun, res2.x


# Wrapper functions for multiprocessing (can't pickle lambdas)
def _worker_sa(args):
    return optimize_worker((args[0], args[1], neg_surface_area))

def _worker_iso(args):
    return optimize_worker((args[0], args[1], neg_isoperimetric))

def _worker_vol(args):
    return optimize_worker((args[0], args[1], neg_volume))


def run_opt(n_vertices, worker_fn, n_restarts=100, n_workers=64):
    work = [(n_vertices, 54321 + n_vertices * 137 + i)
            for i in range(n_restarts)]
    with Pool(min(n_workers, n_restarts)) as pool:
        results = pool.map(worker_fn, work)
    vals = [r[0] for r in results]
    best_idx = np.argmax(vals)
    return vals[best_idx], results[best_idx][1], np.sort(vals)[-5:]


def analyze(pts_4d):
    """Get vol, area, n_facets, facet stats for a point configuration."""
    pts_4d = pts_4d / np.linalg.norm(pts_4d, axis=1, keepdims=True)
    vol, n_tets = ideal_polytope_volume_4d(pts_4d)
    sa, n_facets, fv = ideal_polytope_surface_area_4d(pts_4d)
    hull = ConvexHull(pts_4d)
    return {
        'vol': vol, 'area': sa, 'ratio': vol/sa if sa > 0 else 0,
        'n_tets': n_tets, 'n_facets': n_facets,
        'mean_facet': np.mean(fv), 'max_facet': max(fv), 'min_facet': min(fv),
    }


def main():
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'results', 'data')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, 'surface_area_optimization_4d.txt')
    jsonpath = os.path.join(outdir, 'surface_area_optimization_4d.json')

    f = open(outpath, 'w')

    def tprint(*args):
        msg = ' '.join(str(a) for a in args)
        print(msg); f.write(msg + '\n'); f.flush()

    vertex_counts = [5, 6, 7, 8, 9, 10, 12, 16, 20]
    n_restarts = 150
    all_results = []

    tprint("=" * 100)
    tprint("Ideal 4-polytope: surface area and isoperimetric optimization")
    tprint(f"v₃ (regular ideal tet in H³) = 1.01489466")
    tprint("=" * 100)

    for N in vertex_counts:
        tprint(f"\n{'='*100}")
        tprint(f"N = {N}")
        tprint(f"{'='*100}")

        result = {'n_vertices': N}

        # --- Max surface area ---
        t0 = time.time()
        best_sa, best_x_sa, top5_sa = run_opt(N, _worker_sa, n_restarts)
        dt_sa = time.time() - t0
        pts_sa = angles_to_S3(best_x_sa, N)
        info_sa = analyze(pts_sa)
        tprint(f"\n  MAX SURFACE AREA ({dt_sa:.1f}s):")
        tprint(f"    Area = {info_sa['area']:.8f} ({info_sa['n_facets']} facets)")
        tprint(f"    Vol  = {info_sa['vol']:.8f}, Vol/Area = {info_sa['ratio']:.8f}")
        tprint(f"    Facets: mean={info_sa['mean_facet']:.6f}, "
               f"max={info_sa['max_facet']:.6f}, min={info_sa['min_facet']:.6f}")
        tprint(f"    Top 5 area: {[f'{v:.6f}' for v in top5_sa]}")
        result['max_area'] = info_sa

        # --- Max isoperimetric ratio (Vol/Area) ---
        t0 = time.time()
        best_iso, best_x_iso, top5_iso = run_opt(N, _worker_iso, n_restarts)
        dt_iso = time.time() - t0
        pts_iso = angles_to_S3(best_x_iso, N)
        info_iso = analyze(pts_iso)
        tprint(f"\n  MAX VOL/AREA ({dt_iso:.1f}s):")
        tprint(f"    Vol/Area = {info_iso['ratio']:.8f}")
        tprint(f"    Vol  = {info_iso['vol']:.8f}, Area = {info_iso['area']:.8f}")
        tprint(f"    {info_iso['n_facets']} facets, {info_iso['n_tets']} Delaunay tets")
        tprint(f"    Facets: mean={info_iso['mean_facet']:.6f}, "
               f"max={info_iso['max_facet']:.6f}, min={info_iso['min_facet']:.6f}")
        tprint(f"    Top 5 ratio: {[f'{v:.6f}' for v in top5_iso]}")
        result['max_ratio'] = info_iso

        # --- Max volume (for comparison) ---
        t0 = time.time()
        best_vol, best_x_vol, top5_vol = run_opt(N, _worker_vol, n_restarts)
        dt_vol = time.time() - t0
        pts_vol = angles_to_S3(best_x_vol, N)
        info_vol = analyze(pts_vol)
        tprint(f"\n  MAX VOLUME ({dt_vol:.1f}s):")
        tprint(f"    Vol  = {info_vol['vol']:.8f}")
        tprint(f"    Area = {info_vol['area']:.8f}, Vol/Area = {info_vol['ratio']:.8f}")
        tprint(f"    {info_vol['n_facets']} facets, {info_vol['n_tets']} Delaunay tets")
        result['max_vol'] = info_vol

        all_results.append(result)

    # Summary table
    tprint(f"\n\n{'='*100}")
    tprint("SUMMARY")
    tprint(f"{'='*100}")
    tprint(f"{'N':>4} | {'--- Max Area ---':>30} | {'--- Max Vol/Area ---':>30} | {'--- Max Vol ---':>30}")
    tprint(f"{'':>4} | {'Area':>12} {'Vol':>10} {'V/A':>8} | {'V/A':>8} {'Vol':>10} {'Area':>10} | {'Vol':>10} {'Area':>10} {'V/A':>8}")
    tprint("-" * 100)
    for r in all_results:
        n = r['n_vertices']
        sa = r['max_area']
        iso = r['max_ratio']
        vol = r['max_vol']
        tprint(f"{n:4d} | {sa['area']:12.6f} {sa['vol']:10.6f} {sa['ratio']:8.5f} "
               f"| {iso['ratio']:8.5f} {iso['vol']:10.6f} {iso['area']:10.6f} "
               f"| {vol['vol']:10.6f} {vol['area']:10.6f} {vol['ratio']:8.5f}")

    with open(jsonpath, 'w') as jf:
        json.dump(all_results, jf, indent=2)
    tprint(f"\nSaved to {outpath} and {jsonpath}")
    f.close()


if __name__ == '__main__':
    main()
