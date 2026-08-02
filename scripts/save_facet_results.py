#!/usr/bin/env python3
"""
Run facet-parameterized optimization for multiple f values and save results.

For each f (number of facet normals):
  1. Scan random normals, solve inner ideal constraint
  2. Optimize top starting points via Powell on normals
  3. Save full results as JSON
"""

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial import ConvexHull, Delaunay
from itertools import combinations
from collections import Counter
from multiprocessing import Pool
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ideal_poly_volume_toolkit.geometry_4d import (
    ideal_4simplex_volume_from_tet,
    stereo_project_S3,
    regular_ideal_4simplex_volume,
)

# Import the two-level optimization machinery
from scripts.optimize_facet_full import (
    scan_worker, optimize_worker, solve_inner, compute_volume_from_verts,
    angles_to_S3, normals_to_angles, analyze_best,
)


def full_analysis(angles, f):
    """Return a dict with full analysis of a configuration."""
    normals = angles_to_S3(angles, f)
    h_opt, verts, Am = solve_inner(normals)

    if h_opt is None:
        return None

    r = np.linalg.norm(verts, axis=1)
    vol = compute_volume_from_verts(verts)

    result = {
        'volume': float(vol),
        'n_facet_normals': f,
        'n_vertices': len(verts),
        'vertex_norms_min': float(r.min()),
        'vertex_norms_max': float(r.max()),
        'max_ideal_deviation': float(np.max(np.abs(r - 1))),
        'support_numbers': h_opt.tolist(),
        'normals': normals.tolist(),
        'vertices': verts.tolist(),
        'normal_angles': angles.tolist(),
    }

    # f-vector
    try:
        hull = ConvexHull(verts, qhull_options='QJ')
        edges = set()
        triangles = set()
        for simplex in hull.simplices:
            for i, j in combinations(sorted(simplex), 2):
                edges.add((i, j))
            for tri in combinations(sorted(simplex), 3):
                triangles.add(tri)
        f0 = len(verts)
        f1 = len(edges)
        f2 = len(triangles)
        f3 = len(hull.simplices)
        result['f_vector'] = [f0, f1, f2, f3]
        result['euler_characteristic'] = f0 - f1 + f2 - f3
    except Exception:
        pass

    # Degree sequence
    if 'f_vector' in result:
        deg = Counter()
        for i, j in edges:
            deg[i] += 1
            deg[j] += 1
        result['degree_sequence'] = sorted([deg[i] for i in range(f0)], reverse=True)

    # Delaunay decomposition
    pts_3d, _ = stereo_project_S3(verts, pole_idx=0)
    tri = Delaunay(pts_3d)
    tet_vols = []
    for t in tri.simplices:
        try:
            tv = ideal_4simplex_volume_from_tet(pts_3d[t])
            if tv > 0:
                tet_vols.append(float(tv))
        except Exception:
            pass
    tet_vols_arr = np.array(tet_vols)
    v_reg = regular_ideal_4simplex_volume()
    result['delaunay_simplices'] = len(tet_vols)
    result['tet_vol_min'] = float(tet_vols_arr.min())
    result['tet_vol_max'] = float(tet_vols_arr.max())
    result['tet_vol_mean'] = float(tet_vols_arr.mean())
    result['max_tet_over_regular'] = float(tet_vols_arr.max() / v_reg)

    return result


def run_and_save(f, n_scan=100, n_optimize=10, outdir='results/data'):
    """Run full optimization for one f value and save results."""
    print(f"\n{'='*70}")
    print(f"f = {f} facet normals")
    print(f"{'='*70}")

    # Phase 1: Scan
    print(f"Phase 1: Scanning {n_scan} random normals...")
    t0 = time.time()
    work = [(f, 42000 + f * 1000 + i) for i in range(n_scan)]
    with Pool(processes=32) as pool:
        scan_results = pool.map(scan_worker, work)

    valid_scans = [r for r in scan_results if r is not None]
    valid_scans.sort(key=lambda r: r[0], reverse=True)
    scan_time = time.time() - t0

    print(f"  {len(valid_scans)}/{n_scan} valid, best scan vol = {valid_scans[0][0]:.4f} ({scan_time:.1f}s)")

    if not valid_scans:
        print("  No valid configurations!")
        return None

    # Phase 2: Optimize
    n_opt = min(n_optimize, len(valid_scans))
    print(f"Phase 2: Optimizing top {n_opt}...")
    t1 = time.time()

    opt_work = [(f, valid_scans[i][1], i) for i in range(n_opt)]
    with Pool(processes=min(32, n_opt)) as pool:
        opt_results = pool.map(optimize_worker, opt_work)

    opt_results = [r for r in opt_results if r[0] > 0]
    opt_results.sort(key=lambda r: r[0], reverse=True)
    opt_time = time.time() - t1

    if not opt_results:
        print("  Optimization failed, using best scan result")
        best_angles = valid_scans[0][1]
    else:
        best_angles = opt_results[0][1]
        print(f"  Best optimized vol = {opt_results[0][0]:.4f} ({opt_time:.1f}s)")

    # Phase 3: Full analysis and save
    print("Phase 3: Analyzing and saving...")
    best = full_analysis(best_angles, f)
    if best is None:
        print("  Analysis failed!")
        return None

    # Add metadata
    scan_vols = [r[0] for r in valid_scans]
    opt_vols = [r[0] for r in opt_results] if opt_results else scan_vols
    best['metadata'] = {
        'n_scan': n_scan,
        'n_optimize': n_optimize,
        'scan_success_rate': len(valid_scans) / n_scan,
        'scan_best_volume': float(scan_vols[0]),
        'scan_mean_volume': float(np.mean(scan_vols)),
        'scan_top5': [float(v) for v in scan_vols[:5]],
        'opt_top5': [float(v) for v in opt_vols[:5]],
        'improvement_pct': float(100 * (opt_vols[0] / scan_vols[0] - 1)) if opt_results else 0,
        'scan_time_s': scan_time,
        'opt_time_s': opt_time,
        'total_time_s': time.time() - t0,
        'regular_ideal_4simplex_volume': float(regular_ideal_4simplex_volume()),
    }

    # Save
    outfile = os.path.join(outdir, f'facet_optimization_f{f:02d}.json')
    with open(outfile, 'w') as fh:
        json.dump(best, fh, indent=2)
    print(f"  Saved to {outfile}")
    print(f"  Volume: {best['volume']:.10f}, Vertices: {best['n_vertices']}")

    return best


def main():
    facet_counts = [8, 10, 12, 16, 20, 24]
    n_scan = 100
    n_optimize = 10

    if len(sys.argv) > 1:
        facet_counts = [int(x) for x in sys.argv[1:]]

    outdir = 'results/data'
    os.makedirs(outdir, exist_ok=True)

    v_reg = regular_ideal_4simplex_volume()
    print(f"Facet-parameterized ideal polytope optimization")
    print(f"Regular ideal 4-simplex volume: {v_reg:.8f}")
    print(f"Facet counts: {facet_counts}")
    print(f"Scan: {n_scan}, Optimize: {n_optimize}")

    summary = {}
    for f in facet_counts:
        result = run_and_save(f, n_scan=n_scan, n_optimize=n_optimize, outdir=outdir)
        if result:
            summary[f] = {
                'volume': result['volume'],
                'n_vertices': result['n_vertices'],
                'f_vector': result.get('f_vector'),
                'scan_best': result['metadata']['scan_best_volume'],
                'improvement_pct': result['metadata']['improvement_pct'],
            }

    # Save summary
    print(f"\n\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"{'f':>4s}  {'Volume':>12s}  {'Vertices':>8s}  {'Scan best':>10s}  {'Improv':>8s}  {'Vol/f^2':>8s}")
    for f in sorted(summary.keys()):
        s = summary[f]
        print(f"{f:4d}  {s['volume']:12.6f}  {s['n_vertices']:8d}  "
              f"{s['scan_best']:10.4f}  {s['improvement_pct']:7.1f}%  "
              f"{s['volume']/f**2:8.4f}")

    summary_file = os.path.join(outdir, 'facet_optimization_summary.json')
    with open(summary_file, 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nSummary saved to {summary_file}")


if __name__ == '__main__':
    main()
