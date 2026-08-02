#!/usr/bin/env python3
"""
Optimize ideal 4-polytope volume in H^4.

For each N (number of vertices on S^3), find the configuration maximizing
the total hyperbolic 4-volume.  Analyze the optimizer:
  - 1-skeleton (edge graph of convex hull on S^3)
  - automorphism group via nauty
  - all dihedral angles, checking rationality (θ/π ∈ Q?)
"""

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
from multiprocessing import Pool
from fractions import Fraction
from itertools import combinations
from collections import Counter
import pynauty
import time
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ideal_poly_volume_toolkit.geometry_4d import (
    ideal_4simplex_volume_from_tet,
    tet_dihedral_angles,
    circumsphere,
    face_distances_from_circumcenter,
    stereo_project_S3,
    regular_ideal_4simplex_volume,
)


# ============================================================================
# Parameterization: hyperspherical coordinates on S^3
# ============================================================================

def angles_to_S3(angles, n):
    """Convert 3*n angles to n points on S^3."""
    a = angles[:n]       # [0, π]
    b = angles[n:2*n]    # [0, π]
    c = angles[2*n:3*n]  # [0, 2π]
    sa, ca = np.sin(a), np.cos(a)
    sb, cb = np.sin(b), np.cos(b)
    sc, cc = np.sin(c), np.cos(c)
    return np.column_stack([sa*sb*cc, sa*sb*sc, sa*cb, ca])


def neg_volume_4d(angles):
    """Negative total H^4 volume for optimization."""
    n = len(angles) // 3
    pts = angles_to_S3(angles, n)

    # Stereographic project from first point
    pts_3d, _ = stereo_project_S3(pts, pole_idx=0)

    try:
        tri = Delaunay(pts_3d)
    except Exception:
        return 0.0

    total = 0.0
    for tet_idx in tri.simplices:
        v = pts_3d[tet_idx]
        try:
            total += ideal_4simplex_volume_from_tet(v)
        except Exception:
            pass
    return -total


# ============================================================================
# Optimization
# ============================================================================

def optimize_worker(args):
    """Single optimization run from random initial point."""
    n_vertices, seed = args
    rng = np.random.default_rng(seed)

    # Random initial points on S^3
    pts0 = rng.standard_normal((n_vertices, 4))
    pts0 /= np.linalg.norm(pts0, axis=1, keepdims=True)

    # Convert to hyperspherical coordinates
    a0 = np.arccos(np.clip(pts0[:, 3], -1, 1))
    sin_a = np.sin(a0) + 1e-30
    b0 = np.arccos(np.clip(pts0[:, 2] / sin_a, -1, 1))
    c0 = np.arctan2(pts0[:, 1], pts0[:, 0])
    x0 = np.concatenate([a0, b0, c0])

    # Phase 1: quick
    res = minimize(neg_volume_4d, x0, method='L-BFGS-B',
                   options={'maxiter': 3000, 'ftol': 1e-14})

    # Phase 2: polish
    res2 = minimize(neg_volume_4d, res.x, method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-11})

    return -res2.fun, res2.x


def run_optimization(n_vertices, n_restarts=200, n_workers=64):
    """Run parallel optimization, return best result."""
    work = [(n_vertices, 31415 + n_vertices * 137 + i)
            for i in range(n_restarts)]

    with Pool(processes=min(n_workers, n_restarts)) as pool:
        results = pool.map(optimize_worker, work)

    volumes = [r[0] for r in results]
    best_idx = np.argmax(volumes)
    best_vol = volumes[best_idx]
    best_x = results[best_idx][1]

    # Also track spread
    vols_arr = np.array(volumes)
    top10 = np.sort(vols_arr)[-10:]

    return best_vol, best_x, top10


# ============================================================================
# Analysis of optimal configuration
# ============================================================================

def analyze_1skeleton(pts_4d):
    """
    Compute the 1-skeleton of the convex hull of points on S^3 ⊂ R^4.

    Returns edges, degree sequence, and the adjacency dict for nauty.
    """
    hull = ConvexHull(pts_4d)
    n = len(pts_4d)

    # Extract edges from simplices (boundary facets are tetrahedra in 4D)
    edges = set()
    for simplex in hull.simplices:
        for i, j in combinations(simplex, 2):
            edges.add((min(i, j), max(i, j)))

    edges = sorted(edges)
    degrees = Counter()
    adj = {i: set() for i in range(n)}
    for i, j in edges:
        degrees[i] += 1
        degrees[j] += 1
        adj[i].add(j)
        adj[j].add(i)

    degree_seq = sorted([degrees[i] for i in range(n)], reverse=True)
    return edges, degree_seq, adj


def compute_automorphism_group(adj, n):
    """Compute automorphism group using nauty."""
    g = pynauty.Graph(n)
    for v, nbrs in adj.items():
        g.connect_vertex(v, list(nbrs))

    generators, grpsize1, grpsize2, orbits, numorbits = pynauty.autgrp(g)
    grp_order = int(grpsize1 * 10**grpsize2)
    return grp_order, orbits, numorbits, generators


def collect_dihedral_angles(pts_4d):
    """
    Collect all dihedral angles of all ideal 4-simplices in the triangulation.

    Returns:
        all_angles: list of all angles (10 per tet)
        tet_angles: list of dicts, one per tet
    """
    pts_3d, _ = stereo_project_S3(pts_4d, pole_idx=0)
    tri = Delaunay(pts_3d)

    all_angles = []
    tet_data = []

    for tet_idx in tri.simplices:
        v = pts_3d[tet_idx]
        phi = tet_dihedral_angles(v)
        c, R = circumsphere(v)
        d = face_distances_from_circumcenter(v, c)

        # The 6 Euclidean dihedral angles (= hyperbolic angles θ_{∞,i,j})
        eucl_angles = list(phi.values())

        # The 4 "hemisphere" angles θ_{F_ℓ} = arccos(d_ℓ/R)
        hemi_angles = [np.arccos(np.clip(d[ell] / R, -1.0, 1.0)) for ell in range(4)]

        vol = ideal_4simplex_volume_from_tet(v)

        all_angles.extend(eucl_angles)
        all_angles.extend(hemi_angles)

        tet_data.append({
            'eucl_dihedral': eucl_angles,
            'hemi_angles': hemi_angles,
            'volume': vol,
        })

    return all_angles, tet_data


def check_rationality(angle, max_denom=120):
    """
    Check if angle/π is close to a rational number p/q with q ≤ max_denom.

    Returns (p, q, error) if close, else None.
    """
    ratio = angle / np.pi
    frac = Fraction(ratio).limit_denominator(max_denom)
    p, q = frac.numerator, frac.denominator
    error = abs(ratio - p / q)
    if error < 1e-6:
        return (p, q, error)
    return None


def analyze_angle_spectrum(all_angles, max_denom=120):
    """
    Analyze the spectrum of dihedral angles for rationality.

    Groups angles into clusters and checks each cluster.
    """
    angles = np.array(sorted(all_angles))

    # Cluster angles that are within 1e-6 of each other
    clusters = []
    current = [angles[0]]
    for a in angles[1:]:
        if a - current[-1] < 1e-6:
            current.append(a)
        else:
            clusters.append(current)
            current = [a]
    clusters.append(current)

    results = []
    for cluster in clusters:
        mean_angle = np.mean(cluster)
        mult = len(cluster)
        rat = check_rationality(mean_angle, max_denom)
        results.append({
            'angle': float(mean_angle),
            'angle_deg': float(np.degrees(mean_angle)),
            'multiplicity': mult,
            'rational': f'{rat[0]}/{rat[1]}π' if rat else None,
            'rational_error': float(rat[2]) if rat else None,
        })

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'results', 'data')
    os.makedirs(outdir, exist_ok=True)

    outpath_txt = os.path.join(outdir, 'optimal_ideal_4polytopes.txt')
    outpath_json = os.path.join(outdir, 'optimal_ideal_4polytopes.json')

    txtfile = open(outpath_txt, 'w')

    def tprint(*args, **kwargs):
        msg = ' '.join(str(a) for a in args)
        print(msg, **kwargs)
        txtfile.write(msg + '\n')
        txtfile.flush()

    tprint("=" * 90)
    tprint("Optimal ideal 4-polytopes in H^4")
    tprint(f"Regular ideal 4-simplex volume: {regular_ideal_4simplex_volume():.8f}")
    tprint("=" * 90)

    vertex_counts = [5, 6, 7, 8, 9, 10, 12, 16, 20, 30]
    n_restarts = 200

    all_results = []

    for N in vertex_counts:
        tprint(f"\n{'='*90}")
        tprint(f"N = {N} vertices on S^3")
        tprint(f"{'='*90}")

        t0 = time.time()
        best_vol, best_x, top10 = run_optimization(N, n_restarts=n_restarts)
        dt = time.time() - t0

        tprint(f"Best volume: {best_vol:.10f}")
        tprint(f"Top 10 volumes: {', '.join(f'{v:.8f}' for v in top10)}")
        tprint(f"Optimization time: {dt:.1f}s")

        # Reconstruct optimal points
        pts_4d = angles_to_S3(best_x, N)
        # Renormalize (might drift slightly)
        pts_4d /= np.linalg.norm(pts_4d, axis=1, keepdims=True)

        # 1-skeleton analysis
        edges, degree_seq, adj = analyze_1skeleton(pts_4d)
        tprint(f"\n1-skeleton: {len(edges)} edges")
        tprint(f"Degree sequence: {degree_seq}")
        tprint(f"f-vector: f0={N}, f1={len(edges)}")

        # Get full f-vector from convex hull
        hull = ConvexHull(pts_4d)
        n_facets = len(hull.simplices)  # 3-faces (tetrahedra)
        # 2-faces (triangles): extract from facets
        triangles = set()
        for s in hull.simplices:
            for tri in combinations(s, 3):
                triangles.add(tuple(sorted(tri)))
        n_triangles = len(triangles)
        tprint(f"Full f-vector: ({N}, {len(edges)}, {n_triangles}, {n_facets})")
        # Euler: f0 - f1 + f2 - f3 = 0 for S^3
        euler = N - len(edges) + n_triangles - n_facets
        tprint(f"Euler characteristic: {euler} (should be 0 for S^3)")

        # Automorphism group
        grp_order, orbits, numorbits, generators = compute_automorphism_group(adj, N)
        tprint(f"\nAutomorphism group order: {grp_order}")
        tprint(f"Vertex orbits ({numorbits}): {orbits}")

        # Identify the graph if possible
        if degree_seq == [N-1] * N:
            tprint("Graph: complete graph K_N (every vertex adjacent to every other)")
        elif len(set(degree_seq)) == 1:
            tprint(f"Graph: {degree_seq[0]}-regular on {N} vertices")

        # Dihedral angles
        all_angles, tet_data = collect_dihedral_angles(pts_4d)
        n_tets = len(tet_data)
        tprint(f"\nDelaunay decomposition: {n_tets} ideal 4-simplices")

        # Volume breakdown
        tet_vols = [t['volume'] for t in tet_data]
        tprint(f"Tet volumes: min={min(tet_vols):.8f}, max={max(tet_vols):.8f}, "
               f"mean={np.mean(tet_vols):.8f}")
        if max(tet_vols) > 0:
            tprint(f"Max tet vol / regular: {max(tet_vols)/regular_ideal_4simplex_volume():.6f}")

        # Angle spectrum
        angle_spectrum = analyze_angle_spectrum(all_angles)
        tprint(f"\nAngle spectrum ({len(angle_spectrum)} distinct values):")
        tprint(f"  {'angle':>12} {'degrees':>10} {'mult':>5} {'rational?':>12}")
        tprint(f"  {'-'*45}")
        for a in angle_spectrum:
            rat_str = a['rational'] if a['rational'] else '---'
            tprint(f"  {a['angle']:12.8f} {a['angle_deg']:10.4f}° {a['multiplicity']:5d} "
                   f"{rat_str:>12}")

        # Check: are ALL angles rational multiples of π?
        n_rational = sum(1 for a in angle_spectrum if a['rational'] is not None)
        tprint(f"\n{n_rational}/{len(angle_spectrum)} distinct angles are rational multiples of π")

        # Save result
        result = {
            'n_vertices': N,
            'best_volume': float(best_vol),
            'top10': [float(v) for v in top10],
            'time': dt,
            'f_vector': [N, len(edges), n_triangles, n_facets],
            'degree_sequence': degree_seq,
            'automorphism_order': grp_order,
            'vertex_orbits': orbits,
            'n_orbits': numorbits,
            'n_delaunay_tets': n_tets,
            'tet_volumes': [float(v) for v in tet_vols],
            'angle_spectrum': angle_spectrum,
            'n_rational_angles': n_rational,
            'optimal_points_4d': pts_4d.tolist(),
        }
        all_results.append(result)

    # Summary table
    tprint(f"\n\n{'='*90}")
    tprint("SUMMARY")
    tprint(f"{'='*90}")
    tprint(f"{'N':>4} {'Volume':>14} {'f-vector':>25} {'|Aut|':>8} "
           f"{'#orbits':>7} {'#tets':>5} {'#angles':>7} {'#rat':>5}")
    tprint("-" * 90)
    for r in all_results:
        fv = f"({r['f_vector'][0]},{r['f_vector'][1]},{r['f_vector'][2]},{r['f_vector'][3]})"
        tprint(f"{r['n_vertices']:4d} {r['best_volume']:14.10f} {fv:>25} "
               f"{r['automorphism_order']:8d} {r['n_orbits']:7d} "
               f"{r['n_delaunay_tets']:5d} "
               f"{len(r['angle_spectrum']):7d} {r['n_rational_angles']:5d}")

    # Save JSON
    with open(outpath_json, 'w') as f:
        json.dump(all_results, f, indent=2)

    tprint(f"\nResults saved to {outpath_json}")
    tprint(f"Table saved to {outpath_txt}")
    txtfile.close()


if __name__ == '__main__':
    main()
