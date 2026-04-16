#!/usr/bin/env python3
"""
Generate optimal ideal 4-polytope data and LLM challenge prompts for
missing vertex counts (N=11, 13, 14, 15).

Appends to the existing optimal_ideal_4polytopes.json and regenerates
challenge files.
"""

import json
import numpy as np
import time
import sys
import os
from pathlib import Path
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
from multiprocessing import Pool
from itertools import combinations
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from ideal_poly_volume_toolkit.geometry_4d import (
    ideal_4simplex_volume_from_tet,
    stereo_project_S3,
)


# -- Parameterization -------------------------------------------------------

def angles_to_S3(angles, n):
    a = angles[:n]
    b = angles[n:2*n]
    c = angles[2*n:3*n]
    sa, ca = np.sin(a), np.cos(a)
    sb, cb = np.sin(b), np.cos(b)
    sc, cc = np.sin(c), np.cos(c)
    return np.column_stack([sa*sb*cc, sa*sb*sc, sa*cb, ca])


def neg_volume_4d(angles):
    n = len(angles) // 3
    pts = angles_to_S3(angles, n)
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


def optimize_worker(args):
    n_vertices, seed = args
    rng = np.random.default_rng(seed)
    pts0 = rng.standard_normal((n_vertices, 4))
    pts0 /= np.linalg.norm(pts0, axis=1, keepdims=True)
    a0 = np.arccos(np.clip(pts0[:, 3], -1, 1))
    sin_a = np.sin(a0) + 1e-30
    b0 = np.arccos(np.clip(pts0[:, 2] / sin_a, -1, 1))
    c0 = np.arctan2(pts0[:, 1], pts0[:, 0])
    x0 = np.concatenate([a0, b0, c0])
    res = minimize(neg_volume_4d, x0, method='L-BFGS-B',
                   options={'maxiter': 3000, 'ftol': 1e-14})
    res2 = minimize(neg_volume_4d, res.x, method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-11})
    return -res2.fun, res2.x


def run_optimization(n_vertices, n_restarts=200, n_workers=64):
    work = [(n_vertices, 31415 + n_vertices * 137 + i)
            for i in range(n_restarts)]
    with Pool(processes=min(n_workers, n_restarts)) as pool:
        results = pool.map(optimize_worker, work)
    volumes = [r[0] for r in results]
    best_idx = np.argmax(volumes)
    return volumes[best_idx], results[best_idx][1], np.sort(volumes)[-10:]


def face_lattice_from_hull(pts):
    hull = ConvexHull(pts)
    edges = set()
    triangles = set()
    facets = set()
    for simplex in hull.simplices:
        s = tuple(sorted(simplex))
        facets.add(s)
        for tri in combinations(s, 3):
            triangles.add(tri)
        for edge in combinations(s, 2):
            edges.add(edge)
    return {
        'edges': sorted([[int(x) for x in e] for e in edges]),
        'triangles': sorted([[int(x) for x in t] for t in triangles]),
        'facets': sorted([[int(x) for x in f] for f in facets]),
        'f_vector': [len(pts), len(edges), len(triangles), len(facets)],
    }


def degree_sequence(edges, n):
    deg = Counter()
    for e in edges:
        deg[e[0]] += 1
        deg[e[1]] += 1
    return sorted([deg.get(i, 0) for i in range(n)], reverse=True)


def compute_automorphism_order(adj, n):
    try:
        import pynauty
        g = pynauty.Graph(n)
        for v, nbrs in adj.items():
            g.connect_vertex(v, list(nbrs))
        _, grpsize1, grpsize2, _, _ = pynauty.autgrp(g)
        return int(grpsize1 * 10**grpsize2)
    except ImportError:
        return None


def main():
    base = Path(__file__).parent.parent
    data_dir = base / 'results' / 'data'
    challenges_dir = data_dir / 'llm_challenges'
    challenges_dir.mkdir(exist_ok=True)

    json_path = data_dir / 'optimal_ideal_4polytopes.json'
    existing = json.load(open(json_path)) if json_path.exists() else []
    existing_N = {r['n_vertices'] for r in existing}

    missing = [n for n in [11, 13, 14, 15] if n not in existing_N]
    if not missing:
        print("All N values already present. Regenerating challenge files only.")
    else:
        print(f"Missing N values: {missing}")

    n_restarts = 200

    for N in missing:
        print(f"\n{'='*60}")
        print(f"Optimizing N={N}...")
        print(f"{'='*60}")

        t0 = time.time()
        best_vol, best_x, top10 = run_optimization(N, n_restarts=n_restarts)
        dt = time.time() - t0

        pts_4d = angles_to_S3(best_x, N)
        pts_4d /= np.linalg.norm(pts_4d, axis=1, keepdims=True)

        fl = face_lattice_from_hull(pts_4d)
        fv = fl['f_vector']

        # Euler check
        euler = fv[0] - fv[1] + fv[2] - fv[3]
        assert euler == 0, f"Euler relation fails: {fv}"

        # Adjacency for automorphism
        adj = {i: set() for i in range(N)}
        for u, v in fl['edges']:
            adj[u].add(v)
            adj[v].add(u)
        aut_order = compute_automorphism_order(adj, N)

        degs = degree_sequence(fl['edges'], N)

        rec = {
            'n_vertices': N,
            'best_volume': float(best_vol),
            'f_vector': fv,
            'degree_sequence': degs,
            'automorphism_order': aut_order,
            'optimal_points_4d': pts_4d.tolist(),
            'top10_volumes': top10.tolist(),
        }
        existing.append(rec)

        print(f"  Volume: {best_vol:.8f}")
        print(f"  f-vector: {fv}")
        print(f"  Degrees: {degs}")
        print(f"  |Aut|: {aut_order}")
        print(f"  Top 10: {', '.join(f'{v:.6f}' for v in top10)}")
        print(f"  Time: {dt:.1f}s")

    # Sort by N and save
    existing.sort(key=lambda r: r['n_vertices'])
    with open(json_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved {len(existing)} records to {json_path}")

    # Regenerate ALL challenge files (with updated inscribed prompt)
    print("\nRegenerating challenge files...")
    for rec in existing:
        N = rec['n_vertices']
        challenge = {
            'n_vertices': N,
            'dimension': 4,
            'f_vector': rec['f_vector'],
            'degree_sequence': rec.get('degree_sequence',
                                       degree_sequence(
                                           face_lattice_from_hull(
                                               np.array(rec['optimal_points_4d'])
                                           )['edges'], N)),
            'edges': face_lattice_from_hull(
                np.array(rec['optimal_points_4d']))['edges'],
            'triangles': face_lattice_from_hull(
                np.array(rec['optimal_points_4d']))['triangles'],
            'facets': face_lattice_from_hull(
                np.array(rec['optimal_points_4d']))['facets'],
        }

        with open(challenges_dir / f'face_lattice_N{N}.json', 'w') as f:
            json.dump(challenge, f, indent=2)

        # Also write the text prompt
        fv = challenge['f_vector']
        edges = challenge['edges']
        facets = challenge['facets']
        deg_seq = challenge['degree_sequence']

        prompt = f"""You are given the face lattice of a convex 4-polytope (a polytope in R^4) with {N} vertices.

**Face lattice data:**
- f-vector: {fv}  (vertices, edges, 2-faces, 3-faces)
- Degree sequence (vertex degrees in the 1-skeleton, descending): {deg_seq}
- Edge list (pairs of vertex indices 0..{N-1}):
  {json.dumps(edges)}
- Facets (3-faces, each a tetrahedron given as 4 vertex indices):
  {json.dumps(facets)}

**Your task:** Produce {N} points on the unit 3-sphere S^3 (i.e., each point has unit norm in R^4) whose convex hull has exactly this combinatorial type — the same face lattice up to relabeling of vertices.

**Output format:** Return ONLY a JSON object with a single key "points" containing a list of {N} points, each a list of 4 floats. Example format:
```json
{{"points": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], ...]}}
```

Do not include any other text outside the JSON block. Think carefully about the combinatorial structure before choosing coordinates."""

        with open(challenges_dir / f'prompt_N{N}.txt', 'w') as f:
            f.write(prompt)

        print(f"  N={N:3d}  f={fv}  prompt written")

    # Regenerate combined challenge file
    all_challenges = []
    for p in sorted(challenges_dir.glob('face_lattice_N*.json')):
        all_challenges.append(json.load(open(p)))
    with open(challenges_dir / 'all_challenges.json', 'w') as f:
        json.dump(all_challenges, f, indent=2)

    print(f"\nDone. {len(all_challenges)} challenge files in {challenges_dir}/")


if __name__ == '__main__':
    main()
