#!/usr/bin/env python3
"""
Export face lattices of optimal ideal 4-polytopes as LLM benchmark challenges.

Given N vertices on S^3, compute the convex hull in R^4 and extract the full
face lattice (vertices, edges, 2-faces, 3-faces).  Export as JSON challenges
where the LLM must produce coordinates of a 4-polytope realizing the given
combinatorial type.
"""

import json
import numpy as np
from scipy.spatial import ConvexHull
from itertools import combinations
from pathlib import Path


def face_lattice_from_hull(pts):
    """
    Extract the full face lattice of a convex polytope in R^4.

    Parameters
    ----------
    pts : (N, 4) array — vertices

    Returns
    -------
    dict with keys:
        'vertices': list of vertex indices [0..N-1]
        'edges': sorted list of 2-tuples
        'triangles': sorted list of 3-tuples (2-faces)
        'facets': sorted list of 4-tuples (3-faces = tetrahedra on boundary)
        'f_vector': [f0, f1, f2, f3]
    """
    hull = ConvexHull(pts)
    facets_raw = hull.simplices  # each row is a 3-face (tetrahedron)

    # Collect all k-faces by taking k+1-subsets of each facet
    edges = set()
    triangles = set()
    facets = set()

    for simplex in facets_raw:
        s = tuple(sorted(simplex))
        facets.add(s)
        for tri in combinations(s, 3):
            triangles.add(tri)
        for edge in combinations(s, 2):
            edges.add(edge)

    edges = sorted(edges)
    triangles = sorted(triangles)
    facets = sorted(facets)

    return {
        'vertices': list(range(len(pts))),
        'edges': [[int(x) for x in e] for e in edges],
        'triangles': [[int(x) for x in t] for t in triangles],
        'facets': [[int(x) for x in f] for f in facets],
        'f_vector': [len(pts), len(edges), len(triangles), len(facets)],
    }


def vertex_facet_incidence(face_data):
    """Compute vertex-facet incidence matrix as list of lists."""
    n = len(face_data['vertices'])
    incidence = []
    for v in range(n):
        inc = [i for i, f in enumerate(face_data['facets']) if v in f]
        incidence.append(inc)
    return incidence


def degree_sequence(face_data):
    """Vertex degrees from edge list."""
    from collections import Counter
    deg = Counter()
    for e in face_data['edges']:
        deg[e[0]] += 1
        deg[e[1]] += 1
    return [deg[v] for v in face_data['vertices']]


def main():
    data_path = Path(__file__).parent.parent / 'results' / 'data' / 'optimal_ideal_4polytopes.json'
    out_dir = Path(__file__).parent.parent / 'results' / 'data' / 'llm_challenges'
    out_dir.mkdir(exist_ok=True)

    with open(data_path) as f:
        records = json.load(f)

    challenges = []

    for rec in records:
        N = rec['n_vertices']
        pts = np.array(rec['optimal_points_4d'])

        # Verify points are on S^3
        norms = np.linalg.norm(pts, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6), f"N={N}: points not on S^3"

        face_data = face_lattice_from_hull(pts)

        # Sanity: Euler relation for 4-polytopes
        # f0 - f1 + f2 - f3 = 0 (Euler for 3-sphere boundary)
        fv = face_data['f_vector']
        euler = fv[0] - fv[1] + fv[2] - fv[3]
        assert euler == 0, f"N={N}: Euler relation fails: {fv} -> {euler}"

        # Verify f-vector matches stored one
        assert fv == rec['f_vector'], \
            f"N={N}: f-vector mismatch: computed {fv} vs stored {rec['f_vector']}"

        degs = degree_sequence(face_data)
        vf_inc = vertex_facet_incidence(face_data)

        challenge = {
            'n_vertices': N,
            'dimension': 4,
            'f_vector': fv,
            'degree_sequence': sorted(degs, reverse=True),
            'edges': face_data['edges'],
            'triangles': face_data['triangles'],
            'facets': face_data['facets'],
            'vertex_facet_incidence': vf_inc,
            'automorphism_order': rec.get('automorphism_order'),
            'notes': (
                f"Face lattice of an optimal ideal 4-polytope with {N} vertices. "
                f"Challenge: produce {N} points in R^4 whose convex hull has this "
                f"combinatorial type (same face lattice up to relabeling)."
            ),
        }
        challenges.append(challenge)

        # Individual file
        with open(out_dir / f'face_lattice_N{N}.json', 'w') as f:
            json.dump(challenge, f, indent=2)

        print(f"N={N:3d}  f={fv}  edges={len(face_data['edges'])}  "
              f"triangles={len(face_data['triangles'])}  facets={len(face_data['facets'])}  "
              f"Euler={euler}  aut={rec.get('automorphism_order')}")

    # Combined file
    with open(out_dir / 'all_challenges.json', 'w') as f:
        json.dump(challenges, f, indent=2)

    print(f"\nExported {len(challenges)} challenges to {out_dir}/")

    # Also export the ground-truth coordinates (separate file, for verification)
    ground_truth = []
    for rec in records:
        ground_truth.append({
            'n_vertices': rec['n_vertices'],
            'optimal_points_4d': rec['optimal_points_4d'],
            'volume': rec['best_volume'],
        })
    with open(out_dir / 'ground_truth.json', 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print("Ground truth coordinates saved separately for verification.")


if __name__ == '__main__':
    main()
