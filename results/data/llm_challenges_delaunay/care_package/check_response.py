#!/usr/bin/env python3
"""
Standalone checker for the Delaunay Realizability Benchmark.

Usage:
    python check_response.py benchmark.json response.json

The response.json should contain a dict mapping challenge IDs to answers:
{
    "challenge_00": {"points": [[x0,y0], [x1,y1], ...]},
    "challenge_01": {"result": "None"},
    ...
}

Dependencies: numpy, scipy, pynauty (pip install pynauty)
"""

import json
import sys
import numpy as np
from scipy.spatial import Delaunay
from itertools import combinations
from typing import List, Tuple, Optional, Dict


# ============================================================================
# Graph isomorphism via pynauty
# ============================================================================

def canonical_hash(triangles: List) -> str:
    """Canonical graph hash of a triangulation via pynauty."""
    import pynauty

    vertices = set()
    for tri in triangles:
        vertices.update(int(v) for v in tri)
    n = len(vertices)
    vlist = sorted(vertices)
    v2i = {v: i for i, v in enumerate(vlist)}

    adj = {i: set() for i in range(n)}
    for tri in triangles:
        i0, i1, i2 = v2i[int(tri[0])], v2i[int(tri[1])], v2i[int(tri[2])]
        adj[i0].update([i1, i2])
        adj[i1].update([i0, i2])
        adj[i2].update([i0, i1])

    g = pynauty.Graph(n, directed=False, adjacency_dict=adj)
    return str(pynauty.canon_label(g))


# ============================================================================
# Rivin LP realizability check (self-contained)
# ============================================================================

def check_realizable_lp(triangles: List) -> bool:
    """
    Check Delaunay realizability via Rivin's LP.

    A triangulation of a planar point set is Delaunay-realizable iff
    there exist positive angles theta_c for each corner c of each
    triangle such that:
      (i)   Each triangle's angles sum to pi
      (ii)  Each interior vertex's angles sum to 2*pi
      (iii) Each boundary vertex's angles sum to <= pi
      (iv)  Opposite angles at each interior edge sum to <= pi

    We check feasibility of this LP using scipy.
    """
    from scipy.optimize import linprog

    tris = [tuple(int(v) for v in t) for t in triangles]
    n_tri = len(tris)
    n_corners = 3 * n_tri  # variables: one angle per corner

    # Map corners: corner (tri_idx, local_pos) -> global index
    # corner_idx = tri_idx * 3 + local_pos

    # Identify vertices, edges, boundary
    all_verts = set()
    edge_count = {}
    for ti, (a, b, c) in enumerate(tris):
        all_verts.update([a, b, c])
        for e in [(min(a,b),max(a,b)), (min(b,c),max(b,c)), (min(a,c),max(a,c))]:
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary_edges = {e for e, c in edge_count.items() if c == 1}
    interior_edges = {e for e, c in edge_count.items() if c == 2}

    boundary_verts = set()
    for u, v in boundary_edges:
        boundary_verts.update([u, v])
    interior_verts = all_verts - boundary_verts

    # Build corner lookup: for each (tri_idx, vertex) -> corner_idx
    corner_of = {}
    for ti, (a, b, c) in enumerate(tris):
        corner_of[(ti, a)] = ti * 3 + 0
        corner_of[(ti, b)] = ti * 3 + 1
        corner_of[(ti, c)] = ti * 3 + 2

    # For each edge, find the two opposite corners
    edge_opposite = {}
    for ti, (a, b, c) in enumerate(tris):
        for e, opp in [((min(a,b),max(a,b)), c),
                       ((min(b,c),max(b,c)), a),
                       ((min(a,c),max(a,c)), b)]:
            edge_opposite.setdefault(e, []).append(corner_of[(ti, opp)])

    # LP: minimize 0 (feasibility) subject to:
    # Equality: triangle angle sums = pi
    # Equality: interior vertex angle sums = 2*pi
    # Inequality: boundary vertex angle sums <= pi
    # Inequality: opposite angle sums at interior edges <= pi
    # Bounds: all angles > 0 (use small lower bound)

    A_eq_rows = []
    b_eq = []

    # Triangle sums = pi
    for ti in range(n_tri):
        row = [0.0] * n_corners
        row[ti*3] = 1.0
        row[ti*3+1] = 1.0
        row[ti*3+2] = 1.0
        A_eq_rows.append(row)
        b_eq.append(np.pi)

    # Interior vertex sums = 2*pi
    for v in sorted(interior_verts):
        row = [0.0] * n_corners
        for ti, (a, b, c) in enumerate(tris):
            if v == a: row[ti*3] = 1.0
            elif v == b: row[ti*3+1] = 1.0
            elif v == c: row[ti*3+2] = 1.0
        A_eq_rows.append(row)
        b_eq.append(2 * np.pi)

    A_ub_rows = []
    b_ub = []

    # Boundary vertex sums <= pi
    for v in sorted(boundary_verts):
        row = [0.0] * n_corners
        for ti, (a, b, c) in enumerate(tris):
            if v == a: row[ti*3] = 1.0
            elif v == b: row[ti*3+1] = 1.0
            elif v == c: row[ti*3+2] = 1.0
        A_ub_rows.append(row)
        b_ub.append(np.pi)

    # Interior edge dihedral: opposite angles sum <= pi
    for e in sorted(interior_edges):
        opps = edge_opposite.get(e, [])
        if len(opps) == 2:
            row = [0.0] * n_corners
            row[opps[0]] = 1.0
            row[opps[1]] = 1.0
            A_ub_rows.append(row)
            b_ub.append(np.pi)

    c_obj = [0.0] * n_corners  # feasibility only
    bounds = [(1e-6, np.pi - 1e-6)] * n_corners

    A_eq = np.array(A_eq_rows) if A_eq_rows else None
    b_eq_arr = np.array(b_eq) if b_eq else None
    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub_arr = np.array(b_ub) if b_ub else None

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub_arr,
                     A_eq=A_eq, b_eq=b_eq_arr,
                     bounds=bounds, method='highs')

    return result.success and result.status == 0


# ============================================================================
# Main checker
# ============================================================================

def check_one(challenge: Dict, answer: Dict, ground_truth: Optional[Dict] = None) -> Dict:
    """
    Check a single response.

    Parameters
    ----------
    challenge : dict with 'id', 'n_vertices', 'triangles'
    answer : dict with either 'points' or 'result'='None'
    ground_truth : optional dict with 'is_realizable'

    Returns
    -------
    dict with 'correct', 'reason', 'details'
    """
    cid = challenge['id']
    target_tris = [tuple(t) for t in challenge['triangles']]
    n_verts = challenge['n_vertices']

    if 'result' in answer and answer['result'] == 'None':
        # LLM says not realizable
        if ground_truth and 'is_realizable' in ground_truth:
            is_real = ground_truth['is_realizable']
        else:
            is_real = check_realizable_lp(target_tris)

        if not is_real:
            return {'correct': True, 'reason': 'Correctly identified non-realizable'}
        else:
            return {'correct': False, 'reason': 'Incorrectly said None — triangulation IS realizable'}

    elif 'points' in answer:
        pts = np.array(answer['points'], dtype=float)

        if pts.shape != (n_verts, 2):
            return {'correct': False,
                    'reason': f'Wrong shape: expected ({n_verts}, 2), got {pts.shape}'}

        # Compute Delaunay of proposed points
        try:
            tri = Delaunay(pts)
        except Exception as e:
            return {'correct': False, 'reason': f'Delaunay failed: {e}'}

        llm_tris = [tuple(int(v) for v in s) for s in tri.simplices]

        # Check isomorphism via canonical hash
        target_hash = canonical_hash(target_tris)
        llm_hash = canonical_hash(llm_tris)

        if target_hash == llm_hash:
            return {'correct': True,
                    'reason': 'Valid point set with correct combinatorics',
                    'details': {
                        'target_triangles': len(target_tris),
                        'llm_triangles': len(llm_tris),
                    }}
        else:
            return {'correct': False,
                    'reason': f'Wrong combinatorics: {len(target_tris)} target tri, '
                              f'{len(llm_tris)} produced tri',
                    'details': {
                        'target_triangles': len(target_tris),
                        'llm_triangles': len(llm_tris),
                    }}
    else:
        return {'correct': False, 'reason': 'Invalid response format'}


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} benchmark.json response.json [ground_truth.json]")
        sys.exit(1)

    benchmark = json.load(open(sys.argv[1]))
    responses = json.load(open(sys.argv[2]))
    ground_truth = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}

    challenges = benchmark['challenges']

    print("=" * 70)
    print("DELAUNAY REALIZABILITY BENCHMARK — CHECKER")
    print("=" * 70)
    print()

    n_correct = 0
    n_total = 0

    for challenge in challenges:
        cid = challenge['id']
        if cid not in responses:
            print(f"  {cid}: SKIPPED (no response)")
            continue

        n_total += 1
        gt = ground_truth.get(cid, {})
        result = check_one(challenge, responses[cid], gt)

        status = "PASS" if result['correct'] else "FAIL"
        if result['correct']:
            n_correct += 1

        print(f"  {cid}  N={challenge['n_vertices']:4d}  "
              f"{challenge['n_triangles']:5d} tri  [{status}]  {result['reason']}")

    print()
    print(f"Score: {n_correct}/{n_total}")


if __name__ == '__main__':
    main()
