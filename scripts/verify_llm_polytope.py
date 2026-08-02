#!/usr/bin/env python3
"""
Verify an LLM-generated 4-polytope against a face lattice challenge.

Given:
  - A challenge JSON (face lattice)
  - An LLM response JSON with 'points': [[x1,x2,x3,x4], ...]

Check:
  1. Correct number of vertices
  2. Points are in general position (full-dimensional in R^4)
  3. Convex hull has the same f-vector
  4. Face lattice is isomorphic (via graph isomorphism of the 1-skeleton)
"""

import json
import sys
import numpy as np
from scipy.spatial import ConvexHull
from itertools import combinations


def face_lattice_from_points(pts):
    """Extract face lattice from a point set in R^4."""
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
        'f_vector': [len(pts), len(edges), len(triangles), len(facets)],
        'edges': sorted(edges),
        'facets': sorted(facets),
    }


def edge_set_to_adjacency(n, edges):
    """Build adjacency dict from edge list."""
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def degree_sequence(n, edges):
    """Sorted degree sequence."""
    adj = edge_set_to_adjacency(n, edges)
    return sorted([len(adj[v]) for v in range(n)], reverse=True)


def are_graphs_isomorphic(n1, edges1, n2, edges2):
    """
    Check if two graphs are isomorphic using pynauty if available,
    otherwise fall back to networkx.
    """
    if n1 != n2:
        return False

    try:
        import pynauty
        def to_nauty(n, edges):
            adj = edge_set_to_adjacency(n, edges)
            g = pynauty.Graph(n)
            for v in range(n):
                g.connect_vertex(v, sorted(adj[v]))
            return g
        g1 = to_nauty(n1, edges1)
        g2 = to_nauty(n2, edges2)
        return pynauty.certificate(g1) == pynauty.certificate(g2)
    except ImportError:
        pass

    try:
        import networkx as nx
        from networkx.algorithms.isomorphism import GraphMatcher
        G1 = nx.Graph()
        G1.add_nodes_from(range(n1))
        G1.add_edges_from(edges1)
        G2 = nx.Graph()
        G2.add_nodes_from(range(n2))
        G2.add_edges_from(edges2)
        return nx.is_isomorphic(G1, G2)
    except ImportError:
        pass

    # Fallback: compare degree sequences and edge counts (necessary but not sufficient)
    print("  WARNING: neither pynauty nor networkx available; using weak check")
    ds1 = degree_sequence(n1, edges1)
    ds2 = degree_sequence(n2, edges2)
    return ds1 == ds2 and len(edges1) == len(edges2)


def verify(challenge_path, response_path):
    with open(challenge_path) as f:
        challenge = json.load(f)
    with open(response_path) as f:
        response = json.load(f)

    pts = np.array(response['points'], dtype=float)
    N_expected = challenge['n_vertices']
    f_expected = challenge['f_vector']

    print(f"Challenge: N={N_expected}, f={f_expected}")
    print(f"Response:  {pts.shape[0]} points in R^{pts.shape[1]}")

    # Check 1: vertex count
    if pts.shape[0] != N_expected:
        print(f"FAIL: expected {N_expected} vertices, got {pts.shape[0]}")
        return False
    if pts.shape[1] != 4:
        print(f"FAIL: expected 4-dimensional points, got {pts.shape[1]}D")
        return False

    # Check 2: full-dimensional
    rank = np.linalg.matrix_rank(pts - pts[0], tol=1e-8)
    if rank < 4:
        print(f"FAIL: points span only {rank} dimensions, need 4")
        return False
    print(f"  Dimension check: rank = {rank} ✓")

    # Check 3: f-vector
    lattice = face_lattice_from_points(pts)
    f_got = lattice['f_vector']
    if f_got != f_expected:
        print(f"FAIL: f-vector mismatch: got {f_got}, expected {f_expected}")
        return False
    print(f"  f-vector: {f_got} ✓")

    # Check 4: Euler relation
    euler = f_got[0] - f_got[1] + f_got[2] - f_got[3]
    if euler != 0:
        print(f"FAIL: Euler relation: {euler} ≠ 0")
        return False
    print(f"  Euler relation: {euler} = 0 ✓")

    # Check 5: graph isomorphism of 1-skeleton
    edges_expected = [tuple(e) for e in challenge['edges']]
    edges_got = [tuple(e) for e in lattice['edges']]

    iso = are_graphs_isomorphic(N_expected, edges_expected, N_expected, edges_got)
    if not iso:
        ds_expected = degree_sequence(N_expected, edges_expected)
        ds_got = degree_sequence(N_expected, edges_got)
        print(f"FAIL: 1-skeleton not isomorphic")
        print(f"  Expected degree seq: {ds_expected}")
        print(f"  Got degree seq:      {ds_got}")
        return False
    print(f"  1-skeleton isomorphism: ✓")

    print("PASS: polytope realizes the given face lattice.")
    return True


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <challenge.json> <response.json>")
        print("  response.json should contain: {\"points\": [[x1,x2,x3,x4], ...]}")
        sys.exit(1)
    ok = verify(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)
