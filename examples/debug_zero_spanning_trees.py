#!/usr/bin/env python3
"""
Debug the zero spanning tree cases - these shouldn't exist for 3-connected graphs!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import networkx as nx
from ideal_poly_volume_toolkit.plantri_interface import find_plantri_executable
import subprocess

def get_triangulations_text(n_vertices: int, min_connectivity: int = 3) -> list:
    """Generate triangulations in ASCII format."""
    plantri = find_plantri_executable()
    if plantri is None:
        raise RuntimeError("plantri not found")

    args = [plantri, f'-pc{min_connectivity}', '-a', str(n_vertices)]
    result = subprocess.run(args, capture_output=True, text=True)

    triangulations = []

    for line in result.stdout.split('\n'):
        line = line.strip()
        if not line or line.startswith('>'):
            continue

        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue

        n = int(parts[0])
        adj_str = parts[1]

        # Build adjacency dict
        adj = {}
        vertex_lists = adj_str.split(',')

        for v_idx, neighbor_str in enumerate(vertex_lists):
            neighbors = []
            for letter in neighbor_str:
                neighbor_idx = ord(letter) - ord('a')
                neighbors.append(neighbor_idx)
            adj[v_idx] = neighbors

        # Convert adjacency to triangles
        triangles = []
        for v0 in range(n):
            for v1 in adj[v0]:
                if v1 <= v0:
                    continue
                for v2 in adj[v1]:
                    if v2 <= v1:
                        continue
                    if v2 in adj[v0]:
                        tri = tuple(sorted([v0, v1, v2]))
                        if tri not in triangles:
                            triangles.append(tri)

        if triangles:
            triangulations.append((n, adj, triangles))

    return triangulations


def remove_vertex_to_planar(triangles: list, vertex_to_remove: int) -> list:
    """Remove a vertex to create planar triangulation."""
    return [tri for tri in triangles if vertex_to_remove not in tri]


def triangles_to_graph(triangles: list) -> nx.Graph:
    """Convert triangle list to NetworkX graph."""
    G = nx.Graph()
    for tri in triangles:
        v0, v1, v2 = tri
        G.add_edge(v0, v1)
        G.add_edge(v1, v2)
        G.add_edge(v2, v0)
    return G


def count_spanning_trees_kirchhoff(G: nx.Graph) -> int:
    """Count spanning trees using Kirchhoff's matrix-tree theorem."""
    if len(G.nodes()) == 0:
        return 0
    if len(G.nodes()) == 1:
        return 1

    # Get Laplacian matrix
    L = nx.laplacian_matrix(G).toarray()

    # Remove first row and column (compute cofactor)
    L_reduced = L[1:, 1:]

    # Compute determinant (this is the number of spanning trees)
    det = np.linalg.det(L_reduced)

    # Round to nearest integer (should be exact integer, but floating point)
    return int(round(det))


def debug_zero_cases():
    """Debug cases with zero spanning trees."""

    print("="*70)
    print("DEBUGGING ZERO SPANNING TREE CASES")
    print("="*70)

    # Generate n=10 triangulations
    print("\nGenerating n=10 triangulations...")
    closed_tris = get_triangulations_text(10, min_connectivity=3)
    print(f"Generated {len(closed_tris)} closed triangulations")

    # Find first few zero cases
    zero_cases = []
    for idx, (n, adj, closed_tri) in enumerate(closed_tris):
        planar_tri = remove_vertex_to_planar(closed_tri, 0)
        G = triangles_to_graph(planar_tri)
        n_spanning = count_spanning_trees_kirchhoff(G)

        if n_spanning == 0:
            zero_cases.append((idx, n, adj, closed_tri, planar_tri, G))
            if len(zero_cases) >= 5:
                break

    print(f"\nFound {len(zero_cases)} zero-spanning-tree cases in first {idx+1} triangulations")

    # Analyze each zero case
    for case_num, (idx, n, adj, closed_tri, planar_tri, G) in enumerate(zero_cases, 1):
        print("\n" + "="*70)
        print(f"ZERO CASE #{case_num}: Triangulation index {idx}")
        print("="*70)

        # Original closed triangulation
        print(f"\nOriginal closed triangulation (n={n} vertices):")
        print(f"  Number of triangles: {len(closed_tri)}")
        print(f"  Triangles: {closed_tri[:10]}..." if len(closed_tri) > 10 else f"  Triangles: {closed_tri}")

        # Build graph from closed triangulation
        G_closed = nx.Graph()
        for v in range(n):
            for neighbor in adj[v]:
                G_closed.add_edge(v, neighbor)

        print(f"\nClosed graph properties:")
        print(f"  Nodes: {G_closed.number_of_nodes()}")
        print(f"  Edges: {G_closed.number_of_edges()}")
        print(f"  Connected: {nx.is_connected(G_closed)}")
        print(f"  Node connectivity: {nx.node_connectivity(G_closed)}")

        # Check degrees
        degrees = dict(G_closed.degree())
        print(f"  Degree sequence: {sorted(degrees.values())}")

        # Planar triangulation (after removing vertex 0)
        print(f"\nPlanar triangulation (after removing vertex 0):")
        print(f"  Number of triangles: {len(planar_tri)}")
        print(f"  Triangles: {planar_tri[:10]}..." if len(planar_tri) > 10 else f"  Triangles: {planar_tri}")

        # Planar graph
        print(f"\nPlanar graph properties:")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Connected: {nx.is_connected(G)}")

        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            print(f"  Number of components: {len(components)}")
            print(f"  Component sizes: {[len(c) for c in components]}")
            print(f"  Components: {components}")

        # Check which vertices appear in planar triangulation
        vertices_in_planar = set()
        for tri in planar_tri:
            vertices_in_planar.update(tri)

        print(f"\nVertices analysis:")
        print(f"  Vertices in planar triangulation: {sorted(vertices_in_planar)}")
        print(f"  Expected vertices (1 to {n-1}): {list(range(1, n))}")
        missing = set(range(1, n)) - vertices_in_planar
        if missing:
            print(f"  MISSING vertices: {sorted(missing)}")

        # Check Laplacian
        if G.number_of_nodes() > 0:
            L = nx.laplacian_matrix(G).toarray()
            print(f"\nLaplacian matrix rank: {np.linalg.matrix_rank(L)}")
            print(f"Expected rank for connected graph: {G.number_of_nodes() - 1}")

            eigenvalues = np.linalg.eigvalsh(L)
            print(f"Laplacian eigenvalues (sorted): {eigenvalues[:5]}...")
            print(f"Number of zero eigenvalues: {sum(abs(ev) < 1e-10 for ev in eigenvalues)}")


if __name__ == '__main__':
    debug_zero_cases()
