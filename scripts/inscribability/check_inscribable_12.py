#!/usr/bin/env python3
"""
Check which 12-vertex triangulated spheres are inscribable (realizable as
convex hulls of points on a sphere).

Method:
1. Use plantri to enumerate all triangulations of the sphere with 12 vertices
2. For each triangulation, remove all triangles incident to vertex 0 (vertex at infinity)
3. Check Rivin's strict LP: all dihedral angles < π and all boundary angles < π
4. Count how many pass the test

By Rivin's theorem, a triangulation is inscribable iff the strict LP is feasible.
"""

import subprocess
import sys
from typing import List, Tuple, Set, Generator
from collections import defaultdict
import multiprocessing as mp
import time

sys.path.insert(0, '/home/igor/devel/ideal_poly_volume_toolkit')

from ideal_poly_volume_toolkit.rivin_delaunay import check_delaunay_realizability


def parse_planar_code(data: bytes) -> Generator[List[List[int]], None, None]:
    """
    Parse plantri's planar_code binary output format.

    Yields adjacency lists (0-indexed) for each graph.
    """
    pos = 0

    # Skip header if present
    if data[:15] == b'>>planar_code<<':
        pos = 15

    while pos < len(data):
        n = data[pos]
        pos += 1

        if n == 0:
            break

        adj = []
        for v in range(n):
            neighbors = []
            while pos < len(data):
                neighbor = data[pos]
                pos += 1
                if neighbor == 0:
                    break
                neighbors.append(neighbor - 1)  # Convert to 0-indexed
            adj.append(neighbors)

        yield adj


def adjacency_to_triangles(adj: List[List[int]]) -> List[Tuple[int, int, int]]:
    """
    Convert adjacency list to list of triangles.

    For a triangulation, consecutive neighbors around each vertex form triangles.
    """
    triangles = set()

    for v, neighbors in enumerate(adj):
        n_neighbors = len(neighbors)
        for i in range(n_neighbors):
            u = neighbors[i]
            w = neighbors[(i + 1) % n_neighbors]

            # Check edge u-w exists (forms valid triangle)
            if w in adj[u]:
                tri = tuple(sorted([v, u, w]))
                triangles.add(tri)

    return list(triangles)


def remove_vertex_at_infinity(triangles: List[Tuple[int, int, int]],
                               infinity_vertex: int = 0) -> List[Tuple[int, int, int]]:
    """
    Remove a vertex (send it to infinity) from a triangulation.

    Returns the triangles that don't contain the infinity vertex.
    """
    return [tri for tri in triangles if infinity_vertex not in tri]


def check_single_graph(args):
    """Check if a single triangulation is inscribable."""
    graph_id, adj = args

    # Convert to triangles
    all_triangles = adjacency_to_triangles(adj)

    # Remove vertex 0 (at infinity)
    triangles = remove_vertex_at_infinity(all_triangles, infinity_vertex=0)

    # Check strict realizability
    result = check_delaunay_realizability(
        triangles,
        boundary_vertices=None,  # Auto-detect (neighbors of vertex 0)
        verbose=False,
        strict=True  # Require all angles strictly < π
    )

    return (graph_id, result['realizable'], result)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Find inscribable triangulations of the sphere'
    )
    parser.add_argument('--vertices', '-n', type=int, default=12,
                       help='Number of vertices (default: 12)')
    parser.add_argument('--workers', '-w', type=int, default=mp.cpu_count(),
                       help='Number of parallel workers')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print details of inscribable triangulations')
    args = parser.parse_args()

    n_vertices = args.vertices
    plantri = '/home/igor/miniforge3/envs/HPC/lib/python3.12/site-packages/sage_wheels/bin/plantri'

    print(f"Enumerating triangulations of the sphere with {n_vertices} vertices...")
    print(f"Using plantri: {plantri}")

    # Run plantri (no -p flag = triangulations)
    start_time = time.time()
    result = subprocess.run([plantri, str(n_vertices)], capture_output=True)
    plantri_time = time.time() - start_time

    # Parse output
    graphs = list(parse_planar_code(result.stdout))
    n_graphs = len(graphs)
    print(f"Found {n_graphs:,} triangulations (plantri took {plantri_time:.2f}s)")

    # Verify triangle count for first graph
    test_triangles = adjacency_to_triangles(graphs[0])
    expected = 2 * n_vertices - 4  # Euler: F = 2V - 4 for triangulation of sphere
    print(f"Each triangulation has {len(test_triangles)} triangles (expected: {expected})")

    # Check realizability
    print(f"\nChecking inscribability with {args.workers} workers...")
    print("(Using Rivin's strict LP: dihedral angles < π, boundary angles < π)")
    start_time = time.time()

    inscribable = []
    n_checked = 0

    work_items = [(i, adj) for i, adj in enumerate(graphs)]

    with mp.Pool(args.workers) as pool:
        for graph_id, is_inscribable, result in pool.imap_unordered(check_single_graph, work_items, chunksize=100):
            n_checked += 1

            if is_inscribable:
                inscribable.append((graph_id, result))
                if args.verbose:
                    slack = result.get('slack', 0)
                    print(f"  Graph {graph_id}: INSCRIBABLE (slack ε = {slack:.6f})")

            if n_checked % 1000 == 0:
                elapsed = time.time() - start_time
                rate = n_checked / elapsed
                print(f"  Progress: {n_checked:,}/{n_graphs:,} ({100*n_checked/n_graphs:.1f}%) "
                      f"- {len(inscribable)} inscribable - {rate:.0f}/s")

    check_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"RESULTS for {n_vertices}-vertex triangulations of the sphere")
    print(f"{'='*60}")
    print(f"Total triangulations:     {n_graphs:,}")
    print(f"Inscribable (on sphere):  {len(inscribable)}")
    print(f"Not inscribable:          {n_graphs - len(inscribable):,}")
    print(f"Time: {check_time:.1f}s ({n_graphs/check_time:.0f} graphs/s)")

    if inscribable:
        print(f"\nInscribable graph IDs: {sorted([g[0] for g in inscribable])}")

    return len(inscribable)


if __name__ == '__main__':
    main()
