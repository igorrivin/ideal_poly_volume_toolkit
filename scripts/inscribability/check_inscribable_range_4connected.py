#!/usr/bin/env python3
"""
Check inscribability for 4-connected triangulations of the sphere across a range of vertex counts.
"""

import subprocess
import sys
from typing import List, Tuple, Generator
import multiprocessing as mp
import time

sys.path.insert(0, '/home/igor/devel/ideal_poly_volume_toolkit')

from ideal_poly_volume_toolkit.rivin_delaunay import check_delaunay_realizability


def parse_planar_code(data: bytes) -> Generator[List[List[int]], None, None]:
    """Parse plantri's planar_code binary output format."""
    pos = 0
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
                neighbors.append(neighbor - 1)
            adj.append(neighbors)

        yield adj


def adjacency_to_triangles(adj: List[List[int]]) -> List[Tuple[int, int, int]]:
    """Convert adjacency list to list of triangles."""
    triangles = set()
    for v, neighbors in enumerate(adj):
        n_neighbors = len(neighbors)
        for i in range(n_neighbors):
            u = neighbors[i]
            w = neighbors[(i + 1) % n_neighbors]
            if w in adj[u]:
                tri = tuple(sorted([v, u, w]))
                triangles.add(tri)
    return list(triangles)


def remove_vertex_at_infinity(triangles: List[Tuple[int, int, int]],
                               infinity_vertex: int = 0) -> List[Tuple[int, int, int]]:
    """Remove vertex 0 (at infinity) from triangulation."""
    return [tri for tri in triangles if infinity_vertex not in tri]


def check_single_graph(args):
    """Check if a single triangulation is inscribable."""
    graph_id, adj = args
    all_triangles = adjacency_to_triangles(adj)
    triangles = remove_vertex_at_infinity(all_triangles, infinity_vertex=0)

    result = check_delaunay_realizability(
        triangles,
        boundary_vertices=None,
        verbose=False,
        strict=True
    )
    return (graph_id, result['realizable'], result)


def check_vertex_count(n_vertices: int, n_workers: int, verbose: bool = False):
    """Check all 4-connected triangulations with given vertex count."""
    plantri = '/home/igor/miniforge3/envs/HPC/lib/python3.12/site-packages/sage_wheels/bin/plantri'

    # Run plantri with -c4 flag for 4-connected triangulations
    start_time = time.time()
    result = subprocess.run([plantri, '-c4', str(n_vertices)], capture_output=True)
    plantri_time = time.time() - start_time

    # Parse output
    graphs = list(parse_planar_code(result.stdout))
    n_graphs = len(graphs)

    if n_graphs == 0:
        return n_vertices, 0, 0, 0

    # Check realizability
    check_start = time.time()
    inscribable = []
    n_checked = 0

    work_items = [(i, adj) for i, adj in enumerate(graphs)]

    # Adjust chunksize based on number of graphs
    chunksize = max(1, min(1000, n_graphs // (n_workers * 4)))

    with mp.Pool(n_workers) as pool:
        for graph_id, is_inscribable, res in pool.imap_unordered(check_single_graph, work_items, chunksize=chunksize):
            n_checked += 1
            if is_inscribable:
                inscribable.append(graph_id)

            # Progress for large counts
            if verbose and n_graphs > 10000 and n_checked % 10000 == 0:
                elapsed = time.time() - check_start
                rate = n_checked / elapsed
                eta = (n_graphs - n_checked) / rate if rate > 0 else 0
                print(f"    n={n_vertices}: {n_checked:,}/{n_graphs:,} ({100*n_checked/n_graphs:.1f}%) "
                      f"- {len(inscribable)} inscribable - ETA {eta:.0f}s")

    check_time = time.time() - check_start

    return n_vertices, n_graphs, len(inscribable), check_time


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Check inscribability for 4-connected triangulations')
    parser.add_argument('--min', type=int, default=6, help='Minimum vertices (4-connected needs >= 6)')
    parser.add_argument('--max', type=int, default=20, help='Maximum vertices')
    parser.add_argument('--workers', '-w', type=int, default=mp.cpu_count(), help='Workers')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show progress')
    args = parser.parse_args()

    print(f"Checking inscribability for 4-connected triangulations, n = {args.min} to {args.max} vertices")
    print(f"Using {args.workers} workers")
    print()

    results = []
    total_start = time.time()

    print(f"{'n':>4} | {'Total':>12} | {'Inscribable':>12} | {'Ratio':>10} | {'Time':>8}")
    print("-" * 60)

    for n in range(args.min, args.max + 1):
        n_vert, total, inscr, t = check_vertex_count(n, args.workers, args.verbose)

        if total == 0:
            print(f"{n:4d} | {'NONE':>12} | {'-':>12} | {'-':>10} | {t:>7.1f}s")
            continue

        ratio = inscr / total if total > 0 else 0
        results.append((n_vert, total, inscr, ratio))

        print(f"{n_vert:4d} | {total:>12,} | {inscr:>12,} | {ratio:>10.4f} | {t:>7.1f}s")

    total_time = time.time() - total_start
    print("-" * 60)
    print(f"Total time: {total_time:.1f}s")

    # Summary
    print("\n\nSummary table (for copy/paste):")
    print("n\tTotal\tInscribable\tRatio")
    for n, total, inscr, ratio in results:
        print(f"{n}\t{total}\t{inscr}\t{ratio:.6f}")


if __name__ == '__main__':
    main()
