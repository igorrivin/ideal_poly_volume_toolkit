#!/usr/bin/env python3
"""
Analyze what fraction of planar triangulations are Delaunay realizable.

Strategy:
- Small n (4-10): Enumerate all triangulations, test each
- Medium n (11-15): Count total, sample randomly
- Large n (16+): Random sampling only

Research question: As n increases, what fraction are Delaunay realizable?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ideal_poly_volume_toolkit.plantri_interface import (
    get_triangulation_count,
    enumerate_triangulations,
    find_plantri_executable
)
from ideal_poly_volume_toolkit.rivin_delaunay import check_delaunay_realizability
import subprocess
import tempfile
import json
from collections import defaultdict
import time


def parse_planar_code(data: bytes) -> list:
    """
    Parse plantri's planar_code format.

    Format: Binary format with adjacency lists.
    Header: >>planar_code<<
    Each graph: [n_vertices] [adj list for v0] [0] [adj list for v1] [0] ...

    Returns:
        List of graphs, each as adjacency dict
    """
    # This is complex - for now, let's use plantri's text output instead
    raise NotImplementedError("Binary planar_code parsing not yet implemented")


def get_triangulations_as_text(n_vertices: int, only_polytopes: bool = True) -> list:
    """
    Generate triangulations in readable format.

    Use plantri with -a flag for ASCII output (adjacency lists).

    Returns:
        List of triangulations, each as list of triangles
    """
    plantri = find_plantri_executable()
    if plantri is None:
        raise RuntimeError("plantri not found")

    # Use -a for ASCII adjacency list format
    args = [plantri]
    if only_polytopes:
        args.append('-p')
    args.extend(['-a', str(n_vertices)])

    result = subprocess.run(args, capture_output=True, text=True)

    # Parse ASCII format
    # Format: n adj_list
    # Where adj_list is like "bcd,adc,abd,acb" (letters a,b,c,... for vertices)
    triangulations = []

    for line in result.stdout.split('\n'):
        line = line.strip()
        if not line or line.startswith('>'):
            continue

        # Parse: "4 bcd,adc,abd,acb"
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue

        n = int(parts[0])
        adj_str = parts[1]

        # Build adjacency dict
        # Vertices are a, b, c, ... (convert to 0, 1, 2, ...)
        adj = {}
        vertex_lists = adj_str.split(',')

        for v_idx, neighbor_str in enumerate(vertex_lists):
            neighbors = []
            for letter in neighbor_str:
                neighbor_idx = ord(letter) - ord('a')
                neighbors.append(neighbor_idx)
            adj[v_idx] = neighbors

        # Convert adjacency to triangles (find all 3-cycles)
        triangles = []
        for v0 in range(n):
            for v1 in adj[v0]:
                if v1 <= v0:  # Avoid duplicates
                    continue
                for v2 in adj[v1]:
                    if v2 <= v1:
                        continue
                    if v2 in adj[v0]:
                        # Found triangle
                        tri = tuple(sorted([v0, v1, v2]))
                        if tri not in triangles:
                            triangles.append(tri)

        if triangles:
            triangulations.append(triangles)

    return triangulations


def remove_vertex_to_planar(triangles: list, vertex_to_remove: int) -> list:
    """
    Remove a vertex from a closed triangulation to create a planar triangulation.

    This simulates placing the vertex at infinity, leaving a planar triangulation
    with a boundary.

    Args:
        triangles: List of triangles (closed triangulation)
        vertex_to_remove: Vertex to remove (place at infinity)

    Returns:
        List of triangles not containing the removed vertex
    """
    planar_triangles = []
    for tri in triangles:
        if vertex_to_remove not in tri:
            planar_triangles.append(tri)
    return planar_triangles


def analyze_single_n(n: int, max_test: int = None, verbose: bool = True) -> dict:
    """
    Analyze all (or sample of) triangulations with n vertices.

    Args:
        n: Number of vertices
        max_test: Maximum number to test (None = test all)
        verbose: Print progress

    Returns:
        Dictionary with statistics
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Analyzing n={n} vertices")
        print(f"{'='*70}")

    # Get count
    total_count = get_triangulation_count(n, use_cache=True)
    if verbose:
        print(f"Total closed triangulations: {total_count}")
        print(f"Each gives {n} planar versions (remove each vertex)")
        print(f"Total planar triangulations to test: {total_count * n}")

    # Decide strategy
    if total_count <= 1000 or max_test is None:
        # Enumerate all
        test_count = total_count
        if verbose:
            print(f"Strategy: Test all closed triangulations")
    else:
        # Sample
        test_count = min(max_test or 1000, total_count)
        if verbose:
            print(f"Strategy: Random sample of {test_count}/{total_count} closed triangulations")

    # Get triangulations
    if verbose:
        print(f"\nGenerating closed triangulations...")

    start_time = time.time()
    closed_triangulations = get_triangulations_as_text(n, only_polytopes=True)
    gen_time = time.time() - start_time

    if verbose:
        print(f"  Generated {len(closed_triangulations)} closed triangulations in {gen_time:.2f}s")

    # If we need to sample, shuffle and take subset
    if test_count < len(closed_triangulations):
        import random
        random.seed(42)
        closed_triangulations = random.sample(closed_triangulations, test_count)

    # Convert to planar triangulations by removing each vertex
    triangulations = []
    for closed_tri in closed_triangulations:
        for v in range(n):
            planar_tri = remove_vertex_to_planar(closed_tri, v)
            if planar_tri:  # Only add if non-empty
                triangulations.append(planar_tri)

    if verbose:
        print(f"  Created {len(triangulations)} planar triangulations ({len(closed_triangulations)} × {n})")

    # Test each for Delaunay realizability
    if verbose:
        print(f"\nTesting realizability...")

    realizable_count = 0
    non_realizable_count = 0
    error_count = 0

    start_time = time.time()
    for i, triangles in enumerate(triangulations):
        if verbose and (i + 1) % max(1, len(triangulations) // 10) == 0:
            print(f"  Progress: {i+1}/{len(triangulations)} ({100*(i+1)/len(triangulations):.1f}%)")

        try:
            result = check_delaunay_realizability(triangles, verbose=False)
            if result['realizable']:
                realizable_count += 1
            else:
                non_realizable_count += 1
        except Exception as e:
            error_count += 1
            if verbose and error_count <= 3:
                print(f"  Error testing triangulation {i}: {e}")

    test_time = time.time() - start_time

    # Compute statistics
    tested = realizable_count + non_realizable_count
    fraction_realizable = realizable_count / tested if tested > 0 else 0.0
    total_planar = total_count * n  # Each closed triangulation gives n planar versions

    stats = {
        'n_vertices': n,
        'closed_triangulations': total_count,
        'total_planar_triangulations': total_planar,
        'tested': tested,
        'realizable': realizable_count,
        'non_realizable': non_realizable_count,
        'errors': error_count,
        'fraction_realizable': fraction_realizable,
        'generation_time': gen_time,
        'testing_time': test_time,
    }

    if verbose:
        print(f"\nResults:")
        print(f"  Tested: {tested}/{total_planar} planar triangulations")
        print(f"  Realizable: {realizable_count} ({100*fraction_realizable:.1f}%)")
        print(f"  Non-realizable: {non_realizable_count} ({100*(1-fraction_realizable):.1f}%)")
        if error_count > 0:
            print(f"  Errors: {error_count}")
        print(f"  Testing time: {test_time:.2f}s ({test_time/tested*1000:.1f}ms per triangulation)")

    return stats


def main():
    """Run the analysis."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze Delaunay realizability of planar triangulations')
    parser.add_argument('--min-n', type=int, default=4, help='Minimum number of vertices')
    parser.add_argument('--max-n', type=int, default=10, help='Maximum number of vertices')
    parser.add_argument('--max-test', type=int, default=1000, help='Max triangulations to test per n')
    parser.add_argument('--output', type=str, default='delaunay_fraction_analysis.json', help='Output file')

    args = parser.parse_args()

    print("="*70)
    print("DELAUNAY REALIZABILITY ANALYSIS")
    print("="*70)
    print(f"\nParameters:")
    print(f"  Vertex range: {args.min_n} to {args.max_n}")
    print(f"  Max test per n: {args.max_test}")

    # Run analysis for each n
    results = []
    for n in range(args.min_n, args.max_n + 1):
        try:
            stats = analyze_single_n(n, max_test=args.max_test, verbose=True)
            results.append(stats)
        except Exception as e:
            print(f"\n✗ Error analyzing n={n}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_data = {
        'parameters': {
            'min_n': args.min_n,
            'max_n': args.max_n,
            'max_test': args.max_test,
        },
        'results': results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'n':<4} {'Closed':<8} {'Planar':<8} {'Tested':<8} {'Realizable':<12} {'%Real':<8}")
    print("-" * 70)
    for r in results:
        print(f"{r['n_vertices']:<4} {r['closed_triangulations']:<8} "
              f"{r['total_planar_triangulations']:<8} {r['tested']:<8} "
              f"{r['realizable']:<12} {100*r['fraction_realizable']:>6.1f}%")

    print(f"\nResults saved to: {args.output}")
    print(f"\nNote: 'Closed' = 3-connected polytopes, 'Planar' = with one vertex removed")


if __name__ == '__main__':
    main()
