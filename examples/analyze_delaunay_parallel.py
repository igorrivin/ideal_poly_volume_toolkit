#!/usr/bin/env python3
"""
Parallelized analysis of Delaunay realizability fraction.

Uses multiprocessing to test triangulations on multiple CPUs simultaneously.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ideal_poly_volume_toolkit.plantri_interface import find_plantri_executable
from ideal_poly_volume_toolkit.rivin_delaunay import check_delaunay_realizability
from ideal_poly_volume_toolkit.planar_utils import extract_faces_from_planar_embedding
import subprocess
import json
import time
import multiprocessing as mp
from collections import defaultdict
from typing import Optional, List, Tuple


def get_triangulations_text(n_vertices: int, min_connectivity: int = 3) -> list:
    """
    Generate triangulations in ASCII format.

    Args:
        n_vertices: Number of vertices
        min_connectivity: Minimum connectivity (3 or 4)

    Returns:
        List of triangulations as adjacency dicts
    """
    plantri = find_plantri_executable()
    if plantri is None:
        raise RuntimeError("plantri not found")

    # Use -p for polytopes, -c# for connectivity
    args = [plantri, f'-pc{min_connectivity}', '-a', str(n_vertices)]

    result = subprocess.run(args, capture_output=True, text=True)

    # Parse ASCII format: "n adj_list"
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

        # Extract faces from planar embedding (adjacency lists are in cyclic order)
        triangles = extract_faces_from_planar_embedding(n, adj)

        if triangles:
            triangulations.append(triangles)

    return triangulations


def remove_vertex_to_planar(triangles: list, vertex_to_remove: int) -> list:
    """Remove a vertex to create planar triangulation."""
    return [tri for tri in triangles if vertex_to_remove not in tri]


def test_chunk(args):
    """Test a chunk of triangulations (for parallel processing)."""
    # Import here to ensure each worker has the module
    from ideal_poly_volume_toolkit.rivin_delaunay import check_delaunay_realizability

    chunk, worker_id, strict = args
    realizable = 0
    non_realizable = 0
    errors = 0

    for tri in chunk:
        try:
            result = check_delaunay_realizability(tri, verbose=False, strict=strict)
            if result['realizable']:
                realizable += 1
            else:
                non_realizable += 1
        except Exception as e:
            errors += 1

    return (realizable, non_realizable, errors)


def count_triangulations(n: int, min_connectivity: int = 3) -> int:
    """Count closed triangulations (cached for known values)."""
    # OEIS A000109 (3-connected) and A000108 (4-connected)
    counts_3conn = {4: 1, 5: 1, 6: 7, 7: 34, 8: 257, 9: 2606, 10: 32300, 11: 440564, 12: 6384634}
    counts_4conn = {4: 1, 5: 0, 6: 1, 7: 1, 8: 8, 9: 36, 10: 257, 11: 1734, 12: 13391}

    if min_connectivity == 3 and n in counts_3conn:
        return counts_3conn[n]
    elif min_connectivity == 4 and n in counts_4conn:
        return counts_4conn[n]

    # Compute by generating and counting
    tris = get_triangulations_text(n, min_connectivity)
    return len(tris)


def analyze_single_n_parallel(n: int,
                              n_workers: int = 30,
                              min_connectivity: int = 3,
                              verbose: bool = True,
                              strict: bool = False) -> dict:
    """
    Analyze triangulations with n vertices using parallel processing.

    Args:
        n: Number of vertices
        n_workers: Number of parallel workers
        min_connectivity: Minimum connectivity (3 or 4)
        verbose: Print progress
        strict: Use strict realizability (dihedral < π)

    Returns:
        Statistics dictionary
    """
    mode_str = "strict" if strict else "standard"
    if verbose:
        print(f"\n{'='*70}")
        print(f"Analyzing n={n} vertices ({min_connectivity}-connected, {mode_str} mode)")
        print(f"{'='*70}")

    # Count and generate closed triangulations
    start_time = time.time()
    closed_tris = get_triangulations_text(n, min_connectivity)
    gen_time = time.time() - start_time

    n_closed = len(closed_tris)

    if verbose:
        print(f"Generated {n_closed} closed triangulations in {gen_time:.2f}s")
        print(f"Creating {n_closed} planar versions (remove vertex 0)")

    # Convert to planar triangulations (only remove vertex 0)
    planar_tris = []
    for closed_tri in closed_tris:
        planar_tri = remove_vertex_to_planar(closed_tri, 0)
        if planar_tri:
            planar_tris.append(planar_tri)

    n_planar = len(planar_tris)

    if verbose:
        print(f"Total planar triangulations to test: {n_planar}")
        print(f"Using {n_workers} parallel workers")

    # Split into chunks for parallel processing
    chunk_size = max(1, len(planar_tris) // n_workers)
    chunks = []
    for i in range(0, len(planar_tris), chunk_size):
        chunk = planar_tris[i:i+chunk_size]
        chunks.append((chunk, i // chunk_size, strict))

    # Parallel testing
    if verbose:
        print(f"Split into {len(chunks)} chunks of ~{chunk_size} triangulations each")
        print(f"Testing realizability...")

    start_time = time.time()

    with mp.Pool(n_workers) as pool:
        results = pool.map(test_chunk, chunks)

    test_time = time.time() - start_time

    # Aggregate results
    total_realizable = sum(r[0] for r in results)
    total_non_realizable = sum(r[1] for r in results)
    total_errors = sum(r[2] for r in results)
    tested = total_realizable + total_non_realizable

    fraction = total_realizable / tested if tested > 0 else 0.0

    stats = {
        'n_vertices': n,
        'min_connectivity': min_connectivity,
        'closed_triangulations': n_closed,
        'total_planar_triangulations': n_planar,
        'tested': tested,
        'realizable': total_realizable,
        'non_realizable': total_non_realizable,
        'errors': total_errors,
        'fraction_realizable': fraction,
        'generation_time': gen_time,
        'testing_time': test_time,
        'n_workers': n_workers,
        'strict_mode': strict,
    }

    if verbose:
        print(f"\nResults:")
        print(f"  Tested: {tested}/{n_planar}")
        print(f"  Realizable: {total_realizable} ({100*fraction:.1f}%)")
        print(f"  Non-realizable: {total_non_realizable} ({100*(1-fraction):.1f}%)")
        if total_errors > 0:
            print(f"  Errors: {total_errors}")
        if tested > 0:
            print(f"  Testing time: {test_time:.2f}s ({test_time/tested*1000:.2f}ms per triangulation)")
            print(f"  Speedup: {n_workers * test_time / tested * 1000:.2f}ms sequential equivalent")
        else:
            print(f"  Testing time: {test_time:.2f}s (no successful tests)")

    return stats


def main():
    """Run parallelized analysis."""
    import argparse

    parser = argparse.ArgumentParser(description='Parallel Delaunay realizability analysis')
    parser.add_argument('--min-n', type=int, default=4, help='Minimum vertices')
    parser.add_argument('--max-n', type=int, default=15, help='Maximum vertices')
    parser.add_argument('--workers', type=int, default=30, help='Number of parallel workers')
    parser.add_argument('--connectivity', type=int, default=3, choices=[3, 4],
                       help='Minimum connectivity (3 or 4)')
    parser.add_argument('--time-limit', type=float, default=7200,
                       help='Wall-clock time limit in seconds (default: 2 hours)')
    parser.add_argument('--output', type=str, default='delaunay_parallel_results.json',
                       help='Output file')
    parser.add_argument('--strict', action='store_true',
                       help='Use strict realizability mode (dihedral < π)')

    args = parser.parse_args()

    mode_desc = "STRICT (dihedral < π)" if args.strict else "STANDARD (dihedral ≤ π)"
    print("="*70)
    print(f"PARALLEL DELAUNAY REALIZABILITY ANALYSIS")
    print("="*70)
    print(f"\nParameters:")
    print(f"  Vertex range: {args.min_n} to {args.max_n}")
    print(f"  Parallel workers: {args.workers}")
    print(f"  Min connectivity: {args.connectivity}")
    print(f"  Mode: {mode_desc}")
    print(f"  Time limit: {args.time_limit/3600:.1f} hours")

    # Run analysis
    results = []
    start_time = time.time()

    for n in range(args.min_n, args.max_n + 1):
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed > args.time_limit:
            print(f"\n⏱ Time limit reached ({elapsed/3600:.2f} hours)")
            break

        try:
            # Estimate if we have enough time
            if results:
                avg_time = sum(r['generation_time'] + r['testing_time'] for r in results) / len(results)
                if elapsed + avg_time * 2 > args.time_limit:
                    print(f"\n⏱ Approaching time limit, stopping at n={n-1}")
                    break

            stats = analyze_single_n_parallel(
                n,
                n_workers=args.workers,
                min_connectivity=args.connectivity,
                verbose=True,
                strict=args.strict
            )
            results.append(stats)

        except KeyboardInterrupt:
            print(f"\n\n⚠ Interrupted by user at n={n}")
            break
        except Exception as e:
            print(f"\n✗ Error at n={n}: {e}")
            import traceback
            traceback.print_exc()
            break

    # Save results
    output_data = {
        'parameters': {
            'min_n': args.min_n,
            'max_n': args.max_n,
            'workers': args.workers,
            'min_connectivity': args.connectivity,
            'strict_mode': args.strict,
            'time_limit': args.time_limit,
            'actual_runtime': time.time() - start_time,
        },
        'results': results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'n':<4} {'Closed':<8} {'Planar':<8} {'Realizable':<12} {'%Real':<8} {'Time(s)':<8}")
    print("-" * 70)
    for r in results:
        total_time = r['generation_time'] + r['testing_time']
        print(f"{r['n_vertices']:<4} {r['closed_triangulations']:<8} "
              f"{r['total_planar_triangulations']:<8} {r['realizable']:<12} "
              f"{100*r['fraction_realizable']:>6.1f}%  {total_time:>7.1f}")

    print(f"\nTotal runtime: {(time.time() - start_time)/3600:.2f} hours")
    print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
