#!/usr/bin/env python3
"""
Analyze Delaunay realizability using random sampling with PlanarMap.

This complements the exhaustive enumeration approach by using Gilles Schaeffer's
PlanarMap to generate uniform random samples of triangulations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/igor/devel/PlanarMap')

from planarmap_python import PlanarMapGenerator
from ideal_poly_volume_toolkit.rivin_delaunay import check_delaunay_realizability
import json
import time
import multiprocessing as mp
from collections import defaultdict


def test_chunk(args):
    """Test a chunk of triangulations (for parallel processing)."""
    from ideal_poly_volume_toolkit.rivin_delaunay import check_delaunay_realizability

    chunk, worker_id = args
    realizable = 0
    non_realizable = 0
    errors = 0

    for tri in chunk:
        try:
            result = check_delaunay_realizability(tri, verbose=False)
            if result['realizable']:
                realizable += 1
            else:
                non_realizable += 1
        except Exception as e:
            errors += 1

    return (realizable, non_realizable, errors)


def remove_vertex_to_planar(triangles: list, vertex_to_remove: int) -> list:
    """Remove a vertex to create planar triangulation (place vertex at infinity)."""
    return [tri for tri in triangles if vertex_to_remove not in tri]


def analyze_random_sampling(n_vertices: int,
                            n_samples: int,
                            n_workers: int = 30,
                            seed: int = 42,
                            remove_vertex: bool = True,
                            verbose: bool = True) -> dict:
    """
    Analyze Delaunay realizability using random sampling.

    Args:
        n_vertices: Number of vertices
        n_samples: Number of random samples to generate and test
        n_workers: Number of parallel workers
        seed: Random seed base
        remove_vertex: If True, remove a random vertex to create planar version
        verbose: Print progress

    Returns:
        Statistics dictionary
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Random Sampling: n={n_vertices} vertices, {n_samples} samples")
        if remove_vertex:
            print(f"  (Creating planar versions by removing random vertex)")
        print(f"{'='*70}")

    # Generate random triangulations
    gen = PlanarMapGenerator('/home/igor/devel/PlanarMap/planarmap')

    if verbose:
        print(f"Generating {n_samples} random closed triangulations...")

    start_time = time.time()
    triangulations = []

    import random as pyrandom
    pyrandom.seed(seed)

    for i in range(n_samples):
        sample_seed = seed + i
        try:
            triangles = gen.generate_random_triangulation(n_vertices, sample_seed)

            # Convert to planar by removing a random vertex (place it at infinity)
            if remove_vertex:
                # Remove a random vertex (in 0-indexed format)
                vertex_to_remove = pyrandom.randint(0, n_vertices - 1)
                triangles = remove_vertex_to_planar(triangles, vertex_to_remove)

                if not triangles:  # Skip if removal resulted in empty triangulation
                    continue

            triangulations.append(triangles)
        except Exception as e:
            if verbose and len(triangulations) < 5:
                print(f"  Warning: Failed to generate sample {i}: {e}")

    gen_time = time.time() - start_time

    if verbose:
        print(f"  Generated {len(triangulations)} triangulations in {gen_time:.2f}s")
        print(f"  Using {n_workers} parallel workers")

    # Split into chunks for parallel processing
    chunk_size = max(1, len(triangulations) // n_workers)
    chunks = []
    for i in range(0, len(triangulations), chunk_size):
        chunk = triangulations[i:i+chunk_size]
        chunks.append((chunk, i // chunk_size))

    if verbose:
        print(f"  Testing realizability...")

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
        'n_vertices': n_vertices,
        'n_samples': n_samples,
        'generated': len(triangulations),
        'tested': tested,
        'realizable': total_realizable,
        'non_realizable': total_non_realizable,
        'errors': total_errors,
        'fraction_realizable': fraction,
        'generation_time': gen_time,
        'testing_time': test_time,
        'n_workers': n_workers,
        'method': 'random_sampling',
    }

    if verbose:
        print(f"\n  Results:")
        print(f"    Realizable: {total_realizable}/{tested} ({100*fraction:.1f}%)")
        print(f"    Non-realizable: {total_non_realizable}/{tested} ({100*(1-fraction):.1f}%)")
        if total_errors > 0:
            print(f"    Errors: {total_errors}")
        print(f"    Total time: {gen_time + test_time:.2f}s")

    return stats


def main():
    """Run random sampling analysis."""
    import argparse

    parser = argparse.ArgumentParser(description='Random sampling Delaunay analysis')
    parser.add_argument('--min-n', type=int, default=10, help='Minimum vertices')
    parser.add_argument('--max-n', type=int, default=100, help='Maximum vertices')
    parser.add_argument('--samples', type=int, default=10000, help='Samples per n')
    parser.add_argument('--workers', type=int, default=30, help='Parallel workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed base')
    parser.add_argument('--output', type=str, default='delaunay_random_results.json',
                       help='Output file')

    args = parser.parse_args()

    print("="*70)
    print(f"RANDOM SAMPLING DELAUNAY ANALYSIS (via PlanarMap)")
    print("="*70)
    print(f"\nParameters:")
    print(f"  Vertex range: {args.min_n} to {args.max_n}")
    print(f"  Samples per n: {args.samples}")
    print(f"  Parallel workers: {args.workers}")

    # Run analysis
    results = []

    for n in range(args.min_n, args.max_n + 1):
        try:
            stats = analyze_random_sampling(
                n_vertices=n,
                n_samples=args.samples,
                n_workers=args.workers,
                seed=args.seed,
                verbose=True
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
        'method': 'random_sampling',
        'generator': 'PlanarMap (Gilles Schaeffer)',
        'parameters': {
            'min_n': args.min_n,
            'max_n': args.max_n,
            'samples_per_n': args.samples,
            'workers': args.workers,
            'seed': args.seed,
        },
        'results': results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY (Random Sampling)")
    print(f"{'='*70}")
    print(f"\n{'n':<6} {'Samples':<10} {'Realizable':<12} {'%Real':<8} {'Time(s)':<8}")
    print("-" * 70)
    for r in results:
        total_time = r['generation_time'] + r['testing_time']
        print(f"{r['n_vertices']:<6} {r['tested']:<10} {r['realizable']:<12} "
              f"{100*r['fraction_realizable']:>6.1f}%  {total_time:>7.1f}")

    print(f"\nResults saved to: {args.output}")
    print(f"\nNote: Uses uniform random sampling via PlanarMap")
    print(f"      (complements exhaustive enumeration from plantri)")


if __name__ == '__main__':
    main()
