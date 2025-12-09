#!/usr/bin/env python3
"""
Enumerate distinct combinatorial types of convex polyhedra inscribed in the sphere.

This script generates random point sets on the unit sphere and counts how many
distinct combinatorial types exist for various vertex counts. The lower bounds
on the number of combinatorial types provide insight into the combinatorial
complexity of polyhedra.

Usage:
    python enumerate_types.py
    python enumerate_types.py --vertices 4,5,6,7,8 --samples 10000
    python enumerate_types.py --vertices 10 --samples 100000 --seed 42
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np

from ideal_poly_volume_toolkit.combinatorial_enumeration import (
    enumerate_combinatorial_types,
    enumerate_combinatorial_types_parallel,
    format_enumeration_report,
)
from multiprocessing import cpu_count


def main():
    parser = argparse.ArgumentParser(
        description="Enumerate combinatorial types of convex polyhedra on the sphere"
    )
    parser.add_argument(
        "--vertices",
        type=str,
        default="4,5,6,7,8,9,10",
        help="Comma-separated list of vertex counts to test (default: 4,5,6,7,8,9,10)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of random samples per vertex count (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file to save results (default: None)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing (recommended for large sample sizes)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of parallel workers (default: {cpu_count()} CPUs)",
    )

    args = parser.parse_args()

    # Parse vertex counts
    vertex_counts = [int(v.strip()) for v in args.vertices.split(",")]

    print("="*70)
    print("COMBINATORIAL TYPE ENUMERATION FOR CONVEX POLYHEDRA ON THE SPHERE")
    print("="*70)
    print(f"Vertex counts: {vertex_counts}")
    print(f"Samples per count: {args.samples:,}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    if args.parallel:
        n_workers = args.workers if args.workers else cpu_count()
        print(f"Parallel mode: {n_workers} workers")
    else:
        print("Serial mode (use --parallel for multiprocessing)")
    print()

    # Store all results
    all_results = {}
    summary = []

    for n_vertices in vertex_counts:
        print(f"\n{'='*70}")
        print(f"Enumerating types for {n_vertices} vertices...")
        print(f"{'='*70}")

        start_time = time.time()

        # Run enumeration (serial or parallel)
        if args.parallel:
            result = enumerate_combinatorial_types_parallel(
                n_vertices=n_vertices,
                n_samples=args.samples,
                seed=args.seed,
                n_workers=args.workers,
                verbose=not args.quiet,
            )
        else:
            result = enumerate_combinatorial_types(
                n_vertices=n_vertices,
                n_samples=args.samples,
                seed=args.seed,
                verbose=not args.quiet,
            )

        elapsed = time.time() - start_time

        # Print report
        print(format_enumeration_report(result))
        print(f"\nTime elapsed: {elapsed:.2f} seconds")
        print(f"Samples per second: {args.samples/elapsed:.0f}")

        # Store result (without vertex examples to save space)
        result_summary = {
            'n_vertices': result['n_vertices'],
            'n_samples': result['n_samples'],
            'n_types': result['n_types'],
            'n_failures': result['n_failures'],
            'time_seconds': elapsed,
            'samples_per_second': args.samples / elapsed,
        }

        # Add frequency distribution
        sorted_types = sorted(result['type_counts'].items(), key=lambda x: x[1], reverse=True)
        result_summary['top_types'] = [
            {
                'rank': i + 1,
                'degree_sequence': list(type_hash[1]),  # type_hash = (n_verts, degrees, edges)
                'count': count,
                'frequency': count / args.samples,
            }
            for i, (type_hash, count) in enumerate(sorted_types[:20])  # Top 20
        ]

        all_results[n_vertices] = result_summary
        summary.append((n_vertices, result['n_types'], result['n_failures']))

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Vertices':<12} {'Types Found':<15} {'Degenerate':<12} {'Lower Bound'}")
    print("-"*70)

    for n_vertices, n_types, n_failures in summary:
        print(f"{n_vertices:<12} {n_types:<15} {n_failures:<12} ≥ {n_types}")

    print("="*70)
    print(f"\nTotal samples: {args.samples * len(vertex_counts):,}")
    print(f"Total runtime: {sum(r['time_seconds'] for r in all_results.values()):.2f} seconds")

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            'metadata': {
                'vertex_counts': vertex_counts,
                'samples_per_count': args.samples,
                'seed': args.seed,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            },
            'results': all_results,
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    print("\nNote: These are LOWER BOUNDS on the number of combinatorial types.")
    print("More samples may reveal additional rare types.")


if __name__ == "__main__":
    main()
