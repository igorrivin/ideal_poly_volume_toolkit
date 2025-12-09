#!/usr/bin/env python3
"""
Generate a benchmark for testing LLM geometric reasoning abilities.

This creates a "torture test" where we present 10 triangulations:
- 1 is Delaunay realizable (has a valid point set)
- 9 are NOT realizable (created via edge flips)

The challenge: Given only the triangulation, can the LLM:
1. Produce a valid point set with that combinatorial structure, OR
2. Correctly identify that no such point set exists?

We use pynauty for robust isomorphism checking to verify answers.
"""

import numpy as np
import json
import sys
from pathlib import Path
from scipy.spatial import Delaunay
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from ideal_poly_volume_toolkit.rivin_delaunay import (
    check_delaunay_realizability,
    random_edge_flips,
)


def triangulation_to_dict(triangles, label=""):
    """Convert triangulation to serializable dict."""
    return {
        'label': label,
        'n_vertices': int(len(set(v for tri in triangles for v in tri))),
        'n_triangles': int(len(triangles)),
        'triangles': [[int(v) for v in tri] for tri in triangles],  # Convert numpy int32 to Python int
    }


def generate_benchmark(n_points=150, n_flips=40, n_non_realizable=5, seed=42):
    """
    Generate LLM benchmark with 1 realizable + N non-realizable triangulations.

    Note: Finding valid non-realizable triangulations is challenging because most
    random edge flips either (a) create invalid triangulations (edges in >2 triangles)
    or (b) remain realizable. We aim for n_non_realizable but may get fewer.

    Args:
        n_points: Number of vertices
        n_flips: Number of edge flips per non-realizable triangulation
        n_non_realizable: Target number of non-realizable cases (may get fewer)
        seed: Random seed

    Returns:
        Dict with benchmark data
    """
    np.random.seed(seed)

    print("="*70)
    print("GENERATING LLM GEOMETRIC REASONING BENCHMARK")
    print("="*70)
    print()

    # Generate random points and compute Delaunay triangulation
    print(f"Step 1: Generate {n_points} random points")
    points = np.random.rand(n_points, 2)
    print(f"  ✓ Points generated")
    print()

    print("Step 2: Compute Delaunay triangulation")
    tri = Delaunay(points)
    realizable_triangulation = [tuple(simplex) for simplex in tri.simplices]
    print(f"  ✓ Triangulation: {len(realizable_triangulation)} triangles")
    print()

    # Verify it's realizable
    print("Step 3: Verify realizability")
    result = check_delaunay_realizability(realizable_triangulation, verbose=False)
    if not result['realizable']:
        print("  ✗ ERROR: Base triangulation not realizable (unexpected!)")
        return None
    print(f"  ✓ Confirmed realizable (min angle: {np.degrees(result['min_angle_radians']):.2f}°)")
    print()

    # Generate non-realizable triangulations via edge flips
    print(f"Step 4: Generate up to {n_non_realizable} non-realizable triangulations ({n_flips} flips each)")
    print(f"  (Using check_delaunay_realizability() to verify each is non-realizable)")
    print(f"  Note: Many edge flips create invalid triangulations or remain realizable")
    print()
    non_realizable_triangulations = []

    attempts = 0
    max_attempts = 1000  # Many attempts needed due to filtering

    while len(non_realizable_triangulations) < n_non_realizable and attempts < max_attempts:
        attempts += 1

        # Try different numbers of flips to get variety
        flips_to_try = n_flips + (attempts % 40) - 20  # Vary between n_flips-20 to n_flips+20
        flips_to_try = max(20, flips_to_try)

        flipped = random_edge_flips(
            realizable_triangulation,
            n_flips=flips_to_try,
            seed=seed + attempts
        )

        # IMPORTANT: First check it's a VALID triangulation (no edge in >2 triangles)
        from collections import Counter
        edge_count = Counter()
        for tri in flipped:
            v0, v1, v2 = tri
            for edge in [tuple(sorted([v0, v1])), tuple(sorted([v1, v2])), tuple(sorted([v2, v0]))]:
                edge_count[edge] += 1

        if any(count > 2 for count in edge_count.values()):
            # Invalid triangulation - skip it
            if attempts % 10 == 0:
                print(f"    (attempt {attempts}: invalid triangulation after flips, skipping...)")
            continue

        # IMPORTANT: Verify it's actually non-realizable using Rivin's LP test
        result = check_delaunay_realizability(flipped, verbose=False)

        if not result['realizable']:
            # Confirmed non-realizable AND valid triangulation!
            non_realizable_triangulations.append(flipped)
            lp_status = "infeasible" if result.get('success') and not result['realizable'] else result.get('message', 'unknown')
            print(f"  ✓ Non-realizable #{len(non_realizable_triangulations)}: "
                  f"{len(flipped)} triangles, {flips_to_try} flips, LP: {lp_status}")
        else:
            # Still realizable after flips - skip it
            if attempts % 10 == 0:
                print(f"    (attempt {attempts}: still realizable after {flips_to_try} flips, continuing...)")

    if len(non_realizable_triangulations) < n_non_realizable:
        print()
        print(f"  ⚠ Warning: Only found {len(non_realizable_triangulations)}/{n_non_realizable} non-realizable triangulations")
        print(f"     (This is expected - most edge flips create invalid or still-realizable triangulations)")
    print()

    # Package benchmark data
    print("Step 5: Package benchmark")

    challenges = []

    # Add the realizable triangulation
    # IMPORTANT: Save the points as a certificate/proof
    challenges.append({
        **triangulation_to_dict(realizable_triangulation, label="challenge_0"),
        'is_realizable': True,
        'solution_exists': True,
        'certificate_points': points.tolist(),  # The actual points that realize this triangulation
    })

    # Add non-realizable triangulations
    for i, tri in enumerate(non_realizable_triangulations):
        challenges.append({
            **triangulation_to_dict(tri, label=f"challenge_{i+1}"),
            'is_realizable': False,
            'solution_exists': False,
        })

    # Shuffle challenges so the realizable one isn't always first
    np.random.seed(seed + 999)
    indices = np.arange(len(challenges))
    np.random.shuffle(indices)

    challenges_shuffled = [challenges[i] for i in indices]

    benchmark = {
        'metadata': {
            'description': 'LLM Geometric Reasoning Benchmark',
            'n_vertices': n_points,
            'n_challenges': len(challenges_shuffled),
            'n_realizable': 1,
            'n_non_realizable': len(non_realizable_triangulations),
            'target_non_realizable': n_non_realizable,
            'note': 'Non-realizable count may be less than target due to edge flip constraints',
            'generated': datetime.now().isoformat(),
            'seed': seed,
        },
        'challenges': challenges_shuffled,
        'instructions': (
            "For each challenge, you are given a triangulation specified as a list of triangles. "
            "Each triangle is a tuple of three vertex indices. "
            "Your task: Either (1) produce a set of 2D points such that the Delaunay triangulation "
            "of those points has the same combinatorial structure as the given triangulation, "
            "OR (2) output 'None' if no such point set exists. "
            "Note: Vertex labels may permute - we check graph isomorphism using canonical forms."
        ),
    }

    print(f"  ✓ Created {len(challenges_shuffled)} challenges")
    print(f"  ✓ Challenges have been shuffled")
    print()

    return benchmark


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate LLM geometric reasoning benchmark"
    )
    parser.add_argument(
        "--points",
        type=int,
        default=150,
        help="Number of vertices (default: 150)",
    )
    parser.add_argument(
        "--flips",
        type=int,
        default=40,
        help="Number of edge flips for non-realizable triangulations (default: 40)",
    )
    parser.add_argument(
        "--non-realizable",
        type=int,
        default=5,
        help="Target number of non-realizable cases (default: 5, may get fewer)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="llm_benchmark.json",
        help="Output JSON file (default: llm_benchmark.json)",
    )

    args = parser.parse_args()

    print()
    print("#"*70)
    print("# LLM Geometric Reasoning Benchmark Generator")
    print("#"*70)
    print()

    benchmark = generate_benchmark(
        n_points=args.points,
        n_flips=args.flips,
        n_non_realizable=args.non_realizable,
        seed=args.seed
    )

    if benchmark is None:
        print("✗ Benchmark generation failed")
        return 1

    # Save to JSON
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(benchmark, f, indent=2)

    print("="*70)
    print("BENCHMARK GENERATED")
    print("="*70)
    print(f"Output file: {output_path}")
    print(f"Total challenges: {len(benchmark['challenges'])}")
    print(f"Realizable: {benchmark['metadata']['n_realizable']}")
    print(f"Non-realizable: {benchmark['metadata']['n_non_realizable']}")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
