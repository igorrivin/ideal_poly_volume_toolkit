#!/usr/bin/env python3
"""
Demonstrate that random edge flips break Delaunay realizability.

This script shows the elegant way to find non-realizable triangulations:
1. Start with a large random point set (e.g., 150 points)
2. Compute the Delaunay triangulation (which is realizable)
3. Perform random edge flips (e.g., 10 flips)
4. Check realizability - with high probability, it's now NON-realizable!

This demonstrates that the Rivin algorithm correctly detects when a triangulation
violates the Delaunay edge conditions.
"""

import numpy as np
from scipy.spatial import Delaunay
import sys
sys.path.insert(0, '/home/igor/devel/ideal_poly_volume_toolkit')

from ideal_poly_volume_toolkit.rivin_delaunay import (
    check_delaunay_realizability,
    random_edge_flips,
    format_realizability_report,
)


def test_edge_flips_break_realizability(n_points=150, n_flips=10, seed=42):
    """
    Demonstrate that random edge flips typically break Delaunay realizability.

    Args:
        n_points: Number of random points to generate
        n_flips: Number of random edge flips to perform
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    print("="*70)
    print("EDGE FLIPS BREAKING DELAUNAY REALIZABILITY")
    print("="*70)
    print()
    print(f"Step 1: Generate {n_points} random points")

    # Generate random points
    points = np.random.rand(n_points, 2)
    print(f"  ✓ Generated {n_points} points in unit square")
    print()

    print("Step 2: Compute Delaunay triangulation")
    # Compute Delaunay triangulation
    tri = Delaunay(points)
    original_triangles = [tuple(simplex) for simplex in tri.simplices]
    print(f"  ✓ Delaunay triangulation: {len(original_triangles)} triangles")
    print()

    print("Step 3: Check that original triangulation IS realizable")
    # Check original (should be realizable)
    result_original = check_delaunay_realizability(original_triangles, verbose=False)

    if result_original['realizable']:
        print(f"  ✓ Original Delaunay triangulation IS realizable")
        print(f"    Min angle: {np.degrees(result_original['min_angle_radians']):.2f}°")
    else:
        print(f"  ✗ UNEXPECTED: Original Delaunay triangulation NOT realizable!")
        print(f"    This should not happen - Delaunay triangulations are always realizable")
    print()

    print(f"Step 4: Perform {n_flips} random edge flips")
    # Perform random edge flips
    flipped_triangles = random_edge_flips(original_triangles, n_flips=n_flips, seed=seed)
    print(f"  ✓ Performed {n_flips} random edge flips")
    print(f"    New triangulation: {len(flipped_triangles)} triangles")
    print()

    print("Step 5: Check if flipped triangulation is realizable")
    # Check flipped (should be non-realizable with high probability)
    result_flipped = check_delaunay_realizability(flipped_triangles, verbose=False)

    print()
    if not result_flipped['realizable']:
        print("  ✓ SUCCESS: Flipped triangulation is NOT realizable!")
        print(f"    The edge flips violated the Delaunay edge conditions")
        if result_flipped['success']:
            print(f"    LP solver found: min_angle = {result_flipped['min_angle']:.6f} (≤ 0)")
        else:
            print(f"    LP solver status: {result_flipped['message']}")
    else:
        print(f"  ✗ Flipped triangulation is STILL realizable")
        print(f"    Min angle: {np.degrees(result_flipped['min_angle_radians']):.2f}°")
        print(f"    Note: This can happen with small n_flips - try more flips!")

    print()
    print("="*70)
    print("DETAILED REPORT FOR FLIPPED TRIANGULATION")
    print("="*70)
    print(format_realizability_report(result_flipped))

    return result_original['realizable'], not result_flipped['realizable']


def test_multiple_trials(n_trials=10, n_points=150, n_flips=10):
    """
    Run multiple trials to see how often edge flips break realizability.

    Args:
        n_trials: Number of trials to run
        n_points: Number of points per trial
        n_flips: Number of edge flips per trial
    """
    print("\n" + "="*70)
    print(f"RUNNING {n_trials} TRIALS")
    print("="*70)
    print(f"Each trial: {n_points} points, {n_flips} random edge flips")
    print()

    non_realizable_count = 0

    for trial in range(n_trials):
        np.random.seed(trial)

        # Generate and triangulate
        points = np.random.rand(n_points, 2)
        tri = Delaunay(points)
        original_triangles = [tuple(simplex) for simplex in tri.simplices]

        # Flip
        flipped_triangles = random_edge_flips(original_triangles, n_flips=n_flips, seed=trial)

        # Check
        result = check_delaunay_realizability(flipped_triangles, verbose=False)

        is_realizable = result['realizable']
        status = "realizable  " if is_realizable else "NON-realizable"
        symbol = "✗" if is_realizable else "✓"

        print(f"  Trial {trial+1:2d}: {symbol} {status}")

        if not is_realizable:
            non_realizable_count += 1

    print()
    print("="*70)
    print(f"Results: {non_realizable_count}/{n_trials} trials produced NON-realizable triangulations")
    print(f"Success rate: {100*non_realizable_count/n_trials:.1f}%")

    if non_realizable_count >= n_trials * 0.8:
        print("✓ As expected: Random edge flips break Delaunay realizability with high probability!")
    elif non_realizable_count > 0:
        print(f"⚠ Only {non_realizable_count} successes - consider increasing n_flips")
    else:
        print("✗ No non-realizable triangulations found - increase n_flips!")
    print("="*70)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test that random edge flips break Delaunay realizability"
    )
    parser.add_argument(
        "--points",
        type=int,
        default=150,
        help="Number of random points (default: 150)",
    )
    parser.add_argument(
        "--flips",
        type=int,
        default=10,
        help="Number of random edge flips (default: 10)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trials to run (default: 1, use >1 for statistics)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    print("\n" + "#"*70)
    print("# Testing: Edge Flips Break Delaunay Realizability")
    print("#"*70 + "\n")

    if args.trials == 1:
        # Single detailed test
        original_ok, flipped_broken = test_edge_flips_break_realizability(
            n_points=args.points,
            n_flips=args.flips,
            seed=args.seed
        )

        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Original Delaunay: {'✓ Realizable' if original_ok else '✗ NOT realizable'}")
        print(f"After {args.flips} flips:  {'✓ NON-realizable' if flipped_broken else '✗ Still realizable'}")
        print("="*70)

    else:
        # Multiple trials for statistics
        test_multiple_trials(
            n_trials=args.trials,
            n_points=args.points,
            n_flips=args.flips
        )


if __name__ == "__main__":
    main()
