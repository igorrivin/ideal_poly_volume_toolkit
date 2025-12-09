#!/usr/bin/env python3
"""
Test hyperbolic volume optimization using Rivin's algorithm.

This script demonstrates:
1. Check that a triangulation is Delaunay realizable
2. Optimize the angle assignment to maximize hyperbolic volume
3. The optimization uses exact gradient and Hessian (diagonal!)
4. Rivin's theorem guarantees concavity, so there's a unique maximum
"""

import numpy as np
from scipy.spatial import Delaunay
import sys
sys.path.insert(0, '/home/igor/devel/ideal_poly_volume_toolkit')

from ideal_poly_volume_toolkit.rivin_delaunay import (
    check_delaunay_realizability,
    optimize_hyperbolic_volume,
    format_realizability_report,
)


def test_volume_optimization(n_points=20, seed=42):
    """
    Test volume optimization on a random Delaunay triangulation.

    Args:
        n_points: Number of random points
        seed: Random seed
    """
    np.random.seed(seed)

    print("="*70)
    print("HYPERBOLIC VOLUME OPTIMIZATION VIA RIVIN'S ALGORITHM")
    print("="*70)
    print()

    print(f"Step 1: Generate {n_points} random points and compute Delaunay triangulation")
    points = np.random.rand(n_points, 2)
    tri = Delaunay(points)
    triangles = [tuple(simplex) for simplex in tri.simplices]
    print(f"  ✓ Triangulation: {len(triangles)} triangles, {n_points} vertices")
    print()

    print("Step 2: Check Delaunay realizability and get initial angle assignment")
    result_realizable = check_delaunay_realizability(triangles, verbose=True)
    print()

    if not result_realizable['realizable']:
        print("✗ Triangulation is not realizable - cannot optimize volume")
        return

    print(format_realizability_report(result_realizable))
    print()

    print("Step 3: Optimize hyperbolic volume")
    print("  Using quasi-Newton method with:")
    print("    - Exact gradient: ∇Λ(θ) = -log(2sin(θ))")
    print("    - Exact Hessian: ∇²Λ(θ) = -cot(θ) (diagonal!)")
    print("    - Constraints: Rivin polytope (linear constraints)")
    print()

    result_opt = optimize_hyperbolic_volume(triangles, verbose=True)

    if not result_opt['success']:
        print(f"\n✗ Optimization failed: {result_opt['message']}")
        return

    print()
    print("="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Success: ✓")
    print(f"Iterations: {result_opt['n_iterations']}")
    print(f"Optimal hyperbolic volume: {result_opt['volume']:.6f}")
    print()

    # Compare initial vs optimal angles
    initial_angles = result_realizable['angles_radians']
    optimal_angles = result_opt['angles']

    # Compute initial volume
    from ideal_poly_volume_toolkit.rivin_delaunay import lobachevsky_function
    initial_volume = np.sum([lobachevsky_function(theta) for theta in initial_angles.flatten()])

    print(f"Initial volume (from LP):     {initial_volume:.6f}")
    print(f"Optimal volume (optimized):   {result_opt['volume']:.6f}")
    print(f"Improvement:                  {result_opt['volume'] - initial_volume:.6f}")
    print(f"Relative improvement:         {100*(result_opt['volume'] - initial_volume)/abs(initial_volume):.2f}%")
    print()

    # Angle statistics
    print("Angle statistics (radians):")
    print(f"  Initial - min: {initial_angles.min():.4f}, max: {initial_angles.max():.4f}, mean: {initial_angles.mean():.4f}")
    print(f"  Optimal - min: {optimal_angles.min():.4f}, max: {optimal_angles.max():.4f}, mean: {optimal_angles.mean():.4f}")
    print()

    print("Angle statistics (degrees):")
    print(f"  Initial - min: {np.degrees(initial_angles.min()):.2f}°, max: {np.degrees(initial_angles.max()):.2f}°")
    print(f"  Optimal - min: {np.degrees(optimal_angles.min()):.2f}°, max: {np.degrees(optimal_angles.max()):.2f}°")
    print("="*70)

    return result_opt


def test_hexagon_optimization():
    """
    Test on a simple example: regular hexagon with center.
    """
    print("\n" + "="*70)
    print("SPECIAL CASE: Regular Hexagon Triangulated from Center")
    print("="*70)
    print()

    # Regular hexagon with center
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    center = np.array([[0.0, 0.0]])
    all_points = np.vstack([center, points])

    # Triangulate from center
    triangles = [(0, i+1, ((i+1) % 6) + 1) for i in range(6)]

    print("Configuration: 6 triangles, all sharing the center vertex")
    print()

    print("Step 1: Check realizability")
    result_realizable = check_delaunay_realizability(triangles, verbose=False)

    if not result_realizable['realizable']:
        print("✗ Not realizable")
        return

    print(f"  ✓ Realizable with min angle: {np.degrees(result_realizable['min_angle_radians']):.2f}°")
    print()

    print("Step 2: Optimize volume")
    result_opt = optimize_hyperbolic_volume(triangles, verbose=False)

    if not result_opt['success']:
        print(f"  ✗ Optimization failed")
        return

    from ideal_poly_volume_toolkit.rivin_delaunay import lobachevsky_function
    initial_volume = np.sum([lobachevsky_function(theta)
                            for theta in result_realizable['angles_radians'].flatten()])

    print(f"  ✓ Optimization successful")
    print(f"    Iterations: {result_opt['n_iterations']}")
    print(f"    Initial volume: {initial_volume:.6f}")
    print(f"    Optimal volume: {result_opt['volume']:.6f}")
    print(f"    Improvement: {100*(result_opt['volume']-initial_volume)/abs(initial_volume):.2f}%")
    print()

    # Check if angles are equal (by symmetry, they should be for optimal volume)
    optimal_angles = result_opt['angles']
    print("Optimal angle distribution:")
    print(f"  All angles: {np.degrees(optimal_angles.flatten())}")
    print(f"  Std dev: {np.degrees(optimal_angles.std()):.4f}°")
    print()

    if optimal_angles.std() < 0.01:
        print("  ✓ All angles are approximately equal (as expected by symmetry!)")
    else:
        print("  ⚠ Angles vary (unexpected for this symmetric configuration)")

    print("="*70)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test hyperbolic volume optimization"
    )
    parser.add_argument(
        "--points",
        type=int,
        default=20,
        help="Number of random points (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--example",
        choices=["random", "hexagon", "both"],
        default="random",
        help="Which example to run (default: random)",
    )

    args = parser.parse_args()

    print("\n" + "#"*70)
    print("# Testing: Hyperbolic Volume Optimization")
    print("#"*70 + "\n")

    if args.example in ["random", "both"]:
        test_volume_optimization(n_points=args.points, seed=args.seed)

    if args.example in ["hexagon", "both"]:
        test_hexagon_optimization()


if __name__ == "__main__":
    main()
