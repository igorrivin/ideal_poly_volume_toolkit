#!/usr/bin/env python3
"""
Example: Check if a triangulation is Delaunay realizable using Rivin's algorithm.

This script demonstrates how to use the Rivin LP solver to verify whether
a given triangulation can be realized as a Delaunay triangulation.
"""

import numpy as np
import argparse
from scipy.spatial import Delaunay

import sys
sys.path.insert(0, '/home/igor/devel/ideal_poly_volume_toolkit')

from ideal_poly_volume_toolkit.rivin_delaunay import (
    check_delaunay_realizability,
    format_realizability_report,
)


def check_random_triangulation(n_points=10, seed=None):
    """
    Generate random points, compute Delaunay triangulation, and verify realizability.

    Args:
        n_points: Number of random points to generate
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random points
    points = np.random.rand(n_points, 2)

    # Compute Delaunay triangulation
    tri = Delaunay(points)
    triangles = [tuple(simplex) for simplex in tri.simplices]

    print(f"Generated {n_points} random points")
    print(f"Delaunay triangulation: {len(triangles)} triangles")
    print()

    # Check Delaunay realizability
    result = check_delaunay_realizability(triangles, verbose=True)

    print("\n" + format_realizability_report(result))

    if result['realizable']:
        print("\nAs expected, Delaunay triangulations are Delaunay realizable! ✓")

    return result


def check_custom_triangulation(triangles):
    """
    Check if a custom triangulation is Delaunay realizable.

    Args:
        triangles: List of triangles as (v0, v1, v2) tuples
    """
    print(f"Checking triangulation with {len(triangles)} triangles")
    print(f"Triangles: {triangles}")
    print()

    result = check_delaunay_realizability(triangles, verbose=True)

    print("\n" + format_realizability_report(result))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Check Delaunay realizability of triangulations using Rivin's algorithm"
    )
    parser.add_argument(
        "--points",
        type=int,
        default=10,
        help="Number of random points to generate (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "--example",
        choices=["random", "hexagon", "grid"],
        default="random",
        help="Type of example to run (default: random)",
    )

    args = parser.parse_args()

    print("="*70)
    print("DELAUNAY REALIZABILITY CHECK")
    print("="*70)
    print()

    if args.example == "random":
        print("Example: Random Delaunay triangulation\n")
        check_random_triangulation(n_points=args.points, seed=args.seed)

    elif args.example == "hexagon":
        print("Example: Regular hexagon triangulated from center\n")
        # Regular hexagon with center point
        angles = np.linspace(0, 2*np.pi, 7)[:-1]
        points = np.column_stack([np.cos(angles), np.sin(angles)])

        # Triangulate from center (vertex 0)
        triangles = [(0, i+1, ((i+1) % 6) + 1) for i in range(6)]

        # Need to add center point for Delaunay
        center = np.array([[0.0, 0.0]])
        all_points = np.vstack([center, points])

        check_custom_triangulation(triangles)

    elif args.example == "grid":
        print("Example: 3x3 grid Delaunay triangulation\n")
        # Create 3x3 grid
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 1, 3)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        # Compute Delaunay
        tri = Delaunay(points)
        triangles = [tuple(simplex) for simplex in tri.simplices]

        check_custom_triangulation(triangles)


if __name__ == "__main__":
    main()
