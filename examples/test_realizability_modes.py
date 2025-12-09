#!/usr/bin/env python3
"""
Test the different realizability modes: standard, strict, and Andreev.
"""

import numpy as np
from scipy.spatial import Delaunay
import sys
sys.path.insert(0, '/home/igor/devel/ideal_poly_volume_toolkit')

from ideal_poly_volume_toolkit.rivin_delaunay import check_delaunay_realizability


def test_simple_triangulation():
    """Test a simple triangulation with all three modes."""
    print("=" * 70)
    print("TEST: Simple hexagon triangulation")
    print("=" * 70)

    # Create a simple hexagon triangulation
    # 6 boundary vertices forming a regular hexagon + triangulation
    points = np.array([
        [np.cos(i * np.pi / 3), np.sin(i * np.pi / 3)]
        for i in range(6)
    ])

    tri = Delaunay(points)
    triangles = [tuple(simplex) for simplex in tri.simplices]

    print(f"\nTriangulation: {len(triangles)} triangles, {len(points)} vertices")
    print(f"Triangles: {triangles}")

    # Test standard mode
    print("\n" + "-" * 70)
    print("MODE 1: STANDARD (dihedral ≤ π)")
    print("-" * 70)
    result_standard = check_delaunay_realizability(triangles, verbose=True)
    print(f"\n✓ Realizable: {result_standard['realizable']}")
    if result_standard['realizable']:
        print(f"  Min angle: {result_standard['min_angle_radians']:.6f} rad = {np.degrees(result_standard['min_angle_radians']):.2f}°")

    # Test strict mode
    print("\n" + "-" * 70)
    print("MODE 2: STRICT (dihedral < π)")
    print("-" * 70)
    result_strict = check_delaunay_realizability(triangles, verbose=True, strict=True)
    print(f"\n✓ Realizable (strict): {result_strict['realizable']}")
    if result_strict['realizable']:
        print(f"  Min angle: {result_strict['min_angle_radians']:.6f} rad = {np.degrees(result_strict['min_angle_radians']):.2f}°")
        print(f"  Dihedral slack: {result_strict['slack_radians']:.6f} rad = {np.degrees(result_strict['slack_radians']):.2f}°")
        print(f"  Max dihedral: {result_strict['max_dihedral_radians']:.6f} rad = {np.degrees(result_strict['max_dihedral_radians']):.2f}°")

    # Test Andreev mode
    print("\n" + "-" * 70)
    print("MODE 3: ANDREEV (dihedral ≤ π/2)")
    print("-" * 70)
    result_andreev = check_delaunay_realizability(triangles, verbose=True, andreev=True)
    print(f"\n✓ Realizable (Andreev): {result_andreev['realizable']}")
    if result_andreev['realizable']:
        print(f"  Min angle: {result_andreev['min_angle_radians']:.6f} rad = {np.degrees(result_andreev['min_angle_radians']):.2f}°")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Standard realizable: {result_standard['realizable']}")
    print(f"Strict realizable:   {result_strict['realizable']}")
    print(f"Andreev realizable:  {result_andreev['realizable']}")

    return result_standard, result_strict, result_andreev


def test_random_triangulation():
    """Test a random triangulation."""
    print("\n\n" + "=" * 70)
    print("TEST: Random triangulation (10 points)")
    print("=" * 70)

    np.random.seed(42)
    points = np.random.rand(10, 2)
    tri = Delaunay(points)
    triangles = [tuple(simplex) for simplex in tri.simplices]

    print(f"\nTriangulation: {len(triangles)} triangles, {len(points)} vertices")

    # Quick tests without verbose
    result_standard = check_delaunay_realizability(triangles, verbose=False)
    result_strict = check_delaunay_realizability(triangles, verbose=False, strict=True)
    result_andreev = check_delaunay_realizability(triangles, verbose=False, andreev=True)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Standard realizable: {result_standard['realizable']}")
    if result_standard['realizable']:
        print(f"  Min angle: {np.degrees(result_standard['min_angle_radians']):.2f}°")

    print(f"\nStrict realizable:   {result_strict['realizable']}")
    if result_strict['realizable']:
        print(f"  Min angle: {np.degrees(result_strict['min_angle_radians']):.2f}°")
        print(f"  Dihedral slack: {np.degrees(result_strict['slack_radians']):.2f}°")
        print(f"  Max dihedral: {np.degrees(result_strict['max_dihedral_radians']):.2f}° (< 180°)")

    print(f"\nAndreev realizable:  {result_andreev['realizable']}")
    if result_andreev['realizable']:
        print(f"  Min angle: {np.degrees(result_andreev['min_angle_radians']):.2f}°")

    return result_standard, result_strict, result_andreev


def test_octahedron():
    """Test octahedron (known to be realizable)."""
    print("\n\n" + "=" * 70)
    print("TEST: Octahedron (6 vertices, remove one -> square pyramid)")
    print("=" * 70)

    # Octahedron vertices on sphere
    points_3d = np.array([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
    ])

    # Project to 2D for Delaunay (we removed vertex [0, 0, -1])
    points = points_3d[:, :2]

    tri = Delaunay(points)
    triangles = [tuple(simplex) for simplex in tri.simplices]

    print(f"\nTriangulation: {len(triangles)} triangles, {len(points)} vertices")

    # Test all modes
    result_standard = check_delaunay_realizability(triangles, verbose=False)
    result_strict = check_delaunay_realizability(triangles, verbose=False, strict=True)
    result_andreev = check_delaunay_realizability(triangles, verbose=False, andreev=True)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Standard realizable: {result_standard['realizable']}")
    print(f"Strict realizable:   {result_strict['realizable']}")
    print(f"Andreev realizable:  {result_andreev['realizable']}")

    if result_strict['realizable']:
        print(f"\nStrict mode details:")
        print(f"  Dihedral slack: {np.degrees(result_strict['slack_radians']):.4f}°")
        print(f"  Max dihedral: {np.degrees(result_strict['max_dihedral_radians']):.4f}° (< 180°)")

    return result_standard, result_strict, result_andreev


if __name__ == '__main__':
    print("Testing Rivin Delaunay Realizability with different modes\n")

    # Run tests
    test_simple_triangulation()
    test_random_triangulation()
    test_octahedron()

    print("\n\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
