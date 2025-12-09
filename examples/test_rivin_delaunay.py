#!/usr/bin/env python3
"""
Test the Rivin algorithm for Delaunay realizability.

This script tests triangulations that are known to be Delaunay realizable
and non-realizable to verify the LP solver works correctly.
"""

import numpy as np
from scipy.spatial import Delaunay

from ideal_poly_volume_toolkit.rivin_delaunay import (
    check_delaunay_realizability,
    format_realizability_report,
)


def test_simple_delaunay_triangulation():
    """Test a simple Delaunay triangulation (should be realizable)."""
    print("="*70)
    print("TEST 1: Simple Delaunay triangulation (4 points forming 2 triangles)")
    print("="*70)

    # Create a simple point set
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.5],
        [0.5, 1.0]
    ])

    # Compute Delaunay triangulation
    tri = Delaunay(points)
    triangles = [tuple(simplex) for simplex in tri.simplices]

    print(f"Points:\n{points}")
    print(f"\nTriangles: {triangles}")

    # Check realizability
    result = check_delaunay_realizability(triangles, verbose=True)
    print("\n" + format_realizability_report(result))

    return result['realizable']


def test_non_delaunay_triangulation():
    """Test a non-Delaunay triangulation (should NOT be realizable)."""
    print("\n" + "="*70)
    print("TEST 2: Non-Delaunay triangulation (square with bad diagonal)")
    print("="*70)

    # Square with vertices at (0,0), (1,0), (1,1), (0,1)
    # Delaunay would use diagonal 0-2, but we force diagonal 1-3
    # This violates the Delaunay property

    # Good triangulation (Delaunay):  (0,1,2) and (0,2,3)
    # Bad triangulation (not Delaunay): (0,1,3) and (1,2,3)

    triangles_bad = [(0, 1, 3), (1, 2, 3)]

    print(f"Triangles (forced bad diagonal): {triangles_bad}")
    print("Note: For a square, the Delaunay triangulation uses the shorter diagonal.")
    print("We're forcing the longer diagonal, which should fail the Delaunay test.")

    # Check realizability
    result = check_delaunay_realizability(triangles_bad, verbose=True)
    print("\n" + format_realizability_report(result))

    return result['realizable']


def test_hexagon_triangulation():
    """Test a regular hexagon triangulation."""
    print("\n" + "="*70)
    print("TEST 3: Regular hexagon triangulated from center")
    print("="*70)

    # Regular hexagon with center point
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 vertices
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    center = np.array([[0.0, 0.0]])
    all_points = np.vstack([center, points])

    # Triangulate from center
    triangles = [(0, i+1, ((i+1) % 6) + 1) for i in range(6)]

    print(f"Points: center + 6 vertices on circle")
    print(f"Triangles: {triangles}")

    # Check realizability
    result = check_delaunay_realizability(triangles, verbose=True)
    print("\n" + format_realizability_report(result))

    return result['realizable']


def test_random_delaunay():
    """Test a random Delaunay triangulation."""
    print("\n" + "="*70)
    print("TEST 4: Random Delaunay triangulation (10 points)")
    print("="*70)

    # Random points
    np.random.seed(42)
    points = np.random.rand(10, 2)

    # Compute Delaunay triangulation
    tri = Delaunay(points)
    triangles = [tuple(simplex) for simplex in tri.simplices]

    print(f"Number of triangles: {len(triangles)}")

    # Check realizability
    result = check_delaunay_realizability(triangles, verbose=True)
    print("\n" + format_realizability_report(result))

    return result['realizable']


def main():
    print("\n" + "#"*70)
    print("# Testing Rivin's Delaunay Realizability Algorithm")
    print("#"*70 + "\n")

    results = {}

    # Test 1: Simple Delaunay (should pass)
    results['simple_delaunay'] = test_simple_delaunay_triangulation()

    # Test 2: Non-Delaunay (should fail)
    results['non_delaunay'] = test_non_delaunay_triangulation()

    # Test 3: Hexagon (should pass)
    results['hexagon'] = test_hexagon_triangulation()

    # Test 4: Random Delaunay (should pass)
    results['random'] = test_random_delaunay()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Test 1 (Simple Delaunay):     {'PASS ✓' if results['simple_delaunay'] else 'FAIL ✗'} (expected: realizable)")
    print(f"Test 2 (Non-Delaunay):        {'FAIL ✗' if results['non_delaunay'] else 'PASS ✓'} (expected: NOT realizable)")
    print(f"Test 3 (Hexagon):             {'PASS ✓' if results['hexagon'] else 'FAIL ✗'} (expected: realizable)")
    print(f"Test 4 (Random Delaunay):     {'PASS ✓' if results['random'] else 'FAIL ✗'} (expected: realizable)")

    all_passed = (
        results['simple_delaunay'] and
        not results['non_delaunay'] and
        results['hexagon'] and
        results['random']
    )

    print("="*70)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
    else:
        print("SOME TESTS FAILED! ✗")
    print("="*70)


if __name__ == "__main__":
    main()
