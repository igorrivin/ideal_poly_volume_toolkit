#!/usr/bin/env python3
"""
Test the self-contained volume_threshold module.
"""

import numpy as np
from ideal_poly_volume_toolkit.volume_threshold import (
    volume_exceeds_threshold,
    get_volume,
    compute_volume
)


def test_basic_tetrahedron():
    """Test with basic tetrahedron: 0, 1, i, ∞"""
    print("=" * 70)
    print("Test 1: Basic tetrahedron (0, 1, i, ∞)")
    print("=" * 70)

    vertices = [0+0j, 1+0j, 1j, np.inf]
    volume = get_volume(vertices)
    print(f"Vertices: {vertices}")
    print(f"Volume: {volume:.8f}")

    # Test threshold comparisons
    thresholds = [0.5, 1.0, 1.5, 2.0]
    for threshold in thresholds:
        result = volume_exceeds_threshold(vertices, threshold)
        print(f"  Volume > {threshold}? {result}")

    print()


def test_seven_vertex_config():
    """Test with 7-vertex configuration (4 random + 0, 1, i)"""
    print("=" * 70)
    print("Test 2: Seven-vertex polyhedron (0, 1, i, ∞ + 3 random)")
    print("=" * 70)

    # Use a known configuration from the examples
    np.random.seed(42)
    random_vertices = [
        0.5 + 0.5j,
        -0.3 + 0.7j,
        0.8 - 0.2j
    ]

    vertices = [0+0j, 1+0j, 1j, np.inf] + random_vertices
    volume = get_volume(vertices)

    print(f"Fixed vertices: 0, 1, i, ∞")
    print(f"Random vertices: {random_vertices}")
    print(f"Total vertices: {len(vertices)}")
    print(f"Volume: {volume:.8f}")

    # Test threshold comparisons
    thresholds = [2.0, 3.0, 4.0, 5.0]
    for threshold in thresholds:
        result = volume_exceeds_threshold(vertices, threshold)
        print(f"  Volume > {threshold}? {result}")

    print()


def test_optimized_config():
    """Test with a known optimized configuration"""
    print("=" * 70)
    print("Test 3: Optimized configuration from previous runs")
    print("=" * 70)

    # This is an example - replace with actual optimized values if available
    # Format: real parts, imaginary parts for the 4 free vertices
    # (0, 1, i are fixed, infinity is implicit)
    real_parts = [0.234, -0.456, 0.789, -0.123]
    imag_parts = [0.567, 0.890, -0.234, 0.456]

    free_vertices = [complex(r, i) for r, i in zip(real_parts, imag_parts)]
    vertices = [0+0j, 1+0j, 1j, np.inf] + free_vertices

    volume = get_volume(vertices)

    print(f"Free vertices (4):")
    for i, v in enumerate(free_vertices, 1):
        print(f"  v{i}: {v:.4f}")
    print(f"Total vertices: {len(vertices)}")
    print(f"Volume: {volume:.8f}")

    # Test threshold comparisons
    thresholds = [3.0, 4.0, 5.0, 6.0]
    for threshold in thresholds:
        result = volume_exceeds_threshold(vertices, threshold)
        print(f"  Volume > {threshold}? {result}")

    print()


def test_method_comparison():
    """Compare hull vs Delaunay methods"""
    print("=" * 70)
    print("Test 4: Method comparison (hull vs Delaunay)")
    print("=" * 70)

    vertices = [0+0j, 1+0j, 1j, np.inf, 0.5+0.5j]

    vol_hull = compute_volume(vertices, method="hull")
    vol_delaunay = compute_volume(vertices, method="delaunay")
    vol_auto = compute_volume(vertices, method="auto")

    print(f"Vertices: {vertices}")
    print(f"Volume (hull method):     {vol_hull:.8f}")
    print(f"Volume (Delaunay method): {vol_delaunay:.8f}")
    print(f"Volume (auto method):     {vol_auto:.8f}")
    print(f"Difference (hull-Delaunay): {abs(vol_hull - vol_delaunay):.2e}")

    print()


def test_performance():
    """Test performance with larger vertex sets"""
    print("=" * 70)
    print("Test 5: Performance test")
    print("=" * 70)

    import time

    for n_random in [5, 10, 15, 20]:
        np.random.seed(42)
        # Generate random vertices in unit disk
        random_vertices = []
        for _ in range(n_random):
            r = np.random.uniform(0, 0.9)
            theta = np.random.uniform(0, 2*np.pi)
            random_vertices.append(r * np.exp(1j * theta))

        vertices = [0+0j, 1+0j, 1j, np.inf] + random_vertices

        start = time.time()
        volume = get_volume(vertices)
        elapsed = time.time() - start

        print(f"  {len(vertices)} vertices: volume = {volume:.6f}, time = {elapsed*1000:.2f} ms")

    print()


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("VOLUME THRESHOLD MODULE TEST SUITE")
    print("=" * 70)
    print()

    test_basic_tetrahedron()
    test_seven_vertex_config()
    test_optimized_config()
    test_method_comparison()
    test_performance()

    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
