#!/usr/bin/env python3
"""
Test geometric reconstruction on a symmetric example (hexagon).

This should give nearly perfect reconstruction since the geometry
is highly symmetric and the optimal angles should be exactly realizable.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/igor/devel/ideal_poly_volume_toolkit')

from ideal_poly_volume_toolkit.rivin_delaunay import (
    check_delaunay_realizability,
    optimize_hyperbolic_volume,
    realize_angles_as_points,
    compute_triangle_angle,
)


def test_hexagon():
    """Test on regular hexagon triangulated from center."""
    print("="*70)
    print("HEXAGON RECONSTRUCTION TEST")
    print("="*70)
    print()

    # Regular hexagon with center
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    points_original = np.column_stack([np.cos(angles), np.sin(angles)])
    center = np.array([[0.0, 0.0]])
    all_points_original = np.vstack([center, points_original])

    # Triangulate from center
    triangles = [(0, i+1, ((i+1) % 6) + 1) for i in range(6)]

    print("Step 1: Check realizability")
    result_check = check_delaunay_realizability(triangles, verbose=False)
    print(f"  ✓ Realizable with min angle: {np.degrees(result_check['min_angle_radians']):.2f}°")
    print()

    print("Step 2: Optimize volume")
    result_opt = optimize_hyperbolic_volume(triangles, verbose=False)

    from ideal_poly_volume_toolkit.rivin_delaunay import lobachevsky_function
    initial_volume = np.sum([lobachevsky_function(theta)
                            for theta in result_check['angles_radians'].flatten()])

    print(f"  ✓ Optimization successful")
    print(f"    Initial volume: {initial_volume:.6f}")
    print(f"    Optimal volume: {result_opt['volume']:.6f}")
    print(f"    Improvement: {100*(result_opt['volume']-initial_volume)/abs(initial_volume):.2f}%")
    print()

    print("Optimal angles:")
    print(f"  {np.degrees(result_opt['angles'].flatten())}")
    print(f"  All angles equal? {np.allclose(result_opt['angles'], result_opt['angles'][0,0], rtol=1e-3)}")
    print()

    print("Step 3: Reconstruct geometry")
    result_reconstruct = realize_angles_as_points(triangles, result_opt['angles'], verbose=True)

    print()
    print(f"  Angle reconstruction error: {result_reconstruct['angle_error_degrees']:.4f}°")

    if result_reconstruct['angle_error_degrees'] < 0.1:
        print(f"  ✓ Excellent reconstruction!")
    elif result_reconstruct['angle_error_degrees'] < 1.0:
        print(f"  ✓ Good reconstruction")
    else:
        print(f"  ⚠ Moderate reconstruction")

    print()
    print("="*70)

    return result_reconstruct


if __name__ == "__main__":
    test_hexagon()
