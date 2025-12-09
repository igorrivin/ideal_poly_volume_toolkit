#!/usr/bin/env python3
"""
Test the full Rivin algorithm pipeline:

1. Start with random points in 2D
2. Compute Delaunay triangulation
3. Check realizability (should pass for Delaunay)
4. Optimize angles for maximum hyperbolic volume
5. Reconstruct geometric realization from optimal angles
6. Verify that reconstructed geometry has correct angles

This demonstrates the complete workflow from geometry → angles → optimized angles → new geometry.
"""

import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/igor/devel/ideal_poly_volume_toolkit')

from ideal_poly_volume_toolkit.rivin_delaunay import (
    check_delaunay_realizability,
    optimize_hyperbolic_volume,
    realize_angles_as_points,
    refine_geometry_for_volume,
    compute_triangle_angle,
)


def verify_angles(points, triangles, target_angles, vertex_list):
    """
    Verify that the reconstructed points have the correct angles.

    Args:
        points: np.ndarray of shape (n_vertices, 2)
        triangles: List of triangles
        target_angles: Array of target angles
        vertex_list: Mapping from point index to vertex ID

    Returns:
        RMS angle error in radians
    """
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}

    errors = []
    for tri_id, (v0, v1, v2) in enumerate(triangles):
        p0 = points[vertex_to_idx[v0]]
        p1 = points[vertex_to_idx[v1]]
        p2 = points[vertex_to_idx[v2]]

        # Compute actual angles
        angle0 = compute_triangle_angle(p0, p1, p2)
        angle1 = compute_triangle_angle(p1, p2, p0)
        angle2 = compute_triangle_angle(p2, p0, p1)

        actual = np.array([angle0, angle1, angle2])
        target = target_angles[tri_id]

        errors.extend((actual - target)**2)

    return np.sqrt(np.mean(errors))


def test_full_pipeline(n_points=15, seed=42, plot=False):
    """
    Test the complete pipeline.

    Args:
        n_points: Number of random points
        seed: Random seed
        plot: If True, plot original and reconstructed geometries
    """
    np.random.seed(seed)

    print("="*70)
    print("FULL RIVIN ALGORITHM PIPELINE")
    print("="*70)
    print()

    # Step 1: Generate random points
    print(f"Step 1: Generate {n_points} random points")
    original_points = np.random.rand(n_points, 2)
    print(f"  ✓ Points generated")
    print()

    # Step 2: Compute Delaunay triangulation
    print("Step 2: Compute Delaunay triangulation")
    tri = Delaunay(original_points)
    triangles = [tuple(simplex) for simplex in tri.simplices]
    print(f"  ✓ Triangulation: {len(triangles)} triangles")
    print()

    # Step 3: Check realizability
    print("Step 3: Check Delaunay realizability")
    result_check = check_delaunay_realizability(triangles, verbose=False)
    if not result_check['realizable']:
        print("  ✗ Not realizable (unexpected!)")
        return

    print(f"  ✓ Realizable with min angle: {np.degrees(result_check['min_angle_radians']):.2f}°")
    print()

    # Step 4: Optimize angles for maximum volume
    print("Step 4: Optimize angles for maximum hyperbolic volume")
    result_opt = optimize_hyperbolic_volume(triangles, verbose=False)

    if not result_opt['success']:
        print(f"  ✗ Optimization failed")
        return

    from ideal_poly_volume_toolkit.rivin_delaunay import lobachevsky_function
    initial_volume = np.sum([lobachevsky_function(theta)
                            for theta in result_check['angles_radians'].flatten()])

    print(f"  ✓ Optimization successful")
    print(f"    Iterations: {result_opt['n_iterations']}")
    print(f"    Initial volume: {initial_volume:.6f}")
    print(f"    Optimal volume: {result_opt['volume']:.6f}")
    print(f"    Improvement: {100*(result_opt['volume']-initial_volume)/abs(initial_volume):.2f}%")
    print()

    # Step 5: Reconstruct geometry from optimal angles
    print("Step 5: Reconstruct geometry from optimal angles")
    result_reconstruct = realize_angles_as_points(
        triangles,
        result_opt['angles'],
        verbose=True
    )

    if not result_reconstruct['success']:
        if not result_reconstruct.get('triangulation_preserved', True):
            print(f"  ✗ Reconstruction failed: triangulation not preserved")
        else:
            print(f"  ✗ Reconstruction failed")
        return

    reconstructed_points = result_reconstruct['points']
    vertex_list = result_reconstruct['vertex_list']

    print()

    # Step 6: Refine geometry via direct volume optimization
    print("Step 6: Refine geometry via direct BFGS volume optimization")
    print("  (This nails down the exact geometry while preserving triangulation)")
    print()

    result_refined = refine_geometry_for_volume(
        reconstructed_points,
        vertex_list,
        triangles,
        verbose=True
    )

    if not result_refined['success']:
        print(f"  ✗ Refinement failed or triangulation changed")
        if not result_refined['triangulation_preserved']:
            print(f"    Warning: Delaunay triangulation changed during refinement!")
        refined_points = reconstructed_points  # Fall back to unrefined
    else:
        refined_points = result_refined['points']

    print()

    # Step 7: Verify angles
    print("Step 7: Verify final angles")
    rms_error_refined = verify_angles(refined_points, triangles, result_opt['angles'], vertex_list)

    print(f"  RMS angle error (refined): {rms_error_refined:.6f} radians = {np.degrees(rms_error_refined):.4f}°")
    if rms_error_refined < 0.001:
        print(f"  ✓ Excellent agreement!")
    elif rms_error_refined < 0.01:
        print(f"  ✓ Good agreement")
    else:
        print(f"  ⚠ Moderate agreement")
    print()

    # Compare volumes
    if result_refined['success'] and result_refined['volume'] is not None:
        print(f"Volume comparison:")
        print(f"  Target (from angle opt):  {result_opt['volume']:.6f}")
        print(f"  Achieved (refined geom):  {result_refined['volume']:.6f}")
        print(f"  Difference:               {abs(result_refined['volume'] - result_opt['volume']):.6f}")
        print()

    # Compare original vs reconstructed
    print("="*70)
    print("COMPARISON: Original vs Reconstructed")
    print("="*70)
    print(f"Original points: {original_points.shape[0]} vertices")
    print(f"Refined points: {refined_points.shape[0]} vertices")
    print()

    # Compute angles in original geometry
    print("Angle comparison:")
    vertex_to_idx_orig = {i: i for i in range(n_points)}
    vertex_to_idx_recon = {v: i for i, v in enumerate(vertex_list)}

    angle_diffs = []
    for tri_id, (v0, v1, v2) in enumerate(triangles):
        # Original angles
        p0_orig = original_points[v0]
        p1_orig = original_points[v1]
        p2_orig = original_points[v2]

        angle0_orig = compute_triangle_angle(p0_orig, p1_orig, p2_orig)
        angle1_orig = compute_triangle_angle(p1_orig, p2_orig, p0_orig)
        angle2_orig = compute_triangle_angle(p2_orig, p0_orig, p1_orig)
        orig_angles = np.array([angle0_orig, angle1_orig, angle2_orig])

        # Reconstructed angles
        p0_recon = reconstructed_points[vertex_to_idx_recon[v0]]
        p1_recon = reconstructed_points[vertex_to_idx_recon[v1]]
        p2_recon = reconstructed_points[vertex_to_idx_recon[v2]]

        angle0_recon = compute_triangle_angle(p0_recon, p1_recon, p2_recon)
        angle1_recon = compute_triangle_angle(p1_recon, p2_recon, p0_recon)
        angle2_recon = compute_triangle_angle(p2_recon, p0_recon, p1_recon)
        recon_angles = np.array([angle0_recon, angle1_recon, angle2_recon])

        # Optimal angles
        opt_angles = result_opt['angles'][tri_id]

        angle_diffs.append(np.abs(orig_angles - opt_angles))

    angle_diffs = np.concatenate(angle_diffs)

    print(f"  Average angle change: {np.degrees(np.mean(angle_diffs)):.2f}°")
    print(f"  Max angle change: {np.degrees(np.max(angle_diffs)):.2f}°")
    print()

    # Plot if requested
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot original
        ax1.triplot(original_points[:, 0], original_points[:, 1], tri.simplices, 'b-', linewidth=0.5)
        ax1.plot(original_points[:, 0], original_points[:, 1], 'ro', markersize=4)
        ax1.set_title(f'Original Delaunay Triangulation\n({len(triangles)} triangles)')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)

        # Plot reconstructed
        # Need to create triangles with reconstructed indices
        recon_triangles = []
        for v0, v1, v2 in triangles:
            i0 = vertex_to_idx_recon[v0]
            i1 = vertex_to_idx_recon[v1]
            i2 = vertex_to_idx_recon[v2]
            recon_triangles.append([i0, i1, i2])
        recon_triangles = np.array(recon_triangles)

        ax2.triplot(reconstructed_points[:, 0], reconstructed_points[:, 1], recon_triangles, 'g-', linewidth=0.5)
        ax2.plot(reconstructed_points[:, 0], reconstructed_points[:, 1], 'ro', markersize=4)
        ax2.set_title(f'Reconstructed from Optimal Angles\n(Volume improved by {100*(result_opt["volume"]-initial_volume)/abs(initial_volume):.1f}%)')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('rivin_pipeline_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved to: rivin_pipeline_comparison.png")

    print("="*70)
    print("PIPELINE COMPLETE ✓")
    print("="*70)

    return {
        'original_points': original_points,
        'reconstructed_points': reconstructed_points,
        'triangles': triangles,
        'optimal_angles': result_opt['angles'],
        'volume_improvement': 100*(result_opt['volume']-initial_volume)/abs(initial_volume),
        'angle_reconstruction_error': rms_error_refined,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test full Rivin algorithm pipeline"
    )
    parser.add_argument(
        "--points",
        type=int,
        default=15,
        help="Number of random points (default: 15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plot",
    )

    args = parser.parse_args()

    print("\n" + "#"*70)
    print("# Full Rivin Algorithm Pipeline Test")
    print("#"*70 + "\n")

    result = test_full_pipeline(
        n_points=args.points,
        seed=args.seed,
        plot=args.plot
    )


if __name__ == "__main__":
    main()
