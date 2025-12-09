#!/usr/bin/env python3
"""
Test geometric realization from Rivin LP angles.

With the corrected triangle extraction bug fixed, we should be able to:
1. Check realizability and get angles from the LP
2. Reconstruct point positions from those angles
3. Verify the reconstructed triangulation matches the original
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from ideal_poly_volume_toolkit.plantri_interface import find_plantri_executable
from ideal_poly_volume_toolkit.planar_utils import extract_faces_from_planar_embedding
from ideal_poly_volume_toolkit.rivin_delaunay import (
    check_delaunay_realizability,
    realize_angles_as_points
)
import subprocess


def get_nth_triangulation(n_vertices: int, index: int, min_connectivity: int = 3):
    """Get the nth triangulation for given vertex count."""
    plantri = find_plantri_executable()
    args = [plantri, f'-pc{min_connectivity}', '-a', str(n_vertices)]
    result = subprocess.run(args, capture_output=True, text=True)

    triangulations = []
    for line in result.stdout.split('\n'):
        line = line.strip()
        if not line or line.startswith('>'):
            continue

        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue

        n = int(parts[0])
        adj_str = parts[1]

        # Build adjacency dict
        adj = {}
        for v_idx, neighbor_str in enumerate(adj_str.split(',')):
            neighbors = [ord(c) - ord('a') for c in neighbor_str]
            adj[v_idx] = neighbors

        # Extract faces using CORRECTED method
        closed_tri = extract_faces_from_planar_embedding(n, adj)

        # Remove vertex 0 to get planar
        planar_tri = [tri for tri in closed_tri if 0 not in tri]

        if planar_tri:
            triangulations.append(planar_tri)

    if index < len(triangulations):
        return triangulations[index]
    else:
        return None


def test_octahedron():
    """Test on the octahedron (n=6, the unique strictly realizable case)."""
    print("="*70)
    print("TEST: Octahedron Geometric Realization")
    print("="*70)

    # Get n=6 triangulations
    print("\nLoading n=6 triangulations...")
    triangulations = []

    for i in range(7):  # We know there are 7 of them
        tri = get_nth_triangulation(6, i, min_connectivity=3)
        if tri:
            triangulations.append((i, tri))

    print(f"Found {len(triangulations)} triangulations")

    # Test each one, looking for the octahedron
    for idx, triangles in triangulations:
        print(f"\n{'='*70}")
        print(f"Testing triangulation #{idx}")
        print(f"{'='*70}")
        print(f"Triangles: {triangles}")

        # Check strict realizability
        result = check_delaunay_realizability(triangles, verbose=False, strict=True)

        if not result['realizable']:
            print(f"  ✗ Not strictly realizable, skipping")
            continue

        print(f"  ✓ Strictly realizable!")
        print(f"  Min angle: {result.get('min_angle', 0):.6f} rad")
        print(f"  Max dihedral: {result.get('max_dihedral', 0):.6f} rad (π/2 = {np.pi/2:.6f})")

        # Extract angles from LP solution
        angles = result.get('angles')
        if angles is None:
            print(f"  ✗ No angles in result")
            continue

        # Reshape angles to (n_triangles, 3)
        n_triangles = len(triangles)
        target_angles = angles.reshape((n_triangles, 3))

        print(f"\n  Reconstructing geometry from LP angles...")
        print(f"  Target angles shape: {target_angles.shape}")

        # Realize as points
        realization = realize_angles_as_points(triangles, target_angles, verbose=True)

        if realization['success']:
            print(f"\n  ✓ Geometric realization SUCCESS!")
            print(f"  Angle error (RMS): {realization.get('angle_error', 0):.6e} rad")
            print(f"  Angle error: {realization.get('angle_error_degrees', 0):.6f}°")
            print(f"  Triangulation preserved: {realization.get('triangulation_preserved', False)}")

            points = realization['points']
            print(f"\n  Point coordinates:")
            vertex_list = realization['vertex_list']
            for i, v in enumerate(vertex_list):
                print(f"    v{v}: ({points[i, 0]:8.5f}, {points[i, 1]:8.5f})")
        else:
            print(f"\n  ✗ Geometric realization FAILED")
            print(f"  Message: {realization.get('message', 'Unknown error')}")


def test_simple_case(n: int = 7, index: int = 0):
    """Test on a specific triangulation."""
    print("\n" + "="*70)
    print(f"TEST: n={n} triangulation #{index}")
    print("="*70)

    triangles = get_nth_triangulation(n, index, min_connectivity=3)

    if triangles is None:
        print(f"Could not load triangulation")
        return

    print(f"\nTriangles: {triangles}")
    print(f"Number of triangles: {len(triangles)}")

    # Check realizability
    print("\nChecking realizability (standard mode)...")
    result = check_delaunay_realizability(triangles, verbose=False, strict=False)

    if not result['realizable']:
        print(f"✗ Not realizable")
        return

    print(f"✓ Realizable!")

    # Extract angles
    angles = result.get('angles')
    n_triangles = len(triangles)
    target_angles = angles.reshape((n_triangles, 3))

    print(f"\nReconstructing geometry from LP angles...")
    realization = realize_angles_as_points(triangles, target_angles, verbose=True)

    if realization['success']:
        print(f"\n✓ Geometric realization SUCCESS!")
        print(f"Angle error (RMS): {realization.get('angle_error', 0):.6e} rad")
        print(f"Triangulation preserved: {realization.get('triangulation_preserved', False)}")
    else:
        print(f"\n✗ Geometric realization FAILED")
        print(f"Message: {realization.get('message', 'Unknown error')}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test geometric realization from LP angles')
    parser.add_argument('--test', choices=['octahedron', 'simple'], default='octahedron',
                       help='Which test to run')
    parser.add_argument('--n', type=int, default=7, help='Number of vertices (for simple test)')
    parser.add_argument('--index', type=int, default=0, help='Triangulation index (for simple test)')

    args = parser.parse_args()

    if args.test == 'octahedron':
        test_octahedron()
    else:
        test_simple_case(args.n, args.index)

    print("\n" + "="*70)
