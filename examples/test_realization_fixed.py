#!/usr/bin/env python3
"""Test geometric realization with correct angle scaling."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from ideal_poly_volume_toolkit.plantri_interface import find_plantri_executable
from ideal_poly_volume_toolkit.planar_utils import extract_faces_from_planar_embedding
from ideal_poly_volume_toolkit.rivin_delaunay import (
    check_delaunay_realizability,
    realize_angles_as_points,
    compute_triangle_angle
)
import subprocess


def get_octahedron():
    """Get the octahedron triangulation."""
    plantri = find_plantri_executable()
    args = [plantri, '-pc3', '-a', '6']
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

        adj = {}
        for v_idx, neighbor_str in enumerate(adj_str.split(',')):
            neighbors = [ord(c) - ord('a') for c in neighbor_str]
            adj[v_idx] = neighbors

        closed_tri = extract_faces_from_planar_embedding(n, adj)
        planar_tri = [tri for tri in closed_tri if 0 not in tri]

        if planar_tri:
            triangulations.append(planar_tri)

    return triangulations[6]  # The octahedron


if __name__ == '__main__':
    triangles = get_octahedron()

    print("="*70)
    print("OCTAHEDRON GEOMETRIC REALIZATION (WITH CORRECT SCALING)")
    print("="*70)
    print(f"\nTriangles: {triangles}")

    # Check strict realizability
    result = check_delaunay_realizability(triangles, verbose=False, strict=True)

    print(f"\nRealizability: {result['realizable']}")

    # Extract angles from LP (in scaled units where π = 1)
    angles_scaled = result['angles']
    n_triangles = len(triangles)

    # FIX: Convert from scaled units to radians
    angles_radians = angles_scaled * np.pi

    target_angles = angles_radians.reshape((n_triangles, 3))

    print(f"\nLP angles (scaled, sum=1): {angles_scaled.reshape((n_triangles, 3))}")
    print(f"LP angles (radians, sum=π): {target_angles}")
    print(f"LP angles (degrees): {np.degrees(target_angles)}")

    # Geometric realization
    print(f"\n{'='*70}")
    print("GEOMETRIC REALIZATION")
    print(f"{'='*70}")

    realization = realize_angles_as_points(triangles, target_angles, verbose=True)

    if realization['success']:
        print(f"\n✓ SUCCESS!")
        print(f"Angle error (RMS): {realization.get('angle_error', 0):.6e} rad")
        print(f"Angle error (degrees): {realization.get('angle_error_degrees', 0):.6f}°")
        print(f"Triangulation preserved: {realization.get('triangulation_preserved', False)}")

        points = realization['points']
        vertex_list = realization['vertex_list']
        vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}

        print(f"\nPoint positions:")
        for i, v in enumerate(vertex_list):
            print(f"  v{v}: ({points[i, 0]:8.5f}, {points[i, 1]:8.5f})")

        # Verify angles
        print(f"\n{'='*70}")
        print("VERIFICATION: Target vs Actual Angles")
        print(f"{'='*70}")

        total_error = 0.0
        for i, tri in enumerate(triangles):
            v0, v1, v2 = tri
            p0 = points[vertex_to_idx[v0]]
            p1 = points[vertex_to_idx[v1]]
            p2 = points[vertex_to_idx[v2]]

            angle0 = compute_triangle_angle(p0, p1, p2)
            angle1 = compute_triangle_angle(p1, p2, p0)
            angle2 = compute_triangle_angle(p2, p0, p1)

            actual = np.array([angle0, angle1, angle2])
            error = np.abs(target_angles[i] - actual)
            total_error += np.sum(error**2)

            print(f"\nTriangle {i}: {tri}")
            print(f"  Target (deg): {np.degrees(target_angles[i])}")
            print(f"  Actual (deg): {np.degrees(actual)}")
            print(f"  Error  (deg): {np.degrees(error)}")

        rms_error = np.sqrt(total_error / (n_triangles * 3))
        print(f"\nOverall RMS error: {rms_error:.6e} rad = {np.degrees(rms_error):.6f}°")

    else:
        print(f"\n✗ FAILED")
        print(f"Message: {realization.get('message', 'Unknown')}")
