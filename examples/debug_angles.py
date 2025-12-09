#!/usr/bin/env python3
"""Debug the angles from Rivin LP vs geometric realization."""

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
    """Get the octahedron triangulation (n=6, index=6)."""
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
    print("OCTAHEDRON ANGLE ANALYSIS")
    print("="*70)
    print(f"\nTriangles: {triangles}")

    # Check strict realizability
    result = check_delaunay_realizability(triangles, verbose=True, strict=True)

    print(f"\n{'='*70}")
    print("LP SOLUTION ANGLES")
    print(f"{'='*70}")

    angles = result['angles']
    n_triangles = len(triangles)
    target_angles = angles.reshape((n_triangles, 3))

    for i, tri in enumerate(triangles):
        print(f"\nTriangle {i}: {tri}")
        print(f"  Angles (rad): {target_angles[i]}")
        print(f"  Angles (deg): {np.degrees(target_angles[i])}")
        print(f"  Sum: {target_angles[i].sum():.6f} rad (π = {np.pi:.6f})")

    # Try realization
    print(f"\n{'='*70}")
    print("GEOMETRIC REALIZATION")
    print(f"{'='*70}")

    realization = realize_angles_as_points(triangles, target_angles, verbose=True)

    if realization['success']:
        print(f"\n{'='*70}")
        print("REALIZED ANGLES")
        print(f"{'='*70}")

        points = realization['points']
        vertex_list = realization['vertex_list']
        vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}

        print(f"\nPoint positions:")
        for i, v in enumerate(vertex_list):
            print(f"  v{v}: ({points[i, 0]:8.5f}, {points[i, 1]:8.5f})")

        print(f"\nActual angles achieved:")
        for i, tri in enumerate(triangles):
            v0, v1, v2 = tri
            p0 = points[vertex_to_idx[v0]]
            p1 = points[vertex_to_idx[v1]]
            p2 = points[vertex_to_idx[v2]]

            angle0 = compute_triangle_angle(p0, p1, p2)
            angle1 = compute_triangle_angle(p1, p2, p0)
            angle2 = compute_triangle_angle(p2, p0, p1)

            print(f"\nTriangle {i}: {tri}")
            print(f"  Target angles (rad): {target_angles[i]}")
            print(f"  Target angles (deg): {np.degrees(target_angles[i])}")
            print(f"  Actual angles (rad): [{angle0:.6f}, {angle1:.6f}, {angle2:.6f}]")
            print(f"  Actual angles (deg): {np.degrees([angle0, angle1, angle2])}")
            print(f"  Error (rad): {target_angles[i] - np.array([angle0, angle1, angle2])}")
            print(f"  Error (deg): {np.degrees(target_angles[i] - np.array([angle0, angle1, angle2]))}")
