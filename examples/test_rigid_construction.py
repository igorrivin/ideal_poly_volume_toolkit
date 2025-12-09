#!/usr/bin/env python3
"""Test rigid geometric construction from Rivin LP angles."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.spatial import Delaunay as scipy_Delaunay
from ideal_poly_volume_toolkit.plantri_interface import find_plantri_executable
from ideal_poly_volume_toolkit.planar_utils import extract_faces_from_planar_embedding
from ideal_poly_volume_toolkit.rivin_delaunay import (
    check_delaunay_realizability,
    compute_triangle_angle
)
from ideal_poly_volume_toolkit.geometric_realization import realize_from_angles_rigid
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
    print("RIGID GEOMETRIC CONSTRUCTION TEST - OCTAHEDRON")
    print("="*70)
    print(f"\nTriangles: {triangles}")

    # Check strict realizability
    result = check_delaunay_realizability(triangles, verbose=False, strict=True)

    if not result['realizable']:
        print("Not realizable!")
        sys.exit(1)

    print(f"✓ Realizable (strict mode)")

    # Extract ALL angles from LP (in scaled units where π = 1)
    angles_scaled = result['angles']
    n_triangles = len(triangles)

    # Convert from scaled units to radians
    angles_radians = angles_scaled * np.pi

    print(f"\nLP solution gives ALL angles for ALL triangles:")
    for i, tri in enumerate(triangles):
        ang = angles_radians.reshape((n_triangles, 3))[i]
        print(f"  Triangle {tri}: {np.degrees(ang)} degrees (sum={np.degrees(ang.sum()):.1f}°)")

    # Rigid construction
    print(f"\n{'='*70}")
    print("RIGID CONSTRUCTION")
    print(f"{'='*70}\n")

    construction = realize_from_angles_rigid(
        triangles,
        angles_radians.reshape((n_triangles, 3)),
        verbose=True
    )

    if not construction['success']:
        print(f"\n✗ Construction failed: {construction['message']}")
        sys.exit(1)

    print(f"\n✓ Construction successful!")

    points = construction['points']
    vertex_list = construction['vertex_list']
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}

    print(f"\nPoint positions:")
    for i, v in enumerate(vertex_list):
        print(f"  v{v}: ({points[i, 0]:10.6f}, {points[i, 1]:10.6f})")

    # Verify triangulation
    print(f"\n{'='*70}")
    print("VERIFICATION")
    print(f"{'='*70}")

    # Check Delaunay triangulation
    tri = scipy_Delaunay(points)
    realized_triangles = set()
    for simplex in tri.simplices:
        v0, v1, v2 = [vertex_list[i] for i in simplex]
        realized_triangles.add(tuple(sorted([v0, v1, v2])))

    expected_triangles = set(tuple(sorted(t)) for t in triangles)

    print(f"\nExpected triangles: {len(expected_triangles)}")
    print(f"Realized triangles: {len(realized_triangles)}")

    if expected_triangles == realized_triangles:
        print(f"✓ Triangulation matches perfectly!")
    else:
        print(f"✗ Triangulation mismatch")
        missing = expected_triangles - realized_triangles
        extra = realized_triangles - expected_triangles
        if missing:
            print(f"  Missing: {missing}")
        if extra:
            print(f"  Extra: {extra}")

    # Check angles
    print(f"\nAngle verification:")
    max_error = 0.0
    total_error_sq = 0.0

    for i, tri in enumerate(triangles):
        v0, v1, v2 = tri
        p0 = points[vertex_to_idx[v0]]
        p1 = points[vertex_to_idx[v1]]
        p2 = points[vertex_to_idx[v2]]

        angle0 = compute_triangle_angle(p0, p1, p2)
        angle1 = compute_triangle_angle(p1, p2, p0)
        angle2 = compute_triangle_angle(p2, p0, p1)

        actual = np.array([angle0, angle1, angle2])
        target = angles_radians.reshape((n_triangles, 3))[i]
        error = np.abs(target - actual)

        max_error = max(max_error, np.max(error))
        total_error_sq += np.sum(error**2)

        print(f"\n  Triangle {tri}:")
        print(f"    Target: {np.degrees(target)}")
        print(f"    Actual: {np.degrees(actual)}")
        print(f"    Error:  {np.degrees(error)} deg")

    rms_error = np.sqrt(total_error_sq / (n_triangles * 3))

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Max angle error:  {np.degrees(max_error):.6f}°")
    print(f"RMS angle error:  {np.degrees(rms_error):.6f}°")
    print(f"Triangulation:    {'✓ MATCH' if expected_triangles == realized_triangles else '✗ MISMATCH'}")
