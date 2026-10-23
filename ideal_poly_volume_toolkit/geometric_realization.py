"""
Geometric realization from Rivin LP angles.

Construct Euclidean point positions from triangle angles using rigid construction.
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional


def realize_from_angles_rigid(
    triangles: List[Tuple[int, int, int]],
    angles: np.ndarray,  # Shape: (n_triangles, 3), in RADIANS
    boundary_vertices: Optional[Set[int]] = None,
    verbose: bool = False
) -> Dict:
    """
    Rigidly construct point positions from triangle angles.

    Algorithm:
    1. Place v1 = 0, v2 = 1 (complex plane)
    2. Find first triangle containing both v1 and v2
    3. Place third vertex of that triangle using law of sines
    4. Incrementally place remaining vertices using shared edges

    Args:
        triangles: List of (v0, v1, v2) tuples
        angles: Array of shape (n_triangles, 3) with angles in radians
        boundary_vertices: Optional set of boundary vertices
        verbose: Print progress

    Returns:
        Dict with 'success', 'points', 'vertex_list', etc.
    """
    # Get all vertices
    all_vertices = set()
    for tri in triangles:
        all_vertices.update(tri)
    vertex_list = sorted(all_vertices)
    n_vertices = len(vertex_list)

    if verbose:
        print(f"Rigid construction from {len(triangles)} triangles, {n_vertices} vertices")

    # Build triangle -> angles mapping
    tri_to_angles = {}
    for i, tri in enumerate(triangles):
        tri_to_angles[tuple(sorted(tri))] = angles[i]

    # Build vertex -> triangles mapping
    vertex_to_tris = {v: [] for v in vertex_list}
    for i, (v0, v1, v2) in enumerate(triangles):
        vertex_to_tris[v0].append((i, v0, v1, v2))
        vertex_to_tris[v1].append((i, v0, v1, v2))
        vertex_to_tris[v2].append((i, v0, v1, v2))

    # Use complex numbers for 2D positions
    positions = {}  # vertex -> complex position

    # Step 1: Fix first two vertices
    v1 = vertex_list[0]
    v2 = vertex_list[1]
    positions[v1] = 0.0 + 0.0j
    positions[v2] = 1.0 + 0.0j

    if verbose:
        print(f"  Fixed: v{v1} = 0, v{v2} = 1")

    # Step 2: Find a triangle containing both v1 and v2
    first_tri = None
    first_tri_angles = None

    for tri_id, (v0_t, v1_t, v2_t) in enumerate(triangles):
        tri_verts = {v0_t, v1_t, v2_t}
        if v1 in tri_verts and v2 in tri_verts:
            # Found it!
            first_tri = (v0_t, v1_t, v2_t)
            first_tri_angles = angles[tri_id]
            break

    if first_tri is None:
        return {
            'success': False,
            'message': 'Could not find triangle containing first two vertices',
            'points': None,
            'vertex_list': vertex_list,
        }

    # Step 3: Place third vertex of first triangle
    v3 = [v for v in first_tri if v not in [v1, v2]][0]

    # Get angles: we need to know which angle is at which vertex
    # The triangle is (v0_t, v1_t, v2_t) with angles at positions 0, 1, 2
    angle_at = {}
    for i, v in enumerate(first_tri):
        angle_at[v] = first_tri_angles[i]

    alpha = angle_at[v1]  # angle at v1
    beta = angle_at[v2]   # angle at v2
    gamma = angle_at[v3]  # angle at v3

    if verbose:
        print(f"  First triangle: {first_tri}")
        print(f"    Angle at v{v1}: {np.degrees(alpha):.2f}°")
        print(f"    Angle at v{v2}: {np.degrees(beta):.2f}°")
        print(f"    Angle at v{v3}: {np.degrees(gamma):.2f}°")

    # Edge opposite to v3 is v1-v2, which has length 1
    # Using law of sines: |v1-v3| / sin(beta) = |v2-v3| / sin(alpha) = |v1-v2| / sin(gamma)

    edge_v1_v2 = 1.0  # We fixed this
    edge_v1_v3 = edge_v1_v2 * np.sin(beta) / np.sin(gamma)
    edge_v2_v3 = edge_v1_v2 * np.sin(alpha) / np.sin(gamma)

    # v3 is at distance edge_v1_v3 from v1 = 0
    # and at distance edge_v2_v3 from v2 = 1
    # Solve for position using circles

    p1 = positions[v1]  # = 0
    p2 = positions[v2]  # = 1
    r1 = edge_v1_v3
    r2 = edge_v2_v3

    # Circle intersection: find p such that |p - p1| = r1 and |p - p2| = r2
    # Two solutions (above/below the real axis)

    d = abs(p2 - p1)  # = 1
    if r1 + r2 < d or r1 + d < r2 or r2 + d < r1:
        return {
            'success': False,
            'message': 'Triangle inequality violated for first triangle',
            'points': None,
            'vertex_list': vertex_list,
        }

    # Using formula for circle intersection
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h_sq = r1**2 - a**2

    if h_sq < 0:
        h = 0
    else:
        h = np.sqrt(h_sq)

    # Point along v1-v2 at distance a from v1
    p_mid = p1 + a * (p2 - p1) / d

    # Perpendicular direction (rotate by 90°)
    perp = (p2 - p1) / d * 1j  # Multiply by i to rotate 90°

    # Two solutions
    p3_option1 = p_mid + h * perp
    p3_option2 = p_mid - h * perp

    # Choose the one that gives positive orientation (counterclockwise)
    # For triangle (v1, v2, v3), we want the signed area to be positive
    def signed_area(p1, p2, p3):
        return 0.5 * ((p2 - p1).real * (p3 - p1).imag - (p2 - p1).imag * (p3 - p1).real)

    area1 = signed_area(p1, p2, p3_option1)
    area2 = signed_area(p1, p2, p3_option2)

    # Choose the option with positive area (or larger absolute area if both negative)
    if area1 > 0 or (area1 == 0 and area2 < 0):
        positions[v3] = p3_option1
    else:
        positions[v3] = p3_option2

    if verbose:
        print(f"  Placed v{v3} at {positions[v3]}")
        print(f"    Distances: |v{v1}-v{v3}| = {abs(positions[v3] - positions[v1]):.6f} (target: {r1:.6f})")
        print(f"               |v{v2}-v{v3}| = {abs(positions[v3] - positions[v2]):.6f} (target: {r2:.6f})")

    # Step 4: Incrementally place remaining vertices
    placed = {v1, v2, v3}
    remaining = set(vertex_list) - placed

    max_iterations = 100
    iteration = 0

    while remaining and iteration < max_iterations:
        iteration += 1

        placed_any = False

        for v in list(remaining):
            # Find a triangle containing v and at least two already-placed vertices
            for tri_id, (v0_t, v1_t, v2_t) in enumerate(triangles):
                tri_verts = [v0_t, v1_t, v2_t]
                if v not in tri_verts:
                    continue

                # Get the other two vertices
                others = [u for u in tri_verts if u != v]
                if not all(u in placed for u in others):
                    continue

                # We have a triangle with v and two placed vertices!
                u1, u2 = others

                # Get angles
                angle_at = {}
                for i, vt in enumerate(tri_verts):
                    angle_at[vt] = angles[tri_id][i]

                alpha = angle_at[v]
                beta = angle_at[u1]
                gamma = angle_at[u2]

                # Law of sines to get edge lengths
                edge_u1_u2 = abs(positions[u2] - positions[u1])
                edge_v_u1 = edge_u1_u2 * np.sin(gamma) / np.sin(alpha)
                edge_v_u2 = edge_u1_u2 * np.sin(beta) / np.sin(alpha)

                # Circle intersection
                p1 = positions[u1]
                p2 = positions[u2]
                r1 = edge_v_u1
                r2 = edge_v_u2
                d = abs(p2 - p1)

                # Check triangle inequality
                if r1 + r2 < d - 1e-10 or r1 + d < r2 - 1e-10 or r2 + d < r1 - 1e-10:
                    continue  # Skip this triangle, try another

                a = (r1**2 - r2**2 + d**2) / (2 * d)
                h_sq = r1**2 - a**2

                if h_sq < -1e-10:
                    continue
                elif h_sq < 0:
                    h = 0
                else:
                    h = np.sqrt(h_sq)

                p_mid = p1 + a * (p2 - p1) / d
                perp = (p2 - p1) / d * 1j

                p_option1 = p_mid + h * perp
                p_option2 = p_mid - h * perp

                # Choose the option that doesn't overlap with existing vertices
                # and maintains positive orientation
                min_dist_1 = min(abs(p_option1 - positions[u]) for u in placed)
                min_dist_2 = min(abs(p_option2 - positions[u]) for u in placed)

                # If one option is too close to an existing vertex, use the other
                if min_dist_1 < 1e-6 and min_dist_2 >= 1e-6:
                    positions[v] = p_option2
                elif min_dist_2 < 1e-6 and min_dist_1 >= 1e-6:
                    positions[v] = p_option1
                else:
                    # Both are valid, choose based on positive signed area
                    # Check orientation with the triangle we're using
                    area1 = (p2 - p1).real * (p_option1 - p1).imag - (p2 - p1).imag * (p_option1 - p1).real
                    area2 = (p2 - p1).real * (p_option2 - p1).imag - (p2 - p1).imag * (p_option2 - p1).real

                    # Choose the one that gives positive orientation
                    if abs(area1) > abs(area2):
                        positions[v] = p_option1 if area1 > 0 else p_option2
                    else:
                        positions[v] = p_option2 if area2 > 0 else p_option1

                placed.add(v)
                remaining.remove(v)
                placed_any = True

                if verbose:
                    print(f"  Placed v{v} at {positions[v]:.6f} using triangle {tri_verts}")

                break

            if v in placed:
                break

        if not placed_any:
            if verbose:
                print(f"  Could not place any more vertices. {len(remaining)} remaining.")
            break

    if remaining:
        return {
            'success': False,
            'message': f'Could not place all vertices ({len(remaining)} remaining)',
            'points': None,
            'vertex_list': vertex_list,
            'placed': list(placed),
            'remaining': list(remaining),
        }

    # Convert to numpy array
    points_array = np.zeros((n_vertices, 2))
    for i, v in enumerate(vertex_list):
        pos = positions[v]
        points_array[i, 0] = pos.real
        points_array[i, 1] = pos.imag

    if verbose:
        print(f"  ✓ Successfully placed all {n_vertices} vertices")

    return {
        'success': True,
        'points': points_array,
        'vertex_list': vertex_list,
        'message': 'Rigid construction successful',
    }
