"""
Rivin's algorithm for checking Delaunay realizability of a triangulation.

A triangulation can be realized as a Delaunay triangulation if and only if
there exists an assignment of angles satisfying:
(a) All angles are positive
(b) Sum of angles in each triangle equals π (or 1 after scaling)
(c) Sum of angles around each interior vertex equals 2π (or 2 after scaling)
(d) Sum of angles at each boundary vertex is ≤ π (or ≤ 1 after scaling)
(e) Sum of opposite angles across each interior edge is ≤ π (or ≤ 1 after scaling)

We check this via linear programming: maximize the minimum angle subject to
these constraints. If the optimal minimum angle is > 0, the triangulation
is Delaunay realizable.

References:
- Rivin, I. (1994). "Euclidean structures on simplicial surfaces and hyperbolic volume"
- Leibon, G. (2002). "Characterizing the Delaunay decompositions of compact hyperbolic surfaces"
"""

import numpy as np
from scipy.optimize import linprog, minimize
from collections import defaultdict
from typing import List, Tuple, Set, Dict, Optional
import random
import time


def extract_boundary_vertices(triangles: List[Tuple[int, int, int]]) -> Set[int]:
    """
    Extract boundary vertices from a triangulation.

    A vertex is on the boundary if it's incident to at least one edge that appears
    in exactly one triangle.

    Args:
        triangles: List of triangles, each given as (v0, v1, v2)

    Returns:
        Set of boundary vertex indices
    """
    edge_count = defaultdict(int)

    # Count how many times each edge appears
    for tri in triangles:
        v0, v1, v2 = tri
        # Store edges in canonical form (min vertex first)
        edges = [
            tuple(sorted([v0, v1])),
            tuple(sorted([v1, v2])),
            tuple(sorted([v2, v0]))
        ]
        for edge in edges:
            edge_count[edge] += 1

    # Boundary edges appear exactly once
    boundary_edges = {edge for edge, count in edge_count.items() if count == 1}

    # Collect all vertices on boundary edges
    boundary_vertices = set()
    for edge in boundary_edges:
        boundary_vertices.add(edge[0])
        boundary_vertices.add(edge[1])

    return boundary_vertices


def build_vertex_angle_incidence(triangles: List[Tuple[int, int, int]]) -> Dict[int, List[Tuple[int, int]]]:
    """
    Build mapping from vertex to all angles incident to it.

    Args:
        triangles: List of triangles, each given as (v0, v1, v2)

    Returns:
        Dict mapping vertex_id -> list of (triangle_id, corner_id) where corner_id in {0,1,2}
        indicates which corner of the triangle has this vertex
    """
    vertex_angles = defaultdict(list)

    for tri_id, tri in enumerate(triangles):
        for corner_id, vertex in enumerate(tri):
            vertex_angles[vertex].append((tri_id, corner_id))

    return vertex_angles


def build_edge_adjacency(triangles: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Build mapping from each edge to the angles opposite to it.

    For an interior edge shared by two triangles, we get the two opposite angles.
    For a boundary edge (in only one triangle), we get one opposite angle.

    Args:
        triangles: List of triangles, each given as (v0, v1, v2)

    Returns:
        Dict mapping edge (as sorted tuple) -> list of (triangle_id, corner_id) for opposite angles
        where corner_id is the corner opposite to this edge in the triangle
    """
    edge_opposite_angles = defaultdict(list)

    for tri_id, tri in enumerate(triangles):
        v0, v1, v2 = tri
        # For each edge, find the opposite corner
        edges_and_opposite = [
            (tuple(sorted([v0, v1])), 2),  # edge v0-v1, opposite corner is v2 (index 2)
            (tuple(sorted([v1, v2])), 0),  # edge v1-v2, opposite corner is v0 (index 0)
            (tuple(sorted([v2, v0])), 1),  # edge v2-v0, opposite corner is v1 (index 1)
        ]
        for edge, corner_id in edges_and_opposite:
            edge_opposite_angles[edge].append((tri_id, corner_id))

    return edge_opposite_angles


def find_interior_edges(triangles: List[Tuple[int, int, int]]) -> List[Tuple[int, int]]:
    """
    Find all interior edges (edges shared by exactly 2 triangles).

    Args:
        triangles: List of triangles, each given as (v0, v1, v2)

    Returns:
        List of interior edges as sorted tuples (v1, v2)
    """
    edge_opposite_angles = build_edge_adjacency(triangles)
    interior_edges = [edge for edge, angles in edge_opposite_angles.items() if len(angles) == 2]
    return interior_edges


def compute_rivin_angle_residuals(
    triangles: List[Tuple[int, int, int]],
    angles: np.ndarray,
    boundary_vertices: Optional[Set[int]] = None,
    *,
    tolerance: float = 1e-6,
) -> Dict:
    """Compute primal and inequality residuals for a Rivin angle structure.

    The returned margins use the convention that positive values are
    feasible and negative values are violations.  In particular,
    ``dihedral_margin`` is ``pi`` minus the two angles opposite an interior
    edge, and ``boundary_margin`` is ``pi`` minus the angle sum at a boundary
    vertex.

    This helper intentionally reports both ordinary feasibility (allowing
    equality up to ``tolerance``) and strict-interior status.  Callers that
    construct arithmetic witnesses should apply their own, usually larger,
    positive margin threshold rather than relying on floating-point equality.
    """
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be a finite nonnegative number")
    if not triangles:
        raise ValueError("triangles must be nonempty")

    angle_array = np.asarray(angles, dtype=float)
    expected_shape = (len(triangles), 3)
    if angle_array.shape != expected_shape:
        raise ValueError(
            f"angles must have shape {expected_shape}; got {angle_array.shape}"
        )
    if not np.all(np.isfinite(angle_array)):
        raise ValueError("angles must all be finite")

    if boundary_vertices is None:
        boundary_vertices = extract_boundary_vertices(triangles)
    else:
        boundary_vertices = set(boundary_vertices)

    all_vertices = {vertex for tri in triangles for vertex in tri}
    unknown_boundary = boundary_vertices - all_vertices
    if unknown_boundary:
        raise ValueError(
            "boundary_vertices contains vertices absent from triangles: "
            f"{sorted(unknown_boundary)}"
        )
    interior_vertices = all_vertices - boundary_vertices
    vertex_angles = build_vertex_angle_incidence(triangles)
    edge_opposite_angles = build_edge_adjacency(triangles)

    triangle_residuals = np.sum(angle_array, axis=1) - np.pi
    interior_vertex_residuals = {}
    for vertex in sorted(interior_vertices):
        angle_sum = sum(
            angle_array[tri_id, corner_id]
            for tri_id, corner_id in vertex_angles[vertex]
        )
        interior_vertex_residuals[vertex] = float(angle_sum - 2.0 * np.pi)

    dihedral_margins = {}
    for edge in sorted(edge_opposite_angles):
        opposite_angles = edge_opposite_angles[edge]
        if len(opposite_angles) == 2:
            angle_sum = sum(
                angle_array[tri_id, corner_id]
                for tri_id, corner_id in opposite_angles
            )
            dihedral_margins[edge] = float(np.pi - angle_sum)

    boundary_margins = {}
    for vertex in sorted(boundary_vertices):
        angle_sum = sum(
            angle_array[tri_id, corner_id]
            for tri_id, corner_id in vertex_angles[vertex]
        )
        boundary_margins[vertex] = float(np.pi - angle_sum)

    max_triangle_residual = float(np.max(np.abs(triangle_residuals)))
    max_vertex_residual = max(
        (abs(value) for value in interior_vertex_residuals.values()),
        default=0.0,
    )
    max_equality_residual = max(max_triangle_residual, max_vertex_residual)

    worst_dihedral_edge = min(
        dihedral_margins,
        key=lambda edge: (dihedral_margins[edge], edge),
        default=None,
    )
    min_dihedral_margin = (
        dihedral_margins[worst_dihedral_edge]
        if worst_dihedral_edge is not None
        else None
    )
    worst_boundary_vertex = min(
        boundary_margins,
        key=lambda vertex: (boundary_margins[vertex], vertex),
        default=None,
    )
    min_boundary_margin = (
        boundary_margins[worst_boundary_vertex]
        if worst_boundary_vertex is not None
        else None
    )

    inequality_margins = [*dihedral_margins.values(), *boundary_margins.values()]
    min_inequality_margin = min(inequality_margins, default=None)
    min_angle = float(np.min(angle_array))
    max_angle = float(np.max(angle_array))
    feasible = (
        max_equality_residual <= tolerance
        and min_angle >= -tolerance
        and (
            min_inequality_margin is None
            or min_inequality_margin >= -tolerance
        )
    )
    strict_interior = (
        max_equality_residual <= tolerance
        and min_angle > tolerance
        and (
            min_inequality_margin is None
            or min_inequality_margin > tolerance
        )
    )

    return {
        "feasible": bool(feasible),
        "strict_interior": bool(strict_interior),
        "tolerance": float(tolerance),
        "min_angle": min_angle,
        "max_angle": max_angle,
        "max_triangle_sum_residual": max_triangle_residual,
        "max_interior_vertex_sum_residual": float(max_vertex_residual),
        "max_equality_residual": float(max_equality_residual),
        "min_dihedral_margin": min_dihedral_margin,
        "worst_dihedral_edge": worst_dihedral_edge,
        "max_dihedral_angle": (
            float(np.pi - min_dihedral_margin)
            if min_dihedral_margin is not None
            else None
        ),
        "min_boundary_margin": min_boundary_margin,
        "worst_boundary_vertex": worst_boundary_vertex,
        "max_boundary_angle_sum": (
            float(np.pi - min_boundary_margin)
            if min_boundary_margin is not None
            else None
        ),
        "min_inequality_margin": min_inequality_margin,
        "n_triangles": len(triangles),
        "n_vertices": len(all_vertices),
        "n_interior_vertices": len(interior_vertices),
        "n_boundary_vertices": len(boundary_vertices),
        "n_interior_edges": len(dihedral_margins),
    }


def flip_edge(triangles: List[Tuple[int, int, int]], edge: Tuple[int, int]) -> Optional[List[Tuple[int, int, int]]]:
    """
    Perform an edge flip in a triangulation.

    Given two triangles sharing an edge, flip the edge to the other diagonal
    of the quadrilateral formed by the two triangles.

    For example:
        Triangles: (a, b, c) and (b, d, c) sharing edge (b, c)
        After flip: (a, b, d) and (a, d, c) sharing edge (a, d)

    Args:
        triangles: List of triangles as (v0, v1, v2) tuples
        edge: Edge to flip as (v1, v2) tuple (order doesn't matter)

    Returns:
        New list of triangles with the edge flipped, or None if edge is not flippable
    """
    # Normalize edge to sorted form
    edge_sorted = tuple(sorted(edge))

    # Find triangles incident to this edge
    edge_triangles = []
    other_triangles = []

    for tri in triangles:
        v0, v1, v2 = tri
        tri_edges = [
            tuple(sorted([v0, v1])),
            tuple(sorted([v1, v2])),
            tuple(sorted([v2, v0]))
        ]
        if edge_sorted in tri_edges:
            edge_triangles.append(tri)
        else:
            other_triangles.append(tri)

    # Must be exactly 2 triangles (interior edge)
    if len(edge_triangles) != 2:
        return None

    # Extract the two triangles
    tri1, tri2 = edge_triangles

    # Preserve the cyclic orientation of tri1.  Rotate it to (u, v, a),
    # where u-v is the directed shared edge.  A consistently oriented tri2
    # must then have cyclic form (v, u, d).  Refuse to repair inconsistent
    # input locally: silently doing so can hide a non-orientable or otherwise
    # malformed triangulation from callers that use ``flip_edge`` directly.
    u = v = a = None
    for i in range(3):
        candidate_u = tri1[i]
        candidate_v = tri1[(i + 1) % 3]
        if tuple(sorted((candidate_u, candidate_v))) == edge_sorted:
            u = candidate_u
            v = candidate_v
            a = tri1[(i + 2) % 3]
            break
    if u is None or v is None or a is None:
        return None

    tri2_u = tri2_v = d = None
    for i in range(3):
        candidate_u = tri2[i]
        candidate_v = tri2[(i + 1) % 3]
        if tuple(sorted((candidate_u, candidate_v))) == edge_sorted:
            tri2_u = candidate_u
            tri2_v = candidate_v
            d = tri2[(i + 2) % 3]
            break
    if tri2_u != v or tri2_v != u or d is None:
        return None
    if a == d:
        return None

    # CRITICAL CHECK (from cubic-graph-optimizer):
    # Verify the new edge (a, d) doesn't already exist
    new_edge = tuple(sorted([a, d]))
    for tri in other_triangles:
        v0, v1, v2 = tri
        tri_edges = [
            tuple(sorted([v0, v1])),
            tuple(sorted([v1, v2])),
            tuple(sorted([v2, v0]))
        ]
        if new_edge in tri_edges:
            # Edge already exists - flip would create invalid triangulation
            return None

    # Create new triangles with flipped edge (a, d), preserving tri1's local
    # cyclic orientation.  For tri1=(u,v,a), tri2=(v,u,d), the consistently
    # oriented replacements are (a,u,d) and (a,d,v).
    new_tri1 = (a, u, d)
    new_tri2 = (a, d, v)

    # Return new triangulation
    return other_triangles + [new_tri1, new_tri2]


def random_edge_flips(triangles: List[Tuple[int, int, int]], n_flips: int, seed: Optional[int] = None) -> List[Tuple[int, int, int]]:
    """
    Perform n random edge flips on a triangulation.

    Uses the flip_edge() function which validates each flip to ensure validity.

    Args:
        triangles: Initial triangulation
        n_flips: Number of edge flips to perform
        seed: Random seed for reproducibility

    Returns:
        Triangulation after n_flips random edge flips
    """
    if seed is not None:
        random.seed(seed)

    current = triangles
    successful_flips = 0

    # Try many more attempts than requested, since many edges aren't flippable
    for attempt in range(n_flips * 10):
        if successful_flips >= n_flips:
            break

        # Find all interior edges (edges in exactly 2 triangles)
        interior_edges = find_interior_edges(current)

        if not interior_edges:
            break

        # Pick a random interior edge
        edge = random.choice(interior_edges)

        # Try to flip it (will return None if not flippable)
        flipped = flip_edge(current, edge)

        if flipped is not None:
            current = flipped
            successful_flips += 1

    return current


def check_delaunay_realizability(triangles: List[Tuple[int, int, int]],
                                 boundary_vertices: Optional[Set[int]] = None,
                                 verbose: bool = False,
                                 strict: bool = False,
                                 andreev: bool = False) -> Dict:
    """
    Check if a triangulation can be realized as a Delaunay triangulation.

    Uses Rivin's criterion via linear programming on angle assignments.

    Constraints:
    (a) All angles are positive
    (b) Sum of angles in each triangle equals π (scaled to 1)
    (c) Sum of angles around each interior vertex equals 2π (scaled to 2)
    (d) Sum of angles at each boundary vertex is ≤ π (scaled to ≤ 1)
        - If andreev=True: ≤ π/2 (scaled to ≤ 0.5)
    (e) For each interior edge, sum of two opposite angles (dihedral angle) ≤ π (scaled to ≤ 1)
        - If strict=True: < π (achieved by maximizing slack ε > 0 such that sum ≤ 1 - ε)
        - If andreev=True: ≤ π/2 (scaled to ≤ 0.5)

    Args:
        triangles: List of triangles, each given as (v0, v1, v2) tuple of vertex indices
        boundary_vertices: Set of boundary vertex indices (if None, auto-detect)
        verbose: If True, print detailed information
        strict: If True, require dihedral angles to be strictly < π (not = π)
        andreev: If True, require dihedral angles ≤ π/2 (Andreev's theorem)

    Returns:
        Dict with keys:
        - 'realizable': bool, True if Delaunay realizable
        - 'min_angle': float, the maximum achievable minimum angle (>0 iff realizable)
        - 'slack': float, the dihedral angle slack (only for strict=True)
        - 'angles': np.ndarray of shape (n_triangles, 3), angle assignment (if realizable)
        - 'status': int, optimization status from scipy
        - 'message': str, status message
    """
    n_triangles = len(triangles)
    n_angles = 3 * n_triangles  # 3 angles per triangle

    if boundary_vertices is None:
        boundary_vertices = extract_boundary_vertices(triangles)

    # Get all vertices
    all_vertices = set()
    for tri in triangles:
        all_vertices.update(tri)

    interior_vertices = all_vertices - boundary_vertices

    if verbose:
        print(f"Triangulation: {n_triangles} triangles, {len(all_vertices)} vertices")
        print(f"  Interior vertices: {len(interior_vertices)}")
        print(f"  Boundary vertices: {len(boundary_vertices)}")

    # Build vertex -> angles incidence
    vertex_angles = build_vertex_angle_incidence(triangles)

    # Build edge -> opposite angles mapping
    edge_opposite_angles = build_edge_adjacency(triangles)

    # Classify edges as interior (shared by 2 triangles) or boundary (only 1 triangle)
    interior_edges = {edge: angles for edge, angles in edge_opposite_angles.items() if len(angles) == 2}
    boundary_edges = {edge: angles for edge, angles in edge_opposite_angles.items() if len(angles) == 1}

    # Determine dihedral angle bound based on mode
    if andreev:
        dihedral_bound = 0.5  # π/2 in normalized units
        boundary_bound = 0.5  # π/2 for boundary vertices too
        mode_desc = "Andreev (dihedral ≤ π/2)"
    else:
        dihedral_bound = 1.0  # π in normalized units
        boundary_bound = 1.0  # π for boundary vertices
        mode_desc = "strict (dihedral < π)" if strict else "standard (dihedral ≤ π)"

    # LP formulation:
    # Standard mode: maximize t (minimum angle)
    # Strict mode: maximize ε (dihedral slack), then t
    #
    # Variables:
    # - Standard: [angle[0,0], ..., angle[n-1,2], t]
    # - Strict: [angle[0,0], ..., angle[n-1,2], t, ε]
    #
    # Constraints:
    # - angle[i] >= t for all i (reformulated as -angle[i] + t <= 0)
    # - sum of angles in triangle = 1
    # - sum of angles around interior vertex = 2
    # - sum of angles around boundary vertex <= boundary_bound
    # - sum of two opposite angles across interior edge <= dihedral_bound - ε (if strict)
    #                                                     <= dihedral_bound (if not strict)

    if strict:
        n_vars = n_angles + 2  # angles + t + ε
        # Objective: lexicographic optimization - prioritize slack ε, then min angle t
        # We use a weighted combination: maximize ε with high weight + t with low weight
        # In practice, maximize 1000*ε + t  <=>  minimize -1000*ε - t
        c = np.zeros(n_vars)
        c[-2] = -1.0      # coefficient for t (min angle)
        c[-1] = -1000.0   # coefficient for ε (dihedral slack) - much higher priority
        t_idx = -2
        eps_idx = -1
    else:
        n_vars = n_angles + 1  # angles + t
        # Objective: maximize t <=> minimize -t
        c = np.zeros(n_vars)
        c[-1] = -1.0  # coefficient for t
        t_idx = -1
        eps_idx = None

    # Inequality constraints: A_ub @ x <= b_ub
    # (a) angle[i] >= t  <=>  -angle[i] + t <= 0
    A_ub_list = []
    b_ub_list = []

    for i in range(n_angles):
        row = np.zeros(n_vars)
        row[i] = -1.0        # -angle[i]
        row[t_idx] = 1.0     # +t
        A_ub_list.append(row)
        b_ub_list.append(0.0)

    # (d) sum of angles at boundary vertex <= boundary_bound (< boundary_bound if strict)
    for vertex in boundary_vertices:
        row = np.zeros(n_vars)
        for tri_id, corner_id in vertex_angles[vertex]:
            angle_idx = 3 * tri_id + corner_id
            row[angle_idx] = 1.0
        if strict:
            row[eps_idx] = 1.0  # +ε (so constraint is: sum + ε <= boundary_bound)
        A_ub_list.append(row)
        b_ub_list.append(boundary_bound)

    # (e) sum of two opposite angles across interior edge <= dihedral_bound - ε (Delaunay condition)
    for edge, opposite_angles in interior_edges.items():
        assert len(opposite_angles) == 2, f"Interior edge {edge} should have exactly 2 opposite angles"
        row = np.zeros(n_vars)
        for tri_id, corner_id in opposite_angles:
            angle_idx = 3 * tri_id + corner_id
            row[angle_idx] = 1.0
        if strict:
            row[eps_idx] = 1.0  # +ε (so constraint is: sum + ε <= dihedral_bound)
        A_ub_list.append(row)
        b_ub_list.append(dihedral_bound)

    # Note: For strict mode, ε >= 0 is enforced via bounds below, not as a constraint

    A_ub = np.array(A_ub_list) if A_ub_list else np.zeros((0, n_vars))
    b_ub = np.array(b_ub_list) if b_ub_list else np.zeros(0)

    # Equality constraints: A_eq @ x == b_eq
    # (b) sum of angles in triangle = 1
    # (c) sum of angles around interior vertex = 2
    A_eq_list = []
    b_eq_list = []

    for tri_id in range(n_triangles):
        row = np.zeros(n_vars)
        for corner_id in range(3):
            angle_idx = 3 * tri_id + corner_id
            row[angle_idx] = 1.0
        A_eq_list.append(row)
        b_eq_list.append(1.0)  # scaled: sum = π becomes sum = 1

    for vertex in interior_vertices:
        row = np.zeros(n_vars)
        for tri_id, corner_id in vertex_angles[vertex]:
            angle_idx = 3 * tri_id + corner_id
            row[angle_idx] = 1.0
        A_eq_list.append(row)
        b_eq_list.append(2.0)  # scaled: sum = 2π becomes sum = 2

    A_eq = np.array(A_eq_list)
    b_eq = np.array(b_eq_list)

    if verbose:
        print(f"\nLP formulation ({mode_desc}):")
        if strict:
            print(f"  Variables: {n_vars} (angles: {n_angles}, t: 1, ε: 1)")
        else:
            print(f"  Variables: {n_vars} (angles: {n_angles}, t: 1)")
        print(f"  Inequality constraints: {A_ub.shape[0]}")
        print(f"    - Angle >= t: {n_angles}")
        print(f"    - Boundary vertex angle sum <= {boundary_bound}: {len(boundary_vertices)}")
        if strict:
            print(f"    - Interior edge opposite angles sum + ε <= {dihedral_bound}: {len(interior_edges)}")
        else:
            print(f"    - Interior edge opposite angles sum <= {dihedral_bound}: {len(interior_edges)}")
        print(f"  Equality constraints: {A_eq.shape[0]}")
        print(f"    - Triangle angle sum = 1: {n_triangles}")
        print(f"    - Interior vertex angle sum = 2: {len(interior_vertices)}")
        print("\n  Edge classification:")
        print(f"    - Interior edges: {len(interior_edges)}")
        print(f"    - Boundary edges: {len(boundary_edges)}")

    # Bounds: angles and t can be any real number (LP will constrain them)
    # Actually, we can bound t from below by 0 for numerical stability
    # For strict mode, ε >= 0 (must have positive slack)
    if strict:
        bounds = [(None, None)] * n_angles + [(0, None), (0, None)]  # t >= 0, ε >= 0
    else:
        bounds = [(None, None)] * n_angles + [(0, None)]  # t >= 0

    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=bounds, method='highs')

    if verbose:
        print("\nOptimization result:")
        print(f"  Status: {result.status} - {result.message}")
        print(f"  Success: {result.success}")
        if result.success:
            if strict:
                print(f"  Optimal min angle (t): {result.x[t_idx]:.6f} (scaled units)")
                print(f"  In radians: {result.x[t_idx] * np.pi:.6f}")
                print(f"  Optimal dihedral slack (ε): {result.x[eps_idx]:.6f} (scaled units)")
                print(f"  In radians: {result.x[eps_idx] * np.pi:.6f}")
                print(f"  Max dihedral angle: {dihedral_bound - result.x[eps_idx]:.6f} (< {dihedral_bound})")
            else:
                print(f"  Optimal min angle: {result.x[t_idx]:.6f} (scaled units)")
                print(f"  In radians: {result.x[t_idx] * np.pi:.6f}")

    # Extract angles and other values if successful
    angles = None
    min_angle = 0.0
    slack = 0.0

    if result.success and result.x is not None:
        angles = result.x[:n_angles].reshape(n_triangles, 3)
        min_angle = result.x[t_idx]
        if strict:
            slack = result.x[eps_idx]

    # Realizability check:
    # - Standard/Andreev: realizable if min_angle > 0
    # - Strict: realizable if min_angle > 0 AND slack > 0
    if strict:
        realizable = result.success and min_angle > 1e-6 and slack > 1e-6
    else:
        realizable = result.success and min_angle > 1e-6  # small tolerance for numerical errors

    result_dict = {
        'realizable': realizable,
        'min_angle': min_angle,
        'min_angle_radians': min_angle * np.pi,
        'angles': angles,
        'angles_radians': angles * np.pi if angles is not None else None,
        'status': result.status,
        'message': result.message,
        'success': result.success,
        'n_triangles': n_triangles,
        'n_vertices': len(all_vertices),
        'n_interior': len(interior_vertices),
        'n_boundary': len(boundary_vertices),
        'mode': mode_desc,
        'strict': strict,
        'andreev': andreev,
    }

    # Add slack information for strict mode
    if strict:
        result_dict['slack'] = slack
        result_dict['slack_radians'] = slack * np.pi
        result_dict['max_dihedral'] = dihedral_bound - slack
        result_dict['max_dihedral_radians'] = (dihedral_bound - slack) * np.pi

    return result_dict


def format_realizability_report(result: Dict) -> str:
    """
    Format realizability check results as a readable string.

    Args:
        result: Dictionary returned by check_delaunay_realizability

    Returns:
        Formatted string report
    """
    report = []
    report.append("=" * 60)
    report.append("DELAUNAY REALIZABILITY CHECK (Rivin's Algorithm)")
    report.append("=" * 60)
    report.append(f"Triangulation: {result['n_triangles']} triangles, {result['n_vertices']} vertices")
    report.append(f"  Interior vertices: {result['n_interior']}")
    report.append(f"  Boundary vertices: {result['n_boundary']}")
    report.append("")

    if result['realizable']:
        report.append("✓ REALIZABLE as Delaunay triangulation")
        report.append(f"  Maximum min angle: {result['min_angle']:.6f} (scaled units)")
        report.append(f"  Maximum min angle: {result['min_angle_radians']:.6f} radians")
        report.append(f"  Maximum min angle: {np.degrees(result['min_angle_radians']):.2f}°")
    else:
        report.append("✗ NOT REALIZABLE as Delaunay triangulation")
        if result['success']:
            report.append(f"  Best min angle found: {result['min_angle']:.6f} (≤ 0)")
        else:
            report.append(f"  LP solver status: {result['message']}")

    report.append("")
    report.append(f"Solver status: {result['status']} - {result['message']}")
    report.append("=" * 60)

    return "\n".join(report)


def lobachevsky_function(theta: float) -> float:
    """
    Compute the Lobachevsky function Λ(θ) = -∫₀^θ log|2sin(t)| dt.

    This is related to the Bloch-Wigner dilogarithm and appears in
    hyperbolic volume computations.

    Args:
        theta: Angle in radians

    Returns:
        Value of Lobachevsky function
    """
    from scipy.integrate import quad

    if theta <= 0 or theta >= np.pi:
        return 0.0

    def integrand(t):
        return -np.log(2 * np.sin(t)) if np.sin(t) > 1e-10 else 0.0

    result, _ = quad(integrand, 0, theta, limit=100)
    return result


def lobachevsky_gradient(theta: float) -> float:
    """
    Compute the gradient of the Lobachevsky function: Λ'(θ) = -log(2sin(θ)).

    Args:
        theta: Angle in radians

    Returns:
        Gradient value
    """
    if theta <= 1e-10 or theta >= np.pi - 1e-10:
        # Avoid log(0) at boundaries
        return 0.0

    return -np.log(2 * np.sin(theta))


def lobachevsky_hessian(theta: float) -> float:
    """
    Compute the Hessian (second derivative) of the Lobachevsky function: Λ''(θ) = -cot(θ).

    Args:
        theta: Angle in radians

    Returns:
        Hessian value (diagonal element)
    """
    if theta <= 1e-10 or theta >= np.pi - 1e-10:
        # Avoid division by zero at boundaries
        return -1e10  # Large negative value (concave)

    return -1.0 / np.tan(theta)


def _newton_in_nullspace(x, N, verbose=False, tol=1e-12, max_iter=50, label=""):
    """Run Newton's method for maximizing V(θ) = Σ Λ(θᵢ) projected
    onto the column space of N (null space of equality constraints).

    Only uses Λ'(θ) = -log(2 sin θ) and Λ''(θ) = -cot θ.
    No Lobachevsky function evaluations.

    Returns ``(x, n_iter, converged, reduced_gradient_norm)``.
    """
    EPS = 1e-6

    converged = False
    for it in range(max_iter):
        x = np.clip(x, EPS, np.pi - EPS)
        sin_x = np.sin(x)
        cos_x = np.cos(x)

        grad = -np.log(2.0 * sin_x)
        g_red = N.T @ grad
        grad_norm = np.linalg.norm(g_red)

        if verbose and (it < 3 or it % 10 == 0 or grad_norm < tol):
            print(f"  {label}iter {it:3d}:  |∇V|_red = {grad_norm:.3e}")

        if grad_norm < tol:
            converged = True
            if verbose:
                print(f"  {label}Converged at iteration {it}")
            break

        H_diag = -cos_x / sin_x
        H_red = (N * H_diag[:, None]).T @ N

        try:
            delta_y = np.linalg.solve(H_red, -g_red)
        except np.linalg.LinAlgError:
            if verbose:
                print(f"  {label}Singular Hessian at iteration {it}")
            break

        delta_x = N @ delta_y

        # Backtracking line search: keep angles in (0, π)
        alpha = 1.0
        for _ in range(20):
            x_new = x + alpha * delta_x
            if np.all(x_new > EPS) and np.all(x_new < np.pi - EPS):
                break
            alpha *= 0.5
        x = x + alpha * delta_x

    # Report the residual at the returned iterate.  On an exhausted Newton
    # loop the last update happens after the norm above was computed.
    x = np.clip(x, EPS, np.pi - EPS)
    final_gradient = -np.log(2.0 * np.sin(x))
    reduced_gradient_norm = float(np.linalg.norm(N.T @ final_gradient))
    return x, it + 1, converged, reduced_gradient_norm


def optimize_hyperbolic_volume(triangles: List[Tuple[int, int, int]],
                               initial_angles: Optional[np.ndarray] = None,
                               boundary_vertices: Optional[Set[int]] = None,
                               verbose: bool = False,
                               require_interior: bool = False) -> Dict:
    """
    Find the volume-maximizing angle assignment for a Delaunay realizable
    triangulation via active-set Newton's method.

    Since V(θ) = Σ Λ(θᵢ) is strictly concave on the Rivin polytope, the
    unique maximum is found by:

      1. Newton in the null space of the equality constraints
         (triangle angle sums = π, interior vertex sums = 2π).
      2. Check inequality constraints (dihedral ≤ π, boundary sums ≤ π).
      3. If violated, pin the most-violated constraint at equality and
         repeat with the augmented constraint set.

    Only uses Λ'(θ) = -log(2 sin θ) and Λ''(θ) = -cot θ — no
    Lobachevsky function evaluations.  Each Newton solve converges
    quadratically in ~7 iterations.

    Note: when the optimum is in the interior of the Rivin polytope
    (no active inequality constraints), it is arithmetic.  When the
    optimum lies on the boundary (some dihedrals pinned at π), the
    extra Lagrange multipliers generically break arithmeticity.

    Args:
        triangles: List of triangles as (v0, v1, v2) tuples
        initial_angles: Initial angle assignment (n_triangles, 3) in radians.
            If None, solves LP for a feasible start.
        boundary_vertices: Set of boundary vertex indices (if None, auto-detect)
        verbose: If True, print optimization progress
        require_interior: If True, abort as soon as a DIHEDRAL (cocircular
            tie) constraint becomes active — i.e. the maximum lies on the
            boundary of the Rivin polytope.  Boundary maxima are perfectly
            legitimate volume answers, but they can never be zero-shear
            certified (the tie multipliers break the log-sin cancellation),
            so witness generation for the arithmeticity benchmark should
            reject them immediately and try another combinatorial type
            instead of paying for full convergence.  Boundary-VERTEX
            constraints becoming active are fine either way: their
            multipliers cancel in every shear expression.

    Returns:
        Dict with keys:
        - 'success': bool
        - 'angles': np.ndarray of shape (n_triangles, 3), optimal angles (radians)
        - 'volume': float (computed from the Fourier series at convergence)
        - 'n_active_constraints': int, number of pinned inequality constraints
        - 'message': str
        - 'n_iterations': int (total across all active-set rounds)
    """
    from scipy.linalg import null_space

    n_triangles = len(triangles)
    n_angles = 3 * n_triangles

    if boundary_vertices is None:
        boundary_vertices = extract_boundary_vertices(triangles)

    all_vertices = set()
    for tri in triangles:
        all_vertices.update(tri)
    interior_vertices = all_vertices - boundary_vertices

    vertex_angles = build_vertex_angle_incidence(triangles)
    edge_opposite_angles = build_edge_adjacency(triangles)
    interior_edges = {edge: angles for edge, angles in edge_opposite_angles.items()
                      if len(angles) == 2}

    # --- Feasible starting point from LP ---
    if initial_angles is None:
        if verbose:
            print("Solving LP for feasible starting point...")
        lp_result = check_delaunay_realizability(triangles, boundary_vertices, verbose=False)
        if not lp_result['realizable']:
            return {
                'success': False,
                'angles': None,
                'volume': None,
                'message': 'Triangulation is not Delaunay realizable',
                'n_active_constraints': 0,
                'active_constraint_labels': [],
                'n_iterations': 0,
                'reduced_gradient_norm': None,
                'residuals': None,
            }
        initial_angles = lp_result['angles_radians']

    x = initial_angles.flatten().copy()

    # --- Base equality constraints ---
    # Triangle angle sums = π,  interior vertex angle sums = 2π
    A_base_rows = []

    for tri_id in range(n_triangles):
        row = np.zeros(n_angles)
        for c in range(3):
            row[3 * tri_id + c] = 1.0
        A_base_rows.append(row)

    for vertex in interior_vertices:
        row = np.zeros(n_angles)
        for tri_id, corner_id in vertex_angles[vertex]:
            row[3 * tri_id + corner_id] = 1.0
        A_base_rows.append(row)

    A_base = np.array(A_base_rows)

    # --- Build all inequality constraint rows (for checking violations) ---
    # Each is a row a such that a @ x <= π
    ineq_rows = []
    ineq_labels = []

    for edge, opposite_angles in interior_edges.items():
        row = np.zeros(n_angles)
        for tri_id, corner_id in opposite_angles:
            row[3 * tri_id + corner_id] = 1.0
        ineq_rows.append(row)
        ineq_labels.append(f"dihedral {edge}")

    n_dihedral_rows = len(ineq_rows)  # dihedral rows come first

    for vertex in boundary_vertices:
        row = np.zeros(n_angles)
        for tri_id, corner_id in vertex_angles[vertex]:
            row[3 * tri_id + corner_id] = 1.0
        ineq_rows.append(row)
        ineq_labels.append(f"boundary vertex {vertex}")

    # --- Active-set loop ---
    active_indices = []      # indices into ineq_rows that are pinned at π
    total_iters = 0
    converged = False
    reduced_gradient_norm = None
    MAX_ACTIVE_ROUNDS = len(ineq_rows) + 1  # can't activate more than all

    # RHS for base equality constraints
    b_base = np.array(
        [np.pi] * n_triangles + [2 * np.pi] * len(interior_vertices))

    for active_round in range(MAX_ACTIVE_ROUNDS):
        # Build augmented equality constraint matrix and RHS
        if active_indices:
            A_active = np.array([ineq_rows[i] for i in active_indices])
            A_eq = np.vstack([A_base, A_active])
            b_eq = np.concatenate([b_base, np.full(len(active_indices), np.pi)])
        else:
            A_eq = A_base
            b_eq = b_base

        # Project x onto the augmented constraint manifold:
        # x -= A_eq^T (A_eq A_eq^T)^{-1} (A_eq x - b_eq)
        residual = A_eq @ x - b_eq
        if np.linalg.norm(residual) > 1e-12:
            x -= A_eq.T @ np.linalg.solve(A_eq @ A_eq.T, residual)

        N = null_space(A_eq)

        if N.shape[1] == 0:
            # Fully determined — no free dimensions left
            converged = True
            reduced_gradient_norm = 0.0
            if verbose:
                print(f"  No free dimensions (round {active_round}), "
                      f"system fully determined")
            break

        if verbose:
            if active_round == 0:
                print(f"\nNewton in null space: {n_angles} angles, "
                      f"{A_eq.shape[0]} equality constraints, "
                      f"{N.shape[1]} free dims")
            else:
                print(f"\n  Active-set round {active_round}: "
                      f"{len(active_indices)} pinned, "
                      f"{N.shape[1]} free dims")

        x, n_it, converged, reduced_gradient_norm = _newton_in_nullspace(
            x, N, verbose=verbose,
            label=f"[r{active_round}] " if active_round > 0 else "")
        total_iters += n_it

        if not converged:
            break

        # Check all inequality constraints: a @ x <= π
        worst_violation = 0.0
        worst_idx = -1
        for i, row in enumerate(ineq_rows):
            if i in active_indices:
                continue  # already pinned
            violation = row @ x - np.pi
            if violation > worst_violation:
                worst_violation = violation
                worst_idx = i

        if worst_violation <= 1e-6:
            # All inequalities satisfied — done
            if verbose and active_round > 0:
                print(f"  All inequalities satisfied after "
                      f"{active_round} activation(s)")
            break

        # Early tie-rejection: a dihedral about to be pinned means the
        # maximum is a boundary (cocircular-tie) maximum, which can never
        # be zero-shear certified — bail out now rather than converge.
        if require_interior and worst_idx < n_dihedral_rows:
            rejected_angles = x.reshape(n_triangles, 3)
            angle_residuals = compute_rivin_angle_residuals(
                triangles,
                rejected_angles,
                boundary_vertices,
            )
            if verbose:
                print(f"  REJECT (require_interior): {ineq_labels[worst_idx]} "
                      f"hit π (excess {np.degrees(worst_violation):.2f}°) — "
                      f"boundary maximum, not zero-shear certifiable")
            return {
                'success': False,
                'angles': rejected_angles,
                'volume': None,
                'max_dihedral_angle': float(ineq_rows[worst_idx] @ x),
                'n_active_constraints': len(active_indices),
                'active_constraint_labels': [
                    ineq_labels[index] for index in active_indices
                ],
                'message': (f'Rejected: cocircular tie active '
                            f'({ineq_labels[worst_idx]}) — boundary maximum '
                            f'(require_interior)'),
                'rejected_tie': True,
                'n_iterations': total_iters,
                'reduced_gradient_norm': reduced_gradient_norm,
                'residuals': angle_residuals,
            }

        # Pin the most-violated constraint at equality
        active_indices.append(worst_idx)
        if verbose:
            print(f"  Violated: {ineq_labels[worst_idx]} "
                  f"(excess {np.degrees(worst_violation):.2f}°), "
                  f"pinning at π")

    optimal_angles = x.reshape(n_triangles, 3)

    # --- Compute volume via Fourier series ---
    k = np.arange(1, 65)
    volume = 0.5 * np.sum(np.sin(2.0 * np.outer(x, k)) / (k * k))

    # --- Final feasibility check ---
    max_dihedral = 0.0
    for edge, opposite_angles in interior_edges.items():
        (tri1, c1), (tri2, c2) = opposite_angles
        dihedral = optimal_angles[tri1, c1] + optimal_angles[tri2, c2]
        max_dihedral = max(max_dihedral, dihedral)

    success = converged and (max_dihedral <= np.pi + 1e-6)
    angle_residuals = compute_rivin_angle_residuals(
        triangles,
        optimal_angles,
        boundary_vertices,
    )

    # Asymptotically tight tie (converged onto the boundary without a pin):
    # equally not zero-shear certifiable.
    if require_interior and success and max_dihedral >= np.pi - 1e-9:
        return {
            'success': False,
            'angles': optimal_angles,
            'volume': volume,
            'max_dihedral_angle': max_dihedral,
            'n_active_constraints': len(active_indices),
            'active_constraint_labels': [
                ineq_labels[index] for index in active_indices
            ],
            'message': ('Rejected: dihedral within 1e-9 of π at convergence '
                        '— boundary maximum (require_interior)'),
            'rejected_tie': True,
            'n_iterations': total_iters,
            'reduced_gradient_norm': reduced_gradient_norm,
            'residuals': angle_residuals,
        }

    if verbose:
        print(f"\n  Volume: {volume:.6f}")
        print(f"  Max dihedral: {max_dihedral:.6f} rad ({np.degrees(max_dihedral):.2f}°)")
        print(f"  Active constraints: {len(active_indices)}")
        print(f"  Total Newton iterations: {total_iters}")
        print(f"  Success: {success}")

    return {
        'success': success,
        'angles': optimal_angles,
        'volume': volume,
        'max_dihedral_angle': max_dihedral,
        'n_active_constraints': len(active_indices),
        'active_constraint_labels': [
            ineq_labels[index] for index in active_indices
        ],
        'message': 'Converged' if success else 'Did not converge',
        'n_iterations': total_iters,
        'reduced_gradient_norm': reduced_gradient_norm,
        'residuals': angle_residuals,
    }


def legalize_to_interior_maximum(
    triangles: List[Tuple[int, int, int]],
    boundary_vertices: Optional[Set[int]] = None,
    *,
    max_flips: int = 64,
    margin_tolerance: float = 1e-4,
    constraint_tolerance: float = 1e-6,
    verbose: bool = False,
) -> Dict:
    """Legalize a finite Delaunay disk toward a strict Rivin maximum.

    A rejected unconstrained maximum identifies an interior edge whose
    opposite-angle sum reaches or exceeds ``pi``.  Flipping the worst such
    edge and resolving is the angle-structure analogue of deterministic
    Delaunay legalization.  The process is bounded by ``max_flips`` and by
    orientation-independent cycle detection.

    Acceptance is deliberately stronger than
    ``optimize_hyperbolic_volume(..., require_interior=True)``: the optimizer
    must converge with no active constraints, equality residuals and angle
    positivity must satisfy ``constraint_tolerance``, and both interior-edge
    and boundary-vertex margins must be at least ``margin_tolerance``.  The
    boundary test certifies strictness after the disk is coned to its implicit
    vertex at infinity.

    Returns a structured dictionary.  ``finite_triangulation`` always holds
    the last valid oriented disk, including on failure.  ``termination_reason``
    is one of ``accepted``, ``max_flips``, ``cycle_detected``, ``lp_failure``,
    ``optimizer_failure``, ``residual_failure``, ``insufficient_margin``, or
    ``unflippable_edge``.
    """
    if isinstance(max_flips, bool) or int(max_flips) != max_flips or max_flips < 0:
        raise ValueError("max_flips must be a nonnegative integer")
    max_flips = int(max_flips)
    for name, value in (
        ("margin_tolerance", margin_tolerance),
        ("constraint_tolerance", constraint_tolerance),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")

    current = []
    for face in triangles:
        if len(face) != 3:
            raise ValueError(f"each triangle must contain three vertices: {face}")
        normalized_face = []
        for vertex in face:
            if isinstance(vertex, bool) or int(vertex) != vertex or vertex < 0:
                raise ValueError(
                    f"triangle vertices must be nonnegative integers: {face}"
                )
            normalized_face.append(int(vertex))
        oriented_face = tuple(normalized_face)
        if len(set(oriented_face)) != 3:
            raise ValueError(f"triangle vertices must be distinct: {face}")
        current.append(oriented_face)
    if not current:
        raise ValueError("triangles must be nonempty")

    canonical_faces = [tuple(sorted(face)) for face in current]
    if len(set(canonical_faces)) != len(canonical_faces):
        raise ValueError("triangles contains a duplicate face")

    all_vertices = {vertex for face in current for vertex in face}
    n_finite_vertices = len(all_vertices)
    if all_vertices != set(range(n_finite_vertices)):
        raise ValueError(
            "triangle vertex labels must be contiguous from 0 through "
            f"{n_finite_vertices - 1}"
        )

    # The boundary-completion service owns the project's strict finite-disk
    # validation: one connected boundary cycle, Euler characteristic one,
    # manifold vertex links, face connectivity, and orientability.  Its
    # finite prefix is also a coherently wound copy of the input disk.
    # Import locally to keep this legacy numerical module independent at
    # import time.
    from .optimization import complete_boundary_triangulation

    completed = complete_boundary_triangulation(
        np.asarray(current, dtype=np.int64),
        n_finite_vertices,
    )
    current = [
        tuple(map(int, face)) for face in completed[:len(current)]
    ]
    actual_boundary = extract_boundary_vertices(current)

    if boundary_vertices is None:
        fixed_boundary = actual_boundary
    else:
        fixed_boundary = set()
        for vertex in boundary_vertices:
            if isinstance(vertex, bool) or int(vertex) != vertex or vertex < 0:
                raise ValueError(
                    "boundary_vertices must contain nonnegative integer labels"
                )
            fixed_boundary.add(int(vertex))
        if fixed_boundary != actual_boundary:
            raise ValueError(
                "boundary_vertices must exactly match the triangulated disk "
                f"boundary {sorted(actual_boundary)}"
            )

    start_time = time.perf_counter()
    lp_seconds = 0.0
    optimization_seconds = 0.0
    history = []
    seen_states = set()
    flips_completed = 0
    final_optimization = None
    final_residuals = None

    def finish(reason, message, *, optimization=None, residuals=None):
        result_angles = None if optimization is None else optimization.get("angles")
        result_volume = None if optimization is None else optimization.get("volume")
        return {
            "success": reason == "accepted",
            "termination_reason": reason,
            "message": message,
            "finite_triangulation": list(current),
            "angles": result_angles,
            "volume": result_volume,
            "n_flips": flips_completed,
            "n_states": len(history),
            "residuals": residuals,
            "optimization_result": optimization,
            "history": history,
            "timings": {
                "total_seconds": float(time.perf_counter() - start_time),
                "lp_seconds": float(lp_seconds),
                "optimization_seconds": float(optimization_seconds),
            },
        }

    while True:
        state_key = tuple(sorted(tuple(sorted(face)) for face in current))
        if state_key in seen_states:
            return finish(
                "cycle_detected",
                "Legalization revisited a previously evaluated triangulation",
            )
        seen_states.add(state_key)

        step = flips_completed
        lp_start = time.perf_counter()
        lp_result = check_delaunay_realizability(
            current,
            boundary_vertices=fixed_boundary,
            verbose=False,
        )
        lp_elapsed = time.perf_counter() - lp_start
        lp_seconds += lp_elapsed
        history_row = {
            "step": step,
            "lp_realizable": bool(lp_result["realizable"]),
            "lp_min_angle": float(lp_result.get("min_angle", 0.0)),
            "lp_slack": (
                float(lp_result["slack"])
                if lp_result.get("slack") is not None
                else None
            ),
            "lp_seconds": float(lp_elapsed),
            "optimizer_success": None,
            "optimizer_message": None,
            "optimizer_iterations": None,
            "optimizer_active_constraints": None,
            "optimizer_reduced_gradient_norm": None,
            "optimizer_rejected_tie": False,
            "optimization_seconds": 0.0,
            "residuals": None,
            "flipped_edge": None,
            "status": None,
        }
        if not lp_result["realizable"]:
            history_row["status"] = "lp_failure"
            history.append(history_row)
            return finish(
                "lp_failure",
                f"Rivin feasibility LP failed: {lp_result['message']}",
            )

        optimization_start = time.perf_counter()
        optimization_result = optimize_hyperbolic_volume(
            current,
            initial_angles=lp_result["angles_radians"],
            boundary_vertices=fixed_boundary,
            verbose=False,
            require_interior=True,
        )
        optimization_elapsed = time.perf_counter() - optimization_start
        optimization_seconds += optimization_elapsed
        optimized_angles = optimization_result.get("angles")
        if optimized_angles is not None:
            # Always recompute with the caller's tolerance.  Optimizer-owned
            # residuals use its default tolerance and must not decide this
            # higher-level acceptance contract when the caller overrides it.
            final_residuals = compute_rivin_angle_residuals(
                current,
                optimized_angles,
                fixed_boundary,
                tolerance=constraint_tolerance,
            )
            optimization_result = {
                **optimization_result,
                "residuals": final_residuals,
            }
        else:
            final_residuals = None
        final_optimization = optimization_result

        history_row.update({
            "optimizer_success": bool(optimization_result["success"]),
            "optimizer_message": optimization_result.get("message"),
            "optimizer_iterations": optimization_result.get("n_iterations"),
            "optimizer_active_constraints": optimization_result.get(
                "n_active_constraints"
            ),
            "optimizer_reduced_gradient_norm": optimization_result.get(
                "reduced_gradient_norm"
            ),
            "optimizer_rejected_tie": bool(
                optimization_result.get("rejected_tie", False)
            ),
            "optimization_seconds": float(optimization_elapsed),
            "residuals": final_residuals,
        })

        if optimization_result["success"]:
            equality_ok = (
                final_residuals is not None
                and final_residuals["max_equality_residual"]
                <= constraint_tolerance
                and final_residuals["min_angle"] > constraint_tolerance
            )
            if not equality_ok:
                history_row["status"] = "residual_failure"
                history.append(history_row)
                return finish(
                    "residual_failure",
                    "Optimizer converged but equality or angle residuals failed",
                    optimization=final_optimization,
                    residuals=final_residuals,
                )

            dihedral_margin = final_residuals["min_dihedral_margin"]
            boundary_margin = final_residuals["min_boundary_margin"]
            margins_ok = (
                (dihedral_margin is None or dihedral_margin >= margin_tolerance)
                and (boundary_margin is None or boundary_margin >= margin_tolerance)
                and optimization_result.get("n_active_constraints", 0) == 0
            )
            if not margins_ok:
                history_row["status"] = "insufficient_margin"
                history.append(history_row)
                return finish(
                    "insufficient_margin",
                    "Optimizer converged without the requested strict margins",
                    optimization=final_optimization,
                    residuals=final_residuals,
                )

            history_row["status"] = "accepted"
            history.append(history_row)
            if verbose:
                print(
                    f"Accepted strict interior maximum after "
                    f"{flips_completed} flip(s)"
                )
            return finish(
                "accepted",
                "Strict interior Rivin maximum found",
                optimization=final_optimization,
                residuals=final_residuals,
            )

        if not optimization_result.get("rejected_tie", False):
            history_row["status"] = "optimizer_failure"
            history.append(history_row)
            return finish(
                "optimizer_failure",
                f"Volume optimizer failed: {optimization_result.get('message')}",
                optimization=final_optimization,
                residuals=final_residuals,
            )

        worst_edge = (
            None
            if final_residuals is None
            else final_residuals.get("worst_dihedral_edge")
        )
        if worst_edge is None:
            history_row["status"] = "optimizer_failure"
            history.append(history_row)
            return finish(
                "optimizer_failure",
                "Tie rejection did not identify an interior edge",
                optimization=final_optimization,
                residuals=final_residuals,
            )
        if flips_completed >= max_flips:
            history_row["status"] = "max_flips"
            history.append(history_row)
            return finish(
                "max_flips",
                f"Reached the maximum of {max_flips} edge flips",
                optimization=final_optimization,
                residuals=final_residuals,
            )

        flipped = flip_edge(current, worst_edge)
        history_row["flipped_edge"] = worst_edge
        if flipped is None:
            history_row["status"] = "unflippable_edge"
            history.append(history_row)
            return finish(
                "unflippable_edge",
                f"Worst violating edge {worst_edge} is not flippable",
                optimization=final_optimization,
                residuals=final_residuals,
            )

        history_row["status"] = "flipped"
        history.append(history_row)
        current = flipped
        flips_completed += 1
        if verbose:
            margin = final_residuals.get("min_dihedral_margin")
            print(
                f"Flip {flips_completed}/{max_flips}: edge {worst_edge}, "
                f"dihedral margin {margin:.3e}"
            )


def compute_triangle_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Compute the angle at vertex p1 in triangle (p1, p2, p3).

    Args:
        p1, p2, p3: 2D points as arrays

    Returns:
        Angle at p1 in radians
    """
    # Normalize each ray before taking dot and cross products.  ``hypot`` is
    # scale-stable for very small and very large coordinates, while arccos of
    # a rounded cosine loses near-collinear angles.  A zero-length ray is a
    # degenerate triangle; returning zero keeps optimization penalties finite
    # and makes that behavior explicit rather than relying on atan2(0, 0).
    v1 = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    v2 = np.asarray(p3, dtype=float) - np.asarray(p1, dtype=float)
    if v1.shape != (2,) or v2.shape != (2,):
        raise ValueError("triangle points must be two-dimensional vectors")

    norm1 = float(np.hypot(v1[0], v1[1]))
    norm2 = float(np.hypot(v2[0], v2[1]))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    unit1 = v1 / norm1
    unit2 = v2 / norm2
    cross = float(unit1[0] * unit2[1] - unit1[1] * unit2[0])
    dot = float(np.dot(unit1, unit2))
    return float(np.arctan2(abs(cross), dot))


def realize_angles_as_points(triangles: List[Tuple[int, int, int]],
                            target_angles: np.ndarray,
                            boundary_vertices: Optional[Set[int]] = None,
                            verbose: bool = False) -> Dict:
    """
    Construct a geometric realization of a triangulation with given angles.

    Given target angles for each corner of each triangle, find 2D point positions
    that realize those angles (approximately).

    This is an inverse problem: angles → geometry. We solve it via optimization.

    Args:
        triangles: List of triangles as (v0, v1, v2) tuples
        target_angles: Array of shape (n_triangles, 3) with target angles in radians
        boundary_vertices: Set of boundary vertices (auto-detected if None)
        verbose: If True, print progress

    Returns:
        Dict with keys:
        - 'success': bool
        - 'points': np.ndarray of shape (n_vertices, 2) with 2D coordinates
        - 'angle_error': float, RMS error in angles
        - 'message': str
    """
    if boundary_vertices is None:
        boundary_vertices = extract_boundary_vertices(triangles)
    target_angle_residuals = compute_rivin_angle_residuals(
        triangles,
        target_angles,
        boundary_vertices,
    )

    # Get all vertices
    all_vertices = set()
    for tri in triangles:
        all_vertices.update(tri)
    n_vertices = len(all_vertices)

    # Create vertex index mapping
    vertex_list = sorted(all_vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}

    if verbose:
        print("Reconstructing geometry from angles...")
        print(f"  Vertices: {n_vertices}")
        print(f"  Triangles: {len(triangles)}")

    # Fix translation, rotation, and scale by setting:
    # - First vertex at origin (0, 0) - fixes translation (2 DOF)
    # - Second vertex at (1, 0) - fixes rotation and scale (2 DOF)
    # This gives us exactly 4 constraints for 4 degrees of freedom

    v0 = vertex_list[0]
    v1 = vertex_list[1]

    fixed_points = {}
    fixed_points[v0] = np.array([0.0, 0.0])
    fixed_points[v1] = np.array([1.0, 0.0])

    # Identify free vertices (all except first two)
    free_vertices = [v for v in vertex_list if v not in fixed_points]
    n_free = len(free_vertices)

    if verbose:
        print(f"  Fixed vertices: {len(fixed_points)} (at 0 and 1)")
        print(f"  Free vertices: {n_free}")

    # Incremental graph-based construction
    # Place vertices one by one using angle constraints from already-placed neighbors
    if verbose:
        print("  Constructing geometry incrementally using angle constraints...")

    from scipy.spatial import Delaunay as scipy_Delaunay
    expected_triangulation_set = set(tuple(sorted(tri)) for tri in triangles)

    # Developing-map layout (exact similar-triangle construction).
    #
    # The previous implementation placed each new vertex at the apex of an
    # ISOSCELES triangle over the base edge (height = (L/2)/tan(theta/2)),
    # which satisfies the apex angle but generically violates the other two
    # prescribed angles by tens of degrees, and chose the side of the edge
    # arbitrarily (the pos1/pos2 "scores" were identical).  That produced
    # the 27-38 degree angle errors observed on 50/100-vertex Rivin-optimal
    # angle structures.
    #
    # The correct construction develops the triangulated disk across shared
    # edges: each triangle's shape is determined exactly by its three target
    # angles (law of sines), and the new vertex is placed on the OPPOSITE
    # side of the shared edge from the already-placed triangle, giving a
    # consistent orientation.  At Rivin's critical point the log-sin edge
    # conditions are precisely the closure conditions of this developing
    # map, so re-visited vertices reproduce their positions up to numerical
    # error (reported as closure_error).

    closure_error = [0.0]

    def construct_from_angles():
        """Develop the triangulation across shared edges with exact
        per-triangle shapes given by target_angles.  Returns the point
        array, or None if the dual graph is disconnected."""
        points = np.zeros((n_vertices, 2))

        # edge -> list of (tri_id, opposite_vertex)
        edge_tris = {}
        for tid, tri in enumerate(triangles):
            for k in range(3):
                e = tuple(sorted((tri[k], tri[(k + 1) % 3])))
                edge_tris.setdefault(e, []).append((tid, tri[(k + 2) % 3]))

        def angle_at(tid, v):
            tri = triangles[tid]
            return target_angles[tid][list(tri).index(v)]

        def third_position(tid, u, v, w, side_sign):
            """Position of w in triangle tid given placed u, v; side_sign
            = +1 places w to the left of the directed edge u->v."""
            pu = points[vertex_to_idx[u]]
            pv = points[vertex_to_idx[v]]
            au = angle_at(tid, u)
            aw = angle_at(tid, w)
            av = angle_at(tid, v)
            d = pv - pu
            L = np.hypot(d[0], d[1])
            if L < 1e-14 or np.sin(aw) < 1e-14:
                return None
            r = L * np.sin(av) / np.sin(aw)  # |uw| by the law of sines
            c, s = np.cos(au), np.sin(au) * side_sign
            rot = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]]) / L
            return pu + r * rot

        placed_vertices = set()
        placed_tris = set()

        # Seed: first triangle, first edge along (0,0)-(1,0), apex above.
        t0 = 0
        u0, v0_, w0 = triangles[t0]
        points[vertex_to_idx[u0]] = np.array([0.0, 0.0])
        points[vertex_to_idx[v0_]] = np.array([1.0, 0.0])
        pw = third_position(t0, u0, v0_, w0, +1.0)
        if pw is None:
            return None
        points[vertex_to_idx[w0]] = pw
        placed_vertices.update((u0, v0_, w0))
        placed_tris.add(t0)

        from collections import deque
        queue = deque([t0])
        while queue:
            t1 = queue.popleft()
            tri1 = triangles[t1]
            for k in range(3):
                a, b = tri1[k], tri1[(k + 1) % 3]
                e = tuple(sorted((a, b)))
                for (t2, w2) in edge_tris[e]:
                    if t2 in placed_tris:
                        continue
                    # third vertex of t1 across this edge
                    w1 = tri1[(k + 2) % 3]
                    pa = points[vertex_to_idx[a]]
                    pb = points[vertex_to_idx[b]]
                    pw1 = points[vertex_to_idx[w1]]
                    dx = pb - pa
                    cross1 = dx[0] * (pw1[1] - pa[1]) - dx[1] * (pw1[0] - pa[0])
                    side_sign = -1.0 if cross1 > 0 else +1.0
                    pos = third_position(t2, a, b, w2, side_sign)
                    if pos is None:
                        return None
                    idx = vertex_to_idx[w2]
                    if w2 in placed_vertices:
                        closure_error[0] = max(
                            closure_error[0],
                            float(np.hypot(*(points[idx] - pos))))
                    else:
                        points[idx] = pos
                        placed_vertices.add(w2)
                    placed_tris.add(t2)
                    queue.append(t2)

        if len(placed_tris) < len(triangles) or len(placed_vertices) < n_vertices:
            if verbose:
                print(f"    Developing map incomplete: "
                      f"{len(placed_tris)}/{len(triangles)} triangles, "
                      f"{len(placed_vertices)}/{n_vertices} vertices")
            return None

        if verbose:
            print(f"    Developing map complete; closure error = "
                  f"{closure_error[0]:.3e}")
        return points

    points_constructed = construct_from_angles()

    if points_constructed is None:
        # Construction failed
        if verbose:
            print("    ✗ Incremental construction failed")
        return {
            'success': False,
            'points': None,
            'vertex_list': vertex_list,
            'volume': None,
            'triangulation_preserved': False,
            'angle_error': None,
            'angle_error_degrees': None,
            'max_angle_error': None,
            'max_angle_error_degrees': None,
            'closure_error': None,
            'orientation_consistent': False,
            'min_abs_oriented_area': None,
            'residuals': {
                'rms_angle_error': None,
                'max_angle_error': None,
                'closure_error': None,
                'orientation_consistent': False,
                'min_abs_oriented_area': None,
                'target_angle_structure': target_angle_residuals,
                'realized_angle_structure': None,
            },
            'message': 'Incremental construction failed',
            'n_iterations': 0,
        }

    points = points_constructed

    def triangulation_is_delaunay(pts, tol=1e-7):
        """Is the TARGET triangulation a Delaunay triangulation of pts?

        Robust to ties: at Rivin's volume maximum some interior edges are
        exactly cocircular (active constraints), so scipy's Delaunay picks
        an arbitrary diagonal there and a strict set-equality test fails
        spuriously.  By Delaunay's lemma, an embedded triangulation with a
        convex boundary is (a) Delaunay iff every interior edge is locally
        Delaunay; with angles available, "locally Delaunay" is exactly
        opposite-angle sum <= pi (+ tol for ties).
        Checks: (1) all triangles consistently oriented (embedding),
        (2) every interior edge locally Delaunay within tol,
        (3) boundary convex (angle sums at boundary vertices <= pi + tol).
        """
        # (1) consistent orientation
        signs = []
        for (a, b, c) in triangles:
            pa, pb, pc = (pts[vertex_to_idx[a]], pts[vertex_to_idx[b]],
                          pts[vertex_to_idx[c]])
            cr = (pb[0]-pa[0])*(pc[1]-pa[1]) - (pb[1]-pa[1])*(pc[0]-pa[0])
            signs.append(np.sign(cr))
            if abs(cr) < 1e-14:
                return False
        if len(set(signs)) > 1:
            return False
        # actual corner angles from the layout
        def corner(pa, pb, pc):
            return compute_triangle_angle(pa, pb, pc)
        # (2) local Delaunay per interior edge
        edge_opp = {}
        for tid, (a, b, c) in enumerate(triangles):
            tri_pts = {a: pts[vertex_to_idx[a]], b: pts[vertex_to_idx[b]],
                       c: pts[vertex_to_idx[c]]}
            for (x, y, o) in ((a, b, c), (b, c, a), (c, a, b)):
                e = tuple(sorted((x, y)))
                ang = corner(tri_pts[o], tri_pts[x], tri_pts[y])
                edge_opp.setdefault(e, []).append(ang)
        vertex_sums = {}
        for tid, (a, b, c) in enumerate(triangles):
            pa, pb, pc = (pts[vertex_to_idx[a]], pts[vertex_to_idx[b]],
                          pts[vertex_to_idx[c]])
            vertex_sums[a] = vertex_sums.get(a, 0.0) + corner(pa, pb, pc)
            vertex_sums[b] = vertex_sums.get(b, 0.0) + corner(pb, pc, pa)
            vertex_sums[c] = vertex_sums.get(c, 0.0) + corner(pc, pa, pb)
        for e, angs in edge_opp.items():
            if len(angs) == 2 and angs[0] + angs[1] > np.pi + tol:
                return False
        # (3) convex boundary
        for v in boundary_vertices:
            if vertex_sums.get(v, 0.0) > np.pi + tol:
                return False
        return True

    # Check initial triangulation: exact scipy match, else the tie-robust
    # local-Delaunay certificate.
    try:
        tri = scipy_Delaunay(points)
        current_triangles = set()
        for simplex in tri.simplices:
            v0, v1, v2 = [vertex_list[i] for i in simplex]
            current_triangles.add(tuple(sorted([v0, v1, v2])))
        initial_tri_correct = (current_triangles == expected_triangulation_set)
        if not initial_tri_correct and triangulation_is_delaunay(points):
            initial_tri_correct = True
            if verbose:
                print("    ✓ Target triangulation is Delaunay for the layout "
                      "(scipy picked other diagonals on cocircular ties)")
        elif verbose and initial_tri_correct:
            print("    ✓ Initial geometry has correct triangulation!")
        elif verbose:
            print("    ⚠ Initial triangulation mismatch:")
            print(f"    Expected {len(expected_triangulation_set)} triangles, got {len(current_triangles)}")
            print("    Will refine using optimization...")
    except Exception as e:
        if verbose:
            print(f"    ⚠ Triangulation check failed: {e}")
        initial_tri_correct = False

    # Refine using optimization (if triangulation not perfect)
    if not initial_tri_correct:
        if verbose:
            print("  Refining geometry with optimization...")

        # Extract initial coordinates for free vertices
        initial_coords = np.zeros(n_free * 2)
        for i, v in enumerate(free_vertices):
            initial_coords[2*i:2*i+2] = points[vertex_to_idx[v]]

        def objective_with_penalty(x):
            """Minimize angle error with soft triangulation penalty."""
            # Reconstruct points
            pts = np.zeros((n_vertices, 2))
            for v, pos in fixed_points.items():
                pts[vertex_to_idx[v]] = pos
            for i, v in enumerate(free_vertices):
                pts[vertex_to_idx[v]] = x[2*i:2*i+2]

            # Angle error
            angle_err = 0.0
            for tri_id, (v0, v1, v2) in enumerate(triangles):
                p0 = pts[vertex_to_idx[v0]]
                p1 = pts[vertex_to_idx[v1]]
                p2 = pts[vertex_to_idx[v2]]

                angle0 = compute_triangle_angle(p0, p1, p2)
                angle1 = compute_triangle_angle(p1, p2, p0)
                angle2 = compute_triangle_angle(p2, p0, p1)

                target = target_angles[tri_id]
                angle_err += (angle0 - target[0])**2
                angle_err += (angle1 - target[1])**2
                angle_err += (angle2 - target[2])**2

            # Triangulation penalty
            try:
                tri = scipy_Delaunay(pts)
                curr_tris = set()
                for simplex in tri.simplices:
                    v0, v1, v2 = [vertex_list[i] for i in simplex]
                    curr_tris.add(tuple(sorted([v0, v1, v2])))

                if curr_tris != expected_triangulation_set:
                    n_missing = len(expected_triangulation_set - curr_tris)
                    n_extra = len(curr_tris - expected_triangulation_set)
                    penalty = 1000.0 * (n_missing + n_extra)  # Strong penalty
                else:
                    penalty = 0.0
            except Exception:
                penalty = 10000.0

            return angle_err + penalty

        from scipy.optimize import minimize
        opt_result = minimize(
            objective_with_penalty,
            initial_coords,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-12}  # More iterations
        )

        # Update points with optimized coordinates
        for i, v in enumerate(free_vertices):
            points[vertex_to_idx[v]] = opt_result.x[2*i:2*i+2]

        if verbose:
            print(f"    Optimization completed ({opt_result.nit} iterations)")

    # Final triangulation check (exact match, else tie-robust certificate)
    try:
        tri = scipy_Delaunay(points)
        current_triangles = set()
        for simplex in tri.simplices:
            v0, v1, v2 = [vertex_list[i] for i in simplex]
            current_triangles.add(tuple(sorted([v0, v1, v2])))
        triangulation_correct = (current_triangles == expected_triangulation_set) \
            or triangulation_is_delaunay(points)
    except Exception:
        triangulation_correct = False

    # Compute angle error and volume
    angle_errors = []
    angle_differences = []
    realized_angles = np.zeros_like(np.asarray(target_angles, dtype=float))
    final_volume = 0.0

    for tri_id, (v0, v1, v2) in enumerate(triangles):
        p0 = points[vertex_to_idx[v0]]
        p1 = points[vertex_to_idx[v1]]
        p2 = points[vertex_to_idx[v2]]

        angle0 = compute_triangle_angle(p0, p1, p2)
        angle1 = compute_triangle_angle(p1, p2, p0)
        angle2 = compute_triangle_angle(p2, p0, p1)

        actual = np.array([angle0, angle1, angle2])
        target = target_angles[tri_id]
        difference = actual - target
        realized_angles[tri_id] = actual
        angle_differences.extend(difference)
        angle_errors.extend(difference**2)

        # Compute volume
        final_volume += lobachevsky_function(angle0)
        final_volume += lobachevsky_function(angle1)
        final_volume += lobachevsky_function(angle2)

    rms_error = np.sqrt(np.mean(angle_errors))
    max_angle_error = float(np.max(np.abs(angle_differences)))

    signed_double_areas = []
    for v0, v1, v2 in triangles:
        p0 = points[vertex_to_idx[v0]]
        p1 = points[vertex_to_idx[v1]]
        p2 = points[vertex_to_idx[v2]]
        signed_double_areas.append(
            float(
                (p1[0] - p0[0]) * (p2[1] - p0[1])
                - (p1[1] - p0[1]) * (p2[0] - p0[0])
            )
        )
    orientation_signs = {
        int(np.sign(area)) for area in signed_double_areas if abs(area) > 1e-14
    }
    orientation_consistent = (
        len(orientation_signs) == 1
        and all(abs(area) > 1e-14 for area in signed_double_areas)
    )
    min_abs_oriented_area = 0.5 * min(map(abs, signed_double_areas))
    realized_angle_residuals = compute_rivin_angle_residuals(
        triangles,
        realized_angles,
        boundary_vertices,
    )

    if verbose:
        print("  Construction: complete")
        print(f"  Achieved volume: {final_volume:.6f}")
        print(f"  Triangulation preserved: {'✓' if triangulation_correct else '✗'}")
        print(f"  RMS angle error: {rms_error:.6f} rad = {np.degrees(rms_error):.2f}°")

    # Success requires BOTH the target combinatorics and the target angles
    # to be attained (the old criterion ignored the angle error, so layouts
    # tens of degrees off the Rivin-optimal shape could report success).
    angles_attained = rms_error < 1e-6
    return {
        'success': triangulation_correct and angles_attained,
        'points': points,
        'vertex_list': vertex_list,  # Maps point index to original vertex ID
        'volume': final_volume,
        'triangulation_preserved': triangulation_correct,
        'angle_error': rms_error,
        'angle_error_degrees': np.degrees(rms_error),
        'max_angle_error': max_angle_error,
        'max_angle_error_degrees': float(np.degrees(max_angle_error)),
        'closure_error': closure_error[0],
        'orientation_consistent': orientation_consistent,
        'min_abs_oriented_area': float(min_abs_oriented_area),
        'residuals': {
            'rms_angle_error': float(rms_error),
            'max_angle_error': max_angle_error,
            'closure_error': float(closure_error[0]),
            'orientation_consistent': orientation_consistent,
            'min_abs_oriented_area': float(min_abs_oriented_area),
            'target_angle_structure': target_angle_residuals,
            'realized_angle_structure': realized_angle_residuals,
        },
        'message': 'Developing-map construction completed',
        'n_iterations': 0,
    }


def refine_geometry_for_volume(initial_points: np.ndarray,
                               vertex_list: List[int],
                               expected_triangulation: List[Tuple[int, int, int]],
                               verbose: bool = False) -> Dict:
    """
    Refine approximate geometry by direct volume optimization.

    Starting from approximate point positions, optimize coordinates to maximize
    hyperbolic volume while checking that the Delaunay triangulation remains correct.

    This is the final refinement step after angle-based reconstruction.

    Args:
        initial_points: Array of shape (n_vertices, 2) with initial positions
        vertex_list: List mapping point index to original vertex ID
        expected_triangulation: The expected triangulation (list of triangles)
        verbose: If True, print progress

    Returns:
        Dict with keys:
        - 'success': bool
        - 'points': Optimized point coordinates
        - 'volume': Final hyperbolic volume
        - 'triangulation_preserved': bool, whether Delaunay triangulation is correct
        - 'message': str
    """
    from scipy.spatial import Delaunay as scipy_Delaunay

    n_vertices = len(initial_points)
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}

    # Fix first 2 vertices to eliminate translation/rotation/scale freedom
    # vertex 0 at its current position, vertex 1 at its current position
    # This fixes translation (2 DOF), rotation (1 DOF), and scale (1 DOF) = 4 DOF
    fixed_indices = [0, 1]
    free_indices = [i for i in range(n_vertices) if i not in fixed_indices]

    if verbose:
        print("Refining geometry via direct volume optimization...")
        print(f"  Vertices: {n_vertices} ({len(free_indices)} free)")

    # Check initial triangulation
    try:
        tri_initial = scipy_Delaunay(initial_points)
        initial_triangles = set()
        for simplex in tri_initial.simplices:
            v0, v1, v2 = [vertex_list[i] for i in simplex]
            initial_triangles.add(tuple(sorted([v0, v1, v2])))

        expected_set = set(tuple(sorted(tri)) for tri in expected_triangulation)

        if initial_triangles != expected_set:
            if verbose:
                print("  ⚠ Warning: Initial triangulation does not match expected!")
                print(f"    Expected {len(expected_set)} triangles, got {len(initial_triangles)}")
                print("    Refinement may not preserve triangulation")
    except Exception:
        pass

    # Initial coordinates for free vertices
    x0 = initial_points[free_indices].flatten()

    # Objective: maximize volume = sum of Lobachevsky functions
    # We minimize the negative
    def objective(x):
        # Reconstruct full points
        points = initial_points.copy()
        for i, idx in enumerate(free_indices):
            points[idx] = x[2*i:2*i+2]

        # Compute Delaunay triangulation
        try:
            tri = scipy_Delaunay(points)
            current_triangles = set()
            for simplex in tri.simplices:
                # Map back to original vertex IDs
                v0, v1, v2 = [vertex_list[i] for i in simplex]
                current_triangles.add(tuple(sorted([v0, v1, v2])))

            # Check if triangulation matches
            expected_set = set(tuple(sorted(tri)) for tri in expected_triangulation)

            if current_triangles != expected_set:
                # Triangulation changed - penalize heavily
                return 1e10

            # Compute volume
            volume = 0.0
            for tri in expected_triangulation:
                v0, v1, v2 = tri
                p0 = points[vertex_to_idx[v0]]
                p1 = points[vertex_to_idx[v1]]
                p2 = points[vertex_to_idx[v2]]

                # Compute angles
                angle0 = compute_triangle_angle(p0, p1, p2)
                angle1 = compute_triangle_angle(p1, p2, p0)
                angle2 = compute_triangle_angle(p2, p0, p1)

                # Add Lobachevsky contributions
                volume += lobachevsky_function(angle0)
                volume += lobachevsky_function(angle1)
                volume += lobachevsky_function(angle2)

            return -volume  # Minimize negative volume

        except Exception:
            # Degenerate configuration
            return 1e10

    # Optimize using L-BFGS-B

    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-12}
    )

    # Reconstruct final points
    final_points = initial_points.copy()
    for i, idx in enumerate(free_indices):
        final_points[idx] = result.x[2*i:2*i+2]

    # Verify final triangulation
    tri_final = scipy_Delaunay(final_points)
    final_triangles = set()
    for simplex in tri_final.simplices:
        v0, v1, v2 = [vertex_list[i] for i in simplex]
        final_triangles.add(tuple(sorted([v0, v1, v2])))

    expected_set = set(tuple(sorted(tri)) for tri in expected_triangulation)
    triangulation_preserved = (final_triangles == expected_set)

    final_volume = -result.fun if result.success and result.fun < 1e9 else None

    if verbose:
        print(f"  Optimization: {'converged' if result.success else 'failed'}")
        print(f"  Iterations: {result.nit}")
        if final_volume is not None:
            print(f"  Final volume: {final_volume:.6f}")
        print(f"  Triangulation preserved: {'✓' if triangulation_preserved else '✗'}")

    return {
        'success': result.success and triangulation_preserved,
        'points': final_points,
        'volume': final_volume,
        'triangulation_preserved': triangulation_preserved,
        'message': result.message,
        'n_iterations': result.nit,
    }
