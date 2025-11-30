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
from scipy.sparse import diags
from collections import defaultdict
from typing import List, Tuple, Set, Dict, Optional
import random


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

    # Find the vertices: edge vertices are b, c; opposite vertices are a, d
    b, c = edge_sorted
    tri1_set = set(tri1)
    tri2_set = set(tri2)

    # Find opposite vertices (not on the edge)
    a = (tri1_set - {b, c}).pop()
    d = (tri2_set - {b, c}).pop()

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

    # Create new triangles with flipped edge (a, d)
    new_tri1 = (a, b, d)
    new_tri2 = (a, d, c)

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

    # (d) sum of angles at boundary vertex <= boundary_bound
    for vertex in boundary_vertices:
        row = np.zeros(n_vars)
        for tri_id, corner_id in vertex_angles[vertex]:
            angle_idx = 3 * tri_id + corner_id
            row[angle_idx] = 1.0
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
        print(f"\n  Edge classification:")
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
        print(f"\nOptimization result:")
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

    integrand = lambda t: -np.log(2 * np.sin(t)) if np.sin(t) > 1e-10 else 0.0
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


def optimize_hyperbolic_volume(triangles: List[Tuple[int, int, int]],
                               initial_angles: Optional[np.ndarray] = None,
                               boundary_vertices: Optional[Set[int]] = None,
                               verbose: bool = False) -> Dict:
    """
    Optimize the hyperbolic volume of a Delaunay realizable triangulation.

    Given a realizable triangulation, find the angle assignment that maximizes
    the hyperbolic volume, which is the sum of Lobachevsky functions of the angles.

    By Rivin's theorem, this function is concave on the Rivin polytope, so
    there is a unique maximum.

    Args:
        triangles: List of triangles as (v0, v1, v2) tuples
        initial_angles: Initial angle assignment (n_triangles, 3). If None, solves LP for feasible start.
        boundary_vertices: Set of boundary vertex indices (if None, auto-detect)
        verbose: If True, print optimization progress

    Returns:
        Dict with keys:
        - 'success': bool, whether optimization succeeded
        - 'angles': np.ndarray of shape (n_triangles, 3), optimal angles (radians)
        - 'volume': float, optimal hyperbolic volume
        - 'message': str, optimization message
        - 'n_iterations': int, number of iterations
    """
    n_triangles = len(triangles)
    n_angles = 3 * n_triangles

    if boundary_vertices is None:
        boundary_vertices = extract_boundary_vertices(triangles)

    # Get all vertices and interior vertices
    all_vertices = set()
    for tri in triangles:
        all_vertices.update(tri)
    interior_vertices = all_vertices - boundary_vertices

    # Build data structures
    vertex_angles = build_vertex_angle_incidence(triangles)
    edge_opposite_angles = build_edge_adjacency(triangles)
    interior_edges = {edge: angles for edge, angles in edge_opposite_angles.items() if len(angles) == 2}

    # If no initial angles provided, solve LP to get a feasible starting point
    if initial_angles is None:
        if verbose:
            print("Solving LP to find feasible starting point...")
        lp_result = check_delaunay_realizability(triangles, boundary_vertices, verbose=False)
        if not lp_result['realizable']:
            return {
                'success': False,
                'angles': None,
                'volume': None,
                'message': 'Triangulation is not Delaunay realizable',
                'n_iterations': 0,
            }
        initial_angles = lp_result['angles_radians']  # Already in radians

    # Flatten initial angles
    x0 = initial_angles.flatten()

    # Build constraint matrices (same as in LP, but for angles in radians)
    # Equality constraints: A_eq @ x == b_eq
    A_eq_list = []
    b_eq_list = []

    # Triangle angle sum = π
    for tri_id in range(n_triangles):
        row = np.zeros(n_angles)
        for corner_id in range(3):
            angle_idx = 3 * tri_id + corner_id
            row[angle_idx] = 1.0
        A_eq_list.append(row)
        b_eq_list.append(np.pi)  # Radians, not scaled

    # Interior vertex angle sum = 2π
    for vertex in interior_vertices:
        row = np.zeros(n_angles)
        for tri_id, corner_id in vertex_angles[vertex]:
            angle_idx = 3 * tri_id + corner_id
            row[angle_idx] = 1.0
        A_eq_list.append(row)
        b_eq_list.append(2 * np.pi)  # Radians, not scaled

    A_eq = np.array(A_eq_list) if A_eq_list else np.zeros((0, n_angles))
    b_eq = np.array(b_eq_list) if b_eq_list else np.zeros(0)

    # Inequality constraints: A_ub @ x <= b_ub
    A_ub_list = []
    b_ub_list = []

    # Boundary vertex angle sum <= π
    for vertex in boundary_vertices:
        row = np.zeros(n_angles)
        for tri_id, corner_id in vertex_angles[vertex]:
            angle_idx = 3 * tri_id + corner_id
            row[angle_idx] = 1.0
        A_ub_list.append(row)
        b_ub_list.append(np.pi)  # Radians, not scaled

    # Interior edge opposite angles sum <= π
    for edge, opposite_angles in interior_edges.items():
        row = np.zeros(n_angles)
        for tri_id, corner_id in opposite_angles:
            angle_idx = 3 * tri_id + corner_id
            row[angle_idx] = 1.0
        A_ub_list.append(row)
        b_ub_list.append(np.pi)  # Radians, not scaled

    A_ub = np.array(A_ub_list) if A_ub_list else np.zeros((0, n_angles))
    b_ub = np.array(b_ub_list) if b_ub_list else np.zeros(0)

    # Bounds: all angles in (0, π)
    bounds = [(1e-8, np.pi - 1e-8) for _ in range(n_angles)]

    # Define objective: we MINIMIZE the negative volume
    # Volume = ∑ Lobachevsky(θᵢ)
    # Scipy minimizes, so we minimize -Volume
    def objective(x):
        return -np.sum([lobachevsky_function(theta) for theta in x])

    # Gradient: ∂(-Volume)/∂θᵢ = -Λ'(θᵢ) = log(2sin(θᵢ))
    def gradient(x):
        return np.array([-lobachevsky_gradient(theta) for theta in x])

    # Hessian: ∂²(-Volume)/∂θᵢ² = -Λ''(θᵢ) = cot(θᵢ)
    # Since Hessian is diagonal, we return it as a sparse diagonal matrix
    def hessian(x):
        diag_elements = np.array([-lobachevsky_hessian(theta) for theta in x])
        return diags(diag_elements, format='csr')

    # Build constraints in scipy format
    from scipy.optimize import LinearConstraint

    constraints = []

    if A_eq.shape[0] > 0:
        constraints.append(LinearConstraint(A_eq, b_eq, b_eq))

    if A_ub.shape[0] > 0:
        constraints.append(LinearConstraint(A_ub, -np.inf, b_ub))

    if verbose:
        print(f"\nOptimizing hyperbolic volume...")
        print(f"  Variables: {n_angles} angles")
        print(f"  Equality constraints: {A_eq.shape[0]}")
        print(f"  Inequality constraints: {A_ub.shape[0]}")

    # Use trust-constr method with exact gradient and Hessian
    result = minimize(
        objective,
        x0,
        method='trust-constr',
        jac=gradient,
        hess=hessian,
        constraints=constraints,
        bounds=bounds,
        options={'verbose': 1 if verbose else 0, 'maxiter': 1000}
    )

    if verbose:
        print(f"\nOptimization complete:")
        print(f"  Success: {result.success}")
        print(f"  Iterations: {result.nit}")
        print(f"  Message: {result.message}")

    # Compute final volume (positive, since we minimized the negative)
    optimal_angles = result.x.reshape(n_triangles, 3)
    volume = np.sum([lobachevsky_function(theta) for theta in result.x])

    # Check dihedral angles (sums of opposite angles across interior edges)
    # If these are very close to π, we're in a degenerate situation
    max_dihedral = 0.0
    if verbose:
        print(f"\nChecking dihedral angles (opposite angles across interior edges):")
        dihedral_angles = []
        for edge, opposite_angles in interior_edges.items():
            if len(opposite_angles) == 2:
                (tri1, corner1), (tri2, corner2) = opposite_angles
                angle1 = optimal_angles[tri1, corner1]
                angle2 = optimal_angles[tri2, corner2]
                dihedral = angle1 + angle2
                dihedral_angles.append(dihedral)
                max_dihedral = max(max_dihedral, dihedral)

        if dihedral_angles:
            mean_dihedral = np.mean(dihedral_angles)
            print(f"  Mean dihedral angle: {mean_dihedral:.6f} rad ({np.degrees(mean_dihedral):.2f}°)")
            print(f"  Max dihedral angle:  {max_dihedral:.6f} rad ({np.degrees(max_dihedral):.2f}°)")
            print(f"  π =                  {np.pi:.6f} rad (180.00°)")

            if max_dihedral > np.pi - 0.01:
                print(f"  ⚠ WARNING: Max dihedral angle very close to π!")
                print(f"  This is nearly degenerate - geometric reconstruction may fail!")

    return {
        'success': result.success,
        'angles': optimal_angles,
        'volume': volume,
        'max_dihedral_angle': max_dihedral,
        'message': result.message,
        'n_iterations': result.nit,
        'optimization_result': result,
    }


def compute_triangle_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Compute the angle at vertex p1 in triangle (p1, p2, p3).

    Args:
        p1, p2, p3: 2D points as arrays

    Returns:
        Angle at p1 in radians
    """
    # Vectors from p1 to p2 and p3
    v1 = p2 - p1
    v2 = p3 - p1

    # Compute angle using atan2 for robustness
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical safety
    return np.arccos(cos_angle)


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

    # Get all vertices
    all_vertices = set()
    for tri in triangles:
        all_vertices.update(tri)
    n_vertices = len(all_vertices)

    # Create vertex index mapping
    vertex_list = sorted(all_vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}

    if verbose:
        print(f"Reconstructing geometry from angles...")
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
        print(f"  Constructing geometry incrementally using angle constraints...")

    from scipy.spatial import Delaunay as scipy_Delaunay
    expected_triangulation_set = set(tuple(sorted(tri)) for tri in triangles)

    def construct_from_angles():
        """Incrementally place vertices using angle constraints."""
        points = np.zeros((n_vertices, 2))

        # Set fixed points
        for v, pos in fixed_points.items():
            points[vertex_to_idx[v]] = pos

        placed = set(fixed_points.keys())
        remaining = set(free_vertices)

        # Build adjacency information from triangles
        vertex_triangles = {v: [] for v in vertex_list}
        for tri_id, (v0, v1, v2) in enumerate(triangles):
            vertex_triangles[v0].append((tri_id, v0, v1, v2))
            vertex_triangles[v1].append((tri_id, v0, v1, v2))
            vertex_triangles[v2].append((tri_id, v0, v1, v2))

        iteration = 0
        max_iterations = n_vertices * 3

        while remaining and iteration < max_iterations:
            iteration += 1

            # Find a vertex that can be placed using existing vertices
            best_vertex = None
            best_placement = None
            best_score = -1

            for v in remaining:
                # Find triangles containing this vertex
                candidate_triangles = vertex_triangles[v]

                # Count how many have both other vertices already placed
                valid_triangles = []
                for tri_id, v0, v1, v2 in candidate_triangles:
                    others = [u for u in (v0, v1, v2) if u != v]
                    if all(u in placed for u in others):
                        valid_triangles.append((tri_id, v0, v1, v2))

                if not valid_triangles:
                    continue

                # Use the first valid triangle to compute position
                tri_id, v0, v1, v2 = valid_triangles[0]

                # Get angle at vertex v and positions of other two vertices
                if v == v0:
                    angle_at_v = target_angles[tri_id, 0]
                    p1 = points[vertex_to_idx[v1]]
                    p2 = points[vertex_to_idx[v2]]
                elif v == v1:
                    angle_at_v = target_angles[tri_id, 1]
                    p1 = points[vertex_to_idx[v2]]
                    p2 = points[vertex_to_idx[v0]]
                else:  # v == v2
                    angle_at_v = target_angles[tri_id, 2]
                    p1 = points[vertex_to_idx[v0]]
                    p2 = points[vertex_to_idx[v1]]

                # Compute position that satisfies the angle constraint
                # Place v such that angle(p1, v, p2) = angle_at_v
                edge_vec = p2 - p1
                edge_len = np.linalg.norm(edge_vec)

                if edge_len < 1e-10:
                    continue

                # Midpoint and perpendicular direction
                midpoint = (p1 + p2) / 2
                perp = np.array([-edge_vec[1], edge_vec[0]]) / edge_len

                # Height from edge to vertex (using trigonometry)
                # For angle theta at apex of isosceles triangle with base edge_len:
                # height = (edge_len / 2) / tan(theta / 2)
                if angle_at_v > 0.01 and angle_at_v < np.pi - 0.01:
                    height = (edge_len / 2) / np.tan(angle_at_v / 2)
                else:
                    continue

                # Two possible positions (above/below the edge)
                pos1 = midpoint + perp * height
                pos2 = midpoint - perp * height

                # Choose the position that best satisfies other angle constraints
                score1 = len(valid_triangles)
                score2 = len(valid_triangles)

                # Prefer position with more valid triangles
                if len(valid_triangles) > best_score:
                    best_vertex = v
                    best_placement = pos1  # Default to pos1
                    best_score = len(valid_triangles)

            if best_vertex is None:
                # Can't place any more vertices - break
                if verbose:
                    print(f"    ⚠ Stuck after placing {len(placed)}/{n_vertices} vertices")
                break

            # Place the vertex
            points[vertex_to_idx[best_vertex]] = best_placement
            placed.add(best_vertex)
            remaining.remove(best_vertex)

        if remaining:
            if verbose:
                print(f"    ⚠ Could not place all vertices ({len(remaining)} remaining)")
            return None

        return points

    points_constructed = construct_from_angles()

    if points_constructed is None:
        # Construction failed
        if verbose:
            print(f"    ✗ Incremental construction failed")
        return {
            'success': False,
            'points': None,
            'vertex_list': vertex_list,
            'volume': None,
            'triangulation_preserved': False,
            'angle_error': None,
            'angle_error_degrees': None,
            'message': 'Incremental construction failed',
            'n_iterations': 0,
        }

    points = points_constructed

    # Check initial triangulation
    try:
        tri = scipy_Delaunay(points)
        current_triangles = set()
        for simplex in tri.simplices:
            v0, v1, v2 = [vertex_list[i] for i in simplex]
            current_triangles.add(tuple(sorted([v0, v1, v2])))
        initial_tri_correct = (current_triangles == expected_triangulation_set)

        if verbose:
            if initial_tri_correct:
                print(f"    ✓ Initial geometry has correct triangulation!")
            else:
                print(f"    ⚠ Initial triangulation mismatch:")
                print(f"    Expected {len(expected_triangulation_set)} triangles, got {len(current_triangles)}")
                missing = expected_triangulation_set - current_triangles
                extra = current_triangles - expected_triangulation_set
                if missing:
                    print(f"    Missing {len(missing)} triangles")
                if extra:
                    print(f"    Extra {len(extra)} triangles")
                print(f"    Will refine using optimization...")
    except Exception as e:
        if verbose:
            print(f"    ⚠ Triangulation check failed: {e}")
        initial_tri_correct = False

    # Refine using optimization (if triangulation not perfect)
    if not initial_tri_correct:
        if verbose:
            print(f"  Refining geometry with optimization...")

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
            except:
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

    # Final triangulation check
    try:
        tri = scipy_Delaunay(points)
        current_triangles = set()
        for simplex in tri.simplices:
            v0, v1, v2 = [vertex_list[i] for i in simplex]
            current_triangles.add(tuple(sorted([v0, v1, v2])))
        triangulation_correct = (current_triangles == expected_triangulation_set)
    except:
        triangulation_correct = False

    result = type('Result', (), {'success': True, 'nit': 0})()  # Dummy result

    # Compute angle error and volume
    angle_errors = []
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
        angle_errors.extend((actual - target)**2)

        # Compute volume
        final_volume += lobachevsky_function(angle0)
        final_volume += lobachevsky_function(angle1)
        final_volume += lobachevsky_function(angle2)

    rms_error = np.sqrt(np.mean(angle_errors))

    if verbose:
        print(f"  Construction: complete")
        print(f"  Achieved volume: {final_volume:.6f}")
        print(f"  Triangulation preserved: {'✓' if triangulation_correct else '✗'}")
        print(f"  RMS angle error: {rms_error:.6f} rad = {np.degrees(rms_error):.2f}°")

    return {
        'success': result.success and triangulation_correct,
        'points': points,
        'vertex_list': vertex_list,  # Maps point index to original vertex ID
        'volume': final_volume,
        'triangulation_preserved': triangulation_correct,
        'angle_error': rms_error,
        'angle_error_degrees': np.degrees(rms_error),
        'message': 'Incremental construction completed',
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
        print(f"Refining geometry via direct volume optimization...")
        print(f"  Vertices: {n_vertices} ({len(free_indices)} free)")

    # Check initial triangulation
    from scipy.spatial import Delaunay as scipy_Delaunay
    try:
        tri_initial = scipy_Delaunay(initial_points)
        initial_triangles = set()
        for simplex in tri_initial.simplices:
            v0, v1, v2 = [vertex_list[i] for i in simplex]
            initial_triangles.add(tuple(sorted([v0, v1, v2])))

        expected_set = set(tuple(sorted(tri)) for tri in expected_triangulation)

        if initial_triangles != expected_set:
            if verbose:
                print(f"  ⚠ Warning: Initial triangulation does not match expected!")
                print(f"    Expected {len(expected_set)} triangles, got {len(initial_triangles)}")
                print(f"    Refinement may not preserve triangulation")
    except:
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

        except Exception as e:
            # Degenerate configuration
            return 1e10

    # Optimize using L-BFGS-B
    from scipy.optimize import minimize

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
