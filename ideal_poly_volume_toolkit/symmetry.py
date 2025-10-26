"""
Symmetry group computation for ideal polyhedra using nauty/pynauty.

Computes the automorphism group of the 1-skeleton (graph) of the polyhedron.
"""

import numpy as np
from scipy.spatial import ConvexHull

try:
    import pynauty
    PYNAUTY_AVAILABLE = True
except ImportError:
    PYNAUTY_AVAILABLE = False


def compute_symmetry_group(vertices_3d):
    """
    Compute the automorphism group of a polyhedron's 1-skeleton.

    Args:
        vertices_3d: N x 3 array of vertex coordinates (e.g., from lift_to_sphere_with_inf)

    Returns:
        dict with keys:
        - 'group_size': Total size of automorphism group
        - 'num_generators': Number of group generators
        - 'generators': List of permutations generating the group
        - 'orbits': Orbit partition of vertices
        - 'num_orbits': Number of vertex orbits
        - 'group_name': Human-readable group identification (if recognized)
        - 'available': Whether pynauty is available

    Raises:
        ImportError: If pynauty is not installed
    """
    if not PYNAUTY_AVAILABLE:
        return {
            'available': False,
            'error': 'pynauty not installed. Install with: pip install pynauty'
        }

    # Get convex hull to extract 1-skeleton (edge graph)
    hull = ConvexHull(vertices_3d)

    # Build edge list from hull faces (simplices)
    edges = set()
    for simplex in hull.simplices:
        # Each face is a triangle with 3 edges
        v0, v1, v2 = simplex
        edges.add(tuple(sorted([v0, v1])))
        edges.add(tuple(sorted([v1, v2])))
        edges.add(tuple(sorted([v2, v0])))

    # Build adjacency list for pynauty
    n_vertices = len(vertices_3d)
    adjacency = {i: [] for i in range(n_vertices)}
    for v1, v2 in edges:
        adjacency[v1].append(v2)
        adjacency[v2].append(v1)

    # Create pynauty graph
    g = pynauty.Graph(number_of_vertices=n_vertices, directed=False, adjacency_dict=adjacency)

    # Compute automorphism group
    # Returns: (generators, grpsize1, grpsize2, orbits, numorbits)
    generators, grpsize1, grpsize2, orbits, numorbits = pynauty.autgrp(g)

    # Calculate total group size
    # nauty represents size as grpsize1 × 10^grpsize2
    if isinstance(grpsize2, (int, float)):
        group_size = int(grpsize1 * (10 ** grpsize2))
    else:
        group_size = int(grpsize1)

    # Identify common symmetry groups
    group_name = identify_group(group_size, n_vertices)

    return {
        'available': True,
        'group_size': group_size,
        'num_generators': len(generators),
        'generators': generators,
        'orbits': orbits,
        'num_orbits': numorbits,
        'group_name': group_name,
        'vertex_degrees': [len(adjacency[i]) for i in range(n_vertices)],
        'num_edges': len(edges),
        'num_faces': len(hull.simplices),
    }


def identify_group(size, n_vertices=None):
    """
    Identify common symmetry groups by order.

    Args:
        size: Order of the group
        n_vertices: Number of vertices (helps with identification)

    Returns:
        Human-readable group name
    """
    common_groups = {
        1: "C₁ (trivial, no symmetry)",
        2: "C₂ (single reflection/inversion)",
        3: "C₃ (3-fold rotation)",
        4: "V₄ (Klein four-group) or C₄ (4-fold rotation)",
        6: "S₃ ≅ D₃ (triangle symmetries)",
        8: "D₄ (square symmetries) or other order-8 group",
        12: "A₄ (tetrahedral rotations) or D₆ (hexagon symmetries)",
        24: "S₄ (octahedral symmetries) or other order-24 group",
        48: "Oh (full octahedral symmetries)",
        60: "A₅ (icosahedral rotations)",
        120: "Ih (full icosahedral symmetries) or S₅",
    }

    if size in common_groups:
        return common_groups[size]
    else:
        # Check for dihedral groups D_n (order 2n)
        if size % 2 == 0:
            n = size // 2
            return f"possibly D_{n} (dihedral) or other order-{size} group"
        # Check for cyclic groups C_n
        return f"possibly C_{size} (cyclic) or other order-{size} group"


def format_symmetry_report(sym_info):
    """
    Format symmetry information as a readable string.

    Args:
        sym_info: Dictionary returned by compute_symmetry_group

    Returns:
        Formatted string report
    """
    if not sym_info.get('available', False):
        return f"❌ Symmetry computation unavailable: {sym_info.get('error', 'Unknown error')}"

    report = []
    report.append("## Symmetry Group Analysis\n")
    report.append(f"**Group Order:** {sym_info['group_size']}")
    report.append(f"**Identification:** {sym_info['group_name']}")
    report.append(f"**Generators:** {sym_info['num_generators']}")
    report.append(f"**Vertex Orbits:** {sym_info['num_orbits']}")
    report.append("")
    report.append("**Graph Properties:**")
    report.append(f"- Vertices: {len(sym_info['vertex_degrees'])}")
    report.append(f"- Edges: {sym_info['num_edges']}")
    report.append(f"- Faces: {sym_info['num_faces']}")
    report.append(f"- Vertex degrees: {sym_info['vertex_degrees']}")

    if sym_info['num_generators'] > 0 and sym_info['num_generators'] <= 5:
        report.append("")
        report.append("**Generators (as vertex permutations):**")
        for i, gen in enumerate(sym_info['generators'], 1):
            report.append(f"- σ_{i}: {gen}")

    return "\n".join(report)
