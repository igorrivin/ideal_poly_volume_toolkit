"""
rivin_holonomy.py
-----------------
Reference implementation of the Penner–Rivin holonomy algorithm for an ideal triangulation
with optional shear data. Pure-Python (no external deps).

Model:
- Triangles are labeled 0..F-1, each with sides 0,1,2 in cyclic order (counterclockwise).
- Adjacency maps each oriented side (t, s) to (u, su, eid), where u is the triangle glued across side s,
  su is the side index in u, and eid is a global id for the underlying edge.
- Ori gives, for each global eid, a chosen orientation as ((t_from, s_from), (t_to, s_to)).
- Shears Z: dict eid -> real number (use 0.0 for zero-shear case).

Outputs:
- Generators as 2x2 matrices in SL(2,R) (project to PSL(2,R)).
"""

import math
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Any

def matmul(A, B):
    return [[A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
            [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]]

def matI():
    return [[1.0, 0.0], [0.0, 1.0]]

def matR():
    return [[1.0, 1.0], [-1.0, 0.0]]

def matL():
    return [[0.0, -1.0], [1.0, 1.0]]

def matX(z):
    ez = math.exp(z/2.0)
    return [[0.0, -ez], [1.0/ez, 0.0]]

class Triangulation:
    def __init__(self, F, adjacency, order, orientation):
        """
        F: number of triangles
        adjacency: dict (t, s) -> (u, su, eid)
        order: dict t -> list [0,1,2] (cyclic order of sides in triangle t)
        orientation: dict eid -> ((t_from, s_from), (t_to, s_to))
        """
        self.F = F
        self.Adj = dict(adjacency)
        self.Order = {t: list(order[t]) for t in order}
        self.Ori = dict(orientation)
        # basic sanity
        for (t,s), (u,su,eid) in self.Adj.items():
            rt = self.Adj.get((u,su))
            assert rt is not None and rt[0]==t and rt[1]==s and rt[2]==eid, "Adjacency must be symmetric"
        for t in range(F):
            assert t in self.Order and len(self.Order[t])==3, "Order missing for triangle {}".format(t)

    def dual_graph(self):
        """Return dual graph structure: dict t -> list of (u, s_in_t, s_in_u, eid)."""
        G = defaultdict(list)
        for (t,s), (u,su,eid) in self.Adj.items():
            G[t].append((u, s, su, eid))
        return G

def bfs_spanning_tree(G, root=0):
    parent = {root: None}
    parent_edge = {root: None}   # for v!=root: (eid, side_in_v, side_in_parent)
    Q = deque([root])
    while Q:
        v = Q.popleft()
        for (w, sv, sw, eid) in G[v]:
            if w not in parent:
                parent[w] = v
                parent_edge[w] = (eid, sw, sv)
                Q.append(w)
    return parent, parent_edge

def path_in_tree(parent, v, u):
    """Return list of vertices along the unique path v -> ... -> u in the tree."""
    Pv = []
    x = v
    while x is not None:
        Pv.append(x); x = parent[x]
    Pu = []
    x = u
    while x is not None:
        Pu.append(x); x = parent[x]
    i = len(Pv)-1; j = len(Pu)-1
    while i>=0 and j>=0 and Pv[i]==Pu[j]:
        i -= 1; j -= 1
    up = Pv[:i+1]
    down = list(reversed(Pu[:j+1]))
    return up + [Pv[i+1]] + down

def turn(order, side_in, side_out):
    i = order.index(side_in); j = order.index(side_out)
    d = (j - i) % 3
    if d == 1: return 'L'
    if d == 2: return 'R'
    raise ValueError("Degenerate turn inside triangle")

def tokens_for_path(T: Triangulation, parent, parent_edge, path):
    """
    Convert a path of triangles into a token list of 'L'/'R' and ('X', eid, sign).
    """
    tokens = []
    side_in_prev = None
    for k in range(len(path)-1):
        a, b = path[k], path[k+1]
        if parent[b] == a:
            eid, side_in_b, side_in_a = parent_edge[b]  # from a to b
        elif parent[a] == b:
            eid, side_in_a, side_in_b = parent_edge[a]  # from a to b
        else:
            raise RuntimeError("Non-tree step in tokens_for_path")
        if side_in_prev is not None:
            tokens.append(('T', a, side_in_prev, side_in_a))
        (tf, sf), (tt, st) = T.Ori[eid]
        if (tf == a and tt == b): sign = +1
        elif (tf == b and tt == a): sign = -1
        else: raise RuntimeError("Edge orientation inconsistent with step")
        tokens.append(('X', eid, sign))
        side_in_prev = side_in_b
    return _resolve_turns(T, tokens)

def _resolve_turns(T: Triangulation, tokens):
    resolved = []
    for tok in tokens:
        if isinstance(tok, tuple) and tok and tok[0]=='T':
            _, tri, s_in, s_out = tok
            resolved.append(turn(T.Order[tri], s_in, s_out))
        else:
            resolved.append(tok)
    return resolved

def tokens_to_matrix(tokens, Z):
    M = matI()
    for tok in tokens:
        if tok == 'L':
            M = matmul(M, matL())
        elif tok == 'R':
            M = matmul(M, matR())
        else:
            _, eid, sign = tok
            z = Z.get(eid, 0.0)
            if sign < 0: z = -z
            M = matmul(M, matX(z))
    return M

def generators_from_triangulation(T: Triangulation, Z, root=0):
    """
    Compute a generating set for the holonomy:
      one generator per non-tree dual edge.
    Returns: list of (u, v, tokens, matrix)
    """
    G = T.dual_graph()
    parent, parent_edge = bfs_spanning_tree(G, root=root)

    tree_edges = set()
    for v, p in parent.items():
        if p is None: continue
        tree_edges.add(tuple(sorted((v, p))))

    all_edges = set()
    for v, lst in G.items():
        for (w, sv, sw, eid) in lst:
            if v < w:
                all_edges.add((v, w, eid))

    gens = []
    for (u, v, eid) in all_edges:
        if tuple(sorted((u, v))) in tree_edges:
            continue
        path_u = path_in_tree(parent, root, u)
        path_v = path_in_tree(parent, root, v)

        tok1 = tokens_for_path(T, parent, parent_edge, path_u)          # root -> u
        (tf, sf), (tt, st) = T.Ori[eid]
        if (tf == u and tt == v): sign = +1
        elif (tf == v and tt == u): sign = -1
        else: raise RuntimeError("Orientation mismatch on non-tree edge")
        tok_mid = [('X', eid, sign)]
        tok2 = tokens_for_path(T, parent, parent_edge, list(reversed(path_v)))  # v -> root

        tokens = tok1 + tok_mid + tok2
        M = tokens_to_matrix(tokens, Z)
        gens.append((u, v, tokens, M))
    return gens

def triangulation_from_faces(triangles: List[Tuple[int, int, int]]) -> Tuple[Triangulation, Dict[int, float]]:
    """
    Build a Triangulation object from a list of triangle faces.

    Args:
        triangles: List of triangles, each as (v0, v1, v2) tuple of vertex indices

    Returns:
        Tuple of (Triangulation, Z) where Z is zero shear dict
    """
    F = len(triangles)
    adjacency = {}
    edge_id_map = {}
    edge_id = 0

    for i, tri_i in enumerate(triangles):
        for side_i in range(3):
            v1_i, v2_i = tri_i[side_i], tri_i[(side_i + 1) % 3]
            edge = tuple(sorted([v1_i, v2_i]))

            for j, tri_j in enumerate(triangles):
                if i == j:
                    continue
                for side_j in range(3):
                    v1_j, v2_j = tri_j[side_j], tri_j[(side_j + 1) % 3]
                    if set([v1_j, v2_j]) == set([v1_i, v2_i]):
                        if (i, side_i) not in adjacency:
                            if edge not in edge_id_map:
                                edge_id_map[edge] = edge_id
                                edge_id += 1
                            adjacency[(i, side_i)] = (j, side_j, edge_id_map[edge])

    order = {t: [0, 1, 2] for t in range(F)}
    orientation = {}
    for edge, eid in edge_id_map.items():
        for (t, s), (u, su, e) in adjacency.items():
            if e == eid:
                orientation[eid] = ((t, s), (u, su))
                break

    T = Triangulation(F, adjacency, order, orientation)
    Z = {eid: 0.0 for eid in range(edge_id)}

    return T, Z


def check_arithmeticity(triangles: List[Tuple[int, int, int]],
                        tolerance: float = 0.01,
                        verbose: bool = False) -> Dict[str, Any]:
    """
    Check if a triangulation has arithmetic holonomy (all traces are integers).

    A triangulation is arithmetic if all holonomy generator traces lie in Z.
    This is a necessary condition for the ideal polyhedron to be arithmetic
    (i.e., commensurable with PSL(2, O_K) for some number field K).

    Args:
        triangles: List of triangles as (v0, v1, v2) tuples
        tolerance: Maximum distance from nearest integer to count as integral
        verbose: If True, print detailed trace information

    Returns:
        Dict with keys:
        - 'is_arithmetic': bool, True if all traces are integers
        - 'n_generators': int, number of holonomy generators
        - 'n_integral': int, number of generators with integral traces
        - 'traces': list of trace values
        - 'trace_details': list of dicts with trace analysis for each generator
    """
    T, Z = triangulation_from_faces(triangles)
    gens = generators_from_triangulation(T, Z, root=0)

    trace_details = []
    integral_count = 0

    for u, v, tokens, M in gens:
        trace = M[0][0] + M[1][1]
        nearest_int = round(trace)
        distance = abs(trace - nearest_int)
        is_integral = distance < tolerance

        if is_integral:
            integral_count += 1

        detail = {
            'generator': (u, v),
            'trace': trace,
            'nearest_int': nearest_int,
            'distance': distance,
            'is_integral': is_integral
        }
        trace_details.append(detail)

        if verbose:
            status = "INTEGRAL" if is_integral else ""
            print(f"  Generator {u}-{v}: trace = {trace:.6f}, "
                  f"nearest int = {nearest_int}, dist = {distance:.6f} {status}")

    is_arithmetic = (integral_count == len(gens)) if len(gens) > 0 else True

    return {
        'is_arithmetic': is_arithmetic,
        'n_generators': len(gens),
        'n_integral': integral_count,
        'traces': [d['trace'] for d in trace_details],
        'trace_details': trace_details
    }


def compute_shears_from_vertices(triangles: List[Tuple[int, int, int]],
                                  vertices_complex) -> Dict[int, float]:
    """
    Compute shear parameters from actual vertex positions.

    For each interior edge shared by two triangles, the shear is computed
    from the cross-ratio of the four vertices involved.

    The shear z on edge (v2, v3) shared by triangles (v1, v2, v3) and (v2, v4, v3) is:
        z = log|cross_ratio(v1, v2, v4, v3)|

    where cross_ratio(a, b, c, d) = (a-c)(b-d) / ((a-d)(b-c))

    Args:
        triangles: List of triangles as (v0, v1, v2) tuples
        vertices_complex: Array of complex vertex coordinates

    Returns:
        Dict mapping edge_id -> shear value
    """
    import numpy as np

    # Build edge adjacency structure
    edge_triangles = {}  # edge -> list of (tri_idx, opposite_vertex)

    for tri_idx, tri in enumerate(triangles):
        for i in range(3):
            v1, v2 = tri[i], tri[(i + 1) % 3]
            opposite = tri[(i + 2) % 3]
            edge = tuple(sorted([v1, v2]))

            if edge not in edge_triangles:
                edge_triangles[edge] = []
            edge_triangles[edge].append((tri_idx, opposite, v1, v2))

    # Assign edge IDs and compute shears
    shears = {}
    edge_id = 0

    for edge, tri_list in edge_triangles.items():
        if len(tri_list) == 2:  # Interior edge
            # Get the four vertices: two on the edge, two opposite
            _, opp1, e1, e2 = tri_list[0]
            _, opp2, _, _ = tri_list[1]

            # Get complex coordinates
            z1 = complex(vertices_complex[opp1])  # opposite vertex 1
            z2 = complex(vertices_complex[e1])     # edge vertex 1
            z3 = complex(vertices_complex[e2])     # edge vertex 2
            z4 = complex(vertices_complex[opp2])  # opposite vertex 2

            # Compute cross-ratio: (z1-z3)(z2-z4) / ((z1-z4)(z2-z3))
            num = (z1 - z3) * (z2 - z4)
            den = (z1 - z4) * (z2 - z3)

            if abs(den) > 1e-10:
                cr = num / den
                # Shear is log of absolute value of cross-ratio
                shear = math.log(abs(cr)) if abs(cr) > 1e-10 else 0.0
            else:
                shear = 0.0

            shears[edge_id] = shear
            edge_id += 1

    return shears


def check_arithmeticity_from_vertices(vertices_complex, tolerance: float = 0.01,
                                      verbose: bool = False) -> Dict[str, Any]:
    """
    Check arithmeticity given complex vertex coordinates.

    Computes Delaunay triangulation and checks holonomy traces using
    the actual shear parameters derived from the geometry.

    For random vertex positions, shears are non-zero and traces are
    typically non-integral (non-arithmetic).

    For Rivin-Delaunay optimized positions, shears are zero and traces
    are integers (arithmetic).

    Args:
        vertices_complex: Array of complex numbers (vertex positions)
        tolerance: Maximum distance from nearest integer to count as integral
        verbose: If True, print detailed trace information

    Returns:
        Same as check_arithmeticity(), plus 'shears' key with computed shears
    """
    import numpy as np
    from scipy.spatial import Delaunay

    # Compute Delaunay triangulation
    pts = np.column_stack([np.real(vertices_complex), np.imag(vertices_complex)])
    tri = Delaunay(pts)
    triangles = [tuple(simplex) for simplex in tri.simplices]

    # Build triangulation structure
    T, Z_zero = triangulation_from_faces(triangles)

    # Compute actual shears from geometry
    shears = compute_shears_from_vertices(triangles, vertices_complex)

    # Use actual shears for holonomy computation
    gens = generators_from_triangulation(T, shears, root=0)

    trace_details = []
    integral_count = 0

    for u, v, tokens, M in gens:
        trace = M[0][0] + M[1][1]
        nearest_int = round(trace)
        distance = abs(trace - nearest_int)
        is_integral = distance < tolerance

        if is_integral:
            integral_count += 1

        detail = {
            'generator': (u, v),
            'trace': trace,
            'nearest_int': nearest_int,
            'distance': distance,
            'is_integral': is_integral
        }
        trace_details.append(detail)

        if verbose:
            status = "INTEGRAL" if is_integral else ""
            print(f"  Generator {u}-{v}: trace = {trace:.6f}, "
                  f"nearest int = {nearest_int}, dist = {distance:.6f} {status}")

    is_arithmetic = (integral_count == len(gens)) if len(gens) > 0 else True

    return {
        'is_arithmetic': is_arithmetic,
        'n_generators': len(gens),
        'n_integral': integral_count,
        'traces': [d['trace'] for d in trace_details],
        'trace_details': trace_details,
        'shears': shears,
    }


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Check arithmeticity of an ideal polyhedron via Penner-Rivin holonomy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s config.json              # Check arithmeticity of configuration in JSON file
  %(prog)s config.json --verbose    # Show detailed trace information

JSON file format:
  {
    "vertices_real": [0.0, 1.0, 0.5, ...],
    "vertices_imag": [0.0, 0.0, 0.866, ...]
  }
  OR
  {
    "triangles": [[0, 1, 2], [1, 2, 3], ...]
  }

Output:
  ARITHMETIC    - All holonomy traces are integers (exit code 0)
  NON-ARITHMETIC - Some traces are not integers (exit code 1)
        """
    )
    parser.add_argument('input', nargs='?', help='JSON file with configuration')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed trace information')
    parser.add_argument('--tolerance', '-t', type=float, default=0.01,
                        help='Tolerance for integrality check (default: 0.01)')

    args = parser.parse_args()

    if args.input is None:
        print("Penner-Rivin Holonomy Arithmeticity Checker")
        print("=" * 50)
        print()
        print("This module checks if an ideal polyhedron is arithmetic by computing")
        print("the holonomy generators and verifying that all traces are integers.")
        print()
        print("Usage:")
        print("  python -m ideal_poly_volume_toolkit.rivin_holonomy config.json")
        print("  python -m ideal_poly_volume_toolkit.rivin_holonomy config.json --verbose")
        print()
        print("See --help for more information.")
        sys.exit(0)

    # Load configuration
    with open(args.input, 'r') as f:
        config = json.load(f)

    # Check if we have vertices or triangles
    if 'triangles' in config:
        triangles = [tuple(t) for t in config['triangles']]
        result = check_arithmeticity(triangles, args.tolerance, args.verbose)
    elif 'vertices_real' in config and 'vertices_imag' in config:
        import numpy as np
        vertices = np.array(config['vertices_real']) + 1j * np.array(config['vertices_imag'])
        result = check_arithmeticity_from_vertices(vertices, args.tolerance, args.verbose)
    else:
        print("Error: JSON must contain either 'triangles' or 'vertices_real'/'vertices_imag'")
        sys.exit(2)

    # Output result
    if args.verbose:
        print()
        print("=" * 50)

    print(f"Generators: {result['n_generators']}")
    print(f"Integral traces: {result['n_integral']}/{result['n_generators']}")

    if result['is_arithmetic']:
        print("Result: ARITHMETIC")
        sys.exit(0)
    else:
        print("Result: NON-ARITHMETIC")
        sys.exit(1)
