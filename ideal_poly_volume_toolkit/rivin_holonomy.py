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
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


HolonomyToken = str | tuple[str, int, int]
ExactMatrix2x2 = tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class ClosedSphereTriangulationSummary:
    """Combinatorial invariants of a validated triangulated 2-sphere."""

    vertex_labels: tuple[int, ...]
    n_vertices: int
    n_edges: int
    n_faces: int
    euler_characteristic: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "vertex_labels": list(self.vertex_labels),
            "n_vertices": self.n_vertices,
            "n_edges": self.n_edges,
            "n_faces": self.n_faces,
            "euler_characteristic": self.euler_characteristic,
        }


@dataclass(frozen=True)
class ExactZeroShearGenerator:
    """One holonomy generator evaluated exactly at shear zero."""

    source_triangle: int
    target_triangle: int
    tokens: tuple[HolonomyToken, ...]
    matrix: ExactMatrix2x2

    @property
    def trace(self) -> int:
        return self.matrix[0][0] + self.matrix[1][1]

    @property
    def determinant(self) -> int:
        return (
            self.matrix[0][0] * self.matrix[1][1]
            - self.matrix[0][1] * self.matrix[1][0]
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        tokens = [
            list(token) if isinstance(token, tuple) else token
            for token in self.tokens
        ]
        return {
            "source_triangle": self.source_triangle,
            "target_triangle": self.target_triangle,
            "tokens": tokens,
            "matrix": [list(row) for row in self.matrix],
            "trace": self.trace,
            "determinant": self.determinant,
        }


@dataclass(frozen=True)
class ExactZeroShearCertificate:
    """Exact PSL(2,Z) certificate for a declared zero-shear structure.

    This certificate concerns the abstract zero-shear structure on the supplied
    closed triangulation.  It does not assert that unrelated floating-point
    coordinates realize those zero shears; callers with coordinates must verify
    that geometric premise independently with ``compute_shears_from_vertices``.
    """

    topology: ClosedSphereTriangulationSummary
    root_triangle: int
    psl2z_index: int
    generators: tuple[ExactZeroShearGenerator, ...]
    certificate_type: str = "exact-zero-shear-psl2z-v1"
    scope: str = "abstract-zero-shear-structure"

    @property
    def traces(self) -> tuple[int, ...]:
        return tuple(generator.trace for generator in self.generators)

    @property
    def all_determinants_one(self) -> bool:
        return all(generator.determinant == 1 for generator in self.generators)

    @property
    def all_traces_integral(self) -> bool:
        # The exact evaluator constructs Python integers throughout.  Keep this
        # explicit predicate in the serialized certificate for downstream gates.
        return all(isinstance(trace, int) for trace in self.traces)

    @property
    def is_arithmetic(self) -> bool:
        return self.all_determinants_one and self.all_traces_integral

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "certificate_type": self.certificate_type,
            "scope": self.scope,
            "topology": self.topology.as_dict(),
            "root_triangle": self.root_triangle,
            "psl2z_index": self.psl2z_index,
            "generators": [generator.as_dict() for generator in self.generators],
            "traces": list(self.traces),
            "all_determinants_one": self.all_determinants_one,
            "all_traces_integral": self.all_traces_integral,
            "is_arithmetic": self.is_arithmetic,
        }


def _materialize_integer_faces(
    triangles: Iterable[Sequence[int]],
) -> tuple[tuple[int, int, int], ...]:
    """Materialize and strictly validate the local syntax of triangle faces."""
    faces: list[tuple[int, int, int]] = []
    seen_faces: set[tuple[int, int, int]] = set()

    for face_index, triangle in enumerate(triangles):
        face = tuple(triangle)
        if len(face) != 3:
            raise ValueError(
                f"face {face_index} must contain exactly three vertices"
            )
        if any(
            not isinstance(vertex, Integral) or isinstance(vertex, bool)
            for vertex in face
        ):
            raise TypeError(
                f"face {face_index} vertex labels must be non-negative integers"
            )

        normalized_face = tuple(int(vertex) for vertex in face)
        if any(vertex < 0 for vertex in normalized_face):
            raise ValueError(
                f"face {face_index} vertex labels must be non-negative"
            )
        if len(set(normalized_face)) != 3:
            raise ValueError(f"face {face_index} has repeated vertices")

        unoriented_face = tuple(sorted(normalized_face))
        if unoriented_face in seen_faces:
            raise ValueError(
                f"face {face_index} duplicates an existing unoriented face"
            )
        seen_faces.add(unoriented_face)
        faces.append(normalized_face)

    if not faces:
        raise ValueError("a closed sphere triangulation must contain faces")
    return tuple(faces)


def _analyze_closed_sphere_triangulation(
    triangles: Iterable[Sequence[int]],
) -> tuple[
    tuple[tuple[int, int, int], ...],
    ClosedSphereTriangulationSummary,
]:
    """Return coherently oriented faces and invariants after strict checks."""
    faces = _materialize_integer_faces(triangles)
    edge_occurrences: dict[
        tuple[int, int], list[tuple[int, int, int]]
    ] = defaultdict(list)
    vertex_links: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for face_index, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_occurrences[tuple(sorted((u, v)))].append(
                (face_index, u, v)
            )
        vertex_links[a].append((b, c))
        vertex_links[b].append((c, a))
        vertex_links[c].append((a, b))

    for edge, occurrences in edge_occurrences.items():
        if len(occurrences) != 2:
            raise ValueError(
                f"edge {edge} has incidence {len(occurrences)}; expected 2"
            )

    # Propagate face flips across the dual graph.  This simultaneously checks
    # connectedness and orientability, then supplies coherent cyclic orders to
    # the holonomy evaluator without requiring callers to pre-orient faces.
    dual_constraints: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge, occurrences in edge_occurrences.items():
        (face_a, u_a, v_a), (face_b, u_b, v_b) = occurrences
        direction_a = 1 if (u_a, v_a) == edge else -1
        direction_b = 1 if (u_b, v_b) == edge else -1
        relative_flip = -direction_a * direction_b
        dual_constraints[face_a].append((face_b, relative_flip))
        dual_constraints[face_b].append((face_a, relative_flip))

    face_sign = {0: 1}
    queue = deque([0])
    while queue:
        face_index = queue.popleft()
        for neighbor, relative_flip in dual_constraints[face_index]:
            required_sign = face_sign[face_index] * relative_flip
            if neighbor not in face_sign:
                face_sign[neighbor] = required_sign
                queue.append(neighbor)
            elif face_sign[neighbor] != required_sign:
                raise ValueError("triangulation is not orientable")
    if len(face_sign) != len(faces):
        raise ValueError("face adjacency graph is disconnected")

    # In a simplicial closed surface, the link of every vertex is one cycle.
    # Merely checking edge incidence misses pinched vertices with several link
    # components, so enforce the full local-manifold condition explicitly.
    for vertex, link_edges in vertex_links.items():
        link_adjacency: dict[int, list[int]] = defaultdict(list)
        for u, v in link_edges:
            link_adjacency[u].append(v)
            link_adjacency[v].append(u)
        bad_neighbors = [
            neighbor
            for neighbor, adjacent in link_adjacency.items()
            if len(adjacent) != 2
        ]
        if bad_neighbors:
            raise ValueError(
                f"link of vertex {vertex} is not a cycle: "
                f"degree is not 2 at {bad_neighbors[0]}"
            )
        link_seen: set[int] = set()
        link_queue = deque([next(iter(link_adjacency))])
        while link_queue:
            neighbor = link_queue.popleft()
            if neighbor in link_seen:
                continue
            link_seen.add(neighbor)
            link_queue.extend(link_adjacency[neighbor])
        if len(link_seen) != len(link_adjacency):
            raise ValueError(
                f"link of vertex {vertex} has more than one component"
            )

    vertex_labels = tuple(sorted(vertex_links))
    n_vertices = len(vertex_labels)
    n_edges = len(edge_occurrences)
    n_faces = len(faces)
    euler_characteristic = n_vertices - n_edges + n_faces
    if euler_characteristic != 2:
        raise ValueError(
            "closed connected triangulation is not a 2-sphere: "
            f"Euler characteristic is {euler_characteristic}, expected 2"
        )

    oriented_faces = tuple(
        face if face_sign[index] == 1 else (face[0], face[2], face[1])
        for index, face in enumerate(faces)
    )
    summary = ClosedSphereTriangulationSummary(
        vertex_labels=vertex_labels,
        n_vertices=n_vertices,
        n_edges=n_edges,
        n_faces=n_faces,
        euler_characteristic=euler_characteristic,
    )
    return oriented_faces, summary


def validate_closed_sphere_triangulation(
    triangles: Iterable[Sequence[int]],
) -> ClosedSphereTriangulationSummary:
    """Strictly validate a finite simplicial triangulation of the 2-sphere.

    The check rejects malformed or duplicate faces, non-integral or repeated
    vertex labels, boundary/non-manifold edges, disconnected or non-orientable
    face gluings, singular vertex links, and closed surfaces whose Euler
    characteristic is not two.  Input faces need not already have coherent
    orientations.
    """
    _, summary = _analyze_closed_sphere_triangulation(triangles)
    return summary

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
        Pv.append(x)
        x = parent[x]
    Pu = []
    x = u
    while x is not None:
        Pu.append(x)
        x = parent[x]
    i = len(Pv) - 1
    j = len(Pu) - 1
    while i >= 0 and j >= 0 and Pv[i] == Pu[j]:
        i -= 1
        j -= 1
    up = Pv[:i+1]
    down = list(reversed(Pu[:j+1]))
    return up + [Pv[i+1]] + down

def turn(order, side_in, side_out):
    i = order.index(side_in)
    j = order.index(side_out)
    d = (j - i) % 3
    if d == 1:
        return 'L'
    if d == 2:
        return 'R'
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
        if tf == a and tt == b:
            sign = +1
        elif tf == b and tt == a:
            sign = -1
        else:
            raise RuntimeError("Edge orientation inconsistent with step")
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
    """Evaluate a resolved dual path using its unoriented edge shears.

    The final field in an ``X`` token records which way the path crosses the
    chosen dual-edge orientation, but it does not negate the shear.  Indeed,
    ``X(z) @ X(z) == -I``, so ``X(z)`` is its own inverse in ``PSL(2, R)``.
    Using ``X(-z)`` for the reverse crossing would make an immediate
    backtrack nontrivial.
    """
    M = matI()
    for tok in tokens:
        if tok == 'L':
            M = matmul(M, matL())
        elif tok == 'R':
            M = matmul(M, matR())
        else:
            _, eid, _crossing_direction = tok
            z = Z.get(eid, 0.0)
            M = matmul(M, matX(z))
    return M


def _generator_token_paths(T: Triangulation, root=0):
    """Return closed holonomy words for every non-tree dual edge."""
    if not isinstance(root, Integral) or isinstance(root, bool):
        raise TypeError("root must be an integer triangle index")
    root = int(root)
    if root < 0 or root >= T.F:
        raise ValueError(f"root must be in [0, {T.F})")

    G = T.dual_graph()
    parent, parent_edge = bfs_spanning_tree(G, root=root)
    if len(parent) != T.F:
        raise ValueError("triangulation dual graph is disconnected")

    tree_edge_ids = {
        parent_edge[vertex][0]
        for vertex, parent_vertex in parent.items()
        if parent_vertex is not None
    }

    all_edges = set()
    for v, lst in G.items():
        for (w, sv, sw, eid) in lst:
            if v < w:
                all_edges.add((v, w, eid))

    def tree_crossings(path):
        crossings = []
        for triangle_a, triangle_b in zip(path, path[1:]):
            if parent[triangle_b] == triangle_a:
                eid, side_b, side_a = parent_edge[triangle_b]
            elif parent[triangle_a] == triangle_b:
                eid, side_a, side_b = parent_edge[triangle_a]
            else:
                raise RuntimeError("non-tree step in spanning-tree path")
            crossings.append(
                (triangle_a, triangle_b, eid, side_a, side_b)
            )
        return crossings

    paths = []
    for (u, v, eid) in sorted(all_edges):
        if eid in tree_edge_ids:
            continue
        path_u = path_in_tree(parent, root, u)
        path_v = path_in_tree(parent, root, v)

        middle = next(
            (
                (u, v, edge_id, side_u, side_v)
                for neighbor, side_u, side_v, edge_id in G[u]
                if neighbor == v and edge_id == eid
            ),
            None,
        )
        if middle is None:
            raise RuntimeError("non-tree edge is absent from adjacency")

        crossings = (
            tree_crossings(path_u)
            + [middle]
            + tree_crossings(list(reversed(path_v)))
        )
        tokens: list[HolonomyToken] = []
        root_reference_side = T.Order[root][0]
        first_outgoing_side = crossings[0][3]
        if root_reference_side != first_outgoing_side:
            tokens.append(
                turn(
                    T.Order[root],
                    root_reference_side,
                    first_outgoing_side,
                )
            )

        for index, (triangle_a, triangle_b, edge_id, side_a, side_b) in enumerate(
            crossings
        ):
            (from_triangle, _), (to_triangle, _) = T.Ori[edge_id]
            if from_triangle == triangle_a and to_triangle == triangle_b:
                sign = 1
            elif from_triangle == triangle_b and to_triangle == triangle_a:
                sign = -1
            else:
                raise RuntimeError("edge orientation is inconsistent with path")
            tokens.append(("X", edge_id, sign))

            if index + 1 < len(crossings):
                next_crossing = crossings[index + 1]
                if triangle_b != next_crossing[0]:
                    raise RuntimeError("dual cycle is not continuous")
                next_side = next_crossing[3]
                if side_b != next_side:
                    tokens.append(turn(T.Order[triangle_b], side_b, next_side))

        final_incoming_side = crossings[-1][4]
        if final_incoming_side != root_reference_side:
            tokens.append(
                turn(
                    T.Order[root],
                    final_incoming_side,
                    root_reference_side,
                )
            )

        paths.append((u, v, tokens))
    return paths


def generators_from_triangulation(T: Triangulation, Z, root=0):
    """
    Compute a generating set for the holonomy:
      one generator per non-tree dual edge.
    Returns: list of (u, v, tokens, matrix)
    """
    gens = []
    for u, v, tokens in _generator_token_paths(T, root=root):
        M = tokens_to_matrix(tokens, Z)
        gens.append((u, v, tokens, M))
    return gens


_EXACT_IDENTITY: ExactMatrix2x2 = ((1, 0), (0, 1))
_EXACT_LEFT: ExactMatrix2x2 = ((0, -1), (1, 1))
_EXACT_RIGHT: ExactMatrix2x2 = ((1, 1), (-1, 0))
_EXACT_ZERO_CROSSING: ExactMatrix2x2 = ((0, -1), (1, 0))


def _exact_matmul(
    matrix_a: ExactMatrix2x2,
    matrix_b: ExactMatrix2x2,
) -> ExactMatrix2x2:
    return (
        (
            matrix_a[0][0] * matrix_b[0][0]
            + matrix_a[0][1] * matrix_b[1][0],
            matrix_a[0][0] * matrix_b[0][1]
            + matrix_a[0][1] * matrix_b[1][1],
        ),
        (
            matrix_a[1][0] * matrix_b[0][0]
            + matrix_a[1][1] * matrix_b[1][0],
            matrix_a[1][0] * matrix_b[0][1]
            + matrix_a[1][1] * matrix_b[1][1],
        ),
    )


def _tokens_to_exact_zero_shear_matrix(
    tokens: Iterable[HolonomyToken],
) -> ExactMatrix2x2:
    matrix = _EXACT_IDENTITY
    for token in tokens:
        if token == "L":
            factor = _EXACT_LEFT
        elif token == "R":
            factor = _EXACT_RIGHT
        elif (
            isinstance(token, tuple)
            and len(token) == 3
            and token[0] == "X"
        ):
            # Crossing direction is informational: X(z) is a PSL involution.
            factor = _EXACT_ZERO_CROSSING
        else:
            raise ValueError(f"invalid resolved holonomy token: {token!r}")
        matrix = _exact_matmul(matrix, factor)
    return matrix


def exact_zero_shear_generators(
    triangulation: Triangulation,
    root: int = 0,
) -> tuple[ExactZeroShearGenerator, ...]:
    """Evaluate zero-shear holonomy generators using integer arithmetic.

    Unlike :func:`generators_from_triangulation`, this function has no
    floating-point or exponential step: each word is a product of the three
    integral matrices ``L``, ``R``, and ``X(0)``.  It certifies the declared
    abstract zero-shear structure, not shears inferred from coordinates.
    """
    generators = []
    for source, target, token_list in _generator_token_paths(
        triangulation, root=root
    ):
        tokens = tuple(token_list)
        matrix = _tokens_to_exact_zero_shear_matrix(tokens)
        generator = ExactZeroShearGenerator(
            source_triangle=source,
            target_triangle=target,
            tokens=tokens,
            matrix=matrix,
        )
        if generator.determinant != 1:
            raise RuntimeError("internal error: holonomy generator left SL(2,Z)")
        generators.append(generator)
    return tuple(generators)


def triangulation_from_faces(
    triangles: List[Tuple[int, int, int]],
) -> Tuple[Triangulation, Dict[int, float]]:
    """
    Build a Triangulation object from a list of triangle faces.

    Args:
        triangles: List of triangles, each as (v0, v1, v2) tuple of vertex indices

    Returns:
        Tuple of (Triangulation, Z) where Z is zero shear dict
    """
    faces = [tuple(triangle) for triangle in triangles]
    F = len(faces)
    adjacency = {}
    edge_id_map = {}
    edge_occurrences = {}

    # One pass over face sides replaces the former all-face/all-side scan.
    # Dictionary insertion order preserves the historical first-occurrence
    # edge-id assignment used by compute_shears_from_vertices.
    for face_index, face in enumerate(faces):
        if len(face) != 3:
            raise ValueError("every face must contain exactly three vertices")
        for side in range(3):
            u, v = face[side], face[(side + 1) % 3]
            edge = tuple(sorted((u, v)))
            edge_occurrences.setdefault(edge, []).append((face_index, side))

    orientation = {}
    for edge, occurrences in edge_occurrences.items():
        if len(occurrences) == 1:
            continue
        if len(occurrences) != 2 or occurrences[0][0] == occurrences[1][0]:
            raise ValueError(
                f"edge {edge} must occur in at most two distinct faces"
            )
        eid = len(edge_id_map)
        edge_id_map[edge] = eid
        (face_a, side_a), (face_b, side_b) = occurrences
        adjacency[(face_a, side_a)] = (face_b, side_b, eid)
        adjacency[(face_b, side_b)] = (face_a, side_a, eid)
        orientation[eid] = ((face_a, side_a), (face_b, side_b))

    order = {t: [0, 1, 2] for t in range(F)}

    T = Triangulation(F, adjacency, order, orientation)
    Z = {eid: 0.0 for eid in range(len(edge_id_map))}

    return T, Z


def certify_exact_zero_shear_structure(
    triangles: Iterable[Sequence[int]],
    root: int = 0,
) -> ExactZeroShearCertificate:
    """Certify a closed sphere's abstract zero-shear structure exactly.

    The faces are first subjected to strict closed-sphere validation and are
    coherently oriented internally.  Holonomy words are then evaluated over
    the integers, proving that their matrices lie in ``SL(2, Z)`` and hence
    define elements of ``PSL(2, Z)``.  The modular-group index follows from
    hyperbolic area: each ideal triangle has area pi while a modular
    fundamental domain has area pi/3, so the index is three times the number
    of faces.

    This API deliberately accepts no coordinates.  Its certificate applies to
    the zero-shear structure declared on the abstract triangulation; it does
    not prove that a separate numerical realization has zero shear.
    """
    oriented_faces, topology = _analyze_closed_sphere_triangulation(triangles)
    triangulation, _ = triangulation_from_faces(list(oriented_faces))
    generators = exact_zero_shear_generators(triangulation, root=root)
    certificate = ExactZeroShearCertificate(
        topology=topology,
        root_triangle=int(root),
        psl2z_index=3 * topology.n_faces,
        generators=generators,
    )
    if not certificate.all_determinants_one:
        raise RuntimeError("internal error: exact certificate left SL(2,Z)")
    return certificate


def _validated_shear_mapping(
    triangulation: Triangulation,
    shears: Mapping[int, Real],
) -> dict[int, float]:
    """Require one finite real shear for every triangulation edge."""
    if not isinstance(shears, Mapping):
        raise TypeError("shears must be a mapping from edge ids to real values")

    invalid_keys = [
        key
        for key in shears
        if not isinstance(key, Integral) or isinstance(key, bool)
    ]
    if invalid_keys:
        raise TypeError("shear edge ids must be integers")

    normalized_keys = {int(key) for key in shears}
    expected_keys = set(triangulation.Ori)
    missing = sorted(expected_keys - normalized_keys)
    extra = sorted(normalized_keys - expected_keys)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing edge ids {missing}")
        if extra:
            details.append(f"unexpected edge ids {extra}")
        raise ValueError(
            "shears must cover exactly the triangulation's interior edges: "
            + "; ".join(details)
        )

    normalized = {}
    for edge_id, value in shears.items():
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError(f"shear {edge_id} must be a finite real number")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"shear {edge_id} must be finite")
        normalized[int(edge_id)] = numeric_value
    return normalized


def check_arithmeticity(triangles: List[Tuple[int, int, int]],
                        shears: Optional[Dict[int, float]] = None,
                        vertices_complex=None,
                        tolerance: float = 0.01,
                        verbose: bool = False,
                        shear_tolerance: float = 1e-9) -> Dict[str, Any]:
    """
    Evaluate holonomy diagnostics and, when declared exactly, certify zero shear.

    Near-integrality of the traces of one chosen generating set is exposed as
    ``passes_generator_trace_diagnostic`` only.  It is not Takeuchi's
    criterion and is not an arithmeticity proof: traces of products and the
    invariant trace field are not checked.

    ``is_arithmetic`` is true only when the caller supplies a complete shear
    mapping whose values are all exactly zero.  In that case the separately
    implemented exact evaluator proves that every holonomy generator has
    integral matrix entries and the group is a finite-index subgroup of
    PSL(2,Z).  Zero shears merely inferred from floating-point coordinates are
    reported as a numerical diagnostic, never promoted to an exact proof.

    The certificate REQUIRES the actual geometry: pass either the shear
    parameters (`shears`, keyed by edge id as assigned by
    triangulation_from_faces) or the complex vertex coordinates
    (`vertices_complex`, entries may be None for the ideal vertex at
    infinity), from which shears are computed via
    compute_shears_from_vertices.

    The face complex must be a coherently oriented closed triangulated sphere.
    Shear mappings must cover exactly all edge ids and contain only finite real
    values; missing entries are never silently replaced by zero.

    Args:
        triangles: List of triangles as (v0, v1, v2) tuples
        shears: Complete Dict edge_id -> shear value, mutually exclusive with
            vertices_complex
        vertices_complex: sequence of complex vertex coordinates
        tolerance: Maximum distance from nearest integer to count as integral
        verbose: If True, print detailed trace information
        shear_tolerance: Maximum absolute shear for the numerical zero-shear
            diagnostic; this never creates an exact certificate

    Returns:
        Dict with keys:
        - 'is_arithmetic': bool, True only for declared exact zero shear
        - 'passes_generator_trace_diagnostic': bool, heuristic only
        - 'passes_zero_shear_diagnostic': bool, numerical diagnostic only
        - 'n_generators': int, number of holonomy generators
        - 'n_integral': int, number of generators with integral traces
        - 'traces': list of trace values
        - 'trace_details': list of dicts with trace analysis for each generator
        - 'shears': the shear dict actually used
    """
    if shears is None and vertices_complex is None:
        raise ValueError(
            "check_arithmeticity requires the actual geometry: pass "
            "`shears` or `vertices_complex`. (A geometry-free call would "
            "silently certify the idealized zero-shear structure, which is "
            "integral for ANY triangulation — a vacuous certificate.)")
    if shears is not None and vertices_complex is not None:
        raise ValueError("pass either `shears` or `vertices_complex`, not both")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not math.isfinite(shear_tolerance) or shear_tolerance < 0:
        raise ValueError("shear_tolerance must be finite and non-negative")

    input_faces = _materialize_integer_faces(triangles)
    oriented_faces, topology = _analyze_closed_sphere_triangulation(input_faces)
    if oriented_faces != input_faces:
        raise ValueError(
            "faces must have coherent cyclic orientations for shear holonomy"
        )

    triangulation, _ = triangulation_from_faces(list(input_faces))
    geometry_source = "declared_shears" if shears is not None else "coordinates"
    if shears is None:
        shears = compute_shears_from_vertices(
            list(input_faces), vertices_complex
        )
    validated_shears = _validated_shear_mapping(triangulation, shears)
    gens = generators_from_triangulation(
        triangulation, validated_shears, root=0
    )

    trace_details = []
    integral_count = 0

    for u, v, tokens, M in gens:
        trace = M[0][0] + M[1][1]
        nearest_int = round(trace)
        distance = abs(trace - nearest_int)
        is_integral = distance <= tolerance

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
            status = "NEAR-INTEGER" if is_integral else ""
            print(f"  Generator {u}-{v}: trace = {trace:.6f}, "
                  f"nearest int = {nearest_int}, dist = {distance:.6f} {status}")

    passes_generator_trace_diagnostic = integral_count == len(gens)
    max_abs_shear = max(map(abs, validated_shears.values()), default=0.0)
    passes_zero_shear_diagnostic = max_abs_shear <= shear_tolerance
    has_exact_zero_shears = all(
        value == 0.0 for value in validated_shears.values()
    )

    is_arithmetic = geometry_source == "declared_shears" and has_exact_zero_shears
    if is_arithmetic:
        # This checks topology again intentionally: the public exact API is the
        # authority for the proof represented by the compatibility field.
        certify_exact_zero_shear_structure(input_faces)
        arithmeticity_status = "certified_declared_exact_zero_shear"
        arithmeticity_reason = (
            "exact integral zero-shear matrices place the finite-covolume "
            "holonomy group in PSL(2,Z)"
        )
    else:
        arithmeticity_status = "not_certified"
        arithmeticity_reason = (
            "generator-trace and floating zero-shear checks are diagnostics, "
            "not arithmeticity proofs"
        )

    return {
        'is_arithmetic': is_arithmetic,
        'is_arithmetic_certified': is_arithmetic,
        'arithmeticity_status': arithmeticity_status,
        'arithmeticity_reason': arithmeticity_reason,
        'geometry_source': geometry_source,
        'passes_generator_trace_diagnostic': passes_generator_trace_diagnostic,
        'passes_zero_shear_diagnostic': passes_zero_shear_diagnostic,
        'has_exact_zero_shears': has_exact_zero_shears,
        'max_abs_shear': max_abs_shear,
        'n_generators': len(gens),
        'n_integral': integral_count,
        'traces': [d['trace'] for d in trace_details],
        'trace_details': trace_details,
        'shears': validated_shears,
        'topology': topology.as_dict(),
    }


def compute_shears_from_vertices(triangles: List[Tuple[int, int, int]],
                                  vertices_complex) -> Dict[int, float]:
    """
    Compute shear parameters from actual vertex positions.

    For each interior edge shared by two triangles, the shear is the log
    modulus of the STANDARD cross-ratio of the four vertices involved:
    on edge (x, y) shared by triangles (x, y, a) and (x, y, b),

        shear = log |CR(a, x; b, y)|,   CR = M(a)/M(b),  M(z) = (z-x)/(z-y)
              = log |(a-x)(b-y) / ((a-y)(b-x))|.

    A shear is stored once per unoriented primal edge.  Its sign comes from
    the cyclic surface orientation of the first incident face, not from the
    direction in which a dual path later crosses that edge.  Swapping the two
    coherently oriented incident faces leaves the value unchanged; reversing
    the orientation of the whole surface negates every shear and swaps left
    and right turns.  In either dual crossing direction the transition is the
    same projective involution X(shear).

    This vanishes exactly on Rivin/Delaunay-balanced (zero-shear)
    configurations — e.g. the regular ideal octahedron {0, +-1, +-i, INF}
    and any Moebius image of it.  (An earlier version used a different
    cross-ratio slot arrangement, which is a nonzero Klein-four translate
    of the standard one and reported |shear| ~ 0.35 on provably zero-shear
    configurations.)

    A vertex may be None, denoting the point at infinity of the upper
    half-space model; the cross-ratio is then evaluated by its Moebius
    limit (the two factors containing that vertex cancel).

    Args:
        triangles: List of triangles as (v0, v1, v2) tuples
        vertices_complex: sequence of complex vertex coordinates
            (entries may be None for the ideal vertex at infinity)

    Returns:
        Dict mapping edge_id -> shear value (edge ids consistent with
        triangulation_from_faces: first-occurrence order over interior
        edges)
    """
    def _log_distance(z, w):
        distance = abs(z - w)
        if distance == 0.0 or not math.isfinite(distance):
            raise ValueError("cross-ratio contains a zero or non-finite distance")
        return math.log(distance)

    def _log_cross_ratio_modulus(a, x, b, y):
        """Stable log|CR(a,x;b,y)|, with None denoting infinity."""
        if x is None and y is None:
            raise ValueError("Degenerate edge: both endpoints at infinity")
        if x is None:
            return _log_distance(b, y) - _log_distance(a, y)
        if y is None:
            return _log_distance(a, x) - _log_distance(b, x)
        if a is None and b is None:
            return 0.0
        if a is None:
            return _log_distance(b, y) - _log_distance(b, x)
        if b is None:
            return _log_distance(a, x) - _log_distance(a, y)
        return (
            _log_distance(a, x)
            + _log_distance(b, y)
            - _log_distance(a, y)
            - _log_distance(b, x)
        )

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

    def _coord(idx):
        v = vertices_complex[idx]
        if v is None:
            return None
        coordinate = complex(v)
        if not math.isfinite(coordinate.real) or not math.isfinite(
            coordinate.imag
        ):
            raise ValueError(f"vertex {idx} has a non-finite coordinate")
        return coordinate

    for edge, tri_list in edge_triangles.items():
        if len(tri_list) == 2:  # Interior edge
            # Edge endpoints (x, y) and the two opposite vertices (a, b)
            _, opp1, e1, e2 = tri_list[0]
            _, opp2, _, _ = tri_list[1]

            a = _coord(opp1)
            x = _coord(e1)
            y = _coord(e2)
            b = _coord(opp2)

            try:
                shear = _log_cross_ratio_modulus(a, x, b, y)
            except (OverflowError, ZeroDivisionError, ValueError) as error:
                raise ValueError(
                    f"cannot compute a finite shear for edge {edge}: {error}"
                ) from error
            if not math.isfinite(shear):
                raise ValueError(f"non-finite shear for edge {edge}")

            shears[edge_id] = shear
            edge_id += 1

    return shears


def _closed_delaunay_sphere_from_vertices(vertices_complex):
    """Build a closed N-vertex sphere via an existing vertex at infinity."""
    import numpy as np
    from scipy.spatial import ConvexHull, Delaunay, QhullError

    raw_vertices = list(vertices_complex)
    if len(raw_vertices) < 4:
        raise ValueError("at least four vertices are required for a sphere")

    infinity_indices = [
        index for index, vertex in enumerate(raw_vertices) if vertex is None
    ]
    if len(infinity_indices) > 1:
        raise ValueError("at most one vertex may be the point at infinity")

    finite_vertices: dict[int, complex] = {}
    for index, vertex in enumerate(raw_vertices):
        if vertex is None:
            continue
        coordinate = complex(vertex)
        if not math.isfinite(coordinate.real) or not math.isfinite(
            coordinate.imag
        ):
            raise ValueError(f"vertex {index} has a non-finite coordinate")
        finite_vertices[index] = coordinate
    if len(set(finite_vertices.values())) != len(finite_vertices):
        raise ValueError("vertex coordinates must be pairwise distinct")

    if infinity_indices:
        pole_candidates = infinity_indices
    else:
        # Trying several existing vertices makes the construction robust to a
        # numerically unfortunate choice of Mobius pole without adding a new
        # ideal vertex.
        pole_candidates = list(reversed(range(len(raw_vertices))))

    for pole in pole_candidates:
        pole_coordinate = raw_vertices[pole]
        projected: dict[int, complex] = {}
        try:
            for index, coordinate in finite_vertices.items():
                if index == pole:
                    continue
                if pole_coordinate is None:
                    projected[index] = coordinate
                else:
                    projected[index] = 1.0 / (
                        coordinate - complex(pole_coordinate)
                    )
        except ZeroDivisionError:
            continue

        center = sum(projected.values()) / len(projected)
        scale = max(abs(value - center) for value in projected.values())
        if scale == 0.0 or not math.isfinite(scale):
            continue
        normalized = {
            index: (value - center) / scale
            for index, value in projected.items()
        }
        if any(
            not math.isfinite(value.real) or not math.isfinite(value.imag)
            for value in normalized.values()
        ):
            continue

        finite_indices = list(normalized)
        points = np.array(
            [
                [normalized[index].real, normalized[index].imag]
                for index in finite_indices
            ],
            dtype=float,
        )
        try:
            delaunay = Delaunay(points)
            hull = ConvexHull(points)
        except QhullError:
            continue

        faces: list[tuple[int, int, int]] = []
        for simplex in delaunay.simplices:
            local_a, local_b, local_c = map(int, simplex)
            point_a, point_b, point_c = points[[local_a, local_b, local_c]]
            signed_area = (
                (point_b[0] - point_a[0]) * (point_c[1] - point_a[1])
                - (point_b[1] - point_a[1]) * (point_c[0] - point_a[0])
            )
            if signed_area < 0:
                local_b, local_c = local_c, local_b
            faces.append(
                (
                    finite_indices[local_a],
                    finite_indices[local_b],
                    finite_indices[local_c],
                )
            )

        hull_cycle = [finite_indices[int(local)] for local in hull.vertices]
        hull_area = sum(
            normalized[current].real * normalized[following].imag
            - normalized[current].imag * normalized[following].real
            for current, following in zip(
                hull_cycle, hull_cycle[1:] + hull_cycle[:1]
            )
        )
        if hull_area < 0:
            hull_cycle.reverse()
        for current, following in zip(
            hull_cycle, hull_cycle[1:] + hull_cycle[:1]
        ):
            faces.append((pole, following, current))

        projected_vertices: list[complex | None] = [None] * len(raw_vertices)
        for index, coordinate in normalized.items():
            projected_vertices[index] = coordinate

        try:
            oriented_faces, _ = _analyze_closed_sphere_triangulation(faces)
        except (TypeError, ValueError):
            continue
        if oriented_faces != tuple(faces):
            continue
        return faces, projected_vertices, pole

    raise ValueError(
        "vertices do not admit a nondegenerate closed Delaunay triangulation"
    )


def check_arithmeticity_from_vertices(vertices_complex, tolerance: float = 0.01,
                                      verbose: bool = False,
                                      shear_tolerance: float = 1e-9,
                                      ) -> Dict[str, Any]:
    """
    Compute non-vacuous holonomy diagnostics from complex vertex coordinates.

    One existing vertex is mapped to infinity, the remaining N-1 vertices are
    Delaunay triangulated, and an infinity fan closes the complex using that
    same pole.  Thus the analysis is on a closed sphere with exactly the input
    N vertices, rather than the former finite Delaunay disk whose dual graph
    could be a tree and make the generator diagnostic vacuous.

    Coordinate-derived results are numerical diagnostics only;
    ``is_arithmetic`` remains false.  Use
    :func:`certify_exact_zero_shear_structure` for an exact abstract
    zero-shear certificate.

    Args:
        vertices_complex: Array of complex numbers (vertex positions)
        tolerance: Maximum distance from nearest integer to count as integral
        verbose: If True, print detailed trace information
        shear_tolerance: Numerical zero-shear diagnostic tolerance

    Returns:
        Same as check_arithmeticity(), plus the closed faces and pole index
    """
    triangles, projected_vertices, pole = _closed_delaunay_sphere_from_vertices(
        vertices_complex
    )
    result = check_arithmeticity(
        triangles,
        vertices_complex=projected_vertices,
        tolerance=tolerance,
        verbose=verbose,
        shear_tolerance=shear_tolerance,
    )
    result.update(
        {
            "triangles": triangles,
            "pole_vertex": pole,
            "triangulation_source": "mobius-pole-delaunay-with-infinity-fan",
        }
    )
    return result


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Evaluate Penner-Rivin holonomy arithmeticity diagnostics.",
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
  ARITHMETIC CERTIFIED - Declared exact zero-shear certificate (exit code 0)
  NOT CERTIFIED        - Diagnostics alone do not prove arithmeticity (exit code 1)
        """
    )
    parser.add_argument('input', nargs='?', help='JSON file with configuration')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed trace information')
    parser.add_argument('--tolerance', '-t', type=float, default=0.01,
                        help='Tolerance for integrality check (default: 0.01)')

    args = parser.parse_args()

    if args.input is None:
        print("Penner-Rivin Holonomy Arithmeticity Diagnostics")
        print("=" * 50)
        print()
        print("This module computes holonomy diagnostics. Generator-trace")
        print("integrality alone is not an arithmeticity certificate.")
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

    # Check if we have vertices or triangles.  Triangles alone are no longer
    # sufficient: the certificate requires the actual geometry (coordinates
    # or shears) — see check_arithmeticity.
    if 'triangles' in config and ('vertices_real' in config and 'vertices_imag' in config):
        import numpy as np
        triangles = [tuple(t) for t in config['triangles']]
        vertices = np.array(config['vertices_real']) + 1j * np.array(config['vertices_imag'])
        result = check_arithmeticity(triangles, vertices_complex=vertices,
                                     tolerance=args.tolerance, verbose=args.verbose)
    elif 'triangles' in config:
        print("Error: 'triangles' alone cannot certify arithmeticity — the "
              "zero-shear default is vacuous. Provide 'vertices_real'/'vertices_imag' as well.")
        sys.exit(2)
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
        print("Result: ARITHMETIC CERTIFIED")
        sys.exit(0)
    else:
        print("Result: NOT CERTIFIED")
        sys.exit(1)
