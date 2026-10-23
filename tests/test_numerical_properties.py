"""Deterministic property tests for the core numerical invariants."""

from itertools import combinations

import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st

from ideal_poly_volume_toolkit.fast_logdet import IncrementalLogDet
from ideal_poly_volume_toolkit.geometry import (
    bloch_wigner,
    delaunay_triangulation_indices,
    ideal_poly_volume_from_sphere,
)
from ideal_poly_volume_toolkit.optimization import (
    analyze_ideal_polyhedron_structure,
    complete_boundary_triangulation,
)


PROPERTY_SETTINGS = settings(max_examples=32, derandomize=True)


@st.composite
def _well_separated_complex(draw):
    """Complex values bounded away from the removable points zero and one."""
    real = draw(
        st.floats(
            min_value=-2.0,
            max_value=3.0,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        )
    )
    imag_magnitude = draw(
        st.floats(
            min_value=0.125,
            max_value=3.0,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        )
    )
    imag_sign = draw(st.sampled_from((-1.0, 1.0)))
    return complex(real, imag_sign * imag_magnitude)


@PROPERTY_SETTINGS
@given(st.lists(_well_separated_complex(), min_size=1, max_size=12))
def test_bloch_wigner_symmetries_and_scalar_batch_agreement(values):
    """The vectorized kernel obeys the classical six-fold symmetries."""
    z = np.asarray(values, dtype=np.complex128)
    batch = bloch_wigner(z)
    scalar = np.asarray([bloch_wigner(value) for value in z])

    np.testing.assert_array_equal(batch, scalar)
    np.testing.assert_allclose(
        bloch_wigner(np.conjugate(z)),
        -batch,
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        bloch_wigner(1.0 - z),
        -batch,
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        bloch_wigner(1.0 - 1.0 / z),
        batch,
        rtol=2e-13,
        atol=2e-13,
    )


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    return rotation


def _minimum_affine_four_point_determinant(points: np.ndarray) -> float:
    """Condition against four nearly coplanar sphere points."""
    determinants = (
        abs(np.linalg.det(np.column_stack((points[list(indices)], np.ones(4)))))
        for indices in combinations(range(len(points)), 4)
    )
    return min(determinants)


@settings(max_examples=16, derandomize=True)
@given(
    n_vertices=st.integers(min_value=4, max_value=9),
    point_seed=st.integers(min_value=0, max_value=2**32 - 1),
    transform_seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_sphere_volume_is_invariant_under_scaling_rotation_and_pole(
    n_vertices,
    point_seed,
    transform_seed,
):
    """Generic sphere representatives define a chart-independent volume."""
    point_rng = np.random.default_rng(point_seed)
    points = point_rng.standard_normal((n_vertices, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)

    pairwise_distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    nonzero_distances = pairwise_distances[np.triu_indices(n_vertices, k=1)]
    assume(float(nonzero_distances.min()) > 0.05)
    assume(_minimum_affine_four_point_determinant(points) > 1e-7)

    expected = ideal_poly_volume_from_sphere(points, pole_index=0)
    for pole_index in range(1, n_vertices):
        assert ideal_poly_volume_from_sphere(
            points,
            pole_index=pole_index,
        ) == pytest.approx(expected, rel=2e-11, abs=2e-11)

    transform_rng = np.random.default_rng(transform_seed)
    row_scales = transform_rng.uniform(0.125, 8.0, size=(n_vertices, 1))
    assert ideal_poly_volume_from_sphere(
        points * row_scales,
        pole_index=n_vertices - 1,
    ) == pytest.approx(expected, rel=2e-11, abs=2e-11)

    rotated = points @ _random_rotation(transform_rng)
    assert ideal_poly_volume_from_sphere(
        rotated,
        pole_index=n_vertices // 2,
    ) == pytest.approx(expected, rel=2e-11, abs=2e-11)


def _minimum_planar_triangle_area(points: np.ndarray) -> float:
    twice_areas = (
        abs(
            (points[second, 0] - points[first, 0])
            * (points[third, 1] - points[first, 1])
            - (points[second, 1] - points[first, 1])
            * (points[third, 0] - points[first, 0])
        )
        for first, second, third in combinations(range(len(points)), 3)
    )
    return min(twice_areas) / 2.0


def _minimum_cocircular_determinant(points: np.ndarray) -> float:
    lifted = np.column_stack(
        (points, np.sum(points * points, axis=1), np.ones(len(points)))
    )
    determinants = (
        abs(np.linalg.det(lifted[list(indices)]))
        for indices in combinations(range(len(points)), 4)
    )
    return min(determinants)


def _directed_edge_incidence(faces: np.ndarray):
    incidence: dict[tuple[int, int], list[int]] = {}
    for face in faces:
        for first, second in zip(face, np.roll(face, -1)):
            u, v = int(first), int(second)
            edge = (u, v) if u < v else (v, u)
            incidence.setdefault(edge, []).append(1 if (u, v) == edge else -1)
    return incidence


def _assert_closed_vertex_links_are_cycles(
    faces: np.ndarray,
    n_vertices: int,
) -> None:
    for vertex in range(n_vertices):
        link: dict[int, set[int]] = {}
        for face in faces:
            if vertex not in face:
                continue
            opposite = [int(value) for value in face if value != vertex]
            first, second = opposite
            link.setdefault(first, set()).add(second)
            link.setdefault(second, set()).add(first)

        assert link
        assert all(len(neighbours) == 2 for neighbours in link.values())
        seen: set[int] = set()
        pending = [next(iter(link))]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(link[current] - seen)
        assert seen == set(link)


@settings(max_examples=20, derandomize=True)
@given(
    n_finite_vertices=st.integers(min_value=3, max_value=10),
    point_seed=st.integers(min_value=0, max_value=2**32 - 1),
    winding_seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_delaunay_boundary_closure_is_an_oriented_manifold_sphere(
    n_finite_vertices,
    point_seed,
    winding_seed,
):
    """Closing a conditioned planar Delaunay disk yields an oriented sphere."""
    point_rng = np.random.default_rng(point_seed)
    points = point_rng.uniform(-2.0, 2.0, size=(n_finite_vertices, 2))

    pairwise_distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    nonzero_distances = pairwise_distances[
        np.triu_indices(n_finite_vertices, k=1)
    ]
    assume(float(nonzero_distances.min()) > 1e-4)
    assume(_minimum_planar_triangle_area(points) > 1e-7)
    if n_finite_vertices >= 4:
        assume(_minimum_cocircular_determinant(points) > 1e-7)

    finite_vertices = points[:, 0] + 1j * points[:, 1]
    finite_faces = delaunay_triangulation_indices(finite_vertices)
    winding_rng = np.random.default_rng(winding_seed)
    flipped = winding_rng.random(len(finite_faces)) < 0.5
    finite_faces[flipped] = finite_faces[flipped][:, [0, 2, 1]]

    boundary = complete_boundary_triangulation(
        finite_faces,
        n_finite_vertices,
    )
    incidence = _directed_edge_incidence(boundary)
    n_vertices = n_finite_vertices + 1

    assert set(map(int, boundary.ravel())) == set(range(n_vertices))
    assert all(sorted(directions) == [-1, 1] for directions in incidence.values())
    assert n_vertices - len(incidence) + len(boundary) == 2
    assert len(boundary) == 2 * n_vertices - 4
    _assert_closed_vertex_links_are_cycles(boundary, n_vertices)

    structure = analyze_ideal_polyhedron_structure(finite_vertices, finite_faces)
    assert structure["euler_characteristic"] == 2
    np.testing.assert_array_equal(structure["boundary_triangulation"], boundary)


def _prism_adjacency(n_vertices: int, permutation_seed: int) -> np.ndarray:
    half = n_vertices // 2
    neighbours = [set() for _ in range(n_vertices)]
    for index in range(half):
        for first, second in (
            (index, (index + 1) % half),
            (half + index, half + (index + 1) % half),
            (index, half + index),
        ):
            neighbours[first].add(second)
            neighbours[second].add(first)

    adjacency = np.asarray([sorted(row) for row in neighbours], dtype=np.int32)
    permutation = np.random.default_rng(permutation_seed).permutation(n_vertices)
    relabelled = permutation[adjacency]
    inverse = np.argsort(permutation)
    return np.sort(relabelled[inverse], axis=1).astype(np.int32, copy=False)


def _edge_set(adjacency: np.ndarray) -> set[tuple[int, int]]:
    return {
        (vertex, int(neighbour))
        for vertex, row in enumerate(adjacency)
        for neighbour in row
        if vertex < neighbour
    }


def _fresh_swap(
    adjacency: np.ndarray,
    removed_edges: tuple[tuple[int, int], tuple[int, int]],
    added_edges: tuple[tuple[int, int], tuple[int, int]],
) -> np.ndarray | None:
    neighbours = [set(map(int, row)) for row in adjacency]
    for first, second in removed_edges:
        neighbours[first].remove(second)
        neighbours[second].remove(first)
    for first, second in added_edges:
        if first == second or second in neighbours[first]:
            return None
        neighbours[first].add(second)
        neighbours[second].add(first)
    if any(len(row) != 3 for row in neighbours):
        return None
    return np.asarray([sorted(row) for row in neighbours], dtype=np.int32)


def _is_connected(adjacency: np.ndarray) -> bool:
    seen = {0}
    pending = [0]
    while pending:
        current = pending.pop()
        for neighbour in adjacency[current]:
            neighbour = int(neighbour)
            if neighbour not in seen:
                seen.add(neighbour)
                pending.append(neighbour)
    return len(seen) == len(adjacency)


def _valid_two_switches(adjacency: np.ndarray):
    candidates = []
    for first_edge, second_edge in combinations(sorted(_edge_set(adjacency)), 2):
        a, b = first_edge
        c, d = second_edge
        for added_edges in (((a, c), (b, d)), ((a, d), (b, c))):
            candidate = _fresh_swap(
                adjacency,
                (first_edge, second_edge),
                added_edges,
            )
            if candidate is not None and _is_connected(candidate):
                candidates.append((first_edge, second_edge, added_edges, candidate))
    return candidates


def _fresh_log_spanning_tree_count(adjacency: np.ndarray) -> float:
    n_vertices = len(adjacency)
    laplacian = np.zeros((n_vertices, n_vertices), dtype=np.float64)
    for vertex, row in enumerate(adjacency):
        laplacian[vertex, vertex] = len(row)
        laplacian[vertex, row] = -1.0
    sign, logdet = np.linalg.slogdet(laplacian[:-1, :-1])
    assert sign > 0.0
    return float(logdet)


# Numba may compile the incremental kernel on the first example. A wall-clock
# deadline would measure that one-time compilation rather than this property.
@settings(max_examples=16, derandomize=True, deadline=None)
@given(
    n_vertices=st.sampled_from((6, 8, 10, 12, 14)),
    permutation_seed=st.integers(min_value=0, max_value=2**32 - 1),
    swap_selector=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_incremental_logdet_accepted_swap_matches_full_recomputation(
    n_vertices,
    permutation_seed,
    swap_selector,
):
    adjacency = _prism_adjacency(n_vertices, permutation_seed)
    candidates = _valid_two_switches(adjacency)
    assert candidates
    first_edge, second_edge, added_edges, expected_adjacency = candidates[
        swap_selector % len(candidates)
    ]

    expected_before = _fresh_log_spanning_tree_count(adjacency)
    expected_after = _fresh_log_spanning_tree_count(expected_adjacency)
    incremental = IncrementalLogDet(adjacency)
    adjacency_before = incremental.adj.copy()

    delta = incremental.swap_log_det_ratio(
        first_edge,
        second_edge,
        added_edges[0],
        added_edges[1],
    )
    np.testing.assert_array_equal(incremental.adj, adjacency_before)
    np.testing.assert_allclose(
        delta,
        expected_after - expected_before,
        rtol=5e-11,
        atol=5e-11,
    )

    assert incremental.apply_swap(
        first_edge,
        second_edge,
        added_edges[0],
        added_edges[1],
    )
    np.testing.assert_array_equal(incremental.adj, expected_adjacency)
    np.testing.assert_allclose(
        incremental.log_spanning_trees(),
        expected_after,
        rtol=5e-11,
        atol=5e-11,
    )
