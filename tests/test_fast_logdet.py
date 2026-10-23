"""Regression tests for incremental graph-Laplacian log determinants."""

import numpy as np

from ideal_poly_volume_toolkit.fast_logdet import IncrementalLogDet


CUBE_ADJACENCY = np.array(
    [
        [1, 3, 4],
        [0, 2, 5],
        [1, 3, 6],
        [0, 2, 7],
        [0, 5, 7],
        [1, 4, 6],
        [2, 5, 7],
        [3, 4, 6],
    ],
    dtype=np.int32,
)


def _fresh_log_spanning_tree_count(adjacency):
    n = len(adjacency)
    laplacian = np.zeros((n, n), dtype=float)
    for vertex, neighbours in enumerate(adjacency):
        laplacian[vertex, vertex] = len(neighbours)
        for neighbour in neighbours:
            laplacian[vertex, int(neighbour)] -= 1.0
    sign, logdet = np.linalg.slogdet(laplacian[:-1, :-1])
    return float(logdet) if sign > 0 else -np.inf


def _apply_swap_fresh(adjacency, removed_edges, added_edges):
    neighbours = [set(int(u) for u in row) for row in adjacency]
    for u, v in removed_edges:
        neighbours[u].remove(v)
        neighbours[v].remove(u)
    for u, v in added_edges:
        neighbours[u].add(v)
        neighbours[v].add(u)
    return np.asarray([sorted(row) for row in neighbours], dtype=adjacency.dtype)


def test_apply_swap_commits_adjacency_and_rebuild_preserves_it():
    removed = ((0, 1), (2, 3))
    added = ((0, 2), (1, 3))
    expected_adjacency = _apply_swap_fresh(CUBE_ADJACENCY, removed, added)
    expected_before = _fresh_log_spanning_tree_count(CUBE_ADJACENCY)
    expected_after = _fresh_log_spanning_tree_count(expected_adjacency)

    incremental = IncrementalLogDet(CUBE_ADJACENCY)
    original_adjacency = incremental.adj.copy()
    delta = incremental.swap_log_det_ratio(*removed, *added)

    # A trial ratio must not mutate state.
    np.testing.assert_array_equal(incremental.adj, original_adjacency)
    np.testing.assert_allclose(
        delta, expected_after - expected_before, rtol=1e-12, atol=1e-12
    )

    assert incremental.apply_swap(*removed, *added)
    np.testing.assert_array_equal(incremental.adj, expected_adjacency)
    np.testing.assert_allclose(
        incremental.log_spanning_trees(), expected_after, rtol=1e-12, atol=1e-12
    )

    # This was the original regression: rebuild used the stale initial graph.
    incremental._rebuild()
    np.testing.assert_array_equal(incremental.adj, expected_adjacency)
    np.testing.assert_allclose(
        incremental.log_spanning_trees(), expected_after, rtol=1e-12, atol=1e-12
    )


def test_disconnected_graph_has_negative_infinite_logdet():
    # Disjoint union of two K4 components; every row still has cubic degree.
    disconnected = np.array(
        [
            [1, 2, 3],
            [0, 2, 3],
            [0, 1, 3],
            [0, 1, 2],
            [5, 6, 7],
            [4, 6, 7],
            [4, 5, 7],
            [4, 5, 6],
        ],
        dtype=np.int32,
    )

    incremental = IncrementalLogDet(disconnected)
    assert np.isneginf(incremental.log_spanning_trees())
    assert incremental._chol is None

    incremental._rebuild()
    assert np.isneginf(incremental.log_spanning_trees())
    assert incremental._chol is None


def test_invalid_swap_is_transactional():
    incremental = IncrementalLogDet(CUBE_ADJACENCY)
    adjacency_before = incremental.adj.copy()
    logdet_before = incremental.log_spanning_trees()

    # (0, 2) is not an edge in the cube, so this is not a valid removal.
    assert not incremental.apply_swap((0, 2), (1, 5), (0, 1), (2, 5))
    np.testing.assert_array_equal(incremental.adj, adjacency_before)
    assert incremental.log_spanning_trees() == logdet_before
