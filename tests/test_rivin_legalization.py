"""Focused tests for deterministic Rivin worst-edge legalization."""

import numpy as np
import pytest
from scipy.spatial import Delaunay

import ideal_poly_volume_toolkit.rivin_delaunay as rd


def _random_delaunay(n_vertices, seed):
    points = np.random.default_rng(seed).random((n_vertices, 2))
    triangles = [tuple(int(x) for x in face) for face in Delaunay(points).simplices]
    return points, triangles


def _directed_edge_counts(triangles):
    counts = {}
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((u, v)))
            counts.setdefault(edge, []).append((u, v))
    return counts


def test_flip_edge_preserves_cyclic_orientation():
    triangles = [(0, 1, 2), (0, 2, 3)]

    flipped = rd.flip_edge(triangles, (0, 2))

    assert flipped is not None
    assert {tuple(sorted(face)) for face in flipped} == {
        (1, 2, 3),
        (0, 1, 3),
    }
    for incidences in _directed_edge_counts(flipped).values():
        if len(incidences) == 2:
            assert incidences[0] == incidences[1][::-1]


def test_flip_edge_refuses_inconsistent_shared_edge_winding():
    triangles = [(0, 1, 2), (0, 3, 2)]

    assert rd.flip_edge(triangles, (0, 2)) is None


def test_compute_rivin_angle_residuals_for_equilateral_grid():
    width = height = 3

    def vertex(i, j):
        return j * width + i

    triangles = []
    for j in range(height - 1):
        for i in range(width - 1):
            a = vertex(i, j)
            b = vertex(i + 1, j)
            c = vertex(i, j + 1)
            d = vertex(i + 1, j + 1)
            triangles.extend(((a, b, c), (d, c, b)))
    angles = np.full((len(triangles), 3), np.pi / 3.0)

    residuals = rd.compute_rivin_angle_residuals(triangles, angles)

    assert residuals["feasible"]
    assert not residuals["strict_interior"]
    assert residuals["max_equality_residual"] < 1e-12
    assert residuals["min_dihedral_margin"] == pytest.approx(np.pi / 3.0)
    # Straight lattice sides meet the boundary constraint at equality.
    assert residuals["min_boundary_margin"] == pytest.approx(0.0, abs=1e-12)
    assert residuals["worst_dihedral_edge"] is not None


@pytest.mark.parametrize("scale", [1e-150, 1.0, 1e150])
def test_triangle_angle_is_scale_stable_near_collinear(scale):
    epsilon = 1e-12
    origin = np.array([0.0, 0.0])
    positive_ray = np.array([scale, 0.0])
    almost_parallel = np.array([scale, scale * epsilon])
    almost_opposite = np.array([-scale, scale * epsilon])

    small_angle = rd.compute_triangle_angle(origin, positive_ray, almost_parallel)
    large_angle = rd.compute_triangle_angle(origin, positive_ray, almost_opposite)

    assert small_angle == pytest.approx(np.arctan(epsilon), rel=1e-12)
    assert large_angle == pytest.approx(np.pi - np.arctan(epsilon), rel=1e-15)


def test_triangle_angle_handles_zero_length_ray_explicitly():
    point = np.array([2.0, -3.0])
    assert rd.compute_triangle_angle(point, point, np.array([4.0, 5.0])) == 0.0


def test_legalization_is_deterministic_bounded_and_reports_residuals():
    _, triangles = _random_delaunay(15, seed=2)
    original = list(triangles)

    first = rd.legalize_to_interior_maximum(triangles, max_flips=8)
    second = rd.legalize_to_interior_maximum(triangles, max_flips=8)
    bounded = rd.legalize_to_interior_maximum(triangles, max_flips=0)

    assert triangles == original  # no caller-owned mutation
    assert first["success"]
    assert first["termination_reason"] == "accepted"
    assert first["n_flips"] == 1
    assert first["n_states"] == 2
    assert first["finite_triangulation"] == second["finite_triangulation"]
    assert [row["flipped_edge"] for row in first["history"]] == [
        row["flipped_edge"] for row in second["history"]
    ]
    assert first["history"][0]["status"] == "flipped"
    assert first["history"][-1]["status"] == "accepted"
    for incidences in _directed_edge_counts(first["finite_triangulation"]).values():
        if len(incidences) == 2:
            assert incidences[0] == incidences[1][::-1]
    assert first["residuals"]["max_equality_residual"] <= 1e-6
    assert first["residuals"]["min_dihedral_margin"] >= 1e-4
    assert first["residuals"]["min_boundary_margin"] >= 1e-4
    assert first["optimization_result"]["n_active_constraints"] == 0
    assert first["optimization_result"]["reduced_gradient_norm"] < 1e-10
    assert first["timings"]["total_seconds"] >= 0.0

    assert not bounded["success"]
    assert bounded["termination_reason"] == "max_flips"
    assert bounded["n_flips"] == 0
    assert bounded["finite_triangulation"] == original


@pytest.mark.parametrize(
    "triangles",
    [
        [(0, 1, 2), (3, 4, 5)],
        [(0, 1, 2), (0, 3, 4)],
    ],
    ids=["disconnected", "pinched"],
)
def test_legalization_rejects_non_disk_triangulations(triangles):
    with pytest.raises(ValueError):
        rd.legalize_to_interior_maximum(triangles)


def test_legalization_rejects_noncontiguous_vertex_labels():
    with pytest.raises(ValueError, match="contiguous"):
        rd.legalize_to_interior_maximum([(1, 2, 3)])


def test_legalization_normalizes_mixed_face_winding():
    _, triangles = _random_delaunay(15, seed=2)
    mixed = list(triangles)
    a, b, c = mixed[-1]
    mixed[-1] = (a, c, b)

    result = rd.legalize_to_interior_maximum(mixed, max_flips=8)

    assert result["success"]
    for incidences in _directed_edge_counts(result["finite_triangulation"]).values():
        if len(incidences) == 2:
            assert incidences[0] == incidences[1][::-1]


def test_legalization_recomputes_residuals_with_caller_tolerance():
    _, triangles = _random_delaunay(15, seed=2)
    tolerance = 2.5e-7

    result = rd.legalize_to_interior_maximum(
        triangles,
        max_flips=8,
        constraint_tolerance=tolerance,
    )

    assert result["residuals"]["tolerance"] == tolerance
    assert result["optimization_result"]["residuals"]["tolerance"] == tolerance


def test_legalization_cycle_detection(monkeypatch):
    triangles = [(0, 1, 2), (0, 2, 3)]

    def fake_lp(current, boundary_vertices=None, verbose=False):
        del boundary_vertices, verbose
        return {
            "realizable": True,
            "min_angle": np.pi / 3.0,
            "angles_radians": np.full((len(current), 3), np.pi / 3.0),
            "message": "fake feasible",
        }

    def fake_optimizer(
        current,
        initial_angles=None,
        boundary_vertices=None,
        verbose=False,
        require_interior=False,
    ):
        del initial_angles, verbose, require_interior
        angles = np.full((len(current), 3), np.pi / 3.0)
        adjacency = rd.build_edge_adjacency(current)
        interior_edge = next(
            edge for edge, incidences in adjacency.items() if len(incidences) == 2
        )
        for triangle_id, corner_id in adjacency[interior_edge]:
            angles[triangle_id, corner_id] = 0.6 * np.pi
        residuals = rd.compute_rivin_angle_residuals(
            current,
            angles,
            boundary_vertices,
        )
        return {
            "success": False,
            "angles": angles,
            "volume": None,
            "message": "fake active tie",
            "rejected_tie": True,
            "n_iterations": 1,
            "n_active_constraints": 0,
            "reduced_gradient_norm": 0.0,
            "residuals": residuals,
        }

    monkeypatch.setattr(rd, "check_delaunay_realizability", fake_lp)
    monkeypatch.setattr(rd, "optimize_hyperbolic_volume", fake_optimizer)

    result = rd.legalize_to_interior_maximum(triangles, max_flips=8)

    assert not result["success"]
    assert result["termination_reason"] == "cycle_detected"
    assert result["n_flips"] == 2
    assert result["n_states"] == 2
    assert all(row["status"] == "flipped" for row in result["history"])


def test_realization_reports_extended_residuals():
    _, triangles = _random_delaunay(15, seed=2)
    legalized = rd.legalize_to_interior_maximum(triangles, max_flips=8)
    assert legalized["success"]

    realized = rd.realize_angles_as_points(
        legalized["finite_triangulation"],
        legalized["angles"],
    )

    assert realized["success"]
    assert realized["orientation_consistent"]
    assert realized["max_angle_error"] >= realized["angle_error"]
    assert realized["max_angle_error_degrees"] >= realized["angle_error_degrees"]
    assert realized["residuals"]["closure_error"] == realized["closure_error"]
    assert realized["residuals"]["target_angle_structure"]["strict_interior"]
    assert realized["residuals"]["realized_angle_structure"] is not None
