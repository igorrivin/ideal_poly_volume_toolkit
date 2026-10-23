"""Focused tests for the differentiable fixed-chart H⁴ volume kernel."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import Delaunay
import subprocess
import sys
import textwrap
import torch

from ideal_poly_volume_toolkit.geometry_4d import (
    ideal_4simplex_volume_from_tet,
    ideal_polytope_volume_4d,
    regular_ideal_4simplex_volume,
)
from ideal_poly_volume_toolkit.geometry_4d_torch import (
    ideal_4simplex_volumes_from_projected_tetrahedra,
    ideal_polytope_volume_4d_from_projected,
    projected_similarity_gauge,
)


def _inverse_stereographic_with_infinity(points: np.ndarray) -> np.ndarray:
    squared_norms = np.sum(points * points, axis=1, keepdims=True)
    finite = np.concatenate(
        (
            2.0 * points / (squared_norms + 1.0),
            (squared_norms - 1.0) / (squared_norms + 1.0),
        ),
        axis=1,
    )
    return np.vstack((np.array([[0.0, 0.0, 0.0, 1.0]]), finite))


def _random_projected_configuration(
    seed: int = 1729,
    n_points: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.random.default_rng(seed).standard_normal((n_points, 3))
    simplices = Delaunay(points).simplices.astype(np.int64)
    return points, simplices


def test_regular_ideal_4simplex_matches_numpy_formula() -> None:
    points = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    simplices = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    volumes = ideal_4simplex_volumes_from_projected_tetrahedra(
        points,
        simplices,
    )

    assert volumes.shape == (1,)
    assert float(volumes[0]) == pytest.approx(
        regular_ideal_4simplex_volume(),
        rel=2e-14,
        abs=2e-14,
    )


def test_vectorized_projected_polytope_matches_both_numpy_paths() -> None:
    points, simplices = _random_projected_configuration()
    points_torch = torch.tensor(points, dtype=torch.float64)
    simplices_torch = torch.tensor(simplices, dtype=torch.long)

    simplex_volumes = ideal_4simplex_volumes_from_projected_tetrahedra(
        points_torch,
        simplices_torch,
    )
    expected_simplices = np.array(
        [ideal_4simplex_volume_from_tet(points[simplex]) for simplex in simplices]
    )
    np.testing.assert_allclose(
        simplex_volumes.detach().numpy(),
        expected_simplices,
        rtol=2e-13,
        atol=2e-13,
    )

    expected_polytope, expected_count = ideal_polytope_volume_4d(
        _inverse_stereographic_with_infinity(points),
        pole_idx=0,
    )
    actual_polytope = ideal_polytope_volume_4d_from_projected(
        points_torch,
        simplices_torch,
    )
    assert expected_count == len(simplices)
    assert float(actual_polytope) == pytest.approx(
        expected_polytope,
        rel=2e-13,
        abs=2e-13,
    )


def test_simplex_labels_and_row_order_are_preserved() -> None:
    points, simplices = _random_projected_configuration(seed=2718, n_points=10)
    expected = ideal_polytope_volume_4d_from_projected(
        torch.tensor(points, dtype=torch.float64),
        torch.tensor(simplices, dtype=torch.long),
    )

    point_permutation = np.random.default_rng(314).permutation(len(points))
    inverse_permutation = np.empty_like(point_permutation)
    inverse_permutation[point_permutation] = np.arange(len(points))
    permuted_points = points[point_permutation]
    relabelled_simplices = inverse_permutation[simplices]
    relabelled_simplices = relabelled_simplices[::-1, ::-1].copy()
    actual = ideal_polytope_volume_4d_from_projected(
        torch.tensor(permuted_points, dtype=torch.float64),
        torch.tensor(relabelled_simplices, dtype=torch.long),
    )

    assert float(actual) == pytest.approx(float(expected), rel=3e-14, abs=3e-14)


def test_similarity_gauge_is_differentiable_and_removes_four_null_directions() -> None:
    points, simplices = _random_projected_configuration(seed=99)
    raw = torch.tensor(points, dtype=torch.float64, requires_grad=True)
    simplices_torch = torch.tensor(simplices, dtype=torch.long)
    gauged = projected_similarity_gauge(raw)

    torch.testing.assert_close(
        torch.mean(gauged, dim=0),
        torch.zeros(3, dtype=torch.float64),
        rtol=0.0,
        atol=2e-16,
    )
    mean_squared_radius = torch.mean(torch.sum(gauged * gauged, dim=1)).detach()
    assert float(mean_squared_radius) == pytest.approx(
        1.0,
        rel=2e-15,
        abs=2e-15,
    )

    volume = ideal_polytope_volume_4d_from_projected(
        raw,
        simplices_torch,
        apply_similarity_gauge=True,
    )
    volume.backward()
    assert raw.grad is not None
    torch.testing.assert_close(
        torch.sum(raw.grad, dim=0),
        torch.zeros(3, dtype=torch.float64),
        rtol=0.0,
        atol=5e-14,
    )
    scale_directional_derivative = torch.sum(raw.grad * raw).detach()
    assert float(scale_directional_derivative) == pytest.approx(0.0, abs=5e-13)

    transformed = 7.25 * points + np.array([100.0, -30.0, 4.5])
    transformed_volume = ideal_polytope_volume_4d_from_projected(
        torch.tensor(transformed, dtype=torch.float64),
        simplices_torch,
        apply_similarity_gauge=True,
    )
    assert float(transformed_volume) == pytest.approx(
        float(volume.detach()),
        rel=2e-13,
        abs=2e-13,
    )


def test_autograd_matches_central_finite_difference_in_a_stable_chamber() -> None:
    points, simplices = _random_projected_configuration(seed=8675309, n_points=8)
    raw = torch.tensor(points, dtype=torch.float64, requires_grad=True)
    simplices_torch = torch.tensor(simplices, dtype=torch.long)
    volume = ideal_polytope_volume_4d_from_projected(
        raw,
        simplices_torch,
        apply_similarity_gauge=True,
    )
    (gradient,) = torch.autograd.grad(volume, raw)

    direction = np.random.default_rng(123).standard_normal(points.shape)
    direction /= np.linalg.norm(direction)
    step = 1.0e-6

    def evaluate(offset: float) -> float:
        displaced = torch.tensor(
            points + offset * direction,
            dtype=torch.float64,
        )
        return float(
            ideal_polytope_volume_4d_from_projected(
                displaced,
                simplices_torch,
                apply_similarity_gauge=True,
            )
        )

    finite_difference = (evaluate(step) - evaluate(-step)) / (2.0 * step)
    analytic = float(torch.sum(gradient * torch.tensor(direction)))
    assert analytic == pytest.approx(finite_difference, rel=2e-7, abs=2e-7)


@pytest.mark.parametrize(
    ("points", "simplices", "error", "match"),
    [
        (
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, float("nan")],
                ],
                dtype=torch.float64,
            ),
            torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            ValueError,
            "finite",
        ),
        (
            torch.eye(4, 3, dtype=torch.float64),
            torch.tensor([[0, 1, 2, 4]], dtype=torch.long),
            ValueError,
            "labels",
        ),
        (
            torch.eye(4, 3, dtype=torch.float64),
            torch.tensor([[0, 1, 1, 3]], dtype=torch.long),
            ValueError,
            "distinct",
        ),
        (
            torch.eye(4, 3, dtype=torch.float64),
            torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
            TypeError,
            "int64",
        ),
        (
            torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
                dtype=torch.float64,
            ),
            torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            ValueError,
            "degenerate",
        ),
    ],
)
def test_invalid_projected_inputs_fail_explicitly(
    points: torch.Tensor,
    simplices: torch.Tensor,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        ideal_polytope_volume_4d_from_projected(points, simplices)


def test_similarity_gauge_rejects_collapsed_points() -> None:
    collapsed = torch.ones((5, 3), dtype=torch.float64)
    with pytest.raises(ValueError, match="insufficient finite spread"):
        projected_similarity_gauge(collapsed)


def test_numpy_h4_geometry_stays_importable_without_torch() -> None:
    script = textwrap.dedent(
        """
        import builtins
        import importlib
        import sys

        real_import = builtins.__import__

        def without_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise ModuleNotFoundError("Torch deliberately blocked", name="torch")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = without_torch
        geometry_4d = importlib.import_module("ideal_poly_volume_toolkit.geometry_4d")
        assert geometry_4d.regular_ideal_4simplex_volume() > 0.0
        assert "torch" not in sys.modules

        try:
            importlib.import_module("ideal_poly_volume_toolkit.geometry_4d_torch")
        except ImportError as exc:
            assert "ideal_poly_volume_toolkit[torch]" in str(exc)
        else:
            raise AssertionError("Torch H4 geometry unexpectedly imported")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
