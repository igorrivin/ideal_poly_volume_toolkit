"""Regression tests for the canonical H³ volume kernels."""

import numpy as np
import pytest
import torch

from ideal_poly_volume_toolkit.geometry import (
    bloch_wigner,
    bloch_wigner_mpmath,
    bloch_wigner_torch,
    delaunay_triangulation_indices,
    ideal_poly_volume_from_sphere,
    ideal_poly_volume_via_delaunay,
    lift_to_sphere_with_inf,
    lob_fast,
    stereographic_project_from_pole,
    triangle_volume_bloch_wigner,
    triangle_volume_from_points,
)


REGULAR_IDEAL_TETRAHEDRON_VOLUME = 1.014941606409653625


def _regular_tetrahedron_on_sphere() -> np.ndarray:
    return np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    ) / np.sqrt(3.0)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    return rotation


def _mobius_extended(
    values: np.ndarray,
    a: complex,
    b: complex,
    c: complex,
    d: complex,
) -> np.ndarray:
    transformed = []
    for z in values:
        if np.isinf(z):
            transformed.append(a / c if c != 0.0 else complex(np.inf))
            continue
        denominator = c * z + d
        transformed.append(
            (a * z + b) / denominator if denominator != 0.0 else complex(np.inf)
        )
    return np.asarray(transformed, dtype=np.complex128)


def test_regular_ideal_tetrahedron_volume() -> None:
    shape = np.exp(1j * np.pi / 3.0)
    assert bloch_wigner(shape) == pytest.approx(
        REGULAR_IDEAL_TETRAHEDRON_VOLUME, abs=2e-15
    )
    assert bloch_wigner(shape) == pytest.approx(bloch_wigner_mpmath(shape), abs=2e-15)

    finite_vertices = np.array([0.0j, 1.0 + 0.0j, shape])
    assert ideal_poly_volume_via_delaunay(
        finite_vertices, use_bloch_wigner=True
    ) == pytest.approx(REGULAR_IDEAL_TETRAHEDRON_VOLUME, abs=2e-15)

    sphere_vertices = _regular_tetrahedron_on_sphere()
    for pole_index in range(4):
        assert ideal_poly_volume_from_sphere(
            sphere_vertices, pole_index=pole_index
        ) == pytest.approx(REGULAR_IDEAL_TETRAHEDRON_VOLUME, abs=2e-15)


def test_bloch_wigner_scalar_and_batch_agree() -> None:
    values = np.array([0.3 + 0.7j, -0.4 + 0.8j, 1.2 + 0.4j, 0.0j, 1.0 + 0.0j])
    batch = bloch_wigner(values)
    scalar = np.array([bloch_wigner(z) for z in values])
    np.testing.assert_allclose(batch, scalar, rtol=0.0, atol=0.0)

    z1 = torch.tensor([0.0 + 0.0j, -0.2 + 0.1j], dtype=torch.complex128)
    z2 = torch.tensor([1.0 + 0.0j, 0.8 - 0.3j], dtype=torch.complex128)
    z3 = torch.tensor([np.exp(1j * np.pi / 3.0), 0.4 + 1.1j], dtype=torch.complex128)
    batch_torch = triangle_volume_bloch_wigner(z1, z2, z3)
    scalar_torch = torch.stack(
        [triangle_volume_bloch_wigner(z1[i], z2[i], z3[i]) for i in range(2)]
    )
    torch.testing.assert_close(batch_torch, scalar_torch, rtol=0.0, atol=0.0)


def test_vectorized_delaunay_paths_match_scalar_triangle_sums() -> None:
    vertices = np.array([0.0j, 1.0 + 0.0j, 0.2 + 0.8j, -0.5 + 0.4j, 2.0 - 0.3j])
    triangles = delaunay_triangulation_indices(vertices)

    expected_series = sum(
        triangle_volume_from_points(*vertices[triangle], mode="fast", series_terms=96)
        for triangle in triangles
    )
    actual_series = ideal_poly_volume_via_delaunay(
        vertices, mode="fast", series_terms=96
    )
    assert actual_series == pytest.approx(expected_series, abs=2e-15)

    expected_bloch_wigner = sum(
        abs(bloch_wigner((vertices[i] - vertices[k]) / (vertices[j] - vertices[k])))
        for i, j, k in triangles
    )
    actual_bloch_wigner = ideal_poly_volume_via_delaunay(
        vertices, use_bloch_wigner=True
    )
    assert actual_bloch_wigner == pytest.approx(expected_bloch_wigner, abs=2e-15)


def test_sphere_volume_is_rotation_pole_and_scale_invariant() -> None:
    rng = np.random.default_rng(1234)
    points = rng.standard_normal((9, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    expected = ideal_poly_volume_from_sphere(points, pole_index=0)

    for pole_index in (2, 5, 8):
        assert ideal_poly_volume_from_sphere(
            points, pole_index=pole_index
        ) == pytest.approx(expected, rel=2e-14, abs=2e-14)

    for _ in range(5):
        rotated = points @ _random_rotation(rng)
        assert ideal_poly_volume_from_sphere(rotated) == pytest.approx(
            expected, rel=2e-14, abs=2e-14
        )

    positive_row_scales = rng.uniform(0.1, 10.0, size=(len(points), 1))
    assert ideal_poly_volume_from_sphere(points * positive_row_scales) == pytest.approx(
        expected, rel=2e-14, abs=2e-14
    )

    for extreme_scale in (1e-200, 1e200):
        assert ideal_poly_volume_from_sphere(points * extreme_scale) == pytest.approx(
            expected, rel=2e-14, abs=2e-14
        )


def test_stereographic_projection_mapping_and_actual_degeneracy() -> None:
    points = _regular_tetrahedron_on_sphere()
    projected, original_indices = stereographic_project_from_pole(points, 2)
    assert projected.shape == (3,)
    np.testing.assert_array_equal(original_indices, np.array([0, 1, 3]))

    scaled, scaled_indices = stereographic_project_from_pole(
        points * np.array([[2.0], [0.5], [7.0], [3.0]]), 2
    )
    np.testing.assert_array_equal(scaled_indices, original_indices)
    np.testing.assert_allclose(scaled, projected, rtol=2e-15, atol=2e-15)

    duplicated = points.copy()
    duplicated[3] = duplicated[2]
    with pytest.raises(ValueError, match="projection pole is repeated"):
        stereographic_project_from_pole(duplicated, 2)


def test_stereographic_projection_is_stable_near_the_north_pole() -> None:
    rng = np.random.default_rng(9)
    other_points = rng.standard_normal((8, 3))
    other_points /= np.linalg.norm(other_points, axis=1, keepdims=True)
    angle = 1e-8
    near_north_pole = np.array([np.sin(angle), 0.0, np.cos(angle)])
    points = np.vstack((near_north_pole, other_points))

    expected = ideal_poly_volume_from_sphere(points, pole_index=1)
    assert ideal_poly_volume_from_sphere(points, pole_index=0) == pytest.approx(
        expected, rel=2e-14, abs=2e-14
    )


def test_volume_is_invariant_under_similarity_and_general_mobius_map() -> None:
    shape = np.exp(1j * np.pi / 3.0)
    finite = np.array([0.0j, 1.0 + 0.0j, shape])
    expected = ideal_poly_volume_via_delaunay(finite, use_bloch_wigner=True)

    similarity = (2.7 - 1.3j) * finite + (-0.4 + 0.9j)
    assert ideal_poly_volume_via_delaunay(
        similarity, use_bloch_wigner=True
    ) == pytest.approx(expected, rel=2e-14, abs=2e-14)

    extended = np.array([complex(np.inf), *finite], dtype=np.complex128)
    transformed = _mobius_extended(
        extended,
        a=1.2 + 0.3j,
        b=-0.4 + 0.2j,
        c=0.25 - 0.15j,
        d=0.9 + 0.1j,
    )
    original_sphere = lift_to_sphere_with_inf(extended)
    transformed_sphere = lift_to_sphere_with_inf(transformed)
    assert ideal_poly_volume_from_sphere(transformed_sphere) == pytest.approx(
        ideal_poly_volume_from_sphere(original_sphere),
        rel=2e-14,
        abs=2e-14,
    )


def test_lobachevsky_custom_backward_matches_its_forward_series() -> None:
    theta = torch.tensor([0.1, 0.7, 1.2, 3.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value: lob_fast(value, 96),
        (theta,),
        eps=1e-6,
        atol=2e-6,
        rtol=2e-5,
    )


def test_bloch_wigner_custom_backward_matches_finite_differences() -> None:
    real = torch.tensor([0.3, -0.4, 1.2], dtype=torch.float64, requires_grad=True)
    imag = torch.tensor([0.7, 0.8, 0.4], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        bloch_wigner_torch,
        (real, imag),
        eps=1e-6,
        atol=2e-6,
        rtol=2e-5,
    )

    x = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    y = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    bloch_wigner_torch(x, y).backward()
    step = 1e-6
    x_value = x.detach().item()
    y_value = y.detach().item()
    finite_difference_x = (
        bloch_wigner(complex(x_value + step, y_value))
        - bloch_wigner(complex(x_value - step, y_value))
    ) / (2.0 * step)
    finite_difference_y = (
        bloch_wigner(complex(x_value, y_value + step))
        - bloch_wigner(complex(x_value, y_value - step))
    ) / (2.0 * step)
    assert float(x.grad) == pytest.approx(finite_difference_x, rel=2e-8, abs=2e-8)
    assert float(y.grad) == pytest.approx(finite_difference_y, rel=2e-8, abs=2e-8)


@pytest.mark.parametrize(
    "shape",
    [
        1e-18 + 1e-18j,
        np.nextafter(1.0, 0.0) + 1e-18j,
    ],
)
def test_bloch_wigner_custom_backward_is_stable_near_endpoints(shape) -> None:
    x_value = shape.real
    y_value = shape.imag
    x = torch.tensor(x_value, dtype=torch.float64, requires_grad=True)
    y = torch.tensor(y_value, dtype=torch.float64, requires_grad=True)
    bloch_wigner_torch(x, y).backward()

    radius_sq = x_value * x_value + y_value * y_value
    complement_sq = (1.0 - x_value) ** 2 + y_value * y_value
    if complement_sq < 0.25:
        delta_x = x_value - 1.0
        log_radius = 0.5 * np.log1p(
            2.0 * delta_x + delta_x * delta_x + y_value * y_value
        )
    else:
        log_radius = 0.5 * np.log(radius_sq)
    if radius_sq < 0.25:
        log_complement = 0.5 * np.log1p(
            -2.0 * x_value + x_value * x_value + y_value * y_value
        )
    else:
        log_complement = 0.5 * np.log(complement_sq)

    expected_x = (
        -y_value * log_radius / complement_sq + y_value * log_complement / radius_sq
    )
    expected_y = (
        -(1.0 - x_value) * log_radius / complement_sq
        - x_value * log_complement / radius_sq
    )
    assert float(x.grad) == pytest.approx(expected_x, rel=2e-14, abs=2e-14)
    assert float(y.grad) == pytest.approx(expected_y, rel=2e-14, abs=2e-14)
