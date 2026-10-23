"""Focused regressions for numerical and package-integration bugs."""

import cmath
import importlib

import numpy as np
import pytest


def test_numerical_modules_import_through_package_namespace():
    pointset = importlib.import_module("ideal_poly_volume_toolkit.pointset_to_fuchsian")
    models4 = importlib.import_module(
        "ideal_poly_volume_toolkit.hyperbolic_4_simplex_volume_models"
    )
    models56 = importlib.import_module(
        "ideal_poly_volume_toolkit.hyperbolic_5_6_simplex_volume"
    )

    assert callable(pointset.generators_from_triangulation)
    assert callable(models4.convert_vertices)
    assert callable(models56.convert_vertices)


def test_coplanar_cube_facets_are_reconstructed_as_squares():
    from ideal_poly_volume_toolkit.pointset_to_fuchsian import (
        convex_hull_from_points_on_sphere,
    )

    cube = np.array(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
        dtype=float,
    ) / np.sqrt(3.0)

    first = convex_hull_from_points_on_sphere(cube)
    second = convex_hull_from_points_on_sphere(cube)

    assert len(first["faces"]) == 6
    assert all(len(face) == 4 for face in first["faces"])
    assert len(first["edges"]) == 12
    assert set().union(*(set(face) for face in first["faces"])) == set(range(8))
    assert first["faces"] == second["faces"]


def test_unsigned_bloch_wigner_volume_gradient_matches_finite_difference():
    from ideal_poly_volume_toolkit.pointset_to_fuchsian import (
        volume_and_gradient_from_pointset,
    )

    rng = np.random.default_rng(4)
    points = list(rng.normal(size=6) + 1j * rng.normal(size=6))
    apex = len(points)  # the fixed north pole appended below
    _, gradient = volume_and_gradient_from_pointset(
        points, apex_index=apex, add_north_pole=True
    )

    eps = 1e-6
    finite_difference = np.zeros((len(points), 2), dtype=float)
    for vertex in range(len(points)):
        for coordinate, delta in enumerate((eps, 1j * eps)):
            plus = points.copy()
            minus = points.copy()
            plus[vertex] += delta
            minus[vertex] -= delta
            value_plus, _ = volume_and_gradient_from_pointset(
                plus, apex_index=apex, add_north_pole=True
            )
            value_minus, _ = volume_and_gradient_from_pointset(
                minus, apex_index=apex, add_north_pole=True
            )
            finite_difference[vertex, coordinate] = (value_plus - value_minus) / (
                2.0 * eps
            )

    np.testing.assert_allclose(
        gradient[: len(points)], finite_difference, rtol=2e-6, atol=2e-7
    )


@pytest.mark.parametrize(
    "shape",
    [
        1e-16 + 1e-16j,
        1.0 + 1e-16j,
    ],
)
def test_near_degenerate_bloch_wigner_gradient_is_not_flattened(shape):
    from ideal_poly_volume_toolkit.geometry import bloch_wigner
    from ideal_poly_volume_toolkit.pointset_to_fuchsian import _G_of_shape

    differential = _G_of_shape(shape)
    analytic = np.array([differential.imag, differential.real])

    step = 1e-20
    finite_difference = np.array(
        [
            (bloch_wigner(shape + step) - bloch_wigner(shape - step)) / (2.0 * step),
            (bloch_wigner(shape + 1j * step) - bloch_wigner(shape - 1j * step))
            / (2.0 * step),
        ]
    )

    assert np.linalg.norm(analytic) > 1.0
    np.testing.assert_allclose(analytic, finite_difference, rtol=2e-5, atol=2e-6)


def test_near_coincident_cross_ratio_uses_the_true_reciprocal():
    from ideal_poly_volume_toolkit.pointset_to_fuchsian import (
        INF,
        _G_of_shape,
        _dlog_cr_coeffs,
    )

    separation = 1e-16
    _, coefficient_x, _, _ = _dlog_cr_coeffs(
        INF,
        -separation,
        0.0,
        -1.0,
    )

    assert coefficient_x == pytest.approx(1.0 / separation, rel=1e-15)

    assert _G_of_shape(0.0) == 0.0j
    assert _G_of_shape(1.0) == 0.0j
    _, singular_coefficient_x, _, _ = _dlog_cr_coeffs(INF, 0.0, 0.0, -1.0)
    assert singular_coefficient_x == 0.0j


def test_halfspace_numpy_arrays_are_not_treated_as_infinity_sentinels():
    from ideal_poly_volume_toolkit.hyperbolic_model_transforms import (
        convert_point,
        convert_vertices,
        halfspace_to_minkowski,
    )

    point = np.array([0.1, 0.2, 1.0])
    expected = halfspace_to_minkowski(point.tolist())

    np.testing.assert_allclose(
        np.asarray(halfspace_to_minkowski(point), dtype=float),
        np.asarray(expected, dtype=float),
    )
    converted = convert_point(point, "halfspace", "klein")
    converted_many = convert_vertices([point, point.copy()], "halfspace", "klein")
    assert len(converted) == 3
    assert len(converted_many) == 2


def test_threshold_default_is_accurate_and_explicit_series_terms_propagate():
    from ideal_poly_volume_toolkit.volume_threshold import (
        compute_volume,
        get_volume,
        volume_exceeds_threshold,
    )

    vertices = [
        0j,
        1 + 0j,
        cmath.exp(1j * cmath.pi / 3),
        np.inf,
    ]
    regular_ideal_tetrahedron_volume = 1.0149416064096537

    default_volume = get_volume(vertices)
    assert default_volume == pytest.approx(
        regular_ideal_tetrahedron_volume, rel=0.0, abs=2e-15
    )
    assert volume_exceeds_threshold(vertices, regular_ideal_tetrahedron_volume - 1e-12)
    assert not volume_exceeds_threshold(
        vertices, regular_ideal_tetrahedron_volume + 1e-12
    )

    series_8 = compute_volume(vertices, method="delaunay", series_terms=8)
    series_64 = compute_volume(vertices, method="delaunay", series_terms=64)
    hull_8 = compute_volume(vertices, method="hull", series_terms=8)
    assert series_8 != pytest.approx(series_64, rel=0.0, abs=1e-6)
    assert hull_8 == pytest.approx(series_8, rel=0.0, abs=1e-14)

    with pytest.raises(ValueError, match="series_terms"):
        compute_volume(vertices, series_terms=0)
