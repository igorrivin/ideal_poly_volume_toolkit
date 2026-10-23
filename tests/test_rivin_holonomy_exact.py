"""Exact zero-shear certification and strict topology regression tests."""

import json
import math

import pytest

from ideal_poly_volume_toolkit.rivin_holonomy import (
    certify_exact_zero_shear_structure,
    check_arithmeticity,
    check_arithmeticity_from_vertices,
    compute_shears_from_vertices,
    exact_zero_shear_generators,
    generators_from_triangulation,
    matmul,
    tokens_to_matrix,
    triangulation_from_faces,
    validate_closed_sphere_triangulation,
)


TETRAHEDRON_FACES = [
    (0, 2, 1),
    (0, 1, 3),
    (1, 2, 3),
    (2, 0, 3),
]


def _bipyramid_faces(n: int) -> list[tuple[int, int, int]]:
    """Return coherently oriented faces of an n-gonal bipyramid."""
    top = n + 1
    faces = []
    for index in range(n):
        current = 1 + index
        following = 1 + (index + 1) % n
        faces.append((0, current, following))
        faces.append((top, following, current))
    return faces


def _seven_vertex_torus_faces() -> list[tuple[int, int, int]]:
    """The 14-face Csaszar triangulation of the torus."""
    faces = []
    for index in range(7):
        faces.append((index, (index + 1) % 7, (index + 3) % 7))
        faces.append((index, (index + 2) % 7, (index + 3) % 7))
    return faces


def test_strict_sphere_validation_and_orientation_normalization():
    summary = validate_closed_sphere_triangulation(TETRAHEDRON_FACES)

    assert summary.vertex_labels == (0, 1, 2, 3)
    assert summary.n_vertices == 4
    assert summary.n_edges == 6
    assert summary.n_faces == 4
    assert summary.euler_characteristic == 2

    # Input winding need not be coherent: the exact certificate normalizes it
    # without changing face indices or the resulting based generators.
    one_face_reversed = list(TETRAHEDRON_FACES)
    a, b, c = one_face_reversed[2]
    one_face_reversed[2] = (a, c, b)
    validate_closed_sphere_triangulation(one_face_reversed)
    assert (
        certify_exact_zero_shear_structure(one_face_reversed).generators
        == certify_exact_zero_shear_structure(TETRAHEDRON_FACES).generators
    )


@pytest.mark.parametrize(
    ("faces", "error", "match"),
    [
        ([(0, 1, 2)], ValueError, "incidence 1"),
        (
            TETRAHEDRON_FACES + [(0, 1, 4)],
            ValueError,
            "incidence 3",
        ),
        (
            TETRAHEDRON_FACES + [(10, 12, 11), (10, 11, 13),
                                 (11, 12, 13), (12, 10, 13)],
            ValueError,
            "disconnected",
        ),
        (
            TETRAHEDRON_FACES + [(1, 2, 0)],
            ValueError,
            "duplicates",
        ),
        (_seven_vertex_torus_faces(), ValueError, "Euler characteristic is 0"),
        ([(0, 1, True)], TypeError, "non-negative integers"),
        ([(0, 1, -2)], ValueError, "non-negative"),
        ([(0, 1, 1)], ValueError, "repeated vertices"),
    ],
)
def test_strict_sphere_validation_rejects_invalid_complexes(
    faces, error, match
):
    with pytest.raises(error, match=match):
        validate_closed_sphere_triangulation(faces)


def test_triangulation_from_faces_keeps_edge_ids_and_boundary_behavior():
    faces = [(10, 11, 12), (12, 11, 13), (12, 13, 14)]
    triangulation, shears = triangulation_from_faces(faces)

    assert shears == {0: 0.0, 1: 0.0}
    assert triangulation.Adj == {
        (0, 1): (1, 0, 0),
        (1, 0): (0, 1, 0),
        (1, 2): (2, 0, 1),
        (2, 0): (1, 2, 1),
    }
    assert triangulation.Ori == {
        0: ((0, 1), (1, 0)),
        1: ((1, 2), (2, 0)),
    }


def test_exact_generators_match_zero_shear_float_words_and_matrices():
    triangulation, zero_shears = triangulation_from_faces(TETRAHEDRON_FACES)
    exact = exact_zero_shear_generators(triangulation)
    floating = generators_from_triangulation(triangulation, zero_shears)

    assert len(exact) == 3
    for exact_generator, (source, target, tokens, matrix) in zip(
        exact, floating, strict=True
    ):
        assert (exact_generator.source_triangle, exact_generator.target_triangle) == (
            source,
            target,
        )
        assert exact_generator.tokens == tuple(tokens)
        assert exact_generator.matrix == tuple(tuple(int(x) for x in row) for row in matrix)
        assert exact_generator.determinant == 1
        assert all(type(entry) is int for row in exact_generator.matrix for entry in row)

    # The three fundamental dual cycles encircle punctures, hence are
    # parabolic rather than the spurious trace-zero words produced when turns
    # at concatenation points are omitted.
    assert {generator.trace for generator in exact} <= {-2, 2}
    projective_matrices = {
        min(generator.matrix, tuple(tuple(-x for x in row) for row in generator.matrix))
        for generator in exact
    }
    assert len(projective_matrices) == 3


def test_reverse_crossing_reuses_shear_and_backtrack_is_projective_identity():
    shear = 1.25
    matrix = tokens_to_matrix(
        [("X", 0, 1), ("X", 0, -1)],
        {0: shear},
    )

    assert matrix[0] == pytest.approx([-1.0, 0.0])
    assert matrix[1] == pytest.approx([0.0, -1.0])


def test_coordinate_shears_are_oriented_by_faces_not_crossing_direction():
    coordinates = [0j, 1 + 0j, 0.2 + 1.3j, -0.7 + 0.4j]
    faces = [(0, 1, 2), (1, 0, 3)]
    swapped_faces = [faces[1], faces[0]]
    reversed_surface = [(a, c, b) for a, b, c in faces]

    shear = compute_shears_from_vertices(faces, coordinates)[0]
    swapped_shear = compute_shears_from_vertices(swapped_faces, coordinates)[0]
    reversed_shear = compute_shears_from_vertices(
        reversed_surface, coordinates
    )[0]

    assert shear != pytest.approx(0.0)
    assert swapped_shear == pytest.approx(shear)
    assert reversed_shear == pytest.approx(-shear)


def test_nonzero_complete_shears_still_give_parabolic_cusp_loops():
    coordinates = [0j, 1 + 0j, 0.2 + 1.3j, -0.7 + 0.4j]
    triangulation, _ = triangulation_from_faces(TETRAHEDRON_FACES)
    shears = compute_shears_from_vertices(TETRAHEDRON_FACES, coordinates)

    assert max(abs(shear) for shear in shears.values()) > 0.5
    generators = generators_from_triangulation(triangulation, shears)
    traces = [matrix[0][0] + matrix[1][1] for *_, matrix in generators]
    assert [abs(trace) for trace in traces] == pytest.approx([2.0, 2.0, 2.0])


def test_basis_trace_integrality_is_only_a_diagnostic_in_common_frame():
    coordinates = [0j, 1 + 0j, 1j, None]
    triangulation, _ = triangulation_from_faces(TETRAHEDRON_FACES)
    shears = compute_shears_from_vertices(TETRAHEDRON_FACES, coordinates)
    generators = generators_from_triangulation(triangulation, shears)

    product = matmul(generators[0][3], generators[1][3])
    product_trace = product[0][0] + product[1][1]
    assert product_trace == pytest.approx(6 + 3 * math.sqrt(2))
    assert abs(product_trace - round(product_trace)) > 0.2

    result = check_arithmeticity(
        TETRAHEDRON_FACES,
        vertices_complex=coordinates,
        tolerance=1e-12,
    )
    assert result["passes_generator_trace_diagnostic"]
    assert not result["is_arithmetic"]
    assert not result["is_arithmetic_certified"]
    assert result["arithmeticity_status"] == "not_certified"


def test_declared_exact_zero_shears_are_the_certified_compatibility_path():
    triangulation, _ = triangulation_from_faces(TETRAHEDRON_FACES)
    declared_zero = {edge_id: 0.0 for edge_id in triangulation.Ori}

    result = check_arithmeticity(TETRAHEDRON_FACES, shears=declared_zero)

    assert result["is_arithmetic"]
    assert result["is_arithmetic_certified"]
    assert result["has_exact_zero_shears"]
    assert result["arithmeticity_status"] == (
        "certified_declared_exact_zero_shear"
    )


@pytest.mark.parametrize(
    "shears, match",
    [
        ({}, "missing edge ids"),
        ({**{edge_id: 0.0 for edge_id in range(6)}, 6: 0.0}, "unexpected"),
        ({**{edge_id: 0.0 for edge_id in range(6)}, 2: math.nan}, "finite"),
        ({**{edge_id: 0.0 for edge_id in range(6)}, 2: math.inf}, "finite"),
    ],
)
def test_shear_mapping_requires_exact_finite_edge_coverage(shears, match):
    with pytest.raises(ValueError, match=match):
        check_arithmeticity(TETRAHEDRON_FACES, shears=shears)


def test_near_degenerate_cross_ratio_is_not_clamped_to_zero():
    faces = [(0, 1, 2), (1, 0, 3)]
    coordinates = [0j, 1 + 0j, 1e-12 + 0j, 2 + 0j]

    shear = compute_shears_from_vertices(faces, coordinates)[0]
    expected = math.log(1e-12) - math.log(1 - 1e-12) - math.log(2)

    assert shear == pytest.approx(expected, abs=1e-12)
    assert shear < -28


def test_vertex_entry_point_closes_sphere_without_vacuous_generator_set():
    result = check_arithmeticity_from_vertices(
        [0j, 3 + 0j, 2 + 1j, 2j],
        tolerance=1e-12,
    )

    assert result["topology"] == {
        "vertex_labels": [0, 1, 2, 3],
        "n_vertices": 4,
        "n_edges": 6,
        "n_faces": 4,
        "euler_characteristic": 2,
    }
    assert result["n_generators"] == 3
    assert max(abs(value) for value in result["shears"].values()) > 1
    assert result["passes_generator_trace_diagnostic"]
    assert not result["is_arithmetic"]
    assert result["triangulation_source"] == (
        "mobius-pole-delaunay-with-infinity-fan"
    )


def test_exact_certificate_is_serializable_and_has_area_index():
    certificate = certify_exact_zero_shear_structure(TETRAHEDRON_FACES)
    payload = certificate.as_dict()

    assert certificate.psl2z_index == 12
    assert certificate.all_determinants_one
    assert certificate.all_traces_integral
    assert certificate.is_arithmetic
    assert certificate.scope == "abstract-zero-shear-structure"
    assert len(certificate.generators) == certificate.topology.n_vertices - 1
    assert json.loads(json.dumps(payload)) == payload


def test_exact_certificate_scales_to_exactly_300_vertices():
    certificate = certify_exact_zero_shear_structure(_bipyramid_faces(298))

    assert certificate.topology.n_vertices == 300
    assert certificate.topology.n_edges == 894
    assert certificate.topology.n_faces == 596
    assert certificate.psl2z_index == 1788
    assert len(certificate.generators) == 299
    assert certificate.is_arithmetic


def test_exact_certificate_refuses_open_input_and_invalid_root():
    with pytest.raises(ValueError, match="incidence 1"):
        certify_exact_zero_shear_structure([(0, 1, 2)])
    with pytest.raises(ValueError, match="root must be in"):
        certify_exact_zero_shear_structure(TETRAHEDRON_FACES, root=4)
    with pytest.raises(TypeError, match="root must be an integer"):
        certify_exact_zero_shear_structure(TETRAHEDRON_FACES, root=True)
