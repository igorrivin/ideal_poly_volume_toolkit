"""Core regression tests for robust zero-shear realization audits."""

from __future__ import annotations

import cmath
import json
import math

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from ideal_poly_volume_toolkit.rivin_zero_shear import (
    _ROOTS_OF_UNITY,
    _exact_generalized_circle_sign,
    _normalize_cp1_points,
    _support_predicate,
    _support_report,
    StandardUpperHalfPlaneZeroShearSystem,
    ZeroShearSystem,
    audit_zero_shear_realization,
    lognormal_edge_weights,
    normalize_cp1_points,
    solve_zero_shear_from_initial,
    solve_zero_shear_from_tutte,
    weighted_pole_link_tutte_embedding,
    weighted_tutte_embedding,
)


TETRAHEDRON_FACES = [
    (0, 1, 2),
    (0, 3, 1),
    (1, 3, 2),
    (2, 3, 0),
]
TETRAHEDRON_POINTS = [
    None,
    0.0 + 0.0j,
    1.0 + 0.0j,
    cmath.exp(1j * math.pi / 3.0),
]


def _bipyramid_faces(n: int) -> list[tuple[int, int, int]]:
    top = n + 1
    faces = []
    for index in range(n):
        current = 1 + index
        following = 1 + (index + 1) % n
        faces.append((0, current, following))
        faces.append((top, following, current))
    return faces


OCTAHEDRON_POINTS = [
    0.0 + 0.0j,
    1.0 + 0.0j,
    1.0j,
    -1.0 + 0.0j,
    -1.0j,
    None,
]


EIGHT_VERTEX_FACES = [
    (1, 3, 0),
    (2, 3, 0),
    (2, 7, 3),
    (5, 0, 6),
    (5, 1, 0),
    (5, 2, 7),
    (5, 4, 2),
    (5, 6, 4),
    (5, 7, 1),
    (6, 2, 0),
    (6, 2, 4),
    (7, 3, 1),
]
EIGHT_VERTEX_POINTS = [
    complex(-0.5, -0.8660254037844384),
    complex(1.0, 0.0),
    complex(-0.267949192431131, 1.15e-14),
    complex(-0.5, 0.8660254037844387),
    complex(-0.09807621135332856, -0.16987298107779922),
    complex(0.13397459621554747, -0.23205080756887528),
    complex(-0.18301270189223628, -0.31698729810777054),
    complex(0.13397459621555724, 0.23205080756888002),
]


def _mobius(point: complex | None) -> complex:
    """A nonsingular map which sends infinity to a finite point."""
    if point is None:
        return 2.0 / 0.3
    return (2.0 * point + 1.0) / (0.3 * point + 1.0)


def test_canonical_tetrahedron_has_every_authoritative_certificate():
    audit = audit_zero_shear_realization(
        TETRAHEDRON_FACES,
        TETRAHEDRON_POINTS,
        distinguished_face=(0, 1, 2),
        require_standard_normalization=True,
    )

    assert audit.strict_pass
    assert audit.classification == "target_strict_zero_shear"
    assert audit.normalization["matches"]
    assert audit.normalization["handedness_matches"]
    assert audit.chordal_distinctness["passes"]
    assert audit.topology["passes"]
    assert audit.support["strict_supporting"]
    assert audit.shear["max_abs"] < 1e-14
    assert audit.shear["stable_crosscheck_passes"]
    assert audit.bending["coherent_sign"]
    assert audit.spherical_embedding["embedded"]
    assert audit.qhull_agreement["agrees"]
    json.dumps(audit.as_dict(), allow_nan=False)


def test_octahedron_and_mobius_image_remain_strict_zero_shear():
    octahedron = audit_zero_shear_realization(
        _bipyramid_faces(4),
        OCTAHEDRON_POINTS,
        distinguished_face=(0, 1, 2),
        require_standard_normalization=False,
    )
    transformed = audit_zero_shear_realization(
        TETRAHEDRON_FACES,
        [_mobius(point) for point in TETRAHEDRON_POINTS],
        distinguished_face=(0, 1, 2),
        require_standard_normalization=False,
    )

    for audit in (octahedron, transformed):
        assert audit.strict_pass
        assert audit.classification == "target_strict_zero_shear"
        assert audit.shear["max_abs"] < 1e-13
        assert audit.support["passes"]
        assert audit.bending["passes"]
        assert audit.spherical_embedding["passes"]


def test_normalization_is_label_ordered_and_explicitly_optional():
    permuted_faces = [tuple(3 - label for label in face) for face in TETRAHEDRON_FACES]
    permuted_points = list(reversed(TETRAHEDRON_POINTS))
    audit = audit_zero_shear_realization(
        permuted_faces,
        permuted_points,
        distinguished_face=(3, 2, 1),
        require_standard_normalization=False,
    )
    assert audit.strict_pass
    assert audit.normalization["status"] == "not_applicable"
    assert audit.normalization["label_order_matches"] is False

    required = audit_zero_shear_realization(
        TETRAHEDRON_FACES,
        [0j, 1j, 1 + 0j, 0.5 + 0.5j],
        distinguished_face=(0, 1, 2),
        require_standard_normalization=True,
    )
    assert not required.strict_pass
    assert required.normalization["coordinate_anchors_match"] is False

    finite_mobius_image = [_mobius(point) for point in TETRAHEDRON_POINTS]
    safe_default = audit_zero_shear_realization(
        TETRAHEDRON_FACES,
        finite_mobius_image,
        distinguished_face=(0, 1, 2),
    )
    assert not safe_default.strict_pass
    assert safe_default.classification == "normalization_failure"
    assert safe_default.support["passes"]
    assert safe_default.bending["passes"]


def test_normalization_does_not_collapse_a_near_anchor_finite_point():
    near_anchor = _ROOTS_OF_UNITY[0] + 1e-15j
    normalized = _normalize_cp1_points(
        [*_ROOTS_OF_UNITY, near_anchor],
        (0, 1, 2),
    )

    assert normalized[:3] == (None, 0j, 1 + 0j)
    assert normalized[3] is not None
    assert math.isfinite(normalized[3].real)
    assert math.isfinite(normalized[3].imag)


def test_public_normalization_handles_arbitrary_labels_and_handedness():
    lower = [0j, 1 + 0j, None, 0.5 - math.sqrt(3.0) * 0.5j]
    normalized = normalize_cp1_points(
        lower,
        (2, 0, 1),
        positive_handedness=True,
    )

    assert normalized[:3] == (0j, 1 + 0j, None)
    assert normalized[3] == pytest.approx(0.5 + math.sqrt(3.0) * 0.5j)

    with pytest.raises(ValueError, match="strict half-plane"):
        normalize_cp1_points(
            [None, 0j, 1 + 0j, 0.25 + 0.5j, 0.75 - 0.5j],
            (0, 1, 2),
            positive_handedness=True,
        )


def test_duplicates_and_nonfinite_coordinates_return_finite_audits():
    duplicate = audit_zero_shear_realization(
        TETRAHEDRON_FACES,
        [None, 0j, 1 + 0j, 1 + 0j],
        distinguished_face=(0, 1, 2),
    )
    nonfinite = audit_zero_shear_realization(
        TETRAHEDRON_FACES,
        [None, 0j, 1 + 0j, complex(math.nan, 0.0)],
        distinguished_face=(0, 1, 2),
    )

    assert duplicate.classification == "degenerate_zero_shear"
    assert not duplicate.chordal_distinctness["passes"]
    assert nonfinite.classification == "error"
    assert "finite" in nonfinite.errors[0]
    json.dumps(duplicate.as_dict(), allow_nan=False)
    json.dumps(nonfinite.as_dict(), allow_nan=False)


def test_support_tie_uses_adaptive_exact_sign_and_is_not_strict():
    tied = audit_zero_shear_realization(
        TETRAHEDRON_FACES,
        [None, 0j, 1 + 0j, 0.5 + 0j],
        distinguished_face=(0, 1, 2),
        require_standard_normalization=False,
    )

    assert tied.support["status"] == "degenerate"
    assert tied.support["tie_count"] > 0
    assert tied.support["adaptive_fallback_count"] > 0
    assert not tied.support["passes"]


@pytest.mark.parametrize(
    "points",
    [
        [
            complex(5.1417926449169165e-30, 9.631855483389719e-23),
            complex(1.5168589700044864e35, 4.061792143806927e-21),
            complex(-1.7212942219459936e74, -4.5999088590562494e-61),
            complex(-2.9929099560600775e-33, -1.2567011375392277e47),
        ],
        [
            complex(-0.4570875082404038, 1.2155847937299218),
            complex(-5.353859784719479e-7, 1282446308.759278),
            complex(-530385158.1883303, 1.9222258772374407e-7),
            complex(-626734397.8071846, -159688672.65072924),
        ],
    ],
)
def test_support_filter_matches_exact_sign_across_extreme_scales(points):
    filtered, _, _, _ = _support_predicate(points, (0, 1, 2), 3)
    exact = _exact_generalized_circle_sign(points, (0, 1, 2, 3))

    assert filtered == exact == -1


def test_support_report_cannot_merge_opposite_exact_sides_at_extreme_scale():
    points = [
        complex(5.1417926449169165e-30, 9.631855483389719e-23),
        complex(1.5168589700044864e35, 4.061792143806927e-21),
        complex(-1.7212942219459936e74, -4.5999088590562494e-61),
        complex(-2.9929099560600775e-33, -1.2567011375392277e47),
        complex(1e50, 1e50),
    ]
    exact_signs = [
        _exact_generalized_circle_sign(points, (0, 1, 2, query)) for query in (3, 4)
    ]
    report = _support_report([(0, 1, 2)], points, 0.0)

    assert exact_signs == [-1, 1]
    assert report["status"] == "not_supporting"
    assert not report["passes"]


def test_explicit_embedded_zero_shear_wrong_hull_is_classified():
    audit = audit_zero_shear_realization(
        EIGHT_VERTEX_FACES,
        EIGHT_VERTEX_POINTS,
        distinguished_face=(1, 3, 0),
        shear_tolerance=1e-10,
        require_standard_normalization=False,
    )

    assert not audit.strict_pass
    assert audit.classification == "embedded_wrong_hull_zero_shear"
    assert audit.shear["max_abs"] < 3e-14
    assert audit.planar_embedding["embedded"]
    assert audit.spherical_embedding["embedded"]
    assert audit.support["status"] == "not_supporting"
    assert audit.bending["status"] == "incoherent"
    assert audit.qhull_agreement["agrees"] is False


def test_input_orientation_coherence_is_a_strict_gate():
    reversed_face = list(TETRAHEDRON_FACES)
    reversed_face[1] = (0, 1, 3)
    audit = audit_zero_shear_realization(
        reversed_face,
        TETRAHEDRON_POINTS,
        distinguished_face=(0, 1, 2),
    )

    assert audit.topology["combinatorially_valid"]
    assert not audit.topology["input_orientation_coherent"]
    assert not audit.topology["passes"]
    assert not audit.strict_pass


def test_zero_shear_sparse_jacobian_matches_centered_difference():
    system = ZeroShearSystem(TETRAHEDRON_FACES, (0, 1, 2))
    packed = np.array([0.13, -0.09])
    analytic = system.jacobian(packed)
    step = 1e-7
    numerical = np.column_stack(
        [
            (
                system.residual(packed + step * np.eye(2)[column])
                - system.residual(packed - step * np.eye(2)[column])
            )
            / (2.0 * step)
            for column in range(2)
        ]
    )

    assert isinstance(analytic, csr_matrix)
    assert analytic.shape == (6, 2)
    assert np.max(np.abs(analytic.toarray() - numerical)) < 2e-8
    assert system.fun(packed) == pytest.approx(system.residual(packed))


def test_log_height_sparse_jacobian_matches_centered_difference():
    system = StandardUpperHalfPlaneZeroShearSystem(TETRAHEDRON_FACES, (0, 1, 2))
    packed = np.array([0.37, math.log(0.72)])
    analytic = system.jacobian(packed).toarray()
    step = 1e-7
    numerical = np.column_stack(
        [
            (
                system.residual(packed + step * np.eye(2)[column])
                - system.residual(packed - step * np.eye(2)[column])
            )
            / (2.0 * step)
            for column in range(2)
        ]
    )

    unpacked = system.unpack(packed)
    assert unpacked[:3] == (None, 0j, 1 + 0j)
    assert unpacked[3].imag > 0.0
    assert np.max(np.abs(analytic - numerical)) < 2e-8


def test_weighted_tutte_start_is_planar_and_satisfies_barycenters():
    weights = lognormal_edge_weights(TETRAHEDRON_FACES, sigma=0.7, seed=123)
    embedding = weighted_tutte_embedding(
        TETRAHEDRON_FACES,
        (0, 1, 2),
        edge_weights=weights,
    )

    roots = np.array(
        [
            [1.0, 0.0],
            [-0.5, math.sqrt(3.0) / 2.0],
            [-0.5, -math.sqrt(3.0) / 2.0],
        ]
    )
    assert embedding[:3] == pytest.approx(roots)
    incident = [
        (neighbor, weights[tuple(sorted((3, neighbor)))]) for neighbor in (0, 1, 2)
    ]
    expected = sum(weight * embedding[neighbor] for neighbor, weight in incident) / sum(
        weight for _, weight in incident
    )
    assert embedding[3] == pytest.approx(expected)
    start_audit = audit_zero_shear_realization(
        TETRAHEDRON_FACES,
        [complex(x, y) for x, y in embedding],
        distinguished_face=(0, 1, 2),
        require_standard_normalization=False,
    )
    assert start_audit.planar_embedding["embedded"]


def test_pole_link_tutte_start_fixes_full_regular_boundary_and_disk_barycenter():
    faces = _bipyramid_faces(4)
    top_weights = {(5, label): float(label) for label in range(1, 5)}
    points = weighted_pole_link_tutte_embedding(
        faces,
        (0, 1, 2),
        edge_weights=top_weights,
    )

    assert points[0] is None
    assert points[1:3] == (0j, 1 + 0j)
    assert points[3] == pytest.approx(1 + 1j)
    assert points[4] == pytest.approx(1j)
    assert points[5] == pytest.approx(
        sum(label * complex(points[label]) for label in range(1, 5)) / sum(range(1, 5))
    )
    assert all(complex(points[label]).imag > 0.0 for label in range(3, 6))


def test_lognormal_weights_are_complete_positive_and_reproducible():
    first = lognormal_edge_weights(TETRAHEDRON_FACES, sigma=1.0, seed=9)
    second = lognormal_edge_weights(TETRAHEDRON_FACES, sigma=1.0, seed=9)
    third = lognormal_edge_weights(TETRAHEDRON_FACES, sigma=1.0, seed=10)

    assert first == second
    assert first != third
    assert len(first) == 6
    assert all(value > 0.0 and math.isfinite(value) for value in first.values())
    assert math.fsum(math.log(value) for value in first.values()) == pytest.approx(
        0.0, abs=1e-14
    )


def test_tutte_solver_keeps_disk_coordinates_and_normalizes_separately():
    result = solve_zero_shear_from_tutte(
        TETRAHEDRON_FACES,
        (0, 1, 2),
        max_nfev=100,
        record_trajectory=True,
    )

    assert result.success
    assert result.strict_pass
    assert result.classification == "target_strict_zero_shear"
    assert result.vertices_cp1[:3] == pytest.approx(
        (1 + 0j, -0.5 + math.sqrt(3) * 0.5j, -0.5 - math.sqrt(3) * 0.5j)
    )
    assert result.normalized_vertices_cp1[:3] == (None, 0j, 1 + 0j)
    assert result.audit.planar_embedding["embedded"]
    assert result.trajectory_kind == "accepted_iterates"
    assert result.trajectory[0]["kind"] == "accepted_iterate"
    assert result.trajectory[0]["in_target_convex_cell"] is True
    assert result.trajectory[0]["support_passes"] is True
    assert result.trajectory[0]["bending_coherent"] is True
    assert result.trajectory[0]["planar_embedded"] is True
    assert result.trajectory[0]["spherical_embedded"] is True
    json.dumps(result.as_dict(), allow_nan=False)


def test_generic_cube_solver_is_exactly_the_tutte_wrapper_on_same_start():
    embedding = weighted_tutte_embedding(TETRAHEDRON_FACES, (0, 1, 2))
    initial = tuple(complex(x, y) for x, y in embedding)
    generic = solve_zero_shear_from_initial(
        TETRAHEDRON_FACES,
        (0, 1, 2),
        initial,
        max_nfev=100,
    )
    wrapped = solve_zero_shear_from_tutte(
        TETRAHEDRON_FACES,
        (0, 1, 2),
        max_nfev=100,
    )

    assert generic.success and wrapped.success
    assert generic.vertices_cp1 == wrapped.vertices_cp1
    assert generic.normalized_vertices_cp1 == wrapped.normalized_vertices_cp1
    assert generic.nfev == wrapped.nfev
    assert generic.cost == wrapped.cost
    assert generic.max_abs_residual == wrapped.max_abs_residual
    assert generic.solver_metadata["gauge"] == "cube_roots"
    assert generic.solver_metadata["x_scale"] == "jac"


def test_generic_solver_normalizes_caller_start_and_preserves_positive_heights():
    transformed = [_mobius(point) for point in TETRAHEDRON_POINTS]
    standard = solve_zero_shear_from_initial(
        TETRAHEDRON_FACES,
        (0, 1, 2),
        transformed,
        gauge="standard_upper_half_plane",
        x_scale=1.0,
        lm_polish_steps=2,
        max_nfev=100,
    )
    cube = solve_zero_shear_from_initial(
        TETRAHEDRON_FACES,
        (0, 1, 2),
        transformed,
        gauge="cube_roots",
        max_nfev=100,
    )

    assert standard.success and standard.strict_pass
    assert standard.initial_vertices_cp1[:3] == (None, 0j, 1 + 0j)
    assert standard.vertices_cp1[:3] == (None, 0j, 1 + 0j)
    assert standard.vertices_cp1[3].imag > 0.0
    assert standard.normalized_vertices_cp1 == standard.vertices_cp1
    assert standard.initial_embedding is None
    assert standard.solver_metadata["x_scale"] == 1.0
    assert standard.solver_metadata["lm_polish"]["steps_completed"] == 2
    assert standard.nfev == (
        standard.solver_metadata["trf"]["nfev"]
        + sum(
            stage["nfev"] for stage in standard.solver_metadata["lm_polish"]["stages"]
        )
    )
    assert standard.timings["lm_polish_seconds"] >= 0.0

    assert cube.success and cube.strict_pass
    assert cube.initial_vertices_cp1[:3] == _ROOTS_OF_UNITY
    assert cube.initial_embedding.shape == (4, 2)

    arbitrary_face = (3, 1, 0)
    arbitrary = solve_zero_shear_from_initial(
        TETRAHEDRON_FACES,
        arbitrary_face,
        TETRAHEDRON_POINTS,
        gauge="standard_upper_half_plane",
        max_nfev=100,
    )
    assert arbitrary.success and arbitrary.strict_pass
    assert arbitrary.vertices_cp1[arbitrary_face[0]] is None
    assert arbitrary.vertices_cp1[arbitrary_face[1]] == 0j
    assert arbitrary.vertices_cp1[arbitrary_face[2]] == 1 + 0j
    assert arbitrary.vertices_cp1[2].imag > 0.0


@pytest.mark.parametrize(
    ("points", "gauge", "message"),
    [
        ([None, 0j, 1 + 0j], "cube_roots", "coordinate count"),
        ([None, 0j, 1 + 0j, 1 + 0j], "cube_roots", "pairwise distinct"),
        ([None, 0j, 1 + 0j, 0.5 + 0j], "standard_upper_half_plane", "half-plane"),
        (TETRAHEDRON_POINTS, "unknown", "gauge must be"),
    ],
)
def test_generic_solver_rejects_malformed_or_degenerate_initial_data(
    points, gauge, message
):
    result = solve_zero_shear_from_initial(
        TETRAHEDRON_FACES,
        (0, 1, 2),
        points,
        gauge=gauge,
    )

    assert not result.success
    assert result.classification == "error"
    assert message in result.message
    json.dumps(result.as_dict(), allow_nan=False)


def test_gapped_labels_and_bad_weights_are_rejected_cleanly():
    gapped = [
        (0 if label == 3 else label + 1 if label == 2 else label)
        for face in TETRAHEDRON_FACES
        for label in face
    ]
    gapped_faces = [
        tuple(gapped[index : index + 3]) for index in range(0, len(gapped), 3)
    ]
    audit = audit_zero_shear_realization(
        gapped_faces,
        TETRAHEDRON_POINTS,
        distinguished_face=gapped_faces[0],
    )
    assert audit.classification == "error"

    with pytest.raises(ValueError, match="positive"):
        weighted_tutte_embedding(
            TETRAHEDRON_FACES,
            (0, 1, 2),
            edge_weights={(0, 3): 0.0},
        )
