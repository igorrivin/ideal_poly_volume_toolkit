"""Higher-dimensional geometric-symmetry audit regressions."""

import json
from itertools import product

import numpy as np
import pytest

import ideal_poly_volume_toolkit.symmetry as symmetry_module
from ideal_poly_volume_toolkit.symmetry import (
    PYNAUTY_AVAILABLE,
    audit_hyperbolic_symmetry,
)


pytestmark = pytest.mark.skipif(
    not PYNAUTY_AVAILABLE,
    reason="pynauty is required for geometric symmetry certification",
)


def _ideal_h4_16_cell():
    return np.vstack((np.eye(4), -np.eye(4)))


def _lorentz_boost_boundary_points(points, rapidity=0.7):
    cosh = np.cosh(rapidity)
    sinh = np.sinh(rapidity)
    boost = np.eye(5)
    boost[0, 0] = cosh
    boost[0, 4] = sinh
    boost[4, 0] = sinh
    boost[4, 4] = cosh
    lifts = np.column_stack((points, np.ones(len(points))))
    transformed = lifts @ boost.T
    return transformed[:, :4] / transformed[:, 4, None]


def test_h4_16_cell_has_full_hyperoctahedral_group():
    audit = audit_hyperbolic_symmetry(_ideal_h4_16_cell())

    assert audit["valid_geometry"] is True
    assert audit["topology"]["n_facets"] == 16
    assert audit["combinatorial_group_order"] == 384
    assert audit["geometric_group_order"] == 384
    assert audit["orientation_preserving_count"] == 192
    assert audit["vertex_orbits"] == [list(range(8))]
    assert audit["full_group_certified"] is True
    assert audit["certified_nontrivial_symmetry"] is True
    assert audit["residuals"]["max_accepted_isometry_residual"] < 1e-12
    assert "group_name" not in audit
    json.dumps(audit, allow_nan=False)


def test_group_is_invariant_under_nontrivial_lorentz_boundary_map():
    transformed = _lorentz_boost_boundary_points(_ideal_h4_16_cell())
    assert np.max(np.abs(np.linalg.norm(transformed, axis=1) - 1.0)) < 1e-14

    audit = audit_hyperbolic_symmetry(transformed)

    assert audit["valid_geometry"] is True
    assert audit["combinatorial_group_order"] == 384
    assert audit["geometric_group_order"] == 384
    assert audit["orientation_preserving_count"] == 192
    assert audit["full_group_certified"] is True
    assert audit["residuals"]["max_accepted_isometry_residual"] < 1e-11


@pytest.mark.parametrize("rapidity", [7.0, 8.0, 10.0])
def test_group_certification_survives_strong_lorentz_boost(rapidity):
    transformed = _lorentz_boost_boundary_points(_ideal_h4_16_cell(), rapidity=rapidity)

    audit = audit_hyperbolic_symmetry(transformed)

    assert audit["valid_geometry"] is True
    assert audit["combinatorial_group_order"] == 384
    assert audit["geometric_group_order"] == 384
    assert audit["orientation_preserving_count"] == 192
    assert audit["indeterminate_permutations"] == 0
    assert audit["full_group_certified"] is True
    assert audit["residuals"]["max_accepted_isometry_residual"] < 1e-10


def test_huge_finite_coordinates_fail_with_strict_json_diagnostics():
    audit = audit_hyperbolic_symmetry(np.full((5, 4), 1e200))

    assert audit["valid_geometry"] is False
    assert "norm" in audit["failure_reason"]
    json.dumps(audit, allow_nan=False)


def test_generic_ideal_h4_simplex_filters_combinatorial_symmetries():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(5, 4))
    points /= np.linalg.norm(points, axis=1, keepdims=True)

    audit = audit_hyperbolic_symmetry(points, max_group_order=1_000)

    assert audit["valid_geometry"] is True
    assert audit["combinatorial_group_order"] == 120
    assert audit["geometric_group_order"] == 1
    assert audit["geometric_generators"] == []
    assert audit["vertex_orbits"] == [[0], [1], [2], [3], [4]]
    assert audit["orientation_preserving_count"] == 1
    assert audit["full_group_certified"] is True
    assert audit["certified_nontrivial_symmetry"] is False
    assert audit["residuals"]["min_rejected_isometry_residual"] > 1e-3


def test_almost_symmetric_input_is_not_fully_certified_without_rejection_gap():
    points = _ideal_h4_16_cell()
    points[0] += 2e-6 * np.array([0.0, 1.0, 0.37, -0.21])
    points[0] /= np.linalg.norm(points[0])

    audit = audit_hyperbolic_symmetry(
        points,
        isometry_tolerance=1e-7,
    )

    assert audit["valid_geometry"] is True
    assert audit["geometric_group_order"] == 1
    assert audit["geometric_candidate_count"] == 1
    assert audit["full_group_certified"] is False
    assert audit["certified_nontrivial_symmetry"] is False
    rejected = audit["residuals"]["min_rejected_isometry_residual"]
    assert 1e-7 < rejected <= audit["residuals"]["rejection_gap_threshold"]
    assert "numerically ambiguous" in audit["failure_reason"]


def test_indeterminate_fit_prevents_full_group_certification(monkeypatch):
    real_fit = symmetry_module._lorentz_fit_for_permutation
    identity = tuple(range(8))

    def identity_only_fit(vertices, permutation):
        if tuple(permutation) != identity:
            return None
        return real_fit(vertices, permutation)

    monkeypatch.setattr(
        symmetry_module,
        "_lorentz_fit_for_permutation",
        identity_only_fit,
    )
    audit = audit_hyperbolic_symmetry(_ideal_h4_16_cell())

    assert audit["geometric_group_order"] == 1
    assert audit["geometric_candidate_count"] == 1
    assert audit["indeterminate_permutations"] == 383
    assert audit["full_group_certified"] is False
    assert audit["certified_nontrivial_symmetry"] is False
    assert "indeterminate Lorentz fits" in audit["failure_reason"]


def test_invalid_nonunit_input_is_rejected_before_hull_work():
    points = _ideal_h4_16_cell()
    points[0] *= 0.9

    audit = audit_hyperbolic_symmetry(points)

    assert audit["valid_geometry"] is False
    assert audit["checks"]["unit"] is False
    assert "not unit points" in audit["failure_reason"]
    assert audit["combinatorial_group_order"] is None


def test_nonsimplicial_tesseract_is_not_certified():
    points = np.asarray(list(product((-0.5, 0.5), repeat=4)), dtype=float)

    audit = audit_hyperbolic_symmetry(points)

    assert audit["checks"]["all_vertices_extreme"] is True
    assert audit["checks"]["simplicial"] is False
    assert audit["valid_geometry"] is False
    assert audit["full_group_certified"] is False
    assert audit["topology"]["max_vertices_on_support"] == 8
    assert "nonsimplicial" in audit["failure_reason"]


def test_group_enumeration_cap_preserves_topology_and_order():
    audit = audit_hyperbolic_symmetry(
        _ideal_h4_16_cell(),
        max_group_order=100,
    )

    assert audit["valid_geometry"] is True
    assert audit["combinatorial_group_order"] == 384
    assert audit["geometric_group_order"] is None
    assert audit["candidate_permutations_tested"] == 0
    assert audit["full_group_certified"] is False
    assert "exceeds max_group_order 100" in audit["failure_reason"]


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("unit_tolerance", 0.0),
        ("distinct_tolerance", np.inf),
        ("support_tolerance", -1.0),
        ("isometry_tolerance", np.nan),
        ("max_group_order", 0),
    ],
)
def test_invalid_audit_controls_raise(keyword, value):
    with pytest.raises(ValueError):
        audit_hyperbolic_symmetry(
            _ideal_h4_16_cell(),
            **{keyword: value},
        )
