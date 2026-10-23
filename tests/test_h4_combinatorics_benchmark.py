"""Regressions for coordinate-only ideal H4 combinatorics contracts."""

from __future__ import annotations

from itertools import product
import json

import numpy as np
import pytest

import ideal_poly_volume_toolkit.rl.polytope.h4_combinatorics_benchmark as benchmark


def _regular_ideal_4simplex() -> np.ndarray:
    gram = np.full((5, 5), -0.25)
    np.fill_diagonal(gram, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    positive = eigenvalues > 1.0e-12
    return eigenvectors[:, positive] @ np.diag(np.sqrt(eigenvalues[positive]))


def _sixteen_cell() -> np.ndarray:
    return np.vstack((np.eye(4), -np.eye(4)))


def _random_orthogonal(seed: int) -> np.ndarray:
    matrix = np.random.default_rng(seed).standard_normal((4, 4))
    orthogonal, triangular = np.linalg.qr(matrix)
    signs = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
    return orthogonal * signs[None, :]


def _facets(vertices: np.ndarray) -> list[list[int]]:
    result = benchmark.extract_strict_h4_facets(vertices)
    assert result["valid_geometry"] is True
    return result["facets"]


def test_exact_target_contract_accepts_and_is_strict_json_native() -> None:
    vertices = _regular_ideal_4simplex()
    target = _facets(vertices)

    result = benchmark.audit_h4_combinatorics_candidate(
        vertices,
        target,
        volume_floor=0.26,
    )

    assert result["classification"] == "accepted"
    assert result["accepted"] is True
    assert all(value is True for value in result["checks"].values())
    assert result["geometry"]["facets"] == target
    assert result["incidence"]["uses_full_vertex_facet_incidence"] is True
    assert (
        result["incidence"]["candidate_certificate"]
        == (result["incidence"]["target_certificate"])
    )
    assert result["pole_volume_audit"]["minimum_volume"] == pytest.approx(
        0.26889566016931, rel=2.0e-14
    )
    assert result["pole_facet_consistency"]["all_match"] is True
    assert all(
        entry["facets_match"] is True
        and entry["missing_facets"] == []
        and entry["extra_facets"] == []
        for entry in result["pole_facet_consistency"]["values"]
    )
    assert {
        entry["expected_simplex_count"]
        for entry in result["pole_facet_consistency"]["values"]
    } == {1}
    assert "reference" not in result
    json.dumps(result, allow_nan=False)


def test_contract_is_invariant_under_row_relabeling_and_o4_action() -> None:
    original = _sixteen_cell()
    target = _facets(original)
    permutation = np.array([5, 1, 7, 0, 4, 3, 6, 2])
    transformed = original[permutation] @ _random_orthogonal(20260818)

    result = benchmark.audit_h4_combinatorics_candidate(
        transformed,
        list(reversed([list(reversed(facet)) for facet in target])),
        volume_floor=1.7,
    )

    assert result["classification"] == "accepted"
    assert result["incidence"]["isomorphic"] is True
    assert result["pole_facet_consistency"]["all_match"] is True
    assert result["pole_volume_audit"]["spread"] < 1.0e-12


def test_incidence_certificate_ignores_facet_and_vertex_label_order() -> None:
    facets = _facets(_sixteen_cell())
    relabeling = {old: (3 * old + 1) % 8 for old in range(8)}
    relabeled = [
        [relabeling[vertex] for vertex in reversed(facet)] for facet in reversed(facets)
    ]

    comparison = benchmark.compare_facet_incidence(8, facets, relabeled)

    assert comparison["available"] is True
    assert comparison["backend"] == "pynauty"
    assert comparison["valid_input"] is True
    assert comparison["isomorphic"] is True
    assert comparison["candidate_certificate"] == comparison["target_certificate"]
    assert len(comparison["candidate_certificate_sha256"]) == 64


@pytest.mark.parametrize("change", ["missing", "wrong"])
def test_missing_or_wrong_target_facet_is_rejected(change: str) -> None:
    vertices = _sixteen_cell()
    target = _facets(vertices)
    if change == "missing":
        changed = target[:-1]
    else:
        changed = [facet.copy() for facet in target]
        changed[0] = [0, 1, 2, 4]
        assert sorted(changed[0]) not in target

    result = benchmark.audit_h4_combinatorics_candidate(
        vertices,
        changed,
        volume_floor=0.0,
    )

    assert result["classification"] == "rejected"
    assert result["accepted"] is False
    assert result["checks"]["geometry"] is True
    assert result["checks"]["combinatorics"] is False
    assert (
        "candidate facet incidence is not the target incidence"
        in result["failure_reasons"]
    )


def test_nonsimplicial_unit_hull_is_rejected() -> None:
    tesseract = np.asarray(list(product((-0.5, 0.5), repeat=4)), dtype=float)

    result = benchmark.extract_strict_h4_facets(tesseract)

    assert result["checks"]["unit"] is True
    assert result["checks"]["all_vertices_extreme"] is True
    assert result["checks"]["simplicial"] is False
    assert result["valid_geometry"] is False
    assert "nonsimplicial" in result["failure_reason"]


def test_nonunit_submission_is_rejected() -> None:
    vertices = _regular_ideal_4simplex()
    vertices[2] *= 0.99

    result = benchmark.extract_strict_h4_facets(vertices)

    assert result["checks"]["finite"] is True
    assert result["checks"]["unit"] is False
    assert result["valid_geometry"] is False
    assert "not unit points" in result["failure_reason"]


def test_interior_point_cannot_enter_through_a_relaxed_unit_gate() -> None:
    simplex = _regular_ideal_4simplex()
    vertices = np.vstack((simplex, 0.25 * simplex[0]))

    result = benchmark.extract_strict_h4_facets(vertices, unit_tolerance=1.0)

    # Approximate ideal representatives are normalized after the explicit
    # norm gate.  The interior radial copy then coincides with its boundary
    # representative and is rejected by the strict distinctness gate.
    assert result["checks"]["unit"] is True
    assert result["checks"]["distinct"] is False
    assert result["valid_geometry"] is False


def test_volume_floor_is_applied_to_minimum_all_pole_volume() -> None:
    vertices = _regular_ideal_4simplex()
    result = benchmark.audit_h4_combinatorics_candidate(
        vertices,
        _facets(vertices),
        volume_floor=0.27,
    )

    assert result["classification"] == "rejected"
    assert result["checks"]["all_pole_volume"] is True
    assert result["checks"]["volume_floor"] is False
    assert result["volume_floor_passed"] is False


def test_pole_facet_gate_checks_volume_triangulation_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vertices = _regular_ideal_4simplex()

    def inconsistent_volume_audit(
        points: np.ndarray,
        poles: list[int],
        *,
        all_poles_requested: bool,
        tolerance: float,
    ) -> dict[str, object]:
        return {
            "all_poles_requested": all_poles_requested,
            "requested_indices": poles,
            "requested_count": len(poles),
            "successful_count": len(poles),
            "failure_count": 0,
            "values": [
                {"pole_index": pole, "volume": 1.0, "simplex_count": 2}
                for pole in poles
            ],
            "failures": [],
            "minimum_volume": 1.0,
            "maximum_volume": 1.0,
            "spread": 0.0,
            "spread_tolerance": tolerance,
            "within_tolerance": True,
        }

    monkeypatch.setattr(benchmark, "_pole_volume_audit", inconsistent_volume_audit)
    result = benchmark.audit_h4_combinatorics_candidate(
        vertices,
        _facets(vertices),
        volume_floor=0.5,
    )

    assert result["checks"]["all_pole_volume"] is True
    assert result["checks"]["pole_facet_consistency"] is False
    assert result["classification"] == "rejected"
    assert all(
        entry["facets_match"] is True and entry["simplex_counts_match"] is False
        for entry in result["pole_facet_consistency"]["values"]
    )


def test_missing_pynauty_is_a_controlled_inconclusive_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vertices = _regular_ideal_4simplex()
    target = _facets(vertices)
    monkeypatch.setattr(benchmark, "PYNAUTY_AVAILABLE", False)

    certificate = benchmark.facet_incidence_certificate(5, target)
    result = benchmark.audit_h4_combinatorics_candidate(
        vertices,
        target,
        volume_floor=0.2,
    )

    assert certificate["available"] is False
    assert certificate["backend"] is None
    assert certificate["valid_input"] is True
    assert certificate["canonical_certificate"] is None
    assert "pynauty is required" in certificate["failure_reason"]
    assert result["classification"] == "inconclusive_incidence_backend"
    assert result["accepted"] is False
    assert result["checks"]["incidence_backend"] is False
    assert result["incidence"]["backend"] is None
    assert result["checks"]["combinatorics"] is None
    assert result["failure_reasons"] == []
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "target",
    [
        [[0, 1, 2]],
        [[0, 1, 2, 5]],
        [[0, 1, 2, 2]],
        [[0, 1, 2, 3], [3, 2, 1, 0]],
    ],
)
def test_malformed_target_incidence_is_rejected(target: list[list[int]]) -> None:
    vertices = _regular_ideal_4simplex()

    result = benchmark.audit_h4_combinatorics_candidate(
        vertices,
        target,
        volume_floor=0.0,
    )

    assert result["classification"] == "rejected"
    assert result["checks"]["target_incidence_input"] is False
    assert result["checks"]["combinatorics"] is None
