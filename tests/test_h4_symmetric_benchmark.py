"""H^4 symmetric-candidate mining and diagnostic regressions."""

from __future__ import annotations

import numpy as np
import pytest

import ideal_poly_volume_toolkit.rl.polytope.h4_symmetric_benchmark as h4_benchmark
from ideal_poly_volume_toolkit.rl.polytope import (
    H4OptimizerRun,
    audit_h4_symmetric_candidate,
    optimize_h4_from_initial,
    optimize_h4_unconstrained,
)


def _sixteen_cell() -> np.ndarray:
    return np.vstack((np.eye(4), -np.eye(4)))


def _lorentz_boost_boundary_points(
    points: np.ndarray, *, rapidity: float
) -> np.ndarray:
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


def test_sixteen_cell_all_poles_agree_and_symmetry_is_certified() -> None:
    result = audit_h4_symmetric_candidate(_sixteen_cell())
    pole_audit = result["pole_volume_audit"]

    assert result["classification"] == "symmetric_candidate"
    assert result["independent_classification"] == result["classification"]
    assert result["symmetry"]["valid_geometry"] is True
    assert result["symmetry"]["full_group_certified"] is True
    assert result["symmetry"]["geometric_group_order"] == 384
    assert pole_audit["all_poles_requested"] is True
    assert pole_audit["requested_count"] == 8
    assert pole_audit["successful_count"] == 8
    assert pole_audit["failure_count"] == 0
    assert pole_audit["within_tolerance"] is True
    assert pole_audit["spread"] < 3.0e-14
    assert {entry["simplex_count"] for entry in pole_audit["values"]} == {8}


def test_sixteen_cell_local_maximum_quotients_full_po41_gauge() -> None:
    result = audit_h4_symmetric_candidate(
        _sixteen_cell(),
        local_maximum=True,
        fd_step=1.0e-4,
    )
    local = result["local_maximum_audit"]

    assert local["diagnostic_only"] is True
    assert local["performed"] is True
    assert local["status"] == "completed"
    assert local["chamber_stability"] is True
    assert local["tangent_dimension"] == 24
    assert local["gauge_dimension"] == 10
    assert local["gauge_rank"] == 10
    assert local["physical_dimension"] == 14
    assert local["computed_quotient_dimension"] == 14
    assert local["classification"] == "strict_local_maximum"
    assert local["projected_gradient_norm"] < 2.0e-7
    assert local["negative_eigenvalue_count"] == 14
    assert local["maximum_hessian_eigenvalue"] < -0.9
    assert result["classification"] == "symmetric_strict_local_maximum_diagnostic"


def test_requested_pole_subset_cannot_receive_publication_classification() -> None:
    result = audit_h4_symmetric_candidate(_sixteen_cell(), pole_indices=[0, 4])

    assert result["pole_volume_audit"]["within_tolerance"] is True
    assert result["classification"] == "partial_pole_audit"
    assert result["classification_basis"]["pole_volume_stable_on_requested"] is True
    assert result["classification_basis"]["all_poles_requested"] is False
    assert result["classification_basis"]["all_poles_verified"] is False


def test_explicit_complete_pole_list_counts_as_all_poles_requested() -> None:
    result = audit_h4_symmetric_candidate(
        _sixteen_cell(), pole_indices=list(reversed(range(8)))
    )

    assert result["pole_volume_audit"]["all_poles_requested"] is True
    assert result["classification_basis"]["all_poles_verified"] is True
    assert result["classification"] == "symmetric_candidate"


def test_relaxed_unit_gate_normalizes_before_volume_diagnostics() -> None:
    vertices = 1.000_001 * _sixteen_cell()
    result = audit_h4_symmetric_candidate(
        vertices,
        symmetry_kwargs={"unit_tolerance": 2.0e-6},
    )

    preparation = result["coordinate_preparation"]
    assert result["symmetry"]["checks"]["unit"] is True
    assert preparation["unit_gate_passed"] is True
    assert preparation["unit_normalized_for_diagnostics"] is True
    assert preparation["maximum_input_norm_error"] == pytest.approx(1.0e-6)
    assert preparation["maximum_post_normalization_norm_error"] < 3.0e-16
    assert result["pole_volume_audit"]["within_tolerance"] is True
    assert result["pole_volume_audit"]["minimum_volume"] == pytest.approx(
        1.77142839441536, rel=2.0e-14
    )


def test_local_maximum_is_invariant_under_strong_lorentz_boost() -> None:
    boosted = _lorentz_boost_boundary_points(_sixteen_cell(), rapidity=3.0)
    result = audit_h4_symmetric_candidate(
        boosted,
        pole_indices=[0],
        local_maximum=True,
        fd_step=1.0e-4,
    )
    local = result["local_maximum_audit"]
    balancing = local["conformal_balancing"]

    assert local["coordinate_chart"] == "equal_weight_conformal_barycenter"
    assert balancing["accepted"] is True
    assert balancing["ball_parameter_norm"] == pytest.approx(np.tanh(1.5), abs=2.0e-12)
    assert balancing["balanced_centroid_norm"] < 1.0e-12
    assert local["performed"] is True
    assert local["classification"] == "strict_local_maximum"
    assert local["projected_gradient_norm"] < 2.0e-7
    assert local["maximum_hessian_eigenvalue"] < -0.9


def test_local_maximum_refuses_a_changed_pole_delaunay_chamber(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def unstable_signature(points: np.ndarray, pole: int):
        nonlocal calls
        calls += 1
        return ((calls,),)

    monkeypatch.setattr(h4_benchmark, "_pole_delaunay_signature", unstable_signature)
    result = audit_h4_symmetric_candidate(
        _sixteen_cell(), pole_indices=[0], local_maximum=True
    )
    local = result["local_maximum_audit"]

    assert local["performed"] is False
    assert local["status"] == "skipped_non_strict_chamber"
    assert local["classification"] == "inconclusive_chamber_change"
    assert local["chamber_stability"] is False
    assert local["changed_probes"]


def test_unperformed_strict_label_cannot_certify_local_maximum() -> None:
    symmetry = {
        "valid_geometry": True,
        "full_group_certified": True,
        "certified_nontrivial_symmetry": True,
        "geometric_group_order": 4,
    }
    pole = {"within_tolerance": True, "all_poles_requested": True}
    local = {
        "requested": True,
        "performed": False,
        "classification": "strict_local_maximum",
    }

    classification, _ = h4_benchmark._combined_classification(symmetry, pole, local)

    assert classification == "symmetric_candidate_local_maximum_inconclusive"


def test_unconstrained_optimization_is_deterministic_and_work_counted() -> None:
    first = optimize_h4_unconstrained(5, seed=314159, budget=4)
    second = optimize_h4_unconstrained(5, seed=314159, budget=4)

    assert first["dimension"] == 4
    assert first["n_vertices"] == 5
    assert first["budget"] == 4
    assert first["work"] == {
        "objective_evals": 4,
        "surrogate_evals": 0,
        "gradient_evals": 0,
        "total_evals": 4,
    }
    assert first["optimizer"]["status"] == "budget_exhausted"
    assert first["optimizer"]["selector"] == "powell"
    assert first["optimizer"]["method"] == "scipy_powell"
    assert len(first["initial_coordinate_sha256"]) == 64
    assert first["scores"]["gain"] >= 0.0
    assert first["scores"] == pytest.approx(second["scores"], abs=0.0)
    np.testing.assert_allclose(first["vertices"], second["vertices"], atol=0.0)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(first["vertices"]), axis=1),
        np.ones(5),
        rtol=0.0,
        atol=2.0e-15,
    )
    assert first["timing"]["wall_seconds"] >= 0.0
    assert first["timing"]["process_cpu_seconds"] >= 0.0


def test_explicit_start_digest_matches_across_optimizer_selectors() -> None:
    rng = np.random.default_rng(20262000)
    initial = 7.0 * rng.standard_normal((5, 4))

    powell = optimize_h4_from_initial(
        initial,
        seed=11,
        budget=1,
        optimizer="powell",
    )
    fresh = optimize_h4_from_initial(
        initial,
        seed=11,
        budget=1,
        optimizer="fresh_topology_ad",
    )

    assert powell["initial_coordinate_sha256"] == fresh["initial_coordinate_sha256"]
    assert powell["scores"] == pytest.approx(fresh["scores"], rel=0.0, abs=0.0)
    assert powell["optimizer"]["selector"] == "powell"
    assert fresh["optimizer"]["selector"] == "fresh_topology_ad"
    assert fresh["optimizer"]["diagnostics"]["selector"] == "fresh_topology_ad"
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(fresh["vertices"]), axis=1),
        np.ones(5),
        rtol=0.0,
        atol=2.0e-15,
    )


def test_fresh_topology_explicit_start_is_deterministic_and_nonregressing() -> None:
    initial = np.random.default_rng(20262001).standard_normal((5, 4))
    options = {
        "pole_indices": [0, 2],
        "branch_modes": ["lbfgs", "adam_lbfgs"],
        "adam_steps": 2,
    }

    first = optimize_h4_from_initial(
        initial,
        seed=123,
        budget=20,
        optimizer="fresh_topology_ad",
        optimizer_kwargs=options,
    )
    second = optimize_h4_from_initial(
        initial,
        seed=123,
        budget=20,
        optimizer="fresh_topology_ad",
        optimizer_kwargs=options,
    )

    assert first["scores"]["final"] >= first["scores"]["initial"]
    assert first["scores"] == pytest.approx(second["scores"], rel=0.0, abs=0.0)
    assert first["work"] == second["work"]
    assert first["optimizer"]["diagnostics"] == second["optimizer"]["diagnostics"]
    np.testing.assert_allclose(first["vertices"], second["vertices"], atol=0.0)


def test_explicit_fresh_selector_rejects_an_all_failed_branch_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def all_failed_run(self, solution, task, budget, rng):
        del self, budget, rng
        score = float(task.evaluate(solution))
        return H4OptimizerRun(
            solution=solution,
            values=[score],
            solution_value=score,
            objective_evals=1,
            diagnostics={
                "selector": "fresh_topology_ad",
                "executed_branch_count": 2,
                "failed_branch_count": 2,
            },
        )

    monkeypatch.setattr(
        h4_benchmark.FreshTopologyADHyperbolicD4,
        "run",
        all_failed_run,
    )
    with pytest.raises(RuntimeError, match="failed in every executed"):
        optimize_h4_from_initial(
            np.random.default_rng(91).standard_normal((5, 4)),
            seed=91,
            budget=20,
            optimizer="fresh_topology_ad",
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"optimizer": "newton"}, ValueError, "powell"),
        ({"optimizer": 4}, TypeError, "powell"),
        ({"optimizer_kwargs": []}, TypeError, "mapping"),
        ({"vertices": np.zeros((5, 4))}, ValueError, "nonzero"),
        (
            {"vertices": np.full((5, 4), np.nan)},
            ValueError,
            "finite",
        ),
    ],
)
def test_explicit_start_optimization_rejects_invalid_controls(
    kwargs: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    arguments: dict[str, object] = {
        "vertices": _sixteen_cell(),
        "seed": 0,
        "budget": 1,
    }
    arguments.update(kwargs)
    with pytest.raises(error, match=match):
        optimize_h4_from_initial(**arguments)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"n_vertices": 4, "seed": 0, "budget": 1}, ValueError, "at least 5"),
        ({"n_vertices": True, "seed": 0, "budget": 1}, TypeError, "integer"),
        ({"n_vertices": 5, "seed": -1, "budget": 1}, ValueError, "at least 0"),
        ({"n_vertices": 5, "seed": True, "budget": 1}, TypeError, "integer"),
        ({"n_vertices": 5, "seed": 0, "budget": 0}, ValueError, "at least 1"),
    ],
)
def test_unconstrained_optimization_rejects_invalid_controls(
    kwargs: dict[str, object], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        optimize_h4_unconstrained(**kwargs)


@pytest.mark.parametrize(
    ("vertices", "kwargs", "error", "match"),
    [
        (np.ones((5, 3)), {}, ValueError, "shape"),
        (np.ones((4, 4)), {}, ValueError, "at least 5"),
        (_sixteen_cell(), {"pole_indices": []}, ValueError, "empty"),
        (_sixteen_cell(), {"pole_indices": [0, 0]}, ValueError, "duplicates"),
        (_sixteen_cell(), {"pole_indices": [8]}, ValueError, "outside"),
        (_sixteen_cell(), {"pole_indices": [0.5]}, TypeError, "integer"),
        (
            _sixteen_cell(),
            {"volume_spread_tolerance": -1.0},
            ValueError,
            "nonnegative",
        ),
        (_sixteen_cell(), {"fd_step": 0.0}, ValueError, "positive"),
        (_sixteen_cell(), {"local_maximum": 1}, TypeError, "bool"),
        (_sixteen_cell(), {"symmetry_kwargs": []}, TypeError, "mapping"),
    ],
)
def test_candidate_audit_rejects_invalid_inputs(
    vertices: np.ndarray,
    kwargs: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        audit_h4_symmetric_candidate(vertices, **kwargs)


def test_nonunit_geometry_is_classified_by_delegated_symmetry_audit() -> None:
    vertices = _sixteen_cell()
    vertices[0] *= 2.0

    result = audit_h4_symmetric_candidate(vertices, pole_indices=[0])

    assert result["classification"] == "invalid_geometry"
    assert result["symmetry"]["valid_geometry"] is False
    assert result["symmetry"]["checks"]["unit"] is False
    assert result["coordinate_preparation"]["unit_gate_passed"] is False
    assert result["pole_volume_audit"]["successful_count"] == 0
    assert result["pole_volume_audit"]["failure_count"] == 1
    assert result["local_maximum_audit"]["requested"] is False
