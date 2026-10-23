import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from jsonschema.exceptions import ValidationError
from jsonschema.validators import validator_for

import ideal_poly_volume_toolkit.optimization as optimization_module
from ideal_poly_volume_toolkit.optimization import (
    OBJECTIVE_PENALTY,
    analyze_ideal_polyhedron_structure,
    complete_boundary_triangulation,
    negative_volume_objective,
    optimize_ideal_polyhedron,
    optimization_result_to_dict,
    parameters_to_complex,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _validate_optimization_result_schema(payload):
    schema_path = REPO_ROOT / "docs/schemas/optimization-result-v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator_class = validator_for(schema)
    validator_class.check_schema(schema)
    validator_class(schema).validate(payload)


def test_regular_tetrahedron_objective_and_parameter_validation():
    params = np.array([math.pi / 2.0, math.pi / 3.0])
    vertices = parameters_to_complex(params, n_vertices=4)
    np.testing.assert_allclose(vertices, [0.0j, 1.0 + 0.0j, np.exp(1j * math.pi / 3)])

    regular_volume = 1.0149416064096536
    assert negative_volume_objective(params, 4) == pytest.approx(
        -regular_volume, abs=2e-13
    )

    with pytest.raises(ValueError, match="expected 2 parameters"):
        parameters_to_complex([], n_vertices=4)
    assert negative_volume_objective([math.nan, 0.0], 4) == OBJECTIVE_PENALTY


def test_infinity_fan_closes_planar_delaunay_disk():
    finite_faces = np.array([[0, 1, 2]], dtype=np.int64)
    boundary = complete_boundary_triangulation(finite_faces, n_finite_vertices=3)
    assert boundary.shape == (4, 3)
    assert {tuple(sorted(face)) for face in boundary} == {
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    }

    vertices = np.array([0.0j, 1.0 + 0.0j, np.exp(1j * math.pi / 3)])
    structure = analyze_ideal_polyhedron_structure(vertices, finite_faces)
    assert structure["n_vertices"] == 4
    assert structure["n_finite_vertices"] == 3
    assert structure["implicit_infinity_index"] == 3
    assert structure["n_edges"] == 6
    assert structure["n_faces"] == 4
    assert structure["n_infinity_faces"] == 3
    assert structure["euler_characteristic"] == 2
    assert structure["vertex_degrees"] == [3, 3, 3, 3]

    directed_edge_occurrences = {}
    for face in boundary:
        for u, v in zip(face, np.roll(face, -1)):
            edge = tuple(sorted((int(u), int(v))))
            direction = 1 if (int(u), int(v)) == edge else -1
            directed_edge_occurrences.setdefault(edge, []).append(direction)
    assert all(
        sorted(directions) == [-1, 1]
        for directions in directed_edge_occurrences.values()
    )

    with pytest.raises(ValueError, match="distinct"):
        complete_boundary_triangulation([[0, 0, 1]], 3)
    with pytest.raises(ValueError, match="one connected cycle"):
        complete_boundary_triangulation([[0, 1, 2], [3, 4, 5]], 6)


def test_infinity_fan_rejects_a_punctured_torus():
    # The two translation orbits below triangulate the seven-vertex Csaszar
    # torus.  Removing one face leaves one simple triangular boundary, but the
    # result is a punctured torus rather than a disk.
    torus_faces = []
    for vertex in range(7):
        torus_faces.extend(
            [
                [vertex, (vertex + 1) % 7, (vertex + 3) % 7],
                [vertex, (vertex + 2) % 7, (vertex + 3) % 7],
            ]
        )

    with pytest.raises(ValueError, match="topology of a disk"):
        complete_boundary_triangulation(torus_faces[1:], 7)


def test_infinity_fan_orients_mixed_winding_input_coherently():
    finite_faces = np.array([[0, 1, 2], [0, 3, 2]], dtype=np.int64)
    boundary = complete_boundary_triangulation(finite_faces, 4)

    directed_edge_occurrences = {}
    for face in boundary:
        for u, v in zip(face, np.roll(face, -1)):
            edge = tuple(sorted((int(u), int(v))))
            direction = 1 if (int(u), int(v)) == edge else -1
            directed_edge_occurrences.setdefault(edge, []).append(direction)
    assert all(
        sorted(directions) == [-1, 1]
        for directions in directed_edge_occurrences.values()
    )


def test_optimizer_service_returns_a_complete_positive_result():
    callbacks = []
    result = optimize_ideal_polyhedron(
        4,
        n_trials=1,
        maxiter=1,
        popsize=3,
        seed=0,
        workers=1,
        trial_callback=callbacks.append,
    )

    assert result.best_volume > 1.0
    assert result.n_vertices == 4
    assert result.finite_vertices.shape == (3,)
    assert result.finite_triangulation.shape == (1, 3)
    assert result.boundary_triangulation.shape == (4, 3)
    assert result.structure["euler_characteristic"] == 2
    assert len(result.trials) == 1
    assert callbacks == [result.trials[0]]
    assert result.trials[0].function_evals > 0

    payload = optimization_result_to_dict(
        result,
        source="test",
        timestamp="2026-08-15T00:00:00",
        settings={
            "n_trials": 1,
            "maxiter": 1,
            "popsize": 3,
            "workers": 1,
            "seed_base": 0,
        },
    )
    assert payload["schema_version"] == 1
    assert payload["source"] == "test"
    assert payload["metadata"]["settings"] == {
        "n_trials": 1,
        "maxiter": 1,
        "popsize": 3,
        "workers": 1,
        "seed_base": 0,
    }
    for legacy_key in ("n_trials", "maxiter", "popsize", "workers", "seed_base"):
        assert payload["metadata"][legacy_key] == payload["metadata"]["settings"][
            legacy_key
        ]
    assert (
        payload["best"]["triangulation"]
        == payload["best"]["finite_delaunay_triangulation"]
    )
    assert len(payload["best"]["boundary_triangulation"]) == 4
    assert len(payload["all_trials"]) == 1
    _validate_optimization_result_schema(payload)
    inconsistent_payload = json.loads(json.dumps(payload))
    inconsistent_payload["all_trials"][0]["valid"] = False
    with pytest.raises(ValidationError):
        _validate_optimization_result_schema(inconsistent_payload)
    json.dumps(payload)


def test_optimizer_excludes_penalty_trials_from_volume_statistics(monkeypatch):
    params = np.array([math.pi / 2.0, math.pi / 3.0])
    regular_volume = 1.0149416064096536
    results = iter(
        [
            SimpleNamespace(
                fun=OBJECTIVE_PENALTY,
                x=params,
                success=False,
                message="penalty",
                nit=0,
                nfev=1,
            ),
            SimpleNamespace(
                fun=-regular_volume,
                x=params,
                success=True,
                message="valid",
                nit=1,
                nfev=2,
            ),
        ]
    )
    monkeypatch.setattr(
        optimization_module,
        "differential_evolution",
        lambda *args, **kwargs: next(results),
    )

    result = optimize_ideal_polyhedron(4, n_trials=2, maxiter=1, popsize=1)
    assert result.trials[0].volume is None
    assert result.trials[1].volume == pytest.approx(regular_volume)

    payload = optimization_result_to_dict(result, source="test")
    assert payload["all_trials"][0]["valid"] is False
    assert payload["all_trials"][0]["volume"] is None
    assert payload["statistics"]["all_volumes"] == pytest.approx([regular_volume])
    assert payload["statistics"]["n_valid_trials"] == 1
    assert payload["statistics"]["n_invalid_trials"] == 1


def test_serial_optimizer_retains_scipy_builtin_polish(monkeypatch):
    params = np.array([math.pi / 2.0, math.pi / 3.0])
    differential_evolution_calls = []

    def fake_differential_evolution(*args, **kwargs):
        differential_evolution_calls.append((args, kwargs))
        return SimpleNamespace(
            fun=-1.0,
            x=params.copy(),
            success=True,
            message="serial DE",
            nit=2,
            nfev=11,
        )

    def unexpected_minimize(*args, **kwargs):
        raise AssertionError("serial DE should use SciPy's built-in polish")

    monkeypatch.setattr(
        optimization_module,
        "differential_evolution",
        fake_differential_evolution,
    )
    monkeypatch.setattr(optimization_module, "minimize", unexpected_minimize)

    result = optimize_ideal_polyhedron(
        4,
        n_trials=1,
        maxiter=1,
        popsize=1,
        workers=1,
    )

    assert len(differential_evolution_calls) == 1
    _, kwargs = differential_evolution_calls[0]
    assert kwargs["polish"] is True
    assert kwargs["workers"] == 1
    assert kwargs["updating"] == "immediate"
    assert result.trials[0].function_evals == 11


def test_parallel_optimizer_polishes_serially_and_accepts_improvement(monkeypatch):
    de_params = np.array([math.pi / 2.0, math.pi / 3.0])
    polished_params = np.array([math.pi / 2.0, math.pi / 2.0])
    de_result = SimpleNamespace(
        fun=-1.0,
        x=de_params.copy(),
        success=False,
        message="DE iteration limit",
        nit=3,
        nfev=11,
    )
    polished_jac = np.array([0.1, -0.2])
    calls = {}

    def fake_differential_evolution(*args, **kwargs):
        calls["differential_evolution"] = (args, kwargs)
        return de_result

    def fake_minimize(objective, x, **kwargs):
        calls["minimize"] = (objective, x, kwargs, x is de_result.x)
        return SimpleNamespace(
            fun=-1.1,
            x=polished_params.copy(),
            success=True,
            nfev=7,
            jac=polished_jac,
        )

    monkeypatch.setattr(
        optimization_module,
        "differential_evolution",
        fake_differential_evolution,
    )
    monkeypatch.setattr(optimization_module, "minimize", fake_minimize)

    result = optimize_ideal_polyhedron(
        4,
        n_trials=1,
        maxiter=1,
        popsize=1,
        workers=2,
    )

    _, de_kwargs = calls["differential_evolution"]
    assert de_kwargs["polish"] is False
    assert de_kwargs["workers"] == 2
    assert de_kwargs["updating"] == "deferred"

    objective, polish_start, polish_kwargs, start_was_aliased = calls["minimize"]
    assert objective.func is negative_volume_objective
    assert objective.keywords == {"n_vertices": 4}
    np.testing.assert_array_equal(polish_start, de_params)
    assert start_was_aliased is False
    assert polish_kwargs == {
        "method": "L-BFGS-B",
        "bounds": [(0.1, np.pi - 0.1), (0.0, 2.0 * np.pi)],
    }

    trial = result.trials[0]
    assert trial.volume == pytest.approx(1.1)
    np.testing.assert_array_equal(trial.params, polished_params)
    assert trial.function_evals == 18
    assert trial.iterations == 3
    assert trial.success is False
    assert trial.message == "DE iteration limit"
    np.testing.assert_array_equal(de_result.jac, polished_jac)


@pytest.mark.parametrize(
    ("polished_fun", "polished_success", "polished_params"),
    [
        (-0.9, True, np.array([math.pi / 2.0, math.pi / 2.0])),
        (-1.1, False, np.array([math.pi / 2.0, math.pi / 2.0])),
        (-1.1, True, np.array([0.0, math.pi / 2.0])),
    ],
    ids=["not-improved", "unsuccessful", "outside-bounds"],
)
def test_parallel_optimizer_rejects_invalid_polish_but_counts_its_work(
    monkeypatch,
    polished_fun,
    polished_success,
    polished_params,
):
    de_params = np.array([math.pi / 2.0, math.pi / 3.0])
    de_result = SimpleNamespace(
        fun=-1.0,
        x=de_params.copy(),
        success=False,
        message="DE iteration limit",
        nit=3,
        nfev=11,
    )
    monkeypatch.setattr(
        optimization_module,
        "differential_evolution",
        lambda *args, **kwargs: de_result,
    )
    monkeypatch.setattr(
        optimization_module,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(
            fun=polished_fun,
            x=polished_params,
            success=polished_success,
            nfev=7,
            jac=np.zeros(2),
        ),
    )

    result = optimize_ideal_polyhedron(
        4,
        n_trials=1,
        maxiter=1,
        popsize=1,
        workers=2,
    )

    trial = result.trials[0]
    assert trial.volume == pytest.approx(1.0)
    np.testing.assert_array_equal(trial.params, de_params)
    assert trial.function_evals == 18
    assert trial.iterations == 3
    assert trial.success is False
    assert trial.message == "DE iteration limit"
    assert not hasattr(de_result, "jac")
