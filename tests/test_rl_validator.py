"""Focused protocol tests for the RL controller validation runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
from typing import Any

import numpy as np
import pytest

from ideal_poly_volume_toolkit.rl.generic import OptimizerRun, OptimizerRunError


_VALIDATOR = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "validate_rl_controller.py")
)
PairedState = _VALIDATOR["PairedState"]
_run_solo_from_state = _VALIDATOR["_run_solo_from_state"]
_run_round_robin_from_state = _VALIDATOR["_run_round_robin_from_state"]
aggregate = _VALIDATOR["aggregate"]
_atomic_write_json = _VALIDATOR["_atomic_write_json"]
_exclusive_create_json = _VALIDATOR["_exclusive_create_json"]
_json_sha256 = _VALIDATOR["_json_sha256"]
_model_provenance = _VALIDATOR["_model_provenance"]
_trusted_model_path = _VALIDATOR["_trusted_model_path"]
run_one_task = _VALIDATOR["run_one_task"]
validator_main = _VALIDATOR["main"]


class _FakeTask:
    def copy_solution(self, solution: np.ndarray) -> np.ndarray:
        return np.asarray(solution, dtype=float).copy()

    def evaluate(self, solution: np.ndarray) -> float:
        return float(np.asarray(solution, dtype=float)[0])

    def evaluate_surrogate(self, solution: np.ndarray) -> float:
        return self.evaluate(solution)

    def has_distinct_surrogate(self) -> bool:
        return False

    def to_vector(self, solution: np.ndarray) -> np.ndarray:
        return np.asarray(solution, dtype=float).copy()


def _paired_state() -> Any:
    return PairedState(
        env=None,
        observation=None,
        reset_info={"best_value": 1.0, "total_evals": 1},
        task=_FakeTask(),
        initial_solution=np.array([1.0]),
        initial_value=1.0,
        initial_solution_hash="paired",
        action_ordering=[],
        reset_elapsed_sec=0.25,
        reset_cpu_sec=0.125,
    )


class _ExactWorkOptimizer:
    def __init__(
        self,
        name: str,
        *,
        objective_evals: int | None = None,
        surrogate_evals: int = 0,
        gradient_evals: int = 0,
    ):
        self.name = name
        self.objective_evals = objective_evals
        self.surrogate_evals = surrogate_evals
        self.gradient_evals = gradient_evals
        self.calls: list[tuple[int, np.ndarray]] = []

    def supports(self, task: Any) -> bool:
        return True

    def run(self, solution: Any, task: Any, budget: int, rng: Any) -> OptimizerRun:
        self.calls.append((budget, np.asarray(solution, dtype=float).copy()))
        objective = (
            budget - self.surrogate_evals - self.gradient_evals
            if self.objective_evals is None
            else self.objective_evals
        )
        candidate = np.asarray(solution, dtype=float) + 1.0
        value = float(candidate[0])
        return OptimizerRun(
            solution=candidate,
            values=[value],
            solution_value=value,
            objective_evals=objective,
            surrogate_evals=self.surrogate_evals,
            gradient_evals=self.gradient_evals,
        )


class _StructuredFailureOptimizer:
    name = "structured_failure"

    def supports(self, task: Any) -> bool:
        return True

    def run(self, solution: Any, task: Any, budget: int, rng: Any) -> OptimizerRun:
        partial = OptimizerRun(
            solution=task.copy_solution(solution),
            objective_evals=1,
            surrogate_evals=1,
            gradient_evals=1,
        )
        raise OptimizerRunError("deliberate failure", partial)


def test_solo_passes_the_entire_remaining_cap_once_and_decomposes_work() -> None:
    optimizer = _ExactWorkOptimizer(
        "decomposed",
        objective_evals=2,
        surrogate_evals=3,
        gradient_evals=4,
    )

    row = _run_solo_from_state(
        optimizer,
        _paired_state(),
        seed=7,
        budget=10,
        budget_source="test",
    )

    assert [budget for budget, _ in optimizer.calls] == [9]
    assert np.array_equal(optimizer.calls[0][1], np.array([1.0]))
    assert row["requested_action_budgets"] == [9]
    assert row["objective_evals"] == 3  # reset + optimizer objective work
    assert row["surrogate_evals"] == 3
    assert row["gradient_evals"] == 4
    assert row["total_evals"] == row["budget_cap"] == 10
    assert row["action_counts"] == {"decomposed": 1}
    assert [point["cumulative_work"]["total_evals"] for point in row["anytime_trace"]] == [1, 10]


def test_round_robin_has_exact_stable_budget_sequence_and_total() -> None:
    optimizers = [
        _ExactWorkOptimizer("a"),
        _ExactWorkOptimizer("b"),
        _ExactWorkOptimizer("c"),
    ]

    row = _run_round_robin_from_state(
        optimizers,
        _paired_state(),
        seed=8,
        budget=11,
        budget_source="test",
    )

    assert [optimizer.calls[0][0] for optimizer in optimizers] == [4, 3, 3]
    assert row["requested_action_budgets"] == [4, 3, 3]
    assert row["total_evals"] == row["budget_cap"] == 11
    assert row["action_counts"] == {"a": 1, "b": 1, "c": 1}
    assert [point["cumulative_work"]["total_evals"] for point in row["anytime_trace"]] == [1, 5, 8, 11]


def test_structured_optimizer_failure_preserves_partial_work_row() -> None:
    row = _run_solo_from_state(
        _StructuredFailureOptimizer(),
        _paired_state(),
        seed=9,
        budget=6,
        budget_source="test",
    )

    assert row["status"] == "failed"
    assert row["accounting_complete"] is True
    assert row["error_type"] == "OptimizerRunError"
    assert row["objective_evals"] == 2  # reset + one partial call
    assert row["surrogate_evals"] == 1
    assert row["gradient_evals"] == 1
    assert row["total_evals"] == 4
    assert row["action_failure_count"] == 1


def _row(
    d: int,
    N: int,
    seed: int,
    method: str,
    value: float | None,
    status: str = "ok",
) -> dict[str, Any]:
    return {
        "d": d,
        "N": N,
        "seed": seed,
        "method": method,
        "best_value": value,
        "status": status,
    }


def test_aggregate_uses_task_cell_medians_iqr_and_applicable_denominators() -> None:
    rows = [
        _row(3, 5, 0, "rl", 10.0),
        _row(3, 5, 1, "rl", 14.0),
        _row(3, 5, 0, "solo_a", 9.0),
        _row(3, 5, 1, "solo_a", 11.0),
        _row(3, 5, 0, "solo_b", None, "failed"),
        _row(3, 5, 1, "solo_b", None, "failed"),
        _row(3, 5, 0, "round_robin", 11.0),
        _row(3, 5, 1, "round_robin", 11.0),
        _row(3, 7, 0, "rl", 5.0),
        _row(3, 7, 1, "rl", 7.0),
        _row(3, 7, 0, "solo_b", 8.0),
        _row(3, 7, 1, "solo_b", 10.0),
    ]

    result = aggregate(rows)
    cells = {
        (cell["d"], cell["N"], cell["method"]): cell
        for cell in result["cell_summaries"]
    }
    rl_first = cells[(3, 5, "rl")]
    assert rl_first["median_best_value"] == pytest.approx(12.0)
    assert rl_first["q1_best_value"] == pytest.approx(11.0)
    assert rl_first["q3_best_value"] == pytest.approx(13.0)
    assert rl_first["iqr_best_value"] == pytest.approx(2.0)
    assert cells[(3, 5, "solo_b")]["n_failed"] == 2
    assert cells[(3, 5, "solo_b")]["median_best_value"] is None

    rl_summary = result["methods"]["rl"]
    assert rl_summary["n_ratio_cells"] == 2
    assert rl_summary["geomean_ratio_vs_best_solo"] == pytest.approx(
        np.sqrt((12.0 / 10.0) * (6.0 / 9.0))
    )
    assert rl_summary["win_numerator"] == 1
    assert rl_summary["win_denominator"] == 2
    assert rl_summary["solo_win_numerator"] == 1
    assert rl_summary["solo_win_denominator"] == 2

    # solo_a has data in only one cell, and no other successful solo method
    # is applicable in that cell.  It must not inherit the global task count.
    solo_a = result["methods"]["solo_a"]
    assert solo_a["n_applicable_cells"] == 1
    assert solo_a["solo_win_denominator"] == 0
    assert solo_a["win_denominator"] == 1


def test_checkpoint_create_is_atomic_no_clobber_then_replaceable(tmp_path: Path) -> None:
    output = tmp_path / "checkpoint.json"
    _exclusive_create_json(output, {"generation": 1})

    with pytest.raises(FileExistsError):
        _exclusive_create_json(output, {"generation": 2})
    assert json.loads(output.read_text()) == {"generation": 1}

    _atomic_write_json(output, {"generation": 3})
    assert json.loads(output.read_text()) == {"generation": 3}


def _write_verified_model_pair(tmp_path: Path) -> tuple[str, Path, Path]:
    template = str(tmp_path / "controller_d{d}.zip")
    model_path = tmp_path / "controller_d3.zip"
    config_path = tmp_path / "controller_d3_config.json"
    model_path.write_bytes(b"verified-model")
    config = {
        "schema_version": "ideal-poly-rl-training/1",
        "controller_protocol_version": "ideal-poly-rl-controller/2",
        "status": "complete",
        "task": {"d_range": [3, 3]},
        "environment_config": {
            "action_ordering": ["cma_eucl", "lbfgs_hyp_d3", "sample"],
        },
        "training": {
            "requested_timesteps": 500_000,
            "actual_timesteps": 501_760,
        },
        "artifacts": {
            "model_sha256": hashlib.sha256(b"verified-model").hexdigest(),
        },
    }
    config["config_payload_sha256"] = _json_sha256(config)
    config_path.write_text(json.dumps(config))
    return template, model_path, config_path


def test_model_integrity_verifier_accepts_only_matching_schema_and_hashes(
    tmp_path: Path,
) -> None:
    template, model_path, _ = _write_verified_model_pair(tmp_path)

    metadata = _model_provenance(template, 3)
    trusted_path, error = _trusted_model_path(template, 3, metadata)

    assert metadata["integrity_verified"] is True
    assert metadata["verification_errors"] == []
    assert metadata["sidecar_actual_training_timesteps"] == 501_760
    assert metadata["expected_action_ordering"] == [
        "cma_eucl",
        "lbfgs_hyp_d3",
        "sample",
    ]
    assert trusted_path == model_path
    assert error is None


def test_model_integrity_verifier_rejects_tampered_model(
    tmp_path: Path,
) -> None:
    template, model_path, _ = _write_verified_model_pair(tmp_path)
    model_path.write_bytes(b"tampered-model")

    metadata = _model_provenance(template, 3)
    trusted_path, error = _trusted_model_path(template, 3, metadata)

    assert metadata["integrity_verified"] is False
    assert any("model checksum mismatch" in item for item in metadata["verification_errors"])
    assert trusted_path is None
    assert "model checksum mismatch" in error


def test_model_integrity_verifier_rejects_missing_sidecar(tmp_path: Path) -> None:
    template = str(tmp_path / "controller_d{d}.zip")
    (tmp_path / "controller_d3.zip").write_bytes(b"orphan-model")

    metadata = _model_provenance(template, 3)
    trusted_path, error = _trusted_model_path(template, 3, metadata)

    assert metadata["integrity_verified"] is False
    assert metadata["verification_errors"] == ["training sidecar not found"]
    assert trusted_path is None
    assert error == "training sidecar not found"


def test_model_integrity_verifier_rejects_tampered_sidecar(
    tmp_path: Path,
) -> None:
    template, _, config_path = _write_verified_model_pair(tmp_path)
    config = json.loads(config_path.read_text())
    config["training"]["requested_timesteps"] += 1
    config_path.write_text(json.dumps(config))

    metadata = _model_provenance(template, 3)
    trusted_path, error = _trusted_model_path(template, 3, metadata)

    assert metadata["integrity_verified"] is False
    assert any(
        "payload checksum mismatch" in item
        for item in metadata["verification_errors"]
    )
    assert trusted_path is None
    assert "payload checksum mismatch" in error


def test_invalid_model_integrity_becomes_failed_rl_row_without_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _paired_state()
    state.env = type("ClosableEnv", (), {"close": lambda self: None})()
    globals_dict = run_one_task.__globals__
    monkeypatch.setitem(
        globals_dict,
        "_prepare_paired_state",
        lambda d, N, seed, horizon: state,
    )
    monkeypatch.setitem(globals_dict, "_validation_optimizers", lambda d: [])
    load_called = False

    def forbidden_load(*args: Any, **kwargs: Any) -> Any:
        nonlocal load_called
        load_called = True
        raise AssertionError("invalid controller must not load")

    monkeypatch.setitem(globals_dict, "_run_rl_on_state", forbidden_load)

    rows = run_one_task(
        3,
        5,
        0,
        None,
        fallback_budget=8,
        model_error="model checksum mismatch",
    )

    rl_row = next(row for row in rows if row["method"] == "rl")
    assert load_called is False
    assert rl_row["status"] == "failed"
    assert rl_row["error_type"] == "ModelIntegrityError"
    assert "checksum mismatch" in rl_row["error_message"]
    assert rl_row["budget_source"] == "model_integrity_failed"


def test_main_routes_invalid_sidecar_around_model_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = str(tmp_path / "orphan_d{d}.zip")
    (tmp_path / "orphan_d3.zip").write_bytes(b"not-trusted")
    output = tmp_path / "validation.json"
    captured: dict[str, Any] = {}

    def fake_run_one_task(
        d: int,
        N: int,
        seed: int,
        model_path: Path | None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        captured["model_path"] = model_path
        captured.update(kwargs)
        return [
            {
                "d": d,
                "N": N,
                "seed": seed,
                "method": "rl",
                "status": "failed",
                "best_value": None,
                "actual_training_timesteps": None,
            }
        ]

    monkeypatch.setitem(
        validator_main.__globals__,
        "run_one_task",
        fake_run_one_task,
    )

    validator_main(
        [
            "--smoke",
            "--model-template",
            template,
            "--out",
            str(output),
        ]
    )

    assert captured["model_path"] is None
    assert "training sidecar not found" in captured["model_error"]
    assert output.is_file()
