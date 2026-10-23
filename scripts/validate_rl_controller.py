#!/usr/bin/env python3
"""Validate RL allocation against paired, strict-work optimizer baselines.

The unit of comparison is one ``(dimension, vertex count, seed)`` rollout.
Every applicable method starts from the exact same initial solution and is
charged for the reset score.  Baseline caps are paired to the RL rollout's
actual counted work when a model is available, or to ``--fallback-budget``
otherwise.

The output is a checkpointed JSON document containing seed-level rows,
action-boundary anytime traces, task-cell summaries, and enough provenance to
identify the code, models, configuration, environment, and invocation.  An
existing output is never replaced unless ``--resume`` or ``--overwrite`` is
explicitly requested.

Examples
--------
Full planned grid::

    python scripts/validate_rl_controller.py \
        --model-template 'results/rl_models/ppo_d{d}_controller-v2-seed42.zip' \
        --out results/rl_validation/results.json \
        --seeds 10

Fast protocol smoke (one task, one seed, short horizon/cap)::

    python scripts/validate_rl_controller.py --smoke --out /tmp/rl-smoke.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


# Cap BLAS/OpenMP before importing NumPy/SciPy/Torch.  These effective values
# are also retained in the output provenance.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _thread_variable in THREAD_ENV_VARS:
    os.environ.setdefault(_thread_variable, "2")

import numpy as np  # noqa: E402


try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


REPO = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "ideal-poly-rl-validation/2"
PROTOCOL_VERSION = "2.0"
TRAINING_SCHEMA_VERSION = "ideal-poly-rl-training/1"
CONTROLLER_PROTOCOL_VERSION = "ideal-poly-rl-controller/2"
HORIZON = 40

DEFAULT_GRID: dict[int, list[int]] = {
    3: [5, 7, 10, 15, 20],
    4: [5, 7, 10, 16, 20],
    5: [6, 7, 10, 14, 20],
    6: [7, 10, 16, 20, 24],
}

DEPENDENCY_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "torch",
    "gymnasium",
    "stable-baselines3",
    "cma",
)


@dataclass
class WorkCounts:
    """Cumulative or per-action work decomposition."""

    objective_evals: int = 0
    surrogate_evals: int = 0
    gradient_evals: int = 0
    unclassified_evals: int = 0

    @property
    def total(self) -> int:
        return (
            self.objective_evals
            + self.surrogate_evals
            + self.gradient_evals
            + self.unclassified_evals
        )

    def add(self, other: "WorkCounts") -> None:
        self.objective_evals += int(other.objective_evals)
        self.surrogate_evals += int(other.surrogate_evals)
        self.gradient_evals += int(other.gradient_evals)
        self.unclassified_evals += int(other.unclassified_evals)

    def as_dict(self) -> dict[str, int]:
        return {
            "objective_evals": int(self.objective_evals),
            "surrogate_evals": int(self.surrogate_evals),
            "gradient_evals": int(self.gradient_evals),
            "unclassified_evals": int(self.unclassified_evals),
            "total_evals": int(self.total),
        }


@dataclass
class PairedState:
    """Reset state shared by all methods in one seeded task."""

    env: Any
    observation: Any
    reset_info: Mapping[str, Any]
    task: Any
    initial_solution: Any
    initial_value: float
    initial_solution_hash: str
    action_ordering: list[str]
    reset_elapsed_sec: float
    reset_cpu_sec: float


@dataclass
class OptimizerOutcome:
    """Normalized result of one direct baseline optimizer action."""

    ok: bool
    solution: Any
    value: Optional[float]
    work: WorkCounts
    elapsed_sec: float
    cpu_sec: float
    accounting_complete: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None


class _TaskWorkProbe:
    """Observe canonical/surrogate calls even when an optimizer raises.

    ``OptimizerRun`` remains authoritative for gradient-only work, which
    cannot be instrumented through the generic task interface.  On success we
    take the maximum of reported and observed canonical/surrogate counts.  An
    unstructured exception is marked as incomplete accounting rather than
    pretending that unobservable gradient work was zero with certainty.
    """

    def __init__(self, task: Any):
        self.task = task
        self.objective_evals = 0
        self.surrogate_evals = 0
        self._can_instrument = isinstance(getattr(task, "__dict__", None), dict)

    def __enter__(self) -> "_TaskWorkProbe":
        if not self._can_instrument:
            return self

        attributes = self.task.__dict__
        self._original_evaluate = self.task.evaluate
        self._original_surrogate = self.task.evaluate_surrogate
        self._had_evaluate = "evaluate" in attributes
        self._had_surrogate = "evaluate_surrogate" in attributes
        self._saved_evaluate = attributes.get("evaluate")
        self._saved_surrogate = attributes.get("evaluate_surrogate")
        self._distinct_surrogate = bool(self.task.has_distinct_surrogate())

        def counted_evaluate(solution: Any) -> Any:
            self.objective_evals += 1
            return self._original_evaluate(solution)

        self.task.evaluate = counted_evaluate

        if self._distinct_surrogate:

            def counted_surrogate(solution: Any) -> Any:
                self.surrogate_evals += 1
                return self._original_surrogate(solution)

            self.task.evaluate_surrogate = counted_surrogate
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if not self._can_instrument:
            return
        if self._had_evaluate:
            self.task.evaluate = self._saved_evaluate
        else:
            del self.task.evaluate
        if self._distinct_surrogate:
            if self._had_surrogate:
                self.task.evaluate_surrogate = self._saved_surrogate
            else:
                del self.task.evaluate_surrogate


def _validated_total_budget(budget: int) -> int:
    budget = int(budget)
    if budget < 1:
        raise ValueError("baseline budget must be at least 1")
    return budget


def _coerce_optimizer_run(raw_result: Any) -> Any:
    from ideal_poly_volume_toolkit.rl.generic import OptimizerRun

    if isinstance(raw_result, OptimizerRun):
        return raw_result
    solution, values = raw_result
    return OptimizerRun(
        solution=solution,
        values=list(values),
        objective_evals=len(values),
    )


def _invoke_optimizer(
    optimizer: Any,
    solution: Any,
    task: Any,
    budget_cap: int,
    rng: np.random.Generator,
) -> OptimizerOutcome:
    """Invoke one optimizer once and normalize work/failure accounting."""
    from ideal_poly_volume_toolkit.rl.generic import OptimizerRunError

    budget_cap = _validated_total_budget(budget_cap)
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    probe = _TaskWorkProbe(task)
    run: Any = None
    try:
        with probe:
            try:
                run = _coerce_optimizer_run(
                    optimizer.run(solution, task, budget_cap, rng)
                )
            except OptimizerRunError as exc:
                run = exc.run
                work = WorkCounts(
                    objective_evals=max(
                        int(run.objective_evals), probe.objective_evals
                    ),
                    surrogate_evals=max(
                        int(run.surrogate_evals), probe.surrogate_evals
                    ),
                    gradient_evals=int(run.gradient_evals),
                )
                message = str(exc)
                if work.total > budget_cap:
                    message = (
                        f"{message}; optimizer exceeded work cap "
                        f"{work.total} > {budget_cap}"
                    )
                return OptimizerOutcome(
                    ok=False,
                    solution=run.solution,
                    value=None,
                    work=work,
                    elapsed_sec=time.perf_counter() - wall_start,
                    cpu_sec=time.process_time() - cpu_start,
                    accounting_complete=True,
                    error_type=type(exc).__name__,
                    error_message=message,
                    error_traceback="".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                )

            reported_objective = int(run.objective_evals)
            objective_evals = max(reported_objective, probe.objective_evals)
            surrogate_evals = max(
                int(run.surrogate_evals), probe.surrogate_evals
            )
            gradient_evals = int(run.gradient_evals)
            reported_total = objective_evals + surrogate_evals + gradient_evals

            if reported_total > budget_cap:
                raise RuntimeError(
                    "optimizer exceeded its work budget: "
                    f"{reported_total} > {budget_cap}"
                )

            if run.solution_value is None:
                if reported_total >= budget_cap:
                    raise RuntimeError(
                        "optimizer exhausted its budget without a canonical "
                        "solution_value"
                    )
                canonical_value = float(task.evaluate(run.solution))
                objective_evals = max(
                    reported_objective + 1, probe.objective_evals
                )
            else:
                if objective_evals < 1:
                    raise RuntimeError(
                        "optimizer returned an uncounted canonical solution_value"
                    )
                canonical_value = float(run.solution_value)

            work = WorkCounts(
                objective_evals=objective_evals,
                surrogate_evals=surrogate_evals,
                gradient_evals=gradient_evals,
            )
            if not 1 <= work.total <= budget_cap:
                raise RuntimeError(
                    f"invalid optimizer work count {work.total} for cap {budget_cap}"
                )
    except Exception as exc:
        # Structured OptimizerRunError is handled above.  For other failures,
        # canonical/surrogate observations are exact but generic gradient work
        # is unknowable; expose that limitation in the row.
        work = WorkCounts(
            objective_evals=max(
                int(run.objective_evals) if run is not None else 0,
                probe.objective_evals,
            ),
            surrogate_evals=max(
                int(run.surrogate_evals) if run is not None else 0,
                probe.surrogate_evals,
            ),
            gradient_evals=(int(run.gradient_evals) if run is not None else 0),
        )
        return OptimizerOutcome(
            ok=False,
            solution=solution,
            value=None,
            work=work,
            elapsed_sec=time.perf_counter() - wall_start,
            cpu_sec=time.process_time() - cpu_start,
            accounting_complete=run is not None,
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_traceback="".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        )

    return OptimizerOutcome(
        ok=True,
        solution=run.solution,
        value=canonical_value,
        work=work,
        elapsed_sec=time.perf_counter() - wall_start,
        cpu_sec=time.process_time() - cpu_start,
        accounting_complete=True,
    )


def _consume_optimizer_run(
    raw_result: Any,
    task: Any,
    budget_cap: int,
) -> tuple[Any, float, int]:
    """Compatibility helper for callers that already hold a run result."""
    run = _coerce_optimizer_run(raw_result)
    actual_work = int(run.n_evals)
    if actual_work > budget_cap:
        raise RuntimeError(
            f"optimizer exceeded its work budget: {actual_work} > {budget_cap}"
        )
    if run.solution_value is None:
        if actual_work >= budget_cap:
            raise RuntimeError(
                "optimizer exhausted its budget without a canonical solution_value"
            )
        canonical_value = float(task.evaluate(run.solution))
        actual_work += 1
    else:
        if run.objective_evals < 1:
            raise RuntimeError(
                "optimizer returned an uncounted canonical solution_value"
            )
        canonical_value = float(run.solution_value)
    if not 1 <= actual_work <= budget_cap:
        raise RuntimeError(
            f"invalid optimizer work count {actual_work} for cap {budget_cap}"
        )
    return run.solution, canonical_value, actual_work


def _hash_solution(task: Any, solution: Any) -> str:
    digest = hashlib.sha256()
    try:
        array = np.ascontiguousarray(task.to_vector(solution))
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    except Exception:
        digest.update(repr(solution).encode("utf-8", errors="backslashreplace"))
    return digest.hexdigest()


def _prepare_paired_state(
    d: int,
    N: int,
    seed: int,
    *,
    horizon: int,
) -> PairedState:
    from ideal_poly_volume_toolkit.rl.polytope import make_polytope_env

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    env = make_polytope_env(
        d_range=(d, d),
        N_range=(N, N),
        horizon=horizon,
        seed=seed,
    )
    observation, info = env.reset(seed=seed)
    elapsed = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    task = env.task
    initial_solution = task.copy_solution(env.root.best_solution)
    return PairedState(
        env=env,
        observation=observation,
        reset_info=info,
        task=task,
        initial_solution=initial_solution,
        initial_value=float(env.root.best_value),
        initial_solution_hash=_hash_solution(task, initial_solution),
        action_ordering=list(env.op_names),
        reset_elapsed_sec=elapsed,
        reset_cpu_sec=cpu,
    )


def _trace_entry(
    *,
    action_index: int,
    action: str,
    action_budget: Optional[int],
    action_work: WorkCounts,
    cumulative_work: WorkCounts,
    best_value: float,
    elapsed_sec: float,
    cpu_sec: float,
    status: str,
    error: Optional[str] = None,
    requested_node_index: Optional[int] = None,
    effective_node_index: Optional[int] = None,
    node_index_remapped: Optional[bool] = None,
) -> dict[str, Any]:
    return {
        "action_index": int(action_index),
        "action": action,
        "action_budget": action_budget,
        "action_work": action_work.as_dict(),
        "cumulative_work": cumulative_work.as_dict(),
        "best_value": float(best_value),
        "elapsed_sec": float(elapsed_sec),
        "cpu_sec": float(cpu_sec),
        "status": status,
        "error": error,
        "requested_node_index": requested_node_index,
        "effective_node_index": effective_node_index,
        "node_index_remapped": node_index_remapped,
    }


def _base_row(
    method: str,
    state: PairedState,
    *,
    budget: int,
    budget_source: str,
) -> dict[str, Any]:
    initial_work = WorkCounts(objective_evals=1)
    return {
        "method": method,
        "status": "ok",
        "best_value": float(state.initial_value),
        "initial_value": float(state.initial_value),
        "initial_solution_sha256": state.initial_solution_hash,
        "budget_cap": int(budget),
        "budget_source": budget_source,
        **initial_work.as_dict(),
        "action_count": 0,
        "action_counts": {},
        "action_failure_count": 0,
        "action_failures": {},
        "accounting_complete": True,
        "elapsed_sec": float(state.reset_elapsed_sec),
        "cpu_sec": float(state.reset_cpu_sec),
        "anytime_trace": [
            _trace_entry(
                action_index=0,
                action="reset",
                action_budget=1,
                action_work=initial_work,
                cumulative_work=initial_work,
                best_value=state.initial_value,
                elapsed_sec=state.reset_elapsed_sec,
                cpu_sec=state.reset_cpu_sec,
                status="ok",
            )
        ],
    }


def _apply_outcome_to_row(
    row: dict[str, Any],
    outcome: OptimizerOutcome,
    *,
    action: str,
    action_budget: int,
    cumulative: WorkCounts,
    best_value: float,
) -> None:
    cumulative.add(outcome.work)
    row["action_count"] += 1
    row["action_counts"][action] = row["action_counts"].get(action, 0) + 1
    if not outcome.ok:
        row["action_failure_count"] += 1
        row["action_failures"][action] = row["action_failures"].get(action, 0) + 1
        row["status"] = "failed"
        row["error_type"] = outcome.error_type
        row["error_message"] = outcome.error_message
        row["error_traceback"] = outcome.error_traceback
    row["accounting_complete"] = bool(
        row["accounting_complete"] and outcome.accounting_complete
    )
    row["anytime_trace"].append(
        _trace_entry(
            action_index=row["action_count"],
            action=action,
            action_budget=int(action_budget),
            action_work=outcome.work,
            cumulative_work=cumulative,
            best_value=best_value,
            elapsed_sec=row["elapsed_sec"] + outcome.elapsed_sec,
            cpu_sec=row["cpu_sec"] + outcome.cpu_sec,
            status="ok" if outcome.ok else "failed",
            error=outcome.error_message,
        )
    )
    row.update(cumulative.as_dict())
    row["elapsed_sec"] += outcome.elapsed_sec
    row["cpu_sec"] += outcome.cpu_sec


def _validation_optimizers(d: int) -> list[Any]:
    """Return the registered baseline suite in stable protocol order."""
    from ideal_poly_volume_toolkit.rl.inner_optimizers import (
        CMAESOptimizer,
        RandomRestartOptimizer,
    )
    from ideal_poly_volume_toolkit.rl.polytope.optimizers import (
        LBFGSHyperbolicD3,
        LBFGSHyperbolicD4,
        LBFGSHyperbolicD5,
        LBFGSHyperbolicD6,
    )

    local_types = {
        3: LBFGSHyperbolicD3,
        4: LBFGSHyperbolicD4,
        5: LBFGSHyperbolicD5,
        6: LBFGSHyperbolicD6,
    }
    return [
        RandomRestartOptimizer(name="random_restart"),
        CMAESOptimizer(sigma=0.4, use_surrogate=True, name="cma_eucl"),
        CMAESOptimizer(sigma=0.4, use_surrogate=False, name="cma_true"),
        local_types[d](),
    ]


def _run_solo_from_state(
    optimizer: Any,
    state: PairedState,
    *,
    seed: int,
    budget: int,
    budget_source: str,
) -> dict[str, Any]:
    """Run one optimizer once with the entire post-reset cap."""
    budget = _validated_total_budget(budget)
    row = _base_row(
        f"solo_{optimizer.name}",
        state,
        budget=budget,
        budget_source=budget_source,
    )
    row["optimizer"] = optimizer.name
    row["requested_action_budgets"] = []
    cumulative = WorkCounts(objective_evals=1)
    if budget == 1:
        return row
    if not optimizer.supports(state.task):
        row["status"] = "not_applicable"
        row["reason"] = "optimizer does not support this task"
        return row

    remaining = budget - 1
    row["requested_action_budgets"].append(remaining)
    rng = np.random.default_rng(seed + 1_000_003)
    outcome = _invoke_optimizer(
        optimizer,
        state.task.copy_solution(state.initial_solution),
        state.task,
        remaining,
        rng,
    )
    best_value = state.initial_value
    if outcome.ok and outcome.value is not None and outcome.value > best_value:
        best_value = float(outcome.value)
    _apply_outcome_to_row(
        row,
        outcome,
        action=optimizer.name,
        action_budget=remaining,
        cumulative=cumulative,
        best_value=best_value,
    )
    row["best_value"] = float(best_value)
    if row["total_evals"] > budget:
        row["status"] = "failed"
        row["error_type"] = "BudgetExceeded"
        row["error_message"] = (
            f"total work {row['total_evals']} exceeded cap {budget}"
        )
    return row


def _round_robin_caps(total: int, n_optimizers: int) -> list[int]:
    """Split a cap as evenly as possible, with at most one call per method."""
    if total < 0:
        raise ValueError("round-robin residual budget cannot be negative")
    if n_optimizers < 1:
        raise ValueError("round-robin requires at least one optimizer")
    quotient, remainder = divmod(total, n_optimizers)
    return [
        quotient + (1 if index < remainder else 0)
        for index in range(n_optimizers)
    ]


def _run_round_robin_from_state(
    optimizers: Sequence[Any],
    state: PairedState,
    *,
    seed: int,
    budget: int,
    budget_source: str,
) -> dict[str, Any]:
    budget = _validated_total_budget(budget)
    applicable = [optimizer for optimizer in optimizers if optimizer.supports(state.task)]
    row = _base_row(
        "round_robin",
        state,
        budget=budget,
        budget_source=budget_source,
    )
    row["optimizer_ordering"] = [optimizer.name for optimizer in applicable]
    row["requested_action_budgets"] = []
    if not applicable:
        row["status"] = "not_applicable"
        row["reason"] = "no registered optimizer supports this task"
        return row

    cumulative = WorkCounts(objective_evals=1)
    best_solution = state.task.copy_solution(state.initial_solution)
    best_value = float(state.initial_value)
    rng = np.random.default_rng(seed + 1_000_007)
    caps = _round_robin_caps(budget - 1, len(applicable))
    scheduled = list(zip(applicable, caps))
    schedule_index = 0
    while scheduled:
        optimizer, action_budget = scheduled.pop(0)
        if action_budget == 0:
            continue
        row["requested_action_budgets"].append(action_budget)
        outcome = _invoke_optimizer(
            optimizer,
            state.task.copy_solution(best_solution),
            state.task,
            action_budget,
            rng,
        )
        if outcome.ok and outcome.value is not None and outcome.value > best_value:
            best_value = float(outcome.value)
            best_solution = state.task.copy_solution(outcome.solution)
        _apply_outcome_to_row(
            row,
            outcome,
            action=optimizer.name,
            action_budget=action_budget,
            cumulative=cumulative,
            best_value=best_value,
        )
        if not outcome.ok:
            break
        schedule_index += 1

    # Some methods (notably CMA-ES) may legitimately under-use a cap when the
    # residual cannot form a complete generation.  Redistribute only that
    # unused work in cyclic protocol order so a divisible method can still
    # meet the paired total cap.  The zero-progress guard prevents a broken
    # optimizer contract from creating an infinite loop.
    while row["status"] == "ok" and cumulative.total < budget:
        optimizer = applicable[schedule_index % len(applicable)]
        action_budget = budget - cumulative.total
        row["requested_action_budgets"].append(action_budget)
        outcome = _invoke_optimizer(
            optimizer,
            state.task.copy_solution(best_solution),
            state.task,
            action_budget,
            rng,
        )
        if outcome.ok and outcome.value is not None and outcome.value > best_value:
            best_value = float(outcome.value)
            best_solution = state.task.copy_solution(outcome.solution)
        _apply_outcome_to_row(
            row,
            outcome,
            action=optimizer.name,
            action_budget=action_budget,
            cumulative=cumulative,
            best_value=best_value,
        )
        schedule_index += 1
        if not outcome.ok:
            break
        if outcome.work.total == 0:
            row["status"] = "failed"
            row["error_type"] = "ZeroWorkOptimizer"
            row["error_message"] = (
                f"optimizer {optimizer.name!r} used no work from a positive cap"
            )
            row["accounting_complete"] = False
            break

    row["best_value"] = float(best_value)
    if row["total_evals"] > budget:
        row["status"] = "failed"
        row["error_type"] = "BudgetExceeded"
        row["error_message"] = (
            f"total work {row['total_evals']} exceeded cap {budget}"
        )
    return row


def _baseline_solo(
    opt_name: str,
    d: int,
    N: int,
    seed: int,
    budget: int,
) -> dict[str, Any]:
    """Compatibility entry used by focused budget-contract tests."""
    state = _prepare_paired_state(d, N, seed, horizon=HORIZON)
    optimizer = next(
        optimizer
        for optimizer in _validation_optimizers(d)
        if optimizer.name == opt_name
    )
    row = _run_solo_from_state(
        optimizer,
        state,
        seed=seed,
        budget=budget,
        budget_source="explicit",
    )
    state.env.close()
    return row


def _baseline_round_robin(
    d: int,
    N: int,
    seed: int,
    budget: int,
) -> dict[str, Any]:
    """Compatibility entry used by focused budget-contract tests."""
    state = _prepare_paired_state(d, N, seed, horizon=HORIZON)
    row = _run_round_robin_from_state(
        _validation_optimizers(d),
        state,
        seed=seed,
        budget=budget,
        budget_source="explicit",
    )
    state.env.close()
    return row


def _work_from_env_step(
    *,
    work_delta: int,
    optimizer_metrics: Optional[Mapping[str, Any]],
    failed: bool,
) -> WorkCounts:
    if optimizer_metrics is None:
        if failed:
            return WorkCounts(unclassified_evals=work_delta)
        return WorkCounts(objective_evals=work_delta)
    objective = int(optimizer_metrics.get("objective_evals", 0))
    surrogate = int(optimizer_metrics.get("surrogate_evals", 0))
    gradient = int(optimizer_metrics.get("gradient_evals", 0))
    classified = objective + surrogate + gradient
    return WorkCounts(
        objective_evals=objective,
        surrogate_evals=surrogate,
        gradient_evals=gradient,
        unclassified_evals=max(0, work_delta - classified),
    )


def _requested_timesteps(config: Any) -> Optional[int]:
    preferred_keys = (
        "requested_timesteps",
        "total_timesteps",
        "training_timesteps",
        "timesteps",
    )
    if isinstance(config, Mapping):
        lowered = {str(key).lower(): value for key, value in config.items()}
        for key in preferred_keys:
            value = lowered.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return int(value)
        for value in config.values():
            found = _requested_timesteps(value)
            if found is not None:
                return found
    elif isinstance(config, list):
        for value in config:
            found = _requested_timesteps(value)
            if found is not None:
                return found
    return None


def _run_rl_on_state(
    model_path: Path,
    state: PairedState,
    *,
    requested_timesteps: Optional[int],
    expected_actual_timesteps: Optional[int] = None,
    expected_action_ordering: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    row = _base_row(
        "rl",
        state,
        budget=1,
        budget_source="controller_actual_work",
    )
    row["action_ordering"] = list(state.action_ordering)
    row["model_path"] = _display_path(model_path)
    row["requested_training_timesteps"] = requested_timesteps
    cumulative = WorkCounts(objective_evals=1)
    observation = state.observation
    info = state.reset_info
    previous_total = int(info["total_evals"])
    action_elapsed = 0.0
    action_cpu = 0.0

    try:
        if (
            expected_action_ordering is not None
            and list(expected_action_ordering) != state.action_ordering
        ):
            raise RuntimeError(
                "controller action ordering does not match validation environment: "
                f"{list(expected_action_ordering)!r} != {state.action_ordering!r}"
            )
        from stable_baselines3 import PPO

        model = PPO.load(str(model_path), env=state.env)
        actual_timesteps = getattr(model, "num_timesteps", None)
        row["actual_training_timesteps"] = (
            int(actual_timesteps) if actual_timesteps is not None else None
        )
        if (
            expected_actual_timesteps is not None
            and row["actual_training_timesteps"] != expected_actual_timesteps
        ):
            raise RuntimeError(
                "serialized model timestep count does not match training sidecar: "
                f"{row['actual_training_timesteps']!r} != "
                f"{expected_actual_timesteps!r}"
            )
        done = False
        while not done:
            wall_start = time.perf_counter()
            cpu_start = time.process_time()
            action, _ = model.predict(observation, deterministic=True)
            action_array = np.asarray(action, dtype=int).reshape(-1)
            if action_array.size < 2:
                raise RuntimeError(f"controller returned malformed action {action!r}")
            requested_node = int(action_array[0])
            op_index = int(action_array[1])
            if not 0 <= op_index < len(state.action_ordering):
                raise RuntimeError(
                    f"controller operation index {op_index} is outside "
                    f"the action ordering of length {len(state.action_ordering)}"
                )
            action_name = state.action_ordering[op_index]
            observation, _, terminated, truncated, info = state.env.step(action)
            step_elapsed = time.perf_counter() - wall_start
            step_cpu = time.process_time() - cpu_start
            action_elapsed += step_elapsed
            action_cpu += step_cpu
            current_total = int(info["total_evals"])
            delta = current_total - previous_total
            if delta < 0:
                raise RuntimeError("environment total_evals moved backwards")
            action_work = _work_from_env_step(
                work_delta=delta,
                optimizer_metrics=info.get("last_optimizer_metrics"),
                failed=bool(info.get("last_error")),
            )
            cumulative.add(action_work)
            if cumulative.total != current_total:
                raise RuntimeError(
                    "environment work decomposition does not match total_evals: "
                    f"{cumulative.total} != {current_total}"
                )

            row["action_count"] += 1
            row["action_counts"][action_name] = (
                row["action_counts"].get(action_name, 0) + 1
            )
            action_error = info.get("last_error")
            if action_error:
                row["action_failure_count"] += 1
                row["action_failures"][action_name] = (
                    row["action_failures"].get(action_name, 0) + 1
                )
            row["anytime_trace"].append(
                _trace_entry(
                    action_index=row["action_count"],
                    action=action_name,
                    action_budget=None,
                    action_work=action_work,
                    cumulative_work=cumulative,
                    best_value=float(info["best_value"]),
                    elapsed_sec=state.reset_elapsed_sec + action_elapsed,
                    cpu_sec=state.reset_cpu_sec + action_cpu,
                    status="failed" if action_error else "ok",
                    error=action_error,
                    requested_node_index=info.get(
                        "requested_node_index", requested_node
                    ),
                    effective_node_index=info.get("effective_node_index"),
                    node_index_remapped=info.get("node_index_remapped"),
                )
            )
            previous_total = current_total
            done = bool(terminated or truncated)
    except Exception as exc:
        row["status"] = "failed"
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc)
        row["error_traceback"] = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        row["accounting_complete"] = False

    row["best_value"] = float(info.get("best_value", state.initial_value))
    row.update(cumulative.as_dict())
    row["budget_cap"] = cumulative.total
    row["elapsed_sec"] = state.reset_elapsed_sec + action_elapsed
    row["cpu_sec"] = state.reset_cpu_sec + action_cpu
    return row


def run_rl_episode(
    model_path: Path,
    d: int,
    N: int,
    seed: int,
    *,
    horizon: int = HORIZON,
) -> dict[str, Any]:
    """Compatibility entry for evaluating one controller episode."""
    state = _prepare_paired_state(d, N, seed, horizon=horizon)
    row = _run_rl_on_state(model_path, state, requested_timesteps=None)
    state.env.close()
    return row


def _model_path(template: str, d: int) -> Optional[Path]:
    path = Path(template.format(d=d))
    return path if path.is_file() else None


def _planned_method_names(d: int, include_rl: bool) -> list[str]:
    names = [
        "solo_random_restart",
        "solo_cma_eucl",
        "solo_cma_true",
        f"solo_lbfgs_hyp_d{d}",
        "round_robin",
    ]
    return (["rl"] + names) if include_rl else names


def _setup_failure_rows(
    *,
    d: int,
    N: int,
    seed: int,
    include_rl: bool,
    budget: int,
    exc: Exception,
) -> list[dict[str, Any]]:
    error_tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return [
        {
            "d": d,
            "N": N,
            "seed": seed,
            "method": method,
            "status": "failed",
            "best_value": None,
            "budget_cap": budget,
            "budget_source": "fallback_after_setup_failure",
            **WorkCounts().as_dict(),
            "action_count": 0,
            "action_counts": {},
            "action_failure_count": 0,
            "action_failures": {},
            "accounting_complete": False,
            "elapsed_sec": 0.0,
            "cpu_sec": 0.0,
            "anytime_trace": [],
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "error_traceback": error_tb,
        }
        for method in _planned_method_names(d, include_rl)
    ]


def run_one_task(
    d: int,
    N: int,
    seed: int,
    model_path: Optional[Path],
    *,
    horizon: int = HORIZON,
    fallback_budget: Optional[int] = None,
    requested_timesteps: Optional[int] = None,
    expected_actual_timesteps: Optional[int] = None,
    expected_action_ordering: Optional[Sequence[str]] = None,
    model_error: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Run all arms from one paired initial state, isolating run failures."""
    fallback_budget = _validated_total_budget(
        fallback_budget if fallback_budget is not None else horizon * 12
    )
    try:
        state = _prepare_paired_state(d, N, seed, horizon=horizon)
        optimizers = _validation_optimizers(d)
    except Exception as exc:
        return _setup_failure_rows(
            d=d,
            N=N,
            seed=seed,
            include_rl=model_path is not None or model_error is not None,
            budget=fallback_budget,
            exc=exc,
        )

    rows: list[dict[str, Any]] = []
    if model_error is not None:
        rows.append(
            {
                **_base_row(
                    "rl",
                    state,
                    budget=fallback_budget,
                    budget_source="model_integrity_failed",
                ),
                "status": "failed",
                "error_type": "ModelIntegrityError",
                "error_message": model_error,
                "reason": "controller artifact failed integrity verification",
            }
        )
        budget = fallback_budget
        budget_source = "configured_fallback_model_integrity_failed"
    elif model_path is None:
        rows.append(
            {
                **_base_row(
                    "rl",
                    state,
                    budget=fallback_budget,
                    budget_source="model_missing",
                ),
                "status": "not_applicable",
                "reason": "model file not found",
            }
        )
        budget = fallback_budget
        budget_source = "configured_fallback_model_missing"
    else:
        rl_row = _run_rl_on_state(
            model_path,
            state,
            requested_timesteps=requested_timesteps,
            expected_actual_timesteps=expected_actual_timesteps,
            expected_action_ordering=expected_action_ordering,
        )
        rows.append(rl_row)
        if rl_row["status"] == "ok":
            budget = int(rl_row["total_evals"])
            budget_source = "paired_rl_actual_work"
        else:
            budget = fallback_budget
            budget_source = "configured_fallback_rl_failed"

    for optimizer in optimizers:
        try:
            row = _run_solo_from_state(
                optimizer,
                state,
                seed=seed,
                budget=budget,
                budget_source=budget_source,
            )
        except Exception as exc:
            row = _setup_failure_rows(
                d=d,
                N=N,
                seed=seed,
                include_rl=False,
                budget=budget,
                exc=exc,
            )[0]
            row["method"] = f"solo_{optimizer.name}"
        rows.append(row)

    try:
        rows.append(
            _run_round_robin_from_state(
                optimizers,
                state,
                seed=seed,
                budget=budget,
                budget_source=budget_source,
            )
        )
    except Exception as exc:
        row = _setup_failure_rows(
            d=d,
            N=N,
            seed=seed,
            include_rl=False,
            budget=budget,
            exc=exc,
        )[0]
        row["method"] = "round_robin"
        rows.append(row)

    common = {"d": int(d), "N": int(N), "seed": int(seed)}
    for row in rows:
        row.update(common)
        row["paired_initial_solution_sha256"] = state.initial_solution_hash
    state.env.close()
    return rows


def _percentiles(values: Sequence[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    q1, median, q3 = np.percentile(array, [25, 50, 75])
    return float(q1), float(median), float(q3), float(q3 - q1)


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate seeds inside ``(d, N, method)`` cells before comparisons.

    Failed and non-applicable runs remain visible in cell counts but never
    enter medians.  Each win-rate denominator contains only cells where the
    method and at least one relevant comparator both have a successful median.
    """
    grouped: dict[tuple[int, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        try:
            key = (int(row["d"]), int(row["N"]), str(row["method"]))
        except (KeyError, TypeError, ValueError):
            continue
        grouped.setdefault(key, []).append(row)

    cell_summaries: list[dict[str, Any]] = []
    for (d, N, method), cell_rows in sorted(grouped.items()):
        successful_values = [
            float(row["best_value"])
            for row in cell_rows
            if row.get("status", "ok") == "ok"
            and row.get("best_value") is not None
            and math.isfinite(float(row["best_value"]))
        ]
        summary: dict[str, Any] = {
            "d": d,
            "N": N,
            "method": method,
            "n_runs": len(cell_rows),
            "n_success": len(successful_values),
            "n_failed": sum(row.get("status") == "failed" for row in cell_rows),
            "n_not_applicable": sum(
                row.get("status") == "not_applicable" for row in cell_rows
            ),
            "q1_best_value": None,
            "median_best_value": None,
            "q3_best_value": None,
            "iqr_best_value": None,
        }
        if successful_values:
            q1, median, q3, iqr = _percentiles(successful_values)
            summary.update(
                {
                    "q1_best_value": q1,
                    "median_best_value": median,
                    "q3_best_value": q3,
                    "iqr_best_value": iqr,
                }
            )
        cell_summaries.append(summary)

    by_task: dict[tuple[int, int], dict[str, Mapping[str, Any]]] = {}
    for cell in cell_summaries:
        if cell["median_best_value"] is not None:
            by_task.setdefault((cell["d"], cell["N"]), {})[cell["method"]] = cell

    method_names = sorted({cell["method"] for cell in cell_summaries})
    method_summary: dict[str, dict[str, Any]] = {}
    for method in method_names:
        ratios: list[float] = []
        wins_all = 0
        denominator_all = 0
        wins_solo = 0
        denominator_solo = 0
        applicable_cells = 0
        for task_methods in by_task.values():
            current = task_methods.get(method)
            if current is None:
                continue
            applicable_cells += 1
            value = float(current["median_best_value"])
            other_values = [
                float(cell["median_best_value"])
                for other_method, cell in task_methods.items()
                if other_method != method
            ]
            if other_values:
                denominator_all += 1
                if value > max(other_values):
                    wins_all += 1

            solo_values = [
                float(cell["median_best_value"])
                for other_method, cell in task_methods.items()
                if other_method.startswith("solo_")
                and (not method.startswith("solo_") or other_method != method)
            ]
            if solo_values:
                best_solo = max(solo_values)
                denominator_solo += 1
                if value > best_solo:
                    wins_solo += 1

            all_solo_values = [
                float(cell["median_best_value"])
                for other_method, cell in task_methods.items()
                if other_method.startswith("solo_")
            ]
            if all_solo_values:
                reference = max(all_solo_values)
                if value > 0.0 and reference > 0.0:
                    ratios.append(value / reference)

        if ratios:
            geometric_mean = math.exp(
                sum(math.log(ratio) for ratio in ratios) / len(ratios)
            )
            arithmetic_mean = float(np.mean(ratios))
        else:
            geometric_mean = None
            arithmetic_mean = None
        method_summary[method] = {
            "geomean_ratio_vs_best_solo": geometric_mean,
            "mean_ratio_vs_best_solo": arithmetic_mean,
            "n_ratio_cells": len(ratios),
            "win_rate": (
                wins_all / denominator_all if denominator_all else None
            ),
            "win_numerator": wins_all,
            "win_denominator": denominator_all,
            "win_rate_vs_best_solo": (
                wins_solo / denominator_solo if denominator_solo else None
            ),
            "solo_win_numerator": wins_solo,
            "solo_win_denominator": denominator_solo,
            "n_applicable_cells": applicable_cells,
        }

    return {
        "aggregation_unit": "(d, N) task-cell medians across successful seeds",
        "quartile_method": "numpy.percentile linear interpolation",
        "cell_summaries": cell_summaries,
        "methods": method_summary,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path.resolve())


def _git(*args: str) -> tuple[Optional[bytes], Optional[str]]:
    command = [
        "git",
        "-c",
        "filter.lfs.process=",
        "-c",
        "filter.lfs.required=false",
        "-c",
        "filter.lfs.clean=cat",
        "-c",
        "filter.lfs.smudge=cat",
        *args,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        return None, str(exc)
    if completed.returncode != 0:
        return None, completed.stderr.decode("utf-8", errors="replace").strip()
    return completed.stdout, None


def _repository_provenance() -> dict[str, Any]:
    commit_bytes, commit_error = _git("rev-parse", "HEAD")
    status_bytes, status_error = _git(
        "status", "--porcelain=v1", "-z", "--untracked-files=all"
    )
    diff_bytes, diff_error = _git("diff", "--binary", "HEAD", "--")
    staged_bytes, staged_error = _git("diff", "--binary", "--cached", "--")
    untracked_bytes, untracked_error = _git(
        "ls-files", "--others", "--exclude-standard", "-z"
    )

    errors = [
        error
        for error in (
            commit_error,
            status_error,
            diff_error,
            staged_error,
            untracked_error,
        )
        if error
    ]
    diff_digest = hashlib.sha256()
    for label, content in (
        (b"status\0", status_bytes),
        (b"unstaged\0", diff_bytes),
        (b"staged\0", staged_bytes),
    ):
        diff_digest.update(label)
        diff_digest.update(content or b"")
    if untracked_bytes is not None:
        for raw_path in sorted(filter(None, untracked_bytes.split(b"\0"))):
            diff_digest.update(b"untracked\0" + raw_path + b"\0")
            path = REPO / os.fsdecode(raw_path)
            if path.is_file():
                try:
                    diff_digest.update(path.read_bytes())
                except OSError as exc:
                    errors.append(f"could not hash {path}: {exc}")

    source_files = [Path(__file__).resolve(), REPO / "pyproject.toml"]
    source_files.extend(sorted((REPO / "ideal_poly_volume_toolkit" / "rl").rglob("*.py")))
    source_digest = hashlib.sha256()
    hashed_sources: list[str] = []
    for path in source_files:
        if not path.is_file():
            continue
        relative = _display_path(path)
        source_digest.update(relative.encode() + b"\0")
        source_digest.update(path.read_bytes())
        hashed_sources.append(relative)

    return {
        "commit": (
            commit_bytes.decode().strip() if commit_bytes is not None else None
        ),
        "dirty": bool(status_bytes) if status_bytes is not None else None,
        "diff_sha256": diff_digest.hexdigest(),
        "source_sha256": source_digest.hexdigest(),
        "validator_sha256": _sha256_file(Path(__file__).resolve()),
        "source_files": hashed_sources,
        "git_errors": errors,
    }


def _find_config_path(model_path: Path) -> Optional[Path]:
    candidates = [
        model_path.with_suffix(".json"),
        model_path.with_name(f"{model_path.stem}_config.json"),
        model_path.with_name(f"{model_path.stem.replace('_final', '')}_config.json"),
        model_path.with_name(f"{model_path.stem.replace('_final', '_config')}.json"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _model_provenance(template: str, d: int) -> dict[str, Any]:
    formatted = Path(template.format(d=d))
    model_path = formatted if formatted.is_file() else None
    config_path = _find_config_path(formatted) if model_path is not None else None
    config: Any = None
    config_error: Optional[str] = None
    verification_errors: list[str] = []
    model_sha256: Optional[str] = None
    config_sha256: Optional[str] = None
    if model_path is not None:
        try:
            model_sha256 = _sha256_file(model_path)
        except OSError as exc:
            verification_errors.append(
                f"cannot hash model: {type(exc).__name__}: {exc}"
            )
    if model_path is not None and config_path is None:
        verification_errors.append("training sidecar not found")
    if config_path is not None:
        try:
            config_bytes = config_path.read_bytes()
            config_sha256 = hashlib.sha256(config_bytes).hexdigest()
            config = json.loads(config_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            config_error = f"{type(exc).__name__}: {exc}"
            verification_errors.append(f"cannot read training sidecar: {config_error}")

    sidecar_actual_timesteps: Optional[int] = None
    expected_action_ordering: Optional[list[str]] = None
    if isinstance(config, Mapping):
        if config.get("schema_version") != TRAINING_SCHEMA_VERSION:
            verification_errors.append(
                "training schema mismatch: "
                f"{config.get('schema_version')!r} != {TRAINING_SCHEMA_VERSION!r}"
            )
        if config.get("controller_protocol_version") != CONTROLLER_PROTOCOL_VERSION:
            verification_errors.append(
                "controller protocol mismatch: "
                f"{config.get('controller_protocol_version')!r} != "
                f"{CONTROLLER_PROTOCOL_VERSION!r}"
            )
        if config.get("status") != "complete":
            verification_errors.append(
                f"training sidecar status is {config.get('status')!r}, not 'complete'"
            )

        claimed_payload_hash = config.get("config_payload_sha256")
        payload = dict(config)
        payload.pop("config_payload_sha256", None)
        actual_payload_hash = _json_sha256(payload)
        if not isinstance(claimed_payload_hash, str):
            verification_errors.append("training sidecar payload checksum is missing")
        elif claimed_payload_hash != actual_payload_hash:
            verification_errors.append(
                "training sidecar payload checksum mismatch: "
                f"{claimed_payload_hash} != {actual_payload_hash}"
            )

        artifacts = config.get("artifacts")
        recorded_model_hash = (
            artifacts.get("model_sha256") if isinstance(artifacts, Mapping) else None
        )
        if not isinstance(recorded_model_hash, str):
            verification_errors.append("training sidecar model checksum is missing")
        elif model_sha256 is not None and recorded_model_hash != model_sha256:
            verification_errors.append(
                "model checksum mismatch: "
                f"{recorded_model_hash} != {model_sha256}"
            )

        task_config = config.get("task")
        recorded_d_range = (
            task_config.get("d_range")
            if isinstance(task_config, Mapping)
            else None
        )
        if recorded_d_range != [d, d]:
            verification_errors.append(
                f"training task dimension mismatch: {recorded_d_range!r} != {[d, d]!r}"
            )

        environment_config = config.get("environment_config")
        ordering = (
            environment_config.get("action_ordering")
            if isinstance(environment_config, Mapping)
            else None
        )
        if isinstance(ordering, list) and all(
            isinstance(action, str) for action in ordering
        ):
            expected_action_ordering = list(ordering)
        else:
            verification_errors.append(
                "training sidecar action ordering is missing or invalid"
            )

        training_config = config.get("training")
        actual_value = (
            training_config.get("actual_timesteps")
            if isinstance(training_config, Mapping)
            else None
        )
        if isinstance(actual_value, int) and not isinstance(actual_value, bool):
            sidecar_actual_timesteps = int(actual_value)
        else:
            verification_errors.append(
                "training sidecar actual_timesteps is missing or invalid"
            )
    elif config_path is not None and config_error is None:
        verification_errors.append("training sidecar root must be a JSON object")

    requested_training_timesteps = _requested_timesteps(config)
    if isinstance(config, Mapping) and requested_training_timesteps is None:
        verification_errors.append(
            "training sidecar requested_timesteps is missing or invalid"
        )
    integrity_verified = model_path is not None and not verification_errors
    return {
        "dimension": int(d),
        "model_path": _display_path(formatted),
        "model_exists": model_path is not None,
        "model_sha256": model_sha256,
        "config_path": _display_path(config_path) if config_path is not None else None,
        "config_sha256": config_sha256,
        "config": config,
        "config_error": config_error,
        "integrity_verified": integrity_verified,
        "verification_errors": verification_errors,
        "requested_training_timesteps": requested_training_timesteps,
        "sidecar_actual_training_timesteps": sidecar_actual_timesteps,
        "expected_action_ordering": expected_action_ordering,
        "actual_training_timesteps": None,
    }


def _trusted_model_path(
    template: str,
    d: int,
    metadata: Mapping[str, Any],
) -> tuple[Optional[Path], Optional[str]]:
    model_path = _model_path(template, d)
    if model_path is None:
        return None, None
    if not metadata.get("integrity_verified"):
        errors = metadata.get("verification_errors") or ["unknown integrity error"]
        return None, "; ".join(map(str, errors))
    return model_path, None


def _dependency_versions() -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {}
    for distribution in DEPENDENCY_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    """Atomically replace a checkpoint; caller owns no-clobber policy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


def _exclusive_create_json(path: Path, document: Mapping[str, Any]) -> None:
    """Atomically create the first checkpoint without replacing a race winner."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link is an atomic create-if-absent operation on the same
        # filesystem.  Unlike os.replace, it cannot clobber an output created
        # by another validator after our initial existence check.
        os.link(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except OSError:
            pass


def _effective_range(grid: Mapping[int, Sequence[int]]) -> dict[str, Any]:
    dimensions = sorted(grid)
    vertices = [N for values in grid.values() for N in values]
    return {
        "dimensions": dimensions,
        "dimension_min": min(dimensions) if dimensions else None,
        "dimension_max": max(dimensions) if dimensions else None,
        "vertex_min": min(vertices) if vertices else None,
        "vertex_max": max(vertices) if vertices else None,
        "grid": {str(d): list(grid[d]) for d in dimensions},
    }


def _new_document(
    *,
    args: argparse.Namespace,
    grid: Mapping[int, Sequence[int]],
    horizon: int,
    fallback_budget: int,
    model_metadata: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_ordering = {
        str(d): [optimizer.name for optimizer in _validation_optimizers(d)]
        for d in grid
    }
    action_ordering = {}
    for d in grid:
        from ideal_poly_volume_toolkit.rl.polytope import make_polytope_env

        env = make_polytope_env(d_range=(d, d), N_range=(grid[d][0], grid[d][0]))
        action_ordering[str(d)] = list(env.op_names)
        env.close()

    run_config = {
        "model_template": args.model_template,
        "effective_task_range": _effective_range(grid),
        "seeds": list(range(args.seeds)),
        "horizon": horizon,
        "fallback_budget": fallback_budget,
        "smoke": bool(args.smoke),
        "baseline_optimizer_ordering": baseline_ordering,
        "controller_action_ordering": action_ordering,
    }
    now = _utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "run_config_sha256": _json_sha256(run_config),
        "status": "in_progress",
        "created_at": now,
        "updated_at": now,
        "invocation": {
            "argv": list(sys.argv),
            "cwd": str(Path.cwd().resolve()),
            "thread_settings": {
                variable: os.environ.get(variable) for variable in THREAD_ENV_VARS
            },
        },
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "dependency_versions": _dependency_versions(),
        },
        "repository": _repository_provenance(),
        "models": {str(d): dict(model_metadata[d]) for d in sorted(model_metadata)},
        "protocol": {
            **run_config,
            "work_units": [
                "canonical objective evaluation",
                "surrogate evaluation",
                "gradient-only oracle evaluation",
            ],
            "pairing": (
                "All methods in a seeded task use one copied reset solution; "
                "the reset canonical score is one counted work unit."
            ),
            "solo_policy": (
                "One optimizer invocation receives the entire cap remaining "
                "after reset."
            ),
            "aggregation": (
                "Median and linear-interpolation IQR across successful seeds "
                "inside each (d, N, method) cell; comparisons then operate on "
                "cell medians, with applicability-specific denominators."
            ),
        },
        "completed_tasks": [],
        "rows": [],
        "aggregate": aggregate([]),
    }


def _task_key(d: int, N: int, seed: int) -> str:
    return f"d={d};N={N};seed={seed}"


def _load_resume(
    path: Path,
    fresh_document: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot resume invalid checkpoint {path}: {exc}") from exc
    if document.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"cannot resume schema {document.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    if document.get("run_config_sha256") != fresh_document.get("run_config_sha256"):
        raise RuntimeError(
            "resume arguments do not match the checkpoint run configuration"
        )
    old_source = document.get("repository", {}).get("source_sha256")
    new_source = fresh_document.get("repository", {}).get("source_sha256")
    if old_source != new_source:
        raise RuntimeError(
            "validator/RL source changed since the checkpoint; start a new output"
        )
    old_models = document.get("models", {})
    new_models = fresh_document.get("models", {})
    for dimension, new_model in new_models.items():
        old_model = old_models.get(dimension, {})
        for key in ("model_sha256", "config_sha256"):
            if old_model.get(key) != new_model.get(key):
                raise RuntimeError(
                    f"{key} changed for d={dimension} since the checkpoint; "
                    "start a new output"
                )
    document["status"] = "in_progress"
    return document


def _format_value(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-template",
        type=str,
        default="results/rl_models/ppo_d{d}_controller-v2-seed42.zip",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "results" / "rl_validation" / "results.json",
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--dims", type=str, default="3,4,5,6")
    parser.add_argument(
        "--limit-n",
        type=int,
        default=None,
        help="keep only the first <limit_n> N values per dimension",
    )
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument(
        "--fallback-budget",
        type=int,
        default=None,
        help="total cap when a paired RL rollout is unavailable (default 12*horizon)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one dimension/task/seed, horizon <=2, fallback cap <=8",
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="resume a matching in-progress checkpoint",
    )
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace an existing output with a new run",
    )
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    if args.limit_n is not None and args.limit_n < 1:
        parser.error("--limit-n must be at least 1")
    if args.horizon < 1:
        parser.error("--horizon must be at least 1")
    if args.fallback_budget is not None and args.fallback_budget < 1:
        parser.error("--fallback-budget must be at least 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    if args.out.exists() and not (args.resume or args.overwrite):
        raise SystemExit(
            f"refusing to overwrite existing output {args.out}; "
            "use --resume or --overwrite"
        )

    try:
        dims = [int(value.strip()) for value in args.dims.split(",") if value.strip()]
    except ValueError as exc:
        raise SystemExit(f"invalid --dims value: {exc}") from exc
    unsupported = sorted(set(dims) - set(DEFAULT_GRID))
    if unsupported:
        raise SystemExit(f"unsupported dimensions: {unsupported}")
    dims = list(dict.fromkeys(dims))
    if not dims:
        raise SystemExit("--dims selected no dimensions")

    seeds = args.seeds
    limit_n = args.limit_n
    horizon = args.horizon
    fallback_budget = args.fallback_budget or horizon * 12
    if args.smoke:
        dims = dims[:1]
        seeds = 1
        args.seeds = 1
        limit_n = 1
        horizon = min(horizon, 2)
        fallback_budget = min(fallback_budget, 8)

    grid = {
        d: DEFAULT_GRID[d][:limit_n] if limit_n is not None else DEFAULT_GRID[d]
        for d in dims
    }
    model_metadata = {
        d: _model_provenance(args.model_template, d)
        for d in dims
    }
    fresh_document = _new_document(
        args=args,
        grid=grid,
        horizon=horizon,
        fallback_budget=fallback_budget,
        model_metadata=model_metadata,
    )
    if args.resume:
        if not args.out.exists():
            raise SystemExit(f"cannot resume missing output {args.out}")
        document = _load_resume(args.out, fresh_document)
    else:
        document = fresh_document
        if args.overwrite:
            _atomic_write_json(args.out, document)
        else:
            try:
                _exclusive_create_json(args.out, document)
            except FileExistsError as exc:
                raise SystemExit(
                    f"refusing to overwrite output created concurrently at "
                    f"{args.out}"
                ) from exc

    completed = set(document.get("completed_tasks", []))
    total_tasks = sum(len(values) for values in grid.values()) * seeds
    task_i = 0
    for d, vertex_counts in grid.items():
        metadata = model_metadata[d]
        model_path, model_error = _trusted_model_path(
            args.model_template,
            d,
            metadata,
        )
        if model_error is not None:
            print(
                f"[error] controller integrity verification failed for d={d}: "
                f"{model_error}; using fallback cap"
            )
        elif model_path is None:
            print(
                f"[warn] no model found for d={d} "
                f"(tried {args.model_template.format(d=d)}); using fallback cap"
            )
        for N in vertex_counts:
            for seed in range(seeds):
                task_i += 1
                key = _task_key(d, N, seed)
                if key in completed:
                    print(f"[{task_i:>3}/{total_tasks}] {key} already checkpointed")
                    continue
                wall_start = time.perf_counter()
                task_rows = run_one_task(
                    d,
                    N,
                    seed,
                    model_path,
                    horizon=horizon,
                    fallback_budget=fallback_budget,
                    requested_timesteps=metadata["requested_training_timesteps"],
                    expected_actual_timesteps=metadata[
                        "sidecar_actual_training_timesteps"
                    ],
                    expected_action_ordering=metadata["expected_action_ordering"],
                    model_error=model_error,
                )
                document["rows"].extend(task_rows)
                document["completed_tasks"].append(key)
                completed.add(key)

                rl_row = next(
                    (row for row in task_rows if row["method"] == "rl"), None
                )
                if rl_row is not None and rl_row.get("actual_training_timesteps") is not None:
                    document["models"][str(d)]["actual_training_timesteps"] = (
                        rl_row["actual_training_timesteps"]
                    )
                document["aggregate"] = aggregate(document["rows"])
                document["updated_at"] = _utc_now()
                _atomic_write_json(args.out, document)

                successful_solos = [
                    row["best_value"]
                    for row in task_rows
                    if row["method"].startswith("solo_")
                    and row.get("status") == "ok"
                ]
                print(
                    f"[{task_i:>3}/{total_tasks}] d={d} N={N} seed={seed} "
                    f"rl={_format_value(rl_row.get('best_value') if rl_row and rl_row.get('status') == 'ok' else None)} "
                    f"best_solo={_format_value(max(successful_solos) if successful_solos else None)} "
                    f"({time.perf_counter() - wall_start:.1f}s)"
                )

    document["aggregate"] = aggregate(document["rows"])
    document["status"] = "complete"
    document["updated_at"] = _utc_now()
    _atomic_write_json(args.out, document)
    print(f"\nWrote {len(document['rows'])} rows to {_display_path(args.out)}")

    print("\n--- Task-cell aggregate summary ---")
    print(f"{'method':>24} {'geomean':>10} {'win/all':>12} {'win/solo':>12}")
    methods = document["aggregate"]["methods"]
    def sort_key(item: tuple[str, Mapping[str, Any]]) -> float:
        geomean = item[1]["geomean_ratio_vs_best_solo"]
        return -(geomean if geomean is not None else -math.inf)

    for method, summary in sorted(methods.items(), key=sort_key):
        geomean = summary["geomean_ratio_vs_best_solo"]
        all_wins = f"{summary['win_numerator']}/{summary['win_denominator']}"
        solo_wins = (
            f"{summary['solo_win_numerator']}/{summary['solo_win_denominator']}"
        )
        print(
            f"{method:>24} "
            f"{('n/a' if geomean is None else f'{geomean:.3f}'):>10} "
            f"{all_wins:>12} {solo_wins:>12}"
        )


if __name__ == "__main__":
    main()
