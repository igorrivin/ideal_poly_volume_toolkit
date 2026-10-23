"""Strict work-budget regressions for reusable and polytope optimizers."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys
from typing import Any

import numpy as np
import pytest

import ideal_poly_volume_toolkit.rl.polytope.optimizers as polytope_optimizers
from ideal_poly_volume_toolkit.rl.generic import (
    GenericOptEnv,
    InnerOptimizer,
    OptimizationTask,
    OptimizerRun,
    OptimizerRunError,
    TaskSampler,
)
from ideal_poly_volume_toolkit.rl.inner_optimizers import (
    CMAESOptimizer,
    RandomRestartOptimizer,
    ScipyMinimizeOptimizer,
)
from ideal_poly_volume_toolkit.rl.polytope.optimizers import (
    FreshTopologyADHyperbolicD4,
    H4OptimizerRun,
    LBFGSHyperbolicD3,
    LBFGSHyperbolicD4,
    LBFGSHyperbolicD5,
    LBFGSHyperbolicD6,
)
from ideal_poly_volume_toolkit.rl.polytope.task import IdealPolytopeTask
from ideal_poly_volume_toolkit.rl.training import compare_with_baseline


_VALIDATOR = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "validate_rl_controller.py")
)
_baseline_solo = _VALIDATOR["_baseline_solo"]
_baseline_round_robin = _VALIDATOR["_baseline_round_robin"]


class _CountingVectorTask(OptimizationTask):
    def __init__(self) -> None:
        self.objective_calls = 0

    @staticmethod
    def canonical_value(solution: np.ndarray) -> float:
        return -float(np.sum(np.asarray(solution, dtype=float) ** 2))

    def random_solution(self, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(size=2)

    def evaluate(self, solution: np.ndarray) -> float:
        self.objective_calls += 1
        return self.canonical_value(solution)

    def perturb(self, solution, radius, rng):
        return np.asarray(solution) + radius * rng.normal(size=2)

    def metadata(self) -> np.ndarray:
        return np.array([2.0], dtype=np.float32)

    def copy_solution(self, solution: np.ndarray) -> np.ndarray:
        return np.asarray(solution, dtype=float).copy()

    def to_vector(self, solution: np.ndarray) -> np.ndarray:
        return np.asarray(solution, dtype=float).copy()

    def from_vector(self, vec: np.ndarray) -> np.ndarray:
        return np.asarray(vec, dtype=float).copy()

    def project(self, vec: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(vec, dtype=float), -4.0, 4.0)


class _DistinctSurrogateTask(_CountingVectorTask):
    def __init__(self) -> None:
        super().__init__()
        self.surrogate_calls = 0

    def evaluate_surrogate(self, solution: np.ndarray) -> float:
        self.surrogate_calls += 1
        offset = np.asarray(solution, dtype=float) - 1.0
        return -float(np.sum(offset**2))


class _VectorSampler(TaskSampler):
    def sample(self, rng: np.random.Generator) -> OptimizationTask:
        return _CountingVectorTask()

    def metadata_dim(self) -> int:
        return 1


class _FailsOnFinalD5Score(IdealPolytopeTask):
    def __init__(self) -> None:
        super().__init__(d=5, N=7)
        self.objective_calls = 0

    def evaluate(self, solution) -> float:
        self.objective_calls += 1
        if self.objective_calls > 1:
            raise RuntimeError("deliberate final-score failure")
        return super().evaluate(solution)


class _CountingH4Task(IdealPolytopeTask):
    def __init__(self, n_vertices: int = 5) -> None:
        super().__init__(d=4, N=n_vertices)
        self.objective_calls = 0

    def evaluate(self, solution) -> float:
        self.objective_calls += 1
        return super().evaluate(solution)


class _FailsOnFreshH4Endpoint(_CountingH4Task):
    def evaluate(self, solution) -> float:
        self.objective_calls += 1
        if self.objective_calls > 1:
            raise RuntimeError("deliberate H4 endpoint-score failure")
        return IdealPolytopeTask.evaluate(self, solution)


class _FailingD5Sampler(TaskSampler):
    def sample(self, rng: np.random.Generator) -> OptimizationTask:
        return _FailsOnFinalD5Score()

    def metadata_dim(self) -> int:
        return 3


class _ThreeWorkD5(LBFGSHyperbolicD5):
    default_budget = 3


def _continuous_optimizers(use_surrogate: bool):
    return [
        ScipyMinimizeOptimizer(method="Powell", use_surrogate=use_surrogate),
        CMAESOptimizer(use_surrogate=use_surrogate),
    ]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("objective_evals", -1, ValueError),
        ("surrogate_evals", -1, ValueError),
        ("gradient_evals", -1, ValueError),
        ("objective_evals", True, TypeError),
        ("surrogate_evals", 1.5, TypeError),
    ],
)
def test_optimizer_run_rejects_invalid_work_counts(
    field: str,
    value: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error, match="nonnegative integer"):
        OptimizerRun(solution=None, **{field: value})


def test_optimizer_run_error_is_exported_from_rl_package() -> None:
    from ideal_poly_volume_toolkit.rl import OptimizerRunError as public_error

    assert public_error is OptimizerRunError


@pytest.mark.parametrize("budget", [1, 3, 7])
def test_random_restart_obeys_total_work_cap(budget: int) -> None:
    task = _CountingVectorTask()
    run = RandomRestartOptimizer().run(
        np.array([2.0, -1.0]), task, budget, np.random.default_rng(1)
    )

    assert run.n_evals == budget == task.objective_calls
    assert run.objective_evals == len(run.values)
    assert run.surrogate_evals == 0
    assert run.solution_value == pytest.approx(task.canonical_value(run.solution))


@pytest.mark.parametrize("budget", [1, 3, 7])
@pytest.mark.parametrize("optimizer_index", [0, 1])
def test_continuous_objective_optimizers_obey_total_work_cap(
    budget: int, optimizer_index: int
) -> None:
    task = _CountingVectorTask()
    optimizer = _continuous_optimizers(use_surrogate=False)[optimizer_index]
    run = optimizer.run(np.array([2.0, -1.0]), task, budget, np.random.default_rng(2))

    assert 1 <= run.n_evals <= budget
    assert run.n_evals == task.objective_calls == run.objective_evals
    assert run.surrogate_evals == 0
    assert len(run.values) == run.objective_evals
    assert run.solution_value == pytest.approx(task.canonical_value(run.solution))


@pytest.mark.parametrize("budget", [1, 3, 7])
@pytest.mark.parametrize("optimizer_index", [0, 1])
def test_distinct_surrogate_reserves_a_counted_canonical_score(
    budget: int, optimizer_index: int
) -> None:
    task = _DistinctSurrogateTask()
    optimizer = _continuous_optimizers(use_surrogate=True)[optimizer_index]
    run = optimizer.run(np.array([2.0, -1.0]), task, budget, np.random.default_rng(3))

    assert 1 <= run.n_evals <= budget
    assert run.objective_evals == task.objective_calls == 1
    assert run.surrogate_evals == task.surrogate_calls
    assert run.values == [run.solution_value]
    assert run.solution_value == pytest.approx(task.canonical_value(run.solution))


@pytest.mark.parametrize("optimizer_index", [0, 1])
def test_default_surrogate_fallback_is_classified_as_objective_work(
    optimizer_index: int,
) -> None:
    task = _CountingVectorTask()
    optimizer = _continuous_optimizers(use_surrogate=True)[optimizer_index]
    run = optimizer.run(
        np.array([2.0, -1.0]), task, budget=5, rng=np.random.default_rng(4)
    )

    assert run.surrogate_evals == 0
    assert run.objective_evals == run.n_evals == task.objective_calls
    assert len(run.values) == run.objective_evals
    assert run.solution_value == pytest.approx(task.canonical_value(run.solution))


@pytest.mark.parametrize(
    ("optimizer", "task"),
    [
        (RandomRestartOptimizer(), _CountingVectorTask()),
        (ScipyMinimizeOptimizer(), _CountingVectorTask()),
        (CMAESOptimizer(), _CountingVectorTask()),
        (LBFGSHyperbolicD3(), IdealPolytopeTask(d=3, N=5)),
    ],
)
def test_nonpositive_optimizer_budget_is_rejected(optimizer, task) -> None:
    solution = task.random_solution(np.random.default_rng(5))
    with pytest.raises(ValueError, match="at least 1"):
        optimizer.run(solution, task, budget=0, rng=np.random.default_rng(6))


@pytest.mark.parametrize(
    ("dimension", "n_vertices", "optimizer_type"),
    [
        (3, 5, LBFGSHyperbolicD3),
        (4, 6, LBFGSHyperbolicD4),
    ],
)
def test_powell_polytope_optimizers_use_canonical_budget(
    dimension: int, n_vertices: int, optimizer_type
) -> None:
    rng = np.random.default_rng(10 + dimension)
    task = IdealPolytopeTask(d=dimension, N=n_vertices)
    run = optimizer_type().run(task.random_solution(rng), task, budget=3, rng=rng)

    assert run.n_evals == run.objective_evals == 3
    assert len(run.values) == 3
    assert run.solution_value == pytest.approx(task.evaluate(run.solution))


def test_fixed_pole_chart_roundtrip_preserves_every_vertex_label() -> None:
    rng = np.random.default_rng(20260817)
    vertices = rng.standard_normal((7, 4))
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)

    for pole_index in (0, 3, 6):
        projected, finite_indices, transform = (
            polytope_optimizers._fixed_pole_project(vertices, pole_index)
        )
        restored = polytope_optimizers._fixed_pole_unproject(
            projected,
            finite_indices,
            transform,
            n_vertices=len(vertices),
            pole_index=pole_index,
        )

        assert finite_indices.tolist() == [
            label for label in range(len(vertices)) if label != pole_index
        ]
        np.testing.assert_allclose(restored, vertices, rtol=0.0, atol=4.0e-15)


@pytest.mark.parametrize("budget", [1, 3, 1200])
def test_fresh_topology_h4_obeys_total_work_cap_on_real_geometry(
    budget: int,
) -> None:
    rng = np.random.default_rng(2718)
    task = _CountingH4Task()
    initial = task.random_solution(rng)
    run = FreshTopologyADHyperbolicD4().run(
        initial,
        task,
        budget=budget,
        rng=np.random.default_rng(99),
    )

    assert isinstance(run, H4OptimizerRun)
    assert 1 <= run.n_evals <= budget
    assert run.n_evals == task.objective_calls + run.gradient_evals
    assert run.objective_evals == task.objective_calls == len(run.values)
    assert run.surrogate_evals == 0
    assert run.values[0] == pytest.approx(
        IdealPolytopeTask.evaluate(task, initial), rel=0.0, abs=0.0
    )
    assert run.solution_value == max(run.values)
    assert run.diagnostics["selector"] == "fresh_topology_ad"
    assert run.diagnostics["work"]["total_evals"] == run.n_evals
    assert len(run.diagnostics["branches"]) <= 8
    assert all(
        len(branch["topology_signature_sha256_examples"]) <= 16
        for branch in run.diagnostics["branches"]
    )
    if budget == 1:
        assert run.objective_evals == 1
        assert run.gradient_evals == 0
    elif budget == 3:
        assert run.objective_evals == 2
        assert run.gradient_evals == 1
    else:
        assert run.objective_evals == 9
        assert run.diagnostics["budget_allocation"]["gradient_evals_by_branch"] == [
            149,
            149,
            149,
            149,
            149,
            149,
            149,
            148,
        ]


def test_fresh_topology_h4_rebuilds_delaunay_for_every_gradient_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(1234)
    task = _CountingH4Task()
    initial = task.random_solution(rng)
    real_delaunay = polytope_optimizers.Delaunay
    projected_inputs: list[np.ndarray] = []

    def tracked_delaunay(points: np.ndarray):
        projected_inputs.append(np.asarray(points, dtype=float).copy())
        return real_delaunay(points)

    monkeypatch.setattr(polytope_optimizers, "Delaunay", tracked_delaunay)
    run = FreshTopologyADHyperbolicD4(
        pole_indices=[2],
        branch_modes=("lbfgs",),
    ).run(initial, task, budget=8, rng=np.random.default_rng(5))

    assert len(projected_inputs) == run.gradient_evals
    assert len(projected_inputs) >= 1
    branch = run.diagnostics["branches"][0]
    assert branch["topology_evaluations"] == run.gradient_evals
    assert branch["finite_vertex_labels"] == [0, 1, 3, 4]
    assert branch["topology_first_sha256"] is not None
    assert branch["topology_last_sha256"] is not None
    assert branch["topology_trace_sha256"] is not None


def test_fresh_topology_h4_endpoint_failure_keeps_exact_partial_accounting() -> None:
    rng = np.random.default_rng(9876)
    task = _FailsOnFreshH4Endpoint()
    initial = task.random_solution(rng)
    run = FreshTopologyADHyperbolicD4(
        pole_indices=[0],
        branch_modes=("lbfgs",),
    ).run(initial, task, budget=3, rng=rng)

    assert run.n_evals == 3
    assert run.objective_evals == task.objective_calls == 2
    assert run.gradient_evals == 1
    assert len(run.values) == 1  # Failed canonical calls count but add no value.
    assert run.solution_value == run.values[0]
    assert run.diagnostics["failed_branch_count"] == 1
    assert run.diagnostics["canonical_scored_branch_count"] == 0
    assert run.diagnostics["branches"][0]["error_type"] == "RuntimeError"


def test_fresh_topology_h4_missing_backend_fails_before_counted_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _CountingH4Task()
    initial = task.random_solution(np.random.default_rng(6543))
    optimizer = FreshTopologyADHyperbolicD4()

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match="requires the optional PyTorch"):
        optimizer.run(initial, task, budget=3, rng=np.random.default_rng(0))

    assert task.objective_calls == 0


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"pole_indices": []}, ValueError, "empty"),
        ({"pole_indices": [0, 0]}, ValueError, "duplicates"),
        ({"pole_indices": [True]}, TypeError, "integer"),
        ({"branch_modes": []}, ValueError, "empty"),
        ({"branch_modes": ["newton"]}, ValueError, "lbfgs"),
        ({"branch_modes": ["lbfgs", "lbfgs"]}, ValueError, "duplicates"),
        ({"adam_steps": -1}, ValueError, "nonnegative"),
        ({"adam_lr": 0.0}, ValueError, "positive"),
    ],
)
def test_fresh_topology_h4_rejects_invalid_controls(
    kwargs: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        FreshTopologyADHyperbolicD4(**kwargs)


@pytest.mark.parametrize(
    ("dimension", "n_vertices", "optimizer_type"),
    [
        (5, 7, LBFGSHyperbolicD5),
        (6, 8, LBFGSHyperbolicD6),
    ],
)
def test_gradient_polytope_optimizers_retriangulate_each_iterate(
    dimension: int,
    n_vertices: int,
    optimizer_type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(dimension)
    task = IdealPolytopeTask(d=dimension, N=n_vertices)
    solution = task.random_solution(rng)
    triangulate = task.triangulate
    seen_vertices: list[np.ndarray] = []

    def tracked_triangulation(vertices: np.ndarray):
        seen_vertices.append(np.asarray(vertices).copy())
        return triangulate(vertices)

    monkeypatch.setattr(task, "triangulate", tracked_triangulation)
    run = optimizer_type().run(solution, task, budget=3, rng=rng)

    assert run.n_evals == 3
    assert run.objective_evals == 1
    assert run.gradient_evals == 2
    assert len(seen_vertices) == run.gradient_evals
    assert not np.allclose(seen_vertices[0], seen_vertices[1])
    assert run.values == [run.solution_value]
    assert run.solution_value == pytest.approx(task.evaluate(run.solution))


def test_projected_gradient_failure_preserves_partial_work_accounting() -> None:
    env = GenericOptEnv(
        _FailingD5Sampler(),
        [_ThreeWorkD5()],
        eval_cost_weight=0.0,
        seed=17,
    )
    env.reset(seed=17)

    _, _, _, _, info = env.step((0, 0))

    assert info["total_evals"] == 4  # reset + two gradients + failed score
    assert info["last_error"].startswith("OptimizerRunError:")
    assert info["last_optimizer_metrics"]["objective_evals"] == 1
    assert info["last_optimizer_metrics"]["gradient_evals"] == 2


def test_d5_identical_surrogate_is_counted_as_canonical_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = IdealPolytopeTask(d=5, N=7)
    solution = task.random_solution(np.random.default_rng(18))
    objective_calls = 0
    surrogate_calls = 0
    original_objective = task.evaluate
    original_surrogate = task.evaluate_surrogate

    def counted_objective(candidate):
        nonlocal objective_calls
        objective_calls += 1
        return original_objective(candidate)

    def counted_surrogate(candidate):
        nonlocal surrogate_calls
        surrogate_calls += 1
        return original_surrogate(candidate)

    monkeypatch.setattr(task, "evaluate", counted_objective)
    monkeypatch.setattr(task, "evaluate_surrogate", counted_surrogate)
    run = ScipyMinimizeOptimizer(method="Powell", use_surrogate=True).run(
        solution,
        task,
        budget=3,
        rng=np.random.default_rng(19),
    )

    assert run.objective_evals == objective_calls == 3
    assert run.surrogate_evals == surrogate_calls == 0
    assert run.solution_value is not None


def test_polytope_default_budgets_match_total_work_units() -> None:
    assert LBFGSHyperbolicD3.default_budget == 90
    assert LBFGSHyperbolicD4.default_budget == 60
    assert LBFGSHyperbolicD5.default_budget == 10
    assert LBFGSHyperbolicD6.default_budget == 10


def test_baseline_comparison_records_work_under_cap() -> None:
    def env_factory() -> GenericOptEnv:
        return GenericOptEnv(_VectorSampler(), [], seed=20)

    summary = compare_with_baseline(
        env_factory,
        n_trials=3,
        budget_per_trial=4,
        baselines=[RandomRestartOptimizer()],
        seed=21,
    )["random_restart"]

    assert summary["budget_per_trial"] == 4
    assert summary["evals_per_trial"] == [4, 4, 4]
    assert summary["mean_evals"] == 4.0
    assert summary["max_evals"] == 4


class _OverBudgetOptimizer(InnerOptimizer):
    @property
    def name(self) -> str:
        return "over_budget"

    def run(
        self,
        solution: Any,
        task: OptimizationTask,
        budget: int,
        rng: np.random.Generator,
    ) -> OptimizerRun:
        return OptimizerRun(
            solution=task.copy_solution(solution),
            values=[task.evaluate(solution)],
            solution_value=task.canonical_value(solution),
            objective_evals=budget + 1,
        )


def test_baseline_comparison_rejects_over_budget_optimizer() -> None:
    def env_factory() -> GenericOptEnv:
        return GenericOptEnv(_VectorSampler(), [], seed=30)

    with pytest.raises(RuntimeError, match="exceeded its work budget"):
        compare_with_baseline(
            env_factory,
            n_trials=1,
            budget_per_trial=2,
            baselines=[_OverBudgetOptimizer()],
        )


@pytest.mark.parametrize("budget", [1, 2, 7])
def test_validation_baselines_match_total_budget(budget: int) -> None:
    solo = _baseline_solo("lbfgs_hyp_d3", 3, 5, seed=31, budget=budget)
    round_robin = _baseline_round_robin(3, 5, seed=31, budget=budget)

    assert solo["total_evals"] == solo["budget_cap"] == budget
    assert round_robin["total_evals"] == round_robin["budget_cap"] == budget
