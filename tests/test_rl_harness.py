"""Regression tests for RL search semantics and optimizer accounting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import ideal_poly_volume_toolkit.rl.polytope.task as polytope_task_module
from ideal_poly_volume_toolkit.rl.generic import (
    AnnealRowOptimizer,
    GenericOptEnv,
    GreedyFlipOptimizer,
    HadamardTask,
    InnerOptimizer,
    MAX_FRONTIER,
    MultiFlipOptimizer,
    OptimizationTask,
    OptimizerRun,
    PaleyStartOptimizer,
    RowOperationOptimizer,
    SimulatedAnnealingOptimizer,
    SteepestDescentOptimizer,
    SylvesterStartOptimizer,
    TaskSampler,
    TensorProductOptimizer,
    make_hadamard_env,
)
from ideal_poly_volume_toolkit.rl.polytope.optimizers import LBFGSHyperbolicD3
from ideal_poly_volume_toolkit.rl.polytope.task import IdealPolytopeTask
from ideal_poly_volume_toolkit.rl.polytope.types import PolytopeSolution
from ideal_poly_volume_toolkit.rl.training import compare_with_baseline


class _VectorTask(OptimizationTask):
    def random_solution(self, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(size=3)

    def evaluate(self, solution: np.ndarray) -> float:
        return -float(np.sum(np.asarray(solution) ** 2))

    def evaluate_surrogate(self, solution: np.ndarray) -> float:
        return 12345.0

    def perturb(
        self,
        solution: np.ndarray,
        radius: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return np.asarray(solution) + radius * rng.normal(size=3)

    def metadata(self) -> np.ndarray:
        return np.array([3.0], dtype=np.float32)

    def copy_solution(self, solution: np.ndarray) -> np.ndarray:
        return np.asarray(solution).copy()


class _VectorSampler(TaskSampler):
    def sample(self, rng: np.random.Generator) -> OptimizationTask:
        return _VectorTask()

    def metadata_dim(self) -> int:
        return 1


class _AccountedOptimizer(InnerOptimizer):
    default_budget = 12

    @property
    def name(self) -> str:
        return "accounted"

    def run(
        self,
        solution: Any,
        task: OptimizationTask,
        budget: int,
        rng: np.random.Generator,
    ) -> OptimizerRun:
        return OptimizerRun(
            solution=np.zeros(3),
            values=[0.0],
            solution_value=0.0,
            objective_evals=2,
            surrogate_evals=7,
            gradient_evals=3,
        )


class _UnsupportedOptimizer(InnerOptimizer):
    @property
    def name(self) -> str:
        return "unsupported"

    def supports(self, task: OptimizationTask) -> bool:
        return False

    def run(self, solution, task, budget, rng):  # pragma: no cover
        raise AssertionError("unsupported optimizer must not be run")


class _FailingOptimizer(InnerOptimizer):
    default_budget = 8

    @property
    def name(self) -> str:
        return "failing"

    def run(self, solution, task, budget, rng):
        for _ in range(5):
            task.evaluate(solution)
        raise RuntimeError("deliberate failure")


class _UncountedCachedOptimizer(InnerOptimizer):
    """Return an attractive value without doing the score it claims."""

    @property
    def name(self) -> str:
        return "uncounted_cached"

    def run(self, solution, task, budget, rng):
        return OptimizerRun(
            solution=np.zeros_like(solution),
            values=[],
            solution_value=0.0,
        )


class _WalkingTask(_VectorTask):
    def random_solution(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([0.0])

    def evaluate(self, solution: np.ndarray) -> float:
        return float(np.asarray(solution)[0])

    def perturb(self, solution, radius, rng):
        return np.asarray(solution) + 1.0


class _WalkingSampler(TaskSampler):
    def sample(self, rng: np.random.Generator) -> OptimizationTask:
        return _WalkingTask()

    def metadata_dim(self) -> int:
        return 1


class _FailsDuringSampleTask(_VectorTask):
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, solution: np.ndarray) -> float:
        self.calls += 1
        if self.calls == 3:
            raise RuntimeError("sample score failed")
        return super().evaluate(solution)


class _FailsDuringSampleSampler(TaskSampler):
    def sample(self, rng: np.random.Generator) -> OptimizationTask:
        return _FailsDuringSampleTask()

    def metadata_dim(self) -> int:
        return 1


class _FailsDuringSplitTask(_VectorTask):
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, solution: np.ndarray) -> float:
        self.calls += 1
        if self.calls == 3:
            raise RuntimeError("split score failed")
        return super().evaluate(solution)


class _FailsDuringSplitSampler(TaskSampler):
    def sample(self, rng: np.random.Generator) -> OptimizationTask:
        return _FailsDuringSplitTask()

    def metadata_dim(self) -> int:
        return 1


class _RecordingHadamardTask(HadamardTask):
    """Hadamard task that retains every canonical score and its input."""

    def __init__(self, n: int):
        super().__init__(n)
        self.evaluations: list[tuple[np.ndarray, float]] = []

    def _canonical_value(self, solution: np.ndarray) -> float:
        return float(HadamardTask.evaluate(self, solution))

    def evaluate(self, solution: np.ndarray) -> float:
        value = self._canonical_value(solution)
        self.evaluations.append((np.asarray(solution).copy(), value))
        return value


class _NeverPerfectHadamardTask(_RecordingHadamardTask):
    """Prevent exact constructions from taking their perfect-score shortcut."""

    def _canonical_value(self, solution: np.ndarray) -> float:
        return float(HadamardTask.evaluate(self, solution) - 1.0)


_HADAMARD_OPTIMIZER_TYPES = (
    GreedyFlipOptimizer,
    RowOperationOptimizer,
    SteepestDescentOptimizer,
    SylvesterStartOptimizer,
    PaleyStartOptimizer,
    MultiFlipOptimizer,
    SimulatedAnnealingOptimizer,
    AnnealRowOptimizer,
    TensorProductOptimizer,
)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    return rotation


def test_sample_accepts_numpy_best_solution_and_charges_four_scores() -> None:
    env = GenericOptEnv(_VectorSampler(), [], seed=1)
    env.reset(seed=1)

    _, _, _, _, info = env.step((0, env.OP_SAMPLE))

    assert info["last_error"] is None
    assert info["total_evals"] == 5
    assert env.root.n_evals == 5


def test_sample_failure_preserves_partial_objective_accounting() -> None:
    env = GenericOptEnv(_FailsDuringSampleSampler(), [], seed=11)
    env.reset(seed=11)

    _, _, _, _, info = env.step((0, env.OP_SAMPLE))

    assert info["last_error"] == "RuntimeError: sample score failed"
    assert info["total_evals"] == 3
    assert env.root.n_evals == 3
    assert len(env.root.values_history) == 2


def test_reset_initializes_gym_rng_and_root_statistics() -> None:
    env = GenericOptEnv(_VectorSampler(), [], horizon=2, seed=12)
    observation, _ = env.reset(seed=12)

    assert env.np_random is env._rng
    assert env.root.mean_value == pytest.approx(env.root.best_value)
    assert observation[3] == pytest.approx(env.root.best_value)


def test_horizon_is_a_time_limit_truncation() -> None:
    env = GenericOptEnv(_VectorSampler(), [], horizon=2, seed=13)
    env.reset(seed=13)

    _, _, terminated, truncated, _ = env.step((0, env.OP_PRUNE))
    assert not terminated
    assert not truncated

    _, _, terminated, truncated, _ = env.step((0, env.OP_PRUNE))
    assert not terminated
    assert truncated


def test_hadamard_environment_passes_gymnasium_checker() -> None:
    from gymnasium.utils.env_checker import check_env

    with pytest.warns(UserWarning, match="Box observation space"):
        check_env(make_hadamard_env([4], horizon=2, seed=14), skip_render_check=True)


def test_split_evaluates_distinct_child_centers() -> None:
    env = GenericOptEnv(_VectorSampler(), [], seed=2)
    env.reset(seed=2)
    parent_center = env.root.center.copy()

    _, _, _, _, info = env.step((0, env.OP_SPLIT))

    assert info["total_evals"] == 3
    assert len(env.frontier) == 2
    for child in env.frontier:
        assert not np.array_equal(child.center, parent_center)
        np.testing.assert_array_equal(child.best_solution, child.center)
        assert child.best_value == pytest.approx(env.task.evaluate(child.center))


def test_nonfirst_numpy_node_can_be_split_and_pruned_by_identity() -> None:
    env = GenericOptEnv(_VectorSampler(), [], seed=22)
    env.reset(seed=22)

    _, _, _, _, info = env.step((0, env.OP_SPLIT))
    assert info["last_error"] is None
    assert info["total_evals"] == 3
    assert len(env.frontier) == 2

    _, _, _, _, info = env.step((1, env.OP_SPLIT))
    assert info["last_error"] is None
    assert info["total_evals"] == 5
    assert len(env.frontier) == 3

    _, _, _, _, info = env.step((2, env.OP_PRUNE))
    assert info["last_error"] is None
    assert info["total_evals"] == 5
    assert len(env.frontier) == 2


def test_node_slots_map_evenly_across_two_frontier_nodes() -> None:
    env = GenericOptEnv(
        _VectorSampler(),
        [],
        horizon=MAX_FRONTIER + 2,
        seed=220,
    )
    env.reset(seed=220)
    env.step((0, env.OP_SPLIT))
    assert len(env.frontier) == 2

    effective_indices = []
    for requested_index in range(MAX_FRONTIER):
        _, _, _, _, info = env.step((requested_index, env.OP_SAMPLE))
        effective_indices.append(info["effective_node_index"])
        assert info["requested_node_index"] == requested_index
        assert info["effective_node_index"] == requested_index % 2
        assert info["node_index_remapped"] is (requested_index >= 2)

    assert effective_indices.count(0) == MAX_FRONTIER // 2
    assert effective_indices.count(1) == MAX_FRONTIER // 2
    assert info["node_index_remap_count"] == MAX_FRONTIER - 2


def test_valid_node_indices_are_unchanged_and_remap_telemetry_resets() -> None:
    env = GenericOptEnv(_VectorSampler(), [], horizon=8, seed=221)
    env.reset(seed=221)
    env.step((0, env.OP_SPLIT))

    _, _, _, _, info = env.step((1, env.OP_SAMPLE))
    assert info["requested_node_index"] == 1
    assert info["effective_node_index"] == 1
    assert info["node_index_remapped"] is False
    assert info["node_index_remap_count"] == 0

    _, _, _, _, info = env.step((5, env.OP_SAMPLE))
    assert info["requested_node_index"] == 5
    assert info["effective_node_index"] == 1
    assert info["node_index_remapped"] is True
    assert info["node_index_remap_count"] == 1

    _, info = env.reset(seed=221)
    assert info["requested_node_index"] is None
    assert info["effective_node_index"] is None
    assert info["node_index_remapped"] is False
    assert info["node_index_remap_count"] == 0


def test_node_remapping_preserves_seeded_transition_determinism() -> None:
    envs = [
        GenericOptEnv(_VectorSampler(), [], horizon=8, seed=222)
        for _ in range(2)
    ]
    for env in envs:
        env.reset(seed=222)
        env.step((0, env.OP_SPLIT))

    transitions = []
    for env in envs:
        transitions.append(env.step((15, env.OP_SAMPLE)))

    first, second = transitions
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:] == second[1:]
    assert first[4]["requested_node_index"] == 15
    assert first[4]["effective_node_index"] == 1
    assert first[4]["node_index_remap_count"] == 1


def test_split_failure_preserves_partial_objective_accounting() -> None:
    env = GenericOptEnv(_FailsDuringSplitSampler(), [], seed=23)
    env.reset(seed=23)

    _, _, _, _, info = env.step((0, env.OP_SPLIT))

    assert info["last_error"] == "RuntimeError: split score failed"
    assert info["total_evals"] == 3
    assert env.root.n_evals == 3
    assert len(env.root.values_history) == 2
    assert env.frontier == [env.root]


def test_optimizer_work_is_accounted_without_polluting_objective_history() -> None:
    env = GenericOptEnv(
        _VectorSampler(),
        [_AccountedOptimizer()],
        eval_cost_weight=0.0,
        seed=3,
    )
    env.reset(seed=3)

    _, _, _, _, info = env.step((0, 0))

    # 1 reset score + 2 internal objectives + 7 surrogates + 3 gradients
    # The returned solution's canonical value is cached, so no duplicate final
    # score is charged.
    assert info["total_evals"] == 13
    assert info["last_optimizer_metrics"] == {
        "name": "accounted",
        "objective_evals": 2,
        "surrogate_evals": 7,
        "gradient_evals": 3,
    }
    assert len(env.root.values_history) == 2
    assert 12345.0 not in env.root.values_history


def test_optimizer_run_preserves_two_tuple_sequence_compatibility() -> None:
    run = OptimizerRun(solution="solution", values=[1.0])

    assert len(run) == 2
    assert run[0] == "solution"
    assert run[-1] == [1.0]
    assert run[:] == ("solution", [1.0])


def test_failed_optimizer_is_charged_for_observed_objective_calls() -> None:
    env = GenericOptEnv(_VectorSampler(), [_FailingOptimizer()], seed=4)
    env.reset(seed=4)

    _, _, _, _, info = env.step((0, 0))

    assert info["total_evals"] == 6
    assert info["last_error"] == "RuntimeError: deliberate failure"
    assert info["last_optimizer_metrics"]["objective_evals"] == 5


def test_uncounted_cached_solution_value_is_rejected_and_charged() -> None:
    env = GenericOptEnv(
        _VectorSampler(),
        [_UncountedCachedOptimizer()],
        eval_cost_weight=0.0,
        seed=41,
    )
    env.reset(seed=41)
    old_best = env.best_value

    _, _, _, _, info = env.step((0, 0))

    assert info["total_evals"] == 2
    assert info["last_error"] == (
        "optimizer 'uncounted_cached' returned an uncounted canonical "
        "solution_value"
    )
    assert env.best_value == old_best


def test_sampling_moves_center_so_split_uses_the_best_region() -> None:
    env = GenericOptEnv(_WalkingSampler(), [], seed=5)
    env.reset(seed=5)

    env.step((0, env.OP_SAMPLE))
    assert env.root.best_value == 4.0
    np.testing.assert_array_equal(env.root.center, [4.0])

    env.step((0, env.OP_SPLIT))
    assert [child.best_value for child in env.frontier] == [5.0, 5.0]


def test_h3_task_score_is_rotation_and_projection_pole_invariant() -> None:
    rng = np.random.default_rng(2026)
    points = rng.normal(size=(9, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    task = IdealPolytopeTask(d=3, N=len(points))
    expected = task.evaluate(PolytopeSolution(points, []))

    for _ in range(4):
        rotation = _random_rotation(rng)
        rotated = points @ rotation
        assert task.evaluate(PolytopeSolution(rotated, [])) == pytest.approx(
            expected, rel=3e-14, abs=3e-14
        )

    for pole_index in (2, 5, 8):
        order = np.r_[pole_index, np.delete(np.arange(len(points)), pole_index)]
        assert task.evaluate(PolytopeSolution(points[order], [])) == pytest.approx(
            expected, rel=3e-14, abs=3e-14
        )


def test_d3_powell_reports_internal_objective_calls() -> None:
    rng = np.random.default_rng(7)
    task = IdealPolytopeTask(d=3, N=5)
    solution = task.random_solution(rng)
    run = LBFGSHyperbolicD3().run(solution, task, budget=1, rng=rng)

    assert isinstance(run, OptimizerRun)
    assert run.objective_evals == 1
    assert run.n_evals == run.objective_evals
    assert run.values == [run.solution_value]
    assert run.solution_value == pytest.approx(task.evaluate(run.solution))


def test_dimension_specific_optimizer_rejects_wrong_task() -> None:
    optimizer = LBFGSHyperbolicD3()
    assert optimizer.supports(IdealPolytopeTask(d=3, N=5))
    assert not optimizer.supports(IdealPolytopeTask(d=4, N=6))
    lookalike = _VectorTask()
    lookalike.d = 3
    assert not optimizer.supports(lookalike)


def test_d6_triangulation_cache_detects_vertex_and_simplex_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = polytope_task_module._stereographic_delaunay

    def recording_triangulation(points):
        nonlocal calls
        calls += 1
        return original(points)

    monkeypatch.setattr(
        polytope_task_module,
        "_stereographic_delaunay",
        recording_triangulation,
    )
    monkeypatch.setattr(
        polytope_task_module,
        "_h6_volume",
        lambda points, simplices: float(len(simplices)),
    )

    task = IdealPolytopeTask(d=6, N=9)
    rng = np.random.default_rng(188)
    solution = task.random_solution(rng)
    task.evaluate(solution)
    task.evaluate(solution)
    assert calls == 1

    stale_simplices = [list(simplex) for simplex in solution.simplices]
    solution.vertices[:] = task.random_solution(rng).vertices
    task.evaluate(solution)
    assert calls == 2

    solution.simplices = stale_simplices
    task.evaluate(solution)
    assert calls == 3


def test_baseline_comparison_skips_unsupported_task_optimizer_pairs() -> None:
    def env_factory() -> GenericOptEnv:
        return GenericOptEnv(_VectorSampler(), [], seed=9)

    summary = compare_with_baseline(
        env_factory,
        n_trials=2,
        baselines=[_UnsupportedOptimizer()],
    )["unsupported"]

    assert summary["n_trials"] == 0
    assert summary["n_trials_requested"] == 2
    assert np.isnan(summary["mean_value"])


@pytest.mark.parametrize(
    "optimizer_type",
    _HADAMARD_OPTIMIZER_TYPES,
    ids=lambda optimizer_type: optimizer_type.__name__,
)
@pytest.mark.parametrize("budget", [1, 2, 5])
def test_hadamard_optimizers_respect_total_work_budget_and_return_scored_solution(
    optimizer_type,
    budget: int,
) -> None:
    rng = np.random.default_rng(10_000 + budget)
    task = _RecordingHadamardTask(8)
    initial = task.random_solution(rng)

    run = optimizer_type().run(initial, task, budget=budget, rng=rng)

    assert isinstance(run, OptimizerRun)
    assert 1 <= run.n_evals <= budget
    assert run.n_evals == run.objective_evals + run.gradient_evals
    assert run.surrogate_evals == 0
    assert run.objective_evals == len(task.evaluations)
    assert run.objective_evals == len(run.values)
    assert run.values == pytest.approx([value for _, value in task.evaluations])
    assert run.solution_value == pytest.approx(
        HadamardTask.evaluate(task, run.solution)
    )
    assert any(
        np.array_equal(run.solution, scored_solution)
        and run.solution_value == pytest.approx(scored_value)
        for scored_solution, scored_value in task.evaluations
    )


@pytest.mark.parametrize(
    "optimizer_type",
    _HADAMARD_OPTIMIZER_TYPES,
    ids=lambda optimizer_type: optimizer_type.__name__,
)
@pytest.mark.parametrize("budget", [0, -1])
def test_hadamard_optimizers_reject_nonpositive_work_budgets(
    optimizer_type,
    budget: int,
) -> None:
    rng = np.random.default_rng(11)
    task = _RecordingHadamardTask(8)
    initial = task.random_solution(rng)

    with pytest.raises(ValueError, match="positive integer"):
        optimizer_type().run(initial, task, budget=budget, rng=rng)

    assert task.evaluations == []


def test_steepest_descent_does_not_apply_move_without_budget_to_score_it() -> None:
    rng = np.random.default_rng(12)
    task = _RecordingHadamardTask(8)
    initial = task.random_solution(rng)

    run = SteepestDescentOptimizer().run(initial, task, budget=2, rng=rng)

    assert run.objective_evals == 1
    assert run.gradient_evals == 1
    assert run.n_evals == 2
    np.testing.assert_array_equal(run.solution, initial)
    assert run.solution_value == pytest.approx(task.evaluations[0][1])


def test_steepest_descent_never_returns_a_worse_scored_flip() -> None:
    rng = np.random.default_rng(0)
    task = _RecordingHadamardTask(4)
    initial = task.random_solution(rng)

    run = SteepestDescentOptimizer().run(initial, task, budget=8, rng=rng)

    assert run.solution_value == pytest.approx(max(run.values))
    assert run.solution_value >= run.values[0]


@pytest.mark.parametrize(
    "optimizer",
    [RowOperationOptimizer(), AnnealRowOptimizer()],
    ids=["row-ops", "anneal-row"],
)
def test_row_optimizers_propose_non_symmetric_moves(
    optimizer: InnerOptimizer,
) -> None:
    rng = np.random.default_rng(23)
    task = _RecordingHadamardTask(8)
    initial = task.random_solution(rng)
    initial_gram = initial @ initial.T

    optimizer.run(initial, task, budget=4, rng=rng)

    assert any(
        not np.array_equal(candidate @ candidate.T, initial_gram)
        for candidate, _ in task.evaluations[1:]
    )


def test_exact_sylvester_construction_stops_after_one_score() -> None:
    rng = np.random.default_rng(24)
    task = _RecordingHadamardTask(8)
    initial = task.random_solution(rng)

    run = SylvesterStartOptimizer().run(initial, task, budget=8, rng=rng)

    assert run.n_evals == 1
    assert run.values == pytest.approx([0.0])
    assert run.solution_value == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("optimizer", "n", "budget"),
    [
        pytest.param(PaleyStartOptimizer(), 12, 1, id="paley-construction"),
        pytest.param(TensorProductOptimizer(), 16, 2, id="tensor-candidates"),
    ],
)
def test_nested_hadamard_construction_loops_stop_before_budget_overshoot(
    optimizer: InnerOptimizer,
    n: int,
    budget: int,
) -> None:
    rng = np.random.default_rng(13)
    task = _NeverPerfectHadamardTask(n)
    initial = task.random_solution(rng)

    run = optimizer.run(initial, task, budget=budget, rng=rng)

    assert run.n_evals == budget
    assert run.objective_evals == len(task.evaluations) == budget
    assert any(
        np.array_equal(run.solution, scored_solution)
        and run.solution_value == pytest.approx(scored_value)
        for scored_solution, scored_value in task.evaluations
    )
