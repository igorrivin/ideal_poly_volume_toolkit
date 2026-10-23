"""Exact work-accounting tests for the cage inner optimizers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import ideal_poly_volume_toolkit.rl.cage.task as cage_task
from ideal_poly_volume_toolkit.rl.cage.task import (
    AnnealLogDetOptimizer,
    AnnealSwapOptimizer,
    CubicGirthTask,
    HybridGirthLogDetOptimizer,
    KnownCageOptimizer,
    MultiSwapOptimizer,
    RandomSwapOptimizer,
    ShortCycleBreaker,
    SpanningTreeOptimizer,
    compute_girth,
    is_valid_cubic,
    random_cubic_graph,
)
from ideal_poly_volume_toolkit.rl.generic import InnerOptimizer, OptimizerRun


class _RecordingCubicGirthTask(CubicGirthTask):
    """Record every value actually returned by the canonical objective."""

    def __init__(self, N: int):
        super().__init__(N)
        self.returned_values: list[float] = []

    def evaluate(self, adj: np.ndarray) -> float:
        value = super().evaluate(adj)
        self.returned_values.append(value)
        return value


class _ScriptedCubicGirthTask(CubicGirthTask):
    """Return distinct trial scores so incumbents cannot masquerade as calls."""

    def __init__(self, values: list[float]):
        super().__init__(10)
        self._remaining = iter(values)
        self.returned_values: list[float] = []

    def evaluate(self, adj: np.ndarray) -> float:
        value = float(next(self._remaining))
        self.returned_values.append(value)
        return value


OptimizerFactory = Callable[[], InnerOptimizer]


def _complete_graph_four(offset: int = 0) -> np.ndarray:
    return (
        np.array(
            [
                [1, 2, 3],
                [0, 2, 3],
                [0, 1, 3],
                [0, 1, 2],
            ],
            dtype=np.int32,
        )
        + offset
    )


def _brute_cycle_count(adj: np.ndarray, length: int) -> int:
    """Enumerate simple cycles of one length for small-graph test oracles."""
    count = 0
    for start in range(len(adj)):
        visited = {start}
        path = [start]

        def visit(vertex: int) -> None:
            nonlocal count
            if len(path) == length:
                if start in adj[vertex]:
                    count += 1
                return

            for neighbor_raw in adj[vertex]:
                neighbor = int(neighbor_raw)
                # Requiring start to be the minimum vertex counts each cycle
                # from one start; the two traversal directions remain.
                if neighbor <= start or neighbor in visited:
                    continue
                visited.add(neighbor)
                path.append(neighbor)
                visit(neighbor)
                path.pop()
                visited.remove(neighbor)

        visit(start)
    return count // 2


@pytest.mark.parametrize(
    ("factory", "uses_log_tree_proxy"),
    [
        (RandomSwapOptimizer, False),
        (ShortCycleBreaker, False),
        (AnnealSwapOptimizer, False),
        (lambda: MultiSwapOptimizer(k=3), False),
        (SpanningTreeOptimizer, True),
        (AnnealLogDetOptimizer, True),
        (HybridGirthLogDetOptimizer, True),
        (KnownCageOptimizer, False),
    ],
    ids=[
        "random-swap",
        "short-cycle-breaker",
        "anneal-swap",
        "multi-swap",
        "spanning-tree",
        "anneal-logdet",
        "hybrid",
        "known-cage",
    ],
)
def test_every_cage_optimizer_reports_exact_work(
    factory: OptimizerFactory,
    uses_log_tree_proxy: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = factory()
    task = _RecordingCubicGirthTask(10)
    start = random_cubic_graph(10, np.random.default_rng(101))
    surrogate_values: list[float] = []

    if uses_log_tree_proxy:
        original_proxy = optimizer._log_spanning_trees

        def recording_proxy(adj: np.ndarray) -> float:
            value = float(original_proxy(adj))
            surrogate_values.append(value)
            return value

        monkeypatch.setattr(optimizer, "_log_spanning_trees", recording_proxy)

    run = optimizer.run(
        start,
        task,
        budget=4,
        rng=np.random.default_rng(202),
    )

    assert isinstance(run, OptimizerRun)
    assert run.values == task.returned_values
    assert run.objective_evals == len(task.returned_values)
    assert run.surrogate_evals == len(surrogate_values)
    assert run.gradient_evals == 0
    assert run.n_evals == len(task.returned_values) + len(surrogate_values)
    assert 1 <= run.n_evals <= 4
    assert run.solution_value is not None


def test_random_swap_records_trials_instead_of_repeating_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = random_cubic_graph(10, np.random.default_rng(303))
    task = _ScriptedCubicGirthTask([10.0, 1.0, 2.0, 3.0])

    # Make every proposed swap evaluable while leaving the graph unchanged.
    monkeypatch.setattr(
        cage_task,
        "edge_swap",
        lambda adj, e1, e2, mode=0: adj.copy(),
    )

    run = RandomSwapOptimizer().run(
        start,
        task,
        budget=3,
        rng=np.random.default_rng(404),
    )

    assert run.values == [10.0, 1.0, 2.0]
    assert run.objective_evals == 3
    assert np.array_equal(run.solution, start)


@pytest.mark.parametrize(
    "factory",
    [
        RandomSwapOptimizer,
        ShortCycleBreaker,
        AnnealSwapOptimizer,
        lambda: MultiSwapOptimizer(k=3),
        SpanningTreeOptimizer,
        AnnealLogDetOptimizer,
        HybridGirthLogDetOptimizer,
        KnownCageOptimizer,
    ],
)
def test_one_unit_budget_returns_a_scored_input_without_overshoot(
    factory: OptimizerFactory,
) -> None:
    optimizer = factory()
    task = _RecordingCubicGirthTask(10)
    start = random_cubic_graph(10, np.random.default_rng(707))

    run = optimizer.run(start, task, budget=1, rng=np.random.default_rng(808))

    assert run.n_evals == 1
    assert run.objective_evals == 1
    assert run.surrogate_evals == 0
    assert run.solution_value == run.values[0]


def test_known_cage_does_not_pad_values_to_the_requested_budget() -> None:
    task = _RecordingCubicGirthTask(18)
    start = random_cubic_graph(18, np.random.default_rng(505))

    run = KnownCageOptimizer().run(
        start,
        task,
        budget=20,
        rng=np.random.default_rng(606),
    )

    assert run.values == task.returned_values
    assert len(run.values) == 1
    assert run.objective_evals == 1
    assert run.n_evals == 1


@pytest.mark.parametrize(
    "optimizer",
    [
        SpanningTreeOptimizer(),
        AnnealLogDetOptimizer(),
        HybridGirthLogDetOptimizer(),
    ],
    ids=["spanning-tree", "anneal-logdet", "hybrid"],
)
def test_log_tree_proxy_rejects_disconnected_cubic_graphs(
    optimizer: InnerOptimizer,
) -> None:
    connected = _complete_graph_four()
    disconnected = np.vstack([_complete_graph_four(), _complete_graph_four(offset=4)])

    assert optimizer._log_spanning_trees(connected) == pytest.approx(np.log(16.0))
    assert np.isneginf(optimizer._log_spanning_trees(disconnected))


def test_four_vertex_fallback_is_k4() -> None:
    fallback = cage_task._fallback_cubic(4)

    assert is_valid_cubic(fallback)
    assert compute_girth(fallback) == 3
    np.testing.assert_array_equal(
        np.sort(fallback, axis=1),
        _complete_graph_four(),
    )


@pytest.mark.parametrize(
    ("graph_name", "girth", "cycle_count"),
    [
        ("petersen", 5, 12),
        ("heawood", 6, 28),
    ],
)
def test_known_cages_have_exact_shortest_cycle_counts(
    graph_name: str,
    girth: int,
    cycle_count: int,
) -> None:
    optimizer = KnownCageOptimizer()
    adj = getattr(optimizer, f"_{graph_name}")()
    task = CubicGirthTask(len(adj))

    assert compute_girth(adj) == girth
    assert task._count_short_cycles(adj, girth) == cycle_count
    assert task.evaluate(adj) == pytest.approx(girth - 0.01 * cycle_count / len(adj))


@pytest.mark.parametrize(
    ("N", "seed"),
    [(6, 11), (8, 12), (10, 13), (12, 14), (14, 15)],
)
def test_shortest_cycle_count_matches_brute_force(N: int, seed: int) -> None:
    adj = random_cubic_graph(N, np.random.default_rng(seed))
    task = CubicGirthTask(N)
    girth = compute_girth(adj)

    assert task._count_short_cycles(adj, girth) == _brute_cycle_count(adj, girth)


@pytest.mark.parametrize(
    "adj",
    [
        _complete_graph_four(),
        KnownCageOptimizer()._petersen(),
        KnownCageOptimizer()._heawood(),
        random_cubic_graph(18, np.random.default_rng(16)),
        random_cubic_graph(30, np.random.default_rng(17)),
    ],
    ids=["k4", "petersen", "heawood", "random-18", "random-30"],
)
def test_shortest_cycle_reconstruction_is_simple_and_edge_connected(
    adj: np.ndarray,
) -> None:
    cycle = ShortCycleBreaker()._find_shortest_cycle(adj)

    assert cycle is not None
    assert len(cycle) == compute_girth(adj)
    assert len(set(cycle)) == len(cycle)
    assert all(
        cycle[(index + 1) % len(cycle)] in adj[vertex]
        for index, vertex in enumerate(cycle)
    )


def test_spanning_tree_optimizer_preserves_best_canonical_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = random_cubic_graph(10, np.random.default_rng(18))
    task = _ScriptedCubicGirthTask([10.0, 1.0])
    optimizer = SpanningTreeOptimizer()
    surrogate_values = iter([0.0, 1.0])

    monkeypatch.setattr(
        optimizer,
        "_log_spanning_trees",
        lambda adj: next(surrogate_values),
    )
    monkeypatch.setattr(
        cage_task,
        "edge_swap",
        lambda adj, e1, e2, mode=0: adj.copy(),
    )

    run = optimizer.run(start, task, budget=4, rng=np.random.default_rng(19))

    assert run.values == [10.0, 1.0]
    assert run.solution_value == max(run.values) == 10.0
    assert run.objective_evals == 2
    assert run.surrogate_evals == 2
    assert run.n_evals == 4
