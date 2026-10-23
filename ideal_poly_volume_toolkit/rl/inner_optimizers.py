"""
Problem-agnostic InnerOptimizer implementations.

These classes work for any OptimizationTask that exposes the optional
`to_vector` / `from_vector` / `project` hooks (CMAESOptimizer,
ScipyMinimizeOptimizer) or just the basic interface
(RandomRestartOptimizer).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .generic import InnerOptimizer, OptimizationTask, OptimizerRun


def _validated_budget(budget: int) -> int:
    """Return a positive integer total-work cap."""
    budget = int(budget)
    if budget < 1:
        raise ValueError("optimizer budget must be at least 1")
    return budget


def _has_vector_embedding(task: OptimizationTask) -> bool:
    """Check vector support without sampling or triangulating a solution."""
    task_type = type(task)
    return (
        task_type.to_vector is not OptimizationTask.to_vector
        and task_type.from_vector is not OptimizationTask.from_vector
    )


def _probe_vector_support(task: OptimizationTask, solution: Any) -> None:
    """Raise if the task does not expose a real-vector embedding."""
    if not _has_vector_embedding(task):
        raise TypeError(
            f"Optimizer requires a vector-embeddable task; "
            f"{type(task).__name__} does not override to_vector / from_vector."
        )
    try:
        vec = task.to_vector(solution)
        _ = task.from_vector(vec)
        _ = task.project(vec)
    except NotImplementedError as exc:
        raise TypeError(
            f"Optimizer requires a vector-embeddable task; "
            f"{type(task).__name__} does not override to_vector / from_vector."
        ) from exc


def _has_distinct_surrogate(task: OptimizationTask) -> bool:
    """Whether the task declares a genuinely distinct proxy objective."""
    return bool(task.has_distinct_surrogate())


class _WorkBudgetExhausted(RuntimeError):
    """Private sentinel used to stop scipy at the hard oracle-call cap."""


# =====================================================================
# Random restart
# =====================================================================

class RandomRestartOptimizer(InnerOptimizer):
    """Draw `budget` random solutions from `task.random_solution`, keep the best."""

    def __init__(self, *, name: str = "random_restart"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def run(self, solution: Any, task: OptimizationTask,
            budget: int, rng: np.random.Generator) -> OptimizerRun:
        budget = _validated_budget(budget)
        best_sol = task.copy_solution(solution)
        best_val = float(task.evaluate(solution))
        values = [best_val]
        for _ in range(budget - 1):
            s = task.random_solution(rng)
            v = float(task.evaluate(s))
            values.append(v)
            if v > best_val:
                best_val = v
                best_sol = task.copy_solution(s)
        return OptimizerRun(
            solution=best_sol,
            values=values,
            solution_value=best_val,
            objective_evals=len(values),
        )


# =====================================================================
# CMA-ES on task.evaluate or task.evaluate_surrogate
# =====================================================================

class CMAESOptimizer(InnerOptimizer):
    """CMA-ES via the `cma` package.

    Requires the task to implement `to_vector`, `from_vector`, `project`.
    Maximises either `task.evaluate` or a genuinely distinct
    `task.evaluate_surrogate`, depending on the `use_surrogate` flag.
    ``budget`` is a strict cap on all objective and surrogate calls.  A
    surrogate run reserves one call for a canonical score of its returned
    solution; a canonical-objective run scores its starting point first.
    Consequently every successful run returns a counted ``solution_value``.

    Register multiple instances with different sigmas to give the RL
    controller a choice of exploration scales (e.g. an `explore` variant
    with sigma=0.4 and a `refine` variant with sigma=0.1).
    """

    default_budget = 12

    def __init__(self, *, sigma: float = 0.2, use_surrogate: bool = False,
                 name: str | None = None):
        try:
            import cma
        except ImportError as exc:
            raise ImportError(
                "CMAESOptimizer requires the `cma` package"
            ) from exc
        self._cma = cma
        self.sigma = float(sigma)
        self.use_surrogate = bool(use_surrogate)
        self._name = name or ("cmaes_surrogate" if use_surrogate else "cmaes")

    @property
    def name(self) -> str:
        return self._name

    def supports(self, task: OptimizationTask) -> bool:
        return _has_vector_embedding(task)

    def run(self, solution: Any, task: OptimizationTask,
            budget: int, rng: np.random.Generator) -> OptimizerRun:
        budget = _validated_budget(budget)
        _probe_vector_support(task, solution)
        x0 = task.to_vector(solution).astype(float)
        use_surrogate = self.use_surrogate and _has_distinct_surrogate(task)
        objective = task.evaluate_surrogate if use_surrogate else task.evaluate

        canonical_values: list[float] = []
        calls = 0
        best_sol = task.copy_solution(solution)
        best_search_value = -np.inf

        # Canonical mode scores the incumbent up front.  This both provides a
        # valid fallback and makes the returned solution_value part of the
        # budget even when the remaining cap is too small for a CMA generation.
        if not use_surrogate:
            best_search_value = float(task.evaluate(best_sol))
            canonical_values.append(best_search_value)
            calls = 1

        # Surrogate mode must leave one unit for the canonical score below.
        search_cap = budget - 1 if use_surrogate else budget
        remaining = search_cap - calls

        def neg_obj(vec: np.ndarray) -> float:
            nonlocal calls, best_sol, best_search_value
            try:
                vec = task.project(vec)
                sol = task.from_vector(vec)
                calls += 1
                value = float(objective(sol))
            except Exception:
                return np.finfo(np.float64).max
            if not use_surrogate:
                canonical_values.append(value)
            if np.isfinite(value) and value > best_search_value:
                best_search_value = value
                best_sol = task.copy_solution(sol)
            return -value

        # CMA-ES needs at least two members.  Use complete generations only;
        # a partial final generation would violate CMA's selection assumptions.
        if remaining >= 2:
            default_popsize = 4 + int(3 * np.log(max(len(x0), 1)))
            popsize = min(default_popsize, remaining)
            seed = int(rng.integers(0, 2**31 - 1))
            es = self._cma.CMAEvolutionStrategy(
                x0,
                self.sigma,
                {
                    "maxfevals": remaining,
                    "popsize": popsize,
                    "verbose": -9,
                    "seed": seed,
                },
            )
            while not es.stop() and calls + popsize <= search_cap:
                asked = es.ask()
                fitnesses = [neg_obj(x) for x in asked]
                es.tell(asked, fitnesses)

        if use_surrogate:
            canonical_value = float(task.evaluate(best_sol))
            canonical_values.append(canonical_value)
            objective_evals = 1
            surrogate_evals = calls
        else:
            canonical_value = best_search_value
            objective_evals = calls
            surrogate_evals = 0

        return OptimizerRun(
            solution=best_sol,
            values=canonical_values,
            solution_value=canonical_value,
            objective_evals=objective_evals,
            surrogate_evals=surrogate_evals,
        )


# =====================================================================
# scipy.optimize.minimize wrapper (Powell / L-BFGS-B / Nelder-Mead / ...)
# =====================================================================

class ScipyMinimizeOptimizer(InnerOptimizer):
    """Thin wrapper around `scipy.optimize.minimize`.

    Suitable for tasks with a vector embedding.  `method` picks the scipy
    algorithm; `use_surrogate` selects the objective.  ``budget`` is a hard
    total-work cap, enforced in the objective wrapper rather than relying on
    scipy's method-dependent ``maxfev`` interpretation.
    """

    def __init__(self, *, method: str = "Powell", use_surrogate: bool = False,
                 name: str | None = None):
        self.method = method
        self.use_surrogate = bool(use_surrogate)
        self._name = name or f"scipy_{method.lower()}"

    @property
    def name(self) -> str:
        return self._name

    def supports(self, task: OptimizationTask) -> bool:
        return _has_vector_embedding(task)

    def run(self, solution: Any, task: OptimizationTask,
            budget: int, rng: np.random.Generator) -> OptimizerRun:
        from scipy.optimize import minimize
        budget = _validated_budget(budget)
        _probe_vector_support(task, solution)

        use_surrogate = self.use_surrogate and _has_distinct_surrogate(task)
        objective = task.evaluate_surrogate if use_surrogate else task.evaluate
        x0 = task.to_vector(solution).astype(float)

        canonical_values: list[float] = []
        calls = 0
        best_sol = task.copy_solution(solution)
        best_search_value = -np.inf

        if not use_surrogate:
            best_search_value = float(task.evaluate(best_sol))
            canonical_values.append(best_search_value)
            calls = 1

        search_cap = budget - 1 if use_surrogate else budget

        def neg_obj(vec: np.ndarray) -> float:
            nonlocal calls, best_sol, best_search_value
            if calls >= search_cap:
                raise _WorkBudgetExhausted
            try:
                vec = task.project(vec)
                candidate = task.from_vector(vec)
                calls += 1
                value = float(objective(candidate))
            except Exception:
                return np.finfo(np.float64).max
            if not use_surrogate:
                canonical_values.append(value)
            if np.isfinite(value) and value > best_search_value:
                best_search_value = value
                best_sol = task.copy_solution(candidate)
            return -value

        if calls < search_cap:
            # These remain useful method hints, but the wrapper above is the
            # authoritative cap (not every scipy method honours maxfev alike).
            options: dict = {"maxiter": search_cap, "maxfev": search_cap}
            try:
                minimize(neg_obj, x0, method=self.method, options=options)
            except _WorkBudgetExhausted:
                pass

        if use_surrogate:
            canonical_value = float(task.evaluate(best_sol))
            canonical_values.append(canonical_value)
            objective_evals = 1
            surrogate_evals = calls
        else:
            canonical_value = best_search_value
            objective_evals = calls
            surrogate_evals = 0

        return OptimizerRun(
            solution=best_sol,
            values=canonical_values,
            solution_value=canonical_value,
            objective_evals=objective_evals,
            surrogate_evals=surrogate_evals,
        )
