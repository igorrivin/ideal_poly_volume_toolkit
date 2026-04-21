"""
Problem-agnostic InnerOptimizer implementations.

These classes work for any OptimizationTask that exposes the optional
`to_vector` / `from_vector` / `project` hooks (CMAESOptimizer,
ScipyMinimizeOptimizer) or just the basic interface
(RandomRestartOptimizer).
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from .generic import InnerOptimizer, OptimizationTask


def _probe_vector_support(task: OptimizationTask) -> None:
    """Raise if the task does not expose a real-vector embedding."""
    sol = task.random_solution(np.random.default_rng(0))
    try:
        vec = task.to_vector(sol)
        _ = task.from_vector(vec)
        _ = task.project(vec)
    except NotImplementedError as exc:
        raise TypeError(
            f"Optimizer requires a vector-embeddable task; "
            f"{type(task).__name__} does not override to_vector / from_vector."
        ) from exc


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
            budget: int, rng: np.random.Generator) -> Tuple[Any, List[float]]:
        best_sol = task.copy_solution(solution)
        best_val = task.evaluate(solution)
        values = [best_val]
        for _ in range(budget - 1):
            s = task.random_solution(rng)
            v = task.evaluate(s)
            values.append(v)
            if v > best_val:
                best_val = v
                best_sol = task.copy_solution(s)
        return best_sol, values


# =====================================================================
# CMA-ES on task.evaluate or task.evaluate_surrogate
# =====================================================================

class CMAESOptimizer(InnerOptimizer):
    """CMA-ES via the `cma` package.

    Requires the task to implement `to_vector`, `from_vector`, `project`.
    Maximises either `task.evaluate` or `task.evaluate_surrogate` depending
    on the `use_surrogate` flag.  The returned values list is scored by
    whichever objective the optimizer used; the env will re-score the
    returned best solution via `task.evaluate` for its global best.
    """

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

    def _objective(self, task: OptimizationTask):
        return task.evaluate_surrogate if self.use_surrogate else task.evaluate

    def run(self, solution: Any, task: OptimizationTask,
            budget: int, rng: np.random.Generator) -> Tuple[Any, List[float]]:
        _probe_vector_support(task)
        objective = self._objective(task)

        x0 = task.to_vector(solution).astype(float)
        seed = int(rng.integers(0, 2**31 - 1))
        es = self._cma.CMAEvolutionStrategy(
            x0, self.sigma,
            {"maxfevals": budget, "verbose": -9, "seed": seed},
        )

        def neg_obj(vec: np.ndarray) -> float:
            vec = task.project(vec)
            sol = task.from_vector(vec)
            try:
                return -float(objective(sol))
            except Exception:
                return 0.0

        values: List[float] = []
        while not es.stop():
            asked = es.ask()
            fitnesses = [neg_obj(x) for x in asked]
            es.tell(asked, fitnesses)
            values.extend(-f for f in fitnesses)

        best_vec = task.project(np.asarray(es.result.xbest, dtype=float))
        best_sol = task.from_vector(best_vec)
        return best_sol, values


# =====================================================================
# scipy.optimize.minimize wrapper (Powell / L-BFGS-B / Nelder-Mead / ...)
# =====================================================================

class ScipyMinimizeOptimizer(InnerOptimizer):
    """Thin wrapper around `scipy.optimize.minimize`.

    Suitable for tasks with a vector embedding.  `method` picks the scipy
    algorithm; `use_surrogate` selects the objective.  `budget` caps
    `maxfev` / `maxiter`.
    """

    def __init__(self, *, method: str = "Powell", use_surrogate: bool = False,
                 name: str | None = None):
        self.method = method
        self.use_surrogate = bool(use_surrogate)
        self._name = name or f"scipy_{method.lower()}"

    @property
    def name(self) -> str:
        return self._name

    def run(self, solution: Any, task: OptimizationTask,
            budget: int, rng: np.random.Generator) -> Tuple[Any, List[float]]:
        from scipy.optimize import minimize
        _probe_vector_support(task)

        objective = (task.evaluate_surrogate if self.use_surrogate
                     else task.evaluate)
        x0 = task.to_vector(solution).astype(float)

        values: List[float] = []

        def neg_obj(vec: np.ndarray) -> float:
            vec = task.project(vec)
            try:
                v = float(objective(task.from_vector(vec)))
            except Exception:
                v = 0.0
            values.append(v)
            return -v

        options: dict = {"maxiter": budget, "maxfev": budget}
        res = minimize(neg_obj, x0, method=self.method, options=options)
        best_vec = task.project(np.asarray(res.x, dtype=float))
        best_sol = task.from_vector(best_vec)
        return best_sol, values
