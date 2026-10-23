"""
Generic hierarchical RL optimizer for black-box optimization.

The framework is agnostic to the inner optimization problem.
Users provide:
  - A Task class (defines objective, random solutions, perturbations)
  - Inner optimizers (greedy local search, random perturbation, etc.)

The RL controller learns to orchestrate these across a search tree.

This is the generalization of rl_optimizer.py from ideal polytopes
to arbitrary optimization problems.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# =====================================================================
# Abstract interfaces
# =====================================================================

class OptimizationTask(ABC):
    """Abstract base for an optimization problem instance."""

    @abstractmethod
    def random_solution(self, rng: np.random.Generator) -> Any:
        """Generate a random feasible solution."""

    @abstractmethod
    def evaluate(self, solution: Any) -> float:
        """Evaluate the objective (HIGHER = better)."""

    @abstractmethod
    def perturb(self, solution: Any, radius: float,
                rng: np.random.Generator) -> Any:
        """Perturb a solution by approximately `radius`."""

    @abstractmethod
    def metadata(self) -> np.ndarray:
        """Fixed-size feature vector describing this task instance."""

    @abstractmethod
    def copy_solution(self, solution: Any) -> Any:
        """Deep copy a solution."""

    # ---- Optional hooks for optimizers that need a real-vector embedding ---
    # Tasks whose solutions are (or can be) numpy arrays (needed by CMA-ES,
    # scipy minimize, etc.) override the three methods below.  Combinatorial
    # tasks leave them as-is; continuous optimizers refuse such tasks at
    # __init__ time via a hasattr / try-call probe.

    def to_vector(self, solution: Any) -> np.ndarray:
        """Flatten a solution to a 1-D numpy array."""
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a vector embedding."
        )

    def from_vector(self, vec: np.ndarray) -> Any:
        """Reconstruct a solution from a flat vector (inverse of to_vector)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a vector embedding."
        )

    def project(self, vec: np.ndarray) -> np.ndarray:
        """Project a flat vector back to the feasible set (e.g. normalize
        rows to lie on a sphere, clip to a box).  Default: identity."""
        return vec

    # ---- Optional fast surrogate -------------------------------------------

    def evaluate_surrogate(self, solution: Any) -> float:
        """Cheap proxy for `evaluate`, used by surrogate-aware optimizers
        when the full objective is expensive.  Default: delegates to
        `evaluate`."""
        return self.evaluate(solution)

    def has_distinct_surrogate(self) -> bool:
        """Whether surrogate calls differ from canonical objective calls."""
        return type(self).evaluate_surrogate is not OptimizationTask.evaluate_surrogate


class InnerOptimizer(ABC):
    """Abstract base for a local optimizer."""

    # Per-class preferred evaluation budget.  The env uses this when the RL
    # controller dispatches an inner-optimizer action.  Override in
    # subclasses to request more or fewer steps; the env still passes this
    # via the `budget` argument so implementations may also cap themselves.
    default_budget: int = 8

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name for this optimizer."""

    @abstractmethod
    def run(self, solution: Any, task: OptimizationTask,
            budget: int, rng: np.random.Generator):
        """Run local optimization within a strict positive work budget.

        Objective, surrogate, and gradient/closure evaluations all consume one
        unit, including the canonical score associated with the returned
        solution. Implementations may under-use indivisible residual budget but
        must never report more than ``budget`` total units.

        Prefer returning :class:`OptimizerRun`.  A legacy
        ``(best_solution, values)`` tuple remains accepted by the environment;
        in that form each value is assumed to represent one canonical
        objective evaluation.
        """

    def supports(self, task: OptimizationTask) -> bool:
        """Whether this optimizer is applicable to ``task``."""
        return True


@dataclass
class OptimizerRun:
    """Result and explicit work accounting for one inner-optimizer action.

    ``values`` contains only values of ``task.evaluate``.  Surrogate values
    must not be mixed into node statistics; their calls are tracked separately
    in ``surrogate_evals``.  ``gradient_evals`` counts closures or gradient-only
    oracle calls whose returned scalar is not the canonical task objective.
    ``solution_value`` may identify the already-computed canonical value of the
    returned solution; when omitted, the environment performs one final score.

    Iteration yields ``(solution, values)`` for compatibility with callers that
    historically unpacked the two-tuple returned by ``InnerOptimizer.run``.
    """

    solution: Any
    values: List[float] = field(default_factory=list)
    solution_value: Optional[float] = None
    objective_evals: int = 0
    surrogate_evals: int = 0
    gradient_evals: int = 0
    wall_time: float = 0.0

    def __post_init__(self) -> None:
        """Reject accounting values that could create or erase work."""
        for name in ("objective_evals", "surrogate_evals", "gradient_evals"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be a nonnegative integer")
            if value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            setattr(self, name, int(value))

    @property
    def n_evals(self) -> int:
        return self.objective_evals + self.surrogate_evals + self.gradient_evals

    def __iter__(self):
        yield self.solution
        yield self.values

    def __len__(self):
        """Preserve the length of the historical two-tuple result."""
        return 2

    def __getitem__(self, index):
        """Preserve tuple indexing and slicing for legacy callers."""
        return (self.solution, self.values)[index]


class OptimizerRunError(RuntimeError):
    """Optimizer failure carrying exact work completed before the error."""

    def __init__(self, message: str, run: OptimizerRun):
        super().__init__(message)
        self.run = run


class TaskSampler(ABC):
    """Samples task instances for training."""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> OptimizationTask:
        """Sample a random task instance."""

    @abstractmethod
    def metadata_dim(self) -> int:
        """Dimension of the task metadata vector."""


# =====================================================================
# Search tree (generic)
# =====================================================================

NODE_FEATURE_DIM = 8
MAX_FRONTIER = 16

@dataclass(eq=False)
class SearchNode:
    """A node in the search tree."""
    center: Any                          # current best solution
    radius: float                        # search radius
    depth: int = 0
    parent: Optional['SearchNode'] = None
    children: List['SearchNode'] = field(default_factory=list)

    # Statistics
    n_evals: int = 0
    best_value: float = -np.inf
    best_solution: Any = None
    mean_value: float = 0.0
    var_value: float = 0.0
    recent_improvement: float = 0.0
    values_history: List[float] = field(default_factory=list)
    is_active: bool = True

    def features(self) -> np.ndarray:
        return np.array([
            self.depth,
            np.log1p(self.n_evals),
            self.best_value,
            self.mean_value,
            self.var_value,
            self.recent_improvement,
            self.radius,
            len(self.children),
        ], dtype=np.float32)

    def update_stats(self, values: List[float], *, n_evals: int | None = None):
        """Update running statistics from new evaluations.

        Does NOT touch `best_value` / `best_solution`; those are updated
        only by the caller after it compares `task.evaluate(best_sol)`
        against the current best.  Keeping the "advertised" best strictly
        in sync with the stored solution avoids a latent bug where
        `values` can claim an improvement that the best-solution storage
        never actually captures (caller's `best_val > self.best_value`
        check then sees the already-bumped value and fails to record the
        real solution).
        """
        self.values_history.extend(values)
        self.n_evals += len(values) if n_evals is None else int(n_evals)
        arr = np.array(self.values_history)
        self.mean_value = float(arr.mean())
        self.var_value = float(arr.var()) if len(arr) > 1 else 0.0


# =====================================================================
# Generic Gymnasium Environment
# =====================================================================

class GenericOptEnv(gym.Env):
    """
    Generic hierarchical RL optimization environment.

    Observation: flat vector of (MAX_FRONTIER * (NODE_FEATURE_DIM + task_meta_dim))
    Action: (node_index, operation_index)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        task_sampler: TaskSampler,
        inner_optimizers: List[InnerOptimizer],
        horizon: int = 50,
        eval_cost_weight: float = 0.001,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.task_sampler = task_sampler
        self.inner_optimizers = inner_optimizers
        self.horizon = horizon
        self.eval_cost_weight = eval_cost_weight
        self._rng = np.random.default_rng(seed)
        self._initial_seed = seed
        self._has_reset = False

        # Actions: one per inner optimizer + split + perturb + prune
        self.n_inner = len(inner_optimizers)
        # Operations: [opt_0, opt_1, ..., opt_k, sample, split, perturb, prune]
        self.OP_SAMPLE = self.n_inner
        self.OP_SPLIT = self.n_inner + 1
        self.OP_PERTURB = self.n_inner + 2
        self.OP_PRUNE = self.n_inner + 3
        self.n_ops = self.n_inner + 4

        self.op_names = ([opt.name for opt in inner_optimizers]
                         + ['sample', 'split', 'perturb', 'prune'])

        meta_dim = task_sampler.metadata_dim()
        obs_dim = MAX_FRONTIER * (NODE_FEATURE_DIM + meta_dim)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([MAX_FRONTIER, self.n_ops])
        self._meta_dim = meta_dim
        self.last_error: Optional[str] = None
        self.last_optimizer_metrics: Optional[dict] = None
        self.last_requested_node_index: Optional[int] = None
        self.last_effective_node_index: Optional[int] = None
        self.last_node_index_remapped = False
        self.node_index_remap_count = 0

    def reset(self, seed=None, options=None):
        effective_seed = seed
        if effective_seed is None and not self._has_reset:
            effective_seed = self._initial_seed
        super().reset(seed=effective_seed)
        self._rng = self.np_random
        self._has_reset = True

        self.task = self.task_sampler.sample(self._rng)
        self.task_meta = self.task.metadata()

        init_sol = self.task.random_solution(self._rng)
        init_val = self.task.evaluate(init_sol)

        self.root = SearchNode(
            center=self.task.copy_solution(init_sol),
            radius=1.0, depth=0,
        )
        self.root.best_value = init_val
        self.root.best_solution = self.task.copy_solution(init_sol)
        self.root.n_evals = 1
        self.root.values_history = [init_val]
        self.root.mean_value = float(init_val)

        self.frontier = [self.root]
        self.best_value = init_val
        self.best_solution = self.task.copy_solution(init_sol)
        self.total_evals = 1
        self.step_count = 0
        self.last_error = None
        self.last_optimizer_metrics = None
        self.last_requested_node_index = None
        self.last_effective_node_index = None
        self.last_node_index_remapped = False
        self.node_index_remap_count = 0

        return self._encode_obs(), self._info()

    def step(self, action):
        requested_node_idx = int(action[0])
        node_idx = requested_node_idx % len(self.frontier)
        self.last_requested_node_index = requested_node_idx
        self.last_effective_node_index = node_idx
        self.last_node_index_remapped = node_idx != requested_node_idx
        if self.last_node_index_remapped:
            self.node_index_remap_count += 1

        op_type = int(action[1])
        node = self.frontier[node_idx]

        old_best = self.best_value
        evals_used = 0
        node.recent_improvement = 0.0
        self.last_error = None
        self.last_optimizer_metrics = None

        try:
            if op_type < self.n_inner:
                evals_used = self._do_inner_opt(node, op_type)
            elif op_type == self.OP_SAMPLE:
                evals_used = self._do_sample(node, n=4)
            elif op_type == self.OP_SPLIT:
                evals_used = self._do_split(node)
            elif op_type == self.OP_PERTURB:
                evals_used = self._do_perturb(node)
            elif op_type == self.OP_PRUNE:
                self._do_prune(node)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            # Failed actions are not free.  Optimizers using OptimizerRun
            # report more precise counts before returning normally.
            evals_used = max(evals_used, 1)

        # Update global best
        for n in self.frontier:
            if n.best_value > self.best_value and n.best_solution is not None:
                self.best_value = n.best_value
                self.best_solution = self.task.copy_solution(n.best_solution)

        self.total_evals += evals_used
        self.step_count += 1

        improvement = max(0.0, self.best_value - old_best)
        reward = float(improvement - self.eval_cost_weight * evals_used)

        truncated = self.step_count >= self.horizon
        return self._encode_obs(), reward, False, truncated, self._info()

    def _do_inner_opt(self, node: SearchNode, opt_idx: int) -> int:
        opt = self.inner_optimizers[opt_idx]
        if not opt.supports(self.task):
            raise ValueError(
                f"optimizer {opt.name!r} does not support "
                f"{type(self.task).__name__}"
            )
        start = node.best_solution if node.best_solution is not None else node.center
        budget = getattr(opt, 'default_budget', 8)
        # Legacy heuristic: annealing optimizers get a bigger budget even if
        # the class forgot to override default_budget.
        if 'anneal' in opt.name and opt.default_budget == 8:
            budget = 50

        task = self.task
        objective_calls = 0
        surrogate_calls = 0
        original_evaluate = task.evaluate
        original_surrogate = task.evaluate_surrogate
        task_attributes = getattr(task, "__dict__", None)
        can_instrument = isinstance(task_attributes, dict)
        had_instance_evaluate = can_instrument and "evaluate" in task_attributes
        had_instance_surrogate = (
            can_instrument and "evaluate_surrogate" in task_attributes
        )
        saved_instance_evaluate = (
            task_attributes.get("evaluate") if can_instrument else None
        )
        saved_instance_surrogate = (
            task_attributes.get("evaluate_surrogate") if can_instrument else None
        )

        def counted_evaluate(solution):
            nonlocal objective_calls
            objective_calls += 1
            return original_evaluate(solution)

        surrogate_is_default = not task.has_distinct_surrogate()
        if can_instrument:
            task.evaluate = counted_evaluate
        if can_instrument and not surrogate_is_default:
            def counted_surrogate(solution):
                nonlocal surrogate_calls
                surrogate_calls += 1
                return original_surrogate(solution)

            task.evaluate_surrogate = counted_surrogate

        started = time.perf_counter()
        try:
            raw_result = opt.run(start, task, budget=budget, rng=self._rng)
        except OptimizerRunError as exc:
            elapsed = time.perf_counter() - started
            run = exc.run
            run.wall_time = elapsed
            objective_evals = max(run.objective_evals, objective_calls)
            surrogate_evals = max(run.surrogate_evals, surrogate_calls)
            evals_used = max(
                1,
                objective_evals + surrogate_evals + run.gradient_evals,
            )
            node.update_stats(run.values, n_evals=evals_used)
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_optimizer_metrics = {
                "name": opt.name,
                "objective_evals": objective_evals,
                "surrogate_evals": surrogate_evals,
                "gradient_evals": run.gradient_evals,
                "wall_time": elapsed,
            }
            return evals_used
        except Exception as exc:
            elapsed = time.perf_counter() - started
            evals_used = max(1, objective_calls + surrogate_calls)
            node.update_stats([], n_evals=evals_used)
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_optimizer_metrics = {
                'name': opt.name,
                'objective_evals': objective_calls,
                'surrogate_evals': surrogate_calls,
                'gradient_evals': 0,
                'wall_time': elapsed,
            }
            return evals_used
        finally:
            if can_instrument and had_instance_evaluate:
                task.evaluate = saved_instance_evaluate
            elif can_instrument:
                del task.evaluate
            if can_instrument and not surrogate_is_default:
                if had_instance_surrogate:
                    task.evaluate_surrogate = saved_instance_surrogate
                else:
                    del task.evaluate_surrogate

        elapsed = time.perf_counter() - started
        if isinstance(raw_result, OptimizerRun):
            run = raw_result
        else:
            best_sol, values = raw_result
            run = OptimizerRun(
                solution=best_sol,
                values=list(values),
                objective_evals=len(values),
            )
        run.wall_time = elapsed
        best_sol = run.solution
        old_best = node.best_value
        objective_evals = max(run.objective_evals, objective_calls)
        surrogate_evals = max(run.surrogate_evals, surrogate_calls)
        reported_work = objective_evals + surrogate_evals + run.gradient_evals
        final_eval = 0
        if run.solution_value is not None and objective_evals < 1:
            evals_used = max(1, reported_work)
            node.update_stats(run.values, n_evals=evals_used)
            self.last_error = (
                f"optimizer {opt.name!r} returned an uncounted canonical "
                "solution_value"
            )
            self.last_optimizer_metrics = {
                "name": opt.name,
                "objective_evals": objective_evals,
                "surrogate_evals": surrogate_evals,
                "gradient_evals": run.gradient_evals,
                "wall_time": run.wall_time,
            }
            return evals_used
        if run.solution_value is None:
            if reported_work >= budget:
                node.update_stats(run.values, n_evals=reported_work)
                self.last_error = (
                    f"optimizer {opt.name!r} exhausted its work budget without "
                    "scoring the returned solution"
                )
                self.last_optimizer_metrics = {
                    'name': opt.name,
                    'objective_evals': objective_evals,
                    'surrogate_evals': surrogate_evals,
                    'gradient_evals': run.gradient_evals,
                    'wall_time': run.wall_time,
                }
                return reported_work
            best_val = task.evaluate(best_sol)
            final_eval = 1
            values = [*run.values, best_val]
        else:
            best_val = float(run.solution_value)
            values = run.values
        evals_used = reported_work + final_eval
        if evals_used > budget:
            node.update_stats(values, n_evals=evals_used)
            self.last_error = (
                f"optimizer {opt.name!r} exceeded work budget {budget} "
                f"with {evals_used} evaluations"
            )
            self.last_optimizer_metrics = {
                'name': opt.name,
                'objective_evals': objective_evals + final_eval,
                'surrogate_evals': surrogate_evals,
                'gradient_evals': run.gradient_evals,
                'wall_time': run.wall_time,
            }
            return evals_used
        node.update_stats(values, n_evals=evals_used)
        if best_val > node.best_value:
            node.best_value = best_val
            node.best_solution = self.task.copy_solution(best_sol)
            node.center = self.task.copy_solution(best_sol)
        node.recent_improvement = float(max(0.0, node.best_value - old_best))
        self.last_optimizer_metrics = {
            'name': opt.name,
            'objective_evals': objective_evals + final_eval,
            'surrogate_evals': surrogate_evals,
            'gradient_evals': run.gradient_evals,
            'wall_time': run.wall_time,
        }
        return evals_used

    def _do_sample(self, node: SearchNode, n: int = 4) -> int:
        old_best = node.best_value
        values = []
        attempts = 0
        for _ in range(n):
            base = (
                node.best_solution
                if node.best_solution is not None
                else node.center
            )
            sol = self.task.perturb(
                base,
                node.radius * 0.3, self._rng
            )
            attempts += 1
            try:
                v = self.task.evaluate(sol)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                break
            values.append(v)
            if v > node.best_value:
                node.best_value = v
                node.best_solution = self.task.copy_solution(sol)
                node.center = self.task.copy_solution(sol)
        node.update_stats(values, n_evals=attempts)
        node.recent_improvement = float(max(0.0, node.best_value - old_best))
        return attempts

    def _do_split(self, node: SearchNode) -> int:
        if len(self.frontier) >= MAX_FRONTIER - 1:
            return 0
        r = node.radius * 0.6
        c1 = self.task.perturb(node.center, r * 0.3, self._rng)
        c2 = self.task.perturb(node.center, r * 0.3, self._rng)
        child1 = SearchNode(c1, r, depth=node.depth + 1, parent=node)
        child2 = SearchNode(c2, r, depth=node.depth + 1, parent=node)
        attempts = 0
        values = []
        for child in (child1, child2):
            attempts += 1
            try:
                value = self.task.evaluate(child.center)
            except Exception as exc:
                node.update_stats(values, n_evals=attempts)
                self.last_error = f"{type(exc).__name__}: {exc}"
                return attempts
            values.append(value)
            child.best_solution = self.task.copy_solution(child.center)
            child.best_value = value
            child.n_evals = 1
            child.values_history = [value]
            child.mean_value = value
        node.children = [child1, child2]
        node.is_active = False
        if node in self.frontier:
            self.frontier.remove(node)
        self.frontier.extend([child1, child2])
        return attempts

    def _do_perturb(self, node: SearchNode) -> int:
        if node.best_solution is None:
            return 0
        old_best = node.best_value
        sol = self.task.perturb(node.best_solution, 0.5, self._rng)
        try:
            v = self.task.evaluate(sol)
        except Exception as exc:
            node.update_stats([], n_evals=1)
            self.last_error = f"{type(exc).__name__}: {exc}"
            return 1
        node.update_stats([v])
        if v > node.best_value:
            node.best_value = v
            node.best_solution = self.task.copy_solution(sol)
            node.center = self.task.copy_solution(sol)
        node.recent_improvement = float(max(0.0, node.best_value - old_best))
        return 1

    def _do_prune(self, node: SearchNode):
        if len(self.frontier) > 1 and node in self.frontier:
            self.frontier.remove(node)
            node.is_active = False

    def _encode_obs(self) -> np.ndarray:
        per_node = NODE_FEATURE_DIM + self._meta_dim
        obs = np.zeros(MAX_FRONTIER * per_node, dtype=np.float32)
        for i, node in enumerate(self.frontier[:MAX_FRONTIER]):
            offset = i * per_node
            obs[offset:offset + NODE_FEATURE_DIM] = node.features()
            obs[offset + NODE_FEATURE_DIM:offset + per_node] = self.task_meta
        return obs

    def _info(self) -> dict:
        optimizer_metrics = self.last_optimizer_metrics
        if optimizer_metrics is not None:
            # Wall-clock measurements are useful internally but make seeded
            # Gymnasium transitions nondeterministic.
            optimizer_metrics = {
                key: value
                for key, value in optimizer_metrics.items()
                if key != "wall_time"
            }
        return {
            'best_value': self.best_value,
            'total_evals': self.total_evals,
            'n_frontier': len(self.frontier),
            'last_error': self.last_error,
            'last_optimizer_metrics': optimizer_metrics,
            'requested_node_index': self.last_requested_node_index,
            'effective_node_index': self.last_effective_node_index,
            'node_index_remapped': self.last_node_index_remapped,
            'node_index_remap_count': self.node_index_remap_count,
        }


# =====================================================================
# Hadamard matrix search
# =====================================================================

class HadamardTask(OptimizationTask):
    """Find an n×n Hadamard matrix (H H^T = nI, entries ±1).

    Objective: -||H H^T - nI||_F / n  (negated Frobenius distance, higher = better)
    Perfect Hadamard has objective = 0.
    """

    def __init__(self, n: int):
        if n > 2 and n % 4 != 0:
            raise ValueError(f"Hadamard matrices require n=1,2 or n≡0 mod 4, got {n}")
        self.n = n

    def random_solution(self, rng: np.random.Generator) -> np.ndarray:
        return rng.choice([-1, 1], size=(self.n, self.n)).astype(np.float64)

    def evaluate(self, H: np.ndarray) -> float:
        n = self.n
        gram = H @ H.T
        target = n * np.eye(n)
        return -np.linalg.norm(gram - target, 'fro') / n

    def perturb(self, H: np.ndarray, radius: float,
                rng: np.random.Generator) -> np.ndarray:
        """Flip approximately radius*n^2 random entries."""
        H_new = H.copy()
        n_flips = max(1, int(radius * self.n * self.n))
        for _ in range(n_flips):
            i, j = rng.integers(0, self.n, size=2)
            H_new[i, j] *= -1
        return H_new

    def metadata(self) -> np.ndarray:
        return np.array([self.n, np.log(self.n)], dtype=np.float32)

    def copy_solution(self, H: np.ndarray) -> np.ndarray:
        return H.copy()


class HadamardTaskSampler(TaskSampler):
    """Sample Hadamard tasks with n from a given range."""

    def __init__(self, n_values: List[int] = None):
        if n_values is None:
            # n must be 1, 2, or divisible by 4
            self.n_values = [4, 8, 12, 16, 20, 24, 28, 32]
        else:
            self.n_values = n_values

    def sample(self, rng: np.random.Generator) -> HadamardTask:
        n = rng.choice(self.n_values)
        return HadamardTask(n)

    def metadata_dim(self) -> int:
        return 2


# =====================================================================
# Inner optimizers for Hadamard
# =====================================================================

def _positive_work_budget(budget: int) -> int:
    """Validate the shared objective/gradient work budget."""
    if not isinstance(budget, (int, np.integer)) or budget < 1:
        raise ValueError("budget must be a positive integer")
    return int(budget)


class GreedyFlipOptimizer(InnerOptimizer):
    """Greedily flip the single entry that most improves the objective."""

    @property
    def name(self) -> str:
        return "greedy_flip"

    def run(self, H: np.ndarray, task: HadamardTask,
            budget: int, rng: np.random.Generator) -> OptimizerRun:
        budget = _positive_work_budget(budget)
        n = task.n
        H_best = H.copy()
        best_val = task.evaluate(H_best)
        values = [best_val]

        while len(values) < budget:
            # Try each possible flip, keep the best
            best_flip = None
            best_flip_val = best_val

            # For efficiency, don't try all n^2 flips — sample a subset
            n_candidates = min(n * n, 50)
            candidates = [(rng.integers(n), rng.integers(n)) for _ in range(n_candidates)]

            for (i, j) in candidates:
                if len(values) >= budget:
                    break
                H_trial = H_best.copy()
                H_trial[i, j] *= -1
                v = task.evaluate(H_trial)
                values.append(v)
                if v > best_flip_val:
                    best_flip_val = v
                    best_flip = (i, j)

            if best_flip is not None:
                H_best[best_flip[0], best_flip[1]] *= -1
                best_val = best_flip_val

        return OptimizerRun(
            H_best, values, solution_value=best_val,
            objective_evals=len(values),
        )


class RowOperationOptimizer(InnerOptimizer):
    """Apply productive entry-level moves concentrated within one row.

    Signed row permutations are exact symmetries of the Hadamard residual and
    therefore cannot improve the objective.  The compatibility action name is
    retained, but every proposal now changes relative entries within a row.
    """

    @property
    def name(self) -> str:
        return "row_ops"

    def run(self, H: np.ndarray, task: HadamardTask,
            budget: int, rng: np.random.Generator) -> OptimizerRun:
        budget = _positive_work_budget(budget)
        n = task.n
        H_best = H.copy()
        best_val = task.evaluate(H_best)
        values = [best_val]

        while len(values) < budget:
            op = rng.integers(3)
            H_trial = H_best.copy()
            i = int(rng.integers(n))

            if op == 0:
                j = int(rng.integers(n))
                H_trial[i, j] *= -1
            elif op == 1:
                count = min(n, max(1, n // 4))
                columns = rng.choice(n, size=count, replace=False)
                H_trial[i, columns] *= -1
            else:
                H_trial[i] = rng.choice([-1, 1], size=n).astype(np.float64)

            v = task.evaluate(H_trial)
            values.append(v)
            if v > best_val:
                H_best = H_trial
                best_val = v
        return OptimizerRun(
            H_best, values, solution_value=best_val,
            objective_evals=len(values),
        )


class SteepestDescentOptimizer(InnerOptimizer):
    """Gradient-like descent on the Frobenius objective.

    Computes the 'gradient' G = H H^T - nI, then flips entries
    where H_ij * (G @ H)_ij is most negative (biggest improvement).
    """

    @property
    def name(self) -> str:
        return "steepest"

    def run(self, H: np.ndarray, task: HadamardTask,
            budget: int, rng: np.random.Generator) -> OptimizerRun:
        budget = _positive_work_budget(budget)
        n = task.n
        H_best = H.copy()
        best_val = task.evaluate(H_best)
        values = [best_val]
        gradient_evals = 0

        while len(values) + gradient_evals < budget:
            gram = H_best @ H_best.T
            residual = gram - n * np.eye(n)
            # "Gradient" w.r.t. H: d||G||^2 / dH = 2 * residual @ H
            grad = 2.0 * residual @ H_best
            gradient_evals += 1
            # A flip has delta-H=-2H_ij, so its predicted error reduction is
            # proportional to +H_ij * grad_ij.
            improvement = H_best * grad
            # Pick the best flip
            idx = np.unravel_index(np.argmax(improvement), improvement.shape)
            if improvement[idx] > 0:
                if len(values) + gradient_evals >= budget:
                    break
                H_trial = H_best.copy()
                H_trial[idx] *= -1
                trial_val = task.evaluate(H_trial)
                values.append(trial_val)
                if trial_val > best_val:
                    H_best = H_trial
                    best_val = trial_val
                else:
                    break
            else:
                break

        return OptimizerRun(
            H_best,
            values,
            solution_value=best_val,
            objective_evals=len(values),
            gradient_evals=gradient_evals,
        )


class SylvesterStartOptimizer(InnerOptimizer):
    """Start from a Sylvester-type construction.

    For n = 2^k: exact Hadamard via tensor products.
    For other n: use the largest 2^k ≤ n as a seed, embed in
    a random n×n matrix.
    """

    @property
    def name(self) -> str:
        return "sylvester"

    def _sylvester(self, k: int) -> np.ndarray:
        """Build 2^k × 2^k Hadamard matrix by Sylvester construction."""
        H = np.array([[1.0]])
        for _ in range(k):
            H = np.block([[H, H], [H, -H]])
        return H

    def run(self, H: np.ndarray, task: 'HadamardTask',
            budget: int, rng: np.random.Generator) -> OptimizerRun:
        budget = _positive_work_budget(budget)
        n = task.n
        # Find largest power of 2 ≤ n
        k = int(np.floor(np.log2(n)))
        m = 2 ** k

        if m == n:
            # Exact Sylvester
            H_new = self._sylvester(k)
        else:
            # Embed Sylvester in upper-left, fill rest randomly
            H_syl = self._sylvester(k)
            H_new = rng.choice([-1, 1], size=(n, n)).astype(np.float64)
            H_new[:m, :m] = H_syl

        val = task.evaluate(H_new)
        values = [val]
        if abs(val) < 1e-10:
            return OptimizerRun(
                H_new,
                values,
                solution_value=val,
                objective_evals=1,
            )

        # Polish with greedy flips for remaining budget
        while len(values) < budget:
            i, j = rng.integers(0, n, size=2)
            # Only flip entries outside the Sylvester block
            if m < n:
                if rng.random() < 0.7:
                    i = rng.integers(m, n) if m < n else rng.integers(0, n)
            H_trial = H_new.copy()
            H_trial[i, j] *= -1
            v = task.evaluate(H_trial)
            values.append(v)
            if v > val:
                H_new = H_trial
                val = v

        return OptimizerRun(
            H_new, values, solution_value=val,
            objective_evals=len(values),
        )


class PaleyStartOptimizer(InnerOptimizer):
    """Start from a Paley-type construction when applicable.

    For n = p+1 where p ≡ 3 mod 4 is prime: build from quadratic
    residues. Otherwise fall back to random + greedy.
    """

    @property
    def name(self) -> str:
        return "paley"

    def _is_prime(self, n: int) -> bool:
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def _jacobi_symbol(self, a: int, p: int) -> int:
        """Legendre symbol (a/p) for odd prime p."""
        a = int(a) % int(p)
        if a == 0:
            return 0
        return 1 if pow(a, (int(p) - 1) // 2, int(p)) == 1 else -1

    def _paley_matrix(self, p: int) -> np.ndarray:
        """Build (p+1)×(p+1) Paley Hadamard matrix.
        Requires p ≡ 3 mod 4 and p prime.
        """
        # Quadratic residue matrix Q
        Q = np.zeros((p, p), dtype=np.float64)
        for i in range(p):
            for j in range(p):
                Q[i, j] = self._jacobi_symbol(i - j, p)

        # Paley I construction: H = S + I where S is bordered Q
        # H = [[1,  1...1 ],
        #      [1,  Q - I ]]
        # Actually the standard Paley I:
        S = Q  # S is skew-symmetric since p ≡ 3 mod 4
        # H = I + S bordered
        H = np.ones((p+1, p+1), dtype=np.float64)
        H[1:, 1:] = S
        H[0, 0] = 1
        H[1:, 0] = -1  # first column
        H[0, 1:] = 1   # first row
        # Adjust: H = [[1, j^T], [-j, S+I]] where j = all-ones
        H[1:, 1:] = S + np.eye(p)
        H[0, :] = 1
        H[:, 0] = 1
        H[0, 0] = 1
        H[1:, 0] = -1

        return H

    def run(self, H: np.ndarray, task: 'HadamardTask',
            budget: int, rng: np.random.Generator) -> OptimizerRun:
        budget = _positive_work_budget(budget)
        n = task.n
        p = n - 1
        values = []
        H_best = None
        best_val = -np.inf

        if self._is_prime(p) and p % 4 == 3:
            try:
                H_paley = self._paley_matrix(p)
            except Exception:
                H_paley = None
            if H_paley is not None and len(values) < budget:
                paley_val = task.evaluate(H_paley)
                values.append(paley_val)
                H_best = H_paley
                best_val = paley_val
                if abs(paley_val) < 1e-10 or len(values) >= budget:
                    return OptimizerRun(
                        H_best, values, solution_value=best_val,
                        objective_evals=len(values),
                    )

        # Fallback: random + greedy
        H_new = task.random_solution(rng)
        if len(values) >= budget:
            return OptimizerRun(
                H_best, values, solution_value=best_val,
                objective_evals=len(values),
            )
        val = task.evaluate(H_new)
        values.append(val)
        if H_best is None or val > best_val:
            H_best = H_new.copy()
            best_val = val
        while len(values) < budget:
            i, j = rng.integers(0, n, size=2)
            H_trial = H_new.copy()
            H_trial[i, j] *= -1
            v = task.evaluate(H_trial)
            values.append(v)
            if v > val:
                H_new = H_trial
                val = v
                if val > best_val:
                    H_best = H_new.copy()
                    best_val = val
        return OptimizerRun(
            H_best, values, solution_value=best_val,
            objective_evals=len(values),
        )


class MultiFlipOptimizer(InnerOptimizer):
    """Flip k random entries at once (simulated annealing style)."""

    def __init__(self, k: int = 3):
        self.k = k

    @property
    def name(self) -> str:
        return f"flip_{self.k}"

    def run(self, H: np.ndarray, task: HadamardTask,
            budget: int, rng: np.random.Generator) -> OptimizerRun:
        budget = _positive_work_budget(budget)
        n = task.n
        H_best = H.copy()
        best_val = task.evaluate(H_best)
        values = [best_val]

        while len(values) < budget:
            H_trial = H_best.copy()
            for _ in range(self.k):
                i, j = rng.integers(0, n, size=2)
                H_trial[i, j] *= -1
            v = task.evaluate(H_trial)
            values.append(v)
            if v > best_val:
                H_best = H_trial
                best_val = v

        return OptimizerRun(
            H_best, values, solution_value=best_val,
            objective_evals=len(values),
        )


class SimulatedAnnealingOptimizer(InnerOptimizer):
    """Simulated annealing with adaptive temperature."""

    def __init__(self, t_start: float = 2.0, t_end: float = 0.01):
        self.t_start = t_start
        self.t_end = t_end

    @property
    def name(self) -> str:
        return "anneal"

    def run(self, H: np.ndarray, task: 'HadamardTask',
            budget: int, rng: np.random.Generator) -> OptimizerRun:
        budget = _positive_work_budget(budget)
        n = task.n
        H_curr = H.copy()
        curr_val = task.evaluate(H_curr)
        H_best = H_curr.copy()
        best_val = curr_val
        values = [best_val]
        decay = (self.t_end / self.t_start) ** (1.0 / max(budget, 1))
        T = self.t_start
        while len(values) < budget:
            H_trial = H_curr.copy()
            i, j = rng.integers(0, n, size=2)
            H_trial[i, j] *= -1
            trial_val = task.evaluate(H_trial)
            values.append(trial_val)
            delta = trial_val - curr_val
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                H_curr = H_trial
                curr_val = trial_val
            if curr_val > best_val:
                H_best = H_curr.copy()
                best_val = curr_val
            T *= decay
        return OptimizerRun(
            H_best, values, solution_value=best_val,
            objective_evals=len(values),
        )


class AnnealRowOptimizer(InnerOptimizer):
    """Simulated annealing with productive within-row entry moves."""

    def __init__(self, t_start: float = 3.0, t_end: float = 0.01):
        self.t_start = t_start
        self.t_end = t_end

    @property
    def name(self) -> str:
        return "anneal_row"

    def run(self, H: np.ndarray, task: 'HadamardTask',
            budget: int, rng: np.random.Generator) -> OptimizerRun:
        budget = _positive_work_budget(budget)
        n = task.n
        H_curr = H.copy()
        curr_val = task.evaluate(H_curr)
        H_best = H_curr.copy()
        best_val = curr_val
        values = [best_val]
        decay = (self.t_end / self.t_start) ** (1.0 / max(budget, 1))
        T = self.t_start
        while len(values) < budget:
            H_trial = H_curr.copy()
            op = rng.integers(4)
            if op == 0:
                i, j = rng.integers(0, n, size=2)
                H_trial[i, j] *= -1
            elif op == 1:
                i = int(rng.integers(n))
                count = min(n, 2)
                columns = rng.choice(n, size=count, replace=False)
                H_trial[i, columns] *= -1
            elif op == 2:
                gram = H_curr @ H_curr.T
                residual = gram - n * np.eye(n)
                worst = np.argmax(np.sum(residual ** 2, axis=1))
                H_trial[worst] = rng.choice([-1, 1], size=n).astype(np.float64)
            else:
                i = int(rng.integers(n))
                width = max(1, n // 3)
                start = int(rng.integers(n))
                columns = (start + np.arange(width)) % n
                H_trial[i, columns] *= -1
            trial_val = task.evaluate(H_trial)
            values.append(trial_val)
            delta = trial_val - curr_val
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                H_curr = H_trial
                curr_val = trial_val
            if curr_val > best_val:
                H_best = H_curr.copy()
                best_val = curr_val
            T *= decay
        return OptimizerRun(
            H_best, values, solution_value=best_val,
            objective_evals=len(values),
        )


class TensorProductOptimizer(InnerOptimizer):
    """Construct via tensor product of smaller Hadamard matrices."""

    @property
    def name(self) -> str:
        return "tensor"

    def _is_prime(self, n: int) -> bool:
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def _build_small(self, m: int) -> Optional[np.ndarray]:
        if m == 1:
            return np.array([[1.0]])
        if m == 2:
            return np.array([[1, 1], [1, -1]], dtype=np.float64)
        if m & (m - 1) == 0:
            H = np.array([[1.0]])
            for _ in range(int(np.log2(m))):
                H = np.block([[H, H], [H, -H]])
            return H
        if m >= 4 and m % 4 == 0:
            p = m - 1
            if self._is_prime(p) and p % 4 == 3:
                Q = np.zeros((p, p), dtype=np.float64)
                for i in range(p):
                    for j in range(p):
                        a = int(i - j) % int(p)
                        if a == 0:
                            Q[i, j] = 0
                        else:
                            exponent = (int(p) - 1) // 2
                            Q[i, j] = (
                                1 if pow(int(a), exponent, int(p)) == 1 else -1
                            )
                H = np.ones((p + 1, p + 1), dtype=np.float64)
                H[1:, 1:] = Q + np.eye(p)
                H[1:, 0] = -1
                return H
        return None

    def run(self, H: np.ndarray, task: 'HadamardTask',
            budget: int, rng: np.random.Generator) -> OptimizerRun:
        budget = _positive_work_budget(budget)
        n = task.n
        best_H = None
        best_val = -np.inf
        values = []
        for a in range(2, n):
            if len(values) >= budget:
                break
            if n % a != 0:
                continue
            b = n // a
            Ha = self._build_small(a)
            Hb = self._build_small(b)
            if Ha is not None and Hb is not None:
                H_tensor = np.kron(Ha, Hb)
                if len(values) >= budget:
                    break
                v = task.evaluate(H_tensor)
                values.append(v)
                if v > best_val:
                    best_val = v
                    best_H = H_tensor
                if abs(v) < 1e-10:
                    return OptimizerRun(
                        best_H, values, solution_value=best_val,
                        objective_evals=len(values),
                    )
        if len(values) >= budget:
            return OptimizerRun(
                best_H, values, solution_value=best_val,
                objective_evals=len(values),
            )

        start_val = task.evaluate(H)
        values.append(start_val)
        if best_H is not None and best_val > start_val:
            return OptimizerRun(
                best_H, values, solution_value=best_val,
                objective_evals=len(values),
            )
        # Fallback
        H_curr = H.copy()
        curr_val = start_val
        while len(values) < budget:
            i, j = rng.integers(0, n, size=2)
            H_trial = H_curr.copy()
            H_trial[i, j] *= -1
            v = task.evaluate(H_trial)
            values.append(v)
            if v > curr_val:
                H_curr = H_trial
                curr_val = v
        return OptimizerRun(
            H_curr, values, solution_value=curr_val,
            objective_evals=len(values),
        )


# =====================================================================
# Convenience: Hadamard environment factory
# =====================================================================

def make_hadamard_env(
    n_values: List[int] = None,
    horizon: int = 50,
    eval_cost_weight: float = 0.0005,
    seed: Optional[int] = None,
) -> GenericOptEnv:
    """Create a Gymnasium environment for Hadamard matrix search."""
    sampler = HadamardTaskSampler(n_values)
    optimizers = [
        GreedyFlipOptimizer(),
        RowOperationOptimizer(),
        SteepestDescentOptimizer(),
        MultiFlipOptimizer(k=3),
        SylvesterStartOptimizer(),
        PaleyStartOptimizer(),
        SimulatedAnnealingOptimizer(),
        AnnealRowOptimizer(),
        TensorProductOptimizer(),
    ]
    return GenericOptEnv(
        task_sampler=sampler,
        inner_optimizers=optimizers,
        horizon=horizon,
        eval_cost_weight=eval_cost_weight,
        seed=seed,
    )
