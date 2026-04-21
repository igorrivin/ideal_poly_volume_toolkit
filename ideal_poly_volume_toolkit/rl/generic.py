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

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable

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
            budget: int, rng: np.random.Generator) -> Tuple[Any, List[float]]:
        """Run local optimization from `solution` for `budget` steps.

        Returns (best_solution, list_of_objective_values).
        """


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

@dataclass
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

    def update_stats(self, values: List[float]):
        old_best = self.best_value
        self.values_history.extend(values)
        self.n_evals += len(values)
        if values:
            self.best_value = max(self.best_value, max(values))
        arr = np.array(self.values_history)
        self.mean_value = float(arr.mean())
        self.var_value = float(arr.var()) if len(arr) > 1 else 0.0
        self.recent_improvement = float(self.best_value - old_best)


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

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

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

        self.frontier = [self.root]
        self.best_value = init_val
        self.best_solution = self.task.copy_solution(init_sol)
        self.total_evals = 1
        self.step_count = 0

        return self._encode_obs(), self._info()

    def step(self, action):
        node_idx = min(action[0], len(self.frontier) - 1)
        op_type = action[1]
        node = self.frontier[node_idx]

        old_best = self.best_value
        evals_used = 0

        try:
            if op_type < self.n_inner:
                evals_used = self._do_inner_opt(node, op_type)
            elif op_type == self.OP_SAMPLE:
                evals_used = self._do_sample(node, n=4)
            elif op_type == self.OP_SPLIT:
                self._do_split(node)
            elif op_type == self.OP_PERTURB:
                evals_used = self._do_perturb(node)
            elif op_type == self.OP_PRUNE:
                self._do_prune(node)
        except Exception:
            pass

        # Update global best
        for n in self.frontier:
            if n.best_value > self.best_value and n.best_solution is not None:
                self.best_value = n.best_value
                self.best_solution = self.task.copy_solution(n.best_solution)

        self.total_evals += evals_used
        self.step_count += 1

        improvement = max(0.0, self.best_value - old_best)
        reward = float(improvement - self.eval_cost_weight * evals_used)

        terminated = self.step_count >= self.horizon
        return self._encode_obs(), reward, terminated, False, self._info()

    def _do_inner_opt(self, node: SearchNode, opt_idx: int) -> int:
        opt = self.inner_optimizers[opt_idx]
        start = node.best_solution if node.best_solution is not None else node.center
        budget = getattr(opt, 'default_budget', 8)
        # Legacy heuristic: annealing optimizers get a bigger budget even if
        # the class forgot to override default_budget.
        if 'anneal' in opt.name and opt.default_budget == 8:
            budget = 50
        best_sol, values = opt.run(start, self.task, budget=budget, rng=self._rng)
        node.update_stats(values)
        best_val = self.task.evaluate(best_sol)
        if best_val > node.best_value:
            node.best_value = best_val
            node.best_solution = self.task.copy_solution(best_sol)
            node.center = self.task.copy_solution(best_sol)
        return len(values)

    def _do_sample(self, node: SearchNode, n: int = 4) -> int:
        values = []
        for _ in range(n):
            sol = self.task.perturb(
                node.best_solution or node.center,
                node.radius * 0.3, self._rng
            )
            v = self.task.evaluate(sol)
            values.append(v)
            if v > node.best_value:
                node.best_value = v
                node.best_solution = self.task.copy_solution(sol)
        node.update_stats(values)
        return n

    def _do_split(self, node: SearchNode):
        if len(self.frontier) >= MAX_FRONTIER - 1:
            return
        r = node.radius * 0.6
        c1 = self.task.perturb(node.center, r * 0.3, self._rng)
        c2 = self.task.perturb(node.center, r * 0.3, self._rng)
        child1 = SearchNode(c1, r, depth=node.depth + 1, parent=node)
        child2 = SearchNode(c2, r, depth=node.depth + 1, parent=node)
        if node.best_solution is not None:
            for child in [child1, child2]:
                child.best_solution = self.task.copy_solution(node.best_solution)
                child.best_value = node.best_value
        node.children = [child1, child2]
        node.is_active = False
        if node in self.frontier:
            self.frontier.remove(node)
        self.frontier.extend([child1, child2])

    def _do_perturb(self, node: SearchNode) -> int:
        if node.best_solution is None:
            return 0
        sol = self.task.perturb(node.best_solution, 0.5, self._rng)
        v = self.task.evaluate(sol)
        node.update_stats([v])
        if v > node.best_value:
            node.best_value = v
            node.best_solution = self.task.copy_solution(sol)
            node.center = self.task.copy_solution(sol)
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
        return {
            'best_value': self.best_value,
            'total_evals': self.total_evals,
            'n_frontier': len(self.frontier),
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

class GreedyFlipOptimizer(InnerOptimizer):
    """Greedily flip the single entry that most improves the objective."""

    @property
    def name(self) -> str:
        return "greedy_flip"

    def run(self, H: np.ndarray, task: HadamardTask,
            budget: int, rng: np.random.Generator) -> Tuple[np.ndarray, List[float]]:
        n = task.n
        H_best = H.copy()
        best_val = task.evaluate(H_best)
        values = [best_val]

        for _ in range(budget):
            # Try each possible flip, keep the best
            best_flip = None
            best_flip_val = best_val

            # For efficiency, don't try all n^2 flips — sample a subset
            n_candidates = min(n * n, 50)
            candidates = [(rng.integers(n), rng.integers(n)) for _ in range(n_candidates)]

            for (i, j) in candidates:
                H_trial = H_best.copy()
                H_trial[i, j] *= -1
                v = task.evaluate(H_trial)
                if v > best_flip_val:
                    best_flip_val = v
                    best_flip = (i, j)

            if best_flip is not None:
                H_best[best_flip[0], best_flip[1]] *= -1
                best_val = best_flip_val

            values.append(best_val)

        return H_best, values


class RowOperationOptimizer(InnerOptimizer):
    """Apply row operations: negate a row, swap two rows, or
    negate a row and swap."""

    @property
    def name(self) -> str:
        return "row_ops"

    def run(self, H: np.ndarray, task: HadamardTask,
            budget: int, rng: np.random.Generator) -> Tuple[np.ndarray, List[float]]:
        n = task.n
        H_best = H.copy()
        best_val = task.evaluate(H_best)
        values = [best_val]

        for _ in range(budget):
            op = rng.integers(3)
            H_trial = H_best.copy()

            if op == 0:
                # Negate a random row
                i = rng.integers(n)
                H_trial[i] *= -1
            elif op == 1:
                # Swap two random rows
                i, j = rng.choice(n, size=2, replace=False)
                H_trial[[i, j]] = H_trial[[j, i]]
            else:
                # Negate a row and swap with another
                i, j = rng.choice(n, size=2, replace=False)
                H_trial[i] *= -1
                H_trial[[i, j]] = H_trial[[j, i]]

            v = task.evaluate(H_trial)
            if v > best_val:
                H_best = H_trial
                best_val = v
            values.append(best_val)

        return H_best, values


class SteepestDescentOptimizer(InnerOptimizer):
    """Gradient-like descent on the Frobenius objective.

    Computes the 'gradient' G = H H^T - nI, then flips entries
    where H_ij * (G @ H)_ij is most negative (biggest improvement).
    """

    @property
    def name(self) -> str:
        return "steepest"

    def run(self, H: np.ndarray, task: HadamardTask,
            budget: int, rng: np.random.Generator) -> Tuple[np.ndarray, List[float]]:
        n = task.n
        H_best = H.copy()
        best_val = task.evaluate(H_best)
        values = [best_val]

        for _ in range(budget):
            gram = H_best @ H_best.T
            residual = gram - n * np.eye(n)
            # "Gradient" w.r.t. H: d||G||^2 / dH = 2 * residual @ H
            grad = 2.0 * residual @ H_best
            # Improvement from flipping H[i,j]: proportional to -H[i,j] * grad[i,j]
            improvement = -H_best * grad
            # Pick the best flip
            idx = np.unravel_index(np.argmax(improvement), improvement.shape)
            if improvement[idx] > 0:
                H_best[idx] *= -1
                best_val = task.evaluate(H_best)
            values.append(best_val)

        return H_best, values


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
            budget: int, rng: np.random.Generator) -> Tuple[Any, List[float]]:
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

        # Polish with greedy flips for remaining budget
        for _ in range(budget - 1):
            i, j = rng.integers(0, n, size=2)
            # Only flip entries outside the Sylvester block
            if m < n:
                if rng.random() < 0.7:
                    i = rng.integers(m, n) if m < n else rng.integers(0, n)
            H_trial = H_new.copy()
            H_trial[i, j] *= -1
            v = task.evaluate(H_trial)
            if v > val:
                H_new = H_trial
                val = v
            values.append(val)

        return H_new, values


class PaleyStartOptimizer(InnerOptimizer):
    """Start from a Paley-type construction when applicable.

    For n = p+1 where p ≡ 3 mod 4 is prime: build from quadratic
    residues. Otherwise fall back to random + greedy.
    """

    @property
    def name(self) -> str:
        return "paley"

    def _is_prime(self, n: int) -> bool:
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i+2) == 0: return False
            i += 6
        return True

    def _jacobi_symbol(self, a: int, p: int) -> int:
        """Legendre symbol (a/p) for odd prime p."""
        a = int(a) % int(p)
        if a == 0: return 0
        return 1 if pow(a, (int(p)-1)//2, int(p)) == 1 else -1

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
            budget: int, rng: np.random.Generator) -> Tuple[Any, List[float]]:
        n = task.n
        p = n - 1

        if self._is_prime(p) and p % 4 == 3:
            try:
                H_new = self._paley_matrix(p)
                val = task.evaluate(H_new)
                if abs(val) < 1e-10:
                    return H_new, [val] * budget
            except Exception:
                pass

        # Fallback: random + greedy
        H_new = task.random_solution(rng)
        val = task.evaluate(H_new)
        values = [val]
        for _ in range(budget - 1):
            i, j = rng.integers(0, n, size=2)
            H_trial = H_new.copy()
            H_trial[i, j] *= -1
            v = task.evaluate(H_trial)
            if v > val: H_new = H_trial; val = v
            values.append(val)
        return H_new, values


class MultiFlipOptimizer(InnerOptimizer):
    """Flip k random entries at once (simulated annealing style)."""

    def __init__(self, k: int = 3):
        self.k = k

    @property
    def name(self) -> str:
        return f"flip_{self.k}"

    def run(self, H: np.ndarray, task: HadamardTask,
            budget: int, rng: np.random.Generator) -> Tuple[np.ndarray, List[float]]:
        n = task.n
        H_best = H.copy()
        best_val = task.evaluate(H_best)
        values = [best_val]

        for _ in range(budget):
            H_trial = H_best.copy()
            for _ in range(self.k):
                i, j = rng.integers(0, n, size=2)
                H_trial[i, j] *= -1
            v = task.evaluate(H_trial)
            if v > best_val:
                H_best = H_trial
                best_val = v
            values.append(best_val)

        return H_best, values


class SimulatedAnnealingOptimizer(InnerOptimizer):
    """Simulated annealing with adaptive temperature."""

    def __init__(self, t_start: float = 2.0, t_end: float = 0.01):
        self.t_start = t_start
        self.t_end = t_end

    @property
    def name(self) -> str:
        return "anneal"

    def run(self, H: np.ndarray, task: 'HadamardTask',
            budget: int, rng: np.random.Generator) -> Tuple[Any, List[float]]:
        n = task.n
        H_curr = H.copy()
        curr_val = task.evaluate(H_curr)
        H_best = H_curr.copy()
        best_val = curr_val
        values = [best_val]
        decay = (self.t_end / self.t_start) ** (1.0 / max(budget, 1))
        T = self.t_start
        for _ in range(budget):
            H_trial = H_curr.copy()
            i, j = rng.integers(0, n, size=2)
            H_trial[i, j] *= -1
            trial_val = task.evaluate(H_trial)
            delta = trial_val - curr_val
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                H_curr = H_trial
                curr_val = trial_val
            if curr_val > best_val:
                H_best = H_curr.copy()
                best_val = curr_val
            T *= decay
            values.append(best_val)
        return H_best, values


class AnnealRowOptimizer(InnerOptimizer):
    """Simulated annealing with row-level moves."""

    def __init__(self, t_start: float = 3.0, t_end: float = 0.01):
        self.t_start = t_start
        self.t_end = t_end

    @property
    def name(self) -> str:
        return "anneal_row"

    def run(self, H: np.ndarray, task: 'HadamardTask',
            budget: int, rng: np.random.Generator) -> Tuple[Any, List[float]]:
        n = task.n
        H_curr = H.copy()
        curr_val = task.evaluate(H_curr)
        H_best = H_curr.copy()
        best_val = curr_val
        values = [best_val]
        decay = (self.t_end / self.t_start) ** (1.0 / max(budget, 1))
        T = self.t_start
        for _ in range(budget):
            H_trial = H_curr.copy()
            op = rng.integers(4)
            if op == 0:
                i = rng.integers(n)
                H_trial[i] *= -1
            elif op == 1:
                i, j = rng.choice(n, size=2, replace=False)
                H_trial[[i, j]] = H_trial[[j, i]]
            elif op == 2:
                gram = H_curr @ H_curr.T
                residual = gram - n * np.eye(n)
                worst = np.argmax(np.sum(residual ** 2, axis=1))
                H_trial[worst] = rng.choice([-1, 1], size=n).astype(np.float64)
            else:
                j = rng.integers(n)
                H_trial[:, j] *= -1
            trial_val = task.evaluate(H_trial)
            delta = trial_val - curr_val
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                H_curr = H_trial
                curr_val = trial_val
            if curr_val > best_val:
                H_best = H_curr.copy()
                best_val = curr_val
            T *= decay
            values.append(best_val)
        return H_best, values


class TensorProductOptimizer(InnerOptimizer):
    """Construct via tensor product of smaller Hadamard matrices."""

    @property
    def name(self) -> str:
        return "tensor"

    def _is_prime(self, n: int) -> bool:
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i+2) == 0: return False
            i += 6
        return True

    def _build_small(self, m: int) -> Optional[np.ndarray]:
        if m == 1: return np.array([[1.0]])
        if m == 2: return np.array([[1,1],[1,-1]], dtype=np.float64)
        if m & (m-1) == 0:
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
                        a = int(i-j) % int(p)
                        if a == 0: Q[i,j] = 0
                        else: Q[i,j] = 1 if pow(int(a), (int(p)-1)//2, int(p)) == 1 else -1
                H = np.ones((p+1, p+1), dtype=np.float64)
                H[1:, 1:] = Q + np.eye(p)
                H[1:, 0] = -1
                return H
        return None

    def run(self, H: np.ndarray, task: 'HadamardTask',
            budget: int, rng: np.random.Generator) -> Tuple[Any, List[float]]:
        n = task.n
        best_H = None; best_val = -np.inf
        for a in range(2, n):
            if n % a != 0: continue
            b = n // a
            Ha = self._build_small(a)
            Hb = self._build_small(b)
            if Ha is not None and Hb is not None:
                H_tensor = np.kron(Ha, Hb)
                v = task.evaluate(H_tensor)
                if v > best_val:
                    best_val = v; best_H = H_tensor
                if abs(v) < 1e-10:
                    return best_H, [best_val] * budget
        if best_H is not None and best_val > task.evaluate(H):
            return best_H, [best_val] * budget
        # Fallback
        H_curr = H.copy(); curr_val = task.evaluate(H_curr); values = [curr_val]
        for _ in range(budget - 1):
            i, j = rng.integers(0, n, size=2)
            H_trial = H_curr.copy(); H_trial[i,j] *= -1
            v = task.evaluate(H_trial)
            if v > curr_val: H_curr = H_trial; curr_val = v
            values.append(curr_val)
        return H_curr, values


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
