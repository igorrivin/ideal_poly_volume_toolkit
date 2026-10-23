"""
RL-guided search for high-girth cubic graphs (cage problem).

A (3,g)-cage is the smallest 3-regular (cubic) graph with girth g.
Known cages:
  g=3: K_4 (4 vertices)
  g=4: K_{3,3} (6 vertices)
  g=5: Petersen (10 vertices)
  g=6: Heawood (14 vertices)
  g=7: McGee (24 vertices)
  g=8: Tutte-Coxeter (30 vertices)

For higher girths, exact values are unknown.

The optimization task: given N vertices, find a cubic graph on N
vertices with maximum girth. Equivalently, maximize girth subject
to 3-regularity and fixed vertex count.
"""

from typing import List, Tuple, Optional
from collections import deque

import numpy as np

from ..generic import (
    OptimizationTask, InnerOptimizer, TaskSampler, GenericOptEnv, OptimizerRun
)


# =====================================================================
# Graph utilities
# =====================================================================

def random_cubic_graph(N: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random cubic graph on N vertices (N must be even).

    Uses the pairing model: create 3 copies of each vertex,
    take a random perfect matching, collapse to graph.
    Retry if result has multi-edges or self-loops.

    Returns adjacency list as (N, 3) array of neighbor indices.
    """
    assert N % 2 == 0, "N must be even for cubic graphs"
    for _ in range(1000):
        points = []
        for v in range(N):
            points.extend([v, v, v])
        rng.shuffle(points)
        adj = [[] for _ in range(N)]
        ok = True
        for i in range(0, len(points), 2):
            u, v = points[i], points[i + 1]
            if u == v:  # self-loop
                ok = False
                break
            if v in adj[u]:  # multi-edge
                ok = False
                break
            adj[u].append(v)
            adj[v].append(u)
        if ok and all(len(a) == 3 for a in adj):
            return np.array(adj, dtype=np.int32)
    # Fallback: deterministic construction (cycle + matching)
    return _fallback_cubic(N)


def _fallback_cubic(N: int) -> np.ndarray:
    """Deterministic cubic graph: prism graph (two N/2-cycles + matching)."""
    if N == 4:
        # The two-cycle "prism" degenerates at N=4.  K4 is the unique simple
        # cubic graph on four vertices and is therefore the correct fallback.
        return np.array(
            [
                [1, 2, 3],
                [0, 2, 3],
                [0, 1, 3],
                [0, 1, 2],
            ],
            dtype=np.int32,
        )

    h = N // 2
    adj = [[] for _ in range(N)]
    for i in range(h):
        # Outer cycle
        adj[i].append((i + 1) % h)
        adj[(i + 1) % h].append(i)
        # Inner cycle
        adj[h + i].append(h + (i + 1) % h)
        adj[h + (i + 1) % h].append(h + i)
        # Matching between outer and inner
        adj[i].append(h + i)
        adj[h + i].append(i)
    # Clean up duplicates and ensure degree 3
    result = np.zeros((N, 3), dtype=np.int32)
    for v in range(N):
        neighbors = sorted(set(adj[v]))[:3]
        while len(neighbors) < 3:
            neighbors.append(0)
        result[v] = neighbors
    return result


def adj_list_to_edge_set(adj: np.ndarray) -> set:
    """Convert adjacency list to set of (u,v) edges with u < v."""
    edges = set()
    N = len(adj)
    for u in range(N):
        for v in adj[u]:
            if v > u:
                edges.add((u, int(v)))
    return edges


def compute_girth(adj: np.ndarray) -> int:
    """Compute the girth (shortest cycle length) of a graph.

    Returns N+1 if acyclic (tree/forest).
    """
    N = len(adj)
    best = N + 1

    for source in range(N):
        # BFS from source
        dist = [-1] * N
        dist[source] = 0
        queue = deque([source])
        parent = [-1] * N

        while queue:
            u = queue.popleft()
            for v in adj[u]:
                v = int(v)
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    best = min(best, cycle_len)

            if dist[u] + 1 >= best // 2 + 1:
                break  # can't find shorter cycles from this source

        if best == 3:
            return 3  # can't do better

    return best


def is_valid_cubic(adj: np.ndarray) -> bool:
    """Check that adj represents a valid simple cubic graph."""
    N = len(adj)
    for v in range(N):
        neighbors = list(adj[v])
        if len(neighbors) != 3:
            return False
        if v in neighbors:  # self-loop
            return False
        if len(set(neighbors)) != 3:  # multi-edge
            return False
        for u in neighbors:
            if v not in adj[int(u)]:  # asymmetric
                return False
    return True


# =====================================================================
# Edge swap operations (preserve 3-regularity)
# =====================================================================

def edge_swap(adj: np.ndarray, e1: Tuple[int, int], e2: Tuple[int, int],
              mode: int = 0) -> Optional[np.ndarray]:
    """Swap two edges while preserving 3-regularity.

    Remove edges (a,b) and (c,d).
    Mode 0: add (a,c) and (b,d).
    Mode 1: add (a,d) and (b,c).

    Returns new adj list, or None if the swap creates multi-edges/self-loops.
    """
    a, b = e1
    c, d = e2

    if mode == 0:
        new_edges = [(a, c), (b, d)]
    else:
        new_edges = [(a, d), (b, c)]

    # Check for self-loops
    for u, v in new_edges:
        if u == v:
            return None

    # Check for multi-edges
    adj_new = adj.copy()

    # Remove old edges
    for u, v in [e1, e2]:
        idx_u = np.where(adj_new[u] == v)[0]
        idx_v = np.where(adj_new[v] == u)[0]
        if len(idx_u) == 0 or len(idx_v) == 0:
            return None
        # Replace with placeholder
        adj_new[u, idx_u[0]] = -1
        adj_new[v, idx_v[0]] = -1

    # Add new edges
    for u, v in new_edges:
        if v in adj_new[u]:  # would create multi-edge
            return None
        idx_u = np.where(adj_new[u] == -1)[0]
        idx_v = np.where(adj_new[v] == -1)[0]
        if len(idx_u) == 0 or len(idx_v) == 0:
            return None
        adj_new[u, idx_u[0]] = v
        adj_new[v, idx_v[0]] = u

    return adj_new


# =====================================================================
# Task definition
# =====================================================================

class CubicGirthTask(OptimizationTask):
    """Maximize girth of a cubic graph on N vertices.

    Objective: girth(G) + small_bonus_for_fewer_short_cycles.
    Higher = better.
    """

    def __init__(self, N: int):
        assert N % 2 == 0 and N >= 4
        self.N = N

    def random_solution(self, rng: np.random.Generator) -> np.ndarray:
        return random_cubic_graph(self.N, rng)

    def evaluate(self, adj: np.ndarray) -> float:
        if not is_valid_cubic(adj):
            return 0.0
        g = compute_girth(adj)
        # Main objective: girth
        # Small bonus for "almost higher girth" (fewer short cycles)
        n_short = self._count_short_cycles(adj, g)
        return float(g) - 0.01 * n_short / max(self.N, 1)

    def _count_short_cycles(self, adj: np.ndarray, girth: int) -> int:
        """Count shortest cycles exactly using edge-deleted BFS path counts.

        Removing one edge of a girth-length cycle leaves a shortest path of
        length ``girth - 1`` between its endpoints.  Conversely, every such
        path closes to a girth-length cycle.  Summing these path counts over
        all edges counts every undirected cycle once per edge, hence the final
        division by ``girth``.
        """
        N = len(adj)
        if girth < 3 or girth > N:
            return 0

        target_distance = girth - 1
        edge_cycle_memberships = 0
        for source, target in adj_list_to_edge_set(adj):
            dist = [-1] * N
            path_count = [0] * N
            dist[source] = 0
            path_count[source] = 1
            queue = deque([source])

            while queue:
                u = queue.popleft()
                if dist[u] >= target_distance:
                    continue
                for v in adj[u]:
                    v = int(v)
                    if (u == source and v == target) or (u == target and v == source):
                        continue
                    next_distance = dist[u] + 1
                    if next_distance > target_distance:
                        continue
                    if dist[v] == -1:
                        dist[v] = next_distance
                        path_count[v] = path_count[u]
                        queue.append(v)
                    elif dist[v] == next_distance:
                        path_count[v] += path_count[u]

            if dist[target] == target_distance:
                edge_cycle_memberships += path_count[target]

        return edge_cycle_memberships // girth

    def perturb(self, adj: np.ndarray, radius: float,
                rng: np.random.Generator) -> np.ndarray:
        """Perturb by doing random edge swaps."""
        n_swaps = max(1, int(radius * self.N))
        adj_new = adj.copy()
        edges = list(adj_list_to_edge_set(adj_new))
        for _ in range(n_swaps):
            if len(edges) < 2:
                break
            idx = rng.choice(len(edges), size=2, replace=False)
            e1, e2 = edges[idx[0]], edges[idx[1]]
            mode = rng.integers(2)
            result = edge_swap(adj_new, e1, e2, mode)
            if result is not None:
                adj_new = result
                edges = list(adj_list_to_edge_set(adj_new))
        return adj_new

    def metadata(self) -> np.ndarray:
        return np.array([self.N, np.log(self.N)], dtype=np.float32)

    def copy_solution(self, adj: np.ndarray) -> np.ndarray:
        return adj.copy()


class CubicGirthTaskSampler(TaskSampler):
    """Sample cubic girth tasks with varying N."""

    def __init__(self, N_values: List[int] = None):
        if N_values is None:
            self.N_values = [10, 14, 18, 20, 24, 26, 30]
        else:
            self.N_values = N_values

    def sample(self, rng: np.random.Generator) -> CubicGirthTask:
        N = int(rng.choice(self.N_values))
        return CubicGirthTask(N)

    def metadata_dim(self) -> int:
        return 2


# =====================================================================
# Inner optimizers for cubic graph search
# =====================================================================

def _positive_work_budget(budget: int) -> int:
    budget = int(budget)
    if budget < 1:
        raise ValueError("budget must be positive")
    return budget


class RandomSwapOptimizer(InnerOptimizer):
    """Random edge swaps, accept if girth improves or stays same."""

    @property
    def name(self) -> str:
        return "rand_swap"

    def run(self, adj: np.ndarray, task: CubicGirthTask,
            budget: int, rng: np.random.Generator) -> OptimizerRun:
        budget = _positive_work_budget(budget)
        adj_best = adj.copy()
        best_val = task.evaluate(adj_best)
        values = [best_val]

        edges = list(adj_list_to_edge_set(adj_best))
        for _ in range(budget - 1):
            if len(edges) < 2:
                continue
            idx = rng.choice(len(edges), size=2, replace=False)
            e1, e2 = edges[idx[0]], edges[idx[1]]
            mode = rng.integers(2)
            result = edge_swap(adj_best, e1, e2, mode)
            if result is not None:
                v = task.evaluate(result)
                values.append(v)
                if v >= best_val:
                    adj_best = result
                    best_val = v
                    edges = list(adj_list_to_edge_set(adj_best))

        return OptimizerRun(
            solution=adj_best,
            values=values,
            solution_value=best_val,
            objective_evals=len(values),
        )


class ShortCycleBreaker(InnerOptimizer):
    """Find a shortest cycle and try to break it by swapping one of its edges."""

    @property
    def name(self) -> str:
        return "cycle_break"

    def _find_shortest_cycle(self, adj: np.ndarray) -> Optional[List[int]]:
        """Find one shortest cycle, returned in cyclic vertex order.

        Each non-tree edge in a BFS tree closes a cycle.  The two tree paths
        can share an arbitrary prefix, so reconstruction explicitly finds
        their lowest common ancestor before joining the paths.
        """
        N = len(adj)
        best_len = N + 1
        best_cycle = None

        for source in range(N):
            dist = [-1] * N
            parent = [-1] * N
            dist[source] = 0
            queue = deque([source])

            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    v = int(v)
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        u_to_root = []
                        ancestor_position = {}
                        x = u
                        while x != -1:
                            ancestor_position[x] = len(u_to_root)
                            u_to_root.append(x)
                            x = parent[x]

                        v_to_lca = []
                        x = v
                        while x not in ancestor_position:
                            v_to_lca.append(x)
                            x = parent[x]
                        lca = x
                        u_to_lca = u_to_root[: ancestor_position[lca] + 1]

                        # Traverse LCA -> ... -> u, cross (u, v), then
                        # v -> ... -> child-of-LCA; the implicit closing edge
                        # returns to LCA.
                        cycle = list(reversed(u_to_lca)) + v_to_lca
                        if 3 <= len(cycle) < best_len:
                            best_len = len(cycle)
                            best_cycle = cycle

            if best_len == 3:
                return best_cycle

        return best_cycle

    def run(self, adj: np.ndarray, task: CubicGirthTask,
            budget: int, rng: np.random.Generator) -> OptimizerRun:
        budget = _positive_work_budget(budget)
        adj_best = adj.copy()
        best_val = task.evaluate(adj_best)
        values = [best_val]

        for _ in range(budget - 1):
            cycle = self._find_shortest_cycle(adj_best)
            if cycle is None or len(cycle) < 3:
                continue

            # Try swapping each edge of the cycle with a random non-cycle edge
            improved = False
            cycle_edges = [(cycle[i], cycle[(i+1) % len(cycle)])
                           for i in range(len(cycle))]
            all_edges = list(adj_list_to_edge_set(adj_best))
            non_cycle = [e for e in all_edges
                         if e not in cycle_edges and (e[1], e[0]) not in cycle_edges]

            rng.shuffle(cycle_edges)
            for ce in cycle_edges[:3]:  # try up to 3 cycle edges
                if not non_cycle:
                    break
                ne = non_cycle[rng.integers(len(non_cycle))]
                for mode in [0, 1]:
                    if len(values) >= budget:
                        break
                    result = edge_swap(adj_best, ce, ne, mode)
                    if result is not None:
                        v = task.evaluate(result)
                        values.append(v)
                        if v > best_val:
                            adj_best = result
                            best_val = v
                            improved = True
                            break
                if improved:
                    break
            if len(values) >= budget:
                break

        return OptimizerRun(
            solution=adj_best,
            values=values,
            solution_value=best_val,
            objective_evals=len(values),
        )


class AnnealSwapOptimizer(InnerOptimizer):
    """Simulated annealing with edge swaps."""

    def __init__(self, t_start: float = 2.0, t_end: float = 0.01):
        self.t_start = t_start
        self.t_end = t_end

    @property
    def name(self) -> str:
        return "anneal_swap"

    def run(self, adj: np.ndarray, task: CubicGirthTask,
            budget: int, rng: np.random.Generator) -> OptimizerRun:
        budget = _positive_work_budget(budget)
        adj_curr = adj.copy()
        curr_val = task.evaluate(adj_curr)
        adj_best = adj_curr.copy()
        best_val = curr_val
        values = [best_val]

        decay = (self.t_end / self.t_start) ** (1.0 / max(budget - 1, 1))
        T = self.t_start

        for _ in range(budget - 1):
            edges = list(adj_list_to_edge_set(adj_curr))
            if len(edges) < 2:
                continue

            idx = rng.choice(len(edges), size=2, replace=False)
            e1, e2 = edges[idx[0]], edges[idx[1]]
            mode = rng.integers(2)
            result = edge_swap(adj_curr, e1, e2, mode)

            if result is not None:
                trial_val = task.evaluate(result)
                values.append(trial_val)
                delta = trial_val - curr_val
                if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                    adj_curr = result
                    curr_val = trial_val
                if curr_val > best_val:
                    adj_best = adj_curr.copy()
                    best_val = curr_val

            T *= decay

        return OptimizerRun(
            solution=adj_best,
            values=values,
            solution_value=best_val,
            objective_evals=len(values),
        )


class MultiSwapOptimizer(InnerOptimizer):
    """Multiple simultaneous edge swaps for larger moves."""

    def __init__(self, k: int = 3):
        self.k = k

    @property
    def name(self) -> str:
        return f"multi_{self.k}"

    def run(self, adj: np.ndarray, task: CubicGirthTask,
            budget: int, rng: np.random.Generator) -> OptimizerRun:
        budget = _positive_work_budget(budget)
        adj_best = adj.copy()
        best_val = task.evaluate(adj_best)
        values = [best_val]

        for _ in range(budget - 1):
            adj_trial = adj_best.copy()
            success = True
            for _ in range(self.k):
                edges = list(adj_list_to_edge_set(adj_trial))
                if len(edges) < 2:
                    success = False
                    break
                idx = rng.choice(len(edges), size=2, replace=False)
                result = edge_swap(adj_trial, edges[idx[0]], edges[idx[1]],
                                    rng.integers(2))
                if result is not None:
                    adj_trial = result
                else:
                    success = False
                    break

            if success:
                v = task.evaluate(adj_trial)
                values.append(v)
                if v > best_val:
                    adj_best = adj_trial
                    best_val = v

        return OptimizerRun(
            solution=adj_best,
            values=values,
            solution_value=best_val,
            objective_evals=len(values),
        )


# =====================================================================
# Log-spanning-tree surrogate
# =====================================================================

def _log_spanning_tree_count(adj: np.ndarray) -> float:
    """Return log(number of spanning trees), or -inf if disconnected."""
    N = len(adj)
    laplacian = np.zeros((N, N), dtype=np.float64)
    for v in range(N):
        for u in adj[v]:
            u = int(u)
            laplacian[v, u] -= 1.0
            laplacian[v, v] += 1.0

    eigvals = np.linalg.eigvalsh(laplacian)
    near_zero = np.abs(eigvals) <= 1e-10
    if np.count_nonzero(near_zero) != 1:
        return -np.inf

    positive = eigvals[~near_zero]
    if len(positive) != N - 1 or np.any(positive <= 0.0):
        return -np.inf
    return float(np.sum(np.log(positive)) - np.log(N))


class SpanningTreeOptimizer(InnerOptimizer):
    """Maximize log(spanning tree count) as a proxy for girth.

    The spanning tree count τ(G) = (1/N) det'(L) where L is the
    Laplacian and det' is the product of nonzero eigenvalues.
    log τ correlates with expansion, which correlates with girth.

    This gives a smooth objective for edge swaps, enabling
    gradient-like optimization in the discrete space.
    """

    @property
    def name(self) -> str:
        return "log_det"

    def _log_spanning_trees(self, adj: np.ndarray) -> float:
        """Compute log(spanning tree count) from adjacency list."""
        return _log_spanning_tree_count(adj)

    def run(self, adj: np.ndarray, task: 'CubicGirthTask',
            budget: int, rng: np.random.Generator) -> OptimizerRun:
        budget = _positive_work_budget(budget)
        adj_current = adj.copy()
        adj_best = adj_current.copy()
        best_girth_val = task.evaluate(adj_current)
        values = [best_girth_val]
        surrogate_evals = 0
        if budget == 1:
            return OptimizerRun(
                solution=adj_best,
                values=values,
                solution_value=best_girth_val,
                objective_evals=1,
            )

        current_log_tau = self._log_spanning_trees(adj_current)
        surrogate_evals = 1

        for _ in range(budget):
            if len(values) + surrogate_evals + 2 > budget:
                break
            edges = list(adj_list_to_edge_set(adj_current))
            if len(edges) < 2:
                continue

            idx = rng.choice(len(edges), size=2, replace=False)
            e1, e2 = edges[idx[0]], edges[idx[1]]
            mode = rng.integers(2)
            result = edge_swap(adj_current, e1, e2, mode)

            if result is not None:
                log_tau = self._log_spanning_trees(result)
                surrogate_evals += 1
                if log_tau > current_log_tau:
                    adj_current = result
                    current_log_tau = log_tau
                    girth_val = task.evaluate(adj_current)
                    values.append(girth_val)
                    if girth_val > best_girth_val:
                        adj_best = adj_current.copy()
                        best_girth_val = girth_val

        return OptimizerRun(
            solution=adj_best,
            values=values,
            solution_value=best_girth_val,
            objective_evals=len(values),
            surrogate_evals=surrogate_evals,
        )


class AnnealLogDetOptimizer(InnerOptimizer):
    """Simulated annealing on log(spanning tree count).

    Combines the smooth log-det landscape with SA's ability to
    escape local minima. The temperature schedule allows accepting
    worse log-det values early on.
    """

    def __init__(self, t_start: float = 1.0, t_end: float = 0.01):
        self.t_start = t_start
        self.t_end = t_end

    @property
    def name(self) -> str:
        return "anneal_logdet"

    def _log_spanning_trees(self, adj: np.ndarray) -> float:
        return _log_spanning_tree_count(adj)

    def run(self, adj: np.ndarray, task: 'CubicGirthTask',
            budget: int, rng: np.random.Generator) -> OptimizerRun:
        budget = _positive_work_budget(budget)
        adj_curr = adj.copy()
        adj_best = adj_curr.copy()
        best_girth_val = task.evaluate(adj_best)
        values = [best_girth_val]
        surrogate_evals = 0
        if budget == 1:
            return OptimizerRun(
                solution=adj_best,
                values=values,
                solution_value=best_girth_val,
                objective_evals=1,
            )

        curr_log_tau = self._log_spanning_trees(adj_curr)
        surrogate_evals = 1
        best_log_tau = curr_log_tau

        decay = (self.t_end / self.t_start) ** (1.0 / max((budget - 2) // 2, 1))
        T = self.t_start

        for _ in range(budget):
            if len(values) + surrogate_evals + 2 > budget:
                break
            edges = list(adj_list_to_edge_set(adj_curr))
            if len(edges) < 2:
                continue

            idx = rng.choice(len(edges), size=2, replace=False)
            e1, e2 = edges[idx[0]], edges[idx[1]]
            mode = rng.integers(2)
            result = edge_swap(adj_curr, e1, e2, mode)

            if result is not None:
                log_tau = self._log_spanning_trees(result)
                surrogate_evals += 1
                delta = log_tau - curr_log_tau
                if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                    adj_curr = result
                    curr_log_tau = log_tau

                girth_val = task.evaluate(adj_curr)
                values.append(girth_val)
                if girth_val > best_girth_val or (
                    girth_val == best_girth_val and curr_log_tau > best_log_tau
                ):
                    adj_best = adj_curr.copy()
                    best_girth_val = girth_val
                    best_log_tau = curr_log_tau

            T *= decay

        return OptimizerRun(
            solution=adj_best,
            values=values,
            solution_value=best_girth_val,
            objective_evals=len(values),
            surrogate_evals=surrogate_evals,
        )


class HybridGirthLogDetOptimizer(InnerOptimizer):
    """Alternate between girth-improving and log-det-improving swaps.

    Phase 1: Maximize log-det (find well-structured graph)
    Phase 2: Break shortest cycles (push girth higher)
    Repeat, with decreasing log-det phase length.
    """

    @property
    def name(self) -> str:
        return "hybrid"

    def _log_spanning_trees(self, adj: np.ndarray) -> float:
        return _log_spanning_tree_count(adj)

    def _find_shortest_cycle_edge(self, adj, rng):
        """Find an edge on a shortest cycle."""
        N = len(adj)
        best_len = N + 1
        best_edge = None

        for source in range(N):
            dist = [-1] * N
            parent = [-1] * N
            dist[source] = 0
            queue = deque([source])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    v = int(v)
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and dist[u] + dist[v] + 1 < best_len:
                        best_len = dist[u] + dist[v] + 1
                        best_edge = (u, v)
                if dist[u] >= best_len // 2:
                    break
            if best_len == 3:
                break
        return best_edge, best_len

    def run(self, adj: np.ndarray, task: 'CubicGirthTask',
            budget: int, rng: np.random.Generator) -> OptimizerRun:
        budget = _positive_work_budget(budget)
        adj_best = adj.copy()
        best_girth_val = task.evaluate(adj_best)
        values = [best_girth_val]
        surrogate_evals = 0
        if budget == 1:
            return OptimizerRun(
                solution=adj_best,
                values=values,
                solution_value=best_girth_val,
                objective_evals=1,
            )

        # Phase 1: log-det maximization
        adj_curr = adj_best.copy()
        curr_log_tau = self._log_spanning_trees(adj_curr)
        surrogate_evals = 1
        phase_one_limit = 2 + int((budget - 2) * 0.6)
        for _ in range(budget):
            if len(values) + surrogate_evals + 2 > phase_one_limit:
                break
            edges = list(adj_list_to_edge_set(adj_curr))
            if len(edges) < 2:
                break
            idx = rng.choice(len(edges), size=2, replace=False)
            result = edge_swap(adj_curr, edges[idx[0]], edges[idx[1]], rng.integers(2))
            if result is not None:
                log_tau = self._log_spanning_trees(result)
                surrogate_evals += 1
                if log_tau > curr_log_tau:
                    adj_curr = result
                    curr_log_tau = log_tau
            gv = task.evaluate(adj_curr)
            values.append(gv)
            if gv > best_girth_val:
                adj_best = adj_curr.copy()
                best_girth_val = gv

        # Phase 2: cycle breaking from best log-det graph
        adj_curr = adj_best.copy()
        for _ in range(budget):
            if len(values) + surrogate_evals >= budget:
                break
            cycle_edge, girth = self._find_shortest_cycle_edge(adj_curr, rng)
            if cycle_edge is None:
                continue

            # Try swapping the cycle edge with random non-adjacent edges
            all_edges = list(adj_list_to_edge_set(adj_curr))
            rng.shuffle(all_edges)
            improved = False
            for ne in all_edges[:10]:
                if ne == cycle_edge or (ne[1], ne[0]) == cycle_edge:
                    continue
                for mode in [0, 1]:
                    if len(values) + surrogate_evals >= budget:
                        break
                    result = edge_swap(adj_curr, cycle_edge, ne, mode)
                    if result is not None:
                        gv = task.evaluate(result)
                        values.append(gv)
                        if gv > best_girth_val:
                            adj_curr = result
                            adj_best = result.copy()
                            best_girth_val = gv
                            improved = True
                            break
                if improved:
                    break

        return OptimizerRun(
            solution=adj_best,
            values=values,
            solution_value=best_girth_val,
            objective_evals=len(values),
            surrogate_evals=surrogate_evals,
        )


# =====================================================================
# Known constructions
# =====================================================================


class KnownCageOptimizer(InnerOptimizer):
    """Use known constructions for specific girths."""

    @property
    def name(self) -> str:
        return "known"

    def _petersen(self) -> np.ndarray:
        """Petersen graph: 10 vertices, girth 5."""
        adj = np.array([
            [1, 4, 5], [0, 2, 6], [1, 3, 7], [2, 4, 8], [3, 0, 9],
            [0, 7, 8], [1, 8, 9], [2, 5, 9], [3, 5, 6], [4, 6, 7]
        ], dtype=np.int32)
        return adj

    def _heawood(self) -> np.ndarray:
        """Heawood graph: 14 vertices, girth 6."""
        # Incidence graph of Fano plane
        edges = [
            (0,1),(0,5),(0,13), (1,2),(1,10), (2,3),(2,7),
            (3,4),(3,12), (4,5),(4,9), (5,6), (6,7),(6,11),
            (7,8), (8,9),(8,13), (9,10), (10,11), (11,12), (12,13)
        ]
        adj = [[] for _ in range(14)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return np.array([sorted(a)[:3] for a in adj], dtype=np.int32)

    def run(self, adj: np.ndarray, task: CubicGirthTask,
            budget: int, rng: np.random.Generator) -> OptimizerRun:
        budget = _positive_work_budget(budget)
        N = task.N
        best = adj.copy()
        best_val = task.evaluate(best)
        values = [best_val]

        # Try known constructions if N matches
        if N == 10 and len(values) < budget:
            H = self._petersen()
            v = task.evaluate(H)
            values.append(v)
            if v > best_val:
                best = H
                best_val = v
        elif N == 14 and len(values) < budget:
            H = self._heawood()
            v = task.evaluate(H)
            values.append(v)
            if v > best_val:
                best = H
                best_val = v

        return OptimizerRun(
            solution=best,
            values=values,
            solution_value=best_val,
            objective_evals=len(values),
        )


# =====================================================================
# Environment factory
# =====================================================================

def make_cage_env(
    N_values: List[int] = None,
    horizon: int = 60,
    eval_cost_weight: float = 0.0005,
    seed: Optional[int] = None,
) -> GenericOptEnv:
    """Create a Gymnasium environment for cubic girth maximization."""
    sampler = CubicGirthTaskSampler(N_values)
    optimizers = [
        RandomSwapOptimizer(),
        ShortCycleBreaker(),
        AnnealSwapOptimizer(t_start=3.0, t_end=0.01),
        MultiSwapOptimizer(k=3),
        SpanningTreeOptimizer(),
        AnnealLogDetOptimizer(t_start=1.0, t_end=0.01),
        HybridGirthLogDetOptimizer(),
    ]
    return GenericOptEnv(
        task_sampler=sampler,
        inner_optimizers=optimizers,
        horizon=horizon,
        eval_cost_weight=eval_cost_weight,
        seed=seed,
    )
