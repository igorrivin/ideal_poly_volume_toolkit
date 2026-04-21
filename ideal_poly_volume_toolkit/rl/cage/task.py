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

import math
from typing import List, Tuple, Optional, Any
from collections import deque

import numpy as np

from ..generic import (
    OptimizationTask, InnerOptimizer, TaskSampler, GenericOptEnv
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
        """Count the number of shortest cycles."""
        N = len(adj)
        count = 0
        for source in range(N):
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
                    elif v != parent[u] and dist[u] + dist[v] + 1 == girth:
                        count += 1
                if dist[u] >= girth // 2:
                    break
        # Each cycle counted multiple times
        return count // (2 * girth)

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

class RandomSwapOptimizer(InnerOptimizer):
    """Random edge swaps, accept if girth improves or stays same."""

    @property
    def name(self) -> str:
        return "rand_swap"

    def run(self, adj: np.ndarray, task: CubicGirthTask,
            budget: int, rng: np.random.Generator) -> Tuple[np.ndarray, List[float]]:
        adj_best = adj.copy()
        best_val = task.evaluate(adj_best)
        values = [best_val]

        edges = list(adj_list_to_edge_set(adj_best))
        for _ in range(budget):
            if len(edges) < 2:
                values.append(best_val)
                continue
            idx = rng.choice(len(edges), size=2, replace=False)
            e1, e2 = edges[idx[0]], edges[idx[1]]
            mode = rng.integers(2)
            result = edge_swap(adj_best, e1, e2, mode)
            if result is not None:
                v = task.evaluate(result)
                if v >= best_val:
                    adj_best = result
                    best_val = v
                    edges = list(adj_list_to_edge_set(adj_best))
            values.append(best_val)

        return adj_best, values


class ShortCycleBreaker(InnerOptimizer):
    """Find a shortest cycle and try to break it by swapping one of its edges."""

    @property
    def name(self) -> str:
        return "cycle_break"

    def _find_shortest_cycle(self, adj: np.ndarray) -> Optional[List[int]]:
        """Find one shortest cycle, return as list of vertices."""
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
                    elif v != parent[u] and dist[u] + dist[v] + 1 < best_len:
                        # Reconstruct cycle
                        cycle_len = dist[u] + dist[v] + 1
                        best_len = cycle_len
                        # Build the cycle path
                        path_u = []
                        x = u
                        while x != source:
                            path_u.append(x)
                            x = parent[x]
                        path_u.append(source)
                        path_u.reverse()

                        path_v = []
                        x = v
                        while x != source:
                            path_v.append(x)
                            x = parent[x]
                        path_v.append(source)
                        path_v.reverse()

                        best_cycle = path_u + path_v[-2:0:-1]

                if dist[u] >= best_len // 2:
                    break
            if best_len == 3:
                return best_cycle

        return best_cycle

    def run(self, adj: np.ndarray, task: CubicGirthTask,
            budget: int, rng: np.random.Generator) -> Tuple[np.ndarray, List[float]]:
        adj_best = adj.copy()
        best_val = task.evaluate(adj_best)
        values = [best_val]

        for _ in range(budget):
            cycle = self._find_shortest_cycle(adj_best)
            if cycle is None or len(cycle) < 3:
                values.append(best_val)
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
                    result = edge_swap(adj_best, ce, ne, mode)
                    if result is not None:
                        v = task.evaluate(result)
                        if v > best_val:
                            adj_best = result
                            best_val = v
                            improved = True
                            break
                if improved:
                    break

            values.append(best_val)

        return adj_best, values


class AnnealSwapOptimizer(InnerOptimizer):
    """Simulated annealing with edge swaps."""

    def __init__(self, t_start: float = 2.0, t_end: float = 0.01):
        self.t_start = t_start
        self.t_end = t_end

    @property
    def name(self) -> str:
        return "anneal_swap"

    def run(self, adj: np.ndarray, task: CubicGirthTask,
            budget: int, rng: np.random.Generator) -> Tuple[np.ndarray, List[float]]:
        adj_curr = adj.copy()
        curr_val = task.evaluate(adj_curr)
        adj_best = adj_curr.copy()
        best_val = curr_val
        values = [best_val]

        decay = (self.t_end / self.t_start) ** (1.0 / max(budget, 1))
        T = self.t_start

        for _ in range(budget):
            edges = list(adj_list_to_edge_set(adj_curr))
            if len(edges) < 2:
                values.append(best_val)
                continue

            idx = rng.choice(len(edges), size=2, replace=False)
            e1, e2 = edges[idx[0]], edges[idx[1]]
            mode = rng.integers(2)
            result = edge_swap(adj_curr, e1, e2, mode)

            if result is not None:
                trial_val = task.evaluate(result)
                delta = trial_val - curr_val
                if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                    adj_curr = result
                    curr_val = trial_val
                if curr_val > best_val:
                    adj_best = adj_curr.copy()
                    best_val = curr_val

            T *= decay
            values.append(best_val)

        return adj_best, values


class MultiSwapOptimizer(InnerOptimizer):
    """Multiple simultaneous edge swaps for larger moves."""

    def __init__(self, k: int = 3):
        self.k = k

    @property
    def name(self) -> str:
        return f"multi_{self.k}"

    def run(self, adj: np.ndarray, task: CubicGirthTask,
            budget: int, rng: np.random.Generator) -> Tuple[np.ndarray, List[float]]:
        adj_best = adj.copy()
        best_val = task.evaluate(adj_best)
        values = [best_val]

        for _ in range(budget):
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
                if v > best_val:
                    adj_best = adj_trial
                    best_val = v

            values.append(best_val)

        return adj_best, values


# =====================================================================
# Known constructions
# =====================================================================

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
        N = len(adj)
        # Build Laplacian
        L = np.zeros((N, N), dtype=np.float64)
        for v in range(N):
            for u in adj[v]:
                u = int(u)
                L[v, u] -= 1.0
                L[v, v] += 1.0

        # Eigenvalues of Laplacian
        eigvals = np.linalg.eigvalsh(L)
        # Sort; smallest should be ~0 (connected graph)
        eigvals = np.sort(eigvals)

        # log det' = sum of log of nonzero eigenvalues
        # Skip the smallest eigenvalue (≈0 for connected graph)
        nonzero = eigvals[1:]
        nonzero = nonzero[nonzero > 1e-10]

        if len(nonzero) == 0:
            return -np.inf

        return np.sum(np.log(nonzero)) - np.log(N)

    def run(self, adj: np.ndarray, task: 'CubicGirthTask',
            budget: int, rng: np.random.Generator) -> Tuple[np.ndarray, List[float]]:
        N = task.N
        adj_best = adj.copy()
        best_log_tau = self._log_spanning_trees(adj_best)
        best_girth_val = task.evaluate(adj_best)
        values = [best_girth_val]

        for _ in range(budget):
            edges = list(adj_list_to_edge_set(adj_best))
            if len(edges) < 2:
                values.append(best_girth_val)
                continue

            idx = rng.choice(len(edges), size=2, replace=False)
            e1, e2 = edges[idx[0]], edges[idx[1]]
            mode = rng.integers(2)
            result = edge_swap(adj_best, e1, e2, mode)

            if result is not None:
                log_tau = self._log_spanning_trees(result)
                if log_tau > best_log_tau:
                    adj_best = result
                    best_log_tau = log_tau
                    best_girth_val = task.evaluate(adj_best)

            values.append(best_girth_val)

        return adj_best, values


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
        N = len(adj)
        L = np.zeros((N, N), dtype=np.float64)
        for v in range(N):
            for u in adj[v]:
                u = int(u)
                L[v, u] -= 1.0
                L[v, v] += 1.0
        eigvals = np.sort(np.linalg.eigvalsh(L))
        nonzero = eigvals[1:]
        nonzero = nonzero[nonzero > 1e-10]
        if len(nonzero) == 0:
            return -np.inf
        return np.sum(np.log(nonzero)) - np.log(N)

    def run(self, adj: np.ndarray, task: 'CubicGirthTask',
            budget: int, rng: np.random.Generator) -> Tuple[np.ndarray, List[float]]:
        N = task.N
        adj_curr = adj.copy()
        curr_log_tau = self._log_spanning_trees(adj_curr)
        adj_best = adj_curr.copy()
        best_girth_val = task.evaluate(adj_best)
        best_log_tau = curr_log_tau
        values = [best_girth_val]

        decay = (self.t_end / self.t_start) ** (1.0 / max(budget, 1))
        T = self.t_start

        for _ in range(budget):
            edges = list(adj_list_to_edge_set(adj_curr))
            if len(edges) < 2:
                values.append(best_girth_val)
                continue

            idx = rng.choice(len(edges), size=2, replace=False)
            e1, e2 = edges[idx[0]], edges[idx[1]]
            mode = rng.integers(2)
            result = edge_swap(adj_curr, e1, e2, mode)

            if result is not None:
                log_tau = self._log_spanning_trees(result)
                delta = log_tau - curr_log_tau
                if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                    adj_curr = result
                    curr_log_tau = log_tau

                girth_val = task.evaluate(adj_curr)
                if girth_val > best_girth_val or (
                    girth_val == best_girth_val and curr_log_tau > best_log_tau
                ):
                    adj_best = adj_curr.copy()
                    best_girth_val = girth_val
                    best_log_tau = curr_log_tau

            T *= decay
            values.append(best_girth_val)

        return adj_best, values


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
        N = len(adj)
        L = np.zeros((N, N), dtype=np.float64)
        for v in range(N):
            for u in adj[v]:
                u = int(u)
                L[v, u] -= 1.0
                L[v, v] += 1.0
        eigvals = np.sort(np.linalg.eigvalsh(L))
        nonzero = eigvals[1:]
        nonzero = nonzero[nonzero > 1e-10]
        if len(nonzero) == 0:
            return -np.inf
        return np.sum(np.log(nonzero)) - np.log(N)

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
            budget: int, rng: np.random.Generator) -> Tuple[np.ndarray, List[float]]:
        adj_best = adj.copy()
        best_girth_val = task.evaluate(adj_best)
        values = [best_girth_val]

        # Split budget: 60% log-det, 40% cycle-breaking
        logdet_budget = int(budget * 0.6)
        girth_budget = budget - logdet_budget

        # Phase 1: log-det maximization
        adj_curr = adj_best.copy()
        curr_log_tau = self._log_spanning_trees(adj_curr)
        for _ in range(logdet_budget):
            edges = list(adj_list_to_edge_set(adj_curr))
            if len(edges) < 2:
                break
            idx = rng.choice(len(edges), size=2, replace=False)
            result = edge_swap(adj_curr, edges[idx[0]], edges[idx[1]], rng.integers(2))
            if result is not None:
                log_tau = self._log_spanning_trees(result)
                if log_tau > curr_log_tau:
                    adj_curr = result
                    curr_log_tau = log_tau
            gv = task.evaluate(adj_curr)
            if gv > best_girth_val:
                adj_best = adj_curr.copy()
                best_girth_val = gv
            values.append(best_girth_val)

        # Phase 2: cycle breaking from best log-det graph
        adj_curr = adj_best.copy()
        for _ in range(girth_budget):
            cycle_edge, girth = self._find_shortest_cycle_edge(adj_curr, rng)
            if cycle_edge is None:
                values.append(best_girth_val)
                continue

            # Try swapping the cycle edge with random non-adjacent edges
            all_edges = list(adj_list_to_edge_set(adj_curr))
            rng.shuffle(all_edges)
            improved = False
            for ne in all_edges[:10]:
                if ne == cycle_edge or (ne[1], ne[0]) == cycle_edge:
                    continue
                for mode in [0, 1]:
                    result = edge_swap(adj_curr, cycle_edge, ne, mode)
                    if result is not None:
                        gv = task.evaluate(result)
                        if gv > best_girth_val:
                            adj_curr = result
                            adj_best = result.copy()
                            best_girth_val = gv
                            improved = True
                            break
                if improved:
                    break
            values.append(best_girth_val)

        return adj_best, values


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
            budget: int, rng: np.random.Generator) -> Tuple[np.ndarray, List[float]]:
        N = task.N
        best = adj.copy()
        best_val = task.evaluate(best)

        # Try known constructions if N matches
        if N == 10:
            H = self._petersen()
            v = task.evaluate(H)
            if v > best_val:
                best = H; best_val = v
        elif N == 14:
            H = self._heawood()
            v = task.evaluate(H)
            if v > best_val:
                best = H; best_val = v

        return best, [best_val] * budget


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
