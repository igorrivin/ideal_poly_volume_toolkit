"""
Fast incremental log-det computation for graph Laplacians.

Maintains the Cholesky factor of the reduced Laplacian and updates
it via rank-2 (edge add/remove) operations. An edge swap is two
rank-2 updates = rank-4 total.

For a cubic graph on N vertices:
  - Full eigendecomposition: O(N^3)
  - Incremental update: O(N^2) for Cholesky update, O(1) for det ratio

With numba installed, the inner loops run at compiled speed; otherwise the
same implementation remains available through a pure-Python fallback.
"""

import numpy as np

try:
    import numba as nb
    from numba import float64
except ImportError:  # pragma: no cover - exercised in minimal installations
    class _NumbaFallback:
        @staticmethod
        def njit(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            return lambda function: function

    nb = _NumbaFallback()
    float64 = np.float64


# =====================================================================
# Numba-accelerated Cholesky rank-1 update/downdate
# =====================================================================

@nb.njit(cache=True)
def _cholesky_rank1_update(L, x, sign=1.0):
    """In-place rank-1 update of Cholesky factor L.

    Given L L^T = A, compute the Cholesky of A + sign * x x^T.
    sign = +1 for update, -1 for downdate.

    L is modified in-place. Returns True if successful.
    """
    n = L.shape[0]
    x_work = x.copy()
    for k in range(n):
        r2 = L[k, k] ** 2 + sign * x_work[k] ** 2
        if r2 <= 0.0:
            return False  # downdate failed (matrix not PD)
        r = np.sqrt(r2)
        c = r / L[k, k]
        s = x_work[k] / L[k, k]
        L[k, k] = r
        if k + 1 < n:
            for i in range(k + 1, n):
                L[i, k] = (L[i, k] + sign * s * x_work[i]) / c
                x_work[i] = c * x_work[i] - s * L[i, k]
    return True


@nb.njit(cache=True)
def _log_det_from_cholesky(L):
    """Compute log(det(A)) = 2 * sum(log(diag(L))) from Cholesky L."""
    n = L.shape[0]
    result = 0.0
    for i in range(n):
        if L[i, i] <= 0.0:
            return -np.inf
        result += np.log(L[i, i])
    return 2.0 * result


# =====================================================================
# Reduced Laplacian and its Cholesky
# =====================================================================

@nb.njit(cache=True)
def _build_reduced_laplacian(adj, N):
    """Build the reduced Laplacian (delete last row/col) from adjacency list.

    Returns (N-1) x (N-1) matrix.
    """
    L_red = np.zeros((N - 1, N - 1), dtype=float64)
    for v in range(N - 1):
        for k in range(3):  # cubic graph, degree 3
            u = adj[v, k]
            if u < N - 1:
                L_red[v, u] -= 1.0
            L_red[v, v] += 1.0
    return L_red


@nb.njit(cache=True)
def _edge_swap_laplacian_vectors(u1, v1, u2, v2, N):
    """Compute the rank-1 vectors for an edge swap.

    Remove edge (u1,v1), add edge (u2,v2).
    The Laplacian change is:
      ΔL = -(e_{u1}-e_{v1})(e_{u1}-e_{v1})^T + (e_{u2}-e_{v2})(e_{u2}-e_{v2})^T

    For the reduced Laplacian (last row/col deleted), we truncate
    the vectors to length N-1.

    Returns (remove_vec, add_vec) each of length N-1.
    """
    n = N - 1
    remove_vec = np.zeros(n, dtype=float64)
    add_vec = np.zeros(n, dtype=float64)

    if u1 < n:
        remove_vec[u1] += 1.0
    if v1 < n:
        remove_vec[v1] -= 1.0

    if u2 < n:
        add_vec[u2] += 1.0
    if v2 < n:
        add_vec[v2] -= 1.0

    return remove_vec, add_vec


def _is_connected(adj: np.ndarray) -> bool:
    """Return whether an adjacency-list graph is connected."""
    N = len(adj)
    if N == 0:
        return False
    seen = np.zeros(N, dtype=bool)
    seen[0] = True
    stack = [0]
    while stack:
        vertex = stack.pop()
        for neighbour in adj[vertex]:
            neighbour = int(neighbour)
            if neighbour < 0 or neighbour >= N:
                return False
            if not seen[neighbour]:
                seen[neighbour] = True
                stack.append(neighbour)
    return bool(seen.all())


def _swapped_adjacency(adj: np.ndarray, removed_edges, added_edges):
    """Build and validate the adjacency list after a simple-graph 2-switch."""
    N = len(adj)
    neighbours = [set(int(u) for u in row) for row in adj]
    if any(len(row) != 3 for row in neighbours):
        return None

    for u, v in removed_edges:
        u, v = int(u), int(v)
        if not (0 <= u < N and 0 <= v < N):
            return None
        if v not in neighbours[u] or u not in neighbours[v]:
            return None
        neighbours[u].remove(v)
        neighbours[v].remove(u)

    for u, v in added_edges:
        u, v = int(u), int(v)
        if not (0 <= u < N and 0 <= v < N) or u == v:
            return None
        if v in neighbours[u] or u in neighbours[v]:
            return None
        neighbours[u].add(v)
        neighbours[v].add(u)

    if any(len(row) != 3 for row in neighbours):
        return None
    return np.asarray([sorted(row) for row in neighbours], dtype=adj.dtype)


# =====================================================================
# The main class
# =====================================================================

class IncrementalLogDet:
    """Maintains Cholesky of reduced Laplacian for incremental log-det updates.

    Usage:
        ild = IncrementalLogDet(adj)  # O(N^3) initial Cholesky
        log_tau = ild.log_spanning_trees()  # O(1) from cached Cholesky

        # Test an edge swap without committing
        delta = ild.swap_log_det_ratio(remove_e1, remove_e2, add_e1, add_e2)

        # Commit the swap
        ild.apply_swap(remove_e1, remove_e2, add_e1, add_e2)  # O(N^2)
    """

    def __init__(self, adj: np.ndarray):
        """Initialize from adjacency list (N, 3) of a cubic graph."""
        self.N = len(adj)
        self.adj = adj.copy()
        self._chol = None
        self._log_det = -np.inf
        self._rebuild()

    def log_spanning_trees(self) -> float:
        """Return log(spanning tree count) = log(det(L_reduced))."""
        return self._log_det

    def swap_log_det_ratio(self, e1: tuple, e2: tuple,
                            new_e1: tuple, new_e2: tuple) -> float:
        """Compute log-det change from an edge swap WITHOUT modifying state.

        Remove edges e1=(a,b) and e2=(c,d).
        Add edges new_e1 and new_e2.

        Returns delta = log_det_new - log_det_old.
        Uses a temporary copy of the Cholesky factor.
        """
        candidate = _swapped_adjacency(
            self.adj, (e1, e2), (new_e1, new_e2))
        if candidate is None or not _is_connected(candidate):
            return -np.inf

        chol_tmp = self._incremental_swap_cholesky(e1, e2, new_e1, new_e2)
        if chol_tmp is None:
            # A valid final graph can have a singular intermediate graph after
            # the two removals.  Fall back to a direct factorization rather
            # than incorrectly rejecting that swap.
            chol_tmp = self._factor_adjacency(candidate)
        if chol_tmp is None:
            return -np.inf

        new_log_det = _log_det_from_cholesky(chol_tmp)
        return new_log_det - self._log_det

    def apply_swap(self, e1: tuple, e2: tuple,
                    new_e1: tuple, new_e2: tuple) -> bool:
        """Apply an edge swap and update the Cholesky factor.

        Returns True if successful.
        """
        candidate = _swapped_adjacency(
            self.adj, (e1, e2), (new_e1, new_e2))
        if candidate is None or not _is_connected(candidate):
            return False

        chol_tmp = self._incremental_swap_cholesky(e1, e2, new_e1, new_e2)
        if chol_tmp is None:
            chol_tmp = self._factor_adjacency(candidate)
        if chol_tmp is None:
            return False

        # Commit the factor and graph together.  _rebuild() now reconstructs
        # this accepted graph rather than silently reverting to the initial one.
        self.adj = candidate
        self._chol = chol_tmp
        self._log_det = _log_det_from_cholesky(chol_tmp)
        return True

    def _incremental_swap_cholesky(self, e1, e2, new_e1, new_e2):
        """Return the updated factor without mutating state, or None."""
        if self._chol is None:
            return None
        chol_tmp = self._chol.copy()
        n = self.N - 1
        for (u, v), sign in (
            (e1, -1.0), (e2, -1.0),
            (new_e1, 1.0), (new_e2, 1.0),
        ):
            vec = np.zeros(n)
            if u < n:
                vec[u] += 1.0
            if v < n:
                vec[v] -= 1.0
            if not _cholesky_rank1_update(chol_tmp, vec, sign):
                return None
        return chol_tmp

    def _factor_adjacency(self, adj):
        """Factor a connected graph's reduced Laplacian, or return None."""
        if not _is_connected(adj):
            return None
        L_red = _build_reduced_laplacian(adj.astype(np.int32), self.N)
        try:
            return np.linalg.cholesky(L_red)
        except np.linalg.LinAlgError:
            return None

    def _rebuild(self):
        """Rebuild Cholesky from scratch (fallback after failed update)."""
        self._chol = self._factor_adjacency(self.adj)
        self._log_det = (
            _log_det_from_cholesky(self._chol)
            if self._chol is not None else -np.inf
        )


# =====================================================================
# Benchmark
# =====================================================================

def benchmark(N_values=None):
    """Compare full eigendecomposition vs incremental update timing."""
    import time

    if N_values is None:
        N_values = [20, 40, 80, 160, 320]

    from .rl.cage.task import adj_list_to_edge_set, random_cubic_graph

    print(f"{'N':>5s} {'full eigvalsh':>14s} {'incremental':>14s} {'speedup':>8s}")
    print("-" * 45)

    for N in N_values:
        rng = np.random.default_rng(42)
        adj = random_cubic_graph(N, rng)
        edges = list(adj_list_to_edge_set(adj))

        # Full eigendecomposition
        n_trials = min(500, max(10, 100000 // N**2))
        t0 = time.perf_counter()
        for _ in range(n_trials):
            L = np.zeros((N, N))
            for v in range(N):
                for u in adj[v]:
                    L[v, int(u)] -= 1.0
                    L[v, v] += 1.0
            eigvals = np.linalg.eigvalsh(L)
            _ = np.sum(np.log(eigvals[1:][eigvals[1:] > 1e-10]))
        t_full = (time.perf_counter() - t0) / n_trials

        # Incremental
        ild = IncrementalLogDet(adj)
        t0 = time.perf_counter()
        for _ in range(n_trials):
            idx = rng.choice(len(edges), size=2, replace=False)
            e1, e2 = edges[idx[0]], edges[idx[1]]
            # Test swap without committing
            _ = ild.swap_log_det_ratio(e1, e2, (e1[0], e2[0]), (e1[1], e2[1]))
        t_incr = (time.perf_counter() - t0) / n_trials

        speedup = t_full / t_incr if t_incr > 0 else float('inf')
        print(f"{N:5d} {t_full*1000:12.3f}ms {t_incr*1000:12.3f}ms {speedup:7.1f}x")
