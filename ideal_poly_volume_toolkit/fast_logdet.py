"""
Fast incremental log-det computation for graph Laplacians.

Maintains the Cholesky factor of the reduced Laplacian and updates
it via rank-2 (edge add/remove) operations. An edge swap is two
rank-2 updates = rank-4 total.

For a cubic graph on N vertices:
  - Full eigendecomposition: O(N^3)
  - Incremental update: O(N^2) for Cholesky update, O(1) for det ratio

With numba JIT, the inner loops run at compiled speed.
"""

import numpy as np
import numba as nb
from numba import float64, int32, boolean


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


# =====================================================================
# The main class
# =====================================================================

class IncrementalLogDet:
    """Maintains Cholesky of reduced Laplacian for incremental log-det updates.

    Usage:
        ild = IncrementalLogDet(adj)  # O(N^3) initial Cholesky
        log_tau = ild.log_spanning_trees()  # O(1) from cached Cholesky

        # Test an edge swap without committing
        delta = ild.swap_delta(remove_e1, remove_e2, add_e1, add_e2)

        # Commit the swap
        ild.apply_swap(remove_e1, remove_e2, add_e1, add_e2)  # O(N^2)
    """

    def __init__(self, adj: np.ndarray):
        """Initialize from adjacency list (N, 3) of a cubic graph."""
        self.N = len(adj)
        self.adj = adj.copy()
        n = self.N - 1

        # Build reduced Laplacian and Cholesky
        L_red = _build_reduced_laplacian(adj.astype(np.int32), self.N)
        try:
            self._chol = np.linalg.cholesky(L_red)
        except np.linalg.LinAlgError:
            # Graph might be disconnected; fall back
            self._chol = np.eye(n)

        self._log_det = _log_det_from_cholesky(self._chol)

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
        chol_tmp = self._chol.copy()
        n = self.N - 1

        # Remove e1: downdate by (e_a - e_b)
        a, b = e1
        vec = np.zeros(n)
        if a < n: vec[a] += 1.0
        if b < n: vec[b] -= 1.0
        ok = _cholesky_rank1_update(chol_tmp, vec, -1.0)
        if not ok:
            return -np.inf

        # Remove e2: downdate by (e_c - e_d)
        c, d = e2
        vec = np.zeros(n)
        if c < n: vec[c] += 1.0
        if d < n: vec[d] -= 1.0
        ok = _cholesky_rank1_update(chol_tmp, vec, -1.0)
        if not ok:
            return -np.inf

        # Add new_e1: update by (e_u - e_v)
        u, v = new_e1
        vec = np.zeros(n)
        if u < n: vec[u] += 1.0
        if v < n: vec[v] -= 1.0
        ok = _cholesky_rank1_update(chol_tmp, vec, 1.0)
        if not ok:
            return -np.inf

        # Add new_e2
        u, v = new_e2
        vec = np.zeros(n)
        if u < n: vec[u] += 1.0
        if v < n: vec[v] -= 1.0
        ok = _cholesky_rank1_update(chol_tmp, vec, 1.0)
        if not ok:
            return -np.inf

        new_log_det = _log_det_from_cholesky(chol_tmp)
        return new_log_det - self._log_det

    def apply_swap(self, e1: tuple, e2: tuple,
                    new_e1: tuple, new_e2: tuple) -> bool:
        """Apply an edge swap and update the Cholesky factor.

        Returns True if successful.
        """
        n = self.N - 1

        # Remove e1
        a, b = e1
        vec = np.zeros(n)
        if a < n: vec[a] += 1.0
        if b < n: vec[b] -= 1.0
        ok = _cholesky_rank1_update(self._chol, vec, -1.0)
        if not ok:
            self._rebuild()
            return False

        # Remove e2
        c, d = e2
        vec = np.zeros(n)
        if c < n: vec[c] += 1.0
        if d < n: vec[d] -= 1.0
        ok = _cholesky_rank1_update(self._chol, vec, -1.0)
        if not ok:
            self._rebuild()
            return False

        # Add new_e1
        u, v = new_e1
        vec = np.zeros(n)
        if u < n: vec[u] += 1.0
        if v < n: vec[v] -= 1.0
        ok = _cholesky_rank1_update(self._chol, vec, 1.0)
        if not ok:
            self._rebuild()
            return False

        # Add new_e2
        u, v = new_e2
        vec = np.zeros(n)
        if u < n: vec[u] += 1.0
        if v < n: vec[v] -= 1.0
        ok = _cholesky_rank1_update(self._chol, vec, 1.0)
        if not ok:
            self._rebuild()
            return False

        self._log_det = _log_det_from_cholesky(self._chol)
        return True

    def _rebuild(self):
        """Rebuild Cholesky from scratch (fallback after failed update)."""
        n = self.N - 1
        L_red = _build_reduced_laplacian(self.adj.astype(np.int32), self.N)
        try:
            self._chol = np.linalg.cholesky(L_red)
        except np.linalg.LinAlgError:
            self._chol = np.eye(n)
        self._log_det = _log_det_from_cholesky(self._chol)


# =====================================================================
# Benchmark
# =====================================================================

def benchmark(N_values=None):
    """Compare full eigendecomposition vs incremental update timing."""
    import time

    if N_values is None:
        N_values = [20, 40, 80, 160, 320]

    from .rl_cages import random_cubic_graph, adj_list_to_edge_set, edge_swap

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
