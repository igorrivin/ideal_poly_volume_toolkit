"""
Ideal polytope task for the generic RL harness.

A task instance is a specific (d, N) pair.  Its solution is a
`PolytopeSolution` (vertices on S^{d-1} + Delaunay simplices).  Its objective
is dimension-specific per the design (docs/plans/rl-harness-design.md §3.2a):

  d = 3, 4, 6   →   closed-form hyperbolic volume (fast)
  d = 5         →   Euclidean proxy objective (true H^5 scoring is too slow)

`evaluate_surrogate` returns Euclidean volume for all dimensions, for
optimizers that prefer a uniformly cheap proxy (currently unused but
exposed for future use).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.spatial import ConvexHull, Delaunay

from ..generic import OptimizationTask, TaskSampler
from .types import PolytopeSolution


# =====================================================================
# Helpers
# =====================================================================

def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms == 0.0) or not np.all(np.isfinite(norms)):
        raise ValueError("sphere points must be finite and nonzero")
    return arr / norms


def _random_on_sphere(N: int, d: int,
                      rng: np.random.Generator) -> np.ndarray:
    pts = rng.standard_normal((N, d))
    return _normalize_rows(pts)


def _perturb_on_sphere(pts: np.ndarray, radius: float,
                       rng: np.random.Generator) -> np.ndarray:
    return _normalize_rows(pts + rng.standard_normal(pts.shape) * radius)


def _stereographic_delaunay(pts: np.ndarray) -> List[List[int]]:
    """Cone from vertex 0 over a stereo-projected Delaunay triangulation
    of the remaining vertices.  Matches the old rl_optimizer.triangulate
    helper byte-for-byte."""
    N, d = pts.shape
    pole = pts[0]
    others = np.delete(pts, 0, axis=0)
    basis = []
    for i in range(d):
        e = np.zeros(d)
        e[i] = 1.0
        e -= np.dot(e, pole) * pole
        for b in basis:
            e -= np.dot(e, b) * b
        n = np.linalg.norm(e)
        if n > 1e-10:
            basis.append(e / n)
        if len(basis) == d - 1:
            break
    basis = np.array(basis)
    proj = []
    for p in others:
        dd = 1.0 - np.dot(p, pole)
        if dd <= 0.0:
            raise ValueError("stereographic projection pole is repeated")
        proj.append(basis @ ((p - np.dot(p, pole) * pole) / dd))
    proj = np.array(proj)
    tri = Delaunay(proj)
    return [[0] + [s + 1 for s in simp] for simp in tri.simplices]


def _euclidean_hull_volume(pts: np.ndarray) -> float:
    try:
        return float(ConvexHull(pts).volume)
    except Exception:
        return 0.0


def _current_triangulation(sol: PolytopeSolution) -> List[List[int]]:
    """Return a triangulation certified for the solution's current vertices.

    ``PolytopeSolution`` remains mutable for compatibility, so a nonempty
    simplex list alone cannot prove that it belongs to the current vertex
    array. A byte key makes repeated scoring cheap while forcing a rebuild
    after in-place vertex edits or untrusted simplex replacement.
    """
    vertices = _normalize_rows(sol.vertices)
    key = np.ascontiguousarray(vertices, dtype=np.float64).tobytes()
    signature = tuple(tuple(int(index) for index in simplex) for simplex in sol.simplices)
    if (
        sol.simplices
        and sol._simplices_key == key
        and sol._simplices_signature == signature
    ):
        return sol.simplices

    simplices = _stereographic_delaunay(vertices)
    sol.simplices = [list(simplex) for simplex in simplices]
    sol._simplices_key = key
    sol._simplices_signature = tuple(
        tuple(int(index) for index in simplex) for simplex in sol.simplices
    )
    return sol.simplices


# =====================================================================
# Dimension-specific hyperbolic-volume scoring
# =====================================================================

def _hyperbolic_volume(dim: int, sol: PolytopeSolution) -> float:
    """Return the scoring objective for dimension `dim`.

    H³ and H⁴ build the structures they need internally.  H⁶ consumes a
    supplied triangulation when present and otherwise constructs it lazily;
    every operation that moves vertices clears the cached simplices.
    """
    if dim == 3:
        return _h3_volume(sol.vertices)
    if dim == 4:
        return _h4_volume(sol.vertices)
    if dim == 5:
        # Euclidean proxy: true H^5 scoring is too slow for inner-loop use.
        # Existing experiments found the same optimum on the tested cases,
        # but that empirical agreement is not a functional identity between
        # Euclidean and hyperbolic volume.
        return _euclidean_hull_volume(sol.vertices)
    if dim == 6:
        try:
            simps = _current_triangulation(sol)
        except Exception:
            return 0.0
        return _h6_volume(sol.vertices, simps)
    raise ValueError(f"Unsupported dimension for polytope task: {dim}")


def _h3_volume(pts_S2: np.ndarray) -> float:
    """Closed-form H^3 volume via Bloch–Wigner dilogarithm."""
    from ...geometry import ideal_poly_volume_from_sphere

    try:
        volume = ideal_poly_volume_from_sphere(pts_S2, pole_index=0)
    except Exception:
        return 0.0
    return float(volume) if np.isfinite(volume) else 0.0


def _h4_volume(pts_S3: np.ndarray) -> float:
    """Closed-form H^4 volume via `ideal_polytope_volume_4d`."""
    from ...geometry_4d import ideal_polytope_volume_4d
    try:
        vol, _ = ideal_polytope_volume_4d(pts_S3)
        return float(vol) if np.isfinite(vol) else 0.0
    except Exception:
        return 0.0


def _h6_volume(pts_S5: np.ndarray, simplices: List[List[int]]) -> float:
    """Closed-form H^6 volume via `ideal_h6_polytope_volume` (torch)."""
    import torch
    from ...murakami_torch import ideal_h6_polytope_volume
    from ...li2_torch import li2 as li2_pure
    try:
        t_pts = torch.tensor(pts_S5, dtype=torch.float64)
        t_sim = torch.tensor(simplices, dtype=torch.long)
        with torch.no_grad():
            v = ideal_h6_polytope_volume(t_pts, t_sim, li2_fn=li2_pure).item()
        return float(v) if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


# =====================================================================
# Task
# =====================================================================

class IdealPolytopeTask(OptimizationTask):
    """Find a configuration of N ideal points maximizing hyperbolic
    volume in H^d.

    The combinatorial triangulation is computed lazily.  Any operation that
    changes vertex positions clears it, avoiding both stale topology and
    duplicate Delaunay calls in continuous-optimizer objective evaluations.
    """

    def __init__(self, d: int, N: int):
        if d not in (3, 4, 5, 6):
            raise ValueError(f"Unsupported dimension {d}; expected 3, 4, 5, or 6")
        self.d = int(d)
        self.N = int(N)

    # ---- Core OptimizationTask contract -----------------------------------

    def random_solution(self, rng: np.random.Generator) -> PolytopeSolution:
        verts = _random_on_sphere(self.N, self.d, rng)
        return PolytopeSolution(vertices=verts, simplices=[])

    def evaluate(self, sol: PolytopeSolution) -> float:
        return _hyperbolic_volume(self.d, sol)

    def evaluate_surrogate(self, sol: PolytopeSolution) -> float:
        # Uniform fast proxy: Euclidean hull volume for all dimensions.
        return _euclidean_hull_volume(sol.vertices)

    def has_distinct_surrogate(self) -> bool:
        # In d=5 the canonical objective is this exact Euclidean hull volume.
        return self.d != 5

    def perturb(self, sol: PolytopeSolution, radius: float,
        rng: np.random.Generator) -> PolytopeSolution:
        verts = _perturb_on_sphere(sol.vertices, radius, rng)
        return PolytopeSolution(vertices=verts, simplices=[])

    def metadata(self) -> np.ndarray:
        return np.array(
            [self.d, self.N, np.log(self.N * self.d)], dtype=np.float32
        )

    def copy_solution(self, sol: PolytopeSolution) -> PolytopeSolution:
        return PolytopeSolution(
            vertices=sol.vertices.copy(),
            simplices=[list(s) for s in sol.simplices],
            _simplices_key=sol._simplices_key,
            _simplices_signature=sol._simplices_signature,
        )

    # ---- Continuous-optimizer hooks ---------------------------------------

    def to_vector(self, sol: PolytopeSolution) -> np.ndarray:
        return sol.vertices.ravel().astype(float)

    def from_vector(self, vec: np.ndarray) -> PolytopeSolution:
        verts = _normalize_rows(vec.reshape(self.N, self.d))
        return PolytopeSolution(vertices=verts, simplices=[])

    def project(self, vec: np.ndarray) -> np.ndarray:
        verts = _normalize_rows(vec.reshape(self.N, self.d))
        return verts.ravel()

    def triangulate(self, vertices: np.ndarray) -> List[List[int]]:
        """Construct the dimension-appropriate stereographic triangulation."""
        return _stereographic_delaunay(_normalize_rows(vertices))


# =====================================================================
# Sampler
# =====================================================================

class IdealPolytopeTaskSampler(TaskSampler):
    """Samples (d, N) pairs, mirroring the old env's sampling logic:
    pick d uniformly in `d_range`, then N uniformly in
    [N_range[0], min(N_range[1], 5*d)].
    """

    def __init__(self, d_range: Tuple[int, int] = (4, 6),
                 N_range: Tuple[int, int] = (7, 25)):
        for d in range(d_range[0], d_range[1] + 1):
            if d not in (3, 4, 5, 6):
                raise ValueError(f"Unsupported dimension in d_range: {d}")
        self.d_range = d_range
        self.N_range = N_range

    def sample(self, rng: np.random.Generator) -> IdealPolytopeTask:
        d = int(rng.integers(self.d_range[0], self.d_range[1] + 1))
        # Soft cap at 5*d, but never below the requested N_range[0] —
        # this lets fixed-(d,N) eval envs pass N_range=(N, N) even when
        # N > 5*d (e.g. d=3, N=20).
        N_hi = max(self.N_range[0], min(self.N_range[1], 5 * d))
        N = int(rng.integers(self.N_range[0], N_hi + 1))
        return IdealPolytopeTask(d=d, N=N)

    def metadata_dim(self) -> int:
        return 3
