"""
Ideal polytope task for the generic RL harness.

A task instance is a specific (d, N) pair.  Its solution is a
`PolytopeSolution` (vertices on S^{d-1} + Delaunay simplices).  Its objective
is dimension-specific per the design (docs/plans/rl-harness-design.md §3.2a):

  d = 3, 4, 6   →   closed-form hyperbolic volume (fast)
  d = 5         →   Euclidean volume (paper §7.7: ≈ hyperbolic, MC is too slow)

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
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


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
        dd = max(1.0 - np.dot(p, pole), 1e-14)
        proj.append(basis @ ((p - np.dot(p, pole) * pole) / dd))
    proj = np.array(proj)
    tri = Delaunay(proj)
    return [[0] + [s + 1 for s in simp] for simp in tri.simplices]


def _euclidean_hull_volume(pts: np.ndarray) -> float:
    try:
        return float(ConvexHull(pts).volume)
    except Exception:
        return 0.0


# =====================================================================
# Dimension-specific hyperbolic-volume scoring
# =====================================================================

def _hyperbolic_volume(dim: int, sol: PolytopeSolution) -> float:
    """Return the scoring objective for dimension `dim`.

    For d=6 we defensively retriangulate the vertex set: inner optimizers
    may mutate the vertex positions while carrying along stale simplex
    indices (e.g., LBFGS returns a moved config with the entry-point
    triangulation), and the H^6 volume formula is sensitive to mismatch
    between simplices and vertex positions.  `_h3_volume` and `_h4_volume`
    build their own triangulations internally; `_euclidean_hull_volume`
    doesn't need one.
    """
    if dim == 3:
        return _h3_volume(sol.vertices)
    if dim == 4:
        return _h4_volume(sol.vertices)
    if dim == 5:
        # Euclidean surrogate: true H^5 is Schläfli MC and too slow for
        # inner-loop scoring.  Paper §7.7 establishes Eucl = Hyp to full
        # precision at d = 5, so no hyperbolic signal is lost.
        return _euclidean_hull_volume(sol.vertices)
    if dim == 6:
        simps = sol.simplices
        if not simps or len(simps[0]) != 7:
            try:
                simps = _stereographic_delaunay(sol.vertices)
            except Exception:
                return 0.0
        else:
            # Cheap triangulation refresh — matches the d=3 / d=4 paths.
            try:
                simps = _stereographic_delaunay(sol.vertices)
            except Exception:
                pass  # fall back to the stored simplices
        return _h6_volume(sol.vertices, simps)
    raise ValueError(f"Unsupported dimension for polytope task: {dim}")


def _h3_volume(pts_S2: np.ndarray) -> float:
    """Closed-form H^3 volume via Bloch–Wigner dilogarithm."""
    from scipy.special import spence
    N = pts_S2.shape[0]
    north_dist = [abs(p[2] - 1.0) for p in pts_S2]
    inf_idx = int(np.argmin(north_dist))
    z_arr = []
    for i in range(N):
        if i == inf_idx:
            continue
        p = pts_S2[i]
        denom = max(1.0 - p[2], 1e-12)
        z_arr.append(complex(p[0] / denom, p[1] / denom))
    z_arr = np.array(z_arr)
    pts_2d = np.column_stack([z_arr.real, z_arr.imag])
    try:
        tri = Delaunay(pts_2d)
    except Exception:
        return 0.0
    total = 0.0
    for sx in tri.simplices:
        z1, z2, z3 = z_arr[sx[0]], z_arr[sx[1]], z_arr[sx[2]]
        cr = (z1 - z3) / (z2 - z3) if abs(z2 - z3) > 1e-15 else 0.0
        li2 = spence(1.0 - cr)
        total += abs(
            li2.imag + np.angle(1.0 - cr) * np.log(max(abs(cr), 1e-300))
        )
    return float(total)


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

    The combinatorial triangulation is the stereographic Delaunay
    triangulation of the vertex set; it is recomputed whenever the
    vertices move far enough that the old one might be wrong.  For most
    local moves (`perturb` with small radius, LBFGS gradient steps) the
    simplices are stable and we keep them.
    """

    def __init__(self, d: int, N: int):
        if d not in (3, 4, 5, 6):
            raise ValueError(f"Unsupported dimension {d}; expected 3, 4, 5, or 6")
        self.d = int(d)
        self.N = int(N)

    # ---- Core OptimizationTask contract -----------------------------------

    def random_solution(self, rng: np.random.Generator) -> PolytopeSolution:
        verts = _random_on_sphere(self.N, self.d, rng)
        simps = _stereographic_delaunay(verts)
        return PolytopeSolution(vertices=verts, simplices=simps)

    def evaluate(self, sol: PolytopeSolution) -> float:
        return _hyperbolic_volume(self.d, sol)

    def evaluate_surrogate(self, sol: PolytopeSolution) -> float:
        # Uniform fast proxy: Euclidean hull volume for all dimensions.
        return _euclidean_hull_volume(sol.vertices)

    def perturb(self, sol: PolytopeSolution, radius: float,
                rng: np.random.Generator) -> PolytopeSolution:
        verts = _perturb_on_sphere(sol.vertices, radius, rng)
        # Retriangulate on larger perturbations (matches the old
        # _do_perturb semantics which retriangulated implicitly via LBFGS).
        simps = _stereographic_delaunay(verts) if radius > 0.15 else sol.simplices
        return PolytopeSolution(vertices=verts, simplices=simps)

    def metadata(self) -> np.ndarray:
        return np.array(
            [self.d, self.N, np.log(self.N * self.d)], dtype=np.float32
        )

    def copy_solution(self, sol: PolytopeSolution) -> PolytopeSolution:
        return PolytopeSolution(
            vertices=sol.vertices.copy(),
            simplices=[list(s) for s in sol.simplices],
        )

    # ---- Continuous-optimizer hooks ---------------------------------------

    def to_vector(self, sol: PolytopeSolution) -> np.ndarray:
        return sol.vertices.ravel().astype(float)

    def from_vector(self, vec: np.ndarray) -> PolytopeSolution:
        verts = _normalize_rows(vec.reshape(self.N, self.d))
        # Retriangulate every time: `evaluate` at d=6 consumes simplices
        # (and d=3/4 build their own anyway), so returning stale or empty
        # simplices makes task.evaluate(from_vector(v)) produce zeros at d=6.
        # Delaunay is cheap compared to the closed-form volume call.
        try:
            simps = _stereographic_delaunay(verts)
        except Exception:
            simps = []
        return PolytopeSolution(vertices=verts, simplices=simps)

    def project(self, vec: np.ndarray) -> np.ndarray:
        verts = _normalize_rows(vec.reshape(self.N, self.d))
        return verts.ravel()


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
