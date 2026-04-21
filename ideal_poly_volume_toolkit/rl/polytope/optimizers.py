"""
Polytope-specific InnerOptimizer implementations.

One class per dimension, each LBFGS-on-hyperbolic-volume using whatever
closed-form gradient mechanism is available for that dimension.  The
dimension-specific torch imports are done lazily inside `run()` so the
package stays importable without torch for users who only need the
volume-computation layer.

Design rationale: the previous monolithic `LocalLBFGS` had a four-way
`if self.dimension == k` ladder inside `run()`.  Splitting it into four
small classes removes the dispatch while keeping the per-dimension code
local and lets each dimension carry its own configuration (step count,
learning rate, etc.).
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from ..generic import InnerOptimizer, OptimizationTask
from .types import PolytopeSolution


# =====================================================================
# Common infrastructure
# =====================================================================

def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Return a copy with each row on the unit sphere."""
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


class _LBFGSHyperbolicBase(InnerOptimizer):
    """Common scaffolding for LBFGS-on-hyperbolic-volume optimizers.

    Subclasses supply:
      - `dimension` (class attribute)
      - `_optimize(vertices, simplices, steps) -> final_vertices`
        (dimension-specific gradient step + projection loop)

    `run()` returns (PolytopeSolution, [values]) where `values` contains
    one entry — the task's own `evaluate(final_solution)` — so the env's
    scoring is always through the task and hence consistent.
    """

    dimension: int

    def __init__(self, *, steps: int = 5, name: str | None = None):
        self.steps = int(steps)
        self._name = name or f"lbfgs_hyp_d{self.dimension}"

    @property
    def name(self) -> str:
        return self._name

    # Subclass hook ----------------------------------------------------------
    def _optimize(self, vertices: np.ndarray,
                  simplices: List[List[int]],
                  steps: int) -> np.ndarray:
        raise NotImplementedError

    # InnerOptimizer contract ------------------------------------------------
    def run(self, solution: Any, task: OptimizationTask,
            budget: int, rng: np.random.Generator) -> Tuple[Any, List[float]]:
        if not isinstance(solution, PolytopeSolution):
            raise TypeError(
                f"{self.name} expects a PolytopeSolution; got {type(solution).__name__}"
            )
        # budget is interpreted as an LBFGS step count; default to self.steps
        # when the env passes a generic small number.
        steps = max(1, int(budget)) if budget > 0 else self.steps
        try:
            final = self._optimize(solution.vertices, solution.simplices, steps)
        except Exception:
            # Gradient failures -> no-op (same robustness the old code had).
            return solution, [task.evaluate(solution)]
        final = _normalize_rows(final)
        new_sol = PolytopeSolution(vertices=final, simplices=solution.simplices)
        return new_sol, [task.evaluate(new_sol)]


# =====================================================================
# d = 3 (Bloch–Wigner H^3 volume)
# =====================================================================

class LBFGSHyperbolicD3(_LBFGSHyperbolicBase):
    """H^3 volume via stereographic Delaunay + Bloch–Wigner.  scipy Powell
    on the closed-form H^3 objective — matches the d=3 branch of the
    original LocalLBFGS."""

    dimension = 3

    def _optimize(self, vertices, simplices, steps):
        from scipy.optimize import minimize
        from scipy.special import spence
        from scipy.spatial import Delaunay

        N = vertices.shape[0]

        def h3_vol(pts_S2: np.ndarray) -> float:
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
                    li2.imag
                    + np.angle(1.0 - cr) * np.log(max(abs(cr), 1e-300))
                )
            return total

        def neg(x: np.ndarray) -> float:
            pts = _normalize_rows(x.reshape(vertices.shape))
            try:
                return -h3_vol(pts)
            except Exception:
                return 0.0

        res = minimize(neg, vertices.ravel(), method="Powell",
                       options={"maxiter": steps * 50, "maxfev": steps * 30})
        return res.x.reshape(vertices.shape)


# =====================================================================
# d = 4 (closed-form H^4 polytope volume)
# =====================================================================

class LBFGSHyperbolicD4(_LBFGSHyperbolicBase):
    """H^4 volume via the closed-form `ideal_polytope_volume_4d`; scipy
    Powell optimizer (matches the d=4 branch of the original LocalLBFGS)."""

    dimension = 4

    def _optimize(self, vertices, simplices, steps):
        from scipy.optimize import minimize
        from ...geometry_4d import ideal_polytope_volume_4d

        def neg(x: np.ndarray) -> float:
            pts = _normalize_rows(x.reshape(vertices.shape))
            try:
                vol, _ = ideal_polytope_volume_4d(pts)
                return -vol if np.isfinite(vol) else 0.0
            except Exception:
                return 0.0

        res = minimize(neg, vertices.ravel(), method="Powell",
                       options={"maxiter": steps * 50, "maxfev": steps * 20})
        return res.x.reshape(vertices.shape)


# =====================================================================
# d = 5 (Schläfli surrogate, torch LBFGS)
# =====================================================================

class LBFGSHyperbolicD5(_LBFGSHyperbolicBase):
    """H^5 optimization via the Schläfli surrogate (exact gradient,
    free of the volume value itself).  torch LBFGS with strong-Wolfe
    line search."""

    dimension = 5

    def __init__(self, *, steps: int = 5, lr: float = 0.01,
                 max_iter_per_step: int = 3, name: str | None = None):
        super().__init__(steps=steps, name=name)
        self.lr = float(lr)
        self.max_iter_per_step = int(max_iter_per_step)

    def _optimize(self, vertices, simplices, steps):
        import torch
        from ...h5_volume_torch import schlafli_surrogate_vectorized

        pts_opt = torch.tensor(vertices, dtype=torch.float64,
                               requires_grad=True)
        simps_t = torch.tensor(simplices, dtype=torch.long)
        opt = torch.optim.LBFGS(
            [pts_opt], lr=self.lr, max_iter=self.max_iter_per_step,
            line_search_fn="strong_wolfe",
        )
        for _ in range(steps):
            def closure():
                opt.zero_grad()
                normed = pts_opt / torch.norm(pts_opt, dim=1, keepdim=True)
                s = schlafli_surrogate_vectorized(normed, simps_t)
                (-s).backward()
                return -s
            opt.step(closure)
            with torch.no_grad():
                pts_opt.data /= torch.norm(pts_opt.data, dim=1, keepdim=True)

        return pts_opt.detach().numpy()


# =====================================================================
# d = 6 (closed-form H^6 polytope volume, torch LBFGS)
# =====================================================================

class LBFGSHyperbolicD6(_LBFGSHyperbolicBase):
    """H^6 volume via `ideal_h6_polytope_volume` (Murakami-torch).  torch
    LBFGS with strong-Wolfe line search; differentiable H^6 volume as the
    objective."""

    dimension = 6

    def __init__(self, *, steps: int = 5, lr: float = 0.01,
                 max_iter_per_step: int = 3, name: str | None = None):
        super().__init__(steps=steps, name=name)
        self.lr = float(lr)
        self.max_iter_per_step = int(max_iter_per_step)

    def _optimize(self, vertices, simplices, steps):
        import torch
        from ...murakami_torch import ideal_h6_polytope_volume
        from ...li2_torch import li2 as li2_pure

        pts_opt = torch.tensor(vertices, dtype=torch.float64,
                               requires_grad=True)
        simps_t = torch.tensor(simplices, dtype=torch.long)
        opt = torch.optim.LBFGS(
            [pts_opt], lr=self.lr, max_iter=self.max_iter_per_step,
            line_search_fn="strong_wolfe",
        )
        for _ in range(steps):
            def closure():
                opt.zero_grad()
                normed = pts_opt / torch.norm(pts_opt, dim=1, keepdim=True)
                vol = ideal_h6_polytope_volume(normed, simps_t, li2_fn=li2_pure)
                (-vol).backward()
                return -vol
            opt.step(closure)
            with torch.no_grad():
                pts_opt.data /= torch.norm(pts_opt.data, dim=1, keepdim=True)

        return pts_opt.detach().numpy()


# =====================================================================
# Dispatch by dimension
# =====================================================================

_LBFGS_BY_DIM = {
    3: LBFGSHyperbolicD3,
    4: LBFGSHyperbolicD4,
    5: LBFGSHyperbolicD5,
    6: LBFGSHyperbolicD6,
}


def lbfgs_hyperbolic_for_dimension(d: int, **kwargs) -> _LBFGSHyperbolicBase:
    """Construct the LBFGS-on-hyperbolic optimizer appropriate for
    dimension `d`.  Raises KeyError for unsupported dimensions."""
    cls = _LBFGS_BY_DIM[d]
    return cls(**kwargs)
