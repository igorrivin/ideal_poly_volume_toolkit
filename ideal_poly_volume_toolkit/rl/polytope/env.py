"""
Factory for the polytope RL environment.

Replaces the old `IdealPolytopeOptEnv` by composing a
`GenericOptEnv` with an `IdealPolytopeTaskSampler` and the polytope's
inner optimizers (CMA-ES on the surrogate plus a dimension-specific Powell or
projected-Adam method).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..generic import GenericOptEnv, InnerOptimizer
from ..inner_optimizers import CMAESOptimizer
from .task import IdealPolytopeTaskSampler
from .optimizers import (
    LBFGSHyperbolicD3, LBFGSHyperbolicD4,
    LBFGSHyperbolicD5, LBFGSHyperbolicD6,
)


def _default_inner_optimizers(d_range: Tuple[int, int]) -> List[InnerOptimizer]:
    """Build the inner-optimizer list for the given dimension range.

    Includes CMA-ES on the Euclidean surrogate (for fast exploration) and
    whichever LBFGSHyperbolicD* classes cover the range.  The RL agent
    learns which to dispatch per node.

    The surrogate-CMA sigma is 0.4, matching the old env's root-level
    `radius * 0.2` with radius = 2.0.
    """
    opts: List[InnerOptimizer] = [
        CMAESOptimizer(sigma=0.4, use_surrogate=True, name="cma_eucl"),
    ]
    lo, hi = d_range
    if lo <= 3 <= hi:
        opts.append(LBFGSHyperbolicD3())
    if lo <= 4 <= hi:
        opts.append(LBFGSHyperbolicD4())
    if lo <= 5 <= hi:
        opts.append(LBFGSHyperbolicD5())
    if lo <= 6 <= hi:
        opts.append(LBFGSHyperbolicD6())
    return opts


def make_polytope_env(
    *,
    d_range: Tuple[int, int] = (4, 6),
    N_range: Tuple[int, int] = (7, 25),
    horizon: int = 50,
    eval_cost_weight: float = 0.001,
    seed: Optional[int] = None,
    inner_optimizers: Optional[List[InnerOptimizer]] = None,
) -> GenericOptEnv:
    """Construct the Gymnasium env for RL-guided ideal polytope volume
    maximization.  The replacement for `IdealPolytopeOptEnv`.

    Parameters
    ----------
    d_range : (low, high) range of dimensions to sample.  Only 3..6 supported.
    N_range : (low, high) range of vertex counts; capped at 5*d per task.
    horizon : episode length (env steps).
    eval_cost_weight : cost per function eval in the reward shaping.
    seed : env-level RNG seed.
    inner_optimizers : override the default optimizer list.
    """
    sampler = IdealPolytopeTaskSampler(d_range=d_range, N_range=N_range)
    opts = (inner_optimizers
            if inner_optimizers is not None
            else _default_inner_optimizers(d_range))
    return GenericOptEnv(
        task_sampler=sampler,
        inner_optimizers=opts,
        horizon=horizon,
        eval_cost_weight=eval_cost_weight,
        seed=seed,
    )
