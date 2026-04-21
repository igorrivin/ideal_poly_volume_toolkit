"""
Ideal polytope specialization of the RL harness.

Public entry points will be added in Stage C2:
- `IdealPolytopeTask`, `IdealPolytopeTaskSampler` (rl.polytope.task)
- `make_polytope_env` factory
- `PolytopeSolution` dataclass

In C1 only the dimension-specific LBFGS inner optimizers are available.
"""

from .optimizers import (
    LBFGSHyperbolicD3,
    LBFGSHyperbolicD4,
    LBFGSHyperbolicD5,
    LBFGSHyperbolicD6,
    lbfgs_hyperbolic_for_dimension,
)

__all__ = [
    "LBFGSHyperbolicD3",
    "LBFGSHyperbolicD4",
    "LBFGSHyperbolicD5",
    "LBFGSHyperbolicD6",
    "lbfgs_hyperbolic_for_dimension",
]
