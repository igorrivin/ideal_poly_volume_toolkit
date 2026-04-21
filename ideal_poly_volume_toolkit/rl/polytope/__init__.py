"""
Ideal polytope specialization of the RL harness.

Public entry points:
- `IdealPolytopeTask`, `IdealPolytopeTaskSampler` — problem definition
- `PolytopeSolution` — (vertices, simplices) dataclass
- `make_polytope_env` — constructor for the Gymnasium env
- `LBFGSHyperbolicD{3,4,5,6}` — dimension-specific LBFGS inner optimizers
- `lbfgs_hyperbolic_for_dimension(d)` — dispatcher
"""

from .types import PolytopeSolution
from .task import IdealPolytopeTask, IdealPolytopeTaskSampler
from .env import make_polytope_env
from .optimizers import (
    LBFGSHyperbolicD3,
    LBFGSHyperbolicD4,
    LBFGSHyperbolicD5,
    LBFGSHyperbolicD6,
    lbfgs_hyperbolic_for_dimension,
)

__all__ = [
    "PolytopeSolution",
    "IdealPolytopeTask",
    "IdealPolytopeTaskSampler",
    "make_polytope_env",
    "LBFGSHyperbolicD3",
    "LBFGSHyperbolicD4",
    "LBFGSHyperbolicD5",
    "LBFGSHyperbolicD6",
    "lbfgs_hyperbolic_for_dimension",
]
