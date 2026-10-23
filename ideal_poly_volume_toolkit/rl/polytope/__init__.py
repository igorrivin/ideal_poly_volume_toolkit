"""
Ideal polytope specialization of the RL harness.

Public entry points:
- `IdealPolytopeTask`, `IdealPolytopeTaskSampler` — problem definition
- `PolytopeSolution` — (vertices, simplices) dataclass
- `make_polytope_env` — constructor for the Gymnasium env
- `LBFGSHyperbolicD{3,4,5,6}` — compatibility names for dimension-specific
  Powell or projected-Adam inner optimizers
- `FreshTopologyADHyperbolicD4` — fixed-pole, fresh-Delaunay H4 autograd search
- `lbfgs_hyperbolic_for_dimension(d)` — dispatcher
"""

from .types import PolytopeSolution
from .task import IdealPolytopeTask, IdealPolytopeTaskSampler
from .env import make_polytope_env
from .optimizers import (
    FreshTopologyADHyperbolicD4,
    H4OptimizerRun,
    LBFGSHyperbolicD3,
    LBFGSHyperbolicD4,
    LBFGSHyperbolicD5,
    LBFGSHyperbolicD6,
    lbfgs_hyperbolic_for_dimension,
)
from .h4_symmetric_benchmark import (
    audit_h4_symmetric_candidate,
    optimize_h4_from_initial,
    optimize_h4_unconstrained,
)

__all__ = [
    "PolytopeSolution",
    "IdealPolytopeTask",
    "IdealPolytopeTaskSampler",
    "make_polytope_env",
    "FreshTopologyADHyperbolicD4",
    "H4OptimizerRun",
    "LBFGSHyperbolicD3",
    "LBFGSHyperbolicD4",
    "LBFGSHyperbolicD5",
    "LBFGSHyperbolicD6",
    "lbfgs_hyperbolic_for_dimension",
    "audit_h4_symmetric_candidate",
    "optimize_h4_from_initial",
    "optimize_h4_unconstrained",
]
