"""Back-compat shim for the old `rl_optimizer` import path.

The RL harness was restructured into `ideal_poly_volume_toolkit.rl` during
Stage C of the harness modularization.  This shim re-exports the old
symbols so existing scripts (and the parity test) keep working.

New code should import from:
- `ideal_poly_volume_toolkit.rl.polytope` — `IdealPolytopeTask`,
  `IdealPolytopeTaskSampler`, `make_polytope_env`, `PolytopeSolution`,
  `LBFGSHyperbolicD{3,4,5,6}`, `lbfgs_hyperbolic_for_dimension`.
- `ideal_poly_volume_toolkit.rl.training` — `train_controller`,
  `evaluate_controller`, `compare_with_baseline`.
- `ideal_poly_volume_toolkit.rl.inner_optimizers` — generic
  `CMAESOptimizer`, `ScipyMinimizeOptimizer`, `RandomRestartOptimizer`.

The legacy `IdealPolytopeOptEnv`, `LocalCMAES`, `LocalLBFGS`, and
friends are kept reachable via `rl._legacy_polytope_env` for parity
testing and backwards compatibility; they are deprecated and will be
removed once the parity test is retired.
"""

# Legacy implementation (pre-refactor) — still importable but deprecated.
from .rl._legacy_polytope_env import (  # noqa: F401
    RegionNode,
    NODE_FEATURE_DIM as _LEGACY_NODE_FEATURE_DIM,  # shadowed by rl.generic below
    MAX_FRONTIER as _LEGACY_MAX_FRONTIER,           # shadowed by rl.generic below
    euclidean_volume,
    random_on_sphere,
    perturb_on_sphere,
    LocalCMAES,
    LocalLBFGS,
    triangulate,
    ACT_SAMPLE,
    ACT_CMA_EUCL,
    ACT_LBFGS_HYP,
    ACT_SPLIT,
    ACT_PERTURB,
    ACT_PRUNE,
    NUM_ACTIONS,
    IdealPolytopeOptEnv,
)

# Canonical re-exports from the new subpackage.
from .rl.generic import NODE_FEATURE_DIM, MAX_FRONTIER  # noqa: F401
from .rl.training import (  # noqa: F401
    train_controller as train_rl_controller,
    evaluate_controller,
    compare_with_baseline,
)
from .rl.polytope import (  # noqa: F401
    IdealPolytopeTask,
    IdealPolytopeTaskSampler,
    PolytopeSolution,
    make_polytope_env,
    LBFGSHyperbolicD3,
    LBFGSHyperbolicD4,
    LBFGSHyperbolicD5,
    LBFGSHyperbolicD6,
    lbfgs_hyperbolic_for_dimension,
)
