"""Back-compat shim for the old `rl_generic` import path.

The generic harness moved to `ideal_poly_volume_toolkit.rl.generic` during
Stage C of the harness modularization.  All public symbols are re-exported
here so existing code — `from ideal_poly_volume_toolkit.rl_generic import
GenericOptEnv` and friends — keeps working.

New code should import from `ideal_poly_volume_toolkit.rl` directly.
"""

from .rl.generic import *  # noqa: F401,F403
from .rl.generic import (  # noqa: F401  # explicit static re-exports
    OptimizationTask,
    InnerOptimizer,
    OptimizerRun,
    OptimizerRunError,
    TaskSampler,
    SearchNode,
    GenericOptEnv,
    NODE_FEATURE_DIM,
    MAX_FRONTIER,
)
