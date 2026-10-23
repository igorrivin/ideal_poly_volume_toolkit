"""
Hierarchical RL optimization harness.

The generic layer (task / inner optimizer / search tree / Gym env) lives in
`rl.generic`. Problem-specific customizations live in `rl.polytope` and
`rl.cage`. See docs/plans/rl-harness-design.md for the design.
"""

from .generic import (
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

from .inner_optimizers import (
    RandomRestartOptimizer,
    CMAESOptimizer,
    ScipyMinimizeOptimizer,
)

from .training import (
    train_controller,
    evaluate_controller,
    compare_with_baseline,
)

__all__ = [
    "OptimizationTask",
    "InnerOptimizer",
    "OptimizerRun",
    "OptimizerRunError",
    "TaskSampler",
    "SearchNode",
    "GenericOptEnv",
    "NODE_FEATURE_DIM",
    "MAX_FRONTIER",
    "RandomRestartOptimizer",
    "CMAESOptimizer",
    "ScipyMinimizeOptimizer",
    "train_controller",
    "evaluate_controller",
    "compare_with_baseline",
]
