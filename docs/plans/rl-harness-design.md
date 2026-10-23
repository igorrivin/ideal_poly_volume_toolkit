# RL Harness Modularization — Design

*Status:* historical Stage-B design; Stage C has since been implemented. See
`docs/implementation-status.md` for current status and
`ideal_poly_volume_toolkit/rl/README.md` for the live interfaces. Return types
and optimizer details below describe the pre-accounting design and are retained
for provenance.

## 1. Goal

Factor the "adaptive covariance + tree search + RL meta-controller"
optimization harness into a fully problem-agnostic subpackage, with clean
hooks for each concrete problem (ideal polytopes, cages, Hadamard, future
additions). Eliminate the ~250 lines of duplication between
`rl_optimizer.py` and `rl_generic.py`.

**Guiding principle.** `rl_cages.py` is the template — it uses
`GenericOptEnv` by composition, defines `CubicGirthTask(OptimizationTask)`,
and supplies its own `InnerOptimizer` subclasses. The polytope side should
do the same. The generic layer should not know anything about polytopes,
sphere normalization, torch, scipy, or CMA.

## 2. Target file layout

```
ideal_poly_volume_toolkit/rl/
├── __init__.py            re-exports the public symbols listed in §9
├── generic.py             Task / InnerOptimizer / TaskSampler / SearchNode
│                          / GenericOptEnv.  No problem-specific imports.
├── inner_optimizers.py    Problem-agnostic InnerOptimizer implementations:
│                          RandomRestart, CMAESOptimizer, ScipyMinimizeOptimizer.
├── training.py            train(env_factory, …) and evaluate(model, env, …).
│                          SB3-PPO boilerplate, extracted from rl_optimizer.
├── polytope/
│   ├── __init__.py        re-exports IdealPolytopeTask, IdealPolytopeTaskSampler,
│   │                      make_polytope_env, the polytope-specific optimizers.
│   ├── task.py            IdealPolytopeTask, IdealPolytopeTaskSampler.
│   ├── optimizers.py      LBFGSHyperbolicD{3,4,5,6} and any other
│   │                      polytope-specific InnerOptimizers.  Dispatches
│   │                      on dimension here, not in the harness.
│   └── examples.py        Legacy Hadamard demo (moved out of generic.py).
├── cage/
│   ├── __init__.py        re-exports CubicGirthTask, make_cage_env, etc.
│   └── task.py            Content of today's rl_cages.py (renamed in place).
└── README.md              One-page map of the subpackage with the two
                           working reference pipelines (polytope, cage).
```

Top-level `ideal_poly_volume_toolkit/rl_optimizer.py`, `rl_generic.py`,
`rl_cages.py` become shims that `from .rl.* import *` for one release
(see §9). `fast_logdet.py` stays where it is — it's a numerical utility,
not part of the harness.

## 3. Generic interfaces

### 3.1 Kept as-is

The current `OptimizationTask`, `InnerOptimizer`, `TaskSampler`, `SearchNode`,
and `GenericOptEnv` signatures in `rl_generic.py` are good. **Stage C does not
change their public API.** It only moves them into `rl/generic.py` and removes
the polytope-flavoured comment on line 11–12.

For reference — these are the contracts every problem must implement:

```python
class OptimizationTask(ABC):
    def random_solution(self, rng) -> Any: ...
    def evaluate(self, solution) -> float: ...      # HIGHER is better.
    def perturb(self, solution, radius, rng) -> Any: ...
    def metadata(self) -> np.ndarray: ...           # fixed-size feature vector
    def copy_solution(self, solution) -> Any: ...

class InnerOptimizer(ABC):
    name: str
    def run(self, solution, task, budget, rng) -> tuple[Any, list[float]]: ...
    # Returns (best_solution, list_of_objective_values_seen).

class TaskSampler(ABC):
    def sample(self, rng) -> OptimizationTask: ...
    def metadata_dim(self) -> int: ...
```

`GenericOptEnv` continues to own: frontier management, the action dispatch
(`sample` / `split` / `perturb` / `prune` + N inner-optimizer slots),
reward shaping, observation encoding.

### 3.2 New optional hooks on `OptimizationTask`

For optimizers that need to treat solutions as real vectors (CMA-ES, L-BFGS,
scipy minimize), tasks may optionally implement:

```python
class OptimizationTask(ABC):
    # ... existing abstract methods ...

    # Optional: only tasks with a real-vector embedding need these.
    def to_vector(self, solution) -> np.ndarray: ...
    def from_vector(self, vec) -> Any: ...
    def project(self, vec) -> np.ndarray: ...   # enforce feasibility

    # Optional: fast surrogate used by surrogate-aware optimizers.
    # Defaults to self.evaluate.
    def evaluate_surrogate(self, solution) -> float:
        return self.evaluate(solution)
```

These are **concrete methods with default implementations** (raise
`NotImplementedError` for `to_vector` / `from_vector` / `project`,
return `evaluate(...)` for `evaluate_surrogate`). Optimizers that require
them do an `isinstance` or `hasattr` check in their `__init__` and refuse
tasks that don't support them. Nothing in `GenericOptEnv` touches these.

**Rationale for `evaluate_surrogate`.** The polytope pipeline currently
has `ACT_CMA_EUCL` (CMA-ES on *Euclidean* volume, fast) and `ACT_LBFGS_HYP`
(LBFGS on *hyperbolic* volume, slow, exact gradients). Generic surrogate
support lets both live in generic `inner_optimizers.py` without the
polytope-specific `EUCL` / `HYP` naming leaking into the harness. The
polytope task sets `evaluate_surrogate` = Euclidean volume and `evaluate` =
hyperbolic volume; inner optimizers choose which to call.

### 3.2a Polytope scoring fix (correctness bug identified during C)

The current `rl_optimizer.py` has a latent bug: `_do_cma` always scores by
Euclidean volume (line 489 hardcodes `euclidean_volume` as the CMA
objective), while `_do_lbfgs` writes whatever `LocalLBFGS.run()` returns
into `node.best_value` — and for d ∈ {3, 4, 6} that is **hyperbolic**
volume, while for d = 5 it is Euclidean. So `node.best_value` holds a
different *kind* of number depending on optimizer history, and the
`v > node.best_value` comparison is not well-defined across operations.

The refactor fixes this by giving each task a **single consistent
scoring function**, chosen per dimension by whichever hyperbolic form is
closed and fast:

| d | `task.evaluate` returns      | notes                                           |
|---|-------------------------------|-------------------------------------------------|
| 3 | hyperbolic (Bloch–Wigner)     | closed form, fast                                |
| 4 | hyperbolic (`ideal_polytope_volume_4d`) | closed form, fast                    |
| 5 | Euclidean                     | true H^5 is Schläfli MC (seconds, noisy); paper §7.7 says Eucl ≈ Hyp here |
| 6 | hyperbolic (`ideal_h6_polytope_volume`) | closed form, fast                    |

LBFGS continues to use Schläfli / hyperbolic gradients internally in all
four dimensions (the gradient code already does that). What changes is
that at d ∈ {3, 4, 6}, CMA-ES now also optimizes the hyperbolic objective
(which is cheap for these dimensions), instead of silently optimizing the
wrong function and getting overwritten later.

`task.evaluate_surrogate` still exists for optimizers that want the fast
Euclidean surrogate; for d = 5 it equals `task.evaluate`.

### 3.3 State-persistence contract (resolving B2)

Every existing optimizer on both sides (`LocalCMAES`, `LocalLBFGS`, all the
cage optimizers) is **stateless per call** — each `run()` constructs its own
machinery from scratch. The `local_state` dict on `rl_optimizer.RegionNode`
is actually unused by `LocalCMAES`/`LocalLBFGS`; it's an aspirational hook.

**Decision:** Stage C keeps optimizers stateless. The `SearchNode` in
`generic.py` is sufficient; we do *not* merge `RegionNode` features back in.
If a future optimizer genuinely needs per-region persistence, we add an
optional `state_key: str` on `InnerOptimizer` and a
`node.optimizer_state: dict[str, Any]` field in a follow-up. Not in Stage C.

## 4. Generic inner optimizers (`rl/inner_optimizers.py`)

Three implementations, all problem-agnostic:

```python
class RandomRestartOptimizer(InnerOptimizer):
    """Draw `budget` random solutions, keep the best."""

class CMAESOptimizer(InnerOptimizer):
    """CMA-ES via `cma` package.  Requires the task to implement
    `to_vector` / `from_vector` / `project`.  Maximizes either
    `task.evaluate` or `task.evaluate_surrogate` depending on
    constructor flag."""

    def __init__(self, sigma: float = 0.2, use_surrogate: bool = False): ...

class ScipyMinimizeOptimizer(InnerOptimizer):
    """Wraps `scipy.optimize.minimize` with a configurable method
    ('Powell', 'L-BFGS-B', etc.).  Same vector-protocol requirement
    as CMAES.  Surrogate-aware."""

    def __init__(self, method: str = "Powell", use_surrogate: bool = False): ...
```

These replace `LocalCMAES` (currently in `rl_optimizer.py`) and any
gradient-free scipy calls in the polytope pipeline.

**Not generic:** the dimension-dispatching torch code in `LocalLBFGS`
(4 nested `if dimension == k` blocks, each importing different modules).
That's polytope-specific and moves to `rl/polytope/optimizers.py` as
four small subclasses — `LBFGSHyperbolicD3`, `…D4`, `…D5`, `…D6` — each
with no dispatch logic. The polytope task's sampler picks the right one
for the task's dimension.

## 5. Polytope specialization (`rl/polytope/`)

### 5.1 `task.py`

```python
@dataclass
class PolytopeSolution:
    vertices: np.ndarray           # shape (N, d), each row on S^{d-1}
    simplices: list[list[int]]     # Delaunay triangulation

class IdealPolytopeTask(OptimizationTask):
    def __init__(self, d: int, N: int): ...
    # evaluate -> hyperbolic volume (the true objective)
    # evaluate_surrogate -> Euclidean hull volume (fast, monotone-correlated)
    # to_vector / from_vector / project -> reshape + row-normalize
    # perturb -> random tangent-space step + retriangulate
    # random_solution -> uniform points on S^{d-1} + Delaunay
    # metadata -> [d, N, log(N*d)]

class IdealPolytopeTaskSampler(TaskSampler):
    def __init__(self, d_range=(3, 6), N_range=(7, 25)): ...
```

The sampler owns the `d_range` and `N_range` that are currently hardcoded
inside `IdealPolytopeOptEnv`.

### 5.2 `optimizers.py`

Only the truly polytope-specific inner optimizers live here. Today that is
**just the four LBFGS-on-hyperbolic-volume variants** (d ∈ {3, 4, 5, 6}),
each a thin subclass of `InnerOptimizer` importing its dimension-specific
torch / scipy machinery. Everything else — CMA-ES on Euclidean, random
restarts, Powell — uses the generic optimizers from §4.

### 5.3 Factory

```python
def make_polytope_env(
    *, d_range=(3, 6), N_range=(7, 25),
    horizon=50,
    use_lbfgs_hyp=True,   # include LBFGS-on-hyperbolic for the env's dimension
    use_cma_eucl=True,    # include CMA-ES on Euclidean surrogate
    seed=None,
) -> GenericOptEnv:
    """One-call construction of the polytope RL environment.
    Replaces IdealPolytopeOptEnv."""
```

The factory builds the `inner_optimizers` list by dimension (so d=3 gets
`LBFGSHyperbolicD3`, etc.) and passes it to `GenericOptEnv(task_sampler,
inner_optimizers, horizon=…)`. That's the entire replacement for the
250-line `IdealPolytopeOptEnv`.

### 5.4 Ghost action: `snap-symmetric` (resolving B3)

The comment on `rl_optimizer.py:11` is orphaned — there is no
`ACT_SNAP_SYMMETRIC` and no dispatch. **Stage C deletes the comment.**

If we later want a symmetry-respecting snap action (e.g., project onto the
nearest polytope with prescribed `join_id` structure like `C_a+C_b+I`),
it lives as a *polytope-specific* `InnerOptimizer` in
`rl/polytope/optimizers.py`, not in the generic layer. Candidate name:
`SnapToJoinStructure`. Not in Stage C.

## 6. Cage specialization (`rl/cage/`)

Today's `rl_cages.py` moves to `rl/cage/task.py` with three trivial
rewrites:

- `from .rl_generic import …` → `from ..generic import …`.
- `make_cage_env` stays as-is; re-exported from `rl/cage/__init__.py`.
- Any cage-specific inner optimizers stay inside `task.py` (they're small
  and tightly coupled to `CubicGirthTask`).

No behaviour change. This is the validation that the generic layer is
actually generic: the cage side should compile and test green after the
generic layer moves with literally only the import-path edit.

## 7. Training module (`rl/training.py`)

Extracted from `rl_optimizer.py` lines ~597–707. Three functions:

```python
def train_controller(
    env_factory: Callable[[], GenericOptEnv],
    *, total_timesteps: int = 500_000,
    save_path: str | Path,
    seed: int = 0,
    policy_kwargs: dict | None = None,
    tensorboard_log: str | Path | None = None,
) -> "stable_baselines3.PPO":
    """Train an SB3 PPO policy on the given env.  Agnostic to the problem."""

def evaluate_controller(
    model_path: str | Path,
    env: GenericOptEnv,
    *, n_episodes: int = 10,
    deterministic: bool = True,
    seed: int | None = None,
) -> dict:
    """Load a saved model, run N episodes, return {best_values, episode_returns, …}."""

def compare_with_baseline(
    env: GenericOptEnv,
    *, n_trials: int = 20,
    budget_per_trial: int = 50,
    baselines: list[InnerOptimizer] | None = None,
) -> dict:
    """Run the inner optimizers in isolation (no RL controller) and compare
    to whatever the RL policy achieves.  Baselines default to all optimizers
    configured on the env."""
```

All three accept any `GenericOptEnv`, so the cage side gets them for free
(currently `rl_cages.py` has no training boilerplate). Both
`train_polytope_controller` and `train_cage_controller` become one-line
wrappers in `rl/polytope/__init__.py` and `rl/cage/__init__.py`.

## 8. Validation plan (Stage C exit criteria)

Stage C is done when **all four** of the following hold:

1. `rl_generic.py`, `rl_optimizer.py`, `rl_cages.py` either have moved or
   become ≤10-line shims that import from `rl/*`. The old import paths
   still work.
2. A parity test runs 20 seeds through the pre-refactor
   `IdealPolytopeOptEnv` and the post-refactor `make_polytope_env()` with
   identical RNG. Amended for the §3.2a scoring fix:
   - **d = 5**: bit-for-bit parity on deterministic optimizers; 1e-6
     relative tolerance on CMA-ES. (d = 5 used consistent Euclidean
     scoring before, so behavior is preserved.)
   - **d ∈ {3, 4, 6}**: trajectories diverge because CMA now optimizes
     hyperbolic instead of Euclidean. Assert one-sided: the new env's
     final best hyperbolic volume is ≥ the old env's final best volume
     (after re-scoring the old config with `task.evaluate`).
3. The saved `ppo_d5_*.zip` models in `results/rl_models/` load via
   `evaluate_controller` on the new env and produce best-values within
   float noise of the training-log values.
4. The cage side re-runs (if there are existing cage training artifacts)
   or at least `make_cage_env()` + a 1k-step training smoke test produce
   the same results as before the generic layer moved.

If (2) or (3) fails, **stop and investigate** — the refactor is lying.
Do not tune to make tests pass.

## 9. Migration / shims (no breakage)

For one release cycle after Stage C lands, these files exist as shims:

```python
# ideal_poly_volume_toolkit/rl_generic.py
from .rl.generic import *  # noqa
from .rl.generic import SearchNode, GenericOptEnv, OptimizationTask, \
    InnerOptimizer, TaskSampler

# ideal_poly_volume_toolkit/rl_optimizer.py
from .rl.polytope import (
    IdealPolytopeTask, IdealPolytopeTaskSampler,
    make_polytope_env,
)
from .rl.training import (
    train_controller as train_rl_controller,
    evaluate_controller,
    compare_with_baseline,
)
# Deprecated (warn on import):
from .rl.polytope.optimizers import LBFGSHyperbolicD3 as LocalLBFGS  # etc.
IdealPolytopeOptEnv = None  # removed; see make_polytope_env

# ideal_poly_volume_toolkit/rl_cages.py
from .rl.cage import *  # noqa
```

`scripts/` does not need editing in Stage C — the shims keep it working.
A follow-up PR removes the shims and updates any import sites.

## 10. Explicitly deferred

These are real work, but not part of Stage C. Revisit after:

- **Re-deriving the missing d=5 configs** (N ∈ {12, 14, 20}). This is the
  first research use of the refactored harness, not part of the refactor.
- **Snap-to-join-structure action.** Would be interesting for the
  above re-derivation; add on demand.
- **Per-region CMA-ES state persistence.** Not needed given everything is
  stateless today. If added, it's a single `node.optimizer_state` dict.
- **Merging `rl/polytope/examples.py` back into the polytope tests.**
  The Hadamard task currently sits in `rl_generic.py` lines 338–874; moving
  it out is a cleanup, not a refactor.

## 11. Questions for the user before Stage C

None blocking. If you want to change anything above, the cheapest time is
right now — every item in §8 assumes the decisions in §3–§7 hold.
