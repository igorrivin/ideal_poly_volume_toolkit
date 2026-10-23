# RL optimization harness

This subpackage is the canonical home of the hierarchical optimization
harness. New code should import from `ideal_poly_volume_toolkit.rl` and its
specializations, not from the compatibility modules at the package root.

## Canonical modules

- `generic.py` defines the optimization task, inner-optimizer, task-sampler,
  search-node, and Gymnasium environment interfaces. Its `OptimizerRun` result
  separates canonical `objective_evals`, `surrogate_evals`, and
  `gradient_evals`. A positive `budget` is a hard cap on their total for a
  complete run, including the canonical score of the returned solution; an
  optimizer may under-use a budget when work is indivisible. The environment
  charges actual work and does not mix surrogate values into canonical node
  statistics. An optimizer can supply the counted canonical value of its
  returned solution to avoid a duplicate final score. Failures after partial
  work use `OptimizerRunError` so that the environment still charges every
  completed call. The fixed-width node coordinate is mapped to the live
  frontier with `requested_node_index % len(frontier)`, avoiding the former
  bias that sent every out-of-range index to the last node. Episode `info`
  exposes the requested and effective indices, a per-step remap flag, and a
  cumulative remap count for validation provenance. Action masking remains a
  future enhancement. This module also still contains the legacy Hadamard
  specialization; moving that specialization is deferred.
- `inner_optimizers.py` contains reusable random-restart, CMA-ES, and SciPy
  minimization adapters.
- `training.py` owns controller training, evaluation, and baseline comparison.
- `polytope/` owns the ideal-polytope specialization:
  - `types.py` defines `PolytopeSolution`.
  - `task.py` defines the task and task sampler.
  - `optimizers.py` defines the dimension-specific hyperbolic optimizers.
    Compatibility class names retain the `LBFGSHyperbolicD*` spelling, but
    d=3 and the compatible d=4 path use strict-budget SciPy Powell. H⁴ also
    has the separate `FreshTopologyADHyperbolicD4`, which runs deterministic
    fixed-pole L-BFGS/Adam-to-L-BFGS branches, rebuilds Delaunay at every
    gradient closure, and canonically rescores each branch endpoint. D5 and
    d6 use projected Adam with a fresh triangulation at each gradient iterate.
    In particular,
    `LBFGSHyperbolicD5` applies Adam to the Schläfli differential; the former
    strong-Wolfe L-BFGS path incorrectly treated that gradient-only oracle as
    a scalar objective. D6 also uses Adam so topology and work stay predictable.
  - `h4_symmetric_benchmark.py` owns the reusable exact-N H⁴ benchmark
    primitive: deterministic unconstrained starts with an explicit Powell or
    fresh-topology-AD selector, exact-start SHA-256 binding for paired studies,
    strict all-pole endpoint auditing, conformal-Gram/Lorentz symmetry
    delegation, and an optional finite-difference local-maximum diagnostic
    after quotienting the six rotations and four boosts. The operational
    checkpoint/resume campaign remains in
    `scripts/mine_h4_symmetric_maxima.py`.
  - `h4_combinatorics_benchmark.py` owns strict prescribed-combinatorics H⁴
    grading. It recovers true supporting tetrahedra rather than trusting Qhull
    diagonals, compares the complete two-colored vertex--facet incidence graph
    up to one vertex relabeling, checks every pole's Delaunay decomposition
    against that hull, and applies a reference-free all-pole volume floor.
    `scripts/check_h4_combinatorics_response.py` is the strict-JSON operational
    adapter; private discovery coordinates are never part of its grading gate.

Calibrate the H⁴ verifier before a mining run:

```bash
python scripts/mine_h4_symmetric_maxima.py \
  --starts 0 --calibrate-16-cell --local-max-on-shortlist \
  --budget 1 --output /tmp/h4-16-cell-calibration.json
```

Then run a bounded exact-N pilot, for example:

```bash
python scripts/mine_h4_symmetric_maxima.py \
  --n-vertices 12 --starts 8 --seed 20260817 --budget 1000 \
  --local-max-on-shortlist --max-wall-seconds 180 \
  --output /tmp/h4-n12-symmetric-pilot.json
```

Select the experimental differentiable H⁴ arm explicitly with
`--optimizer fresh_topology_ad`; Powell remains the default so existing
campaigns and checkpoints do not silently change methods.
  - `env.py` assembles the polytope environment.
- `cage/` owns the high-girth cubic-graph specialization.
- `_legacy_polytope_env.py` preserves the pre-refactor environment only for
  compatibility and parity studies.

## Public entry points

Use these imports in new code:

```python
from ideal_poly_volume_toolkit.rl import GenericOptEnv, OptimizerRun, train_controller
from ideal_poly_volume_toolkit.rl.polytope import make_polytope_env
from ideal_poly_volume_toolkit.rl.cage import make_cage_env
```

The package-root modules `rl_generic.py`, `rl_optimizer.py`, and `rl_cages.py`
are compatibility shims for older import paths. They are not the owners of new
RL functionality. `fast_logdet.py` remains a package-root numerical helper
intended for the cage specialization; its incremental path is validated but
not yet wired into the current cage optimizers.

## Dependencies

The RL harness requires Gymnasium. CMA-ES and controller training additionally
use `cma` and Stable-Baselines3, respectively. Gymnasium and CMA are included
in the `dev` extra used by CI; the full research environment in
`requirements.txt` also includes Stable-Baselines3.

## Validation

- `scripts/tests/parity_rl_polytope.py` is the manual legacy-versus-current
  parity harness.
- `scripts/train_polytope_controller.py` trains one controller and atomically
  promotes a `controller-v2` model plus an
  `ideal-poly-rl-training/1` sidecar. The sidecar records the real Git/source
  state, Python/dependency/thread environment, exact PPO/task/action config,
  requested and actual timesteps, and model checksum. Safe run names and
  numeric ranges are validated; a per-target lock and default no-clobber
  policy protect long runs. Model and sidecar are staged as a pair; overwrite
  snapshots are restored if either promotion fails. `--dry-run` validates
  without training or writing.
- `scripts/validate_rl_controller.py` owns the version-2 matched-work study.
  Every method receives the same copied reset solution, the reset score counts
  against its cap, and a solo optimizer receives the entire remaining cap in
  one call. The registered baseline suite is random restart, surrogate CMA,
  true-objective CMA, the applicable dimension-local method, and their
  round-robin allocation. Seed-level rows retain decomposed work, failures,
  action-boundary anytime traces, wall/CPU time, and source/model/environment
  provenance; output is atomic, checkpointed, and no-clobber by default. Use
  `--smoke` for the pre-training protocol gate. The post-accounting full study
  has not yet been run; historical efficiency numbers remain superseded. A
  controller is not loaded until its versioned training sidecar, payload
  checksum, model checksum, dimension, action ordering, and timestep metadata
  pass the validator's integrity gate.

These research validations are intentionally separate from the fast default
pytest suite under `tests/`; `test_rl_harness.py`,
`test_h4_symmetric_benchmark.py`,
`test_cage_optimizer_accounting.py`, and `test_fast_logdet.py` cover the
accounting and incremental-update contracts. `test_rl_budget_contract.py`
exercises the hard work caps and matched-baseline accounting.
`test_rl_validator.py` covers the validator's one-call solo budget, stable
round-robin allocation, partial-failure accounting, task-cell aggregation,
and model/sidecar integrity gate. `test_rl_training_launcher.py` fault-injects
both overwrite and no-clobber pair promotion to verify rollback.
