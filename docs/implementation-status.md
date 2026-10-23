# Implementation Status

Snapshot as of 2026-08-17.

This document tracks what is working, what is broken, and the current reorganization migration. For open research questions see `tasks/todo.md`. For accumulated corrections see `tasks/lessons.md`.

The consolidated execution order and deferred decisions are recorded in
`docs/plans/codebase-improvement-roadmap.md`.

## Working

- **H^3 volume via Delaunay** — validated against known cases; correct for generic point sets.
- **H^3 volume via hull-project-back** — correct for generic point sets.
- **H^3 optimization service** — `optimization.py` is the canonical owner of
  parameterization, objective scoring, repeated differential evolution, and
  result accounting. CLI and GUI code should remain adapters over this service.
- **Installed H^3 optimizer command** — `ideal-poly-optimize` is owned by
  `optimization_cli.py`; the historical `bin/optimize_polyhedron.py` path is a
  compatibility shim over the same adapter.
- **Penner–Rivin holonomy and arithmeticity** — common-frame generators,
  stable coordinate shears, strict closed-sphere checks, and exact integer
  zero-shear PSL(2,Z) certificates run end-to-end. Near-integrality of a
  selected generator basis is retained only as a numerical diagnostic.
- **Strict zero-shear realization audit** — `rivin_zero_shear.py` reports
  normalization, distinctness, coherent topology, support planes, shears,
  signed bending, spherical embeddedness, coherent facet orientation, and
  exact labeled Qhull facet sets as separate predicates. Its reusable
  `solve_zero_shear_from_initial` entry accepts caller-supplied labeled CP1
  starts in either cube-root or standard-upper-half-plane gauge, while
  pole-link and weighted-Tutte construction remain explicit initialization
  helpers.
  `scripts/check_zero_shear_response.py` is the
  strict-JSON benchmark adapter, and
  `scripts/audit_tutte_zero_shear_basins.py` owns reproducible multi-start
  endpoint searches. `scripts/build_zero_shear_holdouts.py` exports
  topology-diverse, independently reaudited public prompts while keeping
  relabeling, source provenance, hardness evidence, and answer keys in a
  disjoint private package.
- **Fixed-combinatorics zero-shear method comparison** —
  `scripts/compare_rivin_zero_shear_methods.py` runs the pole-star Rivin-angle,
  pole-link harmonic/direct, and angle-then-polish routes against the same
  labeled sphere, ordered anchors, audit tolerances, and independent strict
  endpoint gate; the direct and hybrid routes also share identical coordinate
  solver settings. Its schema-v1 output records stage wall/process-CPU costs,
  method-specific solver diagnostics, and the hybrid's upstream dependency
  separately; raw iteration counts across the angle and coordinate objectives
  are not equivalent work measures. The optional outer-face Tutte route is a
  secondary diagnostic. Contract:
  `docs/schemas/rivin-zero-shear-comparison-v1.schema.json`.
- **H^4 volume via Schläfli** — matches known exact values for rectified 5-cell and first doubling (vol = 2π²/9, 4π²/9 respectively).
- **H⁴ symmetric-local-maximum pilot** — the reusable
  `rl/polytope/h4_symmetric_benchmark.py` layer performs strict all-pole
  candidate audits, delegates dimension-generic conformal-Gram/Lorentz
  symmetry certification to `symmetry.py`, and can numerically test
  stationarity and Hessian sign in a conformally balanced chart after
  quotienting the full 10-dimensional hyperbolic-isometry gauge.
  `scripts/mine_h4_symmetric_maxima.py` owns bounded deterministic
  multi-start campaigns and the
  `h4-symmetric-maxima-campaign-v1.schema.json` checkpoint contract. The ideal
  16-cell calibration recovers full group order 384, all-pole volume spread
  below `2e-15`, and a negative quotient Hessian. An initial eight-start N=12
  smoke (seed 20260817, budget 1000 per start) found combinatorial endpoint
  groups of orders 1, 2, 4, and 8 but geometric group order 1 in every case;
  this is a pipeline result, not a holdout. No newly mined N=12 or N=25 result
  becomes a benchmark until the campaign, independent audit, and private
  export stages are completed.
- **Fresh-topology H⁴ AD experiment** — `geometry_4d_torch.py` now supplies a
  vectorized differentiable projected-volume kernel, while
  `FreshTopologyADHyperbolicD4` owns deterministic fixed-pole branches and
  rebuilds Delaunay at every gradient closure. In a frozen four-seed N=25
  paired pilot at a 1200-work cap, the AD arm beat Powell on all four common
  starts: mean final volume `16.59939` versus `15.43897`, median paired gain
  `1.11029`, and median optimizer wall `3.91 s` versus `21.25 s`; every
  endpoint passed the independent strict all-pole audit. A subsequent frozen
  128-start AD campaign completed 128/128 strict endpoints with no failed
  branches (mean/max volume `16.56334`/`16.76475`) but found no exact
  nontrivial geometric group directly. Three endpoints had combinatorial
  order two. Unconstrained continuation preserved those involutions and drove
  their geometric-fit residuals into the `2.35e-7`--`5.46e-7` range. A
  separately recorded post-hoc conformal/orthogonal involution projection of
  one endpoint changed its balanced coordinates by only `5.68e-8` and yielded
  a strict 97-facet candidate of volume `16.7353830588` with full geometric
  group `C2` (orientation-reversing nonidentity). The full, symmetry-free
  65-dimensional quotient diagnostic was chamber-stable at finite-difference
  steps `5e-5`, `1e-4`, and `2e-4`: gradient norms
  `7.47e-7`--`7.51e-7` and Hessian spectrum approximately
  `[-6.49, -2.22]`, with all 65 eigenvalues negative. This is an adaptive,
  double-precision numerical local-maximum certificate, not an interval proof
  or a global maximum claim. A reference-free invariant response checker now
  grades strict ideal geometry, all-pole volume, and the exact full geometric
  group without comparing against private coordinates. The first prompt and
  answer are staged outside the repository pending evaluation; publishing
  either would deliberately burn that holdout.
- **H⁴ prescribed-combinatorics benchmark** —
  `rl/polytope/h4_combinatorics_benchmark.py` and
  `scripts/check_h4_combinatorics_response.py` grade strict ideal S³
  coordinates against the complete target vertex--tetrahedral-facet incidence
  type up to relabeling, together with all-pole Delaunay/facet consistency and
  a minimum-volume bound. They do not use a private-coordinate distance,
  symmetry, or local-Hessian gate. Generated prompts and answer keys remain
  outside the repository until evaluation; the certificate is numerical, not
  interval-validated.
- **Volume threshold oracle** (`volume_threshold.py`).
- **CLI and GUI** — `bin/optimize_polyhedron.py`, `bin/analyze_distribution.py`, `bin/gui.py` run locally.
- **H^3 objective benchmark** — `scripts/benchmark_h3_optimization.py`
  compares deterministic N in {7, 20, 50} inputs against the frozen 0.3
  scalar Torch path, reports numerical differences, and records raw timings
  with source/environment provenance on request.
- **Numerical property checks** — deterministic Hypothesis coverage exercises
  Bloch--Wigner identities, generic sphere-chart invariance, oriented Delaunay
  boundary closure, and incremental log-determinant swaps against direct
  recomputation.

## 0.4 release readiness

The migration guide and release gates live in `docs/release-0.4.0.md`; the
canonical version history is `CHANGELOG.md`. Version 0.4.0 has not been tagged
or published.

The supported metadata range is Python 3.10 through 3.14. A normal install on
official CPython 3.14.7 ARM64 passed the full test suite, all package-module
imports, wheel/editable construction, CLI/GUI/app smoke tests, and CPU
autograd. The x86-64 Python 3.14 CI job remains a release gate, and
free-threaded 3.14t is untested. A canonical performance CSV is intentionally
deferred until it can be generated from the final clean release commit.

## Known issues

- **10–10 duoprism and other highly symmetric point sets in H^4** — Delaunay-based pipeline gives pole-dependent (wildly wrong) answers. Needs cell-by-cell computation or a robust non-generic triangulation. See `tasks/todo.md` §"Volume computation for non-generic ideal polytopes".
- **Symmetric H⁴ benchmark scope** — version 1 certifies only strict simplicial
  candidates with bounded incidence-group enumeration and stable multi-pole
  volume. Non-simplicial cells, numerically ambiguous almost-symmetries, and
  formal interval local-maximality remain out of scope.
- **`scripts/optimize_facet_full.py`** — two-level facet optimization has a degeneracy bug; treat as broken.
- **Non-simplicial facet counting** — floating-point ConvexHull always triangulates; merging coplanar facets needs tolerance calibration. See `tasks/todo.md`.
- **Hodge spectral realization for d ≥ 5** — WIP; naive edge-eigenvector averaging does not work; see `tasks/todo.md` for the Hodge-decomposition plan.
- **Hyperbolic inradius (Chebyshev center)** — optimizer drifts to Klein-ball boundary; the formula itself is correct.
- **Legacy arithmetic benchmark data** — the checked-in 20-, 50-, and
  100-vertex arithmeticity artifacts predate coordinate-aware shears and the
  full infinity fan and fail the current certificate. They are classified as
  `superseded` in `results/manifest.json` and must not be used as ground truth.
- **Strict realization certificates remain numerical** — the zero-shear
  audit uses floating-point predicates and SciPy/Qhull. Positive support and
  bending margins are strong evidence, not an exact proof. Formal acceptance
  still needs a validated root enclosure plus interval-certified strict
  support, embedding, and bending inequalities.

## HF Spaces deployment

- `app.py` (repo root) is a thin shim that imports `bin.gui.create_gui` and launches for HuggingFace Spaces Docker SDK.
- `bin/gui.py` is the actual GUI; its general optimization tab delegates to
  the canonical package optimizer.
- The earlier HF Spaces deployment attempt was not successful. Gradio is now
  constrained to the tested 6.x API (`gradio>=6,<7`), and local GUI
  construction is covered by the release checks; the remote deployment still
  needs to be revalidated before redeployment.
- Neither file is dead on its own — they are complementary. If HF Spaces is formally abandoned, `app.py` can be removed.

## Documentation gaps being filled

The three docs imported by `CLAUDE.md` now exist. They describe the current
mixed layout: mostly flat numerical modules, the feature-grouped `rl/`
subpackage, and the canonical H^3 `optimization.py` service.

## Active reorganization

A staged reorganization is in progress. Stages in order:

- **Stage 0 — docs scaffolding.** Write the three missing `docs/` files above. *Status: done (commit 5b1c45e).*
- **Stage 1 — repository-root cleanup.** Move misplaced scripts, result files, and benchmark JSONs out of the root; archive `paper.zip` and `fivesix/`; add `.gradio/` to `.gitignore`. *Status: done (commit 5b1c45e).*
- **Stage 2 — `results/` cleanup.** Move loose JSON / TXT / log files from `results/` root into `results/data/` or `results/logs/`. *Status: pending.*
- **Stage 3 — `scripts/` grouping.** Partition the mostly flat `scripts/`
  directory into topical subdirectories (`polytopes_4d/`, `rl/`,
  `euclidean_deficit/`, `llm_benchmark/`, `facet_optimization/`,
  `inscribability/`, `_debug/`). *Status: partial — `inscribability/`, `tests/`,
  and `_debug/` exist. The five former flat debug paths are compatibility
  launchers through 0.4.x; the remaining groups are pending.*
- **Stage 4 — package subdivision.** Split `ideal_poly_volume_toolkit/` into feature subpackages where that improves ownership. *Status: partial/deferred — the `rl/` subpackage is in place and `optimization.py` now has explicit H^3 ownership; the remaining flat numerical layout is acceptable for now.*
- **Stage 5 — `examples/` cleanup.** Move loose examples into topical subdirectories. *Status: pending.*

A separate benchmark + data-recovery track runs alongside the reorg:

- **Stage A — H^5 hyperbolic volume recovery + hd-volume-LB benchmark.** *Status: done (commit b29cfa8).* Flagged a data-retention gap: d=5 configs at N ∈ {12, 14, 20} are suboptimal vs paper Table 5.
- **Stage B — RL harness modularization, design.** *Status: done.* Design document: `docs/plans/rl-harness-design.md`.
- **RL controller validation study.** *Status: protocol hardened; fresh
  controller-v2 training launched in the background on 2026-08-15.* Four
  dimension-specific seed-42 jobs use disjoint CPU pairs and a supervisor
  gates model-integrity smokes before the full comparison run. No new study
  result is accepted until that supervisor completes. The schema-v2 validator uses
  exact paired initial solutions, one-call solo caps, random-restart and
  true-objective-CMA baselines, decomposed action-boundary traces, robust
  failure rows, task-cell median/IQR aggregation, full provenance, and atomic
  no-clobber checkpoints. It refuses to load a controller unless the training
  schema/protocol, sidecar self-checksum, model checksum, task dimension,
  action ordering, and serialized timestep metadata verify. Its contracts are in
  `tests/test_rl_validator.py`. No post-hardening result has been written. See
  `docs/plans/rl-controller-validation.md`; the study remains the gate for
  deciding whether to extract the RL harness to its own repo.
  The paired training launcher now uses `controller-v2` artifact names and a
  versioned provenance sidecar with real source/environment hashes, requested
  and actual timesteps, model checksum, validated arguments, target locking,
  pair staging, overwrite snapshots with two-path rollback, and no-clobber
  cleanup. Launch logs live under
  `results/logs/rl_training/20260815T211114Z/`.

- **Arithmetic 300-vertex corpus.** *Status: producer/certificate gates and
  the bounded 20260815T221650Z production run completed.*
  `scripts/generate_arithmeticity_benchmark.py` now owns an atomic/resumable
  schema-v1 campaign. It interprets vertex counts as total vertices, closes
  the infinity fan, deduplicates complete spherical graphs, and separates an
  exact abstract zero-shear PSL(2,Z) proof from coordinate, KKT, realization,
  and shear residuals. The deterministic family supplies two exact
  300-vertex types and fifteen strict, pairwise non-isomorphic 302-vertex
  types. The legalized run accepted 32/40 attempts as 32 distinct
  300-vertex topologies in 234 seconds; every accepted boundary has
  V/E/F=300/894/596, maximum coordinate shear below 1.9e-10, maximum angle
  error below 1.5e-10 radians, and positive recorded strictness margins.
  Those are arithmetic-corpus acceptance gates, not strict convex-realization
  certificates: corpus acceptance does not establish spherical embeddedness,
  the signed Delaunay inequalities, or exact labeled convex-hull facets. Run
  `audit_zero_shear_realization` (or the strict response checker) separately
  before making that geometric claim.
  Generated files remain mutable operational outputs until publication review
  deliberately promotes immutable hashes into `results/manifest.json`.

- **Stage C — RL harness modularization, implementation.** *Status:
  structurally done* (commits c2e8e7a and 6b7af11). Canonical code lives
  under `ideal_poly_volume_toolkit/rl/`, while package-root modules preserve
  old imports. Historical validation covered 40 polytope cases across
  d ∈ {3,4,5,6}, N ∈ {7,10}, seeds {0..4}, loading a saved d=5 controller,
  and a cage smoke test. The previously reported speedup aggregates predate
  consistent optimizer-work accounting and should not be treated as current
  benchmark results.

The follow-up accounting pass introduces `OptimizerRun`, which distinguishes
objective, surrogate, and gradient evaluations. The environment now charges
the reported work while keeping surrogate values out of canonical node
statistics, and reuses an optimizer-supplied canonical solution value instead
of performing a duplicate final score. Compatibility class names are retained,
but their displayed algorithm descriptions reflect reality: the d=3 and d=4
paths use Powell, while d=5 and d=6 use projected Adam with a fresh
triangulation at every gradient iterate. The former strong-Wolfe L-BFGS
treatment of the d=5 Schläfli differential was not a valid scalar-objective
optimization, and unpredictable closures also prevented strict work budgeting.
Re-run the controller validation study before publishing new cross-optimizer
efficiency claims.

The pre-training action audit also replaced last-node clamping with
deterministic modulo mapping for the fixed-width node action. Requested versus
effective node indices and a cumulative remap count are exposed in Gymnasium
`info`; full action masking remains a future enhancement. Search nodes use
identity equality so splitting or pruning a non-first NumPy-backed node cannot
trigger ambiguous array comparison, and partial split failures retain exact
objective-work accounting.

The optional `fast_logdet` helper now commits graph swaps and Cholesky state
atomically, preserves accepted swaps across rebuilds, and returns `-inf` for
disconnected graphs; integration into the cage optimizers remains pending.
Cage optimizers report actual canonical and spectral surrogate calls instead
of padding histories to their requested budgets.

Two correctness bugs surfaced and fixed during C3:
  1. The pre-refactor env tracked an inconsistent mix of Euclidean and
     hyperbolic volumes at d ∈ {3, 4, 6} (§3.2a of the design doc).
     `IdealPolytopeTask` now has one consistent scoring function per
     dimension.
  2. `SearchNode.update_stats` (generic, shared by polytope and cage
     envs) was auto-updating `best_value` from the raw values list,
     which caused a pre-existing latent bug where `best_solution` could
     lag the advertised `best_value`. `update_stats` no longer touches
     `best_value` / `best_solution`; those are updated only by the
     caller after an explicit comparison. This fix is in the generic
     layer and also benefits the cage env.

Current ownership is documented in `docs/architecture.md`,
`docs/module-index.md`, and the nearest feature README, including
`ideal_poly_volume_toolkit/rl/README.md`. Research questions remain in
`tasks/todo.md`.
