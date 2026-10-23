# Changelog

All notable user-facing changes to the Ideal Polyhedron Volume Toolkit are
recorded here. The project follows semantic versioning from the 0.4 release
onward.

## [Unreleased]

The current worktree is being prepared for version 0.4.0. It has not been
tagged or published. See [the 0.4 release and migration
note](docs/release-0.4.0.md) for compatibility details and the release gates.

### Added

- A canonical H^3 optimization service in
  `ideal_poly_volume_toolkit.optimization`, shared by the command-line and GUI
  adapters.
- An installed `ideal-poly-optimize` command, owned by
  `ideal_poly_volume_toolkit.optimization_cli`; the former `bin/` script is a
  compatibility shim.
- A package-level `__version__` sourced from installed distribution metadata.
- Pole-aware sphere projection and a sphere-coordinate H^3 volume entry point.
- Explicit RL optimizer-work accounting for objective, surrogate, and gradient
  evaluations, with strict total-work budgets.
- Transactional incremental graph-Laplacian log-determinant support.
- Regression coverage for geometry invariance, gradients, optimization result
  accounting, RL budgets, cage optimizers, and incremental log determinants.
- A reproducible H^3 objective-throughput benchmark at
  `scripts/benchmark_h3_optimization.py`.
- A dedicated `geometry_torch` owner for optional differentiable H^3 kernels,
  with compatibility forwarding from `geometry`.
- Feature-specific `torch`, `plot`, `gui`, `graph`, `render`, and `rl`
  installation extras.
- Standard package license/project links and a source-distribution manifest
  that omits repository-only tests whose data and operational fixtures are not
  shipped in the sdist.
- Exact integer zero-shear PSL(2,Z) holonomy certificates, strict closed-sphere
  and finite-disk validators, and bounded worst-edge Rivin legalization.
- A versioned arithmetic-corpus schema and atomic, resumable producer with
  explicit infinity topology, canonical graph deduplication, deterministic
  layered-ring constructions, and bounded numerical witness campaigns.

### Changed

- H^3 objective evaluation now uses a vectorized double-precision
  Bloch--Wigner path instead of the former per-triangle Torch series loop.
- CLI and GUI optimization results use the same versioned schema. The full
  spherical boundary includes the explicit index of the implicit infinity
  vertex, while the legacy `triangulation` field remains an alias for the
  finite Delaunay triangulation.
- Optimization trial volumes are consistently positive. Invalid penalty-only
  trials serialize with `volume: null` and are excluded from statistics.
- The CLI `--popsize` option is documented and applied as SciPy's population
  multiplier, and the optimizer exposes an explicit worker count.
- Canonical RL imports now live under `ideal_poly_volume_toolkit.rl`; the
  former package-root modules remain compatibility shims for the 0.4 cycle.
- Gradio is constrained to its tested 6.x API.
- The base install now contains only the NumPy/SciPy/mpmath numerical stack;
  core H^3 geometry and optimization no longer import or install Torch.
- The supported Python range is 3.10 through 3.14. A normal
  dependency-resolved CPython 3.14.7 ARM64 environment passes the package,
  test, import, CLI, GUI, app, and CPU-autograd checks; x86-64 confirmation is
  delegated to CI, and free-threaded 3.14t is not currently claimed.
- Distribution sampling now uses process-count-independent seeded logical
  batches, bounded retries, at most 16 automatic workers, and explicit
  rejection, numerical-failure, completion, and effective-worker metadata.
  Bootstrap resampling uses the CLI seed without modifying NumPy's global RNG.
- Developer-only Torch gradient and optimizer diagnostics now have the
  explicit `scripts/_debug/` owner; the former flat paths remain executable
  compatibility launchers.
- Arithmetic corpus `--n-vertices` now means total ideal vertices, including
  the explicit infinity vertex. Exact abstract holonomy and floating
  coordinate/KKT/realization residuals are recorded as separate claims.

### Fixed

- Corrected Bloch--Wigner and Lobachevsky autodiff behavior.
- Corrected pole projection, positive rescaling invariance, coplanar hull-face
  reconstruction, and near-degenerate volume gradients.
- Corrected optimizer sign handling, population accounting, invalid-trial
  selection, and spherical-boundary topology reporting.
- Corrected RL best-solution bookkeeping, strict budget enforcement, stale
  triangulation caches, failure accounting, and several ineffective legacy
  moves.
- Corrected graph-swap/log-determinant transactions and disconnected-graph
  handling.
- Corrected distribution analysis so serial output is always an ndarray,
  unexpected evaluation errors propagate, tetrahedra are the minimum valid
  configuration, and small or invalid bootstrap counts are handled safely.
- Corrected distribution goodness-of-fit calls for SciPy 1.17/1.18 by binding
  fitted CDF parameters explicitly and selecting the new Anderson p-value
  method without breaking older supported SciPy versions.
- Corrected Penner-Rivin holonomy words to use one root frame and the proper
  orientation-independent crossing involution; missing shears can no longer
  default to zero, and small cross-ratios are no longer clamped to zero.
- Corrected arithmeticity semantics: near-integral traces of a selected
  generator basis are diagnostic only, open Delaunay disks cannot vacuously
  certify, and exact arithmetic claims require the zero-shear PSL(2,Z) path.

### Performance

- Vectorized Bloch--Wigner and triangle accumulation substantially reduce H^3
  objective-evaluation time. Reproducible claims must cite the benchmark
  harness, raw samples, source hashes, and execution environment rather than a
  single ad hoc timing.
- H^3 optimization with worker processes now keeps differential evolution
  parallel but runs L-BFGS-B polishing serially, avoiding fine-grained process
  dispatch overhead while preserving SciPy's acceptance and evaluation-count
  semantics.

### Deprecated

- Package-root RL import modules are compatibility shims; new code should
  import from `ideal_poly_volume_toolkit.rl`.
- Distribution analysis accepts `--series-terms` for compatibility, but the
  vectorized Bloch--Wigner path ignores it.
- Flat `scripts/debug_*.py` and
  `scripts/test_{from_bad_init,gradients,lbfgs_issue}.py` diagnostic paths are
  compatibility launchers through 0.4.x. Use `scripts/_debug/`; the launchers
  may be removed in 0.5.0.

### Known limitations

- Historical RL validation results predate strict work accounting and are not
  current efficiency evidence. Controllers must be retrained and revalidated.
- The H^4 Delaunay decomposition remains unreliable for certain highly
  symmetric non-generic point sets, including the 10--10 duoprism.
- Non-simplicial facet merging and the two-level facet optimizer still have
  documented correctness limitations.
- Incremental `fast_logdet` is validated independently but is not yet wired
  into the cage optimizers.
- Historical 20-, 50-, and 100-vertex arithmetic benchmark artifacts fail the
  current coordinate-shear/full-boundary certificate and are retained only as
  superseded provenance, not ground truth.

## Historical notes

The narrative [2025 development log](docs/CHANGELOG_2025.md) predates this
versioned changelog and is retained for provenance. It is not a statement of
the current support or known-issue status.
