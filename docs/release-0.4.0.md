# Version 0.4.0 Release and Migration Note

**Status:** draft release candidate. Version 0.4.0 has not been tagged or
published.

This document is both the migration guide from 0.3.0 and the release checklist
for 0.4.0. The checklist is intentionally conservative: scientific
correctness, installability, and reproducible performance evidence are release
requirements.

## Migrating from 0.3.0

### Python and installation

- Python 3.10 is the new minimum because the package uses modern type syntax.
- The supported range is Python 3.10 through 3.14, as documented in the
  project README and `pyproject.toml`.
- A normal current-metadata install on official CPython 3.14.7 ARM64 passed
  all declared base/development dependency installation, the full test suite,
  all package-module imports, wheel and editable builds, CLI/GUI/app smoke tests,
  and CPU autograd. GitHub's x86-64 3.14 job remains a release gate.
  Free-threaded 3.14t is untested and is not part of the support claim.
- On that ARM64 environment, `pip check` reported only an upstream
  Torch/`nvidia-cusparselt` platform-tag warning; the x86-64 CI install must
  confirm ordinary resolver health independently.
- Install the NumPy/SciPy/mpmath core with `python3 -m pip install -e .`.
  Torch is no longer installed merely to compute or optimize ordinary H^3
  volumes. Install `.[torch]` for differentiable H^3 and higher-dimensional
  Torch routines.
- Optional feature groups are `plot`, `gui`, `graph`, `render`, and `rl`.
  They can be combined, for example `.[gui,graph]`. The `dev` extra remains
  the complete test/lint/GUI environment used by CI.
- The broader RL/research environment uses the full convenience list in
  `requirements.txt` followed by `python3 -m pip install -e .`.
  `requirements_HPC.txt` is a machine snapshot, not a portable installation
  method.

### H^3 optimization API

Reusable optimization code now belongs to
`ideal_poly_volume_toolkit.optimization`. Import the service rather than
copying the numerical objective from a CLI or GUI module:

```python
from ideal_poly_volume_toolkit.optimization import optimize_ideal_polyhedron

result = optimize_ideal_polyhedron(n_vertices=7, n_trials=2, seed=42)
```

`bin/optimize_polyhedron.py` and `bin/gui.py` are adapters over this service.
They should not acquire independent parameterizations, volume kernels, or
trial-selection rules.

The package now installs `ideal-poly-optimize`, so optimization works from an
installed wheel without relying on the repository's `bin/` directory:

```bash
ideal-poly-optimize --vertices 7 --trials 2 --seed 42
```

`python bin/optimize_polyhedron.py ...` remains a compatible source-tree
spelling and delegates to the same `optimization_cli` adapter.

### Optimization result schema

CLI and GUI output now share schema version 1. Its machine-readable contract
is `docs/schemas/optimization-result-v1.schema.json`; generated payloads are
validated against it in the test suite.

- `best.volume` is positive even though SciPy minimizes an internal negative
  objective.
- Invalid penalty-only trials have `volume: null` and `valid: false`; they do
  not contribute to volume statistics.
- `finite_delaunay_triangulation` contains only finite planar faces.
- `boundary_triangulation` is the complete spherical boundary and uses
  `best.structure.implicit_infinity_index` for the added infinity vertex.
- `triangulation` remains a compatibility alias for the finite triangulation.
- Result filenames identify whether the CLI or GUI produced them.

Consumers should branch on `schema_version`, use the explicit finite or
boundary field they require, and stop inferring a spherical Euler
characteristic from the finite planar disk.

### Arithmetic corpus and holonomy semantics

Arithmetic corpus generation now uses
`ideal-poly-arithmetic-corpus/1` (`docs/schemas/arithmetic-corpus-v1.schema.json`).
Its vertex count is total, including an explicit JSON `null` for infinity, and
its triangulation is the complete closed sphere. Exact integer holonomy for a
declared zero-shear structure is recorded separately from floating coordinate,
KKT, realization, and shear residuals. Near-integral traces of a selected
generator basis are no longer described as an arithmeticity certificate.

The historical 20-, 50-, and 100-vertex arithmetic artifacts predate those
checks and are marked `superseded`; they are provenance only and must not be
used as release or publication ground truth.

### CLI and distribution behavior

- `--popsize` is SciPy's population multiplier. The nominal differential
  evolution population is the multiplier times the parameter dimension.
- `--workers` controls differential-evolution worker processes; `-1` requests
  all available CPUs.
- Distribution analysis uses the vectorized Bloch--Wigner kernel.
  `--series-terms` remains accepted as an ignored compatibility option.
- Distribution requests require at least four vertices and use deterministic
  256-sample logical batches. The seed controls each batch independently, so
  changing `--jobs` does not change the ordered sample stream or counters.
  Automatic concurrency is capped at 16 useful workers. Rejected draws and
  configurations are retried within documented finite limits, with underfill,
  known numerical failures, and the effective worker count recorded explicitly
  in saved metadata. Bootstrap resampling is local and uses the CLI seed.

### Numerical behavior

The standard H^3 path now uses the vectorized SciPy Bloch--Wigner evaluator.
Tiny differences from the former 96-term Lobachevsky series are expected. The
release benchmark reports both absolute and relative differences against a
frozen 0.3 scalar Torch reference before it records timings.

Torch-only H^3 kernels now belong to
`ideal_poly_volume_toolkit.geometry_torch`. Existing imports such as
`from ideal_poly_volume_toolkit.geometry import lob_fast` remain supported by
lazy forwarders. Importing `geometry` or `optimization` no longer imports
Torch; calling a differentiable entry without `.[torch]` raises an actionable
installation error.

Sphere-coordinate callers should prefer `ideal_poly_volume_from_sphere` and
`stereographic_project_from_pole`; those paths normalize sphere
representatives and keep the selected pole/index mapping explicit.

### RL compatibility

New code should import from `ideal_poly_volume_toolkit.rl` and its `polytope`
or `cage` subpackages. The old `rl_generic`, `rl_optimizer`, and `rl_cages`
modules remain shims during the 0.4 compatibility cycle.

An optimizer budget is now a strict cap on total counted work: canonical
objective calls, surrogate calls, and gradient evaluations. Historical
controllers and validation aggregates predate this contract. They may be used
for provenance/loadability checks, but must not be presented as current
efficiency results; retrain and rerun the validation study first.

### Known scientific limitations

The release does not resolve every research limitation. In particular:

- special non-generic H^4 configurations can invalidate the current Delaunay
  decomposition;
- floating-point hull triangulation does not yet provide a calibrated general
  merger for non-simplicial facets;
- `scripts/optimize_facet_full.py` remains affected by a degeneracy bug;
- the hyperbolic inradius optimizer and d >= 5 Hodge realization remain work
  in progress; and
- `fast_logdet` is not yet integrated into cage search.

See `docs/implementation-status.md` and `tasks/todo.md` for the authoritative
details.

### Repository script layout

Manual Torch gradient and optimizer diagnostics now live under
`scripts/_debug/`, their explicit owner. The five former paths directly under
`scripts/` remain executable compatibility launchers throughout 0.4.x and may
be removed in 0.5.0. They are exploratory diagnostics, not installed commands
or pytest tests.

## Reproducible optimization benchmark

Run the objective-throughput benchmark without writing a file:

```bash
python3 scripts/benchmark_h3_optimization.py
```

It uses deterministic parameter vectors for N = 7, 20, and 50, compares the
current objective with the frozen 0.3 scalar Torch/96-term-series path, checks
their documented numerical tolerance, warms both implementations, and prints
median timings from raw `perf_counter_ns` samples.

To inspect raw samples during development:

```bash
python3 scripts/benchmark_h3_optimization.py \
  --output /tmp/h3_optimization_benchmark.csv
```

A canonical release baseline must be generated only from the final clean
release commit:

```bash
python3 scripts/benchmark_h3_optimization.py \
  --require-clean \
  --output results/data/performance/h3_optimization_0.4.0.csv
```

Do not interpret these microbenchmarks as time-to-solution or optimizer-quality
comparisons. They measure steady-state evaluation of the H^3 objective. Any
broader optimization claim needs fixed work, fixed seeds, solution-quality
statistics, and wall-time reporting.

## Release checklist

### Scope and metadata

- [ ] The release work is split into reviewable logical commits, with no
      unrelated user changes included.
- [ ] `CHANGELOG.md` contains the final 0.4.0 date and matches the shipped
      behavior.
- [ ] `pyproject.toml`, built metadata, documentation, and any exposed package
      version agree on `0.4.0`.
- [ ] The compatibility-shim lifetime and all deprecations are documented.

### Python compatibility

- [ ] Python 3.10 installs normally and passes the full suite.
- [ ] Python 3.11 installs normally and passes the full suite.
- [ ] Python 3.12 installs normally and passes the full suite.
- [ ] Python 3.13 installs normally and passes the full suite.
- [ ] Python 3.14 installs normally and passes the full suite in x86-64 CI
      (the official CPython 3.14.7 ARM64 gate already passes).
- [ ] CLI construction and GUI/HF-app construction pass on the versions
      selected in CI for those smoke tests.

### Correctness and static checks

- [ ] `python3 -m pytest -q` passes without an unexplained new warning.
- [ ] `ruff check <changed Python paths>` passes.
- [ ] `git diff --check` passes.
- [ ] Package-module import smoke and `python3 -m compileall -q
      ideal_poly_volume_toolkit` pass.
- [ ] A seeded tetrahedron CLI run produces schema version 1, positive volume,
      four boundary vertices, Euler characteristic two, and JSON that round
      trips through the standard library.
- [ ] `create_gui()` and the root HF app construct without a Gradio API warning.

### Distribution artifacts and packaging

- [ ] `python3 -m build` produces both an sdist and a wheel from a clean tree.
- [ ] Each artifact installs into a fresh environment and `python3 -m pip
      check` passes.
- [ ] The installed wheel imports the package and runs the CLI help path
      without relying on repository-relative files.
- [ ] Wheel/sdist contents contain no certificates, secrets, caches, generated
      result corpus, or unintended large files.
- [ ] A fresh clone with Git LFS has every required object, and LFS rules no
      longer make small source-like metadata unexpectedly unavailable.

### Performance and scientific provenance

- [ ] The benchmark's frozen/current values satisfy its compatibility
      tolerance for N = 7, 20, and 50.
- [ ] Raw benchmark samples are generated from the clean release commit and
      record commit state, source hashes, environment, seed, warmups, and loop
      counts.
- [ ] Changelog performance language says "objective evaluation" unless a
      separate fixed-work time-to-solution study supports a broader claim.
- [ ] Historical RL outputs and the paper retain their superseded warning.
- [ ] Every known scientific limitation above remains visible in
      `docs/implementation-status.md`.

### Finalization

- [ ] CI is green on the final release commit.
- [ ] The exact artifacts intended for release have been reviewed.
- [ ] An annotated `v0.4.0` tag is created from that commit.
- [ ] Publishing to PyPI, Hugging Face, or another external service is
      performed only after explicit authorization.
