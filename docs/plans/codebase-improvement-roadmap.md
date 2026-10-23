# Codebase Improvement Roadmap

**Status:** active, updated 2026-08-15. This roadmap consolidates the
correctness, performance, packaging, provenance, RL-validation, and structural
follow-ups identified during the 0.4 audit. The 0.4 release gates themselves
remain in `docs/release-0.4.0.md`.

## Operating rules

- A speedup is kept only after numerical parity is checked on fixed inputs.
- Scientific or RL efficiency claims require fixed work, paired initial state,
  seeds, robust summaries, wall time, and source/model provenance.
- New product logic receives one package owner. `bin/` and `scripts/` remain
  adapters or operational drivers.
- Data moves are forward-only and manifest-driven. Do not rewrite Git/LFS
  history or delete historical outputs as part of an unrelated refactor.
- Compatibility shims get an explicit lifetime; no large move is done in one
  step.

## Workstreams

| Workstream | Current checkpoint | Next gated work |
|---|---|---|
| Release baseline | 0.4 changelog/migration guide, Python 3.10--3.14 metadata and CI, build/install/CLI/GUI checks | Green x86-64 CI; clean-tree benchmark artifact; version bump/tag only after every release gate; publishing needs separate authorization |
| Numerical correctness | H³ identities, gradients, pole invariance, boundary topology, coplanar hulls, and incremental log determinants have deterministic regressions plus property tests | Robust non-generic H⁴ cell decomposition; calibrated coplanar-facet merging; replace the broken two-level facet optimizer; define tolerances against high-precision references |
| H³ optimization | Vectorized Bloch--Wigner scoring and parallel-DE/serial-polish split are in place | Productionize the analytic objective gradient only after topology-stratum, finite-difference, and end-to-end convergence tests; then benchmark time-to-solution, not just callback speed |
| Distribution analysis | Process-count-independent logical batches, bounded retries, seeded bootstrap, useful-worker cap, and explicit counters | Extract reusable sampling/fitting logic from `bin/` into a package-owned service; add a versioned output schema and an installed command; benchmark adaptive worker selection on representative machines |
| RL harness | Strict total-work accounting, fresh-topology local steps, failure ledgers, unbiased modulo node mapping, and a schema-v2 paired validator are implemented | Run protocol smoke and timing calibration; add action masking; train new controllers; run the preregistered paired study; publish no efficiency claim until the post-accounting study passes |
| Cage search | Cycle/girth logic and direct log-spanning-tree accounting are corrected; `fast_logdet` is transactional and property-tested | Integrate the incremental factorization behind an equivalence gate; vectorize graph validation; keep canonical-best and surrogate-current state distinct in every new optimizer |
| Packaging/API | Base H³ install is NumPy/SciPy/mpmath-only; Torch is lazy; feature extras and `ideal-poly-optimize` exist | Add installed distribution and GUI commands after their logic has package owners; define a top-level export/deprecation policy; add type checking gradually at public boundaries |
| Data/provenance | Scoped LFS rules, a schema-v1 manifest, canonical input hashes, full-clone CI audit, and canonical-first readers are in place | Classify remaining historical outputs before moving them; migrate `bin/results/` consumers; generate new canonical artifacts only from clean commits; document Hugging Face as deployment rather than backup |
| Repository structure | Ownership docs and the RL subpackage are current; the `_debug/` script slice has canonical paths plus 0.4 compatibility launchers | Move one low-coupling script cluster at a time (Euclidean deficit first), then 4D/facet/LLM clusters; do result cleanup before example cleanup; defer package subdivision unless ownership or dependencies demand it |
| Quality automation | Full pytest, deterministic Hypothesis, changed-file Ruff, import/compile, schema, LFS, wheel, CLI, GUI/app, and Docker static checks are defined | Retire the legacy full-tree Ruff backlog in feature-owned batches; add lowest-supported-dependency and scheduled-latest jobs; introduce a type checker module by module; turn the known Rivin integration warning into an explicit numerical contract |

## Execution order

1. **Close the 0.4 code checkpoint.** Run the complete local gate, review the
   diff, and let the Python 3.10--3.14 CI matrix validate x86-64. Keep the
   version at 0.3.0 until this is green.
2. **Record clean provenance.** Commit the reviewed checkpoint, generate the
   canonical H³ benchmark CSV from that clean commit, add its manifest row,
   and repeat the data-integrity gate.
3. **Calibrate RL cost.** Run only the protocol smoke and a short timing job.
   Use those measurements to choose transition counts, seeds, and parallel
   scheduling before authorizing the multi-hour training study.
4. **Run the post-accounting RL study.** Train new model identities, retain
   every configuration and checksum, and validate against the paired baseline
   grid. Update the superseded paper only from the resulting schema-v2 data.
5. **Take the next structural slice.** Move the Euclidean-deficit scripts with
   compatibility launchers and docs, then continue one cluster per change.
6. **Prototype analytic H³ gradients separately.** Require gradient parity on
   fixed combinatorial strata and compare quality per work unit before
   replacing derivative-free polishing.

## Actions that require a separate decision

- Publishing packages, updating the Hugging Face deployment, or pushing tags.
- Spending the full multi-dimensional RL training budget.
- Deleting/moving the legacy result corpus or rewriting Git LFS history.
- Removing compatibility shims before their documented deprecation window.
- Making non-generic H⁴ outputs authoritative before the geometric method is
  independently validated.

## Completion evidence

The fast local gate is:

```bash
python3 -m pytest -q
ruff check <changed Python paths>
python3 -m compileall -q ideal_poly_volume_toolkit
python3 scripts/check_data_integrity.py
python3 scripts/benchmark_h3_optimization.py --warmups 1 --samples 2 --loops 1
python3 -m build
```

Fresh artifact installation, the installed console command, schema-v1 tetra
output, GUI/HF-app construction, Docker validation, `git diff --check`, and the
full Python-version matrix are the remaining release-level checks described in
`docs/release-0.4.0.md`.
