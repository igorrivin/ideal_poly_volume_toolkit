# Module Index

One-line index of every first-party module, script, and entry point. Update whenever you add, rename, or move a module.

Structure note (2026-08): most numerical modules remain flat, while the RL
harness has moved to a feature-grouped subpackage. Further reorganization is
tracked in `docs/implementation-status.md`.

---

## Package: `ideal_poly_volume_toolkit/`

### Volume / geometry — H^3 (ideal polyhedra in hyperbolic 3-space)
- `geometry.py` — canonical H^3 geometry kernel: vectorized `bloch_wigner`,
  pole-aware `stereographic_project_from_pole`, sphere entry point
  `ideal_poly_volume_from_sphere`, and the established Delaunay and
  hull-project-back volume paths. Historical differentiable entries remain as
  lazy compatibility forwarders and do not make Torch a core dependency.
- `geometry_torch.py` — optional differentiable H^3 Lobachevsky and
  Bloch–Wigner Torch kernels, loaded through the `geometry.py` compatibility
  surface only when a Torch API is used.
- `optimization.py` — canonical reusable H^3 parameterization, objective,
  differential-evolution trial orchestration, boundary reconstruction, and
  result accounting shared by CLI/GUI adapters. Public functions:
  `spherical_to_complex`, `parameters_to_complex`,
  `negative_volume_objective`, `complete_boundary_triangulation`,
  `analyze_ideal_polyhedron_structure`, `optimize_ideal_polyhedron`, and
  `optimization_result_to_dict`; result records: `OptimizationTrial` and
  `PolyhedronOptimizationResult`.
- `optimization_cli.py` — installed `ideal-poly-optimize` argument parsing,
  progress reporting, and schema-v1 JSON output; the source-tree
  `bin/optimize_polyhedron.py` is a compatibility shim.
- `visualization.py` — 3D rendering, Klein / Poincaré ball plots, mesh export.

### Volume / geometry — H^4 and higher
- `geometry_4d.py` — Hyperbolic volume of ideal 4-simplices in H^4.
- `geometry_4d_torch.py` — optional vectorized Torch implementation of the
  projected ideal-H⁴ tetrahedral volume, including a differentiable
  translation/scale gauge for fresh-topology optimization.
- `simplex_volume_4d.py` — Arbitrary-precision volume of an ideal 4-simplex `[∞, v₁, v₂, v₃, v₄]` in H^4.
- `hyperbolic_4_simplex_volume.py` — Quadrature-free volume of a hyperbolic 4-simplex from Klein-model vertices.
- `hyperbolic_4_simplex_volume_models.py` — Same, abstracted across hyperbolic models.
- `hyperbolic_5_6_simplex_volume.py` — Numerical volume routines for compact hyperbolic 5- and 6-simplices.
- `hyperbolic_model_transforms.py` — Transformations between Klein / Poincaré / hyperboloid / upper-half-space models.
- `h5_volume_torch.py` — Differentiable ideal H^5 polytope volume optimization via the Schläfli formula.
- `li2_torch.py` — Pure-torch principal-branch dilogarithm used by differentiable higher-dimensional volume routines.
- `murakami_torch.py` — Differentiable Murakami S^3 link volumes and ideal H^6 polytope support.
- `rl/polytope/h4_symmetric_benchmark.py` — reusable unconstrained H⁴
  multi-start primitive, strict multi-pole candidate audit, and optional
  finite-difference local-maximum diagnostic on the tangent quotient by
  `PO(4,1)`.
- `rl/polytope/h4_combinatorics_benchmark.py` — strict prescribed-facet H⁴
  audit: supporting tetrahedra, full colored incidence isomorphism up to vertex
  relabeling, all-pole Delaunay/facet agreement, and reference-free volume
  threshold grading.

### Rivin / Penner holonomy pipeline
- `rivin_holonomy.py` — Penner–Rivin Fuchsian-group generators, strict
  closed-sphere validation, coordinate-shear diagnostics, and exact integer
  zero-shear PSL(2,Z) certificates. Generator-trace near-integrality alone is
  diagnostic and is not an arithmeticity proof.
- `rivin_zero_shear.py` — canonical strict numerical audit for a normalized,
  labeled CP1 realization: anchors, handedness/distinctness, coherent sphere
  topology, support planes and facet orientation, shear equality, signed
  bending, spherical embeddedness, and exact labeled Qhull facet-set
  agreement. It also owns pole-link and weighted-Tutte initialization helpers,
  shared CP1 normalization, and `solve_zero_shear_from_initial` for reusable
  cube-root or standard-upper-half-plane solves. Its planar straight-edge result
  is diagnostic rather than an acceptance gate.
- `rivin_delaunay.py` — Rivin's Delaunay-realizability and volume-critical
  angle solver, including bounded worst-edge legalization, strict disk
  topology/winding validation, and solver/realization residual reports.
- `pointset_to_fuchsian.py` — Full pipeline: point set → triangulation → Fuchsian group → invariant trace field.

### Realization and inscribability
- `geometric_realization.py` — Geometric realization from Rivin LP angles.
- `cone_realization.py` — Inscribed realization of simplicial 4-polytopes via the cone / Volkov method.
- `inscribed_realization.py` — Inscribed realization of simplicial 4-polytopes on S^3.
- `volume_threshold.py` — Self-contained volume-exceeds-threshold oracle.

### Combinatorics
- `combinatorial_enumeration.py` — Combinatorial type enumeration for convex polyhedra inscribed in the sphere.
- `planar_utils.py` — Utilities for planar graphs and embeddings.
- `plantri_interface.py` — Interface to `plantri` for enumerating planar triangulations.
- `symmetry.py` — legacy H³ graph symmetries plus the dimension-generic strict
  ideal-polytope audit that filters full vertex--facet incidence
  automorphisms through conformal Gram/ray-scaling checks for projective
  Lorentz isometries.

### RL-guided search
- `rl/generic.py` — task/optimizer interfaces, `OptimizerRun` accounting
  (`objective_evals`, `surrogate_evals`, `gradient_evals`, and an optional
  cached `solution_value`), search nodes, and the generic Gymnasium
  environment; the legacy Hadamard specialization remains here pending
  cleanup.
- `rl/inner_optimizers.py` — reusable random-restart, CMA-ES, and SciPy minimization adapters.
- `rl/training.py` — controller training, evaluation, and baseline comparison.
- `rl/polytope/types.py` — `PolytopeSolution` value type.
- `rl/polytope/task.py` — ideal-polytope task and task sampler.
- `rl/polytope/optimizers.py` — dimension-specific H^3/H^4/H^5/H^6 inner
  optimizers: compatible Powell for d=3/d=4, a separate fixed-pole
  fresh-Delaunay Torch AD optimizer for H⁴, and fresh-topology projected Adam
  for d=5/d=6; legacy class names remain compatible.
- `rl/polytope/env.py` — `make_polytope_env` environment assembly.
- `rl/cage/task.py` — high-girth cubic-graph task, optimizers, and environment assembly.
- `rl/_legacy_polytope_env.py` — pre-refactor implementation retained only for compatibility and parity studies.
- `rl_generic.py`, `rl_optimizer.py`, `rl_cages.py` — compatibility shims for the former package-root import paths.
- `fast_logdet.py` — transactional incremental graph-Laplacian log determinant
  available to the cage specialization; disconnected states are rejected, but
  the current cage optimizers still use direct eigenspectrum calculations.

See `ideal_poly_volume_toolkit/rl/README.md` for canonical imports and
validation commands.

---

## Entry points

- `ideal-poly-optimize` — installed H^3 optimizer command backed by
  `ideal_poly_volume_toolkit.optimization_cli`.
- `bin/gui.py` — Gradio GUI (local). Main interactive entry point.
- `bin/optimize_polyhedron.py` — compatibility shim for the installed optimizer
  command.
- `bin/analyze_distribution.py` — CLI wrapper for volume-distribution analysis.
- `bin/sample_size_calculator.py` — Sample-size helper.
- `app.py` — Thin HF Spaces shim that imports `bin/gui.py` (see implementation-status.md).

Packaging metadata is owned by `pyproject.toml`; `MANIFEST.in` keeps the sdist
focused on the installable package and omits repository tests whose fixtures
and operational resources are intentionally not distributed.

---

## Scripts

`scripts/` is mostly flat; `inscribability/`, `tests/`, and `_debug/` are the
first grouped areas. Further grouping is planned (see
implementation-status.md).

Main clusters (by filename prefix / purpose):

- **4D / ideal polytopes** — `ideal_4polytope_volume.py`, `ideal_4simplex_volume.py`, `optimize_ideal_4polytopes.py`, `optimize_ideal_by_normals.py`, `sweep_volume_distribution_4d.py`, `optimize_surface_area_4d.py`.
- **Facet-parameterized optimization** — `optimize_by_facets.py`, `optimize_facet_full.py`, `optimize_fixed_facets.py`, `quick_facet_scan.py`, `save_facet_results.py`, `ideal_from_normals.py`, `optimize_projected_volume.py`.
- **Klein / S^3 / annealing** — `optimize_klein_volume.py`, `anneal_S3_volume.py`, `facet_distribution_S3.py`.
- **Euclidean deficit** — `euclidean_deficit.py`, `euclidean_deficit_large_N.py`, `euclidean_deficit_4d.py`, `euclidean_deficit_4d_large.py`.
- **Inscribed-polytope optimization** — `optimal_inscribed_polytopes.py`, `optimal_inscribed_large.py`.
- **Distribution sweeps** — `sweep_vertex_distribution.py`, `run_60vertex_distribution.py`, `fix_sweep_outliers.py`.
- **Cross-validation** — `cross_validate_volume.py`, `verify_lob_derivative.py`.
- **LLM benchmark** — `llm_polytope_benchmark.py`, `generate_random_challenges.py`, `generate_missing_challenges.py`, `generate_arithmeticity_benchmark.py` (schema-v1 atomic/resumable exact layered-ring and legalized arithmetic-corpus producer; contract at `docs/schemas/arithmetic-corpus-v1.schema.json`), `check_zero_shear_response.py` (strict normalized labeled-response grader over the package audit, with optional private-reference gating), `check_h4_symmetric_response.py` (reference-free invariant grader for strict ideal-H⁴ coordinates with an all-pole volume floor and an exact full geometric symmetry group), `check_h4_combinatorics_response.py` (reference-free strict H⁴ grader for a complete public tetrahedral-facet incidence type up to relabeling plus an all-pole volume floor), `audit_tutte_zero_shear_basins.py` (repeatable multi-start endpoint search; contract at `docs/schemas/tutte-zero-shear-audit-v1.schema.json`), `compare_rivin_zero_shear_methods.py` (same-combinatorics angle/direct/hybrid experiment with common strict audits and dependency-aware timing; contract at `docs/schemas/rivin-zero-shear-comparison-v1.schema.json`), `build_zero_shear_holdouts.py` (deterministic topology-diverse prompt export with disjoint public prompts and private answer keys/provenance), `mine_h4_symmetric_maxima.py` (atomic/resumable exact-N unconstrained H⁴ campaign with 16-cell calibration and strict symmetry shortlisting; contract at `docs/schemas/h4-symmetric-maxima-campaign-v1.schema.json`), `generate_hd_volume_lb_benchmark.py` / `check_hd_volume_lb_response.py` (H^d max-volume LB; hyperbolic only), `generate_volume_lb_benchmark.py` / `check_volume_lb_response.py` (5-regime construct-a-record-polytope benchmark: Euclidean d∈{4,5,6} + Hyperbolic d∈{4,6}), `recompute_h5_volumes.py` (recomputes H^5 volumes from stored d=5 configs), `export_face_lattice.py`, `standalone_checker.py`, `verify_llm_polytope.py`, `build_standalone_benchmark.sh`.
- **Inscribability checks** (`scripts/inscribability/`) — `check_inscribable_12.py`, `check_inscribable_range.py`, `check_inscribable_range_4connected.py`.
- **RL controller operations** — `train_polytope_controller.py` validates and
  transactionally promotes a versioned model/provenance-sidecar pair with
  rollback;
  `validate_rl_controller.py` runs the schema-v2 paired-work protocol with
  checkpointed output and a strict controller-integrity gate;
  `try_improve_h3_records.py` performs record searches.
  Manual parity harness: `scripts/tests/parity_rl_polytope.py`.
- **Migration / validation** — `validate_migration.py`;
  `check_data_integrity.py` verifies Git LFS coverage and payloads, the result
  manifest schema and artifact hashes, plus the classification and checksums
  of canonical result inputs.
- **Performance / release validation** —
  `benchmark_h3_optimization.py` compares the canonical vectorized H^3
  objective with a frozen 0.3 scalar Torch reference on deterministic
  N in {7, 20, 50}, checks their documented numerical tolerance, and can emit
  provenance-rich raw timing CSV. It is an objective-throughput benchmark,
  not a time-to-solution study.
- **Manual diagnostics** (`scripts/_debug/`) — `debug_gradient_sign.py`,
  `debug_original.py`, `test_from_bad_init.py`, `test_gradients.py`, and
  `test_lbfgs_issue.py`. These are not pytest tests. Same-named files directly
  under `scripts/` are compatibility launchers retained through 0.4.x.

---

## Examples

`examples/` contains both loose top-level files and subdirectories (`distributions/`, `enumeration/`, `optimization/`, `analysis/`, `visualization/`, `llm_benchmark/`). See `examples/README.md`.

## Plans

- `docs/plans/codebase-improvement-roadmap.md` — active staged roadmap for
  correctness, performance, packaging, provenance, RL revalidation, and
  structural migrations; records gates and decisions that remain deferred.
- `docs/plans/rl-harness-design.md` — historical Stage-B modularization design,
  retained for provenance; live interfaces are documented in
  `ideal_poly_volume_toolkit/rl/README.md`.
- `docs/plans/rl-controller-validation.md` — current design for retraining and
  rerunning the controller study under explicit optimizer-work accounting.

## Release documentation

- `CHANGELOG.md` — canonical versioned user-facing change history.
- `docs/release-0.4.0.md` — draft 0.4 migration guide, reproducible benchmark
  protocol, and release checklist.
- `docs/schemas/optimization-result-v1.schema.json` — machine-readable contract
  for shared CLI/GUI optimizer output.
- `docs/schemas/tutte-zero-shear-audit-v1.schema.json` — checkpointable,
  provenance-bearing contract for the Tutte-start/multi-start zero-shear basin
  audit and its endpoint classifications.
- `docs/schemas/rivin-zero-shear-comparison-v1.schema.json` — provenance-rich
  contract for the fixed-combinatorics angle/direct/hybrid zero-shear
  comparison, including stage wall/CPU timings and hybrid dependency cost.
- `docs/schemas/h4-symmetric-maxima-campaign-v1.schema.json` — checkpointable
  contract for exact-N unconstrained H⁴ searches, ideal-16-cell calibration,
  multi-pole volume checks, geometric-symmetry shortlisting, and optional
  quotient local-maximum diagnostics.
- `docs/schemas/result-manifest-v1.schema.json` and `results/manifest.json` —
  machine-readable provenance contract and the repository's explicit artifact
  inventory.
- `docs/CHANGELOG_2025.md` — historical narrative development log; not the
  current known-issues or release record.

## Tests

Pytest is configured to discover `test_*.py` only under `tests/`. The suite
covers Penner–Rivin regressions (`test_penner_rivin_regression.py`), package
imports, model transforms, coplanar facets, gradients, and threshold behavior
(`test_correctness_regressions.py`), plus the H^3 geometry kernel
(`test_geometry.py`), reusable optimization service (`test_optimization.py`),
deterministic numerical invariants across generic randomized strata
(`test_numerical_properties.py`),
exact/common-frame holonomy and strict topology (`test_rivin_holonomy_exact.py`),
Rivin residual/legalization behavior (`test_rivin_legalization.py`), and the
versioned atomic arithmetic-corpus contract (`test_arithmetic_corpus.py`),
strict realization predicates (`test_rivin_zero_shear.py`) and strict
private-key/model-response grading (`test_zero_shear_checker.py`),
higher-dimensional Lorentz symmetry (`test_hd_geometric_symmetry.py`), H⁴
symmetric-candidate optimization/auditing (`test_h4_symmetric_benchmark.py`),
and its checkpointed campaign contract
(`test_h4_symmetric_maxima_campaign.py`),
process-count-independent distribution sampling, bounded retry accounting, and
seeded bootstrap fitting (`test_distribution.py`),
RL search/accounting semantics (`test_rl_harness.py`), strict optimizer-budget
contracts (`test_rl_budget_contract.py`), validator pairing, failure, trace,
checkpoint, controller integrity, and task-cell aggregation contracts
(`test_rl_validator.py`),
training-launcher naming, validation, provenance, checksum, locking, and
transactional pair-rollback contracts (`test_rl_training_launcher.py`),
cage optimizer work accounting
(`test_cage_optimizer_accounting.py`), and transactional spectral updates
(`test_fast_logdet.py`). The release benchmark's fixed cases, compatibility
check, raw CSV shape, and source-hash provenance have a fast subprocess smoke
test in `test_release_benchmark.py`.
`llm_benchmark_test.py` is a manually invoked benchmark-data generator, not a
pytest module. Volume-threshold demonstrations currently remain under
`examples/`. `test_script_layout.py` protects the staged script moves and
their compatibility launchers without executing developer diagnostics.

## Paper sources

`paper/` holds LaTeX sources for:
- `maximal_ideal_polyhedra.tex` — main paper.
- `deficit_constants.tex` — Euclidean volume deficit note.
- `llm_volume_note.tex` — LLM benchmark note.
- `hyperbolic_5_6_simplex_note.tex` — compact 5/6 simplex volume note.
- `rl_validation_note.tex` — superseded historical RL validation note; its
  pre-accounting numerical claims require a fresh run before publication.
- Prior snapshots in `paper/archive/` (if present).
