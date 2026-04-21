# Module Index

One-line index of every first-party module, script, and entry point. Update whenever you add, rename, or move a module.

Structure note (2026-04): the repository is flat within each directory. A reorganization into feature-grouped subpackages is tracked in `docs/implementation-status.md`.

---

## Package: `ideal_poly_volume_toolkit/`

### Volume / geometry — H^3 (ideal polyhedra in hyperbolic 3-space)
- `geometry.py` — stereographic projection, Delaunay triangulation, hull-project-back, tetrahedral ideal volume via Lobachevsky/Bloch–Wigner.
- `murakami_torch.py` — Differentiable Murakami S^3 tetrahedral volume in PyTorch.
- `li2_torch.py` — Pure-torch principal-branch dilogarithm Li₂(z).
- `visualization.py` — 3D rendering, Klein / Poincaré ball plots, mesh export.

### Volume / geometry — H^4 and higher
- `geometry_4d.py` — Hyperbolic volume of ideal 4-simplices in H^4.
- `simplex_volume_4d.py` — Arbitrary-precision volume of an ideal 4-simplex `[∞, v₁, v₂, v₃, v₄]` in H^4.
- `hyperbolic_4_simplex_volume.py` — Quadrature-free volume of a hyperbolic 4-simplex from Klein-model vertices.
- `hyperbolic_4_simplex_volume_models.py` — Same, abstracted across hyperbolic models.
- `hyperbolic_5_6_simplex_volume.py` — Numerical volume routines for compact hyperbolic 5- and 6-simplices.
- `hyperbolic_model_transforms.py` — Transformations between Klein / Poincaré / hyperboloid / upper-half-space models.
- `h5_volume_torch.py` — Differentiable ideal H^5 polytope volume optimization via the Schläfli formula.

### Rivin / Penner holonomy pipeline
- `rivin_holonomy.py` — Penner–Rivin Fuchsian-group generators from an ideal triangulation.
- `rivin_delaunay.py` — Rivin's algorithm for checking Delaunay realizability of a triangulation.
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
- `symmetry.py` — Symmetry groups of ideal polyhedra via nauty / pynauty.

### RL-guided search
- `rl_optimizer.py` — Hierarchical RL-guided optimizer for ideal-polytope volume maximization.
- `rl_generic.py` — Generic hierarchical RL optimizer for black-box optimization.
- `rl_cages.py` — RL-guided search for high-girth cubic graphs (cage problem).
- `fast_logdet.py` — Fast incremental log-det computation for graph Laplacians (used by `rl_cages`).

---

## Entry points

- `bin/gui.py` — Gradio GUI (local). Main interactive entry point.
- `bin/optimize_polyhedron.py` — CLI wrapper for optimization.
- `bin/analyze_distribution.py` — CLI wrapper for volume-distribution analysis.
- `bin/sample_size_calculator.py` — Sample-size helper.
- `app.py` — Thin HF Spaces shim that imports `bin/gui.py` (see implementation-status.md).

---

## Scripts

`scripts/` is currently flat; a grouping into subdirectories is planned (see implementation-status.md).

Main clusters (by filename prefix / purpose):

- **4D / ideal polytopes** — `ideal_4polytope_volume.py`, `ideal_4simplex_volume.py`, `optimize_ideal_4polytopes.py`, `optimize_ideal_by_normals.py`, `sweep_volume_distribution_4d.py`, `optimize_surface_area_4d.py`.
- **Facet-parameterized optimization** — `optimize_by_facets.py`, `optimize_facet_full.py`, `optimize_fixed_facets.py`, `quick_facet_scan.py`, `save_facet_results.py`, `ideal_from_normals.py`, `optimize_projected_volume.py`.
- **Klein / S^3 / annealing** — `optimize_klein_volume.py`, `anneal_S3_volume.py`, `facet_distribution_S3.py`.
- **Euclidean deficit** — `euclidean_deficit.py`, `euclidean_deficit_large_N.py`, `euclidean_deficit_4d.py`, `euclidean_deficit_4d_large.py`.
- **Inscribed-polytope optimization** — `optimal_inscribed_polytopes.py`, `optimal_inscribed_large.py`.
- **Distribution sweeps** — `sweep_vertex_distribution.py`, `run_60vertex_distribution.py`, `fix_sweep_outliers.py`.
- **Cross-validation** — `cross_validate_volume.py`, `verify_lob_derivative.py`.
- **LLM benchmark** — `llm_polytope_benchmark.py`, `generate_random_challenges.py`, `generate_missing_challenges.py`, `generate_arithmeticity_benchmark.py`, `generate_hd_volume_lb_benchmark.py` (H^d max-volume LB challenge generator), `check_hd_volume_lb_response.py` (grader for same), `recompute_h5_volumes.py` (recomputes H^5 volumes from stored d=5 configs), `export_face_lattice.py`, `standalone_checker.py`, `verify_llm_polytope.py`, `build_standalone_benchmark.sh`.
- **Inscribability checks** (currently at repo root, to be moved) — `check_inscribable_12.py`, `check_inscribable_range.py`, `check_inscribable_range_4connected.py`.
- **Migration / validation** — `validate_migration.py`.
- **Debug / dev** — `debug_gradient_sign.py`, `debug_original.py`, `test_from_bad_init.py`, `test_gradients.py`, `test_lbfgs_issue.py` (not real pytest tests).

---

## Examples

`examples/` contains both loose top-level files and subdirectories (`distributions/`, `enumeration/`, `optimization/`, `analysis/`, `visualization/`, `llm_benchmark/`). See `examples/README.md`.

## Plans

- `docs/plans/rl-harness-design.md` — design for the modularization of the
  RL optimization harness (Stage B). Read before touching any of
  `rl_generic.py`, `rl_optimizer.py`, `rl_cages.py`.

## Tests

`tests/` holds the pytest-style suite. At present it contains `llm_benchmark_test.py` and `test_volume_threshold.py` (moved from repo root).

## Paper sources

`paper/` holds LaTeX sources for:
- `maximal_ideal_polyhedra.tex` — main paper.
- `deficit_constants.tex` — Euclidean volume deficit note.
- `llm_volume_note.tex` — LLM benchmark note.
- `hyperbolic_5_6_simplex_note.tex` — compact 5/6 simplex volume note.
- Prior snapshots in `paper/archive/` (if present).
