# Architecture

## Purpose

A research toolkit for computing, optimizing, and analyzing volumes of ideal hyperbolic polyhedra and polytopes. The primary target dimension is H^3; secondary targets are H^4, H^5, and compact 5- and 6-simplices in H^d.

## Core computational pipelines

### 1. H^3 geometry kernel

`vertices on S² or finite coordinates in C → pole projection → Delaunay triangulation → ideal-tetrahedron volume sum`

Canonical module: `geometry`. Its established entries are
`ideal_poly_volume_via_delaunay` for complex-plane coordinates and
`ideal_poly_volume_via_hull_project_back` for a hull-based decomposition.
`ideal_poly_volume_from_sphere` is the sphere-coordinate entry, with
`stereographic_project_from_pole` exposing the corresponding projection and
index map.

The ordinary double-precision path uses `geometry.bloch_wigner`, while the
arbitrary-precision alternative uses `geometry.lob_exact`. Torch-only H³
implementations live in `geometry_torch`; `geometry.lob_fast` and the other
historical Torch entries are lazy compatibility forwarders, so importing the
core H³ geometry or optimizer does not import Torch. `li2_torch` and
`murakami_torch` support the higher-dimensional differentiable pipelines
described below.

### 2. H^3 optimization

`optimization parameters → normalized ideal vertices → geometry kernel → scored trial → best result`

Canonical module: `optimization`. It owns the reusable H^3 parameterization,
objective evaluation, differential-evolution orchestration, and result
accounting. `optimization_cli` owns the installed `ideal-poly-optimize`
adapter, and `bin/optimize_polyhedron.py` is its source-tree compatibility
shim. The optimization tab in `bin/gui.py` is a separate UI adapter. Adapters
must not maintain separate numerical kernels or reinterpret the optimizer's
objective sign. The service's public functions are `spherical_to_complex`,
`parameters_to_complex`,
`negative_volume_objective`, `complete_boundary_triangulation`,
`analyze_ideal_polyhedron_structure`, and `optimize_ideal_polyhedron`; result
records use `OptimizationTrial` and `PolyhedronOptimizationResult`, and both
adapters serialize them through the versioned `optimization_result_to_dict`
schema.

### 3. H^3 volume via convex hull, projected back

`vertices on S² → hull on S² → ideal-tetrahedron volume sum per face`

Entry: `geometry.ideal_poly_volume_via_hull_project_back`. It is an alternate
H^3 decomposition for point sets where the planar Delaunay path is
inconvenient. The separate H^4 non-generic-cell limitation, including the
10–10 duoprism, is tracked in `tasks/todo.md`.

### 4. Penner–Rivin holonomy

`ideal triangulation → edge invariants → Fuchsian-group generators in PSL(2,C) → invariant trace field`

Entry: `rivin_holonomy.generators_from_triangulation`,
`rivin_holonomy.certify_exact_zero_shear_structure`, and
`pointset_to_fuchsian.group_from_pointset`. Arithmetic claims use the exact
zero-shear PSL(2,Z) certificate; coordinate shears and selected-generator
trace proximity are numerical diagnostics.

### 5. Strict zero-shear realization audit

`coherently oriented labeled sphere + normalized CP1 coordinates → independent geometric predicates → numerical classification`

The owning module is `rivin_zero_shear`. Its public
`audit_zero_shear_realization` entry checks normalization, chordal
distinctness, topology, signed supporting planes, coordinate shears, bending
signs, spherical embeddedness, coherent facet orientation, and exact labeled
convex-hull facet-set agreement. The planar straight-edge drawing is also
reported, but it is a
chart-dependent diagnostic rather than the spherical-embedding gate.

These predicates must remain separate. In particular, zero shear is an
equality and does not imply the Delaunay/convex bending inequalities, preserve
embeddedness, or force the requested hull facets. A Tutte drawing is only an
initialization strategy and carries none of those properties through an
unconstrained nonlinear solve.

The same module owns the reusable coordinate-solver layer.
`weighted_pole_link_tutte_embedding` constructs a harmonic start after deleting
the distinguished pole, and `normalize_cp1_points` provides the shared
ordered-face/handedness normalization. `solve_zero_shear_from_initial`
accepts any labeled CP1 start and supports both the historical finite
cube-root gauge and a standard upper-half-plane gauge with free `(x, log(y))`
variables; it returns the same `TutteSolveResult` diagnostics as
`solve_zero_shear_from_tutte`. Optional variable scaling and
Levenberg--Marquardt polishing are solver controls, not substitutes for the
independent strict audit.

`scripts/check_zero_shear_response.py` owns only the strict private-key/model
JSON protocol and delegates geometry to the package audit. The optional
private-reference chordal error is diagnostic unless its tolerance is
explicitly enabled as an additional gate. `scripts/audit_tutte_zero_shear_basins.py`
is the operational reproducibility search for classifying solver endpoints;
its output contract is
`docs/schemas/tutte-zero-shear-audit-v1.schema.json`.
`scripts/build_zero_shear_holdouts.py` owns deterministic relabeling and the
separation of paste-ready public prompts from private provenance, hardness
evidence, and checker-compatible answer keys. It reuses the package audit and
must not acquire independent geometric acceptance logic.

`scripts/compare_rivin_zero_shear_methods.py` is the operational
fixed-combinatorics comparison. For one labeled sphere and one ordered face it
compares (A) strict-interior Rivin angle optimization and realization after
deleting the distinguished pole star, (B) the pole-link harmonic start followed
by the standard-upper-half-plane zero-shear solve, and (C) the same coordinate
solve initialized by A's realized endpoint. Every route is reattached to the
same full sphere, normalized to the same positive-handed convention, and
independently audited. The hybrid reports both refinement-only cost and its
upstream angle-route dependency; stage-specific iterations are not treated as
interchangeable work units. Its output contract is
`docs/schemas/rivin-zero-shear-comparison-v1.schema.json`. The historical
outer-face Tutte route is available only as an optional secondary comparison.

All of these checks are double-precision numerical evidence. A formal
existence proof would additionally require a validated root enclosure (for
example interval Newton or Krawczyk) and interval-certified strict support,
embedding, and bending inequalities.

### 6. Rivin LP and geometric realization

`planar graph + angles from LP → geometric realization of an ideal polyhedron`

Entries: `rivin_delaunay` (Delaunay realizability oracle), `geometric_realization`, `cone_realization` (Volkov cone method for simplicial 4-polytopes), `inscribed_realization`.

### 7. H^4 ideal-polytope volume via Schläfli

`facet normals → ideal 4-polytope → 4-simplex decomposition → Schläfli-formula integration`

Entries: `geometry_4d`, the optional differentiable/vectorized
`geometry_4d_torch`, `simplex_volume_4d`,
`hyperbolic_4_simplex_volume*`, plus `scripts/` wrappers. The Torch path takes
a tetrahedralization as a detached, per-evaluation combinatorial input; it
does not differentiate through Qhull.

### 8. H^4 symmetric-local-maximum benchmark

`unconstrained S³ vertices → H⁴ volume optimization → strict hull and multi-pole audit → full hyperbolic-symmetry audit → optional quotient local-max audit`

The reusable candidate and optimization layer is owned by
`rl.polytope.h4_symmetric_benchmark`. It reuses the canonical H⁴ task and
offers explicit strict-budget Powell and fresh-topology AD selectors rather
than maintaining a second canonical volume objective. The AD route owns its
fixed-pole charts and branch accounting in `rl.polytope.optimizers`, rebuilds
the Delaunay tetrahedralization at every gradient closure, and compares
branches only after separate canonical endpoint evaluations. Candidate audits
keep pole-by-pole volumes, triangulation counts, strict simplicial margins,
and symmetry results separate. The optional local maximum diagnostic works on
the tangent space modulo all ten infinitesimal `PO(4,1)` directions (six
rotations and four boosts); it must not mistake the coordinate gauge for
physical Hessian nullity.

The dimension-generic geometric-symmetry predicate is owned by `symmetry`.
Unlike the legacy H³ graph helper, it starts from the colored vertex--facet
incidence graph, enumerates its candidate automorphisms up to an explicit cap,
and retains only permutations satisfying the conformal Gram/ray-scaling
equations for the ideal null rays. With a spanning lift these equations are
equivalent to a future-preserving projective Lorentz isometry; direct Lorentz
matrix fits remain diagnostic because strongly boosted charts can make them
ill-conditioned. Orientation comes from the exact action on the coherently
oriented facet sphere. A group order or a 1-skeleton automorphism count alone
is not a geometric-symmetry certificate.

`scripts/mine_h4_symmetric_maxima.py` is the operational, checkpointed
multi-start campaign. Its contract is
`docs/schemas/h4-symmetric-maxima-campaign-v1.schema.json`. It includes an
explicit ideal-16-cell calibration, retains private coordinates only for
fully audited nontrivial candidates by default, and treats generated local
maxima as sources for later volume-threshold prompts rather than as proofs of
global optimality. Version 1 deliberately declines to certify
non-simplicial/wall configurations, where the current H⁴ Delaunay volume path
can depend on the projection pole.

### 8.1 H^4 prescribed-combinatorics benchmark

`public tetrahedral facets + submitted S³ vertices → strict supporting hull → full incidence isomorphism → all-pole Delaunay/facet agreement → volume floor`

The reusable numerical and combinatorial predicate is owned by
`rl.polytope.h4_combinatorics_benchmark`; the strict-JSON command adapter is
`scripts/check_h4_combinatorics_response.py`. A target is an unordered list of
tetrahedral facets, and a response may use any vertex-row ordering. Acceptance
requires one bijection carrying the complete target vertex--facet incidence
structure to the submitted strict hull. A 1-skeleton or f-vector match is not
enough in dimension four. The checker separately requires unit, distinct,
full-dimensional extreme points, strict simplicial supports, agreement between
every pole's Delaunay tetrahedra and the corresponding nonincident hull facets,
and a stable minimum hyperbolic-volume bound across all poles.

The grading path is reference-free: private discovery coordinates, optimizer
history, symmetry, and local-stationarity diagnostics are not acceptance
conditions. Its result remains a double-precision numerical certificate rather
than an interval proof.

### 9. RL-guided optimization

The canonical hierarchical optimizer lives in `ideal_poly_volume_toolkit.rl`:

- `rl.generic` defines the problem-agnostic interfaces, explicit
  `OptimizerRun` work accounting, and search environment. The environment
  charges objective, surrogate, and gradient work separately and keeps
  surrogate values out of canonical node statistics. Optimizers may attach
  the canonical value of their returned solution so it is not scored twice;
- `rl.inner_optimizers` and `rl.training` own reusable optimizers and
  controller lifecycle helpers;
- `rl.polytope` specializes the harness for ideal-polytope volume;
- `rl.cage` specializes it for high-girth cubic graphs. The separate
  `fast_logdet` helper provides validated transactional spectral updates and
  rejects disconnected graphs, but is not yet wired into the cage optimizers.

The package-root `rl_generic`, `rl_optimizer`, and `rl_cages` modules are
backward-compatibility shims, not feature owners. See
`ideal_poly_volume_toolkit/rl/README.md` for the module map.

## Data flow conventions

- **Vertices in the plane**: `numpy.complex128` or float pairs, obtained by
  stereographic projection from a selected sphere pole (the legacy helpers use
  the north pole).
- **Vertices on S²/S³**: unit-norm float arrays; for ideal points in H^{d+1} we use the sphere-at-infinity representation (norm exactly 1).
- **Klein-model points**: unit-ball interior; vertices on the boundary represent ideal points.
- **Triangulations**: list of tuples of vertex indices; orientation conventions documented per function.
- **Volumes**: scalar floats (fast path) or mpmath `mpf` (exact path). Variable names and docstrings must distinguish exact from approximate outputs.

## External dependencies

- The base `pyproject.toml` install is the numerical core: NumPy, SciPy, and
  mpmath. Optional feature groups are `torch`, `plot`, `gui`, `graph`,
  `render`, and `rl`; using a feature-specific API requires its corresponding
  extra. Missing optional dependencies must fail at feature use with an
  actionable installation message, not while importing unrelated core code.
- The `dev` extra is the complete test/lint/GUI environment used by CI. It
  includes Torch, Gymnasium, and `cma` because the fast numerical and RL
  contract tests exercise them.
- `requirements.txt` is the complete RL/research convenience environment,
  including Stable-Baselines3, rendering, and utility dependencies.
- The GUI Docker image installs the `gui` and `graph` extras; it does not pull
  the separate RL/research environment.
- `plantri` is an external executable used only by planar-triangulation
  enumeration.
- `requirements_HPC.txt` is a machine-oriented environment snapshot, not the
  canonical package installation path.

## Repository organization principles (from CLAUDE.md)

- Every feature has **one canonical home**.
- `shared/` and `utils/` are reserved for genuinely cross-cutting code; feature logic does not live there.
- `scripts/` is for operational / driver code, **not** product logic. Its
  first ownership-oriented groups are `inscribability/`, `tests/`, and
  `_debug/`; the last contains manual diagnostics rather than pytest tests.
- `docs/` is authoritative for architecture, module index, and status; it is updated in the same change that introduces new structure.

See `docs/implementation-status.md` for the current deviation from these principles and the active migration plan.
