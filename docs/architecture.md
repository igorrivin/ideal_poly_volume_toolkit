# Architecture

## Purpose

A research toolkit for computing, optimizing, and analyzing volumes of ideal hyperbolic polyhedra and polytopes. The primary target dimension is H^3; secondary targets are H^4, H^5, and compact 5- and 6-simplices in H^d.

## Core computational pipelines

### 1. H^3 volume via Delaunay triangulation

`vertices on S² (stereographic plane) → Delaunay triangulation in C → ideal-tetrahedron volume sum`

Entry: `geometry.ideal_poly_volume_via_delaunay`.
The Lobachevsky / Bloch–Wigner dilogarithm evaluation is done either fast (PyTorch autodiff, `li2_torch`, `murakami_torch`) or exactly (mpmath, `geometry.lob_exact`).

### 2. H^3 volume via convex hull, projected back

`vertices on S² → hull on S² → ideal-tetrahedron volume sum per face`

Entry: `geometry.ideal_poly_volume_via_hull_project_back`. More robust than (1) for degenerate point sets but still fails for highly symmetric configurations (see `tasks/todo.md` on the 10–10 duoprism).

### 3. Penner–Rivin holonomy

`ideal triangulation → edge invariants → Fuchsian-group generators in PSL(2,C) → invariant trace field`

Entry: `rivin_holonomy.generators_from_triangulation`, `pointset_to_fuchsian.group_from_pointset`. Used for arithmeticity detection.

### 4. Rivin LP and geometric realization

`planar graph + angles from LP → geometric realization of an ideal polyhedron`

Entries: `rivin_delaunay` (Delaunay realizability oracle), `geometric_realization`, `cone_realization` (Volkov cone method for simplicial 4-polytopes), `inscribed_realization`.

### 5. H^4 ideal-polytope volume via Schläfli

`facet normals → ideal 4-polytope → 4-simplex decomposition → Schläfli-formula integration`

Entries: `geometry_4d`, `simplex_volume_4d`, `hyperbolic_4_simplex_volume*`, plus `scripts/` wrappers.

### 6. RL-guided optimization

A generic hierarchical optimizer (`rl_generic`) instantiated for:
- ideal-polytope volume (`rl_optimizer`),
- high-girth cubic graphs / the cage problem (`rl_cages`, with `fast_logdet` for incremental spectral updates).

## Data flow conventions

- **Vertices in the plane**: `numpy.complex128` or float pairs, stereographic projection from the north pole.
- **Vertices on S²/S³**: unit-norm float arrays; for ideal points in H^{d+1} we use the sphere-at-infinity representation (norm exactly 1).
- **Klein-model points**: unit-ball interior; vertices on the boundary represent ideal points.
- **Triangulations**: list of tuples of vertex indices; orientation conventions documented per function.
- **Volumes**: scalar floats (fast path) or mpmath `mpf` (exact path). Variable names and docstrings must distinguish exact from approximate outputs.

## External dependencies

- **Required**: numpy, scipy, mpmath, torch, matplotlib.
- **Optional**: pynauty (symmetry groups), plantri binary (planar triangulation enumeration), gradio (GUI).
- A separate `requirements_HPC.txt` pins a CUDA-ready environment used on cluster jobs.

## Repository organization principles (from CLAUDE.md)

- Every feature has **one canonical home**.
- `shared/` and `utils/` are reserved for genuinely cross-cutting code; feature logic does not live there.
- `scripts/` is for operational / driver code, **not** product logic.
- `docs/` is authoritative for architecture, module index, and status; it is updated in the same change that introduces new structure.

See `docs/implementation-status.md` for the current deviation from these principles and the active migration plan.
