# Benchmark & Challenge Index

## LLM Benchmarks

### 1. Inscribed 4-Polytope Reconstruction (`llm_challenges/`, `llm_challenges_random/`)

**Task**: Given the face lattice (f-vector, edges, facets) of a simplicial
4-polytope, produce N points on S^3 whose convex hull realizes that
combinatorial type.

- `llm_challenges/` — **Optimal family**: face lattices from volume-maximizing
  inscribed polytopes (N=5..30). High symmetry, algebraic structure.
  14 challenges with prompts, face lattice JSONs, and ground truth.
- `llm_challenges_random/` — **Random family**: face lattices from random S^3
  points (N=8..20, 3 instances each). Generically trivial automorphism
  group. 18 challenges.
- Checker: `scripts/llm_polytope_benchmark.py` (verify_response)
- Realization tools: `ideal_poly_volume_toolkit/inscribed_realization.py`
  (spectral method), `ideal_poly_volume_toolkit/cone_realization.py`
  (cone/Volkov method)

### 2. Delaunay Realizability (`llm_challenges_delaunay/`)

**Task**: Given a planar triangulation, determine if it is realizable as a
Delaunay triangulation. If yes, produce point coordinates. If no, say None.

- `care_package.zip` — self-contained benchmark (27K):
  11 challenges (6 realizable, 5 non-realizable), standalone checker
  with built-in Rivin LP + pynauty isomorphism.
- Realizable instances: random Delaunay at N=100, 200
- Non-realizable instances: edge-flip constructions (N=100),
  face stellations (N=283, 587)
- Checker: `care_package/check_response.py`

### 3. Arithmetic Ideal Polyhedra (`llm_challenges_arithmetic/`)

**Task**: Construct pairwise non-isometric convex ideal polyhedra in H^3
whose surface is given by an arithmetic Fuchsian group.

- `care_package.zip` — self-contained benchmark (14K):
  prompts at N=50 (25 configs) and N=100 (10 configs), standalone
  checker with Penner-Rivin holonomy, solution guide with Rivin
  theorem references.
- Checker verifies: triangulation validity, volume (Lobachevsky),
  arithmeticity (holonomy trace integrality), pairwise non-isometry
  (cross-ratio invariants).
- `solution_guide.md` — full theorem statement (not given to model).

### 4. Ideal 4-Polytope Volume Band (`llm_challenges_4d_volume/`)

**Task**: Construct N points on S^3 whose convex hull has a specified
geometric measure (volume, surface area, or isoperimetric ratio) within
a narrow tolerance window.

- `care_package.zip` — self-contained benchmark (39K):
  3 separate challenge files by measure type, standalone checker.
  - `benchmark_volume.json`: 17 challenges (N=10,20,30)
  - `benchmark_surface_area.json`: 11 challenges (N=10,20,30)
  - `benchmark_ratio.json`: 5 challenges (N=20)
- Targets drawn from volume distribution at various percentiles,
  not just global maxima.
- `solution_guide.md` — Haagerup-Munkholm formula, parameterization,
  optimization strategy.

### 5. Ideal H^d Max-Volume Lower Bound (`llm_challenges_hd_volume_lb/`)

**Task**: Construct the largest-volume convex ideal polytope in H^d with
exactly N vertices, for each given (d, N). The LLM's claimed volume is
graded against our best optimized lower bound LB(d, N).

- `benchmark.json` — prompt + list of (d, N) challenges (no LBs).
- `ground_truth.json` — LBs; withheld from the LLM.
- `README.md` — full LB table and grading rules.
- Covers d = 4 (N in {5..30 subset}), d = 5 (N in {6..20}), and
  d = 6 (N in {7..24 subset}); 35 challenges in total.
- Generator: `scripts/generate_hd_volume_lb_benchmark.py`.
- Checker: `scripts/check_hd_volume_lb_response.py` — pass if claimed V >=
  0.95 * LB; "strong" if V >= LB (a genuine improvement on our bound).

## Optimal Polytope Data

- `optimal_ideal_4polytopes.json` — volume-optimal inscribed 4-polytopes
  (N=5..30): coordinates, f-vectors, volumes, automorphism orders.
- `optimal_d5_d6_symmetry_analysis.json` — vertex coordinates and symmetry
  groups for d=5 (N=6..20) and d=6 (N=7..24) inscribed-optimal polytopes.
  Note: for d=5 at N in {12, 14, 20} the stored configs are local, not
  global, maxima — they underperform paper Table 5 by 1–4%.
- `ideal_H5_polytope_results.json` — H^5 hyperbolic volumes recomputed
  from the configs in `optimal_d5_d6_symmetry_analysis.json` via
  `scripts/recompute_h5_volumes.py`. Includes per-N `best_lb` (whichever
  is tighter: our recompute or paper Table 5) and `best_lb_source` so the
  discrepancy is explicit.
- `ideal_H6_{polytope_first_results,larger_N,combinatorial_invariance}.json`
  — H^6 results, each with `hyp_vol`.

## LLM Benchmark Results

- `llm_benchmark/` — GPT-5.4 Pro results on face lattice reconstruction.
