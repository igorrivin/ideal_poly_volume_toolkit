# Benchmark & Challenge Index

## Data status and provenance

The following source datasets are canonical, required inputs. They are stored
as ordinary Git text so a reviewer can inspect them without Git LFS:

| Path | Size (bytes) | SHA-256 |
|---|---:|---|
| `optimal_d4_symmetry_analysis.json` | 126922 | `0bea03b4ddd6a91897ee9e9c3cdc4c5362d1a658a9a9de96911345edca2e29b5` |
| `optimal_d5_d6_symmetry_analysis.json` | 101617 | `d50056f9dbc2d5ad2bb8bec1d0a0543700b0dcc92d150fdbbfbdccc61ae7e702` |

`results/data/` contains both canonical inputs and derived benchmark products.
The timestamped files under `bin/results/data/` are historical optimization
outputs; current CLI and GUI writers use `results/data/`. Loose corrected and
uncorrected historical artifacts remain in place pending an approved ownership
migration.

GitHub `origin/main` is the authoritative code and data history. The Hugging
Face remote is a deployment branch that may intentionally lag and must not be
treated as a repository backup. Run `python3 scripts/check_data_integrity.py`
to verify LFS objects, JSON payloads, attributes, the schema-v1
`results/manifest.json` inventory, and canonical hashes; CI additionally
requires every LFS payload to be materialized. The manifest contract is
published at `docs/schemas/result-manifest-v1.schema.json`.

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

**Status: superseded / invalid as ground truth.** The checked-in 50- and
100-vertex files predate coordinate-derived shear validation, treat a finite
Delaunay disk as the full spherical boundary, and do not pass the current
certificate. They are retained only for provenance and must not support
publication claims.

The replacement producer is
`scripts/generate_arithmeticity_benchmark.py`, which writes schema
`ideal-poly-arithmetic-corpus/1`. It uses total-vertex semantics, includes the
explicit infinity vertex and its complete fan, records exact integer
zero-shear holonomy separately from floating coordinate residuals, validates
closed-sphere topology, deduplicates combinatorial types, and checkpoints
atomically. Mutable runs live under `results/data/arithmetic_corpora/` and are
not canonical until a completed file has passed schema/semantic validation and
is deliberately added to `results/manifest.json` with an immutable hash.

**Historical task**: Construct pairwise non-isometric convex ideal polyhedra
in H^3 whose surface is given by an arithmetic Fuchsian group.

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

### 6. Volume Lower Bound — Construct a Record Polytope (`llm_challenges_volume_lb/`)

**Task**: Given a (measure, d, N) triple, provide the sharpest lower bound
you can on the maximum volume of an N-vertex polytope, backed by explicit
vertex coordinates.  The grader computes the volume of the submitted
polytope and cross-checks against both the LLM's claim and our record.

- Five regimes: Euclidean at d in {4, 5, 6} (inscribed in unit ball);
  Hyperbolic at d in {4, 6} (ideal polytopes).  Hyperbolic d=5 is
  excluded because our d=5 records at N in {12, 14, 20} are known
  suboptimal (see Stage A).
- 77 challenges total.
- `benchmark.json`, `ground_truth.json`, `README.md` — same layout as
  benchmark #5 above.
- Anti-hallucination design: `vertices` is required.  Missing → `invalid`.
- Generator: `scripts/generate_volume_lb_benchmark.py`.
- Checker: `scripts/check_volume_lb_response.py`.  Grades each answer
  on (a) claim-vs-computed consistency, (b) computed-vs-record strength
  (≥1.00 = strong, ≥0.95 = pass).

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
