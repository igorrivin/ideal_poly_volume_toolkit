# Results Directory

## Data Files

### Optimization Records

- **`surface_area_optimization_4d.json`** / `.txt` — Ideal 4-polytope optimization for N=5,...,30 vertices. Three objectives per N: max surface area, max Vol/Area ratio, max volume. Summary statistics only (no vertex coordinates).

- **`benchmark_q1_20facets_maxvol.json`** — Max volume ideal 4-polytope with exactly 20 facets. Certificate: 9 vertices, volume = 2.3139. **Includes vertex coordinates.**

- **`benchmark_q2_30vert_maxarea.json`** — Max boundary 3-measure for 30-vertex ideal 4-polytope. Certificate: area = 119.590. **Includes vertex coordinates.**

- **`benchmark_q3_20vert_maxratio.json`** — Max isoperimetric ratio V(P)/A(P) for 20-vertex ideal 4-polytope. Certificate: ratio = 0.16856. **Includes vertex coordinates.**

- **`anneal_N30_euclidean.json`** / `anneal_N30_euclidean_tuned.json` — Simulated annealing optimizer results (N=30, Euclidean volume). Annealing did not improve on L-BFGS-B for this problem.

### Combinatorial Invariance Data

- **`euclidean_vs_hyperbolic_combinatorics.json`** — Volume-only comparison of Euclidean vs hyperbolic optimal combinatorial types for N=5,...,30. **Includes vertex coordinates for both geometries.**

- **`combinatorial_all_metrics.json`** — Full comparison across three metrics (volume, surface area, ratio) in both Euclidean and hyperbolic settings, for N=5,...,30. **Includes vertex coordinates for all 60 configurations.**

- **`combinatorial_d3_comparison.json`** — Same comparison in d=3 (points on S^2), for N=4,...,50. **Includes vertex coordinates.**

### Benchmark Test Suites

- **`benchmarks/`** — 33 hyperbolic benchmark test cases (windowed + facet-constrained). Each file is a self-contained JSON for the standalone checker.
  - `test_XXX_N{n}_{measure}.json` — Windowed tests (volume, surface_area, ratio)
  - `test_XXX_N{n}_{measure}_f{f}.json` — Facet-count constrained tests

- **`benchmarks_euclidean/`** — 33 Euclidean benchmark test cases, same structure.
  - Measures: `euclidean_volume`, `euclidean_area`, `euclidean_ratio`

- **`windowed_benchmarks.json`** — All 33 hyperbolic benchmarks in one file.

- **`windowed_benchmarks_euclidean.json`** — All 33 Euclidean benchmarks in one file.

### Other Data

- **`optimal_configurations.json`** — Earlier optimization results (various N).
- **`facet_optimization_f*.json`** — Facet-based optimization results (various facet counts). **Note:** The f=20 result from the two-level normal-based optimizer is degenerate; use `benchmark_q1_20facets_maxvol.json` (vertex-based) instead.

## Benchmark Format

Each benchmark test case is a JSON file with:

```json
{
  "measure": "volume",
  "target": 11.4591,
  "epsilon": 0.2292,
  "n_vertices": 20,
  "n_facets": 72,
  "vertices": [[...], ...]
}
```

The model's task: produce `n_vertices` points on S^3 such that the specified measure falls within [target - epsilon, target + epsilon]. If `n_facets` is specified, the convex hull must have exactly that many facets.

Supported measures: `volume`, `surface_area`, `ratio` (hyperbolic); `euclidean_volume`, `euclidean_area`, `euclidean_ratio`.

## Standalone Checker

```bash
python scripts/standalone_checker.py answer.json
```

Exit code 0 = PASS, 1 = FAIL.

## Key Results

### Best Known Lower Bounds (N=30 on S^3)

| Metric | Euclidean | Hyperbolic |
|--------|-----------|------------|
| Volume | **2.4909** | **22.262** |
| Surface area | 12.434 | **119.590** |
| V/A ratio | 0.199 | **0.187** |

### Combinatorial Invariance (Conjecture)

The f-vector of the volume-maximizing polytope is the same whether volume is measured Euclidean or hyperbolic:
- d=3: 11/11 match (N=4,...,50)
- d=4: 10/10 match (N=5,...,30)

V_E and V_H are correlated at r > 0.995 across the optimization landscape.

Different metrics (volume vs surface area) can give different optimal combinatorial types (divergence at N=6, 16, 30 in d=4).
