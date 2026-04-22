# Volume Lower Bound Benchmark

94 challenges across the five (measure, dimension) regimes listed in the prompt.

## Task given to the LLM

```
Task: for each challenge below, construct an explicit convex polytope
with exactly N vertices and report your volume as a lower bound.

The challenges come in two flavours, distinguished by the `measure` field:

* measure = "euclidean": a convex polytope inscribed in the closed unit
  ball of R^d, i.e. all N vertices lie on or inside the unit sphere S^{d-1}.
  The reported volume is the Euclidean d-volume of the convex hull.

* measure = "hyperbolic": an ideal convex polytope in hyperbolic d-space
  H^d, i.e. all N vertices lie exactly on the sphere at infinity S^{d-1},
  interpreted as the boundary of the Klein-ball model.  The reported
  volume is the hyperbolic d-volume.

For each challenge you must provide:

  1. `LB`: your claimed lower bound (the volume your polytope achieves).
  2. `vertices`: a list of N explicit vertex coordinates in R^d.  This is
     mandatory — we compute the volume of your submitted polytope and
     cross-check it against your claim, which eliminates the option of
     guessing a number.

Return your answer as a JSON list of objects:

    [
      {"id": <int>, "measure": "euclidean"|"hyperbolic", "d": <int>,
       "N": <int>, "LB": <float>, "vertices": [[...], ...]},
      ...
    ]

Grading per (measure, d, N):

* The grader computes the volume V_comp of your submitted polytope using
  the measure specified.  We compare V_comp to the LB you reported:
  agreement within |V_comp - LB| / max(LB, 1e-12) <= 1e-3 is required,
  otherwise the challenge is marked "inconsistent" and scored on V_comp.
* We then compare V_comp to our record LB (withheld):
      V_comp >= 1.00 * record       -> "strong" (matches/beats our record)
      V_comp >= 0.95 * record       -> "pass"
      V_comp >  0                   -> "fail" (valid polytope, below 95%)
      V_comp <= 0 or invalid input  -> "invalid"
* For hyperbolic challenges, vertices must lie on the unit sphere within
  a tolerance of 1e-6 (after row normalization we check |\|v_i\| - 1| < 1e-6).
  Submissions with vertices off the sphere are normalised defensively,
  but the hyperbolic volume of a clearly non-ideal configuration may be
  small or zero.
```

## Ground truth (withheld from the LLM)

### Euclidean volume in dimension 4

| N | our record | source |
|---|------------|--------|
| 5 | 0.1455773423 | `optimal_d4_symmetry_analysis.json` |
| 6 | 0.2812500000 | `optimal_d4_symmetry_analysis.json` |
| 7 | 0.4330127019 | `optimal_d4_symmetry_analysis.json` |
| 8 | 0.6666666667 | `optimal_d4_symmetry_analysis.json` |
| 9 | 0.7925470969 | `optimal_d4_symmetry_analysis.json` |
| 10 | 0.9078580521 | `optimal_d4_symmetry_analysis.json` |
| 11 | 1.0465289390 | `optimal_d4_symmetry_analysis.json` |
| 12 | 1.1713818649 | `optimal_d4_symmetry_analysis.json` |
| 13 | 1.3035576346 | `optimal_d4_symmetry_analysis.json` |
| 14 | 1.4034382967 | `optimal_d4_symmetry_analysis.json` |
| 15 | 1.5121766980 | `optimal_d4_symmetry_analysis.json` |
| 16 | 1.5979592868 | `optimal_d4_symmetry_analysis.json` |
| 17 | 1.6749751396 | `optimal_d4_symmetry_analysis.json` |
| 18 | 1.7603187215 | `optimal_d4_symmetry_analysis.json` |
| 19 | 1.8610900767 | `optimal_d4_symmetry_analysis.json` |
| 20 | 1.9439499146 | `optimal_d4_symmetry_analysis.json` |
| 21 | 1.9898189645 | `optimal_d4_symmetry_analysis.json` |
| 22 | 2.0636549910 | `optimal_d4_symmetry_analysis.json` |
| 23 | 2.1280951324 | `optimal_d4_symmetry_analysis.json` |
| 24 | 2.1789441449 | `optimal_d4_symmetry_analysis.json` |

### Euclidean volume in dimension 5

| N | our record | source |
|---|------------|--------|
| 6 | 0.0321993789 | `optimal_d5_d6_symmetry_analysis.json` |
| 7 | 0.0666666667 | `optimal_d5_d6_symmetry_analysis.json` |
| 8 | 0.1125000000 | `optimal_d5_d6_symmetry_analysis.json` |
| 9 | 0.1732050808 | `optimal_d5_d6_symmetry_analysis.json` |
| 10 | 0.2666666667 | `optimal_d5_d6_symmetry_analysis.json` |
| 11 | 0.3170188388 | `optimal_d5_d6_symmetry_analysis.json` |
| 12 | 0.3631432208 | `optimal_d5_d6_symmetry_analysis.json` |
| 13 | 0.4186115756 | `optimal_d5_d6_symmetry_analysis.json` |
| 14 | 0.4733622043 | `optimal_d5_d6_symmetry_analysis.json` |
| 15 | 0.5258234744 | `optimal_d5_d6_symmetry_analysis.json` |
| 16 | 0.5878803649 | `optimal_d5_d6_symmetry_analysis.json` |
| 17 | 0.6354331443 | `optimal_d5_d6_symmetry_analysis.json` |
| 18 | 0.6874027788 | `optimal_d5_d6_symmetry_analysis.json` |
| 19 | 0.7507399468 | `optimal_d5_d6_symmetry_analysis.json` |
| 20 | 0.8017328061 | `optimal_d5_d6_symmetry_analysis.json` |

### Euclidean volume in dimension 6

| N | our record | source |
|---|------------|--------|
| 7 | 0.0058352154 | `optimal_d5_d6_symmetry_analysis.json` |
| 8 | 0.0131687243 | `optimal_d5_d6_symmetry_analysis.json` |
| 9 | 0.0243569645 | `optimal_d5_d6_symmetry_analysis.json` |
| 10 | 0.0375000000 | `optimal_d5_d6_symmetry_analysis.json` |
| 11 | 0.0577350269 | `optimal_d5_d6_symmetry_analysis.json` |
| 12 | 0.0888888889 | `optimal_d5_d6_symmetry_analysis.json` |
| 13 | 0.1056729463 | `optimal_d5_d6_symmetry_analysis.json` |
| 14 | 0.1256261802 | `optimal_d5_d6_symmetry_analysis.json` |
| 15 | 0.1493469966 | `optimal_d5_d6_symmetry_analysis.json` |
| 16 | 0.1606639133 | `optimal_d5_d6_symmetry_analysis.json` |
| 17 | 0.1815613358 | `optimal_d5_d6_symmetry_analysis.json` |
| 18 | 0.2020383233 | `optimal_d5_d6_symmetry_analysis.json` |
| 19 | 0.2239431540 | `optimal_d5_d6_symmetry_analysis.json` |
| 20 | 0.2422369613 | `optimal_d5_d6_symmetry_analysis.json` |
| 21 | 0.2616135942 | `optimal_d5_d6_symmetry_analysis.json` |
| 22 | 0.2821318760 | `optimal_d5_d6_symmetry_analysis.json` |
| 23 | 0.3053167710 | `optimal_d5_d6_symmetry_analysis.json` |
| 24 | 0.3288340815 | `optimal_d5_d6_symmetry_analysis.json` |

### Hyperbolic volume in dimension 3

| N | our record | source |
|---|------------|--------|
| 4 | 1.0149416064 | `4vertex_comprehensive_combinatorics.json` |
| 5 | 2.0298832129 | `5vertex_comprehensive_combinatorics.json` |
| 6 | 3.6638623767 | `6vertex_comprehensive_combinatorics.json` |
| 7 | 4.9867734573 | `7vertex_comprehensive_combinatorics.json` |
| 8 | 6.4884689844 | `8vertex_comprehensive_combinatorics.json` |
| 9 | 8.1625383486 | `9vertex_comprehensive_combinatorics.json` |
| 10 | 9.8393154814 | `10vertex_comprehensive_combinatorics.json` |
| 11 | 11.4492900000 | `paper/maximal_ideal_polyhedra.tex (Table 1)` |
| 12 | 13.5296280000 | `paper/maximal_ideal_polyhedra.tex (Table 1)` |
| 20 | 27.6990000000 | `paper/maximal_ideal_polyhedra.tex (Table 1)` |
| 30 | 46.4860000000 | `paper/maximal_ideal_polyhedra.tex (Table 1)` |
| 50 | 84.9710000000 | `paper/maximal_ideal_polyhedra.tex (Table 1)` |
| 70 | 123.8930000000 | `paper/maximal_ideal_polyhedra.tex (Table 1)` |
| 100 | 182.6710000000 | `paper/maximal_ideal_polyhedra.tex (Table 1)` |
| 150 | 279.7267340000 | `h3_large_N.txt` |
| 200 | 377.7294740000 | `h3_large_N.txt` |
| 300 | 573.4290090000 | `h3_large_N.txt` |

### Hyperbolic volume in dimension 4

| N | our record | source |
|---|------------|--------|
| 5 | 0.2688956602 | `optimal_ideal_4polytopes.json` |
| 6 | 0.5878301174 | `optimal_ideal_4polytopes.json` |
| 7 | 1.0236011023 | `optimal_ideal_4polytopes.json` |
| 8 | 1.7714283944 | `optimal_ideal_4polytopes.json` |
| 9 | 2.3138512999 | `optimal_ideal_4polytopes.json` |
| 10 | 3.0091040559 | `optimal_ideal_4polytopes.json` |
| 11 | 3.6223864336 | `optimal_ideal_4polytopes.json` |
| 12 | 4.3771950375 | `optimal_ideal_4polytopes.json` |
| 13 | 5.2487112765 | `optimal_ideal_4polytopes.json` |
| 14 | 6.0493076009 | `optimal_ideal_4polytopes.json` |
| 15 | 6.9970498568 | `optimal_ideal_4polytopes.json` |
| 16 | 7.7986198204 | `optimal_ideal_4polytopes.json` |
| 20 | 11.9357644202 | `optimal_ideal_4polytopes.json` |
| 30 | 22.0036730380 | `optimal_ideal_4polytopes.json` |

### Hyperbolic volume in dimension 6

| N | our record | source |
|---|------------|--------|
| 7 | 0.0102313751 | `ideal_H6_combinatorial_invariance.json` |
| 8 | 0.0250435609 | `ideal_H6_combinatorial_invariance.json` |
| 9 | 0.0502208973 | `ideal_H6_combinatorial_invariance.json` |
| 10 | 0.0839025193 | `ideal_H6_combinatorial_invariance.json` |
| 12 | 0.2328373523 | `ideal_H6_combinatorial_invariance.json` |
| 14 | 0.3692201182 | `ideal_H6_combinatorial_invariance.json` |
| 16 | 0.5405983427 | `ideal_H6_combinatorial_invariance.json` |
| 18 | 0.7417885209 | `ideal_H6_larger_N.json` |
| 20 | 0.9607007050 | `ideal_H6_larger_N.json` |
| 24 | 1.4970305468 | `ideal_H6_larger_N.json` |

## Files

- `benchmark.json` — what is given to the LLM (prompt + (measure, d, N) list, no records).
- `ground_truth.json` — records + sources, withheld from the LLM.
- This README.

## Grading

See `scripts/check_volume_lb_response.py` for the grader.  Summary:

- The LLM must provide `vertices`; missing → `invalid`.
- The grader computes the actual volume from `vertices` and compares to both the LLM's `LB` claim and to our record.
- Consistency tolerance: |claim − computed| / max(record, 1e-12) ≤ 1e-3.
- Strength: ≥ 1.00 × record = strong, ≥ 0.95 × record = pass.
