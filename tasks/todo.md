# TODO

## High Priority

### Inradius computation (Chebyshev center)
- Fix the hyperbolic inradius computation for ideal polytopes in H^4
- Use the polar-point formulation from Rivin (arXiv:1404.1583, Section 7, Eq. 16):
  d(q, p*) = arcsinh((1 - p·q) / (sqrt(p·p - 1) sqrt(1 - q·q)))
  where p* is the polar of a hyperplane in the Klein model
- For convex hull facets with scipy equation a·x + b <= 0, the polar is p* = -a/b
- Previous attempt had a bug in the Chebyshev center optimization (optimizer drifted to boundary of Klein ball); the formula itself is correct
- Goal: compute inradius for optimal ideal polytopes at each N, compare with O(log N) upper bound from Theorem 6.2/6.3, and investigate the isoperimetric limit question (does the inscribed ball capture the volume?)

### Max volume with prescribed facet count (Q1 benchmark)
- **Current status**: simplicial answer (N=9, vol=2.31) is far too small. The question is about non-simplicial ideal polytopes which can have many more vertices.
- **GPT's construction**: double the ideal rectified 5-cell twice to get 20 non-simplicial facets with volume 8pi^2/9 ≈ 8.773.
- **Verified so far**: 
  - Rectified 5-cell: 10 vertices on S^3, volume = 2pi^2/9 (exact). Correct.
  - First doubling (sphere inversion across tetrahedral facet): 16 vertices, volume = 4pi^2/9 (exact). Correct.
- **TODO**: Complete the second doubling. Need to:
  1. Identify the 14 non-simplicial cells of the first-doubled polytope (the 54 simplicial hull facets are a triangulation of 14 cells)
  2. Find a cell orthogonal to 6 others
  3. Perform sphere inversion across it
  4. Verify the result has 20 non-simplicial facets and volume 8pi^2/9
- **Key technique**: hyperbolic reflection of ideal points via sphere inversion. For a geodesic hyperplane n·x = d in the Klein model, inversion sphere has center n/d, radius sqrt(1-d^2)/d. Satisfies |center|^2 - r^2 = 1.
- The two-level facet optimization (optimize_facet_full.py) has a separate degeneracy bug and should be considered broken.

### Volume computation for non-generic ideal polytopes
- Our Delaunay-based pipeline (stereographic project from one vertex, Delaunay in R^3, sum ideal 4-simplex volumes) fails for algebraically special configurations like the 10-10 duoprism on the Clifford torus. Volume varies wildly by choice of projection pole (+23 to -16).
- Root cause: the Delaunay triangulation of the stereographic projection doesn't give a valid ideal decomposition when points have special symmetry/degeneracy.
- Gemini's 10-10 duoprism has exactly 20 non-simplicial facets (verified) and 100 ideal vertices on S^3, but we can't compute its hyperbolic volume.
- Need an alternative approach: either (a) cell-by-cell volume computation using the known cell structure, (b) direct integration (Haagerup-Munkholm integral over the Klein model), or (c) a more robust triangulation that works for non-generic points.
- GPT's doubled rectified 5-cell (16 vertices, first doubling) DOES work with our pipeline (vol = 4pi^2/9, exact) because it's still "generic enough" — but the second doubling may also break.

### Non-simplicial facet counting in the checker
- In floating point, ConvexHull always triangulates — there are no non-simplicial facets. A polytope that "should" have 20 non-simplicial facets will appear to have ~54 simplicial ones.
- To check non-simplicial facet counts, need to merge coplanar adjacent simplicial facets: detect groups of hull tetrahedra whose outward normals agree within tolerance (e.g., 1e-8), then count the merged groups.
- Tolerance design is delicate: too tight = nothing merges, too loose = false merges.
- For polytopes from the doubling construction, coplanarity is exact to machine precision, so 1e-8 should work.
- Required for the non-simplicial Q1 benchmark (20 non-simplicial facets, vol ~ 8pi^2/9).
- The simplicial checker works fine as-is for simplicial polytopes.

### Hodge spectral realization for d≥5
- Graph Laplacian (L₀) spectral embedding works for volume-optimal polytopes in d=3,4 (Blind–Mani–Kalai regime) but fails in d≥5.
- The Hodge 1-Laplacian L₁ = B₁B₁ᵀ + B₂B₂ᵀ on edges encodes the full simplicial structure, not just the graph.
- **Naive approach (averaging edge eigenvectors to vertices) fails** — tested, no improvement over L₀.
- **Better approach**: use Hodge decomposition of edge flows to reconstruct the Gram matrix.
  - Assign target edge flows Y[i,j] = ⟨v_i, v_j⟩ on the 1-skeleton.
  - Since H¹(S^{d-1}) = 0 for the polytope boundary, the harmonic component vanishes.
  - The flow is purely gradient: Y = δ⁰(φ) for some vertex potential φ.
  - Invert the gradient to get vertex coordinates (least-squares in gradient subspace).
- Reference: Wei, Wee, Laurent, Xia (2022) "Hodge theory-based biomolecular data analysis" PMC9188576 — uses Hodge decomposition of edge flows for structural analysis. The gradient/curl/harmonic decomposition is exactly what we need.
- Also: Forman (2003) "Bochner's method for cell complexes" for the Bochner–Weitzenböck decomposition L₁ = □₁ + F₁ and Forman–Ricci curvature.
- Also: sheaf Laplacians (preserving fiber structure over vertices) may be a cleaner framework.
- **Goal**: a spectral realization algorithm that works in all dimensions, not just d≤4.

### Inscribability oracle pipeline
- Use the spectral realization tool (d≤4) or cone/Volkov method as an oracle for inscribability.
- Enumerate simplicial 4-polytope combinatorial types (Firsching's data for small N, polymake for larger).
- For each type, attempt inscribed realization. Success = certified inscribable.
- Extract combinatorial features (f-vector, degree sequence, graph spectrum, flag f-numbers).
- Train a neural net classifier to predict inscribability from features.
- If the classifier achieves high accuracy from simple features, that's evidence for a Rivin-type characterization in d=4.
- GPT-5.4's explicit algorithm (spectral init + facet-support LBFGS on S³) could serve as a fast oracle — needs to be stress-tested on non-inscribable types to confirm it correctly fails.

## Medium Priority

### Benchmark test case generation
- Generate more windowed test cases for larger N (40, 50)
- Consider d=3 benchmarks (points on S^2) — simpler geometry, good warm-up problems
- Add "construct a polytope with this f-vector" as a benchmark type (hardest)

### Paper improvements
- Add the V_E/V_H correlation plot as a figure
- Investigate whether the log-log elasticity approaching 2 at N=30 has a theoretical explanation
- Run combinatorial comparison for N=40, 50 in d=4 (expensive but would strengthen the conjecture)
- Add reference to arXiv:1404.1583 in the inradius/isoperimetric discussion

### Annealing optimizer
- Current implementation underperforms L-BFGS-B for this problem
- The all_on_hull rejection rate is too high (~99.6% rejection)
- Ideas: relax hull constraint temporarily, use basin-hopping (scipy.optimize.basinhopping), or hybrid approach (anneal for combinatorial type, L-BFGS-B within each basin)

### GPU/Torch migration (GH200)
- Port Murakami S^3 volume to torch using Nelson dilogarithm (https://github.com/nelsond/dilogarithm.git) for a differentiable complex Li2
- This gives analytical gradients via autograd, eliminating the 2N+1 finite-difference gradient evaluations that currently dominate runtime
- Proposed three-tier pipeline:
  - **Tier 1 (GPU, fp32)**: 10,000 parallel starts on S^{d-1}, periodic CPU Delaunay re-triangulation, 200 torch gradient steps per cycle. Minutes for N=30.
  - **Tier 2 (GPU, fp32)**: top-100 candidates, 500 hyperbolic gradient steps each.
  - **Tier 3 (CPU, fp64)**: L-BFGS-B polish + mpmath verification on top-50.
- GH200 unified memory eliminates CPU↔GPU transfer bottleneck for Delaunay sync
- H100 fp32 at ~60 TFLOPS is massive overkill; no need for fp16
- The torch dilogarithm would be independently useful (publish as standalone package?) — nothing on PyPI currently, applications in differentiable geometry, QFT loop integrals, number theory
- Key torch components needed: complex Li2 (differentiable), angle_gram_from_klein, dihedral_angles, Murakami formula — all smooth and autograd-friendly
- Only non-differentiable step is Delaunay triangulation: freeze combinatorial structure, optimize within stratum, periodically re-triangulate on CPU

## Low Priority

### Code cleanup
- The standalone checker duplicates volume/area code from geometry_4d.py — consider factoring
- Add type hints and docstrings to anneal_S3_volume.py
- Surface area optimization script could save vertex coordinates (currently only saves statistics)

### Additional experiments
- Spherical codes connection: are our volume-optimal configurations also good spherical codes (max min-distance)?
- Thomson problem: compare with energy-minimizing configurations
- d=5 exploration: is the combinatorial invariance still clean in higher dimensions?
