# Lessons Learned

## Environment (read first in any new session)

- **`pip` and `python3` point to different environments on this machine.** `python3` is `/home/igor/miniforge3/bin/python3`; bare `pip` is `/home/igor/.local/bin/pip`, tied to a *different* Python. Always install with `python3 -m pip install ...`. A bare `pip install` will report success but the package stays invisible to `python3`, giving a confusing `ModuleNotFoundError`. This rule also applies to `pip show`, `pip list`, `pip uninstall`.

## Optimization

- **Cartesian parameterization beats angles.** Use 4N Cartesian coordinates with norm projection (x / |x|) rather than 3N spherical angles. The angle parameterization has coordinate singularities at the poles; Cartesian is smooth everywhere. Confirmed: Cartesian L-BFGS-B reaches 12.4417 for N=30 area, while angles only get 12.4339.

- **Powell beats L-BFGS-B for facet-constrained problems.** When the objective has a discontinuous penalty (wrong facet count), derivative-free Powell handles the transitions better. For 17-facet optimization: Powell 83% valid hit rate vs L-BFGS-B 36%. Powell also finds better optima from random starts (5.035 vs 4.995).

- **Multi-start L-BFGS-B beats annealing** for unconstrained problems. The volume is smooth within each combinatorial stratum, so gradient methods dominate locally. 500 restarts with three-phase L-BFGS-B (progressively tighter tolerances) is currently the best approach for fixed-N optimization.

- **Simulated annealing** with an all-on-hull constraint has ~99.6% rejection rate for N=30 in 4D. The constraint is too restrictive for random tangent-space perturbations. If revisiting, relax the constraint or use basin-hopping.

- **Two-level facet optimization** (outer: normals, inner: solve for support numbers) has a degeneracy failure mode where all vertices collapse to one point. The vertex-based approach (optimize vertices, filter by hull facet count) is more robust.

- **GPT's code contributions:** (a) Cartesian + normalize parameterization, (b) Gram determinant for per-facet 3-volume, (c) `count_actual_facets` via equation rounding to merge coplanar simplices, (d) Powell for facet-constrained optimization. All four are improvements over our original approach.

- **Hybrid Euclidean→Hyperbolic warm start** is faster and better than pure hyperbolic for moderate restarts (200): 22.041 vs 22.006, in 38% less time. But at 500 restarts with tight polish, pure angle hyperbolic (22.262) still beats hybrid (22.172). The Euclidean warm start misses some purely-hyperbolic basins.

- **Cycling E→H→E→H does NOT help at N=30.** Only 9/200 restarts improved, mean gain 0.005. The Euclidean re-optimization pushes off the hyperbolic optimum; the second hyperbolic phase just climbs back. The ridge-walking hypothesis works for N=9 (composite, product structure) but not N=30 (landscape too rugged).

- **Best strategy for production:** Run BOTH pure angle hyperbolic AND hybrid E→H in parallel, take the best of each. The two approaches find different basins.

## Geometry

- **In H^3, boundary area is topological**: each ideal k-gon has H^2 area (k-2)π. Surface area optimization is genuinely geometric only in d ≥ 4, where boundary facets are ideal tetrahedra with continuously varying H^3-volumes.

- **The inradius of ideal polytopes is finite** despite vertices being at infinity. The Chebyshev center is in the interior of the Klein ball, not at the boundary. The earlier inradius code had a bug where the optimizer drifted to |p| → 1 because the penalty function was too weak.

- **Klein model distance formula**: d(q, Π) = arcsinh(|a·q + b| / (sqrt(1-|q|²) sqrt(|a|² - b²))) for facet a·x + b = 0. Equivalently, use the polar point p* = -a/b and Eq. 16 from arXiv:1404.1583.

## Numerical

- **ConvexHull facet equations** from scipy use the convention a·x + b ≤ 0 inside, with a not necessarily unit normal. Always normalize when computing distances.

- **GPT's 9-decimal-place coordinates** are a sign that the values were generated in a forward pass, not by iterative optimization (which would give 15+ digits). Useful forensic indicator.

- **Gemini's numerical claims** should be treated with suspicion. It claimed "numerical optimization confirms" 12.4417 but when asked for code, took an hour and produced code that only reaches 12.423. The 12.4417 number was likely sourced from training data, not computation. Gemini's algebraic derivations (e.g., exact formula for free sum area) are often correct; its numerical claims are often fabricated.

## Free Sum Polytopes

- **Free sum C_p ⊕ C_q** of two regular polygons in orthogonal planes gives a simplicial 4-polytope with pq facets and p+q vertices, all on S^3. Requires both p,q >= 3.

- **Balanced products dominate.** C_4 ⊕ C_5 (vol 0.793) far exceeds C_3 ⊕ C_7 (vol 0.592) despite having fewer vertices. At fixed facet count, the most balanced factorization wins.

- **For unbalanced products, the optimizer beats the free sum** — up to 52% better (e.g., f=33: C_11⊕C_3 = 0.644, optimizer = 0.976).

- **The free sum simultaneously maximizes both Euclidean volume and surface area** for the N=9 / 20-facet case. This dual optimality fails for non-product configurations.

- **The hyperbolic volume maximum is achieved on a manifold** (not an isolated point) for 20 facets, with near-zero Hessian eigenvalues. The free sum and the hyperbolic optimizer's solution are different points on this manifold with identical hyperbolic volume but 10%+ different Euclidean volume. For prime facet counts (17), the maximum is isolated (no near-zero eigenvalues).
