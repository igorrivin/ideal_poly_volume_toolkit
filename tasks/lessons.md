# Lessons Learned

## Optimization

- **Multi-start L-BFGS-B beats annealing** for this problem. The volume is smooth within each combinatorial stratum, so gradient methods dominate locally. 500 restarts with three-phase L-BFGS-B (progressively tighter tolerances) is currently the best approach.

- **Simulated annealing** with an all-on-hull constraint has ~99.6% rejection rate for N=30 in 4D. The constraint is too restrictive for random tangent-space perturbations. If revisiting, relax the constraint or use basin-hopping.

- **Two-level facet optimization** (outer: normals, inner: solve for support numbers) has a degeneracy failure mode where all vertices collapse to one point. The inner solver (least_squares with |v|^2 = 1 constraints) can find trivial solutions. The vertex-based approach (optimize vertices, filter by hull facet count) is more robust.

## Geometry

- **In H^3, boundary area is topological**: each ideal k-gon has H^2 area (k-2)π. Surface area optimization is genuinely geometric only in d ≥ 4, where boundary facets are ideal tetrahedra with continuously varying H^3-volumes.

- **The inradius of ideal polytopes is finite** despite vertices being at infinity. The Chebyshev center is in the interior of the Klein ball, not at the boundary. The earlier inradius code had a bug where the optimizer drifted to |p| → 1 because the penalty function was too weak.

- **Klein model distance formula**: d(q, Π) = arcsinh(|a·q + b| / (sqrt(1-|q|²) sqrt(|a|² - b²))) for facet a·x + b = 0. Equivalently, use the polar point p* = -a/b and Eq. 16 from arXiv:1404.1583.

## Numerical

- **ConvexHull facet equations** from scipy use the convention a·x + b ≤ 0 inside, with a not necessarily unit normal. Always normalize when computing distances.

- **GPT's 9-decimal-place coordinates** are a sign that the values were generated in a forward pass, not by iterative optimization (which would give 15+ digits). Useful forensic indicator.
