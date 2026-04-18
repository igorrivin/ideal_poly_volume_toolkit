# Ideal 4-Polytope Volume Band Challenge: Solution Guide

## Problem Setup

An **ideal polytope** in hyperbolic 4-space H^4 is the convex hull of
N points on the boundary at infinity S^3.  Despite having vertices "at
infinity," such polytopes have finite hyperbolic volume — a consequence
of the exponential shrinking of hyperbolic space toward the boundary.

The challenge asks the model to construct N points on S^3 ⊂ R^4
(unit norm) whose convex hull, viewed as an ideal polytope in H^4,
has a specified geometric measure (volume, surface area, or
isoperimetric ratio) within a narrow tolerance window.

## Geometric Measures

### Hyperbolic 4-volume

The volume is computed by:
1. **Stereographic projection**: project the N points from S^3 to R^3
   using stereographic projection from one vertex (the "pole").
2. **Delaunay triangulation**: triangulate the projected points in R^3
   to obtain a decomposition into tetrahedra.
3. **Ideal 4-simplex volumes**: each tetrahedron in R^3 corresponds to
   an ideal 4-simplex in H^4. Its volume is computed via the
   Haagerup-Munkholm formula, which expresses the volume in terms of
   dihedral angles and the Lobachevsky function:

       V(simplex) = (1/2) * sum over edges e of
           theta_e * L(pi/2 - theta_e)
           + (terms involving the Schläfli function)

   where L(x) = -integral_0^x ln|2 sin t| dt is the Lobachevsky function.

4. **Total volume**: sum the volumes of all ideal 4-simplices.

### Surface area (3-volume of boundary)

The boundary of the ideal 4-polytope consists of tetrahedral 3-faces
(facets). Each facet is an ideal tetrahedron in H^3. Its hyperbolic
3-volume is computed via the classical formula:

    V_3(tet) = sum over corners c of L(alpha_c)

where alpha_c are the dihedral angles and L is the Lobachevsky function.
The total surface area is the sum over all boundary facets.

### Isoperimetric ratio

    ratio = volume / surface_area

This measures the "efficiency" of the polytope — how much 4-volume
it encloses relative to its boundary 3-volume.

## How the Targets Were Chosen

The target values are **not** the global maximum volume. Instead,
they are drawn from the **volume distribution** of random
configurations:

1. For each vertex count N, we sample many random configurations
   of N points on S^3 (uniformly distributed).
2. We compute the measure (volume, surface area, or ratio) for each
   sample.
3. We select targets at various percentiles of the distribution
   (including near-optimal, median, and below-median values).
4. The tolerance epsilon is set to approximately 2% of the target,
   giving a narrow but achievable band.

The key difficulty is that **the model cannot simply maximize volume**
— it must hit a specific band, which may require finding configurations
at non-extremal points of the measure landscape.

## Solution Strategy

### Hitting the volume band

A natural approach:

1. **Start with a random configuration** on S^3.
2. **Optimize toward the target** using gradient-based methods:
   - If the current volume is below the target, optimize to increase it.
   - If above, perturb to decrease it.
3. **Fine-tune within the band** by small perturbations.

Alternatively, since the volume distribution for random points is
well-characterized (approximately Beta-distributed), one can:

1. Sample many random configurations.
2. Compute volumes for all of them.
3. Select one that falls within the band.
4. If none hits exactly, use the closest as a starting point and
   optimize.

For near-optimal targets, the model needs to perform volume
maximization (concave optimization on S^3 via L-BFGS-B or similar).
For median targets, random sampling may suffice.

### Parameterization

Points on S^3 can be parameterized by 3 hyperspherical angles each:

    x = sin(a) * sin(b) * cos(c)
    y = sin(a) * sin(b) * sin(c)
    z = sin(a) * cos(b)
    w = cos(a)

with a, b in [0, pi] and c in [0, 2*pi]. This gives 3N parameters
to optimize. After factoring out the O(4) symmetry group (6 dimensions),
the effective DOF is 3N - 6.

## Verification

The checker (`check_response.py`) verifies:

1. **Correct vertex count**: N points provided.
2. **Unit norm**: all points lie on S^3 (|x| = 1 within tolerance).
3. **Full-dimensionality**: convex hull is 4-dimensional.
4. **Measure in band**: the computed measure lies in
   [target - epsilon, target + epsilon].

The checker is fully standalone (numpy + scipy only) and computes
volumes from first principles using the Klein model and Lobachevsky
function.

## References

- Haagerup, U. and Munkholm, H.J., "Simplices of maximal volume in
  hyperbolic n-space," Acta Math. 147 (1981), 1-11.
- Rivin, I., "Euclidean structures on simplicial surfaces and
  hyperbolic volume," Ann. of Math. 139 (1994), 553-580.
- Rivin, I., "Combinatorial optimization in geometry,"
  Adv. Appl. Math. 31 (2003), 242-271. arXiv:math/0200122.
