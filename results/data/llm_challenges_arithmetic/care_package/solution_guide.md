# Arithmetic Ideal Polyhedra: Background for LLM Benchmark

## The Theorem

**Theorem (Rivin, 1994).** Let $\mathcal{T}$ be a triangulation of the 2-sphere
realizable as the boundary of a convex ideal polyhedron in $\mathbb{H}^3$.
The hyperbolic volume functional $V$ on the space of all ideal polyhedra
with combinatorial type $\mathcal{T}$ is a strictly concave function on the
Rivin polytope (the convex body of valid angle assignments). Its unique
maximum is a critical point where all dihedral angles are rational
multiples of $\pi$.

**Reference:** I. Rivin, "Euclidean structures on simplicial surfaces and
hyperbolic volume," *Ann. of Math.* **139** (1994), 553-580.
Also: I. Rivin, "Combinatorial optimization in geometry," *Adv. in Appl. Math.*
**31** (2003), 242-271. arXiv version: math/0200122.

## Why This Implies Arithmeticity

The surface of an ideal polyhedron in $\mathbb{H}^3$ inherits a complete
hyperbolic metric with cusps at the ideal vertices, making it a finite-area
hyperbolic surface $S = \mathbb{H}^2 / \Gamma$ for a Fuchsian group
$\Gamma < \mathrm{PSL}(2, \mathbb{R})$.

The shape parameters of the ideal triangulation are $z_e = \exp(i\theta_e)$
where $\theta_e$ is the sum of the two angles at edge $e$ from the two
adjacent triangles. When all dihedral angles are rational multiples of
$\pi$, these shape parameters are **roots of unity**. The holonomy
matrices $\rho(\gamma) \in \mathrm{PSL}(2, \mathbb{C})$ are products of
matrices whose entries involve these roots of unity, so all traces
$\mathrm{tr}(\rho(\gamma))$ lie in a cyclotomic field $\mathbb{Q}(\zeta_q)$.

A Fuchsian group is **arithmetic** if it is commensurable with the group
of norm-1 units in an order of a quaternion algebra over a totally real
number field. The trace field being a cyclotomic field is a strong
necessary condition. Our computational evidence confirms that the
holonomy traces are not merely cyclotomic but in fact algebraic integers,
consistent with arithmeticity.

## Practical Construction Method

To construct an arithmetic ideal polyhedron with $N$ vertices:
1. Choose $N$ points in $\mathbb{C} \cup \{\infty\}$ (the boundary of
   the upper half-space model of $\mathbb{H}^3$).
2. Compute the Delaunay triangulation of these points on $S^2$ (via
   stereographic projection).
3. Verify that the triangulation satisfies Rivin's LP conditions
   (positive angles, triangle sums = $\pi$, vertex sums = $2\pi$,
   dihedral sums $\leq 2\pi$).
4. Optimize the vertex positions to maximize hyperbolic volume.
   At the maximum, all dihedral angles are rational multiples of $\pi$,
   and the holonomy representation is arithmetic.

The volume is computed as:
$$V = \sum_{\text{triangles } T} \bigl(
    \Lambda(\alpha_T) + \Lambda(\beta_T) + \Lambda(\gamma_T)
\bigr)$$
where $\Lambda(\theta) = -\int_0^\theta \ln |2 \sin t| \, dt$ is the
Lobachevsky function and $\alpha_T, \beta_T, \gamma_T$ are the angles
of triangle $T$.
