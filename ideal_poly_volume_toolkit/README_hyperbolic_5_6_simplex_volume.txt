hyperbolic_5_6_simplex_volume.py
================================

This module adds practical H^5 and H^6 simplex volume routines on top of the earlier H^4 code.

Files / dependencies
--------------------
- hyperbolic_5_6_simplex_volume.py
- hyperbolic_model_transforms.py  (only needed when model != "klein")
- numpy, scipy, mpmath

What is implemented
-------------------
1. H^5:
   - direct 5D Klein-model QMC volume integral;
   - a Schlaefli-path integrator
         d Vol_5 = -(1/4) sum Vol_3(F^3) d theta_F
     along x_i(t) = t x_i.
     The 3-face volumes are computed intrinsically by extracting 3D Klein coordinates for
     each face from the ambient hyperboloid Gram data, then evaluating the 3D Klein integral.

2. H^6:
   - direct 6D Klein-model QMC volume integral;
   - the even-dimensional Hopf/Poincare reduction
         Vol_6 = (8 pi^3 / 15) * (17/4 - w_4 + (1/2) w_2 - w_0),
     with
         w_4 = (1 / 2 pi)   * sum of the 21 dihedral angles,
         w_2 = (1 / 2 pi^2) * sum of the 35 spherical tetrahedral 2-face links,
         w_0 = (1 / pi^3)   * sum of the 7 spherical 5-simplex vertex links.
     The S^3 links are evaluated by Murakami's explicit log/dilog formula.
     The S^5 links are evaluated numerically by gnomonic coordinates + a 5D QMC integral.

3. Model wrappers:
   All public H^5 / H^6 entry points accept model="klein", "poincare", "halfspace",
   or "minkowski". Internally the vertices are converted to the Klein model first.

Public entry points
-------------------
- hyperbolic_5_simplex_volume(vertices, model="klein", method="schlafli", ...)
- hyperbolic_5_simplex_data(vertices, model="klein", ...)
- hyperbolic_6_simplex_volume(vertices, model="klein", method="reduction", ...)
- hyperbolic_6_simplex_data(vertices, model="klein", ...)
- hyperbolic_simplex_volume_direct_klein(vertices, ...)
- intrinsic_klein_coordinates(vertices)
- intrinsic_hyperbolic_simplex_volume(vertices, ...)
- angle_gram_from_klein(vertices)
- spherical_tetrahedron_volume_murakami(A4, ...)
- spherical_simplex_volume_qmc_from_angle_gram(A, ...)

Compactness assumption
----------------------
This file assumes all simplex vertices are finite, i.e. strictly inside the Klein ball.
It does not currently implement ideal or ultra-ideal H^5 / H^6 simplices.

Typical H^5 example
-------------------
import numpy as np
from hyperbolic_5_6_simplex_volume import hyperbolic_5_simplex_data

verts5 = np.vstack([np.zeros((1, 5)), 0.5 * np.eye(5)])
res5 = hyperbolic_5_simplex_data(verts5)
print("Schlaefli H^5 volume:", res5.volume_schlafli)
print("direct QMC cross-check:", res5.volume_direct)

Typical H^6 example
-------------------
import numpy as np
from hyperbolic_5_6_simplex_volume import hyperbolic_6_simplex_data

verts6 = np.vstack([np.zeros((1, 6)), 0.7 * np.eye(6)])
res6 = hyperbolic_6_simplex_data(verts6)
print("reduction H^6 volume:", res6.volume_reduction)
print("direct QMC cross-check:", res6.volume_direct)
print("w0, w2, w4:", res6.w0, res6.w2, res6.w4)

Example with model conversion
-----------------------------
import numpy as np
from hyperbolic_model_transforms import convert_vertices
from hyperbolic_5_6_simplex_volume import hyperbolic_5_simplex_volume

verts5_klein = np.vstack([np.zeros((1, 5)), 0.3 * np.eye(5)]).tolist()
verts5_poincare = convert_vertices(verts5_klein, from_model="klein", to_model="poincare")
V = hyperbolic_5_simplex_volume(verts5_poincare, model="poincare", method="direct")
print(V)

Recommended defaults and caveats
--------------------------------
- H^5: the Schlaefli method is the main one; the direct 5D Klein integral is a useful cross-check.
- H^6: the reduction method is the main one, but near the Euclidean / very-small-volume regime it
  suffers from cancellation, because it subtracts large angle-sum terms. In that regime the direct
  6D Klein integral is the better diagnostic.
- The default H^6 vertex_link_samples is intentionally fairly large, because the 5D spherical link
  integrals dominate the error budget.
- All Sobol sample counts are rounded up to the next power of two.

Quick smoke-test expectations
-----------------------------
For
    verts5 = [0, 0.5 e_1, ..., 0.5 e_5]
and
    verts6 = [0, 0.7 e_1, ..., 0.7 e_6],
the Schlaefli/reduction answers and the direct QMC cross-checks should agree to within a few
parts in 10^3 with the default settings.
