
hyperbolic_model_transforms.py
hyperbolic_4_simplex_volume_models.py
======================================

These two files are meant to be kept in the same directory.

What they do
------------
1. hyperbolic_model_transforms.py
   General point/vertex transformations between the standard real hyperbolic models:
     - Klein ball
     - Poincare ball
     - Upper half-space
     - Minkowski / hyperboloid model

   The transformation code is dimension-agnostic.

2. hyperbolic_4_simplex_volume_models.py
   Quadrature-free volume of a hyperbolic 4-simplex from vertices in any of those models.
   It supports:
     - compact simplices (all vertices finite), and
     - finite-volume generalized simplices with some ideal vertices.

   Hyperideal vertices are not supported.

Model conventions
-----------------
Minkowski / hyperboloid:
    <X,Y> = -X_0 Y_0 + X_1 Y_1 + ... + X_n Y_n

Finite Minkowski points are returned on the upper sheet:
    <X,X> = -1,  X_0 > 0

Ideal Minkowski points are returned as future-pointing lightlike vectors.

Upper half-space:
    points are lists [x_1, ..., x_{n-1}, t] with t > 0
    ideal boundary points have t = 0
    the point at infinity is represented by the sentinel string

        HALFSPACE_INFINITY = "infinity"

The Cayley transform convention is the standard one sending the north pole on the ball boundary
to HALFSPACE_INFINITY.

Most useful functions
---------------------
From hyperbolic_model_transforms.py:

    convert_point(point, from_model, to_model, dps=50, snap_ideal_tol=None, ambient_dim=None)
    convert_vertices(vertices, from_model, to_model, dps=50, snap_ideal_tol=None)

    to_klein(...)
    to_poincare(...)
    to_halfspace(...)
    to_minkowski(...)

    halfspace_infinity(dim, to_model="halfspace", ...)
    halfspace_infinity_minkowski(dim)

From hyperbolic_4_simplex_volume_models.py:

    hyperbolic_4_simplex_volume(vertices, dps=50, snap_ideal_tol=None)
        # vertices are already in Klein coordinates

    hyperbolic_4_simplex_volume_from_model(vertices, model, dps=50, snap_ideal_tol=None)

    hyperbolic_4_simplex_data(...)
    hyperbolic_4_simplex_data_from_model(...)

    regular_ideal_simplex_vertices(dps=50)

Quick examples
--------------
1. Convert a point from Poincare to half-space:

    from ideal_poly_volume_toolkit.hyperbolic_model_transforms import convert_point

    p = [0.1, -0.2, 0.3, 0.05]
    h = convert_point(p, "poincare", "halfspace", dps=80)

2. Convert a whole simplex from half-space to Klein:

    from ideal_poly_volume_toolkit.hyperbolic_model_transforms import convert_vertices, HALFSPACE_INFINITY

    verts_half = [
        HALFSPACE_INFINITY,
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]

    verts_klein = convert_vertices(verts_half, "halfspace", "klein", dps=80)

3. Compute a 4-simplex volume directly from half-space vertices:

    import mpmath as mp
    from ideal_poly_volume_toolkit.hyperbolic_4_simplex_volume_models import hyperbolic_4_simplex_volume_from_model
    from ideal_poly_volume_toolkit.hyperbolic_model_transforms import HALFSPACE_INFINITY

    verts_half = [
        HALFSPACE_INFINITY,
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]

    V = hyperbolic_4_simplex_volume_from_model(verts_half, model="halfspace", dps=80)
    print(mp.nstr(V, 30))

4. Convert half-space infinity by itself:

    from ideal_poly_volume_toolkit.hyperbolic_model_transforms import halfspace_infinity

    north_pole = halfspace_infinity(4, to_model="klein")
    # returns [0, 0, 0, 1]

    # or, equivalently:
    # convert_point("infinity", "halfspace", "klein", ambient_dim=4)

Notes
-----
- snap_ideal_tol can be used to snap near-ideal input numerically onto the ideal boundary.
- The transformation code is general in dimension n, but the volume code is specific to 4-simplices:
  exactly five vertices in H^4.
- The no-quadrature volume formula reduces the 4-volume to dihedral angles plus spherical
  tetrahedron link volumes, evaluated with log/dilog special functions.
