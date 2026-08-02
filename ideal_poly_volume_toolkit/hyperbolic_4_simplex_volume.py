
"""
Quadrature-free volume of a hyperbolic 4-simplex from its Klein-model vertices.

This version handles both:
  - compact simplices (all five vertices strictly inside the Klein ball), and
  - ideal/generalized simplices of finite volume (some or all vertices on the ideal boundary).

The key change from the earlier compact-only routine is that the angle Gram matrix is no longer
extracted from Q^{-1}.  Instead, it is computed from projective facet normals in homogeneous
Minkowski coordinates.  That keeps the algorithm valid even when the Klein matrix

    Q_{ij} = 1 - x_i · x_j

may become singular or ill-conditioned once ideal vertices are allowed.

Main formula in dimension 4:
    Vol_4(Delta)
      = 4*pi^2/3 - (pi/3) * sum_{0<=i<j<=4} theta_{ij}
        + (2/3) * sum_{finite vertices i} Vol_{S^3}(L_i),

where theta_{ij} is the hyperbolic dihedral angle between the facets opposite x_i and x_j, and
L_i is the spherical tetrahedral link at the finite vertex x_i.  Ideal vertices contribute zero
to the link-volume term.

For compact simplices this agrees with the previous Q^{-1}-based implementation; for ideal
simplices it recovers the exact limiting formula without any extrapolation.

Dependencies:
  - mpmath only.

Optional nicety:
  - If your "ideal" coordinates are only approximate floating-point values, pass
        snap_ideal_tol=1e-12
    (or similar) to radially project vertices whose squared norms are within that tolerance of 1
    onto the unit sphere.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Sequence, Tuple

import mpmath as mp


Vector4 = Sequence[float | int | mp.mpf]
VertexList = Sequence[Vector4]


MINKOWSKI_J = mp.matrix([
    [-1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
])


@dataclass
class Hyperbolic4SimplexResult:
    """Container for the main outputs."""

    volume: mp.mpf
    vertex_types: List[str]
    snapped_to_boundary: List[bool]

    projective_matrix: mp.matrix
    facet_normals: List[mp.matrix]
    facet_normal_norms: List[mp.mpf]

    Q: mp.matrix
    Q_inv: Optional[mp.matrix]

    angle_gram: mp.matrix
    dihedral_angles: Dict[Tuple[int, int], mp.mpf]

    link_angle_grams: List[mp.matrix]
    link_geometries: List[str]
    link_volumes: List[mp.mpf]


def _set_precision(dps: int) -> None:
    mp.mp.dps = int(dps)


def _default_tol(multiplier: float | int = 100) -> mp.mpf:
    """
    A generic numerical tolerance tied to the current working precision.
    """
    return mp.mpf(multiplier) * mp.sqrt(mp.eps)


def _to_mpf(x: float | int | mp.mpf) -> mp.mpf:
    return mp.mpf(x)


def _euclid_dot(u: Sequence[mp.mpf], v: Sequence[mp.mpf]) -> mp.mpf:
    return mp.fsum(ui * vi for ui, vi in zip(u, v))


def _euclid_norm2(u: Sequence[mp.mpf]) -> mp.mpf:
    return _euclid_dot(u, u)


def _minkowski_dot(u: mp.matrix, v: mp.matrix) -> mp.mpf:
    return (u.T * MINKOWSKI_J * v)[0]


def _principal_real(x: complex | mp.mpc | mp.mpf, tol: Optional[mp.mpf] = None) -> mp.mpf:
    """
    Return the real part, insisting that the imaginary part is numerically negligible.
    """
    tol = _default_tol() if tol is None else tol
    z = mp.mpc(x)
    if abs(mp.im(z)) > tol:
        raise ValueError(f"expected a real value, got {x!r}")
    return mp.re(z)


def _clip_unit_interval(x: mp.mpf, tol: Optional[mp.mpf] = None) -> mp.mpf:
    """
    Clip tiny roundoff excursions outside [-1, 1]; raise on a genuine violation.
    """
    tol = _default_tol() if tol is None else tol
    if x > 1 and x - 1 <= tol:
        return mp.mpf("1")
    if x < -1 and -1 - x <= tol:
        return mp.mpf("-1")
    if x < -1 or x > 1:
        raise ValueError(f"value {x!r} is outside [-1,1] by more than tolerance")
    return x


def _as_column(entries: Sequence[mp.mpf]) -> mp.matrix:
    return mp.matrix([[entry] for entry in entries])


def _gram_schmidt_euclidean(vectors: Sequence[Sequence[mp.mpf]]) -> List[List[mp.mpf]]:
    """
    Euclidean Gram-Schmidt in R^n, returning an orthonormal list.
    """
    basis: List[List[mp.mpf]] = []
    for vec in vectors:
        w = [mp.mpf(x) for x in vec]
        for u in basis:
            coeff = _euclid_dot(w, u)
            w = [wi - coeff * ui for wi, ui in zip(w, u)]
        n2 = _euclid_norm2(w)
        if n2 <= 0:
            raise ValueError("Gram-Schmidt encountered dependent vectors")
        nrm = mp.sqrt(n2)
        basis.append([wi / nrm for wi in w])
    return basis


def regular_ideal_simplex_vertices(dps: int = 50) -> List[List[mp.mpf]]:
    """
    Return a regular ideal 4-simplex in Klein coordinates.

    The output is a list of five points on the unit 3-sphere in R^4 with pairwise dot product -1/4.
    This is handy as a built-in self-test.
    """
    _set_precision(dps)

    one = mp.mpf("1")
    five = mp.mpf("5")
    centroid = [one / five] * 5

    # Start in R^5 with the centered simplex e_i - (1/5)(1,1,1,1,1).
    centered: List[List[mp.mpf]] = []
    for i in range(5):
        e = [mp.mpf("0")] * 5
        e[i] = one
        centered.append([e[j] - centroid[j] for j in range(5)])

    # Orthonormal basis of the sum-zero hyperplane in R^5.
    raw_basis = []
    for i in range(4):
        vec = [mp.mpf("0")] * 5
        vec[i] = one
        vec[4] = -one
        raw_basis.append(vec)
    basis = _gram_schmidt_euclidean(raw_basis)

    # Normalize the centered vectors to unit length in R^5, then read off their coordinates in the basis.
    scale = mp.sqrt(mp.mpf("5") / mp.mpf("4"))
    verts: List[List[mp.mpf]] = []
    for w in centered:
        w_unit = [scale * comp for comp in w]
        coords = [_euclid_dot(w_unit, b) for b in basis]
        verts.append(coords)
    return verts


def snap_vertices_to_boundary(
    vertices: VertexList,
    tol: float | int | mp.mpf = mp.mpf("1e-12"),
    dps: int = 50,
) -> Tuple[List[List[mp.mpf]], List[bool]]:
    """
    Radially project nearly-ideal vertices to the unit sphere.

    Any vertex with | ||x||^2 - 1 | <= tol is snapped to the ideal boundary.  Other vertices are left
    unchanged.  This is useful when ideal coordinates are supplied only approximately in machine precision.
    """
    _set_precision(dps)
    tol_mpf = mp.mpf(tol)

    out: List[List[mp.mpf]] = []
    snapped: List[bool] = []
    for v in vertices:
        vv = [_to_mpf(c) for c in v]
        if len(vv) != 4:
            raise ValueError("each vertex must have exactly four coordinates")
        n2 = _euclid_norm2(vv)
        if abs(n2 - 1) <= tol_mpf:
            if n2 <= 0:
                raise ValueError("cannot snap the zero vector to the ideal boundary")
            nrm = mp.sqrt(n2)
            vv = [coord / nrm for coord in vv]
            snapped.append(True)
        else:
            snapped.append(False)
        out.append(vv)
    if len(out) != 5:
        raise ValueError("need exactly five vertices in R^4")
    return out, snapped


def _coerce_vertices(
    vertices: VertexList,
    *,
    snap_ideal_tol: Optional[float | int | mp.mpf] = None,
) -> Tuple[List[List[mp.mpf]], List[str], List[bool]]:
    """
    Convert the input to high-precision coordinates, classify each vertex as finite or ideal, and
    optionally snap nearly-ideal vertices to the unit sphere.
    """
    if len(vertices) != 5:
        raise ValueError("need exactly five vertices in R^4")

    boundary_tol = _default_tol(multiplier=1000)
    if snap_ideal_tol is not None:
        boundary_tol = max(boundary_tol, mp.mpf(snap_ideal_tol))

    verts: List[List[mp.mpf]] = []
    vertex_types: List[str] = []
    snapped: List[bool] = []

    for i, v in enumerate(vertices):
        vv = [_to_mpf(c) for c in v]
        if len(vv) != 4:
            raise ValueError("each vertex must have exactly four coordinates")
        n2 = _euclid_norm2(vv)

        did_snap = False
        if abs(n2 - 1) <= boundary_tol:
            if n2 <= 0:
                raise ValueError("cannot snap the zero vector to the ideal boundary")
            nrm = mp.sqrt(n2)
            vv = [coord / nrm for coord in vv]
            n2 = mp.mpf("1")
            did_snap = True

        if n2 < 1:
            vertex_types.append("finite")
        elif n2 == 1:
            vertex_types.append("ideal")
        else:
            raise ValueError(
                f"vertex x_{i} has squared norm {n2}, so it lies outside the Klein ball; "
                "hyperideal vertices are not supported by this routine"
            )

        verts.append(vv)
        snapped.append(did_snap)

    return verts, vertex_types, snapped


def projective_matrix(
    vertices: VertexList,
    dps: int = 50,
    *,
    snap_ideal_tol: Optional[float | int | mp.mpf] = None,
) -> mp.matrix:
    """
    Return the 5x5 homogeneous-coordinate matrix Y whose i-th column is y_i = (1, x_i).
    """
    _set_precision(dps)
    verts, _, _ = _coerce_vertices(vertices, snap_ideal_tol=snap_ideal_tol)

    Y = mp.matrix(5)
    for j, v in enumerate(verts):
        Y[0, j] = mp.mpf("1")
        for k in range(4):
            Y[k + 1, j] = v[k]
    return Y


def klein_Q(
    vertices: VertexList,
    dps: int = 50,
    *,
    snap_ideal_tol: Optional[float | int | mp.mpf] = None,
) -> mp.matrix:
    """
    Return Q_{ij} = 1 - x_i · x_j.  This matrix is allowed to be singular once ideal vertices are allowed.
    """
    _set_precision(dps)
    verts, _, _ = _coerce_vertices(vertices, snap_ideal_tol=snap_ideal_tol)
    return mp.matrix([[1 - _euclid_dot(verts[i], verts[j]) for j in range(5)] for i in range(5)])


def projective_facet_normals_from_vertices(
    vertices: VertexList,
    dps: int = 50,
    *,
    snap_ideal_tol: Optional[float | int | mp.mpf] = None,
) -> Tuple[List[mp.matrix], List[mp.mpf], mp.matrix, List[str], List[bool]]:
    """
    Compute the projective facet normals from the homogeneous-coordinate matrix Y.

    If Y has columns y_0, ..., y_4 with y_i = (1, x_i), then the i-th row of Y^{-1} is the linear
    functional lambda_i satisfying lambda_i(y_j) = delta_{ij}.  The corresponding Minkowski vector

        n_i = J * lambda_i^T

    is orthogonal to every y_j with j != i, so n_i is a normal to the facet opposite x_i.
    """
    _set_precision(dps)
    verts, vertex_types, snapped = _coerce_vertices(vertices, snap_ideal_tol=snap_ideal_tol)

    Y = mp.matrix(5)
    for j, v in enumerate(verts):
        Y[0, j] = mp.mpf("1")
        for k in range(4):
            Y[k + 1, j] = v[k]

    try:
        Y_inv = Y ** -1
    except ZeroDivisionError as exc:
        raise ValueError(
            "the five vertices are not affinely independent in the affine chart t=1"
        ) from exc

    normals: List[mp.matrix] = []
    scales: List[mp.mpf] = []
    for i in range(5):
        covector = _as_column([Y_inv[i, j] for j in range(5)])
        normal = MINKOWSKI_J * covector
        scale = _principal_real(_minkowski_dot(normal, normal))
        if scale <= 0:
            raise ValueError(
                "the facet opposite vertex x_{} does not define a proper hyperbolic hyperplane; "
                "the simplex may be degenerate or not of finite volume".format(i)
            )
        normals.append(normal / mp.sqrt(scale))
        scales.append(scale)

    return normals, scales, Y, vertex_types, snapped


def angle_gram_from_vertices(
    vertices: VertexList,
    dps: int = 50,
    *,
    snap_ideal_tol: Optional[float | int | mp.mpf] = None,
) -> Tuple[mp.matrix, Optional[mp.matrix], mp.matrix]:
    """
    Return (A, Q_inv, Q), where
      Q_{ij} = 1 - x_i · x_j,
      Q_inv is returned when Q is invertible and otherwise None,
      A is the angle Gram matrix with A_{ii} = 1 and A_{ij} = -cos(theta_{ij}).

    Unlike the earlier compact-only routine, A is now computed from projective facet normals, so it
    remains well-defined even when Q is singular or ill-conditioned in ideal/degenerate configurations.
    """
    _set_precision(dps)

    normals, _, _, _, _ = projective_facet_normals_from_vertices(
        vertices, dps=dps, snap_ideal_tol=snap_ideal_tol
    )

    A = mp.matrix(5)
    for i in range(5):
        A[i, i] = mp.mpf("1")
        for j in range(i + 1, 5):
            entry = _principal_real(_minkowski_dot(normals[i], normals[j]))
            A[i, j] = entry
            A[j, i] = entry

    Q = klein_Q(vertices, dps=dps, snap_ideal_tol=snap_ideal_tol)
    try:
        Q_inv: Optional[mp.matrix] = Q ** -1
    except ZeroDivisionError:
        Q_inv = None

    return A, Q_inv, Q


def dihedral_angles_from_angle_gram(A: mp.matrix) -> Dict[Tuple[int, int], mp.mpf]:
    """
    Return the ten dihedral angles theta_{ij} in radians, with keys (i, j), i < j.
    Here A_{ij} = -cos(theta_{ij}).
    """
    if A.rows != 5 or A.cols != 5:
        raise ValueError("A must be a 5x5 angle Gram matrix")

    angles: Dict[Tuple[int, int], mp.mpf] = {}
    for i in range(5):
        for j in range(i + 1, 5):
            c = _clip_unit_interval(_principal_real(-A[i, j]))
            angles[(i, j)] = mp.acos(c)
    return angles


def link_angle_grams(A: mp.matrix) -> List[mp.matrix]:
    """
    Return the five 4x4 principal minors A^{(i)}, one for each vertex link.

    For a finite vertex this is the angle Gram matrix of the spherical tetrahedral link.
    For an ideal vertex it is the degenerate Euclidean-link angle Gram matrix; in that case the
    corresponding link volume is zero and Murakami's spherical formula must not be applied to it.
    """
    if A.rows != 5 or A.cols != 5:
        raise ValueError("A must be a 5x5 angle Gram matrix")

    links: List[mp.matrix] = []
    for i in range(5):
        idx = [j for j in range(5) if j != i]
        links.append(mp.matrix([[A[r, c] for c in idx] for r in idx]))
    return links


def _murakami_alphas_from_B(
    B: mp.matrix,
) -> Tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf, mp.mpf, mp.mpf]:
    """
    Extract the six dihedral angles (alpha_1, ..., alpha_6) from the 4x4 angle Gram matrix

        [[1, -cos a1, -cos a2, -cos a6],
         [-cos a1, 1, -cos a3, -cos a5],
         [-cos a2, -cos a3, 1, -cos a4],
         [-cos a6, -cos a5, -cos a4, 1]].
    """
    entries = [
        -B[0, 1],  # a1
        -B[0, 2],  # a2
        -B[1, 2],  # a3
        -B[2, 3],  # a4
        -B[1, 3],  # a5
        -B[0, 3],  # a6
    ]
    return tuple(mp.acos(_clip_unit_interval(_principal_real(e))) for e in entries)


def spherical_tetrahedron_volume_from_angle_gram(B: mp.matrix) -> mp.mpf:
    """
    Murakami's log/dilog formula for a spherical tetrahedron in terms of its 4x4 angle Gram matrix.

    The returned value is reduced to the physical branch in (0, pi^2).  This function should only be
    applied to genuine spherical links, not to the degenerate Euclidean links of ideal vertices.
    """
    if B.rows != 4 or B.cols != 4:
        raise ValueError("B must be a 4x4 matrix")

    alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = _murakami_alphas_from_B(B)
    alphas = [alpha1, alpha2, alpha3, alpha4, alpha5, alpha6]
    a1, a2, a3, a4, a5, a6 = [mp.e ** (1j * alpha) for alpha in alphas]

    def Li2(z: complex | mp.mpc) -> mp.mpc:
        return mp.polylog(2, z)

    def L(z: complex | mp.mpc) -> mp.mpc:
        cross_term = (
            mp.log(a1) * mp.log(a4)
            + mp.log(a2) * mp.log(a5)
            + mp.log(a3) * mp.log(a6)
        )
        return mp.mpf("0.5") * (
            Li2(z)
            + Li2((a1 * a2 * a4 * a5) ** -1 * z)
            + Li2((a1 * a3 * a4 * a6) ** -1 * z)
            + Li2((a2 * a3 * a5 * a6) ** -1 * z)
            - Li2(-(a1 * a2 * a3) ** -1 * z)
            - Li2(-(a1 * a5 * a6) ** -1 * z)
            - Li2(-(a2 * a4 * a6) ** -1 * z)
            - Li2(-(a3 * a4 * a5) ** -1 * z)
            + cross_term
        )

    q0 = (
        a1 * a4
        + a2 * a5
        + a3 * a6
        + a1 * a2 * a6
        + a1 * a3 * a5
        + a2 * a3 * a4
        + a4 * a5 * a6
        + a1 * a2 * a3 * a4 * a5 * a6
    )
    q1 = (
        -(a1 - 1 / a1) * (a4 - 1 / a4)
        -(a2 - 1 / a2) * (a5 - 1 / a5)
        -(a3 - 1 / a3) * (a6 - 1 / a6)
    )
    q2 = (
        1 / (a1 * a4)
        + 1 / (a2 * a5)
        + 1 / (a3 * a6)
        + 1 / (a1 * a2 * a6)
        + 1 / (a1 * a3 * a5)
        + 1 / (a2 * a3 * a4)
        + 1 / (a4 * a5 * a6)
        + 1 / (a1 * a2 * a3 * a4 * a5 * a6)
    )

    z0 = (-q1 + mp.sqrt(q1 * q1 - 4 * q0 * q2)) / (2 * q2)

    vol = (
        -mp.re(L(z0))
        + mp.pi * (mp.arg(-q2) + mp.fsum(alphas) / 2)
        - mp.mpf("1.5") * mp.pi ** 2
    )

    # Murakami's formula is naturally modulo 2*pi^2.
    two_pi_sq = 2 * mp.pi ** 2
    vol = _principal_real(vol)
    while vol < 0:
        vol += two_pi_sq
    while vol >= two_pi_sq:
        vol -= two_pi_sq
    if vol > mp.pi ** 2:
        vol = two_pi_sq - vol
    return vol


def hyperbolic_4_simplex_volume(
    vertices: VertexList,
    dps: int = 50,
    *,
    snap_ideal_tol: Optional[float | int | mp.mpf] = None,
) -> mp.mpf:
    """
    Return the hyperbolic 4-volume of the simplex with the given Klein vertices.

    Supports compact simplices and finite-volume generalized simplices with ideal vertices.
    """
    return hyperbolic_4_simplex_data(
        vertices, dps=dps, snap_ideal_tol=snap_ideal_tol
    ).volume


def hyperbolic_4_simplex_data(
    vertices: VertexList,
    dps: int = 50,
    *,
    snap_ideal_tol: Optional[float | int | mp.mpf] = None,
) -> Hyperbolic4SimplexResult:
    """
    Compute the full no-quadrature data package:
        - projective facet normals,
        - Q and (when available) Q^{-1},
        - the 5x5 angle Gram matrix,
        - the ten hyperbolic dihedral angles,
        - the five 4x4 link angle Gram matrices,
        - the five link geometries (spherical or euclidean),
        - the five link volumes (zero for ideal vertices),
        - the final hyperbolic 4-volume.

    The geometry split is:
        finite vertex  -> spherical tetrahedral link -> Murakami formula
        ideal vertex   -> Euclidean tetrahedral link -> zero contribution
    """
    _set_precision(dps)

    normals, normal_scales, Y, vertex_types, snapped = projective_facet_normals_from_vertices(
        vertices, dps=dps, snap_ideal_tol=snap_ideal_tol
    )
    A, Q_inv, Q = angle_gram_from_vertices(vertices, dps=dps, snap_ideal_tol=snap_ideal_tol)
    theta = dihedral_angles_from_angle_gram(A)
    links = link_angle_grams(A)

    link_geometries: List[str] = []
    link_vols: List[mp.mpf] = []
    for i in range(5):
        if vertex_types[i] == "ideal":
            link_geometries.append("euclidean")
            link_vols.append(mp.mpf("0"))
        else:
            link_geometries.append("spherical")
            link_vols.append(spherical_tetrahedron_volume_from_angle_gram(links[i]))

    total_theta = mp.fsum(theta[key] for key in sorted(theta))
    volume = (
        (4 * mp.pi ** 2) / 3
        - (mp.pi / 3) * total_theta
        + (mp.mpf("2") / 3) * mp.fsum(link_vols)
    )

    return Hyperbolic4SimplexResult(
        volume=volume,
        vertex_types=vertex_types,
        snapped_to_boundary=snapped,
        projective_matrix=Y,
        facet_normals=normals,
        facet_normal_norms=normal_scales,
        Q=Q,
        Q_inv=Q_inv,
        angle_gram=A,
        dihedral_angles=theta,
        link_angle_grams=links,
        link_geometries=link_geometries,
        link_volumes=link_vols,
    )


def euclidean_simplex_volume(vertices: VertexList, dps: int = 50) -> mp.mpf:
    """
    Euclidean 4-volume of the affine simplex conv(x_0, ..., x_4) in R^4.

    This is valid for finite and ideal Klein coordinates alike, since it is purely an affine calculation.
    """
    _set_precision(dps)
    verts, _, _ = _coerce_vertices(vertices, snap_ideal_tol=None)

    Y = mp.matrix(4)
    for j in range(4):
        for i in range(4):
            Y[i, j] = verts[j + 1][i] - verts[0][i]
    return abs(mp.det(Y)) / mp.factorial(4)


def monte_carlo_klein_integral(
    vertices: VertexList,
    samples: int = 20000,
    dps: int = 50,
    seed: Optional[int] = None,
) -> mp.mpf:
    """
    Optional sanity check for compact simplices only.

    Estimate the hyperbolic volume by Monte Carlo integration of the Klein density
        (1 - ||x||^2)^(-5/2)
    over the Euclidean simplex.  This is not suitable for ideal vertices because the density becomes
    singular on the boundary.
    """
    _set_precision(dps)
    verts, vertex_types, _ = _coerce_vertices(vertices, snap_ideal_tol=None)
    if any(vt == "ideal" for vt in vertex_types):
        raise ValueError("Monte Carlo checking is implemented only for compact simplices")

    if seed is not None:
        random.seed(seed)

    vol_e = euclidean_simplex_volume(verts, dps=dps)
    accum = mp.mpf("0")
    for _ in range(int(samples)):
        # Dirichlet(1,1,1,1,1) via normalized exponentials
        e = [random.gammavariate(1.0, 1.0) for _ in range(5)]
        s = sum(e)
        lam = [mp.mpf(val / s) for val in e]
        x = [mp.fsum(lam[i] * verts[i][k] for i in range(5)) for k in range(4)]
        dens = (1 - _euclid_norm2(x)) ** (-mp.mpf("2.5"))
        accum += dens
    return vol_e * accum / samples


def format_angles(theta: Dict[Tuple[int, int], mp.mpf], digits: int = 16) -> str:
    """
    Pretty-print the ten dihedral angles.
    """
    items = []
    for key in sorted(theta):
        items.append(f"theta{key} = {mp.nstr(theta[key], digits)}")
    return "\n".join(items)


if __name__ == "__main__":
    mp.mp.dps = 80

    compact_verts = [
        [0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0, 0.0],
        [0.0, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.1],
    ]

    compact = hyperbolic_4_simplex_data(compact_verts, dps=80)
    print("Compact simplex")
    print("  volume =", mp.nstr(compact.volume, 25))
    print("  types  =", compact.vertex_types)
    print()

    ideal_verts = regular_ideal_simplex_vertices(dps=80)
    ideal = hyperbolic_4_simplex_data(ideal_verts, dps=80)
    print("Regular ideal simplex")
    print("  volume =", mp.nstr(ideal.volume, 30))
    print("  types  =", ideal.vertex_types)
    print("  expected ~ 0.268895660169311...")
