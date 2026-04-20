"""
Numerical volume routines for compact hyperbolic 5- and 6-simplices.

What this module does
---------------------
1. H^5:
   - direct QMC evaluation of the Klein-model 5D volume integral;
   - a Schlaefli-path integrator
         d Vol_5 = -(1/4) * sum_{codim-2 faces F^3} Vol_3(F^3) d theta_F,
     along the radial path x_i(t) = t x_i.
     The 3-face volumes are computed intrinsically by extracting Klein coordinates for the
     face from the ambient hyperboloid Gram matrix, then evaluating the 3D Klein integral.

2. H^6:
   - direct QMC evaluation of the Klein-model 6D volume integral;
   - the even-dimensional Poincare-Schlaefli-Hopf reduction
         Vol_6 = (8 pi^3 / 15) * (17/4 - w_4 + (1/2) w_2 - w_0),
     where
         w_4 = (1 / 2 pi)   * sum over 21 codim-2 faces of the dihedral angles,
         w_2 = (1 / 2 pi^2) * sum over 35 two-faces of the S^3 link volumes,
         w_0 = (1 / pi^3)   * sum over 7 vertices of the S^5 link volumes.
     The S^3 links are evaluated by Murakami's log/dilog formula for spherical tetrahedra.
     The S^5 vertex links are evaluated numerically by gnomonic coordinates and a 5D QMC
     integral on the corresponding spherical simplex.

3. Model wrappers:
   All public H^5 / H^6 entry points accept vertices in the Klein model directly, or in the
   Poincare ball, upper half-space, or Minkowski / hyperboloid models via the companion
   module hyperbolic_model_transforms.py.

Scope and limitations
---------------------
- This module assumes the simplices are compact: every vertex must lie strictly inside the
  relevant Klein ball. Ideal and ultra-ideal vertices are not handled here.
- The H^6 reduction formula is exact, but its numerical evaluation can suffer from strong
  cancellation near the Euclidean / very-small-volume regime. The direct 6D QMC integral is
  included as a cross-check for that reason.
- The code aims to be readable and practical rather than maximally optimized.

Dependencies
------------
- numpy
- scipy
- mpmath
- hyperbolic_model_transforms.py (only when model != "klein")

Typical usage
-------------
    from hyperbolic_5_6_simplex_volume import (
        hyperbolic_5_simplex_volume,
        hyperbolic_6_simplex_volume,
    )

    verts5 = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.5],
    ]
    V5 = hyperbolic_5_simplex_volume(verts5, method="schlafli")

    verts6 = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.7, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.7, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.7, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.7, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.7],
    ]
    V6 = hyperbolic_6_simplex_volume(verts6, method="reduction")
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mpmath as mp
import numpy as np
import scipy.linalg
from scipy.stats import qmc

try:
    from hyperbolic_model_transforms import convert_vertices
except Exception:  # pragma: no cover - only matters when model wrappers are used
    convert_vertices = None


ArrayLike = Sequence[Sequence[float | int]]


@dataclass
class Hyperbolic5SimplexResult:
    volume_schlafli: float
    volume_direct: Optional[float]
    t0: float
    steps: int
    angle_gram: np.ndarray
    dihedral_angles: Dict[Tuple[int, int], float]


@dataclass
class Hyperbolic6SimplexResult:
    volume_reduction: float
    volume_direct: Optional[float]
    w0: float
    w2: float
    w4: float
    angle_gram: np.ndarray
    dihedral_angles: Dict[Tuple[int, int], float]
    tetrahedral_link_volumes: Dict[Tuple[int, int, int, int], float]
    vertex_link_volumes: List[float]


def _set_mpmath_precision(dps: int) -> None:
    mp.mp.dps = int(dps)


def _as_numpy_vertices(vertices: ArrayLike, expected_count: int, expected_dim: int) -> np.ndarray:
    arr = np.asarray(vertices, dtype=float)
    if arr.shape != (expected_count, expected_dim):
        raise ValueError(
            f"expected {expected_count} vertices in R^{expected_dim}, got array with shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("vertices must be finite real numbers")
    norms2 = np.sum(arr * arr, axis=1)
    if np.any(norms2 >= 1.0):
        bad = np.where(norms2 >= 1.0)[0][0]
        raise ValueError(
            f"vertex {bad} has squared norm {norms2[bad]!r}, so it is not strictly inside the Klein ball"
        )
    vol_e = _euclidean_simplex_volume(arr)
    if vol_e <= 0:
        raise ValueError("the vertices are not affinely independent")
    return arr


def _maybe_convert_to_klein(
    vertices: Sequence[Sequence[float | int]],
    model: str,
) -> Sequence[Sequence[float | int]]:
    name = model.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "klein": "klein",
        "projective": "klein",
        "beltrami": "klein",
        "beltrami_klein": "klein",
        "poincare": "poincare",
        "poincare_ball": "poincare",
        "ball": "poincare",
        "halfspace": "halfspace",
        "upper_halfspace": "halfspace",
        "upper_half_space": "halfspace",
        "half_space": "halfspace",
        "minkowski": "minkowski",
        "hyperboloid": "minkowski",
        "lorentz": "minkowski",
    }
    if name not in aliases:
        raise ValueError(f"unknown model {model!r}")
    if aliases[name] == "klein":
        return vertices
    if convert_vertices is None:
        raise ImportError(
            "model conversion requested but hyperbolic_model_transforms.py could not be imported"
        )
    converted = convert_vertices(vertices, from_model=aliases[name], to_model="klein")
    return converted


def _euclidean_simplex_volume(vertices: np.ndarray) -> float:
    d = vertices.shape[1]
    edge_matrix = (vertices[1:] - vertices[0]).T
    return abs(float(np.linalg.det(edge_matrix))) / math.factorial(d)


def _next_power_of_two_geq(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


def _uniform_barycentric_samples_on_simplex(
    dimension: int,
    samples: int,
    *,
    scramble: bool = True,
    seed: Optional[int] = 0,
) -> np.ndarray:
    """
    Return approximately-uniform samples on the standard dimension-simplex in barycentric form.

    The simplex is sampled by sorting Sobol points in [0,1]^dimension; the consecutive
    differences give Dirichlet(1,...,1) barycentric coordinates.
    """
    n_points = _next_power_of_two_geq(int(samples))
    engine = qmc.Sobol(d=dimension, scramble=scramble, seed=seed)
    u = engine.random_base2(int(math.log2(n_points)))
    s = np.sort(u, axis=1)
    bary = np.empty((n_points, dimension + 1), dtype=float)
    bary[:, 0] = s[:, 0]
    for i in range(1, dimension):
        bary[:, i] = s[:, i] - s[:, i - 1]
    bary[:, dimension] = 1.0 - s[:, dimension - 1]
    return bary


def _simplex_qmc_integral(
    vertices: np.ndarray,
    density,
    *,
    samples: int,
    scramble: bool = True,
    seed: Optional[int] = 0,
) -> float:
    d = vertices.shape[1]
    bary = _uniform_barycentric_samples_on_simplex(d, samples, scramble=scramble, seed=seed)
    x = bary @ vertices
    vals = np.asarray(density(x), dtype=float)
    return _euclidean_simplex_volume(vertices) * float(np.mean(vals))


def hyperbolic_simplex_volume_direct_klein(
    vertices: ArrayLike,
    *,
    samples: int = 1 << 17,
    scramble: bool = True,
    seed: Optional[int] = 0,
) -> float:
    """
    Direct QMC evaluation of the Klein-model simplex integral

        Vol_n(Delta) = integral_{Delta_E} (1 - ||x||^2)^(-(n+1)/2) dx,

    for a full-dimensional simplex Delta in B^n.
    """
    arr = np.asarray(vertices, dtype=float)
    n_vertices, dim = arr.shape
    if n_vertices != dim + 1:
        raise ValueError(
            "for direct_klein volume, the number of vertices must equal ambient dimension + 1"
        )
    _ = _as_numpy_vertices(arr, n_vertices, dim)
    return _simplex_qmc_integral(
        arr,
        lambda x: (1.0 - np.sum(x * x, axis=1)) ** (-(dim + 1) / 2.0),
        samples=samples,
        scramble=scramble,
        seed=seed,
    )


def _hyperboloid_lifts_from_klein(vertices: np.ndarray) -> np.ndarray:
    r2 = np.sum(vertices * vertices, axis=1)
    den = np.sqrt(1.0 - r2)
    return np.column_stack([1.0 / den, vertices / den[:, None]])


def _minkowski_inner(u: np.ndarray, v: np.ndarray) -> float:
    return float(-u[0] * v[0] + np.dot(u[1:], v[1:]))


def intrinsic_klein_coordinates(vertices: ArrayLike) -> np.ndarray:
    """
    Given k+1 vertices of a k-simplex in the Klein ball B^N (possibly with N > k),
    recover Klein coordinates for the same intrinsic hyperbolic simplex in B^k.

    Numerically, we pass to hyperboloid lifts, build the Lorentzian Gram matrix of the
    lifted vertices, factor it into signature (-,+,...,+), and project back to the
    intrinsic Klein model.
    """
    verts = np.asarray(vertices, dtype=float)
    m, ambient_dim = verts.shape
    k = m - 1
    if k < 1:
        raise ValueError("need at least two vertices")
    if np.any(np.sum(verts * verts, axis=1) >= 1.0):
        raise ValueError("intrinsic_klein_coordinates requires compact (finite) Klein vertices")

    h = _hyperboloid_lifts_from_klein(verts)
    vertex_gram = np.empty((m, m), dtype=float)
    for i in range(m):
        for j in range(m):
            vertex_gram[i, j] = -_minkowski_inner(h[i], h[j])

    minkowski_matrix = -vertex_gram
    evals, evecs = np.linalg.eigh(minkowski_matrix)
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    tol = 1e-12
    n_negative = int(np.sum(evals < -tol))
    n_positive = int(np.sum(evals > tol))
    if n_negative != 1 or n_positive != k:
        raise ValueError(
            "the simplex span does not have the expected Lorentzian signature (1,k); "
            f"found eigenvalues {evals!r}"
        )

    coords_columns = np.diag(np.sqrt(np.abs(evals))) @ evecs.T
    coords = coords_columns.T
    if float(np.mean(coords[:, 0])) < 0.0:
        coords[:, 0] *= -1.0
    if np.any(coords[:, 0] <= 0.0):
        raise ValueError("failed to orient the intrinsic hyperboloid coordinates with positive time coordinate")

    klein = coords[:, 1:] / coords[:, [0]]
    return klein


def intrinsic_hyperbolic_simplex_volume(
    vertices: ArrayLike,
    *,
    samples: int = 1 << 12,
    scramble: bool = True,
    seed: Optional[int] = 0,
) -> float:
    klein = intrinsic_klein_coordinates(vertices)
    return hyperbolic_simplex_volume_direct_klein(
        klein,
        samples=samples,
        scramble=scramble,
        seed=seed,
    )


def angle_gram_from_klein(vertices: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For a compact hyperbolic simplex in the Klein model, return (A, R, Q), where

        Q_ij = 1 - x_i . x_j,
        R    = Q^{-1},
        A_ii = 1,
        A_ij = -cos(theta_ij) = -R_ij / sqrt((-R_ii)(-R_jj)).
    """
    verts = np.asarray(vertices, dtype=float)
    m, dim = verts.shape
    _ = _as_numpy_vertices(verts, m, dim)
    Q = 1.0 - verts @ verts.T
    R = np.linalg.inv(Q)
    A = np.eye(m, dtype=float)
    for i in range(m):
        if R[i, i] >= 0.0:
            raise ValueError(
                "the inverse Klein matrix does not have the expected negative diagonal; "
                "the simplex may be degenerate or non-compact"
            )
        for j in range(m):
            if i == j:
                continue
            A[i, j] = -R[i, j] / math.sqrt((-R[i, i]) * (-R[j, j]))
    return A, R, Q


def dihedral_angles_from_angle_gram(A: np.ndarray) -> Dict[Tuple[int, int], float]:
    m = A.shape[0]
    out: Dict[Tuple[int, int], float] = {}
    for i in range(m):
        for j in range(i + 1, m):
            c = float(-A[i, j])
            c = max(-1.0, min(1.0, c))
            out[(i, j)] = math.acos(c)
    return out


def _numpy_to_mp_matrix(M: np.ndarray) -> mp.matrix:
    return mp.matrix([[mp.mpf(str(M[i, j])) for j in range(M.shape[1])] for i in range(M.shape[0])])


def spherical_tetrahedron_volume_murakami(A4: np.ndarray, *, dps: int = 80) -> float:
    """
    Murakami's volume formula for a spherical tetrahedron from its angle Gram matrix.

    The input A4 must be a 4x4 angle Gram matrix with diagonal 1 and off-diagonal entries
    A_ij = -cos(alpha_ij).
    """
    _set_mpmath_precision(dps)
    B = _numpy_to_mp_matrix(np.asarray(A4, dtype=float))
    if B.rows != 4 or B.cols != 4:
        raise ValueError("Murakami's formula expects a 4x4 angle Gram matrix")

    def clip_unit(x: mp.mpf, tol: mp.mpf = mp.mpf("1e-30")) -> mp.mpf:
        if x > 1 and x - 1 <= tol:
            return mp.mpf("1")
        if x < -1 and -1 - x <= tol:
            return mp.mpf("-1")
        if x < -1 or x > 1:
            raise ValueError(f"value {x!r} lies outside [-1,1]")
        return x

    entries = [-B[0, 1], -B[0, 2], -B[1, 2], -B[2, 3], -B[1, 3], -B[0, 3]]
    alphas = [mp.acos(clip_unit(mp.re(e))) for e in entries]
    a1, a2, a3, a4, a5, a6 = [mp.e ** (1j * alpha) for alpha in alphas]

    def Li2(z):
        return mp.polylog(2, z)

    def L(z):
        cross_term = mp.log(a1) * mp.log(a4) + mp.log(a2) * mp.log(a5) + mp.log(a3) * mp.log(a6)
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

    vol = -mp.re(L(z0)) + mp.pi * (mp.arg(-q2) + mp.fsum(alphas) / 2) - mp.mpf("1.5") * mp.pi**2

    # Murakami's formula is naturally modulo 2*pi^2. For convex spherical tetrahedra we choose
    # the physical branch in [0, pi^2].
    two_pi_sq = 2 * mp.pi**2
    vol = mp.re(vol)
    while vol < 0:
        vol += two_pi_sq
    while vol >= two_pi_sq:
        vol -= two_pi_sq
    if vol > mp.pi**2:
        vol = two_pi_sq - vol
    return float(vol)


def spherical_vertices_from_angle_gram(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct spherical simplex vertices from an angle Gram matrix.

    If A is the spherical angle Gram matrix, then the vertex Gram matrix is
        G_ij = (A^{-1})_ij / sqrt((A^{-1})_ii (A^{-1})_jj).
    """
    A = np.asarray(A, dtype=float)
    R = np.linalg.inv(A)
    d = np.sqrt(np.diag(R))
    G = R / np.outer(d, d)
    G = 0.5 * (G + G.T)
    # Numerical safety near roundoff.
    evals, evecs = np.linalg.eigh(G)
    tol = 1e-12
    if np.min(evals) < -tol:
        raise ValueError(f"reconstructed spherical vertex Gram has negative eigenvalue(s): {evals!r}")
    evals = np.clip(evals, 0.0, None)
    coords = np.diag(np.sqrt(evals)) @ evecs.T
    verts = coords.T  # rows are vertex coordinates in R^{d+1}
    return verts, G


def spherical_simplex_gnomonic_coordinates_from_angle_gram(A: np.ndarray) -> np.ndarray:
    """
    Convert a spherical simplex given by its angle Gram matrix into Euclidean simplex
    coordinates in a tangent hyperplane, via the gnomonic projection.

    We reconstruct spherical vertices v_0,...,v_d on S^d, choose a pole c in the interior
    hemisphere by solving G a = 1 and setting c proportional to sum a_i v_i, and then use
        x_i = proj_{c^\\perp}(v_i) / <v_i, c>.
    """
    verts, G = spherical_vertices_from_angle_gram(A)
    a = np.linalg.solve(G, np.ones(G.shape[0], dtype=float))
    pole = a @ verts
    pole_norm = np.linalg.norm(pole)
    if pole_norm <= 0.0:
        raise ValueError("failed to construct an interior gnomonic pole for the spherical simplex")
    pole = pole / pole_norm

    dots = verts @ pole
    if np.any(dots <= 1e-12):
        raise ValueError(
            "the reconstructed spherical simplex is not contained in the chosen open hemisphere"
        )

    basis = scipy.linalg.null_space(pole.reshape(1, -1)).T
    if basis.shape[0] != verts.shape[0] - 1:
        raise ValueError("unexpected tangent-space basis dimension in the spherical gnomonic step")
    coords = (verts @ basis.T) / dots[:, None]
    return coords


def spherical_simplex_volume_qmc_from_angle_gram(
    A: np.ndarray,
    *,
    samples: int = 1 << 16,
    scramble: bool = True,
    seed: Optional[int] = 0,
) -> float:
    coords = spherical_simplex_gnomonic_coordinates_from_angle_gram(A)
    d = coords.shape[1]
    return _simplex_qmc_integral(
        coords,
        lambda x: (1.0 + np.sum(x * x, axis=1)) ** (-(d + 1) / 2.0),
        samples=samples,
        scramble=scramble,
        seed=seed,
    )


def hyperbolic_5_simplex_data(
    vertices: ArrayLike,
    *,
    model: str = "klein",
    t0: float = 0.2,
    steps: int = 24,
    initial_samples: int = 1 << 15,
    face_samples: int = 1 << 11,
    direct_samples: Optional[int] = 1 << 17,
    scramble: bool = True,
    seed: Optional[int] = 0,
) -> Hyperbolic5SimplexResult:
    """
    Compute the volume of a compact hyperbolic 5-simplex by the Schlaefli path integral,
    with an optional direct 5D Klein QMC cross-check.

    The simplex is scaled radially by x_i(t) = t x_i. We compute the exact 5-volume at t=t0
    by direct QMC, and then integrate Schlaefli's differential formula from t0 to 1.
    """
    klein_vertices = _maybe_convert_to_klein(vertices, model)
    verts = _as_numpy_vertices(klein_vertices, expected_count=6, expected_dim=5)

    if not (0.0 < t0 < 1.0):
        raise ValueError("t0 must satisfy 0 < t0 < 1")
    if steps < 1:
        raise ValueError("steps must be positive")

    V = hyperbolic_simplex_volume_direct_klein(
        t0 * verts,
        samples=initial_samples,
        scramble=scramble,
        seed=seed,
    )

    ts = np.linspace(float(t0), 1.0, int(steps) + 1)
    A0, _, _ = angle_gram_from_klein(ts[0] * verts)
    prev_angles = dihedral_angles_from_angle_gram(A0)
    pairs = sorted(prev_angles)

    for k in range(len(ts) - 1):
        a = ts[k]
        b = ts[k + 1]
        mid = 0.5 * (a + b)
        mid_vertices = mid * verts

        face_volumes: Dict[Tuple[int, int], float] = {}
        for pair_index, (i, j) in enumerate(pairs):
            face_idx = [r for r in range(6) if r not in (i, j)]
            face_seed = None if seed is None else seed + 1000 * k + pair_index
            face_volumes[(i, j)] = intrinsic_hyperbolic_simplex_volume(
                mid_vertices[face_idx],
                samples=face_samples,
                scramble=scramble,
                seed=face_seed,
            )

        A1, _, _ = angle_gram_from_klein(b * verts)
        next_angles = dihedral_angles_from_angle_gram(A1)
        increment = 0.0
        for key in pairs:
            increment += face_volumes[key] * (next_angles[key] - prev_angles[key])
        V += -0.25 * increment
        prev_angles = next_angles

    Afinal, _, _ = angle_gram_from_klein(verts)
    final_dihedral_angles = dihedral_angles_from_angle_gram(Afinal)

    direct_volume = None
    if direct_samples is not None:
        direct_volume = hyperbolic_simplex_volume_direct_klein(
            verts,
            samples=direct_samples,
            scramble=scramble,
            seed=None if seed is None else seed + 999999,
        )

    return Hyperbolic5SimplexResult(
        volume_schlafli=float(V),
        volume_direct=direct_volume,
        t0=float(t0),
        steps=int(steps),
        angle_gram=Afinal,
        dihedral_angles=final_dihedral_angles,
    )


def hyperbolic_5_simplex_volume(
    vertices: ArrayLike,
    *,
    model: str = "klein",
    method: str = "schlafli",
    **kwargs,
) -> float:
    method_name = method.strip().lower()
    if method_name == "direct":
        klein_vertices = _maybe_convert_to_klein(vertices, model)
        verts = _as_numpy_vertices(klein_vertices, expected_count=6, expected_dim=5)
        return hyperbolic_simplex_volume_direct_klein(verts, **kwargs)
    if method_name != "schlafli":
        raise ValueError("method must be 'schlafli' or 'direct'")
    return hyperbolic_5_simplex_data(vertices, model=model, **kwargs).volume_schlafli


def hyperbolic_6_simplex_data(
    vertices: ArrayLike,
    *,
    model: str = "klein",
    vertex_link_samples: int = 1 << 19,
    direct_samples: Optional[int] = 1 << 18,
    murakami_dps: int = 100,
    scramble: bool = True,
    seed: Optional[int] = 0,
) -> Hyperbolic6SimplexResult:
    """
    Compute the volume of a compact hyperbolic 6-simplex by the H^6 Hopf reduction,
    with an optional direct 6D Klein QMC cross-check.

    Formula used:
        Vol_6 = (8 pi^3 / 15) * (17/4 - w_4 + (1/2) w_2 - w_0).
    """
    klein_vertices = _maybe_convert_to_klein(vertices, model)
    verts = _as_numpy_vertices(klein_vertices, expected_count=7, expected_dim=6)
    A, _, _ = angle_gram_from_klein(verts)
    theta = dihedral_angles_from_angle_gram(A)
    total_theta = float(sum(theta.values()))
    w4 = total_theta / (2.0 * math.pi)

    facet_indices = list(range(7))

    tetra_link_volumes: Dict[Tuple[int, int, int, int], float] = {}
    total_link3 = 0.0
    for I in itertools.combinations(facet_indices, 4):
        B = A[np.ix_(I, I)]
        vol = spherical_tetrahedron_volume_murakami(B, dps=murakami_dps)
        tetra_link_volumes[tuple(I)] = vol
        total_link3 += vol
    w2 = total_link3 / (2.0 * math.pi**2)

    vertex_link_volumes: List[float] = []
    total_link5 = 0.0
    for i in facet_indices:
        I = [j for j in facet_indices if j != i]
        B = A[np.ix_(I, I)]
        link_seed = None if seed is None else seed + 10000 + i
        vol = spherical_simplex_volume_qmc_from_angle_gram(
            B,
            samples=vertex_link_samples,
            scramble=scramble,
            seed=link_seed,
        )
        vertex_link_volumes.append(vol)
        total_link5 += vol
    w0 = total_link5 / (math.pi**3)

    volume_reduction = (8.0 * math.pi**3 / 15.0) * (17.0 / 4.0 - w4 + 0.5 * w2 - w0)

    direct_volume = None
    if direct_samples is not None:
        direct_volume = hyperbolic_simplex_volume_direct_klein(
            verts,
            samples=direct_samples,
            scramble=scramble,
            seed=None if seed is None else seed + 999999,
        )

    return Hyperbolic6SimplexResult(
        volume_reduction=float(volume_reduction),
        volume_direct=direct_volume,
        w0=float(w0),
        w2=float(w2),
        w4=float(w4),
        angle_gram=A,
        dihedral_angles=theta,
        tetrahedral_link_volumes=tetra_link_volumes,
        vertex_link_volumes=vertex_link_volumes,
    )


def hyperbolic_6_simplex_volume(
    vertices: ArrayLike,
    *,
    model: str = "klein",
    method: str = "reduction",
    **kwargs,
) -> float:
    method_name = method.strip().lower()
    if method_name == "direct":
        klein_vertices = _maybe_convert_to_klein(vertices, model)
        verts = _as_numpy_vertices(klein_vertices, expected_count=7, expected_dim=6)
        return hyperbolic_simplex_volume_direct_klein(verts, **kwargs)
    if method_name != "reduction":
        raise ValueError("method must be 'reduction' or 'direct'")
    return hyperbolic_6_simplex_data(vertices, model=model, **kwargs).volume_reduction


if __name__ == "__main__":
    # Small smoke tests.
    verts5 = np.vstack([np.zeros((1, 5)), 0.5 * np.eye(5)])
    res5 = hyperbolic_5_simplex_data(verts5, direct_samples=1 << 16)
    print("H^5 simplex")
    print("  Schlaefli volume:", res5.volume_schlafli)
    print("  direct QMC volume:", res5.volume_direct)
    if res5.volume_direct is not None and res5.volume_direct != 0:
        print("  relative difference:", (res5.volume_schlafli - res5.volume_direct) / res5.volume_direct)

    verts6 = np.vstack([np.zeros((1, 6)), 0.7 * np.eye(6)])
    res6 = hyperbolic_6_simplex_data(verts6, direct_samples=1 << 18, vertex_link_samples=1 << 19)
    print("H^6 simplex")
    print("  reduction volume:", res6.volume_reduction)
    print("  direct QMC volume:", res6.volume_direct)
    if res6.volume_direct is not None and res6.volume_direct != 0:
        print("  relative difference:", (res6.volume_reduction - res6.volume_direct) / res6.volume_direct)
