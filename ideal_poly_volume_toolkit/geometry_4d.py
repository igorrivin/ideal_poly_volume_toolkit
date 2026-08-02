"""
Hyperbolic volume of ideal 4-simplices in H^4.

An ideal 4-simplex has 5 vertices on S^3 = ∂H^4.  Sending one vertex to ∞
in the upper half-space model H^4 = {(x,t) : x ∈ R^3, t > 0}, the remaining
four vertices v_1,...,v_4 form a Euclidean tetrahedron T ⊂ R^3.

The volume is given by the angle formula (Hopf / Poincaré–Schläfli):

    Vol_4([∞,v1,v2,v3,v4]) = (π/3)(4π - Σ_{i<j} φ_{ij} - Σ_ℓ arccos(d_ℓ/R))

where:
  - φ_{ij} are the 6 Euclidean dihedral angles of T
  - R is the circumradius of T, c its circumcenter
  - d_ℓ is the signed distance from c to face F_ℓ (positive toward v_ℓ)

Equivalently (Haagerup–Munkholm half-space integral):

    Vol_4 = (1/3) ∫_T d³x / (R² - |x - c|²)^{3/2}

The regular ideal 4-simplex has all 10 angles = arccos(1/3), giving
    Vol ≈ 0.26889.
"""

import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from itertools import combinations


# ============================================================================
# Core geometry: circumsphere, dihedral angles, face distances
# ============================================================================

def circumsphere(v):
    """
    Circumsphere of a tetrahedron in R^3.

    Parameters
    ----------
    v : (4, 3) array
        Vertices of the tetrahedron.

    Returns
    -------
    c : (3,) array — circumcenter
    R : float — circumradius
    """
    # Solve |x - c|^2 = R^2 for all 4 vertices.
    # Subtracting the first equation from the rest gives a 3×3 linear system.
    A = 2 * (v[1:] - v[0])
    b = np.sum(v[1:]**2 - v[0]**2, axis=1)
    c = np.linalg.solve(A, b)
    R = np.linalg.norm(v[0] - c)
    return c, R


def tet_dihedral_angles(v):
    """
    All 6 Euclidean dihedral angles of a tetrahedron.

    Parameters
    ----------
    v : (4, 3) array
        Vertices of the tetrahedron.

    Returns
    -------
    angles : dict mapping (i, j) -> φ_{ij}
        Dihedral angle along edge v_i--v_j, measured inside the tetrahedron.
    """
    # Face normals: face_normals[ℓ] is the outward normal of F_ℓ (opposite v_ℓ).
    face_normals = np.zeros((4, 3))
    face_verts = [
        (1, 2, 3),  # F_0: opposite v_0
        (0, 2, 3),  # F_1: opposite v_1
        (0, 1, 3),  # F_2: opposite v_2
        (0, 1, 2),  # F_3: opposite v_3
    ]

    for ell, (a, b, c) in enumerate(face_verts):
        n = np.cross(v[b] - v[a], v[c] - v[a])
        # Orient outward: n should point away from v_ℓ
        if np.dot(n, v[ell] - v[a]) > 0:
            n = -n
        face_normals[ell] = n / np.linalg.norm(n)

    # Edge (i,j) is shared by faces F_k and F_ℓ where {k,ℓ} = {0,1,2,3}\{i,j}.
    angles = {}
    for i, j in combinations(range(4), 2):
        others = [x for x in range(4) if x != i and x != j]
        k, ell = others
        # Interior dihedral angle = π - angle between outward normals
        cos_angle = np.dot(face_normals[k], face_normals[ell])
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles[(i, j)] = np.pi - np.arccos(cos_angle)

    return angles


def face_distances_from_circumcenter(v, c):
    """
    Signed distance from circumcenter c to each face of tetrahedron v,
    positive toward the opposite vertex.

    Parameters
    ----------
    v : (4, 3) array
    c : (3,) array — circumcenter

    Returns
    -------
    d : (4,) array — d[ℓ] = signed distance from c to face F_ℓ
    """
    face_verts = [
        (1, 2, 3),
        (0, 2, 3),
        (0, 1, 3),
        (0, 1, 2),
    ]
    d = np.zeros(4)
    for ell, (a, b, cc_idx) in enumerate(face_verts):
        # Normal to face, oriented toward v_ℓ
        n = np.cross(v[b] - v[a], v[cc_idx] - v[a])
        n_hat = n / np.linalg.norm(n)
        # Sign: positive if c is on the same side as v_ℓ
        sign = np.sign(np.dot(n_hat, v[ell] - v[a]))
        n_hat *= sign
        d[ell] = np.dot(n_hat, c - v[a])
    return d


# ============================================================================
# Volume of a single ideal 4-simplex
# ============================================================================

def ideal_4simplex_volume_from_tet(v):
    """
    Volume of the ideal 4-simplex [∞, v_1, v_2, v_3, v_4] in H^4,
    given the Euclidean tetrahedron T = conv(v_1,...,v_4) in R^3.

    Uses the angle formula:
        Vol = (π/3)(4π - Σ_{i<j} φ_{ij} - Σ_ℓ arccos(d_ℓ/R))

    Parameters
    ----------
    v : (4, 3) array
        Vertices of the boundary tetrahedron in R^3.

    Returns
    -------
    vol : float
        Hyperbolic 4-volume.
    """
    c, R = circumsphere(v)
    phi = tet_dihedral_angles(v)
    d = face_distances_from_circumcenter(v, c)

    angle_sum = sum(phi.values())
    face_angle_sum = sum(np.arccos(np.clip(d[ell] / R, -1.0, 1.0)) for ell in range(4))

    return (np.pi / 3) * (4 * np.pi - angle_sum - face_angle_sum)


def ideal_4simplex_volume_integral(v, subdivisions=50):
    """
    Volume of ideal 4-simplex via the Haagerup–Munkholm integral:
        Vol = (1/3) ∫_T d³x / (R² - |x - c|²)^{3/2}

    Uses Monte Carlo integration over T for validation.

    Parameters
    ----------
    v : (4, 3) array
        Vertices of the boundary tetrahedron.
    subdivisions : int
        Number of Monte Carlo samples = subdivisions^3.

    Returns
    -------
    vol : float
    """
    c, R = circumsphere(v)

    # Monte Carlo over tetrahedron using barycentric coordinates
    rng = np.random.default_rng(42)
    n_samples = subdivisions ** 3
    # Uniform sampling in a tetrahedron via sorted exponentials
    e = rng.exponential(size=(n_samples, 4))
    lam = e / e.sum(axis=1, keepdims=True)
    pts = lam @ v  # (n_samples, 3)

    # Euclidean volume of tetrahedron (for MC normalization)
    mat = np.column_stack([v[1] - v[0], v[2] - v[0], v[3] - v[0]])
    tet_vol = abs(np.linalg.det(mat)) / 6

    # Integrand
    diff = pts - c
    r2 = np.sum(diff**2, axis=1)
    integrand = 1.0 / (R**2 - r2) ** 1.5

    return (1.0 / 3) * tet_vol * np.mean(integrand)


# ============================================================================
# Stereographic projection S^3 <-> R^3
# ============================================================================

def stereo_project_S3(pts_4d, pole_idx=0):
    """
    Stereographic projection from S^3 to R^3, projecting from pts_4d[pole_idx].

    Maps the pole to ∞ and the remaining points to R^3.

    Parameters
    ----------
    pts_4d : (N, 4) array
        Points on S^3.
    pole_idx : int
        Index of the point to send to infinity.

    Returns
    -------
    pts_3d : (N-1, 3) array
        Projected points in R^3 (pole excluded).
    finite_indices : (N-1,) array
        Original indices of the finite points.
    """
    pole = pts_4d[pole_idx]
    finite_mask = np.arange(len(pts_4d)) != pole_idx
    finite_indices = np.where(finite_mask)[0]
    pts = pts_4d[finite_indices]

    # Stereographic projection from pole p:
    # For a point q on S^3, project to the hyperplane through the origin
    # perpendicular to p.
    #
    # Standard formula: if pole = (0,0,0,1), then
    #   (x1,x2,x3,x4) -> (x1,x2,x3)/(1-x4)
    #
    # For general pole, rotate so pole -> (0,0,0,1), project, done.

    # Build rotation sending pole to e_4 = (0,0,0,1)
    e4 = np.array([0., 0., 0., 1.])

    # Use Householder reflection if pole != ±e4
    if np.allclose(pole, e4):
        rotated = pts
    elif np.allclose(pole, -e4):
        # Antipodal: reflect x4 -> -x4
        rotated = pts.copy()
        rotated[:, 3] = -rotated[:, 3]
    else:
        # Householder: reflect through hyperplane bisecting pole and e4
        u = pole - e4
        u = u / np.linalg.norm(u)
        rotated = pts - 2 * np.outer(pts @ u, u)

    # Now project from (0,0,0,1): (x1,x2,x3,x4) -> (x1,x2,x3)/(1-x4)
    denom = 1.0 - rotated[:, 3]
    # Guard against division by zero (point = pole, shouldn't happen)
    denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
    pts_3d = rotated[:, :3] / denom[:, np.newaxis]

    return pts_3d, finite_indices


# ============================================================================
# Full pipeline: points on S^3 -> ideal polytope volume in H^4
# ============================================================================

def ideal_polytope_volume_4d(pts_4d, pole_idx=0):
    """
    Compute the hyperbolic 4-volume of the ideal polytope with vertices
    on S^3 = ∂H^4.

    Pipeline:
      1. Stereographic project from pts_4d[pole_idx] to R^3
      2. Delaunay triangulate the projected points
      3. Each Delaunay tetrahedron gives an ideal 4-simplex [∞, v1, v2, v3, v4]
      4. Sum the volumes

    Parameters
    ----------
    pts_4d : (N, 4) array
        Points on S^3 (unit norm).
    pole_idx : int
        Which point to send to ∞.

    Returns
    -------
    total_vol : float
        Total hyperbolic 4-volume.
    n_tets : int
        Number of Delaunay tetrahedra.
    """
    pts_3d, _ = stereo_project_S3(pts_4d, pole_idx)

    tri = Delaunay(pts_3d)
    tets = tri.simplices  # (M, 4) indices into pts_3d

    total_vol = 0.0
    for tet_idx in tets:
        v = pts_3d[tet_idx]  # (4, 3)
        total_vol += ideal_4simplex_volume_from_tet(v)

    return total_vol, len(tets)


def regular_ideal_4simplex_volume():
    """
    Exact volume of the regular ideal 4-simplex.

    All 10 dihedral angles = arccos(1/3), giving:
        Vol = (π/3)(4π - 10·arccos(1/3)) ≈ 0.26889
    """
    return (np.pi / 3) * (4 * np.pi - 10 * np.arccos(1.0 / 3))


# ============================================================================
# Surface area: 3-volume of boundary facets (ideal tetrahedra in H^3)
# ============================================================================

def _lobachevsky(theta, n_terms=96):
    """Lobachevsky function Λ(θ) = -(1/2) ∫_0^{2θ} log|2 sin t| dt."""
    k = np.arange(1, n_terms + 1)
    return 0.5 * np.sum(np.sin(2 * k * theta) / (k * k))


def ideal_tet_volume_3d(pts_S3):
    """
    Hyperbolic 3-volume of an ideal tetrahedron in H^3 whose 4 vertices
    lie on S^3 (on a common S^2 ⊂ S^3).

    The 4 points span a 3-plane in R^4; restrict to that plane to get
    4 points on a 2-sphere, then stereographic project to C and compute
    via the Lobachevsky angle formula.

    Parameters
    ----------
    pts_S3 : (4, 4) array
        Four points on S^3.

    Returns
    -------
    vol3 : float
        Hyperbolic 3-volume (always non-negative).
    """
    # 1. Project the 4 points into their affine 3-plane.
    #    Translate so p0 is at origin, then take an ONB of the span.
    p = pts_S3 - pts_S3[0]
    # SVD to get 3D coordinates (the 4 pts span at most a 3D subspace)
    U, S, Vt = np.linalg.svd(p[1:], full_matrices=False)
    # pts in 3D: rows of (p[1:] @ Vt[:3].T)
    coords_3d = p @ Vt[:3].T  # (4, 3), first row is origin

    # 2. The 4 points now lie on a sphere in R^3.  Stereographic
    #    project from coords_3d[0] = origin to get 3 points in R^2.
    #    For projection from a point on a sphere centered at c with radius r,
    #    map q -> c + r^2 * (q - p0) / |q - p0|^2  ... but simpler:
    #    just use the inverse-stereographic through the circumsphere.

    # Find circumsphere of the 4 points in R^3
    v = coords_3d
    A = 2 * (v[1:] - v[0])
    b = np.sum(v[1:]**2 - v[0]**2, axis=1)
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return 0.0  # degenerate
    R = np.linalg.norm(v[0] - c)

    # 3. Stereographic projection from v[0] on the sphere (c, R).
    #    Map v[0] -> ∞, other points -> R^2.
    #    Formula: for a sphere centered at c with radius R, projection
    #    from pole p on the sphere:
    #      σ(q) = p + 2R² (q - p) / |q - p|²
    #    But we only need the resulting 2D triangle's angles.
    #
    #    Simpler: translate so circumcenter = origin, scale to unit sphere.
    w = (v - c) / R  # 4 points on unit sphere in R^3

    # Stereographic project from w[0]: q -> (q - w0·(q·w0)) mapped to
    # the tangent plane at w[0].  Standard formula for unit sphere:
    # σ(q) = (q - w0) / (1 - q·w0), then take components in tangent plane.
    #
    # Even simpler: rotate so w[0] = (0,0,1), then project (x,y,z) -> (x,y)/(1-z).
    pole = w[0]
    e3 = np.array([0., 0., 1.])

    if np.allclose(pole, e3):
        rotated = w
    elif np.allclose(pole, -e3):
        rotated = w.copy()
        rotated[:, 2] = -rotated[:, 2]
    else:
        u = pole - e3
        u = u / np.linalg.norm(u)
        rotated = w - 2 * np.outer(w @ u, u)

    # Project the 3 finite points to R^2
    finite = rotated[1:]
    denom = 1.0 - finite[:, 2]
    denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
    z_complex = (finite[:, 0] + 1j * finite[:, 1]) / denom

    # 4. Compute triangle angles and Lobachevsky sum.
    z1, z2, z3 = z_complex
    def angle_at(a, b, c):
        u = np.array([(b - a).real, (b - a).imag])
        v = np.array([(c - a).real, (c - a).imag])
        dot = u @ v
        cross = u[0] * v[1] - u[1] * v[0]
        return np.arctan2(abs(cross), dot)

    a1 = angle_at(z1, z2, z3)
    a2 = angle_at(z2, z3, z1)
    a3 = angle_at(z3, z1, z2)

    return abs(_lobachevsky(a1) + _lobachevsky(a2) + _lobachevsky(a3))


def ideal_polytope_surface_area_4d(pts_4d):
    """
    Compute the hyperbolic 3-surface area of the ideal 4-polytope with
    vertices on S^3.

    Each boundary facet (a tetrahedron in R^4) is an ideal tetrahedron
    in a totally geodesic H^3 ⊂ H^4.  The surface area is the sum of
    their hyperbolic 3-volumes.

    Parameters
    ----------
    pts_4d : (N, 4) array
        Points on S^3 (unit norm).

    Returns
    -------
    total_area : float
        Total hyperbolic 3-volume of boundary.
    n_facets : int
        Number of boundary facets (tetrahedra).
    facet_vols : list of float
        Individual facet volumes.
    """
    hull = ConvexHull(pts_4d)
    facets = hull.simplices  # (M, 4) — tetrahedral boundary facets

    facet_vols = []
    for facet in facets:
        v = pts_4d[facet]  # (4, 4) — four points on S^3
        vol3 = ideal_tet_volume_3d(v)
        facet_vols.append(vol3)

    return sum(facet_vols), len(facets), facet_vols
