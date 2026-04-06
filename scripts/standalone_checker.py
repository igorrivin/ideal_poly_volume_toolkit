#!/usr/bin/env python3
"""
Standalone checker for ideal 4-polytope benchmark problems.

Verifies that a set of vertices defines a valid ideal 4-polytope in H^4
with a specified geometric measure (volume, surface area, or isoperimetric
ratio) within a given tolerance window.

Usage:
    python standalone_checker.py answer.json

The answer.json must contain:
    - "vertices": list of N points in R^4  (claimed to lie on S^3)
    - "measure": one of "volume", "surface_area", "ratio"
    - "target": the target value
    - "epsilon": the allowed tolerance (answer must be in [target-epsilon, target+epsilon])

Optionally:
    - "n_vertices": expected number of vertices (if checking vertex count)
    - "n_facets": expected number of facets (if checking facet count)

Exit code 0 = PASS, 1 = FAIL. All diagnostics printed to stdout.

Dependencies: numpy, scipy (no custom libraries).
"""

import json
import sys
import numpy as np
from itertools import combinations


# ===========================================================================
# Hyperbolic geometry from first principles.
# We use the Klein model: ideal vertices on S^3 in R^4.
# ===========================================================================

def lobachevsky(theta, n_terms=200):
    """Lobachevsky function: L(theta) = -(1/2) int_0^{2*theta} log|2 sin t| dt.
    Computed via Clausen function: L(theta) = (1/2) sum_{k=1}^N sin(2k*theta)/k^2."""
    k = np.arange(1, n_terms + 1)
    return 0.5 * np.sum(np.sin(2 * k * theta) / (k * k))


# ---------------------------------------------------------------------------
# 4-volume of an ideal 4-simplex via the Haagerup-Munkholm angle formula
# ---------------------------------------------------------------------------

def circumsphere(v):
    """Circumsphere of a tetrahedron in R^3. Returns (center, radius)."""
    A = 2 * (v[1:] - v[0])
    b = np.sum(v[1:]**2 - v[0]**2, axis=1)
    c = np.linalg.solve(A, b)
    R = np.linalg.norm(v[0] - c)
    return c, R


def tet_dihedral_angles(v):
    """All 6 Euclidean dihedral angles of tetrahedron v (4x3 array).
    Returns dict mapping (i,j) -> angle along edge v_i--v_j."""
    face_verts = [(1,2,3), (0,2,3), (0,1,3), (0,1,2)]
    face_normals = np.zeros((4, 3))
    for ell, (a, b, c) in enumerate(face_verts):
        n = np.cross(v[b] - v[a], v[c] - v[a])
        nn = np.linalg.norm(n)
        if nn < 1e-15:
            face_normals[ell] = 0
            continue
        n = n / nn
        if np.dot(n, v[ell] - v[a]) > 0:
            n = -n
        face_normals[ell] = n

    angles = {}
    for i, j in combinations(range(4), 2):
        others = [x for x in range(4) if x != i and x != j]
        k, ell = others
        cos_a = np.dot(face_normals[k], face_normals[ell])
        cos_a = np.clip(cos_a, -1.0, 1.0)
        angles[(i, j)] = np.pi - np.arccos(cos_a)
    return angles


def face_distances_from_circumcenter(v, c):
    """Signed distance from circumcenter c to each face of tetrahedron v.
    Positive toward the opposite vertex."""
    face_verts = [(1,2,3), (0,2,3), (0,1,3), (0,1,2)]
    d = np.zeros(4)
    for ell, (a, b, cc_idx) in enumerate(face_verts):
        n = np.cross(v[b] - v[a], v[cc_idx] - v[a])
        nn = np.linalg.norm(n)
        if nn < 1e-15:
            continue
        n_hat = n / nn
        sign = np.sign(np.dot(n_hat, v[ell] - v[a]))
        n_hat *= sign
        d[ell] = np.dot(n_hat, c - v[a])
    return d


def ideal_4simplex_volume_from_tet(v):
    """
    Volume of the ideal 4-simplex [infinity, v1, v2, v3, v4] in H^4,
    given the Euclidean tetrahedron T = conv(v1,...,v4) in R^3.

    Uses the angle formula (Haagerup-Munkholm / Poincare-Schlafli):
        Vol = (pi/3)(4*pi - sum_{i<j} phi_ij - sum_ell arccos(d_ell/R))

    where phi_ij are the 6 Euclidean dihedral angles, R is the circumradius,
    and d_ell is the signed distance from circumcenter to face ell.
    """
    c, R = circumsphere(v)
    phi = tet_dihedral_angles(v)
    d = face_distances_from_circumcenter(v, c)

    angle_sum = sum(phi.values())
    face_angle_sum = sum(
        np.arccos(np.clip(d[ell] / R, -1.0, 1.0)) for ell in range(4)
    )
    return (np.pi / 3) * (4 * np.pi - angle_sum - face_angle_sum)


# ---------------------------------------------------------------------------
# Stereographic projection S^3 -> R^3
# ---------------------------------------------------------------------------

def stereo_project_S3(pts_4d, pole_idx=0):
    """Stereographic projection from S^3 to R^3, projecting from pts_4d[pole_idx].
    Returns (projected_points_3d, indices_of_finite_points)."""
    pole = pts_4d[pole_idx]
    finite_mask = np.arange(len(pts_4d)) != pole_idx
    finite_indices = np.where(finite_mask)[0]
    pts = pts_4d[finite_indices]

    e4 = np.array([0., 0., 0., 1.])
    if np.allclose(pole, e4):
        rotated = pts
    elif np.allclose(pole, -e4):
        rotated = pts.copy()
        rotated[:, 3] = -rotated[:, 3]
    else:
        u = pole - e4
        u = u / np.linalg.norm(u)
        rotated = pts - 2 * np.outer(pts @ u, u)

    denom = 1.0 - rotated[:, 3]
    denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
    pts_3d = rotated[:, :3] / denom[:, np.newaxis]
    return pts_3d, finite_indices


# ---------------------------------------------------------------------------
# 4-volume of ideal polytope
# ---------------------------------------------------------------------------

def compute_volume_4d(vertices):
    """Volume of ideal 4-polytope with vertices on S^3.
    Uses Delaunay triangulation + Haagerup-Munkholm formula."""
    from scipy.spatial import Delaunay
    pts_3d, _ = stereo_project_S3(vertices, pole_idx=0)
    tri = Delaunay(pts_3d)
    total = 0.0
    for simplex in tri.simplices:
        total += ideal_4simplex_volume_from_tet(pts_3d[simplex])
    return total


# ---------------------------------------------------------------------------
# 3-volume of a single ideal tetrahedron in H^3 (boundary facet)
# ---------------------------------------------------------------------------

def ideal_tet_volume_3d(pts_S3):
    """
    Hyperbolic 3-volume of an ideal tetrahedron in H^3 whose 4 vertices
    lie on S^3 (on a common great S^2).

    Projects the 4 points into their 3D affine span, then stereographically
    projects from one vertex to get a triangle in C, whose angles give
    the volume via the Lobachevsky formula.
    """
    # Project 4 points on S^3 into their 3D affine span
    p = pts_S3 - pts_S3[0]
    U, S, Vt = np.linalg.svd(p[1:], full_matrices=False)
    coords_3d = p @ Vt[:3].T  # (4, 3), first row is origin

    # Find circumsphere
    v = coords_3d
    A = 2 * (v[1:] - v[0])
    b = np.sum(v[1:]**2 - v[0]**2, axis=1)
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return 0.0
    R = np.linalg.norm(v[0] - c)

    # Map to unit sphere
    w = (v - c) / R

    # Stereographic projection from w[0] to R^2
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

    finite = rotated[1:]
    denom = 1.0 - finite[:, 2]
    denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
    z_complex = (finite[:, 0] + 1j * finite[:, 1]) / denom

    # Triangle angles -> Lobachevsky formula
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

    return abs(lobachevsky(a1) + lobachevsky(a2) + lobachevsky(a3))


# ---------------------------------------------------------------------------
# Surface area (3-measure of boundary)
# ---------------------------------------------------------------------------

def compute_surface_area_4d(vertices):
    """3-measure of the boundary of an ideal 4-polytope.
    Each boundary facet (a tetrahedron on S^3) is an ideal tet in H^3."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(vertices)
    facet_vols = []
    for facet in hull.simplices:
        v = vertices[facet]
        vol3 = ideal_tet_volume_3d(v)
        facet_vols.append(vol3)
    return sum(facet_vols), len(hull.simplices), facet_vols


# ===========================================================================
# Checker
# ===========================================================================

def check(answer_file):
    with open(answer_file) as f:
        data = json.load(f)

    vertices = np.array(data['vertices'], dtype=np.float64)
    measure = data['measure']
    target = float(data['target'])
    epsilon = float(data['epsilon'])
    n_expected_verts = data.get('n_vertices', None)
    n_expected_facets = data.get('n_facets', None)

    errors = []
    warnings = []

    print(f"Checking: {measure} = {target} +/- {epsilon}")
    print(f"  Vertices provided: {len(vertices)}")
    print(f"  Expected window: [{target - epsilon:.8f}, {target + epsilon:.8f}]")
    print()

    # --- Check 1: Vertices are in R^4 ---
    if vertices.ndim != 2 or vertices.shape[1] != 4:
        errors.append(f"Vertices must be Nx4 array, got shape {vertices.shape}")
        print("FAIL: " + errors[-1])
        return False, errors

    # --- Check 2: Vertex count ---
    if n_expected_verts is not None and len(vertices) != n_expected_verts:
        errors.append(f"Expected {n_expected_verts} vertices, got {len(vertices)}")
        print(f"FAIL: {errors[-1]}")

    # --- Check 3: All vertices on S^3 ---
    norms = np.linalg.norm(vertices, axis=1)
    max_deviation = np.max(np.abs(norms - 1.0))
    print(f"  Vertex norms: min={norms.min():.12f}, max={norms.max():.12f}")
    print(f"  Max |norm - 1|: {max_deviation:.2e}")
    if max_deviation > 1e-6:
        errors.append(f"Vertices not on S^3: max deviation {max_deviation:.2e} > 1e-6")
        print(f"FAIL: {errors[-1]}")
    else:
        print("  PASS: All vertices on S^3")

    # Normalize to S^3 for computation
    vertices = vertices / norms[:, None]

    # --- Check 4: Convex position (full-dimensional in R^4) ---
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(vertices)
        n_hull_verts = len(set(v for s in hull.simplices for v in s))
        if n_hull_verts < len(vertices):
            warnings.append(f"Only {n_hull_verts}/{len(vertices)} vertices on convex hull")
            print(f"  WARNING: {warnings[-1]}")
        else:
            print(f"  PASS: All {len(vertices)} vertices on convex hull")

        actual_facets = len(hull.simplices)
        print(f"  Convex hull has {actual_facets} facets")
        if n_expected_facets is not None:
            if actual_facets != n_expected_facets:
                errors.append(f"Expected {n_expected_facets} facets, got {actual_facets}")
                print(f"  FAIL: {errors[-1]}")
            else:
                print(f"  PASS: Facet count matches expected {n_expected_facets}")
    except Exception as e:
        errors.append(f"ConvexHull failed: {e}")
        print(f"FAIL: {errors[-1]}")
        return False, errors

    # --- Check 5: Compute the measure ---
    print()
    if measure == 'volume':
        value = compute_volume_4d(vertices)
        print(f"  Computed volume: {value:.10f}")
    elif measure == 'surface_area':
        value, nf, fv = compute_surface_area_4d(vertices)
        print(f"  Computed surface area: {value:.10f}")
        print(f"  Boundary facets: {nf}")
    elif measure == 'ratio':
        vol = compute_volume_4d(vertices)
        area, nf, fv = compute_surface_area_4d(vertices)
        value = vol / area if area > 1e-10 else 0.0
        print(f"  Computed volume: {vol:.10f}")
        print(f"  Computed surface area: {area:.10f}")
        print(f"  Computed V/A ratio: {value:.10f}")
    elif measure == 'euclidean_volume':
        value = hull.volume
        print(f"  Computed Euclidean 4-volume: {value:.10f}")
    elif measure == 'euclidean_area':
        value = hull.area
        print(f"  Computed Euclidean surface area: {value:.10f}")
    elif measure == 'euclidean_ratio':
        value = hull.volume / hull.area if hull.area > 1e-10 else 0.0
        print(f"  Computed Euclidean volume: {hull.volume:.10f}")
        print(f"  Computed Euclidean area: {hull.area:.10f}")
        print(f"  Computed Euclidean V/A ratio: {value:.10f}")
    else:
        errors.append(f"Unknown measure '{measure}'")
        print(f"FAIL: {errors[-1]}")
        return False, errors

    # --- Check 6: Value in window ---
    lo, hi = target - epsilon, target + epsilon
    if lo <= value <= hi:
        print(f"\n  PASS: {measure} = {value:.10f} is in [{lo:.8f}, {hi:.8f}]")
        print(f"  Distance from target: {abs(value - target):.2e} (epsilon = {epsilon:.2e})")
    else:
        if value < lo:
            errors.append(f"{measure} = {value:.10f} < {lo:.8f} (below window)")
        else:
            errors.append(f"{measure} = {value:.10f} > {hi:.8f} (above window)")
        print(f"\n  FAIL: {errors[-1]}")

    # --- Summary ---
    print()
    if errors:
        print(f"RESULT: FAIL ({len(errors)} error(s))")
        for e in errors:
            print(f"  - {e}")
        return False, errors
    else:
        if warnings:
            print(f"RESULT: PASS (with {len(warnings)} warning(s))")
        else:
            print("RESULT: PASS")
        return True, []


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} answer.json")
        sys.exit(2)
    passed, errs = check(sys.argv[1])
    sys.exit(0 if passed else 1)
