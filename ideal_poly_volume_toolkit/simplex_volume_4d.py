"""
Arbitrary-precision volume of an ideal 4-simplex [∞, v₁, v₂, v₃, v₄] in H⁴.

Given four points v₁,...,v₄ ∈ R³ (the boundary tetrahedron T in the upper
half-space model, with the fifth vertex at ∞), compute the hyperbolic 4-volume
using the angle formula:

    Vol = (π/3)(4π - Σ_{i<j} φ_{ij} - Σ_ℓ arccos(d_ℓ/R))

where φ_{ij} are the 6 Euclidean dihedral angles of T, R is the circumradius,
and d_ℓ is the signed distance from the circumcenter to face F_ℓ.

All computation uses mpmath at the requested precision (decimal places).

Usage:
    from ideal_poly_volume_toolkit.simplex_volume_4d import ideal_4simplex_volume

    # Regular ideal 4-simplex (regular tetrahedron boundary)
    v = [[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]]
    vol = ideal_4simplex_volume(v, dps=50)
    print(vol)  # 0.26889566...
"""

import mpmath
from itertools import combinations


def _mp_array(v, dps):
    """Convert a list/array of coordinates to mpmath mpf at given precision."""
    mpmath.mp.dps = dps + 20  # extra guard digits
    return [[mpmath.mpf(str(x)) for x in row] for row in v]


def _dot(a, b):
    """Dot product of two 3-vectors (lists of mpf)."""
    return sum(ai * bi for ai, bi in zip(a, b))


def _sub(a, b):
    """Vector subtraction."""
    return [ai - bi for ai, bi in zip(a, b)]


def _add(a, b):
    """Vector addition."""
    return [ai + bi for ai, bi in zip(a, b)]


def _scale(c, a):
    """Scalar multiplication."""
    return [c * ai for ai in a]


def _cross(a, b):
    """Cross product of two 3-vectors."""
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]


def _norm(a):
    """Euclidean norm."""
    return mpmath.sqrt(_dot(a, a))


def _normalize(a):
    """Return unit vector."""
    n = _norm(a)
    return _scale(1/n, a)


def circumsphere(v):
    """
    Circumsphere of tetrahedron v[0]..v[3] in R³.

    Solves |x - c|² = R² for all four vertices.
    Subtracting the first equation from the rest gives a 3×3 system.

    Returns (c, R) where c is a 3-list of mpf and R is mpf.
    """
    # A[i] = 2*(v[i+1] - v[0]),  b[i] = |v[i+1]|² - |v[0]|²
    A = mpmath.matrix(3, 3)
    b = mpmath.matrix(3, 1)
    v0sq = _dot(v[0], v[0])

    for i in range(3):
        visq = _dot(v[i+1], v[i+1])
        for j in range(3):
            A[i, j] = 2 * (v[i+1][j] - v[0][j])
        b[i] = visq - v0sq

    c_mat = mpmath.lu_solve(A, b)
    c = [c_mat[j] for j in range(3)]
    R = _norm(_sub(v[0], c))
    return c, R


def dihedral_angles(v):
    """
    All 6 Euclidean dihedral angles of tetrahedron v[0]..v[3].

    Returns dict mapping (i,j) -> angle (mpf), where the angle is along
    edge v[i]--v[j], measured inside the tetrahedron.

    Face F_ℓ (opposite v[ℓ]) has outward normal computed from the other
    three vertices, oriented away from v[ℓ].
    """
    # Face vertex indices: face_verts[ℓ] = vertices of face opposite v[ℓ]
    face_verts = [
        (1, 2, 3),
        (0, 2, 3),
        (0, 1, 3),
        (0, 1, 2),
    ]

    # Compute outward normals
    normals = []
    for ell, (a, b, c) in enumerate(face_verts):
        n = _cross(_sub(v[b], v[a]), _sub(v[c], v[a]))
        # Orient outward: should point away from v[ℓ]
        if _dot(n, _sub(v[ell], v[a])) > 0:
            n = _scale(-1, n)
        normals.append(_normalize(n))

    # Edge (i,j) shared by faces F_k, F_ℓ where {k,ℓ} = {0,1,2,3}\{i,j}
    angles = {}
    for i, j in combinations(range(4), 2):
        others = [x for x in range(4) if x != i and x != j]
        k, ell = others
        cos_val = _dot(normals[k], normals[ell])
        # Clamp for numerical safety (should be unnecessary at high precision)
        cos_val = mpmath.mpf(max(-1, min(1, cos_val)))
        # Interior dihedral = π - angle between outward normals
        angles[(i, j)] = mpmath.pi - mpmath.acos(cos_val)

    return angles


def face_distances(v, c):
    """
    Signed distance from circumcenter c to each face of tetrahedron v,
    positive toward the opposite vertex.

    Returns list d[0..3].
    """
    face_verts = [
        (1, 2, 3),
        (0, 2, 3),
        (0, 1, 3),
        (0, 1, 2),
    ]
    d = []
    for ell, (a, b, cc_idx) in enumerate(face_verts):
        n = _cross(_sub(v[b], v[a]), _sub(v[cc_idx], v[a]))
        n_hat = _normalize(n)
        # Sign: positive if c is on same side as v[ℓ]
        sign_test = _dot(n_hat, _sub(v[ell], v[a]))
        if sign_test < 0:
            n_hat = _scale(-1, n_hat)
        d.append(_dot(n_hat, _sub(c, v[a])))
    return d


def ideal_4simplex_volume(vertices, dps=50):
    """
    Volume of ideal 4-simplex [∞, v₁, v₂, v₃, v₄] in H⁴.

    Parameters
    ----------
    vertices : list of 4 points in R³
        Each point is [x, y, z].  These form the boundary tetrahedron T;
        the fifth ideal vertex is at infinity.
    dps : int
        Decimal places of precision (default 50).

    Returns
    -------
    vol : mpmath.mpf
        The hyperbolic 4-volume, computed at the requested precision.

    Notes
    -----
    The formula is:
        Vol = (π/3)(4π - Σ_{i<j} φ_{ij} - Σ_ℓ arccos(d_ℓ/R))

    where φ_{ij} are the 6 Euclidean dihedral angles of T,
    R is the circumradius, d_ℓ the signed distance from the
    circumcenter to face F_ℓ (positive toward v_ℓ).

    Invariant under Euclidean similarities of R³.
    """
    mpmath.mp.dps = dps + 20  # guard digits
    v = _mp_array(vertices, dps)

    c, R = circumsphere(v)
    phi = dihedral_angles(v)
    d = face_distances(v, c)

    angle_sum = sum(phi.values())
    face_angle_sum = sum(mpmath.acos(d[ell] / R) for ell in range(4))

    vol = (mpmath.pi / 3) * (4 * mpmath.pi - angle_sum - face_angle_sum)

    mpmath.mp.dps = dps
    return +vol  # round to requested precision


def ideal_4simplex_angles(vertices, dps=50):
    """
    All 10 dihedral angles of the ideal 4-simplex [∞, v₁, v₂, v₃, v₄].

    Returns
    -------
    dict with keys:
        'euclidean': dict (i,j) -> φ_{ij}  (6 angles, along edges of T)
        'hemisphere': list of 4 angles arccos(d_ℓ/R)
        'all': list of all 10 angles
        'angle_sum': sum of all 10 angles
        'volume': the volume
    """
    mpmath.mp.dps = dps + 20
    v = _mp_array(vertices, dps)

    c, R = circumsphere(v)
    phi = dihedral_angles(v)
    d = face_distances(v, c)

    hemi = [mpmath.acos(d[ell] / R) for ell in range(4)]
    all_angles = list(phi.values()) + hemi
    angle_sum = sum(all_angles)

    vol = (mpmath.pi / 3) * (4 * mpmath.pi - angle_sum)

    mpmath.mp.dps = dps
    return {
        'euclidean': {k: +v for k, v in phi.items()},
        'hemisphere': [+a for a in hemi],
        'all': [+a for a in all_angles],
        'angle_sum': +angle_sum,
        'volume': +vol,
    }


# ============================================================================
# Convenience: regular ideal 4-simplex
# ============================================================================

def regular_ideal_4simplex_volume(dps=50):
    """
    Volume of the regular ideal 4-simplex to arbitrary precision.

    The boundary tetrahedron is a regular Euclidean tetrahedron.
    All 10 dihedral angles equal arccos(1/3).

    Vol = (π/3)(4π - 10·arccos(1/3))
    """
    mpmath.mp.dps = dps + 20
    vol = (mpmath.pi / 3) * (4 * mpmath.pi - 10 * mpmath.acos(mpmath.mpf(1) / 3))
    mpmath.mp.dps = dps
    return +vol


# ============================================================================
# CLI
# ============================================================================

def main():
    import sys

    usage = """Usage: python -m ideal_poly_volume_toolkit.simplex_volume_4d [OPTIONS] x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4

Compute the hyperbolic 4-volume of the ideal 4-simplex [∞, v1, v2, v3, v4]
where vi = (xi, yi, zi) ∈ R³.

Options:
    --dps N      Decimal places of precision (default: 50)
    --angles     Also print all 10 dihedral angles
    --regular    Compute the regular ideal 4-simplex (ignores coordinates)

Examples:
    # Regular tetrahedron
    python -m ideal_poly_volume_toolkit.simplex_volume_4d 1 1 1  1 -1 -1  -1 1 -1  -1 -1 1

    # High precision regular ideal 4-simplex
    python -m ideal_poly_volume_toolkit.simplex_volume_4d --regular --dps 100

    # With angles
    python -m ideal_poly_volume_toolkit.simplex_volume_4d --angles 1 1 1  1 -1 -1  -1 1 -1  -1 -1 1
"""

    args = sys.argv[1:]
    dps = 50
    show_angles = False
    regular = False

    # Parse options
    while args and args[0].startswith('--'):
        if args[0] == '--dps':
            dps = int(args[1])
            args = args[2:]
        elif args[0] == '--angles':
            show_angles = True
            args = args[1:]
        elif args[0] == '--regular':
            regular = True
            args = args[1:]
        elif args[0] == '--help':
            print(usage)
            return
        else:
            print(f"Unknown option: {args[0]}")
            print(usage)
            return

    if regular:
        print(f"Regular ideal 4-simplex volume ({dps} decimal places):")
        vol = regular_ideal_4simplex_volume(dps)
        print(mpmath.nstr(vol, dps))
        if show_angles:
            mpmath.mp.dps = dps
            theta = mpmath.acos(mpmath.mpf(1) / 3)
            print(f"\nAll 10 dihedral angles = arccos(1/3):")
            print(f"  = {mpmath.nstr(theta, dps)} rad")
            print(f"  = {mpmath.nstr(theta * 180 / mpmath.pi, dps)}°")
            print(f"  cos(θ) = 1/3")
        return

    if len(args) != 12:
        print(f"Error: expected 12 coordinates (4 points × 3), got {len(args)}")
        print(usage)
        return

    coords = [args[i] for i in range(12)]  # keep as strings for precision
    vertices = [
        [coords[0], coords[1], coords[2]],
        [coords[3], coords[4], coords[5]],
        [coords[6], coords[7], coords[8]],
        [coords[9], coords[10], coords[11]],
    ]

    if show_angles:
        result = ideal_4simplex_angles(vertices, dps=dps)
        print(f"Ideal 4-simplex volume ({dps} decimal places):")
        print(f"  Vol = {mpmath.nstr(result['volume'], dps)}")
        print(f"\nEuclidean dihedral angles (along edges of T):")
        for (i, j), angle in result['euclidean'].items():
            deg = angle * 180 / mpmath.pi
            cos_val = mpmath.cos(angle)
            print(f"  φ_{{{i},{j}}} = {mpmath.nstr(angle, min(dps, 30))} rad "
                  f"= {mpmath.nstr(deg, min(dps, 30))}°  "
                  f"(cos = {mpmath.nstr(cos_val, min(dps, 30))})")
        print(f"\nHemisphere angles (arccos(d_ℓ/R)):")
        for ell, angle in enumerate(result['hemisphere']):
            deg = angle * 180 / mpmath.pi
            cos_val = mpmath.cos(angle)
            print(f"  θ_F{ell} = {mpmath.nstr(angle, min(dps, 30))} rad "
                  f"= {mpmath.nstr(deg, min(dps, 30))}°  "
                  f"(cos = {mpmath.nstr(cos_val, min(dps, 30))})")
        print(f"\nAngle sum = {mpmath.nstr(result['angle_sum'], min(dps, 30))}")
        print(f"4π        = {mpmath.nstr(4*mpmath.pi, min(dps, 30))}")
        print(f"4π - Σθ   = {mpmath.nstr(4*mpmath.pi - result['angle_sum'], min(dps, 30))}")
    else:
        vol = ideal_4simplex_volume(vertices, dps=dps)
        print(f"Vol = {mpmath.nstr(vol, dps)}")


if __name__ == '__main__':
    main()
