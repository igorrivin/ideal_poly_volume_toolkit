#!/usr/bin/env python3
"""
Arbitrary-precision volume of an ideal 4-simplex in hyperbolic 4-space.

Given four points v1,...,v4 in R^3 (forming a Euclidean tetrahedron T),
compute the hyperbolic volume of the ideal 4-simplex [∞, v1, v2, v3, v4]
in the upper half-space model of H^4.

The key insight is that the volume is given by an EXACT angle formula
(Hopf / Poincaré–Schläfli), NOT by numerical integration:

    Vol = (π/3)(4π - Σ_{i<j} φ_{ij} - Σ_ℓ arccos(d_ℓ/R))

where φ_{ij} are the 6 Euclidean dihedral angles of T, R is the
circumradius, and d_ℓ is the signed distance from the circumcenter
to face F_ℓ (positive toward v_ℓ).  This reduces the computation to
evaluating arccos of algebraic numbers, which mpmath handles to any
requested precision.

By contrast, the equivalent integral representation
    Vol = (1/3) ∫_T d³x / (R² - |x-c|²)^{3/2}
is limited to ~16 digits by floating-point quadrature.

Requires: mpmath  (pip install mpmath)

Usage:
    python ideal_4simplex_volume.py [--dps N] [--angles] x1 y1 z1  x2 y2 z2  x3 y3 z3  x4 y4 z4
    python ideal_4simplex_volume.py --regular [--dps N] [--angles]
    python ideal_4simplex_volume.py --test

Examples:
    # Regular ideal 4-simplex at 100 decimal places
    python ideal_4simplex_volume.py --regular --dps 100

    # Regular tetrahedron boundary, 50 digits, show all angles
    python ideal_4simplex_volume.py --angles --dps 50  1 1 1  1 -1 -1  -1 1 -1  -1 -1 1

    # Run self-tests
    python ideal_4simplex_volume.py --test
"""

import mpmath
from itertools import combinations
import sys


# ============================================================================
# Linear algebra in mpmath (3D vectors)
# ============================================================================

def _dot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))

def _sub(a, b):
    return [ai - bi for ai, bi in zip(a, b)]

def _scale(c, a):
    return [c * ai for ai in a]

def _cross(a, b):
    return [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]

def _norm(a):
    return mpmath.sqrt(_dot(a, a))

def _normalize(a):
    return _scale(1 / _norm(a), a)


# ============================================================================
# Tetrahedron geometry
# ============================================================================

def circumsphere(v):
    """Circumcenter and circumradius of tetrahedron v[0..3] in R^3."""
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
    Six Euclidean dihedral angles of tetrahedron v[0..3].
    Returns dict (i,j) -> angle along edge v_i--v_j.
    """
    face_verts = [(1,2,3), (0,2,3), (0,1,3), (0,1,2)]

    normals = []
    for ell, (a, b, c) in enumerate(face_verts):
        n = _cross(_sub(v[b], v[a]), _sub(v[c], v[a]))
        if _dot(n, _sub(v[ell], v[a])) > 0:
            n = _scale(-1, n)
        normals.append(_normalize(n))

    angles = {}
    for i, j in combinations(range(4), 2):
        others = [x for x in range(4) if x != i and x != j]
        k, ell = others
        cos_val = _dot(normals[k], normals[ell])
        cos_val = max(mpmath.mpf(-1), min(mpmath.mpf(1), cos_val))
        angles[(i, j)] = mpmath.pi - mpmath.acos(cos_val)
    return angles


def face_distances(v, c):
    """Signed distance from circumcenter c to each face, positive toward opposite vertex."""
    face_verts = [(1,2,3), (0,2,3), (0,1,3), (0,1,2)]
    d = []
    for ell, (a, b, cc) in enumerate(face_verts):
        n = _cross(_sub(v[b], v[a]), _sub(v[cc], v[a]))
        n_hat = _normalize(n)
        if _dot(n_hat, _sub(v[ell], v[a])) < 0:
            n_hat = _scale(-1, n_hat)
        d.append(_dot(n_hat, _sub(c, v[a])))
    return d


# ============================================================================
# Volume computation
# ============================================================================

def ideal_4simplex_volume(vertices, dps=50):
    """
    Compute Vol_4([∞, v1, v2, v3, v4]) in H^4 to `dps` decimal places.

    Parameters
    ----------
    vertices : list of 4 triples
        Coordinates of v1,...,v4 in R^3.  Can be strings for exact input.
    dps : int
        Decimal digits of precision.

    Returns
    -------
    mpmath.mpf
    """
    mpmath.mp.dps = dps + 30
    v = [[mpmath.mpf(str(x)) for x in row] for row in vertices]

    c, R = circumsphere(v)
    phi = dihedral_angles(v)
    d = face_distances(v, c)

    total_angle = sum(phi.values()) + sum(mpmath.acos(d[ell] / R) for ell in range(4))

    vol = (mpmath.pi / 3) * (4 * mpmath.pi - total_angle)

    mpmath.mp.dps = dps
    return +vol


def ideal_4simplex_all_angles(vertices, dps=50):
    """
    Compute volume and all 10 dihedral angles of the ideal 4-simplex.

    Returns dict with keys: volume, euclidean (6 angles), hemisphere (4 angles),
    angle_sum, all (list of 10).
    """
    mpmath.mp.dps = dps + 30
    v = [[mpmath.mpf(str(x)) for x in row] for row in vertices]

    c, R = circumsphere(v)
    phi = dihedral_angles(v)
    d = face_distances(v, c)

    hemi = [mpmath.acos(d[ell] / R) for ell in range(4)]
    all_a = list(phi.values()) + hemi
    asum = sum(all_a)
    vol = (mpmath.pi / 3) * (4 * mpmath.pi - asum)

    mpmath.mp.dps = dps
    return {
        'volume': +vol,
        'euclidean': {k: +a for k, a in phi.items()},
        'hemisphere': [+a for a in hemi],
        'all': [+a for a in all_a],
        'angle_sum': +asum,
    }


def regular_ideal_4simplex_volume(dps=50):
    """Vol = (π/3)(4π - 10·arccos(1/3)).  All 10 angles = arccos(1/3)."""
    mpmath.mp.dps = dps + 30
    vol = (mpmath.pi / 3) * (4 * mpmath.pi - 10 * mpmath.acos(mpmath.mpf(1) / 3))
    mpmath.mp.dps = dps
    return +vol


# ============================================================================
# Self-tests
# ============================================================================

def run_tests():
    """Verify the implementation against known values."""
    print("Running self-tests...")
    ok = True

    # Test 1: Regular ideal 4-simplex via formula vs via coordinates
    dps = 60
    v_reg = [[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]]
    vol_coords = ideal_4simplex_volume(v_reg, dps=dps)
    vol_exact = regular_ideal_4simplex_volume(dps=dps)

    mpmath.mp.dps = dps
    err1 = abs(vol_coords - vol_exact)
    status = "PASS" if err1 < mpmath.mpf(10) ** (-(dps - 5)) else "FAIL"
    print(f"  [1] Regular tet coords vs formula ({dps} dps): err={mpmath.nstr(err1, 5)}  [{status}]")
    if status == "FAIL":
        ok = False

    # Test 2: Scaling invariance (volume unchanged under similarity)
    v_scaled = [[2*x for x in row] for row in v_reg]
    vol_scaled = ideal_4simplex_volume(v_scaled, dps=dps)
    err2 = abs(vol_scaled - vol_exact)
    status = "PASS" if err2 < mpmath.mpf(10) ** (-(dps - 5)) else "FAIL"
    print(f"  [2] Scaling invariance (×2): err={mpmath.nstr(err2, 5)}  [{status}]")
    if status == "FAIL":
        ok = False

    # Test 3: Translation invariance
    v_trans = [[x + 100 for x in row] for row in v_reg]
    vol_trans = ideal_4simplex_volume(v_trans, dps=dps)
    err3 = abs(vol_trans - vol_exact)
    status = "PASS" if err3 < mpmath.mpf(10) ** (-(dps - 5)) else "FAIL"
    print(f"  [3] Translation invariance (+100): err={mpmath.nstr(err3, 5)}  [{status}]")
    if status == "FAIL":
        ok = False

    # Test 4: 16-cell tet (known exact angles)
    v_16 = [[0,0,0], [1,0,0], [0,1,0], [0,0,1]]
    result = ideal_4simplex_all_angles(v_16, dps=60)
    # Should have 3 angles at π/2, 6 at arccos(1/√3), 1 at arccos(-1/3)
    mpmath.mp.dps = 60
    expected_vol = (mpmath.pi / 3) * (5*mpmath.pi/2 - 6*mpmath.acos(1/mpmath.sqrt(3)) - mpmath.acos(mpmath.mpf(-1)/3))
    err4 = abs(result['volume'] - expected_vol)
    status = "PASS" if err4 < mpmath.mpf(10) ** -55 else "FAIL"
    print(f"  [4] 16-cell tet volume: err={mpmath.nstr(err4, 5)}  [{status}]")
    if status == "FAIL":
        ok = False

    # Test 5: High precision (200 digits)
    dps = 200
    vol200 = regular_ideal_4simplex_volume(dps=dps)
    # Check it starts with the known digits
    s = mpmath.nstr(vol200, 50)
    expected_start = "0.26889566016931122546929486722724566452499443719303"
    status = "PASS" if s.startswith(expected_start) else "FAIL"
    print(f"  [5] 200-digit regular volume starts correctly: [{status}]")
    if status == "FAIL":
        print(f"      got:      {s}")
        print(f"      expected: {expected_start}...")
        ok = False

    # Test 6: Degenerate detection (flat tetrahedron should give ~0 volume)
    v_flat = [[0,0,0], [1,0,0], [0,1,0], [1,1,0]]
    try:
        vol_flat = ideal_4simplex_volume(v_flat, dps=30)
        # Flat tet has circumsphere at infinity; this should fail or give nonsense
        print(f"  [6] Flat tet: vol={mpmath.nstr(vol_flat, 10)} (degenerate, may be nonsensical) [INFO]")
    except Exception as e:
        print(f"  [6] Flat tet raises {type(e).__name__}: {e} [INFO]")

    print()
    if ok:
        print("All tests passed.")
    else:
        print("SOME TESTS FAILED.")
    return ok


# ============================================================================
# CLI
# ============================================================================

def main():
    args = sys.argv[1:]
    dps = 50
    show_angles = False
    regular = False
    run_test = False

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
        elif args[0] == '--test':
            run_test = True
            args = args[1:]
        elif args[0] in ('--help', '-h'):
            print(__doc__)
            return
        else:
            print(f"Unknown option: {args[0]}")
            print("Use --help for usage.")
            return

    if run_test:
        run_tests()
        return

    if regular:
        vol = regular_ideal_4simplex_volume(dps)
        print(f"Regular ideal 4-simplex volume ({dps} digits):")
        print(mpmath.nstr(vol, dps))
        if show_angles:
            mpmath.mp.dps = dps
            theta = mpmath.acos(mpmath.mpf(1) / 3)
            print(f"\nAll 10 dihedral angles = arccos(1/3):")
            print(f"  {mpmath.nstr(theta, dps)} rad")
            print(f"  {mpmath.nstr(theta * 180 / mpmath.pi, dps)}°")
        return

    if len(args) != 12:
        print(f"Error: need 12 coordinates (4 points × 3), got {len(args)}.")
        print("Use --help for usage.")
        return

    vertices = [[args[3*i], args[3*i+1], args[3*i+2]] for i in range(4)]

    if show_angles:
        result = ideal_4simplex_all_angles(vertices, dps=dps)
        mpmath.mp.dps = dps
        print(f"Volume ({dps} digits):")
        print(f"  {mpmath.nstr(result['volume'], dps)}")
        print(f"\nEuclidean dihedral angles (6, along edges of T):")
        for (i, j), angle in result['euclidean'].items():
            cos_val = mpmath.cos(angle)
            print(f"  edge ({i},{j}): {mpmath.nstr(angle, min(dps, 40))} rad  "
                  f"cos = {mpmath.nstr(cos_val, min(dps, 40))}")
        print(f"\nHemisphere angles (4, arccos(d_ℓ/R) for faces):")
        for ell, angle in enumerate(result['hemisphere']):
            cos_val = mpmath.cos(angle)
            print(f"  face {ell}: {mpmath.nstr(angle, min(dps, 40))} rad  "
                  f"cos = {mpmath.nstr(cos_val, min(dps, 40))}")
        print(f"\nAngle sum:  {mpmath.nstr(result['angle_sum'], min(dps, 40))}")
        print(f"4π:         {mpmath.nstr(4*mpmath.pi, min(dps, 40))}")
        print(f"Deficit:    {mpmath.nstr(4*mpmath.pi - result['angle_sum'], min(dps, 40))}")
    else:
        vol = ideal_4simplex_volume(vertices, dps=dps)
        print(mpmath.nstr(vol, dps))


if __name__ == '__main__':
    main()
