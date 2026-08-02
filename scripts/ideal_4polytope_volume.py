#!/usr/bin/env python3
"""
Arbitrary-precision volume of an ideal 4-polytope in hyperbolic 4-space.

An ideal 4-polytope has N vertices on S^3 = ∂H^4.  Its volume is computed
by triangulating into ideal 4-simplices and summing their volumes via the
exact angle formula.

Input modes:
  --R3   Points in R^3 (one vertex implicitly at ∞).
         The Delaunay triangulation of these points gives the simplex
         decomposition directly.

  --S3   Points on S^3 ⊂ R^4 (given as 4-tuples, must have unit norm).
         One point is stereographically projected to ∞; the rest are
         mapped to R^3 and Delaunay-triangulated there.

The volume is invariant under the choice of projection pole (verified
by the code when --check-poles is given).

Requires: mpmath, scipy, numpy

Usage:
    # Points in R^3 (∞ is the implicit extra vertex)
    python ideal_4polytope_volume.py --R3 --dps 50 < points.txt
    python ideal_4polytope_volume.py --R3 --dps 50 --file points.txt

    # Points on S^3
    python ideal_4polytope_volume.py --S3 --dps 50 --file points_s3.txt

    # Inline R^3 points (separated by semicolons)
    python ideal_4polytope_volume.py --R3 --dps 50 "0 0 0; 1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1"

    # With detailed output
    python ideal_4polytope_volume.py --R3 --dps 50 --verbose --file points.txt

    # Self-tests
    python ideal_4polytope_volume.py --test

File format:
    One point per line, coordinates separated by whitespace.
    Lines starting with # are comments.  Blank lines are ignored.
"""

import mpmath
import numpy as np
from scipy.spatial import Delaunay
from itertools import combinations
import sys
import os


# ============================================================================
# Import the simplex volume calculator (same directory)
# ============================================================================

# Inline the core simplex volume routines so this file is self-contained,
# but delegate to the simplex module if available.
try:
    _dir = os.path.dirname(os.path.abspath(__file__))
    if _dir not in sys.path:
        sys.path.insert(0, _dir)
    from ideal_4simplex_volume import (
        ideal_4simplex_volume as _simplex_vol,
        ideal_4simplex_all_angles as _simplex_angles,
    )
except ImportError:
    # Inline minimal implementation
    def _dot(a, b):
        return sum(ai * bi for ai, bi in zip(a, b))
    def _sub(a, b):
        return [ai - bi for ai, bi in zip(a, b)]
    def _scale(c, a):
        return [c * ai for ai in a]
    def _cross(a, b):
        return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    def _norm(a):
        return mpmath.sqrt(_dot(a, a))
    def _normalize(a):
        return _scale(1/_norm(a), a)

    def _circumsphere(v):
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

    def _dihedral_angles(v):
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
            cos_val = max(mpmath.mpf(-1), min(mpmath.mpf(1), _dot(normals[k], normals[ell])))
            angles[(i, j)] = mpmath.pi - mpmath.acos(cos_val)
        return angles

    def _face_distances(v, c):
        face_verts = [(1,2,3), (0,2,3), (0,1,3), (0,1,2)]
        d = []
        for ell, (a, b, cc) in enumerate(face_verts):
            n = _cross(_sub(v[b], v[a]), _sub(v[cc], v[a]))
            n_hat = _normalize(n)
            if _dot(n_hat, _sub(v[ell], v[a])) < 0:
                n_hat = _scale(-1, n_hat)
            d.append(_dot(n_hat, _sub(c, v[a])))
        return d

    def _simplex_vol(vertices, dps=50):
        mpmath.mp.dps = dps + 30
        v = [[mpmath.mpf(str(x)) for x in row] for row in vertices]
        c, R = _circumsphere(v)
        phi = _dihedral_angles(v)
        d = _face_distances(v, c)
        total = sum(phi.values()) + sum(mpmath.acos(d[ell]/R) for ell in range(4))
        vol = (mpmath.pi / 3) * (4 * mpmath.pi - total)
        mpmath.mp.dps = dps
        return +vol

    def _simplex_angles(vertices, dps=50):
        mpmath.mp.dps = dps + 30
        v = [[mpmath.mpf(str(x)) for x in row] for row in vertices]
        c, R = _circumsphere(v)
        phi = _dihedral_angles(v)
        d = _face_distances(v, c)
        hemi = [mpmath.acos(d[ell]/R) for ell in range(4)]
        all_a = list(phi.values()) + hemi
        asum = sum(all_a)
        vol = (mpmath.pi / 3) * (4 * mpmath.pi - asum)
        mpmath.mp.dps = dps
        return {'volume': +vol, 'euclidean': {k: +a for k, a in phi.items()},
                'hemisphere': [+a for a in hemi], 'all': [+a for a in all_a],
                'angle_sum': +asum}


# ============================================================================
# Stereographic projection S^3 -> R^3  (numpy, for finding triangulation)
# ============================================================================

def stereo_S3_to_R3(pts_4d, pole_idx):
    """
    Stereographic projection from S^3 to R^3.

    Sends pts_4d[pole_idx] to ∞; returns the remaining points in R^3.
    Uses Householder reflection to map the pole to e_4 = (0,0,0,1).
    """
    pole = pts_4d[pole_idx]
    mask = np.arange(len(pts_4d)) != pole_idx
    pts = pts_4d[mask]

    e4 = np.array([0., 0., 0., 1.])
    if np.allclose(pole, e4):
        rotated = pts
    elif np.allclose(pole, -e4):
        rotated = pts.copy()
        rotated[:, 3] = -rotated[:, 3]
    else:
        u = pole - e4
        u /= np.linalg.norm(u)
        rotated = pts - 2 * np.outer(pts @ u, u)

    denom = 1.0 - rotated[:, 3]
    denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
    return rotated[:, :3] / denom[:, np.newaxis], np.where(mask)[0]


# ============================================================================
# Main volume computation
# ============================================================================

def ideal_4polytope_volume_R3(pts_3d_str, dps=50, verbose=False):
    """
    Volume of ideal 4-polytope with vertices at pts (in R^3) and ∞.

    Parameters
    ----------
    pts_3d_str : list of list of str
        Points in R^3, coordinates as strings for exact input.
    dps : int
        Decimal places.
    verbose : bool
        Print per-simplex details.

    Returns
    -------
    total_vol : mpmath.mpf
    n_tets : int
    """
    # Use float64 for Delaunay (topology only, not volume)
    pts_float = np.array([[float(x) for x in row] for row in pts_3d_str])
    tri = Delaunay(pts_float)
    tets = tri.simplices

    if verbose:
        print(f"  {len(pts_3d_str)} points in R^3 + ∞ = {len(pts_3d_str)+1} ideal vertices")
        print(f"  Delaunay triangulation: {len(tets)} tetrahedra")

    mpmath.mp.dps = dps
    total = mpmath.mpf(0)

    for k, tet_idx in enumerate(tets):
        v = [pts_3d_str[i] for i in tet_idx]
        vol = _simplex_vol(v, dps=dps)
        total += vol
        if verbose:
            print(f"  tet {k:4d}: indices={list(tet_idx)}, vol={mpmath.nstr(vol, min(dps, 30))}")

    mpmath.mp.dps = dps
    return +total, len(tets)


def ideal_4polytope_volume_S3(pts_4d_str, dps=50, pole_idx=0, verbose=False):
    """
    Volume of ideal 4-polytope with vertices on S^3.

    Parameters
    ----------
    pts_4d_str : list of list of str
        Points on S^3, coordinates as strings.
    dps : int
        Decimal places.
    pole_idx : int
        Which vertex to send to ∞ via stereographic projection.
    verbose : bool
        Print per-simplex details.

    Returns
    -------
    total_vol : mpmath.mpf
    n_tets : int
    """
    N = len(pts_4d_str)
    pts_float = np.array([[float(x) for x in row] for row in pts_4d_str])
    # Normalize to S^3
    norms = np.linalg.norm(pts_float, axis=1, keepdims=True)
    pts_float = pts_float / norms

    # Stereographic projection (float, for triangulation topology)
    pts_3d_float, finite_idx = stereo_S3_to_R3(pts_float, pole_idx)
    tri = Delaunay(pts_3d_float)
    tets = tri.simplices

    if verbose:
        print(f"  {N} points on S^3, projecting from vertex {pole_idx}")
        print(f"  Delaunay triangulation: {len(tets)} tetrahedra")

    # For high-precision volume: re-do stereographic projection in mpmath
    mpmath.mp.dps = dps + 30
    pole = [mpmath.mpf(str(x)) for x in pts_4d_str[pole_idx]]
    pole_norm = mpmath.sqrt(sum(x**2 for x in pole))
    pole = [x / pole_norm for x in pole]

    # Build Householder matrix to send pole -> e4
    e4 = [mpmath.mpf(0), mpmath.mpf(0), mpmath.mpf(0), mpmath.mpf(1)]
    u = [pole[i] - e4[i] for i in range(4)]
    u_norm = mpmath.sqrt(sum(x**2 for x in u))

    if u_norm < mpmath.mpf(10) ** -20:
        # pole ≈ e4, no rotation needed
        def project(p):
            denom = 1 - p[3]
            return [p[i] / denom for i in range(3)]
    elif mpmath.sqrt(sum((pole[i] + e4[i])**2 for i in range(4))) < mpmath.mpf(10)**-20:
        # pole ≈ -e4
        def project(p):
            denom = 1 + p[3]
            return [p[i] / denom for i in range(3)]
    else:
        u = [x / u_norm for x in u]
        def project(p):
            dot_pu = sum(p[i] * u[i] for i in range(4))
            rotated = [p[i] - 2 * dot_pu * u[i] for i in range(4)]
            denom = 1 - rotated[3]
            return [rotated[i] / denom for i in range(3)]

    # Project all finite points at high precision
    finite_mask = [i for i in range(N) if i != pole_idx]
    pts_3d_hp = {}
    for idx in finite_mask:
        p = [mpmath.mpf(str(x)) for x in pts_4d_str[idx]]
        p_norm = mpmath.sqrt(sum(x**2 for x in p))
        p = [x / p_norm for x in p]
        pts_3d_hp[idx] = project(p)

    # Map Delaunay indices (into the N-1 finite array) back to original indices
    total = mpmath.mpf(0)
    for k, tet_idx in enumerate(tets):
        v = [pts_3d_hp[finite_idx[i]] for i in tet_idx]
        # Convert to string lists for the simplex volume function
        v_str = [[str(x) for x in row] for row in v]
        vol = _simplex_vol(v_str, dps=dps)
        total += vol
        if verbose:
            orig_idx = [int(finite_idx[i]) for i in tet_idx]
            print(f"  tet {k:4d}: vertices={orig_idx}, vol={mpmath.nstr(vol, min(dps, 30))}")

    mpmath.mp.dps = dps
    return +total, len(tets)


# ============================================================================
# Input parsing
# ============================================================================

def parse_points(text):
    """Parse points from text.  One point per line, or semicolon-separated."""
    lines = text.replace(';', '\n').strip().split('\n')
    pts = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        coords = line.split()
        pts.append(coords)  # keep as strings
    return pts


def read_points_from_file(path):
    with open(path) as f:
        return parse_points(f.read())


# ============================================================================
# Self-tests
# ============================================================================

def run_tests():
    print("Running self-tests...")
    ok = True

    # Test 1: 16-cell via R^3 mode
    # 7 points: origin + ±e_i (octahedron + center)
    pts = [['0','0','0'], ['1','0','0'], ['-1','0','0'],
           ['0','1','0'], ['0','-1','0'], ['0','0','1'], ['0','0','-1']]
    vol, nt = ideal_4polytope_volume_R3(pts, dps=60, verbose=False)
    mpmath.mp.dps = 60
    # Expected: 16-cell volume
    from ideal_4simplex_volume import regular_ideal_4simplex_volume
    v_tet = (mpmath.pi / 3) * (5*mpmath.pi/2 - 6*mpmath.acos(1/mpmath.sqrt(3)) - mpmath.acos(mpmath.mpf(-1)/3))
    expected = 8 * v_tet
    err = abs(vol - expected)
    status = "PASS" if err < mpmath.mpf(10)**-55 else "FAIL"
    print(f"  [1] 16-cell via R^3 ({nt} tets): err={mpmath.nstr(err, 5)}  [{status}]")
    if status == "FAIL": ok = False

    # Test 2: 16-cell via S^3 mode (should give same volume)
    pts_s3 = [['1','0','0','0'], ['-1','0','0','0'],
              ['0','1','0','0'], ['0','-1','0','0'],
              ['0','0','1','0'], ['0','0','-1','0'],
              ['0','0','0','1'], ['0','0','0','-1']]
    vol_s3, nt_s3 = ideal_4polytope_volume_S3(pts_s3, dps=60, pole_idx=6)
    err2 = abs(vol_s3 - expected)
    status = "PASS" if err2 < mpmath.mpf(10)**-50 else "FAIL"
    print(f"  [2] 16-cell via S^3 ({nt_s3} tets): err={mpmath.nstr(err2, 5)}  [{status}]")
    if status == "FAIL": ok = False

    # Test 3: Pole independence for S^3 mode
    vols = []
    for pole in range(8):
        v, _ = ideal_4polytope_volume_S3(pts_s3, dps=40, pole_idx=pole)
        vols.append(v)
    spread = max(vols) - min(vols)
    status = "PASS" if spread < mpmath.mpf(10)**-35 else "FAIL"
    print(f"  [3] Pole independence (8 poles): spread={mpmath.nstr(spread, 5)}  [{status}]")
    if status == "FAIL": ok = False

    # Test 4: Single simplex (5 points on S^3 = regular 4-simplex)
    # Vertices are computed in float64 then passed as strings, so only
    # ~16 digits of input precision.  Check accordingly.
    import numpy as np
    v5 = np.eye(5) - 1/5
    v4 = v5[:, :4]
    v4 = v4 / np.linalg.norm(v4, axis=1, keepdims=True)
    pts_5 = [[str(x) for x in row] for row in v4]
    vol5, nt5 = ideal_4polytope_volume_S3(pts_5, dps=60, pole_idx=0)
    expected5 = regular_ideal_4simplex_volume(dps=60)
    err5 = abs(vol5 - expected5)
    # Input is float64 (~16 digits), so expect ~30 digits after mpmath amplification
    status = "PASS" if err5 < mpmath.mpf(10)**-30 else "FAIL"
    print(f"  [4] Regular 4-simplex via S^3 ({nt5} tet): err={mpmath.nstr(err5, 5)} (float64 input)  [{status}]")
    if status == "FAIL": ok = False

    print()
    print("All tests passed." if ok else "SOME TESTS FAILED.")
    return ok


# ============================================================================
# CLI
# ============================================================================

def main():
    args = sys.argv[1:]
    dps = 50
    verbose = False
    mode = None  # 'R3' or 'S3'
    filepath = None
    inline = None
    pole_idx = 0
    check_poles = False
    run_test = False

    i = 0
    while i < len(args):
        a = args[i]
        if a == '--dps':
            dps = int(args[i+1]); i += 2
        elif a == '--R3':
            mode = 'R3'; i += 1
        elif a == '--S3':
            mode = 'S3'; i += 1
        elif a == '--file':
            filepath = args[i+1]; i += 2
        elif a == '--pole':
            pole_idx = int(args[i+1]); i += 2
        elif a == '--verbose':
            verbose = True; i += 1
        elif a == '--check-poles':
            check_poles = True; i += 1
        elif a == '--test':
            run_test = True; i += 1
        elif a in ('--help', '-h'):
            print(__doc__); return
        elif not a.startswith('--'):
            inline = ' '.join(args[i:]); break
        else:
            print(f"Unknown option: {a}"); return

    if run_test:
        run_tests()
        return

    if mode is None:
        print("Error: specify --R3 or --S3.")
        print("Use --help for usage.")
        return

    # Read points
    if filepath:
        pts = read_points_from_file(filepath)
    elif inline:
        pts = parse_points(inline)
    elif not sys.stdin.isatty():
        pts = parse_points(sys.stdin.read())
    else:
        print("Error: no input. Use --file, inline args, or pipe to stdin.")
        return

    dim = 3 if mode == 'R3' else 4
    for k, p in enumerate(pts):
        if len(p) != dim:
            print(f"Error: point {k} has {len(p)} coords, expected {dim} for --{mode}.")
            return

    if mode == 'R3':
        vol, nt = ideal_4polytope_volume_R3(pts, dps=dps, verbose=verbose)
        mpmath.mp.dps = dps
        print(f"Volume ({dps} digits, {len(pts)} pts in R^3 + ∞, {nt} simplices):")
        print(f"  {mpmath.nstr(vol, dps)}")

    else:  # S3
        vol, nt = ideal_4polytope_volume_S3(pts, dps=dps, pole_idx=pole_idx, verbose=verbose)
        mpmath.mp.dps = dps
        print(f"Volume ({dps} digits, {len(pts)} pts on S^3, pole={pole_idx}, {nt} simplices):")
        print(f"  {mpmath.nstr(vol, dps)}")

        if check_poles:
            print(f"\nPole independence check:")
            vols = []
            for p in range(len(pts)):
                v, nt_p = ideal_4polytope_volume_S3(pts, dps=dps, pole_idx=p)
                vols.append(v)
                print(f"  pole {p:3d}: vol={mpmath.nstr(v, min(dps, 30))}, {nt_p} tets")
            spread = max(vols) - min(vols)
            print(f"  Spread: {mpmath.nstr(spread, 10)}")


if __name__ == '__main__':
    main()
