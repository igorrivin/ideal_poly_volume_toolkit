#!/usr/bin/env python3
"""Cross-validate polytope volumes: stereo+angle formula vs 4D Delaunay + Schläfli."""

import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from scipy.optimize import least_squares
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ideal_poly_volume_toolkit.geometry_4d import (
    ideal_4simplex_volume_from_tet, stereo_project_S3, regular_ideal_4simplex_volume
)
from ideal_poly_volume_toolkit.hyperbolic_4_simplex_volume import hyperbolic_4_simplex_volume
import mpmath as mp


def make_ideal_polytope(f, seed=42):
    """Generate a random ideal polytope with f facet normals."""
    rng = np.random.default_rng(seed)
    normals = rng.standard_normal((f, 4))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    for hp in [0.3, 0.1, 0.5, 0.2, 0.7]:
        w = normals / hp
        try:
            hull = ConvexHull(w)
            if np.any(hull.equations[:, -1] > 1e-10):
                continue
        except Exception:
            continue

        Am = []
        ok = True
        for s in hull.simplices:
            U = normals[list(s)]
            try:
                Ui = np.linalg.inv(U)
                A = np.zeros((4, f))
                for k, idx in enumerate(s):
                    A[:, idx] = Ui[:, k]
                Am.append(A)
            except Exception:
                ok = False
                break
        if not ok:
            continue

        result = least_squares(
            lambda h: np.array([np.dot(A @ h, A @ h) - 1.0 for A in Am]),
            np.full(f, hp),
            jac=lambda h: np.array([2.0 * (A.T @ (A @ h)) for A in Am]),
            method='lm', ftol=1e-15, xtol=1e-15, max_nfev=3000
        )
        if result.cost < 1e-20:
            verts = np.array([A @ result.x for A in Am])
            return verts

    return None


def main():
    mp.mp.dps = 30

    # Test 1: Regular ideal 4-simplex
    print("=== Test 1: Regular ideal 4-simplex ===")
    v = np.array([
        [1, 1, 1, -1/np.sqrt(5)],
        [1, -1, -1, -1/np.sqrt(5)],
        [-1, 1, -1, -1/np.sqrt(5)],
        [-1, -1, 1, -1/np.sqrt(5)],
        [0, 0, 0, 4/np.sqrt(5)],
    ], dtype=float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)

    vol_schlafli = float(hyperbolic_4_simplex_volume(v.tolist(), dps=30, snap_ideal_tol=1e-10))
    pts_3d, _ = stereo_project_S3(v, pole_idx=0)
    vol_angle = ideal_4simplex_volume_from_tet(pts_3d)

    print(f"  Schlaefli formula:  {vol_schlafli:.12f}")
    print(f"  Stereo+angle:      {vol_angle:.12f}")
    print(f"  Difference:        {abs(vol_schlafli - vol_angle):.2e}")

    # Test 2: Small polytope (f=8, fewer vertices = fewer simplices = faster)
    print("\n=== Test 2: f=8 ideal polytope ===")
    verts = make_ideal_polytope(8, seed=18)  # seed 18 is known to work
    if verts is None:
        # Try more seeds
        for s in range(100):
            verts = make_ideal_polytope(8, seed=s)
            if verts is not None:
                break

    if verts is None:
        print("  Could not construct ideal polytope")
        return

    r = np.linalg.norm(verts, axis=1)
    print(f"  {len(verts)} vertices, max|r-1|={np.max(np.abs(r-1)):.2e}")

    # Method A: stereo project + 3D Delaunay + angle formula
    pts_3d, _ = stereo_project_S3(verts, pole_idx=0)
    tri3 = Delaunay(pts_3d)
    vol_A = sum(ideal_4simplex_volume_from_tet(pts_3d[t]) for t in tri3.simplices)
    print(f"  Method A (stereo+3D Delaunay): {vol_A:.10f}  ({len(tri3.simplices)} simplices)")

    # Method B: use same 3D Delaunay, reconstruct 5-vertex ideal simplices.
    # Each 3D Delaunay tet has indices into pts_3d = verts[1:] (pole=0 removed).
    # The ideal 4-simplex = verts[0] + verts[idx+1 for idx in tet].
    pole_idx = 0
    print(f"  Method B: {len(tri3.simplices)} simplices via Schlaefli (all-ideal)...")
    vol_B = 0.0
    n_ok = 0
    n_fail = 0
    for i, t in enumerate(tri3.simplices):
        # Map 3D Delaunay indices back to original vertex indices
        vert_4d = [verts[pole_idx].tolist()] + [verts[idx + 1].tolist() for idx in t]
        try:
            v4 = float(hyperbolic_4_simplex_volume(
                vert_4d, dps=30, snap_ideal_tol=1e-10))
            vol_B += v4
            n_ok += 1
        except Exception as e:
            n_fail += 1
            if n_fail <= 3:
                print(f"    Simplex {i} failed: {e}")

    print(f"  Method B (4D Delaunay + Schlaefli): {vol_B:.10f}  ({n_ok} ok, {n_fail} failed)")
    print(f"\n  Agreement: diff={abs(vol_A - vol_B):.2e}, "
          f"rel={abs(vol_A - vol_B) / max(vol_A, 1e-30):.2e}")


if __name__ == '__main__':
    main()
