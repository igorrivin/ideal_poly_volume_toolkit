#!/usr/bin/env python3
"""Quick scan: try random normals, solve ideal constraint, report volumes."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, Delaunay
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ideal_poly_volume_toolkit.geometry_4d import (
    ideal_4simplex_volume_from_tet, stereo_project_S3
)


def try_ideal(normals):
    """Try to find support numbers making all vertices ideal."""
    f = len(normals)

    for hp in [0.3, 0.1, 0.5, 0.2, 0.7]:
        w = normals / hp
        try:
            hull = ConvexHull(w)
            if np.any(hull.equations[:, -1] > 1e-10):
                continue
        except Exception:
            continue

        # Build vertex linear maps
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
            except np.linalg.LinAlgError:
                ok = False
                break
        if not ok:
            continue

        nv = len(Am)

        # Solve |v_j|^2 = 1
        def residuals(h):
            return np.array([np.dot(A @ h, A @ h) - 1.0 for A in Am])

        def jacobian(h):
            return np.array([2.0 * (A.T @ (A @ h)) for A in Am])

        result = least_squares(
            residuals, np.full(f, hp), jac=jacobian,
            method='lm', ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=3000
        )

        if result.cost < 1e-20:
            verts = np.array([A @ result.x for A in Am])
            pts_3d, _ = stereo_project_S3(verts, pole_idx=0)
            try:
                tri = Delaunay(pts_3d)
                vol = sum(
                    ideal_4simplex_volume_from_tet(pts_3d[t])
                    for t in tri.simplices
                )
                return vol, nv, result.x
            except Exception:
                continue

    return None


def main():
    f = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    rng = np.random.default_rng(42)
    results = []

    for trial in range(n_trials):
        normals = rng.standard_normal((f, 4))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        r = try_ideal(normals)
        if r is not None:
            vol, nv, h = r
            results.append((vol, nv))
            print(f"  trial {trial:3d}: vol={vol:10.4f}, {nv:3d} vertices")

    print(f"\nf = {f} normals: {len(results)}/{n_trials} succeeded")
    if results:
        results.sort(reverse=True)
        vols = [r[0] for r in results]
        print(f"  Best:  {vols[0]:.6f} ({results[0][1]} vertices)")
        print(f"  Top 5: {[f'{v:.2f}' for v, _ in results[:5]]}")
        print(f"  Mean:  {np.mean(vols):.4f}")
        print(f"  Vertex counts: {sorted(set(r[1] for r in results))}")


if __name__ == '__main__':
    main()
