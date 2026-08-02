#!/usr/bin/env python3
"""
Migration validation script.

Run this on a new machine to verify that all key computations
produce correct results. Tests volume accuracy, gradient correctness,
GPU availability, and timing benchmarks.

Usage:
    python scripts/validate_migration.py
"""

import sys
import time
import math
import platform
import numpy as np

def header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")

def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return passed

def main():
    all_pass = True
    sys.path.insert(0, '.')

    header("System info")
    print(f"  Platform: {platform.platform()}")
    print(f"  Architecture: {platform.machine()}")
    print(f"  Python: {sys.version.split()[0]}")

    # ---- NumPy / SciPy ----
    header("NumPy / SciPy")
    import scipy
    print(f"  NumPy {np.__version__}, SciPy {scipy.__version__}")
    all_pass &= check("numpy available", True)

    # ---- PyTorch + CUDA ----
    header("PyTorch + CUDA")
    import torch
    print(f"  PyTorch {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    all_pass &= check("CUDA available", cuda_ok)
    if cuda_ok:
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}")
        print(f"  Memory: {props.total_memory // (1024**3)} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")
        all_pass &= check("GPU memory >= 80GB",
                          props.total_memory > 80 * 1024**3,
                          f"{props.total_memory // (1024**3)} GB")
        # Quick complex128 test
        z = torch.randn(1000, dtype=torch.complex128, device='cuda')
        r = torch.abs(z).sum()
        all_pass &= check("complex128 on GPU", r.item() > 0)

    # ---- Core toolkit imports ----
    header("Toolkit imports")
    try:
        from ideal_poly_volume_toolkit.murakami_torch import (
            murakami_volume_batched, ideal_h6_polytope_volume
        )
        all_pass &= check("murakami_torch", True)
    except Exception as e:
        all_pass &= check("murakami_torch", False, str(e))

    try:
        from ideal_poly_volume_toolkit.li2_torch import li2
        all_pass &= check("li2_torch", True)
    except Exception as e:
        all_pass &= check("li2_torch", False, str(e))

    try:
        from ideal_poly_volume_toolkit.h5_volume_torch import (
            schlafli_surrogate_vectorized, lobachevsky_torch
        )
        all_pass &= check("h5_volume_torch", True)
    except Exception as e:
        all_pass &= check("h5_volume_torch", False, str(e))

    try:
        from ideal_poly_volume_toolkit.geometry_4d import ideal_polytope_volume_4d
        all_pass &= check("geometry_4d", True)
    except Exception as e:
        all_pass &= check("geometry_4d", False, str(e))

    # ---- Li2 accuracy ----
    header("Li2 accuracy (pure torch vs scipy)")
    from ideal_poly_volume_toolkit.li2_torch import li2
    from scipy.special import spence

    z_test = np.array([0.1, 0.5, 0.9, 0.99, -1.0, -0.5])
    ref = spence(1.0 - z_test)
    result = li2(torch.tensor(z_test, dtype=torch.float64)).real.numpy()
    max_err = np.max(np.abs(result - ref))
    all_pass &= check("Li2 real values", max_err < 1e-12, f"max err {max_err:.2e}")

    z_complex = np.exp(1j * np.linspace(0.1, 3.0, 20))
    ref_c = spence(1.0 - z_complex)
    result_c = li2(torch.tensor(z_complex, dtype=torch.complex128)).numpy()
    max_err_c = np.max(np.abs(result_c - ref_c))
    all_pass &= check("Li2 complex values", max_err_c < 1e-12, f"max err {max_err_c:.2e}")

    # ---- Lobachevsky function ----
    header("Lobachevsky function")
    from ideal_poly_volume_toolkit.h5_volume_torch import lobachevsky_torch
    from scipy.integrate import quad
    theta_test = torch.tensor([math.pi/6, math.pi/3, math.pi/4])
    lob = lobachevsky_torch(theta_test).numpy()
    for i, t in enumerate([math.pi/6, math.pi/3, math.pi/4]):
        ref_val, _ = quad(lambda s: -math.log(abs(2*math.sin(s)))
                          if abs(math.sin(s)) > 1e-15 else 0, 0, t)
        err = abs(lob[i] - ref_val)
        all_pass &= check(f"Lobachevsky(pi/{[6,3,4][i]})", err < 1e-6, f"err {err:.2e}")

    # ---- H^4 ideal volume ----
    header("H^4 ideal polytope volume")
    rng = np.random.default_rng(42)
    pts4 = rng.standard_normal((8, 4))
    pts4 /= np.linalg.norm(pts4, axis=1, keepdims=True)
    vol4, ntets = ideal_polytope_volume_4d(pts4)
    all_pass &= check("H^4 volume computes", vol4 > 0 and np.isfinite(vol4),
                       f"vol={vol4:.6f}, {ntets} tets")

    # Regular ideal 4-simplex: Vol = (pi/3)(4pi - 10*arccos(1/3))
    e5 = np.eye(5); c = e5.mean(0); v = e5 - c
    u, s, _ = np.linalg.svd(v, full_matrices=False)
    simplex = (u[:, :4] * s[:4]); simplex /= np.linalg.norm(simplex, axis=1, keepdims=True)
    vol_simplex, _ = ideal_polytope_volume_4d(simplex)
    ref_simplex = (math.pi/3) * (4*math.pi - 10*math.acos(1/3))
    err_simplex = abs(vol_simplex - ref_simplex) / ref_simplex
    all_pass &= check("Regular ideal 4-simplex", err_simplex < 0.01,
                       f"vol={vol_simplex:.6f} ref={ref_simplex:.6f} err={err_simplex:.2e}")

    # ---- H^6 vectorized volume + gradients ----
    header("H^6 vectorized volume (CPU)")
    torch.set_default_dtype(torch.float64)

    from ideal_poly_volume_toolkit.murakami_torch import ideal_h6_polytope_volume
    from ideal_poly_volume_toolkit.li2_torch import li2 as li2_pure
    from scipy.spatial import Delaunay

    def stereo_tri(pts):
        N, d = pts.shape; pole = pts[0]; others = np.delete(pts, 0, axis=0)
        basis = []
        for i in range(d):
            e = np.zeros(d); e[i] = 1.0; e -= np.dot(e, pole)*pole
            for b in basis: e -= np.dot(e, b)*b
            n = np.linalg.norm(e)
            if n > 1e-10: basis.append(e/n)
            if len(basis) == d-1: break
        basis = np.array(basis)
        proj = [basis @ ((p - np.dot(p, pole)*pole)/max(1-np.dot(p, pole), 1e-14))
                for p in others]
        tri = Delaunay(np.array(proj))
        return [[0]+[s+1 for s in simp] for simp in tri.simplices]

    pts6 = rng.standard_normal((10, 6))
    pts6 /= np.linalg.norm(pts6, axis=1, keepdims=True)
    simps6 = stereo_tri(pts6)

    pts_t = torch.tensor(pts6, dtype=torch.float64, requires_grad=True)
    simps_t = torch.tensor(simps6, dtype=torch.long)
    vol6 = ideal_h6_polytope_volume(pts_t, simps_t, li2_fn=li2_pure)
    vol6.backward()
    grad_norm = pts_t.grad.norm().item()
    all_pass &= check("H^6 volume (CPU)", vol6.item() > 0, f"vol={vol6.item():.8f}")
    all_pass &= check("H^6 gradient (CPU)", grad_norm > 0, f"|grad|={grad_norm:.6f}")

    # ---- GPU benchmark ----
    if cuda_ok:
        header("GPU benchmark: H^6 N=24")
        pts24 = rng.standard_normal((24, 6))
        pts24 /= np.linalg.norm(pts24, axis=1, keepdims=True)
        simps24 = stereo_tri(pts24)
        simps24_t = torch.tensor(simps24, dtype=torch.long, device='cuda')

        # Warmup
        pts_g = torch.tensor(pts24, dtype=torch.float64, device='cuda', requires_grad=True)
        v = ideal_h6_polytope_volume(pts_g, simps24_t, li2_fn=li2_pure)
        v.backward()
        torch.cuda.synchronize()

        # Timed
        times = []
        for _ in range(10):
            pts_g = torch.tensor(pts24, dtype=torch.float64, device='cuda', requires_grad=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            v = ideal_h6_polytope_volume(pts_g, simps24_t, li2_fn=li2_pure)
            v.backward()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        t_med = np.median(times) * 1000

        all_pass &= check("GPU H^6 fwd+bwd", t_med < 500,
                          f"{t_med:.1f}ms (GH200 baseline: ~14ms)")
        print(f"  {len(simps24)} simplices, {len(simps24)*35*8} Li2 evaluations")

    # ---- CMA-ES ----
    header("CMA-ES")
    try:
        import cma
        from scipy.spatial import ConvexHull
        pts_cma = rng.standard_normal((10, 4))
        pts_cma /= np.linalg.norm(pts_cma, axis=1, keepdims=True)
        def neg_vol(x):
            p = x.reshape(10, 4); p /= np.linalg.norm(p, axis=1, keepdims=True)
            try: return -ConvexHull(p).volume
            except: return 0.0
        es = cma.CMAEvolutionStrategy(pts_cma.ravel(), 0.3,
                                       {'maxfevals': 500, 'verbose': -9, 'seed': 42})
        while not es.stop():
            sols = es.ask(); fits = [neg_vol(x) for x in sols]; es.tell(sols, fits)
        vol_cma = -es.result.fbest
        all_pass &= check("CMA-ES optimization", vol_cma > 0.5, f"vol={vol_cma:.6f}")
    except Exception as e:
        all_pass &= check("CMA-ES", False, str(e))

    # ---- nauty / pynauty ----
    header("pynauty (graph automorphisms)")
    try:
        from pynauty import Graph, autgrp
        g = Graph(4)
        g.connect_vertex(0, [1, 2, 3])
        g.connect_vertex(1, [2, 3])
        g.connect_vertex(2, [3])
        gens, order, _, _, _ = autgrp(g)
        all_pass &= check("pynauty", order == 24, f"|Aut(K4)| = {order}")
    except Exception as e:
        all_pass &= check("pynauty", False, str(e))

    # ---- RL (optional) ----
    header("RL stack (optional)")
    try:
        import gymnasium
        import stable_baselines3
        print(f"  gymnasium {gymnasium.__version__}, SB3 {stable_baselines3.__version__}")
        all_pass &= check("gymnasium + SB3", True)
    except Exception as e:
        print(f"  [SKIP] RL stack not installed: {e}")

    # ---- Summary ----
    header("SUMMARY")
    if all_pass:
        print("  All checks PASSED. Migration validated.")
    else:
        print("  Some checks FAILED. Review output above.")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
