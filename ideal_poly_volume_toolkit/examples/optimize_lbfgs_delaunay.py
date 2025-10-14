import argparse, numpy as np, torch, time
from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    triangle_volume_from_points_torch,  # Tensor->Tensor (keeps autograd)
)

def random_angles(K, rng): 
    return 2*np.pi*rng.random(K)

def build_Z(thetas: torch.Tensor) -> torch.Tensor:
    Z = torch.empty(thetas.numel() + 2, dtype=torch.complex128, device=thetas.device)
    Z[0] = 1 + 0j
    Z[1] = 0 + 0j
    Z[2:] = torch.exp(1j * thetas.to(torch.complex128))
    return Z

def torch_sum_volume(Z_t: torch.Tensor, idx, series_terms: int) -> torch.Tensor:
    # Pure-Torch accumulation; returns a real scalar Tensor
    total = torch.zeros((), dtype=torch.float64, device=Z_t.device)
    for (i, j, k) in idx:
        total = total + triangle_volume_from_points_torch(
            Z_t[i], Z_t[j], Z_t[k], series_terms=series_terms
        )
    return total  # real

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--iters', type=int, default=75)
    ap.add_argument('--series', type=int, default=96)
    ap.add_argument('--print-every', type=int, default=5)
    ap.add_argument('--device', type=str, default='cpu')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    K = 3
    thetas = torch.tensor(
        random_angles(K, rng), dtype=torch.float64, device=args.device, requires_grad=True
    )

    opt = torch.optim.LBFGS([thetas], lr=1.0, max_iter=20, line_search_fn='strong_wolfe')

    history = []
    t0 = time.time()

    for it in range(1, args.iters + 1):
        # Rebuild Delaunay OUTSIDE the graph for this outer iteration
        with torch.no_grad():
            Z_np = build_Z(thetas).detach().cpu().numpy()
            idx = delaunay_triangulation_indices(Z_np)

        # Closure used internally by LBFGS's line search
        def closure():
            opt.zero_grad(set_to_none=True)
            Z_t = build_Z(thetas)
            total = torch_sum_volume(Z_t, idx, args.series)
            loss = -total  # maximize volume
            loss.backward()
            return loss

        _ = opt.step(closure)  # do NOT trust the return value for logging

        # ---- recompute AFTER the step for accurate logging ----
        with torch.no_grad():
            Z_post = build_Z(thetas)
            val_post = torch_sum_volume(Z_post, idx, args.series)  # real tensor
            history.append(float(val_post.item()))
            if it % args.print_every == 0 or it in (1, args.iters):
                print(f'[{it:03d}] fast volume ~ {history[-1]:.10f}   (tris={idx.shape[0]})')

    t1 = time.time()

    # Final exact eval (detached)
    with torch.no_grad():
        Zf = build_Z(thetas).detach().cpu().numpy()
        from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay
        vol_exact = ideal_poly_volume_via_delaunay(Zf, mode='eval_only', dps=250)

    print('\n=== Optimization (Delaunay) done ===')
    print(f'iters={args.iters}, time={t1-t0:.2f}s')
    print(f'final fast volume ~ {history[-1]:.12f}')
    print(f'final exact volume  {vol_exact:.12f}')
    print('final angles (rad):', thetas.detach().cpu().numpy())

if __name__ == '__main__':
    main()
