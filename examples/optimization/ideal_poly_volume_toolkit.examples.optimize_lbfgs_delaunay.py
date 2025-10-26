import argparse, numpy as np, torch, time
from ideal_poly_volume_toolkit.geometry import delaunay_triangulation_indices, triangle_volume_from_points_torch

def random_angles(K, rng): return 2*np.pi*rng.random(K)

def build_Z(thetas):
    Z = torch.empty(thetas.numel()+2, dtype=torch.complex128, device=thetas.device)
    Z[0] = 1+0j; Z[1] = 0+0j
    Z[2:] = torch.exp(1j*thetas.to(torch.complex128))
    return Z

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
    thetas = torch.tensor(random_angles(K, rng), dtype=torch.float64, device=args.device, requires_grad=True)
    opt = torch.optim.LBFGS([thetas], lr=1.0, max_iter=20, line_search_fn='strong_wolfe')

    # Precompute triangulation connectivity outside the graph each outer iter
    def closure_once(idx):
        def _c():
            opt.zero_grad(set_to_none=True)
            Z_t = build_Z(thetas)  # complex torch, depends on real angles
            total = torch.zeros((), dtype=torch.complex128, device=thetas.device)
            for (i,j,k) in idx:
                total = total + triangle_volume_from_points_torch(Z_t[i], Z_t[j], Z_t[k],
                                                                  series_terms=args.series)
            loss = -total.real  # maximize volume
            loss.backward()
            return loss
        return _c

    hist = []; t0 = time.time()
    for it in range(1, args.iters+1):
        # Rebuild Delaunay triangles using current (detached) positions
        with torch.no_grad():
            Z_np = build_Z(thetas).detach().cpu().numpy()
            idx = delaunay_triangulation_indices(Z_np)

        loss = opt.step(closure_once(idx))
        with torch.no_grad(): hist.append(float(-loss.item()))
        if it % args.print_every == 0 or it in (1, args.iters):
            print(f'[{it:03d}] fast volume ~ {hist[-1]:.10f}   (tris={idx.shape[0]})')

    t1 = time.time()

    # Final exact eval
    with torch.no_grad():
        Zf = build_Z(thetas).detach().cpu().numpy()
        from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay as eval_del
        vol_exact = eval_del(Zf, mode='eval_only', dps=250)

    print('\\n=== Optimization (Delaunay) done ===')
    print(f'iters={args.iters}, time={t1-t0:.2f}s')
    print(f'final fast volume ~ {hist[-1]:.12f}')
    print(f'final exact volume  {vol_exact:.12f}')
    print('final angles (rad):', thetas.detach().cpu().numpy())

if __name__ == '__main__':
    main()
