import argparse, numpy as np, torch, time
from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
)

def random_angles(K, rng): 
    return 2*np.pi*rng.random(K)

def build_Z_real(thetas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Z as separate real and imaginary parts to maintain gradient flow"""
    real_parts = torch.zeros(thetas.numel() + 2, dtype=torch.float64, device=thetas.device)
    imag_parts = torch.zeros(thetas.numel() + 2, dtype=torch.float64, device=thetas.device)
    
    # Z[0] = 1 + 0j
    real_parts[0] = 1.0
    imag_parts[0] = 0.0
    
    # Z[1] = 0 + 0j
    real_parts[1] = 0.0
    imag_parts[1] = 0.0
    
    # Z[2:] = exp(i * theta) = cos(theta) + i*sin(theta)
    real_parts[2:] = torch.cos(thetas)
    imag_parts[2:] = torch.sin(thetas)
    
    return real_parts, imag_parts

def _angles_for_triangle_real(x1, y1, x2, y2, x3, y3):
    """Compute triangle angles using real coordinates"""
    def angle_at(ax, ay, bx, by, cx, cy):
        # Vector from a to b
        ux = bx - ax
        uy = by - ay
        # Vector from a to c  
        vx = cx - ax
        vy = cy - ay
        # Dot product
        dot = ux * vx + uy * vy
        # Cross product magnitude
        cross = torch.abs(ux * vy - uy * vx)
        return torch.atan2(cross, dot)
    
    a1 = angle_at(x1, y1, x2, y2, x3, y3)
    a2 = angle_at(x2, y2, x3, y3, x1, y1)
    a3 = angle_at(x3, y3, x1, y1, x2, y2)
    return a1, a2, a3

def _lob_value_series_torch(theta: torch.Tensor, n: int = 64) -> torch.Tensor:
    k = torch.arange(1, n+1, device=theta.device, dtype=theta.dtype)
    return 0.5 * torch.sum(torch.sin(2*k*theta)/(k*k), dim=0)

class _LobFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, n: int = 64):
        ctx.save_for_backward(theta)
        return _lob_value_series_torch(theta, n)
    @staticmethod
    def backward(ctx, gout):
        (theta,) = ctx.saved_tensors
        return gout * torch.log(2.0*torch.sin(theta)), None

def lob_fast(theta: torch.Tensor, n: int = 64) -> torch.Tensor:
    return _LobFn.apply(theta, n)

def triangle_volume_from_real_coords(x1, y1, x2, y2, x3, y3, series_terms=64):
    """Compute triangle volume from real coordinates"""
    a1, a2, a3 = _angles_for_triangle_real(x1, y1, x2, y2, x3, y3)
    return lob_fast(a1, series_terms) + lob_fast(a2, series_terms) + lob_fast(a3, series_terms)

def torch_sum_volume_real(real_parts: torch.Tensor, imag_parts: torch.Tensor, 
                         idx, series_terms: int) -> torch.Tensor:
    """Accumulate volume using real coordinates"""
    total = torch.zeros((), dtype=torch.float64, device=real_parts.device)
    for (i, j, k) in idx:
        total = total + triangle_volume_from_real_coords(
            real_parts[i], imag_parts[i],
            real_parts[j], imag_parts[j], 
            real_parts[k], imag_parts[k],
            series_terms=series_terms
        )
    return total

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
            real_np, imag_np = build_Z_real(thetas)
            Z_np = real_np.detach().cpu().numpy() + 1j * imag_np.detach().cpu().numpy()
            idx = delaunay_triangulation_indices(Z_np)

        # Closure used internally by LBFGS's line search
        def closure():
            opt.zero_grad(set_to_none=True)
            real_parts, imag_parts = build_Z_real(thetas)
            total = torch_sum_volume_real(real_parts, imag_parts, idx, args.series)
            loss = -total  # maximize volume
            loss.backward()
            return loss

        _ = opt.step(closure)  # do NOT trust the return value for logging

        # ---- recompute AFTER the step for accurate logging ----
        with torch.no_grad():
            real_post, imag_post = build_Z_real(thetas)
            val_post = torch_sum_volume_real(real_post, imag_post, idx, args.series)
            history.append(float(val_post.item()))
            if it % args.print_every == 0 or it in (1, args.iters):
                print(f'[{it:03d}] fast volume ~ {history[-1]:.10f}   (tris={idx.shape[0]})')

    t1 = time.time()

    # Final exact eval (detached)
    with torch.no_grad():
        real_f, imag_f = build_Z_real(thetas)
        Zf = real_f.detach().cpu().numpy() + 1j * imag_f.detach().cpu().numpy()
        from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay
        vol_exact = ideal_poly_volume_via_delaunay(Zf, mode='eval_only', dps=250)

    print('\n=== Optimization (Delaunay) done ===')
    print(f'iters={args.iters}, time={t1-t0:.2f}s')
    print(f'final fast volume ~ {history[-1]:.12f}')
    print(f'final exact volume  {vol_exact:.12f}')
    print('final angles (rad):', thetas.detach().cpu().numpy())

if __name__ == '__main__':
    main()