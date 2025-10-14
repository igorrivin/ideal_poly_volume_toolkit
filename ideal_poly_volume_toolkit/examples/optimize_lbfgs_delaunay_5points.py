import argparse, numpy as np, torch, time
from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    triangle_volume_from_points_torch,
)

def random_complex_points(K, rng):
    """Generate K random complex points"""
    # Start with points roughly on unit circle but with some variation
    r = 0.5 + 1.0 * rng.random(K)
    theta = 2 * np.pi * rng.random(K)
    return r * np.exp(1j * theta)

def build_Z_free(real_parts: torch.Tensor, imag_parts: torch.Tensor) -> torch.Tensor:
    """Build complex array from real and imaginary parts"""
    Z = torch.empty(real_parts.numel() + 2, dtype=torch.complex128, device=real_parts.device)
    Z[0] = 0 + 0j  # Fixed at origin
    Z[1] = 1 + 0j  # Fixed at 1
    Z[2:] = torch.complex(real_parts, imag_parts)
    return Z

def torch_sum_volume(Z_t: torch.Tensor, idx, series_terms: int) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float64, device=Z_t.device)
    for (i, j, k) in idx:
        total = total + triangle_volume_from_points_torch(
            Z_t[i], Z_t[j], Z_t[k], series_terms=series_terms
        )
    return total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--series', type=int, default=96)
    ap.add_argument('--print-every', type=int, default=20)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--lr', type=float, default=1.0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    K = 5  # Five free points
    
    # Initialize with random complex points
    z_init = random_complex_points(K, rng)
    real_parts = torch.tensor(z_init.real, dtype=torch.float64, device=args.device, requires_grad=True)
    imag_parts = torch.tensor(z_init.imag, dtype=torch.float64, device=args.device, requires_grad=True)
    
    print(f"Initial points (besides 0 and 1):")
    for i in range(K):
        print(f"  z[{i+2}] = {real_parts[i].item():.4f} + {imag_parts[i].item():.4f}i")

    # Use LBFGS optimizer on both real and imaginary parts
    opt = torch.optim.LBFGS([real_parts, imag_parts], lr=args.lr, max_iter=20, line_search_fn='strong_wolfe')

    history = []
    triangulation_changes = 0
    prev_triangulation = None
    t0 = time.time()

    for it in range(1, args.iters + 1):
        # Rebuild Delaunay triangulation
        with torch.no_grad():
            Z_np = build_Z_free(real_parts, imag_parts).detach().cpu().numpy()
            idx = delaunay_triangulation_indices(Z_np)
            
            # Check if triangulation changed
            if prev_triangulation is not None:
                if idx.shape != prev_triangulation.shape or not np.array_equal(idx, prev_triangulation):
                    triangulation_changes += 1
                    if it <= 10 or it % args.print_every == 0:
                        print(f"  [Triangulation changed at iteration {it}]")
            prev_triangulation = idx.copy()

        # Closure for LBFGS
        def closure():
            opt.zero_grad(set_to_none=True)
            Z_t = build_Z_free(real_parts, imag_parts)
            total = torch_sum_volume(Z_t, idx, args.series)
            loss = -total  # maximize volume
            loss.backward()
            # Clip gradients to prevent instability
            torch.nn.utils.clip_grad_norm_([real_parts, imag_parts], max_norm=10.0)
            return loss

        opt.step(closure)

        # Log progress
        with torch.no_grad():
            Z_post = build_Z_free(real_parts, imag_parts)
            val_post = torch_sum_volume(Z_post, idx, args.series)
            history.append(float(val_post.item()))
            
            if it % args.print_every == 0 or it in (1, args.iters):
                print(f'\n[{it:03d}] volume ~ {history[-1]:.10f}   (tris={idx.shape[0]})')
                # Print current positions
                for i in range(K):
                    z = complex(real_parts[i].item(), imag_parts[i].item())
                    r, theta = abs(z), np.angle(z) * 180/np.pi
                    print(f'      z[{i+2}] = {z.real:+.4f} {z.imag:+.4f}i  (r={r:.4f}, θ={theta:+.1f}°)')

    t1 = time.time()

    # Final exact evaluation
    with torch.no_grad():
        Zf = build_Z_free(real_parts, imag_parts).detach().cpu().numpy()
        from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay
        vol_exact = ideal_poly_volume_via_delaunay(Zf, mode='eval_only', dps=250)

    print('\n=== Optimization (5 free points) done ===')
    print(f'iters={args.iters}, time={t1-t0:.2f}s')
    print(f'triangulation changes: {triangulation_changes}')
    print(f'initial volume ~ {history[0] if history else 0:.10f}')
    print(f'final fast volume ~ {history[-1]:.10f}')
    print(f'final exact volume  {vol_exact:.12f}')
    
    print('\nFinal configuration:')
    print('  z[0] = 0.0000 + 0.0000i (fixed)')
    print('  z[1] = 1.0000 + 0.0000i (fixed)')
    for i in range(K):
        z = complex(real_parts[i].item(), imag_parts[i].item())
        r, theta = abs(z), np.angle(z) * 180/np.pi
        print(f'  z[{i+2}] = {z.real:+.4f} {z.imag:+.4f}i  (r={r:.4f}, θ={theta:+.1f}°)')
    
    print('\nExpected volumes:')
    print('  Regular tetrahedron: ~1.0149')
    print('  Regular cube: ~5.0747 (5 × tetrahedron)')
    print(f'  Our result: {vol_exact:.4f}')
    
    # Check distances between points
    print('\nPairwise distances:')
    Z_final = build_Z_free(real_parts, imag_parts).detach().numpy()
    for i in range(len(Z_final)):
        for j in range(i+1, len(Z_final)):
            dist = abs(Z_final[i] - Z_final[j])
            if dist > 0.01:  # Skip very small distances
                print(f'  |z[{i}] - z[{j}]| = {dist:.4f}')

if __name__ == '__main__':
    main()