import argparse, numpy as np, torch, time
from ideal_poly_volume_toolkit.geometry import triangle_volume_from_points_torch

def build_Z_simplex(theta: torch.Tensor) -> torch.Tensor:
    """Build points for a single triangle: 0, 1, infinity, and one movable point"""
    Z = torch.empty(4, dtype=torch.complex128, device=theta.device)
    Z[0] = 0 + 0j  # origin
    Z[1] = 1 + 0j  # unit point
    Z[2] = 1e10 + 0j  # effectively infinity (large number)
    Z[3] = torch.exp(1j * theta)  # movable point on unit circle
    return Z

def simplex_volume(theta: torch.Tensor, series_terms: int) -> torch.Tensor:
    """Compute volume of the single triangle formed by 0, 1, infinity, and exp(i*theta)"""
    Z = build_Z_simplex(theta)
    # Triangle vertices: 0, 1, exp(i*theta) (skipping infinity)
    return triangle_volume_from_points_torch(Z[0], Z[1], Z[3], series_terms=series_terms)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init-angle', type=float, default=0.5, help='Initial angle in radians')
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--series', type=int, default=96)
    ap.add_argument('--print-every', type=int, default=5)
    ap.add_argument('--device', type=str, default='cpu')
    args = ap.parse_args()

    # Single angle parameter
    theta = torch.tensor(args.init_angle, dtype=torch.float64, device=args.device, requires_grad=True)
    
    print(f"Initial theta: {theta.item():.6f} radians ({theta.item() * 180/np.pi:.2f} degrees)")

    # Use LBFGS optimizer
    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=20, line_search_fn='strong_wolfe')

    history = []
    t0 = time.time()

    for it in range(1, args.iters + 1):
        def closure():
            opt.zero_grad(set_to_none=True)
            volume = simplex_volume(theta, args.series)
            loss = -volume  # maximize volume
            loss.backward()
            return loss

        opt.step(closure)

        # Log progress
        with torch.no_grad():
            vol = simplex_volume(theta, args.series)
            history.append(vol.item())
            if it % args.print_every == 0 or it in (1, args.iters):
                grad_val = theta.grad.item() if theta.grad is not None else 0
                print(f'[{it:03d}] volume = {history[-1]:.10f}, theta = {theta.item():.6f} rad ({theta.item() * 180/np.pi:.2f}°), grad = {grad_val:.6f}')

    t1 = time.time()

    # Final exact evaluation
    with torch.no_grad():
        from ideal_poly_volume_toolkit.geometry import triangle_volume_from_points
        z0, z1, z3 = 0+0j, 1+0j, np.exp(1j * theta.item())
        vol_exact = triangle_volume_from_points(z0, z1, z3, mode='exact', dps=250)

    print('\n=== Simplex optimization done ===')
    print(f'iters={args.iters}, time={t1-t0:.2f}s')
    print(f'initial volume ~ {history[0] if history else args.init_angle:.10f}')
    print(f'final fast volume ~ {history[-1]:.10f}')
    print(f'final exact volume  {vol_exact:.12f}')
    print(f'final theta: {theta.item():.6f} radians ({theta.item() * 180/np.pi:.2f} degrees)')
    
    # The theoretical maximum for a triangle with vertices at 0, 1, exp(i*theta) occurs at theta = π/2
    print(f'\nExpected optimal theta: {np.pi/2:.6f} radians (90.00 degrees)')
    print(f'Distance from optimum: {abs(theta.item() - np.pi/2):.6f} radians')

if __name__ == '__main__':
    main()