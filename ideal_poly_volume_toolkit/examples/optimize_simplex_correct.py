import argparse, numpy as np, torch, time
from ideal_poly_volume_toolkit.geometry import triangle_volume_from_points_torch

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
    grad_history = []
    t0 = time.time()

    for it in range(1, args.iters + 1):
        def closure():
            opt.zero_grad(set_to_none=True)
            # Triangle vertices: 0, 1, exp(i*theta)
            z0 = torch.tensor(0+0j, dtype=torch.complex128, device=theta.device)
            z1 = torch.tensor(1+0j, dtype=torch.complex128, device=theta.device)
            z2 = torch.exp(1j * theta.to(torch.complex128))
            
            volume = triangle_volume_from_points_torch(z0, z1, z2, series_terms=args.series)
            loss = -volume  # maximize volume
            loss.backward()
            return loss

        opt.step(closure)

        # Log progress
        with torch.no_grad():
            z0 = torch.tensor(0+0j, dtype=torch.complex128, device=theta.device)
            z1 = torch.tensor(1+0j, dtype=torch.complex128, device=theta.device)
            z2 = torch.exp(1j * theta.to(torch.complex128))
            vol = triangle_volume_from_points_torch(z0, z1, z2, series_terms=args.series)
            
            history.append(vol.item())
            grad_history.append(theta.grad.item() if theta.grad is not None else 0)
            
            if it % args.print_every == 0 or it in (1, args.iters):
                print(f'[{it:03d}] volume = {history[-1]:.10f}, theta = {theta.item():.6f} rad ({theta.item() * 180/np.pi:.2f}°), |grad| = {abs(grad_history[-1]):.8f}')

    t1 = time.time()

    # Final exact evaluation
    with torch.no_grad():
        from ideal_poly_volume_toolkit.geometry import triangle_volume_from_points
        z0, z1, z2 = 0+0j, 1+0j, np.exp(1j * theta.item())
        vol_exact = triangle_volume_from_points(z0, z1, z2, mode='exact', dps=250)

    print('\n=== Simplex optimization done ===')
    print(f'iters={args.iters}, time={t1-t0:.2f}s')
    print(f'initial volume ~ {vol_exact if it == 1 else history[0]:.10f}')
    print(f'final fast volume ~ {history[-1]:.10f}')
    print(f'final exact volume  {vol_exact:.12f}')
    print(f'final theta: {theta.item():.6f} radians ({theta.item() * 180/np.pi:.2f} degrees)')
    
    # The theoretical maximum for a triangle with vertices at 0, 1, exp(i*theta) occurs at theta = π/2
    print(f'\nExpected optimal theta: {np.pi/2:.6f} radians (90.00 degrees)')
    print(f'Distance from optimum: {abs(theta.item() - np.pi/2):.6f} radians ({abs(theta.item() - np.pi/2) * 180/np.pi:.2f} degrees)')

if __name__ == '__main__':
    main()