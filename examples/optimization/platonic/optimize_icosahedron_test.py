import argparse, numpy as np, torch, time
from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    triangle_volume_from_points_torch,
    ideal_poly_volume_via_delaunay
)

def random_complex_points(K, rng):
    """Generate K random complex points"""
    # Start with points roughly on unit circle but with some variation
    r = 0.5 + 1.5 * rng.random(K)
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

def compute_regular_icosahedron_volume():
    """Compute the volume of regular ideal icosahedron for comparison"""
    # The regular icosahedron can be constructed with vertices at specific positions
    # One standard construction uses golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # This is an approximation - the exact ideal icosahedron volume
    # would need to be computed or looked up from literature
    # For now, we'll compute it if we have a known configuration
    
    # Placeholder - we'll see what volumes we get and compare
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--series', type=int, default=96)
    ap.add_argument('--print-every', type=int, default=20)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--lr', type=float, default=1.0)
    ap.add_argument('--trials', type=int, default=5, help='Number of trials to run')
    args = ap.parse_args()

    K = 9  # Nine free points (plus 0 and 1 makes 11, plus infinity makes 12)
    
    print("Testing 12-vertex configurations (like icosahedron)")
    print(f"Running {args.trials} optimization trials with 9 free complex points...")
    print("Known volumes:")
    print("  Regular tetrahedron (4 vertices): 1.0149")
    print("  Regular octahedron (6 vertices): 3.6639")
    print("  Regular cube (8 vertices): 5.0747")
    print("  Regular icosahedron (12 vertices): ? (to be determined)")
    print("\n")
    
    results = []
    
    for trial in range(args.trials):
        print(f"\n{'='*50}")
        print(f"Trial {trial+1}/{args.trials} (seed={args.seed + trial})")
        print('='*50)
        
        rng = np.random.default_rng(args.seed + trial)
        
        # Initialize with random complex points
        z_init = random_complex_points(K, rng)
        real_parts = torch.tensor(z_init.real, dtype=torch.float64, device=args.device, requires_grad=True)
        imag_parts = torch.tensor(z_init.imag, dtype=torch.float64, device=args.device, requires_grad=True)
        
        if trial == 0:  # Print initial config for first trial
            print(f"Initial points (besides 0 and 1):")
            for i in range(min(K, 5)):  # Just show first 5
                print(f"  z[{i+2}] = {real_parts[i].item():.4f} + {imag_parts[i].item():.4f}i")
            if K > 5:
                print(f"  ... and {K-5} more points")

        # Use LBFGS optimizer
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
                    print(f'[{it:03d}] volume ~ {history[-1]:.10f}   (tris={idx.shape[0]}, changes={triangulation_changes})')

        t1 = time.time()

        # Final exact evaluation
        with torch.no_grad():
            Zf = build_Z_free(real_parts, imag_parts).detach().cpu().numpy()
            vol_exact = ideal_poly_volume_via_delaunay(Zf, mode='eval_only', dps=250)

        print(f'\nOptimization done in {t1-t0:.2f}s')
        print(f'Final volume: {vol_exact:.10f}')
        print(f'Number of triangles: {len(idx)}')
        print(f'Triangulation changes: {triangulation_changes}')
        
        results.append({
            'seed': args.seed + trial,
            'volume': vol_exact,
            'num_triangles': len(idx),
            'config': Zf.copy()
        })
    
    # Summary
    print(f"\n\n{'='*50}")
    print("SUMMARY OF ALL TRIALS")
    print('='*50)
    
    volumes = [r['volume'] for r in results]
    print(f"Volumes found: {[f'{v:.6f}' for v in volumes]}")
    print(f"Maximum: {max(volumes):.10f}")
    print(f"Minimum: {min(volumes):.10f}")
    print(f"Average: {np.mean(volumes):.10f}")
    
    # Find the best configuration
    best = max(results, key=lambda r: r['volume'])
    print(f"\nBest configuration (seed {best['seed']}):")
    print(f"  Volume: {best['volume']:.10f}")
    print(f"  Number of triangles: {best['num_triangles']}")
    
    # Print first few points of best config
    print(f"\n  Points (first 5 of {len(best['config'])}):")
    for i in range(min(5, len(best['config']))):
        z = best['config'][i]
        print(f"    z[{i}] = {z.real:.4f} + {z.imag:.4f}i")
    
    print("\nConclusion:")
    print("Without knowing the exact regular icosahedron volume, we cannot definitively")
    print("say if it's the maximum. But these volumes can be compared with known values")
    print("or the regular icosahedron can be constructed explicitly to check.")

if __name__ == '__main__':
    main()