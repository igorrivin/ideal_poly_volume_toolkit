import numpy as np
import torch
from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    triangle_volume_from_points_torch,
    ideal_poly_volume_via_delaunay
)

def random_complex_points(K, rng):
    """Generate K random complex points"""
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

def run_optimization_trial(seed, max_iters=100, K=5):
    """Run one optimization trial with given seed"""
    rng = np.random.default_rng(seed)
    
    # Initialize with random complex points
    z_init = random_complex_points(K, rng)
    real_parts = torch.tensor(z_init.real, dtype=torch.float64, requires_grad=True)
    imag_parts = torch.tensor(z_init.imag, dtype=torch.float64, requires_grad=True)
    
    # Use LBFGS optimizer
    opt = torch.optim.LBFGS([real_parts, imag_parts], lr=1.0, max_iter=20, line_search_fn='strong_wolfe')
    
    prev_triangulation = None
    triangulation_changes = 0
    
    for it in range(1, max_iters + 1):
        # Rebuild Delaunay triangulation
        with torch.no_grad():
            Z_np = build_Z_free(real_parts, imag_parts).detach().cpu().numpy()
            idx = delaunay_triangulation_indices(Z_np)
            
            if prev_triangulation is not None:
                if idx.shape != prev_triangulation.shape or not np.array_equal(idx, prev_triangulation):
                    triangulation_changes += 1
            prev_triangulation = idx.copy()
        
        # Closure for LBFGS
        def closure():
            opt.zero_grad(set_to_none=True)
            Z_t = build_Z_free(real_parts, imag_parts)
            total = torch_sum_volume(Z_t, idx, 96)
            loss = -total
            loss.backward()
            torch.nn.utils.clip_grad_norm_([real_parts, imag_parts], max_norm=10.0)
            return loss
        
        opt.step(closure)
    
    # Final evaluation
    with torch.no_grad():
        Zf = build_Z_free(real_parts, imag_parts).detach().cpu().numpy()
        vol_fast = torch_sum_volume(build_Z_free(real_parts, imag_parts), idx, 96).item()
        vol_exact = ideal_poly_volume_via_delaunay(Zf, mode='eval_only', dps=250)
    
    return {
        'seed': seed,
        'volume_fast': vol_fast,
        'volume_exact': vol_exact,
        'final_config': Zf.copy(),
        'num_triangles': len(idx),
        'triangulation_changes': triangulation_changes
    }

# Run multiple trials
print("Running multiple optimization trials with 5 free points...")
print("Expected volumes:")
print("  Regular tetrahedron: ~1.0149")
print("  Regular cube: ~5.0747")
print("\n")

num_trials = 10
results = []

for seed in range(num_trials):
    print(f"Trial {seed+1}/{num_trials} (seed={seed})...", end='', flush=True)
    result = run_optimization_trial(seed)
    results.append(result)
    print(f" Volume: {result['volume_exact']:.6f}")

# Analyze results
volumes = [r['volume_exact'] for r in results]
print(f"\n=== Summary of {num_trials} trials ===")
print(f"Minimum volume: {min(volumes):.6f}")
print(f"Maximum volume: {max(volumes):.6f}") 
print(f"Average volume: {np.mean(volumes):.6f}")
print(f"Std deviation:  {np.std(volumes):.6f}")

# Count how many exceed the cube
cube_volume = 5.0747
num_exceed_cube = sum(1 for v in volumes if v > cube_volume)
print(f"\nTrials exceeding cube volume ({cube_volume:.4f}): {num_exceed_cube}/{num_trials}")

# Group by similar volumes
print("\nVolume distribution:")
volume_groups = {}
for r in results:
    v = r['volume_exact']
    found = False
    for group_v in volume_groups:
        if abs(v - group_v) < 0.01:  # Group within 0.01
            volume_groups[group_v].append(r)
            found = True
            break
    if not found:
        volume_groups[v] = [r]

for v in sorted(volume_groups.keys(), reverse=True):
    group = volume_groups[v]
    print(f"  Volume ~{v:.4f}: {len(group)} trial(s) (seeds: {[r['seed'] for r in group]})")

# Check if any special values appear
phi = (1 + np.sqrt(5)) / 2
print(f"\nChecking for special values (φ = {phi:.6f}):")
for v in sorted(volume_groups.keys(), reverse=True):
    ratio = v / 1.0149  # Ratio to tetrahedron
    print(f"  {v:.4f} = {ratio:.4f} × tetrahedron")

# Save the best configuration
best = max(results, key=lambda r: r['volume_exact'])
print(f"\nBest configuration found:")
print(f"  Seed: {best['seed']}")
print(f"  Volume: {best['volume_exact']:.6f}")
print(f"  Points:")
for i, z in enumerate(best['final_config']):
    print(f"    z[{i}] = {z.real:.4f} + {z.imag:.4f}i")