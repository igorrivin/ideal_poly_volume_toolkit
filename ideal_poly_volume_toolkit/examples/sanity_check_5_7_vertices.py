import argparse, numpy as np, torch
from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    triangle_volume_from_points_torch,
    ideal_poly_volume_via_delaunay,
    lift_to_sphere_with_inf
)
from scipy.spatial import ConvexHull

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

def optimize_configuration(K, seed, max_iters=150):
    """Optimize K free points"""
    rng = np.random.default_rng(seed)
    
    # Initialize with random complex points
    z_init = random_complex_points(K, rng)
    real_parts = torch.tensor(z_init.real, dtype=torch.float64, requires_grad=True)
    imag_parts = torch.tensor(z_init.imag, dtype=torch.float64, requires_grad=True)
    
    # Use LBFGS optimizer
    opt = torch.optim.LBFGS([real_parts, imag_parts], lr=1.0, max_iter=20, line_search_fn='strong_wolfe')
    
    for it in range(max_iters):
        # Rebuild Delaunay triangulation
        with torch.no_grad():
            Z_np = build_Z_free(real_parts, imag_parts).detach().cpu().numpy()
            idx = delaunay_triangulation_indices(Z_np)
        
        # Closure for LBFGS
        def closure():
            opt.zero_grad(set_to_none=True)
            Z_t = build_Z_free(real_parts, imag_parts)
            total = torch_sum_volume(Z_t, idx, 96)
            loss = -total  # maximize volume
            loss.backward()
            torch.nn.utils.clip_grad_norm_([real_parts, imag_parts], max_norm=10.0)
            return loss
        
        opt.step(closure)
    
    # Final evaluation
    with torch.no_grad():
        Zf = build_Z_free(real_parts, imag_parts).detach().cpu().numpy()
        vol_exact = ideal_poly_volume_via_delaunay(Zf, mode='eval_only', dps=250)
    
    return {
        'volume': vol_exact,
        'config': Zf,
        'num_triangles': len(idx)
    }

def analyze_configuration(config, num_vertices):
    """Analyze the combinatorial structure"""
    # Project to sphere
    z_with_inf = np.append(config['config'], [np.inf])
    sphere_points = lift_to_sphere_with_inf(z_with_inf)
    
    try:
        hull = ConvexHull(sphere_points)
        
        # Count vertex degrees
        vertex_degrees = {}
        for i in range(num_vertices):
            degree = sum(1 for face in hull.simplices if i in face)
            vertex_degrees[i] = degree
        
        # Degree distribution
        degree_counts = {}
        for d in vertex_degrees.values():
            degree_counts[d] = degree_counts.get(d, 0) + 1
            
        return {
            'num_faces': len(hull.simplices),
            'degree_distribution': degree_counts,
            'vertex_degrees': vertex_degrees
        }
    except:
        return None

# Known volumes for comparison
tetrahedron_vol = 1.01494160640965

print("Sanity Check: Testing 5 and 7 vertices")
print("="*60)

# Test 5 vertices (should be triangular bipyramid)
print("\n5 VERTICES (Triangular Bipyramid)")
print("-"*40)
print("Expected: Regular triangular bipyramid")
print("Expected volume: 2 × tetrahedron = 2.0299")
print("Expected structure: 6 triangular faces, all vertices degree 4")
print()

results_5 = []
for seed in range(5):
    result = optimize_configuration(K=2, seed=seed)  # 2 free + 2 fixed + infinity = 5
    results_5.append(result)
    print(f"Seed {seed}: Volume = {result['volume']:.6f}")

best_5 = max(results_5, key=lambda x: x['volume'])
print(f"\nBest volume: {best_5['volume']:.6f}")
print(f"Ratio to expected: {best_5['volume'] / (2*tetrahedron_vol):.4f}")

# Analyze structure
struct_5 = analyze_configuration(best_5, 5)
if struct_5:
    print(f"Structure: {struct_5['num_faces']} faces")
    print(f"Vertex degrees: {struct_5['degree_distribution']}")

# Test 7 vertices
print("\n\n7 VERTICES")
print("-"*40)
print("Expected: Some type of bipyramid")
print()

results_7 = []
for seed in range(10):
    result = optimize_configuration(K=4, seed=seed)  # 4 free + 2 fixed + infinity = 7
    results_7.append(result)
    print(f"Seed {seed}: Volume = {result['volume']:.6f}")

# Group by similar volumes
volume_groups = {}
for r in results_7:
    v = r['volume']
    found = False
    for group_v in volume_groups:
        if abs(v - group_v) < 0.01:
            volume_groups[group_v].append(r)
            found = True
            break
    if not found:
        volume_groups[v] = [r]

print(f"\nDistinct volume levels found:")
for v in sorted(volume_groups.keys(), reverse=True):
    print(f"  Volume ~{v:.4f}: {len(volume_groups[v])} configurations")

# Analyze best configuration
best_7 = max(results_7, key=lambda x: x['volume'])
print(f"\nBest configuration:")
print(f"Volume: {best_7['volume']:.6f}")

struct_7 = analyze_configuration(best_7, 7)
if struct_7:
    print(f"Structure: {struct_7['num_faces']} faces")
    print(f"Vertex degrees: {struct_7['degree_distribution']}")
    
    # Check if it's a bipyramid
    if struct_7['degree_distribution'].get(3, 0) == 2:
        print("This appears to be a bipyramid (2 vertices of degree 3 are the apex vertices)")

# Compare all results
print("\n\nSUMMARY")
print("="*60)
print("Vertex count | Expected           | Found")
print("-------------|-------------------|------------------")
print(f"4            | Tetrahedron       | {tetrahedron_vol:.4f}")
print(f"5            | Triangular bipyr. | {best_5['volume']:.4f} (ratio: {best_5['volume']/(2*tetrahedron_vol):.3f})")
print(f"6            | Octahedron        | 3.6639")
print(f"7            | Bipyramid         | {best_7['volume']:.4f}")
print(f"8            | Cube              | 5.0747 → 6.4885")
print(f"12           | Icosahedron       | 13.0259 → 13.0321")