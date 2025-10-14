import numpy as np
from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    ideal_poly_volume_via_delaunay,
    lift_to_sphere_with_inf
)
from scipy.spatial import ConvexHull

# First, let's understand the combinatorics of the regular icosahedron
print("Regular Icosahedron Combinatorics:")
print("="*50)
print("Vertices: 12")
print("Edges: 30") 
print("Faces: 20 triangular faces")
print("Vertex degree: 5 (each vertex connects to 5 others)")
print("Face structure: All triangles")
print()

# Now let's re-run the optimization to capture the maximal configurations
# and analyze their structure
def run_single_optimization(seed):
    """Run optimization and return configuration with analysis"""
    import torch
    
    rng = np.random.default_rng(seed)
    K = 9  # 9 free points
    
    # Initialize
    r = 0.5 + 1.5 * rng.random(K)
    theta = 2 * np.pi * rng.random(K)
    z_init = r * np.exp(1j * theta)
    
    real_parts = torch.tensor(z_init.real, dtype=torch.float64, requires_grad=True)
    imag_parts = torch.tensor(z_init.imag, dtype=torch.float64, requires_grad=True)
    
    # Build Z
    def build_Z_free(real_parts, imag_parts):
        Z = torch.empty(real_parts.numel() + 2, dtype=torch.complex128)
        Z[0] = 0 + 0j
        Z[1] = 1 + 0j
        Z[2:] = torch.complex(real_parts, imag_parts)
        return Z
    
    # Optimize
    opt = torch.optim.LBFGS([real_parts, imag_parts], lr=1.0, max_iter=20, line_search_fn='strong_wolfe')
    
    for it in range(100):
        with torch.no_grad():
            Z_np = build_Z_free(real_parts, imag_parts).detach().cpu().numpy()
            idx = delaunay_triangulation_indices(Z_np)
        
        def closure():
            opt.zero_grad(set_to_none=True)
            Z_t = build_Z_free(real_parts, imag_parts)
            total = torch.zeros((), dtype=torch.float64)
            for (i, j, k) in idx:
                from ideal_poly_volume_toolkit.geometry import triangle_volume_from_points_torch
                total = total + triangle_volume_from_points_torch(
                    Z_t[i], Z_t[j], Z_t[k], series_terms=96
                )
            loss = -total
            loss.backward()
            torch.nn.utils.clip_grad_norm_([real_parts, imag_parts], max_norm=10.0)
            return loss
        
        opt.step(closure)
    
    # Final configuration
    with torch.no_grad():
        Zf = build_Z_free(real_parts, imag_parts).detach().cpu().numpy()
        vol = ideal_poly_volume_via_delaunay(Zf, mode='eval_only', dps=150)
        
    return {
        'volume': vol,
        'config': Zf,
        'triangulation': idx
    }

# Run for seeds that gave high volumes
print("\nSearching for maximal configurations...")
maximal_configs = []

for seed in [103, 104]:  # These gave volumes > 13.026
    result = run_single_optimization(seed)
    if result['volume'] > 13.02:
        maximal_configs.append((seed, result))
        print(f"Seed {seed}: Volume = {result['volume']:.6f}")

# Analyze the best one
if maximal_configs:
    best_seed, best = max(maximal_configs, key=lambda x: x[1]['volume'])
    
    print(f"\n\nAnalyzing best configuration (seed {best_seed}):")
    print(f"Volume: {best['volume']:.6f}")
    print(f"Number of Delaunay triangles: {len(best['triangulation'])}")
    
    # Project to sphere and get convex hull
    z_with_inf = np.append(best['config'], [np.inf])
    sphere_points = lift_to_sphere_with_inf(z_with_inf)
    
    try:
        hull = ConvexHull(sphere_points)
        
        print(f"\nCombinatorial structure:")
        print(f"Number of faces: {len(hull.simplices)}")
        
        # Analyze vertex degrees
        vertex_degrees = {}
        for i in range(12):  # 12 vertices total
            degree = sum(1 for face in hull.simplices if i in face)
            vertex_degrees[i] = degree
        
        print(f"\nVertex degrees:")
        degree_counts = {}
        for v, d in vertex_degrees.items():
            degree_counts[d] = degree_counts.get(d, 0) + 1
            
        for degree in sorted(degree_counts.keys()):
            count = degree_counts[degree]
            print(f"  Degree {degree}: {count} vertices")
            
        # Check if all faces are triangles
        face_sizes = [len(face) for face in hull.simplices]
        if all(s == 3 for s in face_sizes):
            print(f"\nAll {len(hull.simplices)} faces are triangles")
        else:
            print(f"\nFace sizes vary: {set(face_sizes)}")
            
        # Compare with icosahedron
        print(f"\n\nComparison with regular icosahedron:")
        print(f"             Icosahedron    Our config")
        print(f"Vertices:    12            12")
        print(f"Faces:       20            {len(hull.simplices)}")
        print(f"Vertex deg:  5 (all)       {list(degree_counts.items())}")
        
        if len(hull.simplices) != 20:
            print(f"\nDIFFERENT COMBINATORIAL TYPE!")
            print(f"This configuration has {len(hull.simplices)} faces instead of 20")
            
    except Exception as e:
        print(f"Error computing convex hull: {e}")

else:
    print("Could not find configurations exceeding icosahedron volume")

# Let's also check the lower-volume configurations to see their structure
print("\n\nChecking other configurations for comparison...")
for seed in [100, 101, 102]:
    result = run_single_optimization(seed)
    triangles = len(result['triangulation'])
    print(f"Seed {seed}: Volume = {result['volume']:.4f}, Delaunay triangles = {triangles}")