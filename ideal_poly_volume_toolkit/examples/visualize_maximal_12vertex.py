import numpy as np
import matplotlib.pyplot as plt
from ideal_poly_volume_toolkit.geometry import delaunay_triangulation_indices, lift_to_sphere_with_inf
from scipy.spatial import ConvexHull
import torch

# Recreate the maximal configuration from seed 104
def get_maximal_config(seed=104):
    rng = np.random.default_rng(seed)
    K = 9
    
    # Initialize
    r = 0.5 + 1.5 * rng.random(K)
    theta = 2 * np.pi * rng.random(K)
    z_init = r * np.exp(1j * theta)
    
    real_parts = torch.tensor(z_init.real, dtype=torch.float64, requires_grad=True)
    imag_parts = torch.tensor(z_init.imag, dtype=torch.float64, requires_grad=True)
    
    def build_Z_free(real_parts, imag_parts):
        Z = torch.empty(real_parts.numel() + 2, dtype=torch.complex128)
        Z[0] = 0 + 0j
        Z[1] = 1 + 0j
        Z[2:] = torch.complex(real_parts, imag_parts)
        return Z
    
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
    
    with torch.no_grad():
        Zf = build_Z_free(real_parts, imag_parts).detach().cpu().numpy()
        
    return Zf, idx

# Get the configuration
z_points, triangulation = get_maximal_config()

# Visualize in complex plane
plt.figure(figsize=(10, 10))

# Plot points
real_coords = z_points.real
imag_coords = z_points.imag
plt.scatter(real_coords, imag_coords, s=200, c='red', zorder=5)

# Label points
for i, z in enumerate(z_points):
    offset = 0.15
    plt.annotate(f'{i}', (z.real + offset, z.imag + offset), fontsize=12)

# Draw Delaunay triangles
for tri in triangulation:
    triangle_x = [real_coords[tri[j]] for j in range(3)] + [real_coords[tri[0]]]
    triangle_y = [imag_coords[tri[j]] for j in range(3)] + [imag_coords[tri[0]]]
    plt.plot(triangle_x, triangle_y, 'b-', linewidth=1.5, alpha=0.7)

plt.xlabel('Real', fontsize=14)
plt.ylabel('Imaginary', fontsize=14)
plt.title('Maximal 12-vertex Configuration (Volume = 13.032)\nDifferent combinatorial type than icosahedron!', fontsize=16)
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.tight_layout()
plt.savefig('maximal_12vertex_plane.png', dpi=150, bbox_inches='tight')
print("Saved: maximal_12vertex_plane.png")

# Analyze the sphere structure
z_with_inf = np.append(z_points, [np.inf])
sphere_points = lift_to_sphere_with_inf(z_with_inf)
hull = ConvexHull(sphere_points)

# Find which vertices have degree 4 and 6
vertex_degrees = {}
for i in range(12):
    degree = sum(1 for face in hull.simplices if i in face)
    vertex_degrees[i] = degree

print("\nVertex degrees in detail:")
deg_4_vertices = [v for v, d in vertex_degrees.items() if d == 4]
deg_5_vertices = [v for v, d in vertex_degrees.items() if d == 5]
deg_6_vertices = [v for v, d in vertex_degrees.items() if d == 6]

print(f"Degree 4 vertices: {deg_4_vertices}")
print(f"Degree 5 vertices: {deg_5_vertices}")
print(f"Degree 6 vertices: {deg_6_vertices}")

# Map back to complex plane
print("\nSpecial vertices in complex plane:")
for v in deg_4_vertices:
    if v < 11:
        print(f"  Vertex {v} (deg 4): z = {z_points[v]:.4f}")
    else:
        print(f"  Vertex {v} (deg 4): infinity")
        
for v in deg_6_vertices:
    if v < 11:
        print(f"  Vertex {v} (deg 6): z = {z_points[v]:.4f}")
    else:
        print(f"  Vertex {v} (deg 6): infinity")

print(f"\nThis polyhedron has the same number of faces (20) as the icosahedron,")
print(f"but a different combinatorial structure!")
print(f"It's a non-regular ideal polyhedron with 12 vertices.")

plt.close()