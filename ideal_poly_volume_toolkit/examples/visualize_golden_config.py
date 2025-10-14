import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    lift_to_sphere_with_inf
)

# The golden ratio configuration from seed 42
phi = (1 + np.sqrt(5)) / 2
z_points = np.array([
    0 + 0j,                    # z[0]
    1 + 0j,                    # z[1]
    2.6180 + 0j,              # z[2] ≈ φ²
    0.5 - 1.5388j,            # z[3] 
    2.8917 - 2.6037j,         # z[4]
    0.5 + 1.5388j,            # z[5]
    -1.6180 + 0j              # z[6] ≈ -φ
])

# Get Delaunay triangulation
triangles = delaunay_triangulation_indices(z_points)

# Figure 1: Complex plane view with Delaunay triangulation
fig1, ax1 = plt.subplots(figsize=(10, 10))

# Plot points
real_coords = z_points.real
imag_coords = z_points.imag
ax1.scatter(real_coords, imag_coords, s=100, c='red', zorder=5)

# Label points
for i, z in enumerate(z_points):
    offset = 0.1
    ax1.annotate(f'z[{i}]', (z.real + offset, z.imag + offset))

# Draw Delaunay triangles
for tri in triangles:
    triangle = plt.Polygon([(real_coords[tri[j]], imag_coords[tri[j]]) for j in range(3)],
                          fill=False, edgecolor='blue', linewidth=1.5)
    ax1.add_patch(triangle)

ax1.set_xlabel('Real')
ax1.set_ylabel('Imaginary')
ax1.set_title('Golden Ratio Configuration in Complex Plane\nwith Delaunay Triangulation')
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# Add circle at radius φ and φ²
circle1 = plt.Circle((0, 0), phi, fill=False, linestyle='--', color='green', alpha=0.5)
circle2 = plt.Circle((0, 0), phi**2, fill=False, linestyle='--', color='green', alpha=0.5)
ax1.add_patch(circle1)
ax1.add_patch(circle2)
ax1.text(0.1, phi + 0.1, f'r = φ ≈ {phi:.3f}', color='green')
ax1.text(0.1, phi**2 + 0.1, f'r = φ² ≈ {phi**2:.3f}', color='green')

plt.savefig('golden_config_plane.png', dpi=150, bbox_inches='tight')
print("Saved: golden_config_plane.png")

# Figure 2: Stereographic projection to unit sphere
fig2 = plt.figure(figsize=(12, 12))
ax2 = fig2.add_subplot(111, projection='3d')

# Add infinity point for sphere projection
z_with_inf = np.append(z_points, [np.inf])
sphere_points = lift_to_sphere_with_inf(z_with_inf)

# Plot sphere points
ax2.scatter(sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2], 
           s=100, c='red', depthshade=False)

# Label points
for i in range(len(sphere_points)):
    if i < len(z_points):
        label = f'z[{i}]'
    else:
        label = '∞'
    ax2.text(sphere_points[i, 0], sphere_points[i, 1], sphere_points[i, 2], 
            label, fontsize=10)

# Draw edges based on convex hull of sphere points
# We need to determine which edges to draw based on the Delaunay triangulation
# Each triangle in the plane corresponds to a face on the sphere
edges_drawn = set()

for tri in triangles:
    # Draw the three edges of each triangle
    for i in range(3):
        j = (i + 1) % 3
        edge = tuple(sorted([tri[i], tri[j]]))
        if edge not in edges_drawn:
            edges_drawn.add(edge)
            p1 = sphere_points[edge[0]]
            p2 = sphere_points[edge[1]]
            ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                    'b-', linewidth=1.5)

# Also draw edges to infinity (north pole)
# In ideal polyhedra, infinity is connected to certain boundary vertices
inf_idx = len(z_points)  # Index of infinity point
# Connect infinity to vertices that are on the boundary of the convex hull in the plane
from scipy.spatial import ConvexHull
hull = ConvexHull(np.column_stack([real_coords, imag_coords]))
for vertex_idx in hull.vertices:
    p1 = sphere_points[vertex_idx]
    p2 = sphere_points[inf_idx]
    ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
            'g-', linewidth=1.5, alpha=0.7)

# Draw unit sphere (transparent)
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
ax2.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='cyan')

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Golden Ratio Configuration\nProjected onto Unit Sphere')

# Set equal aspect ratio
ax2.set_box_aspect([1,1,1])
ax2.view_init(elev=20, azim=45)

plt.savefig('golden_config_sphere.png', dpi=150, bbox_inches='tight')
print("Saved: golden_config_sphere.png")

# Print configuration details
print("\nConfiguration details:")
print(f"Number of vertices: {len(z_points)} (plus infinity)")
print(f"Number of triangles: {len(triangles)}")
print(f"Triangles: {triangles}")

# Analyze symmetries
print("\nDistance analysis:")
distances = {}
for i in range(len(z_points)):
    for j in range(i+1, len(z_points)):
        d = abs(z_points[i] - z_points[j])
        key = f"|z[{i}] - z[{j}]|"
        distances[key] = d

# Group by similar distances
unique_distances = {}
for key, d in distances.items():
    found = False
    for ud in unique_distances:
        if abs(d - ud) < 0.001:
            unique_distances[ud].append(key)
            found = True
            break
    if not found:
        unique_distances[d] = [key]

print("\nGrouped distances:")
for d in sorted(unique_distances.keys()):
    print(f"  {d:.4f}: {unique_distances[d]}")

plt.show()