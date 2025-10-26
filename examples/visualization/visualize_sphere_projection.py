import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ideal_poly_volume_toolkit.geometry import lift_to_sphere_with_inf
from scipy.spatial import ConvexHull

# The golden ratio configuration
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

# Add infinity point for sphere projection
z_with_inf = np.append(z_points, [np.inf])
sphere_points = lift_to_sphere_with_inf(z_with_inf)

print("Sphere points:")
for i, p in enumerate(sphere_points):
    if i < 7:
        print(f"  z[{i}] → ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")
    else:
        print(f"  ∞ → ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) [North pole]")

# Create 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot points
colors = ['red'] * 7 + ['blue']  # Different color for infinity
ax.scatter(sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2], 
           s=200, c=colors, depthshade=False, edgecolors='black', linewidth=2)

# Label points
labels = [f'z[{i}]' for i in range(7)] + ['∞']
for i, (point, label) in enumerate(zip(sphere_points, labels)):
    ax.text(point[0]*1.1, point[1]*1.1, point[2]*1.1, label, fontsize=10)

# Try to construct the convex hull on the sphere
# This will give us the ideal polyhedron
try:
    hull = ConvexHull(sphere_points)
    
    # Plot the faces
    for simplex in hull.simplices:
        # Get the three vertices of each triangular face
        triangle = sphere_points[simplex]
        # Close the triangle
        triangle = np.vstack([triangle, triangle[0]])
        ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], 'b-', linewidth=1.5)
    
    print(f"\nConvex hull has {len(hull.simplices)} faces")
    print("Faces (as vertex indices):")
    for i, face in enumerate(hull.simplices):
        print(f"  Face {i}: {face}")
        
except Exception as e:
    print(f"\nCouldn't compute convex hull: {e}")
    print("Drawing edges manually based on Delaunay triangulation...")
    
    # Fall back to drawing edges from Delaunay triangulation
    from ideal_poly_volume_toolkit.geometry import delaunay_triangulation_indices
    triangles = delaunay_triangulation_indices(z_points)
    
    edges_drawn = set()
    for tri in triangles:
        for i in range(3):
            j = (i + 1) % 3
            edge = tuple(sorted([tri[i], tri[j]]))
            if edge not in edges_drawn:
                edges_drawn.add(edge)
                p1 = sphere_points[edge[0]]
                p2 = sphere_points[edge[1]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                       'b-', linewidth=1.5)
    
    # Connect boundary vertices to infinity
    real_coords = z_points.real
    imag_coords = z_points.imag
    hull_2d = ConvexHull(np.column_stack([real_coords, imag_coords]))
    for vertex_idx in hull_2d.vertices:
        p1 = sphere_points[vertex_idx]
        p2 = sphere_points[7]  # infinity
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
               'g-', linewidth=1.5, alpha=0.7)

# Draw unit sphere wireframe
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))  
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray', linewidth=0.5)

# Add equator and meridians
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta), 'k-', alpha=0.3, linewidth=1)
ax.plot([0, 0], [-1, 1], [0, 0], 'k-', alpha=0.3, linewidth=1)
ax.plot([-1, 1], [0, 0], [0, 0], 'k-', alpha=0.3, linewidth=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Golden Ratio Configuration\nProjected onto Unit Sphere', fontsize=16)

# Set equal aspect ratio and nice viewing angle
ax.set_box_aspect([1,1,1])
ax.view_init(elev=20, azim=45)

# Set axis limits
ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])
ax.set_zlim([-1.2, 1.2])

plt.tight_layout()
plt.savefig('golden_config_sphere.png', dpi=150, bbox_inches='tight')
print("\nSaved: golden_config_sphere.png")

plt.close()