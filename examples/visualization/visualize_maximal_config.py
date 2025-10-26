import numpy as np
import matplotlib.pyplot as plt
from ideal_poly_volume_toolkit.geometry import delaunay_triangulation_indices

# The maximal configuration from seed 0
z_points = np.array([
    0+0j,
    1+0j,
    0.5003 - 0.5829j,
    0.1976 - 1.2890j,
    0.0005 - 0.6598j,
    -0.6504 - 0.7617j,
    1.1527 - 0.9891j
])

# Get Delaunay triangulation
triangles = delaunay_triangulation_indices(z_points)

# Complex plane view
plt.figure(figsize=(10, 10))

# Plot points
real_coords = z_points.real
imag_coords = z_points.imag
plt.scatter(real_coords, imag_coords, s=200, c='red', zorder=5)

# Label points
for i, z in enumerate(z_points):
    offset = 0.1
    plt.annotate(f'z[{i}]', (z.real + offset, z.imag + offset), fontsize=12)

# Draw Delaunay triangles
for tri in triangles:
    triangle_x = [real_coords[tri[j]] for j in range(3)] + [real_coords[tri[0]]]
    triangle_y = [imag_coords[tri[j]] for j in range(3)] + [imag_coords[tri[0]]]
    plt.plot(triangle_x, triangle_y, 'b-', linewidth=1.5)

# Draw convex hull boundary
from scipy.spatial import ConvexHull
points = np.column_stack([real_coords, imag_coords])
hull = ConvexHull(points)
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'g-', linewidth=2.5, alpha=0.5)

plt.xlabel('Real', fontsize=14)
plt.ylabel('Imaginary', fontsize=14)
plt.title('Maximal Volume Configuration (6.4885)\nin Complex Plane', fontsize=16)
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Add unit circle for reference
circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', color='gray', alpha=0.5)
plt.gca().add_patch(circle)

plt.tight_layout()
plt.savefig('maximal_config_plane.png', dpi=150, bbox_inches='tight')
print("Saved: maximal_config_plane.png")

# Print analysis
print("\nMaximal configuration analysis:")
print(f"Volume: 6.4885 (28% larger than regular cube)")
print(f"Number of triangles: {len(triangles)}")
print(f"Triangulation: {triangles}")

# Check if any points are approximately on the unit circle
print("\nPoints near unit circle:")
for i, z in enumerate(z_points):
    r = abs(z)
    if abs(r - 1.0) < 0.1:
        print(f"  z[{i}]: |z| = {r:.4f}")

# Check approximate symmetries
print("\nChecking for approximate symmetries:")
# Look for points that are approximately equally spaced
angles = [np.angle(z) * 180/np.pi for z in z_points if abs(z) > 0.5]
print(f"Angles of points: {[f'{a:.1f}°' for a in sorted(angles)]}")

plt.close()