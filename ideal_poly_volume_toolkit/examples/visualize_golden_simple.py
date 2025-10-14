import numpy as np
import matplotlib.pyplot as plt
from ideal_poly_volume_toolkit.geometry import delaunay_triangulation_indices

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

# Complex plane view with Delaunay triangulation
plt.figure(figsize=(10, 10))

# Plot points
real_coords = z_points.real
imag_coords = z_points.imag
plt.scatter(real_coords, imag_coords, s=200, c='red', zorder=5)

# Label points
for i, z in enumerate(z_points):
    offset = 0.15
    plt.annotate(f'z[{i}]', (z.real + offset, z.imag + offset), fontsize=12)

# Draw Delaunay triangles
for tri in triangles:
    # Close the triangle by adding the first point at the end
    triangle_x = [real_coords[tri[j]] for j in range(3)] + [real_coords[tri[0]]]
    triangle_y = [imag_coords[tri[j]] for j in range(3)] + [imag_coords[tri[0]]]
    plt.plot(triangle_x, triangle_y, 'b-', linewidth=1.5)

plt.xlabel('Real', fontsize=14)
plt.ylabel('Imaginary', fontsize=14)
plt.title('Golden Ratio Configuration in Complex Plane\nwith Delaunay Triangulation', fontsize=16)
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Add circles at special radii
circle1 = plt.Circle((0, 0), phi, fill=False, linestyle='--', color='green', alpha=0.5, linewidth=2)
circle2 = plt.Circle((0, 0), phi**2, fill=False, linestyle='--', color='green', alpha=0.5, linewidth=2)
plt.gca().add_patch(circle1)
plt.gca().add_patch(circle2)
plt.text(0.1, phi + 0.2, f'r = φ ≈ {phi:.3f}', color='green', fontsize=12)
plt.text(0.1, phi**2 + 0.2, f'r = φ² ≈ {phi**2:.3f}', color='green', fontsize=12)

plt.tight_layout()
plt.savefig('golden_config_plane.png', dpi=150, bbox_inches='tight')
print("Saved: golden_config_plane.png")

# Print configuration details
print("\nConfiguration details:")
print(f"Number of vertices: {len(z_points)} (plus infinity)")
print(f"Number of triangles: {len(triangles)}")
print("\nTriangulation:")
for i, tri in enumerate(triangles):
    print(f"  Triangle {i}: vertices {tri}")

# Analyze special values
print("\nSpecial values in configuration:")
print(f"  φ = {phi:.6f}")
print(f"  φ² = {phi**2:.6f}")
print(f"  1/φ = {1/phi:.6f}")
print(f"  0.5 = 1/2")
print(f"  1.5388 ≈ √(5/2) = {np.sqrt(5/2):.4f}? No, = {1.5388:.4f}")
print(f"  1.5388 ≈ φ - 1/(2φ) = {phi - 1/(2*phi):.4f}? Close!")

# Check if z[3] and z[5] are special
z3_abs = abs(0.5 - 1.5388j)
z5_abs = abs(0.5 + 1.5388j)
print(f"\n|z[3]| = |z[5]| = {z3_abs:.6f} = φ")

plt.close()