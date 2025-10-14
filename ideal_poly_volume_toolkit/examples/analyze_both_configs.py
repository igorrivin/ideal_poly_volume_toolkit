import numpy as np
from ideal_poly_volume_toolkit.geometry import delaunay_triangulation_indices, lift_to_sphere_with_inf
from scipy.spatial import ConvexHull

# Configuration 1: The maximal volume (~6.4885) from seed 0
config1_points = np.array([
    0+0j,
    1+0j,
    0.5003 - 0.5829j,
    0.1976 - 1.2890j,
    0.0005 - 0.6598j,
    -0.6504 - 0.7617j,
    1.1527 - 0.9891j
])

# Configuration 2: Golden ratio configuration (~6.0017) from seed 42
phi = (1 + np.sqrt(5)) / 2
config2_points = np.array([
    0+0j,
    1+0j,
    2.6180 + 0j,      # φ²
    0.5 - 1.5388j,
    2.8917 - 2.6037j,
    0.5 + 1.5388j,
    -1.6180 + 0j      # -φ
])

def analyze_configuration(points, name):
    print(f"\n=== {name} ===")
    
    # Get Delaunay triangulation
    triangles = delaunay_triangulation_indices(points)
    print(f"Number of Delaunay triangles: {len(triangles)}")
    
    # Add infinity and project to sphere
    z_with_inf = np.append(points, [np.inf])
    sphere_points = lift_to_sphere_with_inf(z_with_inf)
    
    # Get convex hull on sphere
    try:
        hull = ConvexHull(sphere_points)
        print(f"Number of faces on sphere: {len(hull.simplices)}")
        
        # Count vertices per face
        vertex_counts = {}
        for vertex_idx in range(8):  # 8 vertices total
            count = sum(1 for face in hull.simplices if vertex_idx in face)
            vertex_counts[vertex_idx] = count
        
        print("Vertex degrees (number of faces containing each vertex):")
        for i in range(7):
            print(f"  z[{i}]: {vertex_counts[i]} faces")
        print(f"  ∞: {vertex_counts[7]} faces")
        
        # Analyze face structure
        print("\nFace structure:")
        for i, face in enumerate(hull.simplices):
            vertices = [f"z[{v}]" if v < 7 else "∞" for v in face]
            print(f"  Face {i}: {vertices}")
            
    except Exception as e:
        print(f"Could not compute convex hull: {e}")
    
    # Check symmetries in the plane
    print("\nSymmetries in complex plane:")
    # Check for points on real axis
    on_real = [i for i, z in enumerate(points) if abs(z.imag) < 0.01]
    print(f"  Points on real axis: {on_real}")
    
    # Check for conjugate pairs
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if abs(points[i].real - points[j].real) < 0.01 and abs(points[i].imag + points[j].imag) < 0.01:
                print(f"  z[{i}] and z[{j}] are conjugates")
    
    # Check distances from origin
    print("\nDistances from origin:")
    for i, z in enumerate(points):
        r = abs(z)
        print(f"  |z[{i}]| = {r:.4f}")

# Analyze both configurations
analyze_configuration(config1_points, "Maximal Configuration (volume ~6.4885)")
analyze_configuration(config2_points, "Golden Ratio Configuration (volume ~6.0017)")

# Compare structures
print("\n=== Comparison ===")
print("Maximal config: Unknown structure, volume 6.4885")
print("Golden config: Appears to be hexagonal bipyramid, volume 6.0017")
print("Regular cube would have volume 5.0747")
print("\nBoth configurations significantly exceed the regular cube!")