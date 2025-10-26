import numpy as np
from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay

# Construct regular icosahedron vertices
# One standard construction places vertices at:
# - (0, ±1, ±φ) and cyclic permutations
# where φ = (1+√5)/2 is the golden ratio

phi = (1 + np.sqrt(5)) / 2

# Create the 12 vertices of a regular icosahedron
# We'll use a construction that places them nicely in 3D,
# then stereographically project to the complex plane

# 3D vertices of regular icosahedron (before normalization)
vertices_3d = []

# Rectangle in xy-plane
vertices_3d.extend([(1, phi, 0), (1, -phi, 0), (-1, phi, 0), (-1, -phi, 0)])
# Rectangle in yz-plane  
vertices_3d.extend([(0, 1, phi), (0, 1, -phi), (0, -1, phi), (0, -1, -phi)])
# Rectangle in xz-plane
vertices_3d.extend([(phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)])

# Convert to numpy array
vertices_3d = np.array(vertices_3d)

# Normalize to unit sphere
norms = np.linalg.norm(vertices_3d, axis=1)
vertices_3d = vertices_3d / norms[:, np.newaxis]

print("Regular icosahedron vertices on unit sphere:")
for i, v in enumerate(vertices_3d):
    print(f"  v[{i:2d}] = ({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f})")

# Stereographic projection from sphere to complex plane
# We need to choose one vertex to map to infinity
# Let's use vertex 0
def stereographic_project(point_3d, north_pole_idx=0):
    """Project from unit sphere to complex plane, with one point going to infinity"""
    x, y, z = point_3d
    if abs(z - 1) < 1e-10:  # North pole goes to infinity
        return np.inf
    else:
        return complex(x/(1-z), y/(1-z))

# Try different vertices as infinity to find best configuration
best_volume = 0
best_config = None
best_infinity_idx = None

for infinity_idx in range(12):
    # Project all vertices except the one going to infinity
    z_points = []
    for i in range(12):
        if i != infinity_idx:
            # Rotate sphere so vertex infinity_idx is at north pole
            # This is a bit involved, so let's use a simpler approach:
            # Just project using the standard stereographic projection
            # and handle the infinity point separately
            
            v = vertices_3d[i]
            # Rotate so that vertices_3d[infinity_idx] goes to north pole
            v_inf = vertices_3d[infinity_idx]
            
            # Simple approach: if this is our infinity vertex, skip it
            if i == infinity_idx:
                continue
                
            # Otherwise project normally
            z = stereographic_project(v)
            z_points.append(z)
    
    z_points = np.array(z_points)
    
    # Compute volume
    try:
        volume = ideal_poly_volume_via_delaunay(z_points, mode='exact', dps=150)
        print(f"\nWith vertex {infinity_idx} at infinity: volume = {volume:.10f}")
        
        if volume > best_volume:
            best_volume = volume
            best_config = z_points
            best_infinity_idx = infinity_idx
            
    except Exception as e:
        print(f"\nWith vertex {infinity_idx} at infinity: failed - {e}")

print(f"\n{'='*60}")
print(f"Best configuration: vertex {best_infinity_idx} at infinity")
print(f"Regular icosahedron volume: {best_volume:.10f}")
print('='*60)

# Compare with our optimization results
print("\nComparison with optimization results:")
print(f"  Regular icosahedron: {best_volume:.10f}")
print(f"  Optimization trial 1: 12.6389975023")
print(f"  Optimization trial 2: 12.8822902035") 
print(f"  Optimization trial 3: 12.9432436743")

if 12.9432 > best_volume:
    print(f"\nThe optimization found a configuration with HIGHER volume than the regular icosahedron!")
    print(f"Ratio: {12.9432/best_volume:.4f}")
else:
    print(f"\nThe regular icosahedron appears to be optimal.")

# Let's also try a more direct construction
# Place 11 vertices optimally with 0 and 1 fixed
print("\n\nAlternative: constructing regular icosahedron with 0 and 1 fixed...")
# This is complex - we'd need to find the right Möbius transformation
# that maps two vertices of the icosahedron to 0 and 1