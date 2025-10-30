#!/usr/bin/env python3
"""
Debug 7-vertex optimization.
"""

import numpy as np
import torch
from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    triangle_volume_from_points_torch,
)

# Test a simple configuration
z1 = complex(0, 0)
z2 = complex(1, 0)
z4 = complex(-1, 0)
z5 = complex(0, -1)
z6 = complex(0.5, 0.5)
z7 = complex(-0.5, -0.5)

complex_points = [z1, z2, z3, z4, z5, z6, z7]
Z_np = np.array(complex_points, dtype=np.complex128)
points_2d = np.column_stack([Z_np.real, Z_np.imag])

print("Testing 7-vertex configuration:")
print(f"Points shape: {points_2d.shape}")
print(f"Points:\n{points_2d}")

try:
    idx = delaunay_triangulation_indices(points_2d)
    print(f"\nDelaunay triangulation successful!")
    print(f"Number of triangles: {len(idx)}")
    print(f"Triangles: {idx}")
    
    # Compute volume
    Z_torch = torch.tensor(Z_np, dtype=torch.complex128)
    total_volume = 0
    for i, (a, b, c) in enumerate(idx):
        vol = triangle_volume_from_points_torch(Z_torch[a], Z_torch[b], Z_torch[c], series_terms=96)
        total_volume += vol.item()
        print(f"  Triangle {i} ({a},{b},{c}): volume = {vol.item():.6f}")
    
    print(f"\nTotal volume: {total_volume:.6f}")
    
except Exception as e:
    print(f"\nError: {e}")
    print(f"Error type: {type(e)}")

# Now test with just 4 points (should work)
print("\n" + "="*50)
print("Testing 4-vertex configuration (tetrahedron):")
z1 = complex(0, 0)
z2 = complex(1, 0)
z4 = complex(0.5, 0.5)

complex_points = [z1, z2, z3, z4]
Z_np = np.array(complex_points, dtype=np.complex128)
points_2d = np.column_stack([Z_np.real, Z_np.imag])

try:
    idx = delaunay_triangulation_indices(points_2d)
    print(f"Number of triangles: {len(idx)}")
    
    Z_torch = torch.tensor(Z_np, dtype=torch.complex128)
    total_volume = 0
    for i, (a, b, c) in enumerate(idx):
        vol = triangle_volume_from_points_torch(Z_torch[a], Z_torch[b], Z_torch[c], series_terms=96)
        total_volume += vol.item()
    
    print(f"Total volume: {total_volume:.6f}")
    
except Exception as e:
    print(f"Error: {e}")