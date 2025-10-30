#!/usr/bin/env python3
"""
Quick optimization of 7-vertex ideal polyhedron.
"""

import numpy as np
import torch
from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    triangle_volume_from_points_torch,
)
from scipy.optimize import differential_evolution

def compute_volume(params):
    """Compute volume from spherical parameters."""
    # Fixed vertices
    Z = [complex(0, 0), complex(1, 0)]
    
    # Add 4 parameterized vertices
    for i in range(4):
        theta = params[2*i]
        phi = params[2*i + 1]
        z = np.tan(theta/2) * np.exp(1j * phi)
        Z.append(z)
    
    Z_np = np.array(Z, dtype=np.complex128)
    
    try:
        idx = delaunay_triangulation_indices(Z_np)
        Z_torch = torch.tensor(Z_np, dtype=torch.complex128)
        
        total = 0
        for (i, j, k) in idx:
            vol = triangle_volume_from_points_torch(Z_torch[i], Z_torch[j], Z_torch[k], series_terms=96)
            total += vol.item()
        
        return -total  # Negative for minimization
    except:
        return 1000.0  # Penalty

# Run quick optimization
print("Quick 7-vertex optimization (5 trials)...")

best_volume = 0
best_config = None

for trial in range(5):
    print(f"\nTrial {trial+1}/5...")
    
    bounds = [(0.1, np.pi-0.1), (0, 2*np.pi)] * 4
    
    result = differential_evolution(
        compute_volume, 
        bounds, 
        maxiter=100,  # Reduced iterations
        popsize=15,    # Smaller population
        seed=trial * 42
    )
    
    volume = -result.fun
    print(f"  Volume: {volume:.6f}")
    
    if volume > best_volume:
        best_volume = volume
        best_config = result.x

# Analyze best configuration
print(f"\nBest volume: {best_volume:.6f}")

# Reconstruct for analysis
Z = [complex(0, 0), complex(1, 0)]
for i in range(4):
    theta = best_config[2*i]
    phi = best_config[2*i + 1]
    z = np.tan(theta/2) * np.exp(1j * phi)
    Z.append(z)

Z_np = np.array(Z, dtype=np.complex128)
idx = delaunay_triangulation_indices(Z_np)

# Count faces and vertices
vertices_used = set()
for (i, j, k) in idx:
    vertices_used.update([i, j, k])

print(f"\nCombinatorial structure:")
print(f"  Vertices used: {len(vertices_used)}")
print(f"  Triangular faces: {len(idx)}")

# Check vertex degrees
degrees = [0] * 7
for (i, j, k) in idx:
    degrees[i] += 1
    degrees[j] += 1
    degrees[k] += 1

print(f"  Face counts per vertex: {degrees}")

# Classification
if len(vertices_used) == 6 and len(idx) == 8:
    print("  → Looks like an octahedron")
elif len(vertices_used) == 7 and len(idx) == 10:
    print("  → Looks like a pentagonal bipyramid")
elif len(vertices_used) == 7:
    print(f"  → 7-vertex polyhedron with {len(idx)} faces")

print(f"\nRegular octahedron volume: ≈ 5.333")
print(f"Ratio: {best_volume/5.333:.3f}")