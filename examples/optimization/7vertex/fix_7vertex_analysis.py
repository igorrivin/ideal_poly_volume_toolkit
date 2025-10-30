#!/usr/bin/env python3
"""
Fix the 7-vertex analysis - check which vertices are actually used.
"""

import numpy as np
import torch
from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    triangle_volume_from_points_torch,
)
from scipy.optimize import differential_evolution

# Run a single good optimization
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
        
        # Check if all 7 vertices are used
        vertices_used = set()
        for (i, j, k) in idx:
            vertices_used.update([i, j, k])
        
        if len(vertices_used) < 7:
            return 1000.0  # Penalty if not all vertices used
        
        Z_torch = torch.tensor(Z_np, dtype=torch.complex128)
        
        total = 0
        for (i, j, k) in idx:
            vol = triangle_volume_from_points_torch(Z_torch[i], Z_torch[j], Z_torch[k], series_terms=96)
            total += vol.item()
        
        return -total  # Negative for minimization
    except:
        return 1000.0  # Penalty

print("Running constrained optimization (all 7 vertices must be used)...")

bounds = [(0.1, np.pi-0.1), (0, 2*np.pi)] * 4

result = differential_evolution(
    compute_volume, 
    bounds, 
    maxiter=200,
    popsize=20,
    seed=42
)

volume = -result.fun
print(f"Optimal volume: {volume:.6f}")

# Analyze the result
Z = [complex(0, 0), complex(1, 0)]
for i in range(4):
    theta = result.x[2*i]
    phi = result.x[2*i + 1]
    z = np.tan(theta/2) * np.exp(1j * phi)
    Z.append(z)

print("\nVertex positions:")
for i, z in enumerate(Z):
    print(f"z[{i}] = {z.real:+.4f} + {z.imag:+.4f}i")

Z_np = np.array(Z, dtype=np.complex128)
idx = delaunay_triangulation_indices(Z_np)

# Check which vertices are used
vertices_used = set()
for (i, j, k) in idx:
    vertices_used.update([i, j, k])

print(f"\nVertices used: {sorted(vertices_used)}")
print(f"Number of faces: {len(idx)}")

# Count edges properly
edges = set()
for (a, b, c) in idx:
    edges.add(tuple(sorted([a, b])))
    edges.add(tuple(sorted([b, c])))
    edges.add(tuple(sorted([a, c])))

V = len(vertices_used)
E = len(edges)
F = len(idx)
chi = V - E + F

print(f"\nEuler characteristic: V - E + F = {V} - {E} + {F} = {chi}")

# Vertex degrees
degrees = [0] * 7
for edge in edges:
    if edge[0] < 7 and edge[1] < 7:
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1

print(f"Vertex degrees: {degrees}")

# Volume
Z_torch = torch.tensor(Z_np, dtype=torch.complex128)
total_volume = 0
for (a, b, c) in idx:
    vol = triangle_volume_from_points_torch(Z_torch[a], Z_torch[b], Z_torch[c], series_terms=96)
    total_volume += vol.item()

print(f"\nTotal volume: {total_volume:.6f}")
print(f"Regular octahedron: ≈ 5.333")
print(f"Ratio: {total_volume/5.333:.3f}")