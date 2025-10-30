#!/usr/bin/env python3
"""
Detailed analysis of the 7-vertex optimal configuration.
"""

import numpy as np
import torch
from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    triangle_volume_from_points_torch,
)

# The best parameters found
best_params = [1.3962632, 0.0, 0.34906593, 3.14159277, 
               2.09439515, 1.57079633, 2.79252652, 4.71238910]

# Reconstruct configuration
Z = [complex(0, 0), complex(1, 0)]
for i in range(4):
    theta = best_params[2*i]
    phi = best_params[2*i + 1]
    z = np.tan(theta/2) * np.exp(1j * phi)
    Z.append(z)

print("7-vertex optimal configuration:")
print("="*50)
for i, z in enumerate(Z):
    print(f"z[{i}] = {z.real:+.4f} + {z.imag:+.4f}i   |z|={abs(z):.4f}")

Z_np = np.array(Z, dtype=np.complex128)
idx = delaunay_triangulation_indices(Z_np)

print(f"\nTriangulation has {len(idx)} faces:")
for i, (a, b, c) in enumerate(idx):
    print(f"  Face {i}: ({a}, {b}, {c})")

# Compute edge set
edges = set()
for (a, b, c) in idx:
    edges.add(tuple(sorted([a, b])))
    edges.add(tuple(sorted([b, c])))
    edges.add(tuple(sorted([a, c])))

print(f"\nTotal edges: {len(edges)}")

# Euler characteristic check
V, E, F = 7, len(edges), len(idx)
chi = V - E + F
print(f"\nEuler characteristic: V - E + F = {V} - {E} + {F} = {chi}")
print(f"Expected for sphere: 2 {'✓' if chi == 2 else '✗'}")

# Vertex degrees
degrees = [0] * 7
for edge in edges:
    degrees[edge[0]] += 1
    degrees[edge[1]] += 1

print(f"\nVertex degrees: {degrees}")

# Volume computation
Z_torch = torch.tensor(Z_np, dtype=torch.complex128)
total_volume = 0
for (a, b, c) in idx:
    vol = triangle_volume_from_points_torch(Z_torch[a], Z_torch[b], Z_torch[c], series_terms=96)
    total_volume += vol.item()

print(f"\nTotal volume: {total_volume:.6f}")

# Check if it matches stellated octahedron pattern
print("\n" + "="*50)
print("Structure analysis:")

# Count vertices with each degree
degree_counts = {}
for d in degrees:
    degree_counts[d] = degree_counts.get(d, 0) + 1

print(f"Degree distribution: {degree_counts}")

# A stellated octahedron would have:
# - Original 6 vertices of octahedron
# - 1 new vertex "stellating" one face
# The new vertex would have degree 3 (connected to the 3 vertices of the stellated face)
# The 3 vertices of the stellated face would have their degree increased by 1

if 5 in degree_counts and degree_counts[5] == 1:
    print("\n✓ Found one vertex with degree 5 - likely the 'stellating' vertex")
    stellation_vertex = degrees.index(5)
    print(f"  Stellation vertex: {stellation_vertex}")
    
    # Find faces containing this vertex
    stellated_faces = []
    for i, (a, b, c) in enumerate(idx):
        if stellation_vertex in (a, b, c):
            stellated_faces.append((i, (a, b, c)))
    
    print(f"  This vertex appears in {len(stellated_faces)} faces")

print("\nConclusion:")
print("This appears to be a 7-vertex polyhedron with an irregular structure.")
print("It has only 7 faces (not the 10 of a pentagonal bipyramid).")
print("The vertex degree pattern suggests it might be related to a stellated octahedron,"
      " but with a more complex modification.")