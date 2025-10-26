#!/usr/bin/env python3
"""
Explore what happens when we perturb the regular dodecahedron slightly.
What combinatorial types arise from the Delaunay triangulation?
"""

import numpy as np
from scipy.spatial import Delaunay
from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay
import matplotlib.pyplot as plt
from collections import Counter

def dodecahedron_vertices():
    """
    Generate vertices of regular dodecahedron in stereographic projection.
    The dodecahedron has 20 vertices.
    """
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    
    # Vertices in 3D before stereographic projection
    # Normalized to unit sphere
    vertices_3d = []
    
    # (±1, ±1, ±1)
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                vertices_3d.append([x, y, z])
    
    # (0, ±1/φ, ±φ)
    for y in [-1/phi, 1/phi]:
        for z in [-phi, phi]:
            vertices_3d.append([0, y, z])
            
    # (±1/φ, ±φ, 0)
    for x in [-1/phi, 1/phi]:
        for y in [-phi, phi]:
            vertices_3d.append([x, y, 0])
            
    # (±φ, 0, ±1/φ)
    for x in [-phi, phi]:
        for z in [-1/phi, 1/phi]:
            vertices_3d.append([x, 0, z])
    
    # Normalize to unit sphere
    vertices_3d = np.array(vertices_3d)
    vertices_3d = vertices_3d / np.linalg.norm(vertices_3d, axis=1, keepdims=True)
    
    # Stereographic projection from north pole
    vertices_2d = []
    for v in vertices_3d:
        x, y, z = v
        if z < 0.99:  # Avoid north pole
            X = x / (1 - z)
            Y = y / (1 - z)
            vertices_2d.append(complex(X, Y))
    
    return np.array(vertices_2d)

def analyze_triangulation(vertices, label=""):
    """Analyze the combinatorial type of a Delaunay triangulation."""
    points_2d = np.column_stack([vertices.real, vertices.imag])
    tri = Delaunay(points_2d)
    
    # Count faces by number of triangles
    face_sizes = Counter()
    
    # Group triangles by which vertices they share
    # This is a simplified analysis - full face reconstruction would be more complex
    print(f"\n{label}")
    print(f"Number of vertices: {len(vertices)}")
    print(f"Number of triangles: {len(tri.simplices)}")
    
    # Analyze vertex degrees (how many triangles meet at each vertex)
    vertex_degrees = Counter()
    for simplex in tri.simplices:
        for v in simplex:
            vertex_degrees[v] += 1
    
    degree_distribution = Counter(vertex_degrees.values())
    print(f"Vertex degree distribution: {dict(degree_distribution)}")
    
    # Compute volume
    try:
        volume = ideal_poly_volume_via_delaunay(vertices, mode='fast')
        print(f"Volume: {volume:.8f}")
    except:
        print("Volume computation failed")
        volume = None
    
    return tri, volume

def perturb_dodecahedron(epsilon=0.01, seed=42):
    """Apply small random perturbation to dodecahedron vertices."""
    np.random.seed(seed)
    vertices = dodecahedron_vertices()
    
    # Apply small perturbation
    perturbation = epsilon * (np.random.randn(len(vertices)) + 1j * np.random.randn(len(vertices)))
    perturbed = vertices + perturbation
    
    return perturbed

# Analyze regular dodecahedron
print("="*70)
print("DODECAHEDRON TRIANGULATION ANALYSIS")
print("="*70)

dodeca_vertices = dodecahedron_vertices()
print(f"\nDodecahedron has {len(dodeca_vertices)} finite vertices (plus infinity)")

# Analyze unperturbed dodecahedron
tri_exact, vol_exact = analyze_triangulation(dodeca_vertices, "Exact Dodecahedron:")

# Analyze with different perturbation sizes
perturbation_sizes = [0.001, 0.01, 0.1]
triangulation_types = []

for eps in perturbation_sizes:
    print(f"\n{'='*50}")
    print(f"Perturbation size: {eps}")
    
    # Try multiple random perturbations
    for trial in range(3):
        perturbed = perturb_dodecahedron(epsilon=eps, seed=42+trial)
        tri, vol = analyze_triangulation(perturbed, f"  Trial {trial+1}:")
        
        if vol is not None and vol_exact is not None:
            print(f"  Volume change: {(vol/vol_exact - 1)*100:.2f}%")

# Visualize one perturbation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Original dodecahedron
points_exact = np.column_stack([dodeca_vertices.real, dodeca_vertices.imag])
tri_exact = Delaunay(points_exact)
ax1.triplot(points_exact[:, 0], points_exact[:, 1], tri_exact.simplices, 'b-', alpha=0.5)
ax1.plot(points_exact[:, 0], points_exact[:, 1], 'ro')
ax1.set_title('Exact Dodecahedron Triangulation')
ax1.set_aspect('equal')

# Perturbed dodecahedron
perturbed = perturb_dodecahedron(epsilon=0.01)
points_pert = np.column_stack([perturbed.real, perturbed.imag])
tri_pert = Delaunay(points_pert)
ax2.triplot(points_pert[:, 0], points_pert[:, 1], tri_pert.simplices, 'b-', alpha=0.5)
ax2.plot(points_pert[:, 0], points_pert[:, 1], 'ro')
ax2.set_title('Perturbed Dodecahedron (ε=0.01)')
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('dodecahedron_perturbation.png', dpi=150)
print("\nSaved visualization to dodecahedron_perturbation.png")

# Check if perturbed versions are arithmetic
print("\n\nChecking arithmeticity of perturbed configurations...")
import sys
sys.path.append('.')
from check_local_maxima_arithmetic import check_arithmeticity

for eps in [0.01, 0.1]:
    perturbed = perturb_dodecahedron(epsilon=eps)
    config = {
        'vertices': {f'z{i}': {'real': v.real, 'imag': v.imag} 
                    for i, v in enumerate(perturbed)}
    }
    
    result = check_arithmeticity(config)
    if 'error' not in result:
        print(f"\nPerturbation ε={eps}:")
        print(f"  Integral traces: {result['integral_traces']}/{result['num_generators']}")
        if result['likely_arithmetic']:
            print(f"  *** LIKELY ARITHMETIC ***")
        else:
            print(f"  Not obviously arithmetic")

plt.close()