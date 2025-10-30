#!/usr/bin/env python3
"""
Optimize ideal polyhedron volume for 7 vertices.
User hypothesis: Maximum might be octahedron with one stellated face.
"""

import numpy as np
import torch
from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    triangle_volume_from_points_torch,
)
from scipy.optimize import differential_evolution
import time

def spherical_to_complex(theta, phi):
    """Convert spherical coordinates to complex via stereographic projection."""
    return np.tan(theta/2) * np.exp(1j * phi)

def compute_volume_numpy(params):
    """Compute volume for numpy array input."""
    # First 3 points fixed to break symmetry
    z1 = complex(0, 0)
    z2 = complex(1, 0) 
    # Other 4 points from parameters (spherical coords)
    complex_points = [z1, z2]
    for i in range(4):
        theta = params[2*i]
        phi = params[2*i + 1]
        z = spherical_to_complex(theta, phi)
        complex_points.append(z)
    
    # Convert to numpy array
    Z_np = np.array(complex_points, dtype=np.complex128)
    
    # Delaunay triangulation
    try:
        idx = delaunay_triangulation_indices(Z_np)
    except:
        return 1000.0  # Penalty for degenerate configuration
    
    # Convert to torch for volume computation
    Z_torch = torch.tensor(Z_np, dtype=torch.complex128)
    
    # Compute total volume
    total_volume = 0
    for (i, j, k) in idx:
        try:
            vol = triangle_volume_from_points_torch(Z_torch[i], Z_torch[j], Z_torch[k], series_terms=96)
            total_volume += vol.item()
        except:
            return 1000.0  # Penalty for invalid configuration
    
    return -total_volume  # Negative for minimization

def analyze_hull_structure(Z_np, idx):
    """Analyze the combinatorial structure of the triangulation."""
    n_vertices = len(Z_np)
    n_faces = len(idx)
    
    # Count edges
    edges = set()
    for (i, j, k) in idx:
        edges.add((min(i,j), max(i,j)))
        edges.add((min(i,k), max(i,k)))
        edges.add((min(j,k), max(j,k)))
    n_edges = len(edges)
    
    # Vertex degrees
    vertex_degrees = [0] * n_vertices
    for edge in edges:
        vertex_degrees[edge[0]] += 1
        vertex_degrees[edge[1]] += 1
    
    # Classify polyhedron
    polyhedron_type = "Unknown"
    if n_vertices == 6 and n_faces == 8:
        polyhedron_type = "Octahedron"
    elif n_vertices == 7 and n_faces == 10:
        polyhedron_type = "Pentagonal bipyramid"
    elif n_vertices == 7 and n_faces == 9:
        polyhedron_type = "Stellated octahedron (likely)"
    
    return {
        'n_vertices': n_vertices,
        'n_faces': n_faces,
        'n_edges': n_edges,
        'vertex_degrees': sorted(vertex_degrees),
        'triangles': idx,
        'polyhedron_type': polyhedron_type
    }

# Run optimization with multiple random starts
print("Optimizing 7-vertex ideal polyhedron volume...")
print("="*70)

best_volume = 0
best_params = None
best_hull_info = None
best_complex_points = None

n_trials = 20
for trial in range(n_trials):
    print(f"\nTrial {trial+1}/{n_trials}...")
    
    # Bounds: theta in (0, pi), phi in [0, 2*pi)
    bounds = [(0.1, np.pi-0.1), (0, 2*np.pi)] * 4
    
    result = differential_evolution(
        compute_volume_numpy, 
        bounds, 
        maxiter=500,
        popsize=30,
        polish=True,
        workers=1,
        seed=42 + trial * 13
    )
    
    volume = -result.fun
    
    if volume > best_volume:
        best_volume = volume
        best_params = result.x
        
        # Reconstruct best configuration for analysis
        z1 = complex(0, 0)
        z2 = complex(1, 0)
        complex_points = [z1, z2]
        for i in range(4):
            theta = best_params[2*i]
            phi = best_params[2*i + 1]
            z = spherical_to_complex(theta, phi)
            complex_points.append(z)
        
        Z_np = np.array(complex_points, dtype=np.complex128)
        try:
            idx = delaunay_triangulation_indices(Z_np)
        except:
            continue  # Skip degenerate configurations
        
        best_hull_info = analyze_hull_structure(Z_np, idx)
        best_complex_points = complex_points
        
    print(f"  Volume: {volume:.6f}")

print("\n" + "="*70)
print("RESULTS:")
print("="*70)
print(f"Best volume found: {best_volume:.6f}")

# Analyze the best configuration
if best_hull_info:
    print(f"\nCombinatorial structure:")
    print(f"  Vertices: {best_hull_info['n_vertices']}")
    print(f"  Triangular faces: {best_hull_info['n_faces']}")
    print(f"  Edges: {best_hull_info['n_edges']}")
    print(f"  Vertex degrees: {best_hull_info['vertex_degrees']}")
    print(f"  Classification: {best_hull_info['polyhedron_type']}")
    
    # Print configuration details
    print(f"\nVertex configuration:")
    for i, z in enumerate(best_complex_points):
        print(f"  z[{i}] = {z.real:.4f} + {z.imag:.4f}i")
    
    # Save configuration
    np.savez('optimal_7vertex.npz', 
             params=best_params,
             volume=best_volume,
             hull_info=best_hull_info,
             complex_points=best_complex_points)
    print(f"\nConfiguration saved to optimal_7vertex.npz")

# Compare with regular octahedron
regular_octahedron_vol = 5.3333  # Approximate
print(f"\nComparison:")
print(f"  Regular octahedron (6 vertices): ≈ {regular_octahedron_vol:.4f}")
print(f"  Optimal 7-vertex configuration: {best_volume:.6f}")
if best_volume > regular_octahedron_vol:
    print(f"  ✓ 7-vertex configuration exceeds regular octahedron by {(best_volume/regular_octahedron_vol - 1)*100:.1f}%")
else:
    print(f"  Regular octahedron is {(regular_octahedron_vol/best_volume - 1)*100:.1f}% larger")