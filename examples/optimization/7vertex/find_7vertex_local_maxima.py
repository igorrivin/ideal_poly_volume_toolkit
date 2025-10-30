#!/usr/bin/env python3
"""
Search for multiple local maxima in 7-vertex ideal polyhedra configurations.
"""

import numpy as np
import torch
from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    triangle_volume_from_points_torch,
)
from scipy.optimize import differential_evolution
import json

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

def analyze_configuration(params):
    """Analyze the structure of a configuration."""
    Z = [complex(0, 0), complex(1, 0)]
    for i in range(4):
        theta = params[2*i]
        phi = params[2*i + 1]
        z = np.tan(theta/2) * np.exp(1j * phi)
        Z.append(z)
    
    Z_np = np.array(Z, dtype=np.complex128)
    idx = delaunay_triangulation_indices(Z_np)
    
    # Vertex degrees
    degrees = [0] * 7
    edges = set()
    for (a, b, c) in idx:
        edges.add(tuple(sorted([a, b])))
        edges.add(tuple(sorted([b, c])))
        edges.add(tuple(sorted([a, c])))
    
    for edge in edges:
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1
    
    return {
        'n_faces': len(idx),
        'n_edges': len(edges),
        'degrees': sorted(degrees),
        'degree_signature': tuple(sorted(degrees))
    }

# Run many optimizations with different random seeds
print("Searching for local maxima in 7-vertex configurations...")
print("="*70)

local_maxima = []
seen_signatures = set()

n_trials = 50
for trial in range(n_trials):
    if trial % 10 == 0:
        print(f"\nProgress: {trial}/{n_trials} trials...")
    
    bounds = [(0.1, np.pi-0.1), (0, 2*np.pi)] * 4
    
    result = differential_evolution(
        compute_volume, 
        bounds, 
        maxiter=200,
        popsize=20,
        polish=True,
        seed=trial * 137  # Different seed pattern
    )
    
    volume = -result.fun
    
    if volume < 5.0:  # Skip bad solutions
        continue
    
    # Analyze structure
    try:
        analysis = analyze_configuration(result.x)
        sig = analysis['degree_signature']
        
        # Check if this is a new type of configuration
        is_new = True
        for prev in local_maxima:
            if prev['signature'] == sig and abs(prev['volume'] - volume) < 0.01:
                is_new = False
                break
        
        if is_new:
            local_maxima.append({
                'volume': volume,
                'params': result.x.tolist(),
                'signature': sig,
                'n_faces': analysis['n_faces'],
                'n_edges': analysis['n_edges']
            })
            print(f"  Found local maximum: volume={volume:.6f}, degrees={sig}")
    except:
        pass

# Sort by volume
local_maxima.sort(key=lambda x: x['volume'], reverse=True)

print("\n" + "="*70)
print("SUMMARY OF LOCAL MAXIMA:")
print("="*70)

for i, config in enumerate(local_maxima):
    print(f"\nConfiguration {i+1}:")
    print(f"  Volume: {config['volume']:.6f}")
    print(f"  Faces: {config['n_faces']} (+ face at infinity = {config['n_faces']+1})")
    print(f"  Edges: {config['n_edges']}")
    print(f"  Degree signature: {config['signature']}")
    print(f"  Ratio to octahedron: {config['volume']/5.333:.3f}")

if len(local_maxima) > 1:
    print(f"\nFound {len(local_maxima)} distinct local maxima!")
else:
    print(f"\nOnly one local maximum found.")

# Save results
with open('7vertex_local_maxima.json', 'w') as f:
    json.dump(local_maxima, f, indent=2)
print(f"\nResults saved to 7vertex_local_maxima.json")