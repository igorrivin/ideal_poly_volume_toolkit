#!/usr/bin/env python3
"""
Quick search for local maxima in 7-vertex configurations.
"""

import numpy as np
import torch
from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    triangle_volume_from_points_torch,
)
from scipy.optimize import differential_evolution

def compute_volume(params):
    Z = [complex(0, 0), complex(1, 0)]
    for i in range(4):
        theta = params[2*i]
        phi = params[2*i + 1]
        z = np.tan(theta/2) * np.exp(1j * phi)
        Z.append(z)
    
    Z_np = np.array(Z, dtype=np.complex128)
    
    try:
        idx = delaunay_triangulation_indices(Z_np)
        vertices_used = set()
        for (i, j, k) in idx:
            vertices_used.update([i, j, k])
        
        if len(vertices_used) < 7:
            return 1000.0
        
        Z_torch = torch.tensor(Z_np, dtype=torch.complex128)
        total = 0
        for (i, j, k) in idx:
            vol = triangle_volume_from_points_torch(Z_torch[i], Z_torch[j], Z_torch[k], series_terms=96)
            total += vol.item()
        
        return -total
    except:
        return 1000.0

def get_degree_signature(params):
    Z = [complex(0, 0), complex(1, 0)]
    for i in range(4):
        theta = params[2*i]
        phi = params[2*i + 1]
        z = np.tan(theta/2) * np.exp(1j * phi)
        Z.append(z)
    
    Z_np = np.array(Z, dtype=np.complex128)
    idx = delaunay_triangulation_indices(Z_np)
    
    degrees = [0] * 7
    edges = set()
    for (a, b, c) in idx:
        edges.add(tuple(sorted([a, b])))
        edges.add(tuple(sorted([b, c])))
        edges.add(tuple(sorted([a, c])))
    
    for edge in edges:
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1
    
    return tuple(sorted(degrees))

print("Quick search for 7-vertex local maxima (20 trials)...")

maxima = []
signatures_seen = {}

for trial in range(20):
    print(f"\rTrial {trial+1}/20...", end='', flush=True)
    
    bounds = [(0.1, np.pi-0.1), (0, 2*np.pi)] * 4
    
    result = differential_evolution(
        compute_volume, 
        bounds, 
        maxiter=100,
        popsize=15,
        seed=trial * 42
    )
    
    volume = -result.fun
    
    if volume > 5.0:  # Good solution
        try:
            sig = get_degree_signature(result.x)
            
            if sig not in signatures_seen:
                signatures_seen[sig] = {
                    'volume': volume,
                    'count': 1,
                    'params': result.x
                }
                maxima.append((volume, sig))
            else:
                signatures_seen[sig]['count'] += 1
                if volume > signatures_seen[sig]['volume']:
                    signatures_seen[sig]['volume'] = volume
                    signatures_seen[sig]['params'] = result.x
        except:
            pass

print("\n\nDistinct configurations found:")
print("="*50)

# Sort by volume
maxima.sort(reverse=True)

for i, (vol, sig) in enumerate(maxima):
    count = signatures_seen[sig]['count']
    print(f"\nConfig {i+1}: Volume = {vol:.6f} (found {count} times)")
    print(f"  Degree signature: {sig}")
    print(f"  Ratio to octahedron: {vol/5.333:.3f}")

if len(maxima) == 1:
    print("\nConclusion: Only ONE local maximum found!")
else:
    print(f"\nConclusion: Found {len(maxima)} distinct local maxima!")
    
    # Check if they're arithmetic
    print("\nChecking if maxima might be arithmetic...")
    from ideal_poly_volume_toolkit.examples.six_vertex_distribution import check_arithmeticity
    
    for i, (vol, sig) in enumerate(maxima[:3]):  # Check top 3
        params = signatures_seen[sig]['params']
        Z = [complex(0, 0), complex(1, 0)]
        for j in range(4):
            theta = params[2*j]
            phi = params[2*j + 1]
            z = np.tan(theta/2) * np.exp(1j * phi)
            Z.append(z)
        
        is_arith = check_arithmeticity(Z)
        print(f"  Config {i+1}: {'Arithmetic' if is_arith else 'Non-arithmetic'}")