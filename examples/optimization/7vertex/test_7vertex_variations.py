#!/usr/bin/env python3
"""
Test a few 7-vertex variations to see if there are multiple maxima.
"""

import numpy as np
import torch
from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    triangle_volume_from_points_torch,
)

def evaluate_config(Z):
    """Evaluate volume and structure of a configuration."""
    Z_np = np.array(Z, dtype=np.complex128)
    
    try:
        idx = delaunay_triangulation_indices(Z_np)
        
        # Check vertices used
        vertices_used = set()
        for (i, j, k) in idx:
            vertices_used.update([i, j, k])
        
        if len(vertices_used) < 7:
            return None, None
        
        # Compute volume
        Z_torch = torch.tensor(Z_np, dtype=torch.complex128)
        total = 0
        for (i, j, k) in idx:
            vol = triangle_volume_from_points_torch(Z_torch[i], Z_torch[j], Z_torch[k], series_terms=96)
            total += vol.item()
        
        # Compute degrees
        degrees = [0] * 7
        edges = set()
        for (a, b, c) in idx:
            edges.add(tuple(sorted([a, b])))
            edges.add(tuple(sorted([b, c])))
            edges.add(tuple(sorted([a, c])))
        
        for edge in edges:
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1
        
        return total, tuple(sorted(degrees))
    except:
        return None, None

# Test several hand-crafted configurations
configs = []

# Config 1: Previous optimum
Z1 = [complex(0, 0), complex(1, 0),
      complex(-0.9959, 0.1942), complex(-0.4018, -0.9331),
      complex(0.8441, -1.2772), complex(0.3669, -0.5609)]
configs.append(("Previous optimum", Z1))

# Config 2: Regular hexagon + center
angles = np.linspace(0, 2*np.pi, 7)[:-1]
Z2 = [complex(0, 0), complex(1, 0)]
for a in angles[2:]:
    Z2.append(complex(np.cos(a), np.sin(a)))
configs.append(("Hexagon + center", Z2))

# Config 3: Two triangles
Z3 = [complex(0, 0), complex(1, 0),
      complex(0.5, 0.5), complex(-1, 0), complex(0, -1),
      complex(-0.5, -0.5)]
configs.append(("Two triangles", Z3))

# Config 4: Star pattern
Z4 = [complex(0, 0), complex(1, 0)]
for r, a in [(2, 0.5), (2, 2.5), (0.5, 4), (0.5, 5.5), (1.5, 1.0)]:
    Z4.append(complex(r*np.cos(a), r*np.sin(a)))
configs.append(("Star pattern", Z4))

print("Testing different 7-vertex configurations...")
print("="*50)

results = []
for name, Z in configs:
    vol, sig = evaluate_config(Z)
    if vol is not None:
        results.append((vol, sig, name))
        print(f"\n{name}:")
        print(f"  Volume: {vol:.6f}")
        print(f"  Degree signature: {sig}")
        print(f"  Ratio to octahedron: {vol/5.333:.3f}")

# Group by signature
print("\n" + "="*50)
print("Grouping by degree signature:")
sig_groups = {}
for vol, sig, name in results:
    if sig not in sig_groups:
        sig_groups[sig] = []
    sig_groups[sig].append((vol, name))

for sig, items in sig_groups.items():
    print(f"\nSignature {sig}:")
    for vol, name in sorted(items, reverse=True):
        print(f"  {name}: {vol:.6f}")

# Now try a few random perturbations of the best config
print("\n" + "="*50)
print("Testing small perturbations of the optimum...")

best_Z = Z1
best_vol = results[0][0]

for i in range(5):
    # Perturb slightly
    Z_perturbed = best_Z[:2]  # Keep first 2 fixed (0 and 1)
    for z in best_Z[2:]:
        noise = 0.1 * (np.random.randn() + 1j*np.random.randn())
        Z_perturbed.append(z + noise)
    
    vol, sig = evaluate_config(Z_perturbed)
    if vol is not None:
        print(f"\nPerturbation {i+1}: volume={vol:.6f}, signature={sig}")
        if vol > best_vol * 0.99:  # Within 1% of optimum
            print("  -> Near optimal!")

print("\n" + "="*50)
print("CONCLUSION:")
unique_sigs = len(sig_groups)
if unique_sigs == 1:
    print("All good configurations have the same degree signature!")
else:
    print(f"Found {unique_sigs} different degree signatures.")