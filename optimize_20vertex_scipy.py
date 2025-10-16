#!/usr/bin/env python3
"""
Background optimization to find maximal 20-vertex ideal polyhedron.
Using scipy optimization instead of PyTorch for simplicity.
"""

import numpy as np
from scipy.optimize import minimize
from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay
import json
from datetime import datetime
import time
import os

def objective(x, verbose_freq=100):
    """Negative volume to minimize (we want to maximize volume)."""
    n_free = len(x) // 2
    
    # Build vertices: 0, 1, and n_free complex points
    vertices = [0+0j, 1+0j]
    for i in range(n_free):
        vertices.append(complex(x[2*i], x[2*i+1]))
    
    vertices = np.array(vertices)
    
    try:
        volume = ideal_poly_volume_via_delaunay(vertices, mode='fast', series_terms=200)
        
        # Track progress
        objective.counter += 1
        if objective.counter % verbose_freq == 0:
            print(f"  Evaluation {objective.counter}: volume = {volume:.6f}")
        
        # Track best
        if volume > objective.best_volume:
            objective.best_volume = volume
            objective.best_config = vertices.copy()
            print(f"*** New best! Volume = {volume:.8f} at evaluation {objective.counter}")
            
            # Save intermediate result
            save_result(objective.best_config, objective.best_volume, 'intermediate')
        
        return -volume  # Negative because we minimize
    except:
        return 1e10  # Large penalty for invalid configurations

# Initialize counter and best tracking
objective.counter = 0
objective.best_volume = 0
objective.best_config = None

def save_result(vertices, volume, status='final'):
    """Save configuration to JSON file."""
    result = {
        'status': status,
        'volume': float(volume),
        'dodecahedron_volume': 32.23052334,
        'ratio_to_dodecahedron': float(volume) / 32.23052334,
        'num_vertices': len(vertices) + 1,  # +1 for infinity
        'vertices': {
            f'z{i}': {'real': float(v.real), 'imag': float(v.imag)} 
            for i, v in enumerate(vertices)
        },
        'timestamp': str(datetime.now()),
        'evaluations': objective.counter
    }
    
    filename = f'20vertex_maximal_{status}.json'
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {filename}")

def optimize_20vertex():
    """Run optimization for 20-vertex polyhedron."""
    print(f"\n{'='*70}")
    print(f"20-VERTEX IDEAL POLYHEDRON OPTIMIZATION")
    print(f"Started at {datetime.now()}")
    print(f"Process ID: {os.getpid()}")
    print(f"{'='*70}\n")
    
    # Regular dodecahedron volume
    dodecahedron_vol = 32.23052334
    print(f"Regular dodecahedron volume: {dodecahedron_vol:.8f}")
    print("Goal: Find configuration with larger volume\n")
    
    # Initialize 17 free points (total 20 with 0, 1, infinity)
    n_free = 17
    
    # Try multiple random initializations
    best_overall_volume = 0
    best_overall_config = None
    
    for trial in range(5):
        print(f"\n{'='*50}")
        print(f"TRIAL {trial + 1}/5")
        print(f"{'='*50}")
        
        # Reset counters for this trial
        objective.counter = 0
        
        # Random initialization
        np.random.seed(trial * 42)
        
        if trial == 0:
            # Start near dodecahedron-like configuration
            angles = 2 * np.pi * np.arange(n_free) / n_free
            radii = 1.0 + 0.2 * np.random.randn(n_free)
            x0 = []
            for i in range(n_free):
                x0.extend([radii[i] * np.cos(angles[i]), 
                          radii[i] * np.sin(angles[i])])
        else:
            # Random start
            x0 = 2 * np.random.randn(2 * n_free)
        
        x0 = np.array(x0)
        
        # Optimize using BFGS
        print(f"Starting optimization from {'near-symmetric' if trial == 0 else 'random'} configuration...")
        
        result = minimize(objective, x0, 
                         method='BFGS',
                         options={'maxiter': 2000, 
                                 'gtol': 1e-6,
                                 'disp': True})
        
        print(f"\nTrial {trial + 1} complete:")
        print(f"  Final volume: {-result.fun:.8f}")
        print(f"  Ratio to dodecahedron: {-result.fun/dodecahedron_vol:.6f}")
        print(f"  Function evaluations: {result.nfev}")
        print(f"  Success: {result.success}")
        
        if -result.fun > best_overall_volume:
            best_overall_volume = -result.fun
            best_overall_config = objective.best_config
    
    # Save final result
    if best_overall_config is not None:
        save_result(best_overall_config, best_overall_volume, 'final')
        
        print(f"\n{'='*70}")
        print("OPTIMIZATION COMPLETE!")
        print(f"{'='*70}")
        print(f"Best volume found: {best_overall_volume:.8f}")
        print(f"Regular dodecahedron: {dodecahedron_vol:.8f}")
        print(f"Ratio: {best_overall_volume/dodecahedron_vol:.6f}")
        
        if best_overall_volume > dodecahedron_vol:
            print("\n*** DODECAHEDRON IS NOT MAXIMAL! ***")
            print(f"Found configuration with {(best_overall_volume/dodecahedron_vol - 1)*100:.2f}% more volume!")
        
        return best_overall_config, best_overall_volume
    
    return None, 0

if __name__ == "__main__":
    # Run optimization
    config, volume = optimize_20vertex()
    
    if config is not None and volume > 32.23052334:
        print("\nNext step: Check if this maximal configuration is arithmetic!")
        print("The regular dodecahedron is NOT arithmetic, but this might be.")