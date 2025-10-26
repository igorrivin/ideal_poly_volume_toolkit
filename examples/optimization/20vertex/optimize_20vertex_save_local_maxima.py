#!/usr/bin/env python3
"""
Modified 20-vertex optimization that saves all local maxima for arithmeticity checking.
Based on the conjecture that local maxima are more likely to be arithmetic.
"""

import numpy as np
from scipy.optimize import minimize
from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay
import json
from datetime import datetime
import os

class LocalMaximaTracker:
    def __init__(self, volume_threshold=26.0, min_improvement=0.001):
        """
        Track local maxima during optimization.
        volume_threshold: only save configurations with volume above this
        min_improvement: minimum difference to consider a new local maximum
        """
        self.local_maxima = []
        self.volume_threshold = volume_threshold
        self.min_improvement = min_improvement
        self.last_best = 0
        self.stagnation_count = 0
        self.last_saved_volume = 0
        
    def check_and_save(self, vertices, volume, evaluations):
        """Check if current configuration is a local maximum worth saving."""
        # Check if we've found a new local maximum
        if volume > self.volume_threshold and volume > self.last_saved_volume + self.min_improvement:
            # Check if optimization has stagnated (potential local max)
            if self.stagnation_count > 20 or len(self.local_maxima) == 0:
                config = {
                    'volume': float(volume),
                    'evaluations': evaluations,
                    'vertices': {f'z{i}': {'real': float(v.real), 'imag': float(v.imag)} 
                                for i, v in enumerate(vertices)},
                    'timestamp': str(datetime.now())
                }
                self.local_maxima.append(config)
                self.last_saved_volume = volume
                self.stagnation_count = 0
                
                print(f"*** Saved local maximum #{len(self.local_maxima)}: volume = {volume:.8f}")
                self.save_to_file()
                return True
        
        # Track stagnation
        if abs(volume - self.last_best) < 0.0001:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
            self.last_best = volume
            
        return False
    
    def save_to_file(self):
        """Save all local maxima to JSON file."""
        data = {
            'local_maxima_count': len(self.local_maxima),
            'dodecahedron_volume': 32.23052334,
            'configurations': sorted(self.local_maxima, key=lambda x: x['volume'], reverse=True),
            'timestamp': str(datetime.now())
        }
        
        with open('20vertex_local_maxima.json', 'w') as f:
            json.dump(data, f, indent=2)

# Global tracker
tracker = LocalMaximaTracker()

def objective_with_tracking(x, verbose_freq=100):
    """Objective function that also tracks local maxima."""
    n_free = len(x) // 2
    
    # Build vertices
    vertices = [0+0j, 1+0j]
    for i in range(n_free):
        vertices.append(complex(x[2*i], x[2*i+1]))
    vertices = np.array(vertices)
    
    try:
        volume = ideal_poly_volume_via_delaunay(vertices, mode='fast', series_terms=200)
        
        # Track progress
        objective_with_tracking.counter += 1
        
        if objective_with_tracking.counter % verbose_freq == 0:
            print(f"  Evaluation {objective_with_tracking.counter}: volume = {volume:.6f}")
        
        # Check for local maximum
        tracker.check_and_save(vertices, volume, objective_with_tracking.counter)
        
        # Track best overall
        if volume > objective_with_tracking.best_volume:
            objective_with_tracking.best_volume = volume
            objective_with_tracking.best_config = vertices.copy()
            print(f"*** New global best! Volume = {volume:.8f} at evaluation {objective_with_tracking.counter}")
        
        return -volume
    except:
        return 1e10

# Initialize counter
objective_with_tracking.counter = 0
objective_with_tracking.best_volume = 0
objective_with_tracking.best_config = None

def multi_start_optimization():
    """Run optimization from multiple starting points."""
    print(f"\n{'='*70}")
    print(f"20-VERTEX OPTIMIZATION WITH LOCAL MAXIMA TRACKING")
    print(f"Started at {datetime.now()}")
    print(f"Process ID: {os.getpid()}")
    print(f"{'='*70}\n")
    
    dodecahedron_vol = 32.23052334
    print(f"Regular dodecahedron volume: {dodecahedron_vol:.8f}")
    print("Tracking all local maxima above volume 26.0\n")
    
    n_free = 17
    
    # Try different types of initializations
    initializations = [
        ("symmetric", lambda: symmetric_init(n_free)),
        ("random_circle", lambda: random_circle_init(n_free)),
        ("random_disk", lambda: random_disk_init(n_free)),
        ("clustered", lambda: clustered_init(n_free)),
        ("golden_spiral", lambda: golden_spiral_init(n_free))
    ]
    
    for trial, (init_type, init_func) in enumerate(initializations):
        print(f"\n{'='*50}")
        print(f"TRIAL {trial + 1}/{len(initializations)}: {init_type} initialization")
        print(f"{'='*50}")
        
        # Reset counter for this trial
        objective_with_tracking.counter = 0
        
        # Get initial configuration
        x0 = init_func()
        
        # Run optimization
        result = minimize(objective_with_tracking, x0, 
                         method='BFGS',
                         options={'maxiter': 1000, 'gtol': 1e-5})
        
        print(f"\nTrial {trial + 1} complete:")
        print(f"  Final volume: {-result.fun:.8f}")
        print(f"  Ratio to dodecahedron: {-result.fun/dodecahedron_vol:.6f}")
        print(f"  Local maxima found: {len(tracker.local_maxima)}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("OPTIMIZATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Total local maxima found: {len(tracker.local_maxima)}")
    if tracker.local_maxima:
        volumes = [lm['volume'] for lm in tracker.local_maxima]
        print(f"Volume range: {min(volumes):.6f} - {max(volumes):.6f}")
        print(f"Best volume: {objective_with_tracking.best_volume:.8f}")
        print(f"Ratio to dodecahedron: {objective_with_tracking.best_volume/dodecahedron_vol:.6f}")
        
        # Check if any exceed dodecahedron
        exceeding = [lm for lm in tracker.local_maxima if lm['volume'] > dodecahedron_vol]
        if exceeding:
            print(f"\n*** {len(exceeding)} configurations EXCEED dodecahedron volume! ***")

# Different initialization strategies
def symmetric_init(n_free):
    """Start near symmetric configuration."""
    angles = 2 * np.pi * np.arange(n_free) / n_free
    x0 = []
    for i in range(n_free):
        r = 1.0 + 0.1 * np.random.randn()
        x0.extend([r * np.cos(angles[i]), r * np.sin(angles[i])])
    return np.array(x0)

def random_circle_init(n_free):
    """Random points on circles of varying radii."""
    x0 = []
    for i in range(n_free):
        angle = 2 * np.pi * np.random.rand()
        radius = 0.5 + 2 * np.random.rand()
        x0.extend([radius * np.cos(angle), radius * np.sin(angle)])
    return np.array(x0)

def random_disk_init(n_free):
    """Random points in disk."""
    x0 = []
    for i in range(n_free):
        angle = 2 * np.pi * np.random.rand()
        radius = 2 * np.sqrt(np.random.rand())  # sqrt for uniform distribution
        x0.extend([radius * np.cos(angle), radius * np.sin(angle)])
    return np.array(x0)

def clustered_init(n_free):
    """Points clustered in groups."""
    x0 = []
    n_clusters = 4
    points_per_cluster = n_free // n_clusters
    
    for c in range(n_clusters):
        center_angle = 2 * np.pi * c / n_clusters
        center = 1.5 * np.array([np.cos(center_angle), np.sin(center_angle)])
        
        for _ in range(points_per_cluster):
            offset = 0.3 * np.random.randn(2)
            point = center + offset
            x0.extend(point.tolist())
    
    # Add remaining points randomly
    for _ in range(n_free - n_clusters * points_per_cluster):
        x0.extend(np.random.randn(2).tolist())
    
    return np.array(x0)

def golden_spiral_init(n_free):
    """Points on golden spiral."""
    golden_ratio = (1 + np.sqrt(5)) / 2
    x0 = []
    
    for i in range(n_free):
        angle = 2 * np.pi * i / golden_ratio
        radius = 0.5 + 0.1 * i
        x0.extend([radius * np.cos(angle), radius * np.sin(angle)])
    
    return np.array(x0)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run optimization
    multi_start_optimization()
    
    print("\nLocal maxima saved to 20vertex_local_maxima.json")
    print("Next step: Check each local maximum for arithmeticity!")