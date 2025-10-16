#!/usr/bin/env python3
"""
Background optimization to find maximal 20-vertex ideal polyhedron.
The regular dodecahedron (20 vertices) is NOT arithmetic, but the maximal
configuration might be!

Run with: nohup python optimize_20vertex_background.py > 20vertex_log.txt 2>&1 &
"""

import numpy as np
import torch
import torch.optim as optim
from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay
import json
from datetime import datetime
import time
import signal
import sys
import os

# Global variables for graceful shutdown
should_exit = False
best_volume = 0.0
best_config = None

def signal_handler(sig, frame):
    global should_exit
    print(f"\nReceived signal {sig}. Saving current best result and exiting...")
    should_exit = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def optimize_20vertex_volume(seed=42, max_iterations=10000, lr=0.1):
    """Optimize volume of 20-vertex ideal polyhedron."""
    global best_volume, best_config
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*70}")
    print(f"Starting optimization with seed {seed} at {datetime.now()}")
    print(f"{'='*70}")
    
    # Known volume of regular dodecahedron (20 vertices)
    dodecahedron_vol = 3 * 10.74350778  # Approximately, using Milnor's formula
    print(f"Regular dodecahedron volume (approx): {dodecahedron_vol:.6f}")
    
    # Initialize 17 free vertices (3 fixed at 0, 1, infinity)
    # Start with vertices roughly on a circle but with some randomness
    n_free = 17
    angles = 2 * np.pi * np.arange(n_free) / n_free
    radii = 0.5 + 0.3 * np.random.randn(n_free)
    
    # Add more variation to break symmetry
    angles += 0.1 * np.random.randn(n_free)
    
    real_parts = radii * np.cos(angles)
    imag_parts = radii * np.sin(angles)
    
    # Add some points inside and outside
    for i in range(5):
        idx = np.random.randint(n_free)
        real_parts[idx] += np.random.randn() * 0.3
        imag_parts[idx] += np.random.randn() * 0.3
    
    z_real = torch.tensor(real_parts, dtype=torch.float32, requires_grad=True)
    z_imag = torch.tensor(imag_parts, dtype=torch.float32, requires_grad=True)
    
    # Optimizer with adaptive learning rate
    optimizer = optim.Adam([z_real, z_imag], lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 
                                                     patience=200, factor=0.5)
    
    # Track progress
    history = []
    last_save_time = time.time()
    save_interval = 60  # Save every minute
    
    for iteration in range(max_iterations):
        if should_exit:
            break
            
        optimizer.zero_grad()
        
        # Build vertex list
        vertices = [torch.tensor(0.0+0j, dtype=torch.complex64),
                   torch.tensor(1.0+0j, dtype=torch.complex64)]
        
        # Add free vertices
        for i in range(n_free):
            vertices.append(torch.complex(z_real[i], z_imag[i]))
        
        vertices = torch.stack(vertices)
        
        # Compute volume
        volume = ideal_poly_volume_via_delaunay(vertices, mode='fast', series_terms=200)
        
        if torch.isfinite(volume) and volume > 0:
            # Negative because we want to maximize
            loss = -volume
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([z_real, z_imag], max_norm=1.0)
            
            optimizer.step()
            
            current_vol = volume.item()
            scheduler.step(current_vol)
            
            # Track best configuration
            if current_vol > best_volume:
                best_volume = current_vol
                best_config = {
                    'volume': current_vol,
                    'vertices': vertices.detach().cpu().numpy().tolist(),
                    'seed': seed,
                    'iteration': iteration,
                    'timestamp': str(datetime.now())
                }
                print(f"New best! Iteration {iteration}: volume = {current_vol:.8f} "
                      f"(ratio to dodecahedron: {current_vol/dodecahedron_vol:.6f})")
            
            # Progress update
            if iteration % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Iteration {iteration}: volume = {current_vol:.8f}, "
                      f"lr = {current_lr:.6f}")
            
            history.append(current_vol)
            
            # Periodic save
            current_time = time.time()
            if current_time - last_save_time > save_interval:
                save_intermediate_result(best_config, history)
                last_save_time = current_time
                
    return best_config, history

def save_intermediate_result(config, history):
    """Save intermediate results to file."""
    if config is None:
        return
        
    result = {
        'best_configuration': config,
        'optimization_history': history[-1000:],  # Last 1000 values
        'status': 'in_progress'
    }
    
    filename = f'20vertex_optimization_intermediate.json'
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved intermediate result to {filename}")

def run_multiple_seeds(n_seeds=10, max_iterations=10000):
    """Run optimization with multiple random seeds."""
    global best_volume, best_config
    
    all_results = []
    
    for trial in range(n_seeds):
        if should_exit:
            break
            
        seed = trial * 17  # Use different seeds
        config, history = optimize_20vertex_volume(seed=seed, 
                                                  max_iterations=max_iterations)
        
        if config is not None:
            all_results.append({
                'seed': seed,
                'final_volume': config['volume'],
                'configuration': config,
                'converged_iteration': len(history)
            })
    
    # Save final results
    final_result = {
        'best_configuration': best_config,
        'all_trials': all_results,
        'summary': {
            'num_trials': len(all_results),
            'best_volume': best_volume,
            'dodecahedron_volume': 32.23052334,  # More precise value
            'ratio_to_dodecahedron': best_volume / 32.23052334 if best_volume > 0 else 0
        },
        'timestamp': str(datetime.now()),
        'status': 'completed' if not should_exit else 'interrupted'
    }
    
    with open('20vertex_optimization_final.json', 'w') as f:
        json.dump(final_result, f, indent=2)
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE!")
    print("="*70)
    print(f"Best volume found: {best_volume:.8f}")
    print(f"Ratio to dodecahedron: {best_volume/32.23052334:.6f}")
    print(f"Results saved to 20vertex_optimization_final.json")

if __name__ == "__main__":
    print("Starting 20-vertex ideal polyhedron optimization")
    print("This will run in the background and save results periodically.")
    print("To stop gracefully, use: kill -SIGTERM <pid>")
    print(f"Process ID: {os.getpid()}")
    
    # Import os for getpid
    import os
    
    # Run optimization
    run_multiple_seeds(n_seeds=10, max_iterations=10000)
    
    print("\nOptimization finished!")