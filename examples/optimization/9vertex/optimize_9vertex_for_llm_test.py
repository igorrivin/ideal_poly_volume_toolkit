#!/usr/bin/env python3
"""
Find optimal 9-vertex ideal polyhedron for LLM benchmark test.

9 vertices = 0, 1, ∞ + 6 free vertices = 12 real parameters

Uses Bloch-Wigner dilogarithm for precise volume computation.
"""

import numpy as np
from scipy.optimize import differential_evolution
from functools import partial
import sys
import os
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay


def compute_volume_objective(params, n_free):
    """Objective function: negative volume for minimization.

    Uses Bloch-Wigner dilogarithm for precise, fast volume computation.
    """
    real_parts = params[:n_free]
    imag_parts = params[n_free:]

    # Fixed vertices: 0, 1 (∞ is implicit in the triangulation)
    z_fixed = np.array([0.0 + 0.0j, 1.0 + 0.0j])

    # Free vertices
    z_free = np.array([complex(r, i) for r, i in zip(real_parts, imag_parts)])

    # Combine
    z_all = np.concatenate([z_fixed, z_free])

    try:
        # Use Bloch-Wigner dilogarithm for precise computation
        vol = ideal_poly_volume_via_delaunay(z_all, use_bloch_wigner=True)
        return -vol  # Negative for maximization
    except:
        return 1e10  # Penalty for invalid configurations


def main():
    print("=" * 70)
    print("Optimizing 9-vertex ideal polyhedron (6 free + 0,1,∞)")
    print("=" * 70)

    n_free_vertices = 6  # 9 total = 0, 1, ∞ + 6 free
    n_params = n_free_vertices * 2
    n_trials = 10  # More trials for better optimization

    # Bounds for optimization
    bounds = [(-2, 2)] * n_params

    best_volume = -np.inf
    best_config = None
    best_seed = None

    max_workers = min(32, os.cpu_count() or 1)

    # Create picklable objective function
    objective_func = partial(compute_volume_objective, n_free=n_free_vertices)

    for trial in range(n_trials):
        seed = trial
        print(f"\nTrial {trial + 1}/{n_trials} (seed={seed})...", flush=True)

        result = differential_evolution(
            objective_func,
            bounds,
            maxiter=300,  # Reduced from 500
            popsize=15,   # Reduced from 20
            seed=seed,
            polish=True,
            disp=False,
            workers=max_workers,  # Use multiple cores
            updating='deferred'
        )

        volume = -result.fun
        print(f"  Volume: {volume:.8f}")

        if volume > best_volume:
            best_volume = volume
            best_config = result.x
            best_seed = seed
            print(f"  ✓ New best!")

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Best volume: {best_volume:.8f}")
    print(f"Best seed: {best_seed}")

    # Parse best configuration
    n_free = len(best_config) // 2
    real_parts = best_config[:n_free]
    imag_parts = best_config[n_free:]

    z_fixed = np.array([0.0 + 0.0j, 1.0 + 0.0j])
    z_free = np.array([complex(r, i) for r, i in zip(real_parts, imag_parts)])
    z_all = np.concatenate([z_fixed, z_free])

    print("\nVertices:")
    print("  Fixed: 0, 1, ∞")
    print("  Free:")
    for i, z in enumerate(z_free):
        print(f"    z{i+2}: {z.real:.6f} + {z.imag:.6f}i")

    # Save results
    result_data = {
        "metadata": {
            "description": "Optimal 9-vertex ideal polyhedron for LLM benchmark",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_vertices": 9,
            "n_trials": n_trials,
            "best_seed": int(best_seed),
            "computation_method": "Bloch-Wigner dilogarithm"
        },
        "optimal_configuration": {
            "volume": float(best_volume),
            "vertices": {
                "fixed": [
                    {"real": 0.0, "imag": 0.0},
                    {"real": 1.0, "imag": 0.0}
                ],
                "free": [
                    {"real": float(z.real), "imag": float(z.imag)}
                    for z in z_free
                ],
                "infinity": "implicit"
            },
            "vertices_flat": {
                "real_parts": [float(z.real) for z in z_all],
                "imag_parts": [float(z.imag) for z in z_all]
            }
        }
    }

    # Save to results directory
    os.makedirs("results/data", exist_ok=True)
    output_file = "results/data/9vertex_optimal_for_llm_test.json"

    with open(output_file, 'w') as f:
        json.dump(result_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
