#!/usr/bin/env python3
"""
Comprehensive optimization for maximal volume ideal polyhedra.

This script runs extensive optimization for each vertex count n=4 to n=20,
using many restarts to ensure we find the true global maximum.

For paper preparation, we need verified maximal configurations for all
small vertex counts, with statistics showing convergence confidence.
"""

import numpy as np
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from functools import partial
from scipy.optimize import differential_evolution

sys.path.insert(0, '/home/igor/devel/ideal_poly_volume_toolkit')

from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    triangle_volume_bloch_wigner,
)

import torch

DEVICE = torch.device('cpu')


def spherical_to_complex(theta, phi):
    """Convert spherical coordinates to complex via stereographic projection."""
    return np.tan(theta/2) * np.exp(1j * phi)


def compute_volume(params, n_vertices):
    """Compute volume for a polyhedron with n_vertices."""
    complex_points = [complex(0, 0), complex(1, 0)]
    n_params = n_vertices - 3

    for i in range(n_params):
        theta = params[2*i]
        phi = params[2*i + 1]
        z = spherical_to_complex(theta, phi)
        complex_points.append(z)

    Z_np = np.array(complex_points, dtype=np.complex128)

    try:
        idx = delaunay_triangulation_indices(Z_np)
    except:
        return 1000.0

    Z_torch = torch.tensor(Z_np, dtype=torch.complex128, device=DEVICE)

    total_volume = 0
    for (i, j, k) in idx:
        try:
            vol = triangle_volume_bloch_wigner(
                Z_torch[i], Z_torch[j], Z_torch[k]
            )
            total_volume += vol.item()
        except:
            return 1000.0

    return -total_volume


def run_comprehensive_optimization(
    n_vertices,
    n_trials=100,
    max_iter=500,
    pop_size=15,
    seed=42,
    verbose=True
):
    """
    Run comprehensive optimization for a given vertex count.

    Args:
        n_vertices: Number of vertices (4 to 100)
        n_trials: Number of independent optimization trials
        max_iter: Maximum iterations per trial
        pop_size: Population size for differential evolution
        seed: Base random seed
        verbose: Print progress

    Returns:
        dict with best configuration and statistics
    """
    n_cpus = os.cpu_count()
    max_workers = min(32, n_cpus) if n_cpus else 1

    n_free_vertices = n_vertices - 3
    n_params = n_free_vertices * 2
    bounds = [(0.1, np.pi - 0.1), (0, 2*np.pi)] * n_free_vertices

    # Adaptive settings for large vertex counts
    adaptive_popsize = min(pop_size, max(10, 15 - (n_vertices - 7) // 3))

    best_volume = 0
    best_params = None
    all_volumes = []

    objective_func = partial(compute_volume, n_vertices=n_vertices)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Optimizing n={n_vertices} vertices with {n_trials} trials")
        print(f"Using {max_workers} workers, popsize={adaptive_popsize}")
        print(f"{'='*60}")

    for trial in range(n_trials):
        result = differential_evolution(
            objective_func,
            bounds,
            maxiter=max_iter,
            popsize=adaptive_popsize,
            seed=seed + trial,
            polish=True,
            disp=False,
            workers=max_workers,
            updating='deferred'
        )

        volume = -result.fun
        all_volumes.append(volume)

        if volume > best_volume:
            best_volume = volume
            best_params = result.x.copy()
            if verbose:
                print(f"  Trial {trial+1:3d}: New best volume = {volume:.8f}")
        elif verbose and (trial + 1) % 20 == 0:
            print(f"  Trial {trial+1:3d}: Current best = {best_volume:.8f}")

    # Reconstruct best configuration
    complex_points = [complex(0, 0), complex(1, 0)]
    for i in range(n_free_vertices):
        theta = best_params[2*i]
        phi = best_params[2*i + 1]
        z = spherical_to_complex(theta, phi)
        complex_points.append(z)

    Z_np = np.array(complex_points, dtype=np.complex128)
    idx = delaunay_triangulation_indices(Z_np)

    # Compute statistics
    all_volumes = np.array(all_volumes)
    stats = {
        'n_vertices': n_vertices,
        'n_faces': len(idx),
        'n_trials': n_trials,
        'best_volume': float(best_volume),
        'mean_volume': float(np.mean(all_volumes)),
        'std_volume': float(np.std(all_volumes)),
        'min_volume': float(np.min(all_volumes)),
        'max_volume': float(np.max(all_volumes)),
        'convergence_ratio': float(np.sum(all_volumes >= best_volume * 0.9999) / len(all_volumes)),
    }

    if verbose:
        print(f"\n  Results for n={n_vertices}:")
        print(f"    Best volume:   {stats['best_volume']:.8f}")
        print(f"    Mean volume:   {stats['mean_volume']:.8f}")
        print(f"    Std dev:       {stats['std_volume']:.8f}")
        print(f"    Convergence:   {stats['convergence_ratio']*100:.1f}% reached near-optimal")

    return {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_trials': n_trials,
            'max_iter': max_iter,
            'pop_size': adaptive_popsize,
            'seed': seed,
        },
        'best': {
            'volume': stats['best_volume'],
            'params': best_params.tolist(),
            'vertices_real': Z_np.real.tolist(),
            'vertices_imag': Z_np.imag.tolist(),
            'triangulation': [[int(v) for v in t] for t in idx],
        },
        'statistics': stats,
        'all_volumes': all_volumes.tolist(),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive optimization for maximal volume ideal polyhedra"
    )
    parser.add_argument(
        "--min-vertices", type=int, default=4,
        help="Minimum number of vertices (default: 4)"
    )
    parser.add_argument(
        "--max-vertices", type=int, default=20,
        help="Maximum number of vertices (default: 20)"
    )
    parser.add_argument(
        "--trials", type=int, default=100,
        help="Number of optimization trials per vertex count (default: 100)"
    )
    parser.add_argument(
        "--max-iter", type=int, default=500,
        help="Maximum iterations per trial (default: 500)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/data/comprehensive",
        help="Output directory (default: results/data/comprehensive)"
    )
    parser.add_argument(
        "--single", type=int, default=None,
        help="Run only for a single vertex count"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("COMPREHENSIVE OPTIMIZATION FOR MAXIMAL VOLUME IDEAL POLYHEDRA")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Trials per vertex count: {args.trials}")
    print(f"Max iterations per trial: {args.max_iter}")

    # Determine vertex counts to run
    if args.single is not None:
        vertex_counts = [args.single]
    else:
        vertex_counts = range(args.min_vertices, args.max_vertices + 1)

    results_summary = {}

    for n in vertex_counts:
        result = run_comprehensive_optimization(
            n_vertices=n,
            n_trials=args.trials,
            max_iter=args.max_iter,
            seed=args.seed,
            verbose=True
        )

        # Save individual result
        output_file = output_dir / f"{n}vertex_comprehensive.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        results_summary[n] = {
            'volume': result['statistics']['best_volume'],
            'n_faces': result['statistics']['n_faces'],
            'convergence': result['statistics']['convergence_ratio'],
        }

        print(f"  Saved to: {output_file}")

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'trials': args.trials,
                'max_iter': args.max_iter,
            },
            'results': results_summary,
        }, f, indent=2)

    print("\n" + "="*70)
    print("SUMMARY OF MAXIMAL VOLUMES")
    print("="*70)
    print(f"{'n':>4}  {'Volume':>14}  {'Faces':>6}  {'Convergence':>12}")
    print("-" * 42)

    for n in sorted(results_summary.keys()):
        r = results_summary[n]
        print(f"{n:>4}  {r['volume']:>14.8f}  {r['n_faces']:>6}  {r['convergence']*100:>11.1f}%")

    print("="*70)
    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
