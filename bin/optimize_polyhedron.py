#!/usr/bin/env python3
"""
General-purpose wrapper for optimizing ideal polyhedron volumes.

Usage:
    python bin/optimize_polyhedron.py --vertices 7 --trials 10
    python bin/optimize_polyhedron.py --vertices 12 --trials 20 --output custom_name.json
"""

import argparse
import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import sys

from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    triangle_volume_from_points_torch,
)
from scipy.optimize import differential_evolution


def spherical_to_complex(theta, phi):
    """Convert spherical coordinates to complex via stereographic projection."""
    return np.tan(theta/2) * np.exp(1j * phi)


def compute_volume(params, n_vertices):
    """
    Compute volume for a polyhedron with n_vertices.

    First 3 vertices are fixed to break symmetry:
    - z1 = 0
    - z2 = 1
    - z3 = i

    Remaining (n_vertices - 3) vertices are parameterized by spherical coords.
    """
    # Fixed vertices
    complex_points = [complex(0, 0), complex(1, 0), complex(0, 1)]

    # Parameterized vertices (2 params each: theta, phi)
    n_params = n_vertices - 3
    for i in range(n_params):
        theta = params[2*i]
        phi = params[2*i + 1]
        z = spherical_to_complex(theta, phi)
        complex_points.append(z)

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
            vol = triangle_volume_from_points_torch(
                Z_torch[i], Z_torch[j], Z_torch[k], series_terms=96
            )
            total_volume += vol.item()
        except:
            return 1000.0  # Penalty for invalid configuration

    return -total_volume  # Negative for minimization


def analyze_structure(Z_np, idx):
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

    # Euler characteristic check
    euler_char = n_vertices - n_edges + n_faces

    return {
        'n_vertices': n_vertices,
        'n_faces': n_faces,
        'n_edges': n_edges,
        'euler_characteristic': euler_char,
        'vertex_degrees': sorted(vertex_degrees),
    }


def reconstruct_vertices(params, n_vertices):
    """Reconstruct complex vertices from parameters."""
    complex_points = [complex(0, 0), complex(1, 0), complex(0, 1)]

    n_params = n_vertices - 3
    for i in range(n_params):
        theta = params[2*i]
        phi = params[2*i + 1]
        z = spherical_to_complex(theta, phi)
        complex_points.append(z)

    return np.array(complex_points, dtype=np.complex128)


def main():
    parser = argparse.ArgumentParser(
        description='Optimize ideal polyhedron volumes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --vertices 7 --trials 10
  %(prog)s --vertices 12 --trials 20 --maxiter 300
  %(prog)s --vertices 20 --trials 5 --output results/data/my_20vertex.json
        """
    )

    parser.add_argument('--vertices', '-v', type=int, required=True,
                        help='Number of vertices (must be >= 4)')
    parser.add_argument('--trials', '-t', type=int, default=10,
                        help='Number of optimization trials (default: 10)')
    parser.add_argument('--maxiter', '-m', type=int, default=200,
                        help='Max iterations per trial (default: 200)')
    parser.add_argument('--popsize', '-p', type=int, default=15,
                        help='Population size for differential evolution (default: 15)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON file (default: results/data/{n}vertex_optimization_TIMESTAMP.json)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed base (default: 42)')

    args = parser.parse_args()

    if args.vertices < 4:
        print("Error: Number of vertices must be at least 4")
        sys.exit(1)

    # Setup output file
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/data/{args.vertices}vertex_optimization_{timestamp}.json"
    else:
        output_file = args.output

    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Calculate number of parameters (2 per free vertex)
    n_free_vertices = args.vertices - 3
    n_params = n_free_vertices * 2
    bounds = [(0.1, np.pi - 0.1), (0, 2*np.pi)] * n_free_vertices

    print("=" * 70)
    print(f"Ideal Polyhedron Volume Optimization")
    print("=" * 70)
    print(f"Vertices:       {args.vertices}")
    print(f"Free vertices:  {n_free_vertices} (parameterized)")
    print(f"Parameters:     {n_params} (spherical coordinates)")
    print(f"Trials:         {args.trials}")
    print(f"Max iterations: {args.maxiter}")
    print(f"Population:     {args.popsize}")
    print(f"Output:         {output_file}")
    print("=" * 70)

    best_volume = 0
    best_params = None
    all_results = []

    for trial in range(args.trials):
        print(f"\nTrial {trial + 1}/{args.trials}...")

        result = differential_evolution(
            lambda p: compute_volume(p, args.vertices),
            bounds,
            maxiter=args.maxiter,
            popsize=args.popsize,
            seed=args.seed + trial,
            polish=True,
            disp=False
        )

        volume = -result.fun
        print(f"  Volume: {volume:.8f}")
        print(f"  Success: {result.success}")
        print(f"  Iterations: {result.nit}")

        all_results.append({
            'trial': trial + 1,
            'volume': float(volume),
            'params': result.x.tolist(),
            'success': bool(result.success),
            'iterations': int(result.nit),
            'function_evals': int(result.nfev)
        })

        if volume > best_volume:
            best_volume = volume
            best_params = result.x
            print(f"  → NEW BEST!")

    # Analyze best configuration
    print("\n" + "=" * 70)
    print("BEST RESULT:")
    print("=" * 70)
    print(f"Volume: {best_volume:.10f}")

    # Reconstruct and analyze
    Z_np = reconstruct_vertices(best_params, args.vertices)
    idx = delaunay_triangulation_indices(Z_np)
    structure = analyze_structure(Z_np, idx)

    print(f"\nCombinatorial structure:")
    print(f"  Vertices:    {structure['n_vertices']}")
    print(f"  Edges:       {structure['n_edges']}")
    print(f"  Faces:       {structure['n_faces']}")
    print(f"  Euler char:  {structure['euler_characteristic']} (should be 2 for sphere)")
    print(f"  Vertex degrees: {structure['vertex_degrees']}")

    # Save results
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_vertices': args.vertices,
            'n_trials': args.trials,
            'maxiter': args.maxiter,
            'popsize': args.popsize,
            'seed_base': args.seed,
        },
        'best': {
            'volume': float(best_volume),
            'params': best_params.tolist(),
            'vertices_real': Z_np.real.tolist(),
            'vertices_imag': Z_np.imag.tolist(),
            'structure': structure,
            'triangulation': [list(map(int, tri)) for tri in idx],
        },
        'all_trials': all_results,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()
