#!/usr/bin/env python3
"""
Empirical estimation of mixing time for the triangulation flip graph.

Experimental design:
1. For n=10, we can enumerate all ~32k triangulations (ground truth)
2. Start from a fixed triangulation
3. Generate samples via k-flip chains (for various k)
4. Compute statistics (realizability, spanning trees, etc.) on samples
5. Compare to true distribution from exhaustive enumeration
6. Determine which k gives convergence to true distribution

This addresses the open problem: What is the mixing time of random walks
on the triangulation flip graph? Conjectured to be n^(6/5).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import networkx as nx
from ideal_poly_volume_toolkit.plantri_interface import find_plantri_executable
from ideal_poly_volume_toolkit.rivin_delaunay import (
    check_delaunay_realizability,
    random_edge_flips
)
from ideal_poly_volume_toolkit.planar_utils import extract_faces_from_planar_embedding
import subprocess
import json
from collections import defaultdict
import matplotlib.pyplot as plt


def count_spanning_trees(triangles):
    """Count spanning trees using Kirchhoff's theorem."""
    G = nx.Graph()
    for tri in triangles:
        v0, v1, v2 = tri
        G.add_edge(v0, v1)
        G.add_edge(v1, v2)
        G.add_edge(v2, v0)

    if len(G.nodes()) <= 1:
        return 1

    L = nx.laplacian_matrix(G).toarray()
    L_reduced = L[1:, 1:]
    det = np.linalg.det(L_reduced)
    return int(round(det))


def load_ground_truth(n_vertices=10):
    """Load or compute ground truth distribution for n vertices."""

    cache_file = f'results/ground_truth_n{n_vertices}.json'

    if Path(cache_file).exists():
        print(f"Loading ground truth from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)

    print(f"Computing ground truth for n={n_vertices}...")
    print("This may take a while (testing all triangulations)...")

    # Get all triangulations from plantri
    plantri = find_plantri_executable()
    args = [plantri, f'-pc3', '-a', str(n_vertices)]
    result = subprocess.run(args, capture_output=True, text=True)

    # Parse and analyze all triangulations
    all_data = []

    for line_idx, line in enumerate(result.stdout.split('\n')):
        if line_idx % 1000 == 0 and line_idx > 0:
            print(f"  Processed {line_idx}...")

        line = line.strip()
        if not line or line.startswith('>'):
            continue

        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue

        n = int(parts[0])
        adj_str = parts[1]

        # Build adjacency dict
        adj = {}
        for v_idx, neighbor_str in enumerate(adj_str.split(',')):
            neighbors = [ord(c) - ord('a') for c in neighbor_str]
            adj[v_idx] = neighbors

        # Extract faces
        closed_tri = extract_faces_from_planar_embedding(n, adj)

        # Remove vertex 0 to get planar
        planar_tri = [tri for tri in closed_tri if 0 not in tri]

        if not planar_tri:
            continue

        # Compute statistics
        try:
            n_spanning = count_spanning_trees(planar_tri)

            std_result = check_delaunay_realizability(planar_tri, verbose=False, strict=False)
            strict_result = check_delaunay_realizability(planar_tri, verbose=False, strict=True)

            all_data.append({
                'triangles': planar_tri,
                'n_spanning_trees': n_spanning,
                'standard_realizable': bool(std_result['realizable']),
                'strict_realizable': bool(strict_result['realizable']),
            })
        except Exception as e:
            continue

    print(f"Ground truth: {len(all_data)} triangulations")

    # Compute true statistics
    ground_truth = {
        'n_triangulations': len(all_data),
        'standard_realizable_fraction': sum(1 for d in all_data if d['standard_realizable']) / len(all_data),
        'strict_realizable_fraction': sum(1 for d in all_data if d['strict_realizable']) / len(all_data),
        'spanning_trees_mean': np.mean([d['n_spanning_trees'] for d in all_data]),
        'spanning_trees_median': np.median([d['n_spanning_trees'] for d in all_data]),
        'spanning_trees_std': np.std([d['n_spanning_trees'] for d in all_data]),
        'all_data': all_data,
    }

    # Cache for future use
    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    return ground_truth


def sample_via_flips(starting_triangulation, k_flips, n_samples, seed=42, mode='independent'):
    """
    Generate samples using k-flip chains.

    Args:
        starting_triangulation: Initial triangulation
        k_flips: Number of flips between samples
        n_samples: Number of samples to generate
        seed: Random seed
        mode: 'independent' or 'continuous'
            - 'independent': Each sample starts fresh from starting_triangulation
            - 'continuous': One long chain, sample every k flips (MCMC style)

    Returns:
        List of sampled triangulations
    """
    np.random.seed(seed)
    samples = []

    if mode == 'independent':
        # Independent chains: restart from beginning each time
        for i in range(n_samples):
            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1}/{n_samples} samples...")

            current = random_edge_flips(starting_triangulation, n_flips=k_flips, seed=seed + i)
            samples.append([tuple(sorted(tri)) for tri in current])

    elif mode == 'continuous':
        # Continuous chain: sample every k flips
        current = starting_triangulation
        for i in range(n_samples):
            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1}/{n_samples} samples...")

            current = random_edge_flips(current, n_flips=k_flips, seed=seed + i)
            samples.append([tuple(sorted(tri)) for tri in current])

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return samples


def compute_sample_statistics(samples):
    """Compute statistics on sampled triangulations."""

    stats = {
        'n_spanning_trees': [],
        'standard_realizable': [],
        'strict_realizable': [],
    }

    for i, tri in enumerate(samples):
        if (i + 1) % 100 == 0:
            print(f"  Analyzed {i+1}/{len(samples)} samples...")

        try:
            n_spanning = count_spanning_trees(tri)
            std_result = check_delaunay_realizability(tri, verbose=False, strict=False)
            strict_result = check_delaunay_realizability(tri, verbose=False, strict=True)

            stats['n_spanning_trees'].append(n_spanning)
            stats['standard_realizable'].append(std_result['realizable'])
            stats['strict_realizable'].append(strict_result['realizable'])
        except:
            continue

    # Compute summary
    return {
        'n_samples': len(stats['n_spanning_trees']),
        'standard_realizable_fraction': np.mean(stats['standard_realizable']),
        'strict_realizable_fraction': np.mean(stats['strict_realizable']),
        'spanning_trees_mean': np.mean(stats['n_spanning_trees']),
        'spanning_trees_median': np.median(stats['n_spanning_trees']),
        'spanning_trees_std': np.std(stats['n_spanning_trees']),
    }


def mixing_experiment(n_vertices=10, k_values=None, n_samples=1000, seed=42, mode='continuous'):
    """
    Run mixing time experiment.

    Args:
        n_vertices: Number of vertices (use 10 for exhaustive ground truth)
        k_values: List of flip counts to test
        n_samples: Number of samples per k value
        seed: Random seed
        mode: 'independent' or 'continuous' sampling
    """
    if k_values is None:
        # Test a range of k values
        k_values = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

    print("="*70)
    print("FLIP GRAPH MIXING TIME EXPERIMENT")
    print("="*70)
    print(f"\nParameters:")
    print(f"  n_vertices: {n_vertices}")
    print(f"  k_values: {k_values}")
    print(f"  n_samples per k: {n_samples}")
    print(f"  mode: {mode}")
    print()

    # Load ground truth
    ground_truth = load_ground_truth(n_vertices)

    print("\nGround Truth (exhaustive enumeration):")
    print(f"  Total triangulations: {ground_truth['n_triangulations']}")
    print(f"  Standard realizable: {100*ground_truth['standard_realizable_fraction']:.1f}%")
    print(f"  Strict realizable: {100*ground_truth['strict_realizable_fraction']:.1f}%")
    print(f"  Spanning trees (mean): {ground_truth['spanning_trees_mean']:.1f}")
    print(f"  Spanning trees (median): {ground_truth['spanning_trees_median']:.1f}")
    print()

    # Choose a fixed starting triangulation
    starting_tri = ground_truth['all_data'][0]['triangles']
    print(f"Starting triangulation: {len(starting_tri)} triangles")
    print()

    # Run experiment for each k
    results = {}

    for k in k_values:
        print(f"Testing k={k} flips...")
        print(f"  Generating {n_samples} samples...")

        samples = sample_via_flips(starting_tri, k, n_samples, seed, mode=mode)

        print(f"  Computing statistics...")
        sample_stats = compute_sample_statistics(samples)

        # Compare to ground truth
        std_error = abs(sample_stats['standard_realizable_fraction'] -
                       ground_truth['standard_realizable_fraction'])
        strict_error = abs(sample_stats['strict_realizable_fraction'] -
                          ground_truth['strict_realizable_fraction'])
        spanning_error = abs(sample_stats['spanning_trees_mean'] -
                            ground_truth['spanning_trees_mean'])

        results[k] = {
            'sample_stats': sample_stats,
            'errors': {
                'standard_realizable': std_error,
                'strict_realizable': strict_error,
                'spanning_trees_mean': spanning_error,
            }
        }

        print(f"  Sample: std={100*sample_stats['standard_realizable_fraction']:.1f}%, "
              f"strict={100*sample_stats['strict_realizable_fraction']:.1f}%, "
              f"spanning_mean={sample_stats['spanning_trees_mean']:.1f}")
        print(f"  Error: std={100*std_error:.2f}%, "
              f"strict={100*strict_error:.2f}%, "
              f"spanning={spanning_error:.1f}")
        print()

    # Plot convergence
    plot_convergence(k_values, results, ground_truth)

    return results, ground_truth


def plot_convergence(k_values, results, ground_truth):
    """Plot convergence to ground truth as k increases."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Standard realizability
    ax = axes[0]
    fractions = [results[k]['sample_stats']['standard_realizable_fraction'] for k in k_values]
    ax.semilogx(k_values, [100*f for f in fractions], 'o-', label='Sample')
    ax.axhline(100*ground_truth['standard_realizable_fraction'],
               color='red', linestyle='--', label='Ground Truth')
    ax.set_xlabel('Number of flips (k)')
    ax.set_ylabel('Standard Realizable (%)')
    ax.set_title('Convergence: Standard Realizability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Strict realizability
    ax = axes[1]
    fractions = [results[k]['sample_stats']['strict_realizable_fraction'] for k in k_values]
    ax.semilogx(k_values, [100*f for f in fractions], 'o-', label='Sample')
    ax.axhline(100*ground_truth['strict_realizable_fraction'],
               color='red', linestyle='--', label='Ground Truth')
    ax.set_xlabel('Number of flips (k)')
    ax.set_ylabel('Strict Realizable (%)')
    ax.set_title('Convergence: Strict Realizability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Spanning trees
    ax = axes[2]
    means = [results[k]['sample_stats']['spanning_trees_mean'] for k in k_values]
    ax.semilogx(k_values, means, 'o-', label='Sample')
    ax.axhline(ground_truth['spanning_trees_mean'],
               color='red', linestyle='--', label='Ground Truth')
    ax.set_xlabel('Number of flips (k)')
    ax.set_ylabel('Mean Spanning Trees')
    ax.set_title('Convergence: Spanning Trees')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = 'results/plots/flip_mixing_convergence.png'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Convergence plot saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Measure flip graph mixing time')
    parser.add_argument('--n', type=int, default=10,
                       help='Number of vertices (10 recommended for ground truth)')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of samples per k value')
    parser.add_argument('--k-values', type=int, nargs='+',
                       help='List of k values to test (default: 10 20 50 100 200 500 1000 2000 5000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mode', type=str, default='continuous', choices=['independent', 'continuous'],
                       help='Sampling mode: independent (restart each time) or continuous (MCMC chain)')

    args = parser.parse_args()

    results, ground_truth = mixing_experiment(
        n_vertices=args.n,
        k_values=args.k_values,
        n_samples=args.samples,
        seed=args.seed,
        mode=args.mode
    )
