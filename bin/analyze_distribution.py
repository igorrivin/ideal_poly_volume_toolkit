#!/usr/bin/env python3
"""
General-purpose wrapper for analyzing volume distributions of ideal polyhedra.

Usage:
    python bin/analyze_distribution.py --vertices 4 --samples 10000
    python bin/analyze_distribution.py --vertices 6 --samples 50000 --output custom_plot.png
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import sys

from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    ideal_poly_volume_via_delaunay,
)


def sample_random_vertex():
    """
    Sample a uniform random point on the unit sphere and project to complex plane.
    Uses stereographic projection from north pole.
    """
    # Sample uniform point on sphere using Gaussian method
    vec = np.random.randn(3)
    vec = vec / np.linalg.norm(vec)
    x, y, z = vec

    # Skip near north pole (maps to infinity)
    if z > 0.999:
        return None

    # Stereographic projection
    w = complex(x/(1-z), y/(1-z))
    return w


def analyze_distribution(n_vertices, n_samples, seed=42, series_terms=96):
    """
    Analyze volume distribution for n_vertices polyhedra.

    Args:
        n_vertices: Number of vertices (must be >= 3)
        n_samples: Number of random configurations to sample
        seed: Random seed
        series_terms: Number of terms for Lobachevsky function approximation

    Returns:
        dict with volumes and statistics
    """
    np.random.seed(seed)

    # First 3 vertices are fixed to break symmetry
    fixed_vertices = [complex(0, 0), complex(1, 0), complex(0, 1)]
    n_random = n_vertices - 3

    if n_random < 0:
        raise ValueError("Need at least 3 vertices")

    volumes = []
    print(f"Sampling {n_samples} random {n_vertices}-vertex configurations...")

    for i in range(n_samples):
        if (i + 1) % (n_samples // 10) == 0:
            print(f"  Progress: {i + 1}/{n_samples} ({100*(i+1)/n_samples:.1f}%)")

        # Build configuration
        vertices = fixed_vertices.copy()

        # Add random vertices
        for _ in range(n_random):
            v = sample_random_vertex()
            if v is None:
                continue  # Skip degenerate samples

            # Skip if too close to existing vertices
            too_close = False
            for existing in vertices:
                if abs(v - existing) < 0.01:
                    too_close = True
                    break
            if too_close:
                continue

            vertices.append(v)

        # Only proceed if we have the right number of vertices
        if len(vertices) != n_vertices:
            continue

        # Compute volume
        try:
            vertices_np = np.array(vertices, dtype=np.complex128)
            vol = ideal_poly_volume_via_delaunay(
                vertices_np, mode='fast', series_terms=series_terms
            )

            # Sanity check
            if vol > 0 and vol < 1000:
                volumes.append(vol)
        except:
            pass  # Skip invalid configurations

    volumes = np.array(volumes)

    if len(volumes) == 0:
        raise ValueError("No valid configurations found!")

    print(f"\nSuccessfully analyzed {len(volumes)} valid configurations")

    return {
        'volumes': volumes,
        'n_samples_requested': n_samples,
        'n_valid': len(volumes),
        'mean': np.mean(volumes),
        'median': np.median(volumes),
        'std': np.std(volumes),
        'min': np.min(volumes),
        'max': np.max(volumes),
        'q25': np.percentile(volumes, 25),
        'q75': np.percentile(volumes, 75),
    }


def plot_distribution(volumes, stats, n_vertices, output_file, reference_volume=None):
    """Create histogram plot of volume distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(volumes, bins=50, density=True, alpha=0.7,
             color='steelblue', edgecolor='black', linewidth=0.5)
    ax1.axvline(stats['mean'], color='red', linestyle='--', linewidth=2,
                label=f"Mean: {stats['mean']:.4f}")
    ax1.axvline(stats['median'], color='green', linestyle='--', linewidth=2,
                label=f"Median: {stats['median']:.4f}")

    if reference_volume is not None:
        ax1.axvline(reference_volume, color='orange', linestyle='--', linewidth=2,
                    label=f"Reference: {reference_volume:.4f}")

    ax1.set_xlabel('Volume', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title(f'{n_vertices}-Vertex Ideal Polyhedra Volume Distribution', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot([volumes], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5))
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_title('Volume Distribution (Box Plot)', fontsize=14)
    ax2.set_xticklabels([f'{n_vertices} vertices'])
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze volume distributions of ideal polyhedra',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --vertices 4 --samples 10000
  %(prog)s --vertices 6 --samples 50000 --output my_analysis.png
  %(prog)s --vertices 5 --samples 20000 --reference 3.66
        """
    )

    parser.add_argument('--vertices', '-v', type=int, required=True,
                        help='Number of vertices (must be >= 3)')
    parser.add_argument('--samples', '-n', type=int, default=10000,
                        help='Number of random samples (default: 10000)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output plot file (default: results/plots/{n}vertex_distribution_TIMESTAMP.png)')
    parser.add_argument('--data', '-d', type=str, default=None,
                        help='Output data JSON file (optional)')
    parser.add_argument('--reference', '-r', type=float, default=None,
                        help='Reference volume to mark on plot (optional)')
    parser.add_argument('--series-terms', type=int, default=96,
                        help='Number of series terms for Lobachevsky function (default: 96)')

    args = parser.parse_args()

    if args.vertices < 3:
        print("Error: Number of vertices must be at least 3")
        sys.exit(1)

    # Setup output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output is None:
        plot_file = f"results/plots/{args.vertices}vertex_distribution_{timestamp}.png"
    else:
        plot_file = args.output

    if args.data is not None:
        data_file = args.data
    else:
        data_file = f"results/data/{args.vertices}vertex_distribution_{timestamp}.json"

    # Ensure output directories exist
    Path(plot_file).parent.mkdir(parents=True, exist_ok=True)
    if args.data is not None:
        Path(data_file).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Ideal Polyhedron Volume Distribution Analysis")
    print("=" * 70)
    print(f"Vertices:      {args.vertices}")
    print(f"Samples:       {args.samples}")
    print(f"Random seed:   {args.seed}")
    print(f"Plot output:   {plot_file}")
    if args.data:
        print(f"Data output:   {data_file}")
    print("=" * 70)
    print()

    # Run analysis
    results = analyze_distribution(
        args.vertices,
        args.samples,
        seed=args.seed,
        series_terms=args.series_terms
    )

    # Print statistics
    print("\n" + "=" * 70)
    print("STATISTICS:")
    print("=" * 70)
    print(f"Valid configs:  {results['n_valid']:,} / {results['n_samples_requested']:,}")
    print(f"Mean volume:    {results['mean']:.8f}")
    print(f"Median volume:  {results['median']:.8f}")
    print(f"Std deviation:  {results['std']:.8f}")
    print(f"Min volume:     {results['min']:.8f}")
    print(f"Max volume:     {results['max']:.8f}")
    print(f"25th percentile: {results['q25']:.8f}")
    print(f"75th percentile: {results['q75']:.8f}")

    if args.reference is not None:
        print(f"\nReference volume: {args.reference:.8f}")
        print(f"Mean/Reference:   {results['mean']/args.reference:.4f}")
        print(f"Max/Reference:    {results['max']/args.reference:.4f}")

    # Create plot
    plot_distribution(
        results['volumes'],
        results,
        args.vertices,
        plot_file,
        reference_volume=args.reference
    )

    # Save data if requested
    if args.data is not None:
        output_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_vertices': args.vertices,
                'n_samples_requested': args.samples,
                'n_valid': results['n_valid'],
                'seed': args.seed,
                'series_terms': args.series_terms,
            },
            'statistics': {
                'mean': float(results['mean']),
                'median': float(results['median']),
                'std': float(results['std']),
                'min': float(results['min']),
                'max': float(results['max']),
                'q25': float(results['q25']),
                'q75': float(results['q75']),
            },
            'volumes': results['volumes'].tolist(),
        }

        with open(data_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nData saved to: {data_file}")

    print("=" * 70)


if __name__ == '__main__':
    main()
