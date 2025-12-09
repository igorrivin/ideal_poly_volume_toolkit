#!/usr/bin/env python3
"""
Sample random triangulations using edge flips.

Two strategies depending on size:
1. Small n (≤15): Start with plantri enumeration, then flip
2. Large n (>15): Start with Delaunay of random points, then flip many times

The flip graph is not an expander, so for uniform sampling we need
approximately O(n^2) flips for mixing (diameter is O(n^2)).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.spatial import Delaunay
from ideal_poly_volume_toolkit.rivin_delaunay import random_edge_flips
from ideal_poly_volume_toolkit.plantri_interface import find_plantri_executable
from ideal_poly_volume_toolkit.planar_utils import extract_faces_from_planar_embedding
import subprocess


def sample_small_triangulation(n_vertices: int, n_flips: int, min_connectivity: int = 3, seed: int = 42):
    """
    Sample a random triangulation for small n using plantri + flips.

    Args:
        n_vertices: Number of vertices (recommend ≤15)
        n_flips: Number of random edge flips
        min_connectivity: Minimum connectivity (3 or 4)
        seed: Random seed

    Returns:
        List of triangles
    """
    print(f"Sampling small triangulation (n={n_vertices})...")
    print(f"  Strategy: plantri enumeration → pick one → flip {n_flips} times")

    # Get all triangulations from plantri
    plantri = find_plantri_executable()
    if plantri is None:
        raise RuntimeError("plantri not found")

    args = [plantri, f'-pc{min_connectivity}', '-a', str(n_vertices)]
    result = subprocess.run(args, capture_output=True, text=True)

    # Parse triangulations
    triangulations = []
    for line in result.stdout.split('\n'):
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
        vertex_lists = adj_str.split(',')
        for v_idx, neighbor_str in enumerate(vertex_lists):
            neighbors = []
            for letter in neighbor_str:
                neighbor_idx = ord(letter) - ord('a')
                neighbors.append(neighbor_idx)
            adj[v_idx] = neighbors

        # Extract faces properly
        triangles = extract_faces_from_planar_embedding(n, adj)

        if triangles:
            triangulations.append(triangles)

    print(f"  Found {len(triangulations)} triangulations from plantri")

    # Pick one randomly
    np.random.seed(seed)
    idx = np.random.randint(len(triangulations))
    closed_tri = triangulations[idx]

    # Remove vertex 0 to get planar triangulation
    planar_tri = [tri for tri in closed_tri if 0 not in tri]

    print(f"  Selected triangulation #{idx}")
    print(f"  Planar triangulation: {len(planar_tri)} triangles")

    # Apply random flips
    if n_flips > 0:
        flipped = random_edge_flips(planar_tri, n_flips=n_flips, seed=seed)
        print(f"  Applied {n_flips} random edge flips")
        return flipped
    else:
        return planar_tri


def sample_large_triangulation(n_vertices: int, n_flips: int, seed: int = 42):
    """
    Sample a random triangulation for large n using Delaunay + many flips.

    For uniform sampling, use n_flips ≈ 10 * n_vertices^2 (conservative estimate).

    Args:
        n_vertices: Number of vertices
        n_flips: Number of random edge flips (recommend ≥ 10*n^2 for mixing)
        seed: Random seed

    Returns:
        List of triangles
    """
    print(f"Sampling large triangulation (n={n_vertices})...")
    print(f"  Strategy: Delaunay(random points) → flip {n_flips} times")

    # Generate random points
    np.random.seed(seed)
    points = np.random.rand(n_vertices, 2)

    # Compute Delaunay triangulation
    tri = Delaunay(points)
    initial_triangles = [tuple(simplex) for simplex in tri.simplices]

    print(f"  Generated {n_vertices} random points")
    print(f"  Delaunay triangulation: {len(initial_triangles)} triangles")

    # Apply many random flips for mixing
    if n_flips > 0:
        flipped = random_edge_flips(initial_triangles, n_flips=n_flips, seed=seed)
        print(f"  Applied {n_flips} random edge flips")

        # Note on mixing
        diameter_estimate = n_vertices * n_vertices
        if n_flips < 10 * diameter_estimate:
            print(f"  WARNING: For uniform sampling, recommend ≥{10*diameter_estimate} flips")
            print(f"           (flip graph diameter ≈ n^2 = {diameter_estimate})")

        return flipped
    else:
        return initial_triangles


def sample_triangulation(n_vertices: int, n_flips: int = None, min_connectivity: int = 3, seed: int = 42):
    """
    Sample a random triangulation using the appropriate strategy for the size.

    Args:
        n_vertices: Number of vertices
        n_flips: Number of flips (if None, use default based on size)
        min_connectivity: Minimum connectivity for small n (3 or 4)
        seed: Random seed

    Returns:
        List of triangles
    """
    # Choose strategy and default flips based on size
    if n_vertices <= 15:
        # Small: Use plantri + moderate flips
        if n_flips is None:
            n_flips = 100 * n_vertices  # Conservative
        return sample_small_triangulation(n_vertices, n_flips, min_connectivity, seed)
    else:
        # Large: Use Delaunay + many flips
        if n_flips is None:
            # For mixing, need ~O(n^2) flips (flip graph diameter)
            # Use 10*n^2 to be conservative
            n_flips = 10 * n_vertices * n_vertices
        return sample_large_triangulation(n_vertices, n_flips, seed)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Sample random triangulations')
    parser.add_argument('--n', type=int, required=True, help='Number of vertices')
    parser.add_argument('--flips', type=int, help='Number of edge flips (auto if not specified)')
    parser.add_argument('--connectivity', type=int, default=3, choices=[3, 4],
                       help='Min connectivity for small n (3 or 4)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--samples', type=int, default=1, help='Number of samples to generate')

    args = parser.parse_args()

    print("="*70)
    print("RANDOM TRIANGULATION SAMPLING VIA EDGE FLIPS")
    print("="*70)
    print()

    for i in range(args.samples):
        if args.samples > 1:
            print(f"\nSample {i+1}/{args.samples}:")
            print("-"*70)

        seed = args.seed + i if args.samples > 1 else args.seed
        triangles = sample_triangulation(
            n_vertices=args.n,
            n_flips=args.flips,
            min_connectivity=args.connectivity,
            seed=seed
        )

        print(f"\nResult: {len(triangles)} triangles")

        # Show a few triangles
        print(f"Sample triangles: {triangles[:5]}")

        # Could test realizability here
        # from ideal_poly_volume_toolkit.rivin_delaunay import check_delaunay_realizability
        # result = check_delaunay_realizability(triangles, verbose=False)
        # print(f"Realizable: {result['realizable']}")

    print()
    print("="*70)
