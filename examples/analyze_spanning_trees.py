#!/usr/bin/env python3
"""
Analyze the relationship between spanning tree counts and Delaunay realizability.

Tests the hypothesis that realizable triangulations have more spanning trees.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import networkx as nx
from ideal_poly_volume_toolkit.plantri_interface import find_plantri_executable
from ideal_poly_volume_toolkit.rivin_delaunay import check_delaunay_realizability
from ideal_poly_volume_toolkit.planar_utils import extract_faces_from_planar_embedding
import subprocess
import json
from collections import defaultdict
import matplotlib.pyplot as plt


def get_triangulations_text(n_vertices: int, min_connectivity: int = 3) -> list:
    """Generate triangulations in ASCII format."""
    plantri = find_plantri_executable()
    if plantri is None:
        raise RuntimeError("plantri not found")

    args = [plantri, f'-pc{min_connectivity}', '-a', str(n_vertices)]
    result = subprocess.run(args, capture_output=True, text=True)

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

        # Extract faces from planar embedding (adjacency lists are in cyclic order)
        triangles = extract_faces_from_planar_embedding(n, adj)

        if triangles:
            triangulations.append(triangles)

    return triangulations


def remove_vertex_to_planar(triangles: list, vertex_to_remove: int) -> list:
    """Remove a vertex to create planar triangulation."""
    return [tri for tri in triangles if vertex_to_remove not in tri]


def triangles_to_graph(triangles: list) -> nx.Graph:
    """Convert triangle list to NetworkX graph."""
    G = nx.Graph()
    for tri in triangles:
        v0, v1, v2 = tri
        G.add_edge(v0, v1)
        G.add_edge(v1, v2)
        G.add_edge(v2, v0)
    return G


def count_spanning_trees_kirchhoff(G: nx.Graph) -> int:
    """
    Count spanning trees using Kirchhoff's matrix-tree theorem.

    The number of spanning trees equals any cofactor of the Laplacian matrix.
    We compute the determinant of the Laplacian with one row/column deleted.
    """
    if len(G.nodes()) == 0:
        return 0
    if len(G.nodes()) == 1:
        return 1

    # Get Laplacian matrix
    L = nx.laplacian_matrix(G).toarray()

    # Remove first row and column (compute cofactor)
    L_reduced = L[1:, 1:]

    # Compute determinant (this is the number of spanning trees)
    det = np.linalg.det(L_reduced)

    # Round to nearest integer (should be exact integer, but floating point)
    return int(round(det))


def analyze_n_vertices(n: int, min_connectivity: int = 3, verbose: bool = True):
    """
    Analyze spanning trees vs realizability for n vertices.

    Args:
        n: Number of vertices
        min_connectivity: Minimum connectivity
        verbose: Print progress

    Returns:
        Dictionary with analysis results
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Analyzing n={n} vertices ({min_connectivity}-connected)")
        print(f"{'='*70}")

    # Generate all closed triangulations
    if verbose:
        print(f"\nGenerating closed triangulations...")
    closed_tris = get_triangulations_text(n, min_connectivity)
    if verbose:
        print(f"Generated {len(closed_tris)} closed triangulations")

    # Convert to planar and analyze
    if verbose:
        print(f"Converting to planar (remove vertex 0) and analyzing...")

    results = {
        'n_vertices': n,
        'min_connectivity': min_connectivity,
        'triangulations': [],
    }

    for idx, closed_tri in enumerate(closed_tris):
        if verbose and (idx + 1) % 1000 == 0:
            print(f"  Processed {idx+1}/{len(closed_tris)}...")

        # Convert to planar
        planar_tri = remove_vertex_to_planar(closed_tri, 0)

        # Create graph
        G = triangles_to_graph(planar_tri)

        # Count spanning trees
        n_spanning_trees = count_spanning_trees_kirchhoff(G)

        # Test realizability
        try:
            result_standard = check_delaunay_realizability(planar_tri, verbose=False, strict=False)
            result_strict = check_delaunay_realizability(planar_tri, verbose=False, strict=True)
        except Exception as e:
            # Skip degenerate cases
            if verbose and idx < 10:
                print(f"  Warning: Skipping triangulation {idx}: {e}")
            continue

        # Store results
        results['triangulations'].append({
            'index': idx,
            'n_spanning_trees': n_spanning_trees,
            'standard_realizable': bool(result_standard['realizable']),
            'strict_realizable': bool(result_strict['realizable']),
            'n_edges': G.number_of_edges(),
            'n_vertices': G.number_of_nodes(),
        })

    return results


def compute_statistics(results: dict):
    """Compute statistics from results."""
    tris = results['triangulations']

    # Partition by realizability
    standard_real = [t for t in tris if t['standard_realizable']]
    standard_nonreal = [t for t in tris if not t['standard_realizable']]
    strict_real = [t for t in tris if t['strict_realizable']]
    strict_nonreal = [t for t in tris if not t['strict_realizable']]

    # Among standard realizable, partition by strict
    standard_real_strict_yes = [t for t in standard_real if t['strict_realizable']]
    standard_real_strict_no = [t for t in standard_real if not t['strict_realizable']]

    stats = {
        'total': len(tris),
        'standard_realizable': {
            'count': len(standard_real),
            'spanning_trees': {
                'mean': np.mean([t['n_spanning_trees'] for t in standard_real]) if standard_real else 0,
                'median': np.median([t['n_spanning_trees'] for t in standard_real]) if standard_real else 0,
                'std': np.std([t['n_spanning_trees'] for t in standard_real]) if standard_real else 0,
                'min': min([t['n_spanning_trees'] for t in standard_real]) if standard_real else 0,
                'max': max([t['n_spanning_trees'] for t in standard_real]) if standard_real else 0,
            }
        },
        'standard_non_realizable': {
            'count': len(standard_nonreal),
            'spanning_trees': {
                'mean': np.mean([t['n_spanning_trees'] for t in standard_nonreal]) if standard_nonreal else 0,
                'median': np.median([t['n_spanning_trees'] for t in standard_nonreal]) if standard_nonreal else 0,
                'std': np.std([t['n_spanning_trees'] for t in standard_nonreal]) if standard_nonreal else 0,
                'min': min([t['n_spanning_trees'] for t in standard_nonreal]) if standard_nonreal else 0,
                'max': max([t['n_spanning_trees'] for t in standard_nonreal]) if standard_nonreal else 0,
            }
        },
        'strict_realizable': {
            'count': len(strict_real),
            'spanning_trees': {
                'mean': np.mean([t['n_spanning_trees'] for t in strict_real]) if strict_real else 0,
                'median': np.median([t['n_spanning_trees'] for t in strict_real]) if strict_real else 0,
                'std': np.std([t['n_spanning_trees'] for t in strict_real]) if strict_real else 0,
                'min': min([t['n_spanning_trees'] for t in strict_real]) if strict_real else 0,
                'max': max([t['n_spanning_trees'] for t in strict_real]) if strict_real else 0,
            }
        },
        'strict_non_realizable': {
            'count': len(strict_nonreal),
            'spanning_trees': {
                'mean': np.mean([t['n_spanning_trees'] for t in strict_nonreal]) if strict_nonreal else 0,
                'median': np.median([t['n_spanning_trees'] for t in strict_nonreal]) if strict_nonreal else 0,
                'std': np.std([t['n_spanning_trees'] for t in strict_nonreal]) if strict_nonreal else 0,
                'min': min([t['n_spanning_trees'] for t in strict_nonreal]) if strict_nonreal else 0,
                'max': max([t['n_spanning_trees'] for t in strict_nonreal]) if strict_nonreal else 0,
            }
        },
        'among_standard_realizable': {
            'strict_yes': {
                'count': len(standard_real_strict_yes),
                'spanning_trees': {
                    'mean': np.mean([t['n_spanning_trees'] for t in standard_real_strict_yes]) if standard_real_strict_yes else 0,
                    'median': np.median([t['n_spanning_trees'] for t in standard_real_strict_yes]) if standard_real_strict_yes else 0,
                }
            },
            'strict_no': {
                'count': len(standard_real_strict_no),
                'spanning_trees': {
                    'mean': np.mean([t['n_spanning_trees'] for t in standard_real_strict_no]) if standard_real_strict_no else 0,
                    'median': np.median([t['n_spanning_trees'] for t in standard_real_strict_no]) if standard_real_strict_no else 0,
                }
            }
        }
    }

    return stats


def print_statistics(stats: dict, n: int):
    """Print statistics in readable format."""
    print(f"\n{'='*70}")
    print(f"STATISTICS FOR n={n}")
    print(f"{'='*70}")

    print(f"\nTotal triangulations: {stats['total']}")

    print(f"\n--- STANDARD REALIZABILITY ---")
    print(f"Realizable: {stats['standard_realizable']['count']} ({100*stats['standard_realizable']['count']/stats['total']:.1f}%)")
    print(f"  Spanning trees (mean): {stats['standard_realizable']['spanning_trees']['mean']:.1f}")
    print(f"  Spanning trees (median): {stats['standard_realizable']['spanning_trees']['median']:.1f}")
    print(f"  Spanning trees (range): [{stats['standard_realizable']['spanning_trees']['min']}, {stats['standard_realizable']['spanning_trees']['max']}]")

    print(f"\nNon-realizable: {stats['standard_non_realizable']['count']} ({100*stats['standard_non_realizable']['count']/stats['total']:.1f}%)")
    print(f"  Spanning trees (mean): {stats['standard_non_realizable']['spanning_trees']['mean']:.1f}")
    print(f"  Spanning trees (median): {stats['standard_non_realizable']['spanning_trees']['median']:.1f}")
    if stats['standard_non_realizable']['count'] > 0:
        print(f"  Spanning trees (range): [{stats['standard_non_realizable']['spanning_trees']['min']}, {stats['standard_non_realizable']['spanning_trees']['max']}]")

    # Ratio
    if stats['standard_non_realizable']['spanning_trees']['mean'] > 0:
        ratio = stats['standard_realizable']['spanning_trees']['mean'] / stats['standard_non_realizable']['spanning_trees']['mean']
        print(f"\nRatio (realizable/non-realizable): {ratio:.2f}x")

    print(f"\n--- STRICT REALIZABILITY ---")
    print(f"Strict realizable: {stats['strict_realizable']['count']} ({100*stats['strict_realizable']['count']/stats['total']:.1f}%)")
    print(f"  Spanning trees (mean): {stats['strict_realizable']['spanning_trees']['mean']:.1f}")
    print(f"  Spanning trees (median): {stats['strict_realizable']['spanning_trees']['median']:.1f}")

    print(f"\nStrict non-realizable: {stats['strict_non_realizable']['count']} ({100*stats['strict_non_realizable']['count']/stats['total']:.1f}%)")
    print(f"  Spanning trees (mean): {stats['strict_non_realizable']['spanning_trees']['mean']:.1f}")
    print(f"  Spanning trees (median): {stats['strict_non_realizable']['spanning_trees']['median']:.1f}")

    # Ratio
    if stats['strict_non_realizable']['spanning_trees']['mean'] > 0:
        ratio = stats['strict_realizable']['spanning_trees']['mean'] / stats['strict_non_realizable']['spanning_trees']['mean']
        print(f"\nRatio (strict/non-strict): {ratio:.2f}x")

    print(f"\n--- AMONG STANDARD REALIZABLE: STRICT vs NON-STRICT ---")
    print(f"Strict YES: {stats['among_standard_realizable']['strict_yes']['count']}")
    print(f"  Spanning trees (mean): {stats['among_standard_realizable']['strict_yes']['spanning_trees']['mean']:.1f}")
    print(f"Strict NO: {stats['among_standard_realizable']['strict_no']['count']}")
    print(f"  Spanning trees (mean): {stats['among_standard_realizable']['strict_no']['spanning_trees']['mean']:.1f}")

    if stats['among_standard_realizable']['strict_no']['spanning_trees']['mean'] > 0:
        ratio = stats['among_standard_realizable']['strict_yes']['spanning_trees']['mean'] / stats['among_standard_realizable']['strict_no']['spanning_trees']['mean']
        print(f"\nRatio (strict/non-strict among realizable): {ratio:.2f}x")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze spanning trees vs realizability')
    parser.add_argument('--n', type=int, default=10, help='Number of vertices')
    parser.add_argument('--connectivity', type=int, default=3, choices=[3, 4],
                       help='Minimum connectivity')
    parser.add_argument('--output', type=str, help='Output JSON file')

    args = parser.parse_args()

    # Run analysis
    results = analyze_n_vertices(args.n, args.connectivity, verbose=True)

    # Compute statistics
    stats = compute_statistics(results)

    # Print statistics
    print_statistics(stats, args.n)

    # Save results
    if args.output:
        output_data = {
            'parameters': {
                'n_vertices': args.n,
                'min_connectivity': args.connectivity,
            },
            'statistics': stats,
            'raw_data': results,
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Results saved to: {args.output}")
        print(f"{'='*70}")
