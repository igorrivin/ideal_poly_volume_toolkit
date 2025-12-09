#!/usr/bin/env python3
"""
Compute the automorphism group of a triangulation using pynauty.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
from scipy.spatial import Delaunay
import pynauty


def compute_triangulation_automorphisms(triangulation):
    """
    Compute automorphism group of a triangulation using pynauty.

    We create a bipartite graph where:
    - Vertices 0..n-1 represent the original vertices
    - Vertices n..n+t-1 represent the triangles
    - Edges connect vertices to triangles they belong to
    - Two vertex colors distinguish vertices from triangles

    Returns:
        dict with 'group_size' and 'generators'
    """
    n_vertices = max(max(tri) for tri in triangulation) + 1
    n_triangles = len(triangulation)

    # Total nodes in bipartite graph
    n_total = n_vertices + n_triangles

    # Build adjacency list
    # First n_vertices nodes are original vertices
    # Next n_triangles nodes are triangle nodes
    adjacency = {i: set() for i in range(n_total)}

    for tri_idx, tri in enumerate(triangulation):
        tri_node = n_vertices + tri_idx
        for v in tri:
            # Add edge between vertex v and triangle node
            adjacency[v].add(tri_node)
            adjacency[tri_node].add(v)

    # Convert to pynauty format (list of sets)
    adj_list = [adjacency[i] for i in range(n_total)]

    # Create vertex coloring to distinguish vertices from triangles
    # Color 0 for original vertices, color 1 for triangle nodes
    coloring = [set(range(n_vertices)), set(range(n_vertices, n_total))]

    # Create pynauty graph
    g = pynauty.Graph(
        number_of_vertices=n_total,
        directed=False,
        adjacency_dict={i: adj_list[i] for i in range(n_total)},
        vertex_coloring=coloring
    )

    # Compute automorphism group
    generators, grpsize1, grpsize2, orbits, numorbits = pynauty.autgrp(g)

    # Compute group size: grpsize1 * 10^grpsize2
    group_size = grpsize1 * (10 ** grpsize2)

    # Extract vertex permutations (only look at first n_vertices)
    vertex_generators = []
    for gen in generators:
        # gen is a permutation of all nodes; we only care about vertex nodes
        vertex_perm = [gen[i] for i in range(n_vertices)]
        vertex_generators.append(vertex_perm)

    # Get orbit information for vertices
    # orbits is an array where orbits[i] gives the orbit number of vertex i
    vertex_orbits = {}
    for v in range(n_vertices):
        orbit_idx = orbits[v]
        if orbit_idx not in vertex_orbits:
            vertex_orbits[orbit_idx] = []
        vertex_orbits[orbit_idx].append(v)

    return {
        'group_size': int(group_size),
        'generators': vertex_generators,
        'n_generators': len(vertex_generators),
        'vertex_orbits': list(vertex_orbits.values()),
        'n_orbits': len(vertex_orbits),
    }


def analyze_symmetry(json_file):
    """Analyze symmetry of maximal volume configuration."""

    # Load configuration
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Get vertex positions
    real = np.array(data['best']['vertices_real'])
    imag = np.array(data['best']['vertices_imag'])
    points = np.column_stack([real, imag])

    n_vertices = len(points)
    volume = data['best']['volume']

    print(f"═══════════════════════════════════════════════════════════════")
    print(f"TRIANGULATION SYMMETRY ANALYSIS")
    print(f"═══════════════════════════════════════════════════════════════")
    print(f"\nConfiguration: {Path(json_file).name}")
    print(f"Vertices: {n_vertices}")
    print(f"Volume: {volume:.12f}")

    # Compute Delaunay triangulation
    tri = Delaunay(points)
    triangulation = [tuple(sorted(simplex)) for simplex in tri.simplices]
    triangulation = sorted(set(triangulation))

    print(f"Triangles: {len(triangulation)}")

    # Compute automorphism group
    print(f"\n{'─'*63}")
    print(f"COMPUTING AUTOMORPHISM GROUP...")
    print(f"{'─'*63}")

    result = compute_triangulation_automorphisms(triangulation)

    print(f"\nGroup size: {result['group_size']}")
    print(f"Number of generators: {result['n_generators']}")
    print(f"Number of vertex orbits: {result['n_orbits']}")

    if result['group_size'] == 1:
        print(f"\n✓ TRIVIAL SYMMETRY GROUP (only identity)")
    else:
        print(f"\n✗ NON-TRIVIAL SYMMETRY GROUP!")

        print(f"\nVertex orbits:")
        for i, orbit in enumerate(result['vertex_orbits']):
            print(f"  Orbit {i+1} (size {len(orbit)}): {sorted(orbit)}")

        if result['n_generators'] > 0 and result['n_generators'] <= 10:
            print(f"\nGenerators:")
            for i, gen in enumerate(result['generators']):
                # Check if it's identity
                if gen == list(range(n_vertices)):
                    print(f"  Generator {i+1}: identity")
                else:
                    # Find cycles
                    visited = [False] * n_vertices
                    cycles = []
                    for v in range(n_vertices):
                        if not visited[v] and gen[v] != v:
                            cycle = []
                            curr = v
                            while not visited[curr]:
                                visited[curr] = True
                                cycle.append(curr)
                                curr = gen[curr]
                            if len(cycle) > 1:
                                cycles.append(cycle)

                    if cycles:
                        cycle_str = ' '.join(f"({' '.join(map(str, c))})" for c in cycles)
                        print(f"  Generator {i+1}: {cycle_str}")

    print(f"\n{'─'*63}")

    # Export results
    output_data = {
        'source_file': str(json_file),
        'n_vertices': n_vertices,
        'n_triangles': len(triangulation),
        'volume': float(volume),
        'automorphism_group': {
            'size': int(result['group_size']),
            'n_generators': result['n_generators'],
            'n_orbits': result['n_orbits'],
            'vertex_orbits': result['vertex_orbits'],
        }
    }

    output_file = Path(f"results/data/{Path(json_file).stem}_symmetry.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Results exported to: {output_file}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute automorphism group of triangulation')
    parser.add_argument('json_file', help='Path to optimization result JSON file')
    args = parser.parse_args()

    analyze_symmetry(args.json_file)
