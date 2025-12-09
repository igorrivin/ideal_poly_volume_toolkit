#!/usr/bin/env python3
"""
Complete a finite triangulation by adding vertex at infinity.
Creates a closed triangulation of a sphere.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import json
from scipy.spatial import Delaunay, ConvexHull


def complete_triangulation(triangles, verbose=True):
    """
    Add vertex at infinity to create a closed triangulation of a sphere.

    Args:
        triangles: List of triangles (each a tuple/list of 3 vertex indices)

    Returns:
        dict with:
            - complete_triangles: All triangles including those with ∞
            - n_finite_vertices: Number of finite vertices
            - infinity_vertex_id: ID assigned to vertex at ∞
            - boundary_edges: List of boundary edges
    """

    triangles = [tuple(sorted(tri)) for tri in triangles]
    triangles = sorted(set(triangles))

    # Find all vertices
    all_vertices = set()
    for tri in triangles:
        all_vertices.update(tri)

    n_vertices = len(all_vertices)

    # Find boundary edges (edges that appear in only one triangle)
    edge_count = {}
    for tri in triangles:
        for i in range(3):
            edge = tuple(sorted([tri[i], tri[(i+1)%3]]))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

    if verbose:
        print(f"Finite triangulation:")
        print(f"  Vertices: {n_vertices}")
        print(f"  Triangles: {len(triangles)}")
        print(f"  Boundary edges: {len(boundary_edges)}")

    # Assign ID for vertex at infinity
    # Use the next available integer
    infinity_vertex_id = max(all_vertices) + 1

    # Create triangles with infinity
    # For each boundary edge, create a triangle with ∞
    infinity_triangles = []
    for edge in boundary_edges:
        # Triangle is (v1, v2, ∞) where (v1, v2) is the boundary edge
        tri = tuple(sorted([edge[0], edge[1], infinity_vertex_id]))
        infinity_triangles.append(tri)

    # Combine all triangles
    complete_triangles = triangles + infinity_triangles

    if verbose:
        print(f"\nComplete triangulation (with ∞):")
        print(f"  Vertices: {n_vertices + 1} (including ∞)")
        print(f"  Triangles: {len(complete_triangles)} ({len(triangles)} finite + {len(infinity_triangles)} with ∞)")

    # Verify Euler characteristic
    n_edges = 0
    edge_set = set()
    for tri in complete_triangles:
        for i in range(3):
            edge = tuple(sorted([tri[i], tri[(i+1)%3]]))
            edge_set.add(edge)
    n_edges = len(edge_set)

    V = n_vertices + 1
    E = n_edges
    F = len(complete_triangles)
    chi = V - E + F

    if verbose:
        print(f"\nEuler characteristic:")
        print(f"  V = {V}")
        print(f"  E = {E}")
        print(f"  F = {F}")
        print(f"  χ = V - E + F = {chi}")

        if chi == 2:
            print(f"  ✓ Sphere (χ = 2)")
        else:
            print(f"  ✗ Not a sphere (χ = {chi})")

    # Check if it's a closed manifold (every edge shared by exactly 2 triangles)
    edge_count_complete = {}
    for tri in complete_triangles:
        for i in range(3):
            edge = tuple(sorted([tri[i], tri[(i+1)%3]]))
            edge_count_complete[edge] = edge_count_complete.get(edge, 0) + 1

    max_edge_count = max(edge_count_complete.values())
    min_edge_count = min(edge_count_complete.values())
    is_closed = (max_edge_count == 2 and min_edge_count == 2)

    if verbose:
        print(f"\nManifold check:")
        print(f"  All edges shared by exactly 2 triangles: {is_closed}")
        if not is_closed:
            bad_edges = [edge for edge, count in edge_count_complete.items() if count != 2]
            print(f"  Problem edges: {len(bad_edges)}")

    return {
        'complete_triangles': complete_triangles,
        'finite_triangles': triangles,
        'infinity_triangles': infinity_triangles,
        'n_finite_vertices': n_vertices,
        'infinity_vertex_id': infinity_vertex_id,
        'boundary_edges': boundary_edges,
        'euler_characteristic': chi,
        'is_closed_manifold': is_closed,
    }


def complete_and_save(input_file, output_file=None):
    """Load triangulation, complete it, and save."""

    print(f"Loading: {input_file}")

    with open(input_file, 'r') as f:
        data = json.load(f)

    # Extract triangulation
    triangulation = data['triangulation']

    print(f"\n{'='*70}")

    # Complete the triangulation
    result = complete_triangulation(triangulation, verbose=True)

    # Prepare output
    output_data = {
        'metadata': {
            'source_file': str(input_file),
            'n_finite_vertices': result['n_finite_vertices'],
            'n_total_vertices': result['n_finite_vertices'] + 1,
            'n_finite_triangles': len(result['finite_triangles']),
            'n_infinity_triangles': len(result['infinity_triangles']),
            'n_total_triangles': len(result['complete_triangles']),
            'infinity_vertex_id': result['infinity_vertex_id'],
            'euler_characteristic': result['euler_characteristic'],
            'is_sphere': result['euler_characteristic'] == 2,
            'is_closed_manifold': result['is_closed_manifold'],
        },
        'complete_triangulation': [[int(v) for v in tri] for tri in result['complete_triangles']],
        'finite_triangulation': [[int(v) for v in tri] for tri in result['finite_triangles']],
        'infinity_triangulation': [[int(v) for v in tri] for tri in result['infinity_triangles']],
        'boundary_edges': [[int(v) for v in edge] for edge in result['boundary_edges']],
    }

    # Also copy over vertex positions and angles if they exist
    if 'vertex_positions' in data:
        output_data['vertex_positions'] = data['vertex_positions']
    if 'face_angles' in data:
        output_data['face_angles'] = data['face_angles']
    if 'dihedral_angles' in data:
        output_data['dihedral_angles'] = data['dihedral_angles']

    # Save
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / (input_path.stem + '_complete.json')

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Saved complete triangulation to: {output_file}")

    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Complete a triangulation by adding vertex at infinity'
    )
    parser.add_argument('input_file', help='JSON file with triangulation')
    parser.add_argument('--output', '-o', help='Output file (default: input_complete.json)')

    args = parser.parse_args()

    complete_and_save(args.input_file, args.output)
