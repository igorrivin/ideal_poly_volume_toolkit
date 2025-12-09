#!/usr/bin/env python3
"""
Identify which 6-vertex triangulation is strictly realizable.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ideal_poly_volume_toolkit.plantri_interface import find_plantri_executable
from ideal_poly_volume_toolkit.rivin_delaunay import check_delaunay_realizability
import subprocess
from collections import Counter

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

        # Convert adjacency to triangles
        triangles = []
        for v0 in range(n):
            for v1 in adj[v0]:
                if v1 <= v0:
                    continue
                for v2 in adj[v1]:
                    if v2 <= v1:
                        continue
                    if v2 in adj[v0]:
                        tri = tuple(sorted([v0, v1, v2]))
                        if tri not in triangles:
                            triangles.append(tri)

        if triangles:
            triangulations.append((adj, triangles))

    return triangulations


def remove_vertex_to_planar(triangles: list, vertex_to_remove: int) -> list:
    """Remove a vertex to create planar triangulation."""
    return [tri for tri in triangles if vertex_to_remove not in tri]


def analyze_graph_structure(adj):
    """Analyze the graph structure."""
    n = len(adj)
    degrees = [len(neighbors) for neighbors in adj.values()]
    degree_sequence = tuple(sorted(degrees))

    # Count triangles (3-cliques)
    triangles = 0
    for v0 in range(n):
        for v1 in adj[v0]:
            if v1 <= v0:
                continue
            for v2 in adj[v1]:
                if v2 <= v1:
                    continue
                if v2 in adj[v0]:
                    triangles += 1

    # Check if 4-regular (all degrees = 4)
    is_4_regular = all(d == 4 for d in degrees)

    return {
        'n_vertices': n,
        'n_edges': sum(degrees) // 2,
        'degree_sequence': degree_sequence,
        'degrees': degrees,
        'n_triangles': triangles,
        'is_4_regular': is_4_regular,
    }


print("="*70)
print("IDENTIFYING STRICTLY REALIZABLE 6-VERTEX TRIANGULATION")
print("="*70)

# Generate all 6-vertex 3-connected triangulations
print("\nGenerating all 6-vertex 3-connected triangulations...")
triangulations = get_triangulations_text(6, min_connectivity=3)
print(f"Found {len(triangulations)} closed triangulations")

print("\n" + "="*70)
print("Testing each triangulation in strict mode:")
print("="*70)

for idx, (adj, closed_tri) in enumerate(triangulations):
    # Remove vertex 0 to make planar
    planar_tri = remove_vertex_to_planar(closed_tri, 0)

    # Analyze structure
    struct = analyze_graph_structure(adj)

    # Test realizability (standard and strict)
    result_standard = check_delaunay_realizability(planar_tri, verbose=False, strict=False)
    result_strict = check_delaunay_realizability(planar_tri, verbose=False, strict=True)

    print(f"\nTriangulation #{idx+1}:")
    print(f"  Degree sequence: {struct['degree_sequence']}")
    print(f"  Edges: {struct['n_edges']}, Triangles: {struct['n_triangles']}")
    print(f"  Is 4-regular: {struct['is_4_regular']}")
    if struct['is_4_regular']:
        print(f"  >>> THIS IS THE OCTAHEDRON <<<")

    print(f"  Standard realizable: {result_standard['realizable']}")
    print(f"  Strict realizable:   {result_strict['realizable']}")

    if result_strict['realizable']:
        print(f"  ✓✓✓ STRICTLY REALIZABLE ✓✓✓")
        print(f"      Min angle: {result_strict['min_angle_radians']:.6f} rad")
        print(f"      Slack: {result_strict['slack_radians']:.6f} rad")
        print(f"      Max dihedral: {result_strict['max_dihedral_radians']:.6f} rad")

        # Show adjacency list
        print(f"\n  Adjacency list:")
        for v in sorted(adj.keys()):
            print(f"    {v}: {adj[v]}")

        # Show triangles
        print(f"\n  Triangles (planar, after removing vertex 0):")
        for tri in planar_tri:
            print(f"    {tri}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

strict_count = sum(1 for adj, tri in triangulations
                   if check_delaunay_realizability(remove_vertex_to_planar(tri, 0),
                                                   verbose=False, strict=True)['realizable'])
print(f"Strictly realizable: {strict_count}/7")

# Check if the strictly realizable one is the octahedron
for idx, (adj, closed_tri) in enumerate(triangulations):
    planar_tri = remove_vertex_to_planar(closed_tri, 0)
    result_strict = check_delaunay_realizability(planar_tri, verbose=False, strict=True)
    struct = analyze_graph_structure(adj)

    if result_strict['realizable']:
        if struct['is_4_regular']:
            print(f"\n✓ The strictly realizable triangulation IS the octahedron!")
        else:
            print(f"\n✗ The strictly realizable triangulation is NOT the octahedron")
            print(f"  Degree sequence: {struct['degree_sequence']}")
