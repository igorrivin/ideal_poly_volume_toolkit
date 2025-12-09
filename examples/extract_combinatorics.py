#!/usr/bin/env python3
"""
Extract combinatorics and optimal angles from maximal volume configuration.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import json
from scipy.spatial import Delaunay
from ideal_poly_volume_toolkit.rivin_delaunay import check_delaunay_realizability, build_edge_adjacency


def extract_combinatorics_and_angles(json_file):
    """Extract triangulation and optimal angles from optimization result."""

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
    print(f"MAXIMAL VOLUME IDEAL POLYHEDRON - COMPLETE ANALYSIS")
    print(f"═══════════════════════════════════════════════════════════════")
    print(f"\nVolume: {volume:.12f}")
    print(f"Vertices: {n_vertices}")

    # Compute Delaunay triangulation
    tri = Delaunay(points)
    triangulation = [tuple(sorted(simplex)) for simplex in tri.simplices]
    triangulation = sorted(set(triangulation))  # Remove duplicates, sort

    print(f"Triangles: {len(triangulation)}")

    # Display triangulation
    print(f"\n{'─'*63}")
    print(f"TRIANGULATION (Combinatorial Structure)")
    print(f"{'─'*63}")
    print("\nTriangles (vertex indices):")
    for i, tri in enumerate(triangulation, 1):
        print(f"  {i:2d}. {tri}")

    # Get optimal angles from Rivin LP
    print(f"\n{'─'*63}")
    print(f"OPTIMAL ANGLES FROM RIVIN LP")
    print(f"{'─'*63}")

    result = check_delaunay_realizability(triangulation, verbose=False, strict=False)

    if not result['realizable']:
        print("ERROR: Configuration not realizable!")
        return

    # Extract angles (convert from scaled units to radians)
    angles_scaled = result['angles']
    angles_radians = angles_scaled * np.pi
    n_triangles = len(triangulation)
    angles_array = angles_radians.reshape((n_triangles, 3))

    print("\nFace angles (interior angles of each triangle):")
    print(f"{'Tri#':>5} {'Vertices':>20} {'Vtx':>5} {'Angle (rad)':>14} {'Angle (deg)':>12} {'Angle/π':>10}")
    print(f"{'-'*70}")

    for i, tri in enumerate(triangulation):
        for j, vertex in enumerate(tri):
            angle_rad = angles_array[i, j]
            angle_deg = np.degrees(angle_rad)
            angle_normalized = angle_rad / np.pi
            tri_str = f"({tri[0]},{tri[1]},{tri[2]})"
            print(f"  {i+1:3d}. {tri_str:>18s} v{vertex:2d}  {angle_rad:13.10f}  {angle_deg:11.6f}°  {angle_normalized:9.6f}π")

    # Compute ALL dihedral angles
    print(f"\n{'='*78}")
    print(f"ALL DIHEDRAL ANGLES (Complete Ideal Polyhedron)")
    print(f"{'='*78}")

    from scipy.spatial import ConvexHull
    from fractions import Fraction

    # Find boundary (convex hull)
    hull = ConvexHull(points)
    boundary_edges = set()
    boundary_vertices = set()
    for simplex in hull.simplices:
        edge = tuple(sorted([simplex[0], simplex[1]]))
        boundary_edges.add(edge)
        boundary_vertices.add(simplex[0])
        boundary_vertices.add(simplex[1])

    edge_adjacency = build_edge_adjacency(triangulation)
    interior_edges = [e for e, ops in edge_adjacency.items() if len(ops) == 2]

    print(f"\nTopology:")
    print(f"  Total vertices: {n_vertices}")
    print(f"  Finite triangles: {len(triangulation)}")
    print(f"  Boundary vertices: {len(boundary_vertices)}")
    print(f"  Interior edges: {len(interior_edges)}")
    print(f"  Boundary edges: {len(boundary_edges)}")
    print(f"  Total edges: {len(interior_edges)} + {len(boundary_edges)} + {len(boundary_vertices)} = {len(interior_edges) + len(boundary_edges) + len(boundary_vertices)}")
    print(f"  Expected (3V-6): {3*n_vertices - 6}")

    # TYPE 1: Interior edge dihedrals
    print(f"\n{'─'*78}")
    print(f"TYPE 1: Interior Edge Dihedrals ({len(interior_edges)} edges)")
    print(f"(Edges shared by two finite triangles)")
    print(f"{'─'*78}")

    all_dihedrals = []

    for edge, opposite_corners in sorted(edge_adjacency.items()):
        if len(opposite_corners) == 2:
            angle1 = angles_array[opposite_corners[0][0], opposite_corners[0][1]]
            angle2 = angles_array[opposite_corners[1][0], opposite_corners[1][1]]
            dihedral = angle1 + angle2
            normalized = dihedral / np.pi
            frac = Fraction(normalized).limit_denominator(20)
            rational = f"{frac.numerator}π/{frac.denominator}" if frac.denominator > 1 else f"{frac.numerator}π"

            all_dihedrals.append({
                'type': 'interior',
                'edge': edge,
                'dihedral_rad': dihedral,
                'dihedral_deg': np.degrees(dihedral),
                'normalized': normalized,
                'rational': rational,
                'p': frac.numerator,
                'q': frac.denominator,
            })

    # Print first 10
    for d in all_dihedrals[:10]:
        print(f"  {str(d['edge']):>12}  {d['dihedral_deg']:7.3f}° = {d['normalized']:8.6f}π ≈ {d['rational']:>8}")
    if len(all_dihedrals) > 10:
        print(f"  ... ({len(all_dihedrals) - 10} more)")

    # TYPE 2: Boundary edge dihedrals
    print(f"\n{'─'*78}")
    print(f"TYPE 2: Boundary Edge Dihedrals ({len(boundary_edges)} edges)")
    print(f"(Angle opposite each boundary edge in the triangle containing it)")
    print(f"{'─'*78}")

    boundary_dihedrals = []
    for edge in sorted(boundary_edges):
        # Find the triangle containing this edge
        v1, v2 = edge
        for i, tri in enumerate(triangulation):
            if v1 in tri and v2 in tri:
                # Find the third vertex (opposite to the edge)
                v3 = [v for v in tri if v != v1 and v != v2][0]
                v3_idx = tri.index(v3)
                dihedral = angles_array[i, v3_idx]
                normalized = dihedral / np.pi
                frac = Fraction(normalized).limit_denominator(20)
                rational = f"{frac.numerator}π/{frac.denominator}" if frac.denominator > 1 else f"{frac.numerator}π"

                boundary_dihedrals.append({
                    'type': 'boundary',
                    'edge': edge,
                    'dihedral_rad': dihedral,
                    'dihedral_deg': np.degrees(dihedral),
                    'normalized': normalized,
                    'rational': rational,
                    'p': frac.numerator,
                    'q': frac.denominator,
                })
                print(f"  {str(edge):>12}  {np.degrees(dihedral):7.3f}° = {normalized:8.6f}π ≈ {rational:>8}")
                break

    all_dihedrals.extend(boundary_dihedrals)

    # TYPE 3: Vertex-to-∞ dihedrals
    print(f"\n{'─'*78}")
    print(f"TYPE 3: Vertex-to-∞ Dihedrals ({len(boundary_vertices)} edges)")
    print(f"(Sum of all angles at each boundary vertex)")
    print(f"{'─'*78}")

    vertex_dihedrals = []
    for v in sorted(boundary_vertices):
        # Sum all angles at this vertex
        total_angle = 0
        for i, tri in enumerate(triangulation):
            if v in tri:
                v_idx = tri.index(v)
                total_angle += angles_array[i, v_idx]

        normalized = total_angle / np.pi
        frac = Fraction(normalized).limit_denominator(20)
        rational = f"{frac.numerator}π/{frac.denominator}" if frac.denominator > 1 else f"{frac.numerator}π"

        vertex_dihedrals.append({
            'type': 'to_infinity',
            'edge': (v, 'inf'),
            'dihedral_rad': total_angle,
            'dihedral_deg': np.degrees(total_angle),
            'normalized': normalized,
            'rational': rational,
            'p': frac.numerator,
            'q': frac.denominator,
        })
        print(f"  v{v}→∞  {np.degrees(total_angle):11.3f}° = {normalized:8.6f}π ≈ {rational:>8}  {'< π ✓' if total_angle < np.pi else '≥ π ✗'}")

    all_dihedrals.extend(vertex_dihedrals)
    dihedrals = all_dihedrals  # For later use

    # Summary of rational patterns
    print(f"\n{'─'*63}")
    print(f"RATIONAL ANGLE SUMMARY")
    print(f"{'─'*63}")

    from collections import Counter
    pattern_counts = Counter(d['rational'] for d in dihedrals)

    print(f"\n{'Pattern':>10} {'Count':>8} {'Degrees':>12}")
    print(f"{'-'*32}")
    for pattern, count in pattern_counts.most_common():
        # Get representative angle
        angle_deg = next(d['dihedral_deg'] for d in dihedrals if d['rational'] == pattern)
        print(f"  {pattern:>8}  {count:7d}  {angle_deg:11.3f}°")

    # Determine the dominant denominator
    denominator_counts = Counter(d['q'] for d in dihedrals)
    dominant_q = denominator_counts.most_common(1)[0][0]

    print(f"\n{'─'*63}")
    if len(denominator_counts) == 1 and dominant_q > 1:
        print(f"VERIFICATION: All angles are exact multiples of π/{dominant_q}!")
    else:
        print(f"VERIFICATION: All angles are exact rational multiples of π!")
    print(f"{'─'*63}")

    # Export to JSON
    output_data = {
        'metadata': {
            'source_file': str(json_file),
            'volume': float(volume),
            'n_vertices': int(n_vertices),
            'n_triangles': len(triangulation),
        },
        'combinatorics': {
            'triangles': [[int(v) for v in tri] for tri in triangulation],
        },
        'optimal_angles': {
            'face_angles_radians': angles_array.tolist(),
            'face_angles_degrees': np.degrees(angles_array).tolist(),
            'dihedral_angles': [
                {
                    'type': d['type'],
                    'edge': [int(d['edge'][0]), str(d['edge'][1])],  # Handle 'inf' for vertex-to-∞
                    'angle_radians': float(d['dihedral_rad']),
                    'angle_degrees': float(d['dihedral_deg']),
                    'rational_form': d['rational'],
                    'p': int(d['p']),
                    'q': int(d['q']),
                }
                for d in dihedrals
            ],
        },
        'vertex_positions': {
            'real': real.tolist(),
            'imag': imag.tolist(),
        },
    }

    # Generate output filename based on input file
    input_name = Path(json_file).stem
    output_file = Path(f'results/data/{input_name}_combinatorics.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Exported to: {output_file}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract combinatorics and optimal angles from maximal volume config')
    parser.add_argument('json_file', help='Path to optimization result JSON file')
    args = parser.parse_args()

    extract_combinatorics_and_angles(args.json_file)
