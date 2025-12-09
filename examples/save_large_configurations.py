#!/usr/bin/env python3
"""
Generate and save triangulations with optimal angles for various n.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import json
from datetime import datetime
from scipy.spatial import Delaunay
from ideal_poly_volume_toolkit.rivin_delaunay import check_delaunay_realizability, build_edge_adjacency
from math import gcd
from functools import reduce


def lcm(a, b):
    """Compute least common multiple."""
    return abs(a * b) // gcd(a, b)


def analyze_and_save_configuration(n_vertices, seed, output_dir='results/data/large_configs'):
    """Generate, analyze, and save a configuration."""

    print(f"\n{'='*70}")
    print(f"Configuration: n={n_vertices}, seed={seed}")
    print(f"{'='*70}")

    # Generate random points
    np.random.seed(seed)
    radii = np.sqrt(np.random.uniform(0, 1, n_vertices))
    angles = np.random.uniform(0, 2*np.pi, n_vertices)
    vertices_complex = radii * np.exp(1j * angles)
    points = np.column_stack([vertices_complex.real, vertices_complex.imag])

    print(f"Generated {n_vertices} random points")

    # Compute Delaunay triangulation
    tri = Delaunay(points)
    triangulation = [tuple(sorted(simplex)) for simplex in tri.simplices]
    triangulation = sorted(set(triangulation))

    print(f"Triangulation: {len(triangulation)} triangles")

    # Get optimal angles from Rivin LP
    result = check_delaunay_realizability(triangulation, verbose=False, strict=False)

    if not result['realizable']:
        print("ERROR: Not realizable!")
        return None

    print(f"✓ Realizable")

    # Extract angles
    angles_scaled = result['angles']
    angles_radians = angles_scaled * np.pi
    n_triangles = len(triangulation)
    angles_array = angles_radians.reshape((n_triangles, 3))

    # Compute dihedral angles
    edge_adjacency = build_edge_adjacency(triangulation)
    dihedrals = []

    for edge, opposite_corners in sorted(edge_adjacency.items()):
        if len(opposite_corners) == 2:
            angle1 = angles_array[opposite_corners[0][0], opposite_corners[0][1]]
            angle2 = angles_array[opposite_corners[1][0], opposite_corners[1][1]]
            dihedral = angle1 + angle2
            normalized = dihedral / np.pi

            dihedrals.append({
                'edge': [int(edge[0]), int(edge[1])],
                'angle_radians': float(dihedral),
                'angle_degrees': float(np.degrees(dihedral)),
                'normalized': float(normalized),
            })

    print(f"Computed {len(dihedrals)} interior edge dihedrals")

    # Analyze rational structure
    # Check denominators up to 10*n
    max_denom = 10 * n_vertices
    denominators_found = set()

    for d in dihedrals:
        norm = d['normalized']
        # Try to find rational approximation
        for q in range(1, min(max_denom + 1, 1000)):
            p = round(norm * q)
            if abs(norm - p/q) < 1e-10:
                denominators_found.add(q)
                d['rational_p'] = int(p)
                d['rational_q'] = int(q)
                d['rational_error'] = float(abs(norm - p/q))
                break

    # Find LCM of all denominators
    if denominators_found:
        common_denominator = reduce(lcm, denominators_found)
    else:
        common_denominator = None

    print(f"Denominators found: {sorted(denominators_found)}")
    print(f"Common denominator (LCM): {common_denominator}")
    if common_denominator:
        print(f"Ratio q/n: {common_denominator/n_vertices:.3f}")

    # Check if all angles are rational with common denominator
    all_rational = all('rational_q' in d for d in dihedrals)
    if all_rational and common_denominator:
        # Verify all are multiples of 1/common_denominator
        all_multiples = True
        for d in dihedrals:
            p = round(d['normalized'] * common_denominator)
            error = abs(d['normalized'] - p/common_denominator)
            if error > 1e-10:
                all_multiples = False
                break

        if all_multiples:
            print(f"✓ ALL {len(dihedrals)} angles are exact multiples of π/{common_denominator}")

    # Prepare output data
    output_data = {
        'metadata': {
            'n_vertices': int(n_vertices),
            'n_triangles': int(n_triangles),
            'n_interior_edges': len(dihedrals),
            'seed': int(seed),
            'generated': datetime.now().isoformat(),
        },
        'vertex_positions': {
            'real': vertices_complex.real.tolist(),
            'imag': vertices_complex.imag.tolist(),
        },
        'triangulation': [[int(v) for v in tri] for tri in triangulation],
        'face_angles': {
            'radians': angles_array.tolist(),
            'degrees': np.degrees(angles_array).tolist(),
        },
        'dihedral_angles': dihedrals,
        'rational_structure': {
            'all_rational': all_rational,
            'denominators': sorted(denominators_found),
            'common_denominator': int(common_denominator) if common_denominator else None,
            'ratio_to_n': float(common_denominator / n_vertices) if common_denominator else None,
        }
    }

    # Save to file
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"n{n_vertices:03d}_seed{seed:03d}_triangulation.json"

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Saved to: {output_file}")

    return output_data


def main():
    """Generate and save multiple configurations."""

    print("═"*70)
    print("LARGE CONFIGURATION GENERATOR")
    print("Saving triangulations with optimal angles for various n")
    print("═"*70)

    # Configurations to generate
    configs = [
        (30, 42),
        (40, 42),
        (50, 42),
        (60, 42),
        (70, 42),
        (80, 42),
        (89, 42),
        (100, 42),
        (89, 123),  # Same n, different seed
        (89, 456),  # Another seed for n=89
    ]

    results = []

    for n, seed in configs:
        result = analyze_and_save_configuration(n, seed)
        if result:
            results.append({
                'n': n,
                'seed': seed,
                'common_denom': result['rational_structure']['common_denominator'],
                'ratio': result['rational_structure']['ratio_to_n'],
            })

    # Create summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'n':>4} {'seed':>6} {'q (LCM)':>10} {'q/n':>8}")
    print("-"*30)

    for r in results:
        if r['common_denom']:
            print(f"{r['n']:>4} {r['seed']:>6} {r['common_denom']:>10} {r['ratio']:>8.3f}")
        else:
            print(f"{r['n']:>4} {r['seed']:>6} {'None':>10} {'N/A':>8}")

    # Save summary
    summary_file = Path('results/data/large_configs/SUMMARY.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'generated': datetime.now().isoformat(),
            'configurations': results,
        }, f, indent=2)

    print(f"\n✓ Summary saved to: {summary_file}")
    print(f"\nAll configurations saved to: results/data/large_configs/")


if __name__ == '__main__':
    main()
