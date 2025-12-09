#!/usr/bin/env python3
"""
Generate random points, compute Delaunay triangulation, and analyze optimal angles.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import json
from scipy.spatial import Delaunay
from fractions import Fraction
from collections import Counter
from ideal_poly_volume_toolkit.rivin_delaunay import check_delaunay_realizability, build_edge_adjacency


def continued_fraction_convergents(x, max_terms=20):
    """Compute convergents of continued fraction expansion."""
    convergents = []
    a = []
    remainder = x

    for _ in range(max_terms):
        floor_val = int(np.floor(remainder))
        a.append(floor_val)
        if abs(remainder - floor_val) < 1e-12:
            break
        remainder = 1.0 / (remainder - floor_val)

    p_prev, p_curr = 0, 1
    q_prev, q_curr = 1, 0

    for ai in a:
        p_next = ai * p_curr + p_prev
        q_next = ai * q_curr + q_prev
        convergents.append((p_next, q_next))
        p_prev, p_curr = p_curr, p_next
        q_prev, q_curr = q_curr, q_next

    return convergents


def analyze_random_configuration(n_vertices=89, seed=42):
    """Generate random configuration and analyze optimal angles."""

    print(f"═══════════════════════════════════════════════════════════════")
    print(f"RANDOM CONFIGURATION ANALYSIS")
    print(f"═══════════════════════════════════════════════════════════════")
    print(f"\nNumber of vertices: {n_vertices}")
    print(f"Random seed: {seed}")

    # Generate random points in unit disk
    print(f"\n{'─'*63}")
    print(f"STEP 1: Generate random points")
    print(f"{'─'*63}")

    np.random.seed(seed)

    # Generate points in polar coordinates for better distribution
    radii = np.sqrt(np.random.uniform(0, 1, n_vertices))
    angles = np.random.uniform(0, 2*np.pi, n_vertices)

    vertices_complex = radii * np.exp(1j * angles)
    points = np.column_stack([vertices_complex.real, vertices_complex.imag])

    print(f"Generated {n_vertices} random points in unit disk")

    # Compute Delaunay triangulation
    print(f"\n{'─'*63}")
    print(f"STEP 2: Compute Delaunay triangulation (combinatorics)")
    print(f"{'─'*63}")

    tri = Delaunay(points)
    triangulation = [tuple(sorted(simplex)) for simplex in tri.simplices]
    triangulation = sorted(set(triangulation))

    print(f"Triangulation computed: {len(triangulation)} triangles")

    # Compute optimal angles using Rivin LP
    print(f"\n{'─'*63}")
    print(f"STEP 3: Compute optimal angles via Rivin LP")
    print(f"{'─'*63}")

    result = check_delaunay_realizability(triangulation, verbose=False, strict=False)

    if not result['realizable']:
        print("ERROR: Triangulation not realizable!")
        return None

    print(f"✓ Triangulation is realizable")

    # Extract angles
    angles_scaled = result['angles']
    angles_radians = angles_scaled * np.pi
    n_triangles = len(triangulation)
    angles_array = angles_radians.reshape((n_triangles, 3))

    # Compute interior edge dihedral angles
    print(f"\n{'─'*63}")
    print(f"STEP 4: Compute dihedral angles")
    print(f"{'─'*63}")

    edge_adjacency = build_edge_adjacency(triangulation)
    dihedrals = []

    for edge, opposite_corners in sorted(edge_adjacency.items()):
        if len(opposite_corners) == 2:
            angle1 = angles_array[opposite_corners[0][0], opposite_corners[0][1]]
            angle2 = angles_array[opposite_corners[1][0], opposite_corners[1][1]]
            dihedral = angle1 + angle2
            normalized = dihedral / np.pi

            # Find best rational approximation
            convergents = continued_fraction_convergents(normalized)
            if convergents:
                best_p, best_q = convergents[-1]
                error = abs(normalized - best_p / best_q)

                dihedrals.append({
                    'edge': edge,
                    'angle_rad': float(dihedral),
                    'angle_deg': float(np.degrees(dihedral)),
                    'normalized': float(normalized),
                    'p': int(best_p),
                    'q': int(best_q),
                    'error': float(error),
                    'rational': f"{best_p}π/{best_q}" if best_q > 1 else f"{best_p}π",
                })

    print(f"Computed {len(dihedrals)} interior edge dihedrals")

    # Analyze rational patterns
    print(f"\n{'─'*63}")
    print(f"RATIONAL ANGLE ANALYSIS")
    print(f"{'─'*63}")

    max_denominator = 200
    small_error_threshold = 1e-6

    # Count denominators with small error
    small_q_dihedrals = [d for d in dihedrals if d['q'] <= max_denominator and d['error'] < small_error_threshold]
    denominator_counts = Counter(d['q'] for d in small_q_dihedrals)

    print(f"\nDenominators q ≤ {max_denominator} with error < {small_error_threshold}:")
    print(f"\n{'Denominator':>12} {'Count':>8} {'Relation to n':>20}")
    print(f"{'-'*42}")

    for q in sorted(denominator_counts.keys()):
        count = denominator_counts[q]
        relation = ""
        if q == n_vertices - 2:
            relation = "= n-2"
        elif q == n_vertices - 3:
            relation = "= n-3"
        elif q == n_vertices - 1:
            relation = "= n-1"
        elif q == n_vertices:
            relation = "= n"
        print(f"  q={q:>3}  {count:7d}  {relation:>20}")

    # Check if ALL angles have small denominators
    if len(small_q_dihedrals) == len(dihedrals):
        print(f"\n✓ ALL {len(dihedrals)} interior edges have rational angles with q ≤ {max_denominator}!")

        # Find the dominant denominator
        most_common_q = denominator_counts.most_common(1)[0][0]
        print(f"\nMost common denominator: q = {most_common_q}", end="")
        if most_common_q == n_vertices - 2:
            print(f" = n-2 ★")
        elif most_common_q == n_vertices - 3:
            print(f" = n-3 ★")
        else:
            print()
    else:
        print(f"\n{len(small_q_dihedrals)}/{len(dihedrals)} edges have rational angles")

    # Show pattern distribution
    print(f"\n{'─'*63}")
    print(f"RATIONAL PATTERN DISTRIBUTION")
    print(f"{'─'*63}")

    pattern_counts = Counter(d['rational'] for d in small_q_dihedrals)
    print(f"\n{'Pattern':>10} {'Count':>8} {'Degrees':>12}")
    print(f"{'-'*32}")
    for pattern, count in pattern_counts.most_common(10):
        angle_deg = next(d['angle_deg'] for d in small_q_dihedrals if d['rational'] == pattern)
        print(f"  {pattern:>8}  {count:7d}  {angle_deg:11.3f}°")

    # Sample angles
    print(f"\n{'─'*63}")
    print(f"SAMPLE DIHEDRAL ANGLES (first 10)")
    print(f"{'─'*63}")
    print(f"{'Edge':>12} {'Degrees':>10} {'Rational':>12} {'Error':>12}")
    print(f"{'-'*48}")
    for d in dihedrals[:10]:
        print(f"  {str(d['edge']):>10} {d['angle_deg']:9.3f}° {d['rational']:>12} {d['error']:11.2e}")

    # Save results
    output_data = {
        'n_vertices': n_vertices,
        'n_triangles': n_triangles,
        'n_interior_edges': len(dihedrals),
        'seed': seed,
        'vertex_positions': {
            'real': vertices_complex.real.tolist(),
            'imag': vertices_complex.imag.tolist(),
        },
        'triangulation': [[int(v) for v in tri] for tri in triangulation],
        'denominator_counts': {int(k): int(v) for k, v in denominator_counts.items()},
        'all_rational': len(small_q_dihedrals) == len(dihedrals),
    }

    output_file = Path(f"results/data/{n_vertices}vertex_random_analysis_seed{seed}.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'─'*63}")
    print(f"✓ Results saved to: {output_file}")
    print(f"{'─'*63}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze random triangulation with optimal angles')
    parser.add_argument('--vertices', type=int, default=89, help='Number of vertices')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    analyze_random_configuration(n_vertices=args.vertices, seed=args.seed)
