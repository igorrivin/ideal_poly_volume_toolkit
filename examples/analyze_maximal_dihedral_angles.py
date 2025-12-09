#!/usr/bin/env python3
"""
Analyze dihedral angles in maximal volume triangulations.

Test the conjecture: Maximal volume configurations have dihedral angles
that are rational multiples of π (i.e., θ = pπ/q for small integers p, q).

Strategy:
1. Load maximal volume triangulations from optimization results
2. Compute dihedral angles using Rivin LP
3. Use continued fractions to find best rational approximation to θ/π
4. Report special angles with small denominators
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from fractions import Fraction
from ideal_poly_volume_toolkit.rivin_delaunay import check_delaunay_realizability
import json
import glob


def continued_fraction_convergents(x, max_terms=20):
    """
    Compute continued fraction convergents of x.

    Returns list of (p, q) where p/q are best rational approximations.
    """
    convergents = []

    # Standard continued fraction algorithm
    a = []
    remainder = x

    for _ in range(max_terms):
        floor_val = int(np.floor(remainder))
        a.append(floor_val)

        if abs(remainder - floor_val) < 1e-12:
            break

        remainder = 1.0 / (remainder - floor_val)

    # Compute convergents
    p_prev, p_curr = 0, 1
    q_prev, q_curr = 1, 0

    for ai in a:
        p_next = ai * p_curr + p_prev
        q_next = ai * q_curr + q_prev

        convergents.append((p_next, q_next))

        p_prev, p_curr = p_curr, p_next
        q_prev, q_curr = q_curr, q_next

    return convergents


def find_rational_multiple_of_pi(angle_radians, max_denominator=100, tolerance=1e-6):
    """
    Check if angle is close to pπ/q for small integers p, q.

    Returns (p, q, error) if found, else None.
    """
    # Compute angle / π
    normalized = angle_radians / np.pi

    # Get continued fraction convergents
    convergents = continued_fraction_convergents(normalized, max_terms=30)

    # Find best approximation with small denominator
    best_match = None
    best_error = float('inf')

    for p, q in convergents:
        if q > max_denominator:
            continue

        # Compute error
        error = abs(angle_radians - (p * np.pi / q))

        if error < tolerance and error < best_error:
            best_error = error
            best_match = (p, q, error)

    return best_match


def analyze_dihedral_angles(triangles, name="triangulation"):
    """
    Analyze dihedral angles for a triangulation.

    Returns dict with dihedral angles and their rational approximations.
    """
    # Get angles from Rivin LP
    result = check_delaunay_realizability(triangles, verbose=False, strict=False)

    if not result['realizable']:
        return None

    # Extract angles (convert from scaled units to radians)
    angles_scaled = result['angles']
    angles_radians = angles_scaled * np.pi
    n_triangles = len(triangles)
    angles_array = angles_radians.reshape((n_triangles, 3))

    # Build edge -> opposite angles mapping for dihedral angles
    from ideal_poly_volume_toolkit.rivin_delaunay import build_edge_adjacency

    edge_adjacency = build_edge_adjacency(triangles)

    # Compute dihedral angles
    dihedrals = []

    for edge, opposite_corners in edge_adjacency.items():
        if len(opposite_corners) == 2:  # Interior edge
            # Sum of two opposite angles
            angle1 = angles_array[opposite_corners[0][0], opposite_corners[0][1]]
            angle2 = angles_array[opposite_corners[1][0], opposite_corners[1][1]]
            dihedral = angle1 + angle2

            dihedrals.append({
                'edge': edge,
                'angle': dihedral,
                'angle_deg': np.degrees(dihedral),
                'normalized': dihedral / np.pi,
            })

    # Find rational approximations
    special_angles = []

    for d in dihedrals:
        match = find_rational_multiple_of_pi(d['angle'], max_denominator=100, tolerance=1e-6)

        if match:
            p, q, error = match
            d['rational'] = f"{p}π/{q}"
            d['p'] = p
            d['q'] = q
            d['error'] = error
            d['error_deg'] = np.degrees(error)

            # Consider it "special" if denominator is small
            if q <= 20:
                special_angles.append(d)

    return {
        'name': name,
        'n_triangles': n_triangles,
        'n_vertices': len(set(v for tri in triangles for v in tri)),
        'dihedrals': dihedrals,
        'special_angles': special_angles,
        'volume': result.get('volume', None),
    }


def load_maximal_volume_results(directory='bin/results/data'):
    """Load optimization results and extract maximal volume configurations."""
    from scipy.spatial import Delaunay
    results = []

    pattern = str(Path(directory) / '*vertex_optimization_*.json')
    files = glob.glob(pattern)

    print(f"Found {len(files)} optimization result files")

    for filepath in sorted(files)[:10]:  # Limit to first 10 for now
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Check if we have vertex positions in the "best" field
            if 'best' in data and 'vertices_real' in data['best'] and 'vertices_imag' in data['best']:
                real = np.array(data['best']['vertices_real'])
                imag = np.array(data['best']['vertices_imag'])

                # Construct 2D points
                points = np.column_stack([real, imag])

                # Compute Delaunay triangulation
                tri = Delaunay(points)
                triangulation = [tuple(simplex) for simplex in tri.simplices]

                results.append({
                    'file': Path(filepath).name,
                    'n_vertices': len(points),
                    'volume': data['best'].get('volume'),
                    'triangulation': triangulation,
                    'points': points,
                })
        except Exception as e:
            print(f"  Warning: Could not load {Path(filepath).name}: {e}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze dihedral angles in maximal volume triangulations')
    parser.add_argument('--data-dir', default='bin/results/data', help='Directory with optimization results')
    parser.add_argument('--max-denominator', type=int, default=100, help='Maximum denominator for rational approximation')
    parser.add_argument('--tolerance', type=float, default=1e-6, help='Error tolerance for rational match')

    args = parser.parse_args()

    print("="*70)
    print("DIHEDRAL ANGLE ANALYSIS: Maximal Volume Triangulations")
    print("="*70)
    print("\nConjecture: Dihedral angles are rational multiples of π (pπ/q)\n")

    # Load maximal volume configurations
    print("Loading maximal volume triangulations...")
    configs = load_maximal_volume_results(args.data_dir)

    if not configs:
        print("No configurations found!")
        sys.exit(1)

    print(f"Loaded {len(configs)} configurations\n")

    # Analyze each
    all_special_angles = []

    for config in configs:
        print(f"\n{'='*70}")
        print(f"File: {config['file']}")
        print(f"n={config['n_vertices']}, Volume={config.get('volume', 'N/A')}")
        print(f"{'='*70}")

        analysis = analyze_dihedral_angles(config['triangulation'], name=config['file'])

        if analysis is None:
            print("  Not realizable (skipping)")
            continue

        dihedrals = analysis['dihedrals']
        special = analysis['special_angles']

        print(f"\nDihedral angles: {len(dihedrals)} interior edges")

        if special:
            print(f"\n✨ Found {len(special)} special angles (q ≤ 20):")
            for d in special:
                print(f"  {d['rational']:>8} = {d['angle_deg']:7.3f}° "
                      f"(error: {d['error_deg']:.2e}°, edge: {d['edge']})")

            all_special_angles.extend(special)
        else:
            print("\n  No special angles found with small denominators")

        # Show all dihedral angles with their best rational approximations
        if len(dihedrals) <= 20:  # Only if not too many
            print(f"\nAll dihedral angles:")
            for d in dihedrals:
                if 'rational' in d:
                    print(f"  {d['edge']}: {d['angle_deg']:7.3f}° ≈ {d['rational']} "
                          f"(error: {d['error_deg']:.2e}°)")
                else:
                    print(f"  {d['edge']}: {d['angle_deg']:7.3f}° (no rational match)")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal configurations analyzed: {len(configs)}")
    print(f"Total special angles found: {len(all_special_angles)}")

    if all_special_angles:
        # Count by denominator
        from collections import Counter
        q_counts = Counter(d['q'] for d in all_special_angles)

        print(f"\nDistribution by denominator:")
        for q in sorted(q_counts.keys()):
            print(f"  q={q:2d}: {q_counts[q]} angles")

        # Most common patterns
        pattern_counts = Counter(d['rational'] for d in all_special_angles)
        print(f"\nMost common patterns:")
        for pattern, count in pattern_counts.most_common(10):
            print(f"  {pattern}: {count} occurrences")

    print()
