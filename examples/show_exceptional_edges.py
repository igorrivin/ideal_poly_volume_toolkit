#!/usr/bin/env python3
"""
Show the exceptional edges that don't fit the rational pattern.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.spatial import Delaunay
from fractions import Fraction
from ideal_poly_volume_toolkit.rivin_delaunay import check_delaunay_realizability, build_edge_adjacency


def continued_fraction_convergents(x, max_terms=30):
    """Compute convergents of continued fraction expansion."""
    convergents = []
    a = []
    remainder = x

    for _ in range(max_terms):
        floor_val = int(np.floor(remainder))
        a.append(floor_val)
        if abs(remainder - floor_val) < 1e-12:
            break
        if remainder - floor_val < 1e-12:
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


# Generate same random configuration
n_vertices = 89
seed = 42

np.random.seed(seed)
radii = np.sqrt(np.random.uniform(0, 1, n_vertices))
angles = np.random.uniform(0, 2*np.pi, n_vertices)
vertices_complex = radii * np.exp(1j * angles)
points = np.column_stack([vertices_complex.real, vertices_complex.imag])

# Compute triangulation
tri = Delaunay(points)
triangulation = [tuple(sorted(simplex)) for simplex in tri.simplices]
triangulation = sorted(set(triangulation))

# Get optimal angles
result = check_delaunay_realizability(triangulation, verbose=False, strict=False)
angles_scaled = result['angles']
angles_radians = angles_scaled * np.pi
n_triangles = len(triangulation)
angles_array = angles_radians.reshape((n_triangles, 3))

# Compute dihedrals
edge_adjacency = build_edge_adjacency(triangulation)
all_dihedrals = []

for edge, opposite_corners in sorted(edge_adjacency.items()):
    if len(opposite_corners) == 2:
        angle1 = angles_array[opposite_corners[0][0], opposite_corners[0][1]]
        angle2 = angles_array[opposite_corners[1][0], opposite_corners[1][1]]
        dihedral = angle1 + angle2
        normalized = dihedral / np.pi

        # Find best rational approximation
        convergents = continued_fraction_convergents(normalized, max_terms=30)
        if convergents:
            best_p, best_q = convergents[-1]
            error = abs(normalized - best_p / best_q)

            all_dihedrals.append({
                'edge': edge,
                'angle_deg': float(np.degrees(dihedral)),
                'normalized': float(normalized),
                'p': int(best_p),
                'q': int(best_q),
                'error': float(error),
            })

print(f"═══════════════════════════════════════════════════════════════")
print(f"EXCEPTIONAL EDGES ANALYSIS (89 vertices, seed 42)")
print(f"═══════════════════════════════════════════════════════════════")

# Check which edges don't have denominators dividing 108
max_denominator = 200
small_error = 1e-6

exceptional_edges = []
for d in all_dihedrals:
    # Check if q divides 108
    if d['q'] <= max_denominator and d['error'] < small_error:
        if 108 % d['q'] != 0:
            exceptional_edges.append(d)
    else:
        # Large denominator or large error
        exceptional_edges.append(d)

print(f"\nTotal interior edges: {len(all_dihedrals)}")
print(f"Exceptional edges (q does not divide 108, or q > {max_denominator}, or error > {small_error}): {len(exceptional_edges)}")

if exceptional_edges:
    print(f"\n{'─'*78}")
    print(f"EXCEPTIONAL EDGES")
    print(f"{'─'*78}")
    print(f"\n{'Edge':>12} {'Degrees':>10} {'θ/π':>12} {'Best p/q':>15} {'Error':>12}")
    print(f"{'-'*63}")

    for d in exceptional_edges:
        rational = f"{d['p']}/{d['q']}" if d['q'] > 1 else f"{d['p']}"
        print(f"  {str(d['edge']):>10} {d['angle_deg']:9.3f}° {d['normalized']:11.9f}  {rational:>15} {d['error']:11.2e}")

    print(f"\n{'─'*78}")
    print(f"ANALYSIS OF EXCEPTIONAL DENOMINATORS")
    print(f"{'─'*78}")

    exceptional_qs = sorted(set(d['q'] for d in exceptional_edges))
    print(f"\nDenominators: {exceptional_qs}")

    # Check for patterns
    for q in exceptional_qs:
        count = sum(1 for d in exceptional_edges if d['q'] == q)
        # Factor q
        factors = []
        temp = q
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89]:
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors.append(f"{p}^{exp}" if exp > 1 else str(p))
        if temp > 1:
            factors.append(str(temp))

        factorization = "·".join(factors) if factors else "1"

        # Check divisibility
        divides_108 = (108 % q == 0) if q <= 108 else False

        print(f"\n  q = {q:>4} ({count} edges)")
        print(f"    Factorization: {factorization}")
        print(f"    Divides 108: {divides_108}")
        if not divides_108 and q <= 108:
            # Check GCD with 108
            from math import gcd
            g = gcd(q, 108)
            print(f"    GCD(q, 108) = {g}")

else:
    print("\n✓ No exceptional edges found - all fit the pattern!")

print(f"\n{'─'*78}")
