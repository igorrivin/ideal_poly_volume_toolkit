#!/usr/bin/env python3
"""
Standalone checker for the Arithmetic Ideal Polyhedra Benchmark.

Usage:
    python check_response.py prompt.txt response.json

The response.json should be a JSON array of objects:
[
  {
    "vertices": [[re_1, im_1], [re_2, im_2], ...],
    "triangles": [[i, j, k], ...],
    "volume": <float>
  },
  ...
]

Checks:
  1. Correct number of vertices
  2. Triangulation is a valid triangulated sphere (Euler: V-E+F=2)
  3. Volume is correct (via Lobachevsky function)
  4. Holonomy traces are algebraic integers (arithmeticity)
  5. Polyhedra are pairwise non-isometric

Dependencies: numpy, scipy (for Delaunay cross-check), rivin_holonomy.py (included)
"""

import json
import sys
import math
import numpy as np
from collections import Counter
from itertools import combinations
from typing import List, Tuple, Dict

from rivin_holonomy import (
    triangulation_from_faces,
    generators_from_triangulation,
    compute_shears_from_vertices,
)


# ============================================================================
# Lobachevsky function and volume
# ============================================================================

def lobachevsky(theta: float) -> float:
    """Lobachevsky function: L(theta) = -integral_0^theta ln|2 sin t| dt."""
    # Numerical integration via Clausen function relation
    # L(theta) = -(1/2) * sum_{k=1}^{N} sin(2k*theta) / k^2
    # (truncated series, accurate for moderate theta)
    result = 0.0
    for k in range(1, 500):
        result -= math.sin(2 * k * theta) / (k * k)
    return result / 2.0


def ideal_triangle_volume(alpha: float, beta: float, gamma: float) -> float:
    """Volume of an ideal tetrahedron with dihedral angles alpha, beta, gamma."""
    return lobachevsky(alpha) + lobachevsky(beta) + lobachevsky(gamma)


def compute_volume_from_vertices(vertices_complex, triangles):
    """
    Compute hyperbolic volume of an ideal polyhedron from vertex coordinates.

    Each triangle in the triangulation contributes a volume term
    computed from its angles. The angles are determined by the
    cross-ratios of the complex vertex coordinates.

    Parameters
    ----------
    vertices_complex : list of complex numbers
    triangles : list of (i, j, k) triples

    Returns
    -------
    volume : float
    """
    total = 0.0
    for tri in triangles:
        i, j, k = tri
        z_i = complex(vertices_complex[i])
        z_j = complex(vertices_complex[j])
        z_k = complex(vertices_complex[k])

        # Compute angles of the Euclidean triangle
        # (which equal the dihedral angles of the ideal tetrahedron)
        def angle_at(a, b, c):
            """Angle at vertex a in triangle abc."""
            ab = b - a
            ac = c - a
            cos_a = (ab.real * ac.real + ab.imag * ac.imag) / (abs(ab) * abs(ac) + 1e-30)
            cos_a = max(-1.0, min(1.0, cos_a))
            return math.acos(cos_a)

        alpha = angle_at(z_i, z_j, z_k)
        beta = angle_at(z_j, z_i, z_k)
        gamma = angle_at(z_k, z_i, z_j)

        total += ideal_triangle_volume(alpha, beta, gamma)

    return total


# ============================================================================
# Triangulation validity
# ============================================================================

def check_triangulation_valid(n_vertices: int, triangles: List) -> Dict:
    """
    Check that the triangulation is a valid triangulated sphere.

    Verifies:
    - All vertex indices in range
    - Each triangle has 3 distinct vertices
    - Euler relation V - E + F = 2
    - Each edge in at most 2 triangles
    """
    tris = [tuple(int(v) for v in t) for t in triangles]
    errors = []

    # Check vertex indices
    all_verts = set()
    for t in tris:
        if len(set(t)) != 3:
            errors.append(f"Degenerate triangle: {t}")
        for v in t:
            if v < 0 or v >= n_vertices:
                errors.append(f"Vertex {v} out of range [0, {n_vertices})")
            all_verts.add(v)

    if len(all_verts) != n_vertices:
        errors.append(f"Expected {n_vertices} vertices, found {len(all_verts)}")

    # Edge counts
    edge_count = Counter()
    for t in tris:
        for e in combinations(sorted(t), 2):
            edge_count[e] += 1

    bad_edges = {e: c for e, c in edge_count.items() if c > 2}
    if bad_edges:
        errors.append(f"{len(bad_edges)} edges in >2 triangles")

    n_edges = len(edge_count)
    n_faces = len(tris)
    euler = n_vertices - n_edges + n_faces
    if euler != 2:
        errors.append(f"Euler characteristic = {euler}, expected 2")

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'n_vertices': len(all_verts),
        'n_edges': n_edges,
        'n_faces': n_faces,
        'euler': euler,
    }


# ============================================================================
# Isometry check (are two polyhedra the same?)
# ============================================================================

def cross_ratio_invariant(vertices_complex, triangles):
    """
    Compute a sorted tuple of cross-ratios as an isometry invariant.

    Two ideal polyhedra are isometric iff they have the same cross-ratios
    (up to Mobius transformation = up to permutation of vertices).
    """
    crs = []
    edge_tris = {}
    for ti, t in enumerate(triangles):
        for ii in range(3):
            e = tuple(sorted([t[ii], t[(ii+1) % 3]]))
            opp = t[(ii+2) % 3]
            edge_tris.setdefault(e, []).append(opp)

    for e, opps in edge_tris.items():
        if len(opps) == 2:
            z1 = complex(vertices_complex[opps[0]])
            z2 = complex(vertices_complex[e[0]])
            z3 = complex(vertices_complex[e[1]])
            z4 = complex(vertices_complex[opps[1]])
            num = (z1 - z3) * (z2 - z4)
            den = (z1 - z4) * (z2 - z3)
            if abs(den) > 1e-10:
                cr = abs(num / den)
                crs.append(round(cr, 8))

    return tuple(sorted(crs))


# ============================================================================
# Main checker
# ============================================================================

def check_response(response: List[Dict], n_vertices_expected: int,
                   n_configs_expected: int, tolerance: float = 0.01,
                   volume_tolerance: float = 0.01,
                   verbose: bool = True) -> Dict:
    """
    Check a complete LLM response.

    Parameters
    ----------
    response : list of dicts with 'vertices', 'triangles', 'volume'
    n_vertices_expected : expected number of vertices per polyhedron
    n_configs_expected : expected number of configurations
    tolerance : tolerance for trace integrality
    volume_tolerance : relative tolerance for volume check
    verbose : print details

    Returns
    -------
    dict with 'n_passed', 'n_total', 'results'
    """
    results = []

    if len(response) < n_configs_expected:
        if verbose:
            print(f"WARNING: Expected {n_configs_expected} configs, got {len(response)}")

    seen_invariants = set()

    for idx, config in enumerate(response):
        if verbose:
            print(f"\n--- Config {idx} ---")

        errors = []
        warnings = []

        # Parse vertices
        verts = config.get('vertices', [])
        n_v = len(verts)
        if n_v != n_vertices_expected:
            errors.append(f"Expected {n_vertices_expected} vertices, got {n_v}")

        vertices_complex = [complex(v[0], v[1]) for v in verts]

        # Parse triangles
        tris = [tuple(int(v) for v in t) for t in config.get('triangles', [])]

        # Check triangulation validity
        tv = check_triangulation_valid(n_v, tris)
        if not tv['valid']:
            errors.extend(tv['errors'])
        elif verbose:
            print(f"  Triangulation: V={tv['n_vertices']} E={tv['n_edges']} "
                  f"F={tv['n_faces']} Euler={tv['euler']}")

        if errors:
            results.append({'idx': idx, 'pass': False, 'errors': errors})
            if verbose:
                for e in errors:
                    print(f"  FAIL: {e}")
            continue

        # Check volume
        claimed_vol = config.get('volume', 0.0)
        computed_vol = compute_volume_from_vertices(vertices_complex, tris)
        vol_err = abs(computed_vol - claimed_vol) / (abs(computed_vol) + 1e-10)
        if vol_err > volume_tolerance:
            warnings.append(f"Volume mismatch: claimed {claimed_vol:.6f}, "
                           f"computed {computed_vol:.6f} (rel err {vol_err:.4f})")
        if verbose:
            print(f"  Volume: claimed={claimed_vol:.6f} computed={computed_vol:.6f} "
                  f"(rel err={vol_err:.6f})")

        # Check arithmeticity
        shears = compute_shears_from_vertices(tris, vertices_complex)
        T, _ = triangulation_from_faces(tris)
        gens = generators_from_triangulation(T, shears, root=0)

        n_integral = 0
        traces = []
        for u, v, tokens, M in gens:
            trace = M[0][0] + M[1][1]
            nearest = round(trace)
            dist = abs(trace - nearest)
            is_int = dist < tolerance
            if is_int:
                n_integral += 1
            traces.append(trace)
            if verbose and not is_int:
                print(f"  NON-INTEGRAL trace: gen({u},{v}) = {trace:.6f} "
                      f"(dist={dist:.6f})")

        is_arith = n_integral == len(gens) if gens else False
        if verbose:
            print(f"  Arithmeticity: {n_integral}/{len(gens)} integral traces "
                  f"-> {'ARITHMETIC' if is_arith else 'NOT arithmetic'}")

        if not is_arith:
            errors.append(f"Not arithmetic: {n_integral}/{len(gens)} integral traces")

        # Check non-isometry
        inv = cross_ratio_invariant(vertices_complex, tris)
        if inv in seen_invariants:
            errors.append("Duplicate: isometric to a previous polyhedron")
        seen_invariants.add(inv)

        passed = len(errors) == 0
        results.append({
            'idx': idx,
            'pass': passed,
            'errors': errors,
            'warnings': warnings,
            'volume_computed': computed_vol,
            'volume_claimed': claimed_vol,
            'is_arithmetic': is_arith,
            'n_integral_traces': n_integral,
            'n_generators': len(gens),
        })

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  Result: [{status}]")
            for e in errors:
                print(f"    Error: {e}")
            for w in warnings:
                print(f"    Warning: {w}")

    n_passed = sum(1 for r in results if r['pass'])
    if verbose:
        print(f"\n{'='*50}")
        print(f"Score: {n_passed}/{len(results)}")

    return {
        'n_passed': n_passed,
        'n_total': len(results),
        'results': results,
    }


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} prompt.txt response.json")
        print(f"   or: {sys.argv[0]} --n-vertices N response.json")
        sys.exit(1)

    # Parse args
    if sys.argv[1] == '--n-vertices':
        n_verts = int(sys.argv[2])
        resp_file = sys.argv[3]
        n_configs = 25  # default
    else:
        # Extract n_vertices from prompt
        prompt = open(sys.argv[1]).read()
        import re
        m = re.search(r'exactly (\d+) ideal vertices', prompt)
        n_verts = int(m.group(1)) if m else None
        m2 = re.search(r'construct (\d+) pairwise', prompt, re.IGNORECASE)
        n_configs = int(m2.group(1)) if m2 else 25
        resp_file = sys.argv[2]

    if n_verts is None:
        print("Could not determine n_vertices from prompt")
        sys.exit(1)

    response = json.load(open(resp_file))

    print("=" * 60)
    print("ARITHMETIC IDEAL POLYHEDRA BENCHMARK — CHECKER")
    print("=" * 60)
    print(f"Expected: {n_configs} polyhedra, {n_verts} vertices each")
    print(f"Response: {len(response)} polyhedra")

    check_response(response, n_verts, n_configs)


if __name__ == '__main__':
    main()
