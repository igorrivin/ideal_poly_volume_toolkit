#!/usr/bin/env python3
"""
Generate a benchmark JSON file with:
- 1 optimized configuration (ARITHMETIC)
- 9 random configurations (NON-ARITHMETIC)

This demonstrates the Rivin-Delaunay theorem: maximal volume configurations
for a fixed combinatorial triangulation are always arithmetic.
"""

import numpy as np
import json
from datetime import datetime
import sys
sys.path.insert(0, '.')

from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    ideal_poly_volume_via_delaunay,
)
from ideal_poly_volume_toolkit.rivin_delaunay import (
    check_delaunay_realizability,
    optimize_hyperbolic_volume,
    realize_angles_as_points,
)
from ideal_poly_volume_toolkit.rivin_holonomy import (
    check_arithmeticity,
    check_arithmeticity_from_vertices,
)


def generate_random_sphere_points(n_points, seed=None):
    """Generate n random points uniformly on the unit sphere."""
    if seed is not None:
        np.random.seed(seed)

    points_3d = []
    for _ in range(n_points):
        vec = np.random.randn(3)
        vec = vec / np.linalg.norm(vec)
        points_3d.append(vec)

    points_3d = np.array(points_3d)

    # Stereographic projection from north pole to complex plane
    complex_points = []
    for x, y, z in points_3d:
        if z > 0.9999:
            complex_points.append(complex(np.inf, np.inf))
        else:
            w = complex(x / (1 - z), y / (1 - z))
            complex_points.append(w)

    return np.array(complex_points)


def generate_random_plane_points(n_points, seed=None, scale=2.0):
    """Generate n random points in the complex plane (truly generic)."""
    if seed is not None:
        np.random.seed(seed)

    # Use Gaussian distribution centered at origin
    real_parts = np.random.randn(n_points) * scale
    imag_parts = np.random.randn(n_points) * scale

    return real_parts + 1j * imag_parts


def optimize_for_fixed_combinatorics(complex_points):
    """Optimize volume for fixed combinatorics using Rivin-Delaunay."""
    finite_mask = np.isfinite(complex_points)
    finite_points = complex_points[finite_mask]

    # Get triangulation
    triangulation_indices = delaunay_triangulation_indices(finite_points)
    triangles = [tuple(tri) for tri in triangulation_indices]

    # Check realizability
    realizability = check_delaunay_realizability(triangles, verbose=False)
    if not realizability['realizable']:
        return None, None, triangles

    # Optimize
    opt_result = optimize_hyperbolic_volume(
        triangles,
        initial_angles=realizability['angles_radians'],
        verbose=False
    )

    if not opt_result['success']:
        return None, None, triangles

    # Realize geometry
    realization = realize_angles_as_points(
        triangles,
        opt_result['angles'],
        verbose=False
    )

    if realization['points'] is not None:
        vertex_list = realization['vertex_list']
        points_2d = realization['points']
        optimized_complex = np.zeros(len(vertex_list), dtype=complex)
        for i, v in enumerate(vertex_list):
            optimized_complex[v] = complex(points_2d[i, 0], points_2d[i, 1])
        return optimized_complex, opt_result['volume'], triangles

    return None, None, triangles


def main():
    print("Generating Arithmeticity Benchmark")
    print("=" * 60)

    n_points = 10
    base_seed = 42

    configs = []

    # Generate the first config: optimized (should be arithmetic)
    print(f"\nConfig 0: Optimized (Rivin-Delaunay)")
    print("-" * 40)

    random_points = generate_random_sphere_points(n_points, seed=base_seed)
    optimized_points, opt_volume, triangles = optimize_for_fixed_combinatorics(random_points)

    if optimized_points is not None:
        # Check arithmeticity
        arith_result = check_arithmeticity(triangles, verbose=True)

        finite_optimized = optimized_points[np.isfinite(optimized_points)]
        volume = ideal_poly_volume_via_delaunay(finite_optimized, use_bloch_wigner=True)

        config = {
            "id": 0,
            "type": "optimized",
            "description": "Rivin-Delaunay optimized configuration (maximal volume for fixed combinatorics)",
            "expected_arithmetic": True,
            "n_vertices": int(len(finite_optimized)),
            "n_triangles": int(len(triangles)),
            "volume": float(volume),
            "vertices_real": [float(z.real) for z in finite_optimized],
            "vertices_imag": [float(z.imag) for z in finite_optimized],
            "triangles": [[int(x) for x in t] for t in triangles],
            "holonomy_traces": [float(t) for t in arith_result['traces']],
            "is_arithmetic": bool(arith_result['is_arithmetic']),
            "seed": int(base_seed),
        }
        configs.append(config)

        print(f"  Volume: {volume:.6f}")
        print(f"  Arithmetic: {arith_result['is_arithmetic']}")
        print(f"  Traces: {arith_result['n_integral']}/{arith_result['n_generators']} integral")
    else:
        print("  ERROR: Optimization failed")

    # Generate 9 random configs (should be non-arithmetic)
    # Use random plane points (not sphere points) to get generic configurations
    for i in range(1, 10):
        seed = base_seed + i * 100  # Different seeds for variety

        print(f"\nConfig {i}: Random (seed={seed})")
        print("-" * 40)

        # Use random plane points for truly generic configurations
        finite_points = generate_random_plane_points(n_points, seed=seed)

        if len(finite_points) < 3:
            print("  ERROR: Not enough finite points")
            continue

        # Get triangulation
        triangulation_indices = delaunay_triangulation_indices(finite_points)
        triangles = [tuple(int(x) for x in tri) for tri in triangulation_indices]

        # Check arithmeticity using actual geometry (computes shears from vertices)
        arith_result = check_arithmeticity_from_vertices(finite_points, verbose=True)

        volume = ideal_poly_volume_via_delaunay(finite_points, use_bloch_wigner=True)

        config = {
            "id": i,
            "type": "random",
            "description": f"Random plane configuration (seed={seed})",
            "expected_arithmetic": False,
            "n_vertices": int(len(finite_points)),
            "n_triangles": int(len(triangles)),
            "volume": float(volume),
            "vertices_real": [float(z.real) for z in finite_points],
            "vertices_imag": [float(z.imag) for z in finite_points],
            "triangles": [[int(x) for x in t] for t in triangles],
            "holonomy_traces": [float(t) for t in arith_result['traces']],
            "is_arithmetic": bool(arith_result['is_arithmetic']),
            "seed": int(seed),
        }
        configs.append(config)

        print(f"  Volume: {volume:.6f}")
        print(f"  Arithmetic: {arith_result['is_arithmetic']}")
        print(f"  Traces: {arith_result['n_integral']}/{arith_result['n_generators']} integral")

    # Build final benchmark
    benchmark = {
        "metadata": {
            "title": "Arithmeticity Benchmark: Optimized vs Random Configurations",
            "description": "Demonstrates that Rivin-Delaunay optimized configurations are arithmetic (all holonomy traces are integers), while random configurations are typically not.",
            "created": datetime.now().isoformat(),
            "n_vertices": n_points,
            "theory": "By Rivin's theorem, the maximal volume configuration for a fixed combinatorial triangulation is unique and arithmetic. Random configurations do not achieve this maximum and are generically non-arithmetic.",
        },
        "summary": {
            "total_configs": len(configs),
            "optimized_configs": sum(1 for c in configs if c['type'] == 'optimized'),
            "random_configs": sum(1 for c in configs if c['type'] == 'random'),
            "arithmetic_count": sum(1 for c in configs if c['is_arithmetic']),
            "non_arithmetic_count": sum(1 for c in configs if not c['is_arithmetic']),
        },
        "configurations": configs,
    }

    # Save
    output_file = "arithmeticity_benchmark.json"
    with open(output_file, 'w') as f:
        json.dump(benchmark, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total configurations: {len(configs)}")
    print(f"Arithmetic: {benchmark['summary']['arithmetic_count']}")
    print(f"Non-arithmetic: {benchmark['summary']['non_arithmetic_count']}")
    print(f"\nSaved to: {output_file}")

    # Verify: optimized should be arithmetic, randoms should (mostly) not be
    optimized_configs = [c for c in configs if c['type'] == 'optimized']
    random_configs = [c for c in configs if c['type'] == 'random']

    print("\nVerification:")
    if all(c['is_arithmetic'] for c in optimized_configs):
        print("  [PASS] All optimized configs are arithmetic")
    else:
        print("  [FAIL] Some optimized configs are NOT arithmetic!")

    non_arith_randoms = sum(1 for c in random_configs if not c['is_arithmetic'])
    print(f"  Random configs: {non_arith_randoms}/{len(random_configs)} are non-arithmetic")


if __name__ == "__main__":
    main()
