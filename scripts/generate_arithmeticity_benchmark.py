#!/usr/bin/env python3
"""
Generate a benchmark JSON for testing LLM knowledge of arithmetic ideal
polyhedra in H^3.

Output format (see --help for CLI usage):

    {
      "metadata": {
        "title": str,
        "description": str,
        "theory": str,
        "created": str,              # ISO 8601 UTC
        "generator_version": str,
        "parameters": {
          "n_vertices": int,          # ideal vertices per polyhedron
          "n_random_configs": int,
          "base_seed": int
        }
      },
      "prompt": str,                  # the benchmark prompt for the LLM
      "configurations": [
        {
          "id": int,
          "input": {                  # shown to the LLM
            "vertices": [[re, im], ...],   # complex coords as pairs
            "triangles": [[i, j, k], ...], # 0-indexed face triples
            "n_vertices": int,
            "n_triangles": int
          },
          "ground_truth": {           # the answer
            "is_arithmetic": bool,
            "volume": float,
            "holonomy_traces": [float, ...],
            "generation_method": "optimized" | "random",
            "description": str,
            "seed": int
          }
        },
        ...
      ],
      "summary": {
        "total_configs": int,
        "arithmetic_count": int,
        "non_arithmetic_count": int
      }
    }

Mathematical content: by Rivin's theorem, the maximal-volume ideal
polyhedron for a fixed combinatorial triangulation is unique and
arithmetic, certified via the zero-shear / PSL(2,Z) criterion (all
Penner-Rivin holonomy traces are integers; sufficient, not necessary,
for group-theoretic arithmeticity).  Random configurations generically
fail the certificate.  Only combinatorial types whose Rivin maximum is
interior (no cocircular ties) can be certified; boundary maxima are
rejected early (require_interior).

Usage:
    python generate_arithmeticity_benchmark.py --n-vertices 10
    python generate_arithmeticity_benchmark.py --n-vertices 20 --n-optimized 25 --n-random 0
    python generate_arithmeticity_benchmark.py --n-vertices 20 --n-random 15 --seed 123
"""

import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np

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

GENERATOR_VERSION = "1.0.0"


def generate_random_sphere_points(n_points, rng):
    """Generate n random points uniformly on the unit sphere,
    returned as complex numbers via stereographic projection."""
    points_3d = rng.standard_normal((n_points, 3))
    points_3d /= np.linalg.norm(points_3d, axis=1, keepdims=True)

    complex_points = []
    for x, y, z in points_3d:
        if z > 1 - 1e-4:
            complex_points.append(complex(np.inf, np.inf))
        else:
            complex_points.append(complex(x / (1 - z), y / (1 - z)))

    return np.array(complex_points)


def generate_random_plane_points(n_points, rng, scale=2.0):
    """Generate n random points in the complex plane (Gaussian)."""
    return (rng.standard_normal(n_points) * scale
            + 1j * rng.standard_normal(n_points) * scale)


def optimize_for_fixed_combinatorics(complex_points):
    """Optimize volume for fixed combinatorics using Rivin-Delaunay."""
    finite_mask = np.isfinite(complex_points)
    finite_points = complex_points[finite_mask]

    triangulation_indices = delaunay_triangulation_indices(finite_points)
    triangles = [tuple(tri) for tri in triangulation_indices]

    realizability = check_delaunay_realizability(triangles, verbose=False)
    if not realizability['realizable']:
        return None, None, triangles

    # require_interior: witness generation must reject boundary (cocircular
    # tie) maxima early -- they are legitimate volume answers but can never
    # be zero-shear certified.
    opt_result = optimize_hyperbolic_volume(
        triangles,
        initial_angles=realizability['angles_radians'],
        verbose=False,
        require_interior=True
    )

    if not opt_result['success']:
        return None, None, triangles

    realization = realize_angles_as_points(
        triangles,
        opt_result['angles'],
        verbose=False
    )

    # The realization must actually SUCCEED (angles attained, target
    # combinatorics preserved).  An earlier version only checked
    # `points is not None` and stored the output of failed realizations
    # (27-38 degree angle errors) as "arithmetic" ground truth.
    if (realization['points'] is not None and realization.get('success')
            and realization.get('triangulation_preserved')):
        vertex_list = realization['vertex_list']
        points_2d = realization['points']
        optimized_complex = np.zeros(len(vertex_list), dtype=complex)
        for i, v in enumerate(vertex_list):
            optimized_complex[v] = complex(points_2d[i, 0], points_2d[i, 1])
        return optimized_complex, opt_result['volume'], triangles

    return None, None, triangles


def build_prompt(n_vertices, n_configs):
    """Build a mathematically precise prompt for the LLM benchmark.

    The prompt asks the model to *construct* arithmetic ideal polyhedra
    and return them in the same JSON schema used by the ``input`` block,
    so that responses can be validated programmatically.
    """
    return (
        f"Explicitly construct {n_configs} pairwise non-isometric convex "
        f"ideal polyhedra in hyperbolic 3-space H^3, each with exactly "
        f"{n_vertices} ideal vertices and all triangular faces, such that "
        f"every polyhedron is arithmetic, certified via the zero-shear / "
        f"PSL(2,Z) criterion: the traces of all Penner-Rivin holonomy "
        f"generators of the boundary surface\n"
        f"\n"
        f"    rho: pi_1(boundary surface) -> PSL(2, R)\n"
        f"\n"
        f"must be integers.  (Integer traces of a cofinite Fuchsian group "
        f"certify arithmeticity; the certificate is sufficient, not "
        f"necessary -- an arithmetic group could fail it.)\n"
        f"\n"
        f"(Such configurations exist: whenever the Rivin volume-maximizing "
        f"angle structure of a Delaunay combinatorial type lies in the "
        f"INTERIOR of the admissible angle polytope -- no cocircular ties "
        f"-- the maximizing vertex placement has all shears zero and hence "
        f"all-integer traces.)\n"
        f"\n"
        f"For each polyhedron provide:\n"
        f"  1. The ideal vertices z_1, ..., z_{{{n_vertices}}} as complex "
        f"numbers in C (boundary of the upper half-space model, identified "
        f"with the Riemann sphere via stereographic projection).\n"
        f"  2. The face triangulation as a list of vertex-index triples "
        f"[i, j, k] (0-indexed).\n"
        f"  3. The hyperbolic volume of the polyhedron.\n"
        f"\n"
        f"Return your answer as a JSON array of objects:\n"
        f"\n"
        f"[\n"
        f"  {{\n"
        f'    "vertices": [[re_1, im_1], [re_2, im_2], ...],\n'
        f'    "triangles": [[i, j, k], ...],\n'
        f'    "volume": <float>\n'
        f"  }},\n"
        f"  ...\n"
        f"]"
    )


def generate_optimized_config(config_id, n_vertices, seed, verbose=True):
    """Generate a single optimized (arithmetic) configuration.

    Draws random sphere points with the given seed, computes their
    Delaunay triangulation, then optimizes to the volume-maximizing
    (arithmetic) configuration via Rivin-Delaunay.

    Returns a config dict on success, or None if optimization fails.
    """
    rng = np.random.default_rng(seed)
    random_points = generate_random_sphere_points(n_vertices, rng)
    optimized_points, opt_volume, triangles = optimize_for_fixed_combinatorics(random_points)

    if optimized_points is None:
        return None

    # Certify arithmeticity from the ACTUAL optimized coordinates (shears
    # recomputed from cross-ratios).  The old geometry-free call certified
    # the idealized zero-shear structure — vacuously integral for any
    # triangulation — regardless of the stored coordinates.
    finite_optimized = optimized_points[np.isfinite(optimized_points)]
    arith_result = check_arithmeticity(
        triangles, vertices_complex=finite_optimized, verbose=verbose)
    volume = ideal_poly_volume_via_delaunay(finite_optimized, use_bloch_wigner=True)

    if verbose:
        print(f"  Volume: {volume:.6f}")
        print(f"  Arithmetic: {arith_result['is_arithmetic']}")
        print(f"  Traces: {arith_result['n_integral']}/{arith_result['n_generators']} integral")

    return {
        "id": config_id,
        "input": {
            "vertices": [[float(z.real), float(z.imag)] for z in finite_optimized],
            "triangles": [[int(x) for x in t] for t in triangles],
            "n_vertices": int(len(finite_optimized)),
            "n_triangles": int(len(triangles)),
        },
        "ground_truth": {
            "is_arithmetic": bool(arith_result['is_arithmetic']),
            "volume": float(volume),
            "holonomy_traces": [float(t) for t in arith_result['traces']],
            "generation_method": "optimized",
            "description": f"Rivin-Delaunay volume-maximizing configuration (seed={seed})",
            "seed": int(seed),
        },
    }


def generate_random_config(config_id, n_vertices, seed, verbose=True):
    """Generate a single random (generically non-arithmetic) configuration."""
    rng = np.random.default_rng(seed)
    finite_points = generate_random_plane_points(n_vertices, rng)

    triangulation_indices = delaunay_triangulation_indices(finite_points)
    triangles = [tuple(int(x) for x in tri) for tri in triangulation_indices]

    arith_result = check_arithmeticity_from_vertices(finite_points, verbose=verbose)
    volume = ideal_poly_volume_via_delaunay(finite_points, use_bloch_wigner=True)

    if verbose:
        print(f"  Volume: {volume:.6f}")
        print(f"  Arithmetic: {arith_result['is_arithmetic']}")
        print(f"  Traces: {arith_result['n_integral']}/{arith_result['n_generators']} integral")

    return {
        "id": config_id,
        "input": {
            "vertices": [[float(z.real), float(z.imag)] for z in finite_points],
            "triangles": [[int(x) for x in t] for t in triangles],
            "n_vertices": int(len(finite_points)),
            "n_triangles": int(len(triangles)),
        },
        "ground_truth": {
            "is_arithmetic": bool(arith_result['is_arithmetic']),
            "volume": float(volume),
            "holonomy_traces": [float(t) for t in arith_result['traces']],
            "generation_method": "random",
            "description": f"Random plane configuration (seed={seed})",
            "seed": int(seed),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate an arithmeticity benchmark for ideal polyhedra in H^3."
    )
    parser.add_argument(
        "--n-vertices", type=int, required=True,
        help="Number of ideal vertices per polyhedron."
    )
    parser.add_argument(
        "--n-optimized", type=int, default=1,
        help="Number of optimized (arithmetic) configurations (default: 1)."
    )
    parser.add_argument(
        "--n-random", type=int, default=9,
        help="Number of random (non-arithmetic) configurations (default: 9)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base RNG seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output JSON path (default: arithmeticity_benchmark_<n>v.json)."
    )
    args = parser.parse_args()

    n_vertices = args.n_vertices
    n_optimized = args.n_optimized
    n_random = args.n_random
    base_seed = args.seed
    output_file = args.output or f"arithmeticity_benchmark_{n_vertices}v.json"

    print(f"Generating Arithmeticity Benchmark ({n_vertices} vertices)")
    print(f"  Target: {n_optimized} optimized + {n_random} random configs")
    print("=" * 60)

    configs = []
    config_id = 0

    # --- Optimized configs (should be arithmetic) ---
    # Each seed gives different random sphere points → different Delaunay
    # triangulation → different arithmetic polyhedron after optimization.
    # Some seeds may fail (degenerate geometry), so we try extra seeds.
    n_succeeded = 0
    n_failed = 0
    seed_offset = 0

    while n_succeeded < n_optimized:
        seed_i = base_seed + seed_offset
        seed_offset += 1

        print(f"\nOptimized {n_succeeded}/{n_optimized} (seed={seed_i})")
        print("-" * 40)

        cfg = generate_optimized_config(config_id, n_vertices, seed_i)
        if cfg is not None:
            configs.append(cfg)
            config_id += 1
            n_succeeded += 1
        else:
            n_failed += 1
            print("  Optimization failed, trying next seed...")

        # Safety valve: don't loop forever
        if n_failed > n_optimized * 10:
            print(f"\n  WARNING: {n_failed} failures; stopping with "
                  f"{n_succeeded}/{n_optimized} optimized configs.")
            break

    if n_failed > 0:
        print(f"\n  ({n_failed} seeds failed optimization, {n_succeeded} succeeded)")

    # --- Random configs (generically non-arithmetic) ---
    for i in range(n_random):
        seed_i = base_seed + 10000 + i * 100  # well-separated from optimized seeds
        print(f"\nRandom {i + 1}/{n_random} (seed={seed_i})")
        print("-" * 40)

        cfg = generate_random_config(config_id, n_vertices, seed_i)
        configs.append(cfg)
        config_id += 1

    # --- Assemble benchmark ---
    n_arithmetic = sum(1 for c in configs if c['ground_truth']['is_arithmetic'])

    benchmark = {
        "metadata": {
            "title": "Arithmeticity Benchmark: Ideal Polyhedra in H^3",
            "description": (
                "Tests whether an LLM can construct or classify arithmetic "
                "ideal polyhedra.  Optimized configurations (Rivin-Delaunay) "
                "are arithmetic; random configurations are generically not."
            ),
            "theory": (
                "By Rivin's theorem, for any fixed combinatorial triangulation "
                "of a convex ideal polyhedron in H^3, the unique configuration "
                "maximizing hyperbolic volume is arithmetic: all traces of the "
                "holonomy representation are algebraic integers."
            ),
            "created": datetime.now(timezone.utc).isoformat(),
            "generator_version": GENERATOR_VERSION,
            "parameters": {
                "n_vertices": n_vertices,
                "n_optimized_configs": n_succeeded,
                "n_random_configs": n_random,
                "base_seed": base_seed,
            },
        },
        "prompt": build_prompt(n_vertices, n_configs=n_succeeded),
        "configurations": configs,
        "summary": {
            "total_configs": len(configs),
            "arithmetic_count": n_arithmetic,
            "non_arithmetic_count": len(configs) - n_arithmetic,
        },
    }

    with open(output_file, 'w') as f:
        json.dump(benchmark, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total configurations: {len(configs)}")
    print(f"Arithmetic: {n_arithmetic}")
    print(f"Non-arithmetic: {len(configs) - n_arithmetic}")
    print(f"Saved to: {output_file}")

    # Verification
    optimized = [c for c in configs if c['ground_truth']['generation_method'] == 'optimized']
    randoms = [c for c in configs if c['ground_truth']['generation_method'] == 'random']

    print("\nVerification:")
    if optimized:
        if all(c['ground_truth']['is_arithmetic'] for c in optimized):
            print(f"  [PASS] All {len(optimized)} optimized configs are arithmetic")
        else:
            n_fail = sum(1 for c in optimized if not c['ground_truth']['is_arithmetic'])
            print(f"  [FAIL] {n_fail}/{len(optimized)} optimized configs are NOT arithmetic!")

    if randoms:
        non_arith = sum(1 for c in randoms if not c['ground_truth']['is_arithmetic'])
        print(f"  Random configs: {non_arith}/{len(randoms)} are non-arithmetic")


if __name__ == "__main__":
    main()
