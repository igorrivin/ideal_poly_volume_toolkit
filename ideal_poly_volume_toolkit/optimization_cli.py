"""Installed command-line adapter for H³ ideal-polyhedron optimization.

The numerical implementation belongs to :mod:`ideal_poly_volume_toolkit.optimization`.
This module owns only argument parsing, progress reporting, and JSON output so
the command works from an installed wheel as well as a repository checkout.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from .optimization import (
    analyze_ideal_polyhedron_structure,
    negative_volume_objective,
    optimize_ideal_polyhedron,
    optimization_result_to_dict,
    parameters_to_complex,
    spherical_to_complex as _spherical_to_complex,
)


def spherical_to_complex(theta, phi):
    """Compatibility wrapper around the canonical package projection."""

    return _spherical_to_complex(theta, phi)


def compute_volume(params, n_vertices):
    """Compatibility wrapper around the canonical package objective."""

    return negative_volume_objective(params, n_vertices)


def analyze_structure(vertices, triangulation):
    """Analyze the full boundary, including the implicit infinity vertex."""

    structure = analyze_ideal_polyhedron_structure(vertices, triangulation)
    return {
        key: value
        for key, value in structure.items()
        if key != "boundary_triangulation"
    }


def reconstruct_vertices(params, n_vertices):
    """Reconstruct finite complex vertices from optimization parameters."""

    return parameters_to_complex(params, n_vertices)


def build_parser() -> argparse.ArgumentParser:
    """Build the installed optimizer command's argument parser."""

    parser = argparse.ArgumentParser(
        description="Optimize ideal polyhedron volumes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --vertices 7 --trials 10
  %(prog)s --vertices 12 --trials 20 --maxiter 300
  %(prog)s --vertices 20 --trials 5 --output results/data/my_20vertex.json
        """,
    )
    parser.add_argument(
        "--vertices",
        "-v",
        type=int,
        required=True,
        help="Number of vertices (must be >= 4)",
    )
    parser.add_argument(
        "--trials",
        "-t",
        type=int,
        default=10,
        help="Number of optimization trials (default: 10)",
    )
    parser.add_argument(
        "--maxiter",
        "-m",
        type=int,
        default=200,
        help="Max iterations per trial (default: 200)",
    )
    parser.add_argument(
        "--popsize",
        "-p",
        type=int,
        default=15,
        help="SciPy population multiplier (default: 15)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Worker processes; -1 uses all CPUs (default: 1)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help=(
            "Output JSON file (default: "
            "results/data/{n}vertex_optimization_cli_TIMESTAMP.json)"
        ),
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed base (default: 42)",
    )
    return parser


def _validate_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.vertices < 4:
        parser.error("number of vertices must be at least 4")
    if args.trials < 1 or args.maxiter < 1 or args.popsize < 1:
        parser.error("trials, maxiter, and popsize must all be positive")
    if args.workers == 0 or args.workers < -1:
        parser.error("workers must be -1 or a positive integer")


def _default_output_path(n_vertices: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return Path(
        f"results/data/{n_vertices}vertex_optimization_cli_{timestamp}.json"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the installed optimizer command and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_arguments(parser, args)

    output_file = args.output or _default_output_path(args.vertices)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    n_free_vertices = args.vertices - 3
    n_params = n_free_vertices * 2
    print("=" * 70)
    print("Ideal Polyhedron Volume Optimization")
    print("=" * 70)
    print(f"Vertices:       {args.vertices}")
    print(f"Free vertices:  {n_free_vertices} (parameterized)")
    print(f"Parameters:     {n_params} (spherical coordinates)")
    print(f"Trials:         {args.trials}")
    print(f"Max iterations: {args.maxiter}")
    print(f"Population mult:{args.popsize} (nominal size {args.popsize * n_params})")
    print(f"Workers:        {args.workers}")
    print(f"Output:         {output_file}")
    print("=" * 70)

    running_best = -np.inf

    def report_trial(trial) -> None:
        nonlocal running_best
        is_best = trial.volume is not None and trial.volume > running_best
        if is_best:
            running_best = trial.volume
        print(f"\nTrial {trial.trial}/{args.trials}:")
        if trial.volume is None:
            print("  Volume: invalid configuration")
        else:
            print(f"  Volume: {trial.volume:.8f}")
        print(f"  Success: {trial.success}")
        print(f"  Iterations: {trial.iterations}")
        if is_best:
            print("  → NEW BEST!")

    result = optimize_ideal_polyhedron(
        args.vertices,
        n_trials=args.trials,
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        workers=args.workers,
        trial_callback=report_trial,
    )

    print("\n" + "=" * 70)
    print("BEST RESULT:")
    print("=" * 70)
    print(f"Volume: {result.best_volume:.10f}")

    structure = {
        key: value
        for key, value in result.structure.items()
        if key != "boundary_triangulation"
    }
    print("\nCombinatorial structure:")
    print(f"  Vertices:    {structure['n_vertices']}")
    print(f"  Edges:       {structure['n_edges']}")
    print(f"  Faces:       {structure['n_faces']}")
    print(f"  Euler char:  {structure['euler_characteristic']} (should be 2 for sphere)")
    print(f"  Vertex degrees: {structure['vertex_degrees']}")

    output_data = optimization_result_to_dict(
        result,
        source="cli",
        settings={
            "n_trials": args.trials,
            "maxiter": args.maxiter,
            "popsize": args.popsize,
            "workers": args.workers,
            "seed_base": args.seed,
        },
    )
    with output_file.open("w", encoding="utf-8") as stream:
        json.dump(output_data, stream, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
