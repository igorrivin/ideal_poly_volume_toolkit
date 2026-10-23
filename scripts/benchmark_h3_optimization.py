#!/usr/bin/env python3
"""Reproducible throughput benchmark for the canonical H^3 objective.

This is a microbenchmark, not a time-to-solution study. It evaluates the same
deterministic parameter vectors with:

1. the current vectorized Bloch--Wigner objective; and
2. a local, frozen copy of the 0.3 CLI's scalar Torch/96-term Lobachevsky
   forward path.

The two methods intentionally do not agree bit-for-bit because the historical
path truncates an infinite series. Numerical compatibility is checked before
timings are reported. CSV output is opt-in; canonical release data should be
recorded with ``--require-clean`` from the final release commit.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import math
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Callable

import numpy as np
import scipy
import torch

from ideal_poly_volume_toolkit.geometry import delaunay_triangulation_indices
from ideal_poly_volume_toolkit.optimization import (
    OBJECTIVE_PENALTY,
    negative_volume_objective,
)


BENCHMARK_SCHEMA_VERSION = 1
BENCHMARK_VERTICES = (7, 20, 50)
LEGACY_SERIES_TERMS = 96
DEFAULT_SEED = 42
DEFAULT_RELATIVE_TOLERANCE = 1.0e-4
DEFAULT_ABSOLUTE_TOLERANCE = 1.0e-8


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return result


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return result


def deterministic_parameters(n_vertices: int, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Return the fixed, nondegenerate parameter vector for one benchmark case."""
    if n_vertices not in BENCHMARK_VERTICES:
        raise ValueError(f"benchmark vertices must be one of {BENCHMARK_VERTICES}")

    rng = np.random.default_rng(np.random.SeedSequence([seed, n_vertices]))
    n_free = n_vertices - 3
    params = np.empty(2 * n_free, dtype=np.float64)
    params[0::2] = rng.uniform(0.1, math.pi - 0.1, size=n_free)
    params[1::2] = rng.uniform(0.0, 2.0 * math.pi, size=n_free)
    return params


def _legacy_spherical_to_complex(theta: float, phi: float) -> complex:
    """Frozen scalar parameter conversion from the 0.3 optimizer CLI."""
    return complex(np.tan(theta / 2.0) * np.exp(1j * phi))


def _legacy_angle_at(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """Frozen scalar Torch angle calculation from the 0.3 geometry path."""
    u = torch.stack(((b - a).real, (b - a).imag), dim=-1)
    v = torch.stack(((c - a).real, (c - a).imag), dim=-1)
    dot = (u * v).sum(dim=-1)
    cross = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    return torch.atan2(torch.abs(cross), dot)


def _legacy_lobachevsky_series_value(
    theta: torch.Tensor,
    series_terms: int,
) -> torch.Tensor:
    """Frozen 96-term scalar series value used by the 0.3 CLI."""
    k = torch.arange(
        1,
        series_terms + 1,
        device=theta.device,
        dtype=theta.dtype,
    )
    return 0.5 * torch.sum(torch.sin(2 * k * theta) / (k * k), dim=0)


class _LegacyLobachevskyFn(torch.autograd.Function):
    """Frozen custom-autograd wrapper used by the 0.3 scalar objective."""

    @staticmethod
    def forward(ctx, theta: torch.Tensor, series_terms: int) -> torch.Tensor:
        ctx.save_for_backward(theta)
        return _legacy_lobachevsky_series_value(theta, series_terms)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (theta,) = ctx.saved_tensors
        gradient = -grad_output * torch.log(2.0 * torch.abs(torch.sin(theta)))
        return gradient, None


def _legacy_lobachevsky_series(theta: torch.Tensor) -> torch.Tensor:
    return _LegacyLobachevskyFn.apply(theta, LEGACY_SERIES_TERMS)


def _legacy_triangle_volume(
    z1: torch.Tensor,
    z2: torch.Tensor,
    z3: torch.Tensor,
) -> float:
    a1 = _legacy_angle_at(z1, z2, z3)
    a2 = _legacy_angle_at(z2, z3, z1)
    a3 = _legacy_angle_at(z3, z1, z2)
    return float(
        (
            _legacy_lobachevsky_series(a1)
            + _legacy_lobachevsky_series(a2)
            + _legacy_lobachevsky_series(a3)
        ).item()
    )


def legacy_negative_volume_objective(
    params: np.ndarray,
    n_vertices: int,
) -> float:
    """Evaluate the frozen 0.3 scalar Torch objective.

    This function deliberately keeps the historical Python loops and repeated
    scalar Torch series construction. It is benchmark reference code, not an
    alternate production implementation.
    """
    try:
        points = [0.0j, 1.0 + 0.0j]
        for index in range(n_vertices - 3):
            points.append(
                _legacy_spherical_to_complex(
                    float(params[2 * index]),
                    float(params[2 * index + 1]),
                )
            )

        vertices = np.asarray(points, dtype=np.complex128)
        triangulation = delaunay_triangulation_indices(vertices)
        vertices_torch = torch.tensor(vertices, dtype=torch.complex128)

        total_volume = 0.0
        for i, j, k in triangulation:
            total_volume += _legacy_triangle_volume(
                vertices_torch[i],
                vertices_torch[j],
                vertices_torch[k],
            )
    except Exception:
        # The 0.3 CLI converted any construction/evaluation failure into the
        # same finite differential-evolution penalty.
        return OBJECTIVE_PENALTY

    return -total_volume


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_command(repository: Path, *args: str) -> tuple[int | None, str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, ""
    return completed.returncode, completed.stdout.strip()


def collect_provenance() -> dict[str, object]:
    """Collect enough source and environment identity to interpret raw timings."""
    script_path = Path(__file__).resolve()
    repository = script_path.parent.parent
    commit_code, commit = _git_command(repository, "rev-parse", "HEAD")
    # ``git status`` can spend minutes invoking Git LFS clean filters across
    # the historical result corpus.  ``diff-index`` checks tracked state from
    # the index, while ``ls-files`` checks untracked paths; together they are a
    # complete and much cheaper cleanliness test.  Exit code 1 from
    # ``diff-index --quiet`` means dirty, not command failure.
    tracked_code, _ = _git_command(
        repository,
        "diff-index",
        "--quiet",
        "HEAD",
        "--",
    )
    untracked_code, untracked = _git_command(
        repository,
        "ls-files",
        "--others",
        "--exclude-standard",
    )
    if tracked_code in {0, 1} and untracked_code == 0:
        git_dirty = tracked_code == 1 or bool(untracked)
    else:
        git_dirty = None

    source_paths = {
        "benchmark": script_path,
        "geometry": repository / "ideal_poly_volume_toolkit" / "geometry.py",
        "optimization": repository / "ideal_poly_volume_toolkit" / "optimization.py",
        "pyproject": repository / "pyproject.toml",
    }
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}

    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit if commit_code == 0 else "unknown",
        "git_dirty": git_dirty,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", ""),
        "source_hashes": source_hashes,
    }


def _measure(
    function: Callable[[np.ndarray, int], float],
    params: np.ndarray,
    n_vertices: int,
    loops: int,
) -> tuple[int, float]:
    start = perf_counter_ns()
    value = math.nan
    for _ in range(loops):
        value = float(function(params, n_vertices))
    elapsed_ns = perf_counter_ns() - start
    return elapsed_ns, value


def run_benchmark(
    *,
    seed: int,
    warmups: int,
    samples: int,
    loops: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run all fixed cases and return case summaries plus raw timing samples."""
    summaries: list[dict[str, object]] = []
    raw_samples: list[dict[str, object]] = []
    implementations: tuple[tuple[str, Callable[[np.ndarray, int], float]], ...] = (
        ("current_vectorized_bloch_wigner", negative_volume_objective),
        ("legacy_0_3_torch_series96", legacy_negative_volume_objective),
    )

    for n_vertices in BENCHMARK_VERTICES:
        params = deterministic_parameters(n_vertices, seed)
        current_value = float(negative_volume_objective(params, n_vertices))
        legacy_value = float(legacy_negative_volume_objective(params, n_vertices))
        absolute_difference = abs(current_value - legacy_value)
        relative_difference = absolute_difference / max(abs(current_value), 1.0e-300)
        valid = (
            math.isfinite(current_value)
            and math.isfinite(legacy_value)
            and current_value < 0.0
            and legacy_value < 0.0
        )
        compatible = valid and math.isclose(
            current_value,
            legacy_value,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        )

        for _ in range(warmups):
            for _, function in implementations:
                function(params, n_vertices)

        previous_gc_state = gc.isenabled()
        gc.disable()
        try:
            for sample_index in range(samples):
                # Alternate execution order to reduce systematic thermal/cache
                # bias while retaining the actual order in the raw data.
                ordered = (
                    implementations
                    if sample_index % 2 == 0
                    else tuple(reversed(implementations))
                )
                for order_index, (name, function) in enumerate(ordered):
                    elapsed_ns, measured_value = _measure(
                        function,
                        params,
                        n_vertices,
                        loops,
                    )
                    expected = (
                        current_value if name.startswith("current") else legacy_value
                    )
                    if measured_value != expected:
                        raise RuntimeError(
                            f"{name} returned a nondeterministic value for N={n_vertices}"
                        )
                    raw_samples.append(
                        {
                            "n_vertices": n_vertices,
                            "implementation": name,
                            "sample_index": sample_index,
                            "order_index": order_index,
                            "loops": loops,
                            "elapsed_ns": elapsed_ns,
                            "ns_per_call": elapsed_ns / loops,
                            "objective": measured_value,
                            "current_objective": current_value,
                            "legacy_objective": legacy_value,
                            "absolute_difference": absolute_difference,
                            "relative_difference": relative_difference,
                            "compatible": compatible,
                        }
                    )
        finally:
            if previous_gc_state:
                gc.enable()

        timings: dict[str, list[float]] = {}
        for row in raw_samples:
            if row["n_vertices"] != n_vertices:
                continue
            timings.setdefault(str(row["implementation"]), []).append(
                float(row["ns_per_call"])
            )
        current_median = statistics.median(timings["current_vectorized_bloch_wigner"])
        legacy_median = statistics.median(timings["legacy_0_3_torch_series96"])
        summaries.append(
            {
                "n_vertices": n_vertices,
                "current_objective": current_value,
                "legacy_objective": legacy_value,
                "absolute_difference": absolute_difference,
                "relative_difference": relative_difference,
                "compatible": compatible,
                "current_median_ns": current_median,
                "legacy_median_ns": legacy_median,
                "speedup": legacy_median / current_median,
            }
        )

    return summaries, raw_samples


def _print_report(
    summaries: list[dict[str, object]],
    provenance: dict[str, object],
) -> None:
    hashes = provenance["source_hashes"]
    assert isinstance(hashes, dict)
    print("H^3 objective-throughput benchmark")
    print(
        f"Python {provenance['python_version']} | NumPy {provenance['numpy_version']} "
        f"| SciPy {provenance['scipy_version']} | Torch {provenance['torch_version']}"
    )
    print(
        f"git={provenance['git_commit']} dirty={provenance['git_dirty']} "
        f"machine={provenance['machine']}"
    )
    print(
        "source sha256 "
        f"benchmark={str(hashes['benchmark'])[:12]} "
        f"geometry={str(hashes['geometry'])[:12]} "
        f"optimization={str(hashes['optimization'])[:12]}"
    )
    print()
    print(
        "  N   current ms   legacy ms   speedup   abs difference   rel difference   compatible"
    )
    for summary in summaries:
        print(
            f"{summary['n_vertices']:>3}   "
            f"{float(summary['current_median_ns']) / 1.0e6:>10.4f}   "
            f"{float(summary['legacy_median_ns']) / 1.0e6:>9.4f}   "
            f"{float(summary['speedup']):>7.1f}x   "
            f"{float(summary['absolute_difference']):>14.6g}   "
            f"{float(summary['relative_difference']):>14.6g}   "
            f"{str(summary['compatible']):>10}"
        )
    if provenance["git_dirty"] is not False:
        print(
            "\nThis is not a canonical release baseline: the source tree is dirty "
            "or its Git state is unavailable."
        )


def _write_csv(
    output: Path,
    *,
    raw_samples: list[dict[str, object]],
    provenance: dict[str, object],
    seed: int,
    warmups: int,
    samples: int,
    relative_tolerance: float,
    absolute_tolerance: float,
    force: bool,
) -> None:
    if output.exists() and not force:
        raise FileExistsError(
            f"refusing to replace existing output {output}; pass --force to replace it"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    hashes = provenance["source_hashes"]
    assert isinstance(hashes, dict)
    metadata = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "recorded_at_utc": provenance["recorded_at_utc"],
        "git_commit": provenance["git_commit"],
        "git_dirty": provenance["git_dirty"],
        "python_implementation": provenance["python_implementation"],
        "python_version": provenance["python_version"],
        "numpy_version": provenance["numpy_version"],
        "scipy_version": provenance["scipy_version"],
        "torch_version": provenance["torch_version"],
        "platform": provenance["platform"],
        "machine": provenance["machine"],
        "processor": provenance["processor"],
        "cpu_count": provenance["cpu_count"],
        "torch_num_threads": provenance["torch_num_threads"],
        "omp_num_threads": provenance["omp_num_threads"],
        "mkl_num_threads": provenance["mkl_num_threads"],
        "openblas_num_threads": provenance["openblas_num_threads"],
        "benchmark_sha256": hashes["benchmark"],
        "geometry_sha256": hashes["geometry"],
        "optimization_sha256": hashes["optimization"],
        "pyproject_sha256": hashes["pyproject"],
        "seed": seed,
        "warmups": warmups,
        "samples": samples,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "legacy_series_terms": LEGACY_SERIES_TERMS,
    }
    rows = [{**metadata, **sample} for sample in raw_samples]
    fieldnames = list(rows[0])
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the current H^3 objective against the frozen 0.3 "
            "scalar Torch reference at N=7,20,50."
        )
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmups", type=_nonnegative_int, default=2)
    parser.add_argument("--samples", type=_positive_int, default=9)
    parser.add_argument("--loops", type=_positive_int, default=5)
    parser.add_argument(
        "--rtol",
        type=_nonnegative_float,
        default=DEFAULT_RELATIVE_TOLERANCE,
        help="relative compatibility tolerance (default: %(default)g)",
    )
    parser.add_argument(
        "--atol",
        type=_nonnegative_float,
        default=DEFAULT_ABSOLUTE_TOLERANCE,
        help="absolute compatibility tolerance (default: %(default)g)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional raw-sample CSV path; no file is written by default",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow replacement of an existing --output file",
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="fail unless Git reports a clean worktree",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    provenance = collect_provenance()
    if args.require_clean and provenance["git_dirty"] is not False:
        print(
            "error: --require-clean requested, but the worktree is dirty or "
            "Git state is unavailable",
            file=sys.stderr,
        )
        return 2

    summaries, raw_samples = run_benchmark(
        seed=args.seed,
        warmups=args.warmups,
        samples=args.samples,
        loops=args.loops,
        relative_tolerance=args.rtol,
        absolute_tolerance=args.atol,
    )
    _print_report(summaries, provenance)

    if args.output is not None:
        try:
            _write_csv(
                args.output,
                raw_samples=raw_samples,
                provenance=provenance,
                seed=args.seed,
                warmups=args.warmups,
                samples=args.samples,
                relative_tolerance=args.rtol,
                absolute_tolerance=args.atol,
                force=args.force,
            )
        except OSError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        print(f"\nRaw samples written to {args.output}")

    return 0 if all(bool(summary["compatible"]) for summary in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
