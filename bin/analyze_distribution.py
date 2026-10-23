#!/usr/bin/env python3
"""
General-purpose wrapper for analyzing volume distributions of ideal polyhedra.

Usage:
    python bin/analyze_distribution.py --vertices 4 --samples 10000
    python bin/analyze_distribution.py --vertices 6 --samples 50000 --output custom_plot.png
    python bin/analyze_distribution.py --vertices 5 --samples 20000 --fit beta --data results.json
    python bin/analyze_distribution.py -v 4 -n 5000 --fit gamma --bootstrap 2000 --confidence 0.99
"""

import argparse
import inspect
import json
from multiprocessing import Pool, cpu_count
from numbers import Integral
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import sys
from scipy import stats
from scipy.spatial import QhullError

from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay


# Sampling is divided independently of the process count.  Consequently the
# same seed and request produce the same ordered stream even when ``n_jobs``
# changes.  The retry bounds keep pathological or monkey-patched samplers from
# looping forever while still making under-filled requests explicit.
LOGICAL_BATCH_SIZE = 256
MAX_AUTOMATIC_WORKERS = 16
MAX_VERTEX_DRAW_ATTEMPTS = 100
MAX_CONFIGURATION_ATTEMPTS_PER_SAMPLE = 20
KNOWN_NUMERICAL_EXCEPTIONS = (QhullError, FloatingPointError, np.linalg.LinAlgError)


def sample_random_vertex(rng: np.random.Generator):
    """
    Sample a uniform random point on the unit sphere and project to complex plane.
    Uses stereographic projection from north pole.
    """
    # Sample uniform point on sphere using Gaussian method
    vec = rng.standard_normal(3)
    norm = np.linalg.norm(vec)
    if not np.isfinite(norm) or norm == 0.0:
        return None
    vec = vec / norm
    x, y, z = vec

    # Skip near north pole (maps to infinity)
    if z > 0.999:
        return None

    # Stereographic projection
    w = complex(x/(1-z), y/(1-z))
    return w


def _draw_finite_configuration(n_vertices, rng):
    """Draw one finite configuration and return it with draw counters.

    Each required random vertex gets at most
    :data:`MAX_VERTEX_DRAW_ATTEMPTS` attempts.  Rejected pole-adjacent,
    non-finite, and near-duplicate draws are counted rather than causing the
    whole configuration to be silently short.
    """
    vertices = [complex(0, 0), complex(1, 0)]
    n_draws = 0
    n_rejected_draws = 0

    for _ in range(n_vertices - 3):
        for _ in range(MAX_VERTEX_DRAW_ATTEMPTS):
            n_draws += 1
            vertex = sample_random_vertex(rng)
            if vertex is None or not np.isfinite(vertex):
                n_rejected_draws += 1
                continue
            if any(abs(vertex - existing) < 0.01 for existing in vertices):
                n_rejected_draws += 1
                continue
            vertices.append(vertex)
            break
        else:
            return None, n_draws, n_rejected_draws

    return np.asarray(vertices, dtype=np.complex128), n_draws, n_rejected_draws


def _worker_sample_volumes(args):
    """Produce one deterministic logical batch and its rejection counters."""
    n_vertices, target_samples, seed, batch_index, _series_terms = args
    rng = np.random.default_rng(np.random.SeedSequence([seed, batch_index]))
    max_configuration_attempts = (
        target_samples * MAX_CONFIGURATION_ATTEMPTS_PER_SAMPLE
    )
    volumes = []
    counters = {
        "n_vertex_draws": 0,
        "n_rejected_vertex_draws": 0,
        "n_configurations_attempted": 0,
        "n_rejected_configurations": 0,
        "n_vertex_generation_failures": 0,
        "n_numerical_evaluation_failures": 0,
        "n_invalid_volume_evaluations": 0,
    }

    while (
        len(volumes) < target_samples
        and counters["n_configurations_attempted"] < max_configuration_attempts
    ):
        counters["n_configurations_attempted"] += 1
        vertices, n_draws, n_rejected_draws = _draw_finite_configuration(
            n_vertices, rng
        )
        counters["n_vertex_draws"] += n_draws
        counters["n_rejected_vertex_draws"] += n_rejected_draws

        if vertices is None:
            counters["n_vertex_generation_failures"] += 1
            counters["n_rejected_configurations"] += 1
            continue

        try:
            volume = ideal_poly_volume_via_delaunay(
                vertices, mode="bloch_wigner"
            )
        except KNOWN_NUMERICAL_EXCEPTIONS:
            counters["n_numerical_evaluation_failures"] += 1
            counters["n_rejected_configurations"] += 1
            continue

        if not np.isfinite(volume) or not 0.0 < volume < 1000.0:
            counters["n_invalid_volume_evaluations"] += 1
            counters["n_rejected_configurations"] += 1
            continue

        volumes.append(float(volume))

    return {
        "batch_index": batch_index,
        "target_samples": target_samples,
        "volumes": np.asarray(volumes, dtype=np.float64),
        **counters,
    }


def analyze_distribution(n_vertices, n_samples, seed=42, series_terms=96, n_jobs=None):
    """
    Analyze volume distribution for n_vertices polyhedra.

    Args:
        n_vertices: Number of vertices (must be >= 4)
        n_samples: Number of random configurations to sample
        seed: Random seed
        series_terms: Deprecated compatibility argument; the vectorized
            Bloch--Wigner kernel does not use a truncated series.
        n_jobs: Number of parallel jobs. The automatic default uses at most
            16 processes and never more processes than logical batches.

    Returns:
        dict with volumes and statistics
    """
    if not isinstance(n_vertices, Integral) or isinstance(n_vertices, bool):
        raise TypeError("n_vertices must be an integer")
    if n_vertices < 4:
        raise ValueError("Need at least 4 vertices (including infinity)")
    if not isinstance(n_samples, Integral) or isinstance(n_samples, bool):
        raise TypeError("n_samples must be an integer")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    n_vertices = int(n_vertices)
    n_samples = int(n_samples)
    seed = int(seed)
    n_logical_batches = (n_samples + LOGICAL_BATCH_SIZE - 1) // LOGICAL_BATCH_SIZE

    # Determine number of parallel jobs
    if n_jobs is None:
        effective_workers = min(
            MAX_AUTOMATIC_WORKERS,
            cpu_count() or 1,
            n_logical_batches,
        )
    else:
        if not isinstance(n_jobs, Integral) or isinstance(n_jobs, bool):
            raise TypeError("n_jobs must be an integer or None")
        if n_jobs < 1:
            raise ValueError("n_jobs must be positive")
        effective_workers = min(int(n_jobs), n_logical_batches)

    print(f"Sampling {n_samples} random {n_vertices}-vertex configurations...")
    print(
        f"Using {effective_workers} worker(s) across "
        f"{n_logical_batches} deterministic batch(es)"
    )

    work_batches = []
    for batch_index, start in enumerate(range(0, n_samples, LOGICAL_BATCH_SIZE)):
        target_samples = min(LOGICAL_BATCH_SIZE, n_samples - start)
        work_batches.append(
            (n_vertices, target_samples, seed, batch_index, series_terms)
        )

    if effective_workers == 1:
        batch_results = [_worker_sample_volumes(batch) for batch in work_batches]
    else:
        with Pool(processes=effective_workers) as pool:
            # Pool.map preserves logical-batch order.
            batch_results = pool.map(_worker_sample_volumes, work_batches)

    volumes = np.concatenate(
        [batch["volumes"] for batch in batch_results]
    ).astype(np.float64, copy=False)
    counter_names = (
        "n_vertex_draws",
        "n_rejected_vertex_draws",
        "n_configurations_attempted",
        "n_rejected_configurations",
        "n_vertex_generation_failures",
        "n_numerical_evaluation_failures",
        "n_invalid_volume_evaluations",
    )
    counters = {
        name: sum(int(batch[name]) for batch in batch_results)
        for name in counter_names
    }
    complete = len(volumes) == n_samples
    status = "complete" if complete else "attempt_limit_reached"

    print(f"\nSuccessfully analyzed {len(volumes)} valid configurations")
    print(
        "Rejected "
        f"{counters['n_rejected_vertex_draws']} vertex draw(s), "
        f"{counters['n_rejected_configurations']} configuration(s), and "
        f"encountered {counters['n_numerical_evaluation_failures']} known "
        "numerical evaluation failure(s)."
    )
    if not complete:
        print(
            "WARNING: finite retry limits were reached; "
            f"{n_samples - len(volumes)} requested sample(s) are missing."
        )

    result = {
        'volumes': volumes,
        'n_samples_requested': n_samples,
        'n_valid': len(volumes),
        'n_missing': n_samples - len(volumes),
        'complete': complete,
        'status': status,
        'effective_workers': effective_workers,
        'logical_batch_size': LOGICAL_BATCH_SIZE,
        'n_logical_batches': n_logical_batches,
        'max_vertex_draw_attempts': MAX_VERTEX_DRAW_ATTEMPTS,
        'max_configuration_attempts_per_sample': (
            MAX_CONFIGURATION_ATTEMPTS_PER_SAMPLE
        ),
        **counters,
    }
    if len(volumes):
        result.update({
            'mean': np.mean(volumes),
            'median': np.median(volumes),
            'std': np.std(volumes),
            'min': np.min(volumes),
            'max': np.max(volumes),
            'q25': np.percentile(volumes, 25),
            'q75': np.percentile(volumes, 75),
        })
    else:
        result.update({name: float("nan") for name in (
            "mean", "median", "std", "min", "max", "q25", "q75"
        )})
    return result


def fit_distribution(
    volumes,
    dist_name='beta',
    n_bootstrap=1000,
    confidence_level=0.95,
    seed=42,
):
    """
    Fit a distribution to the volume data with confidence intervals.

    Args:
        volumes: Array of volume values
        dist_name: Name of distribution to fit ('beta', 'gamma', 'lognorm', etc.)
        n_bootstrap: Number of bootstrap samples for confidence intervals
        confidence_level: Confidence level for intervals (default: 0.95)
        seed: Non-negative seed for the function-local bootstrap generator

    Returns:
        dict with fitted parameters, confidence intervals, and goodness-of-fit statistics
    """
    if not isinstance(n_bootstrap, Integral) or isinstance(n_bootstrap, bool):
        raise TypeError("n_bootstrap must be an integer")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    volumes = np.asarray(volumes, dtype=np.float64)
    if volumes.ndim != 1 or volumes.size == 0:
        raise ValueError("volumes must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(volumes)):
        raise ValueError("volumes must contain only finite values")

    # Normalize data to [0,1] for beta distribution
    if dist_name == 'beta':
        # Beta requires data in (0,1), so normalize
        data_min = np.min(volumes)
        data_max = np.max(volumes)
        data_range = data_max - data_min
        if data_range == 0.0:
            raise ValueError("beta fitting requires at least two distinct volumes")
        normalized_data = (volumes - data_min) / data_range
        # Shift slightly away from 0 and 1 to avoid numerical issues
        epsilon = 1e-10
        normalized_data = np.clip(normalized_data, epsilon, 1 - epsilon)
        fit_data = normalized_data
    else:
        fit_data = volumes

    # Get the distribution object
    dist = getattr(stats, dist_name)

    # Fit the distribution
    print(f"\nFitting {dist_name} distribution...")
    params = dist.fit(fit_data)

    # Kolmogorov-Smirnov goodness-of-fit test
    # Passing the distribution name triggers a SciPy 1.18 fast path that can
    # forward ``loc`` and ``scale`` directly to a one-argument special
    # function (for example ``ndtr`` for the normal distribution).  Bind the
    # fitted parameters through the public distribution object instead; this
    # is stable across the supported SciPy/Python combinations.
    def fitted_cdf(values):
        return dist.cdf(values, *params)

    ks_statistic, ks_pvalue = stats.kstest(fit_data, fitted_cdf)

    # SciPy's Anderson implementation supports only a fixed set of families;
    # use its normal diagnostic for the other CLI-supported fits.
    ad_distribution = dist_name if dist_name in ['norm', 'expon'] else 'norm'
    anderson_kwargs = {}
    if "method" in inspect.signature(stats.anderson).parameters:
        # SciPy >= 1.17 requires an explicit p-value method.  We retain only
        # the statistic, so interpolation preserves the inexpensive legacy
        # diagnostic and avoids the pending SciPy 1.19 API transition.
        anderson_kwargs["method"] = "interpolate"
    ad_statistic = stats.anderson(
        fit_data,
        dist=ad_distribution,
        **anderson_kwargs,
    ).statistic

    # Bootstrap for confidence intervals
    print(f"Computing confidence intervals via bootstrap ({n_bootstrap} samples)...")
    bootstrap_params = []
    rng = np.random.default_rng(np.random.SeedSequence(int(seed)))
    progress_interval = max(1, n_bootstrap // 10)
    n_bootstrap_failures = 0

    for i in range(n_bootstrap):
        if (i + 1) % progress_interval == 0 or i + 1 == n_bootstrap:
            print(f"  Bootstrap progress: {i + 1}/{n_bootstrap} ({100*(i+1)/n_bootstrap:.0f}%)")

        # Resample with replacement
        resampled = rng.choice(fit_data, size=len(fit_data), replace=True)
        try:
            bootstrap_params.append(dist.fit(resampled))
        except (stats.FitError, FloatingPointError):
            # These are known numerical fit failures. Unexpected exceptions
            # propagate so programming errors are not mistaken for bad draws.
            n_bootstrap_failures += 1
            continue

    if not bootstrap_params:
        raise RuntimeError("all bootstrap distribution fits failed")
    bootstrap_params = np.asarray(bootstrap_params, dtype=np.float64)

    # Calculate confidence intervals for each parameter
    alpha = 1 - confidence_level
    param_names = []
    if dist_name == 'beta':
        param_names = ['a', 'b', 'loc', 'scale']
    elif dist_name == 'gamma':
        param_names = ['a', 'loc', 'scale']
    elif dist_name == 'lognorm':
        param_names = ['s', 'loc', 'scale']
    else:
        param_names = [f'param_{i}' for i in range(len(params))]

    confidence_intervals = {}
    for i, (param_name, param_value) in enumerate(zip(param_names, params)):
        lower = np.percentile(bootstrap_params[:, i], 100 * alpha / 2)
        upper = np.percentile(bootstrap_params[:, i], 100 * (1 - alpha / 2))
        confidence_intervals[param_name] = {
            'estimate': param_value,
            'lower': lower,
            'upper': upper,
            'ci_level': confidence_level
        }

    # For beta distribution, also report parameters in original scale
    if dist_name == 'beta':
        # params = (a, b, loc, scale) where loc and scale are from normalization
        # We need to transform back to original scale
        result = {
            'distribution': dist_name,
            'params': params,
            'param_names': param_names,
            'confidence_intervals': confidence_intervals,
            'normalization': {
                'data_min': data_min,
                'data_max': data_max,
                'data_range': data_range
            },
            'goodness_of_fit': {
                'ks_statistic': ks_statistic,
                'ks_pvalue': ks_pvalue,
                'ad_statistic': ad_statistic
            },
            'bootstrap': {
                'seed': int(seed),
                'n_requested': int(n_bootstrap),
                'n_successful': len(bootstrap_params),
                'n_failures': n_bootstrap_failures,
            },
        }
    else:
        result = {
            'distribution': dist_name,
            'params': params,
            'param_names': param_names,
            'confidence_intervals': confidence_intervals,
            'goodness_of_fit': {
                'ks_statistic': ks_statistic,
                'ks_pvalue': ks_pvalue,
                'ad_statistic': ad_statistic
            },
            'bootstrap': {
                'seed': int(seed),
                'n_requested': int(n_bootstrap),
                'n_successful': len(bootstrap_params),
                'n_failures': n_bootstrap_failures,
            },
        }

    return result


def plot_distribution(volumes, volume_stats, n_vertices, output_file, reference_volume=None, fit_result=None):
    """Create histogram plot of volume distribution with optional fitted distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(volumes, bins=50, density=True, alpha=0.7,
             color='steelblue', edgecolor='black', linewidth=0.5)
    ax1.axvline(volume_stats['mean'], color='red', linestyle='--', linewidth=2,
                label=f"Mean: {volume_stats['mean']:.4f}")
    ax1.axvline(volume_stats['median'], color='green', linestyle='--', linewidth=2,
                label=f"Median: {volume_stats['median']:.4f}")

    if reference_volume is not None:
        ax1.axvline(reference_volume, color='orange', linestyle='--', linewidth=2,
                    label=f"Reference: {reference_volume:.4f}")

    # Overlay fitted distribution if available
    if fit_result is not None:
        x = np.linspace(volumes.min(), volumes.max(), 500)
        dist_name = fit_result['distribution']
        params = fit_result['params']
        dist = getattr(stats, dist_name)

        if dist_name == 'beta':
            # Transform x to normalized space, get pdf, then transform back
            norm = fit_result['normalization']
            x_normalized = (x - norm['data_min']) / norm['data_range']
            # Beta params: a, b, loc, scale (loc and scale are for the [0,1] domain)
            y = dist.pdf(x_normalized, *params) / norm['data_range']
        else:
            y = dist.pdf(x, *params)

        ax1.plot(x, y, 'r-', linewidth=2.5, alpha=0.8,
                 label=f"Fitted {dist_name.capitalize()}")

    ax1.set_xlabel('Volume', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title(f'{n_vertices}-Vertex Ideal Polyhedra Volume Distribution', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot([volumes], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5))
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_title('Volume Distribution (Box Plot)', fontsize=14)
    ax2.set_xticklabels([f'{n_vertices} vertices'])
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze volume distributions of ideal polyhedra',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --vertices 4 --samples 10000
  %(prog)s --vertices 6 --samples 50000 --output my_analysis.png
  %(prog)s --vertices 5 --samples 20000 --reference 3.66
        """
    )

    parser.add_argument('--vertices', '-v', type=int, required=True,
                        help='Number of vertices (must be >= 4)')
    parser.add_argument('--samples', '-n', type=int, default=10000,
                        help='Number of random samples (default: 10000)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output plot file (default: results/plots/{n}vertex_distribution_TIMESTAMP.png)')
    parser.add_argument('--data', '-d', type=str, default=None,
                        help='Output data JSON file (optional)')
    parser.add_argument('--reference', '-r', type=float, default=None,
                        help='Reference volume to mark on plot (optional)')
    parser.add_argument('--series-terms', type=int, default=96,
                        help='Deprecated compatibility option (ignored by the Bloch-Wigner kernel)')
    parser.add_argument('--fit', '-f', type=str, default=None,
                        choices=['beta', 'gamma', 'lognorm', 'norm'],
                        help='Fit a distribution and compute confidence intervals (default: None)')
    parser.add_argument('--bootstrap', '-b', type=int, default=1000,
                        help='Number of bootstrap samples for CI estimation (default: 1000)')
    parser.add_argument('--confidence', '-c', type=float, default=0.95,
                        help='Confidence level for intervals (default: 0.95)')
    parser.add_argument('--jobs', '-j', type=int, default=None,
                        help='Number of parallel jobs (automatic default: up to 16)')

    args = parser.parse_args()

    if args.vertices < 4:
        print("Error: Number of vertices must be at least 4")
        sys.exit(1)
    if args.samples < 1:
        print("Error: Number of samples must be positive")
        sys.exit(1)
    if args.bootstrap < 1:
        print("Error: Number of bootstrap samples must be positive")
        sys.exit(1)

    # Setup output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output is None:
        plot_file = f"results/plots/{args.vertices}vertex_distribution_{timestamp}.png"
    else:
        plot_file = args.output

    if args.data is not None:
        data_file = args.data
    else:
        data_file = f"results/data/{args.vertices}vertex_distribution_{timestamp}.json"

    # Ensure output directories exist
    Path(plot_file).parent.mkdir(parents=True, exist_ok=True)
    if args.data is not None:
        Path(data_file).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Ideal Polyhedron Volume Distribution Analysis")
    print("=" * 70)
    print(f"Vertices:      {args.vertices}")
    print(f"Samples:       {args.samples}")
    print(f"Random seed:   {args.seed}")
    print(f"Plot output:   {plot_file}")
    if args.data:
        print(f"Data output:   {data_file}")
    print("=" * 70)
    print()

    # Run analysis
    results = analyze_distribution(
        args.vertices,
        args.samples,
        seed=args.seed,
        series_terms=args.series_terms,
        n_jobs=args.jobs
    )

    # Print statistics
    print("\n" + "=" * 70)
    print("STATISTICS:")
    print("=" * 70)
    print(f"Valid configs:  {results['n_valid']:,} / {results['n_samples_requested']:,}")
    print(f"Sampling status: {results['status']}")
    print(f"Workers used:    {results['effective_workers']}")
    print(f"Rejected draws:  {results['n_rejected_vertex_draws']:,}")
    print(f"Rejected configs:{results['n_rejected_configurations']:>10,}")
    print(
        "Numerical fails: "
        f"{results['n_numerical_evaluation_failures']:,}"
    )
    print(f"Mean volume:    {results['mean']:.8f}")
    print(f"Median volume:  {results['median']:.8f}")
    print(f"Std deviation:  {results['std']:.8f}")
    print(f"Min volume:     {results['min']:.8f}")
    print(f"Max volume:     {results['max']:.8f}")
    print(f"25th percentile: {results['q25']:.8f}")
    print(f"75th percentile: {results['q75']:.8f}")

    if args.reference is not None:
        print(f"\nReference volume: {args.reference:.8f}")
        print(f"Mean/Reference:   {results['mean']/args.reference:.4f}")
        print(f"Max/Reference:    {results['max']/args.reference:.4f}")

    # Fit distribution if requested
    fit_result = None
    if args.fit is not None and results['n_valid']:
        print("\n" + "=" * 70)
        print("DISTRIBUTION FITTING:")
        print("=" * 70)
        fit_result = fit_distribution(
            results['volumes'],
            dist_name=args.fit,
            n_bootstrap=args.bootstrap,
            confidence_level=args.confidence,
            seed=args.seed,
        )

        # Print fitted parameters with confidence intervals
        print(f"\nFitted {args.fit.upper()} distribution parameters:")
        print("-" * 70)
        for param_name, ci_info in fit_result['confidence_intervals'].items():
            print(f"{param_name:>10}: {ci_info['estimate']:>12.6f}  "
                  f"[{ci_info['lower']:>10.6f}, {ci_info['upper']:>10.6f}]  "
                  f"({100*ci_info['ci_level']:.0f}% CI)")

        # Print goodness-of-fit statistics
        print("\nGoodness of fit:")
        print("-" * 70)
        gof = fit_result['goodness_of_fit']
        print(f"Kolmogorov-Smirnov statistic: {gof['ks_statistic']:.6f}")
        print(f"Kolmogorov-Smirnov p-value:   {gof['ks_pvalue']:.6f}")
        if gof['ks_pvalue'] > 0.05:
            print("  → Cannot reject the hypothesis that data follows this distribution (p > 0.05)")
        else:
            print("  → Data may not follow this distribution well (p ≤ 0.05)")
        if gof['ad_statistic'] is not None:
            print(f"Anderson-Darling statistic:   {gof['ad_statistic']:.6f}")

    # Histograms and fitted distributions are undefined for an empty result.
    # Preserve the diagnostics/data output instead of failing after a bounded
    # sampler has correctly reported complete underfill.
    if results['n_valid']:
        plot_distribution(
            results['volumes'],
            results,
            args.vertices,
            plot_file,
            reference_volume=args.reference,
            fit_result=fit_result
        )
    else:
        print("No valid configurations were produced; plot and fit skipped.")

    # Save data if requested
    if args.data is not None:
        output_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_vertices': args.vertices,
                'n_samples_requested': args.samples,
                'n_valid': results['n_valid'],
                'n_missing': results['n_missing'],
                'complete': results['complete'],
                'status': results['status'],
                'seed': args.seed,
                'effective_workers': results['effective_workers'],
                'logical_batch_size': results['logical_batch_size'],
                'n_logical_batches': results['n_logical_batches'],
                'max_vertex_draw_attempts': results['max_vertex_draw_attempts'],
                'max_configuration_attempts_per_sample': results[
                    'max_configuration_attempts_per_sample'
                ],
                'n_vertex_draws': results['n_vertex_draws'],
                'n_rejected_vertex_draws': results['n_rejected_vertex_draws'],
                'n_configurations_attempted': results['n_configurations_attempted'],
                'n_rejected_configurations': results['n_rejected_configurations'],
                'n_vertex_generation_failures': results['n_vertex_generation_failures'],
                'n_numerical_evaluation_failures': results['n_numerical_evaluation_failures'],
                'n_invalid_volume_evaluations': results['n_invalid_volume_evaluations'],
                'series_terms': args.series_terms,
                'series_terms_ignored': True,
                'volume_mode': 'bloch_wigner',
            },
            'statistics': {
                name: (
                    float(results[name])
                    if np.isfinite(results[name])
                    else None
                )
                for name in ('mean', 'median', 'std', 'min', 'max', 'q25', 'q75')
            },
            'volumes': results['volumes'].tolist(),
        }

        # Add distribution fitting results if available
        if fit_result is not None:
            # Convert numpy types to Python types for JSON serialization
            fit_data = {
                'distribution': fit_result['distribution'],
                'params': [float(p) for p in fit_result['params']],
                'param_names': fit_result['param_names'],
                'confidence_intervals': {
                    name: {
                        'estimate': float(ci['estimate']),
                        'lower': float(ci['lower']),
                        'upper': float(ci['upper']),
                        'ci_level': float(ci['ci_level'])
                    }
                    for name, ci in fit_result['confidence_intervals'].items()
                },
                'goodness_of_fit': {
                    'ks_statistic': float(fit_result['goodness_of_fit']['ks_statistic']),
                    'ks_pvalue': float(fit_result['goodness_of_fit']['ks_pvalue']),
                    'ad_statistic': float(fit_result['goodness_of_fit']['ad_statistic'])
                                   if fit_result['goodness_of_fit']['ad_statistic'] is not None else None
                },
                'bootstrap': dict(fit_result['bootstrap']),
            }
            if 'normalization' in fit_result:
                fit_data['normalization'] = {
                    'data_min': float(fit_result['normalization']['data_min']),
                    'data_max': float(fit_result['normalization']['data_max']),
                    'data_range': float(fit_result['normalization']['data_range'])
                }
            output_data['distribution_fit'] = fit_data

        with open(data_file, 'w') as f:
            # Reject accidental NaN/Infinity so saved data remains strict JSON.
            json.dump(output_data, f, indent=2, allow_nan=False)

        print(f"\nData saved to: {data_file}")

    print("=" * 70)


if __name__ == '__main__':
    main()
