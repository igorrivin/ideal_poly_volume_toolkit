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
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import sys
from scipy import stats
from multiprocessing import Pool, cpu_count

from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    ideal_poly_volume_via_delaunay,
)


def sample_random_vertex():
    """
    Sample a uniform random point on the unit sphere and project to complex plane.
    Uses stereographic projection from north pole.
    """
    # Sample uniform point on sphere using Gaussian method
    vec = np.random.randn(3)
    vec = vec / np.linalg.norm(vec)
    x, y, z = vec

    # Skip near north pole (maps to infinity)
    if z > 0.999:
        return None

    # Stereographic projection
    w = complex(x/(1-z), y/(1-z))
    return w


def _worker_sample_volumes(args):
    """Worker function for parallel volume sampling."""
    n_vertices, n_samples_chunk, seed_offset, series_terms = args

    np.random.seed(seed_offset)

    fixed_vertices = [complex(0, 0), complex(1, 0)]
    n_random = n_vertices - 3
    volumes = []

    for i in range(n_samples_chunk):
        vertices = fixed_vertices.copy()

        # Add random vertices
        for _ in range(n_random):
            v = sample_random_vertex()
            if v is None:
                continue

            too_close = False
            for existing in vertices:
                if abs(v - existing) < 0.01:
                    too_close = True
                    break
            if too_close:
                continue

            vertices.append(v)

        if len(vertices) != n_vertices - 1:
            continue

        try:
            vertices_np = np.array(vertices, dtype=np.complex128)
            vol = ideal_poly_volume_via_delaunay(
                vertices_np, mode='fast', series_terms=series_terms
            )

            if vol > 0 and vol < 1000:
                volumes.append(vol)
        except:
            pass

    return volumes


def analyze_distribution(n_vertices, n_samples, seed=42, series_terms=96, n_jobs=None):
    """
    Analyze volume distribution for n_vertices polyhedra.

    Args:
        n_vertices: Number of vertices (must be >= 3)
        n_samples: Number of random configurations to sample
        seed: Random seed
        series_terms: Number of terms for Lobachevsky function approximation
        n_jobs: Number of parallel jobs (default: use all CPUs)

    Returns:
        dict with volumes and statistics
    """
    if n_vertices - 3 < 0:
        raise ValueError("Need at least 3 vertices (including infinity)")

    # Determine number of parallel jobs
    if n_jobs is None:
        n_jobs = cpu_count()
    elif n_jobs <= 0:
        n_jobs = 1  # Serial execution

    print(f"Sampling {n_samples} random {n_vertices}-vertex configurations...")
    print(f"Using {n_jobs} parallel workers")

    if n_jobs == 1:
        # Serial execution (original code path)
        np.random.seed(seed)
        volumes = _worker_sample_volumes((n_vertices, n_samples, seed, series_terms))
    else:
        # Parallel execution
        # Split work across workers
        samples_per_worker = n_samples // n_jobs
        remainder = n_samples % n_jobs

        # Create work chunks with different seeds
        work_chunks = []
        for i in range(n_jobs):
            chunk_size = samples_per_worker + (1 if i < remainder else 0)
            seed_offset = seed + i * 1000  # Different seed for each worker
            work_chunks.append((n_vertices, chunk_size, seed_offset, series_terms))

        # Run in parallel
        with Pool(processes=n_jobs) as pool:
            results = pool.map(_worker_sample_volumes, work_chunks)

        # Combine results from all workers
        volumes = []
        for worker_volumes in results:
            volumes.extend(worker_volumes)
        volumes = np.array(volumes)

    if len(volumes) == 0:
        raise ValueError("No valid configurations found!")

    print(f"\nSuccessfully analyzed {len(volumes)} valid configurations")

    return {
        'volumes': volumes,
        'n_samples_requested': n_samples,
        'n_valid': len(volumes),
        'mean': np.mean(volumes),
        'median': np.median(volumes),
        'std': np.std(volumes),
        'min': np.min(volumes),
        'max': np.max(volumes),
        'q25': np.percentile(volumes, 25),
        'q75': np.percentile(volumes, 75),
    }


def fit_distribution(volumes, dist_name='beta', n_bootstrap=1000, confidence_level=0.95):
    """
    Fit a distribution to the volume data with confidence intervals.

    Args:
        volumes: Array of volume values
        dist_name: Name of distribution to fit ('beta', 'gamma', 'lognorm', etc.)
        n_bootstrap: Number of bootstrap samples for confidence intervals
        confidence_level: Confidence level for intervals (default: 0.95)

    Returns:
        dict with fitted parameters, confidence intervals, and goodness-of-fit statistics
    """
    # Normalize data to [0,1] for beta distribution
    if dist_name == 'beta':
        # Beta requires data in (0,1), so normalize
        data_min = np.min(volumes)
        data_max = np.max(volumes)
        data_range = data_max - data_min
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
    ks_statistic, ks_pvalue = stats.kstest(fit_data, dist_name, args=params)

    # Anderson-Darling test (if available for this distribution)
    try:
        ad_result = stats.anderson(fit_data, dist=dist_name if dist_name in ['norm', 'expon'] else 'norm')
        ad_statistic = ad_result.statistic
    except:
        ad_statistic = None

    # Bootstrap for confidence intervals
    print(f"Computing confidence intervals via bootstrap ({n_bootstrap} samples)...")
    bootstrap_params = []

    for i in range(n_bootstrap):
        if (i + 1) % (n_bootstrap // 10) == 0:
            print(f"  Bootstrap progress: {i + 1}/{n_bootstrap} ({100*(i+1)/n_bootstrap:.0f}%)")

        # Resample with replacement
        resampled = np.random.choice(fit_data, size=len(fit_data), replace=True)
        try:
            bootstrap_params.append(dist.fit(resampled))
        except:
            # If fit fails, skip this bootstrap sample
            continue

    bootstrap_params = np.array(bootstrap_params)

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
            }
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
            }
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
                        help='Number of vertices (must be >= 3)')
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
                        help='Number of series terms for Lobachevsky function (default: 96)')
    parser.add_argument('--fit', '-f', type=str, default=None,
                        choices=['beta', 'gamma', 'lognorm', 'norm'],
                        help='Fit a distribution and compute confidence intervals (default: None)')
    parser.add_argument('--bootstrap', '-b', type=int, default=1000,
                        help='Number of bootstrap samples for CI estimation (default: 1000)')
    parser.add_argument('--confidence', '-c', type=float, default=0.95,
                        help='Confidence level for intervals (default: 0.95)')
    parser.add_argument('--jobs', '-j', type=int, default=None,
                        help='Number of parallel jobs (default: use all CPUs)')

    args = parser.parse_args()

    if args.vertices < 3:
        print("Error: Number of vertices must be at least 3")
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
    if args.fit is not None:
        print("\n" + "=" * 70)
        print("DISTRIBUTION FITTING:")
        print("=" * 70)
        fit_result = fit_distribution(
            results['volumes'],
            dist_name=args.fit,
            n_bootstrap=args.bootstrap,
            confidence_level=args.confidence
        )

        # Print fitted parameters with confidence intervals
        print(f"\nFitted {args.fit.upper()} distribution parameters:")
        print("-" * 70)
        for param_name, ci_info in fit_result['confidence_intervals'].items():
            print(f"{param_name:>10}: {ci_info['estimate']:>12.6f}  "
                  f"[{ci_info['lower']:>10.6f}, {ci_info['upper']:>10.6f}]  "
                  f"({100*ci_info['ci_level']:.0f}% CI)")

        # Print goodness-of-fit statistics
        print(f"\nGoodness of fit:")
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

    # Create plot
    plot_distribution(
        results['volumes'],
        results,
        args.vertices,
        plot_file,
        reference_volume=args.reference,
        fit_result=fit_result
    )

    # Save data if requested
    if args.data is not None:
        output_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_vertices': args.vertices,
                'n_samples_requested': args.samples,
                'n_valid': results['n_valid'],
                'seed': args.seed,
                'series_terms': args.series_terms,
            },
            'statistics': {
                'mean': float(results['mean']),
                'median': float(results['median']),
                'std': float(results['std']),
                'min': float(results['min']),
                'max': float(results['max']),
                'q25': float(results['q25']),
                'q75': float(results['q75']),
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
                }
            }
            if 'normalization' in fit_result:
                fit_data['normalization'] = {
                    'data_min': float(fit_result['normalization']['data_min']),
                    'data_max': float(fit_result['normalization']['data_max']),
                    'data_range': float(fit_result['normalization']['data_range'])
                }
            output_data['distribution_fit'] = fit_data

        with open(data_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nData saved to: {data_file}")

    print("=" * 70)


if __name__ == '__main__':
    main()
