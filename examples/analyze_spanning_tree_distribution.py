#!/usr/bin/env python3
"""
Analyze the distribution of spanning trees and identify what makes
some triangulations "vastly more forested" than others.
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


REPO_ROOT = Path(__file__).resolve().parent.parent
CORRECTED_DATA_PATH = REPO_ROOT / "results" / "spanning_trees_n10_CORRECTED.json"
LEGACY_DATA_PATH = REPO_ROOT / "results" / "spanning_trees_n10.json"


def resolve_spanning_tree_data_path(data_file: str | Path | None = None) -> Path:
    """Resolve the corrected dataset, warning before a historical fallback."""

    if data_file is not None:
        requested = Path(data_file)
        if not requested.is_file():
            raise FileNotFoundError(f"spanning-tree data file not found: {requested}")
        return requested

    if CORRECTED_DATA_PATH.is_file():
        return CORRECTED_DATA_PATH
    if LEGACY_DATA_PATH.is_file():
        warnings.warn(
            f"Corrected spanning-tree data is absent; using historical {LEGACY_DATA_PATH}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return LEGACY_DATA_PATH
    raise FileNotFoundError(
        "neither corrected nor historical spanning-tree data is available"
    )


def analyze_distribution(data_file: str | Path | None = None):
    """Analyze spanning tree distribution in detail."""

    resolved_data_file = resolve_spanning_tree_data_path(data_file)
    with resolved_data_file.open(encoding="utf-8") as f:
        data = json.load(f)

    tris = data['raw_data']['triangulations']

    print("="*70)
    print("SPANNING TREE DISTRIBUTION ANALYSIS")
    print("="*70)
    print(f"\nTotal triangulations: {len(tris)}")

    # Extract spanning tree counts
    spanning_trees = np.array([t['n_spanning_trees'] for t in tris])
    log_spanning_trees = np.log1p(spanning_trees)  # log(1+x) to handle zeros

    # Partition by realizability
    standard_real = [t for t in tris if t['standard_realizable']]
    standard_nonreal = [t for t in tris if not t['standard_realizable']]
    strict_real = [t for t in tris if t['strict_realizable']]
    strict_nonreal = [t for t in tris if not t['strict_realizable']]

    st_standard_real = np.array([t['n_spanning_trees'] for t in standard_real])
    st_standard_nonreal = np.array([t['n_spanning_trees'] for t in standard_nonreal])
    st_strict_real = np.array([t['n_spanning_trees'] for t in strict_real])
    st_strict_nonreal = np.array([t['n_spanning_trees'] for t in strict_nonreal])

    # Log-transformed statistics
    print("\n" + "="*70)
    print("LOG-TRANSFORMED STATISTICS: log(1 + spanning_trees)")
    print("="*70)

    print("\n--- STANDARD REALIZABILITY ---")
    print("Realizable:")
    print(f"  Log mean: {np.mean(np.log1p(st_standard_real)):.3f}")
    print(f"  Log median: {np.median(np.log1p(st_standard_real)):.3f}")
    print(f"  Log std: {np.std(np.log1p(st_standard_real)):.3f}")

    print("\nNon-realizable:")
    print(f"  Log mean: {np.mean(np.log1p(st_standard_nonreal)):.3f}")
    print(f"  Log median: {np.median(np.log1p(st_standard_nonreal)):.3f}")
    print(f"  Log std: {np.std(np.log1p(st_standard_nonreal)):.3f}")

    print("\n--- STRICT REALIZABILITY ---")
    print("Strict realizable:")
    print(f"  Log mean: {np.mean(np.log1p(st_strict_real)):.3f}")
    print(f"  Log median: {np.median(np.log1p(st_strict_real)):.3f}")
    print(f"  Log std: {np.std(np.log1p(st_strict_real)):.3f}")

    print("\nStrict non-realizable:")
    print(f"  Log mean: {np.mean(np.log1p(st_strict_nonreal)):.3f}")
    print(f"  Log median: {np.median(np.log1p(st_strict_nonreal)):.3f}")
    print(f"  Log std: {np.std(np.log1p(st_strict_nonreal)):.3f}")

    # Identify extreme cases
    print("\n" + "="*70)
    print("EXTREME CASES: Most Forested Triangulations")
    print("="*70)

    # Sort by spanning trees
    tris_sorted = sorted(tris, key=lambda t: t['n_spanning_trees'], reverse=True)

    print("\nTop 20 most forested triangulations:")
    print(f"{'Rank':<6} {'Index':<8} {'Spanning':<10} {'Vertices':<10} {'Edges':<8} {'Std Real':<10} {'Strict Real':<12}")
    print("-"*70)

    for rank, t in enumerate(tris_sorted[:20], 1):
        print(f"{rank:<6} {t['index']:<8} {t['n_spanning_trees']:<10} "
              f"{t['n_vertices']:<10} {t['n_edges']:<8} "
              f"{'Yes' if t['standard_realizable'] else 'No':<10} "
              f"{'Yes' if t['strict_realizable'] else 'No':<12}")

    # Analyze bottom cases
    print("\n" + "="*70)
    print("EXTREME CASES: Least Forested Triangulations")
    print("="*70)

    # Count zeros
    n_zero = sum(1 for t in tris if t['n_spanning_trees'] == 0)
    print(f"\nTriangulations with ZERO spanning trees: {n_zero} ({100*n_zero/len(tris):.2f}%)")

    if n_zero > 0:
        zero_tris = [t for t in tris if t['n_spanning_trees'] == 0]
        n_real = sum(1 for t in zero_tris if t['standard_realizable'])
        n_strict = sum(1 for t in zero_tris if t['strict_realizable'])
        print(f"  Standard realizable: {n_real} ({100*n_real/n_zero:.1f}%)")
        print(f"  Strict realizable: {n_strict} ({100*n_strict/n_zero:.1f}%)")

    print("\nBottom 20 least forested (non-zero):")
    nonzero_tris = [t for t in tris if t['n_spanning_trees'] > 0]
    tris_sorted_bottom = sorted(nonzero_tris, key=lambda t: t['n_spanning_trees'])

    print(f"{'Rank':<6} {'Index':<8} {'Spanning':<10} {'Vertices':<10} {'Edges':<8} {'Std Real':<10} {'Strict Real':<12}")
    print("-"*70)

    for rank, t in enumerate(tris_sorted_bottom[:20], 1):
        print(f"{rank:<6} {t['index']:<8} {t['n_spanning_trees']:<10} "
              f"{t['n_vertices']:<10} {t['n_edges']:<8} "
              f"{'Yes' if t['standard_realizable'] else 'No':<10} "
              f"{'Yes' if t['strict_realizable'] else 'No':<12}")

    # Analyze correlation with graph properties
    print("\n" + "="*70)
    print("CORRELATION WITH GRAPH PROPERTIES")
    print("="*70)

    vertices = np.array([t['n_vertices'] for t in tris])
    edges = np.array([t['n_edges'] for t in tris])

    # Compute correlations
    corr_vertices = np.corrcoef(spanning_trees, vertices)[0, 1]
    corr_edges = np.corrcoef(spanning_trees, edges)[0, 1]
    corr_log_vertices = np.corrcoef(log_spanning_trees, vertices)[0, 1]
    corr_log_edges = np.corrcoef(log_spanning_trees, edges)[0, 1]

    print("\nPearson correlation (raw):")
    print(f"  Spanning trees vs vertices: {corr_vertices:.4f}")
    print(f"  Spanning trees vs edges: {corr_edges:.4f}")

    print("\nPearson correlation (log-transformed):")
    print(f"  log(spanning trees) vs vertices: {corr_log_vertices:.4f}")
    print(f"  log(spanning trees) vs edges: {corr_log_edges:.4f}")

    # Percentile analysis
    print("\n" + "="*70)
    print("PERCENTILE ANALYSIS")
    print("="*70)

    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    values = np.percentile(spanning_trees, percentiles)

    print(f"\n{'Percentile':<12} {'Value':<12} {'log(1+value)':<15}")
    print("-"*40)
    for p, v in zip(percentiles, values):
        print(f"{p:<12} {int(v):<12} {np.log1p(v):<15.3f}")

    # Create visualizations
    create_visualizations(data, tris, spanning_trees, log_spanning_trees,
                         st_standard_real, st_standard_nonreal,
                         st_strict_real, st_strict_nonreal)


def create_visualizations(data, tris, spanning_trees, log_spanning_trees,
                          st_standard_real, st_standard_nonreal,
                          st_strict_real, st_strict_nonreal):
    """Create distribution plots."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Overall distribution (linear)
    ax = axes[0, 0]
    ax.hist(spanning_trees, bins=100, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of spanning trees')
    ax.set_ylabel('Frequency')
    ax.set_title('Overall Distribution (Linear Scale)')
    ax.axvline(np.mean(spanning_trees), color='red', linestyle='--',
               label=f'Mean: {np.mean(spanning_trees):.1f}')
    ax.axvline(np.median(spanning_trees), color='blue', linestyle='--',
               label=f'Median: {np.median(spanning_trees):.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Overall distribution (log)
    ax = axes[0, 1]
    ax.hist(log_spanning_trees, bins=100, alpha=0.7, edgecolor='black')
    ax.set_xlabel('log(1 + spanning trees)')
    ax.set_ylabel('Frequency')
    ax.set_title('Overall Distribution (Log-Transformed)')
    ax.axvline(np.mean(log_spanning_trees), color='red', linestyle='--',
               label=f'Mean: {np.mean(log_spanning_trees):.2f}')
    ax.axvline(np.median(log_spanning_trees), color='blue', linestyle='--',
               label=f'Median: {np.median(log_spanning_trees):.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Standard realizability comparison (log)
    ax = axes[0, 2]
    ax.hist(np.log1p(st_standard_real), bins=50, alpha=0.5,
            label=f'Realizable (n={len(st_standard_real)})', color='green', edgecolor='black')
    ax.hist(np.log1p(st_standard_nonreal), bins=50, alpha=0.5,
            label=f'Non-realizable (n={len(st_standard_nonreal)})', color='red', edgecolor='black')
    ax.set_xlabel('log(1 + spanning trees)')
    ax.set_ylabel('Frequency')
    ax.set_title('Standard Realizability (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Strict realizability comparison (log)
    ax = axes[1, 0]
    ax.hist(np.log1p(st_strict_real), bins=50, alpha=0.5,
            label=f'Strict (n={len(st_strict_real)})', color='blue', edgecolor='black')
    ax.hist(np.log1p(st_strict_nonreal), bins=50, alpha=0.5,
            label=f'Non-strict (n={len(st_strict_nonreal)})', color='orange', edgecolor='black')
    ax.set_xlabel('log(1 + spanning trees)')
    ax.set_ylabel('Frequency')
    ax.set_title('Strict Realizability (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Scatter: vertices vs log(spanning trees)
    ax = axes[1, 1]
    vertices = np.array([t['n_vertices'] for t in tris])
    ax.scatter(vertices, log_spanning_trees, alpha=0.3, s=10)
    ax.set_xlabel('Number of vertices')
    ax.set_ylabel('log(1 + spanning trees)')
    ax.set_title('Vertices vs Log(Spanning Trees)')
    ax.grid(True, alpha=0.3)

    # 6. Scatter: edges vs log(spanning trees)
    ax = axes[1, 2]
    edges = np.array([t['n_edges'] for t in tris])
    ax.scatter(edges, log_spanning_trees, alpha=0.3, s=10)
    ax.set_xlabel('Number of edges')
    ax.set_ylabel('log(1 + spanning trees)')
    ax.set_title('Edges vs Log(Spanning Trees)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    n = data['parameters']['n_vertices']
    output_path = f'results/plots/spanning_tree_analysis_n{n}.png'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*70}")
    print(f"Plots saved to: {output_path}")
    print(f"{'='*70}")

    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze spanning tree distribution')
    parser.add_argument('--data', type=str,
                       default=None,
                       help=(
                           'Path to spanning trees data JSON (default: corrected '
                           'artifact, with explicit historical fallback)'
                       ))

    args = parser.parse_args()

    analyze_distribution(args.data)
