#!/usr/bin/env python3
"""
Generate paper-quality figures for maximal volume ideal polyhedra.

Creates:
1. Planar Delaunay triangulation drawings
2. Poincaré ball model visualizations
3. Combined figure panels
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import LineCollection
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/igor/devel/ideal_poly_volume_toolkit')

from ideal_poly_volume_toolkit.visualization import plot_polyhedron_poincare


def load_config(filepath):
    """Load a comprehensive optimization result."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    n = data['statistics']['n_vertices']
    real = np.array(data['best']['vertices_real'])
    imag = np.array(data['best']['vertices_imag'])
    triangulation = data['best']['triangulation']
    volume = data['statistics']['best_volume']

    return {
        'n': n,
        'vertices': real + 1j * imag,
        'vertices_2d': np.column_stack([real, imag]),
        'triangulation': triangulation,
        'volume': volume,
    }


def draw_planar_triangulation(ax, config, title=None):
    """
    Draw planar Delaunay triangulation with nice styling.
    """
    vertices = config['vertices_2d']
    triangulation = config['triangulation']
    n = config['n']
    volume = config['volume']

    # Center and scale
    center = vertices.mean(axis=0)
    vertices_centered = vertices - center
    scale = np.max(np.abs(vertices_centered)) * 1.2
    if scale < 0.1:
        scale = 1.0

    # Draw unit circle (boundary at infinity)
    circle = Circle((0, 0), 1.0, fill=False, color='gray',
                    linestyle='--', linewidth=0.5, alpha=0.5)
    ax.add_patch(circle)

    # Normalize vertices to fit in unit disk-ish region
    if scale > 0:
        v_norm = vertices_centered / scale * 0.8
    else:
        v_norm = vertices_centered

    # Draw triangles
    for tri in triangulation:
        triangle = Polygon(v_norm[list(tri)], fill=True,
                          facecolor='lightblue', edgecolor='navy',
                          linewidth=1.5, alpha=0.6)
        ax.add_patch(triangle)

    # Draw vertices
    ax.scatter(v_norm[:, 0], v_norm[:, 1], s=80, c='darkblue',
               zorder=5, edgecolors='white', linewidths=1.5)

    # Label vertices
    for i, (x, y) in enumerate(v_norm):
        ax.annotate(str(i), (x, y), xytext=(5, 5),
                   textcoords='offset points', fontsize=8,
                   fontweight='bold')

    # Add infinity vertex indicator
    ax.annotate('∞', (0.85, 0.85), fontsize=12, fontweight='bold',
               color='gray', ha='center')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')

    if title is None:
        title = f'$n={n}$, $V={volume:.4f}$'
    ax.set_title(title, fontsize=11, fontweight='bold')


def generate_poincare_figure(config, output_path):
    """Generate Poincaré ball visualization."""
    try:
        fig = plot_polyhedron_poincare(
            config['vertices'],
            title=f"n={config['n']}, V={config['volume']:.4f}"
        )
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        return True
    except Exception as e:
        print(f"  Warning: Could not generate Poincaré figure: {e}")
        return False


def generate_panel_figure(configs, output_path, ncols=3):
    """Generate a panel figure with multiple triangulations."""
    n_configs = len(configs)
    nrows = (n_configs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = np.atleast_2d(axes)

    for idx, config in enumerate(configs):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        draw_planar_triangulation(ax, config)

    # Hide unused axes
    for idx in range(n_configs, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].axis('off')

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close(fig)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate paper-quality figures for maximal volume polyhedra"
    )
    parser.add_argument(
        "--data-dir", default="results/data/comprehensive",
        help="Directory with comprehensive results"
    )
    parser.add_argument(
        "--output-dir", default="paper/figures",
        help="Output directory for figures"
    )
    parser.add_argument(
        "--format", default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output format"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATING PAPER FIGURES")
    print("=" * 60)

    # Load all configs
    data_dir = Path(args.data_dir)
    configs = []

    for filepath in sorted(data_dir.glob('*vertex_comprehensive.json')):
        config = load_config(filepath)
        configs.append(config)
        print(f"Loaded n={config['n']}: V={config['volume']:.6f}")

    if not configs:
        print("No configurations found!")
        return

    # Generate individual figures
    print("\nGenerating individual triangulation figures...")
    for config in configs:
        n = config['n']
        output_path = output_dir / f"triangulation_n{n}.{args.format}"

        fig, ax = plt.subplots(figsize=(5, 5))
        draw_planar_triangulation(ax, config)
        fig.savefig(output_path, dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  Saved: {output_path}")

    # Generate Poincaré ball figures
    print("\nGenerating Poincaré ball figures...")
    for config in configs:
        n = config['n']
        output_path = output_dir / f"poincare_n{n}.{args.format}"
        success = generate_poincare_figure(config, output_path)
        if success:
            print(f"  Saved: {output_path}")

    # Generate panel figure
    print("\nGenerating panel figure...")
    panel_path = output_dir / f"triangulations_panel.{args.format}"
    generate_panel_figure(configs, panel_path)
    print(f"  Saved: {panel_path}")

    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
