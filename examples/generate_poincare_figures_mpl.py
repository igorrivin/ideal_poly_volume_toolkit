#!/usr/bin/env python3
"""
Generate Poincaré ball visualizations using matplotlib (no display required).

Uses the Klein model representation transformed to Poincaré ball.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json
from pathlib import Path


def complex_to_klein_3d(z):
    """
    Convert complex coordinate (stereographic projection) to Klein model.

    The complex plane is the boundary of H³ at infinity.
    We use stereographic projection: z -> point on unit sphere S².
    """
    if np.isinf(z) or np.abs(z) > 1e10:
        return np.array([0, 0, 1])  # North pole

    x, y = z.real, z.imag
    r2 = x*x + y*y

    # Inverse stereographic projection from plane to sphere
    denom = 1 + r2
    return np.array([2*x/denom, 2*y/denom, (r2-1)/denom])


def klein_to_poincare(vec):
    """Transform from Klein model to Poincaré ball model."""
    n2 = np.linalg.norm(vec)**2
    if n2 >= 1:
        # On boundary - scale slightly inside
        return vec * 0.999 / np.sqrt(n2) / (1 + np.sqrt(1 - 0.999**2))
    return vec / (1 + np.sqrt(1 - n2))


def subdivide_triangle(v0, v1, v2, depth=3):
    """
    Recursively subdivide a triangle for smooth geodesics.
    Returns list of small triangles (each is a list of 3 vertices).
    """
    if depth == 0:
        return [[v0, v1, v2]]

    # Midpoints
    m01 = (v0 + v1) / 2
    m12 = (v1 + v2) / 2
    m02 = (v0 + v2) / 2

    # Recurse on 4 sub-triangles
    tris = []
    tris.extend(subdivide_triangle(v0, m01, m02, depth-1))
    tris.extend(subdivide_triangle(m01, v1, m12, depth-1))
    tris.extend(subdivide_triangle(m02, m12, v2, depth-1))
    tris.extend(subdivide_triangle(m01, m12, m02, depth-1))
    return tris


def load_config(filepath):
    """Load a comprehensive optimization result."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    n = data['statistics']['n_vertices']
    real = np.array(data['best']['vertices_real'])
    imag = np.array(data['best']['vertices_imag'])
    triangulation = data['best']['triangulation']
    volume = data['statistics']['best_volume']

    # Create complex array, add infinity as additional vertex
    vertices_complex = [complex(r, i) for r, i in zip(real, imag)]
    vertices_complex.append(complex('inf'))  # Add point at infinity

    return {
        'n': n,
        'vertices_complex': vertices_complex,
        'triangulation': triangulation,
        'volume': volume,
    }


def complete_triangulation_with_infinity(triangulation, n_finite):
    """
    Complete the triangulation by adding faces at infinity.
    """
    # Find boundary edges
    edge_count = {}
    for tri in triangulation:
        for i in range(3):
            e = tuple(sorted([tri[i], tri[(i+1)%3]]))
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary_edges = [e for e, c in edge_count.items() if c == 1]

    # Add faces connecting boundary edges to infinity
    inf_vertex = n_finite
    all_faces = list(triangulation)

    for e in boundary_edges:
        all_faces.append([e[0], e[1], inf_vertex])

    return all_faces


def render_poincare_ball(config, output_path, subdivisions=3, elev=25, azim=45):
    """
    Render a Poincaré ball visualization using matplotlib.
    """
    vertices = config['vertices_complex']
    triangulation = config['triangulation']
    n_finite = len(vertices) - 1

    # Complete triangulation with faces at infinity
    all_faces = complete_triangulation_with_infinity(triangulation, n_finite)

    # Get Klein model vertices (on unit sphere)
    klein_vertices = np.array([complex_to_klein_3d(z) for z in vertices])

    # Create figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Collect all Poincaré triangles with subdivision
    all_triangles = []
    for face in all_faces:
        v0 = klein_vertices[face[0]]
        v1 = klein_vertices[face[1]]
        v2 = klein_vertices[face[2]]

        # Subdivide in Klein model, then transform each small triangle
        sub_tris = subdivide_triangle(v0, v1, v2, depth=subdivisions)

        for tri in sub_tris:
            # Transform each vertex to Poincaré
            poincare_tri = [klein_to_poincare(v) for v in tri]
            all_triangles.append(poincare_tri)

    # Create Poly3DCollection
    poly = Poly3DCollection(all_triangles,
                            alpha=0.7,
                            facecolor='lightblue',
                            edgecolor='darkblue',
                            linewidth=0.1)
    ax.add_collection3d(poly)

    # Draw unit sphere wireframe (boundary)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.1, linewidth=0.3)

    # Set axis limits
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])

    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)

    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Title
    ax.set_title(f"Ideal Polyhedron (n={config['n']}, V={config['volume']:.4f})", fontsize=12)

    # Remove axis labels for cleaner look
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Poincaré ball visualizations using matplotlib"
    )
    parser.add_argument(
        "--data-dir", default="results/data/comprehensive",
        help="Directory with comprehensive results"
    )
    parser.add_argument(
        "--output-dir", default="paper/figures",
        help="Output directory"
    )
    parser.add_argument(
        "--subdivisions", type=int, default=3,
        help="Number of triangle subdivisions (default: 3)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATING POINCARÉ BALL VISUALIZATIONS (matplotlib)")
    print("=" * 60)

    data_dir = Path(args.data_dir)

    for filepath in sorted(data_dir.glob('*vertex_comprehensive.json')):
        config = load_config(filepath)
        n = config['n']

        print(f"\nProcessing n={n}...")

        output_path = output_dir / f"poincare_n{n}.png"
        try:
            render_poincare_ball(config, str(output_path),
                                 subdivisions=args.subdivisions)
            print(f"  Saved: {output_path}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Visualizations saved to: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
