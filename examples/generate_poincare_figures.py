#!/usr/bin/env python3
"""
Generate Poincaré ball visualizations of ideal polyhedra.

Uses the hypoviz approach: create mesh in Klein model,
subdivide for smooth geodesics, transform to Poincaré ball.
"""

import numpy as np
import pyvista as pv
import trimesh
import json
import sys
from pathlib import Path
from scipy.spatial import ConvexHull, Delaunay

# Add hypoviz to path
sys.path.insert(0, '/home/igor/devel/hypoviz')
from hyputils import klein_to_poincare
from meshhacks import mungmesh


def complex_to_klein_3d(z):
    """
    Convert complex coordinate (stereographic projection) to Klein model.

    The complex plane is the boundary of H³ at infinity.
    We use stereographic projection: z -> point on unit sphere S².

    Args:
        z: complex number (or inf for the point at infinity)

    Returns:
        3D point on the unit sphere (Klein model boundary)
    """
    if np.isinf(z) or np.abs(z) > 1e10:
        return np.array([0, 0, 1])  # North pole

    x, y = z.real, z.imag
    r2 = x*x + y*y

    # Inverse stereographic projection from plane to sphere
    # (x, y) -> (2x/(1+r²), 2y/(1+r²), (r²-1)/(r²+1))
    denom = 1 + r2
    return np.array([2*x/denom, 2*y/denom, (r2-1)/denom])


def create_ideal_polyhedron_mesh(vertices_complex, triangulation):
    """
    Create a trimesh for an ideal polyhedron in the Klein model.

    Args:
        vertices_complex: list of complex coordinates (boundary points)
        triangulation: list of triangle vertex indices

    Returns:
        trimesh.Trimesh in Klein model coordinates
    """
    # Convert complex vertices to Klein model (on unit sphere)
    klein_vertices = np.array([complex_to_klein_3d(z) for z in vertices_complex])

    # The faces are the triangulation
    faces = np.array(triangulation)

    # Create mesh
    mesh = trimesh.Trimesh(vertices=klein_vertices, faces=faces)

    return mesh


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

    The boundary edges of the finite triangulation connect to the
    vertex at infinity to form additional faces.
    """
    # Find boundary edges
    edge_count = {}
    for tri in triangulation:
        for i in range(3):
            e = tuple(sorted([tri[i], tri[(i+1)%3]]))
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary_edges = [e for e, c in edge_count.items() if c == 1]

    # Add faces connecting boundary edges to infinity
    inf_vertex = n_finite  # Index for the infinity vertex
    all_faces = list(triangulation)

    for e in boundary_edges:
        all_faces.append([e[0], e[1], inf_vertex])

    return all_faces


def render_poincare_ball(config, output_path, show_edges=True,
                         subdivisions=3, opacity=0.7):
    """
    Render a Poincaré ball visualization of an ideal polyhedron.

    Args:
        config: dict with vertices_complex and triangulation
        output_path: where to save the image
        show_edges: whether to show edge lines
        subdivisions: number of mesh subdivisions for smooth geodesics
        opacity: mesh opacity
    """
    vertices = config['vertices_complex']
    triangulation = config['triangulation']
    n_finite = len(vertices) - 1  # Last vertex is infinity

    # Complete triangulation with faces at infinity
    all_faces = complete_triangulation_with_infinity(triangulation, n_finite)

    # Create Klein model mesh
    klein_vertices = np.array([complex_to_klein_3d(z) for z in vertices])
    mesh = trimesh.Trimesh(vertices=klein_vertices, faces=all_faces)

    # Transform to Poincaré ball with subdivision
    def safe_klein_to_poincare(vec):
        """Klein to Poincaré with boundary handling."""
        n2 = np.linalg.norm(vec)**2
        if n2 >= 1:
            # On boundary - scale slightly inside
            return vec * 0.999 / np.sqrt(n2) / (1 + np.sqrt(1 - 0.999**2))
        return vec / (1 + np.sqrt(1 - n2))

    poincare_mesh = mungmesh(mesh, safe_klein_to_poincare, num_iters=subdivisions)

    # Convert to pyvista
    pv_mesh = pv.wrap(poincare_mesh)

    # Create plotter
    plotter = pv.Plotter(off_screen=True, window_size=[800, 800])
    plotter.set_background('white')

    # Add mesh
    plotter.add_mesh(pv_mesh,
                     color='lightblue',
                     opacity=opacity,
                     specular=0.5,
                     specular_power=10,
                     show_edges=show_edges,
                     edge_color='darkblue',
                     line_width=1)

    # Add unit sphere (boundary)
    sphere = pv.Sphere(radius=1.0, theta_resolution=50, phi_resolution=50)
    plotter.add_mesh(sphere, color='gray', opacity=0.1, style='wireframe')

    # Set camera
    plotter.camera.position = (2.5, 2.5, 2.5)
    plotter.camera.focal_point = (0, 0, 0)

    # Add title
    plotter.add_text(f"n={config['n']}, V={config['volume']:.4f}",
                     position='upper_left', font_size=12)

    # Save
    plotter.screenshot(output_path)
    plotter.close()


def render_rotating_gif(config, output_path, fps=10, frames=36):
    """Create a rotating GIF of the Poincaré ball visualization."""
    vertices = config['vertices_complex']
    triangulation = config['triangulation']
    n_finite = len(vertices) - 1

    all_faces = complete_triangulation_with_infinity(triangulation, n_finite)
    klein_vertices = np.array([complex_to_klein_3d(z) for z in vertices])
    mesh = trimesh.Trimesh(vertices=klein_vertices, faces=all_faces)

    def safe_klein_to_poincare(vec):
        n2 = np.linalg.norm(vec)**2
        if n2 >= 1:
            return vec * 0.999 / np.sqrt(n2) / (1 + np.sqrt(1 - 0.999**2))
        return vec / (1 + np.sqrt(1 - n2))

    poincare_mesh = mungmesh(mesh, safe_klein_to_poincare, num_iters=3)
    pv_mesh = pv.wrap(poincare_mesh)

    plotter = pv.Plotter(off_screen=True, window_size=[600, 600])
    plotter.set_background('white')
    plotter.add_mesh(pv_mesh, color='lightblue', opacity=0.7,
                     specular=0.5, show_edges=True, edge_color='darkblue')

    sphere = pv.Sphere(radius=1.0, theta_resolution=30, phi_resolution=30)
    plotter.add_mesh(sphere, color='gray', opacity=0.1, style='wireframe')

    plotter.add_text(f"n={config['n']}, V={config['volume']:.4f}",
                     position='upper_left', font_size=10)

    plotter.open_gif(str(output_path), fps=fps)

    for angle in range(0, 360, 360//frames):
        plotter.camera.azimuth = angle
        plotter.camera.elevation = 20
        plotter.camera.distance = 3.5
        plotter.write_frame()

    plotter.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Poincaré ball visualizations of ideal polyhedra"
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
        "--gif", action="store_true",
        help="Also generate rotating GIFs"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATING POINCARÉ BALL VISUALIZATIONS")
    print("=" * 60)

    data_dir = Path(args.data_dir)

    for filepath in sorted(data_dir.glob('*vertex_comprehensive.json')):
        config = load_config(filepath)
        n = config['n']

        print(f"\nProcessing n={n}...")

        # Static image
        output_path = output_dir / f"poincare_n{n}.png"
        try:
            render_poincare_ball(config, str(output_path))
            print(f"  Saved: {output_path}")
        except Exception as e:
            print(f"  Error: {e}")

        # GIF
        if args.gif:
            gif_path = output_dir / f"poincare_n{n}.gif"
            try:
                render_rotating_gif(config, str(gif_path))
                print(f"  Saved: {gif_path}")
            except Exception as e:
                print(f"  GIF error: {e}")

    print("\n" + "=" * 60)
    print(f"Visualizations saved to: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
