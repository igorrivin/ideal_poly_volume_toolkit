"""
3D Visualization utilities for ideal polyhedra.

Supports:
- Poincaré ball model visualization
- Sphere projection with subdivision
- Interactive plots using plotly
"""

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull


def lift_to_sphere_with_inf(W: np.ndarray) -> np.ndarray:
    """
    Lift complex points to sphere via stereographic projection.

    Args:
        W: Complex array of points

    Returns:
        N x 3 array of points on unit sphere
    """
    P = np.zeros((W.shape[0], 3), dtype=np.float64)
    is_inf = ~np.isfinite(W.real) | ~np.isfinite(W.imag)
    F = ~is_inf
    w = W[F]
    r2 = (w.real**2 + w.imag**2)
    denom = r2 + 1.0
    P[F, 0] = 2.0 * w.real / denom
    P[F, 1] = 2.0 * w.imag / denom
    P[F, 2] = (r2 - 1.0) / denom
    P[is_inf] = np.array([0.0, 0.0, 1.0])
    return P


def subdivide_triangle_euclidean(v1, v2, v3, depth=1):
    """
    Recursively subdivide a triangle using Euclidean (straight line) midpoints.

    This is used for subdividing in the Klein model (unit ball with Euclidean geometry).

    Args:
        v1, v2, v3: Triangle vertices (3D points in the ball)
        depth: Number of subdivision levels

    Returns:
        List of subdivided triangular faces
    """
    if depth == 0:
        return [np.array([v1, v2, v3])]

    # Compute Euclidean midpoints (straight lines in Klein model)
    m12 = (v1 + v2) / 2.0
    m23 = (v2 + v3) / 2.0
    m31 = (v3 + v1) / 2.0

    # Recursively subdivide 4 new triangles
    triangles = []
    triangles.extend(subdivide_triangle_euclidean(v1, m12, m31, depth - 1))
    triangles.extend(subdivide_triangle_euclidean(v2, m23, m12, depth - 1))
    triangles.extend(subdivide_triangle_euclidean(v3, m31, m23, depth - 1))
    triangles.extend(subdivide_triangle_euclidean(m12, m23, m31, depth - 1))

    return triangles


def klein_to_poincare(K: np.ndarray) -> np.ndarray:
    """
    Map Klein ball model to Poincaré ball model.

    The Klein model uses the unit ball with Euclidean (straight line) geodesics.
    The Poincaré model uses the same ball with hyperbolic (curved) geodesics.

    Formula: If k is a point in Klein ball with |k| < 1, then
             p = k / (1 + sqrt(1 - |k|^2))

    Args:
        K: N x 3 array of points in Klein ball

    Returns:
        N x 3 array of points in Poincaré ball
    """
    r_squared = np.sum(K**2, axis=1)

    # Clip to avoid numerical issues near boundary
    r_squared = np.clip(r_squared, 0, 0.9999)

    # Klein to Poincaré transformation
    denom = 1.0 + np.sqrt(1.0 - r_squared)

    result = K / denom[:, np.newaxis]

    return result


def create_polyhedron_mesh(vertices_complex, subdivisions=2):
    """
    Create a subdivided mesh for visualization.

    Algorithm:
    1. Lift to sphere (gives Klein model in the ball)
    2. Get convex hull faces
    3. Subdivide each face using Euclidean midpoints (Klein model)
    4. Map subdivided vertices Klein → Poincaré

    Args:
        vertices_complex: Complex array of vertices
        subdivisions: Number of subdivision levels

    Returns:
        dict with 'klein' and 'poincare' meshes
    """
    # Step 1: Lift to sphere (this gives us the Klein model in the ball)
    klein_vertices = lift_to_sphere_with_inf(vertices_complex)

    # Step 2: Compute convex hull (this is the Klein model of the polyhedron)
    hull = ConvexHull(klein_vertices)

    # Step 3 & 4: Subdivide each face in Klein, then map to Poincaré
    subdivided_triangles_klein = []
    subdivided_triangles_poincare = []

    for simplex in hull.simplices:
        v1, v2, v3 = klein_vertices[simplex]

        # Subdivide in Klein model (Euclidean straight-line subdivision)
        sub_tris_klein = subdivide_triangle_euclidean(v1, v2, v3, depth=subdivisions)
        subdivided_triangles_klein.extend(sub_tris_klein)

        # Map each subdivided triangle to Poincaré ball
        for tri_klein in sub_tris_klein:
            tri_poincare = klein_to_poincare(tri_klein)
            subdivided_triangles_poincare.append(tri_poincare)

    return {
        'klein': {
            'triangles': subdivided_triangles_klein,
            'vertices': klein_vertices,
            'original_faces': hull.simplices
        },
        'poincare': {
            'triangles': subdivided_triangles_poincare,
            'vertices': klein_to_poincare(klein_vertices),
            'original_faces': hull.simplices
        }
    }


def plot_polyhedron_klein(vertices_complex, subdivisions=2, title="Ideal Polyhedron (Klein Model)"):
    """
    Create interactive 3D plot of polyhedron in Klein ball model.

    Args:
        vertices_complex: Complex array of vertices
        subdivisions: Number of subdivision levels
        title: Plot title

    Returns:
        plotly Figure object
    """
    mesh = create_polyhedron_mesh(vertices_complex, subdivisions)
    triangles = mesh['klein']['triangles']

    # Collect all vertices and triangle indices for Mesh3d
    vertices_list = []
    indices_i, indices_j, indices_k = [], [], []
    vertex_map = {}

    for tri in triangles:
        tri_indices = []
        for i in range(3):
            vertex_tuple = tuple(tri[i])
            if vertex_tuple not in vertex_map:
                vertex_map[vertex_tuple] = len(vertices_list)
                vertices_list.append(tri[i])
            tri_indices.append(vertex_map[vertex_tuple])

        # Add triangle indices
        indices_i.append(tri_indices[0])
        indices_j.append(tri_indices[1])
        indices_k.append(tri_indices[2])

    vertices_array = np.array(vertices_list)

    # Create figure
    fig = go.Figure()

    # Add polyhedron as a mesh surface
    fig.add_trace(go.Mesh3d(
        x=vertices_array[:, 0],
        y=vertices_array[:, 1],
        z=vertices_array[:, 2],
        i=indices_i,
        j=indices_j,
        k=indices_k,
        color='lightblue',
        opacity=0.7,
        flatshading=False,
        name='Polyhedron',
        hoverinfo='skip'
    ))

    # Add vertices
    vertices = mesh['klein']['vertices']
    fig.add_trace(go.Scatter3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        mode='markers',
        marker=dict(size=8, color='red'),
        name='Vertices',
        hovertext=[f'Vertex {i}' for i in range(len(vertices))]
    ))

    # Add transparent ball for reference
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_ball = np.outer(np.cos(u), np.sin(v))
    y_ball = np.outer(np.sin(u), np.sin(v))
    z_ball = np.outer(np.ones(np.size(u)), np.cos(v))

    fig.add_trace(go.Surface(
        x=x_ball, y=y_ball, z=z_ball,
        opacity=0.1,
        colorscale=[[0, 'lightgray'], [1, 'lightgray']],
        showscale=False,
        name='Unit Ball',
        hoverinfo='skip'
    ))

    # Layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[-1.2, 1.2], title='X'),
            yaxis=dict(range=[-1.2, 1.2], title='Y'),
            zaxis=dict(range=[-1.2, 1.2], title='Z'),
            aspectmode='cube'
        ),
        showlegend=True,
        width=800,
        height=800
    )

    return fig


def plot_polyhedron_poincare(vertices_complex, subdivisions=2, title="Ideal Polyhedron (Poincaré Ball)"):
    """
    Create interactive 3D plot of polyhedron in Poincaré ball model.

    Args:
        vertices_complex: Complex array of vertices
        subdivisions: Number of subdivision levels
        title: Plot title

    Returns:
        plotly Figure object
    """
    mesh = create_polyhedron_mesh(vertices_complex, subdivisions)
    triangles = mesh['poincare']['triangles']

    # Collect all vertices and triangle indices for Mesh3d
    vertices_list = []
    indices_i, indices_j, indices_k = [], [], []
    vertex_map = {}

    for tri in triangles:
        tri_indices = []
        for i in range(3):
            vertex_tuple = tuple(tri[i])
            if vertex_tuple not in vertex_map:
                vertex_map[vertex_tuple] = len(vertices_list)
                vertices_list.append(tri[i])
            tri_indices.append(vertex_map[vertex_tuple])

        # Add triangle indices
        indices_i.append(tri_indices[0])
        indices_j.append(tri_indices[1])
        indices_k.append(tri_indices[2])

    vertices_array = np.array(vertices_list)

    # Create figure
    fig = go.Figure()

    # Add polyhedron as a mesh surface
    fig.add_trace(go.Mesh3d(
        x=vertices_array[:, 0],
        y=vertices_array[:, 1],
        z=vertices_array[:, 2],
        i=indices_i,
        j=indices_j,
        k=indices_k,
        color='lightblue',
        opacity=0.7,
        flatshading=False,
        name='Polyhedron',
        hoverinfo='skip'
    ))

    # Add vertices
    vertices = mesh['poincare']['vertices']
    fig.add_trace(go.Scatter3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        mode='markers',
        marker=dict(size=8, color='red'),
        name='Vertices',
        hovertext=[f'Vertex {i}' for i in range(len(vertices))]
    ))

    # Add unit sphere boundary
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        opacity=0.1,
        colorscale=[[0, 'lightgray'], [1, 'lightgray']],
        showscale=False,
        name='Unit Ball',
        hoverinfo='skip'
    ))

    # Layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[-1.2, 1.2], title='X'),
            yaxis=dict(range=[-1.2, 1.2], title='Y'),
            zaxis=dict(range=[-1.2, 1.2], title='Z'),
            aspectmode='cube'
        ),
        showlegend=True,
        width=800,
        height=800
    )

    return fig


def plot_delaunay_2d(vertices_complex, triangulation_indices, title="Delaunay Triangulation"):
    """
    Create 2D plot of Delaunay triangulation in complex plane.

    Args:
        vertices_complex: Complex array of vertices
        triangulation_indices: Array of triangle indices
        title: Plot title

    Returns:
        plotly Figure object
    """
    fig = go.Figure()

    # Plot triangulation edges
    for tri in triangulation_indices:
        i, j, k = tri
        vertices_tri = vertices_complex[[i, j, k, i]]  # Close the triangle

        fig.add_trace(go.Scatter(
            x=vertices_tri.real,
            y=vertices_tri.imag,
            mode='lines',
            line=dict(color='blue', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Plot vertices
    fig.add_trace(go.Scatter(
        x=vertices_complex.real,
        y=vertices_complex.imag,
        mode='markers+text',
        marker=dict(size=10, color='red'),
        text=[f'{i}' for i in range(len(vertices_complex))],
        textposition='top center',
        name='Vertices',
        hovertext=[f'Vertex {i}: {z:.3f}' for i, z in enumerate(vertices_complex)]
    ))

    # Layout
    fig.update_layout(
        title=title,
        xaxis_title='Real',
        yaxis_title='Imaginary',
        width=700,
        height=700,
        showlegend=True,
        hovermode='closest'
    )

    fig.update_xaxes(scaleanchor="y", scaleratio=1)

    return fig
