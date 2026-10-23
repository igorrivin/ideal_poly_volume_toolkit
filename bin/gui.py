#!/usr/bin/env python3
"""
Gradio GUI for Ideal Polyhedron Volume Toolkit

Interactive interface for:
- Optimizing polyhedra
- Analyzing distributions
- 3D visualization in sphere and Poincaré ball models

Usage:
    python bin/gui.py           # Local only (127.0.0.1:7860)
    python bin/gui.py --share   # Create shareable public link
    python bin/gui.py --port 8080  # Custom port
"""

import gradio as gr
import numpy as np
import json
import io
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from pathlib import Path
from PIL import Image

from ideal_poly_volume_toolkit import __version__
from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    ideal_poly_volume_via_delaunay,
)
from ideal_poly_volume_toolkit.optimization import (
    negative_volume_objective as _negative_volume_objective,
    optimize_ideal_polyhedron,
    optimization_result_to_dict,
    spherical_to_complex as _spherical_to_complex,
)
from ideal_poly_volume_toolkit.visualization import (
    plot_polyhedron_klein,
    plot_polyhedron_poincare,
    plot_delaunay_2d,
)
from ideal_poly_volume_toolkit.rivin_holonomy import (
    Triangulation,
    generators_from_triangulation,
)
from ideal_poly_volume_toolkit.rivin_delaunay import (
    check_delaunay_realizability,
    optimize_hyperbolic_volume,
    realize_angles_as_points,
)
from ideal_poly_volume_toolkit.symmetry import (
    compute_symmetry_group,
    format_symmetry_report,
)

# ============================================================================
# Optimization Functions
# ============================================================================


def spherical_to_complex(theta, phi):
    """Compatibility wrapper around the canonical package projection."""
    return _spherical_to_complex(theta, phi)


def compute_volume(params, n_vertices):
    """Compatibility wrapper around the canonical package objective."""
    return _negative_volume_objective(params, n_vertices)


def run_optimization(n_vertices, n_trials, max_iter, pop_size, seed, progress=gr.Progress()):
    """Run the canonical package optimizer with GUI progress reporting."""
    try:
        n_vertices = int(n_vertices)
        n_trials = int(n_trials)
        max_iter = int(max_iter)
        pop_size = int(pop_size)
        seed = int(seed)
        if n_vertices < 4:
            return "Error: Number of vertices must be at least 4", None
        if n_vertices > 100:
            return "Error: Number of vertices limited to 100 for practical computation time", None
        if n_trials < 1 or max_iter < 1 or pop_size < 1:
            return "Error: Trials, iterations, and population multiplier must be positive", None
    except (ValueError, TypeError):
        return "Error: Optimization settings must be integers", None

    progress(0, desc="Starting vectorized optimization...")
    effective_popsize = pop_size

    def report_trial(trial):
        volume = "invalid" if trial.volume is None else f"{trial.volume:.6f}"
        progress(
            trial.trial / n_trials,
            desc=f"Completed trial {trial.trial}/{n_trials}: {volume}",
        )

    try:
        result = optimize_ideal_polyhedron(
            n_vertices,
            n_trials=n_trials,
            maxiter=max_iter,
            popsize=effective_popsize,
            seed=seed,
            workers=1,
            trial_callback=report_trial,
        )
    except (ValueError, RuntimeError) as exc:
        return f"Error: {exc}", None

    best_volume = result.best_volume
    Z_np = result.finite_vertices
    idx = result.finite_triangulation
    structure = result.structure

    # Save result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_file = f"results/data/{n_vertices}vertex_optimization_gui_{timestamp}.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    output_data = optimization_result_to_dict(
        result,
        source="gui",
        settings={
            "n_trials": n_trials,
            "maxiter": max_iter,
            "popsize": effective_popsize,
            "workers": 1,
            "seed_base": seed,
        },
    )
    stats = output_data["statistics"]
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Create summary text
    summary = f"""
## Optimization Results

**Configuration:**
- Vertices: {n_vertices}
- Trials: {n_trials}
- Iterations per trial: {max_iter}

**Best Result:**
- Volume: {best_volume:.8f}
- Faces: {structure['n_faces']} (including {structure['n_infinity_faces']} at ∞)

**Statistics over all trials:**
- Mean: {stats['mean_volume']:.8f}
- Std Dev: {stats['std_volume']:.8f}
- Best/Mean: {best_volume/stats['mean_volume']:.4f}

**Saved to:** `{output_file}`
"""

    # Return data as dict for state management
    opt_data = {
        'vertices': Z_np,
        'triangulation': idx,
        'boundary_triangulation': result.boundary_triangulation,
        'stats': stats
    }

    return summary, opt_data


# ============================================================================
# Distribution Analysis Functions
# ============================================================================

def analyze_volume_distribution(n_vertices, n_samples, seed, progress=gr.Progress()):
    """Analyze volume distribution with progress tracking.

    Note: n_vertices is the number of random vertices to add.
    Finite vertices = n_vertices random + 2 fixed (0, 1) = n_vertices + 2
    Total with infinity = n_vertices + 3
    """
    rng = np.random.default_rng(seed)

    # Fixed vertices: 0, 1 (infinity is implicit in the volume computation)
    fixed_vertices = [complex(0, 0), complex(1, 0)]
    n_random = n_vertices
    total_vertices = n_vertices + 2  # 0, 1 + n_vertices random (∞ is implicit, not counted)

    volumes = []

    def sample_random_vertex():
        vec = rng.standard_normal(3)
        vec = vec / np.linalg.norm(vec)
        x, y, z = vec
        if z > 0.999:
            return None
        w = complex(x/(1-z), y/(1-z))
        return w

    for i in range(n_samples):
        if (i + 1) % 100 == 0:
            progress((i + 1) / n_samples, desc=f"Sampling: {i + 1}/{n_samples}")

        vertices = fixed_vertices.copy()

        for _ in range(n_random):
            v = sample_random_vertex()
            if v is None or any(abs(v - existing) < 0.01 for existing in vertices):
                continue
            vertices.append(v)

        if len(vertices) != total_vertices:
            continue

        try:
            vertices_np = np.array(vertices, dtype=np.complex128)
            vol = ideal_poly_volume_via_delaunay(
                vertices_np, mode="bloch_wigner"
            )
            if np.isfinite(vol) and 0 < vol < 1000:
                volumes.append(vol)
        except Exception:
            # Silently skip failed computations
            pass

    volumes = np.array(volumes)

    # Check if we have any valid volumes
    if len(volumes) == 0:
        error_msg = f"""
## Distribution Analysis Failed

**Error:** No valid volume computations succeeded.

**Possible causes:**
- Random vertices may be generating degenerate configurations
- Try increasing the number of samples
- Try a different number of vertices

**Configuration:**
- Random vertices requested: {n_random}
- Samples attempted: {n_samples}
- Valid samples: 0
"""
        return error_msg, None

    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(volumes, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(volumes), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(volumes):.4f}')
    ax.axvline(np.median(volumes), color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(volumes):.4f}')
    ax.set_xlabel('Volume', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{total_vertices}-Vertex Volume Distribution ({len(volumes)} samples)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save plot to BytesIO and convert to PIL Image for Gradio
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()

    # Convert BytesIO to PIL Image for Gradio
    img = Image.open(buf)

    # Statistics summary
    summary = f"""
## Distribution Analysis Results

**Configuration:**
- Random vertices: {n_random}
- Fixed vertices: 2 (at 0, 1)
- **Finite vertices: {total_vertices}**
- **Total (with ∞): {total_vertices + 1}**
- Samples requested: {n_samples}
- Valid samples: {len(volumes)}

**Statistics:**
- Mean: {np.mean(volumes):.8f}
- Median: {np.median(volumes):.8f}
- Std Dev: {np.std(volumes):.8f}
- Min: {np.min(volumes):.8f}
- Max: {np.max(volumes):.8f}
- Q25: {np.percentile(volumes, 25):.8f}
- Q75: {np.percentile(volumes, 75):.8f}
"""

    return summary, img


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_configuration(vertices_real, vertices_imag, vis_type, subdivisions):
    """Create visualization based on user selection."""
    if not vertices_real or not vertices_imag:
        return None, "Please provide vertices"

    try:
        # Parse vertices
        real_parts = [float(x.strip()) for x in vertices_real.split(',')]
        imag_parts = [float(x.strip()) for x in vertices_imag.split(',')]

        if len(real_parts) != len(imag_parts):
            return None, "Real and imaginary parts must have same length"

        vertices = np.array([complex(r, i) for r, i in zip(real_parts, imag_parts)])

        if vis_type == "Delaunay (2D)":
            idx = delaunay_triangulation_indices(vertices)
            fig = plot_delaunay_2d(vertices, idx)
            return fig, f"Showing Delaunay triangulation with {len(idx)} faces"

        elif vis_type == "Klein Ball (3D)":
            fig = plot_polyhedron_klein(vertices, subdivisions=subdivisions)
            return fig, f"Showing Klein model (subdivision level: {subdivisions})"

        elif vis_type == "Poincaré Ball (3D)":
            fig = plot_polyhedron_poincare(vertices, subdivisions=subdivisions)
            return fig, f"Showing Poincaré ball model (subdivision level: {subdivisions})"

    except Exception as e:
        return None, f"Error: {str(e)}"


def load_from_optimization(opt_data):
    """Load vertices from optimization results."""
    if opt_data is None:
        return "", ""

    # opt_data is a dict with vertices
    vertices = opt_data.get('vertices', None)
    if vertices is None:
        return "", ""

    real_str = ", ".join(f"{v.real:.6f}" for v in vertices)
    imag_str = ", ".join(f"{v.imag:.6f}" for v in vertices)

    return real_str, imag_str


# ============================================================================
# Holonomy/Arithmeticity Functions
# ============================================================================

def compute_holonomy_analysis(vertices_real, vertices_imag, progress=gr.Progress()):
    """Compute holonomy generators and check arithmeticity."""
    if not vertices_real or not vertices_imag:
        return "Please provide vertices", None

    try:
        # Parse vertices
        real_parts = [float(x.strip()) for x in vertices_real.split(',')]
        imag_parts = [float(x.strip()) for x in vertices_imag.split(',')]

        if len(real_parts) != len(imag_parts):
            return "Real and imaginary parts must have same length", None

        vertices = np.array([complex(r, i) for r, i in zip(real_parts, imag_parts)])

        progress(0.2, desc="Computing Delaunay triangulation...")

        # Get triangulation
        idx = delaunay_triangulation_indices(vertices)
        F = len(idx)

        progress(0.4, desc="Building adjacency structure...")

        # Build adjacency structure
        adjacency = {}
        edge_id_map = {}
        edge_id = 0

        for i, tri_i in enumerate(idx):
            for side_i in range(3):
                v1_i, v2_i = tri_i[side_i], tri_i[(side_i + 1) % 3]
                edge = tuple(sorted([v1_i, v2_i]))

                # Find matching triangle
                for j, tri_j in enumerate(idx):
                    if i == j:
                        continue
                    for side_j in range(3):
                        v1_j, v2_j = tri_j[side_j], tri_j[(side_j + 1) % 3]
                        if set([v1_j, v2_j]) == set([v1_i, v2_i]):
                            if (i, side_i) not in adjacency:
                                if edge not in edge_id_map:
                                    edge_id_map[edge] = edge_id
                                    edge_id += 1
                                adjacency[(i, side_i)] = (j, side_j, edge_id_map[edge])

        progress(0.6, desc="Computing holonomy generators...")

        # Define order and orientation
        order = {t: [0, 1, 2] for t in range(F)}
        orientation = {}
        for edge, eid in edge_id_map.items():
            for (t, s), (u, su, e) in adjacency.items():
                if e == eid:
                    orientation[eid] = ((t, s), (u, su))
                    break

        # Create triangulation
        T = Triangulation(F, adjacency, order, orientation)

        # Zero shears for ideal polyhedra
        Z = {eid: 0.0 for eid in range(edge_id)}

        # Compute generators
        gens = generators_from_triangulation(T, Z, root=0)

        progress(0.8, desc="Analyzing traces...")

        # Analyze traces
        trace_analysis = []
        traces = []
        integral_count = 0

        for i, (u, v, tokens, M) in enumerate(gens):
            trace = M[0][0] + M[1][1]
            traces.append(trace)

            nearest_int = round(trace)
            distance = abs(trace - nearest_int)
            is_close = distance < 0.01

            if is_close:
                integral_count += 1

            trace_analysis.append({
                'generator': i,
                'edge': (u, v),
                'trace': float(trace),
                'nearest_int': int(nearest_int),
                'distance': float(distance),
                'is_close': is_close
            })

        progress(1.0, desc="Complete!")

        # Create summary
        summary = f"""
## Holonomy Analysis Results

**Configuration:**
- Vertices: {len(vertices)}
- Triangular faces: {F}
- Number of generators: {len(gens)}

**Arithmeticity Test:**
- Generators with integral traces: {integral_count}/{len(gens)}
- Percentage: {100*integral_count/len(gens):.1f}%

**Interpretation:**
"""
        if integral_count == len(gens):
            summary += "✅ **ALL TRACES ARE INTEGERS!**\n\n"
            summary += "This polyhedron is **ARITHMETIC** - it has deep number-theoretic structure!\n"
            summary += "The holonomy lies in PSL(2,O_K) for some number field K."
        elif integral_count > len(gens) * 0.7:
            summary += "⚠️ **MOST TRACES ARE CLOSE TO INTEGERS**\n\n"
            summary += f"This suggests possible arithmetic structure with {integral_count}/{len(gens)} integral traces.\n"
            summary += "May be commensurable with an arithmetic group."
        else:
            summary += "❌ **NOT ARITHMETIC**\n\n"
            summary += "Only a few traces are close to integers. This is likely a generic configuration."

        summary += "\n\n## Trace Details:\n\n"
        summary += "| Generator | Edge | Trace | Nearest Int | Distance | Status |\n"
        summary += "|-----------|------|-------|-------------|----------|--------|\n"

        for ta in trace_analysis:
            status = "✅ INTEGRAL" if ta['is_close'] else "❌"
            summary += f"| {ta['generator']} | {ta['edge'][0]}-{ta['edge'][1]} | "
            summary += f"{ta['trace']:.6f} | {ta['nearest_int']} | "
            summary += f"{ta['distance']:.6f} | {status} |\n"

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot traces
        gen_nums = [ta['generator'] for ta in trace_analysis]
        trace_vals = [ta['trace'] for ta in trace_analysis]
        colors = ['green' if ta['is_close'] else 'red' for ta in trace_analysis]

        ax1.bar(gen_nums, trace_vals, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('Generator', fontsize=12)
        ax1.set_ylabel('Trace', fontsize=12)
        ax1.set_title('Holonomy Generator Traces', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Plot distances from integers
        distances = [ta['distance'] for ta in trace_analysis]
        ax2.bar(gen_nums, distances, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(y=0.01, color='orange', linestyle='--', linewidth=2,
                    label='Threshold (0.01)')
        ax2.set_xlabel('Generator', fontsize=12)
        ax2.set_ylabel('Distance from nearest integer', fontsize=12)
        ax2.set_title('Integrality Test', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        plt.tight_layout()

        # Save plot to BytesIO and convert to PIL Image for Gradio
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close()

        # Convert BytesIO to PIL Image for Gradio
        img = Image.open(buf)

        return summary, img

    except Exception as e:
        return f"Error: {str(e)}", None


def compute_symmetry_analysis(vertices_real, vertices_imag):
    """Compute symmetry group of the polyhedron."""
    if not vertices_real or not vertices_imag:
        return "Please provide vertices"

    try:
        # Parse vertices
        real_parts = [float(x.strip()) for x in vertices_real.split(',')]
        imag_parts = [float(x.strip()) for x in vertices_imag.split(',')]

        if len(real_parts) != len(imag_parts):
            return "Real and imaginary parts must have same length"

        vertices = np.array([complex(r, i) for r, i in zip(real_parts, imag_parts)])

        # Lift to 3D (Klein model in the ball)
        from ideal_poly_volume_toolkit.visualization import lift_to_sphere_with_inf
        vertices_3d = lift_to_sphere_with_inf(vertices)

        # Compute symmetry group
        sym_info = compute_symmetry_group(vertices_3d)

        # Format report
        report = format_symmetry_report(sym_info)

        return report

    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# Fixed Combinatorics Optimization Functions
# ============================================================================

def generate_random_sphere_points(n_points, seed=None):
    """
    Generate n random points uniformly on the unit sphere, plus north pole (infinity).

    Returns complex points via stereographic projection (north pole -> infinity).
    """
    rng = np.random.default_rng(seed)

    # Generate n uniform random points on sphere using Gaussian method
    points_3d = []
    for _ in range(n_points):
        vec = rng.standard_normal(3)
        vec = vec / np.linalg.norm(vec)
        points_3d.append(vec)

    points_3d = np.array(points_3d)

    # Stereographic projection from north pole (0, 0, 1) to complex plane
    # For point (x, y, z), the projection is w = (x + iy) / (1 - z)
    complex_points = []
    for x, y, z in points_3d:
        if z > 0.9999:  # Very close to north pole - treat as infinity
            complex_points.append(complex(np.inf, np.inf))
        else:
            w = complex(x / (1 - z), y / (1 - z))
            complex_points.append(w)

    return np.array(complex_points), points_3d


def inverse_stereographic_to_sphere(complex_points):
    """
    Map complex points back to sphere via inverse stereographic projection.

    For w = x + iy, the sphere point is:
    X = 2x / (|w|^2 + 1)
    Y = 2y / (|w|^2 + 1)
    Z = (|w|^2 - 1) / (|w|^2 + 1)
    """
    points_3d = []
    for w in complex_points:
        if not np.isfinite(w):
            points_3d.append([0, 0, 1])  # North pole
        else:
            r2 = w.real**2 + w.imag**2
            denom = r2 + 1
            X = 2 * w.real / denom
            Y = 2 * w.imag / denom
            Z = (r2 - 1) / denom
            points_3d.append([X, Y, Z])

    return np.array(points_3d)


def optimize_fixed_combinatorics(n_points, seed, progress=gr.Progress()):
    """
    Generate random points, extract combinatorics, optimize volume for fixed combinatorics.

    This uses the Rivin-Delaunay algorithm to find the maximal volume configuration
    while keeping the combinatorial structure (triangulation) fixed.
    """
    progress(0.1, desc="Generating random points on sphere...")

    # Generate random points
    complex_points, sphere_points = generate_random_sphere_points(n_points, seed)

    # Filter out any infinite points for Delaunay computation
    finite_mask = np.isfinite(complex_points)
    finite_points = complex_points[finite_mask]

    if len(finite_points) < 3:
        return "Error: Need at least 3 finite points", None, None

    progress(0.2, desc="Computing Delaunay triangulation...")

    # Compute Delaunay triangulation to get combinatorics
    triangulation_indices = delaunay_triangulation_indices(finite_points)
    triangles = [tuple(tri) for tri in triangulation_indices]

    n_triangles = len(triangles)
    n_vertices = len(finite_points)

    progress(0.3, desc="Checking Delaunay realizability...")

    # Check realizability (should always be realizable since we started from Delaunay)
    realizability = check_delaunay_realizability(triangles, verbose=False)

    if not realizability['realizable']:
        return f"Error: Triangulation not realizable: {realizability['message']}", None, None

    progress(0.5, desc="Optimizing hyperbolic volume...")

    # Optimize volume for this fixed combinatorics
    opt_result = optimize_hyperbolic_volume(
        triangles,
        initial_angles=realizability['angles_radians'],
        verbose=False
    )

    if not opt_result['success']:
        return f"Warning: Optimization may not have converged: {opt_result['message']}", None, None

    optimal_volume = opt_result['volume']
    optimal_angles = opt_result['angles']

    progress(0.7, desc="Reconstructing geometry from optimal angles...")

    # Realize the optimal angles as point positions
    realization = realize_angles_as_points(
        triangles,
        optimal_angles,
        verbose=False
    )

    if not realization['success']:
        # Even if realization didn't fully succeed, we may have useful data
        pass

    # Get the optimized vertices
    if realization['points'] is not None:
        # Map from realization indices to complex numbers
        vertex_list = realization['vertex_list']
        points_2d = realization['points']

        # Create complex array in original vertex order
        optimized_complex = np.zeros(len(vertex_list), dtype=complex)
        for i, v in enumerate(vertex_list):
            optimized_complex[v] = complex(points_2d[i, 0], points_2d[i, 1])
    else:
        optimized_complex = finite_points  # Fall back to original

    progress(0.9, desc="Computing initial volume for comparison...")

    # Compute initial volume for comparison
    initial_volume = ideal_poly_volume_via_delaunay(finite_points, use_bloch_wigner=True)

    progress(1.0, desc="Complete!")

    # Build summary
    summary = f"""
## Fixed Combinatorics Optimization Results

**Input Configuration:**
- Random points generated: {n_points}
- Finite vertices (after projection): {n_vertices}
- Triangular faces: {n_triangles}
- Random seed: {seed}

**Optimization Results:**
- Initial volume (random): {initial_volume:.8f}
- Optimal volume (Rivin): {optimal_volume:.8f}
- Volume improvement: {(optimal_volume/initial_volume - 1)*100:.2f}%

**Geometry Reconstruction:**
- Success: {'Yes' if realization['success'] else 'Partial'}
- Triangulation preserved: {'Yes' if realization.get('triangulation_preserved', False) else 'No'}
- Angle error: {realization.get('angle_error_degrees', 'N/A'):.4f}° (if applicable)

**Interpretation:**
The Rivin-Delaunay algorithm finds the unique maximal volume configuration
for the given combinatorial triangulation structure.
"""

    # Prepare data for visualization and holonomy check
    opt_data = {
        'vertices': optimized_complex,
        'triangulation': triangulation_indices,
        'volume': optimal_volume,
        'initial_volume': initial_volume,
        'angles': optimal_angles,
        'triangles': triangles,
        'n_vertices': n_vertices,
        'n_faces': n_triangles,
    }

    return summary, opt_data, optimized_complex


def batch_fixed_combinatorics_test(n_points, n_trials, seed, progress=gr.Progress()):
    """
    Run multiple trials of fixed combinatorics optimization and check arithmeticity.
    """
    results = []
    all_arithmetic = True

    for trial in range(n_trials):
        progress((trial + 0.5) / n_trials, desc=f"Trial {trial + 1}/{n_trials}...")

        trial_seed = seed + trial

        # Generate and optimize
        summary, opt_data, optimized_complex = optimize_fixed_combinatorics(
            n_points, trial_seed, progress=gr.Progress()
        )

        if opt_data is None:
            results.append({
                'trial': trial + 1,
                'seed': trial_seed,
                'success': False,
                'error': summary
            })
            continue

        # Check arithmeticity via holonomy
        triangles = opt_data['triangles']

        # Build holonomy structure
        try:
            F = len(triangles)
            adjacency = {}
            edge_id_map = {}
            edge_id = 0

            for i, tri_i in enumerate(triangles):
                for side_i in range(3):
                    v1_i, v2_i = tri_i[side_i], tri_i[(side_i + 1) % 3]
                    edge = tuple(sorted([v1_i, v2_i]))

                    for j, tri_j in enumerate(triangles):
                        if i == j:
                            continue
                        for side_j in range(3):
                            v1_j, v2_j = tri_j[side_j], tri_j[(side_j + 1) % 3]
                            if set([v1_j, v2_j]) == set([v1_i, v2_i]):
                                if (i, side_i) not in adjacency:
                                    if edge not in edge_id_map:
                                        edge_id_map[edge] = edge_id
                                        edge_id += 1
                                    adjacency[(i, side_i)] = (j, side_j, edge_id_map[edge])

            order = {t: [0, 1, 2] for t in range(F)}
            orientation = {}
            for edge, eid in edge_id_map.items():
                for (t, s), (u, su, e) in adjacency.items():
                    if e == eid:
                        orientation[eid] = ((t, s), (u, su))
                        break

            T = Triangulation(F, adjacency, order, orientation)
            Z = {eid: 0.0 for eid in range(edge_id)}
            gens = generators_from_triangulation(T, Z, root=0)

            # Check traces
            integral_count = 0
            traces = []
            for u, v, tokens, M in gens:
                trace = M[0][0] + M[1][1]
                traces.append(trace)
                if abs(trace - round(trace)) < 0.01:
                    integral_count += 1

            is_arithmetic = (integral_count == len(gens)) if len(gens) > 0 else False
            if not is_arithmetic:
                all_arithmetic = False

            results.append({
                'trial': trial + 1,
                'seed': trial_seed,
                'success': True,
                'volume': opt_data['volume'],
                'initial_volume': opt_data['initial_volume'],
                'n_generators': len(gens),
                'integral_traces': integral_count,
                'is_arithmetic': is_arithmetic,
                'traces': traces,
            })

        except Exception as e:
            results.append({
                'trial': trial + 1,
                'seed': trial_seed,
                'success': False,
                'error': str(e)
            })
            all_arithmetic = False

    # Build summary report
    summary = f"""
## Batch Fixed Combinatorics Test Results

**Configuration:**
- Points per trial: {n_points}
- Number of trials: {n_trials}
- Starting seed: {seed}

**Results Summary:**
"""

    successful = [r for r in results if r.get('success', False)]
    arithmetic_count = sum(1 for r in successful if r.get('is_arithmetic', False))

    summary += f"- Successful trials: {len(successful)}/{n_trials}\n"
    summary += f"- Arithmetic configurations: {arithmetic_count}/{len(successful)}\n\n"

    if all_arithmetic and len(successful) == n_trials:
        summary += "**ALL CONFIGURATIONS ARE ARITHMETIC!**\n\n"

    summary += "| Trial | Seed | Volume | Init Vol | Improvement | Generators | Integral | Arithmetic |\n"
    summary += "|-------|------|--------|----------|-------------|------------|----------|------------|\n"

    for r in results:
        if r.get('success', False):
            improvement = (r['volume'] / r['initial_volume'] - 1) * 100
            arith_str = "YES" if r['is_arithmetic'] else "NO"
            summary += f"| {r['trial']} | {r['seed']} | {r['volume']:.6f} | {r['initial_volume']:.6f} | {improvement:+.1f}% | {r['n_generators']} | {r['integral_traces']}/{r['n_generators']} | {arith_str} |\n"
        else:
            summary += f"| {r['trial']} | {r['seed']} | ERROR | - | - | - | - | - |\n"

    # Create plot of volumes
    if successful:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        trials = [r['trial'] for r in successful]
        volumes = [r['volume'] for r in successful]
        init_vols = [r['initial_volume'] for r in successful]

        ax1.bar(np.array(trials) - 0.2, init_vols, 0.4, label='Initial (random)', alpha=0.7)
        ax1.bar(np.array(trials) + 0.2, volumes, 0.4, label='Optimized (Rivin)', alpha=0.7)
        ax1.set_xlabel('Trial')
        ax1.set_ylabel('Volume')
        ax1.set_title('Volume Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot integral trace fractions
        fractions = [r['integral_traces'] / r['n_generators'] if r['n_generators'] > 0 else 0
                    for r in successful]
        colors = ['green' if f == 1.0 else 'red' for f in fractions]
        ax2.bar(trials, fractions, color=colors, alpha=0.7)
        ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='100% integral')
        ax2.set_xlabel('Trial')
        ax2.set_ylabel('Fraction of Integral Traces')
        ax2.set_title('Arithmeticity Check')
        ax2.set_ylim(0, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close()

        img = Image.open(buf)
    else:
        img = None

    return summary, img


# ============================================================================
# Gradio Interface
# ============================================================================

APP_THEME = gr.themes.Soft()


def create_gui():
    """Create the main Gradio interface."""

    with gr.Blocks(title="Ideal Polyhedron Volume Toolkit") as demo:
        # Shared state for passing optimization results to visualization
        opt_result_state = gr.State(None)
        gr.Markdown("""
        # 🔺 Ideal Polyhedron Volume Toolkit

        Interactive GUI for computing and optimizing volumes of ideal hyperbolic polyhedra.
        """)

        with gr.Tabs():
            # ================================================================
            # Tab 1: Optimization
            # ================================================================
            with gr.Tab("🎯 Optimization"):
                gr.Markdown("Find maximal volume configurations for ideal polyhedra")

                with gr.Row():
                    with gr.Column():
                        opt_vertices = gr.Number(value=7, label="Number of Vertices",
                                                minimum=4, maximum=100,
                                                info="Recommended: 4-30 (higher is much slower)")
                        opt_trials = gr.Slider(1, 50, value=10, step=1,
                                              label="Number of Trials")
                        opt_maxiter = gr.Slider(50, 500, value=150, step=50,
                                               label="Max Iterations per Trial",
                                               info="150-200 is usually sufficient")
                        opt_popsize = gr.Slider(
                            10,
                            30,
                            value=15,
                            step=5,
                            label="Population Multiplier",
                            info="Nominal population = multiplier × 2(N−3)",
                        )
                        opt_seed = gr.Number(value=42, label="Random Seed")

                        opt_button = gr.Button("Run Optimization", variant="primary")

                    with gr.Column():
                        opt_output = gr.Markdown("Results will appear here...")

                opt_button.click(
                    run_optimization,
                    inputs=[opt_vertices, opt_trials, opt_maxiter, opt_popsize, opt_seed],
                    outputs=[opt_output, opt_result_state]
                )

            # ================================================================
            # Tab 2: Distribution Analysis
            # ================================================================
            with gr.Tab("📊 Distribution Analysis"):
                gr.Markdown("""
                Analyze volume distributions by sampling random configurations.

                **Note:** Vertices are added to fixed base (0, 1, ∞).
                So 4 random vertices = 7 total vertices (4 + 0, 1, ∞).
                """)

                with gr.Row():
                    with gr.Column():
                        dist_vertices = gr.Slider(1, 10, value=4, step=1,
                                                 label="Number of Random Vertices")
                        dist_samples = gr.Slider(100, 50000, value=5000, step=100,
                                                label="Number of Samples")
                        dist_seed = gr.Number(value=42, label="Random Seed")

                        dist_button = gr.Button("Analyze Distribution", variant="primary")

                    with gr.Column():
                        dist_output = gr.Markdown("Results will appear here...")
                        dist_plot = gr.Image(label="Distribution")

                dist_button.click(
                    analyze_volume_distribution,
                    inputs=[dist_vertices, dist_samples, dist_seed],
                    outputs=[dist_output, dist_plot]
                )

            # ================================================================
            # Tab 3: Fixed Combinatorics Optimization (Rivin-Delaunay)
            # ================================================================
            with gr.Tab("📐 Fixed Combinatorics"):
                gr.Markdown("""
                Optimize volume for a **fixed combinatorial triangulation** using the Rivin-Delaunay algorithm.

                This finds the unique maximal-volume configuration while preserving the combinatorial structure.
                All such maximal configurations are **arithmetic** (have integral holonomy traces).
                """)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Single Trial")
                        fc_n_points = gr.Slider(4, 50, value=10, step=1,
                                               label="Number of Random Points",
                                               info="Points on sphere (+ infinity)")
                        fc_seed = gr.Number(value=42, label="Random Seed")

                        fc_single_button = gr.Button("Run Single Optimization", variant="primary")

                        gr.Markdown("---")
                        gr.Markdown("### Batch Test (with Arithmeticity Check)")
                        fc_batch_trials = gr.Slider(1, 20, value=5, step=1,
                                                   label="Number of Trials")
                        fc_batch_button = gr.Button("Run Batch Test", variant="secondary")

                    with gr.Column():
                        fc_output = gr.Markdown("Results will appear here...")
                        fc_plot = gr.Image(label="Results")

                # State for storing optimization results
                fc_result_state = gr.State(None)

                def run_single_fc(n_points, seed):
                    summary, opt_data, vertices = optimize_fixed_combinatorics(
                        int(n_points), int(seed)
                    )
                    return summary, opt_data

                fc_single_button.click(
                    run_single_fc,
                    inputs=[fc_n_points, fc_seed],
                    outputs=[fc_output, fc_result_state]
                )

                fc_batch_button.click(
                    batch_fixed_combinatorics_test,
                    inputs=[fc_n_points, fc_batch_trials, fc_seed],
                    outputs=[fc_output, fc_plot]
                )

            # ================================================================
            # Tab 4: 3D Visualization
            # ================================================================
            with gr.Tab("🔮 Visualization"):
                gr.Markdown("Visualize polyhedra in different models")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Input Vertices")
                        gr.Markdown("Enter vertices as comma-separated values in the complex plane")

                        vis_real = gr.Textbox(
                            label="Real parts",
                            value="0, 1, 0, 0.5",
                            placeholder="0, 1, 0.5, -0.5"
                        )
                        vis_imag = gr.Textbox(
                            label="Imaginary parts",
                            value="0, 0, 1, 0.866",
                            placeholder="0, 0, 0.866, 0.866"
                        )

                        with gr.Row():
                            load_opt_button = gr.Button("Load from Optimization", size="sm")

                        gr.Markdown("### Visualization Options")
                        vis_type = gr.Radio(
                            ["Delaunay (2D)", "Klein Ball (3D)", "Poincaré Ball (3D)"],
                            value="Klein Ball (3D)",
                            label="Visualization Type"
                        )
                        vis_subdivisions = gr.Slider(0, 5, value=3, step=1,
                                                    label="Subdivision Level (3D only)",
                                                    info="Higher = smoother curves (slower rendering)")

                        vis_button = gr.Button("Generate Visualization", variant="primary")

                    with gr.Column():
                        vis_plot = gr.Plot(label="Visualization")
                        vis_status = gr.Textbox(label="Status", interactive=False)

                vis_button.click(
                    visualize_configuration,
                    inputs=[vis_real, vis_imag, vis_type, vis_subdivisions],
                    outputs=[vis_plot, vis_status]
                )

                load_opt_button.click(
                    load_from_optimization,
                    inputs=[opt_result_state],
                    outputs=[vis_real, vis_imag]
                )

            # ================================================================
            # Tab 5: Arithmeticity / Holonomy
            # ================================================================
            with gr.Tab("🔬 Arithmeticity"):
                gr.Markdown("Check if a polyhedron is arithmetic using Penner-Rivin holonomy")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Input Vertices")
                        arith_real = gr.Textbox(
                            label="Real parts",
                            value="0, 1, 0, 0.5",
                            placeholder="0, 1, 0.5, -0.5"
                        )
                        arith_imag = gr.Textbox(
                            label="Imaginary parts",
                            value="0, 0, 1, 0.866",
                            placeholder="0, 0, 0.866, 0.866"
                        )

                        with gr.Row():
                            load_opt_arith_button = gr.Button("Load from Optimization", size="sm")

                        arith_button = gr.Button("Compute Holonomy & Check Arithmeticity", variant="primary")
                        symmetry_button = gr.Button("Compute Symmetry Group", variant="secondary")

                    with gr.Column():
                        arith_output = gr.Markdown("Results will appear here...")
                        arith_plot = gr.Image(label="Trace Analysis")

                arith_button.click(
                    compute_holonomy_analysis,
                    inputs=[arith_real, arith_imag],
                    outputs=[arith_output, arith_plot]
                )

                symmetry_button.click(
                    compute_symmetry_analysis,
                    inputs=[arith_real, arith_imag],
                    outputs=[arith_output]
                )

                load_opt_arith_button.click(
                    load_from_optimization,
                    inputs=[opt_result_state],
                    outputs=[arith_real, arith_imag]
                )

                gr.Markdown("""
                ### About Arithmeticity

                A hyperbolic 3-manifold is **arithmetic** if its holonomy representation lies in PSL(2, O_K)
                where O_K is the ring of integers in a number field K.

                For ideal polyhedra, this can be tested by computing:
                1. Holonomy generators (Penner-Rivin algorithm)
                2. Traces of these generators
                3. Checking if traces are integers (or lie in a number field)

                **Arithmetic polyhedra have deep number-theoretic significance!**

                If all traces are integers, the configuration is arithmetic and related to
                special lattices in hyperbolic space.
                """)

            # ================================================================
            # Tab 6: About
            # ================================================================
            with gr.Tab("ℹ️ About"):
                gr.Markdown(f"""
                ## About This Tool

                This GUI provides an interactive interface to the **Ideal Polyhedron Volume Toolkit**.

                ### Features

                - **Optimization**: Find maximal volume configurations using differential evolution
                - **Distribution Analysis**: Sample random configurations and analyze volume distributions
                - **Fixed Combinatorics**: Optimize volume while keeping triangulation fixed (Rivin-Delaunay)
                - **3D Visualization**: View polyhedra in multiple models:
                  - Delaunay triangulation in complex plane (2D)
                  - Stereographic projection on unit sphere (3D)
                  - Poincaré ball model (3D hyperbolic geometry)
                - **Arithmeticity Testing**: Check if polyhedra have arithmetic holonomy (Penner-Rivin)

                ### Mathematical Background

                Ideal polyhedra are polyhedra in hyperbolic 3-space with all vertices at infinity.
                Their volumes can be computed using:
                - Delaunay triangulation of vertex positions
                - Lobachevsky's formula for ideal tetrahedra
                - Stereographic projection from the complex plane

                ### Usage Tips

                1. Start with **Optimization** to find interesting configurations
                2. Use **Load from Optimization** in the Visualization tab to see results
                3. Adjust **subdivision level** for smoother 3D visualizations
                4. Compare Klein ball and Poincaré ball models to understand hyperbolic geometry

                ### Performance

                - Volume scoring uses the vectorized SciPy Bloch–Wigner kernel
                - The GUI runs one worker to avoid process startup overhead
                - The command-line optimizer offers `--workers` for larger runs

                ### Documentation

                - See `README.md` for installation and Python API
                - See `bin/README.md` for command-line tools
                - See `examples/` for research scripts

                ---

                **Version:** {__version__}
                **License:** MIT
                """)

        gr.Markdown("---")
        gr.Markdown("*Ideal Polyhedron Volume Toolkit GUI*")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch Gradio GUI for Ideal Polyhedron Volume Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Launch locally on 127.0.0.1:7860
  %(prog)s --share            # Create shareable public link
  %(prog)s --port 8080        # Use custom port
  %(prog)s --share --port 8080  # Share with custom port
        """
    )

    parser.add_argument('--share', action='store_true',
                        help='Create a shareable public Gradio link')
    parser.add_argument('--port', type=int, default=7860,
                        help='Port to run the server on (default: 7860)')
    parser.add_argument('--server-name', type=str, default="127.0.0.1",
                        help='Server name/IP to bind to (default: 127.0.0.1)')

    args = parser.parse_args()

    demo = create_gui()

    print("=" * 70)
    print("🎨 Ideal Polyhedron Volume Toolkit - GUI")
    print("=" * 70)
    if args.share:
        print("Creating shareable public link...")
        print("⚠️  WARNING: Public links expose your local server to the internet")
    else:
        print(f"Launching local server at http://{args.server_name}:{args.port}")
    print("=" * 70)

    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.port,
        theme=APP_THEME,
    )
