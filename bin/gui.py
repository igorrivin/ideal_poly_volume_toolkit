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

from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    triangle_volume_from_points_torch,
    triangle_volume_bloch_wigner,
    ideal_poly_volume_via_delaunay,
)
from ideal_poly_volume_toolkit.visualization import (
    plot_polyhedron_klein,
    plot_polyhedron_poincare,
    plot_delaunay_2d,
    create_polyhedron_mesh,
)
from ideal_poly_volume_toolkit.rivin_holonomy import (
    Triangulation,
    generators_from_triangulation,
)
from ideal_poly_volume_toolkit.symmetry import (
    compute_symmetry_group,
    format_symmetry_report,
)

import torch
from scipy.optimize import differential_evolution

# Note: GPU is slower than CPU for this problem due to small tensor sizes
# and transfer overhead, so we use CPU explicitly
DEVICE = torch.device('cpu')


# ============================================================================
# Optimization Functions
# ============================================================================

def spherical_to_complex(theta, phi):
    """Convert spherical coordinates to complex via stereographic projection."""
    return np.tan(theta/2) * np.exp(1j * phi)


def compute_volume(params, n_vertices):
    """Compute volume for a polyhedron with n_vertices.

    Performance optimizations:
    - Reduced series_terms to 64 (good balance of speed/accuracy)
    - Single torch tensor conversion
    - Parallel evaluation via differential_evolution workers

    Args:
        params: Optimization parameters (theta, phi pairs)
        n_vertices: Total number of vertices

    Returns:
        Negative volume (for minimization)

    Note:
        The polishing step (L-BFGS-B) uses finite differences for gradients.
        PyTorch autodiff could be used but the Delaunay triangulation is
        not differentiable, making gradients unreliable near topology changes.
    """
    complex_points = [complex(0, 0), complex(1, 0)]
    n_params = n_vertices - 3  # n_vertices includes 0, 1, ∞, plus (n_vertices-3) free

    for i in range(n_params):
        theta = params[2*i]
        phi = params[2*i + 1]
        z = spherical_to_complex(theta, phi)
        complex_points.append(z)

    Z_np = np.array(complex_points, dtype=np.complex128)

    try:
        idx = delaunay_triangulation_indices(Z_np)
    except:
        return 1000.0

    # Single torch conversion (CPU is faster than GPU for small tensors)
    Z_torch = torch.tensor(Z_np, dtype=torch.complex128, device=DEVICE)

    total_volume = 0
    for (i, j, k) in idx:
        try:
            # Use Bloch-Wigner dilogarithm (faster and more accurate than Lobachevsky series)
            vol = triangle_volume_bloch_wigner(
                Z_torch[i], Z_torch[j], Z_torch[k]
            )
            total_volume += vol.item()
        except:
            return 1000.0

    return -total_volume


def run_optimization(n_vertices, n_trials, max_iter, pop_size, seed, progress=gr.Progress()):
    """Run optimization with progress tracking.

    Uses parallel workers (all CPU cores) for faster optimization.
    """
    import os
    from functools import partial

    # Validate and convert inputs
    try:
        n_vertices = int(n_vertices)
        if n_vertices < 4:
            return "Error: Number of vertices must be at least 4", None
        if n_vertices > 100:
            return "Error: Number of vertices limited to 100 for practical computation time", None
    except (ValueError, TypeError):
        return "Error: Number of vertices must be an integer", None

    n_cpus = os.cpu_count()

    # Limit workers to avoid "too many open files" error
    # Each worker opens file descriptors for IPC
    # Conservative limit: use at most 32 workers
    max_workers = min(32, n_cpus) if n_cpus else 1

    progress(0, desc=f"Starting optimization (using {max_workers} workers)...")

    n_free_vertices = n_vertices - 3
    n_params = n_free_vertices * 2
    bounds = [(0.1, np.pi - 0.1), (0, 2*np.pi)] * n_free_vertices

    # Adaptive settings for large vertex counts
    # For high dimensions, reduce popsize to avoid excessive evaluations
    adaptive_popsize = min(pop_size, max(10, 15 - (n_vertices - 7) // 3))

    best_volume = 0
    best_params = None
    all_volumes = []

    # Create picklable objective function using partial
    # (lambdas can't be pickled for multiprocessing)
    objective_func = partial(compute_volume, n_vertices=n_vertices)

    for trial in range(n_trials):
        progress((trial + 1) / n_trials, desc=f"Trial {trial + 1}/{n_trials}")

        result = differential_evolution(
            objective_func,
            bounds,
            maxiter=max_iter,
            popsize=adaptive_popsize,
            seed=seed + trial,
            polish=True,
            disp=False,
            workers=max_workers,  # Limited to avoid file descriptor exhaustion
            updating='deferred'  # Better for parallel workers
        )

        volume = -result.fun
        all_volumes.append(volume)

        if volume > best_volume:
            best_volume = volume
            best_params = result.x

    # Reconstruct best configuration
    complex_points = [complex(0, 0), complex(1, 0), complex(0, 1)]
    for i in range(n_free_vertices):
        theta = best_params[2*i]
        phi = best_params[2*i + 1]
        z = spherical_to_complex(theta, phi)
        complex_points.append(z)

    Z_np = np.array(complex_points, dtype=np.complex128)
    idx = delaunay_triangulation_indices(Z_np)

    # Compute statistics
    stats = {
        'n_vertices': n_vertices,
        'n_faces': len(idx),
        'best_volume': float(best_volume),
        'mean_volume': float(np.mean(all_volumes)),
        'std_volume': float(np.std(all_volumes)),
        'all_volumes': [float(v) for v in all_volumes],
    }

    # Save result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/data/{n_vertices}vertex_optimization_{timestamp}.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {'timestamp': datetime.now().isoformat()},
            'best': {
                'volume': stats['best_volume'],
                'params': best_params.tolist(),
                'vertices_real': Z_np.real.tolist(),
                'vertices_imag': Z_np.imag.tolist(),
            },
            'statistics': stats,
        }, f, indent=2)

    # Create summary text
    summary = f"""
## Optimization Results

**Configuration:**
- Vertices: {n_vertices}
- Trials: {n_trials}
- Iterations per trial: {max_iter}

**Best Result:**
- Volume: {best_volume:.8f}
- Faces: {len(idx)}

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
        'stats': stats
    }

    return summary, opt_data


# ============================================================================
# Distribution Analysis Functions
# ============================================================================

def analyze_volume_distribution(n_vertices, n_samples, seed, progress=gr.Progress()):
    """Analyze volume distribution with progress tracking.

    Note: n_vertices is the number of random vertices to add.
    Total vertices = n_vertices + 2 fixed (0, 1) + 1 at infinity = n_vertices + 3
    """
    np.random.seed(seed)

    # Fixed vertices: 0, 1 (infinity is implicit in the volume computation)
    fixed_vertices = [complex(0, 0), complex(1, 0)]
    n_random = n_vertices
    total_vertices = n_vertices + 3  # 0, 1, ∞ + n_vertices random

    volumes = []

    def sample_random_vertex():
        vec = np.random.randn(3)
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
            vol = ideal_poly_volume_via_delaunay(vertices_np, mode='fast', series_terms=96)
            if 0 < vol < 1000:
                volumes.append(vol)
        except:
            pass

    volumes = np.array(volumes)

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
- **Total vertices: {total_vertices}** (including ∞)
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
# Gradio Interface
# ============================================================================

def create_gui():
    """Create the main Gradio interface."""

    with gr.Blocks(title="Ideal Polyhedron Volume Toolkit", theme=gr.themes.Soft()) as demo:
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
                        opt_popsize = gr.Slider(10, 30, value=15, step=5,
                                               label="Population Size")
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
            # Tab 3: 3D Visualization
            # ================================================================
            with gr.Tab("🔮 3D Visualization"):
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
            # Tab 4: Arithmeticity / Holonomy
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
            # Tab 5: About
            # ================================================================
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## About This Tool

                This GUI provides an interactive interface to the **Ideal Polyhedron Volume Toolkit**.

                ### Features

                - **Optimization**: Find maximal volume configurations using differential evolution
                - **Distribution Analysis**: Sample random configurations and analyze volume distributions
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

                - **Parallel optimization**: Uses up to 32 CPU cores automatically
                - **7-vertex**: ~10 seconds for 10 trials
                - **20-vertex**: ~2-3 minutes for 10 trials

                **Note**: Worker limit set to 32 to avoid file descriptor exhaustion.
                To use more cores, increase system limit: `ulimit -n 4096`

                ### Documentation

                - See `README.md` for installation and Python API
                - See `bin/README.md` for command-line tools
                - See `examples/` for research scripts

                ---

                **Version:** 0.3.0
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
        server_port=args.port
    )
