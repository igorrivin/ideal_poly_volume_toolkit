#!/usr/bin/env python3
"""
Optimize volume for a large random configuration and analyze arithmetic angles.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import json
from datetime import datetime
from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay
import torch
from scipy.spatial import Delaunay
from fractions import Fraction


def continued_fraction_convergents(x, max_terms=20):
    """Compute convergents of continued fraction expansion."""
    convergents = []
    a = []
    remainder = x

    for _ in range(max_terms):
        floor_val = int(np.floor(remainder))
        a.append(floor_val)
        if abs(remainder - floor_val) < 1e-12:
            break
        remainder = 1.0 / (remainder - floor_val)

    p_prev, p_curr = 0, 1
    q_prev, q_curr = 1, 0

    for ai in a:
        p_next = ai * p_curr + p_prev
        q_next = ai * q_curr + q_prev
        convergents.append((p_next, q_next))
        p_prev, p_curr = p_curr, p_next
        q_prev, q_curr = q_curr, q_next

    return convergents


def optimize_volume(vertices_complex, n_trials=20, max_steps=2000):
    """Optimize volume starting from initial configuration."""

    best_volume = -np.inf
    best_config = None

    print(f"\nOptimizing with {n_trials} trials, {max_steps} steps each...")

    for trial in range(n_trials):
        # Start from initial configuration with small random perturbation
        real_init = vertices_complex.real + np.random.randn(len(vertices_complex)) * 0.1
        imag_init = vertices_complex.imag + np.random.randn(len(vertices_complex)) * 0.1

        # Use parameter array directly (no complex numbers in torch optimization)
        params = torch.tensor(np.concatenate([real_init, imag_init]),
                             dtype=torch.float64, requires_grad=True)

        # Optimizer
        optimizer = torch.optim.Adam([params], lr=0.05)

        for step in range(max_steps):
            optimizer.zero_grad()

            # Extract real and imag from params
            n_verts = len(real_init)
            real_t = params[:n_verts]
            imag_t = params[n_verts:]

            # Compute volume using numpy (convert back temporarily)
            with torch.no_grad():
                vertices_np = real_t.numpy() + 1j * imag_t.numpy()
                volume_np = ideal_poly_volume_via_delaunay(
                    torch.tensor(vertices_np, dtype=torch.complex128)
                )
                volume_np_val = volume_np.item() if isinstance(volume_np, torch.Tensor) else volume_np

            # Now compute with grad for backward
            vertices_torch = torch.complex(real_t, imag_t)
            volume = ideal_poly_volume_via_delaunay(vertices_torch)

            # Maximize volume (minimize negative volume)
            if isinstance(volume, torch.Tensor):
                loss = -volume
                loss.backward()
                optimizer.step()
                vol_val = volume.item()
            else:
                # If returns float, use numerical gradient
                vol_val = volume
                break

            if step % 500 == 0:
                print(f"  Trial {trial+1}/{n_trials}, Step {step}/{max_steps}, Volume: {vol_val:.6f}")

        # Get final volume
        with torch.no_grad():
            real_final = params[:n_verts].numpy()
            imag_final = params[n_verts:].numpy()
            vertices_final = real_final + 1j * imag_final
            final_vol = ideal_poly_volume_via_delaunay(
                torch.tensor(vertices_final, dtype=torch.complex128)
            )
            final_volume = final_vol.item() if isinstance(final_vol, torch.Tensor) else final_vol

        if final_volume > best_volume:
            best_volume = final_volume
            best_config = {
                'vertices_real': real_final.tolist(),
                'vertices_imag': imag_final.tolist(),
                'volume': final_volume,
            }
            print(f"  ★ New best volume: {best_volume:.6f}")

    return best_config


def analyze_dihedral_angles(vertices_complex):
    """Analyze dihedral angles and check for rational patterns."""
    from ideal_poly_volume_toolkit.rivin_delaunay import check_delaunay_realizability, build_edge_adjacency

    points = np.column_stack([vertices_complex.real, vertices_complex.imag])

    # Compute Delaunay triangulation
    tri = Delaunay(points)
    triangulation = [tuple(sorted(simplex)) for simplex in tri.simplices]
    triangulation = sorted(set(triangulation))

    # Get angles from Rivin LP
    result = check_delaunay_realizability(triangulation, verbose=False, strict=False)

    if not result['realizable']:
        print("ERROR: Configuration not realizable!")
        return None

    angles_scaled = result['angles']
    angles_radians = angles_scaled * np.pi
    n_triangles = len(triangulation)
    angles_array = angles_radians.reshape((n_triangles, 3))

    # Compute interior edge dihedral angles
    edge_adjacency = build_edge_adjacency(triangulation)
    dihedrals = []

    for edge, opposite_corners in sorted(edge_adjacency.items()):
        if len(opposite_corners) == 2:
            angle1 = angles_array[opposite_corners[0][0], opposite_corners[0][1]]
            angle2 = angles_array[opposite_corners[1][0], opposite_corners[1][1]]
            dihedral = angle1 + angle2
            normalized = dihedral / np.pi

            # Find best rational approximation
            convergents = continued_fraction_convergents(normalized)
            if convergents:
                best_p, best_q = convergents[-1]
                error = abs(normalized - best_p / best_q)

                dihedrals.append({
                    'edge': edge,
                    'angle_rad': float(dihedral),
                    'angle_deg': float(np.degrees(dihedral)),
                    'normalized': float(normalized),
                    'p': int(best_p),
                    'q': int(best_q),
                    'error': float(error),
                })

    return {
        'n_triangles': n_triangles,
        'n_interior_edges': len(dihedrals),
        'dihedrals': dihedrals,
    }


def main(n_vertices=89, n_trials=20, max_steps=2000):
    """Main optimization and analysis."""

    print(f"═══════════════════════════════════════════════════════════════")
    print(f"LARGE RANDOM CONFIGURATION OPTIMIZATION")
    print(f"═══════════════════════════════════════════════════════════════")
    print(f"\nNumber of vertices: {n_vertices}")
    print(f"Optimization trials: {n_trials}")
    print(f"Steps per trial: {max_steps}")

    # Generate random points in unit disk
    print(f"\nGenerating {n_vertices} random points in unit disk...")
    np.random.seed(42)  # For reproducibility

    # Generate points in polar coordinates for better distribution
    radii = np.sqrt(np.random.uniform(0, 1, n_vertices))
    angles = np.random.uniform(0, 2*np.pi, n_vertices)

    vertices_complex = radii * np.exp(1j * angles)

    print(f"Initial configuration generated")
    init_vol = ideal_poly_volume_via_delaunay(torch.tensor(vertices_complex, dtype=torch.complex128))
    init_vol_val = init_vol.item() if isinstance(init_vol, torch.Tensor) else init_vol
    print(f"Initial volume: {init_vol_val:.6f}")

    # Optimize
    best_config = optimize_volume(vertices_complex, n_trials=n_trials, max_steps=max_steps)

    print(f"\n{'─'*63}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'─'*63}")
    print(f"Best volume: {best_config['volume']:.12f}")

    # Analyze dihedral angles
    print(f"\n{'─'*63}")
    print(f"ANALYZING DIHEDRAL ANGLES")
    print(f"{'─'*63}")

    vertices_opt = np.array(best_config['vertices_real']) + 1j * np.array(best_config['vertices_imag'])
    angle_analysis = analyze_dihedral_angles(vertices_opt)

    if angle_analysis is None:
        return

    print(f"\nTriangles: {angle_analysis['n_triangles']}")
    print(f"Interior edges: {angle_analysis['n_interior_edges']}")

    # Find denominators with small error
    max_denominator = 200
    small_error_threshold = 1e-6

    denominators = {}
    for d in angle_analysis['dihedrals']:
        if d['q'] <= max_denominator and d['error'] < small_error_threshold:
            if d['q'] not in denominators:
                denominators[d['q']] = 0
            denominators[d['q']] += 1

    print(f"\n{'─'*63}")
    print(f"RATIONAL ANGLE DENOMINATORS (q ≤ {max_denominator}, error < {small_error_threshold})")
    print(f"{'─'*63}")

    if denominators:
        print(f"\n{'Denominator':>12} {'Count':>8} {'Relation to n':>20}")
        print(f"{'-'*42}")
        for q in sorted(denominators.keys()):
            count = denominators[q]
            relation = ""
            if q == n_vertices - 2:
                relation = f"= n-2"
            elif q == n_vertices - 3:
                relation = f"= n-3"
            elif q == n_vertices - 1:
                relation = f"= n-1"
            elif q == n_vertices:
                relation = f"= n"
            print(f"  q={q:>3}  {count:7d}  {relation:>20}")

        # Check if ALL angles have small denominators
        total_with_small_q = sum(denominators.values())
        if total_with_small_q == angle_analysis['n_interior_edges']:
            print(f"\n✓ ALL {angle_analysis['n_interior_edges']} interior edges have rational angles with q ≤ {max_denominator}!")
        else:
            print(f"\n  {total_with_small_q}/{angle_analysis['n_interior_edges']} edges have rational angles")
    else:
        print("  No angles found with small denominators")

    # Sample a few angles
    print(f"\n{'─'*63}")
    print(f"SAMPLE DIHEDRAL ANGLES (first 10)")
    print(f"{'─'*63}")
    print(f"{'Edge':>12} {'Degrees':>10} {'Rational':>12} {'Error':>12}")
    print(f"{'-'*48}")
    for i, d in enumerate(angle_analysis['dihedrals'][:10]):
        if d['q'] > 1:
            rational = f"{d['p']}π/{d['q']}"
        else:
            rational = f"{d['p']}π"
        print(f"  {str(d['edge']):>10} {d['angle_deg']:9.3f}° {rational:>12} {d['error']:11.2e}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"results/data/{n_vertices}vertex_random_optimization_{timestamp}.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'n_vertices': n_vertices,
            'n_trials': n_trials,
            'max_steps': max_steps,
            'timestamp': timestamp,
        },
        'best': best_config,
        'angle_analysis': {
            'n_triangles': angle_analysis['n_triangles'],
            'n_interior_edges': angle_analysis['n_interior_edges'],
            'denominator_counts': {str(k): v for k, v in denominators.items()},
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Optimize large random configuration')
    parser.add_argument('--vertices', type=int, default=89, help='Number of vertices')
    parser.add_argument('--trials', type=int, default=20, help='Number of optimization trials')
    parser.add_argument('--steps', type=int, default=2000, help='Steps per trial')
    args = parser.parse_args()

    main(n_vertices=args.vertices, n_trials=args.trials, max_steps=args.steps)
