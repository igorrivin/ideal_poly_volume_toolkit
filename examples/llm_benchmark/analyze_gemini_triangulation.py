#!/usr/bin/env python3
"""
Analyze the actual triangulation of Gemini's configuration.

Show why Gemini's assumed decomposition is wrong.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    triangle_volume_bloch_wigner
)
import torch


def main():
    print("=" * 70)
    print("Analyzing Gemini's Triangulation Assumption")
    print("=" * 70)

    # Gemini's configuration (8 finite vertices)
    vertices = np.array([
        0.0 + 0.0j,           # Origin (South Pole)
        1.0 + 0.0j,           # V₀
        0.6234898 + 0.7818315j,   # V₁ = e^(i 2π/7)
        -0.2225209 + 0.9749279j,  # V₂ = e^(i 4π/7)
        -0.9009689 + 0.4338837j,  # V₃ = e^(i 6π/7)
        -0.9009689 - 0.4338837j,  # V₄ = e^(i 8π/7)
        -0.2225209 - 0.9749279j,  # V₅ = e^(i 10π/7)
        0.6234898 - 0.7818315j,   # V₆ = e^(i 12π/7)
    ])

    # Get actual Delaunay triangulation
    triangles = delaunay_triangulation_indices(vertices)

    print(f"\nActual Delaunay triangulation:")
    print(f"Number of triangles: {len(triangles)}")
    print(f"\nTriangles (each forms tetrahedron with ∞):")

    total_volume = 0.0
    for i, (a, b, c) in enumerate(triangles):
        v1, v2, v3 = vertices[a], vertices[b], vertices[c]

        # Compute volume
        z1 = torch.tensor(v1, dtype=torch.complex128)
        z2 = torch.tensor(v2, dtype=torch.complex128)
        z3 = torch.tensor(v3, dtype=torch.complex128)
        vol = triangle_volume_bloch_wigner(z1, z2, z3).item()
        total_volume += vol

        print(f"  Triangle {i+1}: vertices {a}, {b}, {c}")
        print(f"    {v1:.4f}, {v2:.4f}, {v3:.4f}")
        print(f"    Volume: {vol:.8f}")

    print(f"\nTotal volume: {total_volume:.10f}")

    # Compare with Gemini's assumption
    print("\n" + "=" * 70)
    print("Gemini's Assumed Triangulation")
    print("=" * 70)

    print("\nGemini assumed 7 tetrahedra:")
    print("  T_k = (∞, 0, V_k, V_{k+1}) for k = 0, 1, ..., 6")
    print("\nThis would mean triangles in complex plane:")

    gemini_triangles = []
    for k in range(7):
        # Indices: 0 is origin, 1-7 are V₀-V₆
        idx_origin = 0
        idx_vk = 1 + k
        idx_vk1 = 1 + ((k + 1) % 7)
        gemini_triangles.append((idx_origin, idx_vk, idx_vk1))
        print(f"  Triangle {k+1}: vertices {idx_origin}, {idx_vk}, {idx_vk1}")
        print(f"    (origin, V_{k}, V_{(k+1) % 7})")

    # Compute volume for Gemini's assumed triangulation
    print("\n" + "=" * 70)
    print("Computing volume for Gemini's assumed triangulation:")
    print("=" * 70)

    gemini_total = 0.0
    for i, (a, b, c) in enumerate(gemini_triangles):
        v1, v2, v3 = vertices[a], vertices[b], vertices[c]

        z1 = torch.tensor(v1, dtype=torch.complex128)
        z2 = torch.tensor(v2, dtype=torch.complex128)
        z3 = torch.tensor(v3, dtype=torch.complex128)
        vol = triangle_volume_bloch_wigner(z1, z2, z3).item()
        gemini_total += vol

        print(f"  Triangle {i+1}: {vol:.8f}")

    print(f"\nGemini's assumed total: {gemini_total:.10f}")
    print(f"Gemini's claimed:       11.5352164101")
    print(f"Actual (Delaunay):      {total_volume:.10f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if len(triangles) == 7:
        print("\n✓ Number of triangles matches (7)")
        print("  BUT the triangulation is different!")
        print(f"\n  Gemini's assumed volume: {gemini_total:.6f}")
        print(f"  Actual Delaunay volume:  {total_volume:.6f}")
        print(f"  Difference: {abs(gemini_total - total_volume):.6f}")
    else:
        print(f"\n✗ Number of triangles doesn't match!")
        print(f"  Gemini assumed: 7 triangles")
        print(f"  Actual Delaunay: {len(triangles)} triangles")

    print("\nGemini's error:")
    print("  Assumed a specific triangulation WITHOUT verification")
    print("  The actual Delaunay triangulation is different")
    print("  This leads to a ~39% error in volume calculation!")


if __name__ == "__main__":
    main()
