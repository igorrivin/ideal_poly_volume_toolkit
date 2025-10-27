#!/usr/bin/env python3
"""
Check Gemini's claimed ideal heptagonal bipyramid configuration.

Gemini claims volume ~11.5352 for this configuration.
Let's verify with our Bloch-Wigner computation.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay


def sphere_to_complex_stereographic(x, y, z):
    """
    Stereographic projection from unit sphere to complex plane.

    Projects from North Pole (0,0,1) onto the plane z=0.
    North Pole → ∞, South Pole → 0.

    Formula: w = (x + iy) / (1 - z)
    """
    if abs(z - 1.0) < 1e-10:
        return np.inf  # North Pole → ∞
    return complex(x, y) / (1.0 - z)


def main():
    print("=" * 70)
    print("Checking Gemini's Ideal Heptagonal Bipyramid")
    print("=" * 70)

    # Gemini's 9 points on unit sphere
    sphere_points = [
        (0.0, 0.0, 1.0),          # North Pole → ∞
        (0.0, 0.0, -1.0),         # South Pole → 0
        (1.0, 0.0, 0.0),          # V₀
        (0.62348980, 0.78183148, 0.0),   # V₁ = cos(2π/7), sin(2π/7)
        (-0.22252093, 0.97492791, 0.0),  # V₂ = cos(4π/7), sin(4π/7)
        (-0.90096887, 0.43388374, 0.0),  # V₃ = cos(6π/7), sin(6π/7)
        (-0.90096887, -0.43388374, 0.0), # V₄ = cos(8π/7), sin(8π/7)
        (-0.22252093, -0.97492791, 0.0), # V₅ = cos(10π/7), sin(10π/7)
        (0.62348980, -0.78183148, 0.0),  # V₆ = cos(12π/7), sin(12π/7)
    ]

    print("\nConverting sphere coordinates to complex plane...")
    print("(via stereographic projection from North Pole)")

    complex_points = []
    for i, (x, y, z) in enumerate(sphere_points):
        w = sphere_to_complex_stereographic(x, y, z)
        if np.isinf(w.real) or np.isinf(w.imag):
            print(f"  Point {i+1}: ({x:.6f}, {y:.6f}, {z:.6f}) → ∞ (excluded from finite list)")
        else:
            print(f"  Point {i+1}: ({x:.6f}, {y:.6f}, {z:.6f}) → {w.real:.6f} + {w.imag:.6f}i")
            complex_points.append(w)

    # Convert to numpy array
    vertices = np.array(complex_points)

    print(f"\nTotal finite vertices: {len(vertices)}")
    print("(∞ is implicit in our computation)")

    # Compute volume using Bloch-Wigner
    print("\n" + "=" * 70)
    print("Computing volume...")
    print("=" * 70)

    volume_bw = ideal_poly_volume_via_delaunay(vertices, use_bloch_wigner=True)

    print(f"\nBloch-Wigner volume: {volume_bw:.10f}")

    # Gemini's claim
    gemini_claim = 11.535216410109
    print(f"Gemini's claim:      {gemini_claim:.10f}")
    print(f"Difference:          {abs(volume_bw - gemini_claim):.10f}")
    print(f"Relative error:      {abs(volume_bw - gemini_claim) / gemini_claim * 100:.2f}%")

    # Our optimal 9-vertex
    our_optimal = 8.162538347504972
    print(f"\nOur optimal 9-vertex: {our_optimal:.10f}")
    print(f"Gemini higher by:     {gemini_claim - our_optimal:.10f} ({(gemini_claim/our_optimal - 1)*100:.1f}%)")

    # Check if this exceeds 8.15
    print(f"\nExceeds 8.15 threshold: {volume_bw > 8.15}")

    # Analyze the configuration
    print("\n" + "=" * 70)
    print("Configuration Analysis")
    print("=" * 70)

    print(f"\nFinite vertices:")
    for i, v in enumerate(vertices):
        print(f"  v{i}: {v.real:9.6f} + {v.imag:9.6f}i  (|v| = {abs(v):.6f})")

    # Check if 0 is included
    has_zero = any(abs(v) < 1e-6 for v in vertices)
    print(f"\nIncludes origin (0): {has_zero}")

    # Check for regularity (7 vertices on unit circle + origin)
    vertices_near_unit_circle = sum(1 for v in vertices if abs(abs(v) - 1.0) < 0.1)
    print(f"Vertices near |z|=1: {vertices_near_unit_circle}")

    return volume_bw


if __name__ == "__main__":
    main()
