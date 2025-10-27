#!/usr/bin/env python3
"""
Find what angle gives Cl₂(θ) ≈ 1.6479 (Gemini's claimed value).
"""

import numpy as np
import mpmath as mp
from scipy.optimize import minimize_scalar


def clausen_function(theta):
    """Compute Clausen function Cl₂(θ)."""
    mp.mp.dps = 50
    return float(mp.im(mp.polylog(2, mp.exp(1j * theta))))


def main():
    print("=" * 70)
    print("Finding Gemini's Mystery Angle")
    print("=" * 70)

    gemini_claim = 1.647888058587

    print(f"\nGemini claimed: Cl₂(2π/7) ≈ {gemini_claim}")

    # Compute actual Cl₂(2π/7)
    theta_2pi7 = 2 * np.pi / 7
    cl2_2pi7 = clausen_function(theta_2pi7)

    print(f"Actual Cl₂(2π/7) = {cl2_2pi7:.10f}")
    print(f"Ratio: {gemini_claim / cl2_2pi7:.6f}")

    # Find what angle gives ~1.6479
    print(f"\nSearching for θ such that Cl₂(θ) ≈ {gemini_claim}...")

    def objective(theta):
        return abs(clausen_function(theta) - gemini_claim)

    result = minimize_scalar(objective, bounds=(0, 2*np.pi), method='bounded')
    theta_found = result.x

    print(f"\nFound: θ ≈ {theta_found:.10f} radians")
    print(f"       θ ≈ {theta_found * 180 / np.pi:.6f} degrees")
    print(f"       Cl₂(θ) = {clausen_function(theta_found):.10f}")

    # Check common multiples
    print("\n" + "=" * 70)
    print("Checking common angles:")
    print("=" * 70)

    angles = {
        "2π/7": 2 * np.pi / 7,
        "4π/7": 4 * np.pi / 7,
        "6π/7": 6 * np.pi / 7,
        "8π/7": 8 * np.pi / 7,
        "π": np.pi,
        "2π/3": 2 * np.pi / 3,
        "4π/3": 4 * np.pi / 3,
    }

    for name, angle in angles.items():
        cl2_value = clausen_function(angle)
        match = "✓" if abs(cl2_value - gemini_claim) < 0.001 else " "
        print(f"  {match} Cl₂({name:6s}) = {cl2_value:.10f}")

    # Check if Gemini confused Cl₂ with something else
    print("\n" + "=" * 70)
    print("Possible Confusion:")
    print("=" * 70)

    # Maybe Gemini computed the Clausen function incorrectly?
    # Or used a different normalization?

    # The maximum of Cl₂(θ) occurs at θ = π
    max_angle = np.pi
    max_value = clausen_function(max_angle)
    print(f"\nMaximum value: Cl₂(π) = {max_value:.10f}")

    # Perhaps Gemini computed something like θ × Cl₂(θ)?
    print(f"\n2π/7 × Cl₂(2π/7) = {theta_2pi7 * cl2_2pi7:.10f}")

    # Or perhaps summed multiple terms?
    print(f"7 × Cl₂(2π/7) = {7 * cl2_2pi7:.10f}  (This is Gemini's final answer!)")

    print("\n" + "=" * 70)
    print("HYPOTHESIS:")
    print("=" * 70)

    print("\nGemini claimed each tetrahedron has volume Cl₂(2π/7) ≈ 1.6479")
    print(f"But the actual Cl₂(2π/7) = {cl2_2pi7:.6f}")
    print(f"\nGemini's final volume: 7 × 1.6479 ≈ 11.5352")
    print(f"Our measured volume:   7 × {cl2_2pi7:.6f} ≈ {7 * cl2_2pi7:.6f}")

    print("\nPossible error:")
    print("  Gemini may have cited an INCORRECT reference value for Cl₂(2π/7)")
    print("  Or computed it using a buggy implementation")
    print("  Or confused it with a different function")


if __name__ == "__main__":
    main()
