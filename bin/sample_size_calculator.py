#!/usr/bin/env python3
"""
Calculate required sample size for estimating the mean volume to a desired precision.

Uses the standard error formula: SE = σ/√n
For 95% confidence interval: CI = ±1.96 * SE = ±1.96 * σ/√n

To achieve precision δ (half-width of CI):
  1.96 * σ/√n ≤ δ
  n ≥ (1.96 * σ / δ)²
"""

import argparse
import json
import numpy as np
from pathlib import Path


def calculate_required_samples(std_dev, precision, confidence=0.95):
    """
    Calculate required sample size.

    Args:
        std_dev: Standard deviation of the population (estimated from pilot)
        precision: Desired half-width of confidence interval (e.g., 0.01 for ±0.01)
        confidence: Confidence level (default: 0.95)

    Returns:
        Required sample size (rounded up)
    """
    # Z-score for confidence level (1.96 for 95%, 2.576 for 99%)
    if confidence == 0.95:
        z_score = 1.96
    elif confidence == 0.99:
        z_score = 2.576
    elif confidence == 0.90:
        z_score = 1.645
    else:
        # Use normal distribution approximation
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)

    n_required = (z_score * std_dev / precision) ** 2
    return int(np.ceil(n_required))


def analyze_pilot_data(pilot_file):
    """Analyze pilot data and recommend sample sizes."""
    with open(pilot_file, 'r') as f:
        data = json.load(f)

    n_pilot = data['metadata']['n_valid']
    mean = data['statistics']['mean']
    std = data['statistics']['std']

    print("=" * 70)
    print("PILOT DATA ANALYSIS")
    print("=" * 70)
    print(f"Pilot sample size: {n_pilot:,}")
    print(f"Estimated mean:    {mean:.6f}")
    print(f"Estimated std dev: {std:.6f}")
    print()

    # Current standard error
    current_se = std / np.sqrt(n_pilot)
    current_ci_half_width = 1.96 * current_se

    print(f"Current standard error:     {current_se:.6f}")
    print(f"Current 95% CI half-width:  ±{current_ci_half_width:.6f}")
    print(f"Current 95% CI:             [{mean - current_ci_half_width:.6f}, {mean + current_ci_half_width:.6f}]")
    print()

    # Calculate required samples for different precisions
    print("=" * 70)
    print("REQUIRED SAMPLE SIZES (95% Confidence)")
    print("=" * 70)

    precisions = [
        (0.1, "1 decimal place (±0.1)"),
        (0.05, "1.5 decimal places (±0.05)"),
        (0.01, "2 decimal places (±0.01)"),
        (0.005, "2.5 decimal places (±0.005)"),
        (0.001, "3 decimal places (±0.001)"),
    ]

    recommendations = []
    for precision, desc in precisions:
        n_req = calculate_required_samples(std, precision)
        ratio = n_req / n_pilot

        print(f"\n{desc}:")
        print(f"  Required samples: {n_req:,}")
        print(f"  Ratio to pilot:   {ratio:.1f}x")

        if n_req <= n_pilot:
            print(f"  ✓ Already achieved with pilot data!")
        elif n_req <= 50000:
            print(f"  → Feasible")
            recommendations.append((precision, desc, n_req))
        elif n_req <= 200000:
            print(f"  → Moderately expensive")
            recommendations.append((precision, desc, n_req))
        else:
            print(f"  → Very expensive")

    # Best recommendations
    if recommendations:
        print()
        print("=" * 70)
        print("RECOMMENDATIONS:")
        print("=" * 70)

        # Find the best feasible option (≤50k samples)
        feasible = [r for r in recommendations if r[2] <= 50000]
        if feasible:
            prec, desc, n = feasible[-1]  # Get the most precise feasible one
            print(f"\nBest feasible option: {desc}")
            print(f"  Run with --samples {n:,}")
            print(f"  This will give mean ± {prec}")

        # Show 2 decimal option regardless
        two_dec = [r for r in recommendations if r[0] == 0.01]
        if two_dec:
            prec, desc, n = two_dec[0]
            print(f"\nFor 2 decimal places:")
            print(f"  Run with --samples {n:,}")

    print()
    print("=" * 70)

    return {
        'pilot_n': n_pilot,
        'mean': mean,
        'std': std,
        'current_ci_half_width': current_ci_half_width,
        'recommendations': recommendations
    }


def main():
    parser = argparse.ArgumentParser(
        description='Calculate required sample size for mean estimation'
    )
    parser.add_argument('pilot_file', type=str,
                        help='Path to pilot data JSON file')
    parser.add_argument('--precision', type=float, default=None,
                        help='Desired precision (CI half-width), e.g., 0.01')
    parser.add_argument('--confidence', type=float, default=0.95,
                        help='Confidence level (default: 0.95)')

    args = parser.parse_args()

    if not Path(args.pilot_file).exists():
        print(f"Error: File not found: {args.pilot_file}")
        return 1

    # Analyze pilot data
    results = analyze_pilot_data(args.pilot_file)

    # If specific precision requested, calculate it
    if args.precision is not None:
        print(f"\nCustom calculation for precision ±{args.precision}:")
        n_req = calculate_required_samples(
            results['std'],
            args.precision,
            args.confidence
        )
        print(f"  Required samples: {n_req:,}")
        print(f"  Confidence level: {100*args.confidence:.0f}%")


if __name__ == '__main__':
    main()
