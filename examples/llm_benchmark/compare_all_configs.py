#!/usr/bin/env python3
"""
Compare Gemini's configuration against our benchmark suite.

The embarrassing truth: Gemini's "sophisticated" regular configuration
performs worse than some random configurations!
"""

import json
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay


def load_benchmark_suite():
    """Load our benchmark test suite."""
    with open('test_data/llm_benchmark_test_suite.json', 'r') as f:
        return json.load(f)


def gemini_config_volume():
    """Compute Gemini's configuration volume."""
    vertices = np.array([
        0.0 + 0.0j,           # Origin
        1.0 + 0.0j,           # V₀
        0.6234898 + 0.7818315j,   # V₁
        -0.2225209 + 0.9749279j,  # V₂
        -0.9009689 + 0.4338837j,  # V₃
        -0.9009689 - 0.4338837j,  # V₄
        -0.2225209 - 0.9749279j,  # V₅
        0.6234898 - 0.7818315j,   # V₆
    ])
    return ideal_poly_volume_via_delaunay(vertices, use_bloch_wigner=True)


def main():
    print("=" * 70)
    print("Comparing ALL Configurations")
    print("=" * 70)

    # Load benchmark suite
    suite = load_benchmark_suite()

    # Get Gemini's volume
    gemini_volume = gemini_config_volume()

    # Extract volumes from benchmark
    optimal_volume = suite['configurations'][0]['expected_volume']
    random_volumes = [c['expected_volume'] for c in suite['configurations'][1:]]

    print("\nConfiguration Rankings (by volume):")
    print("─" * 70)

    # Create list of all configs
    all_configs = [
        ("Our Optimal 9-vertex", optimal_volume, True),
        ("Gemini's Heptagonal Bipyramid", gemini_volume, False),
    ] + [(f"Random {i+1}", v, False) for i, v in enumerate(random_volumes)]

    # Sort by volume
    all_configs.sort(key=lambda x: x[1], reverse=True)

    # Display ranking
    threshold = 8.15
    for rank, (name, volume, is_optimal) in enumerate(all_configs, 1):
        exceeds = "✓" if volume > threshold else "✗"
        star = "⭐" if is_optimal else "  "
        gemini_mark = "🤖" if "Gemini" in name else "  "

        print(f"{rank:2d}. {star}{gemini_mark} {name:35s} {volume:.6f} {exceeds}")

    print("─" * 70)
    print(f"Threshold: {threshold}")
    print("⭐ = Our optimized configuration")
    print("🤖 = Gemini's configuration")

    # Statistics
    print("\n" + "=" * 70)
    print("Statistics")
    print("=" * 70)

    # Where does Gemini rank?
    gemini_rank = next(i for i, (name, _, _) in enumerate(all_configs, 1) if "Gemini" in name)
    total_configs = len(all_configs)

    print(f"\nGemini's rank: {gemini_rank} out of {total_configs}")
    print(f"Percentile: {(total_configs - gemini_rank) / total_configs * 100:.1f}th percentile")

    # How many random configs beat Gemini?
    random_better = sum(1 for v in random_volumes if v > gemini_volume)
    print(f"\nRandom configs better than Gemini: {random_better} out of 9")

    if random_better > 0:
        print("\n🎉 Gemini's 'sophisticated' symmetric configuration")
        print("   is BEATEN by random configurations!")

    # Compare to optimal
    optimal_advantage = optimal_volume - gemini_volume
    print(f"\nOur optimal advantage over Gemini: {optimal_advantage:.6f} ({optimal_advantage/gemini_volume*100:.1f}%)")

    # Compare to best random
    best_random = max(random_volumes)
    best_random_advantage = best_random - gemini_volume
    print(f"Best random advantage over Gemini: {best_random_advantage:.6f} ({best_random_advantage/gemini_volume*100:.1f}%)")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    print("\nGemini's Configuration Performance:")
    print(f"  ✗ Below 8.15 threshold ({threshold - gemini_volume:.3f} short)")
    print(f"  ✗ Below our optimal ({optimal_advantage:.3f} lower)")
    if random_better > 0:
        print(f"  ✗ Below {random_better} random config(s) (max {best_random_advantage:.3f} lower)")
    print(f"  ✗ Rank: {gemini_rank}/{total_configs} (bottom {(gemini_rank/total_configs*100):.0f}%)")

    print("\nDespite:")
    print("  • Perfect mathematical reasoning")
    print("  • Symmetric, regular structure")
    print("  • Sophisticated geometric construction")
    print("  • Correct formulas and approach")

    print("\nGemini's configuration is:")
    print("  🎲 Just another mediocre random configuration!")
    print("  😅 Beaten by dumb luck!")


if __name__ == "__main__":
    main()
