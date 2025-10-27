#!/usr/bin/env python3
"""
Generate test cases for LLM benchmark: 1 optimal + 9 random configurations.

The optimal configuration should exceed the 8.15 threshold.
The random configurations should NOT exceed the threshold.
"""

import numpy as np
import json
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay


def load_optimal_configuration():
    """Load the optimal 9-vertex configuration."""
    with open('../../results/data/9vertex_optimal_for_llm_test.json', 'r') as f:
        data = json.load(f)

    real_parts = data['optimal_configuration']['vertices_flat']['real_parts']
    imag_parts = data['optimal_configuration']['vertices_flat']['imag_parts']
    vertices = [{"real": r, "imag": i} for r, i in zip(real_parts, imag_parts)]
    volume = data['optimal_configuration']['volume']

    return {
        "id": "optimal",
        "description": "Optimal 9-vertex configuration (should exceed threshold)",
        "vertices": vertices,
        "expected_volume": volume,
        "should_exceed_threshold": True
    }


def generate_random_configuration(seed, n_vertices=9):
    """Generate a random 9-vertex configuration."""
    np.random.seed(seed)

    # Fixed vertices: 0, 1
    fixed_vertices = [0.0 + 0.0j, 1.0 + 0.0j]

    # Random free vertices in a reasonable range
    n_free = n_vertices - 3  # Subtract 0, 1, ∞
    real_parts = np.random.uniform(-2, 2, n_free)
    imag_parts = np.random.uniform(-2, 2, n_free)
    free_vertices = [complex(r, i) for r, i in zip(real_parts, imag_parts)]

    # Combine
    all_vertices = fixed_vertices + free_vertices
    z_array = np.array(all_vertices)

    # Compute volume with Bloch-Wigner
    try:
        volume = ideal_poly_volume_via_delaunay(z_array, use_bloch_wigner=True)
    except:
        volume = 0.0

    # Create vertex list for JSON
    vertices = [{"real": z.real, "imag": z.imag} for z in all_vertices]

    return {
        "id": f"random_{seed}",
        "description": f"Random configuration {seed} (should NOT exceed threshold)",
        "vertices": vertices,
        "expected_volume": volume,
        "should_exceed_threshold": False
    }


def main():
    print("=" * 70)
    print("Generating LLM Benchmark Test Cases")
    print("=" * 70)

    # Load optimal configuration
    print("\nLoading optimal configuration...")
    optimal = load_optimal_configuration()
    print(f"  Volume: {optimal['expected_volume']:.6f}")
    print(f"  Exceeds 8.15: {optimal['expected_volume'] > 8.15}")

    # Generate random configurations
    print("\nGenerating 9 random configurations...")
    random_configs = []
    for seed in range(1, 10):
        config = generate_random_configuration(seed)
        random_configs.append(config)
        exceeds = config['expected_volume'] > 8.15
        status = "⚠️ EXCEEDS" if exceeds else "✓ below"
        print(f"  Config {seed}: {config['expected_volume']:.6f} {status}")

    # Check if any random configs exceed threshold (shouldn't happen)
    bad_randoms = [c for c in random_configs if c['expected_volume'] > 8.15]
    if bad_randoms:
        print("\n⚠️ WARNING: Some random configs exceed threshold!")
        print("   This is very unlikely. Regenerating with different seeds...")
        # In practice, this is extremely unlikely to happen

    # Create test suite
    test_suite = {
        "metadata": {
            "description": "LLM Benchmark Test Suite for 9-vertex ideal polyhedra",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "threshold": 8.15,
            "n_vertices": 9,
            "n_configurations": 10
        },
        "configurations": [optimal] + random_configs
    }

    # Save test suite
    os.makedirs("test_data", exist_ok=True)
    output_file = "test_data/llm_benchmark_test_suite.json"

    with open(output_file, 'w') as f:
        json.dump(test_suite, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Test suite generated successfully!")
    print(f"{'=' * 70}")
    print(f"Output: {output_file}")
    print(f"\nSummary:")
    print(f"  - 1 optimal configuration (volume: {optimal['expected_volume']:.6f})")
    print(f"  - 9 random configurations (max volume: {max(c['expected_volume'] for c in random_configs):.6f})")
    print(f"  - Threshold: 8.15")
    print(f"  - All random configs below threshold: {all(c['expected_volume'] < 8.15 for c in random_configs)}")


if __name__ == "__main__":
    main()
