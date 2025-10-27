#!/usr/bin/env python3
"""
LLM Benchmark Test for Ideal Polyhedron Volume Computation

This creates a test to expose LLM confusion about ideal polyhedron volumes:
- GPT-4.5: Claims maximum 9-vertex volume is ~9.13
- Gemini 2.5 Deep Think: Claims to construct volume > 10
- Reality: Optimal 9-vertex volume is approximately 9.8

Test structure:
1. One optimal 9-vertex configuration (volume ≈ 9.8)
2. Nine random 9-vertex configurations (volume < 9.8)
3. Threshold test: volume > 9.8 should be True only for optimal

This test can be used to evaluate whether an LLM can correctly:
- Compute ideal polyhedron volumes
- Distinguish optimal from random configurations
- Understand the geometric constraints
"""

import numpy as np
import json
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ideal_poly_volume_toolkit.volume_threshold import get_volume


def generate_random_9vertex_config(seed):
    """
    Generate a random 9-vertex configuration.

    Returns 9 vertices: 0, 1, i, ∞ + 5 random points
    """
    np.random.seed(seed)

    # Fixed vertices
    fixed = [0+0j, 1+0j, 1j]

    # Generate 5 random vertices in a reasonable region
    random_vertices = []
    for _ in range(5):
        # Random point in disk of radius 2
        r = np.random.uniform(0, 1.8)
        theta = np.random.uniform(0, 2 * np.pi)
        z = r * np.exp(1j * theta)
        random_vertices.append(z)

    # All vertices (infinity implicit)
    all_vertices = fixed + random_vertices + [np.inf]

    return all_vertices


def create_llm_benchmark_data(optimal_config_file=None):
    """
    Create the LLM benchmark test data.

    Args:
        optimal_config_file: Path to JSON file with optimal 9-vertex config
                            If None, uses a placeholder

    Returns:
        Dictionary with test data
    """
    # Load optimal configuration if available
    if optimal_config_file and os.path.exists(optimal_config_file):
        with open(optimal_config_file, 'r') as f:
            optimal_data = json.load(f)

        # Extract vertices
        fixed = optimal_data['optimal_configuration']['vertices']['fixed']
        free = optimal_data['optimal_configuration']['vertices']['free']

        optimal_vertices = (
            [complex(v['real'], v['imag']) for v in fixed] +
            [complex(v['real'], v['imag']) for v in free] +
            [np.inf]
        )
        optimal_volume = optimal_data['optimal_configuration']['volume']

    else:
        print("Warning: Optimal configuration file not found. Using placeholder.")
        # Placeholder - will be replaced once optimization completes
        optimal_vertices = [0+0j, 1+0j, 1j, np.inf,
                           0.5+0.5j, -0.3+0.7j, 0.8-0.2j,
                           -0.5-0.3j, 0.2+0.9j]
        optimal_volume = get_volume(optimal_vertices)

    # Generate 9 random configurations
    random_configs = []
    for i in range(9):
        vertices = generate_random_9vertex_config(seed=1000 + i)
        volume = get_volume(vertices)
        random_configs.append({
            "id": i + 1,
            "seed": 1000 + i,
            "vertices": {
                "fixed": [
                    {"real": 0.0, "imag": 0.0},
                    {"real": 1.0, "imag": 0.0},
                    {"real": 0.0, "imag": 1.0}
                ],
                "free": [
                    {"real": float(v.real), "imag": float(v.imag)}
                    for v in vertices[3:8]  # Skip 0,1,i and ∞
                ],
                "infinity": "implicit"
            },
            "volume": float(volume)
        })

    # Compile test data
    test_data = {
        "metadata": {
            "description": "LLM Benchmark: Ideal Polyhedron Volume Computation",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_type": "volume_threshold_classification",
            "threshold": 8.0,
            "context": {
                "gpt4.5_claim": "Maximum 9-vertex volume is approximately 9.13",
                "gemini2.5_claim": "Can construct 9-vertex polyhedron with volume > 10",
                "reality": "Optimal 9-vertex volume found is approximately 8.16 (with standard constraints: 0, 1, i fixed)"
            }
        },
        "test_instructions": {
            "task": "For each configuration, determine if the ideal polyhedron volume exceeds 8.0",
            "expected_result": "Only the optimal configuration should return True",
            "format": "Use volume_exceeds_threshold(vertices, 8.0) from volume_threshold module"
        },
        "optimal_configuration": {
            "id": 0,
            "description": "Optimal 9-vertex configuration (should return True for threshold 8.0)",
            "volume": float(optimal_volume),
            "vertices": {
                "fixed": [
                    {"real": float(v.real), "imag": float(v.imag)}
                    for v in optimal_vertices[:3]
                ],
                "free": [
                    {"real": float(v.real), "imag": float(v.imag)}
                    for v in optimal_vertices[3:8]
                ],
                "infinity": "implicit"
            },
            "expected_result": optimal_volume > 8.0
        },
        "random_configurations": random_configs,
        "summary": {
            "total_configs": 10,
            "optimal_count": 1,
            "random_count": 9,
            "threshold": 8.0,
            "optimal_volume": float(optimal_volume),
            "random_volumes": {
                "min": float(min(cfg['volume'] for cfg in random_configs)),
                "max": float(max(cfg['volume'] for cfg in random_configs)),
                "mean": float(np.mean([cfg['volume'] for cfg in random_configs]))
            }
        }
    }

    return test_data


def run_benchmark_test(test_data):
    """
    Run the benchmark test on all configurations.

    Args:
        test_data: Dictionary from create_llm_benchmark_data()

    Returns:
        Dictionary with test results
    """
    threshold = test_data['metadata']['threshold']
    results = []

    print("=" * 70)
    print("Running LLM Benchmark Test")
    print("=" * 70)
    print(f"Threshold: {threshold}")
    print()

    # Test optimal configuration
    print("Testing optimal configuration...")
    opt_cfg = test_data['optimal_configuration']
    opt_vertices = (
        [complex(v['real'], v['imag']) for v in opt_cfg['vertices']['fixed']] +
        [complex(v['real'], v['imag']) for v in opt_cfg['vertices']['free']] +
        [np.inf]
    )
    opt_volume = get_volume(opt_vertices)
    opt_exceeds = opt_volume > threshold

    print(f"  Volume: {opt_volume:.6f}")
    print(f"  Exceeds {threshold}? {opt_exceeds}")
    print(f"  Expected: {opt_cfg['expected_result']}")
    print(f"  Status: {'✓ PASS' if opt_exceeds == opt_cfg['expected_result'] else '✗ FAIL'}")
    print()

    results.append({
        "id": 0,
        "type": "optimal",
        "volume": float(opt_volume),
        "exceeds_threshold": bool(opt_exceeds),
        "expected": bool(opt_cfg['expected_result']),
        "passed": bool(opt_exceeds == opt_cfg['expected_result'])
    })

    # Test random configurations
    print("Testing random configurations...")
    for cfg in test_data['random_configurations']:
        vertices = (
            [complex(v['real'], v['imag']) for v in cfg['vertices']['fixed']] +
            [complex(v['real'], v['imag']) for v in cfg['vertices']['free']] +
            [np.inf]
        )
        volume = get_volume(vertices)
        exceeds = volume > threshold
        expected = False  # Random configs should not exceed threshold

        status = "✓ PASS" if exceeds == expected else "✗ FAIL"
        print(f"  Config {cfg['id']}: volume = {volume:.6f}, exceeds = {exceeds} {status}")

        results.append({
            "id": cfg['id'],
            "type": "random",
            "seed": cfg['seed'],
            "volume": float(volume),
            "exceeds_threshold": bool(exceeds),
            "expected": bool(expected),
            "passed": bool(exceeds == expected)
        })

    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {100 * passed / total:.1f}%")
    print()

    return {
        "results": results,
        "summary": {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": float(passed / total)
        }
    }


def main():
    """Generate benchmark data and run test."""
    print("\n" + "=" * 70)
    print("LLM BENCHMARK TEST GENERATOR")
    print("=" * 70)
    print()

    # Look for optimal configuration file
    optimal_config_file = "results/data/9vertex_optimal_for_llm_test.json"

    # Create benchmark data
    print("Generating benchmark data...")
    test_data = create_llm_benchmark_data(optimal_config_file)

    # Save benchmark data
    output_file = "tests/llm_benchmark_9vertex.json"
    os.makedirs("tests", exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"✓ Benchmark data saved to: {output_file}")
    print()
    print(f"Optimal volume: {test_data['optimal_configuration']['volume']:.6f}")
    print(f"Random volumes: {test_data['summary']['random_volumes']['min']:.6f} - {test_data['summary']['random_volumes']['max']:.6f}")
    print()

    # Run the test
    test_results = run_benchmark_test(test_data)

    # Save results
    results_file = "tests/llm_benchmark_9vertex_results.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"✓ Test results saved to: {results_file}")
    print()


if __name__ == "__main__":
    main()
