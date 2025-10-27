#!/usr/bin/env python3
"""
LLM Benchmark Driver: Test volume threshold predictions.

This script loads test configurations and checks which ones exceed
the volume threshold. Designed to expose LLM confusion about volumes.

Usage:
    python run_benchmark.py
"""

import json
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay


def load_test_suite(filepath="test_data/llm_benchmark_test_suite.json"):
    """Load the test suite from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def run_test(config, threshold):
    """
    Run a single test case.

    Args:
        config: Configuration dict with vertices and expected results
        threshold: Volume threshold to test against

    Returns:
        dict: Test results
    """
    # Extract vertices and convert to complex numpy array
    vertices = np.array([complex(v['real'], v['imag']) for v in config['vertices']])

    # Compute volume using Bloch-Wigner (most accurate)
    actual_volume = ideal_poly_volume_via_delaunay(vertices, use_bloch_wigner=True)

    # Check if volume exceeds threshold
    exceeds = actual_volume > threshold

    # Expected result
    expected = config['should_exceed_threshold']

    return {
        "id": config['id'],
        "description": config['description'],
        "actual_volume": actual_volume,
        "expected_volume": config.get('expected_volume', None),
        "threshold": threshold,
        "exceeds_threshold": exceeds,
        "expected_to_exceed": expected,
        "passed": exceeds == expected
    }


def main():
    print("=" * 70)
    print("LLM BENCHMARK: Ideal Polyhedron Volume Test")
    print("=" * 70)

    # Load test suite
    print("\nLoading test suite...")
    test_suite = load_test_suite()
    threshold = test_suite['metadata']['threshold']
    n_configs = test_suite['metadata']['n_configurations']

    print(f"  Threshold: {threshold}")
    print(f"  Configurations: {n_configs}")

    # Run tests
    print(f"\n{'=' * 70}")
    print("Running tests...")
    print(f"{'=' * 70}\n")

    results = []
    for config in test_suite['configurations']:
        result = run_test(config, threshold)
        results.append(result)

        # Print result
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        exceeds_str = "EXCEEDS" if result['exceeds_threshold'] else "below"

        print(f"{status} | {result['id']:15s} | Volume: {result['actual_volume']:.6f} | {exceeds_str} threshold")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    n_passed = sum(1 for r in results if r['passed'])
    n_failed = sum(1 for r in results if not r['passed'])

    print(f"Total tests:  {len(results)}")
    print(f"Passed:       {n_passed}")
    print(f"Failed:       {n_failed}")

    if n_failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {n_failed} test(s) failed!")

    # Detailed results
    optimal_result = results[0]
    random_results = results[1:]

    print(f"\nOptimal configuration:")
    print(f"  Volume: {optimal_result['actual_volume']:.8f}")
    print(f"  Exceeds {threshold}: {optimal_result['exceeds_threshold']}")

    print(f"\nRandom configurations:")
    print(f"  Max volume: {max(r['actual_volume'] for r in random_results):.8f}")
    print(f"  Min volume: {min(r['actual_volume'] for r in random_results):.8f}")
    print(f"  Any exceed {threshold}: {any(r['exceeds_threshold'] for r in random_results)}")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
