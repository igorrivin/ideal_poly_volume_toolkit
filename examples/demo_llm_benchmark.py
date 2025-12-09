#!/usr/bin/env python3
"""
Demo: How to use the LLM Geometric Reasoning Benchmark.

This script demonstrates:
1. Loading the benchmark
2. Extracting a challenge
3. Checking responses (both correct and incorrect examples)
"""

import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from check_llm_response import check_response


def demo():
    print()
    print("="*70)
    print("LLM GEOMETRIC REASONING BENCHMARK - DEMO")
    print("="*70)
    print()

    # Load benchmark
    benchmark_path = Path(__file__).parent / "test_benchmark.json"

    if not benchmark_path.exists():
        print(f"Error: Benchmark file not found: {benchmark_path}")
        print("Run: python examples/generate_llm_benchmark.py first")
        return 1

    with open(benchmark_path, 'r') as f:
        benchmark = json.load(f)

    print(f"Loaded benchmark: {benchmark_path}")
    print(f"  Challenges: {len(benchmark['challenges'])}")
    print(f"  Vertices: {benchmark['metadata']['n_vertices']}")
    print()

    # Find a non-realizable challenge
    non_realizable_idx = None
    realizable_idx = None

    for i, challenge in enumerate(benchmark['challenges']):
        if not challenge['solution_exists']:
            non_realizable_idx = i
        else:
            realizable_idx = i

    # Demo 1: Correct response for non-realizable
    if non_realizable_idx is not None:
        print("="*70)
        print(f"DEMO 1: Correct answer for non-realizable challenge")
        print("="*70)
        print()

        challenge = benchmark['challenges'][non_realizable_idx]
        triangulation = [tuple(tri) for tri in challenge['triangles']]

        print(f"Challenge {non_realizable_idx} ({challenge['label']}):")
        print(f"  Vertices: {challenge['n_vertices']}")
        print(f"  Triangles: {challenge['n_triangles']}")
        print()
        print("LLM Response: None")
        print()

        result = check_response(triangulation, None, verbose=True)

        print()
        input("Press Enter to continue...")
        print()

    # Demo 2: Incorrect response for non-realizable (giving random points)
    if non_realizable_idx is not None:
        print("="*70)
        print(f"DEMO 2: Incorrect answer (random points for non-realizable)")
        print("="*70)
        print()

        challenge = benchmark['challenges'][non_realizable_idx]
        triangulation = [tuple(tri) for tri in challenge['triangles']]

        # Generate random points (wrong answer!)
        np.random.seed(999)
        wrong_points = np.random.rand(challenge['n_vertices'], 2)

        print(f"Challenge {non_realizable_idx} ({challenge['label']}):")
        print(f"  Vertices: {challenge['n_vertices']}")
        print(f"  Triangles: {challenge['n_triangles']}")
        print()
        print(f"LLM Response: Random point set (shape {wrong_points.shape})")
        print()

        result = check_response(triangulation, wrong_points, verbose=True)

        print()
        input("Press Enter to continue...")
        print()

    # Demo 3: Wrong answer for realizable (saying None)
    if realizable_idx is not None:
        print("="*70)
        print(f"DEMO 3: Incorrect answer (None for realizable)")
        print("="*70)
        print()

        challenge = benchmark['challenges'][realizable_idx]
        triangulation = [tuple(tri) for tri in challenge['triangles']]

        print(f"Challenge {realizable_idx} ({challenge['label']}):")
        print(f"  Vertices: {challenge['n_vertices']}")
        print(f"  Triangles: {challenge['n_triangles']}")
        print(f"  Actually realizable: {challenge['solution_exists']}")
        print()
        print("LLM Response: None (wrong!)")
        print()

        result = check_response(triangulation, None, verbose=True)

        print()

    print("="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print()
    print("Key takeaways:")
    print("1. Checker verifies realizability using Rivin's LP constraints")
    print("2. Checker uses pynauty for robust isomorphism checking")
    print("3. Vertex relabeling is handled automatically")
    print("4. Both false positives and false negatives are detected")
    print()
    print("Try it yourself:")
    print("  python examples/check_llm_response.py <benchmark.json> <challenge_idx> --points <None|file.npy>")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(demo())
