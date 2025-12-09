#!/usr/bin/env python3
"""
Checker for LLM geometric reasoning benchmark responses.

Given:
1. A target triangulation (from the benchmark)
2. LLM's proposed answer (either point set or "None")

Verify:
- If LLM said "None": Check that triangulation is indeed non-realizable
- If LLM gave points: Check that Delaunay triangulation is isomorphic to target

Uses pynauty for robust graph isomorphism checking.
"""

import numpy as np
import json
import sys
from pathlib import Path
from scipy.spatial import Delaunay
from typing import Optional, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pynauty
except ImportError:
    print("Error: pynauty not installed. Install with: pip install pynauty")
    sys.exit(1)


def compute_canonical_hash(triangles: List[Tuple[int, int, int]]) -> str:
    """
    Compute canonical hash of triangulation using pynauty.

    This gives us a way to check if two triangulations are isomorphic,
    regardless of vertex labeling.

    Args:
        triangles: List of triangles as (v0, v1, v2) tuples

    Returns:
        Canonical hash string
    """
    # Get all vertices
    vertices = set()
    for tri in triangles:
        vertices.update(tri)

    n_vertices = len(vertices)

    # Create vertex mapping
    vertex_list = sorted(vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}

    # Build adjacency sets (graph representation)
    adjacency = {i: set() for i in range(n_vertices)}

    for v0, v1, v2 in triangles:
        i0 = vertex_to_idx[v0]
        i1 = vertex_to_idx[v1]
        i2 = vertex_to_idx[v2]

        # Add edges
        adjacency[i0].add(i1)
        adjacency[i0].add(i2)
        adjacency[i1].add(i0)
        adjacency[i1].add(i2)
        adjacency[i2].add(i0)
        adjacency[i2].add(i1)

    # Create pynauty graph
    g = pynauty.Graph(number_of_vertices=n_vertices, directed=False, adjacency_dict=adjacency)

    # Compute canonical labeling
    canonical_label = pynauty.canon_label(g)

    # Return as string (hashable)
    return str(canonical_label)


def check_response(
    target_triangulation: List[Tuple[int, int, int]],
    llm_response: Optional[np.ndarray],
    verbose: bool = True
) -> Dict:
    """
    Check if LLM's response is correct.

    Args:
        target_triangulation: The triangulation from the benchmark
        llm_response: Either None (LLM says impossible) or np.ndarray of points
        verbose: If True, print diagnostic info

    Returns:
        Dict with:
        - 'correct': bool, whether LLM answer is correct
        - 'reason': str, explanation
        - 'details': dict with additional info
    """
    if verbose:
        print("="*70)
        print("CHECKING LLM RESPONSE")
        print("="*70)
        print()

    # Compute canonical hash of target
    target_hash = compute_canonical_hash(target_triangulation)

    if verbose:
        print(f"Target triangulation:")
        print(f"  Vertices: {len(set(v for tri in target_triangulation for v in tri))}")
        print(f"  Triangles: {len(target_triangulation)}")
        print(f"  Canonical hash: {target_hash[:50]}...")
        print()

    if llm_response is None:
        # LLM claims triangulation is not realizable
        if verbose:
            print("LLM response: None (claims triangulation is not realizable)")
            print()
            print("Verifying claim by checking Rivin constraints...")

        from ideal_poly_volume_toolkit.rivin_delaunay import check_delaunay_realizability

        result = check_delaunay_realizability(target_triangulation, verbose=False)

        if not result['realizable']:
            # LLM correctly identified non-realizable triangulation
            if verbose:
                print("  ✓ Confirmed: Triangulation is NOT realizable")
                print()

            return {
                'correct': True,
                'reason': 'LLM correctly identified non-realizable triangulation',
                'details': {
                    'llm_said': 'None',
                    'actual_realizable': False,
                    'lp_result': result,
                },
            }
        else:
            # LLM incorrectly said it's not realizable
            if verbose:
                print("  ✗ ERROR: Triangulation IS realizable!")
                print(f"     Min angle: {np.degrees(result['min_angle_radians']):.2f}°")
                print()

            return {
                'correct': False,
                'reason': 'LLM incorrectly claimed triangulation is not realizable',
                'details': {
                    'llm_said': 'None',
                    'actual_realizable': True,
                    'lp_result': result,
                },
            }

    # LLM provided a point set
    if verbose:
        print(f"LLM response: Point set with {llm_response.shape[0]} vertices")
        print()

    # Check dimensions
    if llm_response.ndim != 2 or llm_response.shape[1] != 2:
        if verbose:
            print(f"  ✗ ERROR: Expected shape (n, 2), got {llm_response.shape}")
            print()

        return {
            'correct': False,
            'reason': f'Invalid point set shape: {llm_response.shape}',
            'details': {'llm_said': 'points', 'error': 'invalid_shape'},
        }

    n_target_vertices = len(set(v for tri in target_triangulation for v in tri))
    if llm_response.shape[0] != n_target_vertices:
        if verbose:
            print(f"  ✗ ERROR: Expected {n_target_vertices} vertices, got {llm_response.shape[0]}")
            print()

        return {
            'correct': False,
            'reason': f'Wrong number of vertices: expected {n_target_vertices}, got {llm_response.shape[0]}',
            'details': {
                'llm_said': 'points',
                'expected_vertices': n_target_vertices,
                'got_vertices': llm_response.shape[0],
            },
        }

    # Compute Delaunay triangulation of LLM's points
    if verbose:
        print("Computing Delaunay triangulation of proposed points...")

    try:
        tri = Delaunay(llm_response)
        llm_triangulation = [tuple(simplex) for simplex in tri.simplices]

        if verbose:
            print(f"  Triangulation: {len(llm_triangulation)} triangles")
            print()

    except Exception as e:
        if verbose:
            print(f"  ✗ ERROR: Could not compute Delaunay triangulation: {e}")
            print()

        return {
            'correct': False,
            'reason': f'Delaunay triangulation failed: {e}',
            'details': {'llm_said': 'points', 'error': 'delaunay_failed'},
        }

    # Compute canonical hash of LLM's triangulation
    llm_hash = compute_canonical_hash(llm_triangulation)

    if verbose:
        print("Checking combinatorial equivalence (graph isomorphism)...")
        print(f"  Target hash: {target_hash[:50]}...")
        print(f"  LLM hash:    {llm_hash[:50]}...")
        print()

    # Check if hashes match
    if target_hash == llm_hash:
        if verbose:
            print("  ✓ SUCCESS: Triangulations are isomorphic!")
            print("  LLM provided a valid point set with correct combinatorics")
            print()

        return {
            'correct': True,
            'reason': 'LLM provided valid point set with correct combinatorics',
            'details': {
                'llm_said': 'points',
                'isomorphic': True,
                'target_triangles': len(target_triangulation),
                'llm_triangles': len(llm_triangulation),
            },
        }
    else:
        if verbose:
            print("  ✗ INCORRECT: Triangulations are NOT isomorphic")
            print(f"     Target: {len(target_triangulation)} triangles")
            print(f"     LLM:    {len(llm_triangulation)} triangles")
            print()

        return {
            'correct': False,
            'reason': 'Point set produces different combinatorial structure',
            'details': {
                'llm_said': 'points',
                'isomorphic': False,
                'target_triangles': len(target_triangulation),
                'llm_triangles': len(llm_triangulation),
            },
        }


def load_benchmark(filepath: str) -> Dict:
    """Load benchmark JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Check LLM response for geometric reasoning benchmark"
    )
    parser.add_argument(
        "benchmark",
        type=str,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "challenge_idx",
        type=int,
        help="Challenge index (0-based)",
    )
    parser.add_argument(
        "--points",
        type=str,
        default=None,
        help="Path to NPY file with proposed points, or 'None' if claiming non-realizable",
    )

    args = parser.parse_args()

    # Load benchmark
    benchmark = load_benchmark(args.benchmark)

    if args.challenge_idx < 0 or args.challenge_idx >= len(benchmark['challenges']):
        print(f"Error: Invalid challenge index {args.challenge_idx}")
        print(f"Valid range: 0 to {len(benchmark['challenges'])-1}")
        return 1

    challenge = benchmark['challenges'][args.challenge_idx]

    print()
    print("#"*70)
    print("# LLM Response Checker")
    print("#"*70)
    print()
    print(f"Challenge: {challenge['label']}")
    print()

    # Load LLM response
    if args.points is None or args.points.lower() == 'none':
        llm_response = None
    else:
        try:
            llm_response = np.load(args.points)
        except Exception as e:
            print(f"Error loading points file: {e}")
            return 1

    # Check response
    target_triangulation = [tuple(tri) for tri in challenge['triangles']]
    result = check_response(target_triangulation, llm_response, verbose=True)

    # Print summary
    print("="*70)
    print("RESULT")
    print("="*70)
    if result['correct']:
        print("✓ CORRECT")
    else:
        print("✗ INCORRECT")
    print()
    print(f"Reason: {result['reason']}")
    print("="*70)

    return 0 if result['correct'] else 1


if __name__ == "__main__":
    sys.exit(main())
