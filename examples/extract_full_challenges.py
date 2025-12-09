#!/usr/bin/env python3
"""Extract both challenges (realizable + non-realizable) for LLM testing."""

import json

# Load the 150-vertex benchmark
with open('llm_benchmark_150v.json', 'r') as f:
    benchmark = json.load(f)

# Separate challenges
realizable = None
non_realizable = None

for c in benchmark['challenges']:
    if c['is_realizable']:
        realizable = c
    elif non_realizable is None:  # Get first non-realizable
        non_realizable = c

# Create complete challenge package
output = {
    "metadata": {
        "description": "LLM Geometric Reasoning Benchmark - Complete Challenge Package",
        "n_vertices": 150,
        "total_challenges": 2,
        "instructions": (
            "For each challenge, you are given a triangulation specified as a list of triangles. "
            "Each triangle is a tuple of three vertex indices (0-149). "
            "Your task: Either (1) produce a set of 2D points such that the Delaunay triangulation "
            "of those points has the same combinatorial structure as the given triangulation, "
            "OR (2) output 'None' if no such point set exists."
        )
    },
    "challenge_1_realizable": {
        "label": realizable['label'],
        "n_vertices": realizable['n_vertices'],
        "n_triangles": realizable['n_triangles'],
        "triangles": realizable['triangles'],
        "is_realizable": True,
        "solution_exists": True,
        "certificate_points": realizable['certificate_points'],
        "hint": "This triangulation IS Delaunay realizable. Certificate points are provided."
    },
    "challenge_2_non_realizable": {
        "label": non_realizable['label'],
        "n_vertices": non_realizable['n_vertices'],
        "n_triangles": non_realizable['n_triangles'],
        "triangles": non_realizable['triangles'],
        "is_realizable": False,
        "solution_exists": False,
        "hint": "This triangulation is NOT Delaunay realizable. Correct answer: 'None'."
    }
}

# Save complete package
with open('complete_challenge_package.json', 'w') as f:
    json.dump(output, f, indent=2)

print("="*70)
print("COMPLETE CHALLENGE PACKAGE CREATED")
print("="*70)
print(f"\nChallenge 1 (REALIZABLE):")
print(f"  Label: {realizable['label']}")
print(f"  Vertices: {realizable['n_vertices']}")
print(f"  Triangles: {realizable['n_triangles']}")
print(f"  Certificate points: {len(realizable['certificate_points'])} points")
print(f"  Expected answer: Provide the certificate points (or compute Delaunay)")

print(f"\nChallenge 2 (NON-REALIZABLE):")
print(f"  Label: {non_realizable['label']}")
print(f"  Vertices: {non_realizable['n_vertices']}")
print(f"  Triangles: {non_realizable['n_triangles']}")
print(f"  Expected answer: 'None'")

print(f"\nOutput file: complete_challenge_package.json")

# Also create a separate file with just the certificate points for reference
import numpy as np
certificate_array = np.array(realizable['certificate_points'])
np.save('certificate_points.npy', certificate_array)
print(f"Certificate points also saved to: certificate_points.npy")

print("="*70)
