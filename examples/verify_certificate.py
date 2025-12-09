#!/usr/bin/env python3
"""Verify that certificate points produce the correct Delaunay triangulation."""

import json
import numpy as np
from scipy.spatial import Delaunay
from collections import Counter

# Load complete package
with open('complete_challenge_package.json', 'r') as f:
    package = json.load(f)

# Get realizable challenge and certificate
challenge = package['challenge_1_realizable']
certificate_points = np.array(challenge['certificate_points'])
target_triangles = challenge['triangles']

print("="*70)
print("CERTIFICATE VERIFICATION")
print("="*70)

print(f"\nChallenge: {challenge['label']}")
print(f"  Target triangulation: {len(target_triangles)} triangles")
print(f"  Certificate points: {len(certificate_points)} points")

# Compute Delaunay of certificate points
print(f"\nComputing Delaunay triangulation of certificate points...")
tri = Delaunay(certificate_points)
computed_triangles = [tuple(simplex) for simplex in tri.simplices]

print(f"  Result: {len(computed_triangles)} triangles")

# Compare combinatorial structure (normalize for comparison)
def normalize_triangulation(triangles):
    """Sort triangles for comparison."""
    return sorted([tuple(sorted(t)) for t in triangles])

target_normalized = normalize_triangulation(target_triangles)
computed_normalized = normalize_triangulation(computed_triangles)

# Check if identical
if target_normalized == computed_normalized:
    print(f"\n✓✓✓ CERTIFICATE VERIFIED ✓✓✓")
    print(f"  The certificate points produce EXACTLY the target triangulation!")
else:
    print(f"\n✗ Certificate mismatch")
    print(f"  Target has {len(target_normalized)} triangles")
    print(f"  Computed has {len(computed_normalized)} triangles")

    # Find differences
    target_set = set(target_normalized)
    computed_set = set(computed_normalized)

    missing = target_set - computed_set
    extra = computed_set - target_set

    if missing:
        print(f"  Missing {len(missing)} triangles from target")
    if extra:
        print(f"  Extra {len(extra)} triangles not in target")

# Also verify structural validity
print(f"\nStructural validation of certificate triangulation:")
edge_count = Counter()
for t in computed_triangles:
    v0, v1, v2 = t
    for edge in [tuple(sorted([v0, v1])), tuple(sorted([v1, v2])), tuple(sorted([v2, v0]))]:
        edge_count[edge] += 1

max_count = max(edge_count.values()) if edge_count else 0
if max_count <= 2:
    print(f"  ✓ All edges in ≤2 triangles (max: {max_count})")
else:
    print(f"  ✗ Some edges in >2 triangles!")

print("="*70)
