#!/usr/bin/env python3
"""Quick validation of the challenge for LLM testing."""

import json
from collections import Counter

# Load the challenge
with open('challenge_for_llm.json', 'r') as f:
    data = json.load(f)

challenge = data['challenge']
triangles = challenge['triangles']

print("="*70)
print("STRUCTURAL VALIDATION OF CHALLENGE")
print("="*70)
print(f"\nTriangulation size:")
print(f"  Vertices: {challenge['n_vertices']}")
print(f"  Triangles: {challenge['n_triangles']}")

# Check 1: Duplicates
print(f"\n1. Checking for duplicate triangles...")
tri_sorted = [tuple(sorted(t)) for t in triangles]
tri_counts = Counter(tri_sorted)
duplicates = [t for t, count in tri_counts.items() if count > 1]

if duplicates:
    print(f"   ✗ FOUND {len(duplicates)} DUPLICATE TRIANGLES:")
    for tri in duplicates[:5]:
        print(f"      {tri} appears {tri_counts[tri]} times")
else:
    print(f"   ✓ No duplicate triangles")

# Check 2: Edge counts
print(f"\n2. Checking edge counts...")
edge_count = Counter()
for t in triangles:
    v0, v1, v2 = t
    for edge in [tuple(sorted([v0, v1])), tuple(sorted([v1, v2])), tuple(sorted([v2, v0]))]:
        edge_count[edge] += 1

bad_edges = [(e, c) for e, c in edge_count.items() if c > 2]

if bad_edges:
    print(f"   ✗ FOUND {len(bad_edges)} EDGES IN >2 TRIANGLES:")
    for edge, count in sorted(bad_edges, key=lambda x: x[1], reverse=True)[:5]:
        print(f"      Edge {edge} in {count} triangles")
        # Show which triangles
        for i, t in enumerate(triangles):
            v0, v1, v2 = t
            edges = [tuple(sorted([v0, v1])), tuple(sorted([v1, v2])), tuple(sorted([v2, v0]))]
            if edge in edges:
                print(f"         Triangle {i}: {t}")
else:
    max_count = max(edge_count.values())
    boundary_edges = sum(1 for c in edge_count.values() if c == 1)
    interior_edges = sum(1 for c in edge_count.values() if c == 2)
    print(f"   ✓ All edges valid")
    print(f"      Boundary edges (count=1): {boundary_edges}")
    print(f"      Interior edges (count=2): {interior_edges}")
    print(f"      Total edges: {len(edge_count)}")
    print(f"      Max edge count: {max_count}")

# Check 3: Euler characteristic (sanity check)
print(f"\n3. Euler characteristic check...")
V = challenge['n_vertices']
E = len(edge_count)
F = challenge['n_triangles']
chi = V - E + F

print(f"   V = {V} vertices")
print(f"   E = {E} edges")
print(f"   F = {F} faces")
print(f"   χ = V - E + F = {chi}")

# For a planar graph with boundary: χ = 1
if chi == 1:
    print(f"   ✓ χ = 1 (consistent with planar triangulation with boundary)")
elif chi == 2:
    print(f"   ✓ χ = 2 (consistent with closed triangulated surface/sphere)")
else:
    print(f"   ⚠ Unexpected Euler characteristic")

print("\n" + "="*70)
if not duplicates and not bad_edges:
    print("✓✓✓ CHALLENGE IS STRUCTURALLY VALID ✓✓✓")
    print("\nReady for testing with GPT-5 or other LLMs!")
else:
    print("✗✗✗ CHALLENGE HAS STRUCTURAL ISSUES ✗✗✗")
print("="*70)
