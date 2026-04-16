#!/usr/bin/env python3
"""
Generate LLM challenge prompts from RANDOM points on S^3.

These polytopes have (generically) trivial automorphism group and no
exploitable symmetry, testing raw geometric problem-solving ability.

For each N, we sample random points on S^3, compute the convex hull,
verify full-dimensionality and simplicial boundary, then export the
face lattice as a challenge prompt.
"""

import json
import numpy as np
from pathlib import Path
from scipy.spatial import ConvexHull
from itertools import combinations
from collections import Counter


def random_S3_points(n, rng):
    """Sample n points uniformly on S^3."""
    pts = rng.standard_normal((n, 4))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def face_lattice_from_hull(pts):
    hull = ConvexHull(pts)
    edges = set()
    triangles = set()
    facets = set()
    for simplex in hull.simplices:
        s = tuple(sorted(simplex))
        facets.add(s)
        for tri in combinations(s, 3):
            triangles.add(tri)
        for edge in combinations(s, 2):
            edges.add(edge)
    return {
        'edges': sorted([[int(x) for x in e] for e in edges]),
        'triangles': sorted([[int(x) for x in t] for t in triangles]),
        'facets': sorted([[int(x) for x in f] for f in facets]),
        'f_vector': [len(hull.vertices), len(edges), len(triangles), len(facets)],
        'n_hull_vertices': len(hull.vertices),
    }


def degree_sequence(edges, n):
    deg = Counter()
    for e in edges:
        deg[e[0]] += 1
        deg[e[1]] += 1
    return sorted([deg.get(i, 0) for i in range(n)], reverse=True)


def automorphism_order(edges, n):
    try:
        import pynauty
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        g = pynauty.Graph(n)
        for v, nbrs in adj.items():
            g.connect_vertex(v, list(nbrs))
        _, gs1, gs2, _, _ = pynauty.autgrp(g)
        return int(gs1 * 10**gs2)
    except ImportError:
        return None


def build_prompt(N, fv, deg_seq, edges, facets):
    return f"""You are given the face lattice of a convex 4-polytope (a polytope in R^4) with {N} vertices.

**Face lattice data:**
- f-vector: {fv}  (vertices, edges, 2-faces, 3-faces)
- Degree sequence (vertex degrees in the 1-skeleton, descending): {deg_seq}
- Edge list (pairs of vertex indices 0..{N-1}):
  {json.dumps(edges)}
- Facets (3-faces, each a tetrahedron given as 4 vertex indices):
  {json.dumps(facets)}

**Your task:** Produce {N} points on the unit 3-sphere S^3 (i.e., each point has unit norm in R^4) whose convex hull has exactly this combinatorial type — the same face lattice up to relabeling of vertices.

**Output format:** Return ONLY a JSON object with a single key "points" containing a list of {N} points, each a list of 4 floats. Example format:
```json
{{"points": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], ...]}}
```

Do not include any other text outside the JSON block. Think carefully about the combinatorial structure before choosing coordinates."""


def main():
    base = Path(__file__).parent.parent
    out_dir = base / 'results' / 'data' / 'llm_challenges_random'
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # For each N, generate a few random instances and pick one with all
    # vertices on the hull (generic for random S^3 points in 4D)
    vertex_counts = [8, 10, 12, 13, 15, 20]
    n_instances_per_N = 3  # multiple instances to test consistency

    all_challenges = []

    for N in vertex_counts:
        for instance in range(n_instances_per_N):
            seed = 42 + N * 100 + instance
            local_rng = np.random.default_rng(seed)

            # Keep trying until all points are hull vertices
            for attempt in range(20):
                pts = random_S3_points(N, local_rng)
                fl = face_lattice_from_hull(pts)
                if fl['n_hull_vertices'] == N:
                    break
            else:
                print(f"  N={N} instance {instance}: couldn't get all points on hull, skipping")
                continue

            fv = fl['f_vector']
            euler = fv[0] - fv[1] + fv[2] - fv[3]
            assert euler == 0, f"Euler fails: {fv}"

            deg_seq = degree_sequence(fl['edges'], N)
            aut = automorphism_order(fl['edges'], N)

            tag = f"N{N}_r{instance}"

            challenge = {
                'tag': tag,
                'family': 'random',
                'n_vertices': N,
                'dimension': 4,
                'f_vector': fv,
                'degree_sequence': deg_seq,
                'automorphism_order': aut,
                'edges': fl['edges'],
                'triangles': fl['triangles'],
                'facets': fl['facets'],
                'seed': seed,
            }

            # Save ground truth separately
            ground_truth = {
                'tag': tag,
                'n_vertices': N,
                'points': pts.tolist(),
                'seed': seed,
            }

            with open(out_dir / f'face_lattice_{tag}.json', 'w') as f:
                json.dump(challenge, f, indent=2)

            with open(out_dir / f'ground_truth_{tag}.json', 'w') as f:
                json.dump(ground_truth, f, indent=2)

            prompt = build_prompt(N, fv, deg_seq, fl['edges'], fl['facets'])
            with open(out_dir / f'prompt_{tag}.txt', 'w') as f:
                f.write(prompt)

            all_challenges.append(challenge)

            print(f"  {tag:12s}  f={fv}  deg={deg_seq}  |Aut|={aut}")

    with open(out_dir / 'all_challenges.json', 'w') as f:
        json.dump(all_challenges, f, indent=2)

    print(f"\nGenerated {len(all_challenges)} random challenges in {out_dir}/")
    print(f"\nAutomorphism orders:")
    for c in all_challenges:
        print(f"  {c['tag']:12s}  |Aut|={c['automorphism_order']}")


if __name__ == '__main__':
    main()
