#!/usr/bin/env python3
"""
LLM 4-polytope reconstruction benchmark.

Given a face lattice (from our optimal ideal 4-polytopes), ask an LLM to
produce coordinates of a 4-polytope with that combinatorial type.
Then verify the answer.

Usage:
    python scripts/llm_polytope_benchmark.py [--n 8] [--model o3] [--all]
"""

import argparse
import json
import re
import time
import numpy as np
from pathlib import Path
from scipy.spatial import ConvexHull
from itertools import combinations


# ---------------------------------------------------------------------------
# Verification (self-contained, no imports from verify_llm_polytope.py)
# ---------------------------------------------------------------------------

def face_lattice_from_points(pts):
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
        'f_vector': [len(pts), len(edges), len(triangles), len(facets)],
        'edges': sorted(edges),
        'facets': sorted(facets),
    }


def degree_sequence_from_edges(n, edges):
    from collections import Counter
    deg = Counter()
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return sorted([deg.get(i, 0) for i in range(n)], reverse=True)


def check_graph_isomorphism(n, edges1, edges2):
    """Check 1-skeleton isomorphism via pynauty, networkx, or weak fallback."""
    try:
        import pynauty
        def to_nauty(n, edges):
            adj = {i: set() for i in range(n)}
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            g = pynauty.Graph(n)
            for v in range(n):
                g.connect_vertex(v, sorted(adj[v]))
            return g
        return pynauty.certificate(to_nauty(n, edges1)) == pynauty.certificate(to_nauty(n, edges2))
    except ImportError:
        pass
    try:
        import networkx as nx
        G1 = nx.Graph(); G1.add_edges_from(edges1)
        G2 = nx.Graph(); G2.add_edges_from(edges2)
        return nx.is_isomorphic(G1, G2)
    except ImportError:
        pass
    return (degree_sequence_from_edges(n, edges1) ==
            degree_sequence_from_edges(n, edges2) and
            len(edges1) == len(edges2))


def verify_response(challenge, pts):
    """
    Verify LLM-produced points against challenge.
    Returns (pass: bool, details: str).
    """
    N = challenge['n_vertices']
    f_expected = challenge['f_vector']
    details = []

    if pts.shape != (N, 4):
        return False, f"Wrong shape: expected ({N},4), got {pts.shape}"

    # Check inscribed (all points on a common sphere)
    norms = np.linalg.norm(pts, axis=1)
    if np.allclose(norms, norms[0], rtol=1e-4):
        # Normalize to unit sphere for downstream checks
        pts = pts / norms[:, None]
        inscribed = True
        details.append(f"inscribed (r={norms[0]:.4f})")
    else:
        inscribed = False
        details.append(f"NOT inscribed (norms range [{norms.min():.4f}, {norms.max():.4f}])")

    rank = np.linalg.matrix_rank(pts - pts[0], tol=1e-8)
    if rank < 4:
        return False, f"Not full-dimensional: rank={rank}"
    details.append(f"rank={rank}")

    lattice = face_lattice_from_points(pts)
    f_got = lattice['f_vector']
    if f_got != f_expected:
        return False, f"f-vector mismatch: got {f_got}, expected {f_expected}"
    details.append(f"f={f_got}")

    edges_expected = [tuple(e) for e in challenge['edges']]
    edges_got = list(lattice['edges'])
    iso = check_graph_isomorphism(N, edges_expected, edges_got)
    if not iso:
        ds_exp = degree_sequence_from_edges(N, edges_expected)
        ds_got = degree_sequence_from_edges(N, edges_got)
        return False, f"1-skeleton not isomorphic (deg seq: expected {ds_exp}, got {ds_got})"
    details.append("1-skeleton ≅ ✓")

    if not inscribed:
        return False, ", ".join(details) + " [FAIL: not inscribed]"

    return True, ", ".join(details)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(challenge):
    N = challenge['n_vertices']
    fv = challenge['f_vector']
    edges = challenge['edges']
    facets = challenge['facets']
    deg_seq = challenge['degree_sequence']

    prompt = f"""You are given the face lattice of a convex 4-polytope (a polytope in R^4) with {N} vertices.

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

    return prompt


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_gpt(prompt, model="gpt-5.4-pro", reasoning_effort="xhigh"):
    """Call OpenAI Responses API with the given prompt."""
    from openai import OpenAI
    client = OpenAI()

    print(f"  Calling {model} (reasoning={reasoning_effort})...")
    t0 = time.time()

    response = client.responses.create(
        model=model,
        input=prompt,
        reasoning={'effort': reasoning_effort},
    )

    elapsed = time.time() - t0
    content = response.output_text
    usage = response.usage

    in_tok = usage.input_tokens
    out_tok = usage.output_tokens
    reason_tok = usage.output_tokens_details.reasoning_tokens

    print(f"  Response in {elapsed:.1f}s  "
          f"(tokens: {in_tok} in, {out_tok} out, {reason_tok} reasoning)")

    return content, {
        'model': model,
        'reasoning_effort': reasoning_effort,
        'elapsed_s': elapsed,
        'input_tokens': in_tok,
        'output_tokens': out_tok,
        'reasoning_tokens': reason_tok,
        'total_tokens': in_tok + out_tok,
    }


def parse_points_from_response(text):
    """Extract points array from LLM response text."""
    # Try to find JSON block
    # Look for ```json ... ``` first
    m = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            return np.array(data['points'], dtype=float)
        except (json.JSONDecodeError, KeyError):
            pass

    # Try to find raw JSON with "points" key
    m = re.search(r'\{[^{}]*"points"\s*:\s*\[.*?\]\s*\}', text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            return np.array(data['points'], dtype=float)
        except (json.JSONDecodeError, KeyError):
            pass

    # Try to parse the entire response as JSON
    try:
        data = json.loads(text.strip())
        if 'points' in data:
            return np.array(data['points'], dtype=float)
    except json.JSONDecodeError:
        pass

    # Last resort: find all bracketed number lists
    lists = re.findall(r'\[([0-9eE.+\-,\s]+)\]', text)
    points = []
    for lst in lists:
        nums = [float(x.strip()) for x in lst.split(',') if x.strip()]
        if len(nums) == 4:
            points.append(nums)
    if points:
        return np.array(points, dtype=float)

    raise ValueError("Could not parse points from LLM response")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(n_values, model, challenges_dir, results_dir,
                  reasoning_effort="xhigh"):
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for N in n_values:
        challenge_path = challenges_dir / f'face_lattice_N{N}.json'
        if not challenge_path.exists():
            print(f"N={N}: challenge file not found, skipping")
            continue

        with open(challenge_path) as f:
            challenge = json.load(f)

        print(f"\n{'='*60}")
        print(f"N={N}  f={challenge['f_vector']}  aut={challenge.get('automorphism_order')}")
        print(f"{'='*60}")

        prompt = build_prompt(challenge)

        try:
            raw_response, usage_info = call_gpt(prompt, model=model,
                                                   reasoning_effort=reasoning_effort)
        except Exception as e:
            print(f"  API error: {e}")
            all_results.append({
                'n_vertices': N, 'model': model, 'pass': False,
                'error': str(e), 'raw_response': None,
            })
            continue

        # Save raw response
        raw_path = results_dir / f'raw_response_N{N}_{model}.txt'
        with open(raw_path, 'w') as f:
            f.write(raw_response)

        try:
            pts = parse_points_from_response(raw_response)
            print(f"  Parsed {pts.shape[0]} points in R^{pts.shape[1]}")
        except ValueError as e:
            print(f"  PARSE FAIL: {e}")
            all_results.append({
                'n_vertices': N, 'model': model, 'pass': False,
                'error': f'parse: {e}', 'raw_response': raw_response[:500],
            })
            continue

        passed, details = verify_response(challenge, pts)
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {details}")

        result = {
            'n_vertices': N,
            'model': model,
            'pass': passed,
            'details': details,
            'f_vector_expected': challenge['f_vector'],
            'points': pts.tolist() if passed else None,
            **usage_info,
        }
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY ({model})")
    print(f"{'='*60}")
    n_pass = sum(1 for r in all_results if r['pass'])
    n_total = len(all_results)
    for r in all_results:
        status = "✓" if r['pass'] else "✗"
        print(f"  N={r['n_vertices']:3d}  {status}  {r.get('details', r.get('error', ''))}")
    print(f"\n  {n_pass}/{n_total} passed")

    # Save results
    summary_path = results_dir / f'benchmark_{model}.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {summary_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="LLM 4-polytope benchmark")
    parser.add_argument('--n', type=int, nargs='+', default=None,
                        help='Vertex counts to test (default: 5,6,7,8,9,10)')
    parser.add_argument('--model', type=str, default='gpt-5.4-pro',
                        help='OpenAI model (default: gpt-5.4-pro)')
    parser.add_argument('--reasoning', type=str, default='xhigh',
                        help='Reasoning effort: low/medium/high/xhigh (default: xhigh)')
    parser.add_argument('--all', action='store_true',
                        help='Test all available N values')
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    challenges_dir = base / 'results' / 'data' / 'llm_challenges'
    results_dir = base / 'results' / 'data' / 'llm_benchmark'

    if args.all:
        # Find all available challenge files
        n_values = sorted([
            int(p.stem.split('N')[1])
            for p in challenges_dir.glob('face_lattice_N*.json')
        ])
    elif args.n:
        n_values = args.n
    else:
        n_values = [5, 6, 7, 8, 9, 10]

    run_benchmark(n_values, args.model, challenges_dir, results_dir,
                  reasoning_effort=args.reasoning)


if __name__ == '__main__':
    main()
