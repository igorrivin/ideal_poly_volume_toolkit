# LLM Geometric Reasoning Benchmark

A torture test for evaluating Large Language Model capabilities in geometric reasoning and computational geometry.

## Overview

This benchmark tests whether an LLM can:
1. Recognize when a triangulation is geometrically realizable
2. Construct a valid point set that realizes a given triangulation
3. Correctly identify impossible cases

## The Challenge

Given a triangulation specified as a list of triangles (vertex triples), the LLM must either:
- **Produce a set of 2D points** whose Delaunay triangulation has the same combinatorial structure, OR
- **Output "None"** if no such point set exists

## Mathematical Background

Not all triangulations are **Delaunay realizable**. A triangulation can be realized as the Delaunay triangulation of some point set if and only if it satisfies **Rivin's constraints**:

1. All angles positive
2. Triangle angle sums = π
3. Interior vertex angle sums = 2π
4. Boundary vertex angle sums ≤ π
5. **Opposite angles across interior edges ≤ π** (the Delaunay edge condition)

We use linear programming to verify these constraints.

## Benchmark Structure

Each benchmark contains **multiple challenges**:
- **1 is realizable** (has a valid point set solution, includes certificate points)
- **1-5 are NOT realizable** (created via edge flips, verified by Rivin's LP)

**Important**: Generating non-realizable triangulations is challenging. Most random edge flips either:
1. Create invalid triangulations (edges appearing in >2 triangles), OR
2. Remain realizable after flipping

Therefore, benchmarks typically contain 1-3 non-realizable cases rather than the target of 5-9.

The challenges are shuffled so the LLM cannot guess which is realizable.

###Example Benchmark Format

```json
{
  "metadata": {
    "description": "LLM Geometric Reasoning Benchmark",
    "n_vertices": 150,
    "n_challenges": 10,
    "n_realizable": 1,
    "n_non_realizable": 9,
    "generated": "2025-01-13T...",
    "seed": 42
  },
  "challenges": [
    {
      "label": "challenge_5",
      "n_vertices": 150,
      "n_triangles": 283,
      "triangles": [[0, 1, 2], [1, 3, 4], ...],
      "is_realizable": false,
      "solution_exists": false
    },
    {
      "label": "challenge_0",
      "n_vertices": 150,
      "n_triangles": 283,
      "triangles": [[0, 1, 2], [1, 3, 4], ...],
      "is_realizable": true,
      "solution_exists": true,
      "certificate_points": [[x0, y0], [x1, y1], ...]  // Proof that it's realizable
    },
    ...
  ],
  "instructions": "..."
}
```

## Usage

### Generate Benchmark

```bash
python examples/generate_llm_benchmark.py \
    --points 150 \
    --flips 30 \
    --seed 42 \
    --output llm_benchmark.json
```

**Parameters:**
- `--points`: Number of vertices (default: 150)
- `--flips`: Edge flips for non-realizable cases (default: 30)
- `--seed`: Random seed for reproducibility
- `--output`: Output JSON file

### Check LLM Response

```bash
# Example 1: LLM claims not realizable
python examples/check_llm_response.py llm_benchmark.json 0 --points None

# Example 2: LLM provides point set
python examples/check_llm_response.py llm_benchmark.json 3 --points solution.npy
```

The checker:
1. Verifies point set dimensions and vertex count
2. Computes Delaunay triangulation of provided points
3. Uses **pynauty** for robust graph isomorphism checking
4. Returns ✓ CORRECT or ✗ INCORRECT with detailed reasoning

## Verification Method

We use **canonical graph labeling** via pynauty to check combinatorial equivalence:

1. Convert triangulation to graph (vertices + edges)
2. Compute canonical labeling (unique representation up to isomorphism)
3. Compare canonical forms

This handles vertex relabeling: the LLM's point set can use any vertex ordering, and we'll correctly identify if the triangulation structure matches.

## Why This Is Hard

This benchmark is challenging because:

1. **No closed-form solution**: There's no formula to go from triangulation → points
2. **Degenerate cases**: Some triangulations are "almost" realizable (dihedral angles near π)
3. **Combinatorial explosion**: 150 vertices produce ~283 triangles with complex constraints
4. **Non-convex feasible regions**: The Rivin polytope is non-convex in general
5. **Isomorphism checking**: Must recognize equivalent structures despite relabeling

## Expected Performance

**Human expert with code**: Can solve most cases using:
- Rivin's LP test for non-realizable detection
- Iterative optimization for realizable construction
- Success rate: ~80-90% (construction is numerically challenging)

**Current LLMs (o1, Sonnet 4, GPT-4)**:
- Unknown (this benchmark is new!)
- Likely challenges: geometric intuition, numerical optimization, recognizing degeneracies

## Scoring

For each challenge:
- **Correct answer**: 1 point
- **Incorrect answer**: 0 points

Perfect score: **N/N** (typically 2-4 challenges total)

Breakdown:
- Getting all "None" cases: Requires understanding Rivin constraints via LP
- Getting the 1 realizable case: Requires geometric construction skills (OR recognizing the certificate)

**Note**: The realizable challenge includes `certificate_points` as proof. An LLM that simply returns these points would be correct but not demonstrate construction ability.

## Files

- `examples/generate_llm_benchmark.py`: Benchmark generator
- `examples/check_llm_response.py`: Response checker with pynauty
- `llm_benchmark_150v.json`: 150-vertex benchmark (challenging)
- `examples/test_benchmark.json`: 20-vertex benchmark (for testing)

## Dependencies

```bash
pip install numpy scipy pynauty
```

## Theory References

- **Rivin (1994)**: "Euclidean structures on simplicial surfaces and hyperbolic volume"
  - Proves that Delaunay realizability = satisfying angle constraints
  - Characterizes the space of realizable angle assignments (Rivin polytope)

- **De Loera et al. (2010)**: "Triangulations: Structures for Algorithms and Applications"
  - Comprehensive treatment of triangulation theory

## Citation

If you use this benchmark, please cite:
```
@software{llm_geometric_benchmark2025,
  title = {LLM Geometric Reasoning Benchmark},
  author = {Generated via Claude Code},
  year = {2025},
  note = {Delaunay realizability torture test}
}
```

## License

MIT License - feel free to use and extend!

---

**Challenge accepted?** Test your LLM's geometric reasoning skills with this benchmark! 🎯
