# LLM Geometric Reasoning Benchmark

A torture test for evaluating Large Language Model capabilities in geometric reasoning and computational geometry.

This repository also contains a stricter, label-preserving zero-shear
realization protocol. It is different from the older relabeling-insensitive
planar Delaunay benchmark described later on this page.

## Overview

This benchmark tests whether an LLM can:
1. Recognize when a triangulation is geometrically realizable
2. Construct a valid point set that realizes a given triangulation
3. Correctly identify impossible cases

## Strict normalized zero-shear realization protocol

The strict protocol asks for CP1 coordinates with fixed labels and anchors
`z_0 = infinity`, `z_1 = 0`, and `z_2 = 1`. Its private answer key contains
`ground_truth.exists`, `ground_truth.oriented_faces`, normalization metadata,
and optionally `ground_truth.vertices` for a private reference diagnostic. A
model response must be one bare strict-JSON object, with no regex extraction,
Markdown, comments, duplicate keys, `NaN`, or `Infinity` accepted:

```json
{"exists": false}
```

or

```json
{"exists": true, "vertices": [null, [0.0, 0.0], [1.0, 0.0], [0.5, 0.8]]}
```

Run the independent grader with:

```bash
python3 scripts/check_zero_shear_response.py \
  PRIVATE_ANSWER_KEY.json MODEL_RESPONSE.json \
  --output GRADE.json \
  --shear-tolerance 1e-8 \
  --support-tolerance 0
```

The grade is written atomically when `--output` is supplied; otherwise it is
the only JSON object on stdout. An existing output is protected unless
`--overwrite` is explicit. A malformed response or failed realization returns
a nonzero process status. `--reference-tolerance VALUE` promotes the otherwise
diagnostic, label-aligned private-reference chordal error to an additional
gate. It is not needed for independent geometric acceptance.

The package owner is
`ideal_poly_volume_toolkit.rivin_zero_shear.audit_zero_shear_realization`.
Its report keeps the following predicates separate:

- normalization/anchors and handedness, chordal distinctness, and coherent
  closed-sphere topology;
- authoritative strict supporting-plane inequalities and coherent facet
  orientation, with exact labeled Qhull facet-set agreement retained as an
  independent, non-authoritative diagnostic;
- signed bending/Delaunay inequalities;
- zero-shear equalities;
- spherical embeddedness, plus a chart-dependent planar straight-edge
  diagnostic.

This separation is essential: zero shear does not imply Delaunayness,
convexity, the required labeled hull, or embeddedness. A Tutte embedding is an
initial guess only, and an unconstrained solve can cross a cocircular/Delaunay
wall or an embedding wall.

### Tutte-start and multi-start audit

The reproducibility search records whether a Tutte start is unusually reliable
and whether alternative zero-shear roots occur:

```bash
python3 scripts/audit_tutte_zero_shear_basins.py INPUT.json \
  --output AUDIT.json \
  --anchors-per-config 3 \
  --sigmas .25,.5,.75,1 \
  --starts-per-sigma 10 \
  --workers 4
```

Use `--config-id ID` or `--distinguished-face A,B,C` repeatedly to restrict
the search or prescribe ordered anchors. Re-run the same command with
`--resume` after an interruption. The schema is
`docs/schemas/tutte-zero-shear-audit-v1.schema.json`, with schema version
`ideal-poly-tutte-zero-shear-audit/1` and solver protocol
`weighted-tutte-zero-shear-trf/1`. Endpoint classes distinguish the target
strict realization, embedded wrong-hull and nonembedded zero-shear roots,
degenerate roots, residual failures, solver failures, errors, and timeouts.
Root clusters use deterministic task-order representative assignment with a
direct maximum per-label spherical chordal-distance test; a checkpoint keeps
one normalized CP1 representative per discovered cluster, not coordinates for
every start. Resume verifies the corpus, deterministic plan, generator/core
source files (including the holonomy dependency), Python/dependency/platform
versions, and numerical thread environment before adding work. A fully
attempted campaign containing worker errors has status `failed`; wall-time
exhaustion has status `exhausted`. Both return a nonzero CLI status, while only
an error-free `complete` campaign returns zero.
Validate a saved artifact without running a search via:

```bash
python3 scripts/audit_tutte_zero_shear_basins.py --validate-only AUDIT.json
```

### Leakage-resistant holdout packages

`scripts/build_zero_shear_holdouts.py` converts one arithmetic corpus and one
or more finalized basin audits into separate public and private packages. The
public directory contains only opaque sequential IDs, paste-ready prompts,
and prompt hashes. The private directory contains source provenance,
relabeling maps, basin evidence, strict numerical audits, and checker-compatible
answer keys. It must be outside the repository, and neither directory may
already exist:

```bash
python3 scripts/build_zero_shear_holdouts.py \
  CORPUS.json BASIN_A.json BASIN_B.json \
  --public-dir /tmp/zero-shear-holdouts-public \
  --private-dir /SECURE/PATH/zero-shear-holdouts-private \
  --case-id CASE_ID_A --case-id CASE_ID_B
```

Use `--top N` instead of explicit case IDs only after freezing the basin
audits and ranking, before querying any model. Selection admits at most one
case per source configuration/topology, independently reaudits rounded private
coordinates, and requires positive bending orientation. Omitting keys is only
allowed with the explicit `--public-only` flag.

Current holdouts are a positive-only, anchor-mined **Tutte-hard stress set**,
not a representative sample and not a binary existence benchmark. Report
strict-coordinate realization rate as the primary score. An always-true
existence answer is trivial for such a set, and uncertified flipped or
nonconvex examples must not be presented as negative instances. Keep public
prompts and private answer keys in different storage and never commit them
together. Run each holdout one-shot: checker reports containing label-aligned
private-reference errors are internal evaluator artifacts and must not be
returned to a model as an adaptive refinement oracle. A prompt ceases to be a
future holdout once it or its source corpus is published.

Arithmetic-corpus acceptance by
`scripts/generate_arithmeticity_benchmark.py` is not strict convex
certification. Its coordinate-shear, exact abstract PSL(2,Z), KKT, and
realization gates do not by themselves prove spherical embeddedness, signed
bending, or exact labeled hull facets. Apply the strict audit separately.

Finally, the strict audit and checker are double-precision numerical
verification, not formal exact proof. A rigorous existence certificate still
requires a validated root enclosure (for example interval Newton/Krawczyk)
and interval-certified strict support, embedding, and bending inequalities.

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
