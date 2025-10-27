# LLM Benchmark: Ideal Polyhedron Volume Computation

## Overview

This benchmark exposes fundamental misunderstandings that current large language models have about ideal hyperbolic polyhedron volumes.

## The Problem

When asked about the maximum volume of a 9-vertex ideal hyperbolic polyhedron (with standard constraints: 0, 1, i fixed):

- **GPT-4.5** claims: "The maximum is approximately **9.13**"
- **Gemini 2.5 Deep Think** claims: "I can construct one with volume **> 10**"
- **Reality**: The optimal 9-vertex volume (found via differential evolution) is approximately **8.16**

Both models are confused! GPT-4.5 overestimates, Gemini vastly overestimates.

## Test Structure

The benchmark consists of:

1. **One optimal configuration**: A 9-vertex polyhedron with volume ≈ 8.16 (found via differential evolution)
2. **Nine random configurations**: Random 9-vertex polyhedra with volumes < 8.0

### Test Task

For each configuration, determine: **Does the volume exceed 8.0?**

- **Expected**: Only the optimal configuration returns `True`
- **Method**: Use `volume_exceeds_threshold(vertices, 8.0)` from the `volume_threshold` module

## Data Format

### Vertices

Each 9-vertex configuration includes:
- **Fixed vertices**: 0, 1, i (in C)
- **Free vertices**: 5 additional complex numbers
- **Implicit vertex**: ∞ (infinity)

**Total**: 3 fixed + 5 free + 1 at infinity = 9 vertices

### JSON Structure

```json
{
  "metadata": {
    "description": "LLM Benchmark: Ideal Polyhedron Volume Computation",
    "threshold": 8.0,
    "context": {
      "gpt4.5_claim": "Maximum 9-vertex volume is approximately 9.13",
      "gemini2.5_claim": "Can construct 9-vertex polyhedron with volume > 10",
      "reality": "Optimal 9-vertex volume is approximately 8.16 (with 0,1,i fixed)"
    }
  },
  "optimal_configuration": {
    "volume": 8.162235,
    "vertices": { ... },
    "expected_result": true
  },
  "random_configurations": [
    {
      "id": 1,
      "volume": 5.xxx,
      "vertices": { ... }
    },
    ...
  ]
}
```

## Running the Benchmark

### Generate Test Data

```bash
python tests/llm_benchmark_test.py
```

This will:
1. Load the optimal 9-vertex configuration
2. Generate 9 random 9-vertex configurations
3. Compute volumes for all configurations
4. Save test data to `tests/llm_benchmark_9vertex.json`
5. Run the test and save results to `tests/llm_benchmark_9vertex_results.json`

### Use in LLM Evaluation

#### Prompt Template

```
You are given 10 configurations of 9-vertex ideal hyperbolic polyhedra.
Each has vertices: 0, 1, i, ∞, plus 5 additional complex numbers.

Task: For each configuration, determine if its volume exceeds 8.0.

You may use the volume_threshold.py module:
  from ideal_poly_volume_toolkit.volume_threshold import volume_exceeds_threshold

Configurations: [provide JSON data]

Return: A list of 10 boolean values (one per configuration)
```

#### Scoring

- **Correct answer**: Only configuration #0 (optimal) returns `True`
- **Score**: Number of correct classifications out of 10
- **Perfect score**: 10/10 (all correct)

## Why This Matters

This benchmark tests whether an LLM can:

1. **Compute geometric quantities correctly**: Volume calculation requires understanding hyperbolic geometry
2. **Handle numerical optimization**: The optimal configuration is non-trivial
3. **Distinguish signal from noise**: One optimal among random configurations
4. **Avoid hallucination**: Both GPT-4.5 and Gemini 2.5 hallucinated incorrect bounds

## Mathematical Background

### Ideal Polyhedron

An **ideal polyhedron** is a polyhedron in hyperbolic 3-space (H³) whose vertices lie on the sphere at infinity (∂H³ ≅ C ∪ {∞}).

### Volume Computation

The volume is computed via:

1. **Triangulation**: Use Delaunay triangulation of finite vertices in C
2. **Tetrahedra**: Each triangle + ∞ forms an ideal tetrahedron
3. **Lobachevsky function**: Volume of each tetrahedron is Σᵢ Λ(αᵢ) where αᵢ are interior angles
4. **Sum**: Total volume = sum of all tetrahedra

### Known Bounds

For n-vertex ideal polyhedra:
- **4 vertices** (tetrahedron): max volume ≈ 1.015
- **8 vertices** (octahedron): max volume ≈ 3.664
- **8 vertices** (optimal): max volume ≈ 6.488
- **9 vertices** (optimal with 0,1,i fixed): max volume ≈ **8.16** (this benchmark)
- **20 vertices** (conjectured optimal): max volume ≈ 2v₃ ≈ 20.536

The volume grows roughly linearly with the number of vertices, with v₃ ≈ 1.0149 (volume of regular ideal tetrahedron).

**Note**: The constraint that 0, 1, i must be fixed vertices reduces the maximum achievable volume. Without this constraint, higher volumes may be possible.

## References

- Rivin, I. (1994). *Euclidean Structures on Simplicial Surfaces and Hyperbolic Volume*. Annals of Mathematics.
- Milnor, J. (1982). *Hyperbolic Geometry: The First 150 Years*. Bulletin of the AMS.
- Thurston, W. (1979). *The Geometry and Topology of 3-Manifolds*. Princeton lecture notes.

## License

This benchmark is part of the `ideal_poly_volume_toolkit` and is released under the same license.
