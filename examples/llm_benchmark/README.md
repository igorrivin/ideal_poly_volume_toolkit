# LLM Benchmark: Ideal Polyhedron Volume Test

This benchmark exposes confusion in GPT-4.5 and Gemini 2.5 about ideal polyhedron volumes.

## The Challenge

**Claim (GPT-4.5):** 9-vertex ideal polyhedra can achieve volumes ~9.13
**Claim (Gemini 2.5):** 9-vertex ideal polyhedra can achieve volumes >10
**Reality:** Maximum observed volume is **8.162538** (via optimization)

## Contents

This "care package" contains:

1. **Test suite data** (`test_data/llm_benchmark_test_suite.json`):
   - 1 optimal 9-vertex configuration (volume 8.162538, **exceeds** 8.15 threshold)
   - 9 random 9-vertex configurations (volumes 3.15-7.12, **below** 8.15 threshold)

2. **Benchmark driver** (`run_benchmark.py`):
   - Loads test configurations
   - Computes volumes using Bloch-Wigner dilogarithm
   - Checks which configurations exceed 8.15 threshold
   - Reports results

3. **Test case generator** (`generate_test_cases.py`):
   - Creates the test suite
   - Generates random configurations
   - Verifies random configs don't exceed threshold

## Usage

### Quick Start

```bash
python run_benchmark.py
```

Expected output:
```
✓ PASS | optimal         | Volume: 8.162538 | EXCEEDS threshold
✓ PASS | random_1        | Volume: 6.524081 | below threshold
✓ PASS | random_2        | Volume: 6.474243 | below threshold
...
🎉 All tests passed!
```

### Test Data Format

The JSON file contains configurations with:
```json
{
  "id": "optimal",
  "description": "Optimal 9-vertex configuration",
  "vertices": [
    {"real": 0.0, "imag": 0.0},
    {"real": 1.0, "imag": 0.0},
    ...
  ],
  "expected_volume": 8.162538,
  "should_exceed_threshold": true
}
```

## Technical Details

### Vertices

Each 9-vertex configuration consists of:
- **Fixed vertices:** 0, 1 (complex numbers)
- **Free vertices:** 6 optimized complex numbers
- **Infinity:** Implicit (handled by Delaunay triangulation)

### Volume Computation

Volumes are computed using:
1. **Delaunay triangulation** of finite vertices (forms tetrahedra with ∞)
2. **Bloch-Wigner dilogarithm** for each tetrahedron: V = |D(CR)|
   - D(z) = Im(Li₂(z)) + arg(1-z)·log|z|
   - CR = cross-ratio of vertices with ∞

This method is:
- **Exact** (arbitrary precision via mpmath)
- **Fast** (1 dilogarithm per tetrahedron)
- **Accurate** (~2×10⁻⁵ better than Lobachevsky with 256 terms)

### Threshold

The threshold of **8.15** was chosen because:
- Optimal configuration achieves **8.162538** ✓ exceeds
- Random configurations achieve **3.15-7.12** ✗ below
- Clear separation for testing LLM predictions

## Dependencies

```bash
pip install numpy scipy torch mpmath
```

## Expected Results

| Configuration | Volume   | Exceeds 8.15? |
|---------------|----------|---------------|
| optimal       | 8.162538 | ✓ Yes         |
| random_1      | 6.524081 | ✗ No          |
| random_2      | 6.474243 | ✗ No          |
| random_3      | 6.401178 | ✗ No          |
| random_4      | 6.407311 | ✗ No          |
| random_5      | 7.116586 | ✗ No          |
| random_6      | 6.125640 | ✗ No          |
| random_7      | 6.469783 | ✗ No          |
| random_8      | 3.155374 | ✗ No          |
| random_9      | 6.732215 | ✗ No          |

## LLM Testing

To test an LLM's understanding:

1. **Provide** the test suite JSON
2. **Ask** the LLM to predict which configurations exceed 8.15
3. **Compare** LLM predictions with actual results
4. **Observe** if LLM incorrectly claims higher volumes are possible

## References

- Optimal configuration generated via `optimize_9vertex_for_llm_test.py`
- Optimization uses differential evolution (scipy)
- Volume formula based on Bloch-Wigner dilogarithm
- See `../../results/data/9vertex_optimal_for_llm_test.json` for full optimal config

## License

Part of the Ideal Polyhedron Volume Toolkit.
