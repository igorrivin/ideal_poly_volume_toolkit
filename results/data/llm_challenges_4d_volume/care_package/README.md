# Ideal 4-Polytope Volume Band Benchmark

## Task

Construct N points on the unit 3-sphere S^3 in R^4 such that the
convex hull, viewed as an ideal polytope in H^4, has a specified
geometric measure within a narrow tolerance window.

Three separate challenge files, one per measure:

| File | Measure | Challenges | Description |
|------|---------|------------|-------------|
| `benchmark_volume.json` | H^4 volume | 17 | Hyperbolic 4-volume |
| `benchmark_surface_area.json` | H^3 surface area | 11 | Sum of boundary facet volumes |
| `benchmark_ratio.json` | Isoperimetric ratio | 5 | volume / surface_area |

## Files

- `benchmark_volume.json` — volume challenges (N=10,20,30)
- `benchmark_surface_area.json` — surface area challenges (N=10,20,30)
- `benchmark_ratio.json` — isoperimetric ratio challenges (N=20)
- `ground_truth_volume.json` — ground truth for volume (do not give to model)
- `ground_truth_surface_area.json` — ground truth for surface area
- `ground_truth_ratio.json` — ground truth for ratio
- `check_response.py` — standalone checker (numpy + scipy only)
- `solution_guide.md` — mathematical background and solution strategy

## Response format

For each challenge, submit a JSON file:

```json
{
  "vertices": [[x0,y0,z0,w0], [x1,y1,z1,w1], ...],
  "measure": "volume",
  "target": 2.893595,
  "epsilon": 0.057872,
  "n_vertices": 10
}
```

All points must have unit norm (lie on S^3).

## Running the checker

```bash
pip install numpy scipy
python check_response.py response.json
```

The checker verifies: unit norm, full-dimensionality, and measure
in [target - epsilon, target + epsilon].

## Dependencies

- Python 3.8+
- numpy
- scipy
