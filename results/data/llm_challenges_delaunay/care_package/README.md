# Delaunay Realizability Benchmark

## Task

For each triangulation, determine whether it can be realized as the
Delaunay triangulation of some set of points in the plane.

- If **yes**: produce the point coordinates.
- If **no**: output `None`.

## Files

- `benchmark.json` — 11 challenges (shuffled, anonymized IDs). Each challenge
  has `n_vertices`, `n_triangles`, and `triangles` (list of [i,j,k] triples).
- `ground_truth.json` — answers (do not distribute to the model).
- `check_response.py` — standalone checker script.

## Response format

Submit a JSON file mapping challenge IDs to answers:

```json
{
    "challenge_00": {"points": [[0.1, 0.2], [0.3, 0.4], ...]},
    "challenge_01": {"result": "None"},
    ...
}
```

For realizable triangulations, vertex labels may be permuted —
isomorphism is checked via canonical graph labeling (pynauty).

## Running the checker

```bash
pip install numpy scipy pynauty
python check_response.py benchmark.json response.json ground_truth.json
```

## Challenge statistics

- 11 total challenges
- Vertex counts: 100, 200, 283, 587
- Mix of realizable and non-realizable (not revealed which is which)
- Non-realizable instances include edge-flip constructions and
  face stellations

## Dependencies

- Python 3.8+
- numpy
- scipy
- pynauty (`pip install pynauty`)
