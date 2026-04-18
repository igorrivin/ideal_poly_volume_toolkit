# Arithmetic Ideal Polyhedra Benchmark

## Task

Construct pairwise non-isometric convex ideal polyhedra in H^3 whose
surface is given by an arithmetic Fuchsian group. Each polyhedron must
have the specified number of ideal vertices and all triangular faces.

## Files

- `prompt_arithmetic_N50.txt` — prompt asking for 25 polyhedra with 50 vertices
- `prompt_arithmetic_N100.txt` — prompt asking for 10 polyhedra with 100 vertices
- `rivin_holonomy.py` — Penner-Rivin holonomy algorithm (pure Python, no deps)
- `check_response.py` — standalone checker
- `solution_guide.md` — mathematical background (do not give to the model)

## Response format

Submit a JSON array of objects:

```json
[
  {
    "vertices": [[re_1, im_1], [re_2, im_2], ...],
    "triangles": [[i, j, k], ...],
    "volume": 42.123
  },
  ...
]
```

Vertices are complex numbers (real, imaginary parts) on the boundary
of the upper half-space model of H^3.

## What the checker verifies

1. Correct vertex count
2. Valid triangulated sphere (Euler characteristic = 2)
3. Volume matches (recomputed via Lobachevsky function)
4. **Arithmeticity**: all holonomy traces are algebraic integers
   (checked via the Penner-Rivin algorithm with cross-ratio shears)
5. Pairwise non-isometry (via cross-ratio invariants)

## Running the checker

```bash
pip install numpy scipy
python check_response.py prompt_arithmetic_N50.txt response.json
```

Or specify vertex count directly:

```bash
python check_response.py --n-vertices 50 response.json
```

## Dependencies

- Python 3.8+
- numpy, scipy
- `rivin_holonomy.py` (included, pure Python)
