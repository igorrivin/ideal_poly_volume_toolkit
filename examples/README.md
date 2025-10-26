# Examples Directory

This directory contains organized example scripts demonstrating the use of the ideal polyhedra volume toolkit.

## Directory Structure

### `distributions/`
Analysis of volume distributions for various polyhedra:
- `tetrahedron/` - 4-vertex polyhedra volume distributions (10 files)
- `five_vertex/` - 5-vertex polyhedra distributions
- `six_vertex/` - 6-vertex polyhedra distributions
- `euclidean/` - Euclidean tetrahedra analysis and fitting

### `optimization/`
Optimization scripts organized by vertex count:
- `7vertex/` - 7-vertex optimization (octahedron with stellated face hypothesis) - 9 scripts
- `12vertex/` - 12-vertex optimization - 3 scripts
- `20vertex/` - 20-vertex optimization (icosahedron-like) - 6 scripts
- `platonic/` - Platonic solid analysis and perturbations

### `visualization/`
Visualization scripts (5 scripts):
- Sphere projection visualizations
- Golden ratio configurations
- Maximal volume configurations
- Volume landscapes

### `analysis/`
Statistical and theoretical analysis scripts (14 scripts):
- Beta distribution theory
- Central limit theorem analysis
- Combinatorial mixture analysis
- Special configuration analysis

## Running Examples

All examples can be run from their respective directories:

```bash
cd examples/optimization/7vertex
python optimize_7vertex.py
```

Or from the project root using absolute imports (package must be installed with `pip install -e .`).
