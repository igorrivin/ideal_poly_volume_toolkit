# Rivin's Algorithm for Delaunay Realizability

This module implements Rivin's criterion for checking whether a combinatorial triangulation can be realized as a Delaunay triangulation.

## Overview

Given a triangulation (specified combinatorially as a list of triangles), Rivin's algorithm checks whether there exists a geometric embedding where this triangulation is Delaunay. This is determined by solving a linear program on angle assignments.

### The Rivin Criterion

A triangulation is Delaunay realizable if and only if there exists an assignment of angles to each corner of each triangle satisfying:

1. **(a) Positivity**: All angles are positive
2. **(b) Triangle sum**: Sum of angles in each triangle equals π
3. **(c) Interior vertex sum**: Sum of angles around each interior vertex equals 2π
4. **(d) Boundary constraint**: Sum of angles at each boundary vertex is ≤ π
5. **(e) Delaunay edge condition**: For each interior edge, the sum of the two opposite angles is ≤ π

We formulate this as a linear program: **maximize the minimum angle** subject to constraints (a)-(e). If the optimal minimum angle is > 0, the triangulation is realizable.

## Implementation

### Core Functions

```python
from ideal_poly_volume_toolkit.rivin_delaunay import (
    check_delaunay_realizability,
    format_realizability_report,
    extract_boundary_vertices,
)

# Check if triangulation is Delaunay realizable
triangles = [(0, 1, 2), (0, 2, 3), ...]  # List of (v0, v1, v2) tuples
result = check_delaunay_realizability(triangles, verbose=True)

if result['realizable']:
    print("Triangulation IS Delaunay realizable!")
    print(f"Optimal angle assignment:\n{result['angles_radians']}")
else:
    print("Triangulation is NOT Delaunay realizable")
```

### Linear Program Formulation

**Variables:**
- `angle[tri_id, corner_id]` for each corner of each triangle (3N variables for N triangles)
- `t` = the minimum angle (1 variable)

**Objective:**
- Maximize `t` (equivalently, maximize the minimum angle)

**Constraints:**
- `angle[i] >= t` for all i (ensures positivity and captures the minimum)
- Sum of 3 angles in triangle j = 1 (scaled from π)
- Sum of angles around interior vertex v = 2 (scaled from 2π)
- Sum of angles around boundary vertex v ≤ 1 (scaled from π)
- For each interior edge e, sum of two opposite angles ≤ 1 (scaled from π)

We use `scipy.optimize.linprog` with the HiGHS solver for efficiency.

## Usage Examples

### Example 1: Check a Random Delaunay Triangulation

```bash
python examples/check_delaunay_triangulation.py --points 20 --seed 42
```

### Example 2: Check a Hexagon Triangulation

```bash
python examples/check_delaunay_triangulation.py --example hexagon
```

### Example 3: Check a Grid Triangulation

```bash
python examples/check_delaunay_triangulation.py --example grid
```

### Example 4: Programmatic Usage

```python
import numpy as np
from scipy.spatial import Delaunay
from ideal_poly_volume_toolkit.rivin_delaunay import check_delaunay_realizability

# Generate random points
points = np.random.rand(15, 2)

# Compute Delaunay triangulation
tri = Delaunay(points)
triangles = [tuple(simplex) for simplex in tri.simplices]

# Check realizability
result = check_delaunay_realizability(triangles, verbose=True)

if result['realizable']:
    print(f"✓ Realizable with min angle: {np.degrees(result['min_angle_radians']):.2f}°")
    # Access the optimal angle assignment
    angles_rad = result['angles_radians']  # Shape: (n_triangles, 3)
```

## Results Structure

The `check_delaunay_realizability` function returns a dictionary with:

```python
{
    'realizable': bool,                    # True if Delaunay realizable
    'min_angle': float,                    # Max achievable min angle (scaled units)
    'min_angle_radians': float,            # Max achievable min angle in radians
    'angles': np.ndarray,                  # Optimal angle assignment (scaled)
    'angles_radians': np.ndarray,          # Optimal angle assignment in radians
    'status': int,                         # LP solver status code
    'message': str,                        # LP solver message
    'success': bool,                       # Whether LP solver succeeded
    'n_triangles': int,
    'n_vertices': int,
    'n_interior': int,
    'n_boundary': int,
}
```

## Performance

- **Small triangulations** (<20 triangles): < 10 ms
- **Medium triangulations** (20-100 triangles): 10-100 ms
- **Large triangulations** (100-1000 triangles): 100-1000 ms

The LP solver (HiGHS) is very efficient for these types of problems.

## Theoretical Notes

### Delaunay Realizability vs. Delaunay Triangulation

**Important distinction:**

- **Delaunay triangulation** of a point set: The unique triangulation where every triangle's circumcircle contains no other points.
- **Delaunay realizable triangulation**: A combinatorial triangulation for which there EXISTS some point set whose Delaunay triangulation produces it.

Every Delaunay triangulation (from scipy/qhull) is trivially Delaunay realizable. However, you can also have triangulations that are realizable but not the Delaunay triangulation of any specific point set you have in mind.

### Example: The "Bad Diagonal"

Consider a square with vertices (0,0), (1,0), (1,1), (0,1):
- Delaunay triangulation uses diagonal 0-2: triangles (0,1,2) and (0,2,3)
- Alternative uses diagonal 1-3: triangles (0,1,3) and (1,2,3)

Both are Delaunay *realizable* (the LP succeeds for both). The difference is that (0,1,2),(0,2,3) is Delaunay for the *square* embedding, while (0,1,3),(1,2,3) is Delaunay for a *slightly deformed* quadrilateral (e.g., a trapezoid).

### Non-Realizable Triangulations

To construct a truly non-realizable triangulation requires more complex combinatorics. For example, certain triangulations of surfaces with boundary or non-convex configurations can be non-realizable.

## References

1. **Rivin, I. (1994)**. "Euclidean structures on simplicial surfaces and hyperbolic volume." *Annals of Mathematics*, 139(3), 553-580.
   - Original paper introducing the angle criterion for Delaunay realizability

2. **Leibon, G. (2002)**. "Characterizing the Delaunay decompositions of compact hyperbolic surfaces." *Geometry & Topology*, 6(1), 361-391.
   - Extended applications to hyperbolic surfaces

3. **Rivin, I. (1996)**. "A characterization of ideal polyhedra in hyperbolic 3-space." *Annals of Mathematics*, 143(1), 51-70.
   - Connection to ideal hyperbolic polyhedra (relevant for your volume toolkit!)

## Connection to Your Toolkit

This Rivin realizability checker is particularly relevant for your ideal polyhedra volume toolkit:

- **Ideal triangulations**: Many of your triangulations come from convex hull/Delaunay on sphere
- **Consistency check**: Verify that combinatorial modifications preserve Delaunay property
- **Optimization**: When optimizing vertex positions, check if resulting triangulation is still realizable
- **Hyperbolic volume**: Rivin's work connects Delaunay triangulations to hyperbolic structures!

The constraint (e) - opposite angles sum to ≤ π - is exactly the **edge flip criterion** for Delaunay triangulations, which is fundamental to understanding the volume computations you're doing.
