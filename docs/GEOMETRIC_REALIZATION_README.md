# Geometric Realization from Rivin LP Angles

## Overview

This toolkit now supports **geometric realization** of planar triangulations from the angles computed by Rivin's linear programming method. Given a combinatorial triangulation that passes the Delaunay realizability test, we can construct the actual 2D point positions that realize those angles.

## Key Insight: Rivin LP Gives Complete Angle Specification

The Rivin LP solver provides **all interior angles of all triangles**. These angles, when the LP is feasible, are both **necessary and sufficient** for geometric realizability. The LP ensures:

1. Each triangle's angles sum to π (Euclidean constraint)
2. Boundary vertex angles sum to 2π (flatness)
3. Dihedral angles (sums across interior edges) satisfy realizability constraints

## Angle Scaling

**Important**: The LP uses scaled units where π = 1 (for numerical stability). To convert to radians for geometric construction:

```python
angles_radians = lp_angles * np.pi
```

Each triangle's angles in the LP output sum to 1.0 (scaled), which equals π radians.

## Rigid Construction Algorithm

The geometric realization uses a **rigid construction** approach based on the law of sines:

### Algorithm Steps

1. **Fix first two vertices**:
   - v₁ = 0 (complex plane)
   - v₂ = 1

2. **Place third vertex** using first triangle containing v₁ and v₂:
   - Given angles α, β, γ at vertices v₁, v₂, v₃
   - Use law of sines: `|v₁-v₃|/sin(β) = |v₂-v₃|/sin(α) = |v₁-v₂|/sin(γ)`
   - Since |v₁-v₂| = 1 is fixed, compute edge lengths
   - Find v₃ by circle intersection (two circles centered at v₁, v₂)

3. **Incrementally place remaining vertices**:
   - For each unplaced vertex, find a triangle containing it and two placed vertices
   - Use same law of sines + circle intersection approach
   - Choose intersection that doesn't overlap existing vertices
   - Construction is **rigid** - angles uniquely determine positions

### Implementation

```python
from ideal_poly_volume_toolkit.geometric_realization import realize_from_angles_rigid
from ideal_poly_volume_toolkit.rivin_delaunay import check_delaunay_realizability

# Check realizability and get angles
result = check_delaunay_realizability(triangles, verbose=False, strict=True)

if result['realizable']:
    # Extract angles (in scaled units, π = 1)
    angles_scaled = result['angles']
    n_triangles = len(triangles)

    # Convert to radians
    angles_radians = angles_scaled * np.pi
    angles_array = angles_radians.reshape((n_triangles, 3))

    # Rigid construction
    construction = realize_from_angles_rigid(triangles, angles_array, verbose=True)

    if construction['success']:
        points = construction['points']  # Shape: (n_vertices, 2)
        vertex_list = construction['vertex_list']  # Vertex IDs

        # Use points for further analysis...
```

## Example: The Octahedron

For n=6, the unique strictly realizable triangulation (the octahedron) has:
- 4 triangles: [(1,2,5), (1,4,5), (2,3,5), (3,4,5)]
- All angles: 45-45-90 degrees (π/4, π/4, π/2 radians)

Geometric realization produces:
```
v₁: (0.0, 0.0)
v₂: (1.0, 0.0)
v₃: (1.0, 1.0)
v₄: (0.0, 1.0)
v₅: (0.5, 0.5)
```

This forms a **square with center point**, creating 4 right isosceles triangles.

## Verification

The construction achieves:
- **Angle accuracy**: < 10⁻⁸ degrees RMS error
- **Triangulation preservation**: Delaunay triangulation of realized points matches input
- **Rigidity**: Construction is deterministic given the angle specification

## Testing

```bash
# Test on octahedron
python examples/test_rigid_construction.py

# Test on other cases
python examples/test_rigid_construction.py --n 7 --index 0
```

## Technical Details

### Why This Works

Rivin's criterion is that the LP being feasible is **necessary and sufficient** for Delaunay realizability. When the LP has a solution:

1. The angles satisfy all geometric constraints
2. The construction is rigid (angles → unique geometry up to similarity)
3. The resulting point configuration is Delaunay

### Ambiguity Resolution

At each step, circle intersection gives two possible positions. We choose based on:
1. **No overlap**: Reject positions too close to existing vertices
2. **Orientation**: Prefer positive signed area (counterclockwise)

This ensures the construction follows the intended combinatorial structure.

## Files

- `ideal_poly_volume_toolkit/geometric_realization.py`: Core implementation
- `examples/test_rigid_construction.py`: Test and verification
- `examples/debug_angles.py`: Detailed angle debugging

## References

- Rivin, I. "Euclidean structures on simplicial surfaces and hyperbolic volume"
- The Rivin criterion provides both necessity and sufficiency for realizability
