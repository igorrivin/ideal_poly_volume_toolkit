# Delaunay Realizability Modes

This document describes the three realizability modes implemented in the Rivin LP solver.

## Overview

The `check_delaunay_realizability()` function now supports three different modes for testing whether a planar triangulation can be realized as a Delaunay triangulation:

1. **Standard mode** (default): Dihedral angles ≤ π
2. **Strict mode** (`strict=True`): Dihedral angles < π (strictly less than)
3. **Andreev mode** (`andreev=True`): Dihedral angles ≤ π/2 (right-angled)

## Mathematical Background

### Dihedral Angles

In Rivin's formulation, the **dihedral angle** at an interior edge is the sum of the two angles opposite to that edge in the adjacent triangles. This dihedral angle also equals the exterior angle of the convex hull at that edge.

For a Delaunay triangulation:
- **Necessary condition**: All dihedral angles ≤ π
- This is equivalent to the "edge flip criterion" for Delaunay triangulations

### Boundary Angles

Similarly, the sum of angles at a boundary vertex represents the exterior angle at that boundary point.

For a valid triangulation:
- **Necessary condition**: All boundary angle sums ≤ π

## The Three Modes

### 1. Standard Mode (Default)

**Mathematical formulation:**
```
Maximize: t (minimum angle)
Subject to:
  - All angles ≥ t
  - Sum of angles in each triangle = π
  - Sum of angles around interior vertex = 2π
  - Sum of angles at boundary vertex ≤ π
  - Dihedral angle (sum of opposite angles) ≤ π  [KEY CONSTRAINT]
```

**Usage:**
```python
result = check_delaunay_realizability(triangles)
```

**Interpretation:**
- Tests if triangulation is Delaunay realizable
- Allows dihedral angles to equal exactly π (non-strict inequality)
- This can lead to "degenerate" realizations where four points are cocircular

**When to use:**
- Default choice for checking Delaunay realizability
- Matches Rivin's original criterion
- Most permissive mode

### 2. Strict Mode

**Mathematical formulation:**
```
Maximize: 1000·ε + t (prioritize slack, then minimum angle)
Subject to:
  - All angles ≥ t
  - Sum of angles in each triangle = π
  - Sum of angles around interior vertex = 2π
  - Sum of angles at boundary vertex ≤ π
  - Dihedral angle + ε ≤ π  [KEY CONSTRAINT]
  - ε ≥ 0 (slack must be positive)
```

**Usage:**
```python
result = check_delaunay_realizability(triangles, strict=True)
if result['realizable']:
    print(f"Dihedral slack: {result['slack_radians']} rad")
    print(f"Max dihedral: {result['max_dihedral_radians']} rad (< π)")
```

**Interpretation:**
- Tests if triangulation is **strictly** Delaunay realizable
- Requires all dihedral angles < π (strict inequality)
- The slack ε measures "how far" from degenerate the realization is
- Ensures no four points are cocircular

**When to use:**
- When you need a non-degenerate realization
- When geometric robustness is important
- When you want to avoid numerical issues near π

**Key insight:**
- A triangulation can be standard-realizable but not strict-realizable
- This happens when the maximum achievable dihedral angle is exactly π
- Such triangulations force four points to be cocircular

### 3. Andreev Mode

**Mathematical formulation:**
```
Maximize: t (minimum angle)
Subject to:
  - All angles ≥ t
  - Sum of angles in each triangle = π
  - Sum of angles around interior vertex = 2π
  - Sum of angles at boundary vertex ≤ π/2  [ANDREEV CONSTRAINT]
  - Dihedral angle ≤ π/2  [ANDREEV CONSTRAINT]
```

**Usage:**
```python
result = check_delaunay_realizability(triangles, andreev=True)
```

**Interpretation:**
- Tests if triangulation can be realized with **right-angled faces**
- All dihedral angles ≤ π/2 (90°)
- All boundary angles ≤ π/2
- Connected to **Andreev's theorem** on hyperbolic polyhedra with right-angled faces

**When to use:**
- Testing for right-angled hyperbolic polyhedra
- Andreev's theorem applications
- Volume computations in special cases

**Theoretical connection:**
- Andreev (1970) proved existence/uniqueness of hyperbolic polyhedra with prescribed right-angled faces
- This mode tests the combinatorial constraints for such polyhedra
- Much more restrictive than standard/strict modes

## Comparison Table

| Property | Standard | Strict | Andreev |
|----------|----------|--------|---------|
| Dihedral bound | ≤ π | < π | ≤ π/2 |
| Boundary bound | ≤ π | ≤ π | ≤ π/2 |
| Allows cocircular points | Yes | No | No |
| Optimization variable | t (min angle) | ε (slack) + t | t (min angle) |
| Realizability rate | Highest | Medium | Lowest |

## Examples from Tests

### Hexagon Triangulation

```
Standard realizable: True  (min angle: 45°)
Strict realizable:   True  (max dihedral: 90°, slack: 90°)
Andreev realizable:  False
```

The hexagon can be realized as Delaunay, and even strictly (with max dihedral = π/2), but not all dihedral angles can be ≤ π/2 simultaneously in Andreev mode.

### Random Triangulation (10 points)

```
Standard realizable: True  (min angle: 36°)
Strict realizable:   True  (max dihedral: 120°, slack: 60°)
Andreev realizable:  False
```

The random triangulation is strictly realizable with dihedral angles up to 120°, but cannot satisfy the π/2 bound.

### Octahedron (Square Pyramid)

```
Standard realizable: True
Strict realizable:   True  (max dihedral: 90°, slack: 90°)
Andreev realizable:  True
```

The octahedron (after removing one vertex) can be realized in all three modes! This is a special case where all dihedral angles can be exactly π/2.

## Return Values

All modes return a dictionary with these common keys:

```python
{
    'realizable': bool,              # True if realizable in this mode
    'min_angle': float,              # Minimum angle (scaled)
    'min_angle_radians': float,      # Minimum angle in radians
    'angles': np.ndarray,            # Angle assignment (n_triangles, 3)
    'angles_radians': np.ndarray,    # Angles in radians
    'status': int,                   # LP solver status
    'message': str,                  # LP solver message
    'success': bool,                 # LP solver success
    'mode': str,                     # Mode description
    'strict': bool,                  # Whether strict mode was used
    'andreev': bool,                 # Whether Andreev mode was used
}
```

**Additional keys for strict mode:**

```python
{
    'slack': float,                  # Dihedral slack ε (scaled)
    'slack_radians': float,          # Slack in radians
    'max_dihedral': float,           # Maximum dihedral angle (scaled)
    'max_dihedral_radians': float,   # Max dihedral in radians
}
```

## Implementation Details

### LP Variable Structure

**Standard/Andreev:**
```
Variables: [angle[0,0], angle[0,1], ..., angle[n-1,2], t]
```
- `n_angles` angle variables (3 per triangle)
- 1 minimum angle variable `t`

**Strict:**
```
Variables: [angle[0,0], angle[0,1], ..., angle[n-1,2], t, ε]
```
- `n_angles` angle variables
- 1 minimum angle variable `t`
- 1 slack variable `ε`

### Objective Functions

- **Standard/Andreev**: Maximize `t` (or minimize `-t`)
- **Strict**: Maximize `1000·ε + t` (prioritize slack, then min angle)

The weight 1000 on ε ensures lexicographic optimization: first maximize slack, then maximize min angle.

### Constraint Modifications

The key difference is in the dihedral angle constraint:

**Standard:**
```
sum_of_opposite_angles ≤ 1.0  (π in normalized units)
```

**Strict:**
```
sum_of_opposite_angles + ε ≤ 1.0  (π in normalized units)
```

**Andreev:**
```
sum_of_opposite_angles ≤ 0.5  (π/2 in normalized units)
```

## Theoretical Implications

### Standard vs Strict

A triangulation that is standard-realizable but NOT strict-realizable must have at least one interior edge where the sum of opposite angles is **exactly** π. This means:

1. The four vertices forming a quadrilateral around that edge are **cocircular**
2. The edge could be flipped to the other diagonal of the quadrilateral
3. The triangulation is "on the boundary" of the Rivin polytope
4. Small perturbations might make it non-realizable

### Andreev Mode

Andreev's theorem (1970) states that for every 3-connected planar graph with faces of degree ≥ 3, there exists a unique (up to isometry) compact hyperbolic polyhedron with:
- Face combinatorics matching the graph
- All dihedral angles equal to π/2

Our Andreev mode tests whether a triangulation satisfies the **necessary** conditions for this. However, it's not sufficient - Andreev's theorem has additional requirements on the face structure.

## Performance

All three modes have similar performance:
- **Time complexity**: O(LP solve) ≈ O(n²) where n = number of triangles
- **Space complexity**: O(n) for constraint matrices
- **Typical runtime**: 10-100 ms for triangulations with 10-100 triangles

Strict mode adds one extra variable and n_edges constraints, but this has negligible impact on solve time.

## Future Extensions

Potential extensions to consider:

1. **Parameterized mode**: `dihedral_bound=θ` for arbitrary angle bounds
2. **Mixed mode**: Different bounds for different edges
3. **Volume-optimal mode**: Maximize hyperbolic volume subject to realizability
4. **Curvature bounds**: Constrain discrete Gaussian curvature at vertices

## References

1. **Rivin, I. (1996)**. "A characterization of ideal polyhedra in hyperbolic 3-space." *Annals of Mathematics*, 143(1), 51-70.
   - Original Delaunay realizability criterion

2. **Andreev, E.M. (1970)**. "Convex polyhedra in Lobačevskiĭ spaces." *Mat. Sb. (N.S.)*, 81(123), 445-478.
   - Right-angled hyperbolic polyhedra theorem

3. **Leibon, G. (2002)**. "Characterizing the Delaunay decompositions of compact hyperbolic surfaces." *Geometry & Topology*, 6(1), 361-391.
   - Extended Rivin's work to surfaces

4. **Hodgson, C.D., Rivin, I., & Smith, W.D. (1992)**. "A characterization of convex hyperbolic polyhedra and of convex polyhedra inscribed in the sphere." *Bull. Amer. Math. Soc.*, 27(2), 246-251.
   - Connection to sphere packing and convex polyhedra
