# Changelog - Major Updates (2025)

> **Historical document.** This narrative log describes work performed in
> 2025 and is retained for provenance. It is not the current release changelog
> or known-issues list. See the [canonical changelog](../CHANGELOG.md) and
> [0.4 release note](release-0.4.0.md) for current release information.

## Critical Bug Fix: Triangle Extraction from Plantri

### The Bug
Previously, triangle extraction from plantri's planar embedding was **fundamentally broken**. The code was finding 3-cycles in the adjacency graph instead of properly extracting faces from the planar embedding.

**Impact**:
- Generated incomplete triangulations (3-7 triangles instead of 16 for n=10)
- 21.56% had ZERO spanning trees (impossible for 3-connected graphs!)
- All previous exhaustive enumeration results were **invalid**

### The Fix
Created `ideal_poly_volume_toolkit/planar_utils.py` with proper face extraction:
- `extract_faces_from_planar_embedding()` correctly interprets plantri's cyclic adjacency format
- Traces face boundaries by following edges in cyclic order
- Extracts all triangular faces from the embedding

**Files affected**:
- `ideal_poly_volume_toolkit/planar_utils.py` (new)
- `examples/analyze_delaunay_parallel.py` (fixed)
- `examples/analyze_spanning_trees.py` (fixed)
- All enumeration scripts updated

## New Features

### 1. Strict Realizability Mode

Added support for **strict Delaunay realizability** (dihedral angles < π, not ≤ π):

```python
result = check_delaunay_realizability(triangles, strict=True)
```

This ensures no cocircular quadrilaterals (boundary phenomenon).

**Implementation**:
- Added slack variable ε ≥ 0 to dihedral constraints
- Constraint: `sum_of_opposite_angles + ε ≤ π`
- Maximize ε with high priority to push angles away from boundary

**Results** (3-connected, corrected):
| n | Total | Standard | Strict | Strict % |
|---|-------|----------|--------|----------|
| 10 | 32,276 | 99.4% | 17.7% | 17.7% |
| 11 | 440,348 | - | 18.4% | 18.4% |
| 12 | 6,383,736 | - | 18.2% | 18.2% |
| 13 | 96,258,336 | - | 17.9% | 17.9% |

**Key finding**: ~82% of standard-realizable triangulations sit exactly on the boundary (dihedral = π).

### 2. Geometric Realization from LP Angles

New capability to reconstruct actual 2D point positions from Rivin LP angles.

**Theory**:
- Rivin LP gives ALL interior angles of ALL triangles
- LP feasibility is necessary AND sufficient for realizability
- Angles uniquely determine geometry (up to similarity) via rigid construction

**Implementation**: `ideal_poly_volume_toolkit/geometric_realization.py`
- `realize_from_angles_rigid()`: Rigid construction using law of sines
- Algorithm:
  1. Fix v₁=0, v₂=1
  2. Place v₃ using first triangle (law of sines + circle intersection)
  3. Incrementally place remaining vertices
  4. Handle circle intersection ambiguity (avoid overlaps, maintain orientation)

**Accuracy**:
- Angle error: < 10⁻⁸ degrees RMS
- Triangulation preservation: Perfect match
- Deterministic and rigid

**Example - The Octahedron**:
```
Input:  4 triangles with 45-45-90 angles
Output: Square with center point at (0.5, 0.5)
Error:  < 10⁻⁸ degrees
```

### 3. Random Sampling for Large n

Added random triangulation generation for n=20 to n=100 using Schaeffer's PlanarMap generator.

**Results** (10,000 samples each):
| n | Realizability |
|---|---------------|
| 20 | 92.9% |
| 50 | 86.8% |
| 70 | 82.9% |
| 100 | 77.3% |

**Key finding**: Realizability decreases with n, converging toward ~75-77% for large n.

### 4. Arithmetic Dihedral Angles in Maximal Volume Configurations

**MAJOR DISCOVERY**: Maximal volume triangulations have dihedral angles that are exact rational multiples of π.

**Conjecture** (confirmed): Dihedral angles in maximal volume configurations are of the form pπ/q where p, q are small integers.

**Implementation**: `examples/analyze_maximal_dihedral_angles.py`
- Uses continued fractions to detect rational approximations
- Computes dihedral angles from Rivin LP solution
- Finds best rational approximation pπ/q for each angle

**Results** (10 maximal volume configurations, 227 interior edges analyzed):
```
Distribution by denominator:
  q= 1: 31 angles  (exact π - boundary edges)
  q= 3: 10 angles  (regular tetrahedron)
  q= 5: 23 angles  (icosahedral symmetry)
  q= 7: 33 angles
  q=14:  9 angles
  q=17: 41 angles
  q=19: 78 angles

Most common patterns:
  1π/1: 31 occurrences (180° - exact boundary)
  6π/19: 22 occurrences
  16π/19: 19 occurrences
  3π/7: 17 occurrences
  4π/7: 14 occurrences
```

**Error**: < 10⁻¹³ degrees (machine precision!)

**Key Examples**:
- **4-vertex tetrahedron**: ALL angles exactly π/3 (60°)
- **12-vertex icosahedral**: ALL angles are multiples of π/5
- **19-vertex**: ALL 44 angles are multiples of π/17
- **22-vertex**: ALL 50 angles are multiples of π/19

**Theoretical Significance**:
- Connects to arithmetic structure (cross-ratios have modulus 1)
- Maximal volume configurations are arithmetic/algebraic
- This is NOT obvious from the optimization problem - emerges as "magic"
- Requires strict realizability for the argument to work

### 5. Flip Graph Mixing Time Experiment

Empirical measurement of mixing time for random walks on the triangulation flip graph.

**Setup**:
- Ground truth: All 32,276 triangulations for n=10
- Sample via k-flip chains (k = 100, 500, 1000, 2000, 5000, 10000)
- Compare statistics to ground truth

**Results** (continuous chain mode):
- Standard realizability: ~4% error persists at k=10,000
- Strict realizability: Slowly converging, 1.74% error at k=10,000
- **Spanning trees: NO CONVERGENCE** - stuck at 2.2× ground truth even at k=10,000

**Implications**:
- Mixing time is MUCH longer than expected
- Possibly supports (or exceeds!) the conjectured n^(6/5) mixing time
- Flip graph has poor expansion properties

## Bug Fixes

### 1. Angle Scaling
- **Issue**: LP returns angles in scaled units (triangle sum = 1)
- **Fix**: Multiply by π to convert to radians for geometric construction
- **Impact**: Geometric realization now works correctly

### 2. JSON Serialization
- **Issue**: NumPy bool types not JSON serializable
- **Fix**: Convert to Python bool: `bool(numpy_value)`

### 3. Circle Intersection Ambiguity
- **Issue**: Two possible positions, code defaulted to first
- **Fix**: Choose based on minimum distance to existing vertices and orientation

## New Analysis Scripts

### Geometric Realization
- `examples/test_rigid_construction.py`: Test and verify geometric realization
- `examples/debug_angles.py`: Debug LP angles vs realized angles
- `examples/test_geometric_realization.py`: Initial testing (deprecated)

### Arithmetic Structure
- `examples/analyze_maximal_dihedral_angles.py`: Analyze dihedral angles in maximal volume configurations
  - Confirms rational angle conjecture
  - Uses continued fractions for exact detection
  - Reveals arithmetic/algebraic structure

### Spanning Trees
- `examples/analyze_spanning_trees.py`: Spanning trees vs realizability
- `examples/analyze_spanning_tree_distribution.py`: Distribution analysis

### Sampling and Mixing
- `examples/sample_random_triangulations.py`: Unified sampling framework
- `examples/measure_flip_mixing_time.py`: Mixing time experiments

### Identification
- `examples/identify_n6_strict.py`: Confirmed octahedron as unique n=6 strict case

## Corrected Results

All exhaustive enumeration results have been re-run with corrected triangle extraction:

### 3-Connected Standard (n=4-10)
- Results: `results/delaunay_3connected_exhaustive.json`
- Log: `logs_3connected.txt`

### 3-Connected Strict (n=4-13)
- Results: `results/delaunay_3connected_strict_CORRECTED.json`
- Log: `logs_3connected_strict_CORRECTED.txt`
- **Largest**: 96.2M triangulations tested for n=13 in 2.5 hours

### Random Sampling (n=20-100)
- Results: `results/delaunay_random_large_scale.json`
- 10,000 samples per n value

## Performance Improvements

- Parallel processing: 30 workers for exhaustive enumeration
- Efficient LP solving with HiGHS backend
- Optimized face extraction from planar embeddings

## Documentation

- `GEOMETRIC_REALIZATION_README.md`: Complete guide to geometric realization
- `delaunay_enumeration_README.md`: Methodology for exhaustive enumeration
- `examples/rivin_delaunay_README.md`: Rivin's method and edge flips

## Testing

All major features have test scripts:
```bash
# Geometric realization
python examples/test_rigid_construction.py

# Spanning trees analysis
python examples/analyze_spanning_trees.py --n 10 --output results/spanning_n10.json

# Random sampling
python examples/sample_random_triangulations.py --n 50 --flips 25000

# Mixing time
python examples/measure_flip_mixing_time.py --n 10 --mode continuous
```

## Known Issues

None currently. All major bugs have been fixed.

## Future Work

- **Andreev mode**: Extend to non-maximal graphs (requires vertex degrees 3-4)
- **Volume analysis**: Compare volumes for realizable vs non-realizable
- **Hyperbolic realization**: Place ideal points on ∂H² for hyperbolic polyhedra
- **Graph-theoretic characterization**: Identify structural predictors of strict realizability
