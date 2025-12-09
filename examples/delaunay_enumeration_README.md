# Exhaustive Delaunay Realizability Analysis

This document describes the methodology for exhaustively analyzing Delaunay realizability of planar triangulations using plantri enumeration.

## Overview

We analyze **what fraction of combinatorially distinct planar triangulations are Delaunay realizable** using exhaustive enumeration for small vertex counts. This provides exact statistics (not sampling-based estimates) for the realizability question.

### Key Question

Given a planar triangulation (graph with triangular faces and a boundary), can it be realized as a Delaunay triangulation in the plane? Equivalently, is this triangulation the 1-skeleton of some convex polyhedron inscribed in the sphere (with one vertex at infinity)?

## Methodology

### 1. Enumeration Pipeline

```
plantri (closed triangulations)
    ↓
Remove vertex 0 → planar triangulation
    ↓
Rivin's LP (check Delaunay realizability)
    ↓
Statistics (fraction realizable)
```

### 2. Closed vs Planar Triangulations

**plantri** generates **closed triangulations** (spherical triangulations = graphs of convex polyhedra):
- All faces are triangles
- No boundary
- Represents convex polyhedron combinatorics

We convert to **planar triangulations** by removing one vertex:
- Remove vertex and all incident edges
- Result: planar graph with triangular faces and a boundary
- Boundary vertices form the outer face

**Key insight**: A closed triangulation with n vertices becomes a planar triangulation with n-1 finite vertices plus one vertex at infinity (the removed vertex).

### 3. The Isomorphism Issue (CRITICAL)

#### Initial (Incorrect) Approach

Our original implementation removed **all n vertices** from each closed triangulation:

```python
# WRONG: Creates isomorphic duplicates
for closed_tri in plantri_output:
    for v in range(n):  # Remove each vertex
        planar_tri = remove_vertex(closed_tri, v)
        test_realizability(planar_tri)
```

**Problem**: This creates n planar triangulations per closed triangulation, but many are **isomorphic** (the same triangulation under relabeling). Since plantri generates one representative per isomorphism class, removing different vertices can produce the same planar triangulation.

**Consequences**:
- Testing n× too many triangulations
- **Biased statistics**: Some triangulations counted multiple times
- Wasted computation (10-12× slower)

#### Corrected Approach

The fix: Remove **only vertex 0** from each closed triangulation:

```python
# CORRECT: One canonical planar version
for closed_tri in plantri_output:
    planar_tri = remove_vertex(closed_tri, 0)  # Always vertex 0
    test_realizability(planar_tri)
```

**Why this is correct**:
- plantri generates one closed triangulation per isomorphism class
- Removing a fixed vertex (vertex 0) gives one **canonical** planar triangulation per closed triangulation
- Perfect 1-to-1 correspondence: each closed type → one planar type
- No isomorphism duplication
- Correct statistical sampling

**Impact**: 10-12× speedup + unbiased statistics

### 4. Connectivity Filtering

plantri supports connectivity filtering:
- `-c3`: 3-connected triangulations (every vertex cut requires ≥3 vertices)
- `-c4`: 4-connected triangulations (every vertex cut requires ≥4 vertices)

Higher connectivity generally correlates with higher Delaunay realizability:
- 3-connected: ~74-75% realizable (n=13)
- 4-connected: ~90% realizable (n=15)

### 5. Parallel Testing

We use Python multiprocessing to parallelize realizability testing:
- Split planar triangulations into chunks
- Each worker tests its chunk independently
- Aggregate results at the end

**Performance**: With 30 workers on modern hardware:
- ~0.04-0.06 ms per triangulation (amortized)
- Can test millions of triangulations in minutes
- Near-linear speedup for large batches

## Implementation

### Core Script: `analyze_delaunay_parallel.py`

```bash
# 3-connected analysis (exhaustive up to time limit)
python examples/analyze_delaunay_parallel.py \
    --min-n 4 \
    --max-n 20 \
    --workers 30 \
    --connectivity 3 \
    --time-limit 7200 \
    --output results/delaunay_3connected_optimized.json

# 4-connected analysis
python examples/analyze_delaunay_parallel.py \
    --min-n 4 \
    --max-n 20 \
    --workers 30 \
    --connectivity 4 \
    --time-limit 7200 \
    --output results/delaunay_4connected_optimized.json
```

### Parameters

- `--min-n`: Minimum number of vertices (default: 4)
- `--max-n`: Maximum number of vertices (default: 20)
- `--workers`: Number of parallel workers (default: CPU count)
- `--connectivity`: Minimum connectivity (3 or 4)
- `--time-limit`: Maximum runtime in seconds (default: 7200 = 2 hours)
- `--output`: JSON file for results

### plantri Integration

The script calls plantri as a subprocess:

```python
# Generate 3-connected closed triangulations with n vertices
cmd = ['plantri', '-c3', '-m2', str(n)]
result = subprocess.run(cmd, capture_output=True, text=True)

# Parse planar_code format output
closed_tris = parse_planar_code(result.stdout)
```

Flags:
- `-c3` or `-c4`: Minimum connectivity
- `-m2`: Minimum degree 2 (default for triangulations)
- `-q`: Quiet mode (suppress plantri's own output)

## Results

### 3-Connected Triangulations

**Completed**: n=4 through n=13 (2.06 hours)

| n  | Closed | Planar  | Realizable | % Real | Time(s) |
|----|--------|---------|------------|--------|---------|
| 4  | 1      | 1       | 1          | 100.0% | 0.0     |
| 5  | 2      | 2       | 2          | 100.0% | 0.0     |
| 6  | 7      | 7       | 7          | 100.0% | 0.0     |
| 7  | 34     | 34      | 34         | 100.0% | 0.1     |
| 8  | 256    | 255     | 244        | 95.7%  | 0.1     |
| 9  | 2,606  | 2,598   | 2,366      | 91.1%  | 0.2     |
| 10 | 32,298 | 32,276  | 28,166     | 87.3%  | 1.8     |
| 11 | 440,562| 440,348 | 364,421    | 82.8%  | 25.8    |
| 12 | 6,384,625 | 6,383,736 | 5,017,844 | 78.6% | 407.5 |
| 13 | 96,262,921 | 96,258,336 | 71,673,533 | 74.5% | 6677.4 |

**Key observations**:
- Perfect realizability (100%) for n ≤ 7
- Declining realizability: 100% → 74.5% as n increases
- Exponential growth in number of triangulations
- First non-realizable triangulation appears at n=8

### 4-Connected Triangulations

**Completed**: n=4 through n=15 (stopped during n=16 after 143.4s)

| n  | Closed | Planar  | Realizable | % Real | Time(s) |
|----|--------|---------|------------|--------|---------|
| 4  | 0      | 0       | 0          | -      | 0.04    |
| 5  | 0      | 0       | 0          | -      | 0.04    |
| 6  | 1      | 1       | 1          | 100.0% | 0.04    |
| 7  | 1      | 1       | 1          | 100.0% | 0.04    |
| 8  | 4      | 4       | 4          | 100.0% | 0.05    |
| 9  | 10     | 10      | 10         | 100.0% | 0.05    |
| 10 | 53     | 53      | 53         | 100.0% | 0.07    |
| 11 | 292    | 292     | 289        | 99.0%  | 0.09    |
| 12 | 2,224  | 2,224   | 2,184      | 98.2%  | 0.18    |
| 13 | 18,493 | 18,493  | 17,739     | 95.9%  | 1.09    |
| 14 | 167,504| 167,504 | 156,320    | 93.3%  | 9.47    |
| 15 | 1,571,020 | 1,571,020 | 1,419,450 | 90.4% | 92.64 |

**Key observations**:
- No 4-connected triangulations exist for n < 6 (octahedron is the minimum)
- Perfect realizability (100%) for n ≤ 10
- Much higher realizability than 3-connected (90.4% vs 74.5%)
- Fewer total triangulations (4-connectivity is more restrictive)
- First non-realizable triangulation appears at n=11

### Comparison: 3-Connected vs 4-Connected

At n=13 (largest common vertex count):
- **3-connected**: 74.5% realizable (96M triangulations tested)
- **4-connected**: 95.9% realizable (18K triangulations tested)

**Interpretation**: Higher connectivity → higher rigidity → more likely to be Delaunay realizable.

## Computational Performance

### Hardware Requirements

- **CPU**: Multi-core processor (tested with 30 workers)
- **Memory**: ~2-4 GB for n ≤ 15
- **Disk**: ~100 MB for result files

### Runtime Scaling

Approximate runtimes (30 workers):

| n  | 3-connected | 4-connected |
|----|-------------|-------------|
| 10 | 1.8s        | 0.07s       |
| 11 | 26s         | 0.09s       |
| 12 | 408s (7min) | 0.18s       |
| 13 | 6677s (1.9h)| 1.09s       |
| 14 | -           | 9.5s        |
| 15 | -           | 93s         |

### Bottlenecks

1. **plantri generation**: Dominates for large n (n=13: ~1749s generation vs ~4928s testing)
2. **Memory**: plantri must generate all triangulations before we can process them
3. **LP solving**: Each realizability test is ~0.04-0.06 ms (highly optimized)

### Optimization History

- **Original approach**: Remove all n vertices → test n× triangulations
- **Optimized approach**: Remove only vertex 0 → test 1× triangulations
- **Speedup**: 10-12× faster
- **Bonus**: Correct statistics (no isomorphism bias)

## Key Insights

### 1. Why Removing Vertex 0 is Correct

plantri generates graphs modulo isomorphism. Each output represents an **isomorphism class** of closed triangulations. When we remove vertex 0, we get one canonical planar triangulation per isomorphism class.

**Example**: Consider the octahedron (6 vertices). All vertices are equivalent under symmetry. Removing any vertex gives the same planar triangulation (a triangulated pentagon). We only need to test it once, not 6 times.

### 2. Vertex Counting Convention

**Important**: n includes the vertex at infinity.

For n=10:
- 9 finite vertices in the planar triangulation
- 1 vertex at infinity (the removed vertex 0)
- Total: 10 vertices in the original closed triangulation

This matches Rivin's theorem: a planar triangulation with k finite vertices + 1 at infinity corresponds to a convex (k+1)-vertex polyhedron.

### 3. Statistical Interpretation

The realizability fraction answers:

> "If I pick a random n-vertex 3-connected planar triangulation, what's the probability it's Delaunay realizable?"

**Not** (our old approach):

> "If I pick a random vertex to remove from a random closed triangulation..."

The difference is subtle but critical for unbiased statistics.

### 4. Why Realizability Declines

As n increases, triangulations become more complex:
- More constraints in Rivin's LP
- More opportunities for constraint violations
- Harder to satisfy Delaunay edge condition (opposite angles ≤ π)

Higher connectivity partially counteracts this:
- Fewer degrees of freedom
- More rigid structure
- Harder to have "bad" edge configurations

## Theoretical Context

### Connection to Rivin's Theorem

Rivin's 1996 theorem characterizes ideal polyhedra in hyperbolic 3-space:

> A planar triangulation is Delaunay realizable ⟺ it's the 1-skeleton of an ideal convex polyhedron in ℍ³

This connects our combinatorial question to hyperbolic geometry and volume computations.

### Connection to Steinitz's Theorem

Steinitz (1922): A graph is the 1-skeleton of a convex 3-polytope ⟺ it's planar and 3-connected.

Our analysis refines this: Among all Steinitz graphs (planar 3-connected), we identify the fraction that are also Delaunay realizable.

### Open Questions

1. **Asymptotic behavior**: Does realizability → 0 as n → ∞?
2. **Sharp thresholds**: Is there a connectivity k where realizability stays bounded away from 0?
3. **Characterization**: Can we characterize which triangulations are realizable without LP solving?
4. **Random generation**: Can we efficiently sample uniformly from realizable triangulations?

## Files and Dependencies

### Dependencies

- **plantri**: Brinkmann & McKay's planar graph generator
  - Download: http://users.cecs.anu.edu.au/~bdm/plantri/
  - Compile: `gcc -o plantri -O4 plantri.c`
  - Place in PATH or specify location

- **Python packages**:
  - `numpy`: Numerical operations
  - `scipy`: Linear programming (Rivin's criterion)
  - `networkx`: Graph operations (parsing plantri output)

### Output Format

Results are saved as JSON with this structure:

```json
{
  "parameters": {
    "min_n": 4,
    "max_n": 20,
    "workers": 30,
    "min_connectivity": 3,
    "time_limit": 7200.0,
    "actual_runtime": 7433.26
  },
  "results": [
    {
      "n_vertices": 10,
      "min_connectivity": 3,
      "closed_triangulations": 32298,
      "total_planar_triangulations": 32276,
      "tested": 32276,
      "realizable": 28166,
      "non_realizable": 4110,
      "errors": 0,
      "fraction_realizable": 0.8727,
      "generation_time": 0.389,
      "testing_time": 1.450,
      "n_workers": 30
    },
    ...
  ]
}
```

### Result Files

- `results/delaunay_3connected_optimized.json`: 3-connected analysis (n=4-13)
- `results/delaunay_4connected_optimized.json`: 4-connected analysis (n=4-15)
- `logs_3connected_optimized.txt`: Detailed log for 3-connected run
- `logs_4connected_optimized.txt`: Detailed log for 4-connected run

## Usage Examples

### Basic Usage

```bash
# Run optimized analysis for small vertex counts
python examples/analyze_delaunay_parallel.py \
    --min-n 6 \
    --max-n 10 \
    --connectivity 3 \
    --workers 8
```

### Production Run

```bash
# Full analysis with time limit and output
python examples/analyze_delaunay_parallel.py \
    --min-n 4 \
    --max-n 20 \
    --workers 30 \
    --connectivity 4 \
    --time-limit 7200 \
    --output results/my_analysis.json \
    > logs_my_analysis.txt 2>&1
```

### Background Execution

```bash
# Run in background (Linux/Mac)
nohup python examples/analyze_delaunay_parallel.py \
    --min-n 4 \
    --max-n 20 \
    --workers 30 \
    --connectivity 3 \
    --time-limit 7200 \
    --output results/delaunay_3connected.json \
    > logs_3connected.txt 2>&1 &

# Check progress
tail -f logs_3connected.txt
```

## Future Directions

1. **Extend to higher n**:
   - n=14 for 3-connected (estimated ~20 hours)
   - n=16 for 4-connected (estimated ~2-3 hours)

2. **Random sampling**:
   - Implement Boltzmann sampling for n > 20
   - Compare statistics: exhaustive vs sampled

3. **Characterization**:
   - Identify graph-theoretic features predicting realizability
   - Machine learning model to predict realizability

4. **Volume analysis**:
   - For realizable triangulations, compute hyperbolic volume distribution
   - Relate combinatorial features to volume

5. **Extremal triangulations**:
   - Find triangulations with maximum/minimum volume
   - Characterize "most non-realizable" triangulations

## References

1. **Rivin, I. (1996)**. "A characterization of ideal polyhedra in hyperbolic 3-space." *Annals of Mathematics*, 143(1), 51-70.
   - The fundamental theorem connecting Delaunay realizability to hyperbolic geometry

2. **Brinkmann, G. & McKay, B.D. (2007)**. "Fast generation of planar graphs." *MATCH Commun. Math. Comput. Chem.*, 58(2), 323-357.
   - plantri algorithm and implementation

3. **Steinitz, E. (1922)**. "Polyeder und Raumeinteilungen." *Encyclopädie der mathematischen Wissenschaften*, 3, 1-139.
   - Steinitz's theorem on 3-polytope graphs

4. **Hodgson, C., Rivin, I., & Smith, W.D. (1992)**. "A characterization of convex hyperbolic polyhedra and of convex polyhedra inscribed in the sphere." *Bulletin of the AMS*, 27(2), 246-251.
   - Extended characterization results

## Acknowledgments

This analysis was made possible by:
- **plantri** (Brinkmann & McKay): Fast planar graph generation
- **SciPy HiGHS**: Efficient LP solver for Rivin's criterion
- Critical insight on isomorphism issue: Recognizing that plantri's isomorphism-free output + removing a fixed vertex = unbiased statistics
