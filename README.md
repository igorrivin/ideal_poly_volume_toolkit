---
title: Ideal Polyhedron Volume Toolkit
emoji: 🔺
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# Ideal Polyhedra Volume Toolkit

A Python toolkit for computing and optimizing volumes of ideal hyperbolic
polyhedra using Delaunay triangulation, pole-aware sphere projection, and
Bloch–Wigner/Lobachevsky volume kernels.

Supported Python versions: 3.10 through 3.14.

## Installation

Install the NumPy/SciPy/mpmath numerical core in development mode:

```bash
python3 -m pip install -e .
```

Torch is optional for the ordinary H³ volume and optimization paths. Install
it only for differentiable geometry or the Torch-based higher-dimensional
modules:

```bash
python3 -m pip install -e '.[torch]'
```

Other feature extras are `plot`, `gui`, `graph`, `render`, and `rl`; extras
can be combined, for example `python3 -m pip install -e '.[gui,graph]'` for a
GUI with symmetry analysis. The `dev` extra is the complete test/lint/GUI
environment used by CI:

```bash
python3 -m pip install -e '.[dev]'
```

For RL training and the broader research workflow, install the additional
repository requirements first:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

`requirements_HPC.txt` is a machine-oriented environment snapshot, not the
canonical installation path.

Release history is recorded in
[`CHANGELOG.md`](https://github.com/igorrivin/ideal_poly_volume_toolkit/blob/main/CHANGELOG.md).
The draft 0.4
migration guide and release gates are in
[`docs/release-0.4.0.md`](https://github.com/igorrivin/ideal_poly_volume_toolkit/blob/main/docs/release-0.4.0.md).

## Project Structure

```
ideal_poly_volume_toolkit/
├── ideal_poly_volume_toolkit/   # Core package
│   ├── geometry.py              # Core geometry and volume computation functions
│   ├── geometry_torch.py        # Optional differentiable H³ Torch kernels
│   ├── optimization.py          # Canonical reusable H³ optimization service
│   ├── visualization.py         # 3D visualization utilities
│   ├── rivin_holonomy.py        # Penner-Rivin holonomy computation
│   ├── pointset_to_fuchsian.py  # Full Fuchsian group pipeline
│   ├── rl/                       # Generic, polytope, and cage RL features
│   └── __init__.py
│
├── bin/                         # Command-line tools and GUI
│   ├── gui.py                   # 🎨 Interactive Gradio web interface
│   ├── optimize_polyhedron.py  # General optimization wrapper
│   ├── analyze_distribution.py # Distribution analysis wrapper
│   └── README.md
│
├── examples/                    # Organized example scripts
│   ├── distributions/          # Distribution analysis examples
│   │   ├── tetrahedron/       # Tetrahedron volume distributions
│   │   ├── five_vertex/       # 5-vertex polyhedra distributions
│   │   ├── six_vertex/        # 6-vertex polyhedra distributions
│   │   └── euclidean/         # Euclidean tetrahedra analysis
│   ├── optimization/           # Optimization examples by vertex count
│   │   ├── 7vertex/           # 7-vertex optimization (octahedron variants)
│   │   ├── 12vertex/          # 12-vertex optimization
│   │   ├── 20vertex/          # 20-vertex optimization (icosahedron)
│   │   └── platonic/          # Platonic solid analysis
│   ├── visualization/          # Visualization scripts
│   └── analysis/               # Statistical and theoretical analysis
│
├── scripts/                     # Active research/development scripts
├── results/                     # Output files
│   ├── data/                   # JSON configuration files
│   ├── plots/                  # PNG visualization outputs
│   └── logs/                   # Optimization logs
└── docs/                        # Documentation
    ├── architecture.md          # Ownership and computational pipelines
    ├── module-index.md          # Discoverable module/entry-point index
    └── implementation-status.md # Working features and known limitations
```

## Core Functionality

### `ideal_poly_volume_toolkit.geometry` - Volume Computation

- **Stereographic projection**: `lift_to_sphere_with_inf()`, `inverse_stereographic_from_sphere_pts()`
- **Pole-aware sphere projection**: `stereographic_project_from_pole()`
- **Triangulation**: `delaunay_triangulation_indices()`, `hull_tris_projected_back()`
- **Bloch–Wigner kernel**: `bloch_wigner()` (vectorized SciPy path)
- **Lobachevsky function**: `lob_fast()` (lazy PyTorch autodiff from the
  `torch` extra), `lob_exact()` (mpmath high-precision)
- **Volume computation**:
  - `triangle_volume_from_points()` - Single triangle volume
  - `ideal_poly_volume_via_delaunay()` - Full polyhedron via Delaunay
  - `ideal_poly_volume_from_sphere()` - Rotation/pole-invariant sphere entry
  - `ideal_poly_volume_via_hull_project_back()` - Full polyhedron via convex hull

### `ideal_poly_volume_toolkit.optimization` - H³ Optimization

- `negative_volume_objective()` - Validated vectorized SciPy objective
- `optimize_ideal_polyhedron()` - Repeated differential-evolution service
- `analyze_ideal_polyhedron_structure()` - Full boundary combinatorics,
  including the implicit infinity vertex
- `optimization_result_to_dict()` - Shared, versioned CLI/GUI result schema

### `ideal_poly_volume_toolkit.rivin_holonomy` - Penner-Rivin Algorithm

- **Holonomy computation**: `generators_from_triangulation()` - Compute Fuchsian group generators
- **Exact zero-shear certification**:
  `certify_exact_zero_shear_structure()` validates a closed triangulated
  sphere and evaluates its common-frame PSL(2,Z) generators with Python
  integers. Coordinate-derived shears remain a separate numerical check;
  near-integral traces of one generator basis are diagnostic, not a proof.
- **Triangulation structures**: `Triangulation` class for managing ideal triangulations

The versioned, resumable arithmetic-corpus producer is
`scripts/generate_arithmeticity_benchmark.py`. Its `--n-vertices` count is the
total number of ideal vertices, including the explicit point at infinity.

### `ideal_poly_volume_toolkit.pointset_to_fuchsian` - Full Pipeline

- **Group computation**: `group_from_pointset()` - Convert point sets to Fuchsian groups
- **Trace field analysis**: `invariant_trace_field_signature()` - Analyze arithmetic properties
- **Visualization**: `render_snapshot()` - High-quality rendering with iridescence and transparency
- **Mesh export**: `hull_to_mesh()`, `export_mesh_obj()` - Export to OBJ format

## Quick Start

### 🎨 Interactive GUI (Easiest)

The fastest way to get started is with the Gradio web interface:

```bash
python bin/gui.py
```

Then open your browser to `http://127.0.0.1:7860`

**Features:**
- Interactive optimization with real-time progress
- Distribution analysis with automatic plotting
- 3D visualization in sphere and Poincaré ball models
- No need to remember command-line arguments!

### Command-Line Tools

For scripting and batch processing, use the installed optimizer command and
the remaining source-tree wrappers in `bin/`:

```bash
# Optimize a 7-vertex polyhedron (10 trials)
ideal-poly-optimize --vertices 7 --trials 10

# Analyze volume distribution for tetrahedra
python bin/analyze_distribution.py --vertices 4 --samples 10000

# Get optimizer help
ideal-poly-optimize --help
```

`ideal-poly-optimize` works outside a repository checkout. The historical
`python bin/optimize_polyhedron.py ...` spelling remains supported and
delegates to the same package-owned adapter.

See `bin/README.md` for detailed usage and examples.

### Computing a volume (Python API)

```python
import numpy as np
from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay

# Define vertices in the complex plane (stereographic projection)
vertices = np.array([0.0+0.0j, 1.0+0.0j, 0.5+0.866j])
volume = ideal_poly_volume_via_delaunay(vertices, mode="bloch_wigner")
print(f"Volume: {volume}")
```

### Running tests

```bash
python3 -m pytest -q
```

### Reproducing the H^3 objective benchmark

The benchmark compares the current vectorized objective with a frozen copy of
the 0.3 scalar Torch path on deterministic 7-, 20-, and 50-vertex inputs:

```bash
python3 scripts/benchmark_h3_optimization.py
```

It writes no files by default. Pass `--output /tmp/h3-benchmark.csv` to retain
the raw `perf_counter_ns` samples and their source/environment provenance. See
the 0.4 release note before recording a canonical baseline.

### Running examples

Examples are organized by topic. For instance:

```bash
# 7-vertex optimization
cd examples/optimization/7vertex
python optimize_7vertex.py

# Tetrahedron distribution analysis
cd examples/distributions/tetrahedron
python tetrahedron_volume_distribution.py

# Visualization
cd examples/visualization
python visualize_golden_config.py
```

## Key Examples

- **7-vertex optimization**: Testing the hypothesis that the maximum volume is an octahedron with one stellated face
- **20-vertex optimization**: Finding maximal volume configurations for icosahedron-like polyhedra
- **Distribution analysis**: Statistical analysis of volume distributions for various polyhedra
- **Platonic solids**: Analysis of regular polyhedra and their perturbations

## Research Results

See `docs/RESULTS_SUMMARY.md` and `docs/PLATONIC_MAXIMALITY_RESULTS.md` for detailed findings.

## License

See LICENSE file.
