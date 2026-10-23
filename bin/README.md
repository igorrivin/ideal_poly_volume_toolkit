# Command-Line Tools

This directory contains general-purpose wrapper scripts and GUI for common tasks.

The reusable H^3 optimization implementation lives in
`ideal_poly_volume_toolkit.optimization`. The CLI and GUI adapt that service;
numerical objectives, trial selection, and positive-volume result accounting
must stay in the package module. The underlying volume kernel lives in
`ideal_poly_volume_toolkit.geometry`.

## 🎨 Interactive GUI

### `gui.py`

**Launch the interactive Gradio web interface:**

```bash
# Local only (default)
python bin/gui.py

# Create shareable public link
python bin/gui.py --share

# Custom port
python bin/gui.py --port 8080

# Get help
python bin/gui.py --help
```

The GUI will open in your browser at `http://127.0.0.1:7860` (or your custom port)

**Features:**
- **Optimization Tab**: Find maximal volume configurations with adjustable parameters
  - Number of vertices, trials, iterations, and SciPy population multiplier
  - Real-time progress tracking
  - Automatic result saving

- **Distribution Analysis Tab**: Sample random configurations and analyze statistics
  - Configurable sample size
  - Automatic histogram generation
  - Statistics: mean, median, quartiles, std dev

- **Fixed Combinatorics Tab**: Optimize a fixed Rivin-Delaunay triangulation

- **Arithmeticity Tab**: Inspect Penner-Rivin holonomy and symmetry data

- **3D Visualization Tab**: Interactive polyhedron visualization
  - Delaunay triangulation (2D complex plane)
  - Sphere projection (3D)
  - Poincaré ball model (3D hyperbolic geometry)
  - Adjustable subdivision for smooth surfaces
  - Load results directly from optimization

- **About Tab**: Documentation and usage tips

**Advantages:**
- No need to remember command-line arguments
- Visual feedback and progress bars
- Interactive 3D plots you can rotate and zoom
- Seamless workflow: optimize → visualize
- Beginner-friendly interface

---

## Available Tools

### `optimize_polyhedron.py`

General-purpose optimization wrapper for finding maximal volume configurations.
The installed `ideal-poly-optimize` command and this source-tree script both
delegate to `ideal_poly_volume_toolkit.optimization_cli`; output volumes are
always positive even though SciPy minimizes an internal negative-volume
objective.

**Usage:**
```bash
# Optimize a 7-vertex polyhedron with 10 trials
ideal-poly-optimize --vertices 7 --trials 10

# Optimize 12-vertex with more iterations
ideal-poly-optimize --vertices 12 --trials 20 --maxiter 300

# Custom output location
ideal-poly-optimize --vertices 20 --trials 5 --output my_results.json
```

From a checkout, `python bin/optimize_polyhedron.py ...` remains a compatible
spelling with the same arguments and output schema.

**Arguments:**
- `--vertices, -v`: Number of vertices (required, must be >= 4)
- `--trials, -t`: Number of optimization trials (default: 10)
- `--maxiter, -m`: Max iterations per trial (default: 200)
- `--popsize, -p`: SciPy population multiplier; the nominal population is
  `popsize × parameter dimension` (default: 15)
- `--workers, -w`: Worker processes (`-1` uses all available CPUs; default: 1)
- `--output, -o`: Output JSON file (default: auto-generated in results/data/)
- `--seed, -s`: Random seed base (default: 42)

**Output:**
Both the CLI and GUI use schema version 1 and source-specific filenames. The
legacy `triangulation` key remains an alias for
`finite_delaunay_triangulation`. Saves a JSON file with:
- Best configuration found (positive volume, parameters, and finite complex
  vertices; the vertex at infinity is implicit)
- `finite_delaunay_triangulation`, containing only finite Delaunay faces
- `boundary_triangulation`, the full spherical boundary with the explicit
  infinity-vertex index recorded in the metadata
- Combinatorial structure for that full boundary (faces, edges, and Euler
  characteristic)
- All trial results; invalid penalty-only trials have `volume: null` and are
  excluded from aggregate volume statistics

---

### `analyze_distribution.py`

Analyze volume distributions by sampling random polyhedra configurations.

**Usage:**
```bash
# Analyze 10,000 random tetrahedra
python bin/analyze_distribution.py --vertices 4 --samples 10000

# Analyze with custom output
python bin/analyze_distribution.py --vertices 6 --samples 50000 --output my_plot.png

# Include reference volume and save data
python bin/analyze_distribution.py --vertices 5 --samples 20000 --reference 3.66 --data distribution_data.json
```

**Arguments:**
- `--vertices, -v`: Number of vertices (required, must be >= 4)
- `--samples, -n`: Number of random samples (default: 10000)
- `--seed, -s`: Random seed (default: 42)
- `--output, -o`: Output plot file (default: auto-generated in results/plots/)
- `--data, -d`: Output data JSON file (optional)
- `--reference, -r`: Reference volume to mark on plot (optional)
- `--series-terms`: Deprecated compatibility option; ignored by the current
  vectorized Bloch–Wigner kernel
- `--fit, -f`: Optional fitted distribution (`beta`, `gamma`, `lognorm`, or `norm`)
- `--bootstrap, -b`: Bootstrap samples for fitted-parameter intervals (default: 1000)
- `--confidence, -c`: Confidence level for fitted-parameter intervals (default: 0.95)
- `--jobs, -j`: Parallel worker count. The automatic default uses no more
  than 16 workers or the number of useful logical batches; use 1 for serial
  execution.

**Output:**
- PNG plot with histogram and box plot
- Optional JSON file with statistics and all volume samples

Sampling is split into fixed 256-configuration logical batches, each seeded
from the CLI seed and batch index. Therefore the same seed and sample request
produce the same ordered volumes and rejection counters for serial and
parallel runs. Rejected vertex draws are retried up to 100 times per required
vertex, and rejected configurations are retried up to 20 attempts per requested
valid sample within each batch. Saved metadata distinguishes rejected draws,
rejected configurations, known numerical-evaluation failures, invalid volume
evaluations, and the effective worker count. If those finite limits are
exhausted, `status` is `attempt_limit_reached` and `n_missing` is nonzero;
under-filling is never reported as a complete run. Bootstrap confidence
intervals use a function-local generator initialized from `--seed`.

---

### `sample_size_calculator.py`

Estimate the sample count needed to reach a requested confidence-interval
half-width from a pilot distribution JSON file:

```bash
python bin/sample_size_calculator.py results/data/pilot.json --precision 0.01
```

Use `--confidence` to select a confidence level; the default is 0.95.

---

## Examples

### Quick Optimization Run
```bash
# Find best 7-vertex configuration with 5 quick trials
python bin/optimize_polyhedron.py -v 7 -t 5 --maxiter 100
```

### Distribution Analysis Pipeline
```bash
# Analyze distribution, save data, and use max volume as reference
python bin/analyze_distribution.py -v 5 -n 10000 --data dist_data.json
```

### Reproduce Research Results
```bash
# Optimize different vertex counts systematically
for n in 5 6 7 8; do
    python bin/optimize_polyhedron.py -v $n -t 20 -m 300
done
```

---

## Output Conventions

All outputs are saved to standardized locations:

- **Optimization results**: `results/data/{n}vertex_optimization_{cli|gui}_TIMESTAMP.json`
- **Distribution plots**: `results/plots/{n}vertex_distribution_TIMESTAMP.png`
- **Distribution data**: `results/data/{n}vertex_distribution_TIMESTAMP.json`

Optimization timestamps use `YYYYMMDD_HHMMSS_microseconds`; other tools retain
the shorter `YYYYMMDD_HHMMSS` form.
