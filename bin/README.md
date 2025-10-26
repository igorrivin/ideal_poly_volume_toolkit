# Command-Line Tools

This directory contains general-purpose wrapper scripts and GUI for common tasks.

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
  - Number of vertices, trials, iterations
  - Real-time progress tracking
  - Automatic result saving

- **Distribution Analysis Tab**: Sample random configurations and analyze statistics
  - Configurable sample size
  - Automatic histogram generation
  - Statistics: mean, median, quartiles, std dev

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

**Usage:**
```bash
# Optimize a 7-vertex polyhedron with 10 trials
python bin/optimize_polyhedron.py --vertices 7 --trials 10

# Optimize 12-vertex with more iterations
python bin/optimize_polyhedron.py --vertices 12 --trials 20 --maxiter 300

# Custom output location
python bin/optimize_polyhedron.py --vertices 20 --trials 5 --output my_results.json
```

**Arguments:**
- `--vertices, -v`: Number of vertices (required, must be >= 4)
- `--trials, -t`: Number of optimization trials (default: 10)
- `--maxiter, -m`: Max iterations per trial (default: 200)
- `--popsize, -p`: Population size for differential evolution (default: 15)
- `--output, -o`: Output JSON file (default: auto-generated in results/data/)
- `--seed, -s`: Random seed base (default: 42)

**Output:**
Saves a JSON file with:
- Best configuration found (volume, parameters, vertex coordinates)
- Combinatorial structure (faces, edges, Euler characteristic)
- All trial results

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
- `--vertices, -v`: Number of vertices (required, must be >= 3)
- `--samples, -n`: Number of random samples (default: 10000)
- `--seed, -s`: Random seed (default: 42)
- `--output, -o`: Output plot file (default: auto-generated in results/plots/)
- `--data, -d`: Output data JSON file (optional)
- `--reference, -r`: Reference volume to mark on plot (optional)
- `--series-terms`: Number of series terms for Lobachevsky function (default: 96)

**Output:**
- PNG plot with histogram and box plot
- Optional JSON file with statistics and all volume samples

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

- **Optimization results**: `results/data/{n}vertex_optimization_TIMESTAMP.json`
- **Distribution plots**: `results/plots/{n}vertex_distribution_TIMESTAMP.png`
- **Distribution data**: `results/data/{n}vertex_distribution_TIMESTAMP.json`

Timestamps are in format: `YYYYMMDD_HHMMSS`
