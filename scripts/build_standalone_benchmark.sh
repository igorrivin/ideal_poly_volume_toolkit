#!/bin/bash
#
# Build a self-contained care package for the arithmeticity benchmark.
#
# The output directory can be handed to anyone with Python + pip;
# no knowledge of the parent repo is required.
#
# Usage (from repo root):
#   bash scripts/build_standalone_benchmark.sh [output_dir]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-${REPO_ROOT}/standalone_benchmark}"

rm -rf "$OUT"
mkdir -p "$OUT"

# --- Copy library modules (unmodified) ---
cp "$REPO_ROOT/ideal_poly_volume_toolkit/geometry.py"       "$OUT/"
cp "$REPO_ROOT/ideal_poly_volume_toolkit/rivin_delaunay.py" "$OUT/"
cp "$REPO_ROOT/ideal_poly_volume_toolkit/rivin_holonomy.py" "$OUT/"

# --- Copy and patch the benchmark script ---
# Replace  from ideal_poly_volume_toolkit.X import ...  →  from X import ...
# Drop the sys.path hack (no longer needed in a flat directory).
sed -e 's/from ideal_poly_volume_toolkit\./from /g' \
    -e '/sys\.path\.insert/d' \
    "$REPO_ROOT/scripts/generate_arithmeticity_benchmark.py" \
    > "$OUT/generate_arithmeticity_benchmark.py"
chmod +x "$OUT/generate_arithmeticity_benchmark.py"

# --- requirements.txt ---
cat > "$OUT/requirements.txt" <<'EOF'
numpy>=1.20
scipy>=1.7
torch>=1.9
mpmath>=1.2
EOF

# --- Minimal usage note ---
cat > "$OUT/README" <<'EOF'
Arithmeticity Benchmark Generator
==================================

Generates a JSON benchmark testing LLM knowledge of arithmetic ideal
polyhedra in hyperbolic 3-space (Rivin-Delaunay theorem).

Setup:
  pip install -r requirements.txt

Usage:
  python generate_arithmeticity_benchmark.py --n-vertices 10
  python generate_arithmeticity_benchmark.py --n-vertices 20 --n-random 15 --seed 123 -o out.json

The output JSON separates each configuration into "input" (vertices,
triangles) and "ground_truth" (is_arithmetic, volume, holonomy_traces).
See the docstring at the top of generate_arithmeticity_benchmark.py
for the full schema.
EOF

echo "Standalone benchmark package created in: $OUT"
echo "  $(find "$OUT" -type f | wc -l) files, $(du -sh "$OUT" | cut -f1) total"
