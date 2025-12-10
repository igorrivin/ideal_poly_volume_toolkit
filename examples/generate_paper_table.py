#!/usr/bin/env python3
"""
Generate paper-ready table of maximal volume ideal polyhedra.

This script reads the comprehensive optimization results and generates
a LaTeX-formatted table suitable for academic papers.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/igor/devel/ideal_poly_volume_toolkit')


def load_comprehensive_results(directory='results/data/comprehensive'):
    """Load all comprehensive optimization results."""
    results = {}
    dir_path = Path(directory)

    for filepath in sorted(dir_path.glob('*vertex_comprehensive.json')):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            n = data['statistics']['n_vertices']
            results[n] = {
                'volume': data['statistics']['best_volume'],
                'n_faces': data['statistics']['n_faces'],
                'n_triangles': data['statistics']['n_faces'],
                'mean_volume': data['statistics']['mean_volume'],
                'std_volume': data['statistics']['std_volume'],
                'convergence': data['statistics']['convergence_ratio'],
                'n_trials': data['statistics']['n_trials'],
                'vertices_real': data['best'].get('vertices_real', []),
                'vertices_imag': data['best'].get('vertices_imag', []),
            }
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")

    return results


def volume_ratios(results):
    """Compute volume ratios relative to tetrahedron."""
    V4 = results.get(4, {}).get('volume', 1.01494161)
    ratios = {}
    for n, data in results.items():
        ratios[n] = data['volume'] / V4
    return ratios


def generate_latex_table(results):
    """Generate LaTeX table."""
    ratios = volume_ratios(results)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Maximal volume ideal hyperbolic polyhedra with $n$ vertices}",
        r"\label{tab:maximal-volumes}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"$n$ & Volume & Faces & $V/V_4$ & Convergence \\",
        r"\midrule",
    ]

    for n in sorted(results.keys()):
        data = results[n]
        ratio = ratios[n]
        conv = data['convergence'] * 100

        lines.append(
            f"{n} & {data['volume']:.6f} & {data['n_faces']} & "
            f"{ratio:.4f} & {conv:.0f}\\% \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return '\n'.join(lines)


def generate_markdown_table(results):
    """Generate Markdown table."""
    ratios = volume_ratios(results)

    lines = [
        "| n | Volume | Faces | V/V₄ | Convergence |",
        "|---|--------|-------|------|-------------|",
    ]

    for n in sorted(results.keys()):
        data = results[n]
        ratio = ratios[n]
        conv = data['convergence'] * 100

        lines.append(
            f"| {n} | {data['volume']:.6f} | {data['n_faces']} | "
            f"{ratio:.4f} | {conv:.0f}% |"
        )

    return '\n'.join(lines)


def generate_csv(results):
    """Generate CSV output."""
    ratios = volume_ratios(results)

    lines = ["n,volume,faces,volume_ratio,convergence"]

    for n in sorted(results.keys()):
        data = results[n]
        ratio = ratios[n]
        conv = data['convergence']

        lines.append(f"{n},{data['volume']:.8f},{data['n_faces']},{ratio:.6f},{conv:.4f}")

    return '\n'.join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate paper-ready table of maximal volumes"
    )
    parser.add_argument(
        "--data-dir", default="results/data/comprehensive",
        help="Directory with comprehensive results"
    )
    parser.add_argument(
        "--format", choices=["latex", "markdown", "csv", "all"],
        default="all",
        help="Output format"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file (stdout if not specified)"
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.data_dir}...", file=sys.stderr)
    results = load_comprehensive_results(args.data_dir)

    if not results:
        print("No results found!", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(results)} configurations", file=sys.stderr)

    # Generate output
    outputs = []

    if args.format in ["latex", "all"]:
        outputs.append("=" * 60)
        outputs.append("LaTeX Table:")
        outputs.append("=" * 60)
        outputs.append(generate_latex_table(results))
        outputs.append("")

    if args.format in ["markdown", "all"]:
        outputs.append("=" * 60)
        outputs.append("Markdown Table:")
        outputs.append("=" * 60)
        outputs.append(generate_markdown_table(results))
        outputs.append("")

    if args.format in ["csv", "all"]:
        outputs.append("=" * 60)
        outputs.append("CSV:")
        outputs.append("=" * 60)
        outputs.append(generate_csv(results))
        outputs.append("")

    output_text = '\n'.join(outputs)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_text)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
