#!/usr/bin/env python3
"""
Generate a construct-a-record-polytope LLM benchmark across five
(measure, dimension) regimes:

    Euclidean    d = 4   (inscribed in unit ball of R^4)
    Euclidean    d = 5
    Euclidean    d = 6
    Hyperbolic   d = 4   (ideal polytope in H^4)
    Hyperbolic   d = 6

For each (measure, d, N) pair the LLM is asked for the sharpest lower
bound it can give on the maximum volume, together with an *explicit*
set of N vertex coordinates realizing that bound.  The explicit
coordinates are the anti-hallucination device: the grader computes the
volume of the submitted polytope and compares it both to the LLM's
claim and to our record lower bound for that (measure, d, N).

Ground truth sources:
    Eucl d=4       optimal_d4_symmetry_analysis.json      (N = 5..24)
    Eucl d=5       optimal_d5_d6_symmetry_analysis.json   (N = 6..20)
    Eucl d=6       optimal_d5_d6_symmetry_analysis.json   (N = 7..24)
    Hyp  d=4       optimal_ideal_4polytopes.json          (N = 5..16, 20, 30)
    Hyp  d=6       ideal_H6_{first_results,combinatorial_invariance,
                             larger_N}.json               (N = 7..24 subset)

Hyperbolic d=5 is deliberately excluded: our d=5 hyperbolic ground
truth (`ideal_H5_polytope_results.json`) is known to be suboptimal at
N in {12, 14, 20} (see `tasks/todo.md` and Stage A commit message).
Adding it to the benchmark would grade the LLM against a number we
already know is not the true LB.

Output: `results/data/llm_challenges_volume_lb/benchmark.json`,
        `results/data/llm_challenges_volume_lb/ground_truth.json`,
        `results/data/llm_challenges_volume_lb/README.md`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

GENERATOR_VERSION = "1.0.0"

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "results" / "data" / "llm_challenges_volume_lb"


# ---------------------------------------------------------------------------
# Ground-truth loaders — one per (measure, d) source file
# ---------------------------------------------------------------------------

def load_eucl_d4() -> Dict[int, Tuple[float, str]]:
    fn = REPO / "results" / "data" / "optimal_d4_symmetry_analysis.json"
    data = json.loads(fn.read_text())
    out: Dict[int, Tuple[float, str]] = {}
    for key, entry in data.items():
        N = int(entry["N"])
        out[N] = (float(entry["euclidean_volume"]), fn.name)
    return out


def load_eucl_d5_d6() -> Tuple[Dict[int, Tuple[float, str]], Dict[int, Tuple[float, str]]]:
    fn = REPO / "results" / "data" / "optimal_d5_d6_symmetry_analysis.json"
    data = json.loads(fn.read_text())
    d5: Dict[int, Tuple[float, str]] = {}
    d6: Dict[int, Tuple[float, str]] = {}
    for key, entry in data.items():
        N = int(entry["N"])
        vol = float(entry["euclidean_volume"])
        if key.startswith("d5_"):
            d5[N] = (vol, fn.name)
        elif key.startswith("d6_"):
            d6[N] = (vol, fn.name)
    return d5, d6


def load_hyp_d3() -> Dict[int, Tuple[float, str]]:
    """H^3 records for max-volume ideal polyhedra.

    Sources:
      * N = 4..10 — `{N}vertex_comprehensive_combinatorics.json` (metadata.volume).
        Cross-checked to match paper Table 1 to 5+ digits.
      * N = 11, 12, 20, 30, 50, 70, 100 — paper Table 1
        (`paper/maximal_ideal_polyhedra.tex` lines 353–367).
      * N = 150, 200, 300 — `results/data/h3_large_N.txt`.
    """
    out: Dict[int, Tuple[float, str]] = {}

    for N in range(4, 11):
        fn = REPO / "results" / "data" / f"{N}vertex_comprehensive_combinatorics.json"
        if fn.exists():
            meta = json.loads(fn.read_text()).get("metadata", {})
            vol = meta.get("volume")
            if vol is not None:
                out[N] = (float(vol), fn.name)

    # Paper Table 1 values (authoritative).
    paper_values = {
        11: 11.449290,
        12: 13.529628,
        20: 27.699,
        30: 46.486,
        50: 84.971,
        70: 123.893,
        100: 182.671,
    }
    for N, v in paper_values.items():
        out[N] = (float(v), "paper/maximal_ideal_polyhedra.tex (Table 1)")

    # Large-N from the sep-CMA-ES log.
    large_fn = REPO / "results" / "data" / "h3_large_N.txt"
    if large_fn.exists():
        import re
        pat = re.compile(r"^N=(\d+):\s*vol=([\d.]+)")
        for line in large_fn.read_text().splitlines():
            m = pat.match(line)
            if not m:
                continue
            N = int(m.group(1))
            v = float(m.group(2))
            if N not in out or v > out[N][0]:
                out[N] = (v, large_fn.name)
    return out


def load_hyp_d4() -> Dict[int, Tuple[float, str]]:
    fn = REPO / "results" / "data" / "optimal_ideal_4polytopes.json"
    data = json.loads(fn.read_text())
    out: Dict[int, Tuple[float, str]] = {}
    for entry in data:
        N = int(entry["n_vertices"])
        out[N] = (float(entry["best_volume"]), fn.name)
    return out


def load_hyp_d6() -> Dict[int, Tuple[float, str]]:
    files = [
        REPO / "results" / "data" / "ideal_H6_polytope_first_results.json",
        REPO / "results" / "data" / "ideal_H6_combinatorial_invariance.json",
        REPO / "results" / "data" / "ideal_H6_larger_N.json",
    ]
    out: Dict[int, Tuple[float, str]] = {}
    for fn in files:
        for entry in json.loads(fn.read_text()):
            N = int(entry["N"])
            v = float(entry.get("best_hyp_vol", entry["hyp_vol"]))
            if N not in out or v > out[N][0]:
                out[N] = (v, fn.name)
    return out


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PROMPT = """\
Task: for each challenge below, construct an explicit convex polytope
with exactly N vertices and report your volume as a lower bound.

The challenges come in two flavours, distinguished by the `measure` field:

* measure = "euclidean": a convex polytope inscribed in the closed unit
  ball of R^d, i.e. all N vertices lie on or inside the unit sphere S^{d-1}.
  The reported volume is the Euclidean d-volume of the convex hull.

* measure = "hyperbolic": an ideal convex polytope in hyperbolic d-space
  H^d, i.e. all N vertices lie exactly on the sphere at infinity S^{d-1},
  interpreted as the boundary of the Klein-ball model.  The reported
  volume is the hyperbolic d-volume.

For each challenge you must provide:

  1. `LB`: your claimed lower bound (the volume your polytope achieves).
  2. `vertices`: a list of N explicit vertex coordinates in R^d.  This is
     mandatory — we compute the volume of your submitted polytope and
     cross-check it against your claim, which eliminates the option of
     guessing a number.

Return your answer as a JSON list of objects:

    [
      {"id": <int>, "measure": "euclidean"|"hyperbolic", "d": <int>,
       "N": <int>, "LB": <float>, "vertices": [[...], ...]},
      ...
    ]

Grading per (measure, d, N):

* The grader computes the volume V_comp of your submitted polytope using
  the measure specified.  We compare V_comp to the LB you reported:
  agreement within |V_comp - LB| / max(LB, 1e-12) <= 1e-3 is required,
  otherwise the challenge is marked "inconsistent" and scored on V_comp.
* We then compare V_comp to our record LB (withheld):
      V_comp >= 1.00 * record       -> "strong" (matches/beats our record)
      V_comp >= 0.95 * record       -> "pass"
      V_comp >  0                   -> "fail" (valid polytope, below 95%)
      V_comp <= 0 or invalid input  -> "invalid"
* For hyperbolic challenges, vertices must lie on the unit sphere within
  a tolerance of 1e-6 (after row normalization we check |\\|v_i\\| - 1| < 1e-6).
  Submissions with vertices off the sphere are normalised defensively,
  but the hyperbolic volume of a clearly non-ideal configuration may be
  small or zero.
"""


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_entries(sources: Dict[Tuple[str, int], Dict[int, Tuple[float, str]]]
                  ) -> Tuple[List[dict], List[dict]]:
    """Return (challenges_for_prompt, ground_truth_with_records)."""
    challenges: List[dict] = []
    ground_truth: List[dict] = []
    cid = 0
    # Deterministic emission order: measure (euclidean first), then d, then N.
    for (measure, d) in [("euclidean", 4), ("euclidean", 5), ("euclidean", 6),
                         ("hyperbolic", 3), ("hyperbolic", 4), ("hyperbolic", 6)]:
        bounds = sources.get((measure, d), {})
        for N in sorted(bounds):
            lb, src = bounds[N]
            challenges.append({
                "id": cid,
                "input": {
                    "measure": measure,
                    "d": d,
                    "N": N,
                },
            })
            ground_truth.append({
                "id": cid,
                "measure": measure,
                "d": d,
                "N": N,
                "our_record": lb,
                "source": src,
            })
            cid += 1
    return challenges, ground_truth


def build_benchmark(challenges: List[dict]) -> dict:
    return {
        "metadata": {
            "title": "Volume Lower Bound Benchmark: construct a record polytope",
            "description": (
                "For each (measure, d, N) the LLM must give a lower bound on "
                "the maximum volume of an N-vertex polytope and back it with "
                "explicit vertex coordinates.  The grader computes the volume "
                "of the submitted polytope and scores both consistency (claim "
                "vs computed) and strength (computed vs our record)."
            ),
            "measures_covered": ["euclidean", "hyperbolic"],
            "dimensions_covered": {
                "euclidean": [4, 5, 6],
                "hyperbolic": [3, 4, 6],
            },
            "hyperbolic_d5_excluded_because": (
                "Our d=5 hyperbolic ground truth is known suboptimal at "
                "N in {12, 14, 20}; adding it would grade against a number "
                "we already know is not the true LB."
            ),
            "n_challenges": len(challenges),
            "pass_threshold_vs_record": 0.95,
            "claim_consistency_tolerance": 1e-3,
            "created": datetime.now(timezone.utc).isoformat(),
            "generator_version": GENERATOR_VERSION,
        },
        "prompt": PROMPT,
        "challenges": challenges,
    }


def write_readme(bench: dict, gt: List[dict]) -> str:
    by_key: Dict[Tuple[str, int], List[dict]] = {}
    for entry in gt:
        by_key.setdefault((entry["measure"], entry["d"]), []).append(entry)

    lines = [
        "# Volume Lower Bound Benchmark",
        "",
        f"{len(bench['challenges'])} challenges across the five "
        "(measure, dimension) regimes listed in the prompt.",
        "",
        "## Task given to the LLM",
        "",
        "```",
        bench["prompt"].rstrip(),
        "```",
        "",
        "## Ground truth (withheld from the LLM)",
        "",
    ]
    for (measure, d) in sorted(by_key):
        lines.append(f"### {measure.capitalize()} volume in dimension {d}")
        lines.append("")
        lines.append("| N | our record | source |")
        lines.append("|---|------------|--------|")
        for e in sorted(by_key[(measure, d)], key=lambda x: x["N"]):
            lines.append(f"| {e['N']} | {e['our_record']:.10f} | `{e['source']}` |")
        lines.append("")
    lines += [
        "## Files",
        "",
        "- `benchmark.json` — what is given to the LLM (prompt + (measure, d, N) list, no records).",
        "- `ground_truth.json` — records + sources, withheld from the LLM.",
        "- This README.",
        "",
        "## Grading",
        "",
        "See `scripts/check_volume_lb_response.py` for the grader.  Summary:",
        "",
        "- The LLM must provide `vertices`; missing → `invalid`.",
        "- The grader computes the actual volume from `vertices` and compares to "
          "both the LLM's `LB` claim and to our record.",
        "- Consistency tolerance: |claim − computed| / max(record, 1e-12) ≤ 1e-3.",
        "- Strength: ≥ 1.00 × record = strong, ≥ 0.95 × record = pass.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR,
                   help=f"Output directory (default: {OUT_DIR.relative_to(REPO)})")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    eucl_d4 = load_eucl_d4()
    eucl_d5, eucl_d6 = load_eucl_d5_d6()
    hyp_d3 = load_hyp_d3()
    hyp_d4 = load_hyp_d4()
    hyp_d6 = load_hyp_d6()

    sources = {
        ("euclidean", 4): eucl_d4,
        ("euclidean", 5): eucl_d5,
        ("euclidean", 6): eucl_d6,
        ("hyperbolic", 3): hyp_d3,
        ("hyperbolic", 4): hyp_d4,
        ("hyperbolic", 6): hyp_d6,
    }
    for (measure, d), bounds in sources.items():
        print(f"{measure:>10} d={d}: {len(bounds):>2} Ns in {sorted(bounds)}")

    challenges, ground_truth = build_entries(sources)
    benchmark = build_benchmark(challenges)

    (args.out_dir / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2) + "\n"
    )
    (args.out_dir / "ground_truth.json").write_text(
        json.dumps(ground_truth, indent=2) + "\n"
    )
    (args.out_dir / "README.md").write_text(write_readme(benchmark, ground_truth))

    print(f"\nWrote {len(challenges)} challenges to "
          f"{args.out_dir.relative_to(REPO)}")


if __name__ == "__main__":
    main()
