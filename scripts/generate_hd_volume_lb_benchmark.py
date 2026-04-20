#!/usr/bin/env python3
"""
Generate an LLM benchmark of the form:

    "What is the largest-volume convex ideal polytope in H^d with exactly N
     vertices that you can construct?"

We have optimized lower bounds for the maximum volume V*(d, N) at many
(d, N) pairs; those bounds are used as ground truth the LLM is scored
against. The LLM passes if it matches or beats our LB (within a tolerance).

Data sources
------------
- H^4: `results/data/optimal_ideal_4polytopes.txt`
  Lines of the form "N = <N> vertices on S^3" followed by
  "Best volume: <V>". N in {5,6,7,8,9,10,12,16,20,30}.
- H^6: `results/data/ideal_H6_polytope_first_results.json`,
        `results/data/ideal_H6_combinatorial_invariance.json`,
        `results/data/ideal_H6_larger_N.json`.
  JSON lists with `N` and `hyp_vol` fields. N in {7,...,24}.

Output
------
- `results/data/llm_challenges_hd_volume_lb/benchmark.json`
- `results/data/llm_challenges_hd_volume_lb/ground_truth.json` (LB only;
  intended to be withheld from the LLM)
- `results/data/llm_challenges_hd_volume_lb/README.md`
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

GENERATOR_VERSION = "1.0.0"

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "results" / "data" / "llm_challenges_hd_volume_lb"


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------

def load_h4_bounds() -> dict[int, float]:
    """Parse results/data/optimal_ideal_4polytopes.txt → {N: best_volume}."""
    path = REPO / "results" / "data" / "optimal_ideal_4polytopes.txt"
    text = path.read_text()
    # The file has blocks "N = <k> vertices on S^3" followed by "Best volume: <v>".
    # We take the best volume reported for each N.
    n_pat = re.compile(r"^N = (\d+) vertices on S\^3", re.MULTILINE)
    v_pat = re.compile(r"^Best volume: ([\d.+eE-]+)", re.MULTILINE)
    ns = [int(m.group(1)) for m in n_pat.finditer(text)]
    vs = [float(m.group(1)) for m in v_pat.finditer(text)]
    if len(ns) != len(vs):
        raise RuntimeError(
            f"Parse mismatch in {path}: {len(ns)} N-headers, {len(vs)} volumes"
        )
    bounds: dict[int, float] = {}
    for n, v in zip(ns, vs):
        # Keep the best seen if a value repeats.
        bounds[n] = max(bounds.get(n, 0.0), v)
    return bounds


def load_h5_bounds() -> tuple[dict[int, float], dict[int, str]]:
    """Read results/data/ideal_H5_polytope_results.json.

    Returns (bounds, sources) where sources[N] in {"recomputed_from_json",
    "paper_table_5"} records whether the LB came from recomputation of the
    stored config or from Table 5 of maximal_ideal_polyhedra.pdf (used when
    the stored config is suboptimal relative to what the paper reports).
    """
    path = REPO / "results" / "data" / "ideal_H5_polytope_results.json"
    if not path.exists():
        return {}, {}
    data = json.loads(path.read_text())
    bounds: dict[int, float] = {}
    sources: dict[int, str] = {}
    for entry in data:
        N = int(entry["N"])
        bounds[N] = float(entry["best_lb"])
        sources[N] = str(entry.get("best_lb_source", "recomputed_from_json"))
    return bounds, sources


def load_h6_bounds() -> dict[int, float]:
    """Combine the three H^6 JSON files → {N: best hyp_vol}."""
    files = [
        REPO / "results" / "data" / "ideal_H6_polytope_first_results.json",
        REPO / "results" / "data" / "ideal_H6_combinatorial_invariance.json",
        REPO / "results" / "data" / "ideal_H6_larger_N.json",
    ]
    bounds: dict[int, float] = {}
    for fn in files:
        data = json.loads(fn.read_text())
        for entry in data:
            N = int(entry["N"])
            # Prefer best_hyp_vol if present (first_results uses it), else hyp_vol.
            v = float(entry.get("best_hyp_vol", entry["hyp_vol"]))
            bounds[N] = max(bounds.get(N, 0.0), v)
    return bounds


# ---------------------------------------------------------------------------
# Benchmark assembly
# ---------------------------------------------------------------------------

PROMPT = """\
Task: for each (d, N) pair listed below, construct a convex ideal polytope
in hyperbolic d-space H^d with exactly N vertices that maximizes the
hyperbolic volume. The vertices lie on the boundary S^{d-1} of the Klein
ball model of H^d.

Report, for each (d, N):
  * the hyperbolic volume V you achieve (required), and
  * (optional but strongly preferred) the N vertex coordinates on S^{d-1}.

Grading: let V*(d, N) denote the true maximum, which is unknown. We compare
your reported V against our own best-known lower bound LB(d, N) obtained by
numerical optimization:
  * pass   if V >= 0.95 * LB(d, N),
  * strong if V >= LB(d, N) (you have matched or beaten our bound),
  * fail   otherwise.

Note that V*(d, N) >= LB(d, N); matching or beating LB(d, N) is a genuine
mathematical result.

Return your answer as a JSON array of objects of the form
  {"d": <int>, "N": <int>, "V": <float>, "vertices": [[...], ...]}
with `vertices` optional but scored more reliably if present.
"""


def build_challenges(
    h4: dict[int, float], h5: dict[int, float], h6: dict[int, float],
) -> list[dict]:
    challenges = []
    cid = 0
    for d, bounds in ((4, h4), (5, h5), (6, h6)):
        for N in sorted(bounds):
            challenges.append({
                "id": cid,
                "input": {"d": d, "N": N},
            })
            cid += 1
    return challenges


def build_ground_truth(
    h4: dict[int, float],
    h5: dict[int, float],
    h6: dict[int, float],
    h5_sources: dict[int, str],
) -> list[dict]:
    gt = []
    cid = 0
    notes_h4 = {
        5: "regular ideal 4-simplex; volume has a known closed form",
        10: "LB here (~3.009) exceeds the rectified 5-cell volume 2*pi^2/9 ~ 2.193",
        16: "LB here (~7.799) exceeds the doubled rectified 5-cell volume 4*pi^2/9 ~ 4.386",
    }
    notes_h5 = {
        6: "regular ideal 5-simplex",
    }
    notes_h6 = {
        7: "regular ideal 6-simplex; volume has a known closed form",
    }
    for d, bounds, notes in ((4, h4, notes_h4), (5, h5, notes_h5), (6, h6, notes_h6)):
        for N in sorted(bounds):
            entry = {
                "id": cid,
                "d": d,
                "N": N,
                "our_best_lower_bound": bounds[N],
            }
            note_pieces = [notes[N]] if N in notes else []
            if d == 5 and h5_sources.get(N) == "paper_table_5":
                note_pieces.append("LB here is from maximal_ideal_polyhedra.pdf "
                                   "Table 5 (a re-optimization would recover the "
                                   "vertex coords; the config stored on disk is "
                                   "strictly suboptimal)")
            if note_pieces:
                entry["note"] = "; ".join(note_pieces)
            gt.append(entry)
            cid += 1
    return gt


def build_benchmark(
    challenges: list[dict],
    h4: dict[int, float],
    h5: dict[int, float],
    h6: dict[int, float],
) -> dict:
    dims = sorted({4 if h4 else None, 5 if h5 else None, 6 if h6 else None} - {None})
    return {
        "metadata": {
            "title": "H^d Ideal Polytope Max-Volume Lower Bound Benchmark",
            "description": (
                "Given (d, N), the LLM is asked to construct (or at least quote "
                "the volume of) the largest-volume convex ideal polytope in H^d "
                "with N vertices. The LLM's claimed volume is compared against "
                "our optimized lower bound."
            ),
            "theory": (
                "The true supremum V*(d, N) = sup{vol_{H^d}(P) : P an ideal "
                "convex polytope in H^d with |V(P)| = N} is unknown in general. "
                "Numerical optimization produces a lower bound LB(d, N) <= "
                "V*(d, N). A claimed volume V exceeding LB(d, N) would "
                "improve the best-known bound."
            ),
            "dimensions_covered": dims,
            "pass_threshold": 0.95,
            "created": datetime.now(timezone.utc).isoformat(),
            "generator_version": GENERATOR_VERSION,
            "data_sources": [
                "results/data/optimal_ideal_4polytopes.txt",
                "results/data/ideal_H5_polytope_results.json",
                "results/data/ideal_H6_polytope_first_results.json",
                "results/data/ideal_H6_combinatorial_invariance.json",
                "results/data/ideal_H6_larger_N.json",
            ],
            "caveat": (
                "The pipeline fails on highly symmetric non-generic "
                "configurations (e.g. the 10x10 duoprism in H^4); if a model "
                "returns such coordinates our verifier may not produce a "
                "reliable volume, so the model's claimed V is the primary "
                "scored quantity. For d=5, some LBs are taken from paper "
                "Table 5 where the stored configurations are known to be "
                "suboptimal; a re-optimization run is planned."
            ),
        },
        "prompt": PROMPT,
        "challenges": challenges,
    }


def write_readme(bench: dict, gt: list[dict]) -> str:
    n = len(bench["challenges"])
    lines = [
        "# H^d Ideal Polytope Max-Volume Lower Bound Benchmark",
        "",
        f"{n} challenges across dimensions d in "
        f"{bench['metadata']['dimensions_covered']}.",
        "",
        "## Task given to the LLM",
        "",
        "```",
        bench["prompt"].rstrip(),
        "```",
        "",
        "## Ground truth (LB table)",
        "",
        "| d | N | LB on V*(d, N) | note |",
        "|---|---|----------------|------|",
    ]
    for entry in gt:
        lines.append(
            f"| {entry['d']} | {entry['N']} | {entry['our_best_lower_bound']:.10f} "
            f"| {entry.get('note', '')} |"
        )
    lines += [
        "",
        "## Grading",
        "",
        "Pass threshold: `V_claim >= 0.95 * LB(d, N)`.  ",
        "A strong result (genuine improvement): `V_claim >= LB(d, N)`.  ",
        "",
        "Use `scripts/check_hd_volume_lb_response.py` to grade a response JSON.",
        "",
        "## Files",
        "",
        "- `benchmark.json` — what is given to the LLM (no LBs).",
        "- `ground_truth.json` — LBs, intended to be withheld.",
        "",
        "## Caveat",
        "",
        bench["metadata"]["caveat"],
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR,
                   help=f"Output directory (default: {OUT_DIR.relative_to(REPO)})")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    h4 = load_h4_bounds()
    h5, h5_sources = load_h5_bounds()
    h6 = load_h6_bounds()
    print(f"Loaded H^4 LBs for N in {sorted(h4)}")
    if h5:
        paper_Ns = sorted(N for N, s in h5_sources.items() if s == "paper_table_5")
        print(f"Loaded H^5 LBs for N in {sorted(h5)}"
              + (f" (paper-sourced for N in {paper_Ns})" if paper_Ns else ""))
    else:
        print("No H^5 data found — run scripts/recompute_h5_volumes.py first")
    print(f"Loaded H^6 LBs for N in {sorted(h6)}")

    challenges = build_challenges(h4, h5, h6)
    ground_truth = build_ground_truth(h4, h5, h6, h5_sources)
    benchmark = build_benchmark(challenges, h4, h5, h6)

    (args.out_dir / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2) + "\n"
    )
    (args.out_dir / "ground_truth.json").write_text(
        json.dumps(ground_truth, indent=2) + "\n"
    )
    (args.out_dir / "README.md").write_text(write_readme(benchmark, ground_truth))

    print(f"\nWrote benchmark with {len(challenges)} challenges to "
          f"{args.out_dir.relative_to(REPO)}")


if __name__ == "__main__":
    main()
