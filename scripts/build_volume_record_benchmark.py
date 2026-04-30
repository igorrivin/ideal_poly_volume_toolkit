#!/usr/bin/env python3
"""
Assemble the volume-record stress benchmark from per-challenge records
under `results/data/volume_records/d{d}_N{N}_{measure}.json`.

Output layout (under `results/data/llm_challenges_volume_record/`):
    benchmark.json     — what's shown to the LLM (prompt + (d, N, measure) list, no records)
    ground_truth.json  — records + provenance (search budget, top-10 spread, sources)
    records/d{d}_N{N}_{measure}.json — per-challenge winning vertex coordinates
    README.md          — full table + grading rules + record-finding methodology

Each challenge's grading: the LLM submits N coordinates and a claimed
volume; the grader recomputes the volume under the appropriate measure
and compares to our record.  An LLM submission of `>= record - eps`
passes; `>= record` matches our bound; `> record + eps` would be a
genuine new mathematical claim (we'd want to verify it carefully).
"""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parent.parent
SRC_DIR = REPO / "results" / "data" / "volume_records"
OUT_DIR = REPO / "results" / "data" / "llm_challenges_volume_record"

GENERATOR_VERSION = "1.0.0"

# Match-tolerance: LLM volume must equal our record within this relative
# tolerance to count as a match.  10^-4 is tight enough to require real
# numerical work, loose enough to forgive last-digit float jitter.
EPS_REL = 1e-4


PROMPT_BLIND = """\
For each challenge below, construct an explicit convex polytope with
N vertices whose volume meets or exceeds our (withheld) record.

Each challenge specifies:
  - measure: "euclidean" (vertices on or inside the unit ball of R^d) or
             "hyperbolic" (ideal polytope in H^d, vertices on S^{d-1}
             interpreted as the boundary of the Klein-ball model);
  - d: the dimension;
  - N: the number of vertices.

For each challenge return:
  1. `LB`:  the volume your polytope achieves.
  2. `vertices`:  a list of N coordinates in R^d.

Both fields are required.  `LB` is checked for consistency against
the volume of the submitted `vertices` to relative tolerance 10^-4;
mismatched answers are rejected.  Each accepted answer classifies as
`strong` (LB >= our record), `pass` (LB >= 0.95 * our record), or
`fail` otherwise.

Return your answer as a JSON list of objects:

    [
      {"id": 0, "measure": "...", "d": ..., "N": ..., "LB": <float>,
       "vertices": [[...], ...]},
      ...
    ]
"""

PROMPT_OPEN = """\
For each challenge below, construct an explicit convex polytope with
N vertices whose volume meets or exceeds the specified `target`.

Each challenge specifies:
  - measure: "euclidean" (vertices on or inside the unit ball of R^d) or
             "hyperbolic" (ideal polytope in H^d, vertices on S^{d-1}
             interpreted as the boundary of the Klein-ball model);
  - d: the dimension;
  - N: the number of vertices;
  - target: the volume threshold to meet or beat.

For each challenge return:
  1. `LB`:  the volume your polytope achieves.
  2. `vertices`:  a list of N coordinates in R^d.

Both fields are required.  `LB` is checked for consistency against
the volume of the submitted `vertices` to relative tolerance 10^-4;
mismatched answers are rejected.  Each accepted answer classifies as
`strong` (LB >= target), `pass` (LB >= 0.95 * target), or `fail`
otherwise.

Return your answer as a JSON list of objects:

    [
      {"id": 0, "measure": "...", "d": ..., "N": ..., "LB": <float>,
       "vertices": [[...], ...]},
      ...
    ]
"""


def _load_record(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=SRC_DIR)
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir; default depends on --reveal-target.")
    p.add_argument("--reveal-target", action="store_true",
                   help="Include the target volume in each challenge "
                        "(open/control variant).")
    args = p.parse_args()

    if args.out is None:
        args.out = (REPO / "results" / "data"
                    / ("llm_challenges_volume_record_open"
                       if args.reveal_target
                       else "llm_challenges_volume_record"))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "records").mkdir(exist_ok=True)

    src_files = sorted(args.src.glob("d*_N*.json"))
    if not src_files:
        raise SystemExit(f"no per-challenge records found in {args.src}")

    prompt = PROMPT_OPEN if args.reveal_target else PROMPT_BLIND

    challenges = []
    ground_truth = []
    for cid, path in enumerate(src_files):
        rec = _load_record(path)
        d, N, measure = rec["d"], rec["N"], rec["measure"]
        ch_input: Dict[str, Any] = {"measure": measure, "d": d, "N": N}
        if args.reveal_target:
            ch_input["target"] = rec["best_volume"]
        challenges.append({"id": cid, "input": ch_input})
        budget = rec["search_budget"]
        top10 = rec["top_10_after_polish"]
        gt_entry = {
            "id": cid,
            "measure": measure, "d": d, "N": N,
            "our_record": rec["best_volume"],
            "alt_eval": rec.get("best_volume_alt_eval"),
            "top_10_spread": float(max(top10) - min(top10)),
            "search_budget": budget,
            "source": str(path.relative_to(REPO)),
        }
        ground_truth.append(gt_entry)
        # Per-challenge file with the winning vertex coordinates
        (args.out / "records" / path.name).write_text(json.dumps({
            "id": cid, "measure": measure, "d": d, "N": N,
            "best_volume": rec["best_volume"],
            "vertices": rec["vertices"],
        }, indent=2))

    title = ("Volume Record Benchmark (target revealed)"
             if args.reveal_target else "Volume Record Benchmark")
    benchmark = {
        "metadata": {
            "title": title,
            "n_challenges": len(challenges),
            "match_tolerance": EPS_REL,
            "target_revealed": bool(args.reveal_target),
            "created": datetime.now(timezone.utc).isoformat(),
            "generator_version": GENERATOR_VERSION,
        },
        "prompt": prompt,
        "challenges": challenges,
    }

    (args.out / "benchmark.json").write_text(json.dumps(benchmark, indent=2))
    (args.out / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2))

    # README with the human-readable table
    variant_note = (
        "**Variant: target revealed.**  Each challenge includes the target "
        "volume in `input.target`.  This is the *control* version of the "
        "blind benchmark in `llm_challenges_volume_record/`; comparing the "
        "two tells us how much of the difficulty comes from the geometric "
        "construction vs from not knowing the volume to aim for."
        if args.reveal_target else
        "**Variant: target hidden.**  The model is asked to produce a "
        "polytope whose volume meets or exceeds our (withheld) record, "
        "without being told what the record is.  See "
        "`llm_challenges_volume_record_open/` for the corresponding "
        "*control* variant where the target is revealed."
    )
    readme = ["# Volume Record Benchmark"
              + (" (target revealed)" if args.reveal_target else ""), "",
              f"{len(challenges)} challenges, designed to be tighter than the "
              "broader `llm_challenges_volume_lb/` benchmark by:", "",
              "- Choosing `N` deliberately to avoid simplex boundaries and "
              "obvious join structure (which GPT cracked on the windowed "
              "benchmark by structural recognition).",
              "- Investing 500-1000 CMA-ES random restarts + Powell polish "
              "+ paper-record warm starts per challenge, so the records "
              "are near-optimal rather than just \"the best we have on disk\".",
              "- Demanding `LB <= computed_volume` to within "
              f"`{EPS_REL:.0e}` relative tolerance (the windowed benchmark "
              "uses `~2%`).",
              "",
              variant_note, "",
              "## Records (withheld in the blind variant; visible to the LLM in the open variant)", "",
              "| d | N | measure | record | top-10 spread | restarts | wall (s) |",
              "|---|---|---------|--------|---------------|----------|---------:|"]
    for gt in ground_truth:
        b = gt["search_budget"]
        readme.append(
            f"| {gt['d']} | {gt['N']} | {gt['measure']} | "
            f"{gt['our_record']:.8f} | {gt['top_10_spread']:.4f} | "
            f"{b['restarts_run']} | {b['wall_time_sec']:.0f} |"
        )
    readme += ["", "## Files", "",
               "- `benchmark.json` — prompt + challenge list (no records).",
               "- `ground_truth.json` — records + provenance.",
               "- `records/d{d}_N{N}_{measure}.json` — full per-challenge "
               "data including the winning vertex coordinates.",
               "- This README.",
               "",
               "## Grading", "",
               f"- LLM must include `vertices` (anti-hallucination).",
               f"- `|LB_claim - computed_volume| / max(record, 1e-12) <= "
               f"{EPS_REL:.0e}` for consistency.",
               "- `strong` = computed >= record; `pass` = >= 0.95 * record; "
               "`fail` otherwise.",
               "",
               "## Methodology", "",
               "Per-challenge record finding (`scripts/find_record_polytope.py`):",
               "1. Warm-start from existing paper-record configurations of "
               "nearby `N` (when available).",
               "2. Phase 1: 500-1000 random restarts of CMA-ES on the target "
               "objective (Eucl hull volume or closed-form Hyp volume).",
               "3. Phase 2: scipy Powell polish on the top 60-200 candidates.",
               "4. Phase 3: cross-validation via two independent volume "
               "evaluators; degeneracy filter rejects configurations whose "
               "Euclidean hull is below `1e-6` (the closed-form Hyperbolic "
               "formulas misbehave on lower-dimensional point sets).",
               "",
               "Generator: `scripts/build_volume_record_benchmark.py`.",
               ""]
    (args.out / "README.md").write_text("\n".join(readme))

    print(f"Wrote {len(challenges)} challenges to {args.out.relative_to(REPO)}")
    print(f"  - benchmark.json")
    print(f"  - ground_truth.json")
    print(f"  - README.md")
    print(f"  - records/{len(challenges)} per-challenge files")


if __name__ == "__main__":
    main()
