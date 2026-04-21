#!/usr/bin/env python3
"""
Grade an LLM response to the volume-LB construct-a-polytope benchmark.

Each answer must carry (id, measure, d, N, LB, vertices).  The grader:
  1. Recomputes the volume of the submitted polytope via the measure-and-
     dimension-appropriate routine (ConvexHull for Euclidean; the closed-
     form hyperbolic routines in IdealPolytopeTask for Hyperbolic).
  2. Checks consistency between `LB` (claim) and the computed volume.
  3. Compares the computed volume to our record for that (measure, d, N)
     from `ground_truth.json`.

Exit code: 0 if at least one challenge is "strong" or "pass"; 1 otherwise.

Usage:
    python scripts/check_volume_lb_response.py response.json
    python scripts/check_volume_lb_response.py response.json \\
        --ground-truth results/data/llm_challenges_volume_lb/ground_truth.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull

REPO = Path(__file__).resolve().parent.parent
DEFAULT_GT = REPO / "results" / "data" / "llm_challenges_volume_lb" / "ground_truth.json"

STRONG_THRESHOLD = 1.00
PASS_THRESHOLD = 0.95
CLAIM_TOL = 1e-3
SPHERE_TOL = 1e-6


# ---------------------------------------------------------------------------
# Volume recomputation
# ---------------------------------------------------------------------------

def compute_volume(measure: str, d: int, vertices: np.ndarray
                   ) -> Tuple[Optional[float], Optional[str]]:
    """Compute the (measure, d) volume of `vertices`.
    Returns (volume, warning) where warning describes any defensive fix-up.
    """
    if vertices.ndim != 2 or vertices.shape[1] != d:
        return None, f"expected shape (N, {d}); got {vertices.shape}"

    if measure == "euclidean":
        norms = np.linalg.norm(vertices, axis=1)
        if (norms > 1 + SPHERE_TOL).any():
            # Inscribed polytope: clip to unit ball.
            factor = np.maximum(norms, 1.0)
            vertices = vertices / factor[:, None]
        try:
            return float(ConvexHull(vertices).volume), None
        except Exception as exc:
            return None, f"ConvexHull failed: {exc}"

    if measure == "hyperbolic":
        norms = np.linalg.norm(vertices, axis=1)
        warn = None
        if (abs(norms - 1.0) > SPHERE_TOL).any():
            warn = (f"vertices not exactly on S^{d-1} "
                    f"(max dev {abs(norms - 1.0).max():.2e}); "
                    "normalizing defensively")
            vertices = vertices / norms[:, None]
        # Delegate to the same closed-form hyperbolic volume routines that
        # IdealPolytopeTask uses so grading matches the package's own scoring.
        from ideal_poly_volume_toolkit.rl.polytope.types import PolytopeSolution
        from ideal_poly_volume_toolkit.rl.polytope.task import (
            _stereographic_delaunay, _hyperbolic_volume,
        )
        try:
            simps = _stereographic_delaunay(vertices) if d >= 3 else []
        except Exception as exc:
            return None, f"triangulation failed: {exc}"
        sol = PolytopeSolution(vertices=vertices, simplices=simps)
        # `_hyperbolic_volume` handles per-dimension dispatch.  For this
        # benchmark d in {4, 6}, both have fast closed-form routes.
        try:
            return float(_hyperbolic_volume(d, sol)), warn
        except Exception as exc:
            return None, f"hyperbolic volume computation failed: {exc}"

    return None, f"unknown measure: {measure!r}"


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def load_ground_truth(path: Path) -> Dict[Tuple[str, int, int], Tuple[float, str]]:
    out: Dict[Tuple[str, int, int], Tuple[float, str]] = {}
    for entry in json.loads(path.read_text()):
        key = (entry["measure"], int(entry["d"]), int(entry["N"]))
        out[key] = (float(entry["our_record"]), str(entry["source"]))
    return out


def grade_one(ans: Dict[str, Any],
              gt: Dict[Tuple[str, int, int], Tuple[float, str]]
              ) -> Dict[str, Any]:
    measure = str(ans.get("measure", "")).lower()
    try:
        d = int(ans["d"])
        N = int(ans["N"])
    except (KeyError, TypeError, ValueError):
        return {"status": "invalid", "message": "missing d/N"}

    result: Dict[str, Any] = {
        "measure": measure, "d": d, "N": N,
        "status": "invalid",
    }

    gt_key = (measure, d, N)
    if gt_key not in gt:
        result["message"] = f"no ground truth for ({measure}, d={d}, N={N})"
        result["status"] = "unknown_key"
        return result

    record, source = gt[gt_key]
    result["record"] = record
    result["record_source"] = source

    verts_raw = ans.get("vertices")
    if verts_raw is None:
        result["message"] = "missing vertices (required for grading)"
        return result

    try:
        vertices = np.asarray(verts_raw, dtype=float)
    except Exception as exc:
        result["message"] = f"vertices could not be parsed as array: {exc}"
        return result
    if vertices.shape != (N, d):
        result["message"] = (f"vertices shape {vertices.shape} "
                             f"does not match (N={N}, d={d})")
        return result

    V_comp, warn = compute_volume(measure, d, vertices)
    if warn:
        result["warning"] = warn
    if V_comp is None or V_comp <= 0:
        result["V_computed"] = V_comp
        result["message"] = warn or "non-positive computed volume"
        return result
    result["V_computed"] = V_comp

    claim = ans.get("LB")
    if claim is not None:
        try:
            claim_f = float(claim)
            result["V_claimed"] = claim_f
            denom = max(V_comp, 1e-12)
            result["claim_consistency_rel_err"] = abs(claim_f - V_comp) / denom
            result["claim_consistent"] = (
                result["claim_consistency_rel_err"] <= CLAIM_TOL
            )
        except (TypeError, ValueError):
            result["claim_consistent"] = False
            result["message"] = "LB not numeric"

    ratio = V_comp / record
    result["ratio_vs_record"] = ratio
    if ratio >= STRONG_THRESHOLD:
        result["status"] = "strong"
    elif ratio >= PASS_THRESHOLD:
        result["status"] = "pass"
    else:
        result["status"] = "fail"

    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("response", type=Path, help="LLM response JSON")
    p.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    args = p.parse_args()

    answers = json.loads(args.response.read_text())
    if isinstance(answers, dict) and "answers" in answers:
        answers = answers["answers"]
    if not isinstance(answers, list):
        sys.exit(f"expected a JSON list of answers; got {type(answers).__name__}")

    gt = load_ground_truth(args.ground_truth)
    results: List[Dict[str, Any]] = [grade_one(a, gt) for a in answers]

    counts = {"strong": 0, "pass": 0, "fail": 0, "invalid": 0, "unknown_key": 0}
    header = (f"{'measure':>10} {'d':>2} {'N':>3} "
              f"{'V_claim':>12} {'V_comp':>12} {'record':>12} "
              f"{'ratio':>7} {'claim_ok':>8}  status")
    print(header)
    for r in results:
        status = r["status"]
        counts[status] = counts.get(status, 0) + 1
        line = f"{r.get('measure','?'):>10} {r.get('d','?'):>2} {r.get('N','?'):>3} "
        line += f"{r.get('V_claimed', float('nan')):>12.6f} " if r.get("V_claimed") is not None else f"{'-':>12} "
        line += f"{r.get('V_computed', float('nan')):>12.6f} " if r.get("V_computed") is not None else f"{'-':>12} "
        line += f"{r.get('record', float('nan')):>12.6f} " if r.get("record") is not None else f"{'-':>12} "
        line += f"{r.get('ratio_vs_record', float('nan')):>7.3f} " if r.get("ratio_vs_record") is not None else f"{'-':>7} "
        ck = r.get("claim_consistent")
        line += f"{'yes' if ck is True else ('no' if ck is False else '-'):>8}  {status}"
        print(line)
        if r.get("message"):
            print(f"           msg: {r['message']}")

    print(f"\nSummary:")
    for k in ("strong", "pass", "fail", "invalid", "unknown_key"):
        print(f"  {k}: {counts.get(k, 0)}")

    sys.exit(0 if (counts["strong"] + counts["pass"]) > 0 else 1)


if __name__ == "__main__":
    main()
