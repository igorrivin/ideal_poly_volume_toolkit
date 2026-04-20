#!/usr/bin/env python3
"""
Grade an LLM response to the H^d max-volume lower-bound benchmark.

Input
-----
A JSON file listing the LLM's answers, one per (d, N) pair. Each answer
must have "d", "N", "V" (claimed hyperbolic volume); "vertices" is
optional (a list of N points in R^d).

Example:
    [
      {"d": 4, "N": 10, "V": 2.71},
      {"d": 6, "N":  9, "V": 0.042, "vertices": [[...], ...]}
    ]

Grading
-------
Compare V against our LB from ground_truth.json:
    pass   if V >= 0.95 * LB
    strong if V >= LB       (genuine improvement — worth inspecting!)
    fail   otherwise

If `--verify` is passed and "vertices" is present, the volume is
recomputed from the coordinates (H^4 only — uses the existing
standalone checker pattern). Disagreement between claimed and computed V
is flagged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_GT = REPO / "results" / "data" / "llm_challenges_hd_volume_lb" / "ground_truth.json"

PASS_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Optional H^4 volume recomputation.
# We delegate to the standalone checker already present in the repo.
# ---------------------------------------------------------------------------

def recompute_volume_h4(vertices: list[list[float]]) -> float | None:
    import numpy as np
    pkg_path = REPO / "results" / "data" / "llm_challenges_4d_volume" / "care_package"
    if str(pkg_path) not in sys.path:
        sys.path.insert(0, str(pkg_path))
    try:
        import check_response  # type: ignore
    except Exception as exc:
        print(f"  [warn] could not import H^4 checker: {exc}")
        return None
    V = np.asarray(vertices, dtype=float)
    if V.ndim != 2 or V.shape[1] != 4:
        print(f"  [warn] expected shape (N,4); got {V.shape}")
        return None
    try:
        return float(check_response.ideal_polytope_volume_H4(V))
    except AttributeError:
        # Fallback: try common alternative entry points.
        for name in ("compute_volume", "ideal_polytope_volume", "volume"):
            fn = getattr(check_response, name, None)
            if callable(fn):
                return float(fn(V))
        print("  [warn] H^4 checker exposes no volume function we recognise")
        return None


def recompute_volume(d: int, vertices: list[list[float]]) -> float | None:
    if d == 4:
        return recompute_volume_h4(vertices)
    # H^6 (and others) would require the package's h5_volume_torch / geometry_4d
    # pipeline. Not wired in here to keep this checker dependency-light.
    return None


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def load_ground_truth(path: Path) -> dict[tuple[int, int], float]:
    entries = json.loads(path.read_text())
    return {(int(e["d"]), int(e["N"])): float(e["our_best_lower_bound"])
            for e in entries}


def grade_one(ans: dict, gt: dict[tuple[int, int], float], verify: bool) -> dict:
    d = int(ans["d"])
    N = int(ans["N"])
    V_claim = float(ans["V"])
    lb = gt.get((d, N))
    if lb is None:
        return {"d": d, "N": N, "status": "unknown_key",
                "message": f"no ground truth for (d={d}, N={N})"}

    ratio = V_claim / lb
    if ratio >= 1.0:
        status = "strong"
    elif ratio >= PASS_THRESHOLD:
        status = "pass"
    else:
        status = "fail"

    result = {
        "d": d, "N": N, "V_claimed": V_claim, "LB": lb,
        "ratio": ratio, "status": status,
    }

    if verify and "vertices" in ans:
        V_comp = recompute_volume(d, ans["vertices"])
        result["V_recomputed"] = V_comp
        if V_comp is not None:
            result["claim_vs_recomputed_rel_err"] = abs(V_comp - V_claim) / max(abs(V_claim), 1e-12)
            # If recomputed disagrees with the claim, re-grade by recomputed.
            if result["claim_vs_recomputed_rel_err"] > 0.01:
                result["note"] = (f"claimed V={V_claim} but recomputed V={V_comp}; "
                                  f"re-grading on recomputed value.")
                V_eff = V_comp
                r2 = V_eff / lb
                result["status"] = "strong" if r2 >= 1.0 else ("pass" if r2 >= PASS_THRESHOLD else "fail")
                result["ratio"] = r2
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("response", type=Path, help="LLM response JSON")
    p.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    p.add_argument("--verify", action="store_true",
                   help="If vertices are present, recompute volume "
                        "(H^4 only in this minimal checker)")
    args = p.parse_args()

    answers = json.loads(args.response.read_text())
    if isinstance(answers, dict) and "answers" in answers:
        answers = answers["answers"]
    if not isinstance(answers, list):
        sys.exit(f"expected a JSON list of answers; got {type(answers).__name__}")

    gt = load_ground_truth(args.ground_truth)
    results = [grade_one(a, gt, args.verify) for a in answers]

    n_strong = sum(1 for r in results if r["status"] == "strong")
    n_pass = sum(1 for r in results if r["status"] == "pass")
    n_fail = sum(1 for r in results if r["status"] == "fail")
    n_unknown = sum(1 for r in results if r["status"] == "unknown_key")

    print(f"{'d':>3} {'N':>3} {'V_claim':>14} {'LB':>14} {'ratio':>8}  status")
    for r in results:
        if r["status"] == "unknown_key":
            print(f"  ? {r['N']:>3} {'':>14} {'':>14} {'':>8}  {r['status']}")
            continue
        print(f"{r['d']:>3} {r['N']:>3} {r['V_claimed']:>14.6f} "
              f"{r['LB']:>14.6f} {r['ratio']:>8.4f}  {r['status']}")

    print("\nSummary:")
    print(f"  strong (V >= LB): {n_strong}")
    print(f"  pass   (V >= 0.95*LB): {n_pass}")
    print(f"  fail:   {n_fail}")
    if n_unknown:
        print(f"  unknown (d,N): {n_unknown}")

    # Exit 0 unless every single answer failed.
    sys.exit(0 if (n_strong + n_pass) > 0 else 1)


if __name__ == "__main__":
    main()
