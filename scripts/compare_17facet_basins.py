#!/usr/bin/env python3
"""
Compare our independently-found 17-facet records against Gemini's submission.

For each polytope we compute an O(4)-invariant fingerprint: the sorted vector
of pairwise dot products v_i . v_j (which is invariant under any orthogonal
transformation of S^3, and unchanged by vertex relabeling).  Two polytopes
are in the same basin iff their fingerprints match to a tight tolerance.

Usage:
    python scripts/compare_17facet_basins.py \
        --ours results/data/sa_17facet_record.json \
        --gemini /tmp/gemini_17facet.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def fingerprint(vertices: np.ndarray) -> np.ndarray:
    """Sorted vector of pairwise dot products — O(N) x O(d) invariant."""
    G = vertices @ vertices.T
    n = len(vertices)
    iu = np.triu_indices(n, k=1)
    return np.sort(G[iu])


def fp_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def antipodal_summary(vertices: np.ndarray) -> dict:
    """How many antipodal pairs (dot product near -1)?"""
    G = vertices @ vertices.T
    n = len(vertices)
    iu = np.triu_indices(n, k=1)
    dots = G[iu]
    return {
        "n_pairs_below_-0.95": int((dots < -0.95).sum()),
        "n_pairs_below_-0.90": int((dots < -0.90).sum()),
        "min_dot": float(dots.min()),
        "max_dot": float(dots.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ours", type=Path, required=True)
    p.add_argument("--gemini", type=Path, required=True)
    p.add_argument("--basin-tol", type=float, default=0.05,
                   help="fingerprint distance below which two polytopes are "
                        "considered to be in the same basin")
    args = p.parse_args()

    ours = json.load(open(args.ours))
    gem = json.load(open(args.gemini))

    g_verts = np.array(gem["vertices"])
    g_fp = fingerprint(g_verts)
    print(f"Gemini: SA={gem['value']:.6f}, n_facets={gem['n_facets']}")
    print(f"Gemini antipodal: {antipodal_summary(g_verts)}")
    print()

    print(f"Our top-{len(ours['top_k'])} configurations:")
    print(f"{'rank':>4}  {'value':>10}  {'fp-dist':>10}  {'same-basin?':>12}  "
          f"{'pairs<-.95':>10}  {'min_dot':>8}")
    same_basin = 0
    for i, t in enumerate(ours["top_k"], 1):
        verts = np.array(t["vertices"])
        fp = fingerprint(verts)
        d = fp_distance(fp, g_fp)
        anti = antipodal_summary(verts)
        same = d < args.basin_tol
        if same:
            same_basin += 1
        flag = "YES" if same else "no"
        print(f"{i:>4}  {t['value']:>10.6f}  {d:>10.4f}  {flag:>12}  "
              f"{anti['n_pairs_below_-0.95']:>10}  {anti['min_dot']:>8.4f}")

    print()
    print(f"Verdict: {same_basin}/{len(ours['top_k'])} of our top configs are "
          f"in Gemini's basin (fp-dist < {args.basin_tol}).")
    print(f"Best of ours: {ours['best_value']:.6f} vs Gemini {gem['value']:.6f} "
          f"(gap {gem['value']-ours['best_value']:+.6f}).")


if __name__ == "__main__":
    main()
