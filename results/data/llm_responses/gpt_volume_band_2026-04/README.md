# GPT-5.4 Pro submission to the windowed-volume benchmark (2026-04)

Frontier-model response to the 17-challenge windowed-volume task
(`results/data/llm_challenges_4d_volume/care_package/benchmark_volume.json`).

## Files

- `ideal_4polytope_volume_construction.py` — GPT's construction code.
  Two base configurations (`BASE20`, `BASE30`) plus three named families
  (`squash20`, `base30`, `two_pentagons`) parameterised by a single
  scalar that linearly interpolates the volume.
- `ideal_4polytope_volume_points.json` — explicit vertex coordinates
  produced by the constructor for each of the 17 challenges, plus
  GPT's own order-8 Gauss-Legendre volume estimate.
- `ideal_4polytope_volume_summary.csv` — a tabular summary of the
  same.

## Verification result

Cross-checked against the official Haagerup-Munkholm checker
(`results/data/llm_challenges_4d_volume/care_package/check_response.py`)
on 2026-04-29: **17 / 17 pass**.  GPT's order-8 Gauss-Legendre
estimate agrees with the exact closed-form Haagerup-Munkholm volume
to about $2 \times 10^{-3}$ on every challenge; every computed volume
lies inside the target $[T-\epsilon, T+\epsilon]$ window.

This is the first frontier-model submission that solves the windowed
benchmark cleanly across all 17 instances.  See
`paper/llm_volume_note.tex` (Task B section) for discussion.
