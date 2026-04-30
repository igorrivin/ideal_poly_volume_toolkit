# Volume Record Benchmark (target revealed)

13 challenges, designed to be tighter than the broader `llm_challenges_volume_lb/` benchmark by:

- Choosing `N` deliberately to avoid simplex boundaries and obvious join structure (which GPT cracked on the windowed benchmark by structural recognition).
- Investing 500-1000 CMA-ES random restarts + Powell polish + paper-record warm starts per challenge, so the records are near-optimal rather than just "the best we have on disk".
- Demanding `LB <= computed_volume` to within `1e-04` relative tolerance (the windowed benchmark uses `~2%`).

**Variant: target revealed.**  Each challenge includes the target volume in `input.target`.  This is the *control* version of the blind benchmark in `llm_challenges_volume_record/`; comparing the two tells us how much of the difficulty comes from the geometric construction vs from not knowing the volume to aim for.

## Records (withheld in the blind variant; visible to the LLM in the open variant)

| d | N | measure | record | top-10 spread | restarts | wall (s) |
|---|---|---------|--------|---------------|----------|---------:|
| 3 | 21 | hyperbolic | 29.08783023 | 0.1439 | 501 | 143 |
| 3 | 23 | hyperbolic | 32.71518744 | 0.3434 | 752 | 216 |
| 3 | 27 | hyperbolic | 39.37830307 | 0.1645 | 797 | 251 |
| 3 | 33 | hyperbolic | 50.37537892 | 0.8672 | 501 | 209 |
| 4 | 21 | hyperbolic | 12.16463926 | 0.2406 | 914 | 6858 |
| 4 | 23 | hyperbolic | 13.87979902 | 0.3389 | 602 | 5647 |
| 4 | 27 | hyperbolic | 17.19833121 | 0.1874 | 501 | 6321 |
| 4 | 29 | hyperbolic | 19.15361977 | 0.3078 | 501 | 7071 |
| 5 | 21 | euclidean | 0.82664926 | 0.0242 | 501 | 170 |
| 5 | 23 | euclidean | 0.90491551 | 0.0287 | 501 | 197 |
| 5 | 27 | euclidean | 1.02606358 | 0.0085 | 638 | 324 |
| 6 | 23 | hyperbolic | 1.30229515 | 0.0681 | 0 | 691 |
| 6 | 29 | hyperbolic | 1.84707662 | 0.0775 | 501 | 15086 |

## Files

- `benchmark.json` — prompt + challenge list (no records).
- `ground_truth.json` — records + provenance.
- `records/d{d}_N{N}_{measure}.json` — full per-challenge data including the winning vertex coordinates.
- This README.

## Grading

- LLM must include `vertices` (anti-hallucination).
- `|LB_claim - computed_volume| / max(record, 1e-12) <= 1e-04` for consistency.
- `strong` = computed >= record; `pass` = >= 0.95 * record; `fail` otherwise.

## Methodology

Per-challenge record finding (`scripts/find_record_polytope.py`):
1. Warm-start from existing paper-record configurations of nearby `N` (when available).
2. Phase 1: 500-1000 random restarts of CMA-ES on the target objective (Eucl hull volume or closed-form Hyp volume).
3. Phase 2: scipy Powell polish on the top 60-200 candidates.
4. Phase 3: cross-validation via two independent volume evaluators; degeneracy filter rejects configurations whose Euclidean hull is below `1e-6` (the closed-form Hyperbolic formulas misbehave on lower-dimensional point sets).

Generator: `scripts/build_volume_record_benchmark.py`.
