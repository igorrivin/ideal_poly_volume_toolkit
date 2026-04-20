# H^d Ideal Polytope Max-Volume Lower Bound Benchmark

35 challenges across dimensions d in [4, 5, 6].

## Task given to the LLM

```
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
```

## Ground truth (LB table)

| d | N | LB on V*(d, N) | note |
|---|---|----------------|------|
| 4 | 5 | 0.2688956602 | regular ideal 4-simplex; volume has a known closed form |
| 4 | 6 | 0.5878301174 |  |
| 4 | 7 | 1.0236011023 |  |
| 4 | 8 | 1.7714283944 |  |
| 4 | 9 | 2.3138512999 |  |
| 4 | 10 | 3.0091040559 | LB here (~3.009) exceeds the rectified 5-cell volume 2*pi^2/9 ~ 2.193 |
| 4 | 12 | 4.3771950375 |  |
| 4 | 16 | 7.7986198204 | LB here (~7.799) exceeds the doubled rectified 5-cell volume 4*pi^2/9 ~ 4.386 |
| 4 | 20 | 11.9357644202 |  |
| 4 | 30 | 22.0036730380 |  |
| 5 | 6 | 0.0574031981 | regular ideal 5-simplex |
| 5 | 7 | 0.1310737739 |  |
| 5 | 8 | 0.2438755408 |  |
| 5 | 9 | 0.4134572747 |  |
| 5 | 10 | 0.6985865726 |  |
| 5 | 11 | 0.8922824775 |  |
| 5 | 12 | 1.1388600000 | LB here is from maximal_ideal_polyhedra.pdf Table 5 (a re-optimization would recover the vertex coords; the config stored on disk is strictly suboptimal) |
| 5 | 13 | 1.3415213860 |  |
| 5 | 14 | 1.6466000000 | LB here is from maximal_ideal_polyhedra.pdf Table 5 (a re-optimization would recover the vertex coords; the config stored on disk is strictly suboptimal) |
| 5 | 15 | 1.8899751616 |  |
| 5 | 16 | 2.2397507172 |  |
| 5 | 17 | 2.5444916420 |  |
| 5 | 18 | 2.8976942386 |  |
| 5 | 19 | 3.3305907092 |  |
| 5 | 20 | 3.7523100000 | LB here is from maximal_ideal_polyhedra.pdf Table 5 (a re-optimization would recover the vertex coords; the config stored on disk is strictly suboptimal) |
| 6 | 7 | 0.0102313751 | regular ideal 6-simplex; volume has a known closed form |
| 6 | 8 | 0.0250435609 |  |
| 6 | 9 | 0.0502208973 |  |
| 6 | 10 | 0.0839025193 |  |
| 6 | 12 | 0.2328373523 |  |
| 6 | 14 | 0.3692201182 |  |
| 6 | 16 | 0.5405983427 |  |
| 6 | 18 | 0.7417885209 |  |
| 6 | 20 | 0.9607007050 |  |
| 6 | 24 | 1.4970305468 |  |

## Grading

Pass threshold: `V_claim >= 0.95 * LB(d, N)`.  
A strong result (genuine improvement): `V_claim >= LB(d, N)`.  

Use `scripts/check_hd_volume_lb_response.py` to grade a response JSON.

## Files

- `benchmark.json` — what is given to the LLM (no LBs).
- `ground_truth.json` — LBs, intended to be withheld.

## Caveat

The pipeline fails on highly symmetric non-generic configurations (e.g. the 10x10 duoprism in H^4); if a model returns such coordinates our verifier may not produce a reliable volume, so the model's claimed V is the primary scored quantity. For d=5, some LBs are taken from paper Table 5 where the stored configurations are known to be suboptimal; a re-optimization run is planned.
