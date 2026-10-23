# RL Controller Validation Study — Design

*Status:* the version-2 validation protocol and its fake-optimizer contract
tests are implemented. No post-hardening validation result has been generated,
and the historical results remain superseded by the explicit-work-accounting
revision. Fresh controllers and their configuration sidecars are prerequisites
for the expensive run. See `docs/implementation-status.md` and
`ideal_poly_volume_toolkit/rl/README.md` for the live contracts.

## 1. The question we want to answer

Given a task family (ideal polytope volume maximization across d ∈ {3,4,5,6}
and a distribution over N), does a PPO controller trained with `train_controller`
achieve a *systematically* better final volume than running any single inner
optimizer in isolation at the same counted-work budget?

Three possible outcomes, with concrete decision thresholds:

- **RL wins** — geometric mean of the per-task-cell median-volume ratios at
  fixed budget ≥ 1.05 against the best applicable single-baseline median.
  → Publishable contribution; consider extracting
  the harness to its own repo as discussed in the extraction thread.
- **RL ties** — geometric-mean task-cell ratio within ±5 % of the best solo
  reference.  → Keep harness as an
  internal tool; don't market as a meta-optimizer.  Revisit reward shaping
  or state features before drawing hard conclusions.
- **RL loses** — geometric-mean task-cell ratio below 0.95.  → Real finding: either
  the controller is miscalibrated, the problem class is too narrow for
  meta-learning to pay off, or reward signal is noisy at this budget.
  Debug before extraction.

"Budget" means total counted optimizer work per episode: canonical objective,
surrogate, and gradient evaluations are each one work unit. The environment's
`total_evals` field retains its compatibility name but reports this sum, not
just objective-function calls.

## 2. Known complications

### 2.1 The saved `ppo_d*_final.zip` models are stale

Trained on the pre-refactor env (Stage C commit `6b7af11` fixed two
scoring bugs).  Their learned policy optimises an inconsistent objective
for d ∈ {3, 4, 6} and should not be used in the validation run.  They are
kept for provenance and as a loadability smoke test; not as a benchmark
data point.

**Decision**: retrain from scratch on the refactored env.  Four fresh
PPO runs (one per dimension), each 500 k timesteps on CPU — roughly two
hours each on a workstation, running sequentially or four-at-once on a
cluster node.

### 2.2 Mixed-dimension vs single-dimension controllers

The old trained models were per-dimension (`ppo_d5_final.zip`, etc.).
The refactored `IdealPolytopeTaskSampler` supports a range
`d_range=(low, high)` and samples d per episode.  Two choices:

- **Single-d controllers** (what the saved models were): one PPO per
  dimension.  Cleaner comparison; each controller specializes.
- **Mixed-d controllers** (new capability): one PPO across d ∈ {3..6}.
  Tests whether shared structure helps; larger task distribution.

Locked order: train and validate the single-d controllers first. Mixed-d is a
deferred ablation asking "does meta-learning across dimensions help or hurt?"
and must use a separate checkpoint if it is later run.

### 2.3 Baseline choice

Controller actions registered by `make_polytope_env`:
- `cma_eucl` — `CMAESOptimizer(sigma=0.4, use_surrogate=True)`
- `lbfgs_hyp_d{d}` — compatibility action names for dimension-specific local
  methods (Powell at d=3/4, projected Adam at d=5/6 in the current harness)

The validator registers two additional generic baselines alongside those
methods:
- `RandomRestartOptimizer` — worst-reasonable baseline.
- `CMAESOptimizer(use_surrogate=False)` — CMA on the true (hyperbolic)
  objective; expensive per eval at d ∈ {3, 4, 6} but may beat the
  surrogate-CMA baseline.

Two *styles* of baseline run, because meta-learning's value is specifically
in allocation:

- **Solo**: after the shared reset score, one optimizer invocation receives
  the entire remaining cap. It is not restarted in repeated default-sized
  chunks. Strongest single tool, useful as an upper-bound single baseline.
- **Round-robin**: budget split equally across the optimizers.  Models
  "uninformed mixing", which is the dumbest meta-strategy.  If the RL
  controller does not beat round-robin, it is not really learning.

"Solo best-of-k" (run each solo separately, take max) is *not* a deployable
method — it uses k × the RL budget. The best solo *cell median* is retained
only as a labelled reference envelope for ratios, never reported as one run.

### 2.4 Budget matching

Each RL episode consumes some `info["total_evals"]` at horizon=40 (the
default). This is the episode's counted-work total and varies with the selected
operations: sampling is cheap, while CMA and the dimension-specific local
methods consume larger work allocations. Procedure:

1. Run the RL controller for each `(d, N, seed)` task and record that episode's
   exact `total_evals` as its paired budget B.
2. Copy the RL environment's exact reset solution for every baseline. Merely
   reseeding separate environments is insufficient evidence of pairing.
3. For each baseline, enforce B as a strict total-work cap, including the
   shared initial canonical score. A compliant optimizer may under-use an
   indivisible residual but may never exceed it. Round-robin redistributes
   such unused residual work in stable optimizer order.
4. Compare final best-values.

### 2.5 Protocol and output contract

`scripts/validate_rl_controller.py` emits schema
`ideal-poly-rl-validation/2` and protocol `2.0`. Each seed-level row contains:

- canonical-objective, surrogate, gradient-only, and any unclassified work;
- action counts, action failures, and an anytime point after reset and every
  action boundary;
- wall-clock and process-CPU time;
- paired-initial-solution hash, budget source, requested action caps, and a
  structured status/error record.

The document captures the actual Git commit, dirty flag and diff hash; a hash
of the validator/RL source set; model and configuration hashes; Python,
platform, and dependency versions; controller and baseline action ordering;
invocation, thread settings, effective task grid, horizon, and fallback cap;
and requested/actual training timesteps when those values are available.
Before loading a controller, the validator requires training schema
`ideal-poly-rl-training/1`, controller protocol
`ideal-poly-rl-controller/2`, a complete sidecar, a valid sidecar payload
checksum, a matching model SHA-256, the expected task dimension, and a valid
recorded action ordering. It then checks that the recorded ordering matches
the evaluation environment and that the loaded model's serialized timestep
count matches the sidecar. A missing or invalid artifact becomes an explicit
non-applicable or failed RL row; it is never loaded, and baselines use the
documented fallback cap.

Writes are atomic and checkpoint after every `(d, N, seed)` task. Existing
output is a hard error unless `--resume` matches both the protocol
configuration and the recorded source/model/config hashes, or `--overwrite`
is explicit.
`--smoke` is the fast one-task gate and does not change the full-study
defaults.

## 3. Task family and dimensions

Pick a small-but-representative evaluation grid:

| d | N values (eval)      | why                                           |
|---|----------------------|-----------------------------------------------|
| 3 | {5, 7, 10, 15, 20}   | smallest dim, cheapest work units, most tasks |
| 4 | {5, 7, 10, 16, 20}   | sharpest paper-record anchor points           |
| 5 | {6, 7, 10, 14, 20}   | includes three N where records are suspect (§Stage A) |
| 6 | {7, 10, 16, 20, 24}  | closed-form hyp volume, largest dim tested    |

20 tasks total.  Seed each task 10 times → 200 rollouts per method.

This is a *fixed* evaluation grid, separate from the random sampling the
training controller saw.  Only the mixed-d controller sees (d, N) outside
the training distribution, but even single-d controllers evaluate on N
values not in their training range — an intentional stress.

## 4. Metrics

Report for each `(method, d, N)` after aggregating successful seed rows:

- **Median final best volume** (primary).
- **IQR across seeds** (secondary — captures variance).
- **Ratio to record** (tertiary — how close to the known LB?).
- **Anytime curve**: best volume vs counted work units at action boundaries,
  summarized across seeds during analysis.
  Shows whether RL finds good solutions *earlier*, not only *eventually*.

Aggregate across the grid:

- **Geometric-mean ratio** of each method to the best baseline per task.
  Single number per method, directly comparable.
- **Win rate**: fraction of applicable `(d, N)` cells where the method's median
  is strictly greater than every applicable comparator's median. The
  numerator and denominator are always reported; missing, failed, and
  non-applicable comparators never inflate the denominator.

Failures remain first-class raw rows and contribute failure counts, but never
enter medians or IQRs. Quartiles use NumPy's linear percentile interpolation.

## 5. Phases

### Phase 0 — Preparation (~1 h)

- Verify all four `ppo_d*_final.zip` files load on `make_polytope_env(d_range=(d,d))`.
  d=5 already verified in Stage C; check d ∈ {3, 4, 6}.
- **Done:** harden `scripts/validate_rl_controller.py` with exact paired starts,
  one-call solo caps, the expanded baseline registry, decomposed accounting,
  action-boundary traces, structured failures, task-cell aggregation, complete
  provenance, and atomic no-clobber checkpoints. Its compatibility field
  `total_evals` denotes all counted work units.
- **Done:** add fake-optimizer regressions for the solo and round-robin budget
  sequences, decomposed/partial failure accounting, cell medians/IQR, and
  applicability-specific win denominators.
- **Done for future runs:** the training launcher emits a versioned sidecar
  with the exact PPO/task/action configuration, requested and actual
  timesteps, source/environment provenance, and model checksum alongside each
  promoted zip. No fresh model has yet been trained.

### Phase 1 — Retrain on refactored env (~8–10 h total)

- Four single-d controllers: use
  `N_range=(max(5, d+1), 25)` for d ∈ {3, 4, 5, 6}, 500 k requested
  timesteps each, seed=42. The sampler's effective upper bound is
  `max(N_low, min(25, 5*d))`; the training sidecar records both requested and
  effective ranges.
- One mixed-d controller: `d_range=(3, 6)`, 1 M steps (larger task space needs more steps).  Optional for the first validation pass.
- Save to `results/rl_models/ppo_d{d}_controller-v2-seed42.zip` and keep the
  old zips for provenance. The launcher writes the matching
  `ppo_d{d}_controller-v2-seed42_config.json` sidecar under schema
  `ideal-poly-rl-training/1`; it records requested and serialized actual
  timesteps, source/environment/action-ordering provenance, and the promoted
  model checksum. Neither filename claims to be "final".
  Both files are fully staged before promotion. Under `--overwrite`, the
  launcher snapshots the prior pair and restores both paths if either
  promotion fails; no-clobber promotion likewise removes a first new link if
  the second cannot be committed.

### Phase 2 — Run the validation grid (~2–4 h)

- 20 tasks × 10 seeds × (1 RL + 4–5 baselines) = 1000–1200 rollouts.
- Most rollouts take seconds; some dimension-specific local-method rollouts
  take minutes.
- Output: the schema-v2 checkpoint JSON. Derive plotting CSVs from this one
  provenance-bearing source rather than maintaining independently writable
  summaries.

The validator is deliberately serial today. Parallel execution would require
a deterministic shard/merge protocol that preserves no-clobber checkpoints
and model provenance; it is documented as a follow-up instead of exposing an
unsafe `--workers` flag before that design exists.

### Phase 3 — Analysis (~2 h)

- Plot anytime curves per (d, N): x=counted work, y=best volume, lines for each method.
- Heatmap per (d, N): method × task showing ratio-to-record.
- Table: geometric-mean ratio, win rate, per method.
- Write up `docs/results/rl-controller-validation.md` with:
  - One-paragraph summary (did RL win/tie/lose, by how much).
  - Figures in `results/plots/rl_controller_validation/`.
  - Honest discussion of seeds-with-failures and any method-specific caveats.

## 6. Explicitly out of scope

These are interesting but separate work:

- **Controller architecture changes** (attention over frontier, GNN over
  the region tree, transformer policy).  First we need to know if the
  current MLP controller is any good.
- **Reward shaping experiments** (different `eval_cost_weight`, potential-based
  shaping, etc.).  If the current reward cannot beat baselines, shaping may
  rescue; but that's a second-round investigation.
- **Curriculum training** — start with small N and expand.  Orthogonal.
- **Re-deriving the missing d=5 configs** at N ∈ {12, 14, 20}.  Important
  but a different use of the harness.  Independent of validation.
- **Comparison against Optuna / Hyperopt / Ax** on the same tasks.  Worth
  doing *after* we know our harness is competitive internally — otherwise
  we're comparing against external tools with the RL arm untuned.

## 7. Decision gate

After Phase 3, write one paragraph of plain English summarizing:

1. Does the trained RL controller systematically beat the strongest
   single baseline at matched budget?  Yes/No with a number.
2. Does it beat round-robin (i.e., is there any evidence of learned
   allocation)?  Yes/No.
3. If 1 or 2 is No, which single baseline dominates and why might that be?

The paragraph determines whether extraction of the harness to a separate
repo is justified.  If RL wins, extraction is a fine follow-up.  If not,
the harness stays here as an internal tool for problems where we care
about the *task family* more than any single task.

## 8. Rough time budget

| Phase                 | Wall time      | Needed before next |
|-----------------------|----------------|--------------------|
| 0 — prep              | 1 h            | yes                |
| 1 — retrain (4 models)| 8 h (sequential; 2 h on 4-core parallel) | yes — validation runs against fresh models |
| 2 — validation grid   | 2–4 h          | yes                |
| 3 — analysis + writeup| 2 h            | —                  |
| **Total**             | ~13–15 h       |                    |

Running Phase 1 in the background while drafting Phase 2's runner and
Phase 3's plotting harness can compress the wall-clock to one long day.

## 9. Locked defaults before the expensive run

1. Run the four single-d controllers first; defer mixed-d to an ablation.
2. Use training seed 42 so the refactored run is comparable to the historical
   setup.
3. Request 500 k timesteps per single-d controller and record both the request
   and the model's serialized actual `num_timesteps`. Extension to 1 M is a
   follow-up, not an undocumented continuation.
4. Use CPU for the small SB3 MLP and record the effective thread settings.
5. Run `--smoke` to a disposable new path before training, then run the full
   validator only against fresh model/config pairs. Do not mix stale and fresh
   controller rows in one checkpoint.
