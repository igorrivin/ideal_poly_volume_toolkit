# RL Controller Validation Study — Design

*Status:* plan only, no code.  Follows from the Stage-C harness
modularization (`docs/plans/rl-harness-design.md`) and the open question
of whether the trained PPO controller genuinely helps vs. strong classical
baselines.

## 1. The question we want to answer

Given a task family (ideal polytope volume maximization across d ∈ {3,4,5,6}
and a distribution over N), does a PPO controller trained with `train_controller`
achieve a *systematically* better final volume than running any single inner
optimizer in isolation at the same evaluation budget?

Three possible outcomes, with concrete decision thresholds:

- **RL wins** — median volume at fixed budget ≥ 1.05 × best single-baseline
  median across 20+ tasks.  → Publishable contribution; consider extracting
  the harness to its own repo as discussed in the extraction thread.
- **RL ties** — median within ±5 % of best baseline.  → Keep harness as an
  internal tool; don't market as a meta-optimizer.  Revisit reward shaping
  or state features before drawing hard conclusions.
- **RL loses** — median below 0.95 × best baseline.  → Real finding: either
  the controller is miscalibrated, the problem class is too narrow for
  meta-learning to pay off, or reward signal is noisy at this budget.
  Debug before extraction.

"Budget" is total function evaluations per episode, summed across whatever
operations the controller or baseline picks.  The env's `total_evals`
counter is already the correct accounting device.

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

Recommendation: train *both*.  Single-d results are the primary comparison;
mixed-d is an ablation asking "does meta-learning across dimensions help or
hurt?".

### 2.3 Baseline choice

Inner optimizers registered by `make_polytope_env`:
- `cma_eucl` — `CMAESOptimizer(sigma=0.4, use_surrogate=True)`
- `lbfgs_hyp_d{d}` — dimension-specific LBFGS on the hyperbolic surrogate

Plus these ancillary generic baselines to stress-test the harness:
- `RandomRestartOptimizer` — worst-reasonable baseline.
- `CMAESOptimizer(use_surrogate=False)` — CMA on the true (hyperbolic)
  objective; expensive per eval at d ∈ {3, 4, 6} but may beat the
  surrogate-CMA baseline.

Two *styles* of baseline run, because meta-learning's value is specifically
in allocation:

- **Solo**: one optimizer consumes the entire budget.  Strongest single
  tool, useful as an upper-bound single baseline.
- **Round-robin**: budget split equally across the optimizers.  Models
  "uninformed mixing", which is the dumbest meta-strategy.  If the RL
  controller does not beat round-robin, it is not really learning.

"Solo best-of-k" (run each solo separately, take max) is *not* a fair
comparison — it uses k × the RL budget.  Skip it.

### 2.4 Budget matching

Each RL episode consumes some `info["total_evals"]` at horizon=40 (the
default).  Empirically that varies a lot per episode (sample is cheap,
CMA is mid, LBFGS-hyp is expensive).  Procedure:

1. Run the RL controller for n_episodes on each task; record the mean
   `total_evals` per episode as the budget B.
2. For each baseline, run the baseline for exactly B evals on the same
   task's initial random solution, same seed family.
3. Compare final best-values.

## 3. Task family and dimensions

Pick a small-but-representative evaluation grid:

| d | N values (eval)      | why                                           |
|---|----------------------|-----------------------------------------------|
| 3 | {5, 7, 10, 15, 20}   | smallest dim, cheapest evals, most tasks      |
| 4 | {5, 7, 10, 16, 20}   | sharpest paper-record anchor points           |
| 5 | {6, 7, 10, 14, 20}   | includes three N where records are suspect (§Stage A) |
| 6 | {7, 10, 16, 20, 24}  | closed-form hyp volume, largest dim tested    |

20 tasks total.  Seed each task 10 times → 200 rollouts per method.

This is a *fixed* evaluation grid, separate from the random sampling the
training controller saw.  Only the mixed-d controller sees (d, N) outside
the training distribution, but even single-d controllers evaluate on N
values not in their training range — an intentional stress.

## 4. Metrics

Report for each (method, d, N):

- **Median final best volume** (primary).
- **IQR across seeds** (secondary — captures variance).
- **Ratio to record** (tertiary — how close to the known LB?).
- **Anytime curve**: best volume vs evaluations, averaged across seeds.
  Shows whether RL finds good solutions *earlier*, not only *eventually*.

Aggregate across the grid:

- **Geometric-mean ratio** of each method to the best baseline per task.
  Single number per method, directly comparable.
- **Win rate**: fraction of (d, N) tasks where method's median > every
  other method's median.

## 5. Phases

### Phase 0 — Preparation (~1 h)

- Verify all four `ppo_d*_final.zip` files load on `make_polytope_env(d_range=(d,d))`.
  d=5 already verified in Stage C; check d ∈ {3, 4, 6}.
- Implement the validation runner (new script `scripts/validate_rl_controller.py`)
  using `evaluate_controller` for RL and `compare_with_baseline` for baselines.
  The runner emits a JSON: `{(method, d, N, seed): best_value, total_evals, …}`.
- Document the exact training config (n_steps, batch_size, lr, horizon)
  that produced each model, alongside the saved zips.

### Phase 1 — Retrain on refactored env (~8–10 h total)

- Four single-d controllers: `train_controller(lambda: make_polytope_env(d_range=(d,d), N_range=(5, 25)))` for d ∈ {3, 4, 5, 6}, 500 k steps each, seed=42.
- One mixed-d controller: `d_range=(3, 6)`, 1 M steps (larger task space needs more steps).  Optional for the first validation pass.
- Save to `results/rl_models/ppo_d{d}_v2_final.zip` and keep the old zips
  for provenance.  Record training config in a sidecar
  `results/rl_models/ppo_d{d}_v2_config.json`.

### Phase 2 — Run the validation grid (~2–4 h)

- 20 tasks × 10 seeds × (1 RL + 4–5 baselines) = 1000–1200 rollouts.
- Most rollouts take seconds; a few LBFGS-hyp rollouts take minutes.
  Parallelize across tasks via `multiprocessing.Pool`.  The existing
  `compare_with_baseline` does not parallelize — add a `--workers`
  flag to the validator.
- Output: one JSON of raw results + a summary CSV for plotting.

### Phase 3 — Analysis (~2 h)

- Plot anytime curves per (d, N): x=evals, y=best volume, lines for each method.
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

## 9. Open questions for the user before kicking this off

1. **Mixed-d controller — include or defer?** Single-d alone is enough for
   the go/no-go decision; mixed-d is an ablation.  I'd defer to save half
   the training time unless you specifically want it now.
2. **Fresh training seed or reuse 42?** Seed 42 matches the old zips' seed
   and keeps reproducibility straightforward.  No reason to change.
3. **Training step budget.** 500 k was the default in the legacy trainer.
   Is that enough on the refactored env?  Unknown, but matching the old
   setup makes it a controlled comparison.  If RL loses at 500 k we can
   extend to 1 M as a follow-up.
4. **GPU for PPO training?** SB3 PPO with MLP policy is CPU-bound for this
   network size; GPU will not help much.  Default device="cpu".
