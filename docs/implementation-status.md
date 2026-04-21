# Implementation Status

Snapshot as of 2026-04-18.

This document tracks what is working, what is broken, and the current reorganization migration. For open research questions see `tasks/todo.md`. For accumulated corrections see `tasks/lessons.md`.

## Working

- **H^3 volume via Delaunay** — validated against known cases; correct for generic point sets.
- **H^3 volume via hull-project-back** — correct for generic point sets.
- **Penner–Rivin holonomy and arithmeticity** — invariant trace field pipeline runs end-to-end.
- **H^4 volume via Schläfli** — matches known exact values for rectified 5-cell and first doubling (vol = 2π²/9, 4π²/9 respectively).
- **Volume threshold oracle** (`volume_threshold.py`).
- **CLI and GUI** — `bin/optimize_polyhedron.py`, `bin/analyze_distribution.py`, `bin/gui.py` run locally.

## Known issues

- **10–10 duoprism and other highly symmetric point sets in H^4** — Delaunay-based pipeline gives pole-dependent (wildly wrong) answers. Needs cell-by-cell computation or a robust non-generic triangulation. See `tasks/todo.md` §"Volume computation for non-generic ideal polytopes".
- **`scripts/optimize_facet_full.py`** — two-level facet optimization has a degeneracy bug; treat as broken.
- **Non-simplicial facet counting** — floating-point ConvexHull always triangulates; merging coplanar facets needs tolerance calibration. See `tasks/todo.md`.
- **Hodge spectral realization for d ≥ 5** — WIP; naive edge-eigenvector averaging does not work; see `tasks/todo.md` for the Hodge-decomposition plan.
- **Hyperbolic inradius (Chebyshev center)** — optimizer drifts to Klein-ball boundary; the formula itself is correct.

## HF Spaces deployment

- `app.py` (repo root) is a thin shim that imports `bin.gui.create_gui` and launches for HuggingFace Spaces Docker SDK. Last touched 2025-12-09.
- `bin/gui.py` is the actual GUI (last touched 2025-11-30).
- The HF Spaces deployment attempt was not successful. Both files pin an old Gradio version and need a Gradio update before any redeployment.
- Neither file is dead on its own — they are complementary. If HF Spaces is formally abandoned, `app.py` can be removed.

## Documentation gaps being filled

As of 2026-04-18 the CLAUDE.md `@`-imports referenced three docs that did not exist: `architecture.md`, `module-index.md`, `implementation-status.md`. These have now been written (this is one of them). They reflect the **current** layout, not the proposed post-reorganization layout.

## Active reorganization

A staged reorganization is in progress. Stages in order:

- **Stage 0 — docs scaffolding.** Write the three missing `docs/` files above. *Status: done (commit 5b1c45e).*
- **Stage 1 — repository-root cleanup.** Move misplaced scripts, result files, and benchmark JSONs out of the root; archive `paper.zip` and `fivesix/`; add `.gradio/` to `.gitignore`. *Status: done (commit 5b1c45e).*
- **Stage 2 — `results/` cleanup.** Move loose JSON / TXT / log files from `results/` root into `results/data/` or `results/logs/`. *Status: pending.*
- **Stage 3 — `scripts/` grouping.** Partition the flat `scripts/` directory into topical subdirectories (`polytopes_4d/`, `rl/`, `euclidean_deficit/`, `llm_benchmark/`, `facet_optimization/`, `inscribability/`, `_debug/`). *Status: pending.*
- **Stage 4 — package subdivision.** Split `ideal_poly_volume_toolkit/` into `geometry/`, `geometry_hd/`, `rivin/`, `realization/`, `combinatorics/`, `rl/`, `thresholds/`. *Status: deferred — flat package is acceptable for now.*
- **Stage 5 — `examples/` cleanup.** Move loose examples into topical subdirectories. *Status: pending.*

A separate benchmark + data-recovery track runs alongside the reorg:

- **Stage A — H^5 hyperbolic volume recovery + hd-volume-LB benchmark.** *Status: done (commit b29cfa8).* Flagged a data-retention gap: d=5 configs at N ∈ {12, 14, 20} are suboptimal vs paper Table 5.
- **Stage B — RL harness modularization, design.** *Status: done.* Design document: `docs/plans/rl-harness-design.md`.
- **Stage C — RL harness modularization, implementation.** *Status: pending.* Must satisfy the four exit criteria in `docs/plans/rl-harness-design.md` §8.

Ownership map and rationale: see `tasks/todo.md` and the reorganization discussion recorded in session memory.
