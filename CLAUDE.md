# Project Claude Rules

@README.md
@docs/architecture.md
@docs/module-index.md
@docs/implementation-status.md

## Purpose

This repository is organized by functionality. Every new feature must have one clear owning module and one discoverable entry in the docs.

## Working rules

- Before changing code, identify the owning feature/module.
- If the owning module is unclear, stop and ask before creating files.
- Prefer extending an existing feature directory over creating a new top-level directory.
- Do not place feature-specific logic in `shared/` or `utils/` unless it is genuinely cross-cutting.
- For changes touching more than 3 files or more than 1 feature area, propose a plan before editing.

## Documentation rules

- When adding a module, entry point, service, or major file, update `docs/module-index.md` in the same change.
- When changing a specific feature area, update the nearest directory `README.md` if ownership, entry points, commands, or responsibilities changed.
- Do not leave new structure undocumented.

## Commands

- Setup (development): `python3 -m pip install -e '.[dev]'`
- Setup (full RL/research environment): install `requirements.txt`, then run
  `python3 -m pip install -e .`
- Test: `python3 -m pytest -q`
- Lint changed Python files: `ruff check <paths>`
- Typecheck: not configured
- Run GUI: `python3 app.py`
- Run optimizer CLI: `python3 bin/optimize_polyhedron.py --help`

## Placement rules

- `features/` contains product or workflow functionality.
- `shared/` contains genuinely reusable code used by multiple features.
- `scripts/` contains operational scripts, not product logic.
- `docs/` contains architecture, indices, plans, and implementation tracking.

## Refactor rules

- Do not rename or move large sets of files without explicit approval.
- For structural cleanup, first produce a migration plan and proposed ownership map.
- Prefer a sequence of small moves plus doc updates over a single large reorganization.
