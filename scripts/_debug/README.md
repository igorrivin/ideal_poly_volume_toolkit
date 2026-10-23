# Manual Debug Scripts

This directory owns developer-only diagnostics for historical Torch gradient
and optimizer behavior. These files are exploratory executables, not product
logic, supported command-line interfaces, or pytest tests.

Run a diagnostic from the repository root after installing the relevant
optional dependencies, for example:

```bash
python3 scripts/_debug/debug_gradient_sign.py
```

The former paths directly under `scripts/` remain thin compatibility
launchers throughout the 0.4.x release line and may be removed in 0.5.0. New
references should use the `_debug/` paths. Diagnostics that expose a
reproducible correctness contract should be converted into a focused test
under `tests/` rather than added here.
