"""Structural contracts for staged script-directory migrations."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DEBUG_SCRIPTS = (
    "debug_gradient_sign.py",
    "debug_original.py",
    "test_from_bad_init.py",
    "test_gradients.py",
    "test_lbfgs_issue.py",
)


@pytest.mark.parametrize("filename", DEBUG_SCRIPTS)
def test_legacy_debug_launcher_delegates_to_canonical_script(
    filename: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_path = ROOT / "scripts" / filename
    canonical_path = ROOT / "scripts" / "_debug" / filename
    spec = importlib.util.spec_from_file_location(
        f"legacy_{legacy_path.stem}", legacy_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    invocation: dict[str, str] = {}

    def fake_run_path(path: str, *, run_name: str) -> None:
        invocation.update(path=path, run_name=run_name)

    monkeypatch.setattr(module.runpy, "run_path", fake_run_path)
    module.main()

    assert module._TARGET == canonical_path
    assert canonical_path.is_file()
    assert invocation == {"path": str(canonical_path), "run_name": "__main__"}
    compile(canonical_path.read_text(encoding="utf-8"), str(canonical_path), "exec")
