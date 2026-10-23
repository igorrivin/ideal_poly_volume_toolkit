"""Compatibility launcher for ``scripts/_debug/test_lbfgs_issue.py``."""

from pathlib import Path
import runpy

_TARGET = Path(__file__).resolve().parent / "_debug" / "test_lbfgs_issue.py"


def main() -> None:
    runpy.run_path(str(_TARGET), run_name="__main__")


if __name__ == "__main__":
    main()
