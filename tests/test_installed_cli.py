"""Tests for package-owned command entry points."""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import version

import ideal_poly_volume_toolkit
from ideal_poly_volume_toolkit.optimization_cli import build_parser


def test_package_exposes_installed_version() -> None:
    assert ideal_poly_volume_toolkit.__version__ == version(
        "ideal_poly_volume_toolkit"
    )


def test_optimization_cli_parser_accepts_explicit_arguments() -> None:
    args = build_parser().parse_args(
        [
            "--vertices",
            "7",
            "--trials",
            "2",
            "--workers",
            "-1",
        ]
    )

    assert args.vertices == 7
    assert args.trials == 2
    assert args.workers == -1


def test_optimization_cli_module_help() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ideal_poly_volume_toolkit.optimization_cli",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Optimize ideal polyhedron volumes" in completed.stdout
    assert "--workers" in completed.stdout
