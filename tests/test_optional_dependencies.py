"""Regression tests for optional dependency boundaries."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_h3_core_imports_and_evaluates_when_torch_is_unavailable() -> None:
    script = textwrap.dedent(
        """
        import builtins
        import sys

        import numpy as np

        real_import = builtins.__import__

        def without_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise ModuleNotFoundError(
                    "Torch blocked by optional-dependency regression",
                    name="torch",
                )
            return real_import(name, *args, **kwargs)

        builtins.__import__ = without_torch

        from ideal_poly_volume_toolkit import geometry
        from ideal_poly_volume_toolkit.optimization import negative_volume_objective

        finite_vertices = np.array(
            [0.0j, 1.0 + 0.0j, np.exp(1j * np.pi / 3.0)],
            dtype=np.complex128,
        )
        volume = geometry.ideal_poly_volume_via_delaunay(
            finite_vertices,
            mode="bloch_wigner",
        )
        assert abs(volume - 1.0149416064096537) < 2e-15
        assert negative_volume_objective([np.pi / 3.0, np.pi / 2.0], 4) < 0.0
        assert "torch" not in sys.modules

        try:
            geometry.lob_fast(None)
        except ImportError as exc:
            message = str(exc)
            assert "ideal_poly_volume_toolkit[torch]" in message
            assert "python -m pip install" in message
        else:
            raise AssertionError("Torch API unexpectedly worked with Torch blocked")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_private_geometry_torch_compatibility_exports_are_lazy_aliases() -> None:
    from ideal_poly_volume_toolkit import geometry
    from ideal_poly_volume_toolkit import geometry_torch

    assert geometry._LobFn is geometry_torch._LobFn
    assert (
        geometry._angles_for_triangle_torch
        is geometry_torch._angles_for_triangle_torch
    )
    assert geometry._lob_value_series_torch is geometry_torch._lob_value_series_torch
