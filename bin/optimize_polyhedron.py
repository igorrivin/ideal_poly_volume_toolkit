#!/usr/bin/env python3
"""Compatibility shim for the installed H³ optimization command.

New code may run ``ideal-poly-optimize`` after installation or import the
adapter from :mod:`ideal_poly_volume_toolkit.optimization_cli`.
"""

from ideal_poly_volume_toolkit.optimization_cli import (
    analyze_structure,
    build_parser,
    compute_volume,
    main,
    reconstruct_vertices,
    spherical_to_complex,
)


__all__ = [
    "analyze_structure",
    "build_parser",
    "compute_volume",
    "main",
    "reconstruct_vertices",
    "spherical_to_complex",
]


if __name__ == "__main__":
    raise SystemExit(main())
