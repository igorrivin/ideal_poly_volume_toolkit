"""Public package namespace and installed distribution version."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("ideal_poly_volume_toolkit")
except PackageNotFoundError:  # Source tree imported before installation.
    __version__ = "0+unknown"

from ideal_poly_volume_toolkit.hyperbolic_4_simplex_volume import (
    hyperbolic_4_simplex_volume,
    hyperbolic_4_simplex_data,
    regular_ideal_simplex_vertices,
    snap_vertices_to_boundary,
    euclidean_simplex_volume,
    monte_carlo_klein_integral,
)


__all__ = [
    "__version__",
    "euclidean_simplex_volume",
    "hyperbolic_4_simplex_data",
    "hyperbolic_4_simplex_volume",
    "monte_carlo_klein_integral",
    "regular_ideal_simplex_vertices",
    "snap_vertices_to_boundary",
]
