"""Shared data types for the polytope specialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class PolytopeSolution:
    """Solution type for `IdealPolytopeTask`.

    vertices : (N, d) numpy array, rows on S^{d-1}.
    simplices : Delaunay triangulation (list of index lists of length d+1).
    """
    vertices: np.ndarray
    simplices: List[List[int]]
    _simplices_key: bytes | None = field(default=None, repr=False, compare=False)
    _simplices_signature: tuple[tuple[int, ...], ...] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
