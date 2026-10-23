"""Differentiable fixed-infinity-chart kernels for ideal H⁴ volume.

The ordinary :mod:`ideal_poly_volume_toolkit.geometry_4d` implementation
constructs a Delaunay triangulation in NumPy/SciPy and evaluates its ideal
4-simplices one at a time.  This optional-Torch module owns only the smooth
part of that calculation: given finite boundary points in ``R³`` and a
labelled tetrahedralization, it evaluates the same angle formula in one
vectorized autograd graph.

Triangulation remains deliberately outside this module.  A caller performing
optimization must rebuild it after moving the points; differentiating through
a discrete Delaunay choice would not be meaningful.  The supplied simplex
labels are used exactly as given, so no relabelling is hidden in the kernel.
"""

from __future__ import annotations

import math
from numbers import Real

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "PyTorch-backed H^4 geometry requires the 'torch' extra; install it "
        "with `python -m pip install 'ideal_poly_volume_toolkit[torch]'`."
    ) from exc


_FACE_VERTICES = (
    (1, 2, 3),
    (0, 2, 3),
    (0, 1, 3),
    (0, 1, 2),
)
_EDGE_FACE_COMPLEMENTS = (
    (2, 3),
    (1, 3),
    (1, 2),
    (0, 3),
    (0, 2),
    (0, 1),
)
_SUPPORTED_DTYPES = (torch.float32, torch.float64)


def _projected_points(points: torch.Tensor) -> torch.Tensor:
    if not isinstance(points, torch.Tensor):
        raise TypeError("points must be a torch.Tensor")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (M, 3)")
    if points.shape[0] < 4:
        raise ValueError("at least four projected points are required")
    if points.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("points must use torch.float32 or torch.float64")
    if not bool(torch.all(torch.isfinite(points)).item()):
        raise ValueError("points must contain only finite coordinates")
    return points


def _labelled_simplices(
    simplices: torch.Tensor,
    *,
    n_points: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(simplices, torch.Tensor):
        raise TypeError("simplices must be a torch.Tensor")
    if simplices.ndim != 2 or simplices.shape[1] != 4:
        raise ValueError("simplices must have shape (T, 4)")
    if simplices.shape[0] < 1:
        raise ValueError("at least one projected tetrahedron is required")
    if simplices.dtype != torch.long:
        raise TypeError("simplices must be a torch.int64 tensor")

    labels = simplices.to(device=device)
    minimum = int(torch.min(labels).item())
    maximum = int(torch.max(labels).item())
    if minimum < 0 or maximum >= n_points:
        raise ValueError(
            f"simplex labels must lie in [0, {n_points}); "
            f"received minimum {minimum} and maximum {maximum}"
        )
    sorted_labels = torch.sort(labels, dim=1).values
    if bool(torch.any(sorted_labels[:, 1:] == sorted_labels[:, :-1]).item()):
        raise ValueError("every simplex must contain four distinct point labels")
    return labels


def _positive_finite_real(value: Real | None, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite real number or None")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number or None")
    return result


def projected_similarity_gauge(
    points: torch.Tensor,
    *,
    minimum_relative_spread: Real | None = None,
) -> torch.Tensor:
    """Center projected points and give them unit root-mean-square radius.

    Translation and positive scaling of the finite ``R³`` chart are
    boundary actions of hyperbolic isometries fixing infinity.  Removing
    those four redundant directions makes an optimization chart substantially
    better conditioned while preserving the represented ideal polytope.

    The operation is composed entirely of Torch primitives and therefore
    remains differentiable with respect to ``points``.  Rotational gauge
    freedom is intentionally left untouched.
    """

    points = _projected_points(points)
    requested_tolerance = _positive_finite_real(
        minimum_relative_spread,
        name="minimum_relative_spread",
    )
    centered = points - torch.mean(points, dim=0, keepdim=True)
    mean_squared_radius = torch.mean(torch.sum(centered * centered, dim=1))

    dtype_epsilon = torch.finfo(points.dtype).eps
    relative_tolerance = (
        128.0 * dtype_epsilon if requested_tolerance is None else requested_tolerance
    )
    coordinate_scale = torch.max(torch.abs(points.detach()))
    minimum_squared_radius = relative_tolerance * coordinate_scale * coordinate_scale
    if not bool(torch.isfinite(mean_squared_radius).item()) or bool(
        (mean_squared_radius <= minimum_squared_radius).item()
    ):
        raise ValueError(
            "projected points have insufficient finite spread for a stable "
            "similarity gauge"
        )

    gauged = centered / torch.sqrt(mean_squared_radius)
    if not bool(torch.all(torch.isfinite(gauged)).item()):
        raise FloatingPointError("similarity gauging produced non-finite points")
    return gauged


def ideal_4simplex_volumes_from_projected_tetrahedra(
    points: torch.Tensor,
    simplices: torch.Tensor,
    *,
    minimum_relative_determinant: Real | None = None,
) -> torch.Tensor:
    """Evaluate labelled ideal 4-simplex volumes in a fixed infinity chart.

    Parameters
    ----------
    points:
        Finite ideal-boundary coordinates with shape ``(M, 3)``.
    simplices:
        ``torch.int64`` point labels with shape ``(T, 4)``.  Row order and
        point labels are preserved; permuting a row changes no volume.
    minimum_relative_determinant:
        Optional positive degeneracy threshold.  By default a dtype-scaled
        threshold rejects tetrahedra whose determinant is numerically zero
        relative to their longest edge.

    Returns
    -------
    torch.Tensor
        A differentiable vector of shape ``(T,)`` containing the same angle
        formula used by :func:`geometry_4d.ideal_4simplex_volume_from_tet`.
    """

    points = _projected_points(points)
    labels = _labelled_simplices(
        simplices,
        n_points=int(points.shape[0]),
        device=points.device,
    )
    requested_tolerance = _positive_finite_real(
        minimum_relative_determinant,
        name="minimum_relative_determinant",
    )
    tetrahedra = points[labels]
    edge_matrices = tetrahedra[:, 1:] - tetrahedra[:, :1]
    determinants = torch.linalg.det(edge_matrices)
    longest_edges = torch.amax(
        torch.linalg.vector_norm(edge_matrices, dim=2),
        dim=1,
    )
    relative_tolerance = (
        128.0 * torch.finfo(points.dtype).eps
        if requested_tolerance is None
        else requested_tolerance
    )
    determinant_floor = relative_tolerance * longest_edges**3
    if bool(torch.any(torch.abs(determinants) <= determinant_floor).item()):
        raise ValueError("simplices contain a degenerate projected tetrahedron")

    system = 2.0 * edge_matrices
    right_hand_side = torch.sum(
        tetrahedra[:, 1:] ** 2 - tetrahedra[:, :1] ** 2,
        dim=2,
    )
    circumcenters = torch.linalg.solve(
        system,
        right_hand_side.unsqueeze(2),
    ).squeeze(2)
    circumradii = torch.linalg.vector_norm(
        tetrahedra[:, 0] - circumcenters,
        dim=1,
    )

    toward_normals = []
    outward_normals = []
    for opposite, (first, second, third) in enumerate(_FACE_VERTICES):
        normal = torch.linalg.cross(
            tetrahedra[:, second] - tetrahedra[:, first],
            tetrahedra[:, third] - tetrahedra[:, first],
            dim=1,
        )
        direction = torch.sum(
            normal * (tetrahedra[:, opposite] - tetrahedra[:, first]),
            dim=1,
        )
        normal = torch.where((direction < 0.0)[:, None], -normal, normal)
        normal = normal / torch.linalg.vector_norm(normal, dim=1)[:, None]
        toward_normals.append(normal)
        outward_normals.append(-normal)
    toward = torch.stack(toward_normals, dim=1)
    outward = torch.stack(outward_normals, dim=1)

    euclidean_dihedral_angles = []
    for first_face, second_face in _EDGE_FACE_COMPLEMENTS:
        cosine = torch.sum(
            outward[:, first_face] * outward[:, second_face],
            dim=1,
        )
        euclidean_dihedral_angles.append(
            torch.pi - torch.acos(torch.clamp(cosine, -1.0, 1.0))
        )
    euclidean_angle_sum = torch.stack(
        euclidean_dihedral_angles,
        dim=1,
    ).sum(dim=1)

    signed_face_distances = []
    for opposite, (first, _, _) in enumerate(_FACE_VERTICES):
        signed_face_distances.append(
            torch.sum(
                toward[:, opposite] * (circumcenters - tetrahedra[:, first]),
                dim=1,
            )
        )
    distance_ratios = torch.stack(signed_face_distances, dim=1) / (circumradii[:, None])
    hemisphere_angle_sum = torch.acos(torch.clamp(distance_ratios, -1.0, 1.0)).sum(
        dim=1
    )
    volumes = (torch.pi / 3.0) * (
        4.0 * torch.pi - euclidean_angle_sum - hemisphere_angle_sum
    )
    if not bool(torch.all(torch.isfinite(volumes)).item()):
        raise FloatingPointError("ideal 4-simplex evaluation was non-finite")
    return volumes


def ideal_polytope_volume_4d_from_projected(
    points: torch.Tensor,
    simplices: torch.Tensor,
    *,
    apply_similarity_gauge: bool = False,
    minimum_relative_spread: Real | None = None,
    minimum_relative_determinant: Real | None = None,
) -> torch.Tensor:
    """Sum ideal H⁴ volume over a supplied projected tetrahedralization.

    This function does not construct or cache combinatorics.  During moving-
    vertex optimization, callers must compute a fresh valid tetrahedralization
    and pass its original point labels on every evaluation.
    """

    if not isinstance(apply_similarity_gauge, bool):
        raise TypeError("apply_similarity_gauge must be a bool")
    prepared = (
        projected_similarity_gauge(
            points,
            minimum_relative_spread=minimum_relative_spread,
        )
        if apply_similarity_gauge
        else _projected_points(points)
    )
    return torch.sum(
        ideal_4simplex_volumes_from_projected_tetrahedra(
            prepared,
            simplices,
            minimum_relative_determinant=minimum_relative_determinant,
        )
    )


__all__ = [
    "ideal_4simplex_volumes_from_projected_tetrahedra",
    "ideal_polytope_volume_4d_from_projected",
    "projected_similarity_gauge",
]
