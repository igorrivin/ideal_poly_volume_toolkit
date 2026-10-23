"""H^4 candidate mining and diagnostics for symmetry benchmarks.

This module deliberately separates three numerical questions:

* unconstrained volume improvement with no fixed combinatorics;
* geometric-symmetry certification, delegated to :mod:`symmetry`;
* pole stability and an optional, explicitly diagnostic local-maximum test.

The local test works on the quotient of ``(S^3)^N`` by the ten-dimensional
``PO(4, 1)`` action.  It is a finite-difference diagnostic, not a validated
existence or maximality certificate.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
from numbers import Integral, Real
import time
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, Delaunay

from ...geometry_4d import ideal_polytope_volume_4d, stereo_project_S3
from .optimizers import FreshTopologyADHyperbolicD4, LBFGSHyperbolicD4
from .task import IdealPolytopeTask
from .types import PolytopeSolution


DEFAULT_VOLUME_SPREAD_TOLERANCE = 1.0e-8
DEFAULT_FD_STEP = 1.0e-4
_GAUGE_DIMENSION_H4 = 10
_BALANCE_RESIDUAL_TOLERANCE = 1.0e-10
_BALANCE_MAX_FUNCTION_EVALUATIONS = 200


def _vertices_array(vertices: Any) -> np.ndarray:
    """Return an owned H^4 boundary-coordinate array.

    Unit length, distinctness, full dimensionality, extremality, and strict
    simpliciality are intentionally checked by ``audit_hyperbolic_symmetry``.
    Only the representation needed to dispatch that audit is validated here.
    """

    try:
        points = np.asarray(vertices, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("vertices must be a real array of shape (N, 4)") from exc
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("vertices must have shape (N, 4) for ideal H^4 points")
    if points.shape[0] < 5:
        raise ValueError("an H^4 polytope requires at least 5 vertices")
    return np.array(points, dtype=np.float64, copy=True)


def _normalized_optimization_vertices(vertices: Any) -> np.ndarray:
    """Validate and normalize an owned H4 optimization start."""

    points = _vertices_array(vertices)
    norms = np.linalg.norm(points, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0.0):
        raise ValueError("optimization vertices must be finite and nonzero")
    return points / norms[:, None]


def _initial_coordinate_sha256(points: np.ndarray) -> str:
    """Digest the exact normalized float64 start in a fixed byte order."""

    normalized = np.ascontiguousarray(points, dtype="<f8")
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def _positive_real(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number")
    return result


def _nonnegative_real(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a nonnegative finite real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a nonnegative finite real number")
    return result


def _positive_integer(value: Any, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _requested_poles(
    pole_indices: Iterable[int] | None, n: int
) -> tuple[list[int], bool]:
    if pole_indices is None:
        return list(range(n)), True
    if isinstance(pole_indices, (str, bytes)) or isinstance(pole_indices, Integral):
        raise TypeError("pole_indices must be an iterable of distinct integers or None")
    try:
        raw = list(pole_indices)
    except TypeError as exc:
        raise TypeError(
            "pole_indices must be an iterable of distinct integers or None"
        ) from exc
    if not raw:
        raise ValueError("pole_indices must not be empty")
    poles: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("every pole index must be an integer")
        pole = int(value)
        if not 0 <= pole < n:
            raise ValueError(f"pole index {pole} is outside [0, {n})")
        poles.append(pole)
    if len(set(poles)) != len(poles):
        raise ValueError("pole_indices must not contain duplicates")
    return poles, set(poles) == set(range(n))


def _skipped_pole_volume_audit(
    poles: list[int],
    *,
    all_poles_requested: bool,
    tolerance: float,
    reason: str,
) -> dict[str, Any]:
    """Return the ordinary pole-audit schema when coordinates are unusable."""

    failures = [
        {
            "pole_index": pole,
            "error_type": "InvalidBoundaryCoordinates",
            "message": reason,
        }
        for pole in poles
    ]
    return {
        "all_poles_requested": all_poles_requested,
        "requested_indices": poles,
        "requested_count": len(poles),
        "successful_count": 0,
        "failure_count": len(failures),
        "values": [],
        "failures": failures,
        "minimum_volume": None,
        "maximum_volume": None,
        "spread": None,
        "spread_tolerance": tolerance,
        "within_tolerance": False,
    }


def _pole_volume_audit(
    points: np.ndarray,
    poles: list[int],
    *,
    all_poles_requested: bool,
    tolerance: float,
) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for pole in poles:
        try:
            volume, simplex_count = ideal_polytope_volume_4d(points, pole_idx=pole)
            volume = float(volume)
            if not np.isfinite(volume):
                raise FloatingPointError(
                    "volume evaluation returned a non-finite value"
                )
            values.append(
                {
                    "pole_index": pole,
                    "volume": volume,
                    "simplex_count": int(simplex_count),
                }
            )
        except Exception as exc:  # Numerical audit: retain each pole failure.
            failures.append(
                {
                    "pole_index": pole,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    volumes = [entry["volume"] for entry in values]
    minimum = min(volumes) if volumes else None
    maximum = max(volumes) if volumes else None
    spread = maximum - minimum if volumes else None
    complete = len(values) == len(poles)
    within_tolerance = bool(complete and spread is not None and spread <= tolerance)
    return {
        "all_poles_requested": all_poles_requested,
        "requested_indices": poles,
        "requested_count": len(poles),
        "successful_count": len(values),
        "failure_count": len(failures),
        "values": values,
        "failures": failures,
        "minimum_volume": minimum,
        "maximum_volume": maximum,
        "spread": spread,
        "spread_tolerance": tolerance,
        "within_tolerance": within_tolerance,
    }


def _tangent_basis(points: np.ndarray) -> np.ndarray:
    """Block-orthonormal basis for the tangent space of ``(S^3)^N``."""

    n = len(points)
    basis = np.zeros((4 * n, 3 * n), dtype=np.float64)
    for vertex, point in enumerate(points):
        # Full SVD gives three orthonormal vectors perpendicular to point.
        _, _, vh = np.linalg.svd(point.reshape(1, 4), full_matrices=True)
        block = vh[1:].T
        basis[4 * vertex : 4 * vertex + 4, 3 * vertex : 3 * vertex + 3] = block
    return basis


def _po41_infinitesimal_directions(points: np.ndarray) -> np.ndarray:
    """The six rotations and four conformal boosts on the boundary S^3."""

    directions: list[np.ndarray] = []
    for row in range(4):
        for column in range(row + 1, 4):
            generator = np.zeros((4, 4), dtype=np.float64)
            generator[row, column] = 1.0
            generator[column, row] = -1.0
            directions.append((points @ generator.T).reshape(-1))
    for coordinate in range(4):
        axis = np.zeros(4, dtype=np.float64)
        axis[coordinate] = 1.0
        field = axis[None, :] - points[:, coordinate, None] * points
        directions.append(field.reshape(-1))
    return np.column_stack(directions)


def _open_ball_parameter(unconstrained: np.ndarray) -> np.ndarray:
    """Map an unconstrained vector smoothly into the open unit 4-ball."""

    radius = float(np.linalg.norm(unconstrained))
    if radius == 0.0:
        return np.zeros_like(unconstrained)
    return (np.tanh(radius) / radius) * unconstrained


def _mobius_move_to_origin(points: np.ndarray, ball_point: np.ndarray) -> np.ndarray:
    """Apply the boundary ball automorphism taking ``ball_point`` to zero.

    For unit ``x`` and ``a`` in the open ball this is ``(-a) ⊕ x`` in
    Möbius-addition notation.  Renormalizing removes only floating-point
    drift: mathematically the formula maps the sphere to itself exactly.
    """

    squared_radius = float(np.dot(ball_point, ball_point))
    dots = points @ ball_point
    denominators = 1.0 - 2.0 * dots + squared_radius
    minimum_denominator = float(np.min(denominators))
    if minimum_denominator <= 64.0 * np.finfo(np.float64).eps:
        raise FloatingPointError(
            "conformal centering is too close to a boundary singularity"
        )
    transformed = (
        (1.0 - squared_radius) * points - 2.0 * (1.0 - dots)[:, None] * ball_point
    ) / denominators[:, None]
    norms = np.linalg.norm(transformed, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0.0):
        raise FloatingPointError("conformal centering produced invalid points")
    return transformed / norms[:, None]


def _conformally_balance(points: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose the canonical Hersch chart whose boundary centroid is zero.

    Equal-weight, pairwise-distinct configurations have a conformal barycenter
    in the open ball.  Solving for it before finite differencing prevents the
    Euclidean quotient metric from depending on an arbitrary Lorentz-boosted
    boundary chart.  Rotational freedom remains, but it is orthogonal and does
    not change the reported gradient norm or Hessian spectrum.
    """

    initial_centroid_norm = float(np.linalg.norm(np.mean(points, axis=0)))

    def residual(unconstrained: np.ndarray) -> np.ndarray:
        ball_point = _open_ball_parameter(unconstrained)
        return np.mean(_mobius_move_to_origin(points, ball_point), axis=0)

    solve = least_squares(
        residual,
        np.zeros(4, dtype=np.float64),
        method="trf",
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=_BALANCE_MAX_FUNCTION_EVALUATIONS,
    )
    ball_point = _open_ball_parameter(np.asarray(solve.x, dtype=np.float64))
    balanced = _mobius_move_to_origin(points, ball_point)
    centroid = np.mean(balanced, axis=0)
    residual_norm = float(np.linalg.norm(centroid))
    parameter_norm = float(np.linalg.norm(ball_point))
    accepted = bool(
        solve.success
        and np.isfinite(residual_norm)
        and residual_norm <= _BALANCE_RESIDUAL_TOLERANCE
        and parameter_norm < 1.0
    )
    diagnostics = {
        "method": "equal_weight_conformal_barycenter",
        "accepted": accepted,
        "solver_success": bool(solve.success),
        "solver_status": int(solve.status),
        "solver_message": str(solve.message),
        "function_evaluations": int(solve.nfev),
        "initial_centroid_norm": initial_centroid_norm,
        "balanced_centroid": [float(value) for value in centroid],
        "balanced_centroid_norm": residual_norm,
        "residual_tolerance": _BALANCE_RESIDUAL_TOLERANCE,
        "ball_parameter": [float(value) for value in ball_point],
        "ball_parameter_norm": parameter_norm,
    }
    return balanced, diagnostics


def _quotient_directions(points: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    tangent = _tangent_basis(points)
    gauge_ambient = _po41_infinitesimal_directions(points)
    gauge_tangent = tangent.T @ gauge_ambient
    u, singular_values, _ = np.linalg.svd(gauge_tangent, full_matrices=True)
    if singular_values.size:
        tolerance = (
            max(gauge_tangent.shape) * np.finfo(np.float64).eps * singular_values[0]
        )
    else:  # pragma: no cover - H^4 always supplies ten columns.
        tolerance = 0.0
    rank = int(np.sum(singular_values > tolerance))
    physical_tangent = u[:, rank:]
    ambient = tangent @ physical_tangent
    diagnostics = {
        "tangent_dimension": int(tangent.shape[1]),
        "gauge_dimension": _GAUGE_DIMENSION_H4,
        "gauge_rank": rank,
        "gauge_singular_values": [float(value) for value in singular_values],
        "physical_dimension": int(3 * len(points) - _GAUGE_DIMENSION_H4),
        "computed_quotient_dimension": int(ambient.shape[1]),
    }
    return ambient, diagnostics


def _sphere_exponential(points: np.ndarray, tangent_delta: np.ndarray) -> np.ndarray:
    delta = tangent_delta.reshape(points.shape)
    lengths = np.linalg.norm(delta, axis=1)
    result = np.empty_like(points)
    nonzero = lengths > 0.0
    result[~nonzero] = points[~nonzero]
    if np.any(nonzero):
        length = lengths[nonzero, None]
        result[nonzero] = (
            np.cos(length) * points[nonzero]
            + (np.sin(length) / length) * delta[nonzero]
        )
    return result


def _hull_signature(points: np.ndarray) -> tuple[tuple[int, ...], ...]:
    hull = ConvexHull(points)
    return tuple(sorted(tuple(sorted(map(int, facet))) for facet in hull.simplices))


def _pole_delaunay_signature(
    points: np.ndarray, pole: int
) -> tuple[tuple[int, ...], ...]:
    projected, finite_indices = stereo_project_S3(points, pole_idx=pole)
    triangulation = Delaunay(projected)
    simplices = []
    for simplex in triangulation.simplices:
        labels = [pole, *(int(finite_indices[int(index)]) for index in simplex)]
        simplices.append(tuple(sorted(labels)))
    return tuple(sorted(simplices))


def _local_maximum_audit(
    points: np.ndarray,
    *,
    pole: int | None,
    fd_step: float,
    geometry_valid: bool,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "requested": True,
        "performed": False,
        "diagnostic_only": True,
        "status": "pending",
        "classification": "inconclusive",
        "fd_step": fd_step,
        "primary_pole": pole,
        "chamber_stability": None,
        "volume_evaluations": 0,
        "gauge_dimension": _GAUGE_DIMENSION_H4,
        "physical_dimension": int(3 * len(points) - _GAUGE_DIMENSION_H4),
    }

    if not geometry_valid:
        base.update(
            status="skipped_invalid_geometry",
            classification="inconclusive_invalid_geometry",
        )
        return base
    if pole is None:
        base.update(
            status="skipped_no_successful_pole",
            classification="inconclusive_no_volume",
        )
        return base

    try:
        balanced_points, balancing = _conformally_balance(points)
        base["conformal_balancing"] = balancing
    except Exception as exc:
        base.update(
            status="failed_conformal_balancing",
            classification="inconclusive_chart_conditioning",
            failure={"error_type": type(exc).__name__, "message": str(exc)},
        )
        return base
    if not balancing["accepted"]:
        base.update(
            status="skipped_unbalanced_chart",
            classification="inconclusive_chart_conditioning",
        )
        return base
    # Construct the quotient and all finite-difference probes in this
    # canonical representative.  A raw Lorentz boost otherwise changes the
    # Euclidean metric used to select a complement to the gauge orbit.
    points = balanced_points
    base["coordinate_chart"] = "equal_weight_conformal_barycenter"

    try:
        directions, quotient = _quotient_directions(points)
        base.update(quotient)
        if quotient["gauge_rank"] != _GAUGE_DIMENSION_H4:
            base.update(
                status="failed_gauge_rank",
                classification="inconclusive_gauge_rank",
            )
            return base
        physical_dimension = directions.shape[1]
        base_hull = _hull_signature(points)
        base_delaunay = _pole_delaunay_signature(points, pole)
    except Exception as exc:
        base.update(
            status="failed_numerical_setup",
            classification="inconclusive_numerical_failure",
            failure={"error_type": type(exc).__name__, "message": str(exc)},
        )
        return base
    base["base_hull_facet_count"] = len(base_hull)
    base["base_pole_delaunay_simplex_count"] = len(base_delaunay)
    changed_probes: list[dict[str, Any]] = []

    def probe(coefficients: np.ndarray) -> tuple[float, bool]:
        displaced = _sphere_exponential(points, directions @ coefficients)
        hull_same = _hull_signature(displaced) == base_hull
        delaunay_same = _pole_delaunay_signature(displaced, pole) == base_delaunay
        stable = bool(hull_same and delaunay_same)
        if not stable:
            changed_probes.append(
                {
                    "nonzero_coordinates": [
                        int(index) for index in np.flatnonzero(coefficients)
                    ],
                    "hull_signature_same": bool(hull_same),
                    "pole_delaunay_signature_same": bool(delaunay_same),
                }
            )
        volume, _ = ideal_polytope_volume_4d(displaced, pole_idx=pole)
        base["volume_evaluations"] += 1
        volume = float(volume)
        if not np.isfinite(volume):
            raise FloatingPointError("finite-difference volume is non-finite")
        return volume, stable

    try:
        center, _ = ideal_polytope_volume_4d(points, pole_idx=pole)
        center = float(center)
        base["volume_evaluations"] += 1
        plus = np.empty(physical_dimension, dtype=np.float64)
        minus = np.empty(physical_dimension, dtype=np.float64)
        for coordinate in range(physical_dimension):
            offset = np.zeros(physical_dimension, dtype=np.float64)
            offset[coordinate] = fd_step
            plus[coordinate], _ = probe(offset)
            minus[coordinate], _ = probe(-offset)

        gradient = (plus - minus) / (2.0 * fd_step)
        gradient_norm = float(np.linalg.norm(gradient))
        base["center_volume"] = center
        base["projected_gradient"] = [float(value) for value in gradient]
        base["projected_gradient_norm"] = gradient_norm

        if changed_probes:
            base.update(
                status="skipped_non_strict_chamber",
                classification="inconclusive_chamber_change",
                chamber_stability=False,
                changed_probes=changed_probes,
            )
            return base

        hessian = np.zeros((physical_dimension, physical_dimension), dtype=np.float64)
        hessian[np.diag_indices(physical_dimension)] = (plus - 2.0 * center + minus) / (
            fd_step * fd_step
        )
        for row in range(physical_dimension):
            for column in range(row + 1, physical_dimension):
                offset_pp = np.zeros(physical_dimension, dtype=np.float64)
                offset_pm = np.zeros(physical_dimension, dtype=np.float64)
                offset_mp = np.zeros(physical_dimension, dtype=np.float64)
                offset_mm = np.zeros(physical_dimension, dtype=np.float64)
                offset_pp[row] = offset_pp[column] = fd_step
                offset_pm[row], offset_pm[column] = fd_step, -fd_step
                offset_mp[row], offset_mp[column] = -fd_step, fd_step
                offset_mm[row] = offset_mm[column] = -fd_step
                value_pp, _ = probe(offset_pp)
                value_pm, _ = probe(offset_pm)
                value_mp, _ = probe(offset_mp)
                value_mm, _ = probe(offset_mm)
                entry = (value_pp - value_pm - value_mp + value_mm) / (
                    4.0 * fd_step * fd_step
                )
                hessian[row, column] = hessian[column, row] = entry
                if changed_probes:
                    base.update(
                        status="skipped_non_strict_chamber",
                        classification="inconclusive_chamber_change",
                        chamber_stability=False,
                        changed_probes=changed_probes,
                    )
                    return base

        eigenvalues = np.linalg.eigvalsh(hessian)
        gradient_tolerance = max(
            1.0e-6,
            100.0 * np.finfo(np.float64).eps * max(1.0, abs(center)) / fd_step,
        )
        curvature_tolerance = max(
            1.0e-5,
            100.0
            * np.finfo(np.float64).eps
            * max(1.0, abs(center))
            / (fd_step * fd_step),
        )
        maximum_eigenvalue = float(eigenvalues[-1])
        minimum_eigenvalue = float(eigenvalues[0])
        if gradient_norm > gradient_tolerance:
            classification = "not_stationary"
        elif maximum_eigenvalue < -curvature_tolerance:
            classification = "strict_local_maximum"
        elif maximum_eigenvalue <= curvature_tolerance:
            classification = "non_strict_local_maximum_candidate"
        elif minimum_eigenvalue < -curvature_tolerance:
            classification = "saddle"
        else:
            classification = "not_local_maximum"
        base.update(
            performed=True,
            status="completed",
            classification=classification,
            chamber_stability=True,
            changed_probes=[],
            gradient_tolerance=float(gradient_tolerance),
            curvature_tolerance=float(curvature_tolerance),
            projected_hessian_eigenvalues=[float(value) for value in eigenvalues],
            minimum_hessian_eigenvalue=minimum_eigenvalue,
            maximum_hessian_eigenvalue=maximum_eigenvalue,
            negative_eigenvalue_count=int(np.sum(eigenvalues < -curvature_tolerance)),
            near_zero_eigenvalue_count=int(
                np.sum(np.abs(eigenvalues) <= curvature_tolerance)
            ),
            positive_eigenvalue_count=int(np.sum(eigenvalues > curvature_tolerance)),
        )
        return base
    except Exception as exc:
        base.update(
            status="failed_numerical_evaluation",
            classification="inconclusive_numerical_failure",
            failure={"error_type": type(exc).__name__, "message": str(exc)},
            changed_probes=changed_probes,
        )
        return base


def _combined_classification(
    symmetry: Mapping[str, Any],
    pole_audit: Mapping[str, Any],
    local_audit: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    valid_geometry = bool(symmetry.get("valid_geometry", False))
    full_group = bool(symmetry.get("full_group_certified", False))
    order = symmetry.get("geometric_group_order")
    nontrivial = bool(
        full_group
        and symmetry.get("certified_nontrivial_symmetry", False)
        and isinstance(order, Integral)
        and not isinstance(order, bool)
        and int(order) > 1
    )
    pole_stable = bool(pole_audit.get("within_tolerance", False))
    all_poles_requested = bool(pole_audit.get("all_poles_requested", False))
    all_poles_verified = bool(all_poles_requested and pole_stable)
    local_classification = local_audit.get("classification", "not_requested")

    if not valid_geometry:
        label = "invalid_geometry"
    elif not pole_stable:
        label = "pole_unstable_or_incomplete"
    elif not all_poles_requested:
        label = "partial_pole_audit"
    elif not full_group:
        label = "symmetry_inconclusive"
    elif not nontrivial:
        label = "trivial_geometric_symmetry"
    elif not local_audit.get("requested", False):
        label = "symmetric_candidate"
    elif local_audit.get("performed", False) and local_classification == (
        "strict_local_maximum"
    ):
        label = "symmetric_strict_local_maximum_diagnostic"
    elif not local_audit.get("performed", False):
        label = "symmetric_candidate_local_maximum_inconclusive"
    else:
        label = "symmetric_candidate_not_strict_local_maximum"

    basis = {
        "valid_geometry": valid_geometry,
        "pole_volume_stable": pole_stable,
        "pole_volume_stable_on_requested": pole_stable,
        "all_poles_requested": all_poles_requested,
        "all_poles_verified": all_poles_verified,
        "full_group_certified": full_group,
        "geometric_group_order": int(order) if isinstance(order, Integral) else None,
        "nontrivial_geometric_symmetry": nontrivial,
        "local_maximum_classification": local_classification,
    }
    return label, basis


def audit_h4_symmetric_candidate(
    vertices: Any,
    *,
    pole_indices: Iterable[int] | None = None,
    volume_spread_tolerance: float = DEFAULT_VOLUME_SPREAD_TOLERANCE,
    symmetry_kwargs: Mapping[str, Any] | None = None,
    local_maximum: bool = False,
    fd_step: float = DEFAULT_FD_STEP,
) -> dict[str, Any]:
    """Audit one proposed symmetric ideal H^4 benchmark candidate.

    The returned ``classification`` is computed from the independent audit
    results; it is not a restatement of a caller-provided claim.  The optional
    local-maximum result is always marked diagnostic because ordinary finite
    differences and floating-point Delaunay predicates are not a proof.
    """

    points = _vertices_array(vertices)
    tolerance = _nonnegative_real(
        volume_spread_tolerance, name="volume_spread_tolerance"
    )
    step = _positive_real(fd_step, name="fd_step")
    if not isinstance(local_maximum, bool):
        raise TypeError("local_maximum must be a bool")
    if symmetry_kwargs is None:
        symmetry_options: dict[str, Any] = {}
    elif isinstance(symmetry_kwargs, Mapping):
        symmetry_options = dict(symmetry_kwargs)
    else:
        raise TypeError("symmetry_kwargs must be a mapping or None")
    poles, all_poles_requested = _requested_poles(pole_indices, len(points))

    # Imported lazily so this module remains importable while the optional
    # pynauty-backed symmetry feature reports an actionable runtime result.
    from ...symmetry import audit_hyperbolic_symmetry

    symmetry_result = audit_hyperbolic_symmetry(points, **symmetry_options)
    unit_gate_passed = bool(symmetry_result.get("checks", {}).get("unit", False))
    coordinate_preparation: dict[str, Any] = {
        "unit_gate_passed": unit_gate_passed,
        "unit_normalized_for_diagnostics": False,
        "maximum_input_norm_error": symmetry_result.get("topology", {}).get(
            "max_unit_norm_error"
        ),
        "maximum_post_normalization_norm_error": None,
    }
    if unit_gate_passed:
        norms = np.linalg.norm(points, axis=1)
        diagnostic_points = points / norms[:, None]
        coordinate_preparation["unit_normalized_for_diagnostics"] = True
        coordinate_preparation["maximum_post_normalization_norm_error"] = float(
            np.max(np.abs(np.linalg.norm(diagnostic_points, axis=1) - 1.0))
        )
        pole_audit = _pole_volume_audit(
            diagnostic_points,
            poles,
            all_poles_requested=all_poles_requested,
            tolerance=tolerance,
        )
    else:
        diagnostic_points = points
        pole_audit = _skipped_pole_volume_audit(
            poles,
            all_poles_requested=all_poles_requested,
            tolerance=tolerance,
            reason="symmetry audit rejected the submitted unit-sphere coordinates",
        )
    if local_maximum:
        primary_pole = (
            pole_audit["values"][0]["pole_index"] if pole_audit["values"] else None
        )
        local_audit = _local_maximum_audit(
            diagnostic_points,
            pole=primary_pole,
            fd_step=step,
            geometry_valid=bool(symmetry_result.get("valid_geometry", False)),
        )
    else:
        local_audit = {
            "requested": False,
            "performed": False,
            "diagnostic_only": True,
            "status": "not_requested",
            "classification": "not_requested",
            "fd_step": step,
            "physical_dimension": int(3 * len(points) - _GAUGE_DIMENSION_H4),
            "gauge_dimension": _GAUGE_DIMENSION_H4,
        }

    classification, basis = _combined_classification(
        symmetry_result, pole_audit, local_audit
    )
    return {
        "dimension": 4,
        "n_vertices": int(len(points)),
        "classification": classification,
        "independent_classification": classification,
        "classification_basis": basis,
        "coordinate_preparation": coordinate_preparation,
        "symmetry": symmetry_result,
        "pole_volume_audit": pole_audit,
        "local_maximum_audit": local_audit,
    }


def optimize_h4_from_initial(
    vertices: Any,
    *,
    seed: int,
    budget: int,
    optimizer: str = "powell",
    optimizer_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Optimize one explicit H4 start with a selected, work-counted method.

    The start is normalized exactly once and its stable SHA-256 digest is
    returned so independent campaign arms can prove that they received the
    same float64 coordinates.
    """

    points = _normalized_optimization_vertices(vertices)
    seed_value = _positive_integer(seed, name="seed", minimum=0)
    work_budget = _positive_integer(budget, name="budget")
    if not isinstance(optimizer, str):
        raise TypeError("optimizer must be 'powell' or 'fresh_topology_ad'")
    if optimizer not in ("powell", "fresh_topology_ad"):
        raise ValueError("optimizer must be 'powell' or 'fresh_topology_ad'")
    if optimizer_kwargs is None:
        options: dict[str, Any] = {}
    elif isinstance(optimizer_kwargs, Mapping):
        options = dict(optimizer_kwargs)
    else:
        raise TypeError("optimizer_kwargs must be a mapping or None")

    task = IdealPolytopeTask(d=4, N=len(points))
    initial = PolytopeSolution(vertices=points.copy(), simplices=[])
    rng = np.random.default_rng(seed_value)
    if optimizer == "powell":
        optimizer_instance = LBFGSHyperbolicD4(**options)
        method = "scipy_powell"
    else:
        optimizer_instance = FreshTopologyADHyperbolicD4(**options)
        method = "fresh_delaunay_torch_lbfgsb"

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    run = optimizer_instance.run(
        initial,
        task,
        budget=work_budget,
        rng=rng,
    )
    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start

    diagnostics = getattr(run, "diagnostics", None)
    if optimizer == "fresh_topology_ad" and isinstance(diagnostics, Mapping):
        executed = diagnostics.get("executed_branch_count")
        failed = diagnostics.get("failed_branch_count")
        if isinstance(executed, Integral) and executed > 0 and failed == executed:
            raise RuntimeError(
                "fresh_topology_ad failed in every executed optimization branch"
            )

    if not run.values or run.solution_value is None:
        raise RuntimeError("H^4 optimizer returned no counted canonical score")
    initial_score = float(run.values[0])
    final_score = float(run.solution_value)
    if not np.isfinite(initial_score) or not np.isfinite(final_score):
        raise RuntimeError("H^4 optimizer returned a non-finite canonical score")
    total_evals = int(run.n_evals)
    if total_evals > work_budget:
        raise RuntimeError("H^4 optimizer exceeded its work budget")
    status = "budget_exhausted" if total_evals == work_budget else "completed"
    optimizer_metadata: dict[str, Any] = {
        "selector": optimizer,
        "name": optimizer_instance.name,
        "method": method,
        "status": status,
    }
    if isinstance(diagnostics, Mapping):
        optimizer_metadata["diagnostics"] = dict(diagnostics)
    return {
        "dimension": 4,
        "n_vertices": int(len(points)),
        "seed": seed_value,
        "budget": work_budget,
        "initial_coordinate_sha256": _initial_coordinate_sha256(points),
        "vertices": np.asarray(run.solution.vertices, dtype=float).tolist(),
        "scores": {
            "initial": initial_score,
            "final": final_score,
            "gain": final_score - initial_score,
        },
        "work": {
            "objective_evals": int(run.objective_evals),
            "surrogate_evals": int(run.surrogate_evals),
            "gradient_evals": int(run.gradient_evals),
            "total_evals": total_evals,
        },
        "timing": {
            "wall_seconds": float(wall_seconds),
            "process_cpu_seconds": float(cpu_seconds),
        },
        "optimizer": optimizer_metadata,
    }


def optimize_h4_unconstrained(
    n_vertices: int,
    *,
    seed: int,
    budget: int,
    optimizer: str = "powell",
    optimizer_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate one start, then run the selected unconstrained H4 search."""

    n = _positive_integer(n_vertices, name="n_vertices", minimum=5)
    seed_value = _positive_integer(seed, name="seed", minimum=0)
    work_budget = _positive_integer(budget, name="budget")
    rng = np.random.default_rng(seed_value)
    initial = IdealPolytopeTask(d=4, N=n).random_solution(rng)
    return optimize_h4_from_initial(
        initial.vertices,
        seed=seed_value,
        budget=work_budget,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
    )


__all__ = [
    "DEFAULT_FD_STEP",
    "DEFAULT_VOLUME_SPREAD_TOLERANCE",
    "audit_h4_symmetric_candidate",
    "optimize_h4_from_initial",
    "optimize_h4_unconstrained",
]
