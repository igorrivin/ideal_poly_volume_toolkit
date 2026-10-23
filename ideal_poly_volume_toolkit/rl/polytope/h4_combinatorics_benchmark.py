"""Strict coordinate-only contracts for ideal H^4 combinatorics benchmarks.

The benchmark represented here publishes an unordered tetrahedral facet
complex and a volume floor, but never reference coordinates.  A submission is
accepted only when its unit points on ``S^3`` have a strict simplicial hull
with the prescribed full vertex--facet incidence (up to vertex relabeling),
when every pole produces exactly the corresponding facet triangulation, and
when every pole-volume evaluation is stable and clears the floor.

The incidence comparison uses a two-coloured bipartite graph.  In particular,
it does not substitute the 1-skeleton or an f-vector for the complete face
incidence.  ``pynauty`` is optional at package-install time; its absence is
reported as a controlled, inconclusive audit rather than an import failure.
All public return values are JSON-native.
"""

from __future__ import annotations

import hashlib
from numbers import Integral, Real
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError

from ...geometry_4d import stereo_project_S3
from ...symmetry import _strict_simplicial_facets
from .h4_symmetric_benchmark import (
    DEFAULT_VOLUME_SPREAD_TOLERANCE,
    _pole_volume_audit,
    _skipped_pole_volume_audit,
)

try:
    import pynauty

    PYNAUTY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by monkeypatch in tests.
    pynauty = None
    PYNAUTY_AVAILABLE = False


DEFAULT_UNIT_TOLERANCE = 1.0e-9
DEFAULT_DISTINCT_TOLERANCE = 1.0e-8
DEFAULT_SUPPORT_TOLERANCE = 1.0e-9
_DIMENSION = 4


def _positive_tolerance(value: Any, *, name: str) -> float:
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


def _empty_geometry_audit() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dimension": _DIMENSION,
        "n_vertices": None,
        "valid_geometry": False,
        "checks": {
            "shape": False,
            "finite": False,
            "unit": False,
            "distinct": False,
            "full_dimensional": False,
            "all_vertices_extreme": False,
            "simplicial": False,
            "strict_support": False,
        },
        "topology": {
            "affine_rank": None,
            "min_affine_singular_value": None,
            "max_unit_norm_error": None,
            "min_pairwise_distance": None,
            "n_facets": None,
            "min_support_margin": None,
            "max_incident_plane_residual": None,
            "max_vertices_on_support": None,
        },
        "facets": [],
        "failure_reason": None,
    }


def _fail_geometry(result: dict[str, Any], reason: str) -> dict[str, Any]:
    result["failure_reason"] = str(reason)
    return result


def extract_strict_h4_facets(
    vertices: Any,
    *,
    unit_tolerance: float = DEFAULT_UNIT_TOLERANCE,
    distinct_tolerance: float = DEFAULT_DISTINCT_TOLERANCE,
    support_tolerance: float = DEFAULT_SUPPORT_TOLERANCE,
) -> dict[str, Any]:
    """Validate ideal H^4 boundary points and return certified hull facets.

    Facets are sorted lists of four sorted integer labels.  Input rows are
    normalized only after the submitted norm error passes ``unit_tolerance``;
    all remaining geometric checks use those exact unit representatives.
    """

    unit_tol = _positive_tolerance(unit_tolerance, name="unit_tolerance")
    distinct_tol = _positive_tolerance(distinct_tolerance, name="distinct_tolerance")
    support_tol = _positive_tolerance(support_tolerance, name="support_tolerance")
    result = _empty_geometry_audit()

    try:
        points = np.asarray(vertices, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        return _fail_geometry(result, f"vertices cannot be converted to floats: {exc}")
    if points.ndim != 2:
        return _fail_geometry(result, "vertices must be a two-dimensional N x 4 array")
    n_vertices, dimension = points.shape
    result["n_vertices"] = int(n_vertices)
    result["checks"]["shape"] = bool(
        dimension == _DIMENSION and n_vertices >= _DIMENSION + 1
    )
    if not result["checks"]["shape"]:
        return _fail_geometry(result, "vertices must have shape (N, 4) with N >= 5")

    finite = bool(np.all(np.isfinite(points)))
    result["checks"]["finite"] = finite
    if not finite:
        return _fail_geometry(result, "vertices must contain only finite coordinates")

    coordinate_scales = np.max(np.abs(points), axis=1)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled = np.divide(
            points,
            coordinate_scales[:, None],
            out=np.zeros_like(points),
            where=coordinate_scales[:, None] != 0.0,
        )
        norms = coordinate_scales * np.linalg.norm(scaled, axis=1)
    if not np.all(np.isfinite(norms)):
        return _fail_geometry(result, "vertex norms overflowed or are not finite")
    max_unit_error = float(np.max(np.abs(norms - 1.0)))
    result["topology"]["max_unit_norm_error"] = max_unit_error
    result["checks"]["unit"] = bool(np.all(norms > 0.0) and max_unit_error <= unit_tol)
    if not result["checks"]["unit"]:
        return _fail_geometry(
            result,
            f"vertices are not unit points (maximum norm error {max_unit_error:.3e})",
        )
    points = points / norms[:, None]

    pairwise = points[:, None, :] - points[None, :, :]
    upper = np.linalg.norm(pairwise, axis=2)[np.triu_indices(n_vertices, 1)]
    min_distance = float(np.min(upper))
    result["topology"]["min_pairwise_distance"] = min_distance
    result["checks"]["distinct"] = bool(min_distance > distinct_tol)
    if not result["checks"]["distinct"]:
        return _fail_geometry(
            result,
            f"vertices are not pairwise distinct (minimum distance {min_distance:.3e})",
        )

    centered = points - np.mean(points, axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    rank_threshold = max(
        support_tol,
        np.finfo(np.float64).eps * max(centered.shape) * float(singular_values[0]),
    )
    affine_rank = int(np.count_nonzero(singular_values > rank_threshold))
    result["topology"]["affine_rank"] = affine_rank
    result["topology"]["min_affine_singular_value"] = float(
        singular_values[_DIMENSION - 1]
    )
    result["checks"]["full_dimensional"] = bool(affine_rank == _DIMENSION)
    if not result["checks"]["full_dimensional"]:
        return _fail_geometry(
            result,
            "vertices are not full-dimensional "
            f"(affine rank {affine_rank}, expected {_DIMENSION})",
        )

    try:
        hull = ConvexHull(points)
    except (QhullError, ValueError) as exc:
        return _fail_geometry(result, f"convex hull failed: {exc}")
    hull_vertices = {int(vertex) for vertex in hull.vertices}
    result["checks"]["all_vertices_extreme"] = bool(
        hull_vertices == set(range(n_vertices))
    )
    if not result["checks"]["all_vertices_extreme"]:
        return _fail_geometry(result, "not every submitted point is a hull vertex")

    facets, support, failure = _strict_simplicial_facets(points, hull, support_tol)
    if support is not None:
        result["topology"].update(support)
    if failure is not None:
        return _fail_geometry(result, failure)
    assert facets is not None
    result["checks"]["simplicial"] = True
    result["checks"]["strict_support"] = True
    result["topology"]["n_facets"] = int(len(facets))
    result["facets"] = [list(map(int, facet)) for facet in facets]
    result["valid_geometry"] = True
    return result


def _normalize_facets(
    n_vertices: Any, facets: Any
) -> tuple[list[tuple[int, int, int, int]] | None, str | None]:
    if isinstance(n_vertices, bool) or not isinstance(n_vertices, Integral):
        return None, "n_vertices must be an integer"
    n = int(n_vertices)
    if n < _DIMENSION + 1:
        return None, "n_vertices must be at least 5"
    if isinstance(facets, (str, bytes)):
        return None, "facets must be an iterable of four-vertex facets"
    try:
        raw_facets = list(facets)
    except TypeError:
        return None, "facets must be an iterable of four-vertex facets"
    if not raw_facets:
        return None, "facets must not be empty"

    normalized: list[tuple[int, int, int, int]] = []
    for facet_index, raw_facet in enumerate(raw_facets):
        if isinstance(raw_facet, (str, bytes)):
            return None, f"facet {facet_index} must contain exactly four labels"
        try:
            entries = list(raw_facet)
        except TypeError:
            return None, f"facet {facet_index} must be iterable"
        if len(entries) != _DIMENSION:
            return None, f"facet {facet_index} must contain exactly four labels"
        labels: list[int] = []
        for label in entries:
            if isinstance(label, bool) or not isinstance(label, Integral):
                return None, f"facet {facet_index} contains a non-integer label"
            value = int(label)
            if not 0 <= value < n:
                return None, (
                    f"facet {facet_index} contains label {value} outside [0, {n})"
                )
            labels.append(value)
        if len(set(labels)) != _DIMENSION:
            return None, f"facet {facet_index} repeats a vertex label"
        normalized.append(tuple(sorted(labels)))

    if len(set(normalized)) != len(normalized):
        return None, "facets contain a duplicate unordered facet"
    return sorted(normalized), None


def _incidence_graph(n_vertices: int, facets: list[tuple[int, int, int, int]]) -> Any:
    n_facets = len(facets)
    n_nodes = n_vertices + n_facets
    adjacency = {node: [] for node in range(n_nodes)}
    for facet_offset, facet in enumerate(facets):
        facet_node = n_vertices + facet_offset
        for vertex in facet:
            adjacency[vertex].append(facet_node)
            adjacency[facet_node].append(vertex)
    return pynauty.Graph(
        number_of_vertices=n_nodes,
        directed=False,
        adjacency_dict=adjacency,
        vertex_coloring=[
            set(range(n_vertices)),
            set(range(n_vertices, n_nodes)),
        ],
    )


def facet_incidence_certificate(n_vertices: Any, facets: Any) -> dict[str, Any]:
    """Canonicalize a complete two-coloured vertex--facet incidence graph."""

    normalized, input_failure = _normalize_facets(n_vertices, facets)
    result: dict[str, Any] = {
        "schema_version": 1,
        "available": bool(PYNAUTY_AVAILABLE),
        "backend": "pynauty" if PYNAUTY_AVAILABLE else None,
        "valid_input": input_failure is None,
        "n_vertices": int(n_vertices)
        if isinstance(n_vertices, Integral) and not isinstance(n_vertices, bool)
        else None,
        "n_facets": len(normalized) if normalized is not None else None,
        "color_class_sizes": [
            int(n_vertices),
            len(normalized),
        ]
        if normalized is not None
        else None,
        "canonical_certificate": None,
        "certificate_sha256": None,
        "failure_reason": input_failure,
    }
    if input_failure is not None:
        return result
    if not PYNAUTY_AVAILABLE or pynauty is None:
        result["failure_reason"] = (
            "pynauty is required for full vertex-facet incidence certification"
        )
        return result
    assert normalized is not None
    try:
        certificate = pynauty.certificate(_incidence_graph(int(n_vertices), normalized))
        if not isinstance(certificate, bytes):
            raise TypeError("pynauty returned a non-byte certificate")
    except Exception as exc:
        result["failure_reason"] = (
            f"pynauty incidence certification failed: {type(exc).__name__}: {exc}"
        )
        return result
    result["canonical_certificate"] = certificate.hex()
    result["certificate_sha256"] = hashlib.sha256(certificate).hexdigest()
    result["failure_reason"] = None
    return result


def compare_facet_incidence(
    n_vertices: Any, candidate_facets: Any, target_facets: Any
) -> dict[str, Any]:
    """Compare full tetrahedral facet incidence up to one vertex relabeling."""

    candidate = facet_incidence_certificate(n_vertices, candidate_facets)
    target = facet_incidence_certificate(n_vertices, target_facets)
    candidate_certificate = candidate["canonical_certificate"]
    target_certificate = target["canonical_certificate"]
    available = bool(
        candidate["available"]
        and target["available"]
        and candidate_certificate is not None
        and target_certificate is not None
    )
    valid_input = bool(candidate["valid_input"] and target["valid_input"])
    isomorphic = (
        bool(candidate_certificate == target_certificate)
        if available and valid_input
        else None
    )
    failure_reasons = [
        reason
        for reason in (candidate["failure_reason"], target["failure_reason"])
        if reason is not None
    ]
    return {
        "schema_version": 1,
        "available": available,
        "backend": "pynauty" if available else None,
        "valid_input": valid_input,
        "uses_full_vertex_facet_incidence": True,
        "isomorphic": isomorphic,
        "candidate_n_facets": candidate["n_facets"],
        "target_n_facets": target["n_facets"],
        "candidate_certificate": candidate_certificate,
        "target_certificate": target_certificate,
        "candidate_certificate_sha256": candidate["certificate_sha256"],
        "target_certificate_sha256": target["certificate_sha256"],
        "failure_reason": "; ".join(failure_reasons) if failure_reasons else None,
    }


def _skipped_pole_facet_consistency(n_vertices: int, reason: str) -> dict[str, Any]:
    failures = [
        {
            "pole_index": pole,
            "error_type": "InvalidBoundaryCoordinates",
            "message": reason,
        }
        for pole in range(n_vertices)
    ]
    return {
        "all_poles_requested": True,
        "requested_count": n_vertices,
        "successful_count": 0,
        "failure_count": len(failures),
        "all_match": False,
        "values": [],
        "failures": failures,
    }


def _pole_facet_consistency(
    points: np.ndarray,
    facets: list[tuple[int, int, int, int]],
    pole_volume_audit: dict[str, Any],
) -> dict[str, Any]:
    volume_entries = {
        int(entry["pole_index"]): entry for entry in pole_volume_audit.get("values", [])
    }
    values: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for pole in range(len(points)):
        try:
            projected, finite_indices = stereo_project_S3(points, pole_idx=pole)
            delaunay = Delaunay(projected)
            actual = sorted(
                {
                    tuple(
                        sorted(
                            int(finite_indices[int(local_index)])
                            for local_index in simplex
                        )
                    )
                    for simplex in delaunay.simplices
                }
            )
            expected = sorted(facet for facet in facets if pole not in facet)
            actual_set = set(actual)
            expected_set = set(expected)
            volume_entry = volume_entries.get(pole)
            volume_simplex_count = (
                int(volume_entry["simplex_count"]) if volume_entry is not None else None
            )
            facets_match = actual_set == expected_set
            count_match = bool(
                volume_simplex_count is not None
                and volume_simplex_count == len(actual)
                and len(actual) == len(expected)
            )
            values.append(
                {
                    "pole_index": pole,
                    "expected_simplex_count": len(expected),
                    "delaunay_simplex_count": len(actual),
                    "volume_simplex_count": volume_simplex_count,
                    "facets_match": bool(facets_match),
                    "simplex_counts_match": count_match,
                    "matches": bool(facets_match and count_match),
                    "missing_facets": [
                        list(facet) for facet in sorted(expected_set - actual_set)
                    ],
                    "extra_facets": [
                        list(facet) for facet in sorted(actual_set - expected_set)
                    ],
                }
            )
        except Exception as exc:  # Numerical audit: retain every pole failure.
            failures.append(
                {
                    "pole_index": pole,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    all_match = bool(
        len(values) == len(points)
        and not failures
        and all(entry["matches"] for entry in values)
    )
    return {
        "all_poles_requested": True,
        "requested_count": int(len(points)),
        "successful_count": len(values),
        "failure_count": len(failures),
        "all_match": all_match,
        "values": values,
        "failures": failures,
    }


def _normalized_points_if_valid(vertices: Any, geometry: dict[str, Any]) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    norms = np.linalg.norm(points, axis=1)
    if not geometry["valid_geometry"] or np.any(norms == 0.0):
        raise ValueError("strict unit boundary geometry is unavailable")
    return points / norms[:, None]


def audit_h4_combinatorics_candidate(
    vertices: Any,
    target_facets: Any,
    *,
    volume_floor: float,
    volume_spread_tolerance: float = DEFAULT_VOLUME_SPREAD_TOLERANCE,
    unit_tolerance: float = DEFAULT_UNIT_TOLERANCE,
    distinct_tolerance: float = DEFAULT_DISTINCT_TOLERANCE,
    support_tolerance: float = DEFAULT_SUPPORT_TOLERANCE,
) -> dict[str, Any]:
    """Audit a coordinate-only ideal H^4 combinatorics-volume submission.

    No reference coordinates enter this contract.  The only target data are
    ``target_facets`` and ``volume_floor``.  A missing incidence backend is
    classified as inconclusive; every geometric, pole, combinatorial, or
    volume mismatch is a rejection.
    """

    floor = _nonnegative_real(volume_floor, name="volume_floor")
    spread_tolerance = _nonnegative_real(
        volume_spread_tolerance, name="volume_spread_tolerance"
    )
    geometry = extract_strict_h4_facets(
        vertices,
        unit_tolerance=unit_tolerance,
        distinct_tolerance=distinct_tolerance,
        support_tolerance=support_tolerance,
    )
    n_vertices = geometry["n_vertices"]
    candidate_facets = geometry["facets"]

    if isinstance(n_vertices, int) and n_vertices >= _DIMENSION + 1:
        incidence = compare_facet_incidence(n_vertices, candidate_facets, target_facets)
    else:
        incidence = {
            "schema_version": 1,
            "available": False,
            "backend": None,
            "valid_input": False,
            "uses_full_vertex_facet_incidence": True,
            "isomorphic": None,
            "candidate_n_facets": None,
            "target_n_facets": None,
            "candidate_certificate": None,
            "target_certificate": None,
            "candidate_certificate_sha256": None,
            "target_certificate_sha256": None,
            "failure_reason": "candidate vertex count is unavailable or below 5",
        }

    if geometry["valid_geometry"]:
        points = _normalized_points_if_valid(vertices, geometry)
        poles = list(range(len(points)))
        pole_volume = _pole_volume_audit(
            points,
            poles,
            all_poles_requested=True,
            tolerance=spread_tolerance,
        )
        normalized_facets = [tuple(facet) for facet in candidate_facets]
        pole_facets = _pole_facet_consistency(points, normalized_facets, pole_volume)
    else:
        count = n_vertices if isinstance(n_vertices, int) and n_vertices >= 0 else 0
        reason = geometry["failure_reason"] or "invalid H4 boundary geometry"
        pole_volume = _skipped_pole_volume_audit(
            list(range(count)),
            all_poles_requested=True,
            tolerance=spread_tolerance,
            reason=reason,
        )
        pole_facets = _skipped_pole_facet_consistency(count, reason)

    minimum_volume = pole_volume["minimum_volume"]
    volume_floor_passed = bool(
        minimum_volume is not None
        and np.isfinite(minimum_volume)
        and minimum_volume >= floor
    )
    checks: dict[str, Any] = {
        "geometry": bool(geometry["valid_geometry"]),
        "target_incidence_input": bool(incidence["valid_input"]),
        "incidence_backend": bool(incidence["available"]),
        "combinatorics": incidence["isomorphic"],
        "pole_facet_consistency": bool(pole_facets["all_match"]),
        "all_pole_volume": bool(
            pole_volume["all_poles_requested"]
            and pole_volume["failure_count"] == 0
            and pole_volume["successful_count"] == pole_volume["requested_count"]
            and pole_volume["within_tolerance"]
        ),
        "volume_floor": volume_floor_passed,
    }

    known_failures: list[str] = []
    failure_labels = {
        "geometry": "invalid strict H4 boundary geometry",
        "target_incidence_input": "invalid target facet incidence",
        "pole_facet_consistency": "pole triangulations disagree with hull facets",
        "all_pole_volume": "all-pole volume audit failed or was unstable",
        "volume_floor": "minimum all-pole volume is below the required floor",
    }
    for check, label in failure_labels.items():
        if checks[check] is False:
            known_failures.append(label)
    if checks["combinatorics"] is False:
        known_failures.append("candidate facet incidence is not the target incidence")

    incidence_inconclusive = bool(
        not known_failures
        and (checks["incidence_backend"] is False or checks["combinatorics"] is None)
    )
    if known_failures:
        classification = "rejected"
    elif incidence_inconclusive:
        classification = "inconclusive_incidence_backend"
    else:
        classification = "accepted"
    accepted = bool(classification == "accepted")
    if accepted and not all(value is True for value in checks.values()):
        raise RuntimeError("internal error: accepted contract has an unsatisfied gate")

    return {
        "schema_version": 1,
        "dimension": _DIMENSION,
        "n_vertices": n_vertices,
        "classification": classification,
        "accepted": accepted,
        "checks": checks,
        "geometry": geometry,
        "incidence": incidence,
        "pole_facet_consistency": pole_facets,
        "pole_volume_audit": pole_volume,
        "volume_floor": floor,
        "volume_floor_passed": volume_floor_passed,
        "failure_reasons": known_failures,
    }


__all__ = [
    "DEFAULT_DISTINCT_TOLERANCE",
    "DEFAULT_SUPPORT_TOLERANCE",
    "DEFAULT_UNIT_TOLERANCE",
    "DEFAULT_VOLUME_SPREAD_TOLERANCE",
    "PYNAUTY_AVAILABLE",
    "audit_h4_combinatorics_candidate",
    "compare_facet_incidence",
    "extract_strict_h4_facets",
    "facet_incidence_certificate",
]
