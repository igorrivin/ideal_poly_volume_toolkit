"""Robust audits and numerical solves for normalized zero-shear realizations.

The routines in this module deliberately separate four questions which are
easy to conflate:

* does a labelled complex triangulate a closed two-sphere;
* do the supplied points realize zero Penner--Rivin shears;
* is the resulting projective drawing embedded; and
* is the *labelled* triangulation the strict convex/Delaunay hull?

Convex support is decided by filtered projective predicates.  Ambiguous
binary64 cases are recomputed exactly over the rational values represented by
the input floats.  Qhull is useful corroboration, but is never the authority
for the audit.
"""

from __future__ import annotations

import cmath
import inspect
import itertools
import math
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import ConvexHull, QhullError

from .rivin_holonomy import (
    _analyze_closed_sphere_triangulation,
    compute_shears_from_vertices,
)


CP1Point = complex | None
Face = tuple[int, int, int]
Edge = tuple[int, int]

_EPS = np.finfo(float).eps
_MAX_ADAPTIVE_SUPPORT_FALLBACKS = 2048
_MAX_EXACT_SUPPORT_FALLBACKS = 128
_ROOTS_OF_UNITY = (
    1.0 + 0.0j,
    -0.5 + 0.5j * math.sqrt(3.0),
    -0.5 - 0.5j * math.sqrt(3.0),
)


def _finite_json(value: Any) -> Any:
    """Recursively produce strict-JSON-safe builtins (never NaN/Infinity)."""
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, complex):
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            return None
        return [float(value.real), float(value.imag)]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class ZeroShearRealizationAudit:
    """Independent numerical and combinatorial checks for one realization."""

    strict_pass: bool
    classification: str
    distinguished_face: Face | None
    normalization: dict[str, Any]
    chordal_distinctness: dict[str, Any]
    topology: dict[str, Any]
    support: dict[str, Any]
    shear: dict[str, Any]
    bending: dict[str, Any]
    planar_embedding: dict[str, Any]
    spherical_embedding: dict[str, Any]
    qhull_agreement: dict[str, Any]
    errors: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """Compatibility alias for callers which use a generic pass field."""
        return self.strict_pass

    def as_dict(self) -> dict[str, Any]:
        """Return a finite, strict-JSON-compatible representation."""
        return _finite_json(
            {
                "strict_pass": self.strict_pass,
                "passed": self.strict_pass,
                "classification": self.classification,
                "distinguished_face": self.distinguished_face,
                "normalization": self.normalization,
                "chordal_distinctness": self.chordal_distinctness,
                "topology": self.topology,
                "support": self.support,
                "shear": self.shear,
                "bending": self.bending,
                "planar_embedding": self.planar_embedding,
                "spherical_embedding": self.spherical_embedding,
                "qhull_agreement": self.qhull_agreement,
                "errors": self.errors,
            }
        )


@dataclass(frozen=True)
class TutteSolveResult:
    """Result of an initialized zero-shear solve.

    The historical name is retained for API compatibility.  ``vertices_cp1``
    and ``initial_vertices_cp1`` use the requested solver gauge, while
    ``normalized_vertices_cp1`` always sends the ordered distinguished face
    to ``(infinity, 0, 1)``.
    """

    success: bool
    classification: str
    message: str
    vertices_cp1: tuple[CP1Point, ...] | None
    normalized_vertices_cp1: tuple[CP1Point, ...] | None
    audit: ZeroShearRealizationAudit | None
    initial_embedding: np.ndarray | None
    initial_vertices_cp1: tuple[CP1Point, ...] | None
    cost: float | None
    optimality: float | None
    nfev: int
    njev: int | None
    max_abs_residual: float | None
    residual_l2: float | None
    trajectory: tuple[dict[str, Any], ...]
    trajectory_kind: str
    solver_metadata: dict[str, Any]
    timings: dict[str, float]

    @property
    def strict_pass(self) -> bool:
        return bool(self.audit is not None and self.audit.strict_pass)

    def as_dict(self) -> dict[str, Any]:
        """Return a finite, strict-JSON-compatible representation."""
        return _finite_json(
            {
                "success": self.success,
                "strict_pass": self.strict_pass,
                "classification": self.classification,
                "message": self.message,
                "vertices_cp1": self.vertices_cp1,
                "normalized_vertices_cp1": self.normalized_vertices_cp1,
                "audit": None if self.audit is None else self.audit.as_dict(),
                "initial_embedding": self.initial_embedding,
                "initial_vertices_cp1": self.initial_vertices_cp1,
                "cost": self.cost,
                "optimality": self.optimality,
                "nfev": self.nfev,
                "njev": self.njev,
                "max_abs_residual": self.max_abs_residual,
                "residual_l2": self.residual_l2,
                "trajectory": self.trajectory,
                "trajectory_kind": self.trajectory_kind,
                "solver_metadata": self.solver_metadata,
                "timings": self.timings,
            }
        )


def _as_face(face: Sequence[int], *, name: str = "face") -> Face:
    materialized = tuple(face)
    if len(materialized) != 3:
        raise ValueError(f"{name} must contain exactly three labels")
    if any(
        not isinstance(label, (int, np.integer)) or isinstance(label, bool)
        for label in materialized
    ):
        raise TypeError(f"{name} labels must be integers")
    result = tuple(int(label) for label in materialized)
    if len(set(result)) != 3:
        raise ValueError(f"{name} must contain three distinct labels")
    return result  # type: ignore[return-value]


def _materialize_faces(faces: Iterable[Sequence[int]]) -> tuple[Face, ...]:
    return tuple(
        _as_face(face, name=f"face {index}") for index, face in enumerate(faces)
    )


def _coerce_cp1_point(value: Any) -> CP1Point:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise TypeError("coordinate pairs must contain two numbers")
        value = complex(float(value[0]), float(value[1]))
    else:
        try:
            value = complex(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("CP1 coordinates must be complex numbers or None") from exc
    if not (math.isfinite(value.real) and math.isfinite(value.imag)):
        raise ValueError("CP1 coordinates must be finite or None (infinity)")
    return value


def _materialize_vertices(vertices_cp1: Sequence[Any]) -> tuple[CP1Point, ...]:
    return tuple(_coerce_cp1_point(value) for value in vertices_cp1)


def _spinor(point: CP1Point) -> tuple[complex, complex]:
    """Return a safely scaled homogeneous representative ``[u:v]``."""
    if point is None:
        return 1.0 + 0.0j, 0.0 + 0.0j
    scale = max(abs(point.real), abs(point.imag), 1.0)
    return point / scale, complex(1.0 / scale, 0.0)


def _spinor_det(
    left: tuple[complex, complex], right: tuple[complex, complex]
) -> complex:
    return left[0] * right[1] - left[1] * right[0]


def _unit_sphere(spinor: tuple[complex, complex]) -> np.ndarray:
    u, v = spinor
    norm2 = abs(u) ** 2 + abs(v) ** 2
    if norm2 == 0.0 or not math.isfinite(norm2):
        raise ValueError("invalid homogeneous CP1 coordinate")
    uv = u * v.conjugate()
    return np.array(
        [
            2.0 * uv.real / norm2,
            2.0 * uv.imag / norm2,
            (abs(u) ** 2 - abs(v) ** 2) / norm2,
        ],
        dtype=float,
    )


def _edge_records(faces: Sequence[Face]) -> tuple[tuple[Edge, int, int, int, int], ...]:
    """Return edge, opposites, and oriented endpoints in first-use order."""
    incidents: dict[Edge, list[tuple[int, int, int, int]]] = {}
    for face_index, face in enumerate(faces):
        for index in range(3):
            x = face[index]
            y = face[(index + 1) % 3]
            opposite = face[(index + 2) % 3]
            incidents.setdefault(tuple(sorted((x, y))), []).append(
                (face_index, opposite, x, y)
            )
    records = []
    for edge, occurrences in incidents.items():
        if len(occurrences) != 2:
            raise ValueError(
                f"edge {edge} has incidence {len(occurrences)}; expected 2"
            )
        _, a, x, y = occurrences[0]
        _, b, _, _ = occurrences[1]
        records.append((edge, a, x, b, y))
    return tuple(records)


def _fraction(value: float) -> Fraction:
    numerator, denominator = float(value).as_integer_ratio()
    return Fraction(numerator, denominator)


def _determinant_fraction(matrix: Sequence[Sequence[Fraction]]) -> Fraction:
    work = [list(row) for row in matrix]
    size = len(work)
    sign = 1
    determinant = Fraction(1)
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if work[row][column] != 0),
            None,
        )
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            sign *= -1
        pivot_value = work[column][column]
        determinant *= pivot_value
        for entry in range(column, size):
            work[column][entry] /= pivot_value
        for row in range(column + 1, size):
            factor = work[row][column]
            if factor == 0:
                continue
            for entry in range(column, size):
                work[row][entry] -= factor * work[column][entry]
    return determinant if sign > 0 else -determinant


def _exact_generalized_circle_sign(
    points: Sequence[CP1Point], labels: tuple[int, int, int, int]
) -> int:
    rows: list[list[Fraction]] = []
    for label in labels:
        point = points[label]
        if point is None:
            rows.append([Fraction(1), Fraction(0), Fraction(0), Fraction(0)])
            continue
        x = _fraction(point.real)
        y = _fraction(point.imag)
        rows.append([x * x + y * y, x, y, Fraction(1)])
    determinant = _determinant_fraction(rows)
    return (determinant > 0) - (determinant < 0)


def _generalized_circle_rows(points: Sequence[CP1Point]) -> np.ndarray:
    """Return positively scaled generalized-circle rows in extended precision.

    A finite point ``x+iy`` represents ``(x^2+y^2, x, y, 1)`` and infinity
    represents ``(1, 0, 0, 0)``.  Dividing each finite row by
    ``max(|x|, |y|, 1)^2`` preserves every determinant sign while avoiding
    overflow across the full binary64 range.
    """
    rows = np.empty((len(points), 4), dtype=np.longdouble)
    for index, point in enumerate(points):
        if point is None:
            rows[index] = (1.0, 0.0, 0.0, 0.0)
            continue
        x = np.longdouble(point.real)
        y = np.longdouble(point.imag)
        scale = max(abs(x), abs(y), np.longdouble(1.0))
        x_scaled = x / scale
        y_scaled = y / scale
        inverse = np.longdouble(1.0) / scale
        row = np.asarray(
            (
                x_scaled * x_scaled + y_scaled * y_scaled,
                x_scaled * inverse,
                y_scaled * inverse,
                inverse * inverse,
            ),
            dtype=np.longdouble,
        )
        # Some platforms alias longdouble to binary64.  If projective row
        # scaling loses a mathematically nonzero component there, mark the row
        # unusable so the caller takes the exact-rational path.
        lost_component = bool(
            inverse == 0.0
            or row[3] == 0.0
            or (x != 0.0 and row[1] == 0.0)
            or (y != 0.0 and row[2] == 0.0)
            or ((x != 0.0 or y != 0.0) and row[0] == 0.0)
        )
        rows[index] = np.nan if lost_component else row
    return rows


def _det3_and_permanent(matrix: np.ndarray) -> tuple[np.longdouble, np.longdouble]:
    """Return a 3x3 determinant and the absolute Leibniz-term sum."""
    terms = np.asarray(
        [
            matrix[0, 0] * matrix[1, 1] * matrix[2, 2],
            matrix[0, 1] * matrix[1, 2] * matrix[2, 0],
            matrix[0, 2] * matrix[1, 0] * matrix[2, 1],
            -matrix[0, 2] * matrix[1, 1] * matrix[2, 0],
            -matrix[0, 1] * matrix[1, 0] * matrix[2, 2],
            -matrix[0, 0] * matrix[1, 2] * matrix[2, 1],
        ],
        dtype=np.longdouble,
    )
    return np.sum(terms), np.sum(np.abs(terms))


def _generalized_circle_face_filter(
    rows: np.ndarray, face: Face
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Filter all query signs for one face with a conservative error scale.

    The determinant is expanded along its query row.  ``permanents`` includes
    the absolute terms of each 3x3 cofactor, rather than merely the (possibly
    cancellation-damaged) computed cofactors.  The generous factor also
    covers formation of the scaled rows.  A result inside this bound is never
    trusted: callers refine it exactly over the input binary64 rationals.
    """
    face_rows = rows[list(face)]
    cofactors = np.empty(4, dtype=np.longdouble)
    minor_permanents = np.empty(4, dtype=np.longdouble)
    for column in range(4):
        minor = np.delete(face_rows, column, axis=1)
        determinant, permanent = _det3_and_permanent(minor)
        cofactors[column] = determinant if column % 2 else -determinant
        minor_permanents[column] = permanent
    determinants = rows @ cofactors
    permanents = np.abs(rows) @ minor_permanents
    error_bounds = (
        np.longdouble(65536.0) * np.finfo(np.longdouble).eps * permanents
        + np.finfo(np.longdouble).tiny
    )
    margins = np.divide(
        np.abs(determinants),
        permanents,
        out=np.zeros_like(permanents),
        where=permanents != 0.0,
    )
    return determinants, permanents, error_bounds, margins


def _support_predicate(
    points: Sequence[CP1Point], face: Face, query: int
) -> tuple[int | None, float, bool, float]:
    """Filtered generalized-circle support sign and normalized margin."""
    rows = _generalized_circle_rows(points)
    determinants, _, error_bounds, margins = _generalized_circle_face_filter(rows, face)
    determinant = determinants[query]
    error_bound = error_bounds[query]
    normalized_margin = float(margins[query])
    if np.isfinite(determinant) and abs(determinant) > error_bound:
        return (
            1 if determinant > 0 else -1,
            normalized_margin,
            False,
            float(error_bound),
        )
    try:
        exact_sign = _exact_generalized_circle_sign(points, (*face, query))
    except (ArithmeticError, MemoryError, OverflowError, ValueError):
        return None, normalized_margin, True, float(error_bound)
    return exact_sign, normalized_margin, True, float(error_bound)


def _orient2d(
    points: Sequence[complex], a: int, b: int, c: int
) -> tuple[int | None, float, bool]:
    pa, pb, pc = points[a], points[b], points[c]
    left = (pb.real - pa.real) * (pc.imag - pa.imag)
    right = (pb.imag - pa.imag) * (pc.real - pa.real)
    determinant = left - right
    permanent = abs(left) + abs(right)
    error_bound = 32.0 * _EPS * permanent + np.finfo(float).tiny
    margin = abs(determinant) / permanent if permanent else 0.0
    if math.isfinite(determinant) and abs(determinant) > error_bound:
        return (1 if determinant > 0 else -1), margin, False
    try:
        ax, ay = _fraction(pa.real), _fraction(pa.imag)
        bx, by = _fraction(pb.real), _fraction(pb.imag)
        cx, cy = _fraction(pc.real), _fraction(pc.imag)
        exact = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    except (ArithmeticError, MemoryError, OverflowError, ValueError):
        return None, margin, True
    return (exact > 0) - (exact < 0), margin, True


def _mobius_to_standard_matrix(points: Sequence[CP1Point], face: Face) -> np.ndarray:
    """Matrix sending the ordered face vertices to infinity, zero, one."""
    a, b, c = face
    sa, sb, sc = (_spinor(points[label]) for label in (a, b, c))
    l_a_c = sa[1] * sc[0] - sa[0] * sc[1]
    l_b_c = sb[1] * sc[0] - sb[0] * sc[1]
    if l_a_c == 0 or l_b_c == 0:
        raise ValueError("distinguished face has coincident CP1 vertices")
    scale = l_a_c / l_b_c
    matrix = np.array(
        [[scale * sb[1], -scale * sb[0]], [sa[1], -sa[0]]],
        dtype=complex,
    )
    maximum = float(np.max(np.abs(matrix)))
    if maximum == 0.0 or not math.isfinite(maximum):
        raise ValueError("could not construct a finite Mobius normalization")
    return matrix / maximum


def _standard_to_cube_roots_matrix() -> np.ndarray:
    a, b, c = _ROOTS_OF_UNITY
    delta = (c - a) / (b - c)
    return np.array([[a, b * delta], [1.0, delta]], dtype=complex)


def _apply_mobius_spinor(
    matrix: np.ndarray, point: CP1Point
) -> tuple[complex, complex]:
    u, v = _spinor(point)
    output_u = complex(matrix[0, 0] * u + matrix[0, 1] * v)
    output_v = complex(matrix[1, 0] * u + matrix[1, 1] * v)
    scale = max(abs(output_u), abs(output_v))
    if scale == 0.0 or not math.isfinite(scale):
        raise ValueError("Mobius transformation produced invalid coordinates")
    return output_u / scale, output_v / scale


def _spinor_to_cp1(spinor: tuple[complex, complex]) -> CP1Point:
    u, v = spinor
    if v == 0.0:
        return None
    result = u / v
    if not (math.isfinite(result.real) and math.isfinite(result.imag)):
        raise ValueError("Mobius transformation overflowed")
    return complex(result)


def _mobius_points(
    points: Sequence[CP1Point], matrix: np.ndarray
) -> tuple[CP1Point, ...]:
    return tuple(
        _spinor_to_cp1(_apply_mobius_spinor(matrix, point)) for point in points
    )


def _normalization_report(
    points: Sequence[CP1Point],
    distinguished_face: Face,
    bending_tolerance: float,
    *,
    required: bool,
) -> dict[str, Any]:
    labelled_canonical = distinguished_face == (0, 1, 2)
    applicable = labelled_canonical and required
    anchor_tolerance = 64.0 * _EPS
    anchors_match = bool(
        len(points) >= 3
        and points[0] is None
        and points[1] is not None
        and points[2] is not None
        and abs(points[1]) <= anchor_tolerance
        and abs(points[2] - 1.0) <= anchor_tolerance
    )
    imaginary_parts = [point.imag for point in points[3:] if point is not None]
    all_nonanchors_finite = all(point is not None for point in points[3:])
    minimum_imaginary = min(imaginary_parts, default=None)
    handedness_matches = bool(
        all_nonanchors_finite
        and minimum_imaginary is not None
        and minimum_imaginary > bending_tolerance
    )
    if len(points) == 3:
        handedness_matches = True
    matches = labelled_canonical and anchors_match and handedness_matches
    return {
        "status": "applicable" if required and applicable else "not_applicable",
        "applicable": applicable,
        "required": required,
        "label_order_matches": labelled_canonical,
        "coordinate_anchors_match": anchors_match if applicable else None,
        "handedness_matches": handedness_matches if applicable else None,
        "min_nonanchor_imaginary": (minimum_imaginary if applicable else None),
        "anchor_tolerance": anchor_tolerance,
        "handedness_tolerance": bending_tolerance,
        "matches": matches if applicable else None,
        "passes": matches if required else True,
    }


def _chordal_report(points: Sequence[CP1Point]) -> dict[str, Any]:
    spinors = [_spinor(point) for point in points]
    minimum = math.inf
    pair: Edge | None = None
    for left, right in itertools.combinations(range(len(points)), 2):
        determinant = abs(_spinor_det(spinors[left], spinors[right]))
        norm_left = math.hypot(abs(spinors[left][0]), abs(spinors[left][1]))
        norm_right = math.hypot(abs(spinors[right][0]), abs(spinors[right][1]))
        distance = determinant / (norm_left * norm_right)
        if distance < minimum:
            minimum = distance
            pair = (left, right)
    distinct = bool(math.isfinite(minimum) and minimum > 0.0)
    return {
        "status": "distinct" if distinct else "duplicate",
        "passes": distinct,
        "all_distinct": distinct,
        "min_chordal_distance": minimum if math.isfinite(minimum) else None,
        "closest_pair": pair,
    }


def _topology_report(
    input_faces: Sequence[Face], oriented_faces: Sequence[Face], summary: Any
) -> dict[str, Any]:
    directed: dict[Edge, list[tuple[int, int]]] = defaultdict(list)
    for face in input_faces:
        for index in range(3):
            u, v = face[index], face[(index + 1) % 3]
            directed[tuple(sorted((u, v)))].append((u, v))
    input_coherent = all(
        len(uses) == 2 and uses[0] == tuple(reversed(uses[1]))
        for uses in directed.values()
    )
    return {
        "status": "valid_coherent" if input_coherent else "orientation_incoherent",
        "passes": input_coherent,
        "combinatorially_valid": True,
        "closed_sphere": True,
        "input_orientation_coherent": input_coherent,
        "orientation_was_normalized": tuple(input_faces) != tuple(oriented_faces),
        **summary.as_dict(),
    }


def _support_report(
    faces: Sequence[Face], points: Sequence[CP1Point], tolerance: float
) -> dict[str, Any]:
    generalized_rows = _generalized_circle_rows(points)
    fallback_count = 0
    ambiguous_count = 0
    exact_fallback_count = 0
    indeterminate_count = 0
    tie_count = 0
    inconsistent_count = 0
    minimum_margin = math.inf
    worst: dict[str, Any] | None = None
    face_details = []
    labels = np.arange(len(points), dtype=np.int64)
    for face_index, face in enumerate(faces):
        determinant_values, _, error_bounds, margins = _generalized_circle_face_filter(
            generalized_rows, face
        )
        query_mask = np.ones(len(points), dtype=bool)
        query_mask[list(face)] = False
        query_labels = labels[query_mask]
        face_margins = margins[query_mask]
        face_determinants = determinant_values[query_mask]
        face_errors = error_bounds[query_mask]
        reliable = np.isfinite(face_determinants) & (
            np.abs(face_determinants) > face_errors
        )
        sign_values: list[int | None] = [
            1 if value > 0.0 else -1 for value in face_determinants[reliable]
        ]
        ambiguous_indices = np.flatnonzero(~reliable)
        ambiguous_count += len(ambiguous_indices)
        for local_index in ambiguous_indices:
            if fallback_count >= _MAX_ADAPTIVE_SUPPORT_FALLBACKS:
                sign_values.append(None)
                continue
            query = int(query_labels[local_index])
            fallback_count += 1
            try:
                adaptive_sign = None
                if exact_fallback_count < _MAX_EXACT_SUPPORT_FALLBACKS:
                    exact_fallback_count += 1
                    adaptive_sign = _exact_generalized_circle_sign(
                        points, (*face, query)
                    )
                sign_values.append(adaptive_sign)
            except (ArithmeticError, MemoryError, OverflowError, ValueError):
                sign_values.append(None)

        face_indeterminate = any(sign is None for sign in sign_values)
        indeterminate_count += sum(sign is None for sign in sign_values)
        tie_count += sum(sign == 0 for sign in sign_values)
        signs = [int(sign) for sign in sign_values if sign not in (None, 0)]
        local_worst = int(np.argmin(face_margins))
        face_minimum = float(face_margins[local_worst])
        face_witness = int(query_labels[local_worst])
        if face_minimum < minimum_margin:
            minimum_margin = face_minimum
            worst = {
                "face_index": face_index,
                "face": face,
                "query_vertex": face_witness,
                "normalized_margin": face_minimum,
                "filter_error_bound": float(face_errors[local_worst]),
            }
        consistent = bool(signs) and len(set(signs)) == 1
        if signs and not consistent:
            inconsistent_count += 1
        face_details.append(
            {
                "face_index": face_index,
                "sign": signs[0] if consistent else None,
                "min_normalized_margin": (
                    face_minimum if math.isfinite(face_minimum) else None
                ),
                "witness_vertex": face_witness,
                "strict": bool(
                    consistent and not face_indeterminate and face_minimum > tolerance
                ),
            }
        )
    strict = bool(
        indeterminate_count == 0
        and tie_count == 0
        and inconsistent_count == 0
        and all(detail["strict"] for detail in face_details)
    )
    if indeterminate_count:
        status = "indeterminate"
    elif tie_count or minimum_margin <= tolerance:
        status = "degenerate"
    elif inconsistent_count:
        status = "not_supporting"
    else:
        status = "strict_supporting"
    return {
        "status": status,
        "passes": strict,
        "strict_supporting": strict,
        "support_tolerance": tolerance,
        "min_normalized_margin": (
            minimum_margin if math.isfinite(minimum_margin) else None
        ),
        "adaptive_fallback_count": fallback_count,
        "filter_ambiguous_count": ambiguous_count,
        "adaptive_fallback_limit": _MAX_ADAPTIVE_SUPPORT_FALLBACKS,
        "exact_rational_fallback_count": exact_fallback_count,
        "exact_rational_fallback_limit": _MAX_EXACT_SUPPORT_FALLBACKS,
        "adaptive_fallback_limit_reached": ambiguous_count > fallback_count,
        "exact_rational_fallback_limit_reached": (
            exact_fallback_count >= _MAX_EXACT_SUPPORT_FALLBACKS
            and indeterminate_count > 0
        ),
        "indeterminate_count": indeterminate_count,
        "tie_count": tie_count,
        "inconsistent_face_count": inconsistent_count,
        "worst_witness": worst,
        "faces": face_details,
    }


def _log_abs_det(
    left: tuple[complex, complex], right: tuple[complex, complex]
) -> float:
    magnitude = abs(_spinor_det(left, right))
    if magnitude == 0.0 or not math.isfinite(magnitude):
        raise ValueError("cross-ratio contains coincident CP1 vertices")
    return math.log(magnitude)


def _independent_shears_and_bending(
    faces: Sequence[Face], points: Sequence[CP1Point]
) -> tuple[dict[int, float], list[tuple[Edge, float]]]:
    spinors = [_spinor(point) for point in points]
    shears: dict[int, float] = {}
    bending: list[tuple[Edge, float]] = []
    for edge_id, (edge, a, x, b, y) in enumerate(_edge_records(faces)):
        s_a, s_x, s_b, s_y = (spinors[label] for label in (a, x, b, y))
        shear = (
            _log_abs_det(s_a, s_x)
            + _log_abs_det(s_b, s_y)
            - _log_abs_det(s_a, s_y)
            - _log_abs_det(s_b, s_x)
        )
        phase = (
            cmath.phase(_spinor_det(s_a, s_x))
            + cmath.phase(_spinor_det(s_b, s_y))
            - cmath.phase(_spinor_det(s_a, s_y))
            - cmath.phase(_spinor_det(s_b, s_x))
        )
        angle = math.atan2(math.sin(phase), math.cos(phase))
        shears[edge_id] = shear
        bending.append((edge, angle))
    return shears, bending


def _shear_and_bending_reports(
    faces: Sequence[Face],
    points: Sequence[CP1Point],
    shear_tolerance: float,
    bending_tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    independent, bending_values = _independent_shears_and_bending(faces, points)
    stable = compute_shears_from_vertices(list(faces), list(points))
    if stable.keys() != independent.keys():
        raise RuntimeError("stable and independent shear edge ids disagree")
    crosscheck = max(
        (abs(stable[index] - independent[index]) for index in stable),
        default=0.0,
    )
    values = list(stable.values())
    max_abs = max((abs(value) for value in values), default=0.0)
    rms = math.sqrt(math.fsum(value * value for value in values) / len(values))
    shear_report = {
        "status": "within_tolerance"
        if max_abs <= shear_tolerance
        else "residual_failure",
        "passes": max_abs <= shear_tolerance,
        "within_tolerance": max_abs <= shear_tolerance,
        "tolerance": shear_tolerance,
        "n_edges": len(values),
        "max_abs": max_abs,
        "rms": rms,
        "stable_crosscheck_max_abs": crosscheck,
        "stable_crosscheck_passes": crosscheck <= max(1e-12, 64.0 * _EPS),
        "values": values,
    }

    angles = [angle for _, angle in bending_values]
    margins = [min(abs(angle), math.pi - abs(angle)) for angle in angles]
    minimum_margin = min(margins, default=0.0)
    positive = all(angle > bending_tolerance for angle in angles)
    negative = all(angle < -bending_tolerance for angle in angles)
    n_positive = sum(angle > bending_tolerance for angle in angles)
    n_negative = sum(angle < -bending_tolerance for angle in angles)
    n_endpoint = len(angles) - n_positive - n_negative
    endpoint_degenerate = any(margin <= bending_tolerance for margin in margins)
    coherent = (positive or negative) and not endpoint_degenerate
    worst_index = int(np.argmin(margins)) if margins else None
    bending_report = {
        "status": (
            "coherent"
            if coherent
            else "degenerate"
            if endpoint_degenerate
            else "incoherent"
        ),
        "passes": coherent,
        "coherent_sign": coherent,
        "orientation_sign": 1 if positive else -1 if negative else None,
        "tolerance": bending_tolerance,
        "min_endpoint_margin_radians": minimum_margin,
        "max_abs_angle_radians": max(map(abs, angles), default=0.0),
        "worst_edge": (
            bending_values[worst_index][0] if worst_index is not None else None
        ),
        "endpoint_degenerate": endpoint_degenerate,
        "sign_violation_count": min(n_positive, n_negative) + n_endpoint,
        "angles_radians": angles,
    }
    return shear_report, bending_report


def _segments_cross(
    points: Sequence[complex], edge_a: Edge, edge_b: Edge
) -> tuple[str, dict[str, Any] | None]:
    a, b = edge_a
    c, d = edge_b
    tests = [
        _orient2d(points, a, b, c),
        _orient2d(points, a, b, d),
        _orient2d(points, c, d, a),
        _orient2d(points, c, d, b),
    ]
    signs = [test[0] for test in tests]
    if any(sign is None for sign in signs):
        return "indeterminate", {"edges": (edge_a, edge_b)}
    s1, s2, s3, s4 = (int(sign) for sign in signs)
    if s1 * s2 < 0 and s3 * s4 < 0:
        return "crossing", {"edges": (edge_a, edge_b)}

    def on_segment(p: complex, q: complex, r: complex) -> bool:
        return min(p.real, q.real) <= r.real <= max(p.real, q.real) and min(
            p.imag, q.imag
        ) <= r.imag <= max(p.imag, q.imag)

    candidates = ((s1, a, b, c), (s2, a, b, d), (s3, c, d, a), (s4, c, d, b))
    for sign, left, right, query in candidates:
        if sign == 0 and on_segment(points[left], points[right], points[query]):
            return "touching", {"edges": (edge_a, edge_b), "vertex": query}
    return "disjoint", None


def _scan_planar_edge_crossings(
    points: Sequence[complex], edges: Sequence[Edge]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int]:
    """Vector-filter nonincident straight edges, refining ambiguous pairs."""
    coordinates = np.column_stack(
        ([point.real for point in points], [point.imag for point in points])
    )
    edge_array = np.asarray(edges, dtype=np.int64)
    starts = coordinates[edge_array[:, 0]]
    ends = coordinates[edge_array[:, 1]]
    directions = ends - starts
    ambiguous: list[tuple[int, int]] = []
    checked = 0

    def orientations(
        origins: np.ndarray,
        vectors: np.ndarray,
        queries: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        relative = queries - origins
        left = vectors[..., 0] * relative[..., 1]
        right = vectors[..., 1] * relative[..., 0]
        determinants = left - right
        errors = 32.0 * _EPS * (np.abs(left) + np.abs(right))
        return determinants, errors + np.finfo(float).tiny

    for left_index in range(len(edges) - 1):
        right_indices = np.arange(left_index + 1, len(edges))
        nonincident = (
            (edge_array[right_indices, 0] != edge_array[left_index, 0])
            & (edge_array[right_indices, 0] != edge_array[left_index, 1])
            & (edge_array[right_indices, 1] != edge_array[left_index, 0])
            & (edge_array[right_indices, 1] != edge_array[left_index, 1])
        )
        right_indices = right_indices[nonincident]
        if not len(right_indices):
            continue
        checked += len(right_indices)
        first, first_error = orientations(
            starts[left_index], directions[left_index], starts[right_indices]
        )
        second, second_error = orientations(
            starts[left_index], directions[left_index], ends[right_indices]
        )
        third, third_error = orientations(
            starts[right_indices], directions[right_indices], starts[left_index]
        )
        fourth, fourth_error = orientations(
            starts[right_indices], directions[right_indices], ends[left_index]
        )
        reliable = (
            (np.abs(first) > first_error)
            & (np.abs(second) > second_error)
            & (np.abs(third) > third_error)
            & (np.abs(fourth) > fourth_error)
        )
        crossing = (
            reliable
            & (np.signbit(first) != np.signbit(second))
            & (np.signbit(third) != np.signbit(fourth))
        )
        found = np.flatnonzero(crossing)
        if len(found):
            right_index = int(right_indices[int(found[0])])
            return (
                {
                    "edges": (edges[left_index], edges[right_index]),
                    "kind": "crossing",
                },
                None,
                checked,
            )
        ambiguous.extend(
            (left_index, int(right_indices[index]))
            for index in np.flatnonzero(~reliable)
        )

    for left_index, right_index in ambiguous:
        status, witness = _segments_cross(points, edges[left_index], edges[right_index])
        if status in {"crossing", "touching"}:
            assert witness is not None
            witness["kind"] = status
            return witness, None, checked
        if status == "indeterminate":
            return None, witness, checked
    return None, None, checked


def _planar_embedding_report(
    faces: Sequence[Face], points: Sequence[CP1Point], distinguished_face: Face
) -> dict[str, Any]:
    if any(point is None for point in points):
        return {
            "status": "not_applicable",
            "passes": None,
            "embedded": None,
            "reason": "straight-line planar audit requires finite coordinates",
        }
    finite = [complex(point) for point in points if point is not None]
    distinguished_set = frozenset(distinguished_face)
    try:
        distinguished_index = next(
            index
            for index, face in enumerate(faces)
            if frozenset(face) == distinguished_set
        )
    except StopIteration:
        return {
            "status": "error",
            "passes": False,
            "embedded": False,
            "reason": "distinguished face is not in the triangulation",
        }
    outer_face = faces[distinguished_index]
    outer_sign, outer_margin, _ = _orient2d(finite, *outer_face)
    face_degenerate = outer_sign in (None, 0)
    orientation_failure: dict[str, Any] | None = None
    minimum_margin = outer_margin
    for face_index, face in enumerate(faces):
        if face_index == distinguished_index:
            continue
        sign, margin, _ = _orient2d(finite, *face)
        minimum_margin = min(minimum_margin, margin)
        if sign in (None, 0):
            face_degenerate = True
            if orientation_failure is None:
                orientation_failure = {"face_index": face_index, "face": face}
        elif outer_sign not in (None, 0) and sign == outer_sign:
            if orientation_failure is None:
                orientation_failure = {"face_index": face_index, "face": face}

    edges = sorted(
        {
            tuple(sorted((face[index], face[(index + 1) % 3])))
            for face in faces
            for index in range(3)
        }
    )
    crossing, indeterminate, checked_pairs = _scan_planar_edge_crossings(finite, edges)
    embedded = bool(
        not face_degenerate
        and orientation_failure is None
        and crossing is None
        and indeterminate is None
    )
    return {
        "status": (
            "embedded"
            if embedded
            else "indeterminate"
            if indeterminate is not None
            else "degenerate"
            if face_degenerate
            else "nonembedded"
        ),
        "passes": embedded,
        "embedded": embedded,
        "outer_face_index": distinguished_index,
        "min_normalized_face_area": minimum_margin,
        "orientation_failure": orientation_failure,
        "crossing_witness": crossing,
        "indeterminate_witness": indeterminate,
        "n_edges_checked": len(edges),
        "n_nonincident_edge_pairs_checked": checked_pairs,
    }


def _filtered_dot_sign(value: float, scale: float) -> int | None:
    tolerance = 256.0 * _EPS * max(scale, np.finfo(float).tiny)
    if not math.isfinite(value) or abs(value) <= tolerance:
        return None
    return 1 if value > 0 else -1


def _spherical_edges_cross(sphere: np.ndarray, edge_a: Edge, edge_b: Edge) -> str:
    a, b = edge_a
    c, d = edge_b
    normal_ab = np.cross(sphere[a], sphere[b])
    normal_cd = np.cross(sphere[c], sphere[d])
    norm_ab = float(np.linalg.norm(normal_ab))
    norm_cd = float(np.linalg.norm(normal_cd))
    if norm_ab <= 256.0 * _EPS or norm_cd <= 256.0 * _EPS:
        return "indeterminate"
    intersection_axis = np.cross(normal_ab, normal_cd)
    axis_norm = float(np.linalg.norm(intersection_axis))

    def angular_distance(left: np.ndarray, right: np.ndarray) -> float:
        return math.atan2(
            float(np.linalg.norm(np.cross(left, right))),
            float(np.dot(left, right)),
        )

    def arc_position(left: np.ndarray, right: np.ndarray, query: np.ndarray) -> str:
        whole = angular_distance(left, right)
        first = angular_distance(left, query)
        second = angular_distance(query, right)
        error = abs((first + second) - whole)
        tolerance = 4096.0 * _EPS * max(1.0, whole, first + second)
        if error > tolerance:
            return "outside"
        if first <= tolerance or second <= tolerance:
            return "endpoint"
        return "interior"

    if axis_norm <= 1024.0 * _EPS * norm_ab * norm_cd:
        # Coincident great circles need not imply intersecting *minor* arcs.
        positions = (
            arc_position(sphere[a], sphere[b], sphere[c]),
            arc_position(sphere[a], sphere[b], sphere[d]),
            arc_position(sphere[c], sphere[d], sphere[a]),
            arc_position(sphere[c], sphere[d], sphere[b]),
        )
        return "touching" if "interior" in positions else "disjoint"

    candidate = intersection_axis / axis_norm
    for query in (candidate, -candidate):
        first = arc_position(sphere[a], sphere[b], query)
        second = arc_position(sphere[c], sphere[d], query)
        if first == "interior" and second == "interior":
            return "crossing"
        if first != "outside" and second != "outside":
            return "touching"
    return "disjoint"


def _scan_spherical_edge_crossings(
    sphere: np.ndarray, edges: Sequence[Edge]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int]:
    """Vector-filter all nonincident pairs, refining only near-zero signs."""
    edge_array = np.asarray(edges, dtype=np.int64)
    starts = sphere[edge_array[:, 0]]
    ends = sphere[edge_array[:, 1]]
    normals = np.cross(starts, ends)
    normal_norms = np.linalg.norm(normals, axis=1)
    small = np.flatnonzero(normal_norms <= 256.0 * _EPS)
    if len(small):
        index = int(small[0])
        return None, {"edge": edges[index], "reason": "antipodal-or-degenerate-edge"}, 0

    ambiguous: list[tuple[int, int]] = []
    checked = 0
    for left in range(len(edges) - 1):
        right_indices = np.arange(left + 1, len(edges))
        nonincident = (
            (edge_array[right_indices, 0] != edge_array[left, 0])
            & (edge_array[right_indices, 0] != edge_array[left, 1])
            & (edge_array[right_indices, 1] != edge_array[left, 0])
            & (edge_array[right_indices, 1] != edge_array[left, 1])
        )
        right_indices = right_indices[nonincident]
        if not len(right_indices):
            continue
        checked += len(right_indices)
        first = starts[right_indices] @ normals[left]
        second = ends[right_indices] @ normals[left]
        third = normals[right_indices] @ starts[left]
        fourth = normals[right_indices] @ ends[left]
        left_tolerance = 256.0 * _EPS * normal_norms[left]
        right_tolerance = 256.0 * _EPS * normal_norms[right_indices]
        opposite_first = ((first > left_tolerance) & (second < -left_tolerance)) | (
            (first < -left_tolerance) & (second > left_tolerance)
        )
        opposite_second = ((third > right_tolerance) & (fourth < -right_tolerance)) | (
            (third < -right_tolerance) & (fourth > right_tolerance)
        )
        crossings = np.flatnonzero(opposite_first & opposite_second)
        if len(crossings):
            right = int(right_indices[int(crossings[0])])
            return (
                {"edges": (edges[left], edges[right]), "kind": "crossing"},
                None,
                checked,
            )

        near_zero = (
            (np.abs(first) <= left_tolerance)
            | (np.abs(second) <= left_tolerance)
            | (np.abs(third) <= right_tolerance)
            | (np.abs(fourth) <= right_tolerance)
        )
        ambiguous.extend(
            (left, int(right_indices[index])) for index in np.flatnonzero(near_zero)
        )

    # A zero great-circle side test often means only that a third cube root is
    # on the complementary equatorial arc.  Resolve those exceptional pairs
    # with the slower minor-arc test rather than rejecting them wholesale.
    for left, right in ambiguous:
        status = _spherical_edges_cross(sphere, edges[left], edges[right])
        if status in {"crossing", "touching"}:
            return {"edges": (edges[left], edges[right]), "kind": status}, None, checked
        if status == "indeterminate":
            return None, {"edges": (edges[left], edges[right])}, checked
    return None, None, checked


def _spherical_embedding_report(
    faces: Sequence[Face],
    points: Sequence[CP1Point],
    distinguished_face: Face,
    *,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {
            "status": "not_checked",
            "passes": None,
            "embedded": None,
            "chart": "distinguished-face-to-cube-roots",
        }
    try:
        standard = _mobius_to_standard_matrix(points, distinguished_face)
        cube_matrix = _standard_to_cube_roots_matrix() @ standard
        transformed = [_apply_mobius_spinor(cube_matrix, point) for point in points]
        sphere = np.vstack([_unit_sphere(spinor) for spinor in transformed])
    except (ArithmeticError, ValueError, OverflowError) as exc:
        return {
            "status": "indeterminate",
            "passes": False,
            "embedded": False,
            "chart": "distinguished-face-to-cube-roots",
            "error": str(exc),
        }

    distinguished_set = frozenset(distinguished_face)
    try:
        distinguished_index = next(
            index
            for index, face in enumerate(faces)
            if frozenset(face) == distinguished_set
        )
    except StopIteration:
        return {
            "status": "error",
            "passes": False,
            "embedded": False,
            "chart": "distinguished-face-to-cube-roots",
            "error": "distinguished face is absent",
        }

    face_signs: list[int] = []
    min_face_margin = math.inf
    degenerate_face = None
    for face_index, face in enumerate(faces):
        if face_index == distinguished_index:
            continue
        matrix = sphere[list(face)]
        determinant = float(np.linalg.det(matrix))
        permanent_scale = float(
            np.linalg.norm(matrix[0])
            * np.linalg.norm(matrix[1])
            * np.linalg.norm(matrix[2])
        )
        sign = _filtered_dot_sign(determinant, permanent_scale)
        margin = abs(determinant) / permanent_scale if permanent_scale else 0.0
        min_face_margin = min(min_face_margin, margin)
        if sign is None:
            degenerate_face = {"face_index": face_index, "face": face}
            break
        face_signs.append(sign)

    orientation_coherent = bool(face_signs and len(set(face_signs)) == 1)
    edges = sorted(
        {
            tuple(sorted((face[index], face[(index + 1) % 3])))
            for face in faces
            for index in range(3)
        }
    )
    crossing = None
    indeterminate_pair = None
    checked_pairs = 0
    if degenerate_face is None and orientation_coherent:
        crossing, indeterminate_pair, checked_pairs = _scan_spherical_edge_crossings(
            sphere, edges
        )

    embedded = bool(
        degenerate_face is None
        and orientation_coherent
        and crossing is None
        and indeterminate_pair is None
    )
    return {
        "status": (
            "embedded"
            if embedded
            else "indeterminate"
            if degenerate_face or indeterminate_pair
            else "nonembedded"
        ),
        "passes": embedded,
        "embedded": embedded,
        "chart": "distinguished-face-to-cube-roots",
        "distinguished_face_maps_to": [
            [root.real, root.imag] for root in _ROOTS_OF_UNITY
        ],
        "orientation_coherent": orientation_coherent,
        "min_radial_face_margin": (
            min_face_margin if math.isfinite(min_face_margin) else None
        ),
        "degenerate_face": degenerate_face,
        "crossing_witness": crossing,
        "indeterminate_witness": indeterminate_pair,
        "n_edges_checked": len(edges),
        "n_nonincident_edge_pairs_checked": checked_pairs,
    }


def _qhull_report(
    faces: Sequence[Face], points: Sequence[CP1Point], distinguished_face: Face
) -> dict[str, Any]:
    try:
        standard = _mobius_to_standard_matrix(points, distinguished_face)
        matrix = _standard_to_cube_roots_matrix() @ standard
        sphere = np.vstack(
            [_unit_sphere(_apply_mobius_spinor(matrix, point)) for point in points]
        )
        hull = ConvexHull(sphere)
        qhull_faces = {tuple(sorted(map(int, simplex))) for simplex in hull.simplices}
        labelled_faces = {tuple(sorted(face)) for face in faces}
        missing = sorted(labelled_faces - qhull_faces)
        extra = sorted(qhull_faces - labelled_faces)
        agrees = not missing and not extra
        return {
            "status": "agrees" if agrees else "disagrees",
            "passes": agrees,
            "agrees": agrees,
            "diagnostic_only": True,
            "comparison": "exact-labelled-face-sets",
            "n_qhull_faces": len(qhull_faces),
            "n_labelled_faces": len(labelled_faces),
            "missing_faces": missing,
            "extra_faces": extra,
        }
    except (QhullError, ArithmeticError, ValueError, OverflowError) as exc:
        return {
            "status": "indeterminate",
            "passes": None,
            "agrees": None,
            "diagnostic_only": True,
            "comparison": "exact-labelled-face-sets",
            "error": str(exc),
        }


def _empty_audit(
    distinguished_face: Face | None, error: str
) -> ZeroShearRealizationAudit:
    failed = {"status": "error", "passes": False, "error": error}
    return ZeroShearRealizationAudit(
        strict_pass=False,
        classification="error",
        distinguished_face=distinguished_face,
        normalization=dict(failed),
        chordal_distinctness=dict(failed),
        topology=dict(failed),
        support=dict(failed),
        shear=dict(failed),
        bending=dict(failed),
        planar_embedding=dict(failed),
        spherical_embedding=dict(failed),
        qhull_agreement=dict(failed),
        errors=(error,),
    )


def audit_zero_shear_realization(
    faces: Iterable[Sequence[int]],
    vertices_cp1: Sequence[Any],
    *,
    distinguished_face: Sequence[int] | None = None,
    shear_tolerance: float = 1e-8,
    support_tolerance: float = 0.0,
    bending_tolerance: float = 1e-10,
    check_spherical_embedding: bool = True,
    require_standard_normalization: bool | None = None,
) -> ZeroShearRealizationAudit:
    """Audit a labelled CP1 realization without conflating its certificates.

    ``None`` denotes infinity.  Standard normalization, when required, is
    only satisfied by the ordered distinguished face ``(0, 1, 2)``, anchors
    ``(infinity, 0, 1)``, and positive-half-plane handedness for every
    remaining vertex.  ``require_standard_normalization=None`` enables that
    gate automatically whenever the ordered distinguished face is
    ``(0, 1, 2)``; callers intentionally using another gauge must pass
    ``False`` explicitly.  Solvers in the finite cube-root disk gauge do so.

    Invalid input is represented by a finite ``classification='error'`` audit
    rather than leaking a partial result or a non-JSON floating value.
    """
    chosen_face: Face | None = None
    try:
        if not (math.isfinite(shear_tolerance) and shear_tolerance >= 0.0):
            raise ValueError("shear_tolerance must be finite and non-negative")
        if not (math.isfinite(support_tolerance) and support_tolerance >= 0.0):
            raise ValueError("support_tolerance must be finite and non-negative")
        if not (math.isfinite(bending_tolerance) and bending_tolerance >= 0.0):
            raise ValueError("bending_tolerance must be finite and non-negative")
        input_faces = _materialize_faces(faces)
        oriented_faces, summary = _analyze_closed_sphere_triangulation(input_faces)
        points = _materialize_vertices(vertices_cp1)
        expected_labels = tuple(range(len(points)))
        if summary.vertex_labels != expected_labels:
            raise ValueError("face labels must be exactly 0..len(vertices_cp1)-1")
        if distinguished_face is None:
            canonical = next(
                (face for face in input_faces if frozenset(face) == {0, 1, 2}),
                None,
            )
            chosen_face = (0, 1, 2) if canonical is not None else input_faces[0]
        else:
            chosen_face = _as_face(distinguished_face, name="distinguished_face")
        if frozenset(chosen_face) not in {frozenset(face) for face in input_faces}:
            raise ValueError("distinguished_face is not a face of the triangulation")
    except (TypeError, ValueError, ArithmeticError, OverflowError) as exc:
        return _empty_audit(chosen_face, str(exc))

    topology = _topology_report(input_faces, oriented_faces, summary)
    normalization_required = (
        chosen_face == (0, 1, 2)
        if require_standard_normalization is None
        else bool(require_standard_normalization)
    )
    normalization = _normalization_report(
        points,
        chosen_face,
        bending_tolerance,
        required=normalization_required,
    )
    chordal = _chordal_report(points)
    if not chordal["passes"]:
        failed = {
            "status": "not_computable",
            "passes": False,
            "error": "coincident CP1 vertices",
        }
        return ZeroShearRealizationAudit(
            strict_pass=False,
            classification="degenerate_zero_shear",
            distinguished_face=chosen_face,
            normalization=normalization,
            chordal_distinctness=chordal,
            topology=topology,
            support=dict(failed),
            shear=dict(failed),
            bending=dict(failed),
            planar_embedding=_planar_embedding_report(
                oriented_faces, points, chosen_face
            ),
            spherical_embedding=dict(failed),
            qhull_agreement=dict(failed),
            errors=("coincident CP1 vertices",),
        )

    errors: list[str] = []
    try:
        support = _support_report(oriented_faces, points, support_tolerance)
    except (ArithmeticError, ValueError, OverflowError) as exc:
        support = {"status": "indeterminate", "passes": False, "error": str(exc)}
        errors.append(f"support: {exc}")
    try:
        shear, bending = _shear_and_bending_reports(
            oriented_faces, points, shear_tolerance, bending_tolerance
        )
    except (ArithmeticError, ValueError, OverflowError, RuntimeError) as exc:
        shear = {"status": "error", "passes": False, "error": str(exc)}
        bending = {"status": "error", "passes": False, "error": str(exc)}
        errors.append(f"shear/bending: {exc}")
    planar = _planar_embedding_report(oriented_faces, points, chosen_face)
    spherical = _spherical_embedding_report(
        oriented_faces, points, chosen_face, enabled=check_spherical_embedding
    )
    qhull = _qhull_report(oriented_faces, points, chosen_face)

    normalization_gate = bool(normalization["passes"])
    shear_pass = bool(shear.get("passes", False))
    support_pass = bool(support.get("passes", False))
    bending_pass = bool(bending.get("passes", False))
    spherical_pass = bool(spherical.get("passes", False))
    strict = bool(
        topology["passes"]
        and chordal["passes"]
        and normalization_gate
        and shear_pass
        and support_pass
        and bending_pass
    )

    degenerate = bool(
        support.get("status") in {"degenerate", "indeterminate"}
        or bending.get("status") == "degenerate"
    )
    if not shear_pass:
        classification = "residual_failure"
    elif degenerate:
        classification = "degenerate_zero_shear"
    elif check_spherical_embedding and not spherical_pass and not support_pass:
        classification = "nonembedded_zero_shear"
    elif not normalization_gate:
        classification = "normalization_failure"
    elif not support_pass or not bending_pass:
        classification = "embedded_wrong_hull_zero_shear"
    elif strict:
        classification = "target_strict_zero_shear"
    else:
        classification = "error"

    return ZeroShearRealizationAudit(
        strict_pass=strict,
        classification=classification,
        distinguished_face=chosen_face,
        normalization=normalization,
        chordal_distinctness=chordal,
        topology=topology,
        support=support,
        shear=shear,
        bending=bending,
        planar_embedding=planar,
        spherical_embedding=spherical,
        qhull_agreement=qhull,
        errors=tuple(errors),
    )


def _canonical_edges(faces: Sequence[Face]) -> tuple[Edge, ...]:
    return tuple(
        sorted(
            {
                tuple(sorted((face[index], face[(index + 1) % 3])))
                for face in faces
                for index in range(3)
            }
        )
    )


def _validate_tutte_inputs(
    faces: Iterable[Sequence[int]], distinguished_face: Sequence[int]
) -> tuple[tuple[Face, ...], Face, int]:
    input_faces = _materialize_faces(faces)
    oriented_faces, summary = _analyze_closed_sphere_triangulation(input_faces)
    face = _as_face(distinguished_face, name="distinguished_face")
    if frozenset(face) not in {frozenset(item) for item in oriented_faces}:
        raise ValueError("distinguished_face is not a face of the triangulation")
    if summary.vertex_labels != tuple(range(summary.n_vertices)):
        raise ValueError("Tutte embeddings require contiguous labels 0..V-1")
    return oriented_faces, face, summary.n_vertices


def _materialize_edge_weights(
    edges: Sequence[Edge], edge_weights: Mapping[Sequence[int], float] | None
) -> dict[Edge, float]:
    result = {edge: 1.0 for edge in edges}
    if edge_weights is None:
        return result
    edge_set = set(edges)
    for raw_edge, raw_weight in edge_weights.items():
        edge_tuple = tuple(raw_edge)
        if len(edge_tuple) != 2:
            raise ValueError("edge-weight keys must have two labels")
        edge = tuple(sorted((int(edge_tuple[0]), int(edge_tuple[1]))))
        if edge not in edge_set:
            raise ValueError(f"edge weight refers to non-edge {edge}")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("Tutte edge weights must be finite and positive")
        result[edge] = weight
    return result


def weighted_tutte_embedding(
    faces: Iterable[Sequence[int]],
    distinguished_face: Sequence[int],
    edge_weights: Mapping[Sequence[int], float] | None = None,
) -> np.ndarray:
    """Return a positive-weight straight-line embedding with an outer triangle.

    The ordered distinguished vertices are fixed at the three cube roots of
    unity.  Every other vertex is the weighted barycenter of its graph
    neighbors.  Positive weights and the validated sphere topology make the
    reduced graph Laplacian nonsingular.
    """
    oriented_faces, fixed_face, n_vertices = _validate_tutte_inputs(
        faces, distinguished_face
    )
    edges = _canonical_edges(oriented_faces)
    weights = _materialize_edge_weights(edges, edge_weights)
    adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for u, v in edges:
        weight = weights[(u, v)]
        adjacency[u].append((v, weight))
        adjacency[v].append((u, weight))

    fixed_positions = {
        label: np.array([root.real, root.imag], dtype=float)
        for label, root in zip(fixed_face, _ROOTS_OF_UNITY, strict=True)
    }
    free = tuple(label for label in range(n_vertices) if label not in fixed_positions)
    free_index = {label: index for index, label in enumerate(free)}
    result = np.empty((n_vertices, 2), dtype=float)
    for label, position in fixed_positions.items():
        result[label] = position
    if not free:
        return result

    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    right_hand_side = np.zeros((len(free), 2), dtype=float)
    for label in free:
        row = free_index[label]
        total = math.fsum(weight for _, weight in adjacency[label])
        rows.append(row)
        columns.append(row)
        data.append(total)
        for neighbor, weight in adjacency[label]:
            if neighbor in free_index:
                rows.append(row)
                columns.append(free_index[neighbor])
                data.append(-weight)
            else:
                right_hand_side[row] += weight * fixed_positions[neighbor]
    laplacian = coo_matrix(
        (data, (rows, columns)), shape=(len(free), len(free))
    ).tocsr()
    solution = np.asarray(spsolve(laplacian, right_hand_side), dtype=float)
    if solution.ndim == 1:
        solution = solution.reshape((-1, 2))
    if solution.shape != (len(free), 2) or not np.all(np.isfinite(solution)):
        raise RuntimeError("weighted Tutte solve did not produce finite coordinates")
    for label, position in zip(free, solution, strict=True):
        result[label] = position
    return result


def weighted_pole_link_tutte_embedding(
    faces: Iterable[Sequence[int]],
    distinguished_face: Sequence[int],
    *,
    edge_weights: Mapping[Sequence[int], float] | None = None,
) -> tuple[CP1Point, ...]:
    """Harmonically embed the disk obtained by deleting a distinguished pole.

    For the ordered face ``(pole, zero, one)``, the complete oriented link of
    ``pole`` is placed on a regular polygon and affinely normalized so that
    ``zero`` and ``one`` have coordinates 0 and 1.  The pole is returned as
    ``None`` and every remaining vertex lies in the open upper half-plane,
    except for the two real anchors.  Positive graph weights are used on the
    retained disk; weights on deleted pole edges are validated but ignored.
    """
    oriented_faces, fixed_face, n_vertices = _validate_tutte_inputs(
        faces, distinguished_face
    )
    pole, zero, one = fixed_face

    distinguished_set = frozenset(fixed_face)
    oriented_distinguished = next(
        face for face in oriented_faces if frozenset(face) == distinguished_set
    )
    cyclic = {
        oriented_distinguished,
        (
            oriented_distinguished[1],
            oriented_distinguished[2],
            oriented_distinguished[0],
        ),
        (
            oriented_distinguished[2],
            oriented_distinguished[0],
            oriented_distinguished[1],
        ),
    }
    if fixed_face not in cyclic:
        oriented_faces = tuple((a, c, b) for a, b, c in oriented_faces)

    successor: dict[int, int] = {}
    for face in oriented_faces:
        if pole not in face:
            continue
        pole_index = face.index(pole)
        left = face[(pole_index + 1) % 3]
        right = face[(pole_index + 2) % 3]
        if left in successor:
            raise ValueError(f"link of pole {pole} has repeated successor {left}")
        successor[left] = right
    if len(successor) < 3 or set(successor) != set(successor.values()):
        raise ValueError(f"link of pole {pole} is not one oriented cycle")
    if successor.get(zero) != one:
        raise ValueError(
            "distinguished_face orientation does not agree with the pole link"
        )

    link = [zero]
    while True:
        following = successor[link[-1]]
        if following == zero:
            break
        if following in link:
            raise ValueError(f"link of pole {pole} is not one oriented cycle")
        link.append(following)
    if len(link) != len(successor):
        raise ValueError(f"link of pole {pole} has more than one component")

    polygon = [
        cmath.exp(2j * math.pi * index / len(link)) for index in range(len(link))
    ]
    affine_scale = polygon[1] - polygon[0]
    if affine_scale == 0.0:
        raise ValueError("pole link polygon is degenerate")
    boundary_positions = {
        label: (polygon[index] - polygon[0]) / affine_scale
        for index, label in enumerate(link)
    }
    boundary_positions[zero] = 0.0 + 0.0j
    boundary_positions[one] = 1.0 + 0.0j

    all_edges = _canonical_edges(oriented_faces)
    all_weights = _materialize_edge_weights(all_edges, edge_weights)
    disk_edges = tuple(edge for edge in all_edges if pole not in edge)
    adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for u, v in disk_edges:
        weight = all_weights[(u, v)]
        adjacency[u].append((v, weight))
        adjacency[v].append((u, weight))

    boundary = frozenset(link)
    interior = tuple(
        label for label in range(n_vertices) if label != pole and label not in boundary
    )
    interior_index = {label: index for index, label in enumerate(interior)}
    points: list[CP1Point] = [0.0 + 0.0j] * n_vertices
    points[pole] = None
    for label, position in boundary_positions.items():
        points[label] = complex(position)

    if interior:
        rows: list[int] = []
        columns: list[int] = []
        data: list[float] = []
        right_hand_side = np.zeros(len(interior), dtype=complex)
        for label in interior:
            row = interior_index[label]
            total = math.fsum(weight for _, weight in adjacency[label])
            if total <= 0.0:
                raise ValueError(f"disk vertex {label} has no retained neighbors")
            rows.append(row)
            columns.append(row)
            data.append(total)
            for neighbor, weight in adjacency[label]:
                if neighbor in interior_index:
                    rows.append(row)
                    columns.append(interior_index[neighbor])
                    data.append(-weight)
                else:
                    right_hand_side[row] += weight * boundary_positions[neighbor]
        laplacian = coo_matrix(
            (data, (rows, columns)), shape=(len(interior), len(interior))
        ).tocsr()
        solution = np.asarray(spsolve(laplacian, right_hand_side), dtype=complex)
        if solution.shape != (len(interior),) or not np.all(np.isfinite(solution)):
            raise RuntimeError(
                "pole-link Tutte solve did not produce finite coordinates"
            )
        for label, position in zip(interior, solution, strict=True):
            points[label] = complex(position)

    for label, point in enumerate(points):
        if label in (pole, zero, one):
            continue
        if point is None or not point.imag > 0.0:
            raise RuntimeError(
                "pole-link Tutte embedding left the open upper half-plane"
            )
    if not _chordal_report(points)["passes"]:
        raise RuntimeError("pole-link Tutte embedding produced coincident vertices")
    return tuple(points)


def lognormal_edge_weights(
    faces: Iterable[Sequence[int]], *, sigma: float, seed: int
) -> dict[Edge, float]:
    """Generate deterministic positive log-normal weights for all graph edges."""
    materialized = _materialize_faces(faces)
    _analyze_closed_sphere_triangulation(materialized)
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma must be finite and non-negative")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    edges = _canonical_edges(materialized)
    generator = np.random.default_rng(int(seed))
    log_weights = sigma * generator.standard_normal(len(edges))
    log_weights -= float(np.mean(log_weights))
    try:
        samples = np.exp(log_weights)
    except FloatingPointError as exc:
        raise ValueError("sigma produced non-finite edge weights") from exc
    if not np.all(np.isfinite(samples)) or np.any(samples <= 0.0):
        raise ValueError("sigma produced non-finite edge weights")
    return {edge: float(weight) for edge, weight in zip(edges, samples, strict=True)}


class ZeroShearSystem:
    """Zero-shear residual/Jacobian in the finite cube-root disk gauge."""

    def __init__(
        self, faces: Iterable[Sequence[int]], distinguished_face: Sequence[int]
    ) -> None:
        oriented_faces, fixed_face, n_vertices = _validate_tutte_inputs(
            faces, distinguished_face
        )
        self.faces: tuple[Face, ...] = oriented_faces
        self.distinguished_face: Face = fixed_face
        self.n_vertices = n_vertices
        self.free_vertices = tuple(
            label for label in range(n_vertices) if label not in fixed_face
        )
        self._free_index = {
            label: index for index, label in enumerate(self.free_vertices)
        }
        self.edge_records = _edge_records(oriented_faces)
        self.n_residuals = len(self.edge_records)
        self.n_variables = 2 * len(self.free_vertices)

    def unpack(self, packed: Sequence[float]) -> tuple[CP1Point, ...]:
        values = np.asarray(packed, dtype=float)
        if values.shape != (self.n_variables,):
            raise ValueError(
                f"packed coordinates must have shape ({self.n_variables},)"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("packed coordinates must be finite")
        points: list[CP1Point] = [0j] * self.n_vertices
        for label, root in zip(self.distinguished_face, _ROOTS_OF_UNITY, strict=True):
            points[label] = root
        for index, label in enumerate(self.free_vertices):
            points[label] = complex(values[2 * index], values[2 * index + 1])
        return tuple(points)

    def pack(self, vertices_cp1: Sequence[Any]) -> np.ndarray:
        points = _materialize_vertices(vertices_cp1)
        if len(points) != self.n_vertices:
            raise ValueError("coordinate count does not match triangulation")
        packed = np.empty(self.n_variables, dtype=float)
        for index, label in enumerate(self.free_vertices):
            point = points[label]
            if point is None:
                raise ValueError("packed free vertices must be finite")
            packed[2 * index] = point.real
            packed[2 * index + 1] = point.imag
        return packed

    @staticmethod
    def _terms(
        record: tuple[Edge, int, int, int, int],
    ) -> tuple[tuple[float, int, int], ...]:
        _, a, x, b, y = record
        return ((1.0, a, x), (1.0, b, y), (-1.0, a, y), (-1.0, b, x))

    def residual(self, packed: Sequence[float]) -> np.ndarray:
        points = self.unpack(packed)
        result = np.zeros(self.n_residuals, dtype=float)
        for row, record in enumerate(self.edge_records):
            terms = []
            for coefficient, left, right in self._terms(record):
                point_left, point_right = points[left], points[right]
                if point_left is None or point_right is None:
                    continue
                distance = math.hypot(
                    point_left.real - point_right.real,
                    point_left.imag - point_right.imag,
                )
                if distance == 0.0 or not math.isfinite(distance):
                    raise ValueError("zero-shear residual contains a collision")
                terms.append(coefficient * math.log(distance))
            result[row] = math.fsum(terms)
        return result

    def jacobian(self, packed: Sequence[float]) -> csr_matrix:
        points = self.unpack(packed)
        rows: list[int] = []
        columns: list[int] = []
        data: list[float] = []
        accumulator: dict[tuple[int, int], float] = defaultdict(float)
        for row, record in enumerate(self.edge_records):
            for coefficient, left, right in self._terms(record):
                point_left, point_right = points[left], points[right]
                if point_left is None or point_right is None:
                    continue
                dx = point_left.real - point_right.real
                dy = point_left.imag - point_right.imag
                distance = math.hypot(dx, dy)
                if distance == 0.0 or not math.isfinite(distance):
                    raise ValueError("zero-shear Jacobian contains a collision")
                gx = (dx / distance) / distance
                gy = (dy / distance) / distance
                if left in self._free_index:
                    base = 2 * self._free_index[left]
                    accumulator[(row, base)] += coefficient * gx
                    accumulator[(row, base + 1)] += (
                        coefficient
                        * gy
                        * self._imaginary_parameter_derivative(left, point_left)
                    )
                if right in self._free_index:
                    base = 2 * self._free_index[right]
                    accumulator[(row, base)] -= coefficient * gx
                    accumulator[(row, base + 1)] -= (
                        coefficient
                        * gy
                        * self._imaginary_parameter_derivative(right, point_right)
                    )
        for (row, column), value in sorted(accumulator.items()):
            if value != 0.0:
                rows.append(row)
                columns.append(column)
                data.append(value)
        return coo_matrix(
            (data, (rows, columns)),
            shape=(self.n_residuals, self.n_variables),
        ).tocsr()

    def _imaginary_parameter_derivative(self, label: int, point: complex) -> float:
        """Return ``d Im(z_label) / d packed_y_label`` for this gauge."""
        del label, point
        return 1.0

    # SciPy/common numerical naming conveniences.
    fun = residual
    residuals = residual
    jac = jacobian


class StandardUpperHalfPlaneZeroShearSystem(ZeroShearSystem):
    """Zero-shear system in the ``(infinity, 0, 1)`` log-height gauge.

    The ordered distinguished face is fixed at those three CP1 points.  Each
    free vertex has packed coordinates ``(x, t)`` and is unpacked as
    ``x + i exp(t)``.  The sparse analytic Jacobian includes the chain factor
    ``d exp(t) / dt = Im(z)``.
    """

    def unpack(self, packed: Sequence[float]) -> tuple[CP1Point, ...]:
        values = np.asarray(packed, dtype=float)
        if values.shape != (self.n_variables,):
            raise ValueError(
                f"packed coordinates must have shape ({self.n_variables},)"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("packed coordinates must be finite")
        points: list[CP1Point] = [0j] * self.n_vertices
        pole, zero, one = self.distinguished_face
        points[pole] = None
        points[zero] = 0.0 + 0.0j
        points[one] = 1.0 + 0.0j
        for index, label in enumerate(self.free_vertices):
            points[label] = complex(values[2 * index], math.exp(values[2 * index + 1]))
        return tuple(points)

    def pack(self, vertices_cp1: Sequence[Any]) -> np.ndarray:
        points = _materialize_vertices(vertices_cp1)
        if len(points) != self.n_vertices:
            raise ValueError("coordinate count does not match triangulation")
        packed = np.empty(self.n_variables, dtype=float)
        for index, label in enumerate(self.free_vertices):
            point = points[label]
            if point is None:
                raise ValueError("packed free vertices must be finite")
            if not point.imag > 0.0:
                raise ValueError(
                    "standard-upper-half-plane free vertices must have positive "
                    "imaginary part"
                )
            packed[2 * index] = point.real
            packed[2 * index + 1] = math.log(point.imag)
        return packed

    def _imaginary_parameter_derivative(self, label: int, point: complex) -> float:
        del label
        return point.imag


def _normalize_cp1_points(
    points: Sequence[CP1Point], distinguished_face: Face
) -> tuple[CP1Point, ...]:
    matrix = _mobius_to_standard_matrix(points, distinguished_face)
    normalized = list(_mobius_points(points, matrix))
    a, b, c = distinguished_face
    normalized[a] = None
    normalized[b] = 0.0 + 0.0j
    normalized[c] = 1.0 + 0.0j
    return tuple(normalized)


def normalize_cp1_points(
    vertices_cp1: Sequence[Any],
    distinguished_face: Sequence[int],
    *,
    positive_handedness: bool = False,
) -> tuple[CP1Point, ...]:
    """Normalize an ordered CP1 triple to ``(infinity, 0, 1)``.

    When ``positive_handedness`` is true, all non-anchor vertices must occupy
    one strict open half-plane.  A lower-half-plane realization is conjugated;
    mixed or real non-anchor coordinates are rejected.
    """
    points = _materialize_vertices(vertices_cp1)
    face = _as_face(distinguished_face, name="distinguished_face")
    if any(label < 0 or label >= len(points) for label in face):
        raise ValueError("distinguished_face label is outside the coordinate range")
    if not _chordal_report(points)["passes"]:
        raise ValueError("CP1 coordinates must be pairwise distinct")
    normalized = _normalize_cp1_points(points, face)
    if not positive_handedness:
        return normalized

    nonanchors = [point for label, point in enumerate(normalized) if label not in face]
    if any(point is None for point in nonanchors):
        raise ValueError("only the distinguished pole may be infinity")
    imaginary_parts = [complex(point).imag for point in nonanchors]
    if imaginary_parts and all(value < 0.0 for value in imaginary_parts):
        normalized = tuple(
            None if point is None else point.conjugate() for point in normalized
        )
    elif imaginary_parts and not all(value > 0.0 for value in imaginary_parts):
        raise ValueError(
            "normalized non-anchor vertices must lie in one strict half-plane"
        )
    pole, zero, one = face
    normalized_list = list(normalized)
    normalized_list[pole] = None
    normalized_list[zero] = 0.0 + 0.0j
    normalized_list[one] = 1.0 + 0.0j
    return tuple(normalized_list)


def _failed_solve_result(
    exc: Exception,
    *,
    started: float,
    initial_embedding: np.ndarray | None,
    initial_points: tuple[CP1Point, ...] | None,
) -> TutteSolveResult:
    finished = time.perf_counter()
    return TutteSolveResult(
        success=False,
        classification="error",
        message=str(exc),
        vertices_cp1=None,
        normalized_vertices_cp1=None,
        audit=None,
        initial_embedding=initial_embedding,
        initial_vertices_cp1=initial_points,
        cost=None,
        optimality=None,
        nfev=0,
        njev=None,
        max_abs_residual=None,
        residual_l2=None,
        trajectory=(),
        trajectory_kind="not_recorded",
        solver_metadata={},
        timings={
            "total_seconds": finished - started,
            "initialization_seconds": finished - started,
            "solve_seconds": 0.0,
            "audit_seconds": 0.0,
            "trajectory_audit_seconds": 0.0,
        },
    )


def _run_zero_shear_solver(
    materialized_faces: tuple[Face, ...],
    fixed_face: Face,
    initial_points: tuple[CP1Point, ...],
    system: ZeroShearSystem,
    *,
    initial_embedding: np.ndarray | None,
    started: float,
    initialization_finished: float,
    gauge: str,
    x_scale: str | float | Sequence[float] | None,
    max_nfev: int,
    lm_polish_steps: int,
    record_trajectory: bool,
    shear_tolerance: float,
    support_tolerance: float,
    bending_tolerance: float,
    check_spherical_embedding: bool,
    require_standard_normalization: bool,
    ftol: float,
    xtol: float,
    gtol: float,
) -> TutteSolveResult:
    initial = system.pack(initial_points)
    trajectory: list[dict[str, Any]] = []
    trajectory_kind = "not_recorded"
    evaluation = 0

    def residual(values: np.ndarray) -> np.ndarray:
        nonlocal evaluation
        output = system.residual(values)
        evaluation += 1
        return output

    def append_trajectory(values: Sequence[float], *, kind: str) -> None:
        packed = np.asarray(values, dtype=float)
        output = system.residual(packed)
        trajectory.append(
            {
                "kind": kind,
                "iteration": len(trajectory),
                "evaluation_count": evaluation,
                "max_abs_shear": float(np.max(np.abs(output))),
                "residual_l2": float(np.linalg.norm(output)),
                "packed": packed.copy(),
            }
        )

    optimization_kwargs: dict[str, Any] = {}
    objective = residual
    if record_trajectory:
        append_trajectory(initial, kind="accepted_iterate")
        if "callback" in inspect.signature(least_squares).parameters:
            trajectory_kind = "accepted_iterates"

            def callback(intermediate_result: Any) -> None:
                values = (
                    intermediate_result.x
                    if hasattr(intermediate_result, "x")
                    else intermediate_result
                )
                append_trajectory(values, kind="accepted_iterate")

            optimization_kwargs["callback"] = callback
        else:
            trajectory_kind = "residual_evaluations_fallback"
            last_recorded: np.ndarray | None = None

            def recording_residual(values: np.ndarray) -> np.ndarray:
                nonlocal last_recorded
                output = residual(values)
                if last_recorded is None or not np.array_equal(values, last_recorded):
                    append_trajectory(values, kind="residual_evaluation_fallback")
                    last_recorded = np.asarray(values, dtype=float).copy()
                return output

            objective = recording_residual

    solve_started = time.perf_counter()
    trf_optimization = least_squares(
        objective,
        initial,
        jac=system.jacobian,
        method="trf",
        tr_solver="lsmr",
        x_scale=x_scale,
        max_nfev=max_nfev,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        **optimization_kwargs,
    )
    trf_finished = time.perf_counter()
    optimization = trf_optimization
    lm_stages: list[dict[str, Any]] = []
    lm_started = trf_finished
    for step in range(lm_polish_steps):
        optimization = least_squares(
            residual,
            optimization.x,
            jac=lambda values: system.jacobian(values).toarray(),
            method="lm",
            max_nfev=50,
            ftol=1e-15,
            xtol=1e-15,
            gtol=1e-15,
        )
        lm_stages.append(
            {
                "step": step + 1,
                "success": bool(optimization.success),
                "status": int(optimization.status),
                "message": str(optimization.message),
                "nfev": int(optimization.nfev),
                "njev": (None if optimization.njev is None else int(optimization.njev)),
                "cost": float(optimization.cost),
                "optimality": float(optimization.optimality),
            }
        )
        if record_trajectory:
            append_trajectory(optimization.x, kind="lm_polish")
    solve_finished = time.perf_counter()
    if record_trajectory and (
        not trajectory or not np.array_equal(trajectory[-1]["packed"], optimization.x)
    ):
        append_trajectory(optimization.x, kind="accepted_iterate")
    if record_trajectory and lm_polish_steps:
        trajectory_kind = f"{trajectory_kind}_plus_lm_polish"
    points = system.unpack(optimization.x)
    normalized_points = (
        points
        if isinstance(system, StandardUpperHalfPlaneZeroShearSystem)
        else _normalize_cp1_points(points, fixed_face)
    )
    residual_values = system.residual(optimization.x)
    audit_started = time.perf_counter()
    audit = audit_zero_shear_realization(
        materialized_faces,
        points,
        distinguished_face=fixed_face,
        shear_tolerance=shear_tolerance,
        support_tolerance=support_tolerance,
        bending_tolerance=bending_tolerance,
        check_spherical_embedding=check_spherical_embedding,
        require_standard_normalization=require_standard_normalization,
    )
    audit_finished = time.perf_counter()
    final_packed = np.asarray(optimization.x, dtype=float)
    trajectory_audit_started = audit_finished
    if record_trajectory:
        for row in trajectory:
            row_packed = np.asarray(row["packed"], dtype=float)
            iterate_audit = (
                audit
                if np.array_equal(row_packed, final_packed)
                else audit_zero_shear_realization(
                    materialized_faces,
                    system.unpack(row_packed),
                    distinguished_face=fixed_face,
                    shear_tolerance=shear_tolerance,
                    support_tolerance=support_tolerance,
                    bending_tolerance=bending_tolerance,
                    check_spherical_embedding=check_spherical_embedding,
                    require_standard_normalization=require_standard_normalization,
                )
            )
            support_row = iterate_audit.support
            bending_row = iterate_audit.bending
            planar_row = iterate_audit.planar_embedding
            spherical_row = iterate_audit.spherical_embedding
            qhull_row = iterate_audit.qhull_agreement
            in_target_cell = bool(
                iterate_audit.topology.get("passes", False)
                and iterate_audit.chordal_distinctness.get("passes", False)
                and support_row.get("passes", False)
                and bending_row.get("passes", False)
            )
            row.update(
                {
                    "classification": iterate_audit.classification,
                    "strict_target_zero_shear": iterate_audit.strict_pass,
                    "in_target_convex_cell": in_target_cell,
                    "support_status": support_row.get("status"),
                    "support_passes": support_row.get("passes"),
                    "min_support_margin": support_row.get("min_normalized_margin"),
                    "support_inconsistent_face_count": support_row.get(
                        "inconsistent_face_count"
                    ),
                    "support_tie_count": support_row.get("tie_count"),
                    "bending_status": bending_row.get("status"),
                    "bending_coherent": bending_row.get("coherent_sign"),
                    "bending_sign_violation_count": bending_row.get(
                        "sign_violation_count"
                    ),
                    "min_bending_endpoint_margin_radians": bending_row.get(
                        "min_endpoint_margin_radians"
                    ),
                    "planar_status": planar_row.get("status"),
                    "planar_embedded": planar_row.get("embedded"),
                    "spherical_status": spherical_row.get("status"),
                    "spherical_embedded": spherical_row.get("embedded"),
                    "qhull_agrees": qhull_row.get("agrees"),
                    "qhull_missing_face_count": len(qhull_row.get("missing_faces", ())),
                    "qhull_extra_face_count": len(qhull_row.get("extra_faces", ())),
                }
            )
    trajectory_audit_finished = time.perf_counter()
    finished = time.perf_counter()
    residual_pass = bool(audit.shear.get("passes", False))
    if not optimization.success and not residual_pass:
        classification = "solver_failure"
    elif not residual_pass:
        classification = "residual_failure"
    else:
        classification = audit.classification
    success = bool(optimization.success and residual_pass)
    stage_nfev = [int(trf_optimization.nfev), *(row["nfev"] for row in lm_stages)]
    stage_njev = [
        None if trf_optimization.njev is None else int(trf_optimization.njev),
        *(row["njev"] for row in lm_stages),
    ]
    return TutteSolveResult(
        success=success,
        classification=classification,
        message=str(optimization.message),
        vertices_cp1=points,
        normalized_vertices_cp1=normalized_points,
        audit=audit,
        initial_embedding=initial_embedding,
        initial_vertices_cp1=initial_points,
        cost=float(optimization.cost),
        optimality=float(optimization.optimality),
        nfev=sum(stage_nfev),
        njev=(
            None
            if all(value is None for value in stage_njev)
            else sum(value for value in stage_njev if value is not None)
        ),
        max_abs_residual=float(np.max(np.abs(residual_values))),
        residual_l2=float(np.linalg.norm(residual_values)),
        trajectory=tuple(trajectory),
        trajectory_kind=trajectory_kind,
        solver_metadata={
            "gauge": gauge,
            "x_scale": x_scale,
            "trf": {
                "method": "trf",
                "tr_solver": "lsmr",
                "nfev": int(trf_optimization.nfev),
                "njev": (
                    None
                    if trf_optimization.njev is None
                    else int(trf_optimization.njev)
                ),
                "success": bool(trf_optimization.success),
                "status": int(trf_optimization.status),
                "message": str(trf_optimization.message),
                "cost": float(trf_optimization.cost),
                "optimality": float(trf_optimization.optimality),
            },
            "lm_polish": {
                "steps_requested": lm_polish_steps,
                "steps_completed": len(lm_stages),
                "max_nfev_per_step": 50,
                "ftol": 1e-15,
                "xtol": 1e-15,
                "gtol": 1e-15,
                "stages": lm_stages,
            },
        },
        timings={
            "total_seconds": finished - started,
            "initialization_seconds": initialization_finished - started,
            "solve_seconds": solve_finished - solve_started,
            "trf_seconds": trf_finished - solve_started,
            "lm_polish_seconds": solve_finished - lm_started,
            "audit_seconds": audit_finished - audit_started,
            "trajectory_audit_seconds": (
                trajectory_audit_finished - trajectory_audit_started
            ),
        },
    )


def solve_zero_shear_from_initial(
    faces: Iterable[Sequence[int]],
    distinguished_face: Sequence[int],
    initial_vertices_cp1: Sequence[Any],
    *,
    gauge: str = "cube_roots",
    x_scale: str | float | Sequence[float] | None = "jac",
    max_nfev: int = 300,
    lm_polish_steps: int = 0,
    record_trajectory: bool = False,
    shear_tolerance: float = 1e-8,
    support_tolerance: float = 0.0,
    bending_tolerance: float = 1e-10,
    check_spherical_embedding: bool = True,
    ftol: float = 1e-13,
    xtol: float = 1e-13,
    gtol: float = 1e-13,
) -> TutteSolveResult:
    """Solve all edge shears from caller-supplied labelled CP1 coordinates.

    ``gauge='cube_roots'`` preserves the historical unconstrained finite
    ``(x, y)`` parameterization.  ``gauge='standard_upper_half_plane'`` first
    normalizes the ordered distinguished face to ``(infinity, 0, 1)`` and
    uses free ``(x, log(y))`` variables, thereby preserving ``y > 0``.
    ``lm_polish_steps`` optionally follows sparse TRF/LSMR with that many
    dense Levenberg--Marquardt passes (50 evaluations and 1e-15 tolerances per
    pass), matching the reconstructed Sol endgame without changing the legacy
    default.
    """
    started = time.perf_counter()
    embedding: np.ndarray | None = None
    initial_points: tuple[CP1Point, ...] | None = None
    try:
        if not isinstance(max_nfev, int) or isinstance(max_nfev, bool) or max_nfev < 1:
            raise ValueError("max_nfev must be a positive integer")
        if (
            not isinstance(lm_polish_steps, int)
            or isinstance(lm_polish_steps, bool)
            or lm_polish_steps < 0
        ):
            raise ValueError("lm_polish_steps must be a non-negative integer")
        materialized_faces = _materialize_faces(faces)
        fixed_face = _as_face(distinguished_face, name="distinguished_face")
        _, _, n_vertices = _validate_tutte_inputs(materialized_faces, fixed_face)
        supplied_points = _materialize_vertices(initial_vertices_cp1)
        if len(supplied_points) != n_vertices:
            raise ValueError("coordinate count does not match triangulation")
        if not _chordal_report(supplied_points)["passes"]:
            raise ValueError("initial CP1 coordinates must be pairwise distinct")

        if gauge == "cube_roots":
            anchors_match = all(
                supplied_points[label] == root
                for label, root in zip(fixed_face, _ROOTS_OF_UNITY, strict=True)
            )
            if anchors_match:
                initial_points = supplied_points
            else:
                matrix = _standard_to_cube_roots_matrix() @ _mobius_to_standard_matrix(
                    supplied_points, fixed_face
                )
                gauged = list(_mobius_points(supplied_points, matrix))
                for label, root in zip(fixed_face, _ROOTS_OF_UNITY, strict=True):
                    gauged[label] = root
                initial_points = tuple(gauged)
            if any(point is None for point in initial_points):
                raise ValueError("cube-root gauge requires finite transformed vertices")
            embedding = np.asarray(
                [
                    [complex(point).real, complex(point).imag]
                    for point in initial_points
                ],
                dtype=float,
            )
            system: ZeroShearSystem = ZeroShearSystem(materialized_faces, fixed_face)
            normalization_required = False
        elif gauge == "standard_upper_half_plane":
            initial_points = normalize_cp1_points(
                supplied_points,
                fixed_face,
                positive_handedness=True,
            )
            system = StandardUpperHalfPlaneZeroShearSystem(
                materialized_faces, fixed_face
            )
            normalization_required = fixed_face == (0, 1, 2)
        else:
            raise ValueError(
                "gauge must be 'cube_roots' or 'standard_upper_half_plane'"
            )

        initialization_finished = time.perf_counter()
        return _run_zero_shear_solver(
            materialized_faces,
            fixed_face,
            initial_points,
            system,
            initial_embedding=embedding,
            started=started,
            initialization_finished=initialization_finished,
            gauge=gauge,
            x_scale=x_scale,
            max_nfev=max_nfev,
            lm_polish_steps=lm_polish_steps,
            record_trajectory=record_trajectory,
            shear_tolerance=shear_tolerance,
            support_tolerance=support_tolerance,
            bending_tolerance=bending_tolerance,
            check_spherical_embedding=check_spherical_embedding,
            require_standard_normalization=normalization_required,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
        )
    except Exception as exc:  # return a finite operational endpoint
        return _failed_solve_result(
            exc,
            started=started,
            initial_embedding=embedding,
            initial_points=initial_points,
        )


def solve_zero_shear_from_tutte(
    faces: Iterable[Sequence[int]],
    distinguished_face: Sequence[int],
    *,
    edge_weights: Mapping[Sequence[int], float] | None = None,
    x_scale: str | float | Sequence[float] | None = "jac",
    max_nfev: int = 300,
    lm_polish_steps: int = 0,
    record_trajectory: bool = False,
    shear_tolerance: float = 1e-8,
    support_tolerance: float = 0.0,
    bending_tolerance: float = 1e-10,
    check_spherical_embedding: bool = True,
    ftol: float = 1e-13,
    xtol: float = 1e-13,
    gtol: float = 1e-13,
) -> TutteSolveResult:
    """Solve all edge shears from the historical outer-face Tutte start."""
    started = time.perf_counter()
    embedding: np.ndarray | None = None
    initial_points: tuple[CP1Point, ...] | None = None
    try:
        if not isinstance(max_nfev, int) or isinstance(max_nfev, bool) or max_nfev < 1:
            raise ValueError("max_nfev must be a positive integer")
        if (
            not isinstance(lm_polish_steps, int)
            or isinstance(lm_polish_steps, bool)
            or lm_polish_steps < 0
        ):
            raise ValueError("lm_polish_steps must be a non-negative integer")
        materialized_faces = _materialize_faces(faces)
        fixed_face = _as_face(distinguished_face, name="distinguished_face")
        embedding = weighted_tutte_embedding(
            materialized_faces, fixed_face, edge_weights=edge_weights
        )
        initial_points = tuple(complex(x, y) for x, y in embedding)
        system = ZeroShearSystem(materialized_faces, fixed_face)
        initialization_finished = time.perf_counter()
        return _run_zero_shear_solver(
            materialized_faces,
            fixed_face,
            initial_points,
            system,
            initial_embedding=embedding,
            started=started,
            initialization_finished=initialization_finished,
            gauge="cube_roots",
            x_scale=x_scale,
            max_nfev=max_nfev,
            lm_polish_steps=lm_polish_steps,
            record_trajectory=record_trajectory,
            shear_tolerance=shear_tolerance,
            support_tolerance=support_tolerance,
            bending_tolerance=bending_tolerance,
            check_spherical_embedding=check_spherical_embedding,
            require_standard_normalization=False,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
        )
    except Exception as exc:  # return a finite operational endpoint
        return _failed_solve_result(
            exc,
            started=started,
            initial_embedding=embedding,
            initial_points=initial_points,
        )


__all__ = [
    "StandardUpperHalfPlaneZeroShearSystem",
    "TutteSolveResult",
    "ZeroShearRealizationAudit",
    "ZeroShearSystem",
    "audit_zero_shear_realization",
    "lognormal_edge_weights",
    "normalize_cp1_points",
    "solve_zero_shear_from_initial",
    "solve_zero_shear_from_tutte",
    "weighted_pole_link_tutte_embedding",
    "weighted_tutte_embedding",
]
