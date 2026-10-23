#!/usr/bin/env python3
"""Strict invariant grader for symmetric ideal-H4 coordinate responses.

The checker deliberately has no private reference realization.  It grades the
submitted point set itself: strict ideal-boundary geometry, all-pole volume
agreement, a minimum hyperbolic volume, and the full geometric symmetry group.
The result is a double-precision numerical certificate, not an interval proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Integral
import os
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ideal_poly_volume_toolkit.rl.polytope.h4_symmetric_benchmark import (
    audit_h4_symmetric_candidate,
)


SCHEMA_VERSION = "strict-h4-symmetric-response-grade/1"


class CheckerInputError(ValueError):
    """The response or grading contract is malformed."""


def _reject_json_constant(value: str) -> None:
    raise CheckerInputError(f"non-standard JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckerInputError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def load_json_strict_with_sha256(path: str | Path) -> tuple[dict[str, Any], str]:
    """Read and hash one strict JSON object from a single byte snapshot."""
    source = Path(path)
    try:
        raw = source.read_bytes()
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except CheckerInputError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckerInputError(f"cannot load response: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckerInputError("response root must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def load_json_strict(path: str | Path) -> dict[str, Any]:
    """Load one RFC-compatible JSON object without duplicate keys or NaN."""
    return load_json_strict_with_sha256(path)[0]


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckerInputError(f"{name} must be an ordinary JSON number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise CheckerInputError(f"{name} is outside the float64 range") from exc
    if not math.isfinite(result):
        raise CheckerInputError(f"{name} must be finite")
    return result


def _nonnegative_real(value: Any, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result < 0.0:
        raise CheckerInputError(f"{name} must be nonnegative")
    return result


def _positive_real(value: Any, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result <= 0.0:
        raise CheckerInputError(f"{name} must be positive")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CheckerInputError(f"{name} must be a positive integer")
    return value


def _vertices_array(response: Mapping[str, Any], *, n_vertices: int) -> np.ndarray:
    if set(response) != {"vertices"}:
        raise CheckerInputError("response must contain exactly the key 'vertices'")
    vertices = response["vertices"]
    if not isinstance(vertices, list) or len(vertices) != n_vertices:
        raise CheckerInputError(f"vertices must be a list of exactly {n_vertices} rows")
    rows: list[list[float]] = []
    for row_index, row in enumerate(vertices):
        if not isinstance(row, list) or len(row) != 4:
            raise CheckerInputError(
                f"vertices[{row_index}] must contain exactly 4 numbers"
            )
        rows.append(
            [
                _finite_real(value, name=f"vertices[{row_index}][{column}]")
                for column, value in enumerate(row)
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("grade contains a non-finite float")
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"cannot encode {type(value).__name__} as strict JSON")


def strict_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _jsonable(value),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def write_atomic(
    path: str | Path,
    payload: bytes,
    *,
    overwrite: bool = False,
) -> None:
    """Publish a complete grade atomically; refuse clobber by default."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise CheckerInputError(
                    f"output already exists: {target}; use --overwrite to replace it"
                ) from exc
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def grade_response_document(
    response: Mapping[str, Any],
    *,
    n_vertices: int,
    minimum_volume: float,
    geometric_group_order: int,
    orientation_preserving_count: int | None = None,
    volume_spread_tolerance: float = 1.0e-8,
    unit_tolerance: float = 1.0e-9,
    distinct_tolerance: float = 1.0e-8,
    support_tolerance: float = 1.0e-9,
    isometry_tolerance: float = 1.0e-7,
    max_group_order: int = 100_000,
    include_audit: bool = False,
    audit_function: Callable[..., Mapping[str, Any]] = audit_h4_symmetric_candidate,
) -> dict[str, Any]:
    """Grade a parsed response against an invariant H4 challenge contract."""
    n_vertices = _positive_integer(n_vertices, name="n_vertices")
    minimum_volume = _finite_real(minimum_volume, name="minimum_volume")
    geometric_group_order = _positive_integer(
        geometric_group_order, name="geometric_group_order"
    )
    volume_spread_tolerance = _nonnegative_real(
        volume_spread_tolerance, name="volume_spread_tolerance"
    )
    symmetry_tolerances = {
        "unit_tolerance": _positive_real(unit_tolerance, name="unit_tolerance"),
        "distinct_tolerance": _positive_real(
            distinct_tolerance, name="distinct_tolerance"
        ),
        "support_tolerance": _positive_real(
            support_tolerance, name="support_tolerance"
        ),
        "isometry_tolerance": _positive_real(
            isometry_tolerance, name="isometry_tolerance"
        ),
        "max_group_order": _positive_integer(max_group_order, name="max_group_order"),
    }
    if geometric_group_order > symmetry_tolerances["max_group_order"]:
        raise CheckerInputError("geometric_group_order cannot exceed max_group_order")
    if orientation_preserving_count is not None:
        if (
            isinstance(orientation_preserving_count, bool)
            or not isinstance(orientation_preserving_count, int)
            or not 0 <= orientation_preserving_count <= geometric_group_order
        ):
            raise CheckerInputError(
                "orientation_preserving_count must be an integer between 0 and "
                "geometric_group_order"
            )

    points = _vertices_array(response, n_vertices=n_vertices)
    audit = dict(
        audit_function(
            points,
            volume_spread_tolerance=volume_spread_tolerance,
            symmetry_kwargs=symmetry_tolerances,
            local_maximum=False,
        )
    )
    symmetry = audit.get("symmetry", {})
    pole_audit = audit.get("pole_volume_audit", {})
    measured_order = symmetry.get("geometric_group_order")
    measured_orientation_count = symmetry.get("orientation_preserving_count")
    minimum_measured_volume = pole_audit.get("minimum_volume")
    maximum_measured_volume = pole_audit.get("maximum_volume")
    measured_spread = pole_audit.get("spread")
    finite_minimum = (
        isinstance(minimum_measured_volume, (int, float))
        and not isinstance(minimum_measured_volume, bool)
        and math.isfinite(float(minimum_measured_volume))
    )
    finite_maximum = (
        isinstance(maximum_measured_volume, (int, float))
        and not isinstance(maximum_measured_volume, bool)
        and math.isfinite(float(maximum_measured_volume))
    )
    finite_spread = (
        isinstance(measured_spread, (int, float))
        and not isinstance(measured_spread, bool)
        and math.isfinite(float(measured_spread))
        and float(measured_spread) >= 0.0
    )
    volume_summary_consistent = (
        finite_minimum
        and finite_maximum
        and finite_spread
        and float(maximum_measured_volume) >= float(minimum_measured_volume)
        and math.isclose(
            float(measured_spread),
            float(maximum_measured_volume) - float(minimum_measured_volume),
            rel_tol=1.0e-12,
            abs_tol=1.0e-14,
        )
    )
    required_geometry_checks = (
        "shape",
        "finite",
        "unit",
        "distinct",
        "full_dimensional",
        "all_vertices_extreme",
        "simplicial",
        "strict_support",
    )
    exact_integer_order = (
        isinstance(measured_order, Integral)
        and not isinstance(measured_order, (bool, np.bool_))
        and int(measured_order) == geometric_group_order
    )
    exact_orientation_count = isinstance(
        measured_orientation_count, Integral
    ) and not isinstance(measured_orientation_count, (bool, np.bool_))

    gates = {
        "dimension_and_count": audit.get("dimension") == 4
        and audit.get("n_vertices") == n_vertices,
        "strict_ideal_geometry": symmetry.get("valid_geometry") is True
        and all(
            symmetry.get("checks", {}).get(name) is True
            for name in required_geometry_checks
        ),
        "all_poles_verified": pole_audit.get("all_poles_requested") is True
        and pole_audit.get("requested_count") == n_vertices
        and pole_audit.get("successful_count") == n_vertices
        and pole_audit.get("failure_count") == 0
        and audit.get("classification_basis", {}).get("all_poles_verified") is True,
        "pole_volume_stable": pole_audit.get("within_tolerance") is True
        and volume_summary_consistent
        and float(measured_spread) <= volume_spread_tolerance,
        "minimum_volume": finite_minimum
        and float(minimum_measured_volume) >= minimum_volume,
        "full_group_certified": symmetry.get("full_group_certified") is True,
        "exact_geometric_group_order": exact_integer_order,
        "orientation_preserving_count": orientation_preserving_count is None
        or (
            exact_orientation_count
            and int(measured_orientation_count) == orientation_preserving_count
        ),
    }
    strict_pass = all(gates.values())
    grade: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "strict_pass": strict_pass,
        "classification": (
            "strict_h4_symmetric_pass" if strict_pass else "failed_h4_symmetric_audit"
        ),
        "contract": {
            "dimension": 4,
            "n_vertices": n_vertices,
            "minimum_volume": minimum_volume,
            "geometric_group_order": geometric_group_order,
            "orientation_preserving_count": orientation_preserving_count,
            "volume_spread_tolerance": volume_spread_tolerance,
            **symmetry_tolerances,
        },
        "gates": gates,
        "measurements": {
            "minimum_volume": minimum_measured_volume,
            "maximum_volume": maximum_measured_volume,
            "pole_volume_spread": measured_spread,
            "geometric_group_order": measured_order,
            "orientation_preserving_count": measured_orientation_count,
            "facet_count": symmetry.get("topology", {}).get("n_facets"),
            "minimum_support_margin": symmetry.get("topology", {}).get(
                "min_support_margin"
            ),
            "minimum_pairwise_distance": symmetry.get("topology", {}).get(
                "min_pairwise_distance"
            ),
            "maximum_accepted_isometry_residual": symmetry.get("residuals", {}).get(
                "max_accepted_isometry_residual"
            ),
        },
    }
    if include_audit:
        grade["audit"] = audit
    return _jsonable(grade)


def grade_response(
    response_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    response, response_sha256 = load_json_strict_with_sha256(response_path)
    grade = grade_response_document(response, **kwargs)
    grade["response_sha256"] = response_sha256
    return grade


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response", type=Path)
    parser.add_argument("--n-vertices", type=int, required=True)
    parser.add_argument("--minimum-volume", type=float, required=True)
    parser.add_argument("--geometric-group-order", type=int, required=True)
    parser.add_argument("--orientation-preserving-count", type=int)
    parser.add_argument("--volume-spread-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--unit-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--distinct-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--support-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--isometry-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--max-group-order", type=int, default=100_000)
    parser.add_argument(
        "--include-audit",
        action="store_true",
        help="include the full internal numerical audit in the private grade",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _error_grade(classification: str, exc: Exception) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "strict_pass": False,
        "classification": classification,
        "error": {"type": type(exc).__name__, "message": str(exc)},
    }


def _emit_grade(
    grade: Mapping[str, Any],
    *,
    output: Path | None,
    overwrite: bool,
) -> bool:
    """Emit one grade and report whether publication succeeded."""
    try:
        payload = strict_json_bytes(grade)
        if output is None:
            sys.stdout.buffer.write(payload)
        else:
            write_atomic(output, payload, overwrite=overwrite)
    except (CheckerInputError, OSError, TypeError, ValueError) as exc:
        print(f"error: cannot publish grade: {exc}", file=sys.stderr)
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.overwrite and args.output is None:
        exc = CheckerInputError("--overwrite requires --output")
        _emit_grade(
            _error_grade("invalid_response_or_contract", exc),
            output=None,
            overwrite=False,
        )
        return 1
    if args.output is not None and args.output.exists() and not args.overwrite:
        print(
            f"error: output already exists: {args.output}; use --overwrite to replace it",
            file=sys.stderr,
        )
        return 2
    try:
        grade = grade_response(
            args.response,
            n_vertices=args.n_vertices,
            minimum_volume=args.minimum_volume,
            geometric_group_order=args.geometric_group_order,
            orientation_preserving_count=args.orientation_preserving_count,
            volume_spread_tolerance=args.volume_spread_tolerance,
            unit_tolerance=args.unit_tolerance,
            distinct_tolerance=args.distinct_tolerance,
            support_tolerance=args.support_tolerance,
            isometry_tolerance=args.isometry_tolerance,
            max_group_order=args.max_group_order,
            include_audit=args.include_audit,
        )
        if not _emit_grade(grade, output=args.output, overwrite=args.overwrite):
            return 2
        return 0 if grade["strict_pass"] else 1
    except (CheckerInputError, TypeError, ValueError) as exc:
        if not _emit_grade(
            _error_grade("invalid_response_or_contract", exc),
            output=args.output,
            overwrite=args.overwrite,
        ):
            return 2
        return 1
    except Exception as exc:  # Convert dependency/audit failures to strict JSON.
        if not _emit_grade(
            _error_grade("checker_runtime_error", exc),
            output=args.output,
            overwrite=args.overwrite,
        ):
            return 2
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
