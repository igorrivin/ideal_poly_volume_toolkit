#!/usr/bin/env python3
"""Strict reference-free grader for prescribed-combinatorics ideal-H4 responses.

The public target contains only a vertex count, the complete unordered
tetrahedral-facet incidence, a volume floor, and numerical tolerances.  The
checker never loads or compares against a private coordinate realization.
It audits the submitted points themselves, including every projection pole.
The result is a double-precision numerical certificate, not an interval proof.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np

from ideal_poly_volume_toolkit.rl.polytope.h4_combinatorics_benchmark import (
    audit_h4_combinatorics_candidate,
    compare_facet_incidence,
)


TARGET_SCHEMA_VERSION = "strict-h4-combinatorics-response-contract/1"
GRADE_SCHEMA_VERSION = "strict-h4-combinatorics-response-grade/1"
_TARGET_KEYS = {
    "schema",
    "instance_id",
    "dimension",
    "n_vertices",
    "facets",
    "facet_equivalence",
    "minimum_volume",
    "tolerances",
}
_TOLERANCE_KEYS = {
    "unit",
    "distinct",
    "support",
    "pole_volume_spread",
}
_FACET_EQUIVALENCE = "unlabeled_vertex_facet_incidence"
_GEOMETRY_CHECKS = (
    "shape",
    "finite",
    "unit",
    "distinct",
    "full_dimensional",
    "all_vertices_extreme",
    "simplicial",
    "strict_support",
)


class CheckerInputError(ValueError):
    """The response or public target contract is malformed."""


@dataclass(frozen=True)
class TargetContract:
    """Validated, canonical public benchmark contract."""

    instance_id: str
    n_vertices: int
    facets: tuple[tuple[int, int, int, int], ...]
    minimum_volume: float
    unit_tolerance: float
    distinct_tolerance: float
    support_tolerance: float
    volume_spread_tolerance: float


def _reject_json_constant(value: str) -> None:
    raise CheckerInputError(f"non-standard JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckerInputError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def load_json_strict_with_sha256(
    path: str | Path,
    *,
    document_name: str = "JSON document",
) -> tuple[dict[str, Any], str]:
    """Read, parse, and hash exactly one byte snapshot of a strict JSON object."""

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
        raise CheckerInputError(f"cannot load {document_name}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckerInputError(f"{document_name} root must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def load_json_strict(
    path: str | Path,
    *,
    document_name: str = "JSON document",
) -> dict[str, Any]:
    """Load one RFC-compatible JSON object without duplicate keys or NaN."""

    return load_json_strict_with_sha256(path, document_name=document_name)[0]


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


def _integer_at_least(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CheckerInputError(f"{name} must be an integer at least {minimum}")
    return value


def _canonical_facets(
    value: Any,
    *,
    n_vertices: int,
    name: str,
    strict_contract: bool,
) -> tuple[tuple[int, int, int, int], ...] | None:
    """Validate and canonicalize a list of unordered tetrahedral facets.

    Invalid public contracts raise.  Invalid audit payloads return ``None`` so
    an untrusted or partially failed numerical audit becomes a failed gate,
    not a checker crash or a malformed-contract classification.
    """

    def fail(message: str) -> None:
        if strict_contract:
            raise CheckerInputError(message)

    if not isinstance(value, list) or not value:
        fail(f"{name} must be a nonempty list of tetrahedral facets")
        return None
    facets: list[tuple[int, int, int, int]] = []
    for facet_index, facet in enumerate(value):
        allowed_sequence = list if strict_contract else (list, tuple)
        if not isinstance(facet, allowed_sequence) or len(facet) != 4:
            fail(f"{name}[{facet_index}] must contain exactly 4 vertex labels")
            return None
        labels: list[int] = []
        for label_index, label in enumerate(facet):
            if isinstance(label, bool) or not isinstance(label, Integral):
                fail(f"{name}[{facet_index}][{label_index}] must be an integer label")
                return None
            integer = int(label)
            if not 0 <= integer < n_vertices:
                fail(
                    f"{name}[{facet_index}][{label_index}] is outside [0, {n_vertices})"
                )
                return None
            labels.append(integer)
        if len(set(labels)) != 4:
            fail(f"{name}[{facet_index}] must contain four distinct labels")
            return None
        facets.append(tuple(sorted(labels)))
    if len(set(facets)) != len(facets):
        fail(f"{name} must not contain duplicate unordered facets")
        return None
    canonical = tuple(sorted(facets))
    if strict_contract:
        serialized = tuple(tuple(int(label) for label in facet) for facet in value)
        if serialized != canonical:
            raise CheckerInputError(
                f"{name} rows must be strictly increasing and the outer list must "
                "be lexicographically sorted"
            )
        used = {vertex for facet in canonical for vertex in facet}
        if used != set(range(n_vertices)):
            raise CheckerInputError(
                f"{name} must use every label from 0 through {n_vertices - 1}"
            )
        ridge_occurrences: dict[tuple[int, int, int], list[int]] = {}
        for facet_index, facet in enumerate(canonical):
            for omitted in range(4):
                ridge = facet[:omitted] + facet[omitted + 1 :]
                ridge_occurrences.setdefault(ridge, []).append(facet_index)
        if any(len(incidents) != 2 for incidents in ridge_occurrences.values()):
            raise CheckerInputError(
                f"{name} must describe a closed tetrahedral pseudomanifold: "
                "every triangular ridge must occur exactly twice"
            )
        dual_adjacency = [set() for _ in canonical]
        for left, right in ridge_occurrences.values():
            dual_adjacency[left].add(right)
            dual_adjacency[right].add(left)
        reached = {0}
        pending = [0]
        while pending:
            current = pending.pop()
            for neighbor in dual_adjacency[current] - reached:
                reached.add(neighbor)
                pending.append(neighbor)
        if len(reached) != len(canonical):
            raise CheckerInputError(f"{name} facet-dual graph must be connected")
    return canonical


def parse_target_document(document: Mapping[str, Any]) -> TargetContract:
    """Validate a strict public target and return its canonical contract."""

    if not isinstance(document, Mapping):
        raise CheckerInputError("target root must be a JSON object")
    if set(document) != _TARGET_KEYS:
        raise CheckerInputError(
            "target must contain exactly schema, instance_id, dimension, "
            "n_vertices, facets, facet_equivalence, minimum_volume, and tolerances"
        )
    if document["schema"] != TARGET_SCHEMA_VERSION:
        raise CheckerInputError(
            f"target schema must be exactly {TARGET_SCHEMA_VERSION!r}"
        )
    instance_id = document["instance_id"]
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise CheckerInputError("target instance_id must be a nonempty string")
    if document["facet_equivalence"] != _FACET_EQUIVALENCE:
        raise CheckerInputError(
            f"target facet_equivalence must be exactly {_FACET_EQUIVALENCE!r}"
        )
    dimension = _integer_at_least(document["dimension"], name="dimension", minimum=1)
    if dimension != 4:
        raise CheckerInputError("target dimension must be exactly 4")
    n_vertices = _integer_at_least(document["n_vertices"], name="n_vertices", minimum=5)
    facets = _canonical_facets(
        document["facets"],
        n_vertices=n_vertices,
        name="facets",
        strict_contract=True,
    )
    assert facets is not None
    minimum_volume = _nonnegative_real(
        document["minimum_volume"], name="minimum_volume"
    )
    tolerances = document["tolerances"]
    if not isinstance(tolerances, Mapping) or set(tolerances) != _TOLERANCE_KEYS:
        raise CheckerInputError(
            "tolerances must contain exactly unit, distinct, support, and "
            "pole_volume_spread"
        )
    return TargetContract(
        instance_id=instance_id,
        n_vertices=n_vertices,
        facets=facets,
        minimum_volume=minimum_volume,
        unit_tolerance=_positive_real(tolerances["unit"], name="unit"),
        distinct_tolerance=_positive_real(tolerances["distinct"], name="distinct"),
        support_tolerance=_positive_real(tolerances["support"], name="support"),
        volume_spread_tolerance=_nonnegative_real(
            tolerances["pole_volume_spread"],
            name="pole_volume_spread",
        ),
    )


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


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return None
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _pole_measurements(
    pole_audit: Mapping[str, Any],
    *,
    n_vertices: int,
    facets: tuple[tuple[int, int, int, int], ...] | None,
) -> dict[str, Any]:
    """Independently validate pole rows and recompute their volume summary."""

    values = pole_audit.get("values")
    if not isinstance(values, list):
        values = []
    volumes: list[float] = []
    poles: list[int] = []
    simplex_counts_valid = facets is not None
    rows_valid = len(values) == n_vertices
    for entry in values:
        if not isinstance(entry, Mapping):
            rows_valid = False
            simplex_counts_valid = False
            continue
        pole = entry.get("pole_index")
        volume = _finite_number(entry.get("volume"))
        simplex_count = entry.get("simplex_count")
        if (
            isinstance(pole, bool)
            or not isinstance(pole, Integral)
            or not 0 <= int(pole) < n_vertices
            or volume is None
        ):
            rows_valid = False
            simplex_counts_valid = False
            continue
        pole_integer = int(pole)
        poles.append(pole_integer)
        volumes.append(volume)
        if (
            facets is None
            or isinstance(simplex_count, bool)
            or not isinstance(simplex_count, Integral)
            or int(simplex_count) != sum(pole_integer not in facet for facet in facets)
        ):
            simplex_counts_valid = False
    complete_poles = rows_valid and sorted(poles) == list(range(n_vertices))
    if not complete_poles:
        simplex_counts_valid = False
    minimum = min(volumes) if complete_poles else None
    maximum = max(volumes) if complete_poles else None
    spread = maximum - minimum if minimum is not None and maximum is not None else None
    return {
        "complete_poles": complete_poles,
        "simplex_counts_valid": simplex_counts_valid,
        "minimum_volume": minimum,
        "maximum_volume": maximum,
        "spread": spread,
    }


def _summary_matches(
    pole_audit: Mapping[str, Any], measurements: Mapping[str, Any]
) -> bool:
    minimum = measurements.get("minimum_volume")
    maximum = measurements.get("maximum_volume")
    spread = measurements.get("spread")
    if minimum is None or maximum is None or spread is None:
        return False
    reported_minimum = _finite_number(pole_audit.get("minimum_volume"))
    reported_maximum = _finite_number(pole_audit.get("maximum_volume"))
    reported_spread = _finite_number(pole_audit.get("spread"))
    return bool(
        reported_minimum is not None
        and reported_maximum is not None
        and reported_spread is not None
        and reported_minimum == minimum
        and reported_maximum == maximum
        and reported_spread == spread
    )


def _exact_integer(value: Any, expected: int) -> bool:
    return bool(
        isinstance(value, Integral)
        and not isinstance(value, (bool, np.bool_))
        and int(value) == expected
    )


def _complete_requested_indices(value: Any, n_vertices: int) -> bool:
    if not isinstance(value, list) or len(value) != n_vertices:
        return False
    return all(_exact_integer(item, index) for index, item in enumerate(value))


def _pole_facet_consistency_gate(
    result: Mapping[str, Any],
    *,
    n_vertices: int,
    facets: tuple[tuple[int, int, int, int], ...] | None,
) -> bool:
    """Validate the full per-pole facet-set comparison, not merely its counts."""

    if facets is None:
        return False
    values = result.get("values")
    if not isinstance(values, list) or len(values) != n_vertices:
        return False
    if not (
        result.get("all_poles_requested") is True
        and _exact_integer(result.get("requested_count"), n_vertices)
        and _exact_integer(result.get("successful_count"), n_vertices)
        and _exact_integer(result.get("failure_count"), 0)
        and result.get("all_match") is True
        and result.get("failures") == []
    ):
        return False
    seen: set[int] = set()
    for entry in values:
        if not isinstance(entry, Mapping):
            return False
        pole = entry.get("pole_index")
        if (
            isinstance(pole, bool)
            or not isinstance(pole, Integral)
            or not 0 <= int(pole) < n_vertices
            or int(pole) in seen
        ):
            return False
        pole_integer = int(pole)
        seen.add(pole_integer)
        expected_count = sum(pole_integer not in facet for facet in facets)
        if not (
            _exact_integer(entry.get("expected_simplex_count"), expected_count)
            and _exact_integer(entry.get("delaunay_simplex_count"), expected_count)
            and _exact_integer(entry.get("volume_simplex_count"), expected_count)
            and entry.get("facets_match") is True
            and entry.get("simplex_counts_match") is True
            and entry.get("matches") is True
            and entry.get("missing_facets") == []
            and entry.get("extra_facets") == []
        ):
            return False
    return seen == set(range(n_vertices))


def grade_response_document(
    response: Mapping[str, Any],
    target_document: Mapping[str, Any] | TargetContract,
    *,
    include_audit: bool = False,
    audit_function: Callable[..., Mapping[str, Any]] = (
        audit_h4_combinatorics_candidate
    ),
    incidence_function: Callable[..., Mapping[str, Any]] = compare_facet_incidence,
) -> dict[str, Any]:
    """Grade a parsed response against one public combinatorics contract."""

    target = (
        target_document
        if isinstance(target_document, TargetContract)
        else parse_target_document(target_document)
    )
    points = _vertices_array(response, n_vertices=target.n_vertices)
    audit = dict(
        audit_function(
            points,
            target.facets,
            volume_floor=target.minimum_volume,
            volume_spread_tolerance=target.volume_spread_tolerance,
            unit_tolerance=target.unit_tolerance,
            distinct_tolerance=target.distinct_tolerance,
            support_tolerance=target.support_tolerance,
        )
    )

    geometry = audit.get("geometry")
    if not isinstance(geometry, Mapping):
        geometry = {}
    geometry_checks = geometry.get("checks")
    if not isinstance(geometry_checks, Mapping):
        geometry_checks = {}
    geometry_topology = geometry.get("topology")
    if not isinstance(geometry_topology, Mapping):
        geometry_topology = {}
    candidate_facets = _canonical_facets(
        geometry.get("facets"),
        n_vertices=target.n_vertices,
        name="audit.geometry.facets",
        strict_contract=False,
    )
    if candidate_facets is None:
        independent_incidence: Mapping[str, Any] = {
            "available": False,
            "isomorphic": False,
            "failure_reason": "candidate audit did not provide valid facets",
        }
    else:
        independent_incidence = dict(
            incidence_function(target.n_vertices, candidate_facets, target.facets)
        )
    pole_audit = audit.get("pole_volume_audit")
    if not isinstance(pole_audit, Mapping):
        pole_audit = {}
    pole_facet_consistency = audit.get("pole_facet_consistency")
    if not isinstance(pole_facet_consistency, Mapping):
        pole_facet_consistency = {}
    pole_measurements = _pole_measurements(
        pole_audit,
        n_vertices=target.n_vertices,
        facets=candidate_facets,
    )
    minimum_volume = pole_measurements["minimum_volume"]
    spread = pole_measurements["spread"]

    incidence_available = independent_incidence.get("available") is True
    incidence_isomorphic = independent_incidence.get("isomorphic") is True
    all_poles_verified = bool(
        pole_measurements["complete_poles"]
        and pole_audit.get("all_poles_requested") is True
        and _complete_requested_indices(
            pole_audit.get("requested_indices"), target.n_vertices
        )
        and _exact_integer(pole_audit.get("requested_count"), target.n_vertices)
        and _exact_integer(pole_audit.get("successful_count"), target.n_vertices)
        and _exact_integer(pole_audit.get("failure_count"), 0)
        and pole_audit.get("failures") == []
    )
    facet_count_matches = bool(
        candidate_facets is not None
        and len(candidate_facets) == len(target.facets)
        and _exact_integer(
            independent_incidence.get("candidate_n_facets"), len(candidate_facets)
        )
        and _exact_integer(
            independent_incidence.get("target_n_facets"), len(target.facets)
        )
    )
    pole_facets_match = _pole_facet_consistency_gate(
        pole_facet_consistency,
        n_vertices=target.n_vertices,
        facets=candidate_facets,
    )
    core_checks = audit.get("checks")
    if not isinstance(core_checks, Mapping):
        core_checks = {}
    core_audit_consistent = bool(
        audit.get("accepted") is True
        and audit.get("classification") == "accepted"
        and core_checks.get("geometry") is True
        and core_checks.get("target_incidence_input") is True
        and core_checks.get("incidence_backend") is True
        and core_checks.get("combinatorics") is True
        and core_checks.get("pole_facet_consistency") is True
        and core_checks.get("all_pole_volume") is True
        and core_checks.get("volume_floor") is True
    )
    gates = {
        "dimension_and_count": audit.get("dimension") == 4
        and audit.get("n_vertices") == target.n_vertices,
        "strict_ideal_geometry": geometry.get("valid_geometry") is True
        and all(geometry_checks.get(name) is True for name in _GEOMETRY_CHECKS),
        "incidence_backend_available": incidence_available,
        "facet_count_matches": facet_count_matches,
        "combinatorial_type_matches": incidence_available and incidence_isomorphic,
        "all_poles_verified": all_poles_verified,
        "pole_facet_simplex_counts": all_poles_verified
        and pole_measurements["simplex_counts_valid"] is True,
        "pole_facet_consistency": pole_facets_match,
        "pole_volume_stable": all_poles_verified
        and pole_audit.get("within_tolerance") is True
        and _summary_matches(pole_audit, pole_measurements)
        and isinstance(spread, float)
        and spread <= target.volume_spread_tolerance,
        "minimum_volume": isinstance(minimum_volume, float)
        and minimum_volume >= target.minimum_volume,
        "core_audit_consistent": core_audit_consistent,
    }
    strict_pass = all(gates.values())
    classification = (
        "strict_h4_combinatorics_pass"
        if strict_pass
        else (
            "incidence_backend_unavailable"
            if audit.get("classification") == "inconclusive_incidence_backend"
            else "failed_h4_combinatorics_audit"
        )
    )
    grade: dict[str, Any] = {
        "schema": GRADE_SCHEMA_VERSION,
        "strict_pass": strict_pass,
        "classification": classification,
        "contract": {
            "schema": TARGET_SCHEMA_VERSION,
            "instance_id": target.instance_id,
            "dimension": 4,
            "n_vertices": target.n_vertices,
            "facet_count": len(target.facets),
            "facet_equivalence": _FACET_EQUIVALENCE,
            "minimum_volume": target.minimum_volume,
            "tolerances": {
                "unit": target.unit_tolerance,
                "distinct": target.distinct_tolerance,
                "support": target.support_tolerance,
                "pole_volume_spread": target.volume_spread_tolerance,
            },
        },
        "gates": gates,
        "measurements": {
            "minimum_volume": minimum_volume,
            "maximum_volume": pole_measurements["maximum_volume"],
            "pole_volume_spread": spread,
            "candidate_facet_count": (
                len(candidate_facets) if candidate_facets is not None else None
            ),
            "target_facet_count": len(target.facets),
            "incidence_backend": independent_incidence.get("backend"),
            "candidate_incidence_certificate_sha256": independent_incidence.get(
                "candidate_certificate_sha256"
            ),
            "target_incidence_certificate_sha256": independent_incidence.get(
                "target_certificate_sha256"
            ),
            "minimum_support_margin": geometry_topology.get("min_support_margin"),
            "minimum_pairwise_distance": geometry_topology.get("min_pairwise_distance"),
        },
    }
    if include_audit:
        grade["audit"] = audit
        grade["independent_incidence"] = dict(independent_incidence)
    return _jsonable(grade)


def grade_response(
    response_path: str | Path,
    target_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load one response and target snapshot, then grade those exact bytes."""

    response, response_sha256 = load_json_strict_with_sha256(
        response_path, document_name="response"
    )
    target_document, target_sha256 = load_json_strict_with_sha256(
        target_path, document_name="target"
    )
    target = parse_target_document(target_document)
    grade = grade_response_document(response, target, **kwargs)
    grade["response_sha256"] = response_sha256
    grade["contract_sha256"] = target_sha256
    return grade


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response", type=Path)
    parser.add_argument("--target", type=Path, required=True)
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
        "schema": GRADE_SCHEMA_VERSION,
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
            args.target,
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
