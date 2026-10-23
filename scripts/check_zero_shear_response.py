#!/usr/bin/env python3
"""Strictly grade a labeled zero-shear realization response.

The private answer key supplies the coherently oriented target faces and the
expected existence bit.  All geometric predicates are delegated to the
package-owned :mod:`ideal_poly_volume_toolkit.rivin_zero_shear` audit.  A
private reference realization, when present, is used only for a label-aligned
chordal-error diagnostic unless ``--reference-tolerance`` explicitly makes it
an additional grading gate.

This remains a double-precision numerical verifier.  It is not an interval or
exact certificate for the submitted coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ideal_poly_volume_toolkit.rivin_holonomy import (
    validate_closed_sphere_triangulation,
)
from ideal_poly_volume_toolkit.rivin_zero_shear import (
    audit_zero_shear_realization,
)


SCHEMA_VERSION = "strict-zero-shear-response-grade/1"
DEFAULT_SHEAR_TOLERANCE = 1e-8
DEFAULT_SUPPORT_TOLERANCE = 0.0
DISTINGUISHED_FACE = (0, 1, 2)


class CheckerInputError(ValueError):
    """A private key or model response violates the checker contract."""


def _reject_json_constant(value: str) -> None:
    raise CheckerInputError(f"non-standard JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckerInputError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def load_json_strict(path: str | Path, *, label: str) -> dict[str, Any]:
    """Load one RFC-compatible JSON object without duplicate keys or NaN."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_object,
            )
    except CheckerInputError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckerInputError(f"cannot load {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckerInputError(f"{label} root must be a JSON object")
    return value


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    raise TypeError(f"grade value is not JSON serializable: {type(value).__name__}")


def strict_json_bytes(value: Any) -> bytes:
    payload = _jsonable(value)
    text = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_temporary_json(path: Path, value: Any) -> Path:
    """Write and flush one temporary strict-JSON file beside ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(strict_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        return temporary_path
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def exclusive_create_json(path: str | Path, value: Any) -> None:
    """Atomically create ``path`` without replacing an existing grade."""
    destination = Path(path)
    temporary_path = _write_temporary_json(destination, value)
    try:
        try:
            os.link(temporary_path, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to clobber existing grade {destination}; use --overwrite"
            ) from exc
        _fsync_directory(destination.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Atomically replace ``path`` with a flushed strict-JSON grade."""
    destination = Path(path)
    temporary_path = _write_temporary_json(destination, value)
    try:
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def _parse_faces(value: Any) -> tuple[tuple[int, int, int], ...]:
    if not isinstance(value, list) or not value:
        raise CheckerInputError("ground_truth.oriented_faces must be a nonempty list")
    faces: list[tuple[int, int, int]] = []
    for face_index, raw_face in enumerate(value):
        if not isinstance(raw_face, list) or len(raw_face) != 3:
            raise CheckerInputError(
                f"ground_truth.oriented_faces[{face_index}] must have length 3"
            )
        if any(
            not isinstance(vertex, int) or isinstance(vertex, bool)
            for vertex in raw_face
        ):
            raise CheckerInputError(
                f"ground_truth.oriented_faces[{face_index}] labels must be integers"
            )
        faces.append(tuple(raw_face))

    try:
        topology = validate_closed_sphere_triangulation(faces)
    except (TypeError, ValueError) as exc:
        raise CheckerInputError(f"invalid target triangulation: {exc}") from exc
    expected_labels = tuple(range(topology.n_vertices))
    if topology.vertex_labels != expected_labels:
        raise CheckerInputError(
            "target vertex labels must be exactly 0 through N-1 without gaps"
        )

    directed_edges: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for source, target in ((a, b), (b, c), (c, a)):
            directed_edges[(source, target)] = (
                directed_edges.get((source, target), 0) + 1
            )
    incoherent_edge = next(
        (
            tuple(sorted((source, target)))
            for source, target in directed_edges
            if directed_edges[(source, target)] != 1
            or directed_edges.get((target, source), 0) != 1
        ),
        None,
    )
    if incoherent_edge is not None:
        raise CheckerInputError(
            f"target faces are not coherently oriented across edge {incoherent_edge}"
        )

    distinguished_cycles = {
        DISTINGUISHED_FACE,
        (DISTINGUISHED_FACE[1], DISTINGUISHED_FACE[2], DISTINGUISHED_FACE[0]),
        (DISTINGUISHED_FACE[2], DISTINGUISHED_FACE[0], DISTINGUISHED_FACE[1]),
    }
    if sum(face in distinguished_cycles for face in faces) != 1:
        raise CheckerInputError(
            "target must contain the distinguished oriented face (0,1,2) exactly once"
        )
    return tuple(faces)


def _parse_coordinate_row(value: Any, *, path: str) -> complex:
    if not isinstance(value, list) or len(value) != 2:
        raise CheckerInputError(f"{path} must be a two-number JSON array")
    coordinates: list[float] = []
    for component_index, component in enumerate(value):
        if not isinstance(component, (int, float)) or isinstance(component, bool):
            raise CheckerInputError(
                f"{path}[{component_index}] must be an ordinary JSON number"
            )
        number = float(component)
        if not math.isfinite(number):
            raise CheckerInputError(f"{path}[{component_index}] must be finite")
        coordinates.append(number)
    return complex(coordinates[0], coordinates[1])


def _parse_vertices(
    value: Any,
    *,
    n_vertices: int,
    path: str,
) -> tuple[complex | None, ...]:
    if not isinstance(value, list):
        raise CheckerInputError(f"{path} must be a JSON array")
    if len(value) != n_vertices:
        raise CheckerInputError(
            f"{path} must contain exactly {n_vertices} labeled vertices; got {len(value)}"
        )
    if value[0] is not None:
        raise CheckerInputError(f"{path}[0] must be null (the point at infinity)")
    vertices: list[complex | None] = [None]
    for index, row in enumerate(value[1:], start=1):
        if row is None:
            raise CheckerInputError(f"{path}[{index}] must be finite, not null")
        vertices.append(_parse_coordinate_row(row, path=f"{path}[{index}]"))
    return tuple(vertices)


def _parse_answer_key(value: dict[str, Any]) -> dict[str, Any]:
    ground_truth = value.get("ground_truth")
    if not isinstance(ground_truth, dict):
        raise CheckerInputError("answer key must contain a ground_truth object")
    normalization = value.get("normalization")
    if not isinstance(normalization, dict):
        raise CheckerInputError("answer key must contain a normalization object")
    exists = ground_truth.get("exists")
    if not isinstance(exists, bool):
        raise CheckerInputError("ground_truth.exists must be a JSON boolean")
    faces = _parse_faces(ground_truth.get("oriented_faces"))
    n_vertices = max(max(face) for face in faces) + 1

    reference = None
    if "vertices" in ground_truth:
        reference = _parse_vertices(
            ground_truth["vertices"],
            n_vertices=n_vertices,
            path="ground_truth.vertices",
        )
    if exists and reference is None:
        # A reference is not mathematically needed for independent grading,
        # but the current private-key protocol includes one for diagnostics.
        reference_available = False
    else:
        reference_available = reference is not None

    return {
        "exists": exists,
        "faces": faces,
        "n_vertices": n_vertices,
        "reference": reference,
        "reference_available": reference_available,
        "normalization": normalization,
    }


def _parse_response(value: dict[str, Any], *, n_vertices: int) -> dict[str, Any]:
    exists = value.get("exists")
    if not isinstance(exists, bool):
        raise CheckerInputError("response.exists must be a JSON boolean")
    expected_keys = {"exists", "vertices"} if exists else {"exists"}
    if set(value) != expected_keys:
        raise CheckerInputError(
            "response must contain exactly " + ", ".join(sorted(expected_keys))
        )
    vertices = None
    if exists:
        vertices = _parse_vertices(
            value["vertices"],
            n_vertices=n_vertices,
            path="response.vertices",
        )
    return {"exists": exists, "vertices": vertices}


def _sphere_points(vertices: Sequence[complex | None]) -> np.ndarray:
    points = np.empty((len(vertices), 3), dtype=np.float64)
    for index, coordinate in enumerate(vertices):
        if coordinate is None:
            points[index] = (0.0, 0.0, 1.0)
            continue
        real = coordinate.real
        imaginary = coordinate.imag
        scale = max(abs(real), abs(imaginary))
        if scale <= 1.0:
            modulus = math.hypot(real, imaginary)
            modulus_squared = modulus * modulus
            denominator = 1.0 + modulus_squared
            points[index] = (
                2.0 * real / denominator,
                2.0 * imaginary / denominator,
                (modulus_squared - 1.0) / denominator,
            )
        else:
            scaled_real = real / scale
            scaled_imaginary = imaginary / scale
            scaled_modulus = math.hypot(scaled_real, scaled_imaginary)
            inverse_modulus = (1.0 / scale) / scaled_modulus
            inverse_squared = inverse_modulus * inverse_modulus
            denominator = 1.0 + inverse_squared
            points[index] = (
                2.0 * (scaled_real / scaled_modulus) * inverse_modulus / denominator,
                2.0
                * (scaled_imaginary / scaled_modulus)
                * inverse_modulus
                / denominator,
                (1.0 - inverse_squared) / denominator,
            )
    return points


def _private_reference_error(
    submitted: Sequence[complex | None],
    reference: Sequence[complex | None],
    *,
    tolerance: float | None,
) -> dict[str, Any]:
    submitted_points = _sphere_points(submitted)
    reference_points = _sphere_points(reference)
    errors = np.linalg.norm(submitted_points - reference_points, axis=1)
    worst_vertex = int(np.argmax(errors))
    within_tolerance = (
        None if tolerance is None else bool(float(np.max(errors)) <= tolerance)
    )
    return {
        "available": True,
        "metric": "euclidean-chord-on-unit-sphere-label-aligned",
        "n_vertices": len(errors),
        "max_chordal": float(np.max(errors)),
        "median_chordal": float(np.median(errors)),
        "rms_chordal": float(np.sqrt(np.mean(errors * errors))),
        "worst_vertex": worst_vertex,
        "tolerance": tolerance,
        "within_tolerance": within_tolerance,
        "is_acceptance_gate": tolerance is not None,
    }


def grade_response(
    answer_key_path: str | Path,
    response_path: str | Path,
    *,
    shear_tolerance: float = DEFAULT_SHEAR_TOLERANCE,
    support_tolerance: float = DEFAULT_SUPPORT_TOLERANCE,
    reference_tolerance: float | None = None,
) -> dict[str, Any]:
    """Grade one strict response and return a JSON-serializable report."""
    key_raw = load_json_strict(answer_key_path, label="answer key")
    response_raw = load_json_strict(response_path, label="model response")
    key = _parse_answer_key(key_raw)
    response = _parse_response(response_raw, n_vertices=key["n_vertices"])

    existence_matches = response["exists"] == key["exists"]
    audit_dict = None
    audit_pass = False
    reference_report: dict[str, Any] = {
        "available": False,
        "is_acceptance_gate": reference_tolerance is not None,
        "tolerance": reference_tolerance,
    }

    if response["exists"]:
        audit = audit_zero_shear_realization(
            key["faces"],
            response["vertices"],
            distinguished_face=DISTINGUISHED_FACE,
            shear_tolerance=shear_tolerance,
            support_tolerance=support_tolerance,
            require_standard_normalization=True,
        )
        audit_dict = audit.as_dict()
        audit_pass = bool(audit.strict_pass)
        if key["reference"] is not None:
            reference_report = _private_reference_error(
                response["vertices"],
                key["reference"],
                tolerance=reference_tolerance,
            )
        elif reference_tolerance is not None:
            raise CheckerInputError(
                "--reference-tolerance requires ground_truth.vertices in the answer key"
            )

    reference_pass = (
        reference_tolerance is None or reference_report.get("within_tolerance") is True
    )
    if response["exists"]:
        strict_pass = bool(existence_matches and audit_pass and reference_pass)
        if not existence_matches:
            classification = "incorrect_existence_claim"
        elif not audit_pass:
            classification = "failed_strict_realization_audit"
        elif not reference_pass:
            classification = "private_reference_mismatch"
        else:
            classification = "strict_realization_pass"
    else:
        strict_pass = bool(existence_matches)
        classification = (
            "private_key_nonexistence_match"
            if strict_pass
            else "incorrect_existence_claim"
        )

    topology = validate_closed_sphere_triangulation(key["faces"])
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
        "strict_pass": strict_pass,
        "verification_scope": "double-precision-numerical-not-formal",
        "inputs": {
            "answer_key_sha256": _sha256_file(answer_key_path),
            "response_sha256": _sha256_file(response_path),
        },
        "thresholds": {
            "shear_tolerance": float(shear_tolerance),
            "support_tolerance": float(support_tolerance),
            "reference_tolerance": reference_tolerance,
        },
        "existence": {
            "expected": key["exists"],
            "reported": response["exists"],
            "matches": existence_matches,
            "basis_when_false": "private-answer-key-classification",
        },
        "target": {
            "n_vertices": topology.n_vertices,
            "n_edges": topology.n_edges,
            "n_faces": topology.n_faces,
            "distinguished_face": list(DISTINGUISHED_FACE),
            "labels_exact_range": True,
        },
        "audit": audit_dict,
        "private_reference": reference_report,
    }


def _nonnegative_finite(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answer_key", type=Path, help="private answer-key JSON")
    parser.add_argument("response", type=Path, help="strict model-response JSON")
    parser.add_argument(
        "--output",
        type=Path,
        help="atomically write the grade here instead of stdout",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow --output to atomically replace an existing grade",
    )
    parser.add_argument(
        "--shear-tolerance",
        type=_nonnegative_finite,
        default=DEFAULT_SHEAR_TOLERANCE,
    )
    parser.add_argument(
        "--support-tolerance",
        type=_nonnegative_finite,
        default=DEFAULT_SUPPORT_TOLERANCE,
    )
    parser.add_argument(
        "--reference-tolerance",
        type=_nonnegative_finite,
        default=None,
        help="optional private-reference gate; omitted means diagnostic only",
    )
    return parser


def _invalid_grade(exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "invalid_input",
        "strict_pass": False,
        "verification_scope": "double-precision-numerical-not-formal",
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.overwrite and args.output is None:
        parser.error("--overwrite requires --output")
    if args.output is not None and args.output.exists() and not args.overwrite:
        parser.exit(
            2,
            f"error: refusing to clobber existing grade {args.output}; "
            "use --overwrite\n",
        )
    try:
        grade = grade_response(
            args.answer_key,
            args.response,
            shear_tolerance=args.shear_tolerance,
            support_tolerance=args.support_tolerance,
            reference_tolerance=args.reference_tolerance,
        )
    except Exception as exc:  # Convert all data/audit failures to strict JSON.
        grade = _invalid_grade(exc)

    if args.output is None:
        sys.stdout.buffer.write(strict_json_bytes(grade))
        sys.stdout.buffer.flush()
    else:
        try:
            if args.overwrite:
                atomic_write_json(args.output, grade)
            else:
                exclusive_create_json(args.output, grade)
        except FileExistsError as exc:
            parser.exit(2, f"error: {exc}\n")
    return 0 if grade.get("strict_pass") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
