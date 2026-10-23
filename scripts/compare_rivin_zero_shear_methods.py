#!/usr/bin/env python3
"""Compare three fixed-combinatorics Rivin zero-shear workflows.

The advertised methods are:

``angle_optimize_realize``
    Remove the distinguished pole and its entire incident star, optimize the
    Rivin angle structure on the disk bounded by the pole's complete cyclic
    link, and realize the optimized angles geometrically.

``pole_link_tutte_direct``
    Remove the distinguished pole, place its complete cyclic link on a regular
    polygon, harmonically embed the remaining disk, normalize the ordered
    anchors, and run the coordinate zero-shear solver.

``angle_then_polish``
    Feed the angle-realization endpoint into the same coordinate zero-shear
    solver used by the direct method.

An outer-face-triangle Tutte start is available only as an explicitly enabled
secondary method.  Every endpoint is audited afresh by
``audit_zero_shear_realization``.  A direct solver endpoint is never described
as a fixed-combinatorics maximum unless that independent strict convex audit
passes.  These are double-precision numerical comparisons, not formal proofs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any


# This first implementation is deliberately single-process.  Prevent native
# numerical libraries from introducing an unrecorded second layer of fan-out.
BLAS_THREADS = 1
THREAD_ENVIRONMENT = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _thread_variable in THREAD_ENVIRONMENT:
    os.environ[_thread_variable] = str(BLAS_THREADS)

import numpy as np  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


SCHEMA_VERSION = "ideal-poly-rivin-zero-shear-comparison/1"
PROTOCOL_VERSION = "fixed-combinatorics-three-method-comparison/1"
GENERATOR_NAME = "scripts/compare_rivin_zero_shear_methods.py"
GENERATOR_VERSION = "1.0.0"
SCHEMA_PATH = (
    REPOSITORY_ROOT / "docs" / "schemas" / "rivin-zero-shear-comparison-v1.schema.json"
)
ARITHMETIC_SCHEMA_PATH = (
    REPOSITORY_ROOT / "docs" / "schemas" / "arithmetic-corpus-v1.schema.json"
)
ZERO_SHEAR_CORE_PATH = (
    REPOSITORY_ROOT / "ideal_poly_volume_toolkit" / "rivin_zero_shear.py"
)
RIVIN_CORE_PATH = REPOSITORY_ROOT / "ideal_poly_volume_toolkit" / "rivin_delaunay.py"

METHOD_ANGLE = "angle_optimize_realize"
METHOD_POLE = "pole_link_tutte_direct"
METHOD_HYBRID = "angle_then_polish"
METHOD_FACE_TUTTE = "outer_face_tutte_direct"
REQUIRED_METHODS = (METHOD_ANGLE, METHOD_POLE, METHOD_HYBRID)

DEFAULT_MAX_NFEV = 500
DEFAULT_LM_POLISH_STEPS = 3
DEFAULT_SHEAR_TOLERANCE = 1e-8
DEFAULT_SUPPORT_TOLERANCE = 0.0
DEFAULT_BENDING_TOLERANCE = 1e-10
DEFAULT_SOLVER_TOLERANCE = 1e-13


@dataclass(frozen=True)
class ComparisonCase:
    """One validated labeled sphere and its selected normalization face."""

    case_id: str
    faces: tuple[tuple[int, int, int], ...]
    distinguished_face: tuple[int, int, int]
    reference_vertices: tuple[complex | None, ...] | None
    source_pointer: str
    source_kind: str


class StageFailure(RuntimeError):
    """A numerical method stopped at an expected, reportable stage."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert numerical records to finite strict-JSON-compatible values."""
    if is_dataclass(value):
        value = asdict(value)
    elif hasattr(value, "as_dict") and callable(value.as_dict):
        value = value.as_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("non-finite floating-point value is forbidden in JSON")
        return result
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("non-finite complex value is forbidden in JSON")
        return [float(value.real), float(value.imag)]
    if value is None or isinstance(value, (str, int)):
        return value
    raise TypeError(f"cannot serialize {type(value).__name__} to strict JSON")


def strict_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            _jsonable(payload),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r} is forbidden")
        result[key] = value
    return result


def load_json_value_strict(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(
            stream,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )


def load_json_strict(path: str | Path) -> dict[str, Any]:
    value = load_json_value_strict(path)
    if not isinstance(value, dict):
        raise ValueError("input JSON root must be an object")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_temporary_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(strict_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def write_json_atomic(path: str | Path, payload: Any, *, overwrite: bool) -> None:
    """Create a strict JSON output atomically, refusing clobber by default."""
    target = Path(path)
    temporary = _write_temporary_json(target, payload)
    try:
        if overwrite:
            os.replace(temporary, target)
        else:
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"refusing to clobber existing output {target}; use --overwrite"
                ) from exc
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_with_schema(document: Mapping[str, Any], schema_path: Path) -> None:
    try:
        from jsonschema.validators import validator_for
    except ImportError as exc:  # pragma: no cover - dev/test environments include it
        raise RuntimeError(
            "comparison schema validation requires jsonschema; install the dev extra"
        ) from exc
    schema = load_json_strict(schema_path)
    validator_class = validator_for(schema)
    validator_class.check_schema(schema)
    validator_class(schema).validate(document)


def _parse_faces(value: Any, *, path: str) -> tuple[tuple[int, int, int], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a nonempty array")
    faces: list[tuple[int, int, int]] = []
    for index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"{path}[{index}] must contain three labels")
        if any(not isinstance(label, int) or isinstance(label, bool) for label in row):
            raise ValueError(f"{path}[{index}] labels must be integers")
        face = tuple(int(label) for label in row)
        if any(label < 0 for label in face) or len(set(face)) != 3:
            raise ValueError(f"{path}[{index}] labels must be distinct and nonnegative")
        faces.append(face)
    if len({frozenset(face) for face in faces}) != len(faces):
        raise ValueError(f"{path} contains a duplicate unoriented face")

    from ideal_poly_volume_toolkit.rivin_holonomy import (
        validate_closed_sphere_triangulation,
    )

    topology = validate_closed_sphere_triangulation(faces)
    if topology.vertex_labels != tuple(range(topology.n_vertices)):
        raise ValueError("face labels must be exactly 0 through V-1 without gaps")
    return tuple(faces)


def _parse_vertices(
    value: Any, *, n_vertices: int, path: str
) -> tuple[complex | None, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != n_vertices:
        raise ValueError(f"{path} must contain exactly {n_vertices} vertices")
    result: list[complex | None] = []
    for index, row in enumerate(value):
        if row is None:
            result.append(None)
            continue
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"{path}[{index}] must be null or a two-number array")
        coordinates: list[float] = []
        for component_index, component in enumerate(row):
            if not isinstance(component, (int, float)) or isinstance(component, bool):
                raise ValueError(f"{path}[{index}][{component_index}] must be a number")
            number = float(component)
            if not math.isfinite(number):
                raise ValueError(f"{path}[{index}] must be finite")
            coordinates.append(number)
        result.append(complex(*coordinates))
    if sum(vertex is None for vertex in result) > 1:
        raise ValueError(f"{path} may contain at most one point at infinity")
    return tuple(result)


def _parse_face(value: Any, *, path: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(not isinstance(label, int) or isinstance(label, bool) for label in value)
    ):
        raise ValueError(f"{path} must contain three integer labels")
    face = tuple(int(label) for label in value)
    if any(label < 0 for label in face) or len(set(face)) != 3:
        raise ValueError(f"{path} labels must be distinct and nonnegative")
    return face


def _default_distinguished_face(
    faces: Sequence[tuple[int, int, int]], embedded: Any = None
) -> tuple[int, int, int]:
    if embedded is not None:
        requested = _parse_face(embedded, path="distinguished_face")
        if frozenset(requested) not in {frozenset(face) for face in faces}:
            raise ValueError("distinguished_face is absent from the face list")
        return requested
    if frozenset((0, 1, 2)) in {frozenset(face) for face in faces}:
        return (0, 1, 2)
    return tuple(faces[0])


def _case_from_payload(
    *,
    case_id: str,
    faces_value: Any,
    vertices_value: Any,
    distinguished_value: Any,
    source_pointer: str,
    source_kind: str,
    distinguished_override: tuple[int, int, int] | None,
) -> ComparisonCase:
    faces = _parse_faces(faces_value, path=f"{source_pointer}.faces")
    n_vertices = max(max(face) for face in faces) + 1
    distinguished = _default_distinguished_face(
        faces,
        distinguished_override
        if distinguished_override is not None
        else distinguished_value,
    )
    if frozenset(distinguished) not in {frozenset(face) for face in faces}:
        raise ValueError(f"case {case_id!r} distinguished face is absent")
    vertices = _parse_vertices(
        vertices_value,
        n_vertices=n_vertices,
        path=f"{source_pointer}.vertices",
    )
    return ComparisonCase(
        case_id=case_id,
        faces=faces,
        distinguished_face=distinguished,
        reference_vertices=vertices,
        source_pointer=source_pointer,
        source_kind=source_kind,
    )


def load_comparison_cases(
    path: str | Path,
    *,
    requested_ids: Sequence[str] = (),
    distinguished_override: tuple[int, int, int] | None = None,
) -> tuple[str, str | None, list[ComparisonCase]]:
    """Load an arithmetic corpus, checker key, or compact faces document."""
    source = Path(path)
    document = load_json_value_strict(source)
    if isinstance(document, list):
        case = _case_from_payload(
            case_id=source.stem,
            faces_value=document,
            vertices_value=None,
            distinguished_value=None,
            source_pointer="/",
            source_kind="simple_face_list",
            distinguished_override=distinguished_override,
        )
        if requested_ids and list(requested_ids) != [case.case_id]:
            raise ValueError(
                "--config-id on a plain face list must match the input stem"
            )
        return "simple_face_list", None, [case]
    if not isinstance(document, dict):
        raise ValueError("input JSON must be an object or a plain face-list array")
    schema_version = document.get("schema_version")
    cases: list[ComparisonCase] = []
    input_kind: str

    if schema_version == "ideal-poly-arithmetic-corpus/1":
        _validate_with_schema(document, ARITHMETIC_SCHEMA_PATH)
        input_kind = "arithmetic_corpus"
        configurations = document.get("configurations")
        if not isinstance(configurations, list) or not configurations:
            raise ValueError("arithmetic corpus contains no configurations")
        by_id = {entry.get("id"): entry for entry in configurations}
        if any(not isinstance(identifier, str) for identifier in by_id):
            raise ValueError("arithmetic corpus configuration ids must be strings")
        if len(by_id) != len(configurations):
            raise ValueError("arithmetic corpus configuration ids must be unique")
        if requested_ids:
            if len(set(requested_ids)) != len(requested_ids):
                raise ValueError("--config-id values must be unique")
            missing = sorted(set(requested_ids) - set(by_id))
            if missing:
                raise ValueError(f"unknown configuration id: {missing[0]}")
            identifiers = sorted(requested_ids)
        else:
            identifiers = sorted(by_id)
        for identifier in identifiers:
            entry = by_id[identifier]
            input_row = entry.get("input")
            if not isinstance(input_row, dict):
                raise ValueError(f"configuration {identifier!r} has no input object")
            cases.append(
                _case_from_payload(
                    case_id=identifier,
                    faces_value=input_row.get("triangles"),
                    vertices_value=input_row.get("vertices"),
                    distinguished_value=None,
                    source_pointer=f"configurations/{identifier}",
                    source_kind="arithmetic_configuration",
                    distinguished_override=distinguished_override,
                )
            )
    elif isinstance(document.get("ground_truth"), dict) and isinstance(
        document["ground_truth"].get("oriented_faces"), list
    ):
        input_kind = "strict_answer_key"
        ground_truth = document["ground_truth"]
        identifier = str(document.get("instance_id") or source.stem)
        normalization = document.get("normalization")
        embedded_face = (
            normalization.get("distinguished_face")
            if isinstance(normalization, dict)
            else None
        )
        cases.append(
            _case_from_payload(
                case_id=identifier,
                faces_value=ground_truth.get("oriented_faces"),
                vertices_value=ground_truth.get("vertices"),
                distinguished_value=embedded_face,
                source_pointer="ground_truth",
                source_kind="strict_answer_key",
                distinguished_override=distinguished_override,
            )
        )
    else:
        input_kind = "simple_faces"
        input_row = document.get("input")
        input_row = input_row if isinstance(input_row, dict) else document
        faces_value = next(
            (
                input_row[key]
                for key in ("oriented_faces", "faces", "triangles")
                if key in input_row
            ),
            None,
        )
        identifier = str(
            document.get("id") or document.get("instance_id") or source.stem
        )
        cases.append(
            _case_from_payload(
                case_id=identifier,
                faces_value=faces_value,
                vertices_value=input_row.get("vertices"),
                distinguished_value=document.get("distinguished_face"),
                source_pointer="input",
                source_kind="simple_faces",
                distinguished_override=distinguished_override,
            )
        )

    if requested_ids and input_kind != "arithmetic_corpus":
        if len(requested_ids) != 1 or requested_ids[0] != cases[0].case_id:
            raise ValueError(
                "--config-id on a single-case document must match its case id"
            )
    return (
        input_kind,
        (schema_version if isinstance(schema_version, str) else None),
        cases,
    )


def _cp1_pairs(vertices: Sequence[complex | None]) -> list[list[float] | None]:
    return [
        None if vertex is None else [float(vertex.real), float(vertex.imag)]
        for vertex in vertices
    ]


def _materialize_cp1(value: Any, *, n_vertices: int) -> tuple[complex | None, ...]:
    """Adapt a core coordinate result without silently dropping labels."""
    candidates = (
        "normalized_vertices_cp1",
        "vertices_cp1",
        "initial_vertices_cp1",
        "vertices",
        "points",
        "embedding",
    )
    for name in candidates:
        candidate = getattr(value, name, None)
        if candidate is None and isinstance(value, Mapping):
            candidate = value.get(name)
        if candidate is not None:
            value = candidate
            break
    try:
        array = np.asarray(value)
    except ValueError:
        array = np.asarray(value, dtype=object)
    if array.ndim == 2 and array.shape == (n_vertices, 2):
        if not np.all(np.isfinite(np.asarray(array, dtype=float))):
            raise ValueError("coordinate array contains a non-finite value")
        return tuple(complex(float(row[0]), float(row[1])) for row in array)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("coordinate result is not a labeled sequence")
    if len(value) != n_vertices:
        raise ValueError(
            f"coordinate result must contain {n_vertices} labels; got {len(value)}"
        )
    result: list[complex | None] = []
    for index, row in enumerate(value):
        if row is None:
            result.append(None)
        elif isinstance(row, complex):
            if not math.isfinite(row.real) or not math.isfinite(row.imag):
                raise ValueError(f"coordinate {index} is non-finite")
            result.append(complex(row))
        elif isinstance(row, Sequence) and len(row) == 2:
            point = complex(float(row[0]), float(row[1]))
            if not math.isfinite(point.real) or not math.isfinite(point.imag):
                raise ValueError(f"coordinate {index} is non-finite")
            result.append(point)
        else:
            point = complex(row)
            if not math.isfinite(point.real) or not math.isfinite(point.imag):
                raise ValueError(f"coordinate {index} is non-finite")
            result.append(point)
    return tuple(result)


def _normalize_cp1(
    vertices: Sequence[complex | None],
    distinguished_face: tuple[int, int, int],
    *,
    positive_handedness: bool = False,
) -> tuple[complex | None, ...]:
    """Map the ordered anchors to ``(infinity, 0, 1)`` by a Möbius map."""
    a, b, c = distinguished_face
    za, zb, zc = vertices[a], vertices[b], vertices[c]
    if len({za, zb, zc}) != 3:
        raise ValueError("distinguished vertices must be distinct")

    def transform(z: complex | None) -> complex | None:
        if za is None:
            if z is None:
                return None
            assert zb is not None and zc is not None
            return (z - zb) / (zc - zb)
        if zb is None:
            if z is None:
                return 0j
            assert zc is not None
            return (zc - za) / (z - za)
        if zc is None:
            if z is None:
                return 1 + 0j
            return (z - zb) / (z - za)
        if z is None:
            return (zc - za) / (zc - zb)
        return ((z - zb) * (zc - za)) / ((z - za) * (zc - zb))

    normalized = tuple(transform(vertex) for vertex in vertices)
    materialized = list(normalized)
    materialized[a] = None
    materialized[b] = 0j
    materialized[c] = 1 + 0j
    if any(
        vertex is not None
        and (not math.isfinite(vertex.real) or not math.isfinite(vertex.imag))
        for vertex in materialized
    ):
        raise ValueError("Möbius normalization produced non-finite coordinates")
    if positive_handedness:
        nonanchors = [
            vertex
            for label, vertex in enumerate(materialized)
            if label not in distinguished_face
        ]
        if any(vertex is None for vertex in nonanchors):
            raise ValueError("only the distinguished pole may be infinity")
        imaginary_parts = [complex(vertex).imag for vertex in nonanchors]
        if imaginary_parts and all(value < 0.0 for value in imaginary_parts):
            materialized = [
                None if vertex is None else vertex.conjugate()
                for vertex in materialized
            ]
        elif imaginary_parts and not all(value > 0.0 for value in imaginary_parts):
            raise ValueError(
                "normalized non-anchor vertices must lie in one strict half-plane"
            )
        materialized[a] = None
        materialized[b] = 0j
        materialized[c] = 1 + 0j
    return tuple(materialized)


def _coordinate_digest(vertices: Sequence[complex | None]) -> str:
    return _sha256_bytes(strict_json_bytes(_cp1_pairs(vertices)))


def _sphere_points(vertices: Sequence[complex | None]) -> np.ndarray:
    points = np.empty((len(vertices), 3), dtype=float)
    for index, coordinate in enumerate(vertices):
        if coordinate is None:
            points[index] = (0.0, 0.0, 1.0)
            continue
        x, y = coordinate.real, coordinate.imag
        scale = max(abs(x), abs(y))
        if scale <= 1.0:
            squared = x * x + y * y
            denominator = 1.0 + squared
            points[index] = (
                2.0 * x / denominator,
                2.0 * y / denominator,
                (squared - 1.0) / denominator,
            )
        else:
            sx, sy = x / scale, y / scale
            norm = math.hypot(sx, sy)
            inverse = (1.0 / scale) / norm
            inverse_squared = inverse * inverse
            denominator = 1.0 + inverse_squared
            points[index] = (
                2.0 * (sx / norm) * inverse / denominator,
                2.0 * (sy / norm) * inverse / denominator,
                (1.0 - inverse_squared) / denominator,
            )
    return points


def _coordinate_error(
    left: Sequence[complex | None], right: Sequence[complex | None]
) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("coordinate comparison requires equal label counts")
    errors = np.linalg.norm(_sphere_points(left) - _sphere_points(right), axis=1)
    worst = int(np.argmax(errors))
    return {
        "metric": "euclidean-chord-on-unit-sphere-label-aligned",
        "n_vertices": len(errors),
        "max_chordal": float(np.max(errors)),
        "median_chordal": float(np.median(errors)),
        "rms_chordal": float(np.sqrt(np.mean(errors * errors))),
        "worst_vertex": worst,
    }


def _compact_audit(value: Any) -> dict[str, Any]:
    payload = _jsonable(value.as_dict() if hasattr(value, "as_dict") else value)
    if not isinstance(payload, dict):
        raise TypeError("strict audit result must be an object")

    def select(name: str, fields: Sequence[str]) -> dict[str, Any]:
        source = payload.get(name)
        if not isinstance(source, dict):
            return {"passes": False, "status": "missing"}
        return {field: source.get(field) for field in fields}

    qhull = select(
        "qhull_agreement",
        ("passes", "status", "agrees", "n_labelled_faces", "n_qhull_faces"),
    )
    raw_qhull = payload.get("qhull_agreement")
    if isinstance(raw_qhull, dict):
        qhull["missing_face_count"] = len(raw_qhull.get("missing_faces", ()))
        qhull["extra_face_count"] = len(raw_qhull.get("extra_faces", ()))
    return {
        "strict_pass": bool(payload.get("strict_pass", False)),
        "classification": str(payload.get("classification", "error")),
        "errors": [str(item) for item in payload.get("errors", ())],
        "topology": select(
            "topology",
            (
                "passes",
                "status",
                "closed_sphere",
                "combinatorially_valid",
                "input_orientation_coherent",
                "euler_characteristic",
                "n_vertices",
                "n_edges",
                "n_faces",
            ),
        ),
        "normalization": select(
            "normalization",
            (
                "passes",
                "status",
                "matches",
                "coordinate_anchors_match",
                "handedness_matches",
                "label_order_matches",
                "min_nonanchor_imaginary",
            ),
        ),
        "chordal_distinctness": select(
            "chordal_distinctness",
            (
                "passes",
                "status",
                "all_distinct",
                "min_chordal_distance",
                "closest_pair",
            ),
        ),
        "support": select(
            "support",
            (
                "passes",
                "status",
                "strict_supporting",
                "min_normalized_margin",
                "tie_count",
                "indeterminate_count",
                "inconsistent_face_count",
                "filter_ambiguous_count",
            ),
        ),
        "shear": select(
            "shear",
            (
                "passes",
                "status",
                "n_edges",
                "max_abs",
                "rms",
                "tolerance",
                "stable_crosscheck_passes",
                "stable_crosscheck_max_abs",
            ),
        ),
        "bending": select(
            "bending",
            (
                "passes",
                "status",
                "coherent_sign",
                "orientation_sign",
                "sign_violation_count",
                "endpoint_degenerate",
                "min_endpoint_margin_radians",
                "max_abs_angle_radians",
                "worst_edge",
            ),
        ),
        "planar_embedding": select(
            "planar_embedding", ("passes", "status", "embedded", "reason")
        ),
        "spherical_embedding": select(
            "spherical_embedding",
            (
                "passes",
                "status",
                "embedded",
                "orientation_coherent",
                "min_radial_face_margin",
                "n_edges_checked",
                "n_nonincident_edge_pairs_checked",
            ),
        ),
        "qhull_agreement": qhull,
    }


def _compact_solver_result(value: Any) -> dict[str, Any]:
    payload = _jsonable(value.as_dict() if hasattr(value, "as_dict") else value)
    if not isinstance(payload, dict):
        return {"result_type": type(value).__name__}
    omitted = {
        "audit",
        "vertices_cp1",
        "normalized_vertices_cp1",
        "initial_vertices_cp1",
        "initial_embedding",
        "trajectory",
        "angles",
        "points",
    }
    compact = {key: item for key, item in payload.items() if key not in omitted}
    trajectory = payload.get("trajectory")
    if isinstance(trajectory, list):
        compact["trajectory_count"] = len(trajectory)
    return compact


def _stage(
    stages: list[dict[str, Any]],
    name: str,
    operation: Callable[[], Any],
) -> Any:
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        result = operation()
    except BaseException as exc:
        stages.append(
            {
                "name": name,
                "status": "error",
                "reason": type(exc).__name__,
                "message": str(exc)[:2000],
                "wall_seconds": float(time.perf_counter() - wall_started),
                "process_cpu_seconds": float(time.process_time() - cpu_started),
                "metrics": {},
            }
        )
        raise
    stages.append(
        {
            "name": name,
            "status": "completed",
            "reason": "completed",
            "message": None,
            "wall_seconds": float(time.perf_counter() - wall_started),
            "process_cpu_seconds": float(time.process_time() - cpu_started),
            "metrics": {},
        }
    )
    return result


def _mark_last_stage_failed(
    stages: list[dict[str, Any]], reason: str, message: str
) -> None:
    stages[-1].update(status="failed", reason=reason, message=message[:2000])


def _new_method_row(method_id: str, label: str) -> dict[str, Any]:
    return {
        "method_id": method_id,
        "label": label,
        "status": "error",
        "reason": "not_started",
        "message": None,
        "initialization": {},
        "stages": [],
        "solver": {},
        "endpoint": {
            "available": False,
            "coordinate_sha256": None,
            "vertices": None,
        },
        "audit": None,
        "reference_error": None,
        "maximality_assessment": {
            "claimed": False,
            "classification": "not_claimed",
            "scope": "fixed_combinatorics",
            "basis": "no independently audited strict convex endpoint",
            "formal_proof": False,
        },
        "timings": {
            "exclusive_wall_seconds": 0.0,
            "exclusive_process_cpu_seconds": 0.0,
            "dependency_wall_seconds": 0.0,
            "dependency_process_cpu_seconds": 0.0,
            "end_to_end_wall_seconds": 0.0,
            "end_to_end_process_cpu_seconds": 0.0,
        },
    }


def _maximality_assessment(*, strict_pass: bool, method_id: str) -> dict[str, Any]:
    claimed = bool(strict_pass)
    if claimed:
        basis = (
            "Rivin angle optimum plus independent strict convex realization audit"
            if method_id == METHOD_ANGLE
            else "zero shear plus independent strict convex realization audit"
        )
        classification = "strict_audit_supported_fixed_combinatorics_maximum"
    else:
        basis = "no independently audited strict convex endpoint"
        classification = "not_claimed"
    return {
        "claimed": claimed,
        "classification": classification,
        "scope": "fixed_combinatorics",
        "basis": basis,
        "formal_proof": False,
    }


def _finalize_endpoint(
    row: dict[str, Any],
    *,
    case: ComparisonCase,
    vertices: Sequence[complex | None],
    audit_function: Callable[..., Any],
    normalize_function: Callable[..., Any],
    run_config: Mapping[str, Any],
    include_coordinates: bool,
) -> tuple[complex | None, ...]:
    normalized = tuple(
        normalize_function(
            tuple(vertices),
            case.distinguished_face,
            positive_handedness=True,
        )
    )
    pole, zero, one = case.distinguished_face
    nonanchors = [
        vertex
        for label, vertex in enumerate(normalized)
        if label not in case.distinguished_face
    ]
    script_normalization = {
        "passes": bool(
            normalized[pole] is None
            and normalized[zero] == 0j
            and normalized[one] == 1 + 0j
            and all(
                vertex is not None and complex(vertex).imag > 0.0
                for vertex in nonanchors
            )
        ),
        "ordered_anchors_match": bool(
            normalized[pole] is None
            and normalized[zero] == 0j
            and normalized[one] == 1 + 0j
        ),
        "positive_half_plane": bool(
            all(
                vertex is not None and complex(vertex).imag > 0.0
                for vertex in nonanchors
            )
        ),
    }
    audit = _stage(
        row["stages"],
        "independent_strict_audit",
        lambda: audit_function(
            case.faces,
            normalized,
            distinguished_face=case.distinguished_face,
            shear_tolerance=float(run_config["shear_tolerance"]),
            support_tolerance=float(run_config["support_tolerance"]),
            bending_tolerance=float(run_config["bending_tolerance"]),
            check_spherical_embedding=True,
            require_standard_normalization=(case.distinguished_face == (0, 1, 2)),
        ),
    )
    compact = _compact_audit(audit)
    compact["script_normalization"] = script_normalization
    compact["strict_pass"] = bool(
        compact["strict_pass"] and script_normalization["passes"]
    )
    row["audit"] = compact
    row["endpoint"] = {
        "available": True,
        "coordinate_sha256": _coordinate_digest(normalized),
        "vertices": _cp1_pairs(normalized) if include_coordinates else None,
    }
    row["maximality_assessment"] = _maximality_assessment(
        strict_pass=bool(compact["strict_pass"]),
        method_id=str(row["method_id"]),
    )
    if case.reference_vertices is not None:
        reference = tuple(
            normalize_function(
                case.reference_vertices,
                case.distinguished_face,
                positive_handedness=True,
            )
        )
        row["reference_error"] = _coordinate_error(normalized, reference)
    row["status"] = "completed"
    row["reason"] = str(compact["classification"])
    row["message"] = None
    return normalized


def _load_core_api() -> SimpleNamespace:
    rivin = importlib.import_module("ideal_poly_volume_toolkit.rivin_delaunay")
    zero = importlib.import_module("ideal_poly_volume_toolkit.rivin_zero_shear")
    pole_initializer = None
    for name in (
        "weighted_pole_link_tutte_embedding",
        "pole_link_tutte_initialization",
        "pole_link_harmonic_initialization",
        "pole_link_tutte_embedding",
    ):
        candidate = getattr(zero, name, None)
        if callable(candidate):
            pole_initializer = candidate
            break
    if pole_initializer is None:
        raise RuntimeError(
            "rivin_zero_shear core does not expose a pole-link harmonic initializer"
        )
    solve_initial = getattr(zero, "solve_zero_shear_from_initial", None)
    if not callable(solve_initial):
        raise RuntimeError(
            "rivin_zero_shear core does not expose solve_zero_shear_from_initial"
        )
    return SimpleNamespace(
        check_delaunay_realizability=rivin.check_delaunay_realizability,
        optimize_hyperbolic_volume=rivin.optimize_hyperbolic_volume,
        realize_angles_as_points=rivin.realize_angles_as_points,
        extract_boundary_vertices=rivin.extract_boundary_vertices,
        audit_zero_shear_realization=zero.audit_zero_shear_realization,
        normalize_cp1_points=zero.normalize_cp1_points,
        pole_link_initializer=pole_initializer,
        pole_link_initializer_name=pole_initializer.__name__,
        solve_zero_shear_from_initial=solve_initial,
        weighted_tutte_embedding=zero.weighted_tutte_embedding,
    )


def _pole_link_disk(
    case: ComparisonCase,
) -> tuple[tuple[tuple[int, int, int], ...], tuple[int, ...]]:
    """Delete the distinguished pole and return its finite disk and link."""
    pole = case.distinguished_face[0]
    incident = [face for face in case.faces if pole in face]
    disk = tuple(face for face in case.faces if pole not in face)
    if not incident or not disk:
        raise ValueError("distinguished pole must have a nonempty star and complement")
    link_edges: list[tuple[int, int]] = []
    for face in incident:
        opposite = tuple(vertex for vertex in face if vertex != pole)
        if len(opposite) != 2:
            raise ValueError("invalid pole-star face")
        link_edges.append(tuple(sorted(opposite)))
    link_vertices = tuple(sorted({vertex for edge in link_edges for vertex in edge}))
    degrees = Counter(vertex for edge in link_edges for vertex in edge)
    if len(link_edges) != len(link_vertices) or any(
        degrees[vertex] != 2 for vertex in link_vertices
    ):
        raise ValueError("distinguished pole link is not one simple cycle")
    return disk, link_vertices


def _realization_vertices(
    result: Any, *, n_vertices: int, expected_missing: set[int]
) -> tuple[complex | None, ...]:
    payload = result if isinstance(result, Mapping) else vars(result)
    points = payload.get("points")
    vertex_list = payload.get("vertex_list")
    if points is None or vertex_list is None:
        raise StageFailure(
            "realization_missing_coordinates", "realizer returned no points"
        )
    array = np.asarray(points, dtype=float)
    if array.shape != (len(vertex_list), 2) or not np.all(np.isfinite(array)):
        raise StageFailure(
            "realization_invalid_coordinates",
            "realizer returned an invalid point array",
        )
    vertices: list[complex | None] = [None] * n_vertices
    for row, raw_label in enumerate(vertex_list):
        label = int(raw_label)
        if label < 0 or label >= n_vertices or vertices[label] is not None:
            raise StageFailure(
                "realization_invalid_labels", "realizer returned an invalid label map"
            )
        vertices[label] = complex(float(array[row, 0]), float(array[row, 1]))
    missing = {label for label, vertex in enumerate(vertices) if vertex is None}
    if missing != expected_missing:
        raise StageFailure(
            "realization_missing_labels",
            f"realizer omitted labels {sorted(missing)}; expected "
            f"{sorted(expected_missing)}",
        )
    return tuple(vertices)


def _run_angle_method(
    case: ComparisonCase,
    run_config: Mapping[str, Any],
    api: SimpleNamespace,
) -> tuple[dict[str, Any], tuple[complex | None, ...] | None, dict[str, float]]:
    row = _new_method_row(METHOD_ANGLE, "Rivin angle optimize and realize")
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    initialization_timing = {"wall_seconds": 0.0, "process_cpu_seconds": 0.0}
    try:
        disk, link_vertices = _stage(
            row["stages"],
            "remove_distinguished_pole_star",
            lambda: _pole_link_disk(case),
        )
        boundary = set(api.extract_boundary_vertices(list(disk)))
        if boundary != set(link_vertices):
            _mark_last_stage_failed(
                row["stages"],
                "wrong_disk_boundary",
                "removing the distinguished pole did not expose its complete link",
            )
            raise StageFailure(
                "wrong_disk_boundary",
                "removing the distinguished pole did not expose its complete link",
            )
        row["stages"][-1]["metrics"] = {
            "n_disk_faces": len(disk),
            "pole_vertex": case.distinguished_face[0],
            "pole_link_size": len(link_vertices),
            "boundary_vertices": sorted(boundary),
        }

        lp = _stage(
            row["stages"],
            "rivin_feasibility",
            lambda: api.check_delaunay_realizability(
                list(disk), boundary_vertices=boundary, verbose=False
            ),
        )
        row["stages"][-1]["metrics"] = {
            "realizable": bool(lp.get("realizable", False)),
            "success": bool(lp.get("success", False)),
            "min_angle_radians": lp.get("min_angle_radians"),
        }
        if not lp.get("realizable") or lp.get("angles_radians") is None:
            _mark_last_stage_failed(
                row["stages"], "rivin_infeasible", str(lp.get("message", "LP failed"))
            )
            raise StageFailure("rivin_infeasible", str(lp.get("message", "LP failed")))

        optimization = _stage(
            row["stages"],
            "angle_optimization",
            lambda: api.optimize_hyperbolic_volume(
                list(disk),
                initial_angles=lp["angles_radians"],
                boundary_vertices=boundary,
                verbose=False,
                require_interior=True,
            ),
        )
        row["stages"][-1]["metrics"] = {
            "success": bool(optimization.get("success", False)),
            "message": str(optimization.get("message", "")),
            "n_iterations": optimization.get("n_iterations"),
            "n_active_constraints": optimization.get("n_active_constraints"),
            "reduced_gradient_norm": optimization.get("reduced_gradient_norm"),
            "volume": optimization.get("volume"),
        }
        if not optimization.get("success") or optimization.get("angles") is None:
            _mark_last_stage_failed(
                row["stages"],
                "angle_optimization_failure",
                str(optimization.get("message", "angle optimizer failed")),
            )
            raise StageFailure(
                "angle_optimization_failure",
                str(optimization.get("message", "angle optimizer failed")),
            )

        realization = _stage(
            row["stages"],
            "angle_realization",
            lambda: api.realize_angles_as_points(
                list(disk),
                optimization["angles"],
                boundary_vertices=boundary,
                verbose=False,
            ),
        )
        row["stages"][-1]["metrics"] = {
            "success": bool(realization.get("success", False)),
            "triangulation_preserved": bool(
                realization.get("triangulation_preserved", False)
            ),
            "rms_angle_error": realization.get("angle_error"),
            "max_angle_error": realization.get("max_angle_error"),
            "closure_error": realization.get("closure_error"),
            "orientation_consistent": realization.get("orientation_consistent"),
        }
        n_vertices = max(max(face) for face in case.faces) + 1
        raw_vertices = _realization_vertices(
            realization,
            n_vertices=n_vertices,
            expected_missing={case.distinguished_face[0]},
        )
        normalization_wall_started = time.perf_counter()
        normalization_cpu_started = time.process_time()
        normalized = tuple(
            api.normalize_cp1_points(
                raw_vertices,
                case.distinguished_face,
                positive_handedness=True,
            )
        )
        row["stages"].append(
            {
                "name": "mobius_normalization",
                "status": "completed",
                "reason": "completed",
                "message": None,
                "wall_seconds": float(time.perf_counter() - normalization_wall_started),
                "process_cpu_seconds": float(
                    time.process_time() - normalization_cpu_started
                ),
                "metrics": {"coordinate_sha256": _coordinate_digest(normalized)},
            }
        )
        initialization_timing = {
            "wall_seconds": sum(
                float(stage["wall_seconds"])
                for stage in row["stages"]
                if stage["name"] != "independent_strict_audit"
            ),
            "process_cpu_seconds": sum(
                float(stage["process_cpu_seconds"])
                for stage in row["stages"]
                if stage["name"] != "independent_strict_audit"
            ),
        }
        row["initialization"] = {
            "kind": "rivin_angle_optimization_and_developing_map",
            "disk_face_count": len(disk),
            "pole_vertex": case.distinguished_face[0],
            "pole_link_size": len(link_vertices),
            "boundary_vertices": sorted(boundary),
            "coordinate_sha256": _coordinate_digest(normalized),
        }
        row["solver"] = {
            "feasibility": {
                "success": bool(lp.get("success", False)),
                "realizable": bool(lp.get("realizable", False)),
                "message": str(lp.get("message", "")),
                "min_angle_radians": lp.get("min_angle_radians"),
            },
            "optimization": _compact_solver_result(optimization),
            "realization": _compact_solver_result(realization),
        }
        endpoint = _finalize_endpoint(
            row,
            case=case,
            vertices=normalized,
            audit_function=api.audit_zero_shear_realization,
            normalize_function=api.normalize_cp1_points,
            run_config=run_config,
            include_coordinates=bool(run_config["include_coordinates"]),
        )
        if not realization.get("success", False):
            row["message"] = str(realization.get("message", "realization failed"))
        return row, endpoint, initialization_timing
    except StageFailure as exc:
        row.update(status="failed", reason=exc.reason, message=str(exc)[:2000])
        return row, None, initialization_timing
    except BaseException as exc:
        row.update(
            status="error",
            reason=type(exc).__name__,
            message=str(exc)[:2000],
        )
        return row, None, initialization_timing
    finally:
        wall_elapsed = float(time.perf_counter() - wall_started)
        cpu_elapsed = float(time.process_time() - cpu_started)
        row["timings"] = {
            "exclusive_wall_seconds": wall_elapsed,
            "exclusive_process_cpu_seconds": cpu_elapsed,
            "dependency_wall_seconds": 0.0,
            "dependency_process_cpu_seconds": 0.0,
            "end_to_end_wall_seconds": wall_elapsed,
            "end_to_end_process_cpu_seconds": cpu_elapsed,
        }


def _call_solve_from_initial(
    function: Callable[..., Any],
    case: ComparisonCase,
    initial: Sequence[complex | None],
    run_config: Mapping[str, Any],
) -> Any:
    x_scale = 1.0 if run_config["x_scale"] == "unit" else "jac"
    kwargs: dict[str, Any] = {
        "gauge": "standard_upper_half_plane",
        "max_nfev": int(run_config["max_nfev"]),
        "shear_tolerance": float(run_config["shear_tolerance"]),
        "support_tolerance": float(run_config["support_tolerance"]),
        "bending_tolerance": float(run_config["bending_tolerance"]),
        "check_spherical_embedding": True,
        "ftol": float(run_config["ftol"]),
        "xtol": float(run_config["xtol"]),
        "gtol": float(run_config["gtol"]),
    }
    parameters = inspect.signature(function).parameters
    if "x_scale" in parameters:
        kwargs["x_scale"] = x_scale
    if "lm_polish_steps" in parameters:
        kwargs["lm_polish_steps"] = int(run_config["lm_polish_steps"])
    return function(
        case.faces,
        case.distinguished_face,
        initial,
        **kwargs,
    )


def _run_direct_method(
    case: ComparisonCase,
    run_config: Mapping[str, Any],
    api: SimpleNamespace,
    *,
    method_id: str,
    label: str,
    initial_kind: str,
    initializer: Callable[[], Any],
    dependency_timing: Mapping[str, float] | None = None,
    dependency_method: str | None = None,
) -> tuple[dict[str, Any], tuple[complex | None, ...] | None]:
    row = _new_method_row(method_id, label)
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    dependency_timing = (
        {"wall_seconds": 0.0, "process_cpu_seconds": 0.0}
        if dependency_timing is None
        else dependency_timing
    )
    try:
        raw_initial = _stage(row["stages"], "initialization", initializer)
        n_vertices = max(max(face) for face in case.faces) + 1
        initial = _materialize_cp1(raw_initial, n_vertices=n_vertices)
        initial_digest = _coordinate_digest(initial)
        row["stages"][-1]["metrics"] = {
            "coordinate_sha256": initial_digest,
            "n_vertices": n_vertices,
        }
        row["initialization"] = {
            "kind": initial_kind,
            "coordinate_sha256": initial_digest,
            "dependency_method": dependency_method,
            "core_initializer": (
                api.pole_link_initializer_name if method_id == METHOD_POLE else None
            ),
        }
        solve = _stage(
            row["stages"],
            "zero_shear_solve",
            lambda: _call_solve_from_initial(
                api.solve_zero_shear_from_initial, case, initial, run_config
            ),
        )
        row["solver"] = _compact_solver_result(solve)
        solve_success = bool(
            getattr(solve, "success", False)
            if not isinstance(solve, Mapping)
            else solve.get("success", False)
        )
        row["stages"][-1]["metrics"] = {
            "success": solve_success,
            "classification": str(
                getattr(solve, "classification", "unknown")
                if not isinstance(solve, Mapping)
                else solve.get("classification", "unknown")
            ),
            "nfev": (
                getattr(solve, "nfev", None)
                if not isinstance(solve, Mapping)
                else solve.get("nfev")
            ),
            "max_abs_residual": (
                getattr(solve, "max_abs_residual", None)
                if not isinstance(solve, Mapping)
                else solve.get("max_abs_residual")
            ),
        }
        endpoint = _materialize_cp1(solve, n_vertices=n_vertices)
        normalized = _finalize_endpoint(
            row,
            case=case,
            vertices=endpoint,
            audit_function=api.audit_zero_shear_realization,
            normalize_function=api.normalize_cp1_points,
            run_config=run_config,
            include_coordinates=bool(run_config["include_coordinates"]),
        )
        if not solve_success:
            row["message"] = str(
                getattr(solve, "message", "solver reported failure")
                if not isinstance(solve, Mapping)
                else solve.get("message", "solver reported failure")
            )
        return row, normalized
    except StageFailure as exc:
        row.update(status="failed", reason=exc.reason, message=str(exc)[:2000])
        return row, None
    except BaseException as exc:
        row.update(status="error", reason=type(exc).__name__, message=str(exc)[:2000])
        return row, None
    finally:
        exclusive_wall = float(time.perf_counter() - wall_started)
        exclusive_cpu = float(time.process_time() - cpu_started)
        row["timings"] = {
            "exclusive_wall_seconds": exclusive_wall,
            "exclusive_process_cpu_seconds": exclusive_cpu,
            "dependency_wall_seconds": float(dependency_timing["wall_seconds"]),
            "dependency_process_cpu_seconds": float(
                dependency_timing["process_cpu_seconds"]
            ),
            "end_to_end_wall_seconds": float(
                exclusive_wall + dependency_timing["wall_seconds"]
            ),
            "end_to_end_process_cpu_seconds": float(
                exclusive_cpu + dependency_timing["process_cpu_seconds"]
            ),
        }


def _skipped_hybrid(dependency_timing: Mapping[str, float]) -> dict[str, Any]:
    row = _new_method_row(METHOD_HYBRID, "Angle realization then zero-shear polish")
    row.update(
        status="skipped",
        reason="angle_initialization_unavailable",
        message="angle method did not produce labeled finite coordinates",
    )
    row["initialization"] = {
        "kind": "angle_realization_endpoint",
        "dependency_method": METHOD_ANGLE,
        "coordinate_sha256": None,
        "core_initializer": None,
    }
    row["timings"] = {
        "exclusive_wall_seconds": 0.0,
        "exclusive_process_cpu_seconds": 0.0,
        "dependency_wall_seconds": float(dependency_timing["wall_seconds"]),
        "dependency_process_cpu_seconds": float(
            dependency_timing["process_cpu_seconds"]
        ),
        "end_to_end_wall_seconds": float(dependency_timing["wall_seconds"]),
        "end_to_end_process_cpu_seconds": float(
            dependency_timing["process_cpu_seconds"]
        ),
    }
    return row


def _method_comparisons(
    rows: Sequence[Mapping[str, Any]],
    endpoints: Mapping[str, Sequence[complex | None]],
) -> list[dict[str, Any]]:
    order = [str(row["method_id"]) for row in rows]
    comparisons: list[dict[str, Any]] = []
    for left_index, left_id in enumerate(order):
        if left_id not in endpoints:
            continue
        for right_id in order[left_index + 1 :]:
            if right_id not in endpoints:
                continue
            left_row = next(row for row in rows if row["method_id"] == left_id)
            right_row = next(row for row in rows if row["method_id"] == right_id)
            right_wall = float(right_row["timings"]["end_to_end_wall_seconds"])
            wall_ratio = (
                None
                if right_wall == 0.0
                else float(left_row["timings"]["end_to_end_wall_seconds"]) / right_wall
            )
            right_cpu = float(right_row["timings"]["end_to_end_process_cpu_seconds"])
            cpu_ratio = (
                None
                if right_cpu == 0.0
                else float(left_row["timings"]["end_to_end_process_cpu_seconds"])
                / right_cpu
            )
            comparisons.append(
                {
                    "left_method": left_id,
                    "right_method": right_id,
                    "coordinate_error": _coordinate_error(
                        endpoints[left_id], endpoints[right_id]
                    ),
                    "left_to_right_end_to_end_wall_time_ratio": wall_ratio,
                    "left_to_right_end_to_end_process_cpu_time_ratio": cpu_ratio,
                    "timing_ratio_scope": (
                        "single-run-end-to-end-implementation-including-core-"
                        "and-independent-audits"
                    ),
                    "same_strict_outcome": bool(
                        left_row.get("audit", {}).get("strict_pass", False)
                        == right_row.get("audit", {}).get("strict_pass", False)
                    ),
                }
            )
    return comparisons


def compare_case(
    case: ComparisonCase,
    run_config: Mapping[str, Any],
    *,
    api: SimpleNamespace | None = None,
) -> dict[str, Any]:
    """Run the deterministic method order for one fixed labeled sphere."""
    api = _load_core_api() if api is None else api
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    methods: list[dict[str, Any]] = []
    endpoints: dict[str, tuple[complex | None, ...]] = {}

    angle_row, angle_endpoint, angle_initialization_timing = _run_angle_method(
        case, run_config, api
    )
    methods.append(angle_row)
    if angle_endpoint is not None:
        endpoints[METHOD_ANGLE] = angle_endpoint

    pole_row, pole_endpoint = _run_direct_method(
        case,
        run_config,
        api,
        method_id=METHOD_POLE,
        label="Pole-link regular-polygon harmonic start and zero-shear solve",
        initial_kind="pole_link_regular_polygon_harmonic_disk",
        initializer=lambda: api.pole_link_initializer(
            case.faces, case.distinguished_face
        ),
    )
    methods.append(pole_row)
    if pole_endpoint is not None:
        endpoints[METHOD_POLE] = pole_endpoint

    if angle_endpoint is None:
        hybrid_row = _skipped_hybrid(angle_initialization_timing)
        hybrid_endpoint = None
    else:
        hybrid_row, hybrid_endpoint = _run_direct_method(
            case,
            run_config,
            api,
            method_id=METHOD_HYBRID,
            label="Angle realization then zero-shear polish",
            initial_kind="angle_realization_endpoint",
            initializer=lambda: angle_endpoint,
            dependency_timing=angle_initialization_timing,
            dependency_method=METHOD_ANGLE,
        )
    methods.append(hybrid_row)
    if hybrid_endpoint is not None:
        endpoints[METHOD_HYBRID] = hybrid_endpoint

    if run_config["include_face_tutte"]:
        face_row, face_endpoint = _run_direct_method(
            case,
            run_config,
            api,
            method_id=METHOD_FACE_TUTTE,
            label="Outer-face triangle Tutte start and zero-shear solve",
            initial_kind="outer_face_triangle_uniform_tutte",
            initializer=lambda: api.weighted_tutte_embedding(
                case.faces, case.distinguished_face
            ),
        )
        methods.append(face_row)
        if face_endpoint is not None:
            endpoints[METHOD_FACE_TUTTE] = face_endpoint

    statuses = Counter(str(row["status"]) for row in methods)
    strict_methods = [
        str(row["method_id"])
        for row in methods
        if isinstance(row.get("audit"), dict) and row["audit"].get("strict_pass")
    ]
    return {
        "case_id": case.case_id,
        "source_pointer": case.source_pointer,
        "source_kind": case.source_kind,
        "input_summary": {
            "n_vertices": max(max(face) for face in case.faces) + 1,
            "n_faces": len(case.faces),
            "faces_sha256": _sha256_bytes(strict_json_bytes(case.faces)),
            "distinguished_face": list(case.distinguished_face),
            "reference_available": case.reference_vertices is not None,
            "reference_coordinate_sha256": (
                None
                if case.reference_vertices is None
                else _coordinate_digest(
                    tuple(
                        api.normalize_cp1_points(
                            case.reference_vertices,
                            case.distinguished_face,
                            positive_handedness=True,
                        )
                    )
                )
            ),
        },
        "methods": methods,
        "comparisons": _method_comparisons(methods, endpoints),
        "summary": {
            "method_count": len(methods),
            "completed_count": int(statuses["completed"]),
            "failed_count": int(
                statuses["failed"] + statuses["error"] + statuses["skipped"]
            ),
            "strict_method_ids": strict_methods,
            "all_required_methods_completed": all(
                next(row for row in methods if row["method_id"] == method_id)["status"]
                == "completed"
                for method_id in REQUIRED_METHODS
            ),
        },
        "timings": {
            "wall_seconds": float(time.perf_counter() - wall_started),
            "process_cpu_seconds": float(time.process_time() - cpu_started),
        },
    }


def _run_config_hash(run_config: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        json.dumps(_jsonable(run_config), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def _source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        SCHEMA_PATH,
        ARITHMETIC_SCHEMA_PATH,
        ZERO_SHEAR_CORE_PATH,
        RIVIN_CORE_PATH,
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required source file is missing: {missing[0]}")
    return {
        str(path.relative_to(REPOSITORY_ROOT)): _sha256_file(path) for path in paths
    }


def _dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for distribution in ("ideal-poly-volume-toolkit", "numpy", "scipy", "jsonschema"):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = None
    return result


def _git_provenance() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        ).stdout.strip()
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    return (commit if len(commit) == 40 else None), bool(dirty_result.stdout)


def validate_comparison_document(document: Mapping[str, Any]) -> None:
    strict_json_bytes(document)
    _validate_with_schema(document, SCHEMA_PATH)


def run_comparison(
    input_path: str | Path,
    output_path: str | Path,
    run_config: Mapping[str, Any],
    *,
    requested_ids: Sequence[str] = (),
    distinguished_override: tuple[int, int, int] | None = None,
    overwrite: bool = False,
    invocation: Sequence[str] | None = None,
    api: SimpleNamespace | None = None,
) -> dict[str, Any]:
    """Run the bounded single-process comparison and atomically emit JSON."""
    input_path = Path(input_path).resolve()
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to clobber existing output {output_path}; use --overwrite"
        )
    _validate_run_config(run_config)
    input_kind, input_schema_version, cases = load_comparison_cases(
        input_path,
        requested_ids=requested_ids,
        distinguished_override=distinguished_override,
    )
    expected_ids = [case.case_id for case in cases]
    if list(run_config["selected_case_ids"]) != expected_ids:
        raise ValueError(
            "run_config selected_case_ids do not match selected input cases"
        )
    source_hashes = _source_hashes()
    commit, dirty = _git_provenance()
    started_at = _utc_now()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    api = _load_core_api() if api is None else api
    case_rows = [compare_case(case, run_config, api=api) for case in cases]
    all_required = all(
        row["summary"]["all_required_methods_completed"] for row in case_rows
    )
    any_completed = any(row["summary"]["completed_count"] for row in case_rows)
    status = (
        "complete"
        if all_required
        else ("partial_failure" if any_completed else "failed")
    )
    finished_at = _utc_now()
    document = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "generator": {
            "name": GENERATOR_NAME,
            "version": GENERATOR_VERSION,
            "source_sha256": source_hashes[GENERATOR_NAME],
        },
        "status": status,
        "created_at": started_at,
        "finished_at": finished_at,
        "input": {
            "path": str(input_path),
            "sha256": _sha256_file(input_path),
            "kind": input_kind,
            "schema_version": input_schema_version,
            "selected_case_ids": expected_ids,
        },
        "run_config": _jsonable(run_config),
        "run_config_sha256": _run_config_hash(run_config),
        "fairness": {
            "single_process": True,
            "blas_threads": BLAS_THREADS,
            "shared_labeled_faces": True,
            "shared_distinguished_face": True,
            "shared_direct_solver_settings": True,
            "direct_solver_gauge": "standard_upper_half_plane",
            "direct_solver_x_scale": run_config["x_scale"],
            "direct_solver_lm_polish_steps": run_config["lm_polish_steps"],
            "hybrid_reuses_angle_initialization": True,
            "hybrid_reports_reused_dependency_time": True,
            "independent_strict_audit_per_endpoint": True,
            "input_reference_coordinates_diagnostic_only": True,
            "maximality_requires_strict_audit": True,
            "timings_are_single_run_operational_observations": True,
            "speed_claim_requires_external_warmups_and_repeats": True,
            "time_ratios_include_core_and_independent_audits": True,
            "time_ratio_scope": (
                "single-run-end-to-end-implementation-including-core-"
                "and-independent-audits"
            ),
            "verification_scope": "double-precision-numerical-not-formal",
        },
        "provenance": {
            "invocation": list(sys.argv if invocation is None else invocation),
            "git_commit": commit,
            "git_dirty": dirty,
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "dependency_versions": _dependency_versions(),
            "thread_environment": {
                variable: os.environ.get(variable) for variable in THREAD_ENVIRONMENT
            },
            "source_files": source_hashes,
        },
        "cases": case_rows,
        "summary": {
            "case_count": len(case_rows),
            "complete_case_count": sum(
                row["summary"]["all_required_methods_completed"] for row in case_rows
            ),
            "strict_endpoint_count": sum(
                len(row["summary"]["strict_method_ids"]) for row in case_rows
            ),
            "status": status,
        },
        "timings": {
            "wall_seconds": float(time.perf_counter() - wall_started),
            "process_cpu_seconds": float(time.process_time() - cpu_started),
        },
    }
    validate_comparison_document(document)
    write_json_atomic(output_path, document, overwrite=overwrite)
    return document


def _validate_run_config(run_config: Mapping[str, Any]) -> None:
    selected = run_config.get("selected_case_ids")
    if not isinstance(selected, list) or not selected:
        raise ValueError("selected_case_ids must be a nonempty list")
    if len(set(selected)) != len(selected) or any(
        not isinstance(identifier, str) or not identifier for identifier in selected
    ):
        raise ValueError("selected_case_ids must contain unique nonempty strings")
    max_nfev = run_config.get("max_nfev")
    if not isinstance(max_nfev, int) or isinstance(max_nfev, bool) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    lm_polish_steps = run_config.get("lm_polish_steps")
    if (
        not isinstance(lm_polish_steps, int)
        or isinstance(lm_polish_steps, bool)
        or lm_polish_steps < 0
    ):
        raise ValueError("lm_polish_steps must be a nonnegative integer")
    if run_config.get("x_scale") not in {"unit", "jacobian"}:
        raise ValueError("x_scale must be 'unit' or 'jacobian'")
    for key in ("shear_tolerance", "bending_tolerance", "ftol", "xtol", "gtol"):
        value = run_config.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{key} must be finite")
        if float(value) <= 0.0:
            raise ValueError(f"{key} must be positive")
    support = run_config.get("support_tolerance")
    if not isinstance(support, (int, float)) or not math.isfinite(float(support)):
        raise ValueError("support_tolerance must be finite")
    if float(support) < 0.0:
        raise ValueError("support_tolerance must be nonnegative")
    for key in ("include_coordinates", "include_face_tutte"):
        if not isinstance(run_config.get(key), bool):
            raise ValueError(f"{key} must be boolean")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--config-id", action="append", default=[])
    parser.add_argument(
        "--distinguished-face",
        type=lambda value: _parse_face(
            [int(token.strip()) for token in value.split(",")],
            path="--distinguished-face",
        ),
        metavar="A,B,C",
    )
    parser.add_argument("--max-nfev", type=int, default=DEFAULT_MAX_NFEV)
    parser.add_argument("--lm-polish-steps", type=int, default=DEFAULT_LM_POLISH_STEPS)
    parser.add_argument(
        "--x-scale",
        choices=("unit", "jacobian"),
        default="unit",
        help="TRF variable scaling; unit reproduces the external Sol route",
    )
    parser.add_argument(
        "--shear-tolerance", type=float, default=DEFAULT_SHEAR_TOLERANCE
    )
    parser.add_argument(
        "--support-tolerance", type=float, default=DEFAULT_SUPPORT_TOLERANCE
    )
    parser.add_argument(
        "--bending-tolerance", type=float, default=DEFAULT_BENDING_TOLERANCE
    )
    parser.add_argument("--ftol", type=float, default=DEFAULT_SOLVER_TOLERANCE)
    parser.add_argument("--xtol", type=float, default=DEFAULT_SOLVER_TOLERANCE)
    parser.add_argument("--gtol", type=float, default=DEFAULT_SOLVER_TOLERANCE)
    parser.add_argument("--include-coordinates", action="store_true")
    parser.add_argument(
        "--include-face-tutte",
        action="store_true",
        help="add the outer-face-triangle Tutte start as a secondary method",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", type=Path, metavar="PATH")
    return parser


def _run_config_from_args(
    args: argparse.Namespace, selected_case_ids: Sequence[str]
) -> dict[str, Any]:
    return {
        "selected_case_ids": list(selected_case_ids),
        "max_nfev": int(args.max_nfev),
        "lm_polish_steps": int(args.lm_polish_steps),
        "x_scale": str(args.x_scale),
        "shear_tolerance": float(args.shear_tolerance),
        "support_tolerance": float(args.support_tolerance),
        "bending_tolerance": float(args.bending_tolerance),
        "ftol": float(args.ftol),
        "xtol": float(args.xtol),
        "gtol": float(args.gtol),
        "include_coordinates": bool(args.include_coordinates),
        "include_face_tutte": bool(args.include_face_tutte),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.validate_only is not None:
        try:
            validate_comparison_document(load_json_strict(args.validate_only))
        except (RuntimeError, ValueError) as exc:
            parser.exit(2, f"error: {exc}\n")
        print(f"Valid Rivin zero-shear comparison v1: {args.validate_only}")
        return 0
    if args.input is None or args.output is None:
        parser.error("input and --output are required unless --validate-only is used")
    try:
        _, _, cases = load_comparison_cases(
            args.input,
            requested_ids=args.config_id,
            distinguished_override=args.distinguished_face,
        )
        run_config = _run_config_from_args(args, [case.case_id for case in cases])
        _validate_run_config(run_config)
        document = run_comparison(
            args.input,
            args.output,
            run_config,
            requested_ids=args.config_id,
            distinguished_override=args.distinguished_face,
            overwrite=args.overwrite,
            invocation=list(sys.argv if argv is None else [GENERATOR_NAME, *argv]),
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        f"{document['status']}: {document['summary']['case_count']} case(s); "
        f"output={args.output}"
    )
    return 0 if document["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
