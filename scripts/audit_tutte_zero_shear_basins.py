#!/usr/bin/env python3
"""Audit weighted-Tutte basins for an arithmetic zero-shear corpus.

The arithmetic corpus supplies candidate combinatorial types; its acceptance
gates are not strict convex-realization certificates.  This derived study asks
where an unconstrained coordinate zero-shear solve goes when it is initialized
by the ordinary or a positive-weight Tutte embedding, and applies the strict
audit separately at every endpoint.

The mathematical work is owned by :mod:`ideal_poly_volume_toolkit.rivin_zero_shear`.
This file owns only corpus adaptation, deterministic task construction,
bounded process execution, checkpointing, and study-level aggregation.

Examples
--------
Run the canonical uniform start and forty weighted starts for every entry::

    python3 scripts/audit_tutte_zero_shear_basins.py CORPUS.json \
        --output /tmp/tutte-basins.json --workers 4

Resume the exact same study::

    python3 scripts/audit_tutte_zero_shear_basins.py CORPUS.json \
        --output /tmp/tutte-basins.json --workers 4 --resume
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence


# Bound native-library fan-out before importing NumPy/SciPy through the core
# module.  Parallelism belongs to this driver, not to each numerical worker.
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


SCHEMA_VERSION = "ideal-poly-tutte-zero-shear-audit/1"
PROTOCOL_VERSION = "weighted-tutte-zero-shear-trf/1"
GENERATOR_VERSION = "1.1.0"
GENERATOR_NAME = "scripts/audit_tutte_zero_shear_basins.py"
SCHEMA_PATH = REPOSITORY_ROOT / "docs/schemas/tutte-zero-shear-audit-v1.schema.json"
INPUT_SCHEMA_PATH = REPOSITORY_ROOT / "docs/schemas/arithmetic-corpus-v1.schema.json"
CORE_PATH = REPOSITORY_ROOT / "ideal_poly_volume_toolkit/rivin_zero_shear.py"
HOLONOMY_PATH = REPOSITORY_ROOT / "ideal_poly_volume_toolkit/rivin_holonomy.py"
MAX_WORKERS = 8
DEFAULT_SIGMAS = (0.25, 0.5, 0.75, 1.0)
DEFAULT_STARTS_PER_SIGMA = 10
DEFAULT_MAX_NFEV = 300
DEFAULT_MAX_WALL_SECONDS = 3600.0
DEFAULT_ROOT_CLUSTER_TOLERANCE = 1e-8
DEFAULT_SHEAR_TOLERANCE = 1e-8
DEFAULT_SUPPORT_TOLERANCE = 0.0
DEFAULT_BENDING_TOLERANCE = 1e-10
DEFAULT_SOLVER_TOLERANCE = 1e-13

TARGET_OUTCOMES = frozenset({"target_strict_zero_shear"})
WRONG_ZERO_SHEAR_OUTCOMES = frozenset(
    {
        "embedded_wrong_hull_zero_shear",
        "nonembedded_zero_shear",
        "degenerate_zero_shear",
    }
)


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
    """Convert numerical records to strict, finite JSON-compatible values."""
    if is_dataclass(value):
        value = asdict(value)
    elif hasattr(value, "as_dict") and callable(value.as_dict):
        value = value.as_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("non-finite floating-point value is not valid study JSON")
        return result
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("non-finite complex value is not valid study JSON")
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
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def load_json_strict(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(
            stream,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object at the document root")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_temporary_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
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


def exclusive_create_json(path: str | Path, payload: Any) -> None:
    """Atomically create ``path`` without replacing an existing file."""
    target = Path(path)
    temporary = _write_temporary_json(target, payload)
    try:
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to clobber existing output {target}; "
                "use --resume or --overwrite"
            ) from exc
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"atomic replacement target does not exist: {target}")
    temporary = _write_temporary_json(target, payload)
    try:
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_with_schema(document: dict[str, Any], schema_path: Path) -> None:
    try:
        from jsonschema.validators import validator_for
    except ImportError as exc:
        raise RuntimeError(
            "study schema validation requires jsonschema; install the dev extra"
        ) from exc
    schema = load_json_strict(schema_path)
    validator_class = validator_for(schema)
    validator_class.check_schema(schema)
    validator_class(schema).validate(document)


def load_arithmetic_corpus(path: str | Path) -> dict[str, Any]:
    document = load_json_strict(path)
    if document.get("schema_version") != "ideal-poly-arithmetic-corpus/1":
        raise ValueError("input is not an ideal-poly-arithmetic-corpus/1 document")
    _validate_with_schema(document, INPUT_SCHEMA_PATH)
    configurations = document.get("configurations")
    if not isinstance(configurations, list) or not configurations:
        raise ValueError("arithmetic corpus contains no configurations")
    identifiers = [entry.get("id") for entry in configurations]
    if any(not isinstance(identifier, str) for identifier in identifiers):
        raise ValueError("every arithmetic configuration must have a string id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("arithmetic configuration ids must be unique")
    return document


def select_configurations(
    corpus: Mapping[str, Any], requested_ids: Sequence[str] | None
) -> list[dict[str, Any]]:
    by_id = {entry["id"]: entry for entry in corpus["configurations"]}
    if requested_ids:
        if len(set(requested_ids)) != len(requested_ids):
            raise ValueError("--config-id values must be unique")
        missing = sorted(set(requested_ids) - set(by_id))
        if missing:
            raise ValueError(f"unknown configuration id: {missing[0]}")
        identifiers = sorted(requested_ids)
    else:
        identifiers = sorted(by_id)
    return [by_id[identifier] for identifier in identifiers]


def _rotate_face_canonically(face: Sequence[int]) -> tuple[int, int, int]:
    values = tuple(int(vertex) for vertex in face)
    if len(values) != 3 or len(set(values)) != 3:
        raise ValueError("every distinguished face candidate must have three vertices")
    rotations = (values, values[1:] + values[:1], values[2:] + values[:2])
    return min(rotations)


def deterministic_distinguished_faces(
    faces: Iterable[Sequence[int]], count: int
) -> list[tuple[int, int, int]]:
    """Choose oriented faces deterministically and spread them through the list."""
    if count < 1:
        raise ValueError("anchors_per_configuration must be positive")
    by_unoriented: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for raw_face in faces:
        oriented = _rotate_face_canonically(raw_face)
        key = tuple(sorted(oriented))
        if key in by_unoriented:
            raise ValueError(f"duplicate unoriented face {key}")
        by_unoriented[key] = oriented
    ordered = [by_unoriented[key] for key in sorted(by_unoriented)]
    if not ordered:
        raise ValueError("configuration has no faces")
    if count >= len(ordered):
        return ordered
    # Evenly spaced integer indices avoid both RNG/version dependence and a
    # systematic concentration on the lowest vertex labels.
    indices = [(offset * len(ordered)) // count for offset in range(count)]
    return [ordered[index] for index in indices]


def explicit_distinguished_faces(
    faces: Iterable[Sequence[int]], requested: Iterable[Sequence[int]]
) -> list[tuple[int, int, int]]:
    """Validate exact ordered anchors without changing their label order."""
    available = {frozenset(int(vertex) for vertex in face) for face in faces}
    selected: list[tuple[int, int, int]] = []
    for index, raw_face in enumerate(requested):
        face = tuple(int(vertex) for vertex in raw_face)
        if len(face) != 3 or len(set(face)) != 3 or any(vertex < 0 for vertex in face):
            raise ValueError(
                f"explicit distinguished face {index} must contain three "
                "distinct nonnegative labels"
            )
        if frozenset(face) not in available:
            raise ValueError(
                f"explicit distinguished face {face} is not a configuration face"
            )
        selected.append(face)
    if len(set(selected)) != len(selected):
        raise ValueError("explicit distinguished faces must not contain duplicates")
    return selected


def derive_start_seed(
    base_seed: int,
    configuration_id: str,
    distinguished_face: Sequence[int],
    sigma: float,
    replicate: int,
) -> int:
    identity = {
        "base_seed": int(base_seed),
        "configuration_id": configuration_id,
        "distinguished_face": list(map(int, distinguished_face)),
        "sigma": float(sigma).hex(),
        "replicate": int(replicate),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _stable_identifier(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"{prefix}-sha256:{_sha256_bytes(encoded)}"


def build_tasks(
    configurations: Sequence[Mapping[str, Any]], run_config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for configuration in configurations:
        configuration_id = str(configuration["id"])
        faces = configuration["input"]["triangles"]
        requested_anchors = run_config["explicit_distinguished_faces"]
        if requested_anchors:
            anchors = explicit_distinguished_faces(faces, requested_anchors)
        else:
            anchors = deterministic_distinguished_faces(
                faces, int(run_config["anchors_per_configuration"])
            )
        for anchor in anchors:
            case_payload = {
                "configuration_id": configuration_id,
                "distinguished_face": list(anchor),
            }
            case_id = _stable_identifier("case", case_payload)
            starts: list[dict[str, Any]] = [
                {
                    "kind": "uniform",
                    "sigma": None,
                    "replicate": None,
                    "seed": None,
                }
            ]
            for sigma in run_config["sigmas"]:
                for replicate in range(run_config["starts_per_sigma"]):
                    starts.append(
                        {
                            "kind": "lognormal",
                            "sigma": float(sigma),
                            "replicate": int(replicate),
                            "seed": derive_start_seed(
                                run_config["base_seed"],
                                configuration_id,
                                anchor,
                                float(sigma),
                                replicate,
                            ),
                        }
                    )
            for start in starts:
                task_identity = {**case_payload, "start": start}
                tasks.append(
                    {
                        "task_index": -1,
                        "task_id": _stable_identifier("task", task_identity),
                        "case_id": case_id,
                        **case_payload,
                        "start_spec": start,
                    }
                )
    tasks.sort(key=lambda row: row["task_id"])
    # Keep the canonical uniform task first within each case while retaining a
    # stable cross-case order.  This also makes partial checkpoints useful.
    tasks.sort(
        key=lambda row: (
            row["case_id"],
            0 if row["start_spec"]["kind"] == "uniform" else 1,
            row["task_id"],
        )
    )
    for index, task in enumerate(tasks):
        task["task_index"] = index
    return tasks


def _edge_weight_digest(
    faces: Sequence[Sequence[int]], weights: Mapping[tuple[int, int], float] | None
) -> tuple[str, float, float]:
    edges = sorted(
        {
            tuple(sorted((int(u), int(v))))
            for face in faces
            for u, v in zip(face, (*face[1:], face[0]))
        }
    )
    values = np.asarray(
        [1.0 if weights is None else float(weights[edge]) for edge in edges],
        dtype="<f8",
    )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Tutte edge weights must be finite and strictly positive")
    edge_array = np.asarray(edges, dtype="<i8")
    digest = _sha256_bytes(edge_array.tobytes() + values.tobytes())
    return f"sha256:{digest}", float(values.min()), float(values.max())


def _embedding_digest(points: Any) -> str:
    array = np.asarray(points, dtype="<f8")
    if array.ndim != 2 or array.shape[1] != 2 or not np.all(np.isfinite(array)):
        raise ValueError("weighted Tutte embedding must be a finite Vx2 array")
    return f"sha256:{_sha256_bytes(array.tobytes())}"


def _cp1_pairs(vertices: Any) -> list[list[float] | None]:
    try:
        array = np.asarray(vertices)
    except ValueError:
        # NumPy 2 rejects the intentionally ragged ``[None, [x, y], ...]``
        # CP1 representation unless object dtype is explicit.
        array = np.asarray(vertices, dtype=object)
    if array.ndim == 2 and array.shape[1] == 2:
        pairs: list[list[float] | None] = [
            [float(row[0]), float(row[1])] for row in np.asarray(array, dtype=float)
        ]
    else:
        pairs = []
        for vertex in list(vertices):
            if vertex is None:
                pairs.append(None)
            elif isinstance(vertex, complex) or np.iscomplexobj(vertex):
                coordinate = complex(vertex)
                pairs.append([float(coordinate.real), float(coordinate.imag)])
            else:
                pair = list(vertex)
                if len(pair) != 2:
                    raise ValueError(
                        "CP1 vertex must be complex, None, or a coordinate pair"
                    )
                pairs.append([float(pair[0]), float(pair[1])])
    for pair in pairs:
        if pair is not None and not all(math.isfinite(value) for value in pair):
            raise ValueError("endpoint contains a non-finite CP1 coordinate")
    return pairs


def _root_sphere_coordinates(vertices: Any) -> np.ndarray:
    pairs = _cp1_pairs(vertices)
    sphere = np.empty((len(pairs), 3), dtype=float)
    for index, pair in enumerate(pairs):
        if pair is None:
            sphere[index] = (0.0, 0.0, 1.0)
            continue
        x, y = pair
        scale = max(abs(x), abs(y))
        if scale <= 1.0:
            radius = math.hypot(x, y)
            radius_squared = radius * radius
            denominator = 1.0 + radius_squared
            sphere[index] = (
                2.0 * x / denominator,
                2.0 * y / denominator,
                (radius_squared - 1.0) / denominator,
            )
        else:
            scaled_x = x / scale
            scaled_y = y / scale
            scaled_radius = math.hypot(scaled_x, scaled_y)
            inverse_radius = (1.0 / scale) / scaled_radius
            inverse_squared = inverse_radius * inverse_radius
            denominator = 1.0 + inverse_squared
            sphere[index] = (
                2.0 * (scaled_x / scaled_radius) * inverse_radius / denominator,
                2.0 * (scaled_y / scaled_radius) * inverse_radius / denominator,
                (1.0 - inverse_squared) / denominator,
            )
    return sphere


def _root_fingerprint(vertices: Any) -> str:
    """Return an exact digest for a stored cluster representative.

    Tolerance is deliberately absent.  Rounding coordinates onto a tolerance
    grid is not a metric clustering operation: close roots can straddle a grid
    wall and distant roots can share a cell.  Cluster membership is assigned
    below by direct chordal-distance comparisons.
    """
    sphere = np.asarray(_root_sphere_coordinates(vertices), dtype="<f8")
    return f"sha256:{_sha256_bytes(sphere.tobytes())}"


def _root_chordal_distance(left: Any, right: Any) -> float:
    left_sphere = _root_sphere_coordinates(left)
    right_sphere = _root_sphere_coordinates(right)
    if left_sphere.shape != right_sphere.shape:
        raise ValueError("root representatives have different label counts")
    if not len(left_sphere):
        raise ValueError("root representative must contain at least one vertex")
    return max(
        math.hypot(*(float(value) for value in left - right))
        for left, right in zip(left_sphere, right_sphere)
    )


def _assign_root_cluster(
    vertices: Any,
    representatives: list[dict[str, Any]],
    tolerance: float,
) -> str:
    """Assign a root to the first task-ordered representative within tolerance."""
    pairs = _cp1_pairs(vertices)
    for representative in representatives:
        if (
            _root_chordal_distance(pairs, representative["representative_vertices_cp1"])
            <= tolerance
        ):
            return str(representative["fingerprint"])
    fingerprint = _root_fingerprint(pairs)
    if any(row["fingerprint"] == fingerprint for row in representatives):
        raise RuntimeError("SHA-256 collision between distinct root representatives")
    representatives.append(
        {
            "fingerprint": fingerprint,
            "representative_vertices_cp1": pairs,
        }
    )
    return fingerprint


_COORDINATE_KEYS = frozenset(
    {
        "vertices",
        "vertices_cp1",
        "normalized_vertices_cp1",
        "coordinates",
        "endpoint_coordinates",
        "initial_embedding",
        "initial_vertices_cp1",
        "packed",
    }
)


def _strip_coordinate_payload(value: Any) -> Any:
    """Remove bulky coordinates while retaining scalar diagnostics."""
    if isinstance(value, Mapping):
        return {
            str(key): _strip_coordinate_payload(item)
            for key, item in value.items()
            if str(key) not in _COORDINATE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_strip_coordinate_payload(item) for item in value]
    return value


def _compact_audit_payload(value: Any) -> dict[str, Any] | None:
    """Keep fixed audit metrics and witnesses, not O(V)/O(E) detail arrays."""
    if value is None:
        return None
    payload = _jsonable(value.as_dict() if hasattr(value, "as_dict") else value)
    if not isinstance(payload, dict):
        raise TypeError("zero-shear audit payload must be an object")

    topology = payload.get("topology")
    if isinstance(topology, dict):
        topology.pop("vertex_labels", None)

    support = payload.get("support")
    if isinstance(support, dict):
        face_rows = support.pop("faces", None)
        if isinstance(face_rows, list):
            support["face_diagnostic_count"] = len(face_rows)

    shear = payload.get("shear")
    if isinstance(shear, dict):
        values = shear.pop("values", None)
        if isinstance(values, list):
            shear["value_count"] = len(values)

    bending = payload.get("bending")
    if isinstance(bending, dict):
        angles = bending.pop("angles_radians", None)
        if isinstance(angles, list):
            bending["angle_count"] = len(angles)

    qhull = payload.get("qhull_agreement")
    if isinstance(qhull, dict):
        for field, count_field in (
            ("missing_faces", "missing_face_count"),
            ("extra_faces", "extra_face_count"),
        ):
            rows = qhull.pop(field, None)
            if isinstance(rows, list):
                qhull[count_field] = len(rows)
    return payload


def _compact_solve_result(value: Any) -> dict[str, Any]:
    payload = _jsonable(value.as_dict() if hasattr(value, "as_dict") else value)
    if not isinstance(payload, dict):
        raise TypeError("Tutte solve result payload must be an object")
    payload["audit"] = _compact_audit_payload(payload.get("audit"))
    compact = _strip_coordinate_payload(payload)
    if not isinstance(compact, dict):  # pragma: no cover - structural assertion
        raise TypeError("compacted Tutte solve result must be an object")
    return _jsonable(compact)


def _trajectory_summary(
    rows: Sequence[Mapping[str, Any]], *, trajectory_kind: str
) -> dict[str, Any]:
    flags: list[bool] = []
    iterations: list[int] = []
    strict_flags: list[bool] = []
    for index, row in enumerate(rows):
        flag = row.get("in_target_convex_cell")
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError("recorded trajectory row is missing in_target_convex_cell")
        strict_flag = row.get("strict_target_zero_shear", False)
        if not isinstance(strict_flag, (bool, np.bool_)):
            raise ValueError(
                "recorded trajectory row has invalid strict_target_zero_shear"
            )
        iteration = row.get("iteration", index)
        if not isinstance(iteration, (int, np.integer)) or isinstance(iteration, bool):
            raise ValueError("recorded trajectory row has an invalid iteration")
        flags.append(bool(flag))
        strict_flags.append(bool(strict_flag))
        iterations.append(int(iteration))

    first_position = next((index for index, flag in enumerate(flags) if flag), None)
    exit_position = (
        next(
            (
                index
                for index in range(first_position + 1, len(flags))
                if not flags[index]
            ),
            None,
        )
        if first_position is not None
        else None
    )
    return {
        "trajectory_kind": str(trajectory_kind),
        "recorded_iterate_count": len(rows),
        "target_cell_iterate_count": sum(flags),
        "strict_target_iterate_count": sum(strict_flags),
        "first_target_cell_iteration": (
            iterations[first_position] if first_position is not None else None
        ),
        "first_exit_after_target_entry_iteration": (
            iterations[exit_position] if exit_position is not None else None
        ),
        "exited_after_target_entry": exit_position is not None,
        "final_in_target_convex_cell": flags[-1] if flags else None,
    }


def _load_core_api():
    from ideal_poly_volume_toolkit.rivin_zero_shear import (
        audit_zero_shear_realization,
        lognormal_edge_weights,
        solve_zero_shear_from_tutte,
        weighted_tutte_embedding,
    )

    return (
        audit_zero_shear_realization,
        weighted_tutte_embedding,
        lognormal_edge_weights,
        solve_zero_shear_from_tutte,
    )


def _run_task(
    task: Mapping[str, Any],
    configuration: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> dict[str, Any]:
    (
        _audit_zero_shear_realization,
        weighted_tutte_embedding,
        lognormal_edge_weights,
        solve_zero_shear_from_tutte,
    ) = _load_core_api()

    faces = [tuple(map(int, face)) for face in configuration["input"]["triangles"]]
    distinguished_face = tuple(map(int, task["distinguished_face"]))
    start_spec = task["start_spec"]
    if start_spec["kind"] == "uniform":
        weights = None
    else:
        weights = lognormal_edge_weights(
            faces,
            sigma=float(start_spec["sigma"]),
            seed=int(start_spec["seed"]),
        )
    weight_sha256, minimum_weight, maximum_weight = _edge_weight_digest(faces, weights)
    initial_embedding = weighted_tutte_embedding(
        faces, distinguished_face, edge_weights=weights
    )
    initial_embedding_sha256 = _embedding_digest(initial_embedding)

    result = solve_zero_shear_from_tutte(
        faces,
        distinguished_face,
        edge_weights=weights,
        max_nfev=int(run_config["max_nfev"]),
        record_trajectory=bool(run_config["record_trajectory"]),
        shear_tolerance=float(run_config["shear_tolerance"]),
        support_tolerance=float(run_config["support_tolerance"]),
        bending_tolerance=float(run_config["bending_tolerance"]),
        # Strict projective support plus coherent closed topology certifies the
        # convex embedded target.  The all-pairs spherical crossing diagnostic
        # is redundant here and dominates 300-vertex audit time.
        check_spherical_embedding=False,
        xtol=float(run_config["xtol"]),
        ftol=float(run_config["ftol"]),
        gtol=float(run_config["gtol"]),
    )
    classification = str(result.classification)
    result_payload = _compact_solve_result(result)
    trajectory_summary = (
        _trajectory_summary(
            result_payload["trajectory"],
            trajectory_kind=str(result.trajectory_kind),
        )
        if run_config["record_trajectory"]
        else None
    )
    disk_vertices = result.vertices_cp1
    normalized_vertices = result.normalized_vertices_cp1
    # The solver works in the finite disk chart for numerical stability, but
    # normalized endpoints make opt-in coordinates and root identities useful
    # to downstream consumers: the distinguished face is (infinity, 0, 1).
    vertices = normalized_vertices if normalized_vertices is not None else disk_vertices
    root_coordinates = None
    if (
        vertices is not None
        and classification in TARGET_OUTCOMES | WRONG_ZERO_SHEAR_OUTCOMES
    ):
        # This private worker payload is consumed by the parent before the
        # task row is serialized.  Only one representative per discovered
        # cluster is retained in the checkpoint.
        root_coordinates = _cp1_pairs(vertices)
    endpoint_coordinates = (
        _cp1_pairs(vertices)
        if vertices is not None and run_config["include_endpoint_coordinates"]
        else None
    )
    return {
        "outcome": classification,
        "start": {
            **_jsonable(start_spec),
            "weight_sha256": weight_sha256,
            "minimum_weight": minimum_weight,
            "maximum_weight": maximum_weight,
            "initial_embedding_sha256": initial_embedding_sha256,
        },
        "core_result": result_payload,
        "trajectory_summary": trajectory_summary,
        "root_coordinates": root_coordinates,
        "endpoint_coordinates": endpoint_coordinates,
    }


def _task_process_entry(
    task: dict[str, Any],
    configuration: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    started_at = _utc_now()
    started = time.perf_counter()
    try:
        result = _run_task(task, configuration, run_config)
        return {
            "ok": True,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "elapsed_seconds": float(time.perf_counter() - started),
            "result": result,
        }
    except BaseException as exc:
        return {
            "ok": False,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "elapsed_seconds": float(time.perf_counter() - started),
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


_WORKER_CONFIGURATIONS: dict[str, dict[str, Any]] = {}
_WORKER_RUN_CONFIG: dict[str, Any] = {}


def _initialize_task_pool(
    configurations: dict[str, dict[str, Any]], run_config: dict[str, Any]
) -> None:
    global _WORKER_CONFIGURATIONS, _WORKER_RUN_CONFIG
    _WORKER_CONFIGURATIONS = configurations
    _WORKER_RUN_CONFIG = run_config


def _pooled_task_process_entry(task: dict[str, Any]) -> dict[str, Any]:
    return _task_process_entry(
        task,
        _WORKER_CONFIGURATIONS[task["configuration_id"]],
        _WORKER_RUN_CONFIG,
    )


def _take_ready_task_prefix(
    pending: Sequence[Mapping[str, Any]],
    ready_envelopes: dict[str, Mapping[str, Any]],
    next_commit: int,
) -> tuple[list[tuple[Mapping[str, Any], Mapping[str, Any]]], int]:
    """Take only the task-index-ordered ready prefix from a worker buffer."""
    committable: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    while next_commit < len(pending):
        task = pending[next_commit]
        envelope = ready_envelopes.pop(str(task["task_id"]), None)
        if envelope is None:
            break
        committable.append((task, envelope))
        next_commit += 1
    return committable, next_commit


def _task_row(task: Mapping[str, Any], envelope: Mapping[str, Any]) -> dict[str, Any]:
    base = {
        **_jsonable(task),
        "started_at": str(envelope.get("started_at", _utc_now())),
        "finished_at": str(envelope.get("finished_at", _utc_now())),
        "elapsed_seconds": float(envelope.get("elapsed_seconds", 0.0)),
        "status": "error",
        "outcome": "error",
        "start": None,
        "core_result": None,
        "trajectory_summary": None,
        "root_fingerprint": None,
        "endpoint_coordinates": None,
        "error": {
            "type": str(envelope.get("error_type", "WorkerError")),
            "message": str(envelope.get("message", "worker failed"))[:2000],
        },
    }
    if envelope.get("timed_out"):
        base.update(
            status="timed_out",
            outcome="timed_out",
            error={
                "type": "CampaignWallTimeout",
                "message": "task did not finish before the campaign wall limit",
            },
        )
    elif envelope.get("ok"):
        result = envelope["result"]
        base.update(
            status="completed",
            outcome=str(result["outcome"]),
            start=_jsonable(result["start"]),
            core_result=_jsonable(result["core_result"]),
            trajectory_summary=_jsonable(result.get("trajectory_summary")),
            root_fingerprint=result.get("root_fingerprint"),
            endpoint_coordinates=_jsonable(result.get("endpoint_coordinates")),
            error=None,
        )
    return _jsonable(base)


def _prepare_envelope_root_cluster(
    envelope: Mapping[str, Any],
    representatives: list[dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    """Consume a worker-only root payload and assign its persisted cluster id."""
    prepared = dict(envelope)
    if not prepared.get("ok"):
        return prepared
    result = dict(prepared["result"])
    root_coordinates = result.pop("root_coordinates", None)
    if root_coordinates is not None:
        result["root_fingerprint"] = _assign_root_cluster(
            root_coordinates, representatives, tolerance
        )
    else:
        result.setdefault("root_fingerprint", None)
    prepared["result"] = result
    return prepared


def _hardness_for_case(rows: Sequence[Mapping[str, Any]]) -> tuple[str, list[float]]:
    uniform = [row for row in rows if row["start_spec"]["kind"] == "uniform"]
    if len(uniform) != 1:
        return "incomplete", [-1.0, 0.0, 0.0, 0.0]
    uniform_outcome = uniform[0]["outcome"]
    weighted = [row for row in rows if row["start_spec"]["kind"] != "uniform"]
    outcomes = [row["outcome"] for row in rows]
    wrong_count = sum(outcome in WRONG_ZERO_SHEAR_OUTCOMES for outcome in outcomes)
    target_count = sum(outcome in TARGET_OUTCOMES for outcome in outcomes)
    root_count = len(
        {
            row["root_fingerprint"]
            for row in rows
            if row.get("root_fingerprint") is not None
        }
    )
    total = len(rows)
    if uniform_outcome in WRONG_ZERO_SHEAR_OUTCOMES:
        label, tier = "canonical_wrong_root", 4
    elif uniform_outcome not in TARGET_OUTCOMES:
        label, tier = "canonical_nonconvergence", 3
    elif any(row["outcome"] in WRONG_ZERO_SHEAR_OUTCOMES for row in weighted):
        label, tier = "multibasin", 2
    elif any(row["outcome"] not in TARGET_OUTCOMES for row in weighted):
        label, tier = "fragile", 1
    else:
        label, tier = "robust", 0
    key = [
        float(tier),
        float(wrong_count / total) if total else 0.0,
        float(1.0 - target_count / total) if total else 0.0,
        float(root_count),
    ]
    return label, key


def _aggregate_cases(
    task_rows: Sequence[Mapping[str, Any]], planned_tasks: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    planned_by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    rows_by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for task in planned_tasks:
        planned_by_case[task["case_id"]].append(task)
    for row in task_rows:
        rows_by_case[row["case_id"]].append(row)
    cases: list[dict[str, Any]] = []
    for case_id in sorted(planned_by_case):
        planned = planned_by_case[case_id]
        rows = sorted(rows_by_case.get(case_id, []), key=lambda row: row["task_id"])
        first = planned[0]
        outcomes = Counter(str(row["outcome"]) for row in rows)
        uniform_rows = [row for row in rows if row["start_spec"]["kind"] == "uniform"]
        clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get("root_fingerprint") is not None:
                clusters[str(row["root_fingerprint"])].append(row)
        cluster_rows = []
        for fingerprint in sorted(clusters):
            members = clusters[fingerprint]
            cluster_rows.append(
                {
                    "fingerprint": fingerprint,
                    "size": len(members),
                    "outcome_counts": dict(
                        sorted(Counter(str(row["outcome"]) for row in members).items())
                    ),
                    "task_ids": sorted(str(row["task_id"]) for row in members),
                }
            )
        target_count = sum(outcomes[outcome] for outcome in TARGET_OUTCOMES)
        if len(rows) == len(planned):
            hardness_class, hardness_key = _hardness_for_case(rows)
        else:
            hardness_class, hardness_key = "incomplete", [-1.0, 0.0, 0.0, 0.0]
        cases.append(
            {
                "case_id": case_id,
                "configuration_id": first["configuration_id"],
                "distinguished_face": first["distinguished_face"],
                "planned_task_count": len(planned),
                "finished_task_count": len(rows),
                "outcome_counts": dict(sorted(outcomes.items())),
                "uniform_outcome": (
                    str(uniform_rows[0]["outcome"]) if len(uniform_rows) == 1 else None
                ),
                "target_success_count": int(target_count),
                "target_success_fraction": (
                    float(target_count / len(rows)) if rows else 0.0
                ),
                "root_clusters": cluster_rows,
                "hardness_class": hardness_class,
                "hardness_key": hardness_key,
                "hardness_rank": None,
            }
        )
    ranked = sorted(
        (case for case in cases if case["hardness_class"] != "incomplete"),
        key=lambda case: (tuple(case["hardness_key"]), case["case_id"]),
        reverse=True,
    )
    for rank, case in enumerate(ranked, 1):
        case["hardness_rank"] = rank
    return cases


def _run_config_hash(run_config: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        json.dumps(
            _jsonable(run_config), sort_keys=True, separators=(",", ":")
        ).encode()
    )


def _source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        CORE_PATH,
        HOLONOMY_PATH,
        SCHEMA_PATH,
        INPUT_SCHEMA_PATH,
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required source file is missing: {missing[0]}")
    return {
        str(path.relative_to(REPOSITORY_ROOT)): _sha256_file(path) for path in paths
    }


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ("ideal-poly-volume-toolkit", "numpy", "scipy", "jsonschema"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _runtime_provenance() -> dict[str, Any]:
    """Return numerical-runtime fields that must remain fixed across resume."""
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "dependency_versions": _dependency_versions(),
        "thread_environment": {
            variable: os.environ.get(variable) for variable in THREAD_ENVIRONMENT
        },
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if len(commit) == 40 else None


def _initial_document(
    input_path: Path,
    input_sha256: str,
    input_document: Mapping[str, Any],
    selected_ids: Sequence[str],
    run_config: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    planned_tasks: Sequence[Mapping[str, Any]],
    invocation: Sequence[str],
) -> dict[str, Any]:
    timestamp = _utc_now()
    runtime_provenance = _runtime_provenance()
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "generator": {
            "name": GENERATOR_NAME,
            "version": GENERATOR_VERSION,
            "source_sha256": source_hashes[GENERATOR_NAME],
        },
        "status": "in_progress",
        "created_at": timestamp,
        "updated_at": timestamp,
        "input": {
            "path": str(input_path),
            "sha256": input_sha256,
            "schema_version": input_document["schema_version"],
            "selected_configuration_ids": list(selected_ids),
        },
        "run_config": _jsonable(run_config),
        "run_config_sha256": _run_config_hash(run_config),
        "provenance": {
            "invocation": list(invocation),
            "git_commit": _git_commit(),
            "source_files": dict(source_hashes),
            **runtime_provenance,
        },
        "planned_task_count": len(planned_tasks),
        "root_representatives": [],
        "tasks": [],
        "cases": _aggregate_cases([], planned_tasks),
        "summary": {
            "planned": len(planned_tasks),
            "finished": 0,
            "completed": 0,
            "errors": 0,
            "timed_out": 0,
            "pending": len(planned_tasks),
            "outcome_counts": {},
        },
        "timings": {
            "cumulative_active_seconds": 0.0,
            "last_invocation_seconds": 0.0,
        },
    }


def _refresh_document(
    document: dict[str, Any], planned_tasks: Sequence[Mapping[str, Any]]
) -> None:
    document["tasks"].sort(key=lambda row: int(row["task_index"]))
    rows = document["tasks"]
    statuses = Counter(str(row["status"]) for row in rows)
    outcomes = Counter(str(row["outcome"]) for row in rows)
    referenced_roots = {
        str(row["root_fingerprint"])
        for row in rows
        if row.get("root_fingerprint") is not None
    }
    document["root_representatives"] = [
        representative
        for representative in document["root_representatives"]
        if representative["fingerprint"] in referenced_roots
    ]
    document["cases"] = _aggregate_cases(rows, planned_tasks)
    document["summary"] = {
        "planned": len(planned_tasks),
        "finished": len(rows),
        "completed": int(statuses["completed"]),
        "errors": int(statuses["error"]),
        "timed_out": int(statuses["timed_out"]),
        "pending": max(0, len(planned_tasks) - len(rows)),
        "outcome_counts": dict(sorted(outcomes.items())),
    }
    document["status"] = _status_for_summary(document["summary"])
    document["updated_at"] = _utc_now()


def _status_for_summary(summary: Mapping[str, Any]) -> str:
    if int(summary["pending"]) > 0:
        return "in_progress"
    if int(summary["timed_out"]) > 0:
        return "exhausted"
    if int(summary["errors"]) > 0:
        return "failed"
    return "complete"


def _stored_case_plan(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand the persisted case descriptors into an aggregate-only plan."""
    cases = document.get("cases")
    if not isinstance(cases, list):
        raise ValueError("checkpoint cases must be an array")
    case_ids = [case.get("case_id") for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("checkpoint contains duplicate case ids")
    planned: list[dict[str, Any]] = []
    for case in cases:
        count = case.get("planned_task_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("checkpoint case has invalid planned task count")
        descriptor = {
            "case_id": case.get("case_id"),
            "configuration_id": case.get("configuration_id"),
            "distinguished_face": case.get("distinguished_face"),
        }
        planned.extend(dict(descriptor) for _ in range(count))
    return planned


def _validate_root_representatives(document: Mapping[str, Any]) -> None:
    representatives = document.get("root_representatives")
    if not isinstance(representatives, list):
        raise ValueError("checkpoint root_representatives must be an array")
    tolerance = float(document["run_config"]["root_cluster_tolerance"])
    fingerprints: list[str] = []
    for index, representative in enumerate(representatives):
        fingerprint = representative.get("fingerprint")
        vertices = representative.get("representative_vertices_cp1")
        if fingerprint != _root_fingerprint(vertices):
            raise ValueError(
                f"root representative {index} fingerprint does not match its coordinates"
            )
        for earlier in representatives[:index]:
            if (
                _root_chordal_distance(vertices, earlier["representative_vertices_cp1"])
                <= tolerance
            ):
                raise ValueError(
                    "root representatives are not separated by the cluster tolerance"
                )
        fingerprints.append(str(fingerprint))
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("checkpoint contains duplicate root representatives")
    referenced = {
        str(row["root_fingerprint"])
        for row in document.get("tasks", [])
        if row.get("root_fingerprint") is not None
    }
    if referenced != set(fingerprints):
        raise ValueError(
            "checkpoint root representatives do not match task root fingerprints"
        )


def _validate_document_invariants(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("not a Tutte zero-shear audit v1 document")
    if document.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Tutte zero-shear audit protocol mismatch")
    _validate_run_config(document["run_config"])
    if document.get("run_config_sha256") != _run_config_hash(document["run_config"]):
        raise ValueError("stored run_config hash does not match its contents")
    source_files = document.get("provenance", {}).get("source_files", {})
    if document.get("generator", {}).get("source_sha256") != source_files.get(
        GENERATOR_NAME
    ):
        raise ValueError("generator source hash does not match provenance source_files")
    tasks = document.get("tasks", [])
    task_ids = [row.get("task_id") for row in tasks]
    task_indices = [row.get("task_index") for row in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("checkpoint contains duplicate task ids")
    if len(set(task_indices)) != len(task_indices):
        raise ValueError("checkpoint contains duplicate task indices")
    if task_indices != sorted(task_indices):
        raise ValueError("checkpoint tasks must be sorted by task index")
    if any(
        index < 0 or index >= document["planned_task_count"] for index in task_indices
    ):
        raise ValueError("checkpoint contains an out-of-range task index")
    expected_summary = {
        "planned": document["planned_task_count"],
        "finished": len(tasks),
        "completed": sum(row["status"] == "completed" for row in tasks),
        "errors": sum(row["status"] == "error" for row in tasks),
        "timed_out": sum(row["status"] == "timed_out" for row in tasks),
        "pending": document["planned_task_count"] - len(tasks),
        "outcome_counts": dict(
            sorted(Counter(row["outcome"] for row in tasks).items())
        ),
    }
    if document.get("summary") != expected_summary:
        raise ValueError("checkpoint summary does not match task rows")
    expected_status = _status_for_summary(expected_summary)
    if document.get("status") != expected_status:
        raise ValueError(
            f"checkpoint status does not match task rows; expected {expected_status!r}"
        )
    stored_plan = _stored_case_plan(document)
    if len(stored_plan) != document["planned_task_count"]:
        raise ValueError("checkpoint case plan does not match planned task count")
    known_case_ids = {row["case_id"] for row in stored_plan}
    if any(row.get("case_id") not in known_case_ids for row in tasks):
        raise ValueError("checkpoint task refers to an unknown case")
    expected_cases = _aggregate_cases(tasks, stored_plan)
    if document.get("cases") != expected_cases:
        raise ValueError("checkpoint case aggregates do not match task rows")
    _validate_root_representatives(document)


def _validate_checkpoint_matches_plan(
    document: Mapping[str, Any], planned_tasks: Sequence[Mapping[str, Any]]
) -> None:
    if document.get("planned_task_count") != len(planned_tasks):
        raise ValueError("checkpoint planned task count does not match this study")
    identity_fields = (
        "task_index",
        "task_id",
        "case_id",
        "configuration_id",
        "distinguished_face",
        "start_spec",
    )
    for row in document["tasks"]:
        task_index = int(row["task_index"])
        expected = planned_tasks[task_index]
        if any(row.get(field) != expected.get(field) for field in identity_fields):
            raise ValueError(
                f"checkpoint task {task_index} does not match the deterministic plan"
            )
    expected_cases = _aggregate_cases(document["tasks"], planned_tasks)
    if document.get("cases") != expected_cases:
        raise ValueError("checkpoint cases do not match the deterministic plan")


def validate_audit_document(
    document: dict[str, Any], schema_path: str | Path = SCHEMA_PATH
) -> None:
    _validate_document_invariants(document)
    strict_json_bytes(document)
    _validate_with_schema(document, Path(schema_path))


def _validate_run_config(run_config: Mapping[str, Any]) -> None:
    if not (1 <= run_config["workers"] <= MAX_WORKERS):
        raise ValueError(f"--workers must be between 1 and {MAX_WORKERS}")
    if run_config["blas_threads_per_worker"] != 1:
        raise ValueError("Tutte basin workers must use exactly one BLAS thread")
    if run_config["anchors_per_configuration"] < 1:
        raise ValueError("--anchors-per-config must be positive")
    explicit_faces = run_config["explicit_distinguished_faces"]
    materialized_explicit: list[tuple[int, int, int]] = []
    for index, raw_face in enumerate(explicit_faces):
        try:
            face = tuple(int(vertex) for vertex in raw_face)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"explicit distinguished face {index} must contain integer labels"
            ) from exc
        if len(face) != 3 or len(set(face)) != 3 or any(vertex < 0 for vertex in face):
            raise ValueError(
                f"explicit distinguished face {index} must contain three "
                "distinct nonnegative labels"
            )
        materialized_explicit.append(face)
    if len(set(materialized_explicit)) != len(materialized_explicit):
        raise ValueError("explicit distinguished faces must not contain duplicates")
    if explicit_faces and run_config["anchors_per_configuration"] != len(
        explicit_faces
    ):
        raise ValueError(
            "anchors_per_configuration must equal the number of explicit "
            "distinguished faces"
        )
    if run_config["starts_per_sigma"] < 0:
        raise ValueError("--starts-per-sigma must be nonnegative")
    if run_config["max_nfev"] < 1:
        raise ValueError("--max-nfev must be positive")
    if (
        not math.isfinite(run_config["max_wall_seconds"])
        or run_config["max_wall_seconds"] <= 0
    ):
        raise ValueError("--max-wall-seconds must be finite and positive")
    if run_config["base_seed"] < 0 or run_config["base_seed"] >= 1 << 63:
        raise ValueError("--seed must lie in [0, 2**63)")
    sigmas = run_config["sigmas"]
    if len(set(sigmas)) != len(sigmas):
        raise ValueError("--sigmas must not contain duplicates")
    if any(not math.isfinite(sigma) or sigma <= 0.0 for sigma in sigmas):
        raise ValueError("--sigmas must contain finite positive values")
    tolerance = run_config["root_cluster_tolerance"]
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("--root-cluster-tolerance must be finite and positive")
    for key in ("shear_tolerance", "bending_tolerance", "xtol", "ftol", "gtol"):
        if not math.isfinite(run_config[key]) or run_config[key] <= 0.0:
            raise ValueError(f"{key} must be finite and positive")
    support_tolerance = run_config["support_tolerance"]
    if not math.isfinite(support_tolerance) or support_tolerance < 0.0:
        raise ValueError("support_tolerance must be finite and nonnegative")


def _checkpoint(
    output_path: Path,
    document: dict[str, Any],
    planned_tasks: Sequence[Mapping[str, Any]],
    base_elapsed: float,
    invocation_started: float,
) -> None:
    invocation_elapsed = float(time.monotonic() - invocation_started)
    document["timings"] = {
        "cumulative_active_seconds": float(base_elapsed + invocation_elapsed),
        "last_invocation_seconds": invocation_elapsed,
    }
    _refresh_document(document, planned_tasks)
    _validate_document_invariants(document)
    atomic_write_json(output_path, document)


def run_study(
    input_path: str | Path,
    output_path: str | Path,
    run_config: dict[str, Any],
    *,
    requested_ids: Sequence[str] | None = None,
    resume: bool = False,
    overwrite: bool = False,
    invocation: Sequence[str] | None = None,
    use_subprocesses: bool = True,
    task_runner: Callable[
        [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], dict[str, Any]
    ] = _run_task,
) -> dict[str, Any]:
    """Run or resume a bounded audit, checkpointing every finished task."""
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    _validate_run_config(run_config)
    input_path = Path(input_path).resolve()
    output_path = Path(output_path)
    corpus = load_arithmetic_corpus(input_path)
    configurations = select_configurations(corpus, requested_ids)
    # Workers need only the labeled face list.  Exact holonomy certificates
    # can dominate a large corpus file, so do not replicate those unrelated
    # payloads into every spawned process.
    by_id = {
        entry["id"]: {
            "id": entry["id"],
            "input": {"triangles": entry["input"]["triangles"]},
        }
        for entry in configurations
    }
    selected_ids = [entry["id"] for entry in configurations]
    if list(run_config["selected_configuration_ids"]) != selected_ids:
        raise ValueError("run_config selected_configuration_ids do not match selection")
    planned_tasks = build_tasks(configurations, run_config)
    input_sha256 = _sha256_file(input_path)
    source_hashes = _source_hashes()
    invocation = list(sys.argv if invocation is None else invocation)

    if resume:
        if not output_path.is_file():
            raise FileNotFoundError(f"cannot resume missing audit: {output_path}")
        document = load_json_strict(output_path)
        validate_audit_document(document)
        _validate_checkpoint_matches_plan(document, planned_tasks)
        if document["run_config_sha256"] != _run_config_hash(run_config):
            raise ValueError("resume run config does not match the checkpoint")
        if document["input"]["sha256"] != input_sha256:
            raise ValueError("resume refused because the input corpus changed")
        if document["provenance"]["source_files"] != source_hashes:
            raise ValueError("resume refused because audit/core source hashes changed")
        current_runtime = _runtime_provenance()
        for field, current_value in current_runtime.items():
            if document["provenance"].get(field) != current_value:
                raise ValueError(
                    f"resume refused because runtime provenance field {field!r} changed"
                )
        if document["status"] in {"complete", "exhausted", "failed"}:
            return document
    else:
        document = _initial_document(
            input_path,
            input_sha256,
            corpus,
            selected_ids,
            run_config,
            source_hashes,
            planned_tasks,
            invocation,
        )
        _validate_document_invariants(document)
        if overwrite:
            if output_path.exists():
                atomic_write_json(output_path, document)
            else:
                exclusive_create_json(output_path, document)
        else:
            exclusive_create_json(output_path, document)

    finished_ids = {row["task_id"] for row in document["tasks"]}
    pending = [task for task in planned_tasks if task["task_id"] not in finished_ids]
    base_elapsed = float(document["timings"]["cumulative_active_seconds"])
    invocation_started = time.monotonic()
    wall_remaining = float(run_config["max_wall_seconds"]) - base_elapsed
    deadline = invocation_started + max(0.0, wall_remaining)

    def append_envelopes(
        task_envelopes: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    ) -> None:
        for task, envelope in task_envelopes:
            prepared = _prepare_envelope_root_cluster(
                envelope,
                document["root_representatives"],
                float(run_config["root_cluster_tolerance"]),
            )
            document["tasks"].append(_task_row(task, prepared))
        if not task_envelopes:
            return
        _checkpoint(
            output_path, document, planned_tasks, base_elapsed, invocation_started
        )

    def append_envelope(task: Mapping[str, Any], envelope: Mapping[str, Any]) -> None:
        append_envelopes([(task, envelope)])

    def append_wall_timeouts(tasks: Sequence[Mapping[str, Any]]) -> None:
        append_envelopes(
            [
                (
                    task,
                    {
                        "timed_out": True,
                        "started_at": _utc_now(),
                        "finished_at": _utc_now(),
                        "elapsed_seconds": 0.0,
                    },
                )
                for task in tasks
            ]
        )

    try:
        if (
            use_subprocesses
            and task_runner is _run_task
            and pending
            and wall_remaining > 0.0
        ):
            context = multiprocessing.get_context("spawn")
            worker_count = int(run_config["workers"])
            pool = context.Pool(
                processes=worker_count,
                initializer=_initialize_task_pool,
                initargs=(by_id, run_config),
            )
            outstanding: dict[str, tuple[dict[str, Any], Any]] = {}
            ready_envelopes: dict[str, Mapping[str, Any]] = {}
            next_pending = 0
            next_commit = 0

            def fill_worker_slots() -> None:
                nonlocal next_pending
                while len(outstanding) < worker_count and next_pending < len(pending):
                    task = pending[next_pending]
                    next_pending += 1
                    outstanding[task["task_id"]] = (
                        task,
                        pool.apply_async(_pooled_task_process_entry, (task,)),
                    )

            fill_worker_slots()
            try:
                while outstanding and time.monotonic() < deadline:
                    progressed = False
                    for task_id, (task, asynchronous) in list(outstanding.items()):
                        if not asynchronous.ready():
                            continue
                        progressed = True
                        outstanding.pop(task_id)
                        try:
                            envelope = asynchronous.get()
                        except BaseException as exc:
                            envelope = {
                                "ok": False,
                                "started_at": _utc_now(),
                                "finished_at": _utc_now(),
                                "elapsed_seconds": 0.0,
                                "error_type": type(exc).__name__,
                                "message": str(exc),
                            }
                        ready_envelopes[task_id] = envelope
                        fill_worker_slots()
                    committable, next_commit = _take_ready_task_prefix(
                        pending, ready_envelopes, next_commit
                    )
                    append_envelopes(committable)
                    if not progressed:
                        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
                if outstanding:
                    pool.terminate()
                    terminal_rows: list[
                        tuple[Mapping[str, Any], Mapping[str, Any]]
                    ] = []
                    for task in pending[next_commit:]:
                        envelope = ready_envelopes.get(task["task_id"])
                        if envelope is None:
                            envelope = {
                                "timed_out": True,
                                "started_at": _utc_now(),
                                "finished_at": _utc_now(),
                                "elapsed_seconds": 0.0,
                            }
                        terminal_rows.append((task, envelope))
                    append_envelopes(terminal_rows)
                else:
                    pool.close()
            except BaseException:
                pool.terminate()
                raise
            finally:
                pool.join()
        else:
            for task_index, task in enumerate(pending):
                if time.monotonic() >= deadline:
                    append_wall_timeouts(pending[task_index:])
                    break
                started_at = _utc_now()
                started = time.perf_counter()
                try:
                    result = task_runner(
                        task, by_id[task["configuration_id"]], run_config
                    )
                    envelope = {
                        "ok": True,
                        "started_at": started_at,
                        "finished_at": _utc_now(),
                        "elapsed_seconds": float(time.perf_counter() - started),
                        "result": result,
                    }
                except BaseException as exc:
                    envelope = {
                        "ok": False,
                        "started_at": started_at,
                        "finished_at": _utc_now(),
                        "elapsed_seconds": float(time.perf_counter() - started),
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                append_envelope(task, envelope)
    except KeyboardInterrupt:
        document["status"] = "in_progress"
        _checkpoint(
            output_path, document, planned_tasks, base_elapsed, invocation_started
        )
        raise

    _checkpoint(output_path, document, planned_tasks, base_elapsed, invocation_started)
    validate_audit_document(document)
    return document


def _parse_sigmas(value: str) -> tuple[float, ...]:
    try:
        values = tuple(
            float(token.strip()) for token in value.split(",") if token.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "sigmas must be comma-separated floats"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one sigma is required")
    return values


def _parse_distinguished_face(value: str) -> tuple[int, int, int]:
    tokens = [token.strip() for token in value.split(",")]
    if len(tokens) != 3:
        raise argparse.ArgumentTypeError(
            "distinguished face must be three comma-separated labels"
        )
    try:
        face = tuple(int(token) for token in tokens)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "distinguished face labels must be integers"
        ) from exc
    if any(vertex < 0 for vertex in face) or len(set(face)) != 3:
        raise argparse.ArgumentTypeError(
            "distinguished face labels must be distinct and nonnegative"
        )
    return face


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit uniform and weighted-Tutte zero-shear solver basins."
    )
    parser.add_argument(
        "input", nargs="?", type=Path, help="Arithmetic corpus v1 JSON."
    )
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--config-id", action="append", default=[])
    parser.add_argument("--anchors-per-config", type=int, default=1)
    parser.add_argument(
        "--distinguished-face",
        action="append",
        type=_parse_distinguished_face,
        default=[],
        metavar="A,B,C",
        help=(
            "Use this exact ordered anchor face for every selected configuration; "
            "repeat to audit multiple explicit anchors."
        ),
    )
    parser.add_argument(
        "--sigmas",
        type=_parse_sigmas,
        default=DEFAULT_SIGMAS,
        help="Comma-separated positive lognormal sigmas.",
    )
    parser.add_argument(
        "--starts-per-sigma", type=int, default=DEFAULT_STARTS_PER_SIGMA
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-nfev", type=int, default=DEFAULT_MAX_NFEV)
    parser.add_argument(
        "--shear-tolerance", type=float, default=DEFAULT_SHEAR_TOLERANCE
    )
    parser.add_argument(
        "--support-tolerance", type=float, default=DEFAULT_SUPPORT_TOLERANCE
    )
    parser.add_argument(
        "--bending-tolerance", type=float, default=DEFAULT_BENDING_TOLERANCE
    )
    parser.add_argument("--xtol", type=float, default=DEFAULT_SOLVER_TOLERANCE)
    parser.add_argument("--ftol", type=float, default=DEFAULT_SOLVER_TOLERANCE)
    parser.add_argument("--gtol", type=float, default=DEFAULT_SOLVER_TOLERANCE)
    parser.add_argument(
        "--max-wall-seconds", type=float, default=DEFAULT_MAX_WALL_SECONDS
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--record-trajectory", action="store_true")
    parser.add_argument("--include-endpoint-coordinates", action="store_true")
    parser.add_argument(
        "--root-cluster-tolerance",
        type=float,
        default=DEFAULT_ROOT_CLUSTER_TOLERANCE,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", type=Path, metavar="PATH")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one configuration, one anchor, one weighted start, and one worker.",
    )
    return parser


def _run_config_from_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    selected_configuration_ids: Sequence[str],
) -> dict[str, Any]:
    sigmas = tuple(args.sigmas)
    starts_per_sigma = args.starts_per_sigma
    workers = args.workers
    anchors = args.anchors_per_config
    explicit_faces = sorted(tuple(face) for face in args.distinguished_face)
    max_wall_seconds = args.max_wall_seconds
    if args.smoke:
        sigmas = sigmas[:1]
        starts_per_sigma = min(1, starts_per_sigma)
        workers = 1
        anchors = 1
        explicit_faces = explicit_faces[:1]
        max_wall_seconds = min(60.0, max_wall_seconds)
    if explicit_faces:
        anchors = len(explicit_faces)
    run_config = {
        "selected_configuration_ids": list(selected_configuration_ids),
        "anchors_per_configuration": int(anchors),
        "explicit_distinguished_faces": [list(face) for face in explicit_faces],
        "sigmas": [float(value) for value in sigmas],
        "starts_per_sigma": int(starts_per_sigma),
        "base_seed": int(args.seed),
        "max_nfev": int(args.max_nfev),
        "shear_tolerance": float(args.shear_tolerance),
        "support_tolerance": float(args.support_tolerance),
        "bending_tolerance": float(args.bending_tolerance),
        "xtol": float(args.xtol),
        "ftol": float(args.ftol),
        "gtol": float(args.gtol),
        "record_trajectory": bool(args.record_trajectory),
        "include_endpoint_coordinates": bool(args.include_endpoint_coordinates),
        "root_cluster_tolerance": float(args.root_cluster_tolerance),
        "max_wall_seconds": float(max_wall_seconds),
        "workers": int(workers),
        "blas_threads_per_worker": int(BLAS_THREADS),
    }
    try:
        _validate_run_config(run_config)
    except ValueError as exc:
        parser.error(str(exc))
    return run_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.validate_only is not None:
        validate_audit_document(load_json_strict(args.validate_only))
        print(f"Valid Tutte zero-shear audit v1: {args.validate_only}")
        return 0
    if args.input is None or args.output is None:
        parser.error("input and --output are required unless --validate-only is used")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    try:
        corpus = load_arithmetic_corpus(args.input)
        configurations = select_configurations(corpus, args.config_id)
        if args.smoke:
            configurations = configurations[:1]
        selected_ids = [entry["id"] for entry in configurations]
        run_config = _run_config_from_args(args, parser, selected_ids)
        document = run_study(
            args.input,
            args.output,
            run_config,
            requested_ids=selected_ids,
            resume=args.resume,
            overwrite=args.overwrite,
            invocation=list(sys.argv if argv is None else [GENERATOR_NAME, *argv]),
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        f"{document['status']}: {document['summary']['finished']}/"
        f"{document['summary']['planned']} tasks; output={args.output}"
    )
    return 0 if document["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
