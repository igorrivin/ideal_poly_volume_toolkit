#!/usr/bin/env python3
"""Mine and audit unconstrained H4 volume maxima with geometric symmetry.

Product numerical logic belongs to
``ideal_poly_volume_toolkit.rl.polytope.h4_symmetric_benchmark``.  This
operational adapter owns only deterministic task construction, bounded serial
execution, checkpoint/resume semantics, provenance, compact result retention,
and the versioned JSON contract.

Use ``--starts 0 --calibrate-16-cell`` for the deterministic verifier-only
calibration on the eight vertices ``+/- e_i``.  Ordinary mining uses one
process at a time; task identifiers and seeds do not depend on execution order.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any


BLAS_THREADS = 1
THREAD_ENVIRONMENT = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _thread_variable in THREAD_ENVIRONMENT:
    os.environ[_thread_variable] = str(BLAS_THREADS)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

SCHEMA_VERSION = "ideal-poly-h4-symmetric-maxima-campaign/1"
PROTOCOL_VERSION = "h4-unconstrained-symmetric-maxima-mining/1"
GENERATOR_NAME = "scripts/mine_h4_symmetric_maxima.py"
GENERATOR_VERSION = "1.1.0"
SCHEMA_PATH = (
    REPOSITORY_ROOT / "docs" / "schemas" / "h4-symmetric-maxima-campaign-v1.schema.json"
)
CORE_PATH = (
    REPOSITORY_ROOT
    / "ideal_poly_volume_toolkit"
    / "rl"
    / "polytope"
    / "h4_symmetric_benchmark.py"
)
TRANSITIVE_SOURCE_PATHS = (
    REPOSITORY_ROOT / "ideal_poly_volume_toolkit" / "__init__.py",
    REPOSITORY_ROOT / "ideal_poly_volume_toolkit" / "rl" / "__init__.py",
    REPOSITORY_ROOT / "ideal_poly_volume_toolkit" / "rl" / "polytope" / "__init__.py",
    REPOSITORY_ROOT / "ideal_poly_volume_toolkit" / "symmetry.py",
    REPOSITORY_ROOT / "ideal_poly_volume_toolkit" / "geometry_4d.py",
    REPOSITORY_ROOT / "ideal_poly_volume_toolkit" / "geometry_4d_torch.py",
    REPOSITORY_ROOT / "ideal_poly_volume_toolkit" / "rl" / "polytope" / "optimizers.py",
    REPOSITORY_ROOT / "ideal_poly_volume_toolkit" / "rl" / "polytope" / "task.py",
    REPOSITORY_ROOT / "ideal_poly_volume_toolkit" / "rl" / "generic.py",
    REPOSITORY_ROOT / "ideal_poly_volume_toolkit" / "rl" / "inner_optimizers.py",
    REPOSITORY_ROOT / "ideal_poly_volume_toolkit" / "rl" / "polytope" / "types.py",
)

DEFAULT_N_VERTICES = 12
DEFAULT_STARTS = 16
DEFAULT_SEED = 20260817
DEFAULT_BUDGET = 500
DEFAULT_MAX_WALL_SECONDS = 3600.0
DEFAULT_OPTIMIZER = "powell"
OPTIMIZERS = (DEFAULT_OPTIMIZER, "fresh_topology_ad")
MIN_VERTICES = 8
MAX_VERTICES = 40
MAX_STARTS = 100_000
MAX_SEED = (1 << 63) - 1

_COORDINATE_KEYS = frozenset(
    {
        "vertices",
        "vertices_s3",
        "points",
        "points_s3",
        "coordinates",
        "initial_vertices",
        "initial_points",
        "best_vertices",
        "best_points",
    }
)
_BULK_AUDIT_KEYS = frozenset(
    {
        "combinatorial_generators",
        "geometric_generators",
        "generator_matrices",
        "hessian",
        "projected_hessian",
        "gradient",
        "projected_gradient",
        "facets",
        "simplices",
        "values",
        "projected_hessian_eigenvalues",
        "gauge_singular_values",
        "changed_probes",
        "vertex_orbits",
        "combinatorial_vertex_orbits",
        "requested_indices",
    }
)
_BULK_OPTIMIZATION_KEYS = frozenset(
    {
        "finite_vertex_labels",
        "topology_signature_sha256_examples",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _jsonable(value.as_dict())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numeric value is forbidden in campaign JSON")
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)
    if not math.isfinite(number):
        raise ValueError("non-finite numeric value is forbidden in campaign JSON")
    return number


def strict_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            _jsonable(payload),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


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
        raise ValueError("campaign JSON must be an object")
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
    target = Path(path)
    temporary = _write_temporary_json(target, payload)
    try:
        os.link(temporary, target)
        _fsync_directory(target.parent)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to clobber existing output {target}; use --resume or --overwrite"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    temporary = _write_temporary_json(target, payload)
    try:
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_with_schema(document: Mapping[str, Any]) -> None:
    try:
        from jsonschema.exceptions import SchemaError, ValidationError
        from jsonschema.validators import validator_for
    except ImportError as exc:  # pragma: no cover - dev environments include it
        raise RuntimeError(
            "campaign validation requires jsonschema; install the dev extra"
        ) from exc
    schema = load_json_strict(SCHEMA_PATH)
    validator_class = validator_for(schema)
    try:
        validator_class.check_schema(schema)
        # Validate the exact JSON-native representation that will reach disk.
        # This prevents NaN/Infinity from slipping through Python's permissive
        # numeric type checks and becoming a schema-invalid ``null`` on write.
        validator_class(schema).validate(_jsonable(document))
    except (SchemaError, ValidationError) as exc:
        raise ValueError(
            f"campaign JSON schema validation failed: {exc.message}"
        ) from exc


def derive_task_seed(base_seed: int, task_index: int) -> int:
    seed = int(base_seed) + int(task_index)
    if seed < 0 or seed > MAX_SEED:
        raise ValueError("derived task seed must lie in [0, 2**63 - 1]")
    return seed


def _stable_task_id(payload: Mapping[str, Any]) -> str:
    digest = _sha256_bytes(
        json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return f"h4-start-{int(payload['task_index']):06d}-{digest[:12]}"


def build_tasks(run_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for task_index in range(int(run_config["starts"])):
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "task_index": task_index,
            "n_vertices": int(run_config["n_vertices"]),
            "seed": derive_task_seed(int(run_config["base_seed"]), task_index),
            "budget": int(run_config["budget"]),
            "optimizer": str(run_config["optimizer"]),
        }
        tasks.append({**payload, "task_id": _stable_task_id(payload)})
    return tasks


def _load_core_api() -> SimpleNamespace:
    module = importlib.import_module(
        "ideal_poly_volume_toolkit.rl.polytope.h4_symmetric_benchmark"
    )
    optimize = getattr(module, "optimize_h4_unconstrained", None)
    audit = getattr(module, "audit_h4_symmetric_candidate", None)
    if not callable(optimize) or not callable(audit):
        raise RuntimeError(
            "H4 benchmark core must expose optimize_h4_unconstrained and "
            "audit_h4_symmetric_candidate"
        )
    return SimpleNamespace(optimize=optimize, audit=audit)


def _function_accepts(function: Callable[..., Any], name: str) -> bool:
    parameters = inspect.signature(function).parameters
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _call_optimize(function: Callable[..., Any], task: Mapping[str, Any]) -> Any:
    optimizer = str(task["optimizer"])
    kwargs = {
        "n_vertices": int(task["n_vertices"]),
        "seed": int(task["seed"]),
        "budget": int(task["budget"]),
        "optimizer": optimizer,
    }
    if not _function_accepts(function, "optimizer"):
        if optimizer != DEFAULT_OPTIMIZER:
            raise TypeError(
                "optimizer callable does not accept optimizer="
                f"{optimizer!r}; refusing an ambiguous fallback"
            )
        # Historical callables selected Powell implicitly.  This compatibility
        # path is scientifically unambiguous only for that default arm.
        kwargs.pop("optimizer")
    accepted = {
        key: value for key, value in kwargs.items() if _function_accepts(function, key)
    }
    return function(**accepted)


def _call_audit(
    function: Callable[..., Any],
    vertices: Sequence[Sequence[float]],
    *,
    check_local_max: bool,
    budget: int,
) -> Any:
    kwargs: dict[str, Any] = {}
    local_aliases = (
        "local_maximum",
        "check_local_max",
        "check_local_maximum",
        "perform_local_max_audit",
        "audit_local_maximum",
    )
    for name in local_aliases:
        if _function_accepts(function, name):
            kwargs[name] = bool(check_local_max)
            break
    for name in ("budget", "local_max_budget"):
        if _function_accepts(function, name):
            kwargs[name] = int(budget)
            break
    if _function_accepts(function, "pole_indices"):
        # Publication-level classifications require stability at every pole.
        kwargs["pole_indices"] = None
    return function(vertices, **kwargs)


def _payload(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        value = asdict(value)
    elif hasattr(value, "as_dict") and callable(value.as_dict):
        value = value.as_dict()
    elif not isinstance(value, Mapping) and hasattr(value, "__dict__"):
        value = vars(value)
    converted = _jsonable(value)
    if not isinstance(converted, dict):
        raise TypeError(f"core result must be an object, got {type(value).__name__}")
    return converted


def _extract_vertices(value: Any, *, n_vertices: int) -> list[list[float]]:
    candidates = (
        "vertices",
        "vertices_s3",
        "best_vertices",
        "points",
        "points_s3",
        "coordinates",
    )
    candidate = None
    for name in candidates:
        candidate = getattr(value, name, None)
        if candidate is None and isinstance(value, Mapping):
            candidate = value.get(name)
        if candidate is not None:
            break
    if candidate is None:
        raise ValueError("optimization result contains no candidate coordinates")
    if hasattr(candidate, "tolist"):
        candidate = candidate.tolist()
    if not isinstance(candidate, Sequence) or isinstance(
        candidate, (str, bytes, bytearray)
    ):
        raise ValueError("candidate coordinates must be a sequence")
    if len(candidate) != n_vertices:
        raise ValueError(
            f"candidate must contain {n_vertices} vertices; got {len(candidate)}"
        )
    vertices: list[list[float]] = []
    for vertex_index, row in enumerate(candidate):
        if hasattr(row, "tolist"):
            row = row.tolist()
        if (
            not isinstance(row, Sequence)
            or isinstance(row, (str, bytes, bytearray))
            or len(row) != 4
        ):
            raise ValueError(f"vertex {vertex_index} must contain four coordinates")
        converted: list[float] = []
        for component in row:
            if isinstance(component, bool):
                raise ValueError("coordinate components must be real numbers")
            number = float(component)
            if not math.isfinite(number):
                raise ValueError("candidate coordinates must be finite")
            converted.append(number)
        vertices.append(converted)
    return vertices


def _strip_payload(
    value: Any,
    *,
    audit: bool = False,
    optimization: bool = False,
) -> Any:
    converted = _jsonable(value)
    if isinstance(converted, list):
        return [
            _strip_payload(item, audit=audit, optimization=optimization)
            for item in converted
        ]
    if not isinstance(converted, dict):
        return converted
    omitted = _COORDINATE_KEYS
    if audit:
        omitted |= _BULK_AUDIT_KEYS
    if optimization:
        omitted |= _BULK_OPTIMIZATION_KEYS
    result: dict[str, Any] = {}
    for key, item in converted.items():
        if key.lower() in omitted:
            continue
        result[key] = _strip_payload(
            item,
            audit=audit,
            optimization=optimization,
        )
    return result


def _compact_optimization(
    value: Any,
    task: Mapping[str, Any],
    *,
    allow_legacy_powell: bool = False,
) -> dict[str, Any]:
    compact = _strip_payload(_payload(value), optimization=True)
    if not isinstance(compact, dict):  # pragma: no cover - guarded by _payload
        compact = {}
    compact.setdefault("dimension", 4)
    compact.setdefault("n_vertices", int(task["n_vertices"]))
    compact.setdefault("seed", int(task["seed"]))
    compact.setdefault("budget", int(task["budget"]))
    for name in ("scores", "work", "timing", "optimizer"):
        if not isinstance(compact.get(name), dict):
            compact[name] = {}
    expected_optimizer = str(task["optimizer"])
    reported_optimizer = compact["optimizer"].get("selector")
    if reported_optimizer is None:
        if not (allow_legacy_powell and expected_optimizer == DEFAULT_OPTIMIZER):
            raise ValueError("optimizer result omitted optimizer.selector")
        # Pre-selector Powell diagnostics are unambiguous only when the
        # callable itself predates the selector argument.
        compact["optimizer"]["selector"] = DEFAULT_OPTIMIZER
    elif reported_optimizer != expected_optimizer:
        raise ValueError("optimizer result selector does not match its task")
    return compact


def _orbit_sizes(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    if not value:
        return []
    if all(isinstance(item, list) for item in value):
        return sorted(len(item) for item in value)
    if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return sorted(Counter(int(item) for item in value).values())
    return []


def _compact_classification(
    symmetry: Mapping[str, Any],
    pole: Mapping[str, Any],
    local: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Recompute the public classification from retained primitive fields."""
    valid_geometry = symmetry.get("valid_geometry") is True
    full_group = symmetry.get("full_group_certified") is True
    order = symmetry.get("geometric_group_order")
    nontrivial = bool(
        full_group
        and symmetry.get("certified_nontrivial_symmetry") is True
        and isinstance(order, int)
        and not isinstance(order, bool)
        and order > 1
    )
    pole_stable = pole.get("within_tolerance") is True
    all_poles_requested = pole.get("all_poles_requested") is True
    all_poles_verified = bool(all_poles_requested and pole_stable)
    local_requested = local.get("requested") is True
    local_performed = local.get("performed") is True
    local_classification = str(local.get("classification", "not_requested"))

    if not valid_geometry:
        classification = "invalid_geometry"
    elif not pole_stable:
        classification = "pole_unstable_or_incomplete"
    elif not all_poles_requested:
        classification = "partial_pole_audit"
    elif not full_group:
        classification = "symmetry_inconclusive"
    elif not nontrivial:
        classification = "trivial_geometric_symmetry"
    elif not local_requested:
        classification = "symmetric_candidate"
    elif local_performed and local_classification == "strict_local_maximum":
        classification = "symmetric_strict_local_maximum_diagnostic"
    elif not local_performed:
        classification = "symmetric_candidate_local_maximum_inconclusive"
    else:
        classification = "symmetric_candidate_not_strict_local_maximum"
    return classification, {
        "valid_geometry": valid_geometry,
        "pole_volume_stable": pole_stable,
        "pole_volume_stable_on_requested": pole_stable,
        "all_poles_requested": all_poles_requested,
        "all_poles_verified": all_poles_verified,
        "full_group_certified": full_group,
        "geometric_group_order": (
            int(order)
            if isinstance(order, int) and not isinstance(order, bool)
            else None
        ),
        "nontrivial_geometric_symmetry": nontrivial,
        "local_maximum_classification": local_classification,
    }


def _compact_audit(value: Any) -> dict[str, Any]:
    payload = _payload(value)
    symmetry_raw = payload.get("symmetry")
    if not isinstance(symmetry_raw, dict):
        symmetry_raw = {}
    pole_raw = payload.get("pole_volume_audit")
    if not isinstance(pole_raw, dict):
        pole_raw = {}
    local_raw = payload.get("local_maximum_audit")
    if not isinstance(local_raw, dict):
        local_raw = {}

    symmetry = _strip_payload(symmetry_raw, audit=True)
    pole = _strip_payload(pole_raw, audit=True)
    local = _strip_payload(local_raw, audit=True)
    assert isinstance(symmetry, dict)
    assert isinstance(pole, dict)
    assert isinstance(local, dict)

    group_order = symmetry_raw.get("geometric_group_order")
    if isinstance(group_order, bool) or not isinstance(group_order, int):
        group_order = None
    symmetry["valid_geometry"] = symmetry_raw.get("valid_geometry") is True
    symmetry["full_group_certified"] = symmetry_raw.get("full_group_certified") is True
    symmetry["certified_nontrivial_symmetry"] = (
        symmetry_raw.get("certified_nontrivial_symmetry") is True
    )
    symmetry["geometric_group_order"] = group_order
    symmetry["combinatorial_generator_count"] = len(
        symmetry_raw.get("combinatorial_generators", ())
    )
    symmetry["geometric_generator_count"] = len(
        symmetry_raw.get("geometric_generators", ())
    )
    symmetry["orbit_sizes"] = _orbit_sizes(symmetry_raw.get("vertex_orbits"))

    pole.setdefault("within_tolerance", None)
    pole.setdefault("spread", None)
    pole.setdefault("successful_count", None)
    pole.setdefault("failure_count", None)
    pole.setdefault("all_poles_requested", False)
    pole.setdefault("requested_count", None)
    local.setdefault("requested", False)
    local.setdefault("performed", False)
    local.setdefault("status", "not_requested")

    classification, basis = _compact_classification(symmetry, pole, local)
    reported_classification = str(payload.get("classification", "unclassified"))
    reported_independent = str(
        payload.get("independent_classification", reported_classification)
    )
    if (
        reported_classification != classification
        or reported_independent != classification
    ):
        raise ValueError(
            "core H4 classification disagrees with independently recomputed fields"
        )
    reported_basis = payload.get("classification_basis")
    if not isinstance(reported_basis, Mapping) or any(
        reported_basis.get(key) != expected for key, expected in basis.items()
    ):
        raise ValueError("core H4 classification basis is inconsistent")
    eligible = basis["nontrivial_geometric_symmetry"]
    return {
        "dimension": payload.get("dimension", 4),
        "n_vertices": payload.get("n_vertices"),
        "classification": classification,
        "independent_classification": classification,
        "classification_basis": basis,
        "fully_certified_nontrivial_geometric_symmetry": eligible,
        "symmetry": symmetry,
        "pole_volume_audit": pole,
        "local_maximum_audit": local,
    }


def _symmetry_eligible(audit: Mapping[str, Any]) -> bool:
    symmetry = audit.get("symmetry")
    if not isinstance(symmetry, Mapping):
        return False
    order = symmetry.get("geometric_group_order")
    return bool(
        symmetry.get("valid_geometry") is True
        and symmetry.get("full_group_certified") is True
        and symmetry.get("certified_nontrivial_symmetry") is True
        and isinstance(order, int)
        and not isinstance(order, bool)
        and order > 1
    )


def _finite_number(value: Any) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_compact_audit_semantics(
    audit: Mapping[str, Any], *, expected_n_vertices: int
) -> None:
    if audit.get("dimension") != 4 or audit.get("n_vertices") != expected_n_vertices:
        raise ValueError("audit dimension or vertex count is inconsistent")
    symmetry = audit.get("symmetry")
    pole = audit.get("pole_volume_audit")
    local = audit.get("local_maximum_audit")
    if not all(isinstance(value, Mapping) for value in (symmetry, pole, local)):
        raise ValueError("audit is missing a required primitive section")

    full_group = symmetry.get("full_group_certified") is True
    certified_nontrivial = symmetry.get("certified_nontrivial_symmetry") is True
    order = symmetry.get("geometric_group_order")
    if full_group and (
        not isinstance(order, int) or isinstance(order, bool) or order < 1
    ):
        raise ValueError("certified symmetry requires a positive group order")
    if certified_nontrivial != bool(
        full_group and isinstance(order, int) and order > 1
    ):
        raise ValueError("nontrivial-symmetry flag is inconsistent with group order")
    if full_group:
        orbit_sizes = symmetry.get("orbit_sizes")
        if not (
            isinstance(orbit_sizes, list)
            and orbit_sizes
            and all(
                isinstance(size, int) and not isinstance(size, bool) and size > 0
                for size in orbit_sizes
            )
            and sum(orbit_sizes) == expected_n_vertices
        ):
            raise ValueError("certified symmetry has an invalid vertex-orbit partition")
        generator_count = symmetry.get("geometric_generator_count")
        if (
            not isinstance(generator_count, int)
            or isinstance(generator_count, bool)
            or generator_count < (1 if certified_nontrivial else 0)
        ):
            raise ValueError("certified symmetry has an invalid generator count")

    requested = pole.get("requested_count")
    successful = pole.get("successful_count")
    failures = pole.get("failure_count")
    counts = (requested, successful, failures)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in counts
    ):
        raise ValueError("pole-audit counts must be nonnegative integers")
    if pole.get("all_poles_requested") is not True or requested != expected_n_vertices:
        raise ValueError("publication audit must request every pole exactly once")
    if successful + failures != requested:
        raise ValueError("pole-audit success and failure counts are inconsistent")
    if pole.get("within_tolerance") is True:
        spread = pole.get("spread")
        tolerance = pole.get("spread_tolerance")
        if not (
            successful == requested
            and failures == 0
            and _finite_number(spread)
            and float(spread) >= 0.0
            and _finite_number(tolerance)
            and float(tolerance) >= 0.0
            and float(spread) <= float(tolerance)
        ):
            raise ValueError("successful all-pole audit has inconsistent metrics")

    local_requested = local.get("requested") is True
    local_performed = local.get("performed") is True
    if local_performed and not local_requested:
        raise ValueError("performed local-maximum audit was not requested")
    if local_performed and (
        local.get("status") != "completed" or local.get("diagnostic_only") is not True
    ):
        raise ValueError("performed local-maximum audit is not a completed diagnostic")
    if local.get("classification") == "strict_local_maximum" and not local_performed:
        raise ValueError("strict local-maximum classification was not performed")
    if not local_requested and local.get("classification") != "not_requested":
        raise ValueError("unrequested local-maximum audit has a classification")


def _validate_optimization_semantics(
    optimization: Mapping[str, Any], task: Mapping[str, Any]
) -> None:
    for key, expected in (
        ("dimension", 4),
        ("n_vertices", task["n_vertices"]),
        ("seed", task["seed"]),
        ("budget", task["budget"]),
    ):
        if optimization.get(key) != expected:
            raise ValueError(f"task optimization has inconsistent {key}")
    initial_digest = optimization.get("initial_coordinate_sha256")
    if not (
        isinstance(initial_digest, str)
        and len(initial_digest) == 64
        and all(character in "0123456789abcdef" for character in initial_digest)
    ):
        raise ValueError(
            "task optimization initial_coordinate_sha256 must be lowercase 64-hex"
        )
    scores = optimization.get("scores")
    work = optimization.get("work")
    timing = optimization.get("timing")
    optimizer = optimization.get("optimizer")
    if not isinstance(scores, Mapping) or not isinstance(work, Mapping):
        raise ValueError("task optimization is missing scores or work accounting")
    if not isinstance(timing, Mapping):
        raise ValueError("task optimization is missing timing")
    if (
        not isinstance(optimizer, Mapping)
        or optimizer.get("selector") != task.get("optimizer")
    ):
        raise ValueError("task optimization has inconsistent optimizer selector")
    if any(not _finite_number(scores.get(key)) for key in ("initial", "final", "gain")):
        raise ValueError("task optimization scores must be finite")
    expected_gain = float(scores["final"]) - float(scores["initial"])
    if not math.isclose(
        float(scores["gain"]), expected_gain, rel_tol=1.0e-12, abs_tol=1.0e-12
    ):
        raise ValueError("task optimization gain is inconsistent")
    work_keys = ("objective_evals", "surrogate_evals", "gradient_evals")
    if any(
        not isinstance(work.get(key), int)
        or isinstance(work.get(key), bool)
        or work[key] < 0
        for key in (*work_keys, "total_evals")
    ):
        raise ValueError("task work counters must be nonnegative integers")
    if work["total_evals"] != sum(work[key] for key in work_keys):
        raise ValueError("task total work counter is inconsistent")
    if work["total_evals"] > int(task["budget"]):
        raise ValueError("task work counter exceeds its budget")
    if task.get("optimizer") == "fresh_topology_ad":
        _validate_fresh_optimizer_diagnostics(
            optimization=optimization,
            optimizer=optimizer,
            work=work,
        )
    if any(
        not _finite_number(timing.get(key)) or float(timing[key]) < 0.0
        for key in ("wall_seconds", "process_cpu_seconds")
    ):
        raise ValueError("task optimization timing must be finite and nonnegative")


def _nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_fresh_optimizer_diagnostics(
    *,
    optimization: Mapping[str, Any],
    optimizer: Mapping[str, Any],
    work: Mapping[str, Any],
) -> None:
    prefix = "fresh optimizer diagnostics"
    diagnostics = optimizer.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError(f"{prefix} are missing")
    if diagnostics.get("selector") != "fresh_topology_ad":
        raise ValueError(f"{prefix} selector disagrees with the outer selector")
    if diagnostics.get("budget") != optimization.get("budget"):
        raise ValueError(f"{prefix} budget disagrees with the optimization budget")
    diagnostics_work = diagnostics.get("work")
    if not isinstance(diagnostics_work, Mapping) or dict(diagnostics_work) != dict(work):
        raise ValueError(f"{prefix} work disagrees with top-level work")

    count_names = (
        "requested_branch_count",
        "executed_branch_count",
        "completed_branch_count",
        "failed_branch_count",
        "canonical_scored_branch_count",
    )
    if any(not _nonnegative_integer(diagnostics.get(name)) for name in count_names):
        raise ValueError(f"{prefix} branch counters must be nonnegative integers")
    requested = int(diagnostics["requested_branch_count"])
    executed = int(diagnostics["executed_branch_count"])
    completed = int(diagnostics["completed_branch_count"])
    failed = int(diagnostics["failed_branch_count"])
    canonical_scored = int(diagnostics["canonical_scored_branch_count"])

    pole_indices = diagnostics.get("pole_indices")
    branch_modes = diagnostics.get("branch_modes")
    branches = diagnostics.get("branches")
    if not all(isinstance(value, list) for value in (pole_indices, branch_modes, branches)):
        raise ValueError(f"{prefix} branch-plan lists are missing")
    if requested != len(pole_indices) * len(branch_modes) or executed > requested:
        raise ValueError(f"{prefix} requested/executed branch plan is inconsistent")
    if len(branches) != executed:
        raise ValueError(f"{prefix} branch list length disagrees with executed count")
    if completed + failed != executed:
        raise ValueError(f"{prefix} completed/failed counts disagree with executed count")
    if canonical_scored > executed:
        raise ValueError(f"{prefix} canonical-scored count exceeds executed count")

    allocation = diagnostics.get("budget_allocation")
    if not isinstance(allocation, Mapping):
        raise ValueError(f"{prefix} budget allocation is missing")
    initial = allocation.get("initial_objective_evals")
    reserved = allocation.get("reserved_endpoint_objective_evals")
    gradient_allocations = allocation.get("gradient_evals_by_branch")
    if not (
        _nonnegative_integer(initial)
        and _nonnegative_integer(reserved)
        and isinstance(gradient_allocations, list)
        and all(_nonnegative_integer(value) for value in gradient_allocations)
    ):
        raise ValueError(f"{prefix} budget allocation is malformed")
    if len(gradient_allocations) != executed or reserved != executed:
        raise ValueError(f"{prefix} allocation lengths disagree with executed count")
    if initial != 1:
        raise ValueError(f"{prefix} must reserve exactly one initial objective eval")
    allocated_work = int(initial) + int(reserved) + sum(gradient_allocations)
    if allocated_work > int(optimization["budget"]):
        raise ValueError(f"{prefix} allocation exceeds the optimization budget")
    if work.get("objective_evals") != int(initial) + executed:
        raise ValueError(f"{prefix} objective work disagrees with branch allocation")

    summed_gradients = 0
    observed_failed = 0
    for index, (branch, branch_allocation) in enumerate(
        zip(branches, gradient_allocations, strict=True)
    ):
        if not isinstance(branch, Mapping):
            raise ValueError(f"{prefix} branch {index} is not an object")
        branch_budget = branch.get("gradient_budget")
        branch_gradients = branch.get("gradient_evals")
        topology_evaluations = branch.get("topology_evaluations")
        if not all(
            _nonnegative_integer(value)
            for value in (branch_budget, branch_gradients, topology_evaluations)
        ):
            raise ValueError(f"{prefix} branch {index} work is malformed")
        if branch_budget != branch_allocation or branch_gradients > branch_budget:
            raise ValueError(f"{prefix} branch {index} exceeds its gradient allocation")
        if topology_evaluations > branch_gradients:
            raise ValueError(
                f"{prefix} branch {index} topology work exceeds gradient work"
            )
        summed_gradients += int(branch_gradients)
        if branch.get("status") == "failed":
            observed_failed += 1
    if summed_gradients != work.get("gradient_evals"):
        raise ValueError(f"{prefix} branch gradient work disagrees with top-level work")
    if observed_failed != failed or executed - observed_failed != completed:
        raise ValueError(f"{prefix} branch statuses disagree with summary counts")


def _reject_forbidden_compact_keys(
    value: Any, forbidden: frozenset[str], *, path: str
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in forbidden:
                raise ValueError(
                    f"compact payload contains forbidden key {path}.{key_text}"
                )
            _reject_forbidden_compact_keys(item, forbidden, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_compact_keys(item, forbidden, path=f"{path}[{index}]")


def _local_maximum_shortlist(audit: Mapping[str, Any]) -> bool:
    """Return whether the cheap all-pole audit warrants the costly diagnostic."""

    basis = audit.get("classification_basis")
    return bool(
        _symmetry_eligible(audit)
        and audit.get("classification") == "symmetric_candidate"
        and isinstance(basis, Mapping)
        and basis.get("all_poles_requested") is True
        and basis.get("all_poles_verified") is True
    )


def _publication_candidate(audit: Mapping[str, Any]) -> bool:
    """Return whether all cheap publication gates, not local maximality, pass."""

    return bool(
        _local_maximum_shortlist(audit)
        or (
            _symmetry_eligible(audit)
            and audit.get("classification")
            == "symmetric_strict_local_maximum_diagnostic"
            and isinstance(audit.get("classification_basis"), Mapping)
            and audit["classification_basis"].get("all_poles_requested") is True
            and audit["classification_basis"].get("all_poles_verified") is True
        )
    )


def _coordinate_digest(vertices: Sequence[Sequence[float]]) -> str:
    return _sha256_bytes(strict_json_bytes(vertices))


def _error_row(
    task: Mapping[str, Any],
    exc: BaseException,
    *,
    started: float,
    cpu_started: float,
    optimization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        **dict(task),
        "status": "failed",
        "reason": type(exc).__name__,
        "message": str(exc)[:2000],
        "elapsed_wall_seconds": float(time.perf_counter() - started),
        "process_cpu_seconds": float(time.process_time() - cpu_started),
        "optimization": None if optimization is None else dict(optimization),
        "audit": None,
        "coordinate_sha256": None,
        "coordinates_retained": False,
        "vertices": None,
    }


def _run_task(
    task: Mapping[str, Any],
    run_config: Mapping[str, Any],
    api: SimpleNamespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    optimization_summary: dict[str, Any] | None = None
    vertices: list[list[float]] | None = None
    try:
        allow_legacy_powell = bool(
            task["optimizer"] == DEFAULT_OPTIMIZER
            and not _function_accepts(api.optimize, "optimizer")
        )
        optimization_result = _call_optimize(api.optimize, task)
        optimization_summary = _compact_optimization(
            optimization_result,
            task,
            allow_legacy_powell=allow_legacy_powell,
        )
        vertices = _extract_vertices(
            optimization_result, n_vertices=int(task["n_vertices"])
        )
        preliminary = _compact_audit(
            _call_audit(
                api.audit,
                vertices,
                check_local_max=False,
                budget=int(task["budget"]),
            )
        )
        audit = preliminary
        if bool(run_config["local_max_on_shortlist"]) and _local_maximum_shortlist(
            preliminary
        ):
            try:
                audit = _compact_audit(
                    _call_audit(
                        api.audit,
                        vertices,
                        check_local_max=True,
                        budget=int(task["budget"]),
                    )
                )
            except Exception as exc:  # retain the independently useful first audit
                audit = dict(preliminary)
                audit["local_maximum_audit"] = {
                    "requested": True,
                    "performed": False,
                    "diagnostic_only": True,
                    "status": "error",
                    "classification": "inconclusive_numerical_failure",
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:2000],
                }
                classification, basis = _compact_classification(
                    audit["symmetry"],
                    audit["pole_volume_audit"],
                    audit["local_maximum_audit"],
                )
                audit["classification"] = classification
                audit["independent_classification"] = classification
                audit["classification_basis"] = basis
        eligible = _symmetry_eligible(audit)
        retain = bool(run_config["retain_all_coordinates"]) or eligible
        return {
            **dict(task),
            "status": "completed",
            "reason": str(audit["classification"]),
            "message": None,
            "elapsed_wall_seconds": float(time.perf_counter() - started),
            "process_cpu_seconds": float(time.process_time() - cpu_started),
            "optimization": optimization_summary,
            "audit": audit,
            "coordinate_sha256": _coordinate_digest(vertices),
            "coordinates_retained": retain,
            "vertices": vertices if retain else None,
        }
    except Exception as exc:
        row = _error_row(
            task,
            exc,
            started=started,
            cpu_started=cpu_started,
            optimization=optimization_summary,
        )
        if bool(run_config["retain_all_coordinates"]) and vertices is not None:
            row["coordinate_sha256"] = _coordinate_digest(vertices)
            row["coordinates_retained"] = True
            row["vertices"] = vertices
        return row


def _sixteen_cell_vertices() -> list[list[float]]:
    vertices: list[list[float]] = []
    for sign in (1.0, -1.0):
        for axis in range(4):
            row = [0.0, 0.0, 0.0, 0.0]
            row[axis] = sign
            vertices.append(row)
    return vertices


def run_16_cell_calibration(
    run_config: Mapping[str, Any], api: SimpleNamespace
) -> dict[str, Any]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    vertices = _sixteen_cell_vertices()
    try:
        audit = _compact_audit(
            _call_audit(
                api.audit,
                vertices,
                check_local_max=bool(run_config["local_max_on_shortlist"]),
                budget=int(run_config["budget"]),
            )
        )
        symmetry = audit["symmetry"]
        passes = bool(
            _publication_candidate(audit)
            and symmetry.get("geometric_group_order") == 384
        )
        return {
            "requested": True,
            "status": "completed" if passes else "failed",
            "passes": passes,
            "expected_geometric_group_order": 384,
            "coordinate_sha256": _coordinate_digest(vertices),
            "audit": audit,
            "message": None if passes else "16-cell symmetry certification failed",
            "elapsed_wall_seconds": float(time.perf_counter() - started),
            "process_cpu_seconds": float(time.process_time() - cpu_started),
        }
    except Exception as exc:
        return {
            "requested": True,
            "status": "failed",
            "passes": False,
            "expected_geometric_group_order": 384,
            "coordinate_sha256": _coordinate_digest(vertices),
            "audit": None,
            "message": f"{type(exc).__name__}: {str(exc)[:1900]}",
            "elapsed_wall_seconds": float(time.perf_counter() - started),
            "process_cpu_seconds": float(time.process_time() - cpu_started),
        }


def _process_entry(
    kind: str,
    payload: dict[str, Any],
    run_config: dict[str, Any],
    result_connection: Any,
) -> None:
    try:
        api = _load_core_api()
        if kind == "task":
            result = _run_task(payload, run_config, api)
        elif kind == "calibration":
            result = run_16_cell_calibration(run_config, api)
        else:  # pragma: no cover - internal invariant
            raise ValueError(f"unknown child operation: {kind}")
        result_connection.send({"ok": True, "result": result})
    except BaseException as exc:
        result_connection.send(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "message": str(exc)[:2000],
            }
        )
    finally:
        result_connection.close()


def _execute_with_deadline(
    kind: str,
    payload: dict[str, Any],
    run_config: dict[str, Any],
    *,
    remaining_seconds: float,
    api: SimpleNamespace | None,
) -> tuple[str, Any]:
    if remaining_seconds <= 0.0:
        return "timed_out", None
    if api is not None:
        if kind == "task":
            return "finished", _run_task(payload, run_config, api)
        return "finished", run_16_cell_calibration(run_config, api)

    methods = mp.get_all_start_methods()
    context = mp.get_context("fork" if "fork" in methods else "spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_process_entry,
        args=(kind, payload, run_config, send_connection),
    )
    process.start()
    send_connection.close()
    envelope = None
    try:
        if receive_connection.poll(timeout=max(0.0, remaining_seconds)):
            envelope = receive_connection.recv()
    except EOFError:
        envelope = None
    if envelope is None:
        process.terminate()
        process.join(timeout=2.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=2.0)
        receive_connection.close()
        if process.exitcode is None or process.exitcode < 0:
            return "timed_out", None
        return "error", {
            "ok": False,
            "error_type": "WorkerExit",
            "message": f"child exited with code {process.exitcode} without a result",
        }
    receive_connection.close()
    process.join(timeout=2.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=2.0)
    if not envelope.get("ok"):
        return "error", envelope
    return "finished", envelope["result"]


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
        CORE_PATH,
        *TRANSITIVE_SOURCE_PATHS,
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required source file is missing: {missing[0]}")
    return {
        str(path.relative_to(REPOSITORY_ROOT)): _sha256_file(path) for path in paths
    }


def _assert_source_hashes_unchanged(expected: Mapping[str, str]) -> None:
    if _source_hashes() != dict(expected):
        raise RuntimeError(
            "campaign source files changed during execution; refusing to mix results"
        )


def _dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for distribution in (
        "ideal-poly-volume-toolkit",
        "numpy",
        "scipy",
        "jsonschema",
        "pynauty",
        "torch",
    ):
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
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return (commit if len(commit) == 40 else None), bool(dirty)


def _empty_calibration(requested: bool) -> dict[str, Any]:
    return {
        "requested": requested,
        "status": "pending" if requested else "not_requested",
        "passes": False,
        "expected_geometric_group_order": 384,
        "coordinate_sha256": None,
        "audit": None,
        "message": None,
        "elapsed_wall_seconds": 0.0,
        "process_cpu_seconds": 0.0,
    }


def _initial_document(
    run_config: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    invocation: Sequence[str],
) -> dict[str, Any]:
    timestamp = _utc_now()
    commit, dirty = _git_provenance()
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
        "run_config": _jsonable(run_config),
        "run_config_sha256": _run_config_hash(run_config),
        "provenance": {
            "invocation": list(invocation),
            "git_commit": commit,
            "git_dirty": dirty,
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "dependency_versions": _dependency_versions(),
            "thread_environment": {
                variable: os.environ.get(variable) for variable in THREAD_ENVIRONMENT
            },
            "source_files": dict(source_hashes),
        },
        "planned_task_count": len(tasks),
        "calibration": _empty_calibration(bool(run_config["calibrate_16_cell"])),
        "tasks": [],
        "summary": {
            "planned": len(tasks),
            "finished": 0,
            "completed": 0,
            "failed": 0,
            "pending": len(tasks),
            "certified_nontrivial_symmetry": 0,
            "local_maximum_audits_performed": 0,
            "coordinates_retained": 0,
        },
        "timings": {
            "cumulative_active_seconds": 0.0,
            "last_invocation_seconds": 0.0,
        },
    }


def _expected_status(document: Mapping[str, Any]) -> str:
    summary = document["summary"]
    calibration = document["calibration"]
    if calibration["requested"] and calibration["status"] == "failed":
        return "failed"
    if int(summary["pending"]) > 0 or calibration["status"] == "pending":
        return "in_progress"
    if calibration["requested"] and calibration["status"] != "completed":
        return "failed"
    if int(summary["planned"]) == 0:
        return "complete" if calibration.get("passes") is True else "failed"
    if int(summary["completed"]) == 0:
        return "failed"
    return "complete"


def _refresh_document(
    document: dict[str, Any], planned_tasks: Sequence[Mapping[str, Any]]
) -> None:
    document["tasks"].sort(key=lambda row: int(row["task_index"]))
    rows = document["tasks"]
    statuses = Counter(str(row["status"]) for row in rows)
    document["summary"] = {
        "planned": len(planned_tasks),
        "finished": len(rows),
        "completed": int(statuses["completed"]),
        "failed": int(statuses["failed"]),
        "pending": len(planned_tasks) - len(rows),
        "certified_nontrivial_symmetry": sum(
            bool(
                row.get("audit", {}).get(
                    "fully_certified_nontrivial_geometric_symmetry", False
                )
            )
            for row in rows
            if isinstance(row.get("audit"), dict)
        ),
        "local_maximum_audits_performed": sum(
            bool(
                row.get("audit", {})
                .get("local_maximum_audit", {})
                .get("performed", False)
            )
            for row in rows
            if isinstance(row.get("audit"), dict)
        ),
        "coordinates_retained": sum(
            bool(row.get("coordinates_retained", False)) for row in rows
        ),
    }
    document["updated_at"] = _utc_now()
    document["status"] = _expected_status(document)


def _checkpoint(
    path: Path,
    document: dict[str, Any],
    planned_tasks: Sequence[Mapping[str, Any]],
    base_elapsed: float,
    invocation_started: float,
) -> None:
    elapsed = float(time.perf_counter() - invocation_started)
    document["timings"] = {
        "cumulative_active_seconds": float(base_elapsed + elapsed),
        "last_invocation_seconds": elapsed,
    }
    _refresh_document(document, planned_tasks)
    validate_campaign_document(document)
    atomic_write_json(path, document)


def _validate_run_config(run_config: Mapping[str, Any]) -> None:
    n_vertices = run_config.get("n_vertices")
    if (
        not isinstance(n_vertices, int)
        or isinstance(n_vertices, bool)
        or not MIN_VERTICES <= n_vertices <= MAX_VERTICES
    ):
        raise ValueError(f"n_vertices must lie in [{MIN_VERTICES}, {MAX_VERTICES}]")
    starts = run_config.get("starts")
    if (
        not isinstance(starts, int)
        or isinstance(starts, bool)
        or not 0 <= starts <= MAX_STARTS
    ):
        raise ValueError(f"starts must lie in [0, {MAX_STARTS}]")
    if starts == 0 and run_config.get("calibrate_16_cell") is not True:
        raise ValueError("starts=0 requires calibrate_16_cell=true")
    optimizer = run_config.get("optimizer")
    if not isinstance(optimizer, str) or optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
    for name in ("base_seed", "budget"):
        value = run_config.get(name)
        minimum = 0 if name == "base_seed" else 1
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if int(run_config["base_seed"]) + max(0, int(starts) - 1) > MAX_SEED:
        raise ValueError("derived task seeds exceed 2**63 - 1")
    wall = run_config.get("max_wall_seconds")
    if (
        not isinstance(wall, (int, float))
        or isinstance(wall, bool)
        or not math.isfinite(float(wall))
        or float(wall) <= 0.0
    ):
        raise ValueError("max_wall_seconds must be finite and positive")
    for name in (
        "local_max_on_shortlist",
        "retain_all_coordinates",
        "calibrate_16_cell",
    ):
        if not isinstance(run_config.get(name), bool):
            raise ValueError(f"{name} must be boolean")
    if run_config.get("workers") != 1 or run_config.get("blas_threads") != 1:
        raise ValueError("campaign v1 requires workers=1 and blas_threads=1")


def _validate_document_invariants(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported H4 campaign schema version")
    if document.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unsupported H4 campaign protocol version")
    run_config = document["run_config"]
    _validate_run_config(run_config)
    if document.get("run_config_sha256") != _run_config_hash(run_config):
        raise ValueError("stored run_config hash does not match its contents")
    source_files = document["provenance"]["source_files"]
    required_sources = {
        GENERATOR_NAME,
        str(SCHEMA_PATH.relative_to(REPOSITORY_ROOT)),
        str(CORE_PATH.relative_to(REPOSITORY_ROOT)),
        *(str(path.relative_to(REPOSITORY_ROOT)) for path in TRANSITIVE_SOURCE_PATHS),
    }
    if not required_sources.issubset(source_files):
        raise ValueError("provenance is missing a required source hash")
    if document["generator"]["source_sha256"] != source_files[GENERATOR_NAME]:
        raise ValueError("generator hash disagrees with source-file provenance")
    plan = build_tasks(run_config)
    if document.get("planned_task_count") != len(plan):
        raise ValueError("planned_task_count does not match deterministic plan")
    rows = document["tasks"]
    if rows != sorted(rows, key=lambda row: int(row["task_index"])):
        raise ValueError("tasks must be sorted by task_index")
    indices = [int(row["task_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("checkpoint contains duplicate task indices")
    if any(index < 0 or index >= len(plan) for index in indices):
        raise ValueError("checkpoint contains an out-of-range task index")
    for row in rows:
        expected = plan[int(row["task_index"])]
        for key in (
            "protocol_version",
            "task_index",
            "task_id",
            "n_vertices",
            "seed",
            "budget",
            "optimizer",
        ):
            if row.get(key) != expected[key]:
                raise ValueError(f"task {row.get('task_index')} has invalid {key}")
        retained = row.get("coordinates_retained") is True
        if retained != (row.get("vertices") is not None):
            raise ValueError("coordinate retention flag and payload disagree")
        if retained:
            vertices = _extract_vertices(row, n_vertices=int(run_config["n_vertices"]))
            if row.get("coordinate_sha256") != _coordinate_digest(vertices):
                raise ValueError("retained coordinate digest is incorrect")
        audit = row.get("audit")
        if row["status"] == "completed":
            if not isinstance(audit, Mapping):
                raise ValueError("completed task requires an audit")
            optimization = row.get("optimization")
            if not isinstance(optimization, Mapping):
                raise ValueError("completed task requires optimization diagnostics")
            _reject_forbidden_compact_keys(
                optimization,
                _COORDINATE_KEYS | _BULK_OPTIMIZATION_KEYS,
                path="task.optimization",
            )
            _reject_forbidden_compact_keys(
                audit,
                _COORDINATE_KEYS | _BULK_AUDIT_KEYS,
                path="task.audit",
            )
            _validate_optimization_semantics(optimization, row)
            if audit.get("n_vertices") != row["n_vertices"]:
                raise ValueError("task audit vertex count is inconsistent")
            if audit.get("classification") != audit.get("independent_classification"):
                raise ValueError("task classifications disagree")
            recomputed_classification, recomputed_basis = _compact_classification(
                audit["symmetry"],
                audit["pole_volume_audit"],
                audit["local_maximum_audit"],
            )
            if audit.get("classification") != recomputed_classification:
                raise ValueError(
                    "task classification is inconsistent with audit fields"
                )
            if audit.get("classification_basis") != recomputed_basis:
                raise ValueError("task classification basis is inconsistent")
            _validate_compact_audit_semantics(
                audit, expected_n_vertices=int(row["n_vertices"])
            )
            if audit.get(
                "fully_certified_nontrivial_geometric_symmetry"
            ) is not _symmetry_eligible(audit):
                raise ValueError("task symmetry summary is inconsistent")
            pole_audit = audit.get("pole_volume_audit")
            if not isinstance(pole_audit, Mapping) or (
                pole_audit.get("all_poles_requested") is not True
            ):
                raise ValueError("completed task must audit every pole")
            if run_config["retain_all_coordinates"] and not retained:
                raise ValueError("retain-all campaign omitted completed coordinates")
            if _symmetry_eligible(audit) and not retained:
                raise ValueError("certified symmetric task omitted its coordinates")
        if (
            retained
            and not run_config["retain_all_coordinates"]
            and not _symmetry_eligible(audit or {})
        ):
            raise ValueError("uncertified task retained coordinates without opt-in")

    expected_summary_document = {
        "tasks": [dict(row) for row in rows],
        "calibration": document["calibration"],
        "summary": {},
        "status": "in_progress",
        "updated_at": document["updated_at"],
    }
    _refresh_document(expected_summary_document, plan)
    if document["summary"] != expected_summary_document["summary"]:
        raise ValueError("campaign summary does not match task rows")
    expected = _expected_status(document)
    if document["status"] not in {expected, "timed_out"}:
        raise ValueError(
            f"campaign status does not match contents; expected {expected!r}"
        )
    if document["status"] == "timed_out" and not (
        document["summary"]["pending"] > 0
        or document["calibration"]["status"] == "pending"
    ):
        raise ValueError("timed_out status requires pending work")
    calibration = document["calibration"]
    if calibration["requested"] != bool(run_config["calibrate_16_cell"]):
        raise ValueError("calibration request does not match run_config")
    if calibration["status"] == "completed":
        audit = calibration.get("audit")
        if not isinstance(audit, Mapping):
            raise ValueError("completed calibration requires an audit")
        symmetry = audit.get("symmetry")
        if not isinstance(symmetry, Mapping):
            raise ValueError("completed calibration requires a symmetry audit")
        recomputed_classification, recomputed_basis = _compact_classification(
            audit["symmetry"],
            audit["pole_volume_audit"],
            audit["local_maximum_audit"],
        )
        _reject_forbidden_compact_keys(
            audit,
            _COORDINATE_KEYS | _BULK_AUDIT_KEYS,
            path="calibration.audit",
        )
        _validate_compact_audit_semantics(audit, expected_n_vertices=8)
        if not (
            calibration.get("passes") is True
            and calibration.get("coordinate_sha256")
            == _coordinate_digest(_sixteen_cell_vertices())
            and audit.get("dimension") == 4
            and audit.get("n_vertices") == 8
            and audit.get("classification") == recomputed_classification
            and audit.get("independent_classification") == recomputed_classification
            and audit.get("classification_basis") == recomputed_basis
            and _publication_candidate(audit)
            and symmetry.get("valid_geometry") is True
            and symmetry.get("full_group_certified") is True
            and symmetry.get("geometric_group_order") == 384
        ):
            raise ValueError("completed 16-cell calibration must certify order 384")


def validate_campaign_document(document: Mapping[str, Any]) -> None:
    strict_json_bytes(document)
    _validate_with_schema(document)
    _validate_document_invariants(document)


def _validate_checkpoint_matches(
    document: Mapping[str, Any],
    run_config: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> None:
    if document["run_config_sha256"] != _run_config_hash(run_config):
        raise ValueError("resume run config does not match checkpoint")
    if document["provenance"]["source_files"] != dict(source_hashes):
        raise ValueError("resume refused because source hashes changed")
    provenance = document["provenance"]
    current_environment = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "dependency_versions": _dependency_versions(),
        "thread_environment": {
            variable: os.environ.get(variable) for variable in THREAD_ENVIRONMENT
        },
    }
    for key, current in current_environment.items():
        if provenance.get(key) != current:
            raise ValueError(f"resume refused because {key} changed")


def run_campaign(
    output_path: str | Path,
    run_config: dict[str, Any],
    *,
    resume: bool = False,
    overwrite: bool = False,
    invocation: Sequence[str] | None = None,
    api: SimpleNamespace | None = None,
) -> dict[str, Any]:
    """Run or resume the bounded serial campaign with atomic checkpoints."""
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    _validate_run_config(run_config)
    output = Path(output_path)
    tasks = build_tasks(run_config)
    source_hashes = _source_hashes()
    invocation = list(sys.argv if invocation is None else invocation)

    if resume:
        if not output.is_file():
            raise FileNotFoundError(f"cannot resume missing campaign: {output}")
        document = load_json_strict(output)
        validate_campaign_document(document)
        _validate_checkpoint_matches(document, run_config, source_hashes)
    else:
        document = _initial_document(run_config, tasks, source_hashes, invocation)
        validate_campaign_document(document)
        if overwrite:
            atomic_write_json(output, document)
        else:
            exclusive_create_json(output, document)

    base_elapsed = float(document["timings"]["cumulative_active_seconds"])
    invocation_started = time.perf_counter()
    deadline = invocation_started + float(run_config["max_wall_seconds"])
    timed_out = False

    if document["calibration"]["status"] == "pending":
        _assert_source_hashes_unchanged(source_hashes)
        outcome, result = _execute_with_deadline(
            "calibration",
            {},
            run_config,
            remaining_seconds=deadline - time.perf_counter(),
            api=api,
        )
        if outcome == "timed_out":
            timed_out = True
        elif outcome == "error":
            document["calibration"] = {
                **_empty_calibration(True),
                "status": "failed",
                "message": f"{result['error_type']}: {result['message']}",
            }
        else:
            document["calibration"] = result
        _assert_source_hashes_unchanged(source_hashes)
        _checkpoint(output, document, tasks, base_elapsed, invocation_started)
        if document["calibration"]["status"] == "failed":
            return document

    if document["calibration"]["status"] == "failed":
        return document

    completed_indices = {int(row["task_index"]) for row in document["tasks"]}
    for task in tasks:
        if timed_out or int(task["task_index"]) in completed_indices:
            continue
        _assert_source_hashes_unchanged(source_hashes)
        outcome, result = _execute_with_deadline(
            "task",
            task,
            run_config,
            remaining_seconds=deadline - time.perf_counter(),
            api=api,
        )
        if outcome == "timed_out":
            timed_out = True
            break
        if outcome == "error":
            now = time.perf_counter()
            result = _error_row(
                task,
                RuntimeError(f"{result['error_type']}: {result['message']}"),
                started=now,
                cpu_started=time.process_time(),
            )
        _assert_source_hashes_unchanged(source_hashes)
        document["tasks"].append(result)
        completed_indices.add(int(task["task_index"]))
        _checkpoint(output, document, tasks, base_elapsed, invocation_started)

    _assert_source_hashes_unchanged(source_hashes)
    _checkpoint(output, document, tasks, base_elapsed, invocation_started)
    if timed_out:
        document["status"] = "timed_out"
        document["updated_at"] = _utc_now()
        _assert_source_hashes_unchanged(source_hashes)
        validate_campaign_document(document)
        atomic_write_json(output, document)
    return document


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-vertices", type=int, default=DEFAULT_N_VERTICES, metavar="N"
    )
    parser.add_argument("--starts", type=int, default=DEFAULT_STARTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument(
        "--optimizer",
        choices=OPTIMIZERS,
        default=DEFAULT_OPTIMIZER,
        help="unconstrained optimizer arm (default: %(default)s)",
    )
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument(
        "--max-wall-seconds", type=float, default=DEFAULT_MAX_WALL_SECONDS
    )
    parser.add_argument(
        "--local-max-on-shortlist",
        action="store_true",
        help="run the expensive local-maximum audit only after symmetry passes",
    )
    parser.add_argument(
        "--retain-all-coordinates",
        action="store_true",
        help="retain coordinates even when full nontrivial symmetry is uncertified",
    )
    parser.add_argument(
        "--calibrate-16-cell",
        action="store_true",
        help="audit the exact +/-e_i 16-cell before mining; use --starts 0 alone",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", type=Path, metavar="PATH")
    return parser


def _run_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    run_config = {
        "n_vertices": int(args.n_vertices),
        "starts": int(args.starts),
        "base_seed": int(args.seed),
        "budget": int(args.budget),
        "optimizer": str(args.optimizer),
        "max_wall_seconds": float(args.max_wall_seconds),
        "local_max_on_shortlist": bool(args.local_max_on_shortlist),
        "retain_all_coordinates": bool(args.retain_all_coordinates),
        "calibrate_16_cell": bool(args.calibrate_16_cell),
        "workers": 1,
        "blas_threads": BLAS_THREADS,
    }
    _validate_run_config(run_config)
    return run_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.validate_only is not None:
        try:
            validate_campaign_document(load_json_strict(args.validate_only))
        except (RuntimeError, ValueError) as exc:
            parser.exit(2, f"error: {exc}\n")
        print(f"Valid H4 symmetric-maxima campaign v1: {args.validate_only}")
        return 0
    if args.output is None:
        parser.error("--output is required unless --validate-only is used")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    try:
        run_config = _run_config_from_args(args)
        document = run_campaign(
            args.output,
            run_config,
            resume=bool(args.resume),
            overwrite=bool(args.overwrite),
            invocation=list(
                sys.argv if argv is None else [GENERATOR_NAME, *list(argv)]
            ),
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        f"{document['status']}: {document['summary']['finished']}/"
        f"{document['summary']['planned']} starts; "
        f"certified={document['summary']['certified_nontrivial_symmetry']}; "
        f"output={args.output}"
    )
    return 0 if document["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
