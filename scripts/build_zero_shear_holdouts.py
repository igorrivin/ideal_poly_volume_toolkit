#!/usr/bin/env python3
"""Build leakage-resistant zero-shear LLM holdout packages.

The public package contains paste-ready prompts and a manifest with only
opaque sequential identifiers and public-artifact hashes.  The optional
private package contains answer keys, the relabeling maps, input provenance,
basin evidence, and an independent strict realization audit.

Both outputs are staged beside their destinations.  Their destination
directories are reserved with exclusive creation, and manifests are moved
last as completion markers.  Existing destinations are never replaced.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ideal_poly_volume_toolkit.rivin_holonomy import (  # noqa: E402
    validate_closed_sphere_triangulation,
)
from ideal_poly_volume_toolkit.rivin_zero_shear import (  # noqa: E402
    _normalize_cp1_points,
    audit_zero_shear_realization,
)


PUBLIC_SCHEMA_VERSION = "ideal-poly-zero-shear-holdouts-public/1"
PRIVATE_SCHEMA_VERSION = "ideal-poly-zero-shear-holdouts-private/1"
ANSWER_KEY_SCHEMA_VERSION = "strict-zero-shear-answer-key/1"
PROMPT_PROTOCOL_VERSION = "strict-zero-shear-prompt/1"
GENERATOR_VERSION = "1.0.0"
GENERATOR_NAME = "scripts/build_zero_shear_holdouts.py"
CHECKER_NAME = "scripts/check_zero_shear_response.py"
CHECKER_SCHEMA_VERSION = "strict-zero-shear-response-grade/1"
CORPUS_SCHEMA_VERSION = "ideal-poly-arithmetic-corpus/1"
BASIN_SCHEMA_VERSION = "ideal-poly-tutte-zero-shear-audit/1"
BASIN_PROTOCOL_VERSION = "weighted-tutte-zero-shear-trf/1"
DEFAULT_RELABEL_SEED = 20260817
DEFAULT_TOP_COUNT = 8
DEFAULT_SHEAR_TOLERANCE = 1e-8
DEFAULT_SUPPORT_TOLERANCE = 0.0
DEFAULT_BENDING_TOLERANCE = 1e-10
BENCHMARK_SCOPE = {
    "suite_type": "positive-only-anchor-mined-tutte-hard-stress-set",
    "primary_metric": "strict-coordinate-realization-rate",
    "existence_accuracy_is_meaningful": False,
    "certificate_scope": "double-precision-numerical-not-formal",
}
HARDNESS_CLASSES = frozenset(
    {"canonical_wrong_root", "canonical_nonconvergence", "multibasin", "fragile"}
)
FINAL_HARDNESS_CLASSES = HARDNESS_CLASSES | {"robust"}
TARGET_OUTCOME = "target_strict_zero_shear"
WRONG_ZERO_SHEAR_OUTCOMES = frozenset(
    {
        "embedded_wrong_hull_zero_shear",
        "nonembedded_zero_shear",
        "degenerate_zero_shear",
    }
)


class HoldoutInputError(ValueError):
    """An input or requested package violates the holdout contract."""


def _reject_json_constant(value: str) -> None:
    raise HoldoutInputError(f"non-standard JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HoldoutInputError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def load_json_strict(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_object,
            )
    except HoldoutInputError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HoldoutInputError(f"cannot load {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise HoldoutInputError(f"{label} root must be a JSON object")
    return value


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HoldoutInputError("non-finite values are forbidden in package JSON")
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise HoldoutInputError(f"cannot encode {type(value).__name__} as strict JSON")


def strict_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _jsonable(value),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(*parts: Any) -> bytes:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).digest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":")
    ).encode()
    return _sha256_bytes(payload)


def _ordinary_integer(value: Any, *, path: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise HoldoutInputError(f"{path} must be an integer >= {minimum}")
    return value


def _ordinary_number(value: Any, *, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise HoldoutInputError(f"{path} must be an ordinary JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise HoldoutInputError(f"{path} must be finite")
    return result


def _parse_faces(value: Any, *, path: str) -> tuple[tuple[int, int, int], ...]:
    if not isinstance(value, list) or not value:
        raise HoldoutInputError(f"{path} must be a nonempty array")
    faces: list[tuple[int, int, int]] = []
    for index, raw_face in enumerate(value):
        if not isinstance(raw_face, list) or len(raw_face) != 3:
            raise HoldoutInputError(f"{path}[{index}] must have three labels")
        face = tuple(
            _ordinary_integer(label, path=f"{path}[{index}][{offset}]")
            for offset, label in enumerate(raw_face)
        )
        if len(set(face)) != 3:
            raise HoldoutInputError(f"{path}[{index}] repeats a vertex")
        faces.append(face)
    try:
        topology = validate_closed_sphere_triangulation(faces)
    except (TypeError, ValueError) as exc:
        raise HoldoutInputError(
            f"invalid closed triangulation at {path}: {exc}"
        ) from exc
    expected = tuple(range(topology.n_vertices))
    if topology.vertex_labels != expected:
        raise HoldoutInputError(f"{path} labels must be exactly 0 through N-1")
    return tuple(faces)


def _parse_vertices(
    value: Any, *, n_vertices: int, path: str
) -> tuple[complex | None, ...]:
    if not isinstance(value, list) or len(value) != n_vertices:
        raise HoldoutInputError(f"{path} must contain exactly {n_vertices} vertices")
    points: list[complex | None] = []
    for index, row in enumerate(value):
        if row is None:
            points.append(None)
            continue
        if not isinstance(row, list) or len(row) != 2:
            raise HoldoutInputError(
                f"{path}[{index}] must be null or a two-number array"
            )
        points.append(
            complex(
                _ordinary_number(row[0], path=f"{path}[{index}][0]"),
                _ordinary_number(row[1], path=f"{path}[{index}][1]"),
            )
        )
    if sum(point is None for point in points) != 1:
        raise HoldoutInputError(f"{path} must contain exactly one point at infinity")
    return tuple(points)


def _validate_exact_arithmetic_certificate(
    configuration: Mapping[str, Any], *, identifier: str, topology: Any
) -> int:
    ground_truth = configuration.get("ground_truth")
    if (
        not isinstance(ground_truth, dict)
        or ground_truth.get("is_arithmetic") is not True
    ):
        raise HoldoutInputError(
            f"configuration {identifier!r} lacks arithmetic ground truth"
        )
    if ground_truth.get("classification") != "zero-shear-psl2z":
        raise HoldoutInputError(
            f"configuration {identifier!r} has the wrong arithmetic classification"
        )
    certificate = configuration.get("certificate")
    if not isinstance(certificate, dict):
        raise HoldoutInputError(
            f"configuration {identifier!r} has no exact arithmetic certificate"
        )
    if certificate.get("protocol_version") != "strict-zero-shear-psl2z/1":
        raise HoldoutInputError(
            f"configuration {identifier!r} has an unsupported certificate protocol"
        )
    checks = certificate.get("checks")
    required_checks = (
        "accepted",
        "closed_sphere",
        "coordinate_zero_shear",
        "exact_determinants_one",
        "exact_integral_traces",
        "exact_total_vertex_count",
    )
    if not isinstance(checks, dict) or any(
        checks.get(key) is not True for key in required_checks
    ):
        raise HoldoutInputError(
            f"configuration {identifier!r} did not pass every arithmetic certificate gate"
        )
    exact = certificate.get("exact_zero_shear")
    if not isinstance(exact, dict) or not (
        exact.get("certificate_type") == "exact-zero-shear-psl2z-v1"
        and exact.get("all_determinants_one") is True
        and exact.get("all_traces_integral") is True
        and exact.get("is_arithmetic") is True
        and exact.get("scope") == "abstract-zero-shear-structure"
    ):
        raise HoldoutInputError(
            f"configuration {identifier!r} has no exact PSL(2,Z) determinant/trace proof"
        )
    certificate_topology = certificate.get("topology")
    expected_topology = {
        "n_vertices": topology.n_vertices,
        "n_edges": topology.n_edges,
        "n_faces": topology.n_faces,
        "euler_characteristic": 2,
    }
    if not isinstance(certificate_topology, dict) or any(
        certificate_topology.get(key) != value
        for key, value in expected_topology.items()
    ):
        raise HoldoutInputError(
            f"configuration {identifier!r} certificate topology is inconsistent"
        )
    exact_topology = exact.get("topology")
    expected_labels = list(range(topology.n_vertices))
    if (
        not isinstance(exact_topology, dict)
        or any(
            exact_topology.get(key) != value for key, value in expected_topology.items()
        )
        or exact_topology.get("vertex_labels") != expected_labels
    ):
        raise HoldoutInputError(
            f"configuration {identifier!r} nested exact topology is inconsistent"
        )
    # For this torsion-free zero-shear ideal-triangle quotient, area gives
    # index = 3F.  Require the exact certificate to state that same value.
    psl2z_index = exact.get("psl2z_index")
    if (
        not isinstance(psl2z_index, int)
        or isinstance(psl2z_index, bool)
        or psl2z_index != 3 * topology.n_faces
    ):
        raise HoldoutInputError(
            f"configuration {identifier!r} has an inconsistent exact PSL(2,Z) index"
        )
    return psl2z_index


def _validate_corpus(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if document.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise HoldoutInputError(f"corpus must use {CORPUS_SCHEMA_VERSION}")
    if document.get("status") != "complete":
        raise HoldoutInputError("arithmetic corpus must have status 'complete'")
    raw_configurations = document.get("configurations")
    if not isinstance(raw_configurations, list) or not raw_configurations:
        raise HoldoutInputError("arithmetic corpus must contain configurations")
    configurations: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_configurations):
        if not isinstance(raw, dict):
            raise HoldoutInputError(f"configurations[{index}] must be an object")
        identifier = raw.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise HoldoutInputError(f"configurations[{index}].id must be a string")
        if identifier in configurations:
            raise HoldoutInputError(f"duplicate corpus configuration id {identifier!r}")
        input_data = raw.get("input")
        if not isinstance(input_data, dict):
            raise HoldoutInputError(f"configuration {identifier!r} has no input object")
        faces = _parse_faces(
            input_data.get("triangles"),
            path=f"configuration {identifier}.input.triangles",
        )
        topology = validate_closed_sphere_triangulation(faces)
        n_vertices = _ordinary_integer(
            input_data.get("n_vertices"),
            path=f"configuration {identifier}.input.n_vertices",
            minimum=4,
        )
        if n_vertices != topology.n_vertices:
            raise HoldoutInputError(
                f"configuration {identifier!r} vertex count disagrees with its faces"
            )
        points = _parse_vertices(
            input_data.get("vertices"),
            n_vertices=n_vertices,
            path=f"configuration {identifier}.input.vertices",
        )
        infinity_index = _ordinary_integer(
            input_data.get("infinity_index"),
            path=f"configuration {identifier}.input.infinity_index",
        )
        if infinity_index >= n_vertices or points[infinity_index] is not None:
            raise HoldoutInputError(
                f"configuration {identifier!r} infinity_index is inconsistent"
            )
        psl2z_index = _validate_exact_arithmetic_certificate(
            raw, identifier=identifier, topology=topology
        )
        topology_id = raw.get("deduplication", {}).get("topology_id")
        if not isinstance(topology_id, str) or not topology_id:
            raise HoldoutInputError(
                f"configuration {identifier!r} requires a nonempty topology_id"
            )
        configurations[identifier] = {
            "raw": raw,
            "faces": faces,
            "points": points,
            "topology": topology,
            "infinity_index": infinity_index,
            "psl2z_index": psl2z_index,
            "topology_id": topology_id,
        }
    return configurations


def _recompute_hardness(rows: Sequence[Mapping[str, Any]]) -> tuple[str, list[float]]:
    uniform = [row for row in rows if row["start_spec"]["kind"] == "uniform"]
    if len(uniform) != 1:
        raise HoldoutInputError(
            "every finalized basin case must have one uniform start"
        )
    uniform_outcome = uniform[0]["outcome"]
    weighted = [row for row in rows if row["start_spec"]["kind"] != "uniform"]
    outcomes = [row["outcome"] for row in rows]
    wrong_count = sum(outcome in WRONG_ZERO_SHEAR_OUTCOMES for outcome in outcomes)
    target_count = sum(outcome == TARGET_OUTCOME for outcome in outcomes)
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
    elif uniform_outcome != TARGET_OUTCOME:
        label, tier = "canonical_nonconvergence", 3
    elif any(row["outcome"] in WRONG_ZERO_SHEAR_OUTCOMES for row in weighted):
        label, tier = "multibasin", 2
    elif any(row["outcome"] != TARGET_OUTCOME for row in weighted):
        label, tier = "fragile", 1
    else:
        label, tier = "robust", 0
    return label, [
        float(tier),
        float(wrong_count / total),
        float(1.0 - target_count / total),
        float(root_count),
    ]


def _validate_basin(
    document: Mapping[str, Any], *, corpus_sha256: str
) -> dict[str, dict[str, Any]]:
    if document.get("schema_version") != BASIN_SCHEMA_VERSION:
        raise HoldoutInputError(f"basin audit must use {BASIN_SCHEMA_VERSION}")
    if document.get("protocol_version") != BASIN_PROTOCOL_VERSION:
        raise HoldoutInputError("basin audit protocol does not match this exporter")
    if document.get("status") != "complete":
        raise HoldoutInputError("basin audit must be finalized with status 'complete'")
    run_config = document.get("run_config")
    if not isinstance(run_config, dict) or document.get(
        "run_config_sha256"
    ) != _canonical_json_sha256(run_config):
        raise HoldoutInputError("basin run_config hash does not match its contents")
    basin_input = document.get("input")
    if not isinstance(basin_input, dict) or basin_input.get("sha256") != corpus_sha256:
        raise HoldoutInputError(
            "basin audit is not derived from the supplied corpus bytes"
        )
    selected_ids = basin_input.get("selected_configuration_ids")
    if not isinstance(selected_ids, list) or any(
        not isinstance(item, str) for item in selected_ids
    ):
        raise HoldoutInputError("basin selected_configuration_ids must be strings")
    if len(set(selected_ids)) != len(selected_ids):
        raise HoldoutInputError("basin selected_configuration_ids must be unique")

    summary = document.get("summary")
    if not isinstance(summary, dict):
        raise HoldoutInputError("basin audit has no summary object")
    counts = {
        key: _ordinary_integer(summary.get(key), path=f"basin.summary.{key}")
        for key in (
            "planned",
            "finished",
            "completed",
            "errors",
            "timed_out",
            "pending",
        )
    }
    if not (
        counts["planned"] == counts["finished"] == counts["completed"]
        and counts["errors"] == counts["timed_out"] == counts["pending"] == 0
    ):
        raise HoldoutInputError(
            "final basin audit must have every task completed cleanly"
        )

    tasks = document.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != counts["finished"]:
        raise HoldoutInputError("basin task rows disagree with the finalized summary")
    task_ids: set[str] = set()
    task_indices: set[int] = set()
    tasks_by_case: Counter[str] = Counter()
    task_outcomes_by_case: dict[str, Counter[str]] = {}
    task_rows_by_case: dict[str, list[dict[str, Any]]] = {}
    global_outcomes: Counter[str] = Counter()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise HoldoutInputError(f"basin.tasks[{index}] must be an object")
        task_id = task.get("task_id")
        case_id = task.get("case_id")
        task_index = task.get("task_index")
        outcome = task.get("outcome")
        if not all(
            isinstance(item, str) and item for item in (task_id, case_id, outcome)
        ):
            raise HoldoutInputError(f"basin.tasks[{index}] has invalid identity fields")
        if task.get("status") != "completed":
            raise HoldoutInputError(f"basin.tasks[{index}] is not completed")
        core_result = task.get("core_result")
        if (
            not isinstance(core_result, dict)
            or core_result.get("classification") != outcome
        ):
            raise HoldoutInputError(
                f"basin.tasks[{index}] outcome disagrees with its core result"
            )
        if outcome in {TARGET_OUTCOME, *WRONG_ZERO_SHEAR_OUTCOMES}:
            core_audit = core_result.get("audit")
            expected_strict = outcome == TARGET_OUTCOME
            if not isinstance(core_audit, dict) or (
                core_audit.get("classification") != outcome
                or core_audit.get("strict_pass") is not expected_strict
                or core_result.get("strict_pass") is not expected_strict
            ):
                raise HoldoutInputError(
                    f"basin.tasks[{index}] zero-shear outcome disagrees with its audit"
                )
            if not isinstance(task.get("root_fingerprint"), str):
                raise HoldoutInputError(
                    f"basin.tasks[{index}] zero-shear root has no fingerprint"
                )
        elif outcome in {"solver_failure", "residual_failure"}:
            if (
                core_result.get("success") is not False
                or task.get("root_fingerprint") is not None
            ):
                raise HoldoutInputError(
                    f"basin.tasks[{index}] failure semantics are inconsistent"
                )
        else:
            raise HoldoutInputError(f"basin.tasks[{index}] has an unknown outcome")
        start_spec = task.get("start_spec")
        if not isinstance(start_spec, dict) or start_spec.get("kind") not in {
            "uniform",
            "lognormal",
        }:
            raise HoldoutInputError(f"basin.tasks[{index}] has an invalid start_spec")
        if not isinstance(task.get("configuration_id"), str) or not isinstance(
            task.get("distinguished_face"), list
        ):
            raise HoldoutInputError(
                f"basin.tasks[{index}] lacks configuration/anchor identity"
            )
        _ordinary_integer(task_index, path=f"basin.tasks[{index}].task_index")
        if task_id in task_ids or task_index in task_indices:
            raise HoldoutInputError("basin task identities must be unique")
        task_ids.add(task_id)
        task_indices.add(task_index)
        tasks_by_case[case_id] += 1
        task_outcomes_by_case.setdefault(case_id, Counter())[outcome] += 1
        task_rows_by_case.setdefault(case_id, []).append(task)
        global_outcomes[outcome] += 1
    if task_indices != set(range(counts["planned"])):
        raise HoldoutInputError("basin task indices must be the complete range")
    if summary.get("outcome_counts") != dict(sorted(global_outcomes.items())):
        raise HoldoutInputError("basin summary outcome counts disagree with task rows")

    raw_cases = document.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise HoldoutInputError("basin audit contains no cases")
    cases: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(raw_cases):
        if not isinstance(case, dict):
            raise HoldoutInputError(f"basin.cases[{index}] must be an object")
        case_id = case.get("case_id")
        configuration_id = case.get("configuration_id")
        if not isinstance(case_id, str) or not isinstance(configuration_id, str):
            raise HoldoutInputError(f"basin.cases[{index}] has invalid identifiers")
        if case_id in cases:
            raise HoldoutInputError(f"duplicate basin case id {case_id!r}")
        if configuration_id not in selected_ids:
            raise HoldoutInputError(
                f"basin case {case_id!r} is outside selected_configuration_ids"
            )
        anchor = case.get("distinguished_face")
        if not isinstance(anchor, list) or len(anchor) != 3:
            raise HoldoutInputError(f"basin case {case_id!r} has an invalid anchor")
        anchor_tuple = tuple(
            _ordinary_integer(value, path=f"basin case {case_id}.distinguished_face")
            for value in anchor
        )
        if len(set(anchor_tuple)) != 3:
            raise HoldoutInputError(f"basin case {case_id!r} anchor repeats a label")
        planned = _ordinary_integer(
            case.get("planned_task_count"),
            path=f"basin case {case_id}.planned_task_count",
            minimum=1,
        )
        finished = _ordinary_integer(
            case.get("finished_task_count"),
            path=f"basin case {case_id}.finished_task_count",
        )
        if planned != finished or tasks_by_case[case_id] != planned:
            raise HoldoutInputError(f"basin case {case_id!r} is incomplete")
        case_tasks = task_rows_by_case[case_id]
        if any(
            task["configuration_id"] != configuration_id
            or task["distinguished_face"] != list(anchor_tuple)
            for task in case_tasks
        ):
            raise HoldoutInputError(
                f"basin case {case_id!r} identity disagrees with its task rows"
            )
        outcome_counts = case.get("outcome_counts")
        if not isinstance(outcome_counts, dict):
            raise HoldoutInputError(f"basin case {case_id!r} has no outcome counts")
        parsed_outcomes = {
            str(key): _ordinary_integer(
                value, path=f"basin case {case_id}.outcome_counts"
            )
            for key, value in outcome_counts.items()
        }
        if parsed_outcomes != dict(
            sorted(task_outcomes_by_case.get(case_id, {}).items())
        ):
            raise HoldoutInputError(
                f"basin case {case_id!r} outcome counts are inconsistent"
            )
        target_count = _ordinary_integer(
            case.get("target_success_count"),
            path=f"basin case {case_id}.target_success_count",
        )
        if target_count != parsed_outcomes.get(TARGET_OUTCOME, 0):
            raise HoldoutInputError(
                f"basin case {case_id!r} target count is inconsistent"
            )
        if case.get("target_success_fraction") != float(target_count / planned):
            raise HoldoutInputError(
                f"basin case {case_id!r} target fraction is inconsistent"
            )
        hardness = case.get("hardness_class")
        rank = case.get("hardness_rank")
        if hardness not in FINAL_HARDNESS_CLASSES:
            raise HoldoutInputError(
                f"basin case {case_id!r} has no finalized hardness class: {hardness!r}"
            )
        _ordinary_integer(rank, path=f"basin case {case_id}.hardness_rank", minimum=1)
        recomputed_hardness, recomputed_key = _recompute_hardness(case_tasks)
        uniform_outcome = next(
            task["outcome"]
            for task in case_tasks
            if task["start_spec"]["kind"] == "uniform"
        )
        if (
            hardness != recomputed_hardness
            or case.get("hardness_key") != recomputed_key
            or case.get("uniform_outcome") != uniform_outcome
        ):
            raise HoldoutInputError(
                f"basin case {case_id!r} hardness evidence is inconsistent"
            )
        cases[case_id] = {**case, "distinguished_face": list(anchor_tuple)}
    if set(tasks_by_case) != set(cases):
        raise HoldoutInputError("basin cases and task rows do not cover the same cases")
    ranked = sorted(
        cases.values(),
        key=lambda case: (tuple(case["hardness_key"]), case["case_id"]),
        reverse=True,
    )
    expected_ranks = {case["case_id"]: index for index, case in enumerate(ranked, 1)}
    if any(
        case["hardness_rank"] != expected_ranks[case_id]
        for case_id, case in cases.items()
    ):
        raise HoldoutInputError("basin hardness ranks disagree with recomputed ranks")
    return cases


def select_cases(
    cases: Mapping[str, dict[str, Any]],
    configurations: Mapping[str, dict[str, Any]],
    *,
    requested_case_ids: Sequence[str] | None,
    top_count: int,
    relabel_seed: int,
    excluded_configuration_ids: Sequence[str] = (),
) -> list[dict[str, Any]]:
    def topology_id(case: Mapping[str, Any]) -> str | None:
        configuration = configurations.get(str(case["configuration_id"]), {})
        return configuration.get("topology_id")

    excluded = set(excluded_configuration_ids)
    if len(excluded) != len(excluded_configuration_ids):
        raise HoldoutInputError("excluded configuration IDs must be unique")
    unknown_exclusions = sorted(excluded - set(configurations))
    if unknown_exclusions:
        raise HoldoutInputError(
            f"unknown excluded configuration ID: {unknown_exclusions[0]!r}"
        )

    if requested_case_ids:
        if len(set(requested_case_ids)) != len(requested_case_ids):
            raise HoldoutInputError("--case-id values must be unique")
        missing = sorted(set(requested_case_ids) - set(cases))
        if missing:
            raise HoldoutInputError(f"unknown basin case id {missing[0]!r}")
        selected = [cases[case_id] for case_id in requested_case_ids]
        nonhard = [
            case["case_id"]
            for case in selected
            if case["hardness_class"] not in HARDNESS_CLASSES
        ]
        if nonhard:
            raise HoldoutInputError(
                f"selected basin case is not classified as hard: {nonhard[0]!r}"
            )
        exposed = [
            case["case_id"] for case in selected if case["configuration_id"] in excluded
        ]
        if exposed:
            raise HoldoutInputError(
                f"selected case belongs to an excluded configuration: {exposed[0]!r}"
            )
    else:
        if top_count < 1:
            raise HoldoutInputError("--top must be positive")
        ranked = sorted(
            (
                case
                for case in cases.values()
                if case["hardness_class"] in HARDNESS_CLASSES
                and case["configuration_id"] not in excluded
            ),
            key=lambda case: (tuple(case["hardness_key"]), case["case_id"]),
            reverse=True,
        )
        selected = []
        seen_configurations: set[str] = set()
        seen_topologies: set[str] = set()
        for case in ranked:
            configuration_id = case["configuration_id"]
            source_topology = topology_id(case)
            if configuration_id in seen_configurations or (
                source_topology is not None and source_topology in seen_topologies
            ):
                continue
            selected.append(case)
            seen_configurations.add(configuration_id)
            if source_topology is not None:
                seen_topologies.add(source_topology)
            if len(selected) == top_count:
                break
        if len(selected) != top_count:
            raise HoldoutInputError(
                f"requested {top_count} topology-diverse hard cases but only "
                f"{len(selected)} are eligible"
            )
    configuration_ids = [case["configuration_id"] for case in selected]
    source_topologies = [topology_id(case) for case in selected]
    if len(set(configuration_ids)) != len(configuration_ids) or len(
        {value for value in source_topologies if value is not None}
    ) != sum(value is not None for value in source_topologies):
        raise HoldoutInputError(
            "selected holdouts must use distinct configurations and source topologies"
        )
    # Public sequential IDs must not reveal hardness rank or source ordering.
    return sorted(
        selected,
        key=lambda case: _stable_digest(
            "instance-order", relabel_seed, case["case_id"]
        ),
    )


def _relabeling(
    n_vertices: int,
    anchor: tuple[int, int, int],
    *,
    case_id: str,
    relabel_seed: int,
) -> tuple[dict[int, int], list[int]]:
    if any(label >= n_vertices for label in anchor):
        raise HoldoutInputError(f"case {case_id!r} anchor is outside its vertex range")
    remaining = [label for label in range(n_vertices) if label not in anchor]
    remaining.sort(
        key=lambda label: _stable_digest("vertex", relabel_seed, case_id, label)
    )
    new_to_old = [*anchor, *remaining]
    old_to_new = {old: new for new, old in enumerate(new_to_old)}
    return old_to_new, new_to_old


def _cyclic_rotations(face: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
    a, b, c = face
    return face, (b, c, a), (c, a, b)


def _relabel_faces(
    faces: Sequence[tuple[int, int, int]],
    old_to_new: Mapping[int, int],
    *,
    case_id: str,
    relabel_seed: int,
) -> tuple[tuple[int, int, int], ...]:
    rotated: list[tuple[int, int, int]] = []
    distinguished: tuple[int, int, int] | None = None
    for source_index, face in enumerate(faces):
        relabeled = tuple(old_to_new[label] for label in face)
        if frozenset(relabeled) == {0, 1, 2}:
            cycles = _cyclic_rotations(relabeled)
            if (0, 1, 2) not in cycles:
                raise HoldoutInputError(
                    "selected ordered anchor has the opposite winding from its source face"
                )
            distinguished = (0, 1, 2)
            continue
        offset = (
            int.from_bytes(
                _stable_digest("face-rotation", relabel_seed, case_id, source_index)[
                    :2
                ],
                "big",
            )
            % 3
        )
        rotated.append(_cyclic_rotations(relabeled)[offset])
    if distinguished is None:
        raise HoldoutInputError(
            "selected ordered anchor is not an oriented source face"
        )
    rotated.sort(
        key=lambda face: _stable_digest("face-order", relabel_seed, case_id, face)
    )
    return (distinguished, *rotated)


def _normalize_and_relabel_points(
    points: Sequence[complex | None],
    anchor: tuple[int, int, int],
    new_to_old: Sequence[int],
) -> tuple[tuple[complex | None, ...], bool]:
    normalized = list(_normalize_cp1_points(points, anchor))
    nonanchor_imaginary = [
        point.imag
        for label, point in enumerate(normalized)
        if label not in anchor and point is not None
    ]
    if not nonanchor_imaginary:
        raise HoldoutInputError("normalization has no non-anchor finite vertices")
    positive = all(value > DEFAULT_BENDING_TOLERANCE for value in nonanchor_imaginary)
    negative = all(value < -DEFAULT_BENDING_TOLERANCE for value in nonanchor_imaginary)
    if not (positive or negative):
        raise HoldoutInputError(
            "normalized target does not lie strictly in one open half-plane"
        )
    conjugated = negative
    if conjugated:
        normalized = [
            None if point is None else point.conjugate() for point in normalized
        ]
    relabeled = tuple(normalized[old_label] for old_label in new_to_old)
    if relabeled[:3] != (None, 0.0 + 0.0j, 1.0 + 0.0j):
        raise HoldoutInputError(
            "normalization did not produce exact infinity, zero, one anchors"
        )
    return relabeled, conjugated


def _coordinate_rows(points: Sequence[complex | None]) -> list[Any]:
    rows: list[Any] = []
    for point in points:
        if point is None:
            rows.append(None)
        else:
            if not (math.isfinite(point.real) and math.isfinite(point.imag)):
                raise HoldoutInputError(
                    "normalized coordinates must be finite away from infinity"
                )
            rows.append([float(point.real), float(point.imag)])
    return rows


def _round_points_significantly(
    points: Sequence[complex | None], digits: int
) -> tuple[complex | None, ...]:
    """Round finite components exactly as a minimally compliant response may."""
    return tuple(
        None
        if point is None
        else complex(
            float(format(point.real, f".{digits}g")),
            float(format(point.imag, f".{digits}g")),
        )
        for point in points
    )


def _audit_evidence(audit: Any) -> dict[str, Any]:
    value = audit.as_dict()
    support = value["support"]
    shear = value["shear"]
    bending = value["bending"]
    spherical = value["spherical_embedding"]
    qhull = value["qhull_agreement"]
    return {
        "verification_scope": "double-precision-numerical-not-formal",
        "strict_pass": value["strict_pass"],
        "classification": value["classification"],
        "normalization": {
            "coordinate_anchors_match": value["normalization"][
                "coordinate_anchors_match"
            ],
            "handedness_matches": value["normalization"]["handedness_matches"],
            "min_nonanchor_imaginary": value["normalization"][
                "min_nonanchor_imaginary"
            ],
        },
        "topology": {
            key: value["topology"][key]
            for key in ("n_vertices", "n_edges", "n_faces", "euler_characteristic")
        },
        "chordal_distinctness": {
            "passes": value["chordal_distinctness"]["passes"],
            "min_chordal_distance": value["chordal_distinctness"][
                "min_chordal_distance"
            ],
        },
        "support": {
            key: support[key]
            for key in (
                "passes",
                "min_normalized_margin",
                "adaptive_fallback_count",
                "exact_rational_fallback_count",
                "indeterminate_count",
                "tie_count",
                "inconsistent_face_count",
            )
        },
        "shear": {
            key: shear[key]
            for key in (
                "passes",
                "tolerance",
                "max_abs",
                "rms",
                "stable_crosscheck_max_abs",
            )
        },
        "bending": {
            key: bending[key]
            for key in (
                "passes",
                "orientation_sign",
                "tolerance",
                "min_endpoint_margin_radians",
                "sign_violation_count",
            )
        },
        "spherical_embedding": {
            "passes": spherical["passes"],
            "embedded": spherical["embedded"],
            "orientation_coherent": spherical["orientation_coherent"],
        },
        "qhull_agreement": {
            "passes": qhull["passes"],
            "agrees": qhull["agrees"],
            "n_qhull_faces": qhull["n_qhull_faces"],
            "n_labelled_faces": qhull["n_labelled_faces"],
            "missing_face_count": len(qhull["missing_faces"]),
            "extra_face_count": len(qhull["extra_faces"]),
        },
    }


def _prompt_text(*, n_vertices: int, n_faces: int, psl2z_index: int, faces: Any) -> str:
    last = n_vertices - 1
    face_json = json.dumps(faces, ensure_ascii=True, separators=(",", ":"))
    return f"""You are given a labeled, coherently oriented triangulation T of the 2-sphere. It has {n_vertices} vertices labeled 0 through {last} and {n_faces} triangular faces. The distinguished oriented face is (0,1,2).

Determine whether there exist pairwise-distinct points z_0,...,z_{last} in the Riemann sphere C-hat satisfying all of the following conditions:

1. Normalization and handedness:
   - z_0 = infinity,
   - z_1 = 0,
   - z_2 = 1,
   - Im(z_i) > 0 for every i = 3,...,{last}.

2. Exact labeled combinatorics:
   The strict convex ideal hull of these points in hyperbolic 3-space has exactly the labeled triangular boundary faces listed below. Cyclic rotations of an oriented face are equivalent, but vertex relabeling is not allowed. "Strict" means that no listed edge is flat and no two adjacent listed triangles are coplanar.

3. Zero shear:
   For every edge {{i,j}}, let its two coherently oriented incident faces be (i,j,k) and (j,i,l), up to cyclic rotation. Then

       log | ((z_k-z_i)(z_l-z_j)) / ((z_k-z_j)(z_l-z_i)) | = 0,

   using the projective limiting value whenever one of the vertices is infinity.

Equivalently, this asks whether T itself, without deleting or flipping edges, is the strict face triangulation of the convex ideal realization of its zero-shear metric. If it exists, its intrinsic holonomy is conjugate to a torsion-free subgroup of PSL(2,Z) of index {psl2z_index}.

Output requirements:

- If no such strict realization exists, return exactly:
  {{"exists": false}}

- If it exists, return exactly one JSON object of the form:
  {{"exists": true, "vertices": [null, [0.0, 0.0], [1.0, 0.0], [x_3, y_3], ..., [x_{last}, y_{last}]]}}

Use ordinary JSON numbers, preserve the vertex labels, provide at least 12 significant decimal digits per non-anchor coordinate, and include no prose or Markdown outside the JSON object.

The oriented face list T is:
{face_json}
"""


def _public_manifest(instances: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "instance_id": item["instance_id"],
            "prompt_file": item["prompt_file"],
            "prompt_sha256": item["prompt_sha256"],
            "n_vertices": item["n_vertices"],
            "n_edges": item["n_edges"],
            "n_faces": item["n_faces"],
            "psl2z_index": item["psl2z_index"],
        }
        for item in instances
    ]
    return {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "protocol_version": PROMPT_PROTOCOL_VERSION,
        "benchmark_scope": dict(BENCHMARK_SCOPE),
        "generator": {
            "name": GENERATOR_NAME,
            "version": GENERATOR_VERSION,
            "source_sha256": _sha256_file(__file__),
        },
        "grading_contract": {
            "checker": CHECKER_NAME,
            "checker_schema_version": CHECKER_SCHEMA_VERSION,
            "implementation_source_sha256": {
                CHECKER_NAME: _sha256_file(REPOSITORY_ROOT / CHECKER_NAME),
                "ideal_poly_volume_toolkit/rivin_zero_shear.py": _sha256_file(
                    REPOSITORY_ROOT / "ideal_poly_volume_toolkit/rivin_zero_shear.py"
                ),
                "ideal_poly_volume_toolkit/rivin_holonomy.py": _sha256_file(
                    REPOSITORY_ROOT / "ideal_poly_volume_toolkit/rivin_holonomy.py"
                ),
            },
            "dependency_versions": {
                name: importlib.metadata.version(name) for name in ("numpy", "scipy")
            },
            "runtime": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "system": platform.system(),
                "machine": platform.machine(),
            },
            "verification_scope": "double-precision-numerical-not-formal",
            "thresholds": {
                "shear_tolerance": DEFAULT_SHEAR_TOLERANCE,
                "support_tolerance": DEFAULT_SUPPORT_TOLERANCE,
                "bending_tolerance": DEFAULT_BENDING_TOLERANCE,
            },
        },
        "instance_count": len(rows),
        "instances": rows,
    }


def _validate_public_manifest(document: Mapping[str, Any]) -> None:
    if set(document) != {
        "schema_version",
        "protocol_version",
        "benchmark_scope",
        "generator",
        "grading_contract",
        "instance_count",
        "instances",
    }:
        raise HoldoutInputError("public manifest has an unexpected field")
    if document.get("schema_version") != PUBLIC_SCHEMA_VERSION:
        raise HoldoutInputError("public manifest schema mismatch")
    if document.get("benchmark_scope") != BENCHMARK_SCOPE:
        raise HoldoutInputError("public benchmark scope is malformed")
    generator = document.get("generator")
    if not isinstance(generator, dict) or set(generator) != {
        "name",
        "version",
        "source_sha256",
    }:
        raise HoldoutInputError("public generator provenance has an unexpected field")
    grading = document.get("grading_contract")
    if not isinstance(grading, dict) or set(grading) != {
        "checker",
        "checker_schema_version",
        "implementation_source_sha256",
        "dependency_versions",
        "runtime",
        "verification_scope",
        "thresholds",
    }:
        raise HoldoutInputError("public grading contract has an unexpected field")
    thresholds = grading.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != {
        "shear_tolerance",
        "support_tolerance",
        "bending_tolerance",
    }:
        raise HoldoutInputError("public grading thresholds have an unexpected field")
    implementation_sources = grading.get("implementation_source_sha256")
    expected_sources = {
        CHECKER_NAME,
        "ideal_poly_volume_toolkit/rivin_zero_shear.py",
        "ideal_poly_volume_toolkit/rivin_holonomy.py",
    }
    if (
        not isinstance(implementation_sources, dict)
        or set(implementation_sources) != expected_sources
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in implementation_sources.values()
        )
    ):
        raise HoldoutInputError("public grading source provenance is malformed")
    dependencies = grading.get("dependency_versions")
    if not isinstance(dependencies, dict) or set(dependencies) != {"numpy", "scipy"}:
        raise HoldoutInputError("public dependency provenance is malformed")
    runtime = grading.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {
        "python",
        "python_implementation",
        "system",
        "machine",
    }:
        raise HoldoutInputError("public runtime provenance is malformed")
    instances = document.get("instances")
    if not isinstance(instances, list) or document.get("instance_count") != len(
        instances
    ):
        raise HoldoutInputError("public manifest instance count mismatch")
    expected_ids = [
        f"ipr-holdout-{index:03d}" for index in range(1, len(instances) + 1)
    ]
    if [row.get("instance_id") for row in instances] != expected_ids:
        raise HoldoutInputError("public manifest IDs are not opaque sequential IDs")
    allowed = {
        "instance_id",
        "prompt_file",
        "prompt_sha256",
        "n_vertices",
        "n_edges",
        "n_faces",
        "psl2z_index",
    }
    if any(set(row) != allowed for row in instances):
        raise HoldoutInputError("public instance metadata contains a private field")
    strict_json_bytes(document)


def _validate_private_manifest(document: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "protocol_version",
        "benchmark_scope",
        "generator",
        "grading_contract",
        "inputs",
        "run_config",
        "provenance",
        "public_manifest_sha256",
        "instance_count",
        "instances",
    }
    if (
        set(document) != required
        or document.get("schema_version") != PRIVATE_SCHEMA_VERSION
        or document.get("benchmark_scope") != BENCHMARK_SCOPE
    ):
        raise HoldoutInputError("private manifest schema mismatch")
    instances = document.get("instances")
    if not isinstance(instances, list) or document.get("instance_count") != len(
        instances
    ):
        raise HoldoutInputError("private manifest instance count mismatch")
    expected_ids = [
        f"ipr-holdout-{index:03d}" for index in range(1, len(instances) + 1)
    ]
    if [row.get("instance_id") for row in instances] != expected_ids:
        raise HoldoutInputError("private manifest instance identities mismatch")
    for name in ("public_manifest_sha256",):
        value = document.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise HoldoutInputError(f"private manifest {name} is not a SHA-256 digest")
    strict_json_bytes(document)


def _source_files() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        REPOSITORY_ROOT / "ideal_poly_volume_toolkit/rivin_zero_shear.py",
        REPOSITORY_ROOT / "ideal_poly_volume_toolkit/rivin_holonomy.py",
        REPOSITORY_ROOT / CHECKER_NAME,
    )
    return {
        str(path.relative_to(REPOSITORY_ROOT)): _sha256_file(path) for path in paths
    }


def _write_file(path: Path, payload: bytes, *, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _new_stage(destination: Path, *, private: bool) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    )
    os.chmod(stage, 0o700 if private else 0o755)
    return stage


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve(strict=False)
    right_resolved = right.resolve(strict=False)
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def _commit_stages(
    public_stage: Path,
    public_directory: Path,
    private_stage: Path | None,
    private_directory: Path | None,
) -> None:
    def commit_one(stage: Path, destination: Path, *, private: bool) -> None:
        try:
            os.mkdir(destination, 0o700 if private else 0o755)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to clobber {destination}") from exc
        try:
            files = sorted(
                stage.iterdir(),
                key=lambda path: (path.name.endswith("manifest.json"), path.name),
            )
            # Public manifest.json and private manifest.private.json are moved
            # last, so consumers can use manifest presence as completion.
            files.sort(key=lambda path: ("manifest" in path.name, path.name))
            for source in files:
                os.rename(source, destination / source.name)
            _fsync_directory(destination)
            stage.rmdir()
            _fsync_directory(destination.parent)
        except BaseException:
            shutil.rmtree(destination)
            _fsync_directory(destination.parent)
            raise

    committed_private = False
    try:
        if private_stage is not None and private_directory is not None:
            commit_one(private_stage, private_directory, private=True)
            committed_private = True
        commit_one(public_stage, public_directory, private=False)
    except BaseException:
        if committed_private and private_directory is not None:
            shutil.rmtree(private_directory)
            _fsync_directory(private_directory.parent)
        raise


def _validate_answer_key(
    document: Mapping[str, Any], *, n_vertices: int, prompt_sha256: str
) -> None:
    if document.get("schema_version") != ANSWER_KEY_SCHEMA_VERSION:
        raise HoldoutInputError("answer-key schema mismatch")
    if document.get("prompt_sha256") != prompt_sha256:
        raise HoldoutInputError("answer key is not bound to its exact prompt bytes")
    ground_truth = document.get("ground_truth")
    if not isinstance(ground_truth, dict) or ground_truth.get("exists") is not True:
        raise HoldoutInputError("answer key must contain a positive ground truth")
    faces = _parse_faces(
        ground_truth.get("oriented_faces"), path="ground_truth.oriented_faces"
    )
    _parse_vertices(
        ground_truth.get("vertices"),
        n_vertices=n_vertices,
        path="ground_truth.vertices",
    )
    if faces[0] != (0, 1, 2):
        raise HoldoutInputError("answer key must begin with distinguished face (0,1,2)")
    strict_json_bytes(document)


def build_holdout_package(
    corpus_path: str | Path,
    basin_paths: str | Path | Sequence[str | Path],
    *,
    public_directory: str | Path,
    private_directory: str | Path | None = None,
    requested_case_ids: Sequence[str] | None = None,
    top_count: int = DEFAULT_TOP_COUNT,
    relabel_seed: int = DEFAULT_RELABEL_SEED,
    excluded_configuration_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Build and no-clobber commit one deterministic holdout package."""
    public_target = Path(public_directory)
    private_target = None if private_directory is None else Path(private_directory)
    if public_target.exists():
        raise FileExistsError(f"refusing to clobber {public_target}")
    if private_target is not None:
        if private_target.exists():
            raise FileExistsError(f"refusing to clobber {private_target}")
        if _paths_overlap(public_target, private_target):
            raise HoldoutInputError(
                "public and private directories must be disjoint and non-nested"
            )
        private_resolved = private_target.resolve(strict=False)
        repository_resolved = REPOSITORY_ROOT.resolve()
        if (
            private_resolved == repository_resolved
            or repository_resolved in private_resolved.parents
        ):
            raise HoldoutInputError(
                "private key directory must be outside the repository tree"
            )
    if relabel_seed < 0 or relabel_seed >= 1 << 63:
        raise HoldoutInputError("relabel seed must lie in [0, 2**63)")

    corpus_source = Path(corpus_path)
    corpus_sha256 = _sha256_file(corpus_source)
    corpus = load_json_strict(corpus_source, label="arithmetic corpus")
    configurations = _validate_corpus(corpus)
    if isinstance(basin_paths, (str, os.PathLike)):
        basin_sources = [Path(basin_paths)]
    else:
        basin_sources = [Path(path) for path in basin_paths]
    if not basin_sources:
        raise HoldoutInputError("at least one finalized basin audit is required")
    resolved_basins = [path.resolve(strict=False) for path in basin_sources]
    if len(set(resolved_basins)) != len(resolved_basins):
        raise HoldoutInputError("basin audit paths must be unique")
    basin_inputs: list[dict[str, str]] = []
    cases: dict[str, dict[str, Any]] = {}
    for basin_source in basin_sources:
        basin_sha256 = _sha256_file(basin_source)
        basin = load_json_strict(basin_source, label="basin audit")
        basin_cases = _validate_basin(basin, corpus_sha256=corpus_sha256)
        basin_record = {
            "file": basin_source.name,
            "sha256": basin_sha256,
        }
        for case_id, case in basin_cases.items():
            candidate = {**case, "_basin_inputs": [basin_record]}
            previous = cases.get(case_id)
            if previous is None:
                cases[case_id] = candidate
                continue
            if (
                previous["configuration_id"] != case["configuration_id"]
                or previous["distinguished_face"] != case["distinguished_face"]
            ):
                raise HoldoutInputError(
                    f"case ID collision across basin audits: {case_id!r}"
                )
            evidence_inputs = [*previous["_basin_inputs"], basin_record]
            previous_tasks = int(previous["planned_task_count"])
            candidate_tasks = int(candidate["planned_task_count"])
            if candidate_tasks > previous_tasks:
                cases[case_id] = {**candidate, "_basin_inputs": evidence_inputs}
            elif candidate_tasks < previous_tasks:
                previous["_basin_inputs"] = evidence_inputs
            elif any(
                previous.get(key) != candidate.get(key)
                for key in (
                    "hardness_class",
                    "hardness_key",
                    "uniform_outcome",
                    "outcome_counts",
                )
            ):
                raise HoldoutInputError(
                    f"equal-sized basin evidence conflicts for case {case_id!r}"
                )
            else:
                previous["_basin_inputs"] = evidence_inputs
        basin_inputs.append(basin_record)
    selected = select_cases(
        cases,
        configurations,
        requested_case_ids=requested_case_ids,
        top_count=top_count,
        relabel_seed=relabel_seed,
        excluded_configuration_ids=excluded_configuration_ids,
    )
    if not selected:
        raise HoldoutInputError("no hard cases were selected")

    public_stage = _new_stage(public_target, private=False)
    private_stage = _new_stage(private_target, private=True) if private_target else None
    public_instances: list[dict[str, Any]] = []
    private_instances: list[dict[str, Any]] = []
    try:
        for sequence, case in enumerate(selected, start=1):
            instance_id = f"ipr-holdout-{sequence:03d}"
            configuration_id = case["configuration_id"]
            configuration = configurations.get(configuration_id)
            if configuration is None:
                raise HoldoutInputError(
                    f"basin case refers to unknown corpus configuration {configuration_id!r}"
                )
            anchor = tuple(case["distinguished_face"])
            if frozenset(anchor) not in {
                frozenset(face) for face in configuration["faces"]
            }:
                raise HoldoutInputError(
                    f"basin anchor {anchor} is not a face of {configuration_id!r}"
                )
            n_vertices = configuration["topology"].n_vertices
            old_to_new, new_to_old = _relabeling(
                n_vertices,
                anchor,
                case_id=case["case_id"],
                relabel_seed=relabel_seed,
            )
            faces = _relabel_faces(
                configuration["faces"],
                old_to_new,
                case_id=case["case_id"],
                relabel_seed=relabel_seed,
            )
            points, conjugated = _normalize_and_relabel_points(
                configuration["points"], anchor, new_to_old
            )

            # This is intentionally independent of corpus acceptance and basin
            # endpoint classifications: the published target itself must pass.
            strict_audit = audit_zero_shear_realization(
                faces,
                points,
                distinguished_face=(0, 1, 2),
                shear_tolerance=DEFAULT_SHEAR_TOLERANCE,
                support_tolerance=DEFAULT_SUPPORT_TOLERANCE,
                bending_tolerance=DEFAULT_BENDING_TOLERANCE,
                require_standard_normalization=True,
            )
            exporter_gates = (
                strict_audit.strict_pass,
                strict_audit.topology.get("input_orientation_coherent") is True,
                strict_audit.topology.get("orientation_was_normalized") is False,
                strict_audit.normalization.get("handedness_matches") is True,
                strict_audit.bending.get("passes") is True,
                strict_audit.bending.get("orientation_sign") == 1,
                strict_audit.spherical_embedding.get("embedded") is True,
                strict_audit.qhull_agreement.get("agrees") is True,
            )
            if not all(exporter_gates):
                raise HoldoutInputError(
                    f"independent strict positive-orientation audit failed for selected "
                    f"case {case['case_id']!r}: {strict_audit.classification}; "
                    f"bending_sign={strict_audit.bending.get('orientation_sign')!r}"
                )
            evidence = _audit_evidence(strict_audit)
            rounded_audit = audit_zero_shear_realization(
                faces,
                _round_points_significantly(points, 12),
                distinguished_face=(0, 1, 2),
                shear_tolerance=DEFAULT_SHEAR_TOLERANCE,
                support_tolerance=DEFAULT_SUPPORT_TOLERANCE,
                bending_tolerance=DEFAULT_BENDING_TOLERANCE,
                require_standard_normalization=True,
            )
            if not (
                rounded_audit.strict_pass
                and rounded_audit.topology.get("input_orientation_coherent") is True
                and rounded_audit.topology.get("orientation_was_normalized") is False
                and rounded_audit.bending.get("orientation_sign") == 1
                and rounded_audit.qhull_agreement.get("agrees") is True
            ):
                raise HoldoutInputError(
                    f"12-significant-digit target failed the strict audit for "
                    f"{case['case_id']!r}: {rounded_audit.classification}"
                )
            evidence["twelve_significant_digit_response_gate"] = {
                "passes": True,
                "max_abs_shear": rounded_audit.shear["max_abs"],
                "min_support_margin": rounded_audit.support["min_normalized_margin"],
                "min_bending_margin_radians": rounded_audit.bending[
                    "min_endpoint_margin_radians"
                ],
            }
            topology = validate_closed_sphere_triangulation(faces)
            psl2z_index = configuration["psl2z_index"]
            if psl2z_index != 3 * topology.n_faces:
                raise HoldoutInputError(
                    f"exact arithmetic index is inconsistent for {configuration_id!r}"
                )
            prompt_payload = _prompt_text(
                n_vertices=topology.n_vertices,
                n_faces=topology.n_faces,
                psl2z_index=psl2z_index,
                faces=[list(face) for face in faces],
            ).encode("utf-8")
            prompt_file = f"{instance_id}.prompt.txt"
            prompt_sha256 = _sha256_bytes(prompt_payload)
            _write_file(public_stage / prompt_file, prompt_payload, mode=0o644)
            public_instances.append(
                {
                    "instance_id": instance_id,
                    "prompt_file": prompt_file,
                    "prompt_sha256": prompt_sha256,
                    "n_vertices": topology.n_vertices,
                    "n_edges": topology.n_edges,
                    "n_faces": topology.n_faces,
                    "psl2z_index": psl2z_index,
                }
            )

            answer_key_file = f"{instance_id}.answer-key.json"
            answer_key = {
                "schema_version": ANSWER_KEY_SCHEMA_VERSION,
                "instance_id": instance_id,
                "status": "private-answer-key",
                "prompt_sha256": prompt_sha256,
                "ground_truth": {
                    "exists": True,
                    "oriented_faces": [list(face) for face in faces],
                    "vertices": _coordinate_rows(points),
                },
                "normalization": {
                    "distinguished_face": [0, 1, 2],
                    "handedness": "all non-anchor vertices have positive imaginary part",
                    "source_anchor_face": list(anchor),
                    "source_infinity_vertex": configuration["infinity_index"],
                    "complex_conjugation_applied": conjugated,
                    "relabel_seed": relabel_seed,
                    "old_to_new": {
                        str(key): value for key, value in sorted(old_to_new.items())
                    },
                    "new_to_old": list(new_to_old),
                },
                "source": {
                    "configuration_id": configuration_id,
                    "basin_case_id": case["case_id"],
                    "source_seed": configuration["raw"].get("seed"),
                    "source_topology_id": configuration["raw"]
                    .get("deduplication", {})
                    .get("topology_id"),
                    "basin_audits": case["_basin_inputs"],
                },
                "validation": evidence,
            }
            _validate_answer_key(
                answer_key,
                n_vertices=topology.n_vertices,
                prompt_sha256=prompt_sha256,
            )
            answer_key_payload = strict_json_bytes(answer_key)
            answer_key_sha256 = _sha256_bytes(answer_key_payload)
            if private_stage is not None:
                _write_file(
                    private_stage / answer_key_file, answer_key_payload, mode=0o600
                )
            private_instances.append(
                {
                    "instance_id": instance_id,
                    "answer_key_file": answer_key_file
                    if private_stage is not None
                    else None,
                    "answer_key_sha256": answer_key_sha256,
                    "prompt_file": prompt_file,
                    "prompt_sha256": prompt_sha256,
                    "configuration_id": configuration_id,
                    "basin_case_id": case["case_id"],
                    "source_anchor_face": list(anchor),
                    "basin_audits": case["_basin_inputs"],
                    "hardness": {
                        key: case.get(key)
                        for key in (
                            "hardness_class",
                            "hardness_rank",
                            "hardness_key",
                            "uniform_outcome",
                            "planned_task_count",
                            "outcome_counts",
                            "target_success_count",
                            "target_success_fraction",
                            "root_clusters",
                        )
                    },
                    "strict_audit": evidence,
                }
            )

        public_manifest = _public_manifest(public_instances)
        _validate_public_manifest(public_manifest)
        public_manifest_payload = strict_json_bytes(public_manifest)
        _write_file(public_stage / "manifest.json", public_manifest_payload, mode=0o644)
        _fsync_directory(public_stage)

        private_manifest = {
            "schema_version": PRIVATE_SCHEMA_VERSION,
            "protocol_version": PROMPT_PROTOCOL_VERSION,
            "benchmark_scope": public_manifest["benchmark_scope"],
            "generator": {
                "name": GENERATOR_NAME,
                "version": GENERATOR_VERSION,
                "source_sha256": _sha256_file(__file__),
            },
            "grading_contract": public_manifest["grading_contract"],
            "inputs": {
                "corpus_file": corpus_source.name,
                "corpus_sha256": corpus_sha256,
                "basin_audits": basin_inputs,
            },
            "run_config": {
                "relabel_seed": relabel_seed,
                "requested_case_ids": list(requested_case_ids or []),
                "top_count": None if requested_case_ids else top_count,
                "public_only": private_stage is None,
                "excluded_configuration_ids": list(excluded_configuration_ids),
            },
            "provenance": {"source_files": _source_files()},
            "public_manifest_sha256": _sha256_bytes(public_manifest_payload),
            "instance_count": len(private_instances),
            "instances": private_instances,
        }
        _validate_private_manifest(private_manifest)
        if private_stage is not None:
            _write_file(
                private_stage / "manifest.private.json",
                strict_json_bytes(private_manifest),
                mode=0o600,
            )
            _fsync_directory(private_stage)
        _commit_stages(public_stage, public_target, private_stage, private_target)
    except BaseException:
        if public_stage.exists():
            shutil.rmtree(public_stage)
        if private_stage is not None and private_stage.exists():
            shutil.rmtree(private_stage)
        raise
    return {
        "public_directory": str(public_target),
        "private_directory": None if private_target is None else str(private_target),
        "instance_count": len(public_instances),
        "public_manifest_sha256": _sha256_file(public_target / "manifest.json"),
        "private_keys_persisted": private_target is not None,
    }


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _seed(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result < 0 or result >= 1 << 63:
        raise argparse.ArgumentTypeError("must lie in [0, 2**63)")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="complete arithmetic corpus JSON")
    parser.add_argument(
        "basin_audit",
        type=Path,
        nargs="+",
        help="one or more finalized basin-audit JSON files",
    )
    parser.add_argument(
        "--public-dir", type=Path, required=True, help="new public package directory"
    )
    privacy = parser.add_mutually_exclusive_group(required=True)
    privacy.add_argument(
        "--private-dir",
        type=Path,
        help="new disjoint private-key directory outside the repository",
    )
    privacy.add_argument(
        "--public-only",
        action="store_true",
        help="explicitly discard grading keys after validating the public prompts",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="selected basin case ID; repeatable",
    )
    selection.add_argument(
        "--top",
        type=_positive_integer,
        default=DEFAULT_TOP_COUNT,
        help=f"select the top ranked hard cases (default: {DEFAULT_TOP_COUNT})",
    )
    parser.add_argument("--relabel-seed", type=_seed, default=DEFAULT_RELABEL_SEED)
    parser.add_argument(
        "--exclude-config-id",
        action="append",
        default=[],
        help="private source configuration to exclude; repeatable",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.public_only:
        print(
            "warning: --public-only selected; grading keys will not be persisted",
            file=sys.stderr,
        )
    try:
        result = build_holdout_package(
            args.corpus,
            args.basin_audit,
            public_directory=args.public_dir,
            private_directory=args.private_dir,
            requested_case_ids=args.case_ids,
            top_count=args.top,
            relabel_seed=args.relabel_seed,
            excluded_configuration_ids=args.exclude_config_id,
        )
    except (FileExistsError, HoldoutInputError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    sys.stdout.buffer.write(strict_json_bytes(result))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
