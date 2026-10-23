"""Tests for deterministic, leakage-resistant zero-shear holdout export."""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts/build_zero_shear_holdouts.py"
SPEC = importlib.util.spec_from_file_location("build_zero_shear_holdouts", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
holdouts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(holdouts)
CHECKER_PATH = REPOSITORY_ROOT / "scripts/check_zero_shear_response.py"
CHECKER_SPEC = importlib.util.spec_from_file_location(
    "check_zero_shear_response_for_holdouts", CHECKER_PATH
)
assert CHECKER_SPEC is not None and CHECKER_SPEC.loader is not None
checker = importlib.util.module_from_spec(CHECKER_SPEC)
CHECKER_SPEC.loader.exec_module(checker)


FACES = [
    [0, 1, 2],
    [0, 3, 1],
    [1, 3, 2],
    [2, 3, 0],
]
VERTICES = [
    None,
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, math.sqrt(3.0) / 2.0],
]


def _strict_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, allow_nan=False, indent=2, sort_keys=True, separators=(",", ": ")
        )
        + "\n"
    ).encode()


def _write(path: Path, value: object) -> Path:
    path.write_bytes(_strict_bytes(value))
    return path


def _inputs(
    tmp_path: Path, *, two_cases: bool = True, lower_half: bool = False
) -> tuple[Path, Path]:
    configuration_ids = ["secret-configuration-alpha"]
    if two_cases:
        configuration_ids.append("secret-configuration-beta")
    vertices = json.loads(json.dumps(VERTICES))
    if lower_half:
        vertices[3][1] *= -1.0
    configurations = [
        {
            "id": configuration_id,
            "seed": 9000 + index,
            "deduplication": {"topology_id": f"private-topology-{index}"},
            "ground_truth": {
                "classification": "zero-shear-psl2z",
                "is_arithmetic": True,
            },
            "certificate": {
                "protocol_version": "strict-zero-shear-psl2z/1",
                "checks": {
                    "accepted": True,
                    "closed_sphere": True,
                    "coordinate_zero_shear": True,
                    "exact_determinants_one": True,
                    "exact_integral_traces": True,
                    "exact_total_vertex_count": True,
                },
                "exact_zero_shear": {
                    "certificate_type": "exact-zero-shear-psl2z-v1",
                    "all_determinants_one": True,
                    "all_traces_integral": True,
                    "is_arithmetic": True,
                    "scope": "abstract-zero-shear-structure",
                    "psl2z_index": 12,
                    "topology": {
                        "n_vertices": 4,
                        "n_edges": 6,
                        "n_faces": 4,
                        "euler_characteristic": 2,
                        "vertex_labels": [0, 1, 2, 3],
                    },
                },
                "topology": {
                    "n_vertices": 4,
                    "n_edges": 6,
                    "n_faces": 4,
                    "euler_characteristic": 2,
                },
            },
            "input": {
                "n_vertices": 4,
                "n_triangles": 4,
                "infinity_index": 0,
                "triangles": FACES,
                "vertices": vertices,
            },
        }
        for index, configuration_id in enumerate(configuration_ids)
    ]
    corpus = {
        "schema_version": holdouts.CORPUS_SCHEMA_VERSION,
        "status": "complete",
        "configurations": configurations,
    }
    corpus_path = _write(tmp_path / "corpus.json", corpus)
    corpus_sha = holdouts._sha256_file(corpus_path)

    cases = []
    tasks = []
    for index, configuration_id in enumerate(configuration_ids):
        case_id = f"case-private-{index}"
        outcome = "embedded_wrong_hull_zero_shear" if index == 0 else "solver_failure"
        is_zero_shear = index == 0
        tasks.append(
            {
                "task_index": index,
                "task_id": f"task-private-{index}",
                "case_id": case_id,
                "configuration_id": configuration_id,
                "distinguished_face": [0, 1, 2],
                "start_spec": {
                    "kind": "uniform",
                    "sigma": None,
                    "replicate": None,
                    "seed": None,
                },
                "status": "completed",
                "outcome": outcome,
                "root_fingerprint": "root-private-alpha" if is_zero_shear else None,
                "core_result": (
                    {
                        "classification": outcome,
                        "success": True,
                        "strict_pass": False,
                        "audit": {
                            "classification": outcome,
                            "strict_pass": False,
                        },
                    }
                    if is_zero_shear
                    else {
                        "classification": outcome,
                        "success": False,
                        "strict_pass": False,
                        "audit": {
                            "classification": "residual_failure",
                            "strict_pass": False,
                        },
                    }
                ),
            }
        )
        cases.append(
            {
                "case_id": case_id,
                "configuration_id": configuration_id,
                "distinguished_face": [0, 1, 2],
                "planned_task_count": 1,
                "finished_task_count": 1,
                "outcome_counts": {outcome: 1},
                "uniform_outcome": outcome,
                "target_success_count": 0,
                "target_success_fraction": 0.0,
                "root_clusters": [],
                "hardness_class": (
                    "canonical_wrong_root" if index == 0 else "canonical_nonconvergence"
                ),
                "hardness_key": (
                    [4.0, 1.0, 1.0, 1.0] if index == 0 else [3.0, 0.0, 1.0, 0.0]
                ),
                "hardness_rank": index + 1,
            }
        )
    run_config = {"test_protocol_marker": "deterministic"}
    basin = {
        "schema_version": holdouts.BASIN_SCHEMA_VERSION,
        "protocol_version": holdouts.BASIN_PROTOCOL_VERSION,
        "status": "complete",
        "run_config": run_config,
        "run_config_sha256": holdouts._canonical_json_sha256(run_config),
        "input": {
            "sha256": corpus_sha,
            "selected_configuration_ids": configuration_ids,
        },
        "summary": {
            "planned": len(tasks),
            "finished": len(tasks),
            "completed": len(tasks),
            "errors": 0,
            "timed_out": 0,
            "pending": 0,
            "outcome_counts": dict(
                sorted(
                    {
                        task["outcome"]: sum(
                            row["outcome"] == task["outcome"] for row in tasks
                        )
                        for task in tasks
                    }.items()
                )
            ),
        },
        "tasks": tasks,
        "cases": cases,
    }
    basin_path = _write(tmp_path / "basin.json", basin)
    return corpus_path, basin_path


def _tree_bytes(path: Path) -> dict[str, bytes]:
    return {
        str(file.relative_to(path)): file.read_bytes()
        for file in sorted(path.iterdir())
        if file.is_file()
    }


def test_export_is_deterministic_private_and_publicly_leakage_resistant(tmp_path: Path):
    corpus, basin = _inputs(tmp_path)
    first_public = tmp_path / "public-one"
    first_private = tmp_path / "private-one"
    second_public = tmp_path / "public-two"
    second_private = tmp_path / "private-two"

    first = holdouts.build_holdout_package(
        corpus,
        basin,
        public_directory=first_public,
        private_directory=first_private,
        top_count=2,
        relabel_seed=314159,
    )
    second = holdouts.build_holdout_package(
        corpus,
        basin,
        public_directory=second_public,
        private_directory=second_private,
        top_count=2,
        relabel_seed=314159,
    )

    assert first["instance_count"] == second["instance_count"] == 2
    assert _tree_bytes(first_public) == _tree_bytes(second_public)
    assert _tree_bytes(first_private) == _tree_bytes(second_private)
    public_bytes = b"\n".join(_tree_bytes(first_public).values())
    for forbidden in (
        b"secret-configuration",
        b"case-private",
        b"314159",
        b"hardness",
        b"answer-key",
        b"9000",
        b"private-topology",
        holdouts._sha256_file(corpus).encode(),
        holdouts._sha256_file(basin).encode(),
    ):
        assert forbidden not in public_bytes
    manifest = json.loads((first_public / "manifest.json").read_text())
    assert [row["instance_id"] for row in manifest["instances"]] == [
        "ipr-holdout-001",
        "ipr-holdout-002",
    ]
    assert manifest["benchmark_scope"] == {
        "suite_type": "positive-only-anchor-mined-tutte-hard-stress-set",
        "primary_metric": "strict-coordinate-realization-rate",
        "existence_accuracy_is_meaningful": False,
        "certificate_scope": "double-precision-numerical-not-formal",
    }
    assert (
        "The oriented face list T is:"
        in (first_public / "ipr-holdout-001.prompt.txt").read_text()
    )
    assert manifest["grading_contract"]["checker"] == holdouts.CHECKER_NAME
    assert all(not path.is_symlink() for path in first_public.rglob("*"))
    assert {path.name for path in first_public.iterdir()} == {
        "manifest.json",
        "ipr-holdout-001.prompt.txt",
        "ipr-holdout-002.prompt.txt",
    }
    assert oct(os.stat(first_private).st_mode & 0o777) == "0o700"
    assert (
        oct(os.stat(first_private / "ipr-holdout-001.answer-key.json").st_mode & 0o777)
        == "0o600"
    )


def test_answer_key_preserves_winding_and_passes_independent_audit(tmp_path: Path):
    corpus, basin = _inputs(tmp_path, two_cases=False)
    public = tmp_path / "public"
    private = tmp_path / "private"
    holdouts.build_holdout_package(
        corpus,
        basin,
        public_directory=public,
        private_directory=private,
        top_count=1,
        relabel_seed=7,
    )
    key = json.loads((private / "ipr-holdout-001.answer-key.json").read_text())
    old_to_new = {
        int(old): new for old, new in key["normalization"]["old_to_new"].items()
    }
    expected_cycles = {
        cycle
        for face in FACES
        for cycle in holdouts._cyclic_rotations(
            tuple(old_to_new[label] for label in face)
        )
    }
    output_faces = [tuple(face) for face in key["ground_truth"]["oriented_faces"]]
    assert output_faces[0] == (0, 1, 2)
    assert all(face in expected_cycles for face in output_faces)
    points = [
        None if row is None else complex(row[0], row[1])
        for row in key["ground_truth"]["vertices"]
    ]
    audit = holdouts.audit_zero_shear_realization(
        output_faces,
        points,
        distinguished_face=(0, 1, 2),
        require_standard_normalization=True,
    )
    assert audit.strict_pass
    assert min(point.imag for point in points[3:] if point is not None) > 0.0
    assert key["validation"]["strict_pass"] is True
    rounded_vertices = [
        None
        if row is None
        else [float(format(row[0], ".12g")), float(format(row[1], ".12g"))]
        for row in key["ground_truth"]["vertices"]
    ]
    response = _write(
        tmp_path / "rounded-response.json",
        {"exists": True, "vertices": rounded_vertices},
    )
    grade = checker.grade_response(
        private / "ipr-holdout-001.answer-key.json", response
    )
    assert grade["strict_pass"] is True


def test_lower_half_plane_source_is_conjugated_with_positive_bending(tmp_path: Path):
    corpus, basin = _inputs(tmp_path, two_cases=False, lower_half=True)
    public = tmp_path / "public"
    private = tmp_path / "private"
    holdouts.build_holdout_package(
        corpus,
        basin,
        public_directory=public,
        private_directory=private,
        top_count=1,
    )
    key = json.loads((private / "ipr-holdout-001.answer-key.json").read_text())
    assert key["normalization"]["complex_conjugation_applied"] is True
    assert key["validation"]["normalization"]["handedness_matches"] is True
    assert key["validation"]["bending"]["orientation_sign"] == 1
    assert key["validation"]["twelve_significant_digit_response_gate"]["passes"]


def test_refuses_nonfinal_tampered_or_overlapping_outputs(tmp_path: Path):
    corpus, basin = _inputs(tmp_path, two_cases=False)
    document = json.loads(basin.read_text())
    document["status"] = "in_progress"
    bad_basin = _write(tmp_path / "bad-basin.json", document)
    with pytest.raises(holdouts.HoldoutInputError, match="finalized"):
        holdouts.build_holdout_package(
            corpus,
            bad_basin,
            public_directory=tmp_path / "unused-public",
            top_count=1,
        )

    with pytest.raises(holdouts.HoldoutInputError, match="disjoint"):
        holdouts.build_holdout_package(
            corpus,
            basin,
            public_directory=tmp_path / "package",
            private_directory=tmp_path / "package/private",
            top_count=1,
        )

    with pytest.raises(holdouts.HoldoutInputError, match="outside the repository"):
        holdouts.build_holdout_package(
            corpus,
            basin,
            public_directory=tmp_path / "other-public",
            private_directory=REPOSITORY_ROOT / ".should-never-be-created-private",
            top_count=1,
        )

    public = tmp_path / "public"
    holdouts.build_holdout_package(corpus, basin, public_directory=public, top_count=1)
    with pytest.raises(FileExistsError, match="clobber"):
        holdouts.build_holdout_package(
            corpus, basin, public_directory=public, top_count=1
        )


def test_recomputes_hardness_and_rejects_duplicate_selected_topologies(tmp_path: Path):
    corpus_path, basin_path = _inputs(tmp_path)
    configurations = holdouts._validate_corpus(
        holdouts.load_json_strict(corpus_path, label="corpus")
    )
    cases = holdouts._validate_basin(
        holdouts.load_json_strict(basin_path, label="basin"),
        corpus_sha256=holdouts._sha256_file(corpus_path),
    )
    first = next(iter(cases.values()))
    duplicate = {**first, "case_id": "another-private-case"}
    with pytest.raises(holdouts.HoldoutInputError, match="distinct configurations"):
        holdouts.select_cases(
            {first["case_id"]: first, duplicate["case_id"]: duplicate},
            configurations,
            requested_case_ids=[first["case_id"], duplicate["case_id"]],
            top_count=2,
            relabel_seed=1,
        )
    with pytest.raises(holdouts.HoldoutInputError, match="unknown excluded"):
        holdouts.select_cases(
            cases,
            configurations,
            requested_case_ids=None,
            top_count=1,
            relabel_seed=1,
            excluded_configuration_ids=["typoed-private-id"],
        )

    tampered = json.loads(basin_path.read_text())
    tampered["cases"][0]["hardness_class"] = "robust"
    with pytest.raises(holdouts.HoldoutInputError, match="hardness evidence"):
        holdouts._validate_basin(
            tampered, corpus_sha256=holdouts._sha256_file(corpus_path)
        )

    fabricated_outcome = json.loads(basin_path.read_text())
    fabricated_outcome["tasks"][0]["outcome"] = "target_strict_zero_shear"
    with pytest.raises(holdouts.HoldoutInputError, match="core result"):
        holdouts._validate_basin(
            fabricated_outcome, corpus_sha256=holdouts._sha256_file(corpus_path)
        )


def test_rejects_false_index_and_incoherent_prompt_winding(tmp_path: Path):
    corpus_path, basin_path = _inputs(tmp_path, two_cases=False)
    false_index = json.loads(corpus_path.read_text())
    false_index["configurations"][0]["certificate"]["exact_zero_shear"][
        "psl2z_index"
    ] = 13
    with pytest.raises(holdouts.HoldoutInputError, match=r"PSL\(2,Z\) index"):
        holdouts._validate_corpus(false_index)

    incoherent = json.loads(corpus_path.read_text())
    incoherent["configurations"][0]["input"]["triangles"][1] = [0, 1, 3]
    incoherent_path = _write(tmp_path / "incoherent-corpus.json", incoherent)
    basin = json.loads(basin_path.read_text())
    basin["input"]["sha256"] = holdouts._sha256_file(incoherent_path)
    incoherent_basin = _write(tmp_path / "incoherent-basin.json", basin)
    with pytest.raises(holdouts.HoldoutInputError, match="positive-orientation audit"):
        holdouts.build_holdout_package(
            incoherent_path,
            incoherent_basin,
            public_directory=tmp_path / "incoherent-public",
            top_count=1,
        )


def test_multiple_basin_audits_merge_into_one_private_provenance(tmp_path: Path):
    corpus, combined_path = _inputs(tmp_path)
    combined = json.loads(combined_path.read_text())
    basin_paths = []
    for index in range(2):
        task = {**combined["tasks"][index], "task_index": 0}
        case = {**combined["cases"][index], "hardness_rank": 1}
        one = {
            **combined,
            "input": {
                **combined["input"],
                "selected_configuration_ids": [case["configuration_id"]],
            },
            "tasks": [task],
            "cases": [case],
            "summary": {
                "planned": 1,
                "finished": 1,
                "completed": 1,
                "errors": 0,
                "timed_out": 0,
                "pending": 0,
                "outcome_counts": {task["outcome"]: 1},
            },
        }
        basin_paths.append(_write(tmp_path / f"basin-{index}.json", one))
    public = tmp_path / "public"
    private = tmp_path / "private"
    holdouts.build_holdout_package(
        corpus,
        basin_paths,
        public_directory=public,
        private_directory=private,
        top_count=2,
    )
    manifest = json.loads((private / "manifest.private.json").read_text())
    assert len(manifest["inputs"]["basin_audits"]) == 2
    assert {row["configuration_id"] for row in manifest["instances"]} == {
        "secret-configuration-alpha",
        "secret-configuration-beta",
    }


def test_public_only_cli_warns_and_persists_no_keys(tmp_path: Path, capsys):
    corpus, basin = _inputs(tmp_path, two_cases=False)
    public = tmp_path / "public"
    return_code = holdouts.main(
        [
            str(corpus),
            str(basin),
            "--public-dir",
            str(public),
            "--top",
            "1",
            "--public-only",
        ]
    )
    captured = capsys.readouterr()
    assert return_code == 0
    assert "grading keys will not be persisted" in captured.err
    assert not list(public.glob("*.answer-key.json"))
    assert json.loads(captured.out)["private_keys_persisted"] is False
