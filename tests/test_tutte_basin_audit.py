from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from jsonschema.exceptions import ValidationError


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


basin = _load_script(
    "tutte_basin_audit_script", "scripts/audit_tutte_zero_shear_basins.py"
)
arithmetic = _load_script(
    "tutte_basin_arithmetic_script", "scripts/generate_arithmeticity_benchmark.py"
)


@pytest.fixture()
def arithmetic_corpus(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "arithmetic.json"
    run_config = {
        "mode": "layered-ring",
        "vertex_semantics": "total-including-explicit-infinity",
        "n_vertices": 6,
        "target_count": 1,
        "max_attempts": 1,
        "max_wall_seconds": 30.0,
        "workers": 1,
        "base_seed": 42,
        "blas_threads_per_worker": arithmetic.BLAS_THREADS,
        "max_flips": 8,
        "shear_tolerance": 1e-9,
        "margin_tolerance": 1e-4,
        "constraint_tolerance": 1e-6,
        "qhull_audit": False,
    }
    document = arithmetic.run_campaign(
        path,
        run_config,
        invocation=["pytest", "make-small-arithmetic-corpus"],
        use_subprocesses=False,
    )
    return path, document["configurations"][0]["id"]


def _study_config(configuration_id: str, **overrides):
    config = {
        "selected_configuration_ids": [configuration_id],
        "anchors_per_configuration": 1,
        "explicit_distinguished_faces": [],
        "sigmas": [0.5],
        "starts_per_sigma": 1,
        "base_seed": 17,
        "max_nfev": 50,
        "shear_tolerance": 1e-8,
        "support_tolerance": 0.0,
        "bending_tolerance": 1e-10,
        "xtol": 1e-13,
        "ftol": 1e-13,
        "gtol": 1e-13,
        "record_trajectory": False,
        "include_endpoint_coordinates": False,
        "root_cluster_tolerance": 1e-8,
        "max_wall_seconds": 30.0,
        "workers": 1,
        "blas_threads_per_worker": basin.BLAS_THREADS,
    }
    config.update(overrides)
    return config


def _fake_result(task, _configuration, _run_config):
    uniform = task["start_spec"]["kind"] == "uniform"
    outcome = (
        "target_strict_zero_shear" if uniform else "embedded_wrong_hull_zero_shear"
    )
    root_coordinates = [
        None,
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5 if uniform else 0.8, 0.5],
    ]
    start = {
        **task["start_spec"],
        "weight_sha256": "sha256:" + "3" * 64,
        "minimum_weight": 1.0,
        "maximum_weight": 1.0,
        "initial_embedding_sha256": "sha256:" + "4" * 64,
    }
    return {
        "outcome": outcome,
        "start": start,
        "core_result": {
            "success": True,
            "classification": outcome,
            "audit": {"strict_pass": uniform},
        },
        "root_coordinates": root_coordinates,
        "endpoint_coordinates": None,
    }


def _task_projection(document):
    return [
        {
            "task_index": row["task_index"],
            "task_id": row["task_id"],
            "case_id": row["case_id"],
            "configuration_id": row["configuration_id"],
            "distinguished_face": row["distinguished_face"],
            "start_spec": row["start_spec"],
            "outcome": row["outcome"],
            "root_fingerprint": row["root_fingerprint"],
        }
        for row in document["tasks"]
    ]


def test_task_expansion_is_worker_count_independent(arithmetic_corpus):
    path, configuration_id = arithmetic_corpus
    corpus = basin.load_arithmetic_corpus(path)
    configurations = basin.select_configurations(corpus, [configuration_id])
    one_worker = basin.build_tasks(
        configurations,
        _study_config(
            configuration_id,
            workers=1,
            sigmas=[0.25, 0.5],
            starts_per_sigma=3,
        ),
    )
    eight_workers = basin.build_tasks(
        configurations,
        _study_config(
            configuration_id,
            workers=8,
            sigmas=[0.25, 0.5],
            starts_per_sigma=3,
        ),
    )
    assert one_worker == eight_workers
    assert len(one_worker) == 7
    assert one_worker[0]["start_spec"] == {
        "kind": "uniform",
        "sigma": None,
        "replicate": None,
        "seed": None,
    }
    weighted_seeds = [row["start_spec"]["seed"] for row in one_worker[1:]]
    assert len(set(weighted_seeds)) == len(weighted_seeds)
    assert all(0 <= seed < 1 << 63 for seed in weighted_seeds)


def test_explicit_ordered_anchor_is_preserved(arithmetic_corpus):
    path, configuration_id = arithmetic_corpus
    corpus = basin.load_arithmetic_corpus(path)
    configurations = basin.select_configurations(corpus, [configuration_id])
    ordered_anchor = [2, 1, 0]
    tasks = basin.build_tasks(
        configurations,
        _study_config(
            configuration_id,
            anchors_per_configuration=1,
            explicit_distinguished_faces=[ordered_anchor],
        ),
    )
    assert {tuple(task["distinguished_face"]) for task in tasks} == {
        tuple(ordered_anchor)
    }
    with pytest.raises(ValueError, match="not a configuration face"):
        basin.build_tasks(
            configurations,
            _study_config(
                configuration_id,
                anchors_per_configuration=1,
                explicit_distinguished_faces=[[0, 1, 5]],
            ),
        )


def test_checkpoint_resume_no_clobber_schema_and_hardness(
    arithmetic_corpus, tmp_path: Path
):
    input_path, configuration_id = arithmetic_corpus
    output = tmp_path / "audit.json"
    run_config = _study_config(configuration_id)
    document = basin.run_study(
        input_path,
        output,
        run_config,
        requested_ids=[configuration_id],
        invocation=["pytest"],
        use_subprocesses=False,
        task_runner=_fake_result,
    )
    assert document["status"] == "complete"
    assert document["summary"] == {
        "planned": 2,
        "finished": 2,
        "completed": 2,
        "errors": 0,
        "timed_out": 0,
        "pending": 0,
        "outcome_counts": {
            "embedded_wrong_hull_zero_shear": 1,
            "target_strict_zero_shear": 1,
        },
    }
    case = document["cases"][0]
    assert case["hardness_class"] == "multibasin"
    assert case["hardness_rank"] == 1
    assert len(case["root_clusters"]) == 2
    assert all(row["endpoint_coordinates"] is None for row in document["tasks"])
    basin.validate_audit_document(basin.load_json_strict(output))

    original = output.read_bytes()
    resumed = basin.run_study(
        input_path,
        output,
        run_config,
        requested_ids=[configuration_id],
        resume=True,
        invocation=["pytest", "--resume"],
        use_subprocesses=False,
        task_runner=_fake_result,
    )
    assert resumed["status"] == "complete"
    assert output.read_bytes() == original

    with pytest.raises(FileExistsError, match="refusing to clobber"):
        basin.run_study(
            input_path,
            output,
            run_config,
            requested_ids=[configuration_id],
            invocation=["pytest"],
            use_subprocesses=False,
            task_runner=_fake_result,
        )

    changed = dict(run_config, max_nfev=51)
    with pytest.raises(ValueError, match="run config"):
        basin.run_study(
            input_path,
            output,
            changed,
            requested_ids=[configuration_id],
            resume=True,
            invocation=["pytest", "--resume"],
            use_subprocesses=False,
            task_runner=_fake_result,
        )

    partial_path = tmp_path / "partial.json"
    partial = basin.load_json_strict(output)
    corpus = basin.load_arithmetic_corpus(input_path)
    configurations = basin.select_configurations(corpus, [configuration_id])
    planned_tasks = basin.build_tasks(configurations, run_config)
    partial["tasks"] = partial["tasks"][:1]
    partial["status"] = "in_progress"
    basin._refresh_document(partial, planned_tasks)
    partial_path.write_bytes(basin.strict_json_bytes(partial))
    resumed_partial = basin.run_study(
        input_path,
        partial_path,
        run_config,
        requested_ids=[configuration_id],
        resume=True,
        invocation=["pytest", "--resume-partial"],
        use_subprocesses=False,
        task_runner=_fake_result,
    )
    assert resumed_partial["status"] == "complete"
    assert resumed_partial["summary"]["finished"] == 2
    assert len({row["task_id"] for row in resumed_partial["tasks"]}) == 2

    out_of_order_path = tmp_path / "out-of-order-partial.json"
    out_of_order = basin.load_json_strict(output)
    out_of_order["tasks"] = out_of_order["tasks"][1:]
    basin._refresh_document(out_of_order, planned_tasks)
    basin.validate_audit_document(out_of_order)
    out_of_order_path.write_bytes(basin.strict_json_bytes(out_of_order))
    resumed_out_of_order = basin.run_study(
        input_path,
        out_of_order_path,
        run_config,
        requested_ids=[configuration_id],
        resume=True,
        invocation=["pytest", "--resume-out-of-order"],
        use_subprocesses=False,
        task_runner=_fake_result,
    )
    assert resumed_out_of_order["status"] == "complete"
    assert [row["task_index"] for row in resumed_out_of_order["tasks"]] == [0, 1]

    tampered_path = tmp_path / "tampered.json"
    tampered = basin.load_json_strict(output)
    tampered["tasks"][0]["distinguished_face"] = [97, 98, 99]
    tampered_path.write_bytes(basin.strict_json_bytes(tampered))
    with pytest.raises(ValueError, match="does not match the deterministic plan"):
        basin.run_study(
            input_path,
            tampered_path,
            run_config,
            requested_ids=[configuration_id],
            resume=True,
            invocation=["pytest", "--resume-tampered"],
            use_subprocesses=False,
            task_runner=_fake_result,
        )


def test_validator_and_resume_reject_fabricated_aggregates_and_terminal_status(
    arithmetic_corpus, tmp_path: Path
):
    input_path, configuration_id = arithmetic_corpus
    output = tmp_path / "audit.json"
    run_config = _study_config(configuration_id)
    document = basin.run_study(
        input_path,
        output,
        run_config,
        requested_ids=[configuration_id],
        invocation=["pytest"],
        use_subprocesses=False,
        task_runner=_fake_result,
    )

    fabricated_cases = json.loads(json.dumps(document))
    fabricated_cases["cases"][0]["hardness_class"] = "robust"
    with pytest.raises(ValueError, match="case aggregates"):
        basin.validate_audit_document(fabricated_cases)

    corpus = basin.load_arithmetic_corpus(input_path)
    configurations = basin.select_configurations(corpus, [configuration_id])
    planned_tasks = basin.build_tasks(configurations, run_config)
    fabricated_terminal = json.loads(json.dumps(document))
    fabricated_terminal["tasks"] = fabricated_terminal["tasks"][:1]
    basin._refresh_document(fabricated_terminal, planned_tasks)
    fabricated_terminal["status"] = "complete"
    with pytest.raises(ValueError, match="status does not match"):
        basin.validate_audit_document(fabricated_terminal)

    terminal_path = tmp_path / "fabricated-terminal.json"
    terminal_path.write_bytes(basin.strict_json_bytes(fabricated_terminal))
    with pytest.raises(ValueError, match="status does not match"):
        basin.run_study(
            input_path,
            terminal_path,
            run_config,
            requested_ids=[configuration_id],
            resume=True,
            invocation=["pytest", "--resume"],
            use_subprocesses=False,
            task_runner=_fake_result,
        )

    fabricated_generator = json.loads(json.dumps(document))
    fabricated_generator["generator"]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="generator source hash"):
        basin.validate_audit_document(fabricated_generator)


def test_resume_rejects_runtime_dependency_platform_and_thread_drift(
    arithmetic_corpus, tmp_path: Path
):
    input_path, configuration_id = arithmetic_corpus
    output = tmp_path / "audit.json"
    run_config = _study_config(configuration_id)
    document = basin.run_study(
        input_path,
        output,
        run_config,
        requested_ids=[configuration_id],
        invocation=["pytest"],
        use_subprocesses=False,
        task_runner=_fake_result,
    )
    assert (
        "ideal_poly_volume_toolkit/rivin_holonomy.py"
        in document["provenance"]["source_files"]
    )

    mutations = {
        "python": lambda value: value.__setitem__(
            "python", f"{value['python']}-different"
        ),
        "platform": lambda value: value.__setitem__(
            "platform", f"{value['platform']}-different"
        ),
        "dependency_versions": lambda value: value["dependency_versions"].__setitem__(
            "numpy", f"{value['dependency_versions']['numpy']}-different"
        ),
        "thread_environment": lambda value: value["thread_environment"].__setitem__(
            "OMP_NUM_THREADS",
            f"{value['thread_environment']['OMP_NUM_THREADS']}-different",
        ),
    }
    for field, mutate in mutations.items():
        tampered = json.loads(json.dumps(document))
        mutate(tampered["provenance"])
        basin.validate_audit_document(tampered)
        tampered_path = tmp_path / f"runtime-{field}.json"
        tampered_path.write_bytes(basin.strict_json_bytes(tampered))
        with pytest.raises(ValueError, match=field):
            basin.run_study(
                input_path,
                tampered_path,
                run_config,
                requested_ids=[configuration_id],
                resume=True,
                invocation=["pytest", "--resume"],
                use_subprocesses=False,
                task_runner=_fake_result,
            )


def test_root_clusters_use_direct_chordal_tolerance_without_integer_grids():
    def root(variable: float):
        return [None, [0.0, 0.0], [1.0, 0.0], [variable, 0.0]]

    tolerance = 1e-8
    representatives = []
    first = basin._assign_root_cluster(root(0.0), representatives, tolerance)
    nearby = basin._assign_root_cluster(root(1e-10), representatives, tolerance)
    distant = basin._assign_root_cluster(root(1e-7), representatives, tolerance)
    assert nearby == first
    assert distant != first
    assert len(representatives) == 2
    assert basin._root_chordal_distance(root(0.0), root(1e-10)) < tolerance
    assert basin._root_chordal_distance(root(0.0), root(1e-7)) > tolerance

    tiny_tolerance_representatives = []
    tiny_first = basin._assign_root_cluster(
        root(1.0), tiny_tolerance_representatives, 1e-320
    )
    tiny_second = basin._assign_root_cluster(
        root(2.0), tiny_tolerance_representatives, 1e-320
    )
    assert tiny_first != tiny_second

    huge_positive = root(1e308)
    huge_negative = root(-1e308)
    assert np.all(np.isfinite(basin._root_sphere_coordinates(huge_positive)))
    assert basin._root_fingerprint(huge_positive) != basin._root_fingerprint(
        huge_negative
    )
    huge_representatives = []
    assert basin._assign_root_cluster(
        huge_positive, huge_representatives, 1e-320
    ) != basin._assign_root_cluster(huge_negative, huge_representatives, 1e-320)


def test_ready_worker_results_are_committed_in_task_order():
    pending = [{"task_id": "task-0"}, {"task_id": "task-1"}]
    ready = {"task-1": {"ok": True}}
    committable, next_commit = basin._take_ready_task_prefix(pending, ready, 0)
    assert committable == []
    assert next_commit == 0

    ready["task-0"] = {"ok": True}
    committable, next_commit = basin._take_ready_task_prefix(
        pending, ready, next_commit
    )
    assert [task["task_id"] for task, _ in committable] == ["task-0", "task-1"]
    assert next_commit == 2
    assert ready == {}


def test_fake_study_outcomes_do_not_depend_on_worker_setting(
    arithmetic_corpus, tmp_path: Path
):
    input_path, configuration_id = arithmetic_corpus
    documents = []
    for workers in (1, 8):
        output = tmp_path / f"audit-{workers}.json"
        documents.append(
            basin.run_study(
                input_path,
                output,
                _study_config(configuration_id, workers=workers),
                requested_ids=[configuration_id],
                invocation=["pytest", f"--workers={workers}"],
                use_subprocesses=False,
                task_runner=_fake_result,
            )
        )
    assert _task_projection(documents[0]) == _task_projection(documents[1])
    comparable_cases = []
    for document in documents:
        comparable_cases.append(
            [
                {key: value for key, value in case.items() if key not in {"case_id"}}
                for case in document["cases"]
            ]
        )
    assert comparable_cases[0] == comparable_cases[1]


def test_errors_are_checkpointed_and_make_a_target_start_fragile(
    arithmetic_corpus, tmp_path: Path, monkeypatch
):
    input_path, configuration_id = arithmetic_corpus

    def fails_on_weighted(task, configuration, run_config):
        if task["start_spec"]["kind"] == "lognormal":
            raise RuntimeError("bounded synthetic failure")
        return _fake_result(task, configuration, run_config)

    output = tmp_path / "errors.json"
    document = basin.run_study(
        input_path,
        output,
        _study_config(configuration_id),
        requested_ids=[configuration_id],
        invocation=["pytest"],
        use_subprocesses=False,
        task_runner=fails_on_weighted,
    )
    assert document["status"] == "failed"
    assert document["summary"]["errors"] == 1
    assert document["cases"][0]["hardness_class"] == "fragile"
    failed = next(row for row in document["tasks"] if row["status"] == "error")
    assert failed["outcome"] == "error"
    assert failed["error"]["type"] == "RuntimeError"
    assert "synthetic failure" in failed["error"]["message"]
    basin.validate_audit_document(document)

    monkeypatch.setattr(basin, "run_study", lambda *args, **kwargs: document)
    assert (
        basin.main(
            [
                str(input_path),
                "--output",
                str(tmp_path / "unused-cli-output.json"),
                "--config-id",
                configuration_id,
            ]
        )
        == 1
    )


def test_schema_and_strict_json_reject_unversioned_or_nonfinite_data(
    arithmetic_corpus, tmp_path: Path
):
    input_path, configuration_id = arithmetic_corpus
    output = tmp_path / "audit.json"
    document = basin.run_study(
        input_path,
        output,
        _study_config(configuration_id),
        requested_ids=[configuration_id],
        invocation=["pytest"],
        use_subprocesses=False,
        task_runner=_fake_result,
    )
    malformed = json.loads(json.dumps(document))
    malformed["unknown_root_field"] = True
    with pytest.raises(ValidationError):
        basin.validate_audit_document(malformed)
    with pytest.raises(ValueError, match="non-finite"):
        basin.strict_json_bytes({"bad": math.nan})

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        basin.load_json_strict(duplicate)


def test_worker_and_numerical_bounds_are_explicit():
    with pytest.raises(ValueError, match="workers"):
        basin._validate_run_config(_study_config("x", workers=9))
    with pytest.raises(ValueError, match="sigmas"):
        basin._validate_run_config(_study_config("x", sigmas=[0.5, 0.5]))
    with pytest.raises(ValueError, match="root-cluster"):
        basin._validate_run_config(_study_config("x", root_cluster_tolerance=0.0))


def test_coordinate_payload_filter_keeps_trajectory_metrics_only():
    filtered = basin._strip_coordinate_payload(
        {
            "vertices_cp1": [[0.0, 0.0]],
            "normalized_vertices_cp1": [None],
            "initial_vertices_cp1": [[1.0, 0.0]],
            "trajectory": [
                {
                    "iteration": 0,
                    "max_abs_shear": 0.25,
                    "packed": [1.0, 2.0],
                }
            ],
        }
    )
    assert filtered == {"trajectory": [{"iteration": 0, "max_abs_shear": 0.25}]}


def test_compact_audit_replaces_linear_diagnostic_arrays_with_counts():
    compact = basin._compact_audit_payload(
        {
            "topology": {"n_vertices": 4, "vertex_labels": [0, 1, 2, 3]},
            "support": {"passes": True, "faces": [{}, {}, {}, {}]},
            "shear": {"passes": True, "values": [0.0] * 6},
            "bending": {"passes": True, "angles_radians": [1.0] * 6},
            "qhull_agreement": {"missing_faces": [[0, 1, 2]], "extra_faces": []},
        }
    )
    assert compact == {
        "topology": {"n_vertices": 4},
        "support": {"passes": True, "face_diagnostic_count": 4},
        "shear": {"passes": True, "value_count": 6},
        "bending": {"passes": True, "angle_count": 6},
        "qhull_agreement": {"missing_face_count": 1, "extra_face_count": 0},
    }


def test_trajectory_summary_reports_target_cell_entry_and_later_exit():
    summary = basin._trajectory_summary(
        [
            {
                "iteration": 0,
                "in_target_convex_cell": False,
                "strict_target_zero_shear": False,
            },
            {
                "iteration": 1,
                "in_target_convex_cell": True,
                "strict_target_zero_shear": False,
            },
            {
                "iteration": 2,
                "in_target_convex_cell": True,
                "strict_target_zero_shear": True,
            },
            {
                "iteration": 3,
                "in_target_convex_cell": False,
                "strict_target_zero_shear": False,
            },
        ],
        trajectory_kind="accepted_iterates",
    )
    assert summary == {
        "trajectory_kind": "accepted_iterates",
        "recorded_iterate_count": 4,
        "target_cell_iterate_count": 2,
        "strict_target_iterate_count": 1,
        "first_target_cell_iteration": 1,
        "first_exit_after_target_entry_iteration": 3,
        "exited_after_target_entry": True,
        "final_in_target_convex_cell": False,
    }


def test_real_script_task_skips_spherical_scan_and_compacts_audit(
    arithmetic_corpus,
):
    path, configuration_id = arithmetic_corpus
    corpus = basin.load_arithmetic_corpus(path)
    configuration = basin.select_configurations(corpus, [configuration_id])[0]
    run_config = _study_config(configuration_id, max_nfev=100, record_trajectory=True)
    task = basin.build_tasks([configuration], run_config)[0]
    result = basin._run_task(task, configuration, run_config)
    assert result["outcome"] == "target_strict_zero_shear"
    audit = result["core_result"]["audit"]
    assert audit["spherical_embedding"]["status"] == "not_checked"
    assert "faces" not in audit["support"]
    assert audit["support"]["face_diagnostic_count"] == 8
    assert "values" not in audit["shear"]
    assert audit["shear"]["value_count"] == 12
    assert "angles_radians" not in audit["bending"]
    assert result["endpoint_coordinates"] is None
    assert result["trajectory_summary"]["recorded_iterate_count"] >= 1
    assert result["trajectory_summary"]["first_target_cell_iteration"] is not None
    assert all(
        "packed" not in row and "in_target_convex_cell" in row
        for row in result["core_result"]["trajectory"]
    )


def test_small_real_uniform_tutte_solve_reaches_strict_target():
    from ideal_poly_volume_toolkit.rivin_zero_shear import (
        solve_zero_shear_from_tutte,
    )

    faces = [
        (0, 1, 2),
        (0, 3, 1),
        (1, 3, 2),
        (2, 3, 0),
    ]
    result = solve_zero_shear_from_tutte(
        faces,
        (0, 1, 2),
        max_nfev=100,
        record_trajectory=True,
    )
    assert result.success
    assert result.classification == "target_strict_zero_shear"
    assert result.audit.strict_pass
    assert len(result.trajectory) >= 1
    assert result.normalized_vertices_cp1[:3] == (None, 0j, 1 + 0j)
