from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from jsonschema.exceptions import ValidationError


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "mine_h4_symmetric_maxima.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "h4_symmetric_maxima_campaign_script", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


campaign = _load_script()


def _run_config(**overrides):
    result = {
        "n_vertices": 8,
        "starts": 3,
        "base_seed": 20,
        "budget": 7,
        "optimizer": "powell",
        "max_wall_seconds": 30.0,
        "local_max_on_shortlist": True,
        "retain_all_coordinates": False,
        "calibrate_16_cell": False,
        "workers": 1,
        "blas_threads": 1,
    }
    result.update(overrides)
    return result


def _vertices(n_vertices: int, marker: float) -> list[list[float]]:
    rows: list[list[float]] = []
    for index in range(n_vertices):
        row = [0.0, 0.0, 0.0, 0.0]
        row[index % 4] = marker if index == 0 else 1.0
        rows.append(row)
    return rows


def _is_sixteen_cell(vertices) -> bool:
    if len(vertices) != 8:
        return False
    expected = {
        tuple(sign if coordinate == axis else 0.0 for coordinate in range(4))
        for sign in (-1.0, 1.0)
        for axis in range(4)
    }
    return {tuple(float(value) for value in row) for row in vertices} == expected


class FakeCore:
    def __init__(self):
        self.optimize_calls: list[tuple[int, int, int, str]] = []
        self.audit_calls: list[dict[str, object]] = []

    def optimize_h4_unconstrained(
        self,
        n_vertices: int,
        *,
        seed: int,
        budget: int,
        optimizer: str = "powell",
    ):
        self.optimize_calls.append((n_vertices, seed, budget, optimizer))
        marker = 1.0 if seed % 2 == 0 else -1.0
        if optimizer == "fresh_topology_ad":
            work = {
                "objective_evals": 2,
                "surrogate_evals": 0,
                "gradient_evals": 1,
                "total_evals": 3,
            }
        else:
            work = {
                "objective_evals": budget,
                "surrogate_evals": 0,
                "gradient_evals": 0,
                "total_evals": budget,
            }
        optimizer_details = {
            "selector": optimizer,
            "name": "fake",
            "method": "deterministic",
            "status": "completed",
        }
        if optimizer == "fresh_topology_ad":
            optimizer_details["diagnostics"] = {
                "selector": "fresh_topology_ad",
                "requested_branch_count": 1,
                "executed_branch_count": 1,
                "completed_branch_count": 1,
                "failed_branch_count": 0,
                "canonical_scored_branch_count": 1,
                "pole_indices": [0],
                "branch_modes": ["lbfgs"],
                "budget": budget,
                "budget_allocation": {
                    "initial_objective_evals": 1,
                    "reserved_endpoint_objective_evals": 1,
                    "gradient_evals_by_branch": [1],
                },
                "branches": [
                    {
                        "gradient_budget": 1,
                        "gradient_evals": 1,
                        "topology_evaluations": 1,
                        "status": "completed",
                    }
                ],
                "work": dict(work),
            }
        return {
            "dimension": 4,
            "n_vertices": n_vertices,
            "seed": seed,
            "budget": budget,
            "initial_coordinate_sha256": f"{seed:064x}",
            "vertices": _vertices(n_vertices, marker),
            "scores": {"initial": 0.25, "final": 0.75, "gain": 0.5},
            "work": work,
            "timing": {"wall_seconds": 0.01, "process_cpu_seconds": 0.01},
            "optimizer": optimizer_details,
        }

    def audit_h4_symmetric_candidate(
        self,
        vertices,
        *,
        pole_indices=None,
        local_maximum: bool = False,
    ):
        calibration = _is_sixteen_cell(vertices)
        good = calibration or float(vertices[0][0]) > 0.0
        group_order = 384 if calibration else (8 if good else 1)
        self.audit_calls.append(
            {
                "n_vertices": len(vertices),
                "pole_indices": pole_indices,
                "local_maximum": local_maximum,
                "group_order": group_order,
            }
        )
        if good and local_maximum:
            classification = "symmetric_strict_local_maximum_diagnostic"
            local_classification = "strict_local_maximum"
        elif good:
            classification = "symmetric_candidate"
            local_classification = "not_requested"
        else:
            classification = "trivial_geometric_symmetry"
            local_classification = "not_requested"
        symmetry = {
            "valid_geometry": True,
            "full_group_certified": True,
            "certified_nontrivial_symmetry": good,
            "combinatorial_group_order": group_order,
            "geometric_group_order": group_order,
            "combinatorial_generators": [[0, 1]],
            "geometric_generators": [[1, 0]],
            "vertex_orbits": [list(range(len(vertices)))],
            "residuals": {"maximum_isometry_residual": 1.0e-12},
        }
        pole_audit = {
            "all_poles_requested": pole_indices is None,
            "requested_indices": list(range(len(vertices))),
            "requested_count": len(vertices),
            "successful_count": len(vertices),
            "failure_count": 0,
            "values": [
                {"pole_index": index, "volume": 1.0, "simplex_count": 4}
                for index in range(len(vertices))
            ],
            "failures": [],
            "minimum_volume": 1.0,
            "maximum_volume": 1.0,
            "spread": 0.0,
            "spread_tolerance": 1.0e-8,
            "within_tolerance": True,
        }
        local_audit = {
            "requested": local_maximum,
            "performed": local_maximum,
            "diagnostic_only": True,
            "status": "completed" if local_maximum else "not_requested",
            "classification": local_classification,
            "physical_dimension": 3 * len(vertices) - 10,
            "projected_hessian_eigenvalues": [-1.0, -0.5],
        }
        basis = {
            "valid_geometry": True,
            "pole_volume_stable": True,
            "pole_volume_stable_on_requested": True,
            "all_poles_requested": True,
            "all_poles_verified": True,
            "full_group_certified": True,
            "geometric_group_order": group_order,
            "nontrivial_geometric_symmetry": good,
            "local_maximum_classification": local_classification,
        }
        return {
            "dimension": 4,
            "n_vertices": len(vertices),
            "classification": classification,
            "independent_classification": classification,
            "classification_basis": basis,
            "symmetry": symmetry,
            "pole_volume_audit": pole_audit,
            "local_maximum_audit": local_audit,
        }

    def namespace(self) -> SimpleNamespace:
        return SimpleNamespace(
            optimize=self.optimize_h4_unconstrained,
            audit=self.audit_h4_symmetric_candidate,
        )


def test_task_plan_is_deterministic_and_config_is_bounded():
    config = _run_config(starts=4, base_seed=91, n_vertices=12)
    first = campaign.build_tasks(config)
    second = campaign.build_tasks(dict(reversed(list(config.items()))))
    assert first == second
    assert [task["seed"] for task in first] == [91, 92, 93, 94]
    assert {task["optimizer"] for task in first} == {"powell"}
    assert len({task["task_id"] for task in first}) == 4

    fresh = campaign.build_tasks(
        _run_config(
            starts=4,
            base_seed=91,
            n_vertices=12,
            optimizer="fresh_topology_ad",
        )
    )
    assert {task["optimizer"] for task in fresh} == {"fresh_topology_ad"}
    assert [task["task_id"] for task in fresh] == [
        task["task_id"]
        for task in campaign.build_tasks(
            _run_config(
                starts=4,
                base_seed=91,
                n_vertices=12,
                optimizer="fresh_topology_ad",
            )
        )
    ]
    assert [task["task_id"] for task in fresh] != [
        task["task_id"] for task in first
    ]

    campaign._validate_run_config(config)
    with pytest.raises(ValueError, match="n_vertices"):
        campaign._validate_run_config(_run_config(n_vertices=7))
    with pytest.raises(ValueError, match="n_vertices"):
        campaign._validate_run_config(_run_config(n_vertices=41))
    with pytest.raises(ValueError, match="starts=0"):
        campaign._validate_run_config(_run_config(starts=0))
    with pytest.raises(ValueError, match="workers=1"):
        campaign._validate_run_config(_run_config(workers=2))
    with pytest.raises(ValueError, match="optimizer"):
        campaign._validate_run_config(_run_config(optimizer="unknown"))
    with pytest.raises(ValueError, match="optimizer"):
        campaign._validate_run_config(
            {key: value for key, value in config.items() if key != "optimizer"}
        )

    default_args = campaign._build_parser().parse_args(["--output", "unused.json"])
    assert campaign._run_config_from_args(default_args)["optimizer"] == "powell"
    fresh_args = campaign._build_parser().parse_args(
        ["--optimizer", "fresh_topology_ad", "--output", "unused.json"]
    )
    assert (
        campaign._run_config_from_args(fresh_args)["optimizer"]
        == "fresh_topology_ad"
    )


def test_optimizer_dispatch_and_legacy_powell_compatibility(tmp_path: Path):
    output = tmp_path / "fresh-topology-ad.json"
    fake = FakeCore()
    config = _run_config(
        starts=1,
        base_seed=20,
        optimizer="fresh_topology_ad",
        local_max_on_shortlist=False,
    )
    document = campaign.run_campaign(
        output,
        config,
        api=fake.namespace(),
        invocation=["pytest", "fresh-topology-ad"],
    )

    assert fake.optimize_calls == [(8, 20, 7, "fresh_topology_ad")]
    row = document["tasks"][0]
    assert document["run_config"]["optimizer"] == "fresh_topology_ad"
    assert (
        "ideal_poly_volume_toolkit/geometry_4d_torch.py"
        in document["provenance"]["source_files"]
    )
    assert row["optimizer"] == "fresh_topology_ad"
    assert row["optimization"]["optimizer"]["selector"] == "fresh_topology_ad"
    campaign.validate_campaign_document(document)

    class LegacyPowellCore(FakeCore):
        def optimize_h4_unconstrained(
            self, n_vertices: int, *, seed: int, budget: int
        ):
            result = super().optimize_h4_unconstrained(
                n_vertices,
                seed=seed,
                budget=budget,
                optimizer="powell",
            )
            result["optimizer"].pop("selector")
            return result

    legacy = LegacyPowellCore()
    legacy_task = campaign.build_tasks(_run_config(starts=1))[0]
    legacy_row = campaign._run_task(
        legacy_task,
        _run_config(starts=1, local_max_on_shortlist=False),
        legacy.namespace(),
    )
    assert legacy_row["status"] == "completed"
    assert legacy_row["optimization"]["optimizer"]["selector"] == "powell"

    fresh_task = campaign.build_tasks(
        _run_config(starts=1, optimizer="fresh_topology_ad")
    )[0]
    with pytest.raises(TypeError, match="ambiguous fallback"):
        campaign._call_optimize(legacy.optimize_h4_unconstrained, fresh_task)

    class SelectorOmittingCore(FakeCore):
        def optimize_h4_unconstrained(
            self,
            n_vertices: int,
            *,
            seed: int,
            budget: int,
            optimizer: str = "powell",
        ):
            result = super().optimize_h4_unconstrained(
                n_vertices,
                seed=seed,
                budget=budget,
                optimizer=optimizer,
            )
            result["optimizer"].pop("selector")
            return result

    omitted_row = campaign._run_task(
        fresh_task,
        _run_config(
            starts=1,
            optimizer="fresh_topology_ad",
            local_max_on_shortlist=False,
        ),
        SelectorOmittingCore().namespace(),
    )
    assert omitted_row["status"] == "failed"
    assert "optimizer.selector" in omitted_row["message"]


def test_worker_entry_preserves_deterministic_optimizer_arm(monkeypatch):
    class CaptureConnection:
        def __init__(self):
            self.envelope = None

        def send(self, value):
            self.envelope = value

        def close(self):
            return None

    fake = FakeCore()
    monkeypatch.setattr(campaign, "_load_core_api", fake.namespace)
    config = _run_config(
        starts=1,
        base_seed=20,
        optimizer="fresh_topology_ad",
        local_max_on_shortlist=False,
    )
    task = campaign.build_tasks(config)[0]
    connections = [CaptureConnection(), CaptureConnection()]
    for connection in connections:
        campaign._process_entry("task", task, config, connection)

    rows = [connection.envelope["result"] for connection in connections]
    for row in rows:
        row["elapsed_wall_seconds"] = 0.0
        row["process_cpu_seconds"] = 0.0
    assert rows[0] == rows[1]
    assert rows[0]["task_id"] == task["task_id"]
    assert rows[0]["optimizer"] == "fresh_topology_ad"
    assert fake.optimize_calls == [
        (8, 20, 7, "fresh_topology_ad"),
        (8, 20, 7, "fresh_topology_ad"),
    ]


def test_compact_fresh_ad_topology_diagnostics_have_bounded_size():
    task = campaign.build_tasks(
        _run_config(starts=1, optimizer="fresh_topology_ad")
    )[0]

    def compact_with_bulk_size(size: int):
        result = FakeCore().optimize_h4_unconstrained(
            task["n_vertices"],
            seed=task["seed"],
            budget=task["budget"],
            optimizer=task["optimizer"],
        )
        result["optimizer"]["diagnostics"] = {
            "branches": [
                {
                    "finite_vertex_labels": list(range(size)),
                    "topology_signature_sha256_examples": ["a" * 64] * size,
                    "topology_evaluations": 17,
                    "topology_change_count": 3,
                    "unique_topology_count": 4,
                    "topology_first_sha256": "b" * 64,
                    "topology_last_sha256": "c" * 64,
                    "topology_trace_sha256": "d" * 64,
                }
            ]
        }
        return campaign._compact_optimization(result, task)

    small = compact_with_bulk_size(1)
    large = compact_with_bulk_size(20_000)
    assert small == large
    assert len(campaign.strict_json_bytes(small)) == len(
        campaign.strict_json_bytes(large)
    )

    encoded = campaign.strict_json_bytes(large).decode("utf-8")
    assert "finite_vertex_labels" not in encoded
    assert "topology_signature_sha256_examples" not in encoded
    branch = large["optimizer"]["diagnostics"]["branches"][0]
    assert branch == {
        "topology_evaluations": 17,
        "topology_change_count": 3,
        "unique_topology_count": 4,
        "topology_first_sha256": "b" * 64,
        "topology_last_sha256": "c" * 64,
        "topology_trace_sha256": "d" * 64,
    }


def test_campaign_shortlists_locally_and_retains_only_certified_coordinates(
    tmp_path: Path,
):
    output = tmp_path / "campaign.json"
    fake = FakeCore()
    document = campaign.run_campaign(
        output,
        _run_config(),
        api=fake.namespace(),
        invocation=["pytest", "fake-h4-campaign"],
    )

    assert document["status"] == "complete"
    assert document["summary"] == {
        "planned": 3,
        "finished": 3,
        "completed": 3,
        "failed": 0,
        "pending": 0,
        "certified_nontrivial_symmetry": 2,
        "local_maximum_audits_performed": 2,
        "coordinates_retained": 2,
    }
    assert [call[1] for call in fake.optimize_calls] == [20, 21, 22]
    assert all(call["pole_indices"] is None for call in fake.audit_calls)
    assert [call["local_maximum"] for call in fake.audit_calls] == [
        False,
        True,
        False,
        False,
        True,
    ]

    by_seed = {row["seed"]: row for row in document["tasks"]}
    assert by_seed[20]["coordinates_retained"] is True
    assert by_seed[21]["coordinates_retained"] is False
    assert by_seed[21]["vertices"] is None
    assert by_seed[22]["coordinates_retained"] is True
    for row in document["tasks"]:
        assert "vertices" not in row["optimization"]
        assert "combinatorial_generators" not in row["audit"]["symmetry"]
        assert "geometric_generators" not in row["audit"]["symmetry"]
        assert "values" not in row["audit"]["pole_volume_audit"]
        assert (
            "projected_hessian_eigenvalues" not in row["audit"]["local_maximum_audit"]
        )
    campaign.validate_campaign_document(document)
    assert json.loads(output.read_text(encoding="utf-8")) == document
    assert "NaN" not in output.read_text(encoding="utf-8")


def test_local_audit_error_retains_candidate_with_inconclusive_classification(
    tmp_path: Path,
):
    class LocalFailureCore(FakeCore):
        def audit_h4_symmetric_candidate(
            self, vertices, *, pole_indices=None, local_maximum=False
        ):
            if local_maximum:
                raise RuntimeError("local finite-difference failure")
            return super().audit_h4_symmetric_candidate(
                vertices,
                pole_indices=pole_indices,
                local_maximum=local_maximum,
            )

    output = tmp_path / "local-failure.json"
    document = campaign.run_campaign(
        output,
        _run_config(starts=1, base_seed=20),
        api=LocalFailureCore().namespace(),
        invocation=["pytest", "local-failure"],
    )

    row = document["tasks"][0]
    assert row["status"] == "completed"
    assert row["audit"]["classification"] == (
        "symmetric_candidate_local_maximum_inconclusive"
    )
    assert row["audit"]["local_maximum_audit"]["status"] == "error"
    campaign.validate_campaign_document(document)


def test_no_clobber_checkpoint_resume_and_validate_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    output = tmp_path / "resume.json"
    config = _run_config(local_max_on_shortlist=False)
    initial_core = FakeCore()
    complete = campaign.run_campaign(
        output,
        config,
        api=initial_core.namespace(),
        invocation=["pytest", "initial"],
    )
    with pytest.raises(FileExistsError, match="refusing to clobber"):
        campaign.run_campaign(
            output,
            config,
            api=FakeCore().namespace(),
            invocation=["pytest", "no-clobber"],
        )

    partial = deepcopy(complete)
    missing = partial["tasks"].pop()
    campaign._refresh_document(partial, campaign.build_tasks(config))
    campaign.validate_campaign_document(partial)
    campaign.atomic_write_json(output, partial)

    resumed_core = FakeCore()
    resumed = campaign.run_campaign(
        output,
        config,
        resume=True,
        api=resumed_core.namespace(),
        invocation=["pytest", "resume"],
    )
    assert resumed["status"] == "complete"
    assert resumed["summary"]["finished"] == 3
    assert resumed_core.optimize_calls == [
        (
            missing["n_vertices"],
            missing["seed"],
            missing["budget"],
            missing["optimizer"],
        )
    ]
    assert campaign.main(["--validate-only", str(output)]) == 0
    assert "Valid H4 symmetric-maxima campaign v1" in capsys.readouterr().out

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        campaign.main(["--validate-only", str(invalid)])
    assert exit_info.value.code == 2
    assert "schema validation failed" in capsys.readouterr().err


def test_resume_rejects_optimizer_arm_drift(tmp_path: Path):
    output = tmp_path / "optimizer-drift.json"
    powell_config = _run_config(starts=1, local_max_on_shortlist=False)
    campaign.run_campaign(
        output,
        powell_config,
        api=FakeCore().namespace(),
        invocation=["pytest", "powell"],
    )

    with pytest.raises(ValueError, match="run config"):
        campaign.run_campaign(
            output,
            _run_config(
                starts=1,
                optimizer="fresh_topology_ad",
                local_max_on_shortlist=False,
            ),
            resume=True,
            api=FakeCore().namespace(),
            invocation=["pytest", "fresh-topology-ad-resume"],
        )


def test_timeout_checkpoint_is_resumable(tmp_path: Path, monkeypatch):
    output = tmp_path / "timed-out.json"
    config = _run_config(starts=2, local_max_on_shortlist=False)
    with monkeypatch.context() as context:
        context.setattr(
            campaign,
            "_execute_with_deadline",
            lambda *args, **kwargs: ("timed_out", None),
        )
        timed_out = campaign.run_campaign(
            output,
            config,
            api=FakeCore().namespace(),
            invocation=["pytest", "timeout"],
        )
    assert timed_out["status"] == "timed_out"
    assert timed_out["summary"]["pending"] == 2
    campaign.validate_campaign_document(timed_out)

    resumed = campaign.run_campaign(
        output,
        config,
        resume=True,
        api=FakeCore().namespace(),
        invocation=["pytest", "resume-after-timeout"],
    )
    assert resumed["status"] == "complete"
    assert resumed["summary"]["pending"] == 0


def test_semantic_and_finite_json_gates_reject_tampering(tmp_path: Path):
    output = tmp_path / "gates.json"
    document = campaign.run_campaign(
        output,
        _run_config(local_max_on_shortlist=False),
        api=FakeCore().namespace(),
        invocation=["pytest", "gates"],
    )

    bad_summary = deepcopy(document)
    bad_summary["summary"]["pending"] += 1
    with pytest.raises(ValueError, match="summary"):
        campaign.validate_campaign_document(bad_summary)

    leaked = deepcopy(document)
    row = next(item for item in leaked["tasks"] if item["seed"] % 2)
    row["vertices"] = _vertices(row["n_vertices"], -1.0)
    row["coordinates_retained"] = True
    row["coordinate_sha256"] = campaign._coordinate_digest(row["vertices"])
    with pytest.raises(ValueError, match="uncertified"):
        campaign.validate_campaign_document(leaked)

    nonfinite = deepcopy(document)
    nonfinite["tasks"][0]["elapsed_wall_seconds"] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        campaign.validate_campaign_document(nonfinite)

    fabricated = deepcopy(document)
    fabricated["tasks"][0]["audit"]["classification"] = "trivial_geometric_symmetry"
    fabricated["tasks"][0]["audit"]["independent_classification"] = (
        "trivial_geometric_symmetry"
    )
    with pytest.raises(ValueError, match="classification"):
        campaign.validate_campaign_document(fabricated)

    bad_work = deepcopy(document)
    bad_work["tasks"][0]["optimization"]["work"]["total_evals"] = 999_999
    with pytest.raises(ValueError, match="work"):
        campaign.validate_campaign_document(bad_work)

    bad_scores = deepcopy(document)
    bad_scores["tasks"][0]["optimization"]["scores"] = {}
    with pytest.raises(ValueError, match="scores"):
        campaign.validate_campaign_document(bad_scores)

    bad_optimizer = deepcopy(document)
    bad_optimizer["tasks"][0]["optimization"]["optimizer"]["selector"] = (
        "fresh_topology_ad"
    )
    with pytest.raises(ValueError, match="optimizer"):
        campaign.validate_campaign_document(bad_optimizer)

    missing_initial_digest = deepcopy(document)
    missing_optimization = missing_initial_digest["tasks"][0]["optimization"]
    del missing_optimization["initial_coordinate_sha256"]
    with pytest.raises(ValueError, match="schema validation failed"):
        campaign.validate_campaign_document(missing_initial_digest)
    with pytest.raises(ValueError, match="initial_coordinate_sha256"):
        campaign._validate_optimization_semantics(
            missing_optimization,
            missing_initial_digest["tasks"][0],
        )

    malformed_initial_digest = deepcopy(document)
    malformed_optimization = malformed_initial_digest["tasks"][0]["optimization"]
    malformed_optimization["initial_coordinate_sha256"] = "A" * 64
    with pytest.raises(ValueError, match="schema validation failed"):
        campaign.validate_campaign_document(malformed_initial_digest)
    with pytest.raises(ValueError, match="initial_coordinate_sha256"):
        campaign._validate_optimization_semantics(
            malformed_optimization,
            malformed_initial_digest["tasks"][0],
        )

    bad_task_optimizer = deepcopy(document)
    bad_task_optimizer["tasks"][0]["optimizer"] = "fresh_topology_ad"
    with pytest.raises(ValueError, match="optimizer"):
        campaign.validate_campaign_document(bad_task_optimizer)

    bad_poles = deepcopy(document)
    pole = bad_poles["tasks"][0]["audit"]["pole_volume_audit"]
    pole.update(
        requested_count=0,
        successful_count=0,
        failure_count=0,
        within_tolerance=True,
        spread=None,
    )
    with pytest.raises(ValueError, match="pole"):
        campaign.validate_campaign_document(bad_poles)

    leaked_optimization = deepcopy(document)
    leaked_optimization["tasks"][0]["optimization"]["vertices"] = [[1, 2, 3, 4]]
    with pytest.raises(ValueError, match="forbidden key"):
        campaign.validate_campaign_document(leaked_optimization)

    leaked_topology_bulk = deepcopy(document)
    leaked_topology_bulk["tasks"][0]["optimization"]["optimizer"][
        "diagnostics"
    ] = {
        "branches": [
            {
                "finite_vertex_labels": [0, 1, 2],
                "topology_signature_sha256_examples": ["a" * 64],
            }
        ]
    }
    with pytest.raises(ValueError, match="forbidden key"):
        campaign.validate_campaign_document(leaked_topology_bulk)

    leaked_audit = deepcopy(document)
    leaked_audit["tasks"][0]["audit"]["symmetry"]["geometric_generators"] = [[0]]
    with pytest.raises(ValueError, match="forbidden key"):
        campaign.validate_campaign_document(leaked_audit)

    fake_strict = deepcopy(document)
    local = fake_strict["tasks"][0]["audit"]["local_maximum_audit"]
    local.update(
        requested=True,
        performed=True,
        status="error",
        classification="strict_local_maximum",
    )
    classification, basis = campaign._compact_classification(
        fake_strict["tasks"][0]["audit"]["symmetry"],
        fake_strict["tasks"][0]["audit"]["pole_volume_audit"],
        local,
    )
    fake_strict["tasks"][0]["audit"].update(
        classification=classification,
        independent_classification=classification,
        classification_basis=basis,
    )
    with pytest.raises(ValueError, match="completed diagnostic"):
        campaign.validate_campaign_document(fake_strict)


def test_fresh_optimizer_diagnostic_invariants_reject_mutations(tmp_path: Path):
    document = campaign.run_campaign(
        tmp_path / "fresh-diagnostics.json",
        _run_config(
            starts=1,
            base_seed=20,
            optimizer="fresh_topology_ad",
            local_max_on_shortlist=False,
        ),
        api=FakeCore().namespace(),
        invocation=["pytest", "fresh-diagnostics"],
    )
    campaign.validate_campaign_document(document)

    def replace(document_copy, path, value):
        target = document_copy["tasks"][0]["optimization"]["optimizer"][
            "diagnostics"
        ]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    mutations = (
        (("selector",), "powell"),
        (
            ("work",),
            {
                "objective_evals": 2,
                "surrogate_evals": 0,
                "gradient_evals": 0,
                "total_evals": 2,
            },
        ),
        (("executed_branch_count",), 2),
        (("completed_branch_count",), 0),
        (("branches", 0, "gradient_evals"), 0),
        (("branches", 0, "topology_evaluations"), 2),
        (("budget_allocation", "gradient_evals_by_branch"), [8]),
        (("canonical_scored_branch_count",), 2),
    )
    for path, value in mutations:
        tampered = deepcopy(document)
        replace(tampered, path, value)
        with pytest.raises(ValueError, match="fresh optimizer diagnostics"):
            campaign.validate_campaign_document(tampered)


def test_fake_calibration_requires_exact_certified_order_384():
    config = _run_config(starts=0, calibrate_16_cell=True)
    result = campaign.run_16_cell_calibration(config, FakeCore().namespace())
    assert result["status"] == "completed"
    assert result["passes"] is True
    assert result["audit"]["symmetry"]["valid_geometry"] is True
    assert result["audit"]["symmetry"]["full_group_certified"] is True
    assert result["audit"]["symmetry"]["geometric_group_order"] == 384

    tampered = deepcopy(result)
    tampered["audit"]["symmetry"]["geometric_group_order"] = 383
    assert not (
        tampered["audit"]["symmetry"]["geometric_group_order"]
        == tampered["expected_geometric_group_order"]
    )


def test_real_n8_sixteen_cell_calibration_smoke(tmp_path: Path):
    output = tmp_path / "real-calibration.json"
    config = _run_config(
        starts=0,
        calibrate_16_cell=True,
        local_max_on_shortlist=False,
        budget=1,
    )
    document = campaign.run_campaign(
        output,
        config,
        api=campaign._load_core_api(),
        invocation=["pytest", "real-16-cell-calibration"],
    )
    calibration = document["calibration"]
    symmetry = calibration["audit"]["symmetry"]
    assert document["status"] == "complete"
    assert document["summary"]["planned"] == 0
    assert calibration["passes"] is True
    assert symmetry["valid_geometry"] is True
    assert symmetry["full_group_certified"] is True
    assert symmetry["certified_nontrivial_symmetry"] is True
    assert symmetry["geometric_group_order"] == 384
    assert calibration["audit"]["pole_volume_audit"]["all_poles_requested"] is True
    campaign.validate_campaign_document(document)

    wrong_order = deepcopy(document)
    wrong_order["calibration"]["audit"]["symmetry"]["geometric_group_order"] = 383
    with pytest.raises((ValidationError, ValueError)):
        campaign.validate_campaign_document(wrong_order)

    incomplete = deepcopy(document)
    incomplete["calibration"]["audit"]["symmetry"]["full_group_certified"] = False
    with pytest.raises((ValidationError, ValueError)):
        campaign.validate_campaign_document(incomplete)

    wrong_digest = deepcopy(document)
    wrong_digest["calibration"]["coordinate_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="calibration"):
        campaign.validate_campaign_document(wrong_digest)

    bad_poles = deepcopy(document)
    bad_poles["calibration"]["audit"]["pole_volume_audit"]["within_tolerance"] = False
    with pytest.raises(ValueError, match="calibration"):
        campaign.validate_campaign_document(bad_poles)

    bad_orbits = deepcopy(document)
    bad_orbits["calibration"]["audit"]["symmetry"]["orbit_sizes"] = []
    bad_orbits["calibration"]["audit"]["symmetry"]["geometric_generator_count"] = 0
    with pytest.raises(ValueError, match="orbit"):
        campaign.validate_campaign_document(bad_orbits)


def test_failed_calibration_stops_before_mining(tmp_path: Path):
    class BrokenCalibrationCore(FakeCore):
        def audit_h4_symmetric_candidate(self, *args, **kwargs):
            raise RuntimeError("broken verifier")

    core = BrokenCalibrationCore()
    document = campaign.run_campaign(
        tmp_path / "failed-calibration.json",
        _run_config(starts=2, calibrate_16_cell=True),
        api=core.namespace(),
        invocation=["pytest", "failed-calibration"],
    )

    assert document["status"] == "failed"
    assert document["calibration"]["status"] == "failed"
    assert document["summary"]["pending"] == 2
    assert core.optimize_calls == []

    resumed_core = BrokenCalibrationCore()
    resumed = campaign.run_campaign(
        tmp_path / "failed-calibration.json",
        _run_config(starts=2, calibrate_16_cell=True),
        resume=True,
        api=resumed_core.namespace(),
        invocation=["pytest", "failed-calibration-resume"],
    )
    assert resumed["status"] == "failed"
    assert resumed["summary"]["pending"] == 2
    assert resumed_core.optimize_calls == []


def test_resume_rejects_changed_runtime_environment(tmp_path: Path):
    output = tmp_path / "environment.json"
    config = _run_config(starts=1, local_max_on_shortlist=False)
    document = campaign.run_campaign(
        output,
        config,
        api=FakeCore().namespace(),
        invocation=["pytest", "environment"],
    )
    document["provenance"]["python"] = "0.0.0"
    campaign.atomic_write_json(output, document)

    with pytest.raises(ValueError, match="python changed"):
        campaign.run_campaign(
            output,
            config,
            resume=True,
            api=FakeCore().namespace(),
            invocation=["pytest", "environment-resume"],
        )
