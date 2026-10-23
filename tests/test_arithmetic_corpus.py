import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from jsonschema.exceptions import ValidationError


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts/generate_arithmeticity_benchmark.py"
SPEC = importlib.util.spec_from_file_location("arithmetic_corpus_script", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
arithmetic_corpus = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = arithmetic_corpus
SPEC.loader.exec_module(arithmetic_corpus)


def _run_config(**overrides):
    config = {
        "mode": "layered-ring",
        "vertex_semantics": "total-including-explicit-infinity",
        "n_vertices": 6,
        "target_count": 1,
        "max_attempts": 1,
        "max_wall_seconds": 30.0,
        "workers": 1,
        "base_seed": 42,
        "blas_threads_per_worker": arithmetic_corpus.BLAS_THREADS,
        "max_flips": 64,
        "shear_tolerance": 1e-9,
        "margin_tolerance": 1e-4,
        "constraint_tolerance": 1e-6,
        "qhull_audit": False,
    }
    config.update(overrides)
    return config


def _edge_incidence(faces):
    incidence = {}
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((int(u), int(v))))
            incidence[edge] = incidence.get(edge, 0) + 1
    return incidence


def test_layered_ring_factorizations_for_300_and_302_vertices():
    assert arithmetic_corpus.layered_ring_factor_pairs(300) == [
        (1, 298),
        (2, 149),
    ]
    assert arithmetic_corpus.layered_ring_factor_pairs(302) == [
        (1, 300),
        (2, 150),
        (3, 100),
        (4, 75),
        (5, 60),
        (6, 50),
        (10, 30),
        (12, 25),
        (15, 20),
        (20, 15),
        (25, 12),
        (30, 10),
        (50, 6),
        (60, 5),
        (75, 4),
    ]
    assert arithmetic_corpus.layered_ring_factor_pairs(5) == []


@pytest.mark.parametrize("total_vertices", [300, 302])
def test_every_layered_ring_factor_has_exact_closed_sphere_counts(total_vertices):
    for k, n in arithmetic_corpus.layered_ring_factor_pairs(total_vertices):
        construction = arithmetic_corpus.construct_layered_ring(k, n)
        assert construction.total_vertices == total_vertices
        assert len(construction.finite_vertices) == total_vertices - 1
        assert construction.infinity_index == total_vertices - 1
        assert len(construction.boundary_triangles) == 2 * total_vertices - 4
        assert len(construction.finite_triangles) == 2 * k * n - n
        assert np.isfinite(construction.finite_vertices).all()

        incidence = _edge_incidence(construction.boundary_triangles)
        assert len(incidence) == 3 * total_vertices - 6
        assert set(incidence.values()) == {2}
        topology = arithmetic_corpus.validate_closed_sphere_triangulation(
            construction.boundary_triangles
        )
        assert (topology.n_vertices, topology.n_edges, topology.n_faces) == (
            total_vertices,
            3 * total_vertices - 6,
            2 * total_vertices - 4,
        )
        assert topology.euler_characteristic == 2


@pytest.mark.parametrize("k,n", [(1, 298), (2, 149), (1, 300), (75, 4)])
def test_large_layered_ring_coordinates_pass_default_shear_gate(k, n):
    configuration = arithmetic_corpus.generate_layered_ring_configuration(
        k,
        n,
        qhull_audit=False,
    )
    total_vertices = k * n + 2
    assert configuration["input"]["n_vertices"] == total_vertices
    assert configuration["input"]["vertices"][-1] is None
    assert configuration["input"]["n_triangles"] == 2 * total_vertices - 4
    shear = configuration["certificate"]["coordinate_zero_shear"]
    assert shear["all_finite"]
    assert shear["within_tolerance"]
    assert shear["max_abs"] < 2e-13
    assert configuration["certificate"]["checks"]["accepted"]


def test_exact_holonomy_integers_are_decimal_strings_at_json_boundary():
    configuration = arithmetic_corpus.generate_layered_ring_configuration(
        2,
        4,
        qhull_audit=False,
    )
    exact = configuration["certificate"]["exact_zero_shear"]
    assert exact["traces"] and all(isinstance(trace, str) for trace in exact["traces"])
    for generator in exact["generators"]:
        assert isinstance(generator["trace"], str)
        assert all(
            isinstance(entry, str)
            for row in generator["matrix"]
            for entry in row
        )
        assert generator["determinant"] == 1
    assert configuration["certificate"]["exact_trace_summary"][
        "float_integrality_test_used"
    ] is False
    assert "q + q^-1" in configuration["construction"]["parameters"][
        "radius_ratio_equation"
    ]


def test_strict_json_rejects_nonfinite_values_and_duplicate_keys(tmp_path):
    with pytest.raises(ValueError, match="non-finite"):
        arithmetic_corpus.strict_json_bytes({"value": math.nan})

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        arithmetic_corpus.load_json_strict(duplicate)


def test_atomic_create_is_no_clobber_and_atomic_replace_is_strict(tmp_path):
    output = tmp_path / "nested/corpus.json"
    arithmetic_corpus.exclusive_create_json(output, {"value": 1})
    with pytest.raises(FileExistsError, match="refusing to clobber"):
        arithmetic_corpus.exclusive_create_json(output, {"value": 2})
    assert arithmetic_corpus.load_json_strict(output) == {"value": 1}

    arithmetic_corpus.atomic_write_json(output, {"value": 3})
    assert arithmetic_corpus.load_json_strict(output) == {"value": 3}
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_complete_campaign_validates_schema_and_resumes_without_clobber(tmp_path):
    output = tmp_path / "corpus.json"
    run_config = _run_config()
    document = arithmetic_corpus.run_campaign(
        output,
        run_config,
        invocation=["pytest"],
        use_subprocesses=False,
    )
    assert document["status"] == "complete"
    assert document["counters"] == {
        "attempted": 1,
        "accepted": 1,
        "rejected": 0,
        "duplicates": 0,
        "errors": 0,
        "timed_out": 0,
    }
    arithmetic_corpus.validate_corpus_document(
        arithmetic_corpus.load_json_strict(output)
    )

    original_bytes = output.read_bytes()
    resumed = arithmetic_corpus.run_campaign(
        output,
        run_config,
        resume=True,
        invocation=["pytest", "--resume"],
        use_subprocesses=False,
    )
    assert resumed["status"] == "complete"
    assert output.read_bytes() == original_bytes

    with pytest.raises(FileExistsError):
        arithmetic_corpus.run_campaign(
            output,
            run_config,
            invocation=["pytest"],
            use_subprocesses=False,
        )
    changed = dict(run_config, shear_tolerance=1e-8)
    with pytest.raises(ValueError, match="run config"):
        arithmetic_corpus.run_campaign(
            output,
            changed,
            resume=True,
            invocation=["pytest", "--resume"],
            use_subprocesses=False,
        )


def test_schema_rejects_unknown_root_fields(tmp_path):
    output = tmp_path / "corpus.json"
    document = arithmetic_corpus.run_campaign(
        output,
        _run_config(),
        invocation=["pytest"],
        use_subprocesses=False,
    )
    malformed = json.loads(json.dumps(document))
    malformed["unversioned_extra"] = True
    with pytest.raises(ValidationError):
        arithmetic_corpus.validate_corpus_document(malformed)


def test_networkx_fallback_uses_exact_isomorphism_after_relabeling():
    construction = arithmetic_corpus.construct_layered_ring(1, 4)
    faces = construction.boundary_triangles
    labels = list(range(construction.total_vertices))
    permutation = dict(zip(labels, reversed(labels)))
    relabeled = [tuple(permutation[vertex] for vertex in face) for face in faces]
    backend = "networkx-exact-isomorphism-v1"
    topology_id, _ = arithmetic_corpus.topology_identifier(faces, backend)
    existing = {
        "id": "arithmetic-000000",
        "input": {"triangles": [list(face) for face in faces]},
        "deduplication": {"method": backend, "topology_id": topology_id},
    }
    _, duplicate_of = arithmetic_corpus.find_duplicate_configuration(
        relabeled,
        [existing],
        backend,
    )
    assert duplicate_of == "arithmetic-000000"


def test_pynauty_identifier_is_relabeling_invariant_when_available():
    if arithmetic_corpus.available_deduplication_backend() != (
        "pynauty-canonical-certificate-v1"
    ):
        pytest.skip("pynauty is not installed")
    construction = arithmetic_corpus.construct_layered_ring(2, 4)
    faces = construction.boundary_triangles
    count = construction.total_vertices
    relabeled = [tuple(count - 1 - vertex for vertex in face) for face in faces]
    backend = "pynauty-canonical-certificate-v1"
    assert arithmetic_corpus.topology_identifier(faces, backend)[0] == (
        arithmetic_corpus.topology_identifier(relabeled, backend)[0]
    )


def test_legalized_mode_calls_settled_apis_and_closes_infinity(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        arithmetic_corpus,
        "delaunay_triangulation_indices",
        lambda points: np.asarray([[0, 1, 2]], dtype=np.int64),
    )

    def fake_legalize(faces, **kwargs):
        calls["legalize"] = (faces, kwargs)
        return {
            "success": True,
            "termination_reason": "accepted",
            "message": "accepted",
            "finite_triangulation": [(0, 1, 2)],
            "angles": np.full((1, 3), math.pi / 3.0),
            "volume": 1.0149416064096536,
            "n_flips": 0,
            "n_states": 1,
            "residuals": None,
            "optimization_result": {
                "reduced_gradient_norm": 0.0,
                "n_active_constraints": 0,
            },
            "history": [],
            "timings": {
                "total_seconds": 0.01,
                "lp_seconds": 0.002,
                "optimization_seconds": 0.008,
            },
        }

    monkeypatch.setattr(
        arithmetic_corpus,
        "legalize_to_interior_maximum",
        fake_legalize,
    )

    equilateral = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]]
    )

    def fake_realize(faces, angles, **kwargs):
        calls["realize"] = (faces, angles, kwargs)
        return {
            "success": True,
            "points": equilateral,
            "vertex_list": [0, 1, 2],
            "triangulation_preserved": True,
            "angle_error": 0.0,
            "angle_error_degrees": 0.0,
            "max_angle_error": 0.0,
            "max_angle_error_degrees": 0.0,
            "closure_error": 0.0,
            "orientation_consistent": True,
            "min_abs_oriented_area": math.sqrt(3.0) / 4.0,
            "residuals": {},
            "message": "ok",
            "n_iterations": 0,
        }

    monkeypatch.setattr(arithmetic_corpus, "realize_angles_as_points", fake_realize)
    configuration = arithmetic_corpus.generate_legalized_configuration(4, seed=1002)
    assert calls["legalize"][1]["max_flips"] == 64
    assert calls["realize"][0] == [(0, 1, 2)]
    assert configuration["input"]["n_vertices"] == 4
    assert configuration["input"]["infinity_index"] == 3
    assert configuration["input"]["vertices"][-1] is None
    assert len(configuration["input"]["triangles"]) == 4
    assert configuration["certificate"]["coordinate_zero_shear"][
        "within_tolerance"
    ]

    def bad_realize(faces, angles, **kwargs):
        result = fake_realize(faces, angles, **kwargs)
        result["max_angle_error"] = 1e-4
        result["max_angle_error_degrees"] = float(np.degrees(1e-4))
        return result

    monkeypatch.setattr(arithmetic_corpus, "realize_angles_as_points", bad_realize)
    with pytest.raises(
        arithmetic_corpus.CandidateRejected,
        match="maximum-angle, closure, or winding gate",
    ):
        arithmetic_corpus.generate_legalized_configuration(4, seed=1002)


def test_failed_attempts_are_bounded_and_checkpointed(tmp_path):
    output = tmp_path / "exhausted.json"
    run_config = _run_config(
        mode="legalized",
        n_vertices=20,
        target_count=1,
        max_attempts=2,
    )

    def reject_attempt(config, attempt_index):
        return {
            "outcome": "rejected",
            "reason": "test_rejection",
            "message": "bounded rejection",
            "candidate_key": f"test-{attempt_index}",
            "seed": arithmetic_corpus.derive_attempt_seed(
                config["base_seed"], attempt_index
            ),
            "diagnostics": {},
        }

    document = arithmetic_corpus.run_campaign(
        output,
        run_config,
        invocation=["pytest"],
        use_subprocesses=False,
        attempt_runner=reject_attempt,
    )
    assert document["status"] == "exhausted"
    assert document["counters"]["attempted"] == 2
    assert document["counters"]["rejected"] == 2
    assert document["next_attempt_index"] == 2
    assert arithmetic_corpus.load_json_strict(output)["status"] == "exhausted"


def test_deprecated_cli_alias_and_bounds():
    parser = arithmetic_corpus._build_parser()
    args = parser.parse_args(
        ["--n-vertices", "300", "--n-optimized", "2", "--n-random", "0"]
    )
    with pytest.warns(FutureWarning, match="deprecated"):
        config = arithmetic_corpus._run_config_from_args(args, parser)
    assert config["target_count"] == 2
    assert config["n_vertices"] == 300

    with pytest.raises(ValueError, match="workers"):
        arithmetic_corpus._validate_run_config(_run_config(workers=9))
    with pytest.raises(ValueError, match="only 2 strict"):
        arithmetic_corpus._validate_run_config(
            _run_config(n_vertices=300, target_count=3, max_attempts=3)
        )
