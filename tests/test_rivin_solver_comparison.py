from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from jsonschema.validators import validator_for

from ideal_poly_volume_toolkit.rivin_zero_shear import normalize_cp1_points


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "compare_rivin_zero_shear_methods.py"
SCHEMA_PATH = ROOT / "docs" / "schemas" / "rivin-zero-shear-comparison-v1.schema.json"
SPEC = importlib.util.spec_from_file_location(
    "compare_rivin_zero_shear_methods", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
comparison = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = comparison
SPEC.loader.exec_module(comparison)


FACES = [
    [0, 1, 2],
    [0, 3, 1],
    [1, 3, 2],
    [2, 3, 0],
]
CANONICAL_VERTICES = [
    None,
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, math.sqrt(3.0) / 2.0],
]


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return path


def _answer_key(*, reference_vertices: object = CANONICAL_VERTICES) -> dict:
    return {
        "instance_id": "synthetic-tetrahedron",
        "normalization": {"distinguished_face": [0, 1, 2]},
        "ground_truth": {
            "exists": True,
            "oriented_faces": FACES,
            "vertices": reference_vertices,
        },
    }


def _run_config(case_id: str = "synthetic-tetrahedron", **overrides) -> dict:
    result = {
        "selected_case_ids": [case_id],
        "max_nfev": 500,
        "lm_polish_steps": 3,
        "x_scale": "unit",
        "shear_tolerance": 1e-8,
        "support_tolerance": 0.0,
        "bending_tolerance": 1e-10,
        "ftol": 1e-13,
        "xtol": 1e-13,
        "gtol": 1e-13,
        "include_coordinates": False,
        "include_face_tutte": False,
    }
    result.update(overrides)
    return result


def _strict_audit(*_args, **_kwargs) -> dict:
    return {
        "strict_pass": True,
        "classification": "target_strict_zero_shear",
        "errors": [],
    }


class FakeCore:
    def __init__(self) -> None:
        self.solve_calls: list[dict] = []
        self.audit_calls: list[dict] = []
        self.pole_initializer_name = "weighted_pole_link_tutte_embedding"

    @staticmethod
    def extract_boundary_vertices(faces):
        return sorted({label for face in faces for label in face})

    @staticmethod
    def check_delaunay_realizability(faces, *, boundary_vertices, verbose) -> dict:
        assert all(0 not in face for face in faces)
        assert set(boundary_vertices) == {1, 2, 3}
        assert verbose is False
        return {
            "realizable": True,
            "success": True,
            "message": "ok",
            "angles_radians": [math.pi / 3.0] * 3,
            "min_angle_radians": math.pi / 3.0,
        }

    @staticmethod
    def optimize_hyperbolic_volume(
        faces, *, initial_angles, boundary_vertices, verbose, require_interior
    ) -> dict:
        assert all(0 not in face for face in faces)
        assert initial_angles
        assert set(boundary_vertices) == {1, 2, 3}
        assert verbose is False
        assert require_interior is True
        return {
            "success": True,
            "message": "ok",
            "angles": [math.pi / 3.0] * 3,
            "n_iterations": 1,
            "n_active_constraints": 0,
            "reduced_gradient_norm": 0.0,
            "volume": 1.0,
        }

    @staticmethod
    def realize_angles_as_points(faces, angles, *, boundary_vertices, verbose) -> dict:
        assert all(0 not in face for face in faces)
        assert angles
        assert set(boundary_vertices) == {1, 2, 3}
        assert verbose is False
        return {
            "success": True,
            "message": "ok",
            "points": [[0.0, 0.0], [1.0, 0.0], CANONICAL_VERTICES[3]],
            "vertex_list": [1, 2, 3],
            "triangulation_preserved": True,
            "angle_error": 0.0,
            "max_angle_error": 0.0,
            "closure_error": 0.0,
            "orientation_consistent": True,
        }

    @staticmethod
    def normalize_cp1_points(vertices, face, *, positive_handedness=False):
        return normalize_cp1_points(
            vertices, face, positive_handedness=positive_handedness
        )

    @staticmethod
    def pole_link_initializer(faces, distinguished_face):
        assert tuple(distinguished_face) == (0, 1, 2)
        assert len(faces) == 4
        return tuple(
            None if row is None else complex(float(row[0]), float(row[1]))
            for row in CANONICAL_VERTICES
        )

    @staticmethod
    def weighted_tutte_embedding(faces, distinguished_face):
        del faces, distinguished_face
        return CANONICAL_VERTICES

    def solve_zero_shear_from_initial(
        self,
        faces,
        distinguished_face,
        initial_vertices_cp1,
        *,
        gauge,
        x_scale,
        max_nfev,
        lm_polish_steps,
        record_trajectory=False,
        shear_tolerance,
        support_tolerance,
        bending_tolerance,
        check_spherical_embedding,
        ftol,
        xtol,
        gtol,
    ) -> dict:
        call = {
            "faces": tuple(tuple(face) for face in faces),
            "distinguished_face": tuple(distinguished_face),
            "initial": tuple(initial_vertices_cp1),
            "gauge": gauge,
            "x_scale": x_scale,
            "max_nfev": max_nfev,
            "lm_polish_steps": lm_polish_steps,
            "record_trajectory": record_trajectory,
            "shear_tolerance": shear_tolerance,
            "support_tolerance": support_tolerance,
            "bending_tolerance": bending_tolerance,
            "check_spherical_embedding": check_spherical_embedding,
            "ftol": ftol,
            "xtol": xtol,
            "gtol": gtol,
        }
        self.solve_calls.append(call)
        return {
            # Deliberately false: a strict independently audited endpoint is
            # still a valid numerical maximality certificate for this scope.
            "success": False,
            "classification": "maximum_evaluations_exceeded",
            "message": "TRF budget exhausted before reporting success",
            "normalized_vertices_cp1": CANONICAL_VERTICES,
            "nfev": max_nfev,
            "njev": max_nfev,
            "max_abs_residual": 1e-12,
            "residual_l2": 2e-12,
            "solver_metadata": {
                "gauge": gauge,
                "x_scale": x_scale,
                "trf": {"success": False},
                "lm_polish": {"requested_steps": lm_polish_steps},
            },
            "timings": {
                "initialization_seconds": 0.001,
                "solve_seconds": 0.002,
                "trf_seconds": 0.0015,
                "lm_polish_seconds": 0.0005,
                "audit_seconds": 0.001,
            },
        }

    def audit_zero_shear_realization(self, *_args, **kwargs) -> dict:
        self.audit_calls.append(dict(kwargs))
        return _strict_audit()


def _api(fake: FakeCore) -> SimpleNamespace:
    return SimpleNamespace(
        extract_boundary_vertices=fake.extract_boundary_vertices,
        check_delaunay_realizability=fake.check_delaunay_realizability,
        optimize_hyperbolic_volume=fake.optimize_hyperbolic_volume,
        realize_angles_as_points=fake.realize_angles_as_points,
        normalize_cp1_points=fake.normalize_cp1_points,
        pole_link_initializer=fake.pole_link_initializer,
        pole_link_initializer_name=fake.pole_initializer_name,
        weighted_tutte_embedding=fake.weighted_tutte_embedding,
        solve_zero_shear_from_initial=fake.solve_zero_shear_from_initial,
        audit_zero_shear_realization=fake.audit_zero_shear_realization,
    )


def test_schema_is_valid_draft_2020_12() -> None:
    schema = comparison.load_json_strict(SCHEMA_PATH)
    validator = validator_for(schema)
    validator.check_schema(schema)


def test_pole_link_disk_removes_entire_star_not_one_face() -> None:
    case = comparison.ComparisonCase(
        case_id="tetrahedron",
        faces=tuple(tuple(face) for face in FACES),
        distinguished_face=(0, 1, 2),
        reference_vertices=None,
        source_pointer="/",
        source_kind="simple_face_list",
    )

    disk, link = comparison._pole_link_disk(case)

    assert disk == ((1, 3, 2),)
    assert link == (1, 2, 3)
    assert all(0 not in face for face in disk)


def test_plain_face_list_and_answer_key_inputs_are_supported(tmp_path: Path) -> None:
    face_path = _write_json(tmp_path / "plain.json", FACES)
    kind, version, cases = comparison.load_comparison_cases(face_path)
    assert (kind, version) == ("simple_face_list", None)
    assert cases[0].case_id == "plain"
    assert cases[0].reference_vertices is None

    key_path = _write_json(tmp_path / "key.json", _answer_key())
    kind, version, cases = comparison.load_comparison_cases(key_path)
    assert (kind, version) == ("strict_answer_key", None)
    assert cases[0].case_id == "synthetic-tetrahedron"
    assert cases[0].reference_vertices is not None


def test_arithmetic_corpus_selection_is_sorted_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = {
        "schema_version": "ideal-poly-arithmetic-corpus/1",
        "configurations": [
            {"id": "z", "input": {"triangles": FACES, "vertices": None}},
            {"id": "a", "input": {"triangles": FACES, "vertices": None}},
        ],
    }
    path = _write_json(tmp_path / "corpus.json", corpus)
    monkeypatch.setattr(comparison, "_validate_with_schema", lambda *_args: None)

    kind, version, cases = comparison.load_comparison_cases(
        path, requested_ids=["z", "a"]
    )

    assert kind == "arithmetic_corpus"
    assert version == "ideal-poly-arithmetic-corpus/1"
    assert [case.case_id for case in cases] == ["a", "z"]


def test_direct_contract_and_strict_endpoint_are_independent_of_solver_status(
    tmp_path: Path,
) -> None:
    key_path = _write_json(tmp_path / "key.json", _answer_key())
    _, _, cases = comparison.load_comparison_cases(key_path)
    fake = FakeCore()

    row, endpoint = comparison._run_direct_method(
        cases[0],
        _run_config(),
        _api(fake),
        method_id=comparison.METHOD_POLE,
        label="direct",
        initial_kind="pole_link_regular_polygon_harmonic_disk",
        initializer=lambda: fake.pole_link_initializer(
            cases[0].faces, cases[0].distinguished_face
        ),
    )

    assert endpoint is not None
    assert row["status"] == "completed"
    assert row["solver"]["success"] is False
    assert row["audit"]["strict_pass"] is True
    assert row["maximality_assessment"]["claimed"] is True
    assert fake.solve_calls[0]["gauge"] == "standard_upper_half_plane"
    assert fake.solve_calls[0]["x_scale"] == 1.0
    assert fake.solve_calls[0]["max_nfev"] == 500
    assert fake.solve_calls[0]["lm_polish_steps"] == 3


def test_noncanonical_face_uses_script_anchor_gate_not_core_label_gate() -> None:
    face = (3, 2, 1)
    vertices = (
        complex(0.5, math.sqrt(3.0) / 2.0),
        1 + 0j,
        0j,
        None,
    )
    case = comparison.ComparisonCase(
        case_id="relabelled",
        faces=tuple(tuple(3 - label for label in row) for row in FACES),
        distinguished_face=face,
        reference_vertices=None,
        source_pointer="/",
        source_kind="simple_face_list",
    )
    received: dict = {}

    def audit(*_args, **kwargs):
        received.update(kwargs)
        return _strict_audit()

    row = comparison._new_method_row(comparison.METHOD_ANGLE, "angle")
    normalized = comparison._finalize_endpoint(
        row,
        case=case,
        vertices=vertices,
        audit_function=audit,
        normalize_function=normalize_cp1_points,
        run_config=_run_config("relabelled"),
        include_coordinates=False,
    )

    assert received["require_standard_normalization"] is False
    assert normalized[3] is None
    assert normalized[2] == 0j
    assert normalized[1] == 1 + 0j
    assert row["audit"]["script_normalization"] == {
        "passes": True,
        "ordered_anchors_match": True,
        "positive_half_plane": True,
    }
    assert row["maximality_assessment"]["claimed"] is True


def test_complete_fake_run_validates_schema_and_never_initializes_from_reference(
    tmp_path: Path,
) -> None:
    reference = [None, [0.0, 0.0], [1.0, 0.0], [0.4, 0.8]]
    input_path = _write_json(
        tmp_path / "answer-key.json", _answer_key(reference_vertices=reference)
    )
    output_path = tmp_path / "comparison.json"
    fake = FakeCore()

    document = comparison.run_comparison(
        input_path,
        output_path,
        _run_config(),
        overwrite=False,
        invocation=["pytest", "fake-comparison"],
        api=_api(fake),
    )

    assert document["status"] == "complete"
    assert output_path.is_file()
    assert comparison.load_json_strict(output_path) == document
    methods = document["cases"][0]["methods"]
    assert [row["method_id"] for row in methods] == list(comparison.REQUIRED_METHODS)
    assert all(row["status"] == "completed" for row in methods)
    assert all(row["audit"]["strict_pass"] for row in methods)
    assert all(row["maximality_assessment"]["claimed"] for row in methods)
    assert all(row["reference_error"] is not None for row in methods)
    assert document["fairness"]["input_reference_coordinates_diagnostic_only"]
    assert document["fairness"]["time_ratios_include_core_and_independent_audits"]
    assert len(fake.solve_calls) == 2
    assert all(call["initial"][3] != complex(0.4, 0.8) for call in fake.solve_calls)
    assert all(
        set(stage) >= {"wall_seconds", "process_cpu_seconds"}
        for row in methods
        for stage in row["stages"]
    )
    assert set(document["timings"]) == {"wall_seconds", "process_cpu_seconds"}


def test_output_is_no_clobber_by_default_and_strict_json(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "result.json"
    comparison.write_json_atomic(output, {"value": 1.0}, overwrite=False)
    with pytest.raises(FileExistsError, match="refusing to clobber"):
        comparison.write_json_atomic(output, {"value": 2.0}, overwrite=False)
    assert comparison.load_json_strict(output) == {"value": 1.0}
    with pytest.raises(ValueError, match="non-finite"):
        comparison.strict_json_bytes({"value": math.nan})


def test_partial_failure_cli_exit_is_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    case = comparison.ComparisonCase(
        case_id="case",
        faces=tuple(tuple(face) for face in FACES),
        distinguished_face=(0, 1, 2),
        reference_vertices=None,
        source_pointer="/",
        source_kind="simple_face_list",
    )
    monkeypatch.setattr(
        comparison,
        "load_comparison_cases",
        lambda *_args, **_kwargs: ("simple_face_list", None, [case]),
    )
    monkeypatch.setattr(
        comparison,
        "run_comparison",
        lambda *_args, **_kwargs: {
            "status": "partial_failure",
            "summary": {"case_count": 1},
        },
    )

    assert comparison.main([str(input_path), "--output", str(output_path)]) == 1
