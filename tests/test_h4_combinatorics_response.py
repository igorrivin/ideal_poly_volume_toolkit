"""Reference-free ideal-H4 prescribed-combinatorics checker tests."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "check_h4_combinatorics_response.py"
SPEC = importlib.util.spec_from_file_location(
    "check_h4_combinatorics_response", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def _sixteen_cell_vertices() -> np.ndarray:
    return np.vstack((np.eye(4), -np.eye(4)))


def _sixteen_cell_facets() -> list[list[int]]:
    facets = []
    for signs in itertools.product((0, 1), repeat=4):
        facets.append(sorted(axis + 4 * sign for axis, sign in enumerate(signs)))
    return sorted(facets)


def _target(*, minimum_volume: float = 1.7) -> dict[str, object]:
    return {
        "schema": checker.TARGET_SCHEMA_VERSION,
        "instance_id": "test-ideal-16-cell",
        "dimension": 4,
        "n_vertices": 8,
        "facets": _sixteen_cell_facets(),
        "facet_equivalence": "unlabeled_vertex_facet_incidence",
        "minimum_volume": minimum_volume,
        "tolerances": {
            "unit": 1.0e-9,
            "distinct": 1.0e-8,
            "support": 1.0e-9,
            "pole_volume_spread": 1.0e-8,
        },
    }


def _response(vertices: np.ndarray | None = None) -> dict[str, object]:
    points = _sixteen_cell_vertices() if vertices is None else vertices
    return {"vertices": points.tolist()}


def _fake_audit(
    points: np.ndarray,
    target_facets: object,
    *,
    volume: float = 1.8,
    candidate_facets: list[list[int]] | None = None,
    **_kwargs: object,
) -> dict[str, object]:
    facets = _sixteen_cell_facets() if candidate_facets is None else candidate_facets
    rows = [
        {
            "pole_index": pole,
            "volume": volume,
            "simplex_count": sum(pole not in facet for facet in facets),
        }
        for pole in range(len(points))
    ]
    facet_rows = [
        {
            "pole_index": pole,
            "expected_simplex_count": sum(pole not in facet for facet in facets),
            "delaunay_simplex_count": sum(pole not in facet for facet in facets),
            "volume_simplex_count": sum(pole not in facet for facet in facets),
            "facets_match": True,
            "simplex_counts_match": True,
            "matches": True,
            "missing_facets": [],
            "extra_facets": [],
        }
        for pole in range(len(points))
    ]
    return {
        "dimension": 4,
        "n_vertices": len(points),
        "classification": "accepted",
        "accepted": True,
        "checks": {
            "geometry": True,
            "target_incidence_input": True,
            "incidence_backend": True,
            "combinatorics": True,
            "pole_facet_consistency": True,
            "all_pole_volume": True,
            "volume_floor": True,
        },
        "geometry": {
            "valid_geometry": True,
            "checks": {name: True for name in checker._GEOMETRY_CHECKS},
            "facets": facets,
            "topology": {
                "min_support_margin": 0.1,
                "min_pairwise_distance": 1.0,
            },
        },
        "incidence": {"available": True, "isomorphic": True},
        "pole_facet_consistency": {
            "all_poles_requested": True,
            "requested_count": len(points),
            "successful_count": len(points),
            "failure_count": 0,
            "all_match": True,
            "values": facet_rows,
            "failures": [],
        },
        "pole_volume_audit": {
            "all_poles_requested": True,
            "requested_indices": list(range(len(points))),
            "requested_count": len(points),
            "successful_count": len(points),
            "failure_count": 0,
            "values": rows,
            "failures": [],
            "minimum_volume": volume,
            "maximum_volume": volume,
            "spread": 0.0,
            "within_tolerance": True,
        },
        "target_facets_echo": target_facets,
    }


def test_real_sixteen_cell_passes_reference_free_grade() -> None:
    grade = checker.grade_response_document(_response(), _target())

    assert grade["strict_pass"] is True
    assert grade["classification"] == "strict_h4_combinatorics_pass"
    assert all(grade["gates"].values())
    assert grade["measurements"]["candidate_facet_count"] == 16
    assert grade["measurements"]["minimum_volume"] > 1.7714


def test_invalid_geometry_is_not_mislabeled_as_missing_incidence_backend() -> None:
    vertices = _sixteen_cell_vertices()
    vertices[1] = vertices[0]

    grade = checker.grade_response_document(_response(vertices), _target())

    assert grade["strict_pass"] is False
    assert grade["classification"] == "failed_h4_combinatorics_audit"
    assert grade["gates"]["strict_ideal_geometry"] is False


def test_response_row_permutation_is_accepted_as_unlabeled_incidence() -> None:
    permutation = np.array([6, 1, 4, 3, 0, 7, 2, 5])
    grade = checker.grade_response_document(
        _response(_sixteen_cell_vertices()[permutation]), _target()
    )

    assert grade["strict_pass"] is True
    assert grade["gates"]["combinatorial_type_matches"] is True


def test_incidence_is_recomputed_from_facets_not_audit_summary() -> None:
    wrong_facets = _sixteen_cell_facets()
    wrong_facets[-1] = [0, 1, 2, 4]
    wrong_facets = sorted(wrong_facets)

    def audit(*args: Any, **kwargs: Any) -> dict[str, object]:
        return _fake_audit(*args, candidate_facets=wrong_facets, **kwargs)

    grade = checker.grade_response_document(
        _response(), _target(), audit_function=audit
    )

    assert grade["strict_pass"] is False
    assert grade["gates"]["strict_ideal_geometry"] is True
    assert grade["gates"]["combinatorial_type_matches"] is False


def test_pole_simplex_counts_and_volume_rows_are_recomputed() -> None:
    def audit(*args: Any, **kwargs: Any) -> dict[str, object]:
        result = _fake_audit(*args, **kwargs)
        result["pole_volume_audit"]["values"][3]["simplex_count"] += 1
        result["pole_volume_audit"]["minimum_volume"] = 100.0
        return result

    grade = checker.grade_response_document(
        _response(), _target(), audit_function=audit
    )

    assert grade["strict_pass"] is False
    assert grade["gates"]["pole_facet_simplex_counts"] is False
    assert grade["gates"]["pole_volume_stable"] is False
    assert grade["measurements"]["minimum_volume"] == pytest.approx(1.8)


def test_same_count_wrong_pole_facet_set_cannot_be_spoofed() -> None:
    def audit(*args: Any, **kwargs: Any) -> dict[str, object]:
        result = _fake_audit(*args, **kwargs)
        consistency = result["pole_facet_consistency"]
        consistency["all_match"] = False
        consistency["values"][2]["facets_match"] = False
        consistency["values"][2]["matches"] = False
        consistency["values"][2]["missing_facets"] = [[0, 1, 2, 3]]
        consistency["values"][2]["extra_facets"] = [[0, 1, 2, 4]]
        return result

    grade = checker.grade_response_document(
        _response(), _target(), audit_function=audit
    )

    assert grade["strict_pass"] is False
    assert grade["gates"]["pole_facet_simplex_counts"] is True
    assert grade["gates"]["pole_facet_consistency"] is False


def test_even_tiny_pole_summary_drift_is_rejected() -> None:
    def audit(*args: Any, **kwargs: Any) -> dict[str, object]:
        result = _fake_audit(*args, **kwargs)
        result["pole_volume_audit"]["minimum_volume"] = 1.8 + 1.0e-13
        return result

    grade = checker.grade_response_document(
        _response(), _target(), audit_function=audit
    )

    assert grade["strict_pass"] is False
    assert grade["gates"]["pole_volume_stable"] is False


def test_conservative_minimum_volume_controls_floor() -> None:
    def audit(*args: Any, **kwargs: Any) -> dict[str, object]:
        result = _fake_audit(*args, volume=1.8, **kwargs)
        result["pole_volume_audit"]["values"][5]["volume"] = 1.69
        result["pole_volume_audit"]["minimum_volume"] = 1.69
        result["pole_volume_audit"]["spread"] = 0.11
        result["pole_volume_audit"]["within_tolerance"] = False
        return result

    grade = checker.grade_response_document(
        _response(), _target(), audit_function=audit
    )

    assert grade["strict_pass"] is False
    assert grade["gates"]["minimum_volume"] is False
    assert grade["measurements"]["minimum_volume"] == pytest.approx(1.69)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda value: value.update(extra=True), "exactly"),
        (lambda value: value.update(dimension=3), "dimension"),
        (
            lambda value: value.update(facet_equivalence="labeled_facets"),
            "facet_equivalence",
        ),
        (
            lambda value: value["facets"].reverse(),
            "lexicographically sorted",
        ),
        (
            lambda value: value["facets"].append(value["facets"][0]),
            "duplicate",
        ),
    ],
)
def test_target_contract_is_strict(mutate: Any, message: str) -> None:
    target = _target()
    mutate(target)

    with pytest.raises(checker.CheckerInputError, match=message):
        checker.parse_target_document(target)


def test_target_contract_rejects_huge_float_conversion() -> None:
    target = _target()
    target["minimum_volume"] = 10**400

    with pytest.raises(checker.CheckerInputError, match="float64 range"):
        checker.parse_target_document(target)


def test_target_contract_rejects_negative_volume_floor() -> None:
    target = _target()
    target["minimum_volume"] = -0.1

    with pytest.raises(checker.CheckerInputError, match="nonnegative"):
        checker.parse_target_document(target)


@pytest.mark.parametrize(
    "response, message",
    [
        ({"vertices": [[1.0, 0.0, 0.0, 0.0]] * 8, "claim": 1}, "exactly"),
        ({"vertices": [[1.0, 0.0, 0.0, 0.0]]}, "exactly 8 rows"),
        ({"vertices": [[True, 0.0, 0.0, 0.0]] * 8}, "ordinary JSON number"),
        ({"vertices": [[10**400, 0.0, 0.0, 0.0]] * 8}, "float64 range"),
    ],
)
def test_response_shape_and_number_contract_is_strict(
    response: dict[str, object], message: str
) -> None:
    with pytest.raises(checker.CheckerInputError, match=message):
        checker.grade_response_document(response, _target())


def test_strict_loader_rejects_duplicate_keys_and_nonfinite(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"vertices": [], "vertices": []}', encoding="utf-8")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"vertices": [[NaN, 0, 0, 0]]}', encoding="utf-8")

    with pytest.raises(checker.CheckerInputError, match="duplicate"):
        checker.load_json_strict(duplicate)
    with pytest.raises(checker.CheckerInputError, match="non-standard"):
        checker.load_json_strict(nonfinite)


def test_atomic_grade_output_refuses_clobber(tmp_path: Path) -> None:
    target = tmp_path / "grade.json"
    payload = checker.strict_json_bytes({"strict_pass": True})
    checker.write_atomic(target, payload)

    with pytest.raises(checker.CheckerInputError, match="already exists"):
        checker.write_atomic(target, payload)

    checker.write_atomic(
        target, checker.strict_json_bytes({"strict_pass": False}), overwrite=True
    )
    assert json.loads(target.read_text(encoding="utf-8"))["strict_pass"] is False


def test_cli_converts_huge_json_integer_to_structured_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    response = tmp_path / "huge.json"
    response.write_text(
        json.dumps({"vertices": [[10**400, 0, 0, 0]] * 8}), encoding="utf-8"
    )
    target = tmp_path / "target.json"
    target.write_text(json.dumps(_target()), encoding="utf-8")

    exit_code = checker.main([str(response), "--target", str(target)])

    grade = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert grade["strict_pass"] is False
    assert grade["classification"] == "invalid_response_or_contract"
    assert "float64 range" in grade["error"]["message"]


def test_response_and_contract_hashes_use_one_byte_snapshot(tmp_path: Path) -> None:
    response_path = tmp_path / "response.json"
    response_bytes = json.dumps(_response(), separators=(",", ":")).encode()
    response_path.write_bytes(response_bytes)
    target_path = tmp_path / "target.json"
    target_bytes = json.dumps(_target(), separators=(",", ":")).encode()
    target_path.write_bytes(target_bytes)

    def replacing_audit(*args: Any, **kwargs: Any) -> dict[str, object]:
        response_path.write_text('{"vertices":[]}', encoding="utf-8")
        target_path.write_text("{}", encoding="utf-8")
        return _fake_audit(*args, **kwargs)

    grade = checker.grade_response(
        response_path, target_path, audit_function=replacing_audit
    )

    assert grade["response_sha256"] == hashlib.sha256(response_bytes).hexdigest()
    assert grade["contract_sha256"] == hashlib.sha256(target_bytes).hexdigest()
    assert (
        grade["response_sha256"]
        != hashlib.sha256(response_path.read_bytes()).hexdigest()
    )
    assert (
        grade["contract_sha256"] != hashlib.sha256(target_path.read_bytes()).hexdigest()
    )


def test_existing_output_is_rejected_before_loading_or_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "grade.json"
    output.write_text("preserve", encoding="utf-8")

    def unexpected_grade(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("inputs and audit should not be touched")

    monkeypatch.setattr(checker, "grade_response", unexpected_grade)
    exit_code = checker.main(
        [
            str(tmp_path / "missing-response.json"),
            "--target",
            str(tmp_path / "missing-target.json"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 2
    assert output.read_text(encoding="utf-8") == "preserve"
    assert "already exists" in capsys.readouterr().err


def test_overwrite_requires_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = checker.main(
        [
            str(tmp_path / "unused-response.json"),
            "--target",
            str(tmp_path / "unused-target.json"),
            "--overwrite",
        ]
    )

    grade = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert grade["classification"] == "invalid_response_or_contract"
    assert "requires --output" in grade["error"]["message"]


def test_publish_oserror_is_controlled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        checker,
        "grade_response",
        lambda *_args, **_kwargs: {"strict_pass": True},
    )

    def failed_publish(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated publish failure")

    monkeypatch.setattr(checker, "write_atomic", failed_publish)
    exit_code = checker.main(
        [
            str(tmp_path / "response.json"),
            "--target",
            str(tmp_path / "target.json"),
            "--output",
            str(tmp_path / "grade.json"),
        ]
    )

    assert exit_code == 2
    assert "simulated publish failure" in capsys.readouterr().err
