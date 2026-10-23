"""Invariant H4 symmetry-response checker tests."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "check_h4_symmetric_response.py"
SPEC = importlib.util.spec_from_file_location(
    "check_h4_symmetric_response", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def _sixteen_cell_response() -> dict[str, object]:
    vertices = np.vstack((np.eye(4), -np.eye(4))).tolist()
    return {"vertices": vertices}


def test_real_sixteen_cell_passes_invariant_grade() -> None:
    grade = checker.grade_response_document(
        _sixteen_cell_response(),
        n_vertices=8,
        minimum_volume=1.77,
        geometric_group_order=384,
        orientation_preserving_count=192,
    )

    assert grade["strict_pass"] is True
    assert grade["classification"] == "strict_h4_symmetric_pass"
    assert all(grade["gates"].values())
    assert grade["measurements"]["facet_count"] == 16
    assert grade["measurements"]["minimum_volume"] > 1.7714


def test_volume_floor_and_exact_full_group_are_independent_gates() -> None:
    volume_failure = checker.grade_response_document(
        _sixteen_cell_response(),
        n_vertices=8,
        minimum_volume=2.0,
        geometric_group_order=384,
    )
    group_failure = checker.grade_response_document(
        _sixteen_cell_response(),
        n_vertices=8,
        minimum_volume=1.7,
        geometric_group_order=192,
    )

    assert volume_failure["strict_pass"] is False
    assert volume_failure["gates"]["minimum_volume"] is False
    assert volume_failure["gates"]["full_group_certified"] is True
    assert group_failure["strict_pass"] is False
    assert group_failure["gates"]["exact_geometric_group_order"] is False


@pytest.mark.parametrize(
    "response, message",
    [
        ({"vertices": [[1.0, 0.0, 0.0, 0.0]], "claim": 1}, "exactly the key"),
        ({"vertices": [[1.0, 0.0, 0.0, 0.0]]}, "exactly 8 rows"),
        (
            {"vertices": [[True, 0.0, 0.0, 0.0]] * 8},
            "ordinary JSON number",
        ),
        (
            {"vertices": [[10**400, 0.0, 0.0, 0.0]] * 8},
            "outside the float64 range",
        ),
    ],
)
def test_response_shape_and_number_contract_is_strict(
    response: dict[str, object], message: str
) -> None:
    with pytest.raises(checker.CheckerInputError, match=message):
        checker.grade_response_document(
            response,
            n_vertices=8,
            minimum_volume=1.7,
            geometric_group_order=384,
        )


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
        json.dumps({"vertices": [[10**400, 0, 0, 0]] * 5}), encoding="utf-8"
    )

    exit_code = checker.main(
        [
            str(response),
            "--n-vertices",
            "5",
            "--minimum-volume",
            "0",
            "--geometric-group-order",
            "1",
        ]
    )

    grade = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert grade["strict_pass"] is False
    assert grade["classification"] == "invalid_response_or_contract"
    assert "float64 range" in grade["error"]["message"]


def test_response_hash_and_parse_use_one_byte_snapshot(tmp_path: Path) -> None:
    response = tmp_path / "response.json"
    original = json.dumps(
        {"vertices": [[1.0, 0.0, 0.0, 0.0]] * 5}, separators=(",", ":")
    ).encode()
    response.write_bytes(original)

    def replacing_audit(*_args: object, **_kwargs: object) -> dict[str, object]:
        response.write_text('{"vertices":[]}', encoding="utf-8")
        return {
            "dimension": 4,
            "n_vertices": 5,
            "classification_basis": {},
            "symmetry": {},
            "pole_volume_audit": {},
        }

    grade = checker.grade_response(
        response,
        n_vertices=5,
        minimum_volume=0.0,
        geometric_group_order=1,
        audit_function=replacing_audit,
    )

    assert grade["response_sha256"] == hashlib.sha256(original).hexdigest()
    assert grade["response_sha256"] != hashlib.sha256(response.read_bytes()).hexdigest()


def test_symmetry_tolerances_must_be_strictly_positive() -> None:
    with pytest.raises(checker.CheckerInputError, match="must be positive"):
        checker.grade_response_document(
            _sixteen_cell_response(),
            n_vertices=8,
            minimum_volume=1.7,
            geometric_group_order=384,
            isometry_tolerance=0.0,
        )


def test_semantic_gates_do_not_trust_only_summary_booleans() -> None:
    def tampered_audit(*args: object, **kwargs: object) -> dict[str, object]:
        audit = checker.audit_h4_symmetric_candidate(*args, **kwargs)
        audit["symmetry"]["geometric_group_order"] = 384.0
        audit["pole_volume_audit"]["successful_count"] = 0
        return audit

    grade = checker.grade_response_document(
        _sixteen_cell_response(),
        n_vertices=8,
        minimum_volume=1.7,
        geometric_group_order=384,
        audit_function=tampered_audit,
    )

    assert grade["strict_pass"] is False
    assert grade["gates"]["exact_geometric_group_order"] is False
    assert grade["gates"]["all_poles_verified"] is False


def test_existing_output_is_rejected_before_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    response = tmp_path / "response.json"
    response.write_text(json.dumps(_sixteen_cell_response()), encoding="utf-8")
    output = tmp_path / "grade.json"
    output.write_text("preserve", encoding="utf-8")

    def unexpected_grade(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("audit should not run")

    monkeypatch.setattr(checker, "grade_response", unexpected_grade)
    exit_code = checker.main(
        [
            str(response),
            "--n-vertices",
            "8",
            "--minimum-volume",
            "1.7",
            "--geometric-group-order",
            "384",
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
    response = tmp_path / "unused.json"
    exit_code = checker.main(
        [
            str(response),
            "--n-vertices",
            "8",
            "--minimum-volume",
            "1.7",
            "--geometric-group-order",
            "384",
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
    response = tmp_path / "response.json"
    response.write_text(json.dumps(_sixteen_cell_response()), encoding="utf-8")

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
            str(response),
            "--n-vertices",
            "8",
            "--minimum-volume",
            "1.7",
            "--geometric-group-order",
            "384",
            "--output",
            str(tmp_path / "grade.json"),
        ]
    )

    assert exit_code == 2
    assert "simulated publish failure" in capsys.readouterr().err
