"""Strict JSON and independent-geometry checks for the zero-shear grader."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "check_zero_shear_response.py"
SPEC = importlib.util.spec_from_file_location("check_zero_shear_response", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


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


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return path


def _key(*, exists: bool = True, vertices: object = VERTICES) -> dict[str, object]:
    ground_truth: dict[str, object] = {
        "exists": exists,
        "oriented_faces": FACES,
    }
    if vertices is not None:
        ground_truth["vertices"] = vertices
    return {
        "instance_id": "synthetic-tetrahedron",
        "ground_truth": ground_truth,
        "normalization": {
            "handedness": "all non-anchor vertices have positive imaginary part"
        },
    }


def _paths(
    tmp_path: Path,
    response: object,
    *,
    key: object | None = None,
) -> tuple[Path, Path]:
    key_path = _write_json(tmp_path / "answer-key.json", _key() if key is None else key)
    response_path = _write_json(tmp_path / "response.json", response)
    return key_path, response_path


def test_valid_labeled_normalized_tetrahedron_passes(tmp_path: Path):
    key_path, response_path = _paths(
        tmp_path,
        {"exists": True, "vertices": VERTICES},
    )

    grade = checker.grade_response(key_path, response_path)

    assert grade["strict_pass"] is True
    assert grade["classification"] == "strict_realization_pass"
    assert grade["existence"] == {
        "expected": True,
        "reported": True,
        "matches": True,
        "basis_when_false": "private-answer-key-classification",
    }
    assert grade["target"]["distinguished_face"] == [0, 1, 2]
    assert grade["target"]["labels_exact_range"] is True
    assert grade["audit"]["strict_pass"] is True
    assert grade["audit"]["normalization"]["handedness_matches"] is True


@pytest.mark.parametrize(
    "vertices",
    [
        [None, [0.01, 0.0], [1.0, 0.0], VERTICES[3]],
        [None, [0.0, 0.0], [1.01, 0.0], VERTICES[3]],
    ],
)
def test_wrong_anchor_fails_normalization(tmp_path: Path, vertices: object):
    key_path, response_path = _paths(
        tmp_path,
        {"exists": True, "vertices": vertices},
    )

    grade = checker.grade_response(key_path, response_path)

    assert grade["strict_pass"] is False
    assert grade["classification"] == "failed_strict_realization_audit"
    assert grade["audit"]["normalization"]["coordinate_anchors_match"] is False


def test_wrong_handedness_is_not_accepted_as_the_same_realization(tmp_path: Path):
    reflected = [row[:] if isinstance(row, list) else row for row in VERTICES]
    reflected[3][1] *= -1.0
    key_path, response_path = _paths(
        tmp_path,
        {"exists": True, "vertices": reflected},
    )

    grade = checker.grade_response(key_path, response_path)

    assert grade["strict_pass"] is False
    assert grade["classification"] == "failed_strict_realization_audit"
    assert grade["audit"]["normalization"]["handedness_matches"] is False


def test_target_labels_must_be_exact_contiguous_range(tmp_path: Path):
    bad_key = _key()
    bad_key["ground_truth"]["oriented_faces"] = [
        [4 if vertex == 3 else vertex for vertex in face] for face in FACES
    ]
    key_path, response_path = _paths(
        tmp_path,
        {"exists": True, "vertices": VERTICES},
        key=bad_key,
    )

    with pytest.raises(checker.CheckerInputError, match="without gaps"):
        checker.grade_response(key_path, response_path)


def test_private_key_requires_normalization_metadata(tmp_path: Path):
    bad_key = _key()
    del bad_key["normalization"]
    key_path, response_path = _paths(
        tmp_path,
        {"exists": True, "vertices": VERTICES},
        key=bad_key,
    )

    with pytest.raises(checker.CheckerInputError, match="normalization object"):
        checker.grade_response(key_path, response_path)


def test_target_orientation_must_already_be_coherent(tmp_path: Path):
    bad_key = _key()
    bad_faces = [face[:] for face in FACES]
    bad_faces[1] = [0, 1, 3]
    bad_key["ground_truth"]["oriented_faces"] = bad_faces
    key_path, response_path = _paths(
        tmp_path,
        {"exists": True, "vertices": VERTICES},
        key=bad_key,
    )

    with pytest.raises(checker.CheckerInputError, match="coherently oriented"):
        checker.grade_response(key_path, response_path)


@pytest.mark.parametrize(
    "response_text",
    [
        '{"exists": true, "exists": false}',
        '{"exists": true, "vertices": [null, [0, 0], [1, 0], [NaN, 1]]}',
        "not JSON",
    ],
)
def test_main_reports_malformed_or_nonstandard_json_as_strict_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    response_text: str,
):
    key_path = _write_json(tmp_path / "answer-key.json", _key())
    response_path = tmp_path / "response.json"
    response_path.write_text(response_text, encoding="utf-8")

    exit_code = checker.main([str(key_path), str(response_path)])
    captured = capsys.readouterr()
    grade = json.loads(captured.out)

    assert exit_code == 1
    assert captured.err == ""
    assert grade["classification"] == "invalid_input"
    assert grade["strict_pass"] is False


def test_response_contract_rejects_extra_keys(tmp_path: Path):
    key_path, response_path = _paths(
        tmp_path,
        {"exists": False, "explanation": "none"},
    )

    with pytest.raises(checker.CheckerInputError, match="exactly exists"):
        checker.grade_response(key_path, response_path)


@pytest.mark.parametrize(
    ("expected", "reported", "passes", "classification"),
    [
        (False, False, True, "private_key_nonexistence_match"),
        (True, False, False, "incorrect_existence_claim"),
    ],
)
def test_false_answer_uses_only_the_private_existence_bit(
    tmp_path: Path,
    expected: bool,
    reported: bool,
    passes: bool,
    classification: str,
):
    key_path, response_path = _paths(
        tmp_path,
        {"exists": reported},
        key=_key(exists=expected, vertices=None),
    )

    grade = checker.grade_response(key_path, response_path)

    assert grade["strict_pass"] is passes
    assert grade["classification"] == classification
    assert grade["audit"] is None


def test_private_reference_is_diagnostic_unless_tolerance_is_requested(
    tmp_path: Path,
):
    shifted_reference = [row[:] if isinstance(row, list) else row for row in VERTICES]
    shifted_reference[3][0] += 0.05
    key_path, response_path = _paths(
        tmp_path,
        {"exists": True, "vertices": VERTICES},
        key=_key(vertices=shifted_reference),
    )

    diagnostic_grade = checker.grade_response(key_path, response_path)
    gated_grade = checker.grade_response(
        key_path,
        response_path,
        reference_tolerance=0.0,
    )

    diagnostic = diagnostic_grade["private_reference"]
    assert diagnostic_grade["strict_pass"] is True
    assert diagnostic["available"] is True
    assert diagnostic["max_chordal"] > 0.0
    assert diagnostic["is_acceptance_gate"] is False
    assert diagnostic["within_tolerance"] is None
    assert gated_grade["strict_pass"] is False
    assert gated_grade["classification"] == "private_reference_mismatch"
    assert gated_grade["private_reference"]["within_tolerance"] is False


def test_atomic_output_is_complete_and_stdout_stays_empty(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    key_path, response_path = _paths(
        tmp_path,
        {"exists": True, "vertices": VERTICES},
    )
    output_path = tmp_path / "reports" / "grade.json"

    exit_code = checker.main(
        [str(key_path), str(response_path), "--output", str(output_path)]
    )
    captured = capsys.readouterr()
    grade = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert captured.out == ""
    assert captured.err == ""
    assert grade["strict_pass"] is True
    assert output_path.read_bytes().endswith(b"\n")
    assert list(output_path.parent.glob(".grade.json.*.tmp")) == []


def test_output_refuses_to_clobber_without_explicit_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    key_path, response_path = _paths(
        tmp_path,
        {"exists": True, "vertices": VERTICES},
    )
    output_path = tmp_path / "grade.json"
    output_path.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(SystemExit) as caught:
        checker.main([str(key_path), str(response_path), "--output", str(output_path)])

    captured = capsys.readouterr()
    assert caught.value.code == 2
    assert captured.out == ""
    assert "refusing to clobber" in captured.err
    assert output_path.read_text(encoding="utf-8") == "sentinel\n"


def test_explicit_overwrite_atomically_replaces_existing_grade(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    key_path, response_path = _paths(
        tmp_path,
        {"exists": True, "vertices": VERTICES},
    )
    output_path = tmp_path / "grade.json"
    output_path.write_text("sentinel\n", encoding="utf-8")

    exit_code = checker.main(
        [
            str(key_path),
            str(response_path),
            "--output",
            str(output_path),
            "--overwrite",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(output_path.read_text(encoding="utf-8"))["strict_pass"] is True
    assert list(tmp_path.glob(".grade.json.*.tmp")) == []
