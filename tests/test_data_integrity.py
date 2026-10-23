import json
from pathlib import Path

import pytest

from examples import analyze_maximal_dihedral_angles as angle_analysis
from examples import analyze_spanning_tree_distribution as spanning_analysis
from scripts.check_data_integrity import (
    IntegrityError,
    RESULT_MANIFEST_PATH,
    RESULT_MANIFEST_SCHEMA_PATH,
    check_lfs_attributes,
    check_ordinary_git_attributes,
    check_required_inputs,
    check_result_manifest,
    is_lfs_pointer,
    list_lfs_paths,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _copy_result_manifest_fixture(tmp_path):
    fixture_paths = (
        RESULT_MANIFEST_SCHEMA_PATH,
        RESULT_MANIFEST_PATH,
        "results/data/optimal_d4_symmetry_analysis.json",
        "results/data/optimal_d5_d6_symmetry_analysis.json",
    )
    for relative_path in fixture_paths:
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO_ROOT / relative_path).read_bytes())
    return tmp_path / RESULT_MANIFEST_PATH


def test_existing_lfs_paths_remain_attribute_filtered():
    paths = list_lfs_paths(REPO_ROOT)

    assert len(paths) == 407
    check_lfs_attributes(REPO_ROOT, paths)


def test_canonical_inputs_are_ordinary_git_with_expected_hashes():
    check_ordinary_git_attributes(REPO_ROOT)
    check_required_inputs(REPO_ROOT)


def test_result_manifest_validates_schema_paths_and_hashes():
    assert check_result_manifest(REPO_ROOT) == 2


def test_legacy_arithmetic_collections_are_explicitly_superseded():
    manifest = json.loads(
        (REPO_ROOT / RESULT_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    statuses = {
        collection["path_glob"]: collection["status"]
        for collection in manifest["collections"]
    }

    assert statuses["results/data/arithmeticity_benchmark.json"] == "superseded"
    assert statuses["results/data/20vertex_arithmetic_analysis.json"] == (
        "superseded"
    )
    assert statuses["results/data/llm_challenges_arithmetic/**"] == "superseded"


def test_result_manifest_rejects_tampered_artifact_hash(tmp_path):
    manifest_path = _copy_result_manifest_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(IntegrityError, match="SHA-256 mismatch"):
        check_result_manifest(tmp_path)


def test_result_manifest_rejects_weakened_canonical_classification(tmp_path):
    manifest_path = _copy_result_manifest_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["status"] = "historical"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(IntegrityError, match="role=canonical_input"):
        check_result_manifest(tmp_path)


def test_lfs_pointer_detection():
    assert is_lfs_pointer(
        b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 3\n"
    )
    assert not is_lfs_pointer(b'{"ordinary": "json"}\n')


def test_angle_analysis_prefers_canonical_results(monkeypatch, tmp_path):
    canonical = tmp_path / "canonical"
    legacy = tmp_path / "legacy"
    canonical.mkdir()
    legacy.mkdir()
    canonical_file = canonical / "4vertex_optimization_cli_1.json"
    legacy_file = legacy / "4vertex_optimization_legacy_1.json"
    canonical_file.write_text("{}", encoding="utf-8")
    legacy_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(angle_analysis, "CANONICAL_RESULTS_DIRECTORY", canonical)
    monkeypatch.setattr(angle_analysis, "LEGACY_RESULTS_DIRECTORY", legacy)

    files, selected = angle_analysis.find_optimization_result_files()

    assert selected == canonical
    assert files == [canonical_file]


def test_angle_analysis_warns_before_legacy_fallback(monkeypatch, tmp_path):
    canonical = tmp_path / "canonical"
    legacy = tmp_path / "legacy"
    canonical.mkdir()
    legacy.mkdir()
    legacy_file = legacy / "4vertex_optimization_legacy_1.json"
    legacy_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(angle_analysis, "CANONICAL_RESULTS_DIRECTORY", canonical)
    monkeypatch.setattr(angle_analysis, "LEGACY_RESULTS_DIRECTORY", legacy)

    with pytest.warns(RuntimeWarning, match="historical results"):
        files, selected = angle_analysis.find_optimization_result_files()

    assert selected == legacy
    assert files == [legacy_file]


def test_spanning_tree_analysis_prefers_corrected_data(monkeypatch, tmp_path):
    corrected = tmp_path / "corrected.json"
    legacy = tmp_path / "legacy.json"
    corrected.write_text("{}", encoding="utf-8")
    legacy.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(spanning_analysis, "CORRECTED_DATA_PATH", corrected)
    monkeypatch.setattr(spanning_analysis, "LEGACY_DATA_PATH", legacy)

    assert spanning_analysis.resolve_spanning_tree_data_path() == corrected


def test_spanning_tree_analysis_warns_before_legacy_fallback(
    monkeypatch,
    tmp_path,
):
    corrected = tmp_path / "corrected.json"
    legacy = tmp_path / "legacy.json"
    legacy.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(spanning_analysis, "CORRECTED_DATA_PATH", corrected)
    monkeypatch.setattr(spanning_analysis, "LEGACY_DATA_PATH", legacy)

    with pytest.warns(RuntimeWarning, match="historical"):
        selected = spanning_analysis.resolve_spanning_tree_data_path()

    assert selected == legacy
