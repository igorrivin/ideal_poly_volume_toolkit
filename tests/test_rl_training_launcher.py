"""Artifact/provenance contracts for the polytope controller launcher."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import runpy
from types import SimpleNamespace
from typing import Any

import pytest


_TRAINER = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "train_polytope_controller.py")
)
_json_sha256 = _TRAINER["_json_sha256"]
_parse_args = _TRAINER["_parse_args"]
_promote_artifact_pair = _TRAINER["_promote_artifact_pair"]
_resolved_paths = _TRAINER["_resolved_paths"]
main = _TRAINER["main"]


class _FakeSpace:
    def __init__(self, label: str):
        self.label = label

    def __repr__(self) -> str:
        return self.label


class _FakeEnv:
    op_names = ["random_restart", "cma_true", "sample"]
    action_space = _FakeSpace("FakeActionSpace(3)")
    observation_space = _FakeSpace("FakeObservationSpace(8)")

    def close(self) -> None:
        pass


def test_default_artifact_name_uses_current_protocol_not_final() -> None:
    args = _parse_args(["--d", "3"])
    model_path, config_path, lock_path = _resolved_paths(args)

    assert model_path.name == "ppo_d3_controller-v2-seed42.zip"
    assert config_path.name == "ppo_d3_controller-v2-seed42_config.json"
    assert lock_path.name == "ppo_d3_controller-v2-seed42.training-lock.json"
    assert "final" not in model_path.name


@pytest.mark.parametrize(
    "arguments",
    [
        ["--d", "3", "--steps", "0"],
        ["--d", "4", "--n-low", "4"],
        ["--d", "3", "--n-low", "8", "--n-high", "7"],
        ["--d", "3", "--n-steps", "1", "--batch-size", "1"],
        ["--d", "3", "--n-steps", "10", "--batch-size", "6"],
        ["--d", "3", "--learning-rate", "nan"],
        ["--d", "3", "--run-name", "../../escape"],
        ["--d", "3", "--output", "model.bin"],
    ],
)
def test_invalid_training_arguments_fail_before_training(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        _parse_args(arguments)


def _install_fake_training(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail: bool = False,
) -> list[dict[str, Any]]:
    import ideal_poly_volume_toolkit.rl.polytope as polytope_module
    import ideal_poly_volume_toolkit.rl.training as training_module

    calls: list[dict[str, Any]] = []

    def fake_make_polytope_env(**kwargs: Any) -> _FakeEnv:
        calls.append({"kind": "env", **kwargs})
        return _FakeEnv()

    def fake_train_controller(factory: Any, **kwargs: Any) -> Any:
        calls.append({"kind": "train", **kwargs})
        if fail:
            raise RuntimeError("deliberate training failure")
        env = factory()
        env.close()
        Path(kwargs["save_path"]).write_bytes(b"fake-sb3-model")
        return SimpleNamespace(num_timesteps=8)

    monkeypatch.setattr(polytope_module, "make_polytope_env", fake_make_polytope_env)
    monkeypatch.setattr(training_module, "train_controller", fake_train_controller)
    return calls


def test_launcher_writes_truthful_atomic_model_and_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_training(monkeypatch)
    for variable in _TRAINER["THREAD_ENV_VARS"]:
        monkeypatch.setenv(variable, "9")
    model_path = tmp_path / "controller.zip"

    result = main(
        [
            "--d",
            "3",
            "--steps",
            "4",
            "--horizon",
            "2",
            "--n-steps",
            "4",
            "--batch-size",
            "2",
            "--threads",
            "1",
            "--output",
            str(model_path),
        ]
    )

    config_path = tmp_path / "controller_config.json"
    lock_path = tmp_path / "controller.training-lock.json"
    assert result == 0
    assert model_path.read_bytes() == b"fake-sb3-model"
    assert config_path.is_file()
    assert not lock_path.exists()

    config = json.loads(config_path.read_text())
    checksum = config.pop("config_payload_sha256")
    assert checksum == _json_sha256(config)
    assert config["schema_version"] == "ideal-poly-rl-training/1"
    assert config["controller_protocol_version"] == "ideal-poly-rl-controller/2"
    assert config["status"] == "complete"
    assert config["training"]["requested_timesteps"] == 4
    assert config["training"]["actual_timesteps"] == 8
    assert config["training"]["n_steps"] == 4
    assert config["training"]["batch_size"] == 2
    assert config["task"]["requested_N_range"] == [5, 25]
    assert config["task"]["effective_N_range"] == [5, 15]
    assert config["environment"]["thread_settings"] == {
        variable: "1" for variable in _TRAINER["THREAD_ENV_VARS"]
    }
    assert config["environment_config"]["action_ordering"] == _FakeEnv.op_names
    assert config["repository"]["commit"]
    assert config["repository"]["commit"] != "run git rev-parse HEAD to recover"
    assert len(config["repository"]["diff_sha256"]) == 64
    assert len(config["repository"]["source_sha256"]) == 64
    assert config["artifacts"]["model_sha256"] == hashlib.sha256(
        b"fake-sb3-model"
    ).hexdigest()
    assert config["artifacts"]["model_bytes"] == len(b"fake-sb3-model")

    train_call = next(call for call in calls if call["kind"] == "train")
    assert train_call["total_timesteps"] == 4
    assert train_call["n_steps"] == 4
    assert train_call["batch_size"] == 2

    train_calls_before = sum(call["kind"] == "train" for call in calls)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(["--d", "3", "--output", str(model_path)])
    assert sum(call["kind"] == "train" for call in calls) == train_calls_before


def test_training_failure_leaves_no_partial_artifact_or_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_training(monkeypatch, fail=True)
    model_path = tmp_path / "failed.zip"

    with pytest.raises(RuntimeError, match="deliberate training failure"):
        main(
            [
                "--d",
                "3",
                "--steps",
                "1",
                "--n-steps",
                "2",
                "--batch-size",
                "2",
                "--output",
                str(model_path),
            ]
        )

    assert not model_path.exists()
    assert not (tmp_path / "failed_config.json").exists()
    assert not (tmp_path / "failed.training-lock.json").exists()


def test_overwrite_pair_promotion_restores_both_old_artifacts_on_sidecar_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_target = tmp_path / "controller.zip"
    config_target = tmp_path / "controller_config.json"
    model_target.write_bytes(b"old-model")
    config_target.write_bytes(b"old-config")
    staging = tmp_path / "staging"
    staging.mkdir()
    staged_model = staging / model_target.name
    staged_config = staging / config_target.name
    staged_model.write_bytes(b"new-model")
    staged_config.write_bytes(b"new-config")

    real_replace = os.replace

    def fail_sidecar_promotion(source: Any, target: Any) -> None:
        if Path(source) == staged_config and Path(target) == config_target:
            raise OSError("deliberate sidecar promotion failure")
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_sidecar_promotion)

    with pytest.raises(OSError, match="deliberate sidecar promotion failure"):
        _promote_artifact_pair(
            staged_model,
            staged_config,
            model_target,
            config_target,
            overwrite=True,
        )

    assert model_target.read_bytes() == b"old-model"
    assert config_target.read_bytes() == b"old-config"
    assert not list(staging.glob(".rollback.*"))


def test_no_clobber_pair_promotion_removes_first_artifact_if_second_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_target = tmp_path / "controller.zip"
    config_target = tmp_path / "controller_config.json"
    staging = tmp_path / "staging"
    staging.mkdir()
    staged_model = staging / model_target.name
    staged_config = staging / config_target.name
    staged_model.write_bytes(b"new-model")
    staged_config.write_bytes(b"new-config")

    real_link = os.link

    def fail_sidecar_link(source: Any, target: Any, **kwargs: Any) -> None:
        if Path(source) == staged_config and Path(target) == config_target:
            raise OSError("deliberate sidecar link failure")
        real_link(source, target, **kwargs)

    monkeypatch.setattr(os, "link", fail_sidecar_link)

    with pytest.raises(OSError, match="deliberate sidecar link failure"):
        _promote_artifact_pair(
            staged_model,
            staged_config,
            model_target,
            config_target,
            overwrite=False,
        )

    assert not model_target.exists()
    assert not config_target.exists()


def test_dry_run_writes_nothing_and_never_calls_trainer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _install_fake_training(monkeypatch)
    model_path = tmp_path / "dry.zip"

    assert main(["--d", "3", "--output", str(model_path), "--dry-run"]) == 0
    assert not model_path.exists()
    assert not (tmp_path / "dry_config.json").exists()
    assert not any(call["kind"] == "train" for call in calls)
    assert '"status": "dry_run"' in capsys.readouterr().out
