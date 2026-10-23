#!/usr/bin/env python3
"""Train one provenance-bearing PPO polytope controller artifact.

The launcher writes a model plus ``*_config.json`` sidecar under an explicit
controller protocol name.  Both outputs are staged before atomic promotion,
are no-clobber by default, and are protected by a per-target training lock.

Example::

    python scripts/train_polytope_controller.py --d 3 --steps 500000

The default artifact is
``results/rl_models/ppo_d3_controller-v2-seed42.zip``.  Use ``--dry-run`` to
validate and print the resolved contract without training or writing files.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


REPO = Path(__file__).resolve().parent.parent
TRAINING_SCHEMA_VERSION = "ideal-poly-rl-training/1"
CONTROLLER_PROTOCOL_VERSION = "ideal-poly-rl-controller/2"
DEFAULT_RUN_NAME_TEMPLATE = "controller-v2-seed{seed}"
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
DEPENDENCY_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "torch",
    "gymnasium",
    "stable-baselines3",
    "cma",
)
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path.resolve())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seal_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-normalized config with a verifiable payload checksum."""
    sealed = json.loads(json.dumps(config, allow_nan=False))
    sealed.pop("config_payload_sha256", None)
    sealed["config_payload_sha256"] = _json_sha256(sealed)
    return sealed


def _write_json_handle(handle: Any, document: Mapping[str, Any]) -> None:
    json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())


def _exclusive_create_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            _write_json_handle(handle, document)
        os.link(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except OSError:
            pass


def _promote_artifact_pair(
    staged_model: Path,
    staged_config: Path,
    model_target: Path,
    config_target: Path,
    *,
    overwrite: bool,
) -> None:
    """Promote the staged model/config pair, rolling both back on failure.

    A filesystem cannot rename two paths in one syscall.  The training lock
    excludes cooperating writers; for overwrite mode, hard-linked snapshots
    of both prior artifacts provide rollback if either promotion fails.  In
    no-clobber mode, any newly linked first artifact is removed if linking the
    second one fails.
    """
    staged = (
        (staged_model, model_target),
        (staged_config, config_target),
    )
    for source, _ in staged:
        with source.open("rb") as handle:
            os.fsync(handle.fileno())

    backups: dict[Path, Path] = {}
    promoted: set[Path] = set()
    try:
        if overwrite:
            for _, target in staged:
                if target.exists():
                    backup = staged_model.parent / f".rollback.{target.name}"
                    os.link(target, backup, follow_symlinks=False)
                    backups[target] = backup

        for source, target in staged:
            if overwrite:
                os.replace(source, target)
            else:
                os.link(source, target)
            promoted.add(target)
    except BaseException as promotion_error:
        rollback_errors: list[str] = []
        for _, target in reversed(staged):
            backup = backups.get(target)
            try:
                if backup is not None:
                    os.replace(backup, target)
                elif target in promoted:
                    target.unlink()
            except OSError as rollback_error:
                rollback_errors.append(
                    f"{target}: {type(rollback_error).__name__}: {rollback_error}"
                )
        if rollback_errors:
            details = "; ".join(rollback_errors)
            raise RuntimeError(
                "controller pair promotion failed and rollback was incomplete: "
                f"{details}"
            ) from promotion_error
        raise
    finally:
        for backup in backups.values():
            try:
                backup.unlink()
            except OSError:
                pass


def _git(*args: str) -> tuple[Optional[bytes], Optional[str]]:
    command = [
        "git",
        "-c",
        "filter.lfs.process=",
        "-c",
        "filter.lfs.required=false",
        "-c",
        "filter.lfs.clean=cat",
        "-c",
        "filter.lfs.smudge=cat",
        *args,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        return None, str(exc)
    if completed.returncode != 0:
        return None, completed.stderr.decode("utf-8", errors="replace").strip()
    return completed.stdout, None


def _repository_provenance() -> dict[str, Any]:
    commit_bytes, commit_error = _git("rev-parse", "HEAD")
    status_bytes, status_error = _git(
        "status", "--porcelain=v1", "-z", "--untracked-files=all"
    )
    diff_bytes, diff_error = _git("diff", "--binary", "HEAD", "--")
    staged_bytes, staged_error = _git("diff", "--binary", "--cached", "--")
    untracked_bytes, untracked_error = _git(
        "ls-files", "--others", "--exclude-standard", "-z"
    )
    errors = [
        error
        for error in (
            commit_error,
            status_error,
            diff_error,
            staged_error,
            untracked_error,
        )
        if error
    ]

    diff_digest = hashlib.sha256()
    for label, content in (
        (b"status\0", status_bytes),
        (b"unstaged\0", diff_bytes),
        (b"staged\0", staged_bytes),
    ):
        diff_digest.update(label)
        diff_digest.update(content or b"")
    if untracked_bytes is not None:
        for raw_path in sorted(filter(None, untracked_bytes.split(b"\0"))):
            diff_digest.update(b"untracked\0" + raw_path + b"\0")
            path = REPO / os.fsdecode(raw_path)
            if path.is_file():
                try:
                    diff_digest.update(path.read_bytes())
                except OSError as exc:
                    errors.append(f"could not hash {path}: {exc}")

    source_files = [Path(__file__).resolve(), REPO / "pyproject.toml"]
    source_files.extend(
        sorted((REPO / "ideal_poly_volume_toolkit" / "rl").rglob("*.py"))
    )
    source_digest = hashlib.sha256()
    hashed_sources: list[str] = []
    for path in source_files:
        if not path.is_file():
            continue
        display = _display_path(path)
        source_digest.update(display.encode() + b"\0")
        source_digest.update(path.read_bytes())
        hashed_sources.append(display)

    return {
        "commit": (
            commit_bytes.decode().strip() if commit_bytes is not None else None
        ),
        "dirty": bool(status_bytes) if status_bytes is not None else None,
        "diff_sha256": diff_digest.hexdigest(),
        "source_sha256": source_digest.hexdigest(),
        "launcher_sha256": _sha256_file(Path(__file__).resolve()),
        "source_files": hashed_sources,
        "git_errors": errors,
    }


def _dependency_versions() -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {}
    for distribution in DEPENDENCY_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _resolved_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.output is not None:
        model_path = args.output.expanduser().resolve()
    else:
        model_path = (
            REPO
            / "results"
            / "rl_models"
            / f"ppo_d{args.d}_{args.run_name}.zip"
        )
    if model_path.suffix.lower() != ".zip":
        raise ValueError("controller output must have a .zip suffix")
    config_path = model_path.with_name(f"{model_path.stem}_config.json")
    lock_path = model_path.with_name(f"{model_path.stem}.training-lock.json")
    return model_path, config_path, lock_path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, required=True, choices=(3, 4, 5, 6))
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument(
        "--n-low",
        type=int,
        default=None,
        help="low end of N range; default=max(5, d+1)",
    )
    parser.add_argument("--n-high", type=int, default=25)
    identity = parser.add_mutually_exclusive_group()
    identity.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="artifact identity (default controller-v2-seed<seed>)",
    )
    identity.add_argument(
        "--tag",
        type=str,
        default=None,
        help="deprecated alias for --run-name",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="explicit .zip model path; config is written alongside it",
    )
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--eval-cost-weight", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace existing model/config after successful training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the resolved contract without writing/training",
    )
    args = parser.parse_args(argv)
    args.invocation_argv = (
        list(sys.argv)
        if argv is None
        else [str(Path(__file__).resolve()), *map(str, argv)]
    )

    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.seed < 0 or args.seed >= 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if args.horizon < 1:
        parser.error("--horizon must be at least 1")
    if args.n_low is None:
        args.n_low = max(5, args.d + 1)
    if args.n_low < args.d + 1:
        parser.error(f"--n-low must be at least d+1 ({args.d + 1})")
    if args.n_high < args.n_low:
        parser.error("--n-high must be greater than or equal to --n-low")
    if args.n_steps < 2:
        parser.error("--n-steps must be at least 2 for PPO advantage normalization")
    if args.batch_size < 2 or args.batch_size > args.n_steps:
        parser.error("--batch-size must be in [2, n-steps]")
    if args.n_steps % args.batch_size:
        parser.error("--batch-size must divide --n-steps")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        parser.error("--learning-rate must be finite and positive")
    if not math.isfinite(args.gamma) or not 0.0 < args.gamma <= 1.0:
        parser.error("--gamma must be in (0, 1]")
    if not math.isfinite(args.ent_coef) or args.ent_coef < 0.0:
        parser.error("--ent-coef must be finite and nonnegative")
    if not math.isfinite(args.eval_cost_weight) or args.eval_cost_weight < 0.0:
        parser.error("--eval-cost-weight must be finite and nonnegative")
    if args.threads < 1:
        parser.error("--threads must be at least 1")

    args.run_name = (
        args.run_name
        or args.tag
        or DEFAULT_RUN_NAME_TEMPLATE.format(seed=args.seed)
    )
    if not RUN_NAME_PATTERN.fullmatch(args.run_name):
        parser.error(
            "--run-name/--tag must be 1-80 safe filename characters "
            "(letters, digits, dot, underscore, hyphen)"
        )
    if args.output is not None and args.output.suffix.lower() != ".zip":
        parser.error("--output must have a .zip suffix")
    return args


def _base_config(
    args: argparse.Namespace,
    *,
    model_path: Path,
    config_path: Path,
    action_ordering: Sequence[str],
    action_space: str,
    observation_space: str,
) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "controller_protocol_version": CONTROLLER_PROTOCOL_VERSION,
        "status": "in_progress",
        "started_at": _utc_now(),
        "invocation": {
            "argv": list(args.invocation_argv),
            "cwd": str(Path.cwd().resolve()),
        },
        "repository": _repository_provenance(),
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "dependency_versions": _dependency_versions(),
            "thread_settings": {
                variable: os.environ.get(variable) for variable in THREAD_ENV_VARS
            },
        },
        "task": {
            "family": "ideal_polytope",
            "d_range": [args.d, args.d],
            "requested_N_range": [args.n_low, args.n_high],
            "effective_N_range": [
                args.n_low,
                max(args.n_low, min(args.n_high, 5 * args.d)),
            ],
        },
        "environment_config": {
            "factory": "ideal_poly_volume_toolkit.rl.polytope.make_polytope_env",
            "horizon": args.horizon,
            "eval_cost_weight": args.eval_cost_weight,
            "action_ordering": list(action_ordering),
            "action_space": action_space,
            "observation_space": observation_space,
        },
        "training": {
            "algorithm": "stable_baselines3.PPO",
            "policy": "MlpPolicy",
            "requested_timesteps": args.steps,
            "actual_timesteps": None,
            "seed": args.seed,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "ent_coef": args.ent_coef,
            "device": "cpu",
            "policy_kwargs": None,
            "tensorboard_log": None,
        },
        "artifacts": {
            "model_path": _display_path(model_path),
            "config_path": _display_path(config_path),
            "model_sha256": None,
            "model_bytes": None,
        },
    }


def _print_contract(config: Mapping[str, Any]) -> None:
    print(json.dumps(_seal_config(config), indent=2, sort_keys=True))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    for variable in THREAD_ENV_VARS:
        os.environ[variable] = str(args.threads)

    try:
        model_path, config_path, lock_path = _resolved_paths(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if not args.overwrite:
        existing = [path for path in (model_path, config_path) if path.exists()]
        if existing:
            rendered = ", ".join(str(path) for path in existing)
            raise SystemExit(
                f"refusing to overwrite existing artifact(s): {rendered}; "
                "use --overwrite with exact targets"
            )
    if lock_path.exists():
        raise SystemExit(
            f"training lock already exists at {lock_path}; verify that no run "
            "is active before removing the stale lock"
        )

    from ideal_poly_volume_toolkit.rl.polytope import make_polytope_env

    preview_env = make_polytope_env(
        d_range=(args.d, args.d),
        N_range=(args.n_low, args.n_high),
        horizon=args.horizon,
        eval_cost_weight=args.eval_cost_weight,
        seed=args.seed,
    )
    try:
        config = _base_config(
            args,
            model_path=model_path,
            config_path=config_path,
            action_ordering=preview_env.op_names,
            action_space=repr(preview_env.action_space),
            observation_space=repr(preview_env.observation_space),
        )
    finally:
        preview_env.close()

    if args.dry_run:
        config["status"] = "dry_run"
    print(f"[d={args.d}] resolved training contract:")
    _print_contract(config)
    if args.dry_run:
        print("dry run: no training started and no artifact written")
        return 0

    model_path.parent.mkdir(parents=True, exist_ok=True)
    lock_document = _seal_config(
        {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "status": "training_lock",
            "pid": os.getpid(),
            "started_at": config["started_at"],
            "model_path": _display_path(model_path),
            "config_path": _display_path(config_path),
            "repository": config["repository"],
        }
    )
    try:
        _exclusive_create_json(lock_path, lock_document)
    except FileExistsError as exc:
        raise SystemExit(f"training lock was acquired concurrently: {lock_path}") from exc

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        from ideal_poly_volume_toolkit.rl.training import train_controller

        def factory():
            return make_polytope_env(
                d_range=(args.d, args.d),
                N_range=(args.n_low, args.n_high),
                horizon=args.horizon,
                eval_cost_weight=args.eval_cost_weight,
                seed=args.seed,
            )

        with tempfile.TemporaryDirectory(
            prefix=f".{model_path.stem}.", dir=model_path.parent
        ) as temporary_directory:
            temporary_model = Path(temporary_directory) / model_path.name
            model = train_controller(
                factory,
                total_timesteps=args.steps,
                save_path=temporary_model,
                seed=args.seed,
                device="cpu",
                verbose=1,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                gamma=args.gamma,
                ent_coef=args.ent_coef,
            )
            if not temporary_model.is_file():
                raise RuntimeError(
                    f"trainer returned without writing expected model {temporary_model}"
                )
            elapsed = time.perf_counter() - wall_start
            cpu_time = time.process_time() - cpu_start
            actual_timesteps = getattr(model, "num_timesteps", None)
            config["status"] = "complete"
            config["ended_at"] = _utc_now()
            config["elapsed_sec"] = float(elapsed)
            config["cpu_sec"] = float(cpu_time)
            config["training"]["actual_timesteps"] = (
                int(actual_timesteps) if actual_timesteps is not None else None
            )
            config["artifacts"]["model_sha256"] = _sha256_file(temporary_model)
            config["artifacts"]["model_bytes"] = temporary_model.stat().st_size
            sealed_config = _seal_config(config)
            temporary_config = Path(temporary_directory) / config_path.name
            _exclusive_create_json(temporary_config, sealed_config)

            # Both complete artifacts exist before either final path changes.
            # The pair helper snapshots overwrite targets and restores them if
            # either promotion fails.
            _promote_artifact_pair(
                temporary_model,
                temporary_config,
                model_path,
                config_path,
                overwrite=args.overwrite,
            )
    except BaseException as exc:
        print(
            f"training failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise
    finally:
        try:
            lock_path.unlink()
        except OSError:
            pass

    config_file_sha256 = _sha256_file(config_path)
    print(
        f"[d={args.d}] complete in {elapsed / 60:.1f} min\n"
        f"  model: {_display_path(model_path)} "
        f"sha256={config['artifacts']['model_sha256']}\n"
        f"  config: {_display_path(config_path)} "
        f"sha256={config_file_sha256}"
    )
    return 0


if __name__ == "__main__":
    main()
