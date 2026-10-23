#!/usr/bin/env python3
"""Verify repository result data, Git LFS coverage, and provenance contracts.

The default mode accepts a valid Git LFS pointer in the working tree and reads
its locally available object through ``git lfs smudge``. CI uses
``--require-materialized`` after an LFS-enabled checkout to catch incomplete
checkouts as well.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

from jsonschema.exceptions import SchemaError, ValidationError
from jsonschema.validators import validator_for


REPO_ROOT = Path(__file__).resolve().parents[1]
OPTIMIZATION_RESULT_SCHEMA_PATH = "docs/schemas/optimization-result-v1.schema.json"
ARITHMETIC_CORPUS_SCHEMA_PATH = "docs/schemas/arithmetic-corpus-v1.schema.json"
RESULT_MANIFEST_PATH = "results/manifest.json"
RESULT_MANIFEST_SCHEMA_PATH = "docs/schemas/result-manifest-v1.schema.json"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"
REQUIRED_INPUT_HASHES = {
    "results/data/optimal_d4_symmetry_analysis.json": (
        "0bea03b4ddd6a91897ee9e9c3cdc4c5362d1a658a9a9de96911345edca2e29b5"
    ),
    "results/data/optimal_d5_d6_symmetry_analysis.json": (
        "d50056f9dbc2d5ad2bb8bec1d0a0543700b0dcc92d150fdbbfbdccc61ae7e702"
    ),
}
ORDINARY_GIT_PATHS = (
    OPTIMIZATION_RESULT_SCHEMA_PATH,
    ARITHMETIC_CORPUS_SCHEMA_PATH,
    RESULT_MANIFEST_PATH,
    RESULT_MANIFEST_SCHEMA_PATH,
    *REQUIRED_INPUT_HASHES,
)


class IntegrityError(RuntimeError):
    """Raised when a repository data-integrity invariant is violated."""


def _run_git(
    root: Path,
    *args: str,
    input_text: str | None = None,
) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise IntegrityError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout


def _smudge_pointer(root: Path, relative_path: str, pointer: bytes) -> bytes:
    completed = subprocess.run(
        ["git", "lfs", "smudge", "--", relative_path],
        cwd=root,
        input=pointer,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()
        raise IntegrityError(f"cannot read LFS object for {relative_path}: {detail}")
    return completed.stdout


def is_lfs_pointer(data: bytes) -> bool:
    """Return whether *data* starts with the canonical Git LFS pointer header."""

    return data.startswith(LFS_POINTER_PREFIX)


def list_lfs_paths(root: Path) -> list[str]:
    """Return all paths represented by LFS pointers in the current index."""

    output = _run_git(root, "lfs", "ls-files", "--name-only")
    paths = [line for line in output.splitlines() if line]
    if not paths:
        raise IntegrityError("no Git LFS paths found")
    return paths


def _filter_attributes(root: Path, paths: Iterable[str]) -> dict[str, str]:
    path_list = list(paths)
    if not path_list:
        return {}
    output = _run_git(
        root,
        "check-attr",
        "--stdin",
        "filter",
        input_text="".join(f"{path}\n" for path in path_list),
    )
    attributes: dict[str, str] = {}
    separator = ": filter: "
    for line in output.splitlines():
        if separator not in line:
            raise IntegrityError(f"unexpected git check-attr output: {line!r}")
        path, value = line.rsplit(separator, 1)
        attributes[path] = value
    return attributes


def check_lfs_attributes(root: Path, lfs_paths: Iterable[str]) -> None:
    """Require every indexed LFS pointer to remain covered by an LFS rule."""

    paths = list(lfs_paths)
    attributes = _filter_attributes(root, paths)
    uncovered = [path for path in paths if attributes.get(path) != "lfs"]
    if uncovered:
        sample = ", ".join(uncovered[:5])
        raise IntegrityError(
            f"{len(uncovered)} indexed LFS paths lack filter=lfs: {sample}"
        )


def check_ordinary_git_attributes(root: Path) -> None:
    """Require canonical provenance contracts to remain outside Git LFS."""

    attributes = _filter_attributes(root, ORDINARY_GIT_PATHS)
    filtered = [path for path, value in attributes.items() if value == "lfs"]
    if filtered:
        raise IntegrityError(
            "canonical ordinary-Git paths are covered by LFS: " + ", ".join(filtered)
        )


def _read_lfs_worktree_payload(root: Path, relative_path: str) -> tuple[bytes, bool]:
    path = root / relative_path
    if not path.is_file():
        raise IntegrityError(
            f"tracked LFS path is absent from the checkout: {relative_path}"
        )
    data = path.read_bytes()
    pointer = is_lfs_pointer(data)
    if pointer:
        data = _smudge_pointer(root, relative_path, data)
    return data, pointer


def check_lfs_json(
    root: Path,
    lfs_paths: Iterable[str],
) -> tuple[int, list[str]]:
    """Parse every LFS-managed JSON payload and report unsmudged paths."""

    json_count = 0
    pointer_paths: list[str] = []
    for relative_path in lfs_paths:
        path = Path(relative_path)
        if path.suffix.lower() != ".json":
            payload = (root / relative_path).read_bytes()
            if is_lfs_pointer(payload):
                pointer_paths.append(relative_path)
            continue

        payload, pointer = _read_lfs_worktree_payload(root, relative_path)
        if pointer:
            pointer_paths.append(relative_path)
        try:
            json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IntegrityError(
                f"invalid JSON payload at {relative_path}: {exc}"
            ) from exc
        json_count += 1
    return json_count, pointer_paths


def check_required_inputs(root: Path) -> None:
    """Validate required ordinary-Git inputs against their recorded hashes."""

    for relative_path, expected_hash in REQUIRED_INPUT_HASHES.items():
        path = root / relative_path
        if not path.is_file():
            raise IntegrityError(
                f"required provenance input is missing: {relative_path}"
            )
        payload = path.read_bytes()
        if is_lfs_pointer(payload):
            raise IntegrityError(
                f"required provenance input must be ordinary Git, not LFS: {relative_path}"
            )
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != expected_hash:
            raise IntegrityError(
                f"SHA-256 mismatch for {relative_path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        try:
            json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IntegrityError(
                f"invalid required JSON input {relative_path}: {exc}"
            ) from exc


def _read_json_object(path: Path, *, description: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise IntegrityError(f"cannot read {description} at {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise IntegrityError(f"invalid JSON in {description} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"{description} must contain a JSON object: {path}")
    return value


def _validate_json_schema(instance: object, schema: Mapping, *, label: str) -> None:
    """Validate *instance* and turn jsonschema diagnostics into one stable error."""

    validator_class = validator_for(schema)
    try:
        validator_class.check_schema(schema)
        validator_class(schema).validate(instance)
    except SchemaError as exc:
        raise IntegrityError(f"invalid JSON schema for {label}: {exc.message}") from exc
    except ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "<root>"
        raise IntegrityError(
            f"{label} does not satisfy its schema at {location}: {exc.message}"
        ) from exc


def _safe_repository_path(root: Path, raw_path: str, *, label: str) -> Path:
    """Resolve a normalized POSIX repository path without permitting escape."""

    if "\\" in raw_path:
        raise IntegrityError(f"{label} must use POSIX separators: {raw_path!r}")
    relative = PurePosixPath(raw_path)
    if (
        relative.is_absolute()
        or relative.as_posix() != raw_path
        or ".." in relative.parts
    ):
        raise IntegrityError(f"unsafe or non-normalized {label}: {raw_path!r}")

    resolved_root = root.resolve()
    try:
        resolved = (resolved_root / Path(*relative.parts)).resolve()
    except (OSError, ValueError) as exc:
        raise IntegrityError(f"invalid {label}: {raw_path!r}") from exc
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise IntegrityError(f"{label} escapes the repository: {raw_path!r}") from exc
    return resolved


def check_result_manifest(root: Path) -> int:
    """Validate the result manifest's schema, paths, hashes, and canonical rows."""

    root = root.resolve()
    schema = _read_json_object(
        root / RESULT_MANIFEST_SCHEMA_PATH,
        description="result-manifest schema",
    )
    manifest = _read_json_object(
        root / RESULT_MANIFEST_PATH,
        description="result manifest",
    )
    _validate_json_schema(manifest, schema, label="result manifest")

    artifacts_by_path: dict[str, Mapping] = {}
    for artifact in manifest["artifacts"]:
        relative_path = artifact["path"]
        if relative_path in artifacts_by_path:
            raise IntegrityError(f"duplicate result-manifest artifact: {relative_path}")
        if any(character in relative_path for character in "*?["):
            raise IntegrityError(
                f"result-manifest artifact path must not be a glob: {relative_path!r}"
            )

        path = _safe_repository_path(
            root,
            relative_path,
            label="result-manifest artifact path",
        )
        if not path.is_file():
            raise IntegrityError(f"result-manifest artifact is missing: {relative_path}")
        payload = path.read_bytes()
        if is_lfs_pointer(payload):
            payload = _smudge_pointer(root, relative_path, payload)
        actual_size = len(payload)
        if actual_size != artifact["size_bytes"]:
            raise IntegrityError(
                f"size mismatch for {relative_path}: expected "
                f"{artifact['size_bytes']}, got {actual_size}"
            )
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != artifact["sha256"]:
            raise IntegrityError(
                f"SHA-256 mismatch for {relative_path}: expected "
                f"{artifact['sha256']}, got {actual_hash}"
            )
        artifacts_by_path[relative_path] = artifact

    collection_globs: set[str] = set()
    for collection in manifest["collections"]:
        path_glob = collection["path_glob"]
        _safe_repository_path(
            root,
            path_glob,
            label="result-manifest collection glob",
        )
        if path_glob in collection_globs:
            raise IntegrityError(f"duplicate result-manifest collection: {path_glob}")
        collection_globs.add(path_glob)

    for relative_path, expected_hash in REQUIRED_INPUT_HASHES.items():
        artifact = artifacts_by_path.get(relative_path)
        if artifact is None:
            raise IntegrityError(
                f"canonical input is absent from result manifest: {relative_path}"
            )
        if artifact["sha256"] != expected_hash:
            raise IntegrityError(
                f"result manifest has the wrong canonical hash for {relative_path}"
            )
        if artifact["role"] != "canonical_input" or artifact["status"] != "canonical":
            raise IntegrityError(
                f"result manifest must classify {relative_path} as "
                "role=canonical_input and status=canonical"
            )

    return len(artifacts_by_path)


def check_data_integrity(root: Path, *, require_materialized: bool = False) -> None:
    """Run every repository data-integrity check."""

    root = root.resolve()
    _run_git(root, "lfs", "fsck", "--objects", "--pointers", "HEAD")
    lfs_paths = list_lfs_paths(root)
    check_lfs_attributes(root, lfs_paths)
    check_ordinary_git_attributes(root)
    json_count, pointer_paths = check_lfs_json(root, lfs_paths)
    check_required_inputs(root)
    manifest_artifact_count = check_result_manifest(root)

    if require_materialized and pointer_paths:
        sample = ", ".join(pointer_paths[:5])
        raise IntegrityError(
            f"{len(pointer_paths)} LFS paths are not materialized: {sample}"
        )

    print(f"Git LFS: {len(lfs_paths)} paths verified")
    print(f"LFS JSON: {json_count} payloads parsed")
    print(f"Canonical inputs: {len(REQUIRED_INPUT_HASHES)} hashes verified")
    print(f"Result manifest: {manifest_artifact_count} artifacts verified")
    if pointer_paths:
        print(
            f"Working tree: {len(pointer_paths)} valid LFS pointers "
            "(objects were read from local LFS storage)"
        )
    else:
        print("Working tree: all LFS paths materialized")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to inspect (default: inferred from this script)",
    )
    parser.add_argument(
        "--require-materialized",
        action="store_true",
        help="Fail if the checkout contains an LFS pointer instead of payload data",
    )
    args = parser.parse_args(argv)

    try:
        check_data_integrity(
            args.repo_root,
            require_materialized=args.require_materialized,
        )
    except IntegrityError as exc:
        print(f"Data integrity check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
