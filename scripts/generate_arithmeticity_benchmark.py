#!/usr/bin/env python3
"""Generate a resumable, certified arithmetic ideal-polyhedron corpus.

The v1 corpus has two deliberately separate construction modes:

``layered-ring``
    Deterministic closed-form zero-shear configurations.  For ``k`` rings of
    ``n`` vertices the construction has exactly ``k*n + 2`` ideal vertices,
    including a centre and an explicit vertex at infinity.  This is the fast,
    reproducible mode intended for numerical-stability campaigns.

``legalized``
    Seeded planar Delaunay disks legalized to a strict interior Rivin maximum,
    geometrically realized, and then closed by the full fan at infinity.

Every accepted entry carries both (1) an exact PSL(2,Z) certificate for the
abstract zero-shear structure and (2) a coordinate-derived shear residual for
the stored floating-point realization.  Float holonomy traces are not used as
an integrality certificate.

Writes are checkpointed atomically.  Existing paths are never replaced unless
``--overwrite`` is explicit, and ``--resume`` verifies the schema, run config,
deduplication backend, and source hashes before doing more work.  Legacy
schema-less benchmark files are intentionally not resumable.

Examples
--------
Generate both deterministic 300-vertex layered-ring types::

    python scripts/generate_arithmeticity_benchmark.py \
        --n-vertices 300 --mode layered-ring

Generate all 15 strict layered-ring types with 302 total vertices::

    python scripts/generate_arithmeticity_benchmark.py \
        --n-vertices 302 --mode layered-ring --workers 4

Search for five legalized 300-vertex types with bounded work::

    python scripts/generate_arithmeticity_benchmark.py \
        --n-vertices 300 --mode legalized --target-count 5 \
        --max-attempts 100 --max-wall-seconds 1800 --workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing.connection import wait as wait_for_connections
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


# Bound numerical-library fan-out before NumPy/SciPy are imported.  A campaign
# can opt in to a different value through this purpose-specific environment
# variable; the safe default is one BLAS thread per campaign worker.
try:
    BLAS_THREADS = int(os.environ.get("IDEAL_POLY_ARITHMETIC_BLAS_THREADS", "1"))
except ValueError:
    BLAS_THREADS = 1
BLAS_THREADS = min(8, max(1, BLAS_THREADS))
for _thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = str(BLAS_THREADS)

import numpy as np  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ideal_poly_volume_toolkit.geometry import (  # noqa: E402
    bloch_wigner,
    delaunay_triangulation_indices,
)
from ideal_poly_volume_toolkit.optimization import (  # noqa: E402
    complete_boundary_triangulation,
)
from ideal_poly_volume_toolkit.rivin_delaunay import (  # noqa: E402
    legalize_to_interior_maximum,
    realize_angles_as_points,
)
from ideal_poly_volume_toolkit.rivin_holonomy import (  # noqa: E402
    certify_exact_zero_shear_structure,
    compute_shears_from_vertices,
    validate_closed_sphere_triangulation,
)


GENERATOR_VERSION = "2.0.0"
SCHEMA_VERSION = "ideal-poly-arithmetic-corpus/1"
PROTOCOL_VERSION = "strict-zero-shear-psl2z/1"
GENERATOR_NAME = "scripts/generate_arithmeticity_benchmark.py"
SCHEMA_PATH = REPOSITORY_ROOT / "docs/schemas/arithmetic-corpus-v1.schema.json"
MAX_WORKERS = 8
DEFAULT_MAX_WALL_SECONDS = 300.0
DEFAULT_SHEAR_TOLERANCE = 1e-9
DEFAULT_MARGIN_TOLERANCE = 1e-4
DEFAULT_CONSTRAINT_TOLERANCE = 1e-6
DEFAULT_MAX_ANGLE_ERROR_TOLERANCE = 1e-7
DEFAULT_CLOSURE_ERROR_TOLERANCE = 1e-8
DEFAULT_MAX_FLIPS = 64

_SOURCE_PATHS = (
    Path(__file__).resolve(),
    REPOSITORY_ROOT / "ideal_poly_volume_toolkit/geometry.py",
    REPOSITORY_ROOT / "ideal_poly_volume_toolkit/optimization.py",
    REPOSITORY_ROOT / "ideal_poly_volume_toolkit/rivin_delaunay.py",
    REPOSITORY_ROOT / "ideal_poly_volume_toolkit/rivin_holonomy.py",
    SCHEMA_PATH,
)


class CandidateRejected(RuntimeError):
    """A bounded attempt completed but did not meet the corpus protocol."""

    def __init__(self, reason: str, message: str, diagnostics: dict | None = None):
        super().__init__(message)
        self.reason = reason
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class LayeredRingConstruction:
    """Closed-form layered-ring realization with an explicit infinity vertex."""

    k: int
    n: int
    h: float
    log_radius_ratio: float
    finite_vertices: np.ndarray
    finite_triangles: tuple[tuple[int, int, int], ...]
    boundary_triangles: tuple[tuple[int, int, int], ...]
    infinity_index: int

    @property
    def total_vertices(self) -> int:
        return self.k * self.n + 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    hashes = {}
    for path in _SOURCE_PATHS:
        if path.is_file():
            hashes[str(path.relative_to(REPOSITORY_ROOT))] = _sha256_file(path)
    return hashes


def _jsonable(value: Any) -> Any:
    """Normalize first-party values and reject non-finite/non-JSON data."""
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _jsonable(value.as_dict())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float is forbidden in corpus JSON: {value}")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"corpus JSON object key must be a string: {key!r}")
            normalized[key] = _jsonable(item)
        return normalized
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"value is not JSON serializable under the corpus protocol: {value!r}")


def strict_json_bytes(payload: Any) -> bytes:
    """Serialize canonical strict JSON (notably with NaN/Infinity disabled)."""
    normalized = _jsonable(payload)
    text = json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def load_json_strict(path: str | Path) -> dict[str, Any]:
    """Load RFC-compatible JSON while rejecting NaN and duplicate keys."""
    with Path(path).open("r", encoding="utf-8") as handle:
        result = json.load(
            handle,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    if not isinstance(result, dict):
        raise ValueError("corpus root must be a JSON object")
    return result


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_temp_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(strict_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        return temporary_path
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def exclusive_create_json(path: str | Path, payload: Any) -> None:
    """Atomically create ``path`` without ever replacing an existing file."""
    destination = Path(path)
    temporary_path = _write_temp_json(destination, payload)
    try:
        os.link(temporary_path, destination)
        _fsync_directory(destination.parent)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to clobber existing corpus: {destination}; "
            "use --resume or --overwrite"
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Atomically replace ``path`` with a fully flushed strict-JSON file."""
    destination = Path(path)
    temporary_path = _write_temp_json(destination, payload)
    try:
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def layered_ring_factor_pairs(total_vertices: int) -> list[tuple[int, int]]:
    """Return every strict ``(k, n)`` layered-ring type for a total size.

    Strict constructions require ``n > 3``.  In particular, the geometrically
    flat ``n == 3, k >= 3`` family is deliberately excluded.
    """
    if isinstance(total_vertices, bool) or int(total_vertices) != total_vertices:
        raise ValueError("total_vertices must be an integer")
    total_vertices = int(total_vertices)
    if total_vertices < 6:
        return []
    ring_vertices = total_vertices - 2
    return [
        (k, ring_vertices // k)
        for k in range(1, ring_vertices + 1)
        if ring_vertices % k == 0 and ring_vertices // k > 3
    ]


def construct_layered_ring(k: int, n: int) -> LayeredRingConstruction:
    """Construct the stable closed-form ``k``-by-``n`` zero-shear family."""
    if isinstance(k, bool) or int(k) != k or int(k) < 1:
        raise ValueError("k must be a positive integer")
    if isinstance(n, bool) or int(n) != n or int(n) <= 3:
        raise ValueError("n must be an integer greater than 3 for strict mode")
    k = int(k)
    n = int(n)

    h = math.pi / n
    cosine = math.cos(h)
    # Evaluate S-2 directly: forming S=4+2c-4c^2 loses most significant
    # digits as n grows.  Since S/2 = 1 + (S-2)/2, this keeps acosh stable.
    s_minus_two = 4.0 * math.sin(h / 2.0) ** 2 * (2.0 * cosine + 1.0)
    log_radius_ratio = (
        0.0 if k == 1 else math.acosh(1.0 + 0.5 * s_minus_two)
    )

    vertices: list[complex] = [0.0j]
    ring_indices = np.arange(n, dtype=np.float64)
    for layer in range(k):
        log_radius = (layer - (k - 1) / 2.0) * log_radius_ratio
        radius = math.exp(log_radius)
        phases = (2.0 * ring_indices + layer) * h
        vertices.extend(radius * np.exp(1j * phases))

    def vertex(layer: int, index: int) -> int:
        return 1 + layer * n + (index % n)

    infinity_index = 1 + k * n
    finite_faces: list[tuple[int, int, int]] = []
    for index in range(n):
        finite_faces.append((0, vertex(0, index), vertex(0, index + 1)))
    for layer in range(k - 1):
        for index in range(n):
            finite_faces.append(
                (vertex(layer, index), vertex(layer + 1, index), vertex(layer, index + 1))
            )
            finite_faces.append(
                (
                    vertex(layer, index),
                    vertex(layer + 1, index - 1),
                    vertex(layer + 1, index),
                )
            )

    boundary_faces = list(finite_faces)
    for index in range(n):
        boundary_faces.append(
            (infinity_index, vertex(k - 1, index + 1), vertex(k - 1, index))
        )

    construction = LayeredRingConstruction(
        k=k,
        n=n,
        h=h,
        log_radius_ratio=log_radius_ratio,
        finite_vertices=np.asarray(vertices, dtype=np.complex128),
        finite_triangles=tuple(finite_faces),
        boundary_triangles=tuple(boundary_faces),
        infinity_index=infinity_index,
    )
    expected_vertices = k * n + 2
    expected_faces = 2 * k * n
    if len(construction.finite_vertices) + 1 != expected_vertices:
        raise AssertionError("layered-ring vertex count invariant failed")
    if len(construction.boundary_triangles) != expected_faces:
        raise AssertionError("layered-ring face count invariant failed")
    return construction


def generate_random_sphere_points(n_points: int, rng: np.random.Generator) -> np.ndarray:
    """Compatibility helper returning finite stereographic sphere samples."""
    points = rng.standard_normal((n_points, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    # Resample an astronomically unlikely exact pole instead of emitting a
    # second infinity vertex into a finite planar triangulation.
    denominator = 1.0 - points[:, 2]
    while np.any(denominator <= np.finfo(float).eps):
        mask = denominator <= np.finfo(float).eps
        replacement = rng.standard_normal((int(mask.sum()), 3))
        replacement /= np.linalg.norm(replacement, axis=1, keepdims=True)
        points[mask] = replacement
        denominator = 1.0 - points[:, 2]
    return points[:, 0] / denominator + 1j * points[:, 1] / denominator


def generate_random_plane_points(
    n_points: int,
    rng: np.random.Generator,
    scale: float = 2.0,
) -> np.ndarray:
    """Generate deterministic Gaussian finite-plane coordinates."""
    return scale * (rng.standard_normal(n_points) + 1j * rng.standard_normal(n_points))


def build_prompt(n_vertices: int, n_configs: int) -> str:
    """Build the legacy LLM-facing prompt with corrected total-vertex semantics."""
    return (
        f"Construct {n_configs} pairwise non-isometric convex ideal polyhedra "
        f"in H^3, each with exactly {n_vertices} total ideal vertices and "
        f"triangular boundary faces. Represent one vertex explicitly as JSON "
        f"null (the point at infinity) and every other vertex as [real, imag]. "
        f"Provide the complete closed-sphere triangulation, including every "
        f"face incident to infinity. Certify arithmeticity by an exact "
        f"zero-shear PSL(2,Z) holonomy calculation, and separately report the "
        f"maximum coordinate-derived shear residual for the stored realization."
    )


def _triangle_volume(finite_vertices: np.ndarray, finite_faces: Sequence[Sequence[int]]) -> float:
    faces = np.asarray(finite_faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("finite_faces must be a nonempty array of triangles")
    z0 = finite_vertices[faces[:, 0]]
    z1 = finite_vertices[faces[:, 1]]
    z2 = finite_vertices[faces[:, 2]]
    cross_ratios = (z0 - z2) / (z1 - z2)
    values = np.asarray(bloch_wigner(cross_ratios), dtype=float)
    if not np.all(np.isfinite(values)):
        raise CandidateRejected("nonfinite_volume", "Bloch-Wigner volume is non-finite")
    return float(np.sum(np.abs(values)))


def _coordinate_shear_summary(
    full_faces: Sequence[Sequence[int]],
    finite_vertices: np.ndarray,
    infinity_index: int,
    tolerance: float,
) -> dict[str, Any]:
    coordinates: list[complex | None] = [complex(value) for value in finite_vertices]
    if infinity_index != len(coordinates):
        raise ValueError("infinity_index must immediately follow the finite vertices")
    coordinates.append(None)
    shears = compute_shears_from_vertices(
        [tuple(map(int, face)) for face in full_faces],
        coordinates,
    )
    values = np.asarray(list(shears.values()), dtype=float)
    all_finite = bool(values.size and np.all(np.isfinite(values)))
    max_abs = float(np.max(np.abs(values))) if all_finite else None
    rms = float(np.sqrt(np.mean(values * values))) if all_finite else None
    return {
        "n_edges": int(values.size),
        "all_finite": all_finite,
        "max_abs": max_abs,
        "rms": rms,
        "tolerance": float(tolerance),
        "within_tolerance": bool(all_finite and max_abs is not None and max_abs <= tolerance),
    }


def _qhull_audit(
    finite_vertices: np.ndarray,
    analytic_faces: Sequence[Sequence[int]],
) -> dict[str, Any]:
    expected = {tuple(sorted(map(int, face))) for face in analytic_faces}
    try:
        qhull_faces = delaunay_triangulation_indices(finite_vertices)
        observed = {tuple(sorted(map(int, face))) for face in qhull_faces}
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        return {
            "attempted": True,
            "succeeded": True,
            "matches_analytic_finite_faces": not missing and not unexpected,
            "analytic_face_count": len(expected),
            "qhull_face_count": len(observed),
            "missing_face_count": len(missing),
            "unexpected_face_count": len(unexpected),
            "missing_face_examples": [list(face) for face in missing[:5]],
            "unexpected_face_examples": [list(face) for face in unexpected[:5]],
            "error": None,
        }
    except Exception as exc:  # Qhull errors are diagnostic, never authoritative.
        return {
            "attempted": True,
            "succeeded": False,
            "matches_analytic_finite_faces": False,
            "analytic_face_count": len(expected),
            "qhull_face_count": None,
            "missing_face_count": None,
            "unexpected_face_count": None,
            "missing_face_examples": [],
            "unexpected_face_examples": [],
            "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:500]}",
        }


def _skipped_qhull_audit(analytic_face_count: int) -> dict[str, Any]:
    return {
        "attempted": False,
        "succeeded": False,
        "matches_analytic_finite_faces": False,
        "analytic_face_count": int(analytic_face_count),
        "qhull_face_count": None,
        "missing_face_count": None,
        "unexpected_face_count": None,
        "missing_face_examples": [],
        "unexpected_face_examples": [],
        "error": None,
    }


_RIVIN_RESIDUAL_KEYS = (
    "feasible",
    "strict_interior",
    "tolerance",
    "min_angle",
    "max_angle",
    "max_triangle_sum_residual",
    "max_interior_vertex_sum_residual",
    "max_equality_residual",
    "min_dihedral_margin",
    "worst_dihedral_edge",
    "max_dihedral_angle",
    "min_boundary_margin",
    "worst_boundary_vertex",
    "max_boundary_angle_sum",
    "min_inequality_margin",
    "n_triangles",
    "n_vertices",
    "n_interior_vertices",
    "n_boundary_vertices",
    "n_interior_edges",
)


def _compact_rivin_residuals(residuals: dict | None) -> dict | None:
    if residuals is None:
        return None
    return _jsonable({key: residuals.get(key) for key in _RIVIN_RESIDUAL_KEYS})


def _compact_legalization(result: dict[str, Any]) -> dict[str, Any]:
    history = []
    for row in result.get("history", []):
        history.append(
            {
                "step": row.get("step"),
                "status": row.get("status"),
                "lp_realizable": row.get("lp_realizable"),
                "lp_min_angle": row.get("lp_min_angle"),
                "lp_slack": row.get("lp_slack"),
                "lp_seconds": row.get("lp_seconds"),
                "optimizer_success": row.get("optimizer_success"),
                "optimizer_message": row.get("optimizer_message"),
                "optimizer_iterations": row.get("optimizer_iterations"),
                "optimizer_active_constraints": row.get("optimizer_active_constraints"),
                "optimizer_reduced_gradient_norm": row.get(
                    "optimizer_reduced_gradient_norm"
                ),
                "optimizer_rejected_tie": row.get("optimizer_rejected_tie"),
                "optimization_seconds": row.get("optimization_seconds"),
                "residuals": _compact_rivin_residuals(row.get("residuals")),
                "flipped_edge": row.get("flipped_edge"),
            }
        )
    optimization = result.get("optimization_result") or {}
    return _jsonable(
        {
            "success": result.get("success"),
            "termination_reason": result.get("termination_reason"),
            "message": result.get("message"),
            "n_flips": result.get("n_flips"),
            "n_states": result.get("n_states"),
            "volume": result.get("volume"),
            "residuals": _compact_rivin_residuals(result.get("residuals")),
            "reduced_gradient_norm": optimization.get("reduced_gradient_norm"),
            "n_active_constraints": optimization.get("n_active_constraints"),
            "timings": result.get("timings", {}),
            "history": history,
        }
    )


def _compact_realization(result: dict[str, Any]) -> dict[str, Any]:
    return _jsonable(
        {
            "success": result.get("success"),
            "message": str(result.get("message", "")),
            "triangulation_preserved": result.get("triangulation_preserved"),
            "angle_error": result.get("angle_error"),
            "angle_error_degrees": result.get("angle_error_degrees"),
            "max_angle_error": result.get("max_angle_error"),
            "max_angle_error_degrees": result.get("max_angle_error_degrees"),
            "closure_error": result.get("closure_error"),
            "orientation_consistent": result.get("orientation_consistent"),
            "min_abs_oriented_area": result.get("min_abs_oriented_area"),
            "n_iterations": result.get("n_iterations"),
            "target_angle_structure": _compact_rivin_residuals(
                (result.get("residuals") or {}).get("target_angle_structure")
            ),
            "realized_angle_structure": _compact_rivin_residuals(
                (result.get("residuals") or {}).get("realized_angle_structure")
            ),
        }
    )


def _configuration_from_geometry(
    *,
    generation_method: str,
    construction_id: str,
    finite_vertices: np.ndarray,
    finite_faces: Sequence[Sequence[int]],
    full_faces: Sequence[Sequence[int]],
    infinity_index: int,
    construction: dict[str, Any],
    diagnostics: dict[str, Any],
    proof_basis: str,
    seed: int | None,
    shear_tolerance: float,
    stage_timings: dict[str, float],
) -> dict[str, Any]:
    certification_started = time.perf_counter()
    finite_vertices = np.asarray(finite_vertices, dtype=np.complex128)
    if finite_vertices.ndim != 1 or not np.all(np.isfinite(finite_vertices)):
        raise CandidateRejected(
            "nonfinite_coordinates", "finite realization contains invalid coordinates"
        )
    finite_faces = tuple(tuple(map(int, face)) for face in finite_faces)
    full_faces = tuple(tuple(map(int, face)) for face in full_faces)
    total_vertices = len(finite_vertices) + 1
    if infinity_index != total_vertices - 1:
        raise CandidateRejected(
            "vertex_count_mismatch", "explicit infinity index does not close the vertex list"
        )

    topology = validate_closed_sphere_triangulation(full_faces)
    expected_faces = 2 * total_vertices - 4
    expected_edges = 3 * total_vertices - 6
    if (
        topology.n_vertices != total_vertices
        or topology.n_faces != expected_faces
        or topology.n_edges != expected_edges
    ):
        raise CandidateRejected(
            "topology_mismatch",
            "closed triangulation does not have the expected simplicial-sphere counts",
            topology.as_dict(),
        )

    shear_summary = _coordinate_shear_summary(
        full_faces,
        finite_vertices,
        infinity_index,
        shear_tolerance,
    )
    exact_certificate_raw = certify_exact_zero_shear_structure(full_faces).as_dict()
    traces = exact_certificate_raw["traces"]
    max_abs_trace = max((abs(int(trace)) for trace in traces), default=0)
    # JSON numbers are commonly consumed through IEEE-754 doubles.  Exact
    # holonomy entries can rapidly exceed 2**53, so encode every unbounded
    # matrix entry and trace as a signed decimal string at the file boundary.
    exact_certificate = dict(exact_certificate_raw)
    exact_certificate["traces"] = [str(int(trace)) for trace in traces]
    exact_generators = []
    for generator in exact_certificate_raw["generators"]:
        serialized_generator = dict(generator)
        serialized_generator["matrix"] = [
            [str(int(entry)) for entry in row] for row in generator["matrix"]
        ]
        serialized_generator["trace"] = str(int(generator["trace"]))
        exact_generators.append(serialized_generator)
    exact_certificate["generators"] = exact_generators
    trace_summary = {
        "n_traces": len(traces),
        "max_abs_trace": str(max_abs_trace),
        "max_abs_trace_decimal_digits": len(str(max_abs_trace)),
        "float_integrality_test_used": False,
    }
    checks = {
        "closed_sphere": True,
        "exact_total_vertex_count": topology.n_vertices == total_vertices,
        "coordinate_zero_shear": shear_summary["within_tolerance"],
        "exact_determinants_one": exact_certificate["all_determinants_one"],
        "exact_integral_traces": exact_certificate["all_traces_integral"],
        "accepted": bool(
            shear_summary["within_tolerance"] and exact_certificate["is_arithmetic"]
        ),
    }
    if not checks["accepted"]:
        reason = (
            "coordinate_shear_failure"
            if not shear_summary["within_tolerance"]
            else "exact_certificate_failure"
        )
        raise CandidateRejected(
            reason,
            "candidate did not satisfy both coordinate and exact certificate gates",
            {"coordinate_zero_shear": shear_summary, "checks": checks},
        )

    volume = _triangle_volume(finite_vertices, finite_faces)
    vertices_json: list[list[float] | None] = [
        [float(value.real), float(value.imag)] for value in finite_vertices
    ]
    vertices_json.append(None)
    stage_timings = dict(stage_timings)
    stage_timings["certification_seconds"] = float(
        time.perf_counter() - certification_started
    )

    return {
        "id": "pending",
        "attempt_index": -1,
        "construction_id": construction_id,
        "generation_method": generation_method,
        "seed": seed,
        "input": {
            "coordinate_model": "upper-half-space-boundary-complex-plus-infinity",
            "vertex_semantics": "total-including-explicit-infinity",
            "vertices": vertices_json,
            "infinity_index": infinity_index,
            "n_vertices": total_vertices,
            "n_finite_vertices": len(finite_vertices),
            "triangles": [list(face) for face in full_faces],
            "finite_triangulation": [list(face) for face in finite_faces],
            "n_triangles": len(full_faces),
            "n_finite_triangles": len(finite_faces),
        },
        "construction": _jsonable(construction),
        "ground_truth": {
            "is_arithmetic": True,
            "classification": "zero-shear-psl2z",
            "proof_basis": proof_basis,
            "volume": volume,
            "volume_method": "analytic-finite-faces-vectorized-bloch-wigner",
            "description": (
                "Exact abstract zero-shear PSL(2,Z) certificate paired with a "
                "coordinate-derived shear audit of the stored realization."
            ),
        },
        "certificate": {
            "protocol_version": PROTOCOL_VERSION,
            "topology": topology.as_dict(),
            "coordinate_zero_shear": shear_summary,
            "exact_zero_shear": exact_certificate,
            "exact_trace_summary": trace_summary,
            "checks": checks,
        },
        "diagnostics": _jsonable(diagnostics),
        "deduplication": {
            "method": "pending",
            "topology_id": "pending",
        },
        "timings": _jsonable(stage_timings),
    }


def generate_layered_ring_configuration(
    k: int,
    n: int,
    *,
    shear_tolerance: float = DEFAULT_SHEAR_TOLERANCE,
    qhull_audit: bool = True,
) -> dict[str, Any]:
    """Generate and fully certify one deterministic layered-ring entry."""
    started = time.perf_counter()
    construction = construct_layered_ring(k, n)
    constructed = time.perf_counter()
    audit = (
        _qhull_audit(construction.finite_vertices, construction.finite_triangles)
        if qhull_audit
        else _skipped_qhull_audit(len(construction.finite_triangles))
    )
    return _configuration_from_geometry(
        generation_method="layered-ring-exact",
        construction_id=f"layered-ring-k{k}-n{n}",
        finite_vertices=construction.finite_vertices,
        finite_faces=construction.finite_triangles,
        full_faces=construction.boundary_triangles,
        infinity_index=construction.infinity_index,
        construction={
            "family": "layered-ring-zero-shear",
            "face_source": "closed-form-analytic-authoritative",
            "parameters": {
                "k": k,
                "n": n,
                "h_radians": construction.h,
                "log_radius_ratio": construction.log_radius_ratio,
                "radius_ratio": math.exp(construction.log_radius_ratio),
                "radius_ratio_definition": (
                    "q=1 (single-ring family)"
                    if k == 1
                    else "q=exp(acosh(S/2))"
                ),
                "radius_ratio_equation": (
                    "not-applicable-without-a-belt"
                    if k == 1
                    else "q + q^-1 = 4 + 2*cos(pi/n) - 4*cos(pi/n)^2"
                ),
                "stable_s_minus_two_formula": (
                    "4*sin(pi/(2*n))^2*(2*cos(pi/n)+1)"
                ),
                "radius_exponent_centering": "layer-(k-1)/2",
                "phase_formula": "(2*j+layer)*pi/n",
                "zero_shear_proof_identity": (
                    "the radius-ratio equation makes every belt cross-ratio "
                    "have unit modulus; rotational symmetry handles the two fans"
                ),
                "strict": True,
            },
        },
        diagnostics={
            "qhull_audit": audit,
            "legalization": None,
            "realization": None,
        },
        proof_basis="closed-form-layered-ring-zero-shear-identity",
        seed=None,
        shear_tolerance=shear_tolerance,
        stage_timings={
            "construction_seconds": float(constructed - started),
            "qhull_audit_seconds": float(time.perf_counter() - constructed),
        },
    )


def _realized_complex_points(result: dict[str, Any], n_finite_vertices: int) -> np.ndarray:
    points = result.get("points")
    vertex_list = result.get("vertex_list")
    if points is None or vertex_list is None:
        raise CandidateRejected("realization_failure", "realizer returned no points")
    points = np.asarray(points, dtype=float)
    if points.shape != (len(vertex_list), 2):
        raise CandidateRejected("realization_failure", "realizer returned invalid shape")
    coordinates = np.empty(n_finite_vertices, dtype=np.complex128)
    seen = set()
    for row, vertex in enumerate(vertex_list):
        vertex = int(vertex)
        if vertex < 0 or vertex >= n_finite_vertices or vertex in seen:
            raise CandidateRejected("realization_failure", "invalid realization vertex map")
        seen.add(vertex)
        coordinates[vertex] = complex(points[row, 0], points[row, 1])
    if seen != set(range(n_finite_vertices)):
        raise CandidateRejected("realization_failure", "realization omitted finite vertices")
    return coordinates


def generate_legalized_configuration(
    total_vertices: int,
    seed: int,
    *,
    shear_tolerance: float = DEFAULT_SHEAR_TOLERANCE,
    max_flips: int = DEFAULT_MAX_FLIPS,
    margin_tolerance: float = DEFAULT_MARGIN_TOLERANCE,
    constraint_tolerance: float = DEFAULT_CONSTRAINT_TOLERANCE,
) -> dict[str, Any]:
    """Generate one seeded strict-interior Rivin candidate and certify it."""
    if total_vertices < 4:
        raise ValueError("total_vertices must be at least 4")
    started = time.perf_counter()
    n_finite_vertices = total_vertices - 1
    rng = np.random.default_rng(seed)
    initial_points = generate_random_sphere_points(n_finite_vertices, rng)
    initial_faces = [
        tuple(map(int, face))
        for face in delaunay_triangulation_indices(initial_points)
    ]
    delaunay_finished = time.perf_counter()

    legalization = legalize_to_interior_maximum(
        initial_faces,
        max_flips=max_flips,
        margin_tolerance=margin_tolerance,
        constraint_tolerance=constraint_tolerance,
        verbose=False,
    )
    legalization_finished = time.perf_counter()
    compact_legalization = _compact_legalization(legalization)
    if not legalization.get("success"):
        raise CandidateRejected(
            f"legalization_{legalization.get('termination_reason', 'failure')}",
            str(legalization.get("message", "Rivin legalization failed")),
            {"legalization": compact_legalization},
        )

    finite_faces = [
        tuple(map(int, face)) for face in legalization["finite_triangulation"]
    ]
    realization = realize_angles_as_points(
        finite_faces,
        legalization["angles"],
        verbose=False,
    )
    realization_finished = time.perf_counter()
    compact_realization = _compact_realization(realization)
    if not (
        realization.get("success")
        and realization.get("triangulation_preserved")
        and realization.get("points") is not None
    ):
        raise CandidateRejected(
            "realization_failure",
            str(realization.get("message", "geometric realization failed")),
            {
                "legalization": compact_legalization,
                "realization": compact_realization,
            },
        )

    # The realizer's historical success flag is based on RMS angle error.
    # At hundreds of vertices a single bad corner can be diluted by thousands
    # of otherwise accurate angles, so the corpus protocol also gates the
    # worst corner, developing-map closure, and global winding explicitly.
    max_angle_error = realization.get("max_angle_error")
    closure_error = realization.get("closure_error")
    realization_residuals_pass = (
        isinstance(max_angle_error, (int, float, np.integer, np.floating))
        and math.isfinite(float(max_angle_error))
        and float(max_angle_error) <= DEFAULT_MAX_ANGLE_ERROR_TOLERANCE
        and isinstance(closure_error, (int, float, np.integer, np.floating))
        and math.isfinite(float(closure_error))
        and float(closure_error) <= DEFAULT_CLOSURE_ERROR_TOLERANCE
        and realization.get("orientation_consistent") is True
    )
    if not realization_residuals_pass:
        raise CandidateRejected(
            "realization_residual_failure",
            "realization exceeded a maximum-angle, closure, or winding gate",
            {
                "thresholds": {
                    "max_angle_error_radians": (
                        DEFAULT_MAX_ANGLE_ERROR_TOLERANCE
                    ),
                    "closure_error_relative": (
                        DEFAULT_CLOSURE_ERROR_TOLERANCE
                    ),
                    "orientation_consistent": True,
                },
                "legalization": compact_legalization,
                "realization": compact_realization,
            },
        )

    realized_points = _realized_complex_points(realization, n_finite_vertices)
    full_faces_array = complete_boundary_triangulation(
        np.asarray(finite_faces, dtype=np.int64),
        n_finite_vertices,
    )
    full_faces = [tuple(map(int, face)) for face in full_faces_array]

    return _configuration_from_geometry(
        generation_method="legalized-rivin",
        construction_id=f"legalized-seed-{seed}",
        finite_vertices=realized_points,
        finite_faces=finite_faces,
        full_faces=full_faces,
        infinity_index=n_finite_vertices,
        construction={
            "family": "seeded-rivin-legalization",
            "face_source": "legalized-finite-disk-plus-explicit-infinity-fan",
            "parameters": {
                "seed": seed,
                "initial_sampler": "uniform-sphere-stereographic-projection",
                "max_flips": max_flips,
                "margin_tolerance": margin_tolerance,
                "constraint_tolerance": constraint_tolerance,
                "max_angle_error_tolerance_radians": (
                    DEFAULT_MAX_ANGLE_ERROR_TOLERANCE
                ),
                "closure_error_tolerance_relative": (
                    DEFAULT_CLOSURE_ERROR_TOLERANCE
                ),
            },
        },
        diagnostics={
            "qhull_audit": None,
            "legalization": compact_legalization,
            "realization": compact_realization,
        },
        proof_basis="strict-interior-rivin-maximum-with-numerical-realization",
        seed=seed,
        shear_tolerance=shear_tolerance,
        stage_timings={
            "delaunay_seconds": float(delaunay_finished - started),
            "legalization_seconds": float(legalization_finished - delaunay_finished),
            "realization_seconds": float(realization_finished - legalization_finished),
        },
    )


def optimize_for_fixed_combinatorics(complex_points: Iterable[complex]):
    """Deprecated compatibility wrapper around strict Rivin legalization."""
    warnings.warn(
        "optimize_for_fixed_combinatorics is deprecated; use the corpus campaign",
        DeprecationWarning,
        stacklevel=2,
    )
    points = np.asarray(list(complex_points), dtype=np.complex128)
    points = points[np.isfinite(points)]
    faces = [tuple(map(int, face)) for face in delaunay_triangulation_indices(points)]
    result = legalize_to_interior_maximum(faces, verbose=False)
    if not result.get("success"):
        return None, None, result["finite_triangulation"]
    realized = realize_angles_as_points(
        result["finite_triangulation"], result["angles"], verbose=False
    )
    if not realized.get("success") or not realized.get("triangulation_preserved"):
        return None, None, result["finite_triangulation"]
    optimized = _realized_complex_points(realized, len(points))
    return optimized, result.get("volume"), result["finite_triangulation"]


def generate_optimized_config(
    config_id: int,
    n_vertices: int,
    seed: int,
    verbose: bool = True,
):
    """Deprecated single-entry adapter using corrected total-vertex semantics."""
    warnings.warn(
        "generate_optimized_config is deprecated; use run_campaign",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        configuration = generate_legalized_configuration(n_vertices, seed)
    except CandidateRejected:
        return None
    configuration["id"] = f"arithmetic-{int(config_id):06d}"
    if verbose:
        print(
            f"  volume={configuration['ground_truth']['volume']:.12g}; "
            f"max_shear={configuration['certificate']['coordinate_zero_shear']['max_abs']:.3e}"
        )
    return configuration


def _edges_from_faces(faces: Sequence[Sequence[int]]) -> tuple[set[int], set[tuple[int, int]]]:
    vertices: set[int] = set()
    edges: set[tuple[int, int]] = set()
    for face in faces:
        a, b, c = map(int, face)
        vertices.update((a, b, c))
        edges.update((tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))))
    return vertices, edges


def available_deduplication_backend() -> str:
    """Select pynauty, with exact NetworkX isomorphism as the fallback."""
    try:
        import pynauty  # noqa: F401
    except ImportError:
        try:
            import networkx  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "topology deduplication requires pynauty or NetworkX; "
                "install the project graph extra"
            ) from exc
        return "networkx-exact-isomorphism-v1"
    return "pynauty-canonical-certificate-v1"


def _networkx_graph(faces: Sequence[Sequence[int]]):
    import networkx as nx

    vertices, edges = _edges_from_faces(faces)
    graph = nx.Graph()
    graph.add_nodes_from(sorted(vertices))
    graph.add_edges_from(sorted(edges))
    return graph


def topology_identifier(
    faces: Sequence[Sequence[int]],
    backend: str,
) -> tuple[str, Any]:
    """Return an invariant identifier and the graph used for exact fallback."""
    vertices, edges = _edges_from_faces(faces)
    if backend == "pynauty-canonical-certificate-v1":
        import pynauty

        labels = sorted(vertices)
        relabel = {vertex: index for index, vertex in enumerate(labels)}
        adjacency = {index: set() for index in range(len(labels))}
        for u, v in edges:
            adjacency[relabel[u]].add(relabel[v])
            adjacency[relabel[v]].add(relabel[u])
        graph = pynauty.Graph(
            number_of_vertices=len(labels),
            directed=False,
            adjacency_dict=adjacency,
        )
        certificate = pynauty.certificate(graph)
        return f"pynauty-sha256:{_sha256_bytes(certificate)}", None
    if backend == "networkx-exact-isomorphism-v1":
        import networkx as nx

        graph = _networkx_graph(faces)
        # This hash is only a bucket/identifier.  Equality is always confirmed
        # below with nx.is_isomorphic, so hash collisions cannot deduplicate data.
        nx.set_node_attributes(graph, "vertex", "corpus_node_kind")
        wl_hash = nx.weisfeiler_lehman_graph_hash(
            graph,
            node_attr="corpus_node_kind",
            iterations=8,
        )
        return f"networkx-wl:{wl_hash}", graph
    raise ValueError(f"unknown deduplication backend: {backend}")


def find_duplicate_configuration(
    candidate_faces: Sequence[Sequence[int]],
    configurations: Sequence[dict[str, Any]],
    backend: str,
) -> tuple[str, str | None]:
    """Find an isomorphic accepted topology, never trusting a weak hash alone."""
    topology_id, candidate_graph = topology_identifier(candidate_faces, backend)
    if backend == "pynauty-canonical-certificate-v1":
        for configuration in configurations:
            if configuration["deduplication"]["topology_id"] == topology_id:
                return topology_id, str(configuration["id"])
        return topology_id, None

    import networkx as nx

    for configuration in configurations:
        existing_faces = configuration["input"]["triangles"]
        existing_graph = _networkx_graph(existing_faces)
        if nx.is_isomorphic(candidate_graph, existing_graph):
            return topology_id, str(configuration["id"])
    return topology_id, None


def derive_attempt_seed(base_seed: int, attempt_index: int) -> int:
    """Derive the documented consecutive campaign seed."""
    seed = int(base_seed) + int(attempt_index)
    if seed < 0 or seed >= 1 << 63:
        raise ValueError("derived attempt seed must lie in [0, 2**63)")
    return seed


def _run_attempt(run_config: dict[str, Any], attempt_index: int) -> dict[str, Any]:
    mode = run_config["mode"]
    if mode == "layered-ring":
        pairs = layered_ring_factor_pairs(run_config["n_vertices"])
        if attempt_index >= len(pairs):
            return {
                "outcome": "rejected",
                "reason": "no_more_layered_ring_factor_pairs",
                "message": "all strict factor pairs have been exhausted",
                "candidate_key": f"factor-index-{attempt_index}",
                "seed": None,
                "diagnostics": {},
            }
        k, n = pairs[attempt_index]
        candidate_key = f"layered-ring-k{k}-n{n}"
        seed = None
        def generator() -> dict[str, Any]:
            return generate_layered_ring_configuration(
                k,
                n,
                shear_tolerance=run_config["shear_tolerance"],
                qhull_audit=run_config["qhull_audit"],
            )
    elif mode == "legalized":
        seed = derive_attempt_seed(run_config["base_seed"], attempt_index)
        candidate_key = f"legalized-seed-{seed}"
        def generator() -> dict[str, Any]:
            return generate_legalized_configuration(
                run_config["n_vertices"],
                seed,
                shear_tolerance=run_config["shear_tolerance"],
                max_flips=run_config["max_flips"],
                margin_tolerance=run_config["margin_tolerance"],
                constraint_tolerance=run_config["constraint_tolerance"],
            )
    else:
        raise ValueError(f"unknown campaign mode: {mode}")

    try:
        configuration = generator()
    except CandidateRejected as exc:
        return {
            "outcome": "rejected",
            "reason": exc.reason,
            "message": str(exc),
            "candidate_key": candidate_key,
            "seed": seed,
            "diagnostics": _jsonable(exc.diagnostics),
        }
    return {
        "outcome": "candidate",
        "reason": "certified",
        "message": "candidate passed construction and certificate gates",
        "candidate_key": candidate_key,
        "seed": seed,
        "configuration": configuration,
        "diagnostics": {},
    }


def _attempt_process_entry(
    connection,
    run_config: dict[str, Any],
    attempt_index: int,
) -> None:
    started_at = _utc_now()
    started = time.perf_counter()
    try:
        result = _run_attempt(run_config, attempt_index)
        envelope = {
            "ok": True,
            "attempt_index": attempt_index,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "elapsed_seconds": float(time.perf_counter() - started),
            "result": result,
        }
    except BaseException as exc:
        envelope = {
            "ok": False,
            "attempt_index": attempt_index,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "elapsed_seconds": float(time.perf_counter() - started),
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    try:
        connection.send(envelope)
    finally:
        connection.close()


def _execute_attempt_batch(
    run_config: dict[str, Any],
    attempt_indices: Sequence[int],
    deadline: float,
) -> dict[int, dict[str, Any]]:
    """Run a bounded process batch and terminate it at the campaign deadline."""
    context = multiprocessing.get_context("spawn")
    active: dict[int, tuple[multiprocessing.Process, Any]] = {}
    launched_at: dict[int, str] = {}
    for attempt_index in attempt_indices:
        receive, send = context.Pipe(duplex=False)
        process = context.Process(
            target=_attempt_process_entry,
            args=(send, run_config, attempt_index),
            name=f"arithmetic-corpus-attempt-{attempt_index}",
        )
        launched_at[attempt_index] = _utc_now()
        process.start()
        send.close()
        active[attempt_index] = (process, receive)

    results: dict[int, dict[str, Any]] = {}
    try:
        while active:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            connections = [receive for _, receive in active.values()]
            ready = wait_for_connections(
                connections,
                timeout=min(0.2, remaining),
            )
            for receive in ready:
                attempt_index = next(
                    index
                    for index, (_, candidate_receive) in active.items()
                    if candidate_receive is receive
                )
                process, _ = active.pop(attempt_index)
                try:
                    results[attempt_index] = receive.recv()
                except EOFError:
                    results[attempt_index] = {
                        "ok": False,
                        "attempt_index": attempt_index,
                        "started_at": launched_at[attempt_index],
                        "finished_at": _utc_now(),
                        "elapsed_seconds": 0.0,
                        "error_type": "WorkerProcessError",
                        "message": f"worker exited with code {process.exitcode}",
                    }
                receive.close()
                process.join(timeout=1.0)

            # A crashed worker closes its pipe, which will normally appear in
            # ``ready``.  This fallback prevents a platform-specific dead pipe
            # from lingering until the whole wall limit.
            for attempt_index, (process, receive) in list(active.items()):
                if not process.is_alive() and process.exitcode is not None:
                    try:
                        if receive.poll(0.05):
                            continue
                    except (EOFError, OSError):
                        pass
                    active.pop(attempt_index)
                    receive.close()
                    process.join(timeout=1.0)
                    results[attempt_index] = {
                        "ok": False,
                        "attempt_index": attempt_index,
                        "started_at": launched_at[attempt_index],
                        "finished_at": _utc_now(),
                        "elapsed_seconds": 0.0,
                        "error_type": "WorkerProcessError",
                        "message": f"worker exited with code {process.exitcode}",
                    }
    finally:
        for attempt_index, (process, receive) in active.items():
            if process.is_alive():
                process.terminate()
            process.join(timeout=2.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=2.0)
            receive.close()
            results[attempt_index] = {
                "ok": False,
                "attempt_index": attempt_index,
                "started_at": launched_at[attempt_index],
                "finished_at": _utc_now(),
                "elapsed_seconds": 0.0,
                "error_type": "CampaignWallTimeout",
                "message": "attempt terminated at the campaign wall-time limit",
                "timed_out": True,
            }
    return results


def _run_config_hash(run_config: dict[str, Any]) -> str:
    normalized = json.dumps(_jsonable(run_config), sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(normalized.encode("utf-8"))


def _dependency_versions() -> dict[str, str | None]:
    versions = {}
    for distribution in (
        "ideal-poly-volume-toolkit",
        "numpy",
        "scipy",
        "networkx",
        "pynauty",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit if len(commit) >= 7 else None


def _initial_document(
    run_config: dict[str, Any],
    source_hashes: dict[str, str],
    backend: str,
    invocation: Sequence[str],
) -> dict[str, Any]:
    timestamp = _utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "generator": {
            "name": GENERATOR_NAME,
            "version": GENERATOR_VERSION,
            "source_sha256": source_hashes.get(GENERATOR_NAME, "0" * 64),
        },
        "status": "in_progress",
        "created_at": timestamp,
        "updated_at": timestamp,
        "run_config": _jsonable(run_config),
        "run_config_sha256": _run_config_hash(run_config),
        "provenance": {
            "invocation": list(invocation),
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "git_commit": _git_commit(),
            "git_dirty": None,
            "dependency_versions": _dependency_versions(),
            "thread_environment": {
                variable: os.environ.get(variable)
                for variable in (
                    "IDEAL_POLY_ARITHMETIC_BLAS_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "source_files": source_hashes,
            "deduplication_backend": backend,
            "legacy_dataset_policy": (
                "schema-less and pre-v1 datasets are historical/quarantined and "
                "are never resumed as v1 corpora"
            ),
        },
        "prompt": build_prompt(run_config["n_vertices"], run_config["target_count"]),
        "next_attempt_index": 0,
        "counters": {
            "attempted": 0,
            "accepted": 0,
            "rejected": 0,
            "duplicates": 0,
            "errors": 0,
            "timed_out": 0,
        },
        "attempts": [],
        "configurations": [],
        "summary": {
            "target_count": run_config["target_count"],
            "produced_count": 0,
            "remaining_count": run_config["target_count"],
            "all_arithmetic": True,
            "n_vertices": run_config["n_vertices"],
        },
        "timings": {
            "cumulative_active_seconds": 0.0,
            "last_invocation_seconds": 0.0,
        },
    }


_ATTEMPT_COUNTER = {
    "accepted": "accepted",
    "rejected": "rejected",
    "duplicate": "duplicates",
    "error": "errors",
    "timed_out": "timed_out",
}


def _refresh_document(document: dict[str, Any]) -> None:
    statuses = [attempt["status"] for attempt in document["attempts"]]
    document["next_attempt_index"] = len(document["attempts"])
    document["counters"] = {
        "attempted": len(statuses),
        "accepted": statuses.count("accepted"),
        "rejected": statuses.count("rejected"),
        "duplicates": statuses.count("duplicate"),
        "errors": statuses.count("error"),
        "timed_out": statuses.count("timed_out"),
    }
    produced = len(document["configurations"])
    target = document["run_config"]["target_count"]
    document["summary"] = {
        "target_count": target,
        "produced_count": produced,
        "remaining_count": max(0, target - produced),
        "all_arithmetic": all(
            configuration["ground_truth"]["is_arithmetic"]
            for configuration in document["configurations"]
        ),
        "n_vertices": document["run_config"]["n_vertices"],
    }
    document["updated_at"] = _utc_now()


def _validate_document_invariants(document: dict[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "not an arithmetic corpus v1 file; legacy datasets must remain "
            "quarantined and cannot be resumed"
        )
    if document.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("corpus protocol version mismatch")
    if document.get("run_config_sha256") != _run_config_hash(document["run_config"]):
        raise ValueError("stored run_config hash does not match its contents")
    attempts = document.get("attempts", [])
    if [attempt.get("attempt_index") for attempt in attempts] != list(range(len(attempts))):
        raise ValueError("attempt indices must be contiguous from zero")
    if document.get("next_attempt_index") != len(attempts):
        raise ValueError("next_attempt_index does not match checkpointed attempts")
    expected_statuses = {status for status in _ATTEMPT_COUNTER}
    if any(attempt.get("status") not in expected_statuses for attempt in attempts):
        raise ValueError("corpus contains an unknown attempt status")
    expected_counters = {
        "attempted": len(attempts),
        "accepted": sum(a["status"] == "accepted" for a in attempts),
        "rejected": sum(a["status"] == "rejected" for a in attempts),
        "duplicates": sum(a["status"] == "duplicate" for a in attempts),
        "errors": sum(a["status"] == "error" for a in attempts),
        "timed_out": sum(a["status"] == "timed_out" for a in attempts),
    }
    if document.get("counters") != expected_counters:
        raise ValueError("corpus counters do not match its attempts")
    configurations = document.get("configurations", [])
    if len(configurations) != expected_counters["accepted"]:
        raise ValueError("accepted-attempt count does not match configuration count")
    expected_ids = [f"arithmetic-{index:06d}" for index in range(len(configurations))]
    if [configuration.get("id") for configuration in configurations] != expected_ids:
        raise ValueError("configuration identifiers are not contiguous")
    n_vertices = document["run_config"]["n_vertices"]
    for configuration in configurations:
        entry = configuration["input"]
        if entry["n_vertices"] != n_vertices or len(entry["vertices"]) != n_vertices:
            raise ValueError("configuration violates total vertex semantics")
        infinity_index = entry["infinity_index"]
        if infinity_index != n_vertices - 1 or entry["vertices"][infinity_index] is not None:
            raise ValueError("configuration lacks the explicit final infinity vertex")
        if entry["n_triangles"] != 2 * n_vertices - 4:
            raise ValueError("configuration is not a full simplicial sphere")
        validate_closed_sphere_triangulation(entry["triangles"])


def validate_corpus_document(
    document: dict[str, Any],
    schema_path: str | Path = SCHEMA_PATH,
) -> None:
    """Validate both the published JSON schema and cross-field invariants."""
    _validate_document_invariants(document)
    strict_json_bytes(document)
    try:
        from jsonschema.validators import validator_for
    except ImportError as exc:
        raise RuntimeError(
            "schema validation requires jsonschema; install the project dev extra"
        ) from exc
    schema = load_json_strict(schema_path)
    validator_class = validator_for(schema)
    validator_class.check_schema(schema)
    validator_class(schema).validate(document)


def _validate_run_config(run_config: dict[str, Any]) -> None:
    n_vertices = run_config["n_vertices"]
    if isinstance(n_vertices, bool) or int(n_vertices) != n_vertices or n_vertices < 4:
        raise ValueError("--n-vertices must be at least 4 and includes infinity")
    if (
        isinstance(run_config["target_count"], bool)
        or int(run_config["target_count"]) != run_config["target_count"]
        or run_config["target_count"] < 1
    ):
        raise ValueError("--target-count must be positive")
    if (
        isinstance(run_config["max_attempts"], bool)
        or int(run_config["max_attempts"]) != run_config["max_attempts"]
        or run_config["max_attempts"] < run_config["target_count"]
    ):
        raise ValueError("--max-attempts must be at least --target-count")
    if not (1 <= run_config["workers"] <= MAX_WORKERS):
        raise ValueError(f"--workers must be between 1 and {MAX_WORKERS}")
    if not math.isfinite(run_config["max_wall_seconds"]) or run_config["max_wall_seconds"] <= 0:
        raise ValueError("--max-wall-seconds must be finite and positive")
    if run_config["base_seed"] < 0 or run_config["base_seed"] >= 1 << 63:
        raise ValueError("--seed must lie in [0, 2**63)")
    if (
        isinstance(run_config["max_flips"], bool)
        or int(run_config["max_flips"]) != run_config["max_flips"]
        or run_config["max_flips"] < 0
    ):
        raise ValueError("--max-flips must be a nonnegative integer")
    for key in ("shear_tolerance", "margin_tolerance", "constraint_tolerance"):
        if not math.isfinite(run_config[key]) or run_config[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if run_config["mode"] == "layered-ring":
        pairs = layered_ring_factor_pairs(n_vertices)
        if not pairs:
            raise ValueError(
                f"no strict layered-ring factorization exists for V={n_vertices}"
            )
        if run_config["target_count"] > len(pairs):
            raise ValueError(
                f"V={n_vertices} has only {len(pairs)} strict layered-ring "
                "combinatorial types; lower --target-count or use --mode legalized"
            )
        if run_config["max_attempts"] > len(pairs):
            raise ValueError(
                f"--max-attempts cannot exceed the {len(pairs)} deterministic factor pairs"
            )


def _append_attempt_result(
    document: dict[str, Any],
    attempt_index: int,
    envelope: dict[str, Any],
    backend: str,
) -> None:
    started_at = str(envelope.get("started_at", _utc_now()))
    attempt = {
        "attempt_index": attempt_index,
        "candidate_key": f"attempt-{attempt_index}",
        "seed": None,
        "started_at": started_at,
        "finished_at": str(envelope.get("finished_at", _utc_now())),
        "elapsed_seconds": float(envelope.get("elapsed_seconds", 0.0)),
        "status": "error",
        "reason": "worker_error",
        "message": str(envelope.get("message", "worker failed")),
        "error_type": envelope.get("error_type"),
        "configuration_id": None,
        "diagnostics": {},
    }
    if envelope.get("timed_out"):
        attempt.update(
            status="timed_out",
            reason="campaign_wall_timeout",
            error_type="CampaignWallTimeout",
        )
    elif envelope.get("ok"):
        result = envelope["result"]
        attempt.update(
            candidate_key=result["candidate_key"],
            seed=result["seed"],
            reason=result["reason"],
            message=result["message"],
            error_type=None,
            diagnostics=result.get("diagnostics", {}),
        )
        if result["outcome"] == "rejected":
            attempt["status"] = "rejected"
        else:
            configuration = result["configuration"]
            topology_id, duplicate_of = find_duplicate_configuration(
                configuration["input"]["triangles"],
                document["configurations"],
                backend,
            )
            if duplicate_of is not None:
                attempt.update(
                    status="duplicate",
                    reason="isomorphic_topology",
                    message=f"exact topology duplicate of {duplicate_of}",
                )
            else:
                configuration_id = f"arithmetic-{len(document['configurations']):06d}"
                configuration["id"] = configuration_id
                configuration["attempt_index"] = attempt_index
                configuration["deduplication"] = {
                    "method": backend,
                    "topology_id": topology_id,
                }
                document["configurations"].append(configuration)
                attempt.update(
                    status="accepted",
                    configuration_id=configuration_id,
                )
    attempt["finished_at"] = str(envelope.get("finished_at", _utc_now()))
    document["attempts"].append(_jsonable(attempt))
    _refresh_document(document)


def _checkpoint(
    output_path: Path,
    document: dict[str, Any],
    base_elapsed: float,
    invocation_started: float,
) -> None:
    invocation_elapsed = float(time.monotonic() - invocation_started)
    document["timings"] = {
        "cumulative_active_seconds": float(base_elapsed + invocation_elapsed),
        "last_invocation_seconds": invocation_elapsed,
    }
    _refresh_document(document)
    _validate_document_invariants(document)
    atomic_write_json(output_path, document)


def run_campaign(
    output_path: str | Path,
    run_config: dict[str, Any],
    *,
    resume: bool = False,
    overwrite: bool = False,
    invocation: Sequence[str] | None = None,
    use_subprocesses: bool = True,
    attempt_runner: Callable[[dict[str, Any], int], dict[str, Any]] = _run_attempt,
) -> dict[str, Any]:
    """Run or resume a bounded campaign, checkpointing every completed batch."""
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    _validate_run_config(run_config)
    output_path = Path(output_path)
    backend = available_deduplication_backend()
    source_hashes = _source_hashes()
    invocation = list(sys.argv if invocation is None else invocation)

    if resume:
        if not output_path.is_file():
            raise FileNotFoundError(f"cannot resume missing corpus: {output_path}")
        document = load_json_strict(output_path)
        validate_corpus_document(document)
        if document["run_config_sha256"] != _run_config_hash(run_config):
            raise ValueError("resume run config does not match the checkpoint")
        if document["provenance"]["source_files"] != source_hashes:
            raise ValueError("resume refused because generator/core source hashes changed")
        if document["provenance"]["deduplication_backend"] != backend:
            raise ValueError("resume refused because the exact deduplication backend changed")
        if document["status"] == "complete":
            return document
    else:
        document = _initial_document(run_config, source_hashes, backend, invocation)
        _validate_document_invariants(document)
        if overwrite:
            atomic_write_json(output_path, document)
        else:
            exclusive_create_json(output_path, document)

    base_elapsed = float(document["timings"]["cumulative_active_seconds"])
    invocation_started = time.monotonic()
    wall_remaining = run_config["max_wall_seconds"] - base_elapsed
    if wall_remaining <= 0.0:
        document["status"] = "exhausted"
        _checkpoint(output_path, document, base_elapsed, invocation_started)
        return document
    deadline = invocation_started + wall_remaining

    try:
        while (
            len(document["configurations"]) < run_config["target_count"]
            and document["next_attempt_index"] < run_config["max_attempts"]
            and time.monotonic() < deadline
        ):
            remaining_target = run_config["target_count"] - len(document["configurations"])
            remaining_attempts = run_config["max_attempts"] - document["next_attempt_index"]
            batch_size = min(run_config["workers"], remaining_target, remaining_attempts)
            attempt_indices = list(
                range(
                    document["next_attempt_index"],
                    document["next_attempt_index"] + batch_size,
                )
            )
            if use_subprocesses and attempt_runner is _run_attempt:
                envelopes = _execute_attempt_batch(run_config, attempt_indices, deadline)
            else:
                envelopes = {}
                for attempt_index in attempt_indices:
                    started_at = _utc_now()
                    started = time.perf_counter()
                    try:
                        result = attempt_runner(run_config, attempt_index)
                        envelopes[attempt_index] = {
                            "ok": True,
                            "attempt_index": attempt_index,
                            "started_at": started_at,
                            "finished_at": _utc_now(),
                            "elapsed_seconds": float(time.perf_counter() - started),
                            "result": result,
                        }
                    except BaseException as exc:
                        envelopes[attempt_index] = {
                            "ok": False,
                            "attempt_index": attempt_index,
                            "started_at": started_at,
                            "finished_at": _utc_now(),
                            "elapsed_seconds": float(time.perf_counter() - started),
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
            for attempt_index in attempt_indices:
                _append_attempt_result(document, attempt_index, envelopes[attempt_index], backend)
            _checkpoint(output_path, document, base_elapsed, invocation_started)
    except KeyboardInterrupt:
        document["status"] = "in_progress"
        _checkpoint(output_path, document, base_elapsed, invocation_started)
        raise

    if len(document["configurations"]) >= run_config["target_count"]:
        document["status"] = "complete"
    else:
        document["status"] = "exhausted"
    _checkpoint(output_path, document, base_elapsed, invocation_started)
    validate_corpus_document(document)
    return document


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a versioned arithmetic ideal-polyhedron corpus with "
            "exact total-vertex and full-boundary semantics."
        )
    )
    parser.add_argument(
        "--n-vertices",
        type=int,
        help="Total ideal vertices, including the explicit infinity vertex.",
    )
    parser.add_argument(
        "--mode",
        choices=("layered-ring", "legalized"),
        default="layered-ring",
        help="Construction family (default: layered-ring).",
    )
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument(
        "--n-optimized",
        type=int,
        default=None,
        help="Deprecated alias for --target-count.",
    )
    parser.add_argument(
        "--n-random",
        type=int,
        default=None,
        help="Deprecated; v1 arithmetic corpora require this to be zero.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed for legalized mode.")
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=DEFAULT_MAX_WALL_SECONDS,
        help="Cumulative active wall limit; running attempt workers are terminated at it.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-flips", type=int, default=DEFAULT_MAX_FLIPS)
    parser.add_argument("--shear-tolerance", type=float, default=DEFAULT_SHEAR_TOLERANCE)
    parser.add_argument("--margin-tolerance", type=float, default=DEFAULT_MARGIN_TOLERANCE)
    parser.add_argument(
        "--constraint-tolerance", type=float, default=DEFAULT_CONSTRAINT_TOLERANCE
    )
    parser.add_argument(
        "--no-qhull-audit",
        action="store_true",
        help="Skip the non-authoritative Qhull comparison in layered-ring mode.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-only",
        type=Path,
        default=None,
        metavar="PATH",
        help="Validate an existing v1 corpus and exit.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Bound the run to one target/attempt, one worker, and at most 60 seconds.",
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    return parser


def _run_config_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict[str, Any]:
    if args.n_vertices is None:
        parser.error("--n-vertices is required unless --validate-only is used")
    if args.n_random not in (None, 0):
        parser.error(
            "--n-random is deprecated; arithmetic corpus v1 contains only "
            "certified arithmetic configurations, so use --n-random 0"
        )
    if args.n_optimized is not None:
        warnings.warn("--n-optimized is deprecated; use --target-count", FutureWarning)
        if args.target_count is not None and args.target_count != args.n_optimized:
            parser.error("--target-count and --n-optimized disagree")
    target_count = args.target_count if args.target_count is not None else args.n_optimized
    pairs = layered_ring_factor_pairs(args.n_vertices) if args.mode == "layered-ring" else []
    if target_count is None:
        target_count = len(pairs) if args.mode == "layered-ring" else 1
    max_attempts = args.max_attempts
    if max_attempts is None:
        max_attempts = len(pairs) if args.mode == "layered-ring" else max(10, target_count * 20)
    workers = args.workers
    max_wall_seconds = args.max_wall_seconds
    if args.smoke:
        target_count = 1
        max_attempts = 1
        workers = 1
        max_wall_seconds = min(max_wall_seconds, 60.0)
    run_config = {
        "mode": args.mode,
        "vertex_semantics": "total-including-explicit-infinity",
        "n_vertices": args.n_vertices,
        "target_count": target_count,
        "max_attempts": max_attempts,
        "max_wall_seconds": max_wall_seconds,
        "workers": workers,
        "base_seed": args.seed,
        "blas_threads_per_worker": BLAS_THREADS,
        "max_flips": args.max_flips,
        "shear_tolerance": args.shear_tolerance,
        "margin_tolerance": args.margin_tolerance,
        "constraint_tolerance": args.constraint_tolerance,
        "qhull_audit": not args.no_qhull_audit,
    }
    try:
        _validate_run_config(run_config)
    except ValueError as exc:
        parser.error(str(exc))
    return run_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.validate_only is not None:
        document = load_json_strict(args.validate_only)
        validate_corpus_document(document)
        print(f"Valid arithmetic corpus v1: {args.validate_only}")
        return 0
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    run_config = _run_config_from_args(args, parser)
    output = args.output or (
        REPOSITORY_ROOT
        / "results/data/arithmetic_corpora"
        / f"arithmetic-corpus-v1-{args.n_vertices}v-{args.mode}.json"
    )
    try:
        document = run_campaign(
            output,
            run_config,
            resume=args.resume,
            overwrite=args.overwrite,
            invocation=list(sys.argv if argv is None else [GENERATOR_NAME, *argv]),
        )
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        f"{document['status']}: {document['summary']['produced_count']}/"
        f"{document['summary']['target_count']} configurations; output={output}"
    )
    return 0 if document["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
