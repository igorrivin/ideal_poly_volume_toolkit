"""Optimization services for ideal hyperbolic polyhedra in H^3.

This module owns the reusable, non-UI part of the general H^3 optimizer.
Command-line and GUI entry points should adapt these functions rather than
maintaining independent objectives.

The parameterization fixes three ideal vertices at ``0``, ``1``, and
``infinity``.  Each remaining vertex is represented by spherical
``(theta, phi)`` coordinates and stereographically projected to the complex
plane.  Volume evaluation uses the vectorized Bloch--Wigner implementation in
``geometry``; SciPy's differential-evolution driver does not benefit from
building Torch autograd graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Callable, Iterable, Mapping

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import QhullError

from .geometry import (
    delaunay_triangulation_indices,
    ideal_poly_volume_via_delaunay,
)


OBJECTIVE_PENALTY = 1_000.0


def spherical_to_complex(theta, phi):
    """Stereographically project spherical coordinates to the complex plane."""
    return np.tan(np.asarray(theta) / 2.0) * np.exp(1j * np.asarray(phi))


def parameters_to_complex(params: Iterable[float], n_vertices: int) -> np.ndarray:
    """Build the finite vertices for an ``n_vertices`` ideal polyhedron.

    The returned array contains ``0``, ``1``, and the ``n_vertices - 3`` free
    vertices.  The final vertex at infinity is implicit.
    """
    if n_vertices < 4:
        raise ValueError("n_vertices must be at least 4")

    values = np.asarray(params, dtype=np.float64)
    expected = 2 * (n_vertices - 3)
    if values.ndim != 1 or values.size != expected:
        raise ValueError(
            f"expected {expected} parameters for {n_vertices} vertices; "
            f"received shape {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("optimization parameters must be finite")

    free = spherical_to_complex(values[0::2], values[1::2]).astype(
        np.complex128, copy=False
    )
    if not np.all(np.isfinite(free.real)) or not np.all(np.isfinite(free.imag)):
        raise ValueError("stereographic projection produced a non-finite vertex")
    return np.concatenate((np.array([0.0j, 1.0 + 0.0j]), free))


def negative_volume_objective(params: Iterable[float], n_vertices: int) -> float:
    """Return negative H^3 volume, or a finite penalty when triangulation fails."""
    try:
        vertices = parameters_to_complex(params, n_vertices)
        volume = ideal_poly_volume_via_delaunay(
            vertices,
            mode="bloch_wigner",
        )
    except (
        QhullError,
        ValueError,
        FloatingPointError,
        OverflowError,
        ZeroDivisionError,
    ):
        return OBJECTIVE_PENALTY

    if not np.isfinite(volume) or volume <= 0.0:
        return OBJECTIVE_PENALTY
    return -float(volume)


def _serially_polish_differential_evolution_result(
    result,
    objective: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
) -> None:
    """Apply SciPy's differential-evolution polish semantics without a pool.

    Recent SciPy versions pass the differential-evolution worker map into
    L-BFGS-B polishing.  That is useful for expensive objectives, but the H^3
    objective is fine-grained enough that repeated process dispatch dominates
    the finite-difference evaluations.  Running the same minimizer directly
    keeps polishing serial while preserving SciPy's result acceptance and
    function-evaluation accounting rules.
    """
    polished = minimize(
        objective,
        np.copy(result.x),
        method="L-BFGS-B",
        bounds=bounds,
    )
    result.nfev += polished.nfev

    limits = np.asarray(bounds, dtype=np.float64).T
    if (
        polished.fun < result.fun
        and polished.success
        and np.all(polished.x <= limits[1])
        and np.all(limits[0] <= polished.x)
    ):
        result.fun = polished.fun
        result.x = polished.x
        result.jac = polished.jac


def _canonical_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def complete_boundary_triangulation(
    finite_triangulation: np.ndarray,
    n_finite_vertices: int,
) -> np.ndarray:
    """Close a planar Delaunay disk by adding the fan at infinity.

    ``finite_triangulation`` consists of the faces not incident to infinity.
    Edges occurring once form its boundary; coning each of them to the
    implicit vertex ``n_finite_vertices`` yields the full triangulated sphere.
    """
    if int(n_finite_vertices) != n_finite_vertices or n_finite_vertices < 3:
        raise ValueError("n_finite_vertices must be an integer at least 3")
    n_finite_vertices = int(n_finite_vertices)
    triangles = np.asarray(finite_triangulation, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("finite_triangulation must have shape (n_faces, 3)")
    if not len(triangles):
        raise ValueError("finite_triangulation must contain at least one face")
    if triangles.min() < 0 or triangles.max() >= n_finite_vertices:
        raise ValueError("triangulation contains an out-of-range vertex index")
    if any(len(set(map(int, triangle))) != 3 for triangle in triangles):
        raise ValueError("triangulation faces must contain three distinct vertices")
    if set(map(int, triangles.ravel())) != set(range(n_finite_vertices)):
        raise ValueError("every finite vertex must occur in the triangulation")

    # Record both face incidence and the direction in which each input face
    # traverses an edge.  The latter lets us orient mixed-winding input before
    # adding the fan at infinity.
    edge_faces: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_index, (i, j, k) in enumerate(triangles):
        for u, v in ((int(i), int(j)), (int(j), int(k)), (int(k), int(i))):
            edge = _canonical_edge(u, v)
            direction = 1 if (u, v) == edge else -1
            edge_faces.setdefault(edge, []).append((face_index, direction))

    if any(len(incidences) > 2 for incidences in edge_faces.values()):
        raise ValueError("a triangulation edge cannot belong to more than two faces")

    boundary_edges = sorted(
        edge for edge, incidences in edge_faces.items() if len(incidences) == 1
    )
    boundary_adjacency: dict[int, set[int]] = {}
    for u, v in boundary_edges:
        boundary_adjacency.setdefault(u, set()).add(v)
        boundary_adjacency.setdefault(v, set()).add(u)
    if not boundary_adjacency or any(
        len(neighbours) != 2 for neighbours in boundary_adjacency.values()
    ):
        raise ValueError("finite triangulation boundary must be a simple cycle")
    seen = set()
    pending = [next(iter(boundary_adjacency))]
    while pending:
        vertex = pending.pop()
        if vertex in seen:
            continue
        seen.add(vertex)
        pending.extend(boundary_adjacency[vertex] - seen)
    if seen != set(boundary_adjacency):
        raise ValueError("finite triangulation boundary must be one connected cycle")

    # A connected triangulated disk has Euler characteristic one.  This rules
    # out, for example, a punctured torus whose boundary is also one simple
    # cycle and whose edge multiplicities otherwise look valid.
    finite_euler = n_finite_vertices - len(edge_faces) + len(triangles)
    if finite_euler != 1:
        raise ValueError("finite triangulation must have the topology of a disk")

    # Check the local manifold condition.  The link of an interior vertex is
    # one cycle; the link of a boundary vertex is one path.
    boundary_vertices = set(boundary_adjacency)
    for vertex in range(n_finite_vertices):
        link_adjacency: dict[int, set[int]] = {}
        link_edges: set[tuple[int, int]] = set()
        for triangle in triangles:
            if vertex not in triangle:
                continue
            opposite = [int(item) for item in triangle if item != vertex]
            link_edge = _canonical_edge(*opposite)
            if link_edge in link_edges:
                raise ValueError("finite triangulation has a non-manifold vertex link")
            link_edges.add(link_edge)
            u, v = link_edge
            link_adjacency.setdefault(u, set()).add(v)
            link_adjacency.setdefault(v, set()).add(u)

        link_seen: set[int] = set()
        link_pending = [next(iter(link_adjacency))]
        while link_pending:
            neighbour = link_pending.pop()
            if neighbour in link_seen:
                continue
            link_seen.add(neighbour)
            link_pending.extend(link_adjacency[neighbour] - link_seen)
        if link_seen != set(link_adjacency):
            raise ValueError("finite triangulation has a disconnected vertex link")

        link_degrees = [len(neighbours) for neighbours in link_adjacency.values()]
        if vertex in boundary_vertices:
            if link_degrees.count(1) != 2 or any(
                degree not in (1, 2) for degree in link_degrees
            ):
                raise ValueError("finite triangulation has a non-manifold boundary")
        elif any(degree != 2 for degree in link_degrees):
            raise ValueError("finite triangulation has a non-manifold interior")

    # Propagate a coherent orientation across the finite disk.  Two incident
    # faces must traverse their shared edge in opposite directions.
    face_signs = np.zeros(len(triangles), dtype=np.int8)
    face_signs[0] = 1
    face_pending = [0]
    face_neighbours: dict[int, list[tuple[int, int, int]]] = {
        face_index: [] for face_index in range(len(triangles))
    }
    for incidences in edge_faces.values():
        if len(incidences) != 2:
            continue
        (first_face, first_direction), (second_face, second_direction) = incidences
        face_neighbours[first_face].append(
            (second_face, first_direction, second_direction)
        )
        face_neighbours[second_face].append(
            (first_face, second_direction, first_direction)
        )

    while face_pending:
        face = face_pending.pop()
        for neighbour, face_direction, neighbour_direction in face_neighbours[face]:
            required_sign = (
                -int(face_signs[face]) * face_direction * neighbour_direction
            )
            if face_signs[neighbour] == 0:
                face_signs[neighbour] = required_sign
                face_pending.append(neighbour)
            elif face_signs[neighbour] != required_sign:
                raise ValueError("finite triangulation is not orientable")
    if np.any(face_signs == 0):
        raise ValueError("finite triangulation faces must be edge-connected")

    oriented_triangles = triangles.copy()
    flipped = face_signs == -1
    oriented_triangles[flipped] = oriented_triangles[flipped][:, [0, 2, 1]]

    infinity = n_finite_vertices
    infinity_faces = []
    for edge in boundary_edges:
        face_index, input_direction = edge_faces[edge][0]
        oriented_direction = input_direction * int(face_signs[face_index])
        u, v = edge if oriented_direction == 1 else tuple(reversed(edge))
        # Reverse the disk's directed boundary edge in the adjacent fan face.
        infinity_faces.append((v, u, infinity))
    return np.vstack((oriented_triangles, np.asarray(infinity_faces, dtype=np.int64)))


def analyze_ideal_polyhedron_structure(
    finite_vertices: np.ndarray,
    finite_triangulation: np.ndarray | None = None,
) -> dict:
    """Describe the full spherical boundary, including the implicit infinity."""
    vertices = np.asarray(finite_vertices, dtype=np.complex128)
    if vertices.ndim != 1:
        raise ValueError("finite_vertices must be a one-dimensional complex array")
    triangles = (
        delaunay_triangulation_indices(vertices)
        if finite_triangulation is None
        else np.asarray(finite_triangulation, dtype=np.int64)
    )
    boundary = complete_boundary_triangulation(triangles, len(vertices))

    edges: set[tuple[int, int]] = set()
    for i, j, k in boundary:
        edges.update(
            {
                _canonical_edge(int(i), int(j)),
                _canonical_edge(int(j), int(k)),
                _canonical_edge(int(k), int(i)),
            }
        )

    n_vertices = len(vertices) + 1
    degrees = [0] * n_vertices
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1

    n_faces = len(boundary)
    n_edges = len(edges)
    return {
        "n_vertices": n_vertices,
        "n_finite_vertices": len(vertices),
        "implicit_infinity_index": len(vertices),
        "n_faces": n_faces,
        "n_finite_faces": len(triangles),
        "n_infinity_faces": n_faces - len(triangles),
        "n_edges": n_edges,
        "euler_characteristic": n_vertices - n_edges + n_faces,
        "vertex_degrees": sorted(degrees),
        "boundary_triangulation": boundary,
    }


@dataclass(frozen=True)
class OptimizationTrial:
    """Serializable summary of one differential-evolution run."""

    trial: int
    volume: float | None
    params: np.ndarray
    success: bool
    message: str
    iterations: int
    function_evals: int

    def as_dict(self) -> dict:
        return {
            "trial": self.trial,
            "volume": self.volume,
            "valid": self.volume is not None,
            "params": self.params.tolist(),
            "success": self.success,
            "message": self.message,
            "iterations": self.iterations,
            "function_evals": self.function_evals,
        }


@dataclass(frozen=True)
class PolyhedronOptimizationResult:
    """Best configuration and all trial summaries."""

    n_vertices: int
    best_volume: float
    best_params: np.ndarray
    finite_vertices: np.ndarray
    finite_triangulation: np.ndarray
    structure: dict
    trials: tuple[OptimizationTrial, ...]

    @property
    def boundary_triangulation(self) -> np.ndarray:
        return self.structure["boundary_triangulation"]


def optimization_result_to_dict(
    result: PolyhedronOptimizationResult,
    *,
    source: str,
    timestamp: str | None = None,
    settings: Mapping[str, object] | None = None,
) -> dict:
    """Serialize an optimization result using the shared CLI/GUI schema.

    ``source`` identifies the adapter that initiated the run. Additional run
    settings are nested under ``metadata.settings`` so callers cannot replace
    schema-owned fields accidentally.
    """
    if not source or not source.strip():
        raise ValueError("source must be a nonempty string")

    structure = {
        key: value
        for key, value in result.structure.items()
        if key != "boundary_triangulation"
    }
    finite = result.finite_triangulation.astype(int, copy=False).tolist()
    boundary = result.boundary_triangulation.astype(int, copy=False).tolist()
    volumes = [
        float(trial.volume) for trial in result.trials if trial.volume is not None
    ]
    run_settings = dict(settings or {})
    metadata = {
        "timestamp": timestamp or datetime.now().isoformat(),
        "n_vertices": int(result.n_vertices),
        "objective": "vectorized_scipy_bloch_wigner",
        "implicit_infinity_index": int(len(result.finite_vertices)),
        "settings": run_settings,
    }
    # Schema v1 owns a nested settings namespace. Retain the settings exposed
    # by the two legacy writers at metadata level so existing result readers
    # can migrate without losing configuration data.
    for legacy_key in ("n_trials", "maxiter", "popsize", "workers", "seed_base"):
        if legacy_key in run_settings:
            metadata[legacy_key] = run_settings[legacy_key]
    return {
        "schema_version": 1,
        "source": source.strip(),
        "metadata": metadata,
        "best": {
            "volume": float(result.best_volume),
            "params": result.best_params.tolist(),
            "vertices_real": result.finite_vertices.real.tolist(),
            "vertices_imag": result.finite_vertices.imag.tolist(),
            "structure": structure,
            # Historical consumers used ``triangulation`` for the finite disk.
            "triangulation": finite,
            "finite_delaunay_triangulation": finite,
            "boundary_triangulation": boundary,
        },
        "statistics": {
            "n_vertices": int(result.n_vertices),
            "n_faces": int(structure["n_faces"]),
            "n_finite_faces": int(structure["n_finite_faces"]),
            "best_volume": float(result.best_volume),
            "mean_volume": float(np.mean(volumes)),
            "std_volume": float(np.std(volumes)),
            "all_volumes": volumes,
            "n_valid_trials": len(volumes),
            "n_invalid_trials": len(result.trials) - len(volumes),
        },
        "all_trials": [trial.as_dict() for trial in result.trials],
    }


def optimize_ideal_polyhedron(
    n_vertices: int,
    *,
    n_trials: int = 10,
    maxiter: int = 200,
    popsize: int = 15,
    seed: int = 42,
    workers: int = 1,
    trial_callback: Callable[[OptimizationTrial], None] | None = None,
) -> PolyhedronOptimizationResult:
    """Maximize ideal-polyhedron volume with repeated differential evolution.

    ``popsize`` retains SciPy's meaning: it is a multiplier of the parameter
    dimension, not an absolute population count.  ``workers=-1`` uses every
    available CPU; values greater than one use that many worker processes for
    differential evolution.  L-BFGS-B polishing remains serial because this
    fine-grained objective costs less than repeated process dispatch.
    """
    if n_vertices < 4:
        raise ValueError("n_vertices must be at least 4")
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if maxiter < 1:
        raise ValueError("maxiter must be at least 1")
    if popsize < 1:
        raise ValueError("popsize must be at least 1")
    if workers == 0 or workers < -1:
        raise ValueError("workers must be -1 or a positive integer")

    n_free = n_vertices - 3
    bounds = [(0.1, np.pi - 0.1), (0.0, 2.0 * np.pi)] * n_free
    objective = partial(negative_volume_objective, n_vertices=n_vertices)

    trials: list[OptimizationTrial] = []
    best_volume = -np.inf
    best_params: np.ndarray | None = None
    for trial_index in range(n_trials):
        serial_polish = workers != 1
        result = differential_evolution(
            objective,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=seed + trial_index,
            polish=not serial_polish,
            disp=False,
            workers=workers,
            updating="deferred" if workers != 1 else "immediate",
        )
        if serial_polish:
            _serially_polish_differential_evolution_result(
                result,
                objective,
                bounds,
            )
        objective_value = float(result.fun)
        trial_volume = (
            -objective_value
            if np.isfinite(objective_value) and objective_value < 0.0
            else None
        )
        trial = OptimizationTrial(
            trial=trial_index + 1,
            volume=trial_volume,
            params=np.asarray(result.x, dtype=np.float64).copy(),
            success=bool(result.success),
            message=str(result.message),
            iterations=int(result.nit),
            function_evals=int(result.nfev),
        )
        trials.append(trial)
        if trial_callback is not None:
            trial_callback(trial)
        if trial.volume is not None and trial.volume > best_volume:
            best_volume = trial.volume
            best_params = trial.params.copy()

    if best_params is None or not np.isfinite(best_volume) or best_volume <= 0.0:
        raise RuntimeError("optimization did not produce a valid configuration")

    finite_vertices = parameters_to_complex(best_params, n_vertices)
    finite_triangulation = delaunay_triangulation_indices(finite_vertices)
    structure = analyze_ideal_polyhedron_structure(
        finite_vertices, finite_triangulation
    )
    return PolyhedronOptimizationResult(
        n_vertices=n_vertices,
        best_volume=best_volume,
        best_params=best_params,
        finite_vertices=finite_vertices,
        finite_triangulation=finite_triangulation,
        structure=structure,
        trials=tuple(trials),
    )


__all__ = [
    "OBJECTIVE_PENALTY",
    "OptimizationTrial",
    "PolyhedronOptimizationResult",
    "analyze_ideal_polyhedron_structure",
    "complete_boundary_triangulation",
    "negative_volume_objective",
    "optimize_ideal_polyhedron",
    "optimization_result_to_dict",
    "parameters_to_complex",
    "spherical_to_complex",
]
