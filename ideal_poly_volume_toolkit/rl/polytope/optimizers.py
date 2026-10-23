"""
Polytope-specific InnerOptimizer implementations.

One compatibility class per dimension, using the best available local
method for that objective.  The
dimension-specific torch imports are done lazily inside `run()` so the
package stays importable without torch for users who only need the
volume-computation layer.

Design rationale: the previous monolithic `LocalLBFGS` had a four-way
`if self.dimension == k` ladder inside `run()`.  Splitting it into four
small classes removes the dispatch while keeping the per-dimension code
local and lets each dimension carry its own configuration (step count,
learning rate, etc.).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import math
from numbers import Integral, Real
import time
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

from ..generic import (
    InnerOptimizer,
    OptimizationTask,
    OptimizerRun,
    OptimizerRunError,
)
from .types import PolytopeSolution


# =====================================================================
# Common infrastructure
# =====================================================================

def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Return a copy with each row on the unit sphere."""
    arr = np.asarray(arr, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms == 0.0) or not np.all(np.isfinite(norms)):
        raise ValueError("sphere points must be finite and nonzero")
    return arr / norms


def _normalize_rows_unless_already_unit(arr: np.ndarray) -> np.ndarray:
    """Preserve an already-normalized float64 start byte for byte."""

    points = np.array(arr, dtype=np.float64, copy=True)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    if np.any(norms == 0.0) or not np.all(np.isfinite(norms)):
        raise ValueError("sphere points must be finite and nonzero")
    if float(np.max(np.abs(norms - 1.0))) <= 32.0 * np.finfo(np.float64).eps:
        return points
    return points / norms


def _positive_finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number")
    return result


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


class _LBFGSHyperbolicBase(InnerOptimizer):
    """Common scaffolding for dimension-specific hyperbolic optimizers.

    The class name is retained for compatibility with saved controllers and
    downstream imports; subclasses use the local method appropriate to their
    objective.

    ``budget`` is a strict total-work cap.  Subclasses return a canonical
    ``solution_value`` whose objective call is included in that cap, so the
    environment never needs an unbudgeted final score.
    """

    dimension: int
    default_budget = 8

    def __init__(self, *, steps: int = 5, name: str | None = None):
        self.steps = int(steps)
        self._name = name or f"lbfgs_hyp_d{self.dimension}"

    @property
    def name(self) -> str:
        return self._name

    def supports(self, task: OptimizationTask) -> bool:
        from .task import IdealPolytopeTask

        return isinstance(task, IdealPolytopeTask) and task.d == self.dimension

    # Subclass hook ----------------------------------------------------------
    def _run(self, solution: PolytopeSolution, task: OptimizationTask,
             budget: int, rng: np.random.Generator) -> OptimizerRun:
        raise NotImplementedError

    # InnerOptimizer contract ------------------------------------------------
    def run(self, solution: Any, task: OptimizationTask,
            budget: int, rng: np.random.Generator) -> OptimizerRun:
        if not isinstance(solution, PolytopeSolution):
            raise TypeError(
                f"{self.name} expects a PolytopeSolution; got {type(solution).__name__}"
            )
        if not self.supports(task):
            raise ValueError(
                f"{self.name} only supports d={self.dimension}; "
                f"task has d={getattr(task, 'd', None)}"
            )
        budget = int(budget)
        if budget < 1:
            raise ValueError("optimizer budget must be at least 1")
        return self._run(solution, task, budget, rng)


def _run_powell(
    solution: PolytopeSolution,
    task: OptimizationTask,
    budget: int,
    rng: np.random.Generator,
) -> OptimizerRun:
    """Run strict-budget Powell against the task's canonical objective."""
    from ..inner_optimizers import ScipyMinimizeOptimizer

    return ScipyMinimizeOptimizer(method="Powell").run(
        solution, task, budget=budget, rng=rng
    )


def _run_projected_adam(
    solution: PolytopeSolution,
    task: OptimizationTask,
    budget: int,
    *,
    lr: float,
    gradient_objective: Callable[[Any, Any], Any],
) -> OptimizerRun:
    """Use ``budget - 1`` fresh-topology gradients and one canonical score."""
    import torch

    triangulate = getattr(task, "triangulate", None)
    if triangulate is None:
        raise TypeError(f"{type(task).__name__} does not expose triangulate()")

    pts_opt = torch.tensor(
        _normalize_rows(solution.vertices), dtype=torch.float64, requires_grad=True
    )
    optimizer = torch.optim.Adam([pts_opt], lr=lr)
    gradient_evals = 0
    last_valid = pts_opt.detach().clone()

    for _ in range(budget - 1):
        try:
            optimizer.zero_grad(set_to_none=True)
            normed = pts_opt / torch.linalg.vector_norm(
                pts_opt, dim=1, keepdim=True
            )

            # Delaunay combinatorics can change after every vertex update.
            simplices = triangulate(np.from_dlpack(normed.detach().cpu()))
            simps_t = torch.as_tensor(simplices, dtype=torch.long)
            gradient_evals += 1
            objective = gradient_objective(normed, simps_t)
            if not bool(torch.isfinite(objective).item()):
                break
            (-objective).backward()
            optimizer.step()
            with torch.no_grad():
                norms = torch.linalg.vector_norm(pts_opt, dim=1, keepdim=True)
                if not bool(
                    torch.all(torch.isfinite(norms) & (norms > 0.0)).item()
                ):
                    pts_opt.copy_(last_valid)
                    break
                pts_opt.div_(norms)
                last_valid.copy_(pts_opt)
        except Exception:
            with torch.no_grad():
                pts_opt.copy_(last_valid)
            break

    try:
        final = _normalize_rows(np.from_dlpack(pts_opt.detach().cpu()))
    except Exception as exc:
        partial = OptimizerRun(
            solution=PolytopeSolution(
                vertices=_normalize_rows(solution.vertices),
                simplices=[],
            ),
            gradient_evals=gradient_evals,
        )
        raise OptimizerRunError(
            f"projected Adam result conversion failed: {exc}", partial
        ) from exc
    # Vertex motion invalidates any cached entry topology.  The canonical task
    # builds the topology it needs from these final vertices.
    new_solution = PolytopeSolution(vertices=final, simplices=[])
    try:
        canonical_value = float(task.evaluate(new_solution))
    except Exception as exc:
        partial = OptimizerRun(
            solution=new_solution,
            gradient_evals=gradient_evals,
            objective_evals=1,
        )
        raise OptimizerRunError(
            f"projected Adam canonical scoring failed: {exc}", partial
        ) from exc
    return OptimizerRun(
        solution=new_solution,
        values=[canonical_value],
        solution_value=canonical_value,
        objective_evals=1,
        gradient_evals=gradient_evals,
    )


# =====================================================================
# d = 3 (Bloch–Wigner H^3 volume)
# =====================================================================

class LBFGSHyperbolicD3(_LBFGSHyperbolicBase):
    """H^3 volume via stereographic Delaunay + Bloch–Wigner.  scipy Powell
    on the closed-form H^3 objective — matches the d=3 branch of the
    original LocalLBFGS."""

    dimension = 3
    # The old default of three outer steps allowed up to 3 * 30 calls.
    default_budget = 90

    def _run(self, solution, task, budget, rng):
        return _run_powell(solution, task, budget, rng)


# =====================================================================
# d = 4 (closed-form H^4 polytope volume)
# =====================================================================

class LBFGSHyperbolicD4(_LBFGSHyperbolicBase):
    """H^4 volume via the closed-form `ideal_polytope_volume_4d`; scipy
    Powell optimizer (matches the d=4 branch of the original LocalLBFGS)."""

    dimension = 4
    # The old default of three outer steps allowed up to 3 * 20 calls.
    default_budget = 60

    def _run(self, solution, task, budget, rng):
        return _run_powell(solution, task, budget, rng)


@dataclass
class H4OptimizerRun(OptimizerRun):
    """Work-counted H4 run with JSON-native branch diagnostics."""

    diagnostics: dict[str, Any] = field(default_factory=dict)


class _GradientBudgetExhausted(RuntimeError):
    """Private sentinel used to stop SciPy before an extra oracle call."""


def _fixed_pole_project(
    vertices: np.ndarray,
    pole_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project labelled S3 vertices from one fixed pole.

    Returns the finite R3 coordinates, their original vertex labels, and the
    involutive orthogonal map taking the selected pole to ``(0, 0, 0, 1)``.
    The label array is deliberately explicit so inverse projection cannot
    silently permute vertices.
    """

    points = _normalize_rows(vertices)
    n_vertices = len(points)
    if not 0 <= pole_index < n_vertices:
        raise ValueError(f"pole index {pole_index} is outside [0, {n_vertices})")

    pole = points[pole_index]
    north = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    displacement = pole - north
    if np.array_equal(pole, north):
        transform = np.eye(4, dtype=np.float64)
    else:
        displacement_norm = float(np.linalg.norm(displacement))
        if not np.isfinite(displacement_norm) or displacement_norm == 0.0:
            raise FloatingPointError("could not construct the fixed-pole chart")
        direction = displacement / displacement_norm
        transform = np.eye(4, dtype=np.float64) - 2.0 * np.outer(
            direction, direction
        )

    mapped_pole = pole @ transform
    if not np.allclose(mapped_pole, north, rtol=0.0, atol=5.0e-14):
        raise FloatingPointError("fixed-pole transform did not map the pole exactly")

    finite_indices = np.delete(np.arange(n_vertices, dtype=np.int64), pole_index)
    rotated = points[finite_indices] @ transform
    denominators = 1.0 - rotated[:, 3]
    if not np.all(np.isfinite(denominators)) or np.any(
        denominators <= 128.0 * np.finfo(np.float64).eps
    ):
        raise ValueError("a finite vertex is repeated or too close to the chart pole")
    projected = rotated[:, :3] / denominators[:, None]
    if not np.all(np.isfinite(projected)):
        raise FloatingPointError("fixed-pole projection produced non-finite points")
    return projected, finite_indices, transform


def _projected_similarity_gauge_numpy(points: np.ndarray) -> np.ndarray:
    """NumPy counterpart of the differentiable Torch similarity gauge."""

    projected = np.asarray(points, dtype=np.float64)
    if projected.ndim != 2 or projected.shape[1] != 3:
        raise ValueError("projected points must have shape (M, 3)")
    if not np.all(np.isfinite(projected)):
        raise ValueError("projected points must be finite")
    centered = projected - np.mean(projected, axis=0, keepdims=True)
    mean_squared_radius = float(np.mean(np.sum(centered * centered, axis=1)))
    coordinate_scale = float(np.max(np.abs(projected)))
    floor = 128.0 * np.finfo(np.float64).eps * coordinate_scale * coordinate_scale
    if not np.isfinite(mean_squared_radius) or mean_squared_radius <= floor:
        raise ValueError("projected points have insufficient finite spread")
    gauged = centered / math.sqrt(mean_squared_radius)
    if not np.all(np.isfinite(gauged)):
        raise FloatingPointError("similarity gauging produced non-finite points")
    return gauged


def _fixed_pole_unproject(
    projected: np.ndarray,
    finite_indices: np.ndarray,
    transform: np.ndarray,
    *,
    n_vertices: int,
    pole_index: int,
) -> np.ndarray:
    """Inverse-project finite points without changing their original labels."""

    finite = np.asarray(projected, dtype=np.float64)
    labels = np.asarray(finite_indices, dtype=np.int64)
    if finite.shape != (n_vertices - 1, 3):
        raise ValueError("projected points have the wrong shape for this chart")
    if labels.shape != (n_vertices - 1,) or set(labels.tolist()) != (
        set(range(n_vertices)) - {pole_index}
    ):
        raise ValueError("finite_indices do not preserve the chart labels")
    squared_norms = np.sum(finite * finite, axis=1, keepdims=True)
    denominators = squared_norms + 1.0
    rotated = np.concatenate(
        (
            2.0 * finite / denominators,
            (squared_norms - 1.0) / denominators,
        ),
        axis=1,
    )
    vertices = np.empty((n_vertices, 4), dtype=np.float64)
    vertices[labels] = rotated @ transform
    vertices[pole_index] = np.array([0.0, 0.0, 0.0, 1.0]) @ transform
    return _normalize_rows(vertices)


def _topology_digest(
    simplices: np.ndarray,
    finite_indices: np.ndarray,
) -> str:
    """Stable digest of labelled tetrahedra, independent of Qhull ordering."""

    labelled = np.sort(finite_indices[np.asarray(simplices, dtype=np.int64)], axis=1)
    canonical = np.asarray(
        sorted(tuple(int(label) for label in row) for row in labelled),
        dtype="<i8",
    )
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


class FreshTopologyADHyperbolicD4(_LBFGSHyperbolicBase):
    """Fresh-Delaunay autograd optimizer for unconstrained ideal H4 volume.

    Each branch fixes one labelled ideal vertex at infinity, optimizes the
    remaining points in a similarity-gauged R3 chart, and reconstructs the
    full labelled S3 configuration before a counted canonical score.  The
    Delaunay tetrahedralization is rebuilt on every joint objective/gradient
    oracle call, so the search is free to cross topology walls.
    """

    dimension = 4
    default_budget = 1200
    _ALLOWED_BRANCH_MODES = ("lbfgs", "adam_lbfgs")

    def __init__(
        self,
        *,
        pole_indices: Sequence[int] | None = None,
        branch_modes: Sequence[str] = _ALLOWED_BRANCH_MODES,
        adam_steps: int = 64,
        adam_lr: float = 0.03,
        name: str = "fresh_topology_ad_h4",
    ) -> None:
        super().__init__(steps=1, name=name)
        if pole_indices is None:
            self.pole_indices: tuple[int, ...] | None = None
        else:
            if isinstance(pole_indices, (str, bytes)) or isinstance(
                pole_indices, Integral
            ):
                raise TypeError("pole_indices must be a sequence of distinct integers")
            parsed_poles: list[int] = []
            for value in pole_indices:
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise TypeError("every pole index must be an integer")
                parsed_poles.append(int(value))
            if not parsed_poles:
                raise ValueError("pole_indices must not be empty")
            if len(set(parsed_poles)) != len(parsed_poles):
                raise ValueError("pole_indices must not contain duplicates")
            self.pole_indices = tuple(parsed_poles)

        if isinstance(branch_modes, (str, bytes)):
            raise TypeError("branch_modes must be a sequence of mode names")
        parsed_modes = tuple(branch_modes)
        if not parsed_modes:
            raise ValueError("branch_modes must not be empty")
        if any(not isinstance(mode, str) for mode in parsed_modes):
            raise TypeError("every branch mode must be a string")
        invalid_modes = [
            mode for mode in parsed_modes if mode not in self._ALLOWED_BRANCH_MODES
        ]
        if invalid_modes:
            raise ValueError(
                "branch_modes entries must be 'lbfgs' or 'adam_lbfgs'; "
                f"received {invalid_modes[0]!r}"
            )
        if len(set(parsed_modes)) != len(parsed_modes):
            raise ValueError("branch_modes must not contain duplicates")
        self.branch_modes = parsed_modes
        self.adam_steps = _nonnegative_integer(adam_steps, name="adam_steps")
        self.adam_lr = _positive_finite_real(adam_lr, name="adam_lr")

    def _resolved_poles(self, n_vertices: int) -> tuple[int, ...]:
        poles = (
            tuple(range(min(4, n_vertices)))
            if self.pole_indices is None
            else self.pole_indices
        )
        for pole in poles:
            if not 0 <= pole < n_vertices:
                raise ValueError(
                    f"pole index {pole} is outside [0, {n_vertices})"
                )
        return poles

    @staticmethod
    def _load_gradient_backend() -> tuple[Any, Callable[..., Any]]:
        """Load the explicit AD backend or fail before any counted work."""

        try:
            import torch

            from ...geometry_4d_torch import (
                ideal_polytope_volume_4d_from_projected,
            )
        except ImportError as exc:
            raise ImportError(
                "fresh_topology_ad requires the optional PyTorch H4 backend; "
                "install the project's torch extra before selecting this optimizer"
            ) from exc
        return torch, ideal_polytope_volume_4d_from_projected

    @staticmethod
    def _branch_plan(
        poles: tuple[int, ...],
        modes: tuple[str, ...],
        budget: int,
    ) -> tuple[list[tuple[int, str]], list[int]]:
        requested = [(pole, mode) for pole in poles for mode in modes]
        branch_count = min(len(requested), (budget - 1) // 2)
        branches = requested[:branch_count]
        if not branches:
            return [], []
        gradient_work = budget - 1 - branch_count
        quotient, remainder = divmod(gradient_work, branch_count)
        allocations = [
            quotient + (1 if index < remainder else 0)
            for index in range(branch_count)
        ]
        return branches, allocations

    def _run_branch(
        self,
        vertices: np.ndarray,
        *,
        pole_index: int,
        mode: str,
        gradient_budget: int,
    ) -> tuple[np.ndarray, dict[str, Any], int]:
        """Run one chart branch and return its best search-kernel iterate."""

        torch, ideal_polytope_volume_4d_from_projected = (
            self._load_gradient_backend()
        )

        projected, finite_indices, transform = _fixed_pole_project(
            vertices, pole_index
        )
        projected = _projected_similarity_gauge_numpy(projected)
        shape = projected.shape
        current = projected.reshape(-1).copy()
        best_coordinates = current.copy()
        best_search_value: float | None = None
        gradient_evals = 0
        delaunay_evals = 0
        topology_changes = 0
        topology_hashes: set[str] = set()
        topology_examples: list[str] = []
        topology_trace = hashlib.sha256()
        first_topology: str | None = None
        last_topology: str | None = None
        previous_topology: str | None = None

        def oracle(vector: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal best_coordinates
            nonlocal best_search_value
            nonlocal delaunay_evals
            nonlocal gradient_evals
            nonlocal first_topology
            nonlocal last_topology
            nonlocal previous_topology
            nonlocal topology_changes

            if gradient_evals >= gradient_budget:
                raise _GradientBudgetExhausted
            gradient_evals += 1
            raw = np.asarray(vector, dtype=np.float64).reshape(shape)
            gauged = _projected_similarity_gauge_numpy(raw)
            delaunay_evals += 1
            simplices = np.asarray(Delaunay(gauged).simplices, dtype=np.int64)
            digest = _topology_digest(simplices, finite_indices)
            topology_trace.update(bytes.fromhex(digest))
            if first_topology is None:
                first_topology = digest
            last_topology = digest
            if previous_topology is not None and digest != previous_topology:
                topology_changes += 1
            previous_topology = digest
            if digest not in topology_hashes:
                topology_hashes.add(digest)
                if len(topology_examples) < 16:
                    topology_examples.append(digest)

            points_torch = torch.tensor(
                raw,
                dtype=torch.float64,
                requires_grad=True,
            )
            simplices_torch = torch.tensor(simplices, dtype=torch.long)
            volume = ideal_polytope_volume_4d_from_projected(
                points_torch,
                simplices_torch,
                apply_similarity_gauge=True,
            )
            (gradient,) = torch.autograd.grad(volume, points_torch)
            value = float(volume.detach())
            gradient_array = gradient.detach().cpu().numpy().reshape(-1)
            if not np.isfinite(value) or not np.all(np.isfinite(gradient_array)):
                raise FloatingPointError("H4 gradient oracle returned non-finite data")
            if best_search_value is None or value > best_search_value:
                best_search_value = value
                best_coordinates = np.array(vector, dtype=np.float64, copy=True)
            return -value, -gradient_array

        status = "completed"
        error_type: str | None = None
        error_message: str | None = None
        scipy_status: int | None = None
        scipy_message: str | None = None
        adam_completed = 0
        try:
            if mode == "adam_lbfgs":
                first_moment = np.zeros_like(current)
                second_moment = np.zeros_like(current)
                for step_index in range(min(self.adam_steps, gradient_budget)):
                    _, gradient = oracle(current)
                    adam_completed += 1
                    first_moment = 0.9 * first_moment + 0.1 * gradient
                    second_moment = 0.999 * second_moment + 0.001 * gradient * gradient
                    corrected_first = first_moment / (1.0 - 0.9 ** (step_index + 1))
                    corrected_second = second_moment / (
                        1.0 - 0.999 ** (step_index + 1)
                    )
                    current -= self.adam_lr * corrected_first / (
                        np.sqrt(corrected_second) + 1.0e-8
                    )
                    current = _projected_similarity_gauge_numpy(
                        current.reshape(shape)
                    ).reshape(-1)

            remaining = gradient_budget - gradient_evals
            if remaining > 0:
                try:
                    result = minimize(
                        oracle,
                        current,
                        jac=True,
                        method="L-BFGS-B",
                        options={
                            "maxfun": remaining,
                            "maxiter": remaining,
                            "maxls": 20,
                            "ftol": 1.0e-13,
                            "gtol": 1.0e-10,
                        },
                    )
                    scipy_status = int(result.status)
                    scipy_message = str(result.message)
                    if not bool(result.success):
                        status = (
                            "budget_exhausted"
                            if gradient_evals >= gradient_budget
                            else "stopped"
                        )
                except _GradientBudgetExhausted:
                    status = "budget_exhausted"
            elif gradient_evals >= gradient_budget:
                status = "budget_exhausted"
        except _GradientBudgetExhausted:
            status = "budget_exhausted"
        except Exception as exc:
            status = "failed"
            error_type = type(exc).__name__
            error_message = str(exc)

        try:
            if best_search_value is None:
                candidate = np.array(vertices, dtype=np.float64, copy=True)
            else:
                best_projected = _projected_similarity_gauge_numpy(
                    best_coordinates.reshape(shape)
                )
                candidate = _fixed_pole_unproject(
                    best_projected,
                    finite_indices,
                    transform,
                    n_vertices=len(vertices),
                    pole_index=pole_index,
                )
        except Exception as exc:
            candidate = np.array(vertices, dtype=np.float64, copy=True)
            status = "failed"
            error_type = type(exc).__name__
            error_message = str(exc)
        diagnostics: dict[str, Any] = {
            "pole_index": int(pole_index),
            "mode": mode,
            "gradient_budget": int(gradient_budget),
            "gradient_evals": int(gradient_evals),
            "status": status,
            "adam_steps_completed": int(adam_completed),
            "scipy_status": scipy_status,
            "scipy_message": scipy_message,
            "error_type": error_type,
            "error_message": error_message,
            "search_best_value": best_search_value,
            "canonical_endpoint_value": None,
            "finite_vertex_labels": [int(label) for label in finite_indices],
            "topology_evaluations": int(delaunay_evals),
            "topology_change_count": int(topology_changes),
            "unique_topology_count": len(topology_hashes),
            "topology_first_sha256": first_topology,
            "topology_last_sha256": last_topology,
            "topology_trace_sha256": (
                topology_trace.hexdigest() if delaunay_evals else None
            ),
            "topology_signature_sha256_examples": topology_examples,
        }
        return candidate, diagnostics, gradient_evals

    def _run(self, solution, task, budget, rng):
        # An explicit AD selection must never degrade into a successful-looking
        # no-op merely because its optional backend is absent.
        self._load_gradient_backend()
        del rng  # Every branch is deterministic from its submitted coordinates.
        vertices = _normalize_rows_unless_already_unit(solution.vertices)
        poles = self._resolved_poles(len(vertices))
        branch_plan, gradient_allocations = self._branch_plan(
            poles, self.branch_modes, budget
        )
        wall_start = time.perf_counter()

        diagnostics: dict[str, Any] = {
            "selector": "fresh_topology_ad",
            "requested_branch_count": int(len(poles) * len(self.branch_modes)),
            "executed_branch_count": int(len(branch_plan)),
            "completed_branch_count": 0,
            "failed_branch_count": 0,
            "canonical_scored_branch_count": 0,
            "pole_indices": [int(pole) for pole in poles],
            "branch_modes": list(self.branch_modes),
            "adam_steps": int(self.adam_steps),
            "adam_lr": float(self.adam_lr),
            "budget": int(budget),
            "budget_allocation": {
                "initial_objective_evals": 1,
                "reserved_endpoint_objective_evals": int(len(branch_plan)),
                "gradient_evals_by_branch": [
                    int(value) for value in gradient_allocations
                ],
            },
            "branches": [],
        }

        initial_solution = PolytopeSolution(vertices=vertices.copy(), simplices=[])
        try:
            initial_score = float(task.evaluate(initial_solution))
        except Exception as exc:
            partial = H4OptimizerRun(
                solution=initial_solution,
                objective_evals=1,
                diagnostics=diagnostics,
                wall_time=time.perf_counter() - wall_start,
            )
            raise OptimizerRunError(
                f"fresh-topology H4 initial scoring failed: {exc}", partial
            ) from exc
        if not np.isfinite(initial_score):
            partial = H4OptimizerRun(
                solution=initial_solution,
                objective_evals=1,
                diagnostics=diagnostics,
                wall_time=time.perf_counter() - wall_start,
            )
            raise OptimizerRunError(
                "fresh-topology H4 initial scoring was non-finite", partial
            )

        values = [initial_score]
        objective_evals = 1
        gradient_evals = 0
        best_solution = initial_solution
        best_score = initial_score

        for (pole_index, mode), branch_budget in zip(
            branch_plan, gradient_allocations, strict=True
        ):
            try:
                candidate, branch_diagnostics, used_gradients = self._run_branch(
                    vertices,
                    pole_index=pole_index,
                    mode=mode,
                    gradient_budget=branch_budget,
                )
            except Exception as exc:
                candidate = vertices.copy()
                used_gradients = 0
                branch_diagnostics = {
                    "pole_index": int(pole_index),
                    "mode": mode,
                    "gradient_budget": int(branch_budget),
                    "gradient_evals": 0,
                    "status": "failed",
                    "adam_steps_completed": 0,
                    "scipy_status": None,
                    "scipy_message": None,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "search_best_value": None,
                    "canonical_endpoint_value": None,
                    "finite_vertex_labels": [],
                    "topology_evaluations": 0,
                    "topology_change_count": 0,
                    "unique_topology_count": 0,
                    "topology_first_sha256": None,
                    "topology_last_sha256": None,
                    "topology_trace_sha256": None,
                    "topology_signature_sha256_examples": [],
                }
            gradient_evals += int(used_gradients)

            try:
                endpoint_vertices = _normalize_rows(candidate)
            except Exception as exc:
                endpoint_vertices = vertices.copy()
                branch_diagnostics["status"] = "failed"
                branch_diagnostics["error_type"] = type(exc).__name__
                branch_diagnostics["error_message"] = str(exc)
            endpoint_solution = PolytopeSolution(vertices=endpoint_vertices, simplices=[])
            objective_evals += 1
            try:
                endpoint_score = float(task.evaluate(endpoint_solution))
                if not np.isfinite(endpoint_score):
                    raise FloatingPointError("canonical endpoint score was non-finite")
            except Exception as exc:
                endpoint_score = None
                branch_diagnostics["status"] = "failed"
                branch_diagnostics["error_type"] = type(exc).__name__
                branch_diagnostics["error_message"] = str(exc)
            else:
                values.append(endpoint_score)
                branch_diagnostics["canonical_endpoint_value"] = endpoint_score
                diagnostics["canonical_scored_branch_count"] += 1
                if endpoint_score > best_score:
                    best_score = endpoint_score
                    best_solution = endpoint_solution

            if branch_diagnostics["status"] == "failed":
                diagnostics["failed_branch_count"] += 1
            else:
                diagnostics["completed_branch_count"] += 1
            diagnostics["branches"].append(branch_diagnostics)

        diagnostics["work"] = {
            "objective_evals": int(objective_evals),
            "surrogate_evals": 0,
            "gradient_evals": int(gradient_evals),
            "total_evals": int(objective_evals + gradient_evals),
        }
        wall_seconds = time.perf_counter() - wall_start
        run = H4OptimizerRun(
            solution=best_solution,
            values=values,
            solution_value=best_score,
            objective_evals=objective_evals,
            gradient_evals=gradient_evals,
            wall_time=wall_seconds,
            diagnostics=diagnostics,
        )
        if run.n_evals > budget:  # Defensive assertion around external SciPy.
            raise RuntimeError("fresh-topology H4 optimizer exceeded its work budget")
        return run


# =====================================================================
# d = 5 (Schläfli differential, torch Adam)
# =====================================================================

class LBFGSHyperbolicD5(_LBFGSHyperbolicBase):
    """H^5 optimization via gradient-only Adam on the Schläfli differential.

    The historical class name is retained for API compatibility.  A
    Schläfli differential is not a globally consistent scalar objective, so
    it must not be passed to a line-search method such as strong-Wolfe LBFGS.
    """

    dimension = 5
    # Nine gradients plus one canonical score, matching the old nominal work.
    default_budget = 10

    def __init__(self, *, steps: int = 5, lr: float = 0.01,
                 max_iter_per_step: int = 3, name: str | None = None):
        super().__init__(steps=steps, name=name)
        self.lr = float(lr)
        self.max_iter_per_step = int(max_iter_per_step)

    def _run(self, solution, task, budget, rng):
        from ...h5_volume_torch import schlafli_surrogate_vectorized

        return _run_projected_adam(
            solution,
            task,
            budget,
            lr=self.lr,
            gradient_objective=schlafli_surrogate_vectorized,
        )


# =====================================================================
# d = 6 (closed-form H^6 polytope volume, projected Adam)
# =====================================================================

class LBFGSHyperbolicD6(_LBFGSHyperbolicBase):
    """H^6 volume via `ideal_h6_polytope_volume` (Murakami-torch).  torch
    projected Adam with a fresh Delaunay triangulation at every gradient.
    The historical class name is retained for API compatibility."""

    dimension = 6
    # Nine gradients plus one canonical score; unlike LBFGS, this is exact.
    default_budget = 10

    def __init__(self, *, steps: int = 5, lr: float = 0.01,
                 max_iter_per_step: int = 3, name: str | None = None):
        super().__init__(steps=steps, name=name)
        self.lr = float(lr)
        self.max_iter_per_step = int(max_iter_per_step)

    def _run(self, solution, task, budget, rng):
        from ...murakami_torch import ideal_h6_polytope_volume
        from ...li2_torch import li2 as li2_pure

        def gradient_objective(points, simplices):
            return ideal_h6_polytope_volume(
                points, simplices, li2_fn=li2_pure
            )

        return _run_projected_adam(
            solution,
            task,
            budget,
            lr=self.lr,
            gradient_objective=gradient_objective,
        )


# =====================================================================
# Dispatch by dimension
# =====================================================================

_LBFGS_BY_DIM = {
    3: LBFGSHyperbolicD3,
    4: LBFGSHyperbolicD4,
    5: LBFGSHyperbolicD5,
    6: LBFGSHyperbolicD6,
}


def lbfgs_hyperbolic_for_dimension(d: int, **kwargs) -> _LBFGSHyperbolicBase:
    """Construct the compatible dimension-specific local optimizer.

    The function name is retained for callers of the original API. Raises
    ``KeyError`` for unsupported dimensions.
    """
    cls = _LBFGS_BY_DIM[d]
    return cls(**kwargs)
