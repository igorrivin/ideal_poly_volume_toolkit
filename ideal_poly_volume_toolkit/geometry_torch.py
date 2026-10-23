"""Differentiable PyTorch helpers for the H³ geometry kernel.

This module owns the Torch-only implementation that historically lived in
``geometry.py``.  Importing :mod:`ideal_poly_volume_toolkit.geometry` does not
import Torch; its compatibility functions load this module only when a
differentiable API is used.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - covered via lazy facade
    raise ImportError(
        "PyTorch-backed geometry requires the 'torch' extra; install it with "
        "`python -m pip install 'ideal_poly_volume_toolkit[torch]'`."
    ) from exc

from .geometry import bloch_wigner


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Share a detached CPU tensor with NumPy without Python scalar loops."""
    return np.from_dlpack(tensor.detach().cpu())


def _angles_for_triangle_torch(z1, z2, z3):
    def angle_at(a, b, c):
        u = torch.stack(((b - a).real, (b - a).imag), dim=-1)
        v = torch.stack(((c - a).real, (c - a).imag), dim=-1)
        dot = (u * v).sum(dim=-1)
        cross = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
        return torch.atan2(torch.abs(cross), dot)

    a1 = angle_at(z1, z2, z3)
    a2 = angle_at(z2, z3, z1)
    a3 = angle_at(z3, z1, z2)
    return a1, a2, a3


def _series_frequencies_torch(theta: torch.Tensor, n: int) -> torch.Tensor:
    if n < 1:
        raise ValueError("n must be a positive integer")
    k = torch.arange(1, n + 1, device=theta.device, dtype=theta.dtype)
    return k.reshape((n,) + (1,) * theta.ndim)


def _lob_value_series_torch(theta: torch.Tensor, n: int = 64) -> torch.Tensor:
    k = _series_frequencies_torch(theta, n)
    return 0.5 * torch.sum(torch.sin(2 * k * theta) / (k * k), dim=0)


def _lob_derivative_series_torch(theta: torch.Tensor, n: int = 64) -> torch.Tensor:
    """Derivative of the same truncated series used in the forward pass."""
    k = _series_frequencies_torch(theta, n)
    return torch.sum(torch.cos(2 * k * theta) / k, dim=0)


class _LobFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, n: int = 64):
        ctx.save_for_backward(theta)
        ctx.n = int(n)
        return _lob_value_series_torch(theta, n)

    @staticmethod
    def backward(ctx, gout):
        (theta,) = ctx.saved_tensors
        derivative = _lob_derivative_series_torch(theta, ctx.n)
        return gout * derivative, None


def lob_fast(theta: torch.Tensor, n: int = 64) -> torch.Tensor:
    """Evaluate the truncated Lobachevsky series with Torch autograd."""
    return _LobFn.apply(theta, n)


class _BlochWignerFn(torch.autograd.Function):
    """Autograd bridge for the real-valued Bloch–Wigner dilogarithm."""

    @staticmethod
    def forward(ctx, z_real, z_imag):
        ctx.save_for_backward(z_real, z_imag)

        # DLPack avoids Python scalar loops and PyTorch's optional NumPy bridge;
        # the latter can be unavailable when Torch and NumPy use different ABIs.
        z_real_np = np.asarray(_tensor_to_numpy(z_real), dtype=np.float64)
        z_imag_np = np.asarray(_tensor_to_numpy(z_imag), dtype=np.float64)
        z_complex = z_real_np + 1j * z_imag_np
        result = np.asarray(bloch_wigner(z_complex), dtype=np.float64)
        return torch.from_dlpack(result).to(
            dtype=z_real.dtype,
            device=z_real.device,
        )

    @staticmethod
    def backward(ctx, grad_output):
        """Apply the real differential of the Bloch–Wigner function."""
        z_real, z_imag = ctx.saved_tensors

        x = z_real
        y = z_imag
        radius_sq = x * x + y * y
        complement_sq = (1.0 - x) ** 2 + y * y
        if torch.any((radius_sq == 0.0) | (complement_sq == 0.0)):
            raise RuntimeError("the Bloch-Wigner gradient is singular at z=0 and z=1")

        # Directly taking log((1-x)^2+y^2) loses the linear term in x near
        # z=0; the symmetric issue occurs for log(x^2+y^2) near z=1.  Evaluate
        # those two cancellation-prone regions with log1p.  Mask the unused
        # log1p arguments to zero so autograd never encounters a non-selected
        # singular branch such as log1p(-1).
        near_one = complement_sq < 0.25
        delta_x = x - 1.0
        radius_log1p_argument = torch.where(
            near_one,
            2.0 * delta_x + delta_x * delta_x + y * y,
            torch.zeros_like(x),
        )
        log_radius = torch.where(
            near_one,
            0.5 * torch.log1p(radius_log1p_argument),
            0.5 * torch.log(radius_sq),
        )

        near_zero = radius_sq < 0.25
        complement_log1p_argument = torch.where(
            near_zero,
            -2.0 * x + x * x + y * y,
            torch.zeros_like(x),
        )
        log_complement = torch.where(
            near_zero,
            0.5 * torch.log1p(complement_log1p_argument),
            0.5 * torch.log(complement_sq),
        )

        derivative_x = (
            -y * log_radius / complement_sq + y * log_complement / radius_sq
        )
        derivative_y = (
            -(1.0 - x) * log_radius / complement_sq
            - x * log_complement / radius_sq
        )

        grad_real = grad_output * derivative_x
        grad_imag = grad_output * derivative_y
        return grad_real, grad_imag


def bloch_wigner_torch(z_real, z_imag):
    """Evaluate Bloch–Wigner values with PyTorch autograd support."""
    z_real, z_imag = torch.broadcast_tensors(z_real, z_imag)
    return _BlochWignerFn.apply(z_real, z_imag)


def cross_ratio_torch(
    z1_real, z1_imag, z2_real, z2_imag, z3_real, z3_imag, z4_real, z4_imag
):
    """Compute the complex cross ratio from separate real and imaginary tensors."""
    num1_real = z1_real - z3_real
    num1_imag = z1_imag - z3_imag
    num2_real = z2_real - z4_real
    num2_imag = z2_imag - z4_imag

    num_real = num1_real * num2_real - num1_imag * num2_imag
    num_imag = num1_real * num2_imag + num1_imag * num2_real

    den1_real = z1_real - z4_real
    den1_imag = z1_imag - z4_imag
    den2_real = z2_real - z3_real
    den2_imag = z2_imag - z3_imag

    den_real = den1_real * den2_real - den1_imag * den2_imag
    den_imag = den1_real * den2_imag + den1_imag * den2_real

    den_abs_sq = den_real**2 + den_imag**2
    if torch.any(den_abs_sq == 0.0):
        raise ValueError("cross ratio is undefined because its denominator is zero")
    cr_real = (num_real * den_real + num_imag * den_imag) / den_abs_sq
    cr_imag = (num_imag * den_real - num_real * den_imag) / den_abs_sq
    return cr_real, cr_imag


def triangle_volume_from_points_torch(z1t, z2t, z3t, series_terms=64):
    """Accumulate one ideal-tetrahedron volume using Torch operations."""
    a1, a2, a3 = _angles_for_triangle_torch(z1t, z2t, z3t)
    return (
        lob_fast(a1, series_terms)
        + lob_fast(a2, series_terms)
        + lob_fast(a3, series_terms)
    )


def triangle_volume_bloch_wigner(z1t, z2t, z3t):
    """Compute one ideal-tetrahedron volume with the Torch BW bridge."""
    z1_real = z1t.real
    z1_imag = z1t.imag
    z2_real = z2t.real
    z2_imag = z2t.imag
    z3_real = z3t.real
    z3_imag = z3t.imag

    # For infinity as the fourth point, the cross ratio reduces to
    # (z1 - z3) / (z2 - z3).
    diff13_real = z1_real - z3_real
    diff13_imag = z1_imag - z3_imag
    diff23_real = z2_real - z3_real
    diff23_imag = z2_imag - z3_imag

    den_abs_sq = diff23_real**2 + diff23_imag**2
    if torch.any(den_abs_sq == 0.0):
        raise ValueError("tetrahedron is degenerate because z2 equals z3")
    cr_real = (diff13_real * diff23_real + diff13_imag * diff23_imag) / den_abs_sq
    cr_imag = (diff13_imag * diff23_real - diff13_real * diff23_imag) / den_abs_sq

    return torch.abs(bloch_wigner_torch(cr_real, cr_imag))


__all__ = [
    "bloch_wigner_torch",
    "cross_ratio_torch",
    "lob_fast",
    "triangle_volume_bloch_wigner",
    "triangle_volume_from_points_torch",
]
