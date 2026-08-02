"""
Pure-torch principal-branch dilogarithm Li2(z).

Uses Bernoulli series in w = -log(1-z) with argument reduction:
  |z| > 1:     Li2(z) = -Li2(1/z) - pi^2/6 - (1/2)log(-z)^2
  Re(z) > 1/2: Li2(z) = pi^2/6 - log(z)log(1-z) - Li2(1-z)

This maps any z to the reduced region |z| <= 1, Re(z) <= 1/2
where the Bernoulli series converges rapidly.

Reference: Ginsberg & de Doncker (1980), "Li2 in arbitrary precision".
"""

import math
import torch
from torch import nn

_PI2_OVER_6 = math.pi * math.pi / 6.0

# Bernoulli-series coefficients B_n / (n+1)! with B1 = -1/2.
_COEFFS64 = (
    1.0,
    -0.25,
    1.0 / 36.0,
    -1.0 / 3600.0,
    1.0 / 211680.0,
    -1.0 / 10886400.0,
    1.0 / 526901760.0,
)

_COEFFS128 = (
    1.0,
    -0.25,
    1.0 / 36.0,
    -1.0 / 3600.0,
    1.0 / 211680.0,
    -1.0 / 10886400.0,
    1.0 / 526901760.0,
    -691.0 / 16999766784000.0,
    1.0 / 1120863744000.0,
    -3617.0 / 181400588328960000.0,
)


def _real_to_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(x, -torch.zeros_like(x))


def _check_dtype(z: torch.Tensor) -> None:
    if z.dtype not in (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ):
        raise TypeError(
            "li2 supports float32/float64/complex64/complex128 inputs"
        )


def _coeffs_for(zc: torch.Tensor) -> torch.Tensor:
    if zc.dtype == torch.complex64:
        return zc.real.new_tensor(_COEFFS64).to(zc.dtype)
    if zc.dtype == torch.complex128:
        return zc.real.new_tensor(_COEFFS128).to(zc.dtype)
    raise TypeError(f"expected complex64/complex128, got {zc.dtype}")


def _li2_core_reduced(zc: torch.Tensor) -> torch.Tensor:
    """
    Evaluate Li2 on the reduced region:
        |z| <= 1 and Re(z) <= 1/2
    via the Bernoulli series in w = -log1p(-z).
    """
    coeffs = _coeffs_for(zc)
    w = -torch.log1p(-zc)
    w2 = w * w

    out = w + w2 * (coeffs[1] + w * coeffs[2])

    if coeffs.numel() > 3:
        p = coeffs[-1]
        for k in range(coeffs.numel() - 2, 2, -1):
            p = coeffs[k] + w2 * p
        out = out + (w * w2 * w2) * p

    return out


def _li2_principal_complex(zc: torch.Tensor) -> torch.Tensor:
    """Principal-branch dilogarithm for complex inputs."""
    pi2_6 = zc.new_tensor(_PI2_OVER_6)
    out = torch.zeros_like(zc)

    is_zero = (zc == 0)
    is_one = (zc == 1)
    regular = ~(is_zero | is_one)

    if regular.any():
        z = zc[regular].clone()
        acc = torch.zeros_like(z)
        sgn = torch.ones_like(z)

        # Li2(z) = -Li2(1/z) - pi^2/6 - 1/2 log(-z)^2
        m = (torch.abs(z) > 1)
        if m.any():
            L = torch.log(-z[m])
            c = -pi2_6 - 0.5 * L * L
            acc[m] = acc[m] + sgn[m] * c
            sgn[m] = -sgn[m]
            z[m] = 1.0 / z[m]

        # Li2(z) = pi^2/6 - log(z) log(1-z) - Li2(1-z)
        m = (z.real > 0.5)
        if m.any():
            c = pi2_6 - torch.log(z[m]) * torch.log1p(-z[m])
            acc[m] = acc[m] + sgn[m] * c
            sgn[m] = -sgn[m]
            z[m] = 1.0 - z[m]

        out[regular] = acc + sgn * _li2_core_reduced(z)

    if is_one.any():
        out[is_one] = pi2_6

    return out


def _li2_derivative_complex(zc: torch.Tensor) -> torch.Tensor:
    """d/dz Li2(z) = -log(1-z)/z, with removable value 1 at z=0."""
    safe_z = zc.clone()
    safe_z[zc == 0] = 1.0 + 0.0j
    deriv = -torch.log1p(-zc) / safe_z
    deriv[zc == 0] = 1.0 + 0.0j
    return deriv


class _Li2Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: torch.Tensor) -> torch.Tensor:
        _check_dtype(z)
        ctx.save_for_backward(z)

        if torch.is_complex(z):
            zc = z
        else:
            zc = _real_to_complex(z)

        return _li2_principal_complex(zc)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (z,) = ctx.saved_tensors

        if torch.is_complex(z):
            zc = z
            deriv = _li2_derivative_complex(zc)
            return grad_output * deriv.conj()

        zc = _real_to_complex(z)
        deriv = _li2_derivative_complex(zc)
        grad = grad_output * deriv.conj()
        return grad.real.to(z.dtype)


def li2(z: torch.Tensor) -> torch.Tensor:
    """
    Stable principal-branch dilogarithm.

    Real inputs are promoted to complex and the output is complex-valued.
    For strictly real x < 1, use li2(x).real if you want a real tensor.
    """
    return _Li2Fn.apply(z)


class Li2(nn.Module):
    """
    Thin nn.Module wrapper around li2(z).

    Parameters
    ----------
    cut : {"below", "above"}
        Only matters for exact real inputs:
          - "below": promote x to x - 0j
          - "above": promote x to x + 0j
    """

    def __init__(self, cut: str = "below"):
        super().__init__()
        if cut not in ("below", "above"):
            raise ValueError("cut must be 'below' or 'above'")
        self.cut = cut

    def extra_repr(self) -> str:
        return f"cut={self.cut!r}"

    def get_extra_state(self):
        return self.cut

    def set_extra_state(self, state):
        if state not in ("below", "above"):
            raise ValueError(f"invalid saved cut setting: {state!r}")
        self.cut = state

    def forward(self, z):
        if not isinstance(z, torch.Tensor):
            z = torch.as_tensor(z)

        _check_dtype(z)

        if torch.is_complex(z):
            return li2(z)

        if self.cut == "below":
            return li2(z)

        zc = torch.complex(z, torch.zeros_like(z))
        return li2(zc)
