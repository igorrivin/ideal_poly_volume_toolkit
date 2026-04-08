"""
Differentiable Murakami S^3 tetrahedral volume in PyTorch.

Uses torch.autograd with a custom Li2 (dilogarithm) function that
delegates to scipy.special.spence in the forward pass and provides
the analytical derivative -log(1-z)/z in the backward pass.

All complex arithmetic uses torch's native complex128 tensors.

Requires: torch, numpy, scipy

Example usage:
    import torch
    from murakami_torch import murakami_volume, ideal_6simplex_volume

    # S^3 tetrahedral link volume from 4x4 angle Gram matrix
    A4 = torch.tensor([[1, -c01, -c02, -c03],
                       [-c01, 1, -c12, -c13],
                       [-c02, -c12, 1, -c23],
                       [-c03, -c13, -c23, 1]], dtype=torch.float64,
                      requires_grad=True)
    vol = murakami_volume(A4)
    vol.backward()  # A4.grad now contains dVol/dA_ij
"""

import math
from itertools import combinations

import numpy as np
import torch
from scipy.special import spence


class Li2(torch.autograd.Function):
    """Dilogarithm Li2(z) = spence(1-z) with analytical backward pass.

    Forward: scipy.special.spence (fast, accurate, handles complex).
    Backward: dLi2/dz = -log(1-z)/z, applied via Wirtinger conjugate convention.
    """

    @staticmethod
    def forward(ctx, z):
        z_np = z.detach().cpu().numpy()
        result = spence(1.0 - z_np)
        ctx.save_for_backward(z.detach().clone())
        return torch.tensor(result, dtype=z.dtype, device=z.device)

    @staticmethod
    def backward(ctx, grad_output):
        (z,) = ctx.saved_tensors
        deriv = -torch.log(1.0 - z) / z
        return grad_output * deriv.conj()


def li2(z):
    """Differentiable dilogarithm Li2(z) for complex torch tensors."""
    return Li2.apply(z)


def murakami_volume(A4):
    """
    Differentiable Murakami volume of a spherical tetrahedron.

    Parameters
    ----------
    A4 : torch.Tensor, shape (4, 4), dtype float64
        Angle Gram matrix with diagonal 1 and off-diagonal A_ij = -cos(alpha_ij).
        May require grad.

    Returns
    -------
    vol : torch.Tensor, scalar
        Volume of the spherical tetrahedron, in [0, pi^2].
    """
    entries = [-A4[0, 1], -A4[0, 2], -A4[1, 2], -A4[2, 3], -A4[1, 3], -A4[0, 3]]
    alphas = [torch.acos(torch.clamp(e, -1 + 1e-7, 1 - 1e-7)) for e in entries]
    a = [torch.exp(1j * alpha) for alpha in alphas]

    cross_term = (
        torch.log(a[0]) * torch.log(a[3])
        + torch.log(a[1]) * torch.log(a[4])
        + torch.log(a[2]) * torch.log(a[5])
    )

    def L(z):
        return 0.5 * (
            li2(z)
            + li2(z / (a[0] * a[1] * a[3] * a[4]))
            + li2(z / (a[0] * a[2] * a[3] * a[5]))
            + li2(z / (a[1] * a[2] * a[4] * a[5]))
            - li2(-z / (a[0] * a[1] * a[2]))
            - li2(-z / (a[0] * a[4] * a[5]))
            - li2(-z / (a[1] * a[3] * a[5]))
            - li2(-z / (a[2] * a[3] * a[4]))
            + cross_term
        )

    q0 = (
        a[0] * a[3]
        + a[1] * a[4]
        + a[2] * a[5]
        + a[0] * a[1] * a[5]
        + a[0] * a[2] * a[4]
        + a[1] * a[2] * a[3]
        + a[3] * a[4] * a[5]
        + a[0] * a[1] * a[2] * a[3] * a[4] * a[5]
    )

    q1 = (
        -(a[0] - 1 / a[0]) * (a[3] - 1 / a[3])
        - (a[1] - 1 / a[1]) * (a[4] - 1 / a[4])
        - (a[2] - 1 / a[2]) * (a[5] - 1 / a[5])
    )

    q2 = (
        1 / (a[0] * a[3])
        + 1 / (a[1] * a[4])
        + 1 / (a[2] * a[5])
        + 1 / (a[0] * a[1] * a[5])
        + 1 / (a[0] * a[2] * a[4])
        + 1 / (a[1] * a[2] * a[3])
        + 1 / (a[3] * a[4] * a[5])
        + 1 / (a[0] * a[1] * a[2] * a[3] * a[4] * a[5])
    )

    disc = q1 * q1 - 4 * q0 * q2
    z0 = (-q1 + torch.sqrt(disc)) / (2 * q2)

    Lz0 = L(z0)
    alpha_sum = sum(alphas)
    vol_raw = -Lz0.real + math.pi * (torch.angle(-q2) + alpha_sum / 2) - 1.5 * math.pi**2

    # Branch selection: convex spherical tetrahedra have vol in [0, pi^2]
    two_pi_sq = 2 * math.pi**2
    vol = vol_raw % two_pi_sq
    vol = torch.where(vol > math.pi**2, two_pi_sq - vol, vol)

    return vol


def angle_gram_from_klein_torch(vertices):
    """
    Compute the angle Gram matrix for a compact hyperbolic simplex in the Klein model.

    Parameters
    ----------
    vertices : torch.Tensor, shape (m, d), dtype float64
        Klein-model vertices, strictly inside the unit ball.

    Returns
    -------
    A : torch.Tensor, shape (m, m)
        Angle Gram matrix with A_ii = 1, A_ij = -cos(theta_ij).
    """
    Q = 1.0 - vertices @ vertices.T
    R = torch.linalg.inv(Q)
    m = vertices.shape[0]
    diag_R = torch.diag(R)
    # A_ij = -R_ij / sqrt((-R_ii)(-R_jj))
    denom = torch.sqrt((-diag_R).unsqueeze(1) * (-diag_R).unsqueeze(0))
    A = -R / denom
    # Fix diagonal
    A = A - torch.diag(torch.diag(A)) + torch.eye(m, dtype=vertices.dtype, device=vertices.device)
    return A


def ideal_6simplex_volume_torch(verts_S5):
    """
    Differentiable volume of an ideal 6-simplex with 7 vertices on S^5.

    Uses the Hopf reduction with w0=0:
        Vol_6 = (8*pi^3/15) * (17/4 - w4 + w2/2)

    Parameters
    ----------
    verts_S5 : torch.Tensor, shape (7, 6), dtype float64
        Vertices on S^5 (unit norm). May require grad.

    Returns
    -------
    vol : torch.Tensor, scalar
    """
    verts = verts_S5 / torch.norm(verts_S5, dim=1, keepdim=True) * 0.9999
    A = angle_gram_from_klein_torch(verts)

    # w4: sum of 21 dihedral angles / (2*pi)
    m = 7
    w4 = torch.tensor(0.0, dtype=verts.dtype, device=verts.device)
    for i in range(m):
        for j in range(i + 1, m):
            c = torch.clamp(-A[i, j], -1 + 1e-7, 1 - 1e-7)
            w4 = w4 + torch.acos(c)
    w4 = w4 / (2 * math.pi)

    # w2: sum of 35 S^3 Murakami link volumes / (2*pi^2)
    w2_sum = torch.tensor(0.0, dtype=verts.dtype, device=verts.device)
    for subset in combinations(range(m), 4):
        idx = list(subset)
        A_sub = A[idx][:, idx]
        w2_sum = w2_sum + murakami_volume(A_sub)
    w2 = w2_sum / (2 * math.pi**2)

    return (8 * math.pi**3 / 15) * (17.0 / 4 - w4 + 0.5 * w2)
