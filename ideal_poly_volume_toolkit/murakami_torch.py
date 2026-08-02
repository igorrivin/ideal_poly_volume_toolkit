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

# Precomputed index arrays for C(7,4)=35 4-element subsets of {0,...,6}
_SUBSETS_7_4 = list(combinations(range(7), 4))
# Upper triangle indices for 7x7 matrix (21 pairs)
_TRIU_I, _TRIU_J = [], []
for i in range(7):
    for j in range(i + 1, 7):
        _TRIU_I.append(i)
        _TRIU_J.append(j)
_TRIU_I = tuple(_TRIU_I)
_TRIU_J = tuple(_TRIU_J)


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
    # Batched: collect all 35 4x4 submatrices, one Li2 call
    all_subsets = list(combinations(range(m), 4))
    A_subs = torch.stack([A[list(s)][:, list(s)] for s in all_subsets])  # (35, 4, 4)
    link_vols = murakami_volume_batched(A_subs)  # (35,)
    w2 = link_vols.sum() / (2 * math.pi**2)

    return (8 * math.pi**3 / 15) * (17.0 / 4 - w4 + 0.5 * w2)


def murakami_volume_batched(A):
    """
    Batched Murakami volume for B spherical tetrahedra at once.

    Parameters
    ----------
    A : torch.Tensor, shape (B, 4, 4), dtype float64
        Batch of angle Gram matrices.

    Returns
    -------
    vols : torch.Tensor, shape (B,)
        Volumes of the B spherical tetrahedra.
    """
    B = A.shape[0]

    entries = torch.stack([
        -A[:, 0, 1], -A[:, 0, 2], -A[:, 1, 2],
        -A[:, 2, 3], -A[:, 1, 3], -A[:, 0, 3]
    ], dim=1)  # (B, 6)

    alphas = torch.acos(torch.clamp(entries, -1 + 1e-7, 1 - 1e-7))
    a = torch.exp(1j * alphas)  # (B, 6)

    log_a = torch.log(a)
    cross = log_a[:, 0] * log_a[:, 3] + log_a[:, 1] * log_a[:, 4] + log_a[:, 2] * log_a[:, 5]

    q0 = (a[:, 0] * a[:, 3] + a[:, 1] * a[:, 4] + a[:, 2] * a[:, 5]
          + a[:, 0] * a[:, 1] * a[:, 5] + a[:, 0] * a[:, 2] * a[:, 4]
          + a[:, 1] * a[:, 2] * a[:, 3] + a[:, 3] * a[:, 4] * a[:, 5]
          + a[:, 0] * a[:, 1] * a[:, 2] * a[:, 3] * a[:, 4] * a[:, 5])

    q1 = (-(a[:, 0] - 1 / a[:, 0]) * (a[:, 3] - 1 / a[:, 3])
          - (a[:, 1] - 1 / a[:, 1]) * (a[:, 4] - 1 / a[:, 4])
          - (a[:, 2] - 1 / a[:, 2]) * (a[:, 5] - 1 / a[:, 5]))

    q2 = (1 / (a[:, 0] * a[:, 3]) + 1 / (a[:, 1] * a[:, 4]) + 1 / (a[:, 2] * a[:, 5])
          + 1 / (a[:, 0] * a[:, 1] * a[:, 5]) + 1 / (a[:, 0] * a[:, 2] * a[:, 4])
          + 1 / (a[:, 1] * a[:, 2] * a[:, 3]) + 1 / (a[:, 3] * a[:, 4] * a[:, 5])
          + 1 / (a[:, 0] * a[:, 1] * a[:, 2] * a[:, 3] * a[:, 4] * a[:, 5]))

    disc = q1 * q1 - 4 * q0 * q2
    z0 = (-q1 + torch.sqrt(disc)) / (2 * q2)

    z0c = z0.unsqueeze(1)
    li2_args = torch.cat([
        z0c,
        z0c / (a[:, 0] * a[:, 1] * a[:, 3] * a[:, 4]).unsqueeze(1),
        z0c / (a[:, 0] * a[:, 2] * a[:, 3] * a[:, 5]).unsqueeze(1),
        z0c / (a[:, 1] * a[:, 2] * a[:, 4] * a[:, 5]).unsqueeze(1),
        -z0c / (a[:, 0] * a[:, 1] * a[:, 2]).unsqueeze(1),
        -z0c / (a[:, 0] * a[:, 4] * a[:, 5]).unsqueeze(1),
        -z0c / (a[:, 1] * a[:, 3] * a[:, 5]).unsqueeze(1),
        -z0c / (a[:, 2] * a[:, 3] * a[:, 4]).unsqueeze(1),
    ], dim=1)  # (B, 8)

    # ONE Li2 call for all B*8 arguments
    li2_vals = Li2.apply(li2_args)

    L = 0.5 * (li2_vals[:, :4].sum(dim=1) - li2_vals[:, 4:].sum(dim=1) + cross)

    alpha_sum = alphas.sum(dim=1)
    vol_raw = -L.real + math.pi * (torch.angle(-q2) + alpha_sum / 2) - 1.5 * math.pi ** 2

    two_pi_sq = 2 * math.pi ** 2
    vol = vol_raw % two_pi_sq
    vol = torch.where(vol > math.pi ** 2, two_pi_sq - vol, vol)

    return vol


# =====================================================================
# Fully vectorized H^6 ideal polytope volume
# =====================================================================

def _angle_gram_batch(klein, simplices_idx):
    """
    Batched angle Gram matrices for S simplices.

    Parameters
    ----------
    klein : torch.Tensor, shape (N, 6)
        Klein-model vertices (norm < 1).
    simplices_idx : torch.Tensor, shape (S, 7), dtype long
        Vertex indices for each 6-simplex.

    Returns
    -------
    A : torch.Tensor, shape (S, 7, 7)
        Angle Gram matrices.
    valid : torch.Tensor, shape (S,), dtype bool
        True for simplices with well-defined angle Gram (all diag(R) < 0).
    """
    verts = klein[simplices_idx]           # (S, 7, 6)
    Q = 1.0 - torch.bmm(verts, verts.transpose(1, 2))  # (S, 7, 7)
    R = torch.linalg.inv(Q)               # (S, 7, 7)
    diag_R = torch.diagonal(R, dim1=1, dim2=2)  # (S, 7)
    valid = (diag_R < 0).all(dim=1)       # (S,)

    neg_diag = (-diag_R).clamp(min=1e-30)
    denom = torch.sqrt(neg_diag.unsqueeze(2) * neg_diag.unsqueeze(1))  # (S, 7, 7)
    A = -R / denom
    # Fix diagonal to 1
    eye = torch.eye(7, dtype=klein.dtype, device=klein.device).unsqueeze(0)
    A = A * (1 - eye) + eye

    return A, valid


def _w4_batch(A):
    """
    Batched w4 = sum of 21 dihedral angles / (2*pi).

    Parameters
    ----------
    A : torch.Tensor, shape (S, 7, 7)

    Returns
    -------
    w4 : torch.Tensor, shape (S,)
    """
    cosines = -A[:, _TRIU_I, _TRIU_J]     # (S, 21)
    angles = torch.acos(cosines.clamp(-1 + 1e-7, 1 - 1e-7))
    return angles.sum(dim=1) / (2 * math.pi)


def _w2_batch(A, li2_fn=None):
    """
    Batched w2 = sum of 35 Murakami link volumes / (2*pi^2).

    Parameters
    ----------
    A : torch.Tensor, shape (S, 7, 7)
    li2_fn : callable, optional
        Dilogarithm function. Default: module-level li2 (scipy-backed).

    Returns
    -------
    w2 : torch.Tensor, shape (S,)
    """
    S = A.shape[0]
    idx = A.new_tensor(_SUBSETS_7_4, dtype=torch.long)  # (35, 4)
    row_idx = idx.unsqueeze(1).expand(35, 4, 4)   # (35, 4, 4)
    col_idx = idx.unsqueeze(2).expand(35, 4, 4)   # (35, 4, 4)
    A_subs = A[:, row_idx, col_idx]                # (S, 35, 4, 4)
    A_flat = A_subs.reshape(S * 35, 4, 4)          # (S*35, 4, 4)

    link_vols = _murakami_batched_with_li2(A_flat, li2_fn)  # (S*35,)
    link_vols = link_vols.reshape(S, 35)
    return link_vols.sum(dim=1) / (2 * math.pi ** 2)


def _murakami_batched_with_li2(A, li2_fn=None):
    """
    Batched Murakami using a specified Li2 implementation.
    Same as murakami_volume_batched but allows swapping the Li2 function.
    """
    if li2_fn is None:
        li2_fn = li2

    B = A.shape[0]
    entries = torch.stack([
        -A[:, 0, 1], -A[:, 0, 2], -A[:, 1, 2],
        -A[:, 2, 3], -A[:, 1, 3], -A[:, 0, 3]
    ], dim=1)

    alphas = torch.acos(torch.clamp(entries, -1 + 1e-7, 1 - 1e-7))
    a = torch.exp(1j * alphas)

    log_a = torch.log(a)
    cross = log_a[:, 0] * log_a[:, 3] + log_a[:, 1] * log_a[:, 4] + log_a[:, 2] * log_a[:, 5]

    q0 = (a[:, 0] * a[:, 3] + a[:, 1] * a[:, 4] + a[:, 2] * a[:, 5]
          + a[:, 0] * a[:, 1] * a[:, 5] + a[:, 0] * a[:, 2] * a[:, 4]
          + a[:, 1] * a[:, 2] * a[:, 3] + a[:, 3] * a[:, 4] * a[:, 5]
          + a[:, 0] * a[:, 1] * a[:, 2] * a[:, 3] * a[:, 4] * a[:, 5])

    q1 = (-(a[:, 0] - 1 / a[:, 0]) * (a[:, 3] - 1 / a[:, 3])
          - (a[:, 1] - 1 / a[:, 1]) * (a[:, 4] - 1 / a[:, 4])
          - (a[:, 2] - 1 / a[:, 2]) * (a[:, 5] - 1 / a[:, 5]))

    q2 = (1 / (a[:, 0] * a[:, 3]) + 1 / (a[:, 1] * a[:, 4]) + 1 / (a[:, 2] * a[:, 5])
          + 1 / (a[:, 0] * a[:, 1] * a[:, 5]) + 1 / (a[:, 0] * a[:, 2] * a[:, 4])
          + 1 / (a[:, 1] * a[:, 2] * a[:, 3]) + 1 / (a[:, 3] * a[:, 4] * a[:, 5])
          + 1 / (a[:, 0] * a[:, 1] * a[:, 2] * a[:, 3] * a[:, 4] * a[:, 5]))

    disc = q1 * q1 - 4 * q0 * q2
    z0 = (-q1 + torch.sqrt(disc)) / (2 * q2)

    z0c = z0.unsqueeze(1)
    li2_args = torch.cat([
        z0c,
        z0c / (a[:, 0] * a[:, 1] * a[:, 3] * a[:, 4]).unsqueeze(1),
        z0c / (a[:, 0] * a[:, 2] * a[:, 3] * a[:, 5]).unsqueeze(1),
        z0c / (a[:, 1] * a[:, 2] * a[:, 4] * a[:, 5]).unsqueeze(1),
        -z0c / (a[:, 0] * a[:, 1] * a[:, 2]).unsqueeze(1),
        -z0c / (a[:, 0] * a[:, 4] * a[:, 5]).unsqueeze(1),
        -z0c / (a[:, 1] * a[:, 3] * a[:, 5]).unsqueeze(1),
        -z0c / (a[:, 2] * a[:, 3] * a[:, 4]).unsqueeze(1),
    ], dim=1)  # (B, 8)

    li2_vals = li2_fn(li2_args.reshape(-1)).reshape(B, 8)

    L = 0.5 * (li2_vals[:, :4].sum(dim=1) - li2_vals[:, 4:].sum(dim=1) + cross)

    alpha_sum = alphas.sum(dim=1)
    vol_raw = -L.real + math.pi * (torch.angle(-q2) + alpha_sum / 2) - 1.5 * math.pi ** 2

    two_pi_sq = 2 * math.pi ** 2
    vol = vol_raw % two_pi_sq
    vol = torch.where(vol > math.pi ** 2, two_pi_sq - vol, vol)

    return vol


def ideal_h6_polytope_volume(pts_S5, simplices, li2_fn=None):
    """
    Fully vectorized H^6 ideal polytope volume.

    No Python loops over simplices — everything is batched tensor ops.

    Parameters
    ----------
    pts_S5 : torch.Tensor, shape (N, 6)
        Points on S^5 (or near it). Will be normalized and scaled to Klein.
    simplices : list of lists or tensor, shape (S, 7)
        Each row gives 7 vertex indices forming a 6-simplex.
    li2_fn : callable, optional
        Dilogarithm function. Use li2_torch.li2 for pure-torch / GPU.
        Default: scipy-backed Li2.

    Returns
    -------
    total_vol : torch.Tensor, scalar
        Sum of ideal 6-simplex volumes.
    """
    klein = pts_S5 / torch.norm(pts_S5, dim=1, keepdim=True) * 0.9999

    if not isinstance(simplices, torch.Tensor):
        simplices = torch.tensor(simplices, dtype=torch.long, device=klein.device)

    A, valid = _angle_gram_batch(klein, simplices)

    if not valid.any():
        return torch.tensor(0.0, dtype=klein.dtype, device=klein.device)

    A_valid = A[valid]
    w4 = _w4_batch(A_valid)
    w2 = _w2_batch(A_valid, li2_fn=li2_fn)

    simplex_vols = (8 * math.pi ** 3 / 15) * (17.0 / 4 - w4 + 0.5 * w2)

    good = (simplex_vols > 0) & torch.isfinite(simplex_vols)
    return simplex_vols[good].sum()
