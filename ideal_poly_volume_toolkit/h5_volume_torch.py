"""
Differentiable ideal H^5 polytope volume optimization via the Schläfli formula.

The key insight: for OPTIMIZATION we never need the volume itself.
The Schläfli formula gives exact gradients in closed form:

    ∂Vol_5/∂x = -(1/4) Σ_{codim-2 faces F} Vol_3(F) · ∂θ_F/∂x

where Vol_3(F) are ideal H^3 tetrahedron volumes (Lobachevsky, closed form)
and θ_F are dihedral angles from the angle Gram matrix (differentiable).

We build a "surrogate" function S(x) = -(1/4) Σ Vol_3(F) · θ_F(x)
whose gradient ∂S/∂x equals the Schläfli gradient exactly.
LBFGS/Newton can optimize S without ever computing the H^5 volume.

The actual volume is computed once at the end via the Schläfli path integral.

For the Hessian (if needed): the truncated edge lengths are
    l*_ij = ln((1 - x_i · x_j) / 2)
which is closed-form from Klein coordinates.
"""

import math
from itertools import combinations

import numpy as np
import torch
from scipy.spatial import Delaunay

from .li2_torch import li2 as li2_pure

# Precomputed indices for a 5-simplex (6 vertices)
# C(6,2) = 15 codim-2 face pairs
_PAIRS_6_2 = list(combinations(range(6), 2))
_TRIU6_I = tuple(i for i, j in _PAIRS_6_2)
_TRIU6_J = tuple(j for i, j in _PAIRS_6_2)

# For each of the 15 pairs (i,j), the 4 face vertices (complement)
_FACE_IDX = []
for i, j in _PAIRS_6_2:
    _FACE_IDX.append([k for k in range(6) if k not in (i, j)])
_FACE_IDX = torch.tensor(_FACE_IDX, dtype=torch.long)  # (15, 4)


# =====================================================================
# Lobachevsky function: Λ(θ) = (1/2) Im(Li₂(e^{2iθ}))
# =====================================================================

def lobachevsky_torch(theta):
    """Lobachevsky function Λ(θ) via Clausen/Li₂ identity."""
    z = torch.exp(2j * theta.to(torch.complex128))
    return 0.5 * li2_pure(z).imag


# =====================================================================
# Ideal H^3 tetrahedron volume (Milnor formula)
# =====================================================================

def ideal_tet_volume_from_angle_gram(A4):
    """
    Volume of an ideal tetrahedron from its 4×4 angle Gram matrix.

    Milnor: Vol = Λ(θ_{12}) + Λ(θ_{13}) + Λ(θ_{23})
    where θ_{ab} are dihedral angles at edges meeting at vertex 0.

    The angle Gram entry A[i,j] = -cos(dihedral angle between facets i,j).
    The dihedral angle at the edge connecting vertices NOT in {i,j}
    equals arccos(-A[i,j]).

    Edges meeting at vertex 0: (0,1), (0,2), (0,3).
    Their dihedral angles are θ_{23}, θ_{13}, θ_{12} respectively.

    Parameters
    ----------
    A4 : torch.Tensor, shape (..., 4, 4)

    Returns
    -------
    vol : torch.Tensor, shape (...)
    """
    theta_12 = torch.acos((-A4[..., 1, 2]).clamp(-1 + 1e-7, 1 - 1e-7))
    theta_13 = torch.acos((-A4[..., 1, 3]).clamp(-1 + 1e-7, 1 - 1e-7))
    theta_23 = torch.acos((-A4[..., 2, 3]).clamp(-1 + 1e-7, 1 - 1e-7))

    return lobachevsky_torch(theta_12) + lobachevsky_torch(theta_13) + lobachevsky_torch(theta_23)


# =====================================================================
# Schläfli surrogate: a function whose gradient = Schläfli gradient
# =====================================================================

def schlafli_surrogate_5simplex(klein_verts):
    """
    Surrogate function for a single ideal 5-simplex.

    S = -(1/4) Σ_{(i,j)} Vol_3(F_{ij}).detach() * θ_{ij}

    where θ_{ij} = arccos(-A6[i,j]) are dihedral angles of the 5-simplex
    (differentiable via the angle Gram), and Vol_3(F_{ij}) are ideal H^3
    tetrahedron volumes (Milnor/Lobachevsky, detached as constants).

    ∂S/∂x = ∂Vol_5/∂x exactly (Schläfli formula).

    Parameters
    ----------
    klein_verts : torch.Tensor, shape (6, 5)
        Klein-model vertices of the 5-simplex.

    Returns
    -------
    S : torch.Tensor, scalar (differentiable)
    """
    # 5-simplex angle Gram
    Q = 1.0 - klein_verts @ klein_verts.T
    R = torch.linalg.inv(Q)
    diag_R = torch.diag(R)
    neg_diag = (-diag_R).clamp(min=1e-30)
    denom = torch.sqrt(neg_diag.unsqueeze(1) * neg_diag.unsqueeze(0))
    A6 = -R / denom
    eye6 = torch.eye(6, dtype=klein_verts.dtype, device=klein_verts.device)
    A6 = A6 * (1 - eye6) + eye6

    S = torch.tensor(0.0, dtype=klein_verts.dtype, device=klein_verts.device)

    for i in range(6):
        for j in range(i + 1, 6):
            # Dihedral angle of 5-simplex (differentiable)
            theta_ij = torch.acos((-A6[i, j]).clamp(-1 + 1e-7, 1 - 1e-7))

            # Face F_{ij}: ideal tetrahedron with 4 vertices
            face_idx = [k for k in range(6) if k not in (i, j)]
            fk = klein_verts[face_idx]
            Qf = 1.0 - fk @ fk.T
            Rf = torch.linalg.inv(Qf)
            dRf = torch.diag(Rf)
            neg_dRf = (-dRf).clamp(min=1e-30)
            denomf = torch.sqrt(neg_dRf.unsqueeze(1) * neg_dRf.unsqueeze(0))
            A4 = -Rf / denomf
            eye4 = torch.eye(4, dtype=klein_verts.dtype, device=klein_verts.device)
            A4 = A4 * (1 - eye4) + eye4

            # Face volume (detached — constant in Schläfli gradient)
            with torch.no_grad():
                face_vol = ideal_tet_volume_from_angle_gram(A4)

            S = S + face_vol * theta_ij

    return -0.25 * S


# =====================================================================
# Vectorized Schläfli surrogate (no Python loops)
# =====================================================================

def _angle_gram_batch_5(klein, simplices_idx):
    """Batched 6×6 angle Gram matrices for S 5-simplices."""
    verts = klein[simplices_idx]                        # (S, 6, 5)
    Q = 1.0 - torch.bmm(verts, verts.transpose(1, 2))  # (S, 6, 6)
    R = torch.linalg.inv(Q)                             # (S, 6, 6)
    diag_R = torch.diagonal(R, dim1=1, dim2=2)          # (S, 6)
    valid = (diag_R < 0).all(dim=1)                     # (S,)
    neg_diag = (-diag_R).clamp(min=1e-30)
    denom = torch.sqrt(neg_diag.unsqueeze(2) * neg_diag.unsqueeze(1))
    A = -R / denom
    eye = torch.eye(6, dtype=klein.dtype, device=klein.device).unsqueeze(0)
    A = A * (1 - eye) + eye
    return A, valid


def _face_angle_grams_batch(klein, simplices_idx):
    """
    Compute intrinsic 4×4 angle Gram matrices for all 15 codim-2 faces
    of all S 5-simplices.

    Each face is a tetrahedron with 4 vertices in H^5. Its intrinsic
    angle Gram must be computed from those 4 Klein-model vertices directly
    (NOT by extracting a submatrix of the 6×6 Gram, which gives
    the wrong dihedral angles).

    Parameters
    ----------
    klein : torch.Tensor, shape (N, 5)
    simplices_idx : torch.Tensor, shape (S, 6)

    Returns
    -------
    A4_all : torch.Tensor, shape (S, 15, 4, 4)
    """
    S = simplices_idx.shape[0]
    face_idx = _FACE_IDX.to(klein.device)  # (15, 4)

    # For each simplex and each face, gather the 4 Klein vertices
    # simplices_idx[s, face_idx[f, k]] = global vertex index
    global_face_idx = simplices_idx[:, face_idx]  # (S, 15, 4)
    face_verts = klein[global_face_idx]            # (S, 15, 4, 5)

    # Reshape to (S*15, 4, 5) for batched computation
    fv = face_verts.reshape(S * 15, 4, 5)

    # Intrinsic angle Gram for each face
    Q = 1.0 - torch.bmm(fv, fv.transpose(1, 2))  # (S*15, 4, 4)
    R = torch.linalg.inv(Q)
    diag_R = torch.diagonal(R, dim1=1, dim2=2)    # (S*15, 4)
    neg_diag = (-diag_R).clamp(min=1e-30)
    denom = torch.sqrt(neg_diag.unsqueeze(2) * neg_diag.unsqueeze(1))
    A4 = -R / denom
    eye4 = torch.eye(4, dtype=klein.dtype, device=klein.device).unsqueeze(0)
    A4 = A4 * (1 - eye4) + eye4

    return A4.reshape(S, 15, 4, 4)


def _face_volumes_batch(klein, simplices_idx):
    """
    Batched ideal H^3 tetrahedron volumes for all 15 codim-2 faces.

    Computes intrinsic face angle Grams from the Klein-model vertices,
    then applies the Milnor formula.

    Parameters
    ----------
    klein : torch.Tensor, shape (N, 5)
    simplices_idx : torch.Tensor, shape (S, 6)

    Returns
    -------
    face_vols : torch.Tensor, shape (S, 15)
    """
    A4_all = _face_angle_grams_batch(klein, simplices_idx)  # (S, 15, 4, 4)

    # Milnor: Vol = Λ(θ_{12}) + Λ(θ_{13}) + Λ(θ_{23})
    theta_12 = torch.acos((-A4_all[..., 1, 2]).clamp(-1 + 1e-7, 1 - 1e-7))
    theta_13 = torch.acos((-A4_all[..., 1, 3]).clamp(-1 + 1e-7, 1 - 1e-7))
    theta_23 = torch.acos((-A4_all[..., 2, 3]).clamp(-1 + 1e-7, 1 - 1e-7))

    return lobachevsky_torch(theta_12) + lobachevsky_torch(theta_13) + lobachevsky_torch(theta_23)


def schlafli_surrogate_vectorized(pts_S4, simplices):
    """
    Fully vectorized Schläfli surrogate for an ideal H^5 polytope.

    S = -(1/4) Σ_{simplices} Σ_{(i,j)} Vol_3(F_{ij}).detach() * θ_{ij}

    No Python loops over simplices or face pairs.

    Parameters
    ----------
    pts_S4 : torch.Tensor, shape (N, 5), requires_grad=True
    simplices : list/tensor, shape (S, 6)

    Returns
    -------
    S : torch.Tensor, scalar (differentiable)
    """
    klein = pts_S4 / torch.norm(pts_S4, dim=1, keepdim=True) * 0.9999

    if not isinstance(simplices, torch.Tensor):
        simplices = torch.tensor(simplices, dtype=torch.long, device=klein.device)

    A6, valid = _angle_gram_batch_5(klein, simplices)

    if not valid.any():
        return torch.tensor(0.0, dtype=klein.dtype, device=klein.device)

    A6_valid = A6[valid]  # (S', 6, 6)
    simps_valid = simplices[valid]

    # 15 dihedral angles per simplex (differentiable)
    dihedral = torch.acos((-A6_valid[:, _TRIU6_I, _TRIU6_J]).clamp(-1 + 1e-7, 1 - 1e-7))  # (S', 15)

    # 15 face volumes per simplex (detached constants)
    with torch.no_grad():
        fvols = _face_volumes_batch(klein, simps_valid)  # (S', 15)

    # Surrogate: -(1/4) Σ fvol * θ
    return -0.25 * (fvols * dihedral).sum()


def schlafli_surrogate_polytope(pts_S4, simplices):
    """
    Schläfli surrogate for a full ideal H^5 polytope.

    Sum of surrogates over all 5-simplices in the triangulation.

    Parameters
    ----------
    pts_S4 : torch.Tensor, shape (N, 5), requires_grad=True
        Points on S^4.
    simplices : list of lists or tensor, shape (S, 6)
        Vertex indices for each 5-simplex.

    Returns
    -------
    S : torch.Tensor, scalar (differentiable)
    """
    klein = pts_S4 / torch.norm(pts_S4, dim=1, keepdim=True) * 0.9999
    S = torch.tensor(0.0, dtype=pts_S4.dtype, device=pts_S4.device)

    for simp in simplices:
        simp_klein = klein[simp]
        # Check validity
        Q = 1.0 - simp_klein @ simp_klein.T
        try:
            R = torch.linalg.inv(Q)
        except Exception:
            continue
        diag_R = torch.diag(R)
        if (diag_R >= 0).any():
            continue
        S = S + schlafli_surrogate_5simplex(simp_klein)

    return S


# =====================================================================
# Triangulation: S^4 → R^4 stereographic + Delaunay + cone
# =====================================================================

def _stereo_project_S4(pts_S4_np, pole_idx=0):
    """Stereographic S^4 → R^4."""
    pole = pts_S4_np[pole_idx]
    others = np.delete(pts_S4_np, pole_idx, axis=0)
    basis = []
    for i in range(5):
        e = np.zeros(5)
        e[i] = 1.0
        e -= np.dot(e, pole) * pole
        for b in basis:
            e -= np.dot(e, b) * b
        n = np.linalg.norm(e)
        if n > 1e-10:
            basis.append(e / n)
        if len(basis) == 4:
            break
    basis = np.array(basis)
    proj = []
    for p in others:
        d = max(1 - np.dot(p, pole), 1e-14)
        proj.append(basis @ ((p - np.dot(p, pole) * pole) / d))
    return np.array(proj)


def triangulate_S4(pts_S4_np, pole_idx=0):
    """Cone from pole over Delaunay of remaining vertices."""
    proj = _stereo_project_S4(pts_S4_np, pole_idx)
    tri = Delaunay(proj)
    simplices = []
    for simp in tri.simplices:
        global_idx = [pole_idx]
        for s in simp:
            orig = s if s < pole_idx else s + 1
            global_idx.append(orig)
        simplices.append(global_idx)
    return simplices


# =====================================================================
# Volume evaluation (Schläfli path integral, for final reporting only)
# =====================================================================

def schlafli_volume_ideal_5simplex(
    verts_S4_np,
    t0=0.15,
    steps=32,
    face_samples=1 << 13,
    seed=0,
):
    """
    H^5 volume of an ideal 5-simplex via Schläfli path integral.

    Only needed for final reporting, NOT during optimization.
    """
    from .hyperbolic_5_6_simplex_volume import (
        angle_gram_from_klein,
        dihedral_angles_from_angle_gram,
        intrinsic_hyperbolic_simplex_volume,
        hyperbolic_simplex_volume_direct_klein,
    )

    klein = verts_S4_np * 0.9999
    pairs = sorted(combinations(range(6), 2))

    V = hyperbolic_simplex_volume_direct_klein(
        t0 * klein, samples=1 << 15, scramble=True, seed=seed,
    )

    ts = np.linspace(t0, 1.0, steps + 1)
    A0, _, _ = angle_gram_from_klein(ts[0] * klein)
    prev_angles = dihedral_angles_from_angle_gram(A0)

    for k in range(len(ts) - 1):
        a, b = ts[k], ts[k + 1]
        mid = 0.5 * (a + b)
        mid_verts = mid * klein

        face_vols = {}
        for pi, (i, j) in enumerate(pairs):
            face_idx = [r for r in range(6) if r not in (i, j)]
            fseed = None if seed is None else seed + 1000 * k + pi
            face_vols[(i, j)] = intrinsic_hyperbolic_simplex_volume(
                mid_verts[face_idx],
                samples=face_samples,
                scramble=True,
                seed=fseed,
            )

        A1, _, _ = angle_gram_from_klein(b * klein)
        next_angles = dihedral_angles_from_angle_gram(A1)

        increment = 0.0
        for key in pairs:
            increment += face_vols[key] * (next_angles[key] - prev_angles[key])
        V += -0.25 * increment
        prev_angles = next_angles

    return float(V)


def schlafli_volume_ideal_polytope(pts_S4_np, simplices=None, **kwargs):
    """Total H^5 volume of an ideal polytope. For final reporting only."""
    if simplices is None:
        pts = pts_S4_np / np.linalg.norm(pts_S4_np, axis=1, keepdims=True)
        simplices = triangulate_S4(pts)

    total = 0.0
    for simp in simplices:
        try:
            v = schlafli_volume_ideal_5simplex(pts_S4_np[simp], **kwargs)
            if np.isfinite(v) and v > 0:
                total += v
        except Exception:
            pass
    return total
