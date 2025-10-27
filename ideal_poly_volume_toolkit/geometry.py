import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import torch
import mpmath as mp
import cmath

def lift_to_sphere_with_inf(W: np.ndarray) -> np.ndarray:
    P = np.zeros((W.shape[0], 3), dtype=np.float64)
    is_inf = ~np.isfinite(W.real) | ~np.isfinite(W.imag)
    F = ~is_inf
    w = W[F]
    r2 = (w.real**2 + w.imag**2)
    denom = r2 + 1.0
    P[F, 0] = 2.0 * w.real / denom
    P[F, 1] = 2.0 * w.imag / denom
    P[F, 2] = (r2 - 1.0) / denom
    P[is_inf] = np.array([0.0, 0.0, 1.0])
    return P

def inverse_stereographic_from_sphere_pts(X: np.ndarray) -> np.ndarray:
    x, y, z = X[:,0], X[:,1], X[:,2]
    denom = (1.0 - z)
    return (x/denom) + 1j*(y/denom)

def hull_tris_projected_back(W: np.ndarray, index_inf: int = 0) -> list:
    P3 = lift_to_sphere_with_inf(W)
    hull = ConvexHull(P3)
    tris = []
    for f in hull.simplices:
        if index_inf in f:
            continue
        X = P3[np.asarray(f)]
        tri = tuple(inverse_stereographic_from_sphere_pts(X))
        tris.append(tri)
    return tris

def delaunay_triangulation_indices(z_finite: np.ndarray) -> np.ndarray:
    pts = np.column_stack([z_finite.real, z_finite.imag])
    tri = Delaunay(pts)
    return tri.simplices.astype(np.int64)

def _angles_for_triangle_torch(z1, z2, z3):
    def angle_at(a, b, c):
        u = torch.stack(((b-a).real, (b-a).imag), dim=-1)
        v = torch.stack(((c-a).real, (c-a).imag), dim=-1)
        dot = (u * v).sum(dim=-1)
        cross = u[...,0]*v[...,1] - u[...,1]*v[...,0]
        return torch.atan2(torch.abs(cross), dot)
    a1 = angle_at(z1, z2, z3)
    a2 = angle_at(z2, z3, z1)
    a3 = angle_at(z3, z1, z2)
    return a1, a2, a3

def _angles_for_triangle_np(z1, z2, z3):
    def angle_at(a, b, c):
        u = np.array([(b-a).real, (b-a).imag])
        v = np.array([(c-a).real, (c-a).imag])
        dot = (u*v).sum()
        cross = u[0]*v[1] - u[1]*v[0]
        return np.arctan2(abs(cross), dot)
    return angle_at(z1,z2,z3), angle_at(z2,z3,z1), angle_at(z3,z1,z2)

def _lob_value_series_torch(theta: torch.Tensor, n: int = 64) -> torch.Tensor:
    k = torch.arange(1, n+1, device=theta.device, dtype=theta.dtype)
    return 0.5 * torch.sum(torch.sin(2*k*theta)/(k*k), dim=0)

class _LobFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, n: int = 64):
        ctx.save_for_backward(theta)
        return _lob_value_series_torch(theta, n)
    @staticmethod
    def backward(ctx, gout):
        (theta,) = ctx.saved_tensors
        return -gout * torch.log(2.0*torch.abs(torch.sin(theta))), None

def lob_fast(theta: torch.Tensor, n: int = 64) -> torch.Tensor:
    return _LobFn.apply(theta, n)

def lob_exact(theta: float, dps: int = 120) -> float:
    mp.mp.dps = dps
    return 0.5 * mp.clsin(2, 2*float(theta))


# ============================================================================
# Bloch-Wigner dilogarithm approach (faster and more accurate)
# ============================================================================

def bloch_wigner_mpmath(z: complex) -> float:
    """
    Compute Bloch-Wigner dilogarithm using mpmath.

    D(z) = Im(Li_2(z)) + arg(1-z) * log|z|

    This is exact and should be used for validation/testing.
    """
    return float(mp.im(mp.polylog(2, z)) + mp.arg(1 - z) * mp.log(abs(z)))


class _BlochWignerFn(torch.autograd.Function):
    """
    Custom autograd function for Bloch-Wigner dilogarithm.

    Forward: D(z) = Im(Li_2(z)) + arg(1-z) * log|z|
    Backward: Uses the derivative formula for the dilogarithm
    """

    @staticmethod
    def forward(ctx, z_real, z_imag):
        """
        Compute Bloch-Wigner dilogarithm.

        Args:
            z_real: Real part of z (torch tensor)
            z_imag: Imaginary part of z (torch tensor)

        Returns:
            D(z) as a real tensor
        """
        # Save for backward pass
        ctx.save_for_backward(z_real, z_imag)

        # Convert to numpy for mpmath computation
        z_real_np = z_real.detach().cpu().numpy()
        z_imag_np = z_imag.detach().cpu().numpy()

        # Handle scalar vs array
        is_scalar = z_real_np.shape == ()
        if is_scalar:
            z_complex = complex(float(z_real_np), float(z_imag_np))
            result = bloch_wigner_mpmath(z_complex)
        else:
            z_complex = z_real_np + 1j * z_imag_np
            result = np.array([bloch_wigner_mpmath(complex(z)) for z in z_complex.flat])
            result = result.reshape(z_real_np.shape)

        return torch.tensor(result, dtype=z_real.dtype, device=z_real.device)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute gradient of Bloch-Wigner dilogarithm.

        The derivative is: dD/dz = -log(1-z) / z (complex derivative)
        We need to split this into real and imaginary parts.
        """
        z_real, z_imag = ctx.saved_tensors

        # Compute 1 - z
        one_minus_z_real = 1.0 - z_real
        one_minus_z_imag = -z_imag

        # Compute log(1-z)
        # log(a + bi) = log|a+bi| + i*arg(a+bi)
        abs_one_minus_z = torch.sqrt(one_minus_z_real**2 + one_minus_z_imag**2)
        log_abs = torch.log(abs_one_minus_z + 1e-10)  # Add epsilon for stability
        arg = torch.atan2(one_minus_z_imag, one_minus_z_real)

        log_real = log_abs
        log_imag = arg

        # Compute 1/z
        z_abs_sq = z_real**2 + z_imag**2 + 1e-10
        inv_z_real = z_real / z_abs_sq
        inv_z_imag = -z_imag / z_abs_sq

        # Compute -log(1-z) / z (complex multiplication)
        # (-log_real - i*log_imag) * (inv_z_real + i*inv_z_imag)
        result_real = -(log_real * inv_z_real - log_imag * inv_z_imag)
        result_imag = -(log_real * inv_z_imag + log_imag * inv_z_real)

        # The Bloch-Wigner is real-valued, so we take the imaginary part of dD/dz for the real gradient
        # This is a bit subtle: for f: C -> R, df/dx and df/dy come from Re(df/dz) and Im(df/dz)
        # But D is already real, so we use the Wirtinger derivatives
        grad_real = grad_output * result_real
        grad_imag = grad_output * result_imag

        return grad_real, grad_imag


def bloch_wigner_torch(z_real, z_imag):
    """
    Compute Bloch-Wigner dilogarithm with PyTorch autograd support.

    Args:
        z_real: Real part of z (torch tensor)
        z_imag: Imaginary part of z (torch tensor)

    Returns:
        D(z) as a torch tensor with gradient support
    """
    return _BlochWignerFn.apply(z_real, z_imag)


def cross_ratio_torch(z1_real, z1_imag, z2_real, z2_imag, z3_real, z3_imag, z4_real, z4_imag):
    """
    Compute cross ratio of four complex points in torch.

    CR(z1, z2, z3, z4) = (z1 - z3)(z2 - z4) / ((z1 - z4)(z2 - z3))

    Args:
        z*_real, z*_imag: Real and imaginary parts of the four points

    Returns:
        Real and imaginary parts of the cross ratio
    """
    # Compute z1 - z3
    num1_real = z1_real - z3_real
    num1_imag = z1_imag - z3_imag

    # Compute z2 - z4
    num2_real = z2_real - z4_real
    num2_imag = z2_imag - z4_imag

    # Compute (z1 - z3)(z2 - z4)
    num_real = num1_real * num2_real - num1_imag * num2_imag
    num_imag = num1_real * num2_imag + num1_imag * num2_real

    # Compute z1 - z4
    den1_real = z1_real - z4_real
    den1_imag = z1_imag - z4_imag

    # Compute z2 - z3
    den2_real = z2_real - z3_real
    den2_imag = z2_imag - z3_imag

    # Compute (z1 - z4)(z2 - z3)
    den_real = den1_real * den2_real - den1_imag * den2_imag
    den_imag = den1_real * den2_imag + den1_imag * den2_real

    # Compute num / den
    den_abs_sq = den_real**2 + den_imag**2 + 1e-10
    cr_real = (num_real * den_real + num_imag * den_imag) / den_abs_sq
    cr_imag = (num_imag * den_real - num_real * den_imag) / den_abs_sq

    return cr_real, cr_imag

def triangle_volume_from_points(z1, z2, z3, mode="fast", series_terms=64, dps=120):
    if mode == "fast":
        z1t = torch.tensor(z1, dtype=torch.complex128)
        z2t = torch.tensor(z2, dtype=torch.complex128)
        z3t = torch.tensor(z3, dtype=torch.complex128)
        a1,a2,a3 = _angles_for_triangle_torch(z1t,z2t,z3t)
        return (lob_fast(a1, series_terms) + lob_fast(a2, series_terms) + lob_fast(a3, series_terms)).item()
    else:
        a1,a2,a3 = _angles_for_triangle_np(z1,z2,z3)
        return float(lob_exact(a1, dps=dps) + lob_exact(a2, dps=dps) + lob_exact(a3, dps=dps))

def triangle_volume_from_points_torch(z1t, z2t, z3t, series_terms=64):
    """
    Pure-Torch triangle accumulator: angles → Λ → sum. Returns a real Tensor.
    """
    a1, a2, a3 = _angles_for_triangle_torch(z1t, z2t, z3t)  # float64 tensors
    return lob_fast(a1, series_terms) + lob_fast(a2, series_terms) + lob_fast(a3, series_terms)


def triangle_volume_bloch_wigner(z1t, z2t, z3t):
    """
    Compute volume of ideal tetrahedron using Bloch-Wigner dilogarithm.

    For a tetrahedron with vertices z1, z2, z3, ∞, the volume is:
    V = |D(CR(z1, z2, z3, ∞))|

    where D is the Bloch-Wigner dilogarithm and CR is the cross ratio.

    This is faster and more accurate than the Lobachevsky series approach.

    Args:
        z1t, z2t, z3t: Complex torch tensors for the three finite vertices

    Returns:
        Volume as a real torch tensor
    """
    # Extract real and imaginary parts
    z1_real = z1t.real
    z1_imag = z1t.imag
    z2_real = z2t.real
    z2_imag = z2t.imag
    z3_real = z3t.real
    z3_imag = z3t.imag

    # For infinity, we use the cross ratio formula with ∞ as the 4th point:
    # CR(z1, z2, z3, ∞) = (z1 - z3) / (z2 - z3)
    # This is a simplified version since (z - ∞) factors cancel

    diff13_real = z1_real - z3_real
    diff13_imag = z1_imag - z3_imag
    diff23_real = z2_real - z3_real
    diff23_imag = z2_imag - z3_imag

    # Compute (z1 - z3) / (z2 - z3)
    den_abs_sq = diff23_real**2 + diff23_imag**2 + 1e-10
    cr_real = (diff13_real * diff23_real + diff13_imag * diff23_imag) / den_abs_sq
    cr_imag = (diff13_imag * diff23_real - diff13_real * diff23_imag) / den_abs_sq

    # Compute Bloch-Wigner dilogarithm
    vol = bloch_wigner_torch(cr_real, cr_imag)

    # Return absolute value (volume is positive)
    return torch.abs(vol)

def ideal_poly_volume_via_hull_project_back(W, index_inf=0, mode="fast", dps=120, series_terms=64):
    tris = hull_tris_projected_back(W, index_inf=index_inf)
    total = 0.0
    for (z1, z2, z3) in tris:
        total += triangle_volume_from_points(z1, z2, z3, mode=mode, series_terms=series_terms, dps=dps)
    return total

def ideal_poly_volume_via_delaunay(z_finite_complex: np.ndarray, mode="fast", dps=120, series_terms=64):
    idx_tris = delaunay_triangulation_indices(z_finite_complex)
    total = 0.0
    for (i,j,k) in idx_tris:
        z1, z2, z3 = z_finite_complex[i], z_finite_complex[j], z_finite_complex[k]
        total += triangle_volume_from_points(z1, z2, z3, mode=mode, series_terms=series_terms, dps=dps)
    return total
