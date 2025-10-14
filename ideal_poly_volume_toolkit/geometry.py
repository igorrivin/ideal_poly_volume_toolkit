import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import torch
import mpmath as mp

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
