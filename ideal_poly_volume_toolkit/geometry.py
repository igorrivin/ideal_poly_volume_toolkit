from importlib import import_module

import mpmath as mp
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.special import spence


_TORCH_PRIVATE_EXPORTS = frozenset(
    {
        "_LobFn",
        "_angles_for_triangle_torch",
        "_lob_value_series_torch",
    }
)


def _geometry_torch_module():
    """Load the optional differentiable H³ implementation on demand."""
    try:
        return import_module(".geometry_torch", __package__)
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ImportError(
                "PyTorch-backed geometry requires the 'torch' extra; install it "
                "with `python -m pip install 'ideal_poly_volume_toolkit[torch]'`."
            ) from exc
        raise


def __getattr__(name: str):
    """Preserve the repository's historical private Torch imports lazily."""
    if name in _TORCH_PRIVATE_EXPORTS:
        return getattr(_geometry_torch_module(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _TORCH_PRIVATE_EXPORTS)


def _as_complex_array(values) -> tuple[np.ndarray, bool]:
    """Return a complex128 ndarray and whether the input was scalar."""
    arr = np.asarray(values, dtype=np.complex128)
    return arr, arr.ndim == 0


def lift_to_sphere_with_inf(W: np.ndarray) -> np.ndarray:
    P = np.zeros((W.shape[0], 3), dtype=np.float64)
    is_inf = ~np.isfinite(W.real) | ~np.isfinite(W.imag)
    F = ~is_inf
    w = W[F]
    r2 = w.real**2 + w.imag**2
    denom = r2 + 1.0
    P[F, 0] = 2.0 * w.real / denom
    P[F, 1] = 2.0 * w.imag / denom
    P[F, 2] = (r2 - 1.0) / denom
    P[is_inf] = np.array([0.0, 0.0, 1.0])
    return P


def inverse_stereographic_from_sphere_pts(X: np.ndarray) -> np.ndarray:
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    denom = 1.0 - z
    return (x / denom) + 1j * (y / denom)


def stereographic_project_from_pole(
    points_sphere: np.ndarray,
    pole_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project points on S² from one of the points to the complex plane.

    ``points_sphere[pole_index]`` is sent to infinity and omitted from the
    returned finite coordinates.  The second return value maps each projected
    coordinate back to its original row index.

    Rows are normalized first, so positive rescaling of sphere representatives
    does not change the result.  Coincident points are rejected explicitly;
    near-coincident but distinct points are projected without adding an
    absolute epsilon to the denominator.
    """
    points = np.asarray(points_sphere, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_sphere must have shape (N, 3)")
    if points.shape[0] < 2:
        raise ValueError("at least two sphere points are required")
    if not np.isfinite(points).all():
        raise ValueError("points_sphere must contain only finite values")

    row_scales = np.max(np.abs(points), axis=1)
    if np.any(row_scales == 0.0):
        raise ValueError("sphere representatives must be nonzero")
    scaled = points / row_scales[:, None]
    unit = scaled / np.linalg.norm(scaled, axis=1)[:, None]

    pole_index = int(pole_index)
    if pole_index < 0:
        pole_index += len(unit)
    if not 0 <= pole_index < len(unit):
        raise IndexError("pole_index is out of range")

    pole = unit[pole_index]

    # Construct the chart directly from a well-conditioned tangent basis at
    # the pole.  A Householder vector proportional to ``pole - north`` loses
    # its longitudinal component when the pole is within roughly sqrt(eps) of
    # the north pole, so the resulting transform need not actually send the
    # pole to north.  Projecting the least-aligned Cartesian axis into the
    # tangent plane keeps the normalization bounded away from zero instead.
    reference = np.eye(3)[int(np.argmin(np.abs(pole)))]
    tangent_real = reference - np.dot(reference, pole) * pole
    tangent_real /= np.linalg.norm(tangent_real)
    tangent_imag = np.cross(pole, tangent_real)
    tangent_imag /= np.linalg.norm(tangent_imag)

    original_indices = np.delete(np.arange(len(unit), dtype=np.int64), pole_index)
    finite = unit[original_indices]

    # For unit vectors this is exactly 1 - <point, pole>, evaluated as a
    # squared distance to avoid cancellation for points close to the pole.
    denominator = 0.5 * np.sum((finite - pole) ** 2, axis=1)
    if np.any(denominator == 0.0):
        raise ValueError("the projection pole is repeated in points_sphere")

    numerator = finite @ tangent_real + 1j * (finite @ tangent_imag)
    projected = numerator / denominator
    return projected.astype(np.complex128, copy=False), original_indices


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


def _angles_for_triangle_np(z1, z2, z3):
    def angle_at(a, b, c):
        u = b - a
        v = c - a
        dot = u.real * v.real + u.imag * v.imag
        cross = u.real * v.imag - u.imag * v.real
        return np.arctan2(abs(cross), dot)

    return angle_at(z1, z2, z3), angle_at(z2, z3, z1), angle_at(z3, z1, z2)


def _lob_value_series_np(theta, n: int = 64):
    """Vectorized NumPy evaluation of the truncated Lobachevsky series."""
    if n < 1:
        raise ValueError("n must be a positive integer")
    theta_arr = np.asarray(theta, dtype=np.float64)
    k = np.arange(1, n + 1, dtype=np.float64).reshape((n,) + (1,) * theta_arr.ndim)
    return 0.5 * np.sum(np.sin(2 * k * theta_arr) / (k * k), axis=0)


def lob_fast(theta, n: int = 64):
    """Evaluate the Torch Lobachevsky kernel via the optional ``torch`` extra."""
    return _geometry_torch_module().lob_fast(theta, n)


def lob_exact(theta: float, dps: int = 120) -> float:
    mp.mp.dps = dps
    return 0.5 * mp.clsin(2, 2 * float(theta))


# ============================================================================
# Bloch-Wigner dilogarithm approach (faster and more accurate)
# ============================================================================


def bloch_wigner(z):
    """Evaluate the Bloch–Wigner dilogarithm in double precision.

    Parameters
    ----------
    z : complex or array-like of complex
        Cross-ratio value(s).

    Returns
    -------
    float or numpy.ndarray
        A scalar for scalar input and an array of matching shape otherwise.

    Notes
    -----
    SciPy's vectorized complex ``spence`` implementation evaluates
    ``Li₂(z) = spence(1-z)``.  The removable values at z=0 and z=1 are
    handled explicitly instead of relying on products involving ``log(0)``.
    """
    z_arr, is_scalar = _as_complex_array(z)
    out = np.empty(z_arr.shape, dtype=np.float64)
    removable = (z_arr == 0.0) | (z_arr == 1.0)
    out[removable] = 0.0

    regular = ~removable
    if np.any(regular):
        zr = z_arr[regular]
        li2 = spence(1.0 - zr)
        out[regular] = li2.imag + np.angle(1.0 - zr) * np.log(np.abs(zr))

    return float(out) if is_scalar else out


def bloch_wigner_mpmath(z: complex) -> float:
    """
    Compute Bloch-Wigner dilogarithm using mpmath.

    D(z) = Im(Li_2(z)) + arg(1-z) * log|z|

    This is exact and should be used for validation/testing.
    """
    return float(mp.im(mp.polylog(2, z)) + mp.arg(1 - z) * mp.log(abs(z)))


def bloch_wigner_torch(z_real, z_imag):
    """Evaluate Bloch–Wigner values via the optional ``torch`` extra."""
    return _geometry_torch_module().bloch_wigner_torch(z_real, z_imag)


def cross_ratio_torch(
    z1_real, z1_imag, z2_real, z2_imag, z3_real, z3_imag, z4_real, z4_imag
):
    """Compute a cross ratio via the optional ``torch`` extra."""
    return _geometry_torch_module().cross_ratio_torch(
        z1_real,
        z1_imag,
        z2_real,
        z2_imag,
        z3_real,
        z3_imag,
        z4_real,
        z4_imag,
    )


def triangle_volume_from_points(z1, z2, z3, mode="fast", series_terms=64, dps=120):
    if mode == "fast":
        a1, a2, a3 = _angles_for_triangle_np(z1, z2, z3)
        value = (
            _lob_value_series_np(a1, series_terms)
            + _lob_value_series_np(a2, series_terms)
            + _lob_value_series_np(a3, series_terms)
        )
        return float(value)
    if mode in ("exact", "eval_only"):
        a1, a2, a3 = _angles_for_triangle_np(z1, z2, z3)
        return float(
            lob_exact(a1, dps=dps) + lob_exact(a2, dps=dps) + lob_exact(a3, dps=dps)
        )
    raise ValueError("mode must be 'fast', 'exact', or the legacy alias 'eval_only'")


def triangle_volume_from_points_torch(z1t, z2t, z3t, series_terms=64):
    """Accumulate one tetrahedron via the optional ``torch`` extra."""
    return _geometry_torch_module().triangle_volume_from_points_torch(
        z1t, z2t, z3t, series_terms
    )


def triangle_volume_bloch_wigner(z1t, z2t, z3t):
    """Compute one BW tetrahedron via the optional ``torch`` extra."""
    return _geometry_torch_module().triangle_volume_bloch_wigner(z1t, z2t, z3t)


def ideal_poly_volume_via_hull_project_back(
    W, index_inf=0, mode="fast", dps=120, series_terms=64
):
    tris = hull_tris_projected_back(W, index_inf=index_inf)
    total = 0.0
    for z1, z2, z3 in tris:
        total += triangle_volume_from_points(
            z1, z2, z3, mode=mode, series_terms=series_terms, dps=dps
        )
    return total


def _triangle_cross_ratios(z: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Vectorized cross ratios for tetrahedra [z1,z2,z3,infinity]."""
    triangle_points = z[np.asarray(triangles, dtype=np.int64)]
    numerator = triangle_points[:, 0] - triangle_points[:, 2]
    denominator = triangle_points[:, 1] - triangle_points[:, 2]
    if np.any(denominator == 0.0):
        raise ValueError("Delaunay simplex contains coincident vertices")
    return numerator / denominator


def _volume_from_triangles_fast_series(
    z: np.ndarray,
    triangles: np.ndarray,
    series_terms: int,
) -> float:
    triangle_points = z[np.asarray(triangles, dtype=np.int64)]
    a1, a2, a3 = _angles_for_triangle_np(
        triangle_points[:, 0], triangle_points[:, 1], triangle_points[:, 2]
    )
    values = (
        _lob_value_series_np(a1, series_terms)
        + _lob_value_series_np(a2, series_terms)
        + _lob_value_series_np(a3, series_terms)
    )
    return float(np.sum(values))


def _volume_from_triangles_bloch_wigner(
    z: np.ndarray,
    triangles: np.ndarray,
) -> float:
    cross_ratios = _triangle_cross_ratios(z, triangles)
    return float(np.sum(np.abs(bloch_wigner(cross_ratios))))


def ideal_poly_volume_via_delaunay(
    z_finite_complex: np.ndarray,
    mode="fast",
    dps=120,
    series_terms=64,
    use_bloch_wigner=False,
):
    """
    Compute volume of ideal polyhedron using Delaunay triangulation.

    Args:
        z_finite_complex: Array of finite vertices (complex numbers)
        mode: "bloch_wigner"/"bw", "fast" (series), or "exact" (mpmath)
        dps: Decimal precision for exact mode
        series_terms: Number of terms for fast mode
        use_bloch_wigner: If True, use Bloch-Wigner dilogarithm (most accurate)

    Returns:
        Total volume as float
    """
    z_finite_complex = np.asarray(z_finite_complex, dtype=np.complex128)
    if z_finite_complex.ndim != 1:
        raise ValueError("z_finite_complex must be a one-dimensional array")
    if len(z_finite_complex) < 3:
        raise ValueError("at least three finite vertices are required")

    idx_tris = delaunay_triangulation_indices(z_finite_complex)

    if use_bloch_wigner or mode in ("bloch_wigner", "bw"):
        return _volume_from_triangles_bloch_wigner(z_finite_complex, idx_tris)

    if mode == "fast":
        return _volume_from_triangles_fast_series(
            z_finite_complex, idx_tris, series_terms
        )
    if mode in ("exact", "eval_only"):
        total = 0.0
        for i, j, k in idx_tris:
            total += triangle_volume_from_points(
                z_finite_complex[i],
                z_finite_complex[j],
                z_finite_complex[k],
                mode=mode,
                series_terms=series_terms,
                dps=dps,
            )
        return total
    raise ValueError(
        "mode must be 'bloch_wigner', 'fast', 'exact', or the legacy alias 'eval_only'"
    )


def ideal_poly_volume_from_sphere(
    points_sphere: np.ndarray,
    pole_index: int = 0,
    mode: str = "bloch_wigner",
    series_terms: int = 64,
    dps: int = 120,
) -> float:
    """Volume of an ideal H³ polyhedron from all of its points on S².

    The chosen input vertex is sent to infinity by
    :func:`stereographic_project_from_pole`; unlike a fixed-north projection,
    this is invariant under rotations of the sphere and under changing the
    chosen pole for generic configurations.

    ``mode='bloch_wigner'`` is the accurate vectorized double-precision path.
    ``mode='fast'`` uses the truncated Lobachevsky series, while ``'exact'``
    (and its legacy alias ``'eval_only'``) uses mpmath.
    """
    projected, _ = stereographic_project_from_pole(points_sphere, pole_index)
    if mode in ("bloch_wigner", "bw"):
        return ideal_poly_volume_via_delaunay(projected, use_bloch_wigner=True)
    if mode in ("fast", "exact", "eval_only"):
        return ideal_poly_volume_via_delaunay(
            projected,
            mode=mode,
            dps=dps,
            series_terms=series_terms,
            use_bloch_wigner=False,
        )
    raise ValueError("mode must be 'bloch_wigner', 'fast', 'exact', or 'eval_only'")
