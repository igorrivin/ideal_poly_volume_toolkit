
"""
Transformations between the standard real hyperbolic models:

  - Klein ball
  - Poincare ball
  - Upper half-space
  - Minkowski / hyperboloid model

The conversion routines are dimension-agnostic. They work in H^n for any n >= 1,
provided the input coordinate lengths are compatible.

Conventions
-----------
1. The Minkowski bilinear form has signature (-,+,...,+):
       <X,Y> = -X_0 Y_0 + X_1 Y_1 + ... + X_n Y_n.

2. Finite points in the Minkowski model are represented on the upper sheet
       H^n = { X : <X,X> = -1, X_0 > 0 }.
   Ideal points are represented by future-pointing lightlike vectors, returned in
   canonical normalization when possible.

3. The half-space model is
       U^n = { (x,t) in R^{n-1} x R_{>0} }.
   Ideal boundary points may be given either as points with last coordinate t = 0,
   or as the sentinel string HALFSPACE_INFINITY for the point at infinity.

4. The Cayley transform convention is the standard one sending the north pole
   (0,...,0,1) on the ball boundary to HALFSPACE_INFINITY.

Typical usage
-------------
    from hyperbolic_model_transforms import convert_vertices, HALFSPACE_INFINITY

    verts_half = [
        HALFSPACE_INFINITY,
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ]
    verts_klein = convert_vertices(verts_half, from_model="halfspace", to_model="klein")
"""

from __future__ import annotations

from typing import Any, Sequence

import mpmath as mp



HALFSPACE_INFINITY = "infinity"


def halfspace_infinity_minkowski(dim: int) -> list[mp.mpf]:
    """
    Canonical future-lightlike lift of the half-space point at infinity in H^dim.
    """
    if dim < 1:
        raise ValueError("the ambient dimension must be positive")
    return [mp.mpf("1")] + [mp.mpf("0")] * (dim - 1) + [mp.mpf("1")]


def halfspace_infinity(
    dim: int,
    to_model: str = "halfspace",
    *,
    dps: int = 50,
    snap_ideal_tol: float | int | mp.mpf | None = None,
) -> Any:
    """
    Return the half-space point at infinity expressed in the requested model.

    Examples:
        halfspace_infinity(4, to_model="klein")     -> [0,0,0,1]
        halfspace_infinity(4, to_model="minkowski") -> [1,0,0,0,1]
    """
    dst = _canonical_model_name(to_model)
    if dst == "halfspace":
        return HALFSPACE_INFINITY
    X = halfspace_infinity_minkowski(dim)
    if dst == "minkowski":
        return X
    if dst == "klein":
        return minkowski_to_klein(X, dps=dps, snap_ideal_tol=snap_ideal_tol)
    if dst == "poincare":
        return minkowski_to_poincare(X, dps=dps, snap_ideal_tol=snap_ideal_tol)
    return minkowski_to_halfspace(X, dps=dps, snap_ideal_tol=snap_ideal_tol)


def _set_precision(dps: int) -> None:
    mp.mp.dps = int(dps)


def _default_tol(multiplier: float | int = 1000) -> mp.mpf:
    return mp.mpf(multiplier) * mp.sqrt(mp.eps)


def _to_mpf(x: Any) -> mp.mpf:
    return mp.mpf(x)


def _dot(u: Sequence[mp.mpf], v: Sequence[mp.mpf]) -> mp.mpf:
    return mp.fsum(ui * vi for ui, vi in zip(u, v))


def _norm2(u: Sequence[mp.mpf]) -> mp.mpf:
    return _dot(u, u)


def minkowski_inner(x: Sequence[Any], y: Sequence[Any]) -> mp.mpf:
    """
    Minkowski bilinear form with signature (-,+,...,+).
    """
    xx = [_to_mpf(v) for v in x]
    yy = [_to_mpf(v) for v in y]
    if len(xx) != len(yy):
        raise ValueError("Minkowski vectors must have the same length")
    return -xx[0] * yy[0] + mp.fsum(xx[i] * yy[i] for i in range(1, len(xx)))


def _canonical_model_name(model: str) -> str:
    name = model.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "klein": "klein",
        "projective": "klein",
        "beltrami_klein": "klein",
        "beltrami": "klein",
        "poincare": "poincare",
        "poincare_ball": "poincare",
        "poincareball": "poincare",
        "ball": "poincare",
        "halfspace": "halfspace",
        "upper_halfspace": "halfspace",
        "upper_half_space": "halfspace",
        "half_space": "halfspace",
        "upperhalfspace": "halfspace",
        "uhs": "halfspace",
        "minkowski": "minkowski",
        "hyperboloid": "minkowski",
        "lorentz": "minkowski",
        "lorentzian": "minkowski",
        "hyperboloid_model": "minkowski",
    }
    if name not in aliases:
        raise ValueError(f"unknown model {model!r}")
    return aliases[name]


def _is_halfspace_infinity(point: Any) -> bool:
    if isinstance(point, str):
        return point.strip().lower() in {"inf", "infinity", "halfspace_infinity"}
    # Sequence-like points (especially NumPy arrays) implement elementwise
    # equality, which cannot be used as a scalar truth value.
    return False


def _coerce_ball_point(
    point: Sequence[Any],
    *,
    snap_ideal_tol: mp.mpf,
    model_name: str,
) -> tuple[list[mp.mpf], str]:
    out = [_to_mpf(v) for v in point]
    if len(out) < 1:
        raise ValueError(f"{model_name} point must have positive dimension")
    r2 = _norm2(out)
    if abs(r2 - 1) <= snap_ideal_tol:
        if r2 <= 0:
            raise ValueError(f"cannot snap the zero vector on the {model_name} boundary")
        r = mp.sqrt(r2)
        out = [v / r for v in out]
        r2 = mp.mpf("1")
    if r2 < 1:
        return out, "finite"
    if r2 == 1:
        return out, "ideal"
    raise ValueError(f"{model_name} point has squared norm {r2}, so it lies outside the unit ball")


def _normalize_minkowski_point(
    point: Sequence[Any],
    *,
    snap_ideal_tol: mp.mpf,
) -> tuple[list[mp.mpf], str]:
    out = [_to_mpf(v) for v in point]
    if len(out) < 2:
        raise ValueError("Minkowski point must have length at least 2")
    q = minkowski_inner(out, out)
    if abs(q) <= snap_ideal_tol:
        # Future-pointing lightlike ray; canonicalize by making X_0 > 0 and, if possible, X_0 = 1.
        if out[0] < 0:
            out = [-v for v in out]
        if abs(out[0]) <= snap_ideal_tol:
            raise ValueError("lightlike Minkowski vector must have nonzero time coordinate")
        scale = out[0]
        out = [v / scale for v in out]
        return out, "ideal"
    if q < 0:
        scale = mp.sqrt(-q)
        out = [v / scale for v in out]
        if out[0] < 0:
            out = [-v for v in out]
        return out, "finite"
    raise ValueError(
        f"Minkowski input has <X,X> = {q}, so it is spacelike; "
        "expected a timelike hyperboloid point or a lightlike ideal point"
    )


def klein_to_minkowski(
    point: Sequence[Any],
    *,
    dps: int = 50,
    snap_ideal_tol: float | int | mp.mpf | None = None,
) -> list[mp.mpf]:
    """
    Convert a Klein-ball point to the Minkowski / hyperboloid model.

    Finite point x in B^n:
        X = (1, x) / sqrt(1 - ||x||^2)

    Ideal point x on S^{n-1}:
        X = (1, x)
    """
    _set_precision(dps)
    tol = _default_tol() if snap_ideal_tol is None else mp.mpf(snap_ideal_tol)
    x, geometry = _coerce_ball_point(point, snap_ideal_tol=tol, model_name="Klein")
    r2 = _norm2(x)
    if geometry == "finite":
        den = mp.sqrt(1 - r2)
        return [mp.mpf("1") / den] + [v / den for v in x]
    return [mp.mpf("1")] + x


def poincare_to_minkowski(
    point: Sequence[Any],
    *,
    dps: int = 50,
    snap_ideal_tol: float | int | mp.mpf | None = None,
) -> list[mp.mpf]:
    """
    Convert a Poincare-ball point to the Minkowski / hyperboloid model.

    Finite point p in B^n:
        X_0 = (1 + ||p||^2) / (1 - ||p||^2),
        X_i = 2 p_i / (1 - ||p||^2)

    Ideal point p on S^{n-1}:
        X = (1, p)
    """
    _set_precision(dps)
    tol = _default_tol() if snap_ideal_tol is None else mp.mpf(snap_ideal_tol)
    p, geometry = _coerce_ball_point(point, snap_ideal_tol=tol, model_name="Poincare")
    r2 = _norm2(p)
    if geometry == "finite":
        den = 1 - r2
        return [(1 + r2) / den] + [2 * v / den for v in p]
    return [mp.mpf("1")] + p


def halfspace_to_minkowski(
    point: Any,
    *,
    dps: int = 50,
    snap_ideal_tol: float | int | mp.mpf | None = None,
) -> list[mp.mpf]:
    """
    Convert an upper-half-space point to the Minkowski / hyperboloid model.

    Finite point (x,t), with x in R^{n-1}, t > 0:
        X_0 = (||x||^2 + t^2 + 1) / (2t),
        X_i = x_i / t,   1 <= i <= n-1,
        X_n = (||x||^2 + t^2 - 1) / (2t)

    Boundary point (x,0):
        X_0 = (||x||^2 + 1) / 2,
        X_i = x_i,
        X_n = (||x||^2 - 1) / 2

    Infinity:
        X = (1, 0, ..., 0, 1)
    """
    _set_precision(dps)
    tol = _default_tol() if snap_ideal_tol is None else mp.mpf(snap_ideal_tol)

    if _is_halfspace_infinity(point):
        raise ValueError(
            "halfspace_to_minkowski cannot infer the ambient dimension from the infinity sentinel alone; "
            "use halfspace_infinity_minkowski(dim), halfspace_infinity(dim, to_model=...), or convert_vertices"
        )

    h = [_to_mpf(v) for v in point]
    if len(h) < 1:
        raise ValueError("half-space point must have positive dimension")
    x = h[:-1]
    t = h[-1]
    r2 = _norm2(x)

    if abs(t) <= tol:
        t = mp.mpf("0")
    if t > 0:
        return [(r2 + t * t + 1) / (2 * t)] + [v / t for v in x] + [(r2 + t * t - 1) / (2 * t)]
    if t == 0:
        return [(r2 + 1) / 2] + x + [(r2 - 1) / 2]
    raise ValueError(f"half-space height must be nonnegative; got t = {t}")


def minkowski_to_klein(
    point: Sequence[Any],
    *,
    dps: int = 50,
    snap_ideal_tol: float | int | mp.mpf | None = None,
) -> list[mp.mpf]:
    """
    Convert a Minkowski / hyperboloid point to the Klein model.

    Finite X on the hyperboloid:
        x = X_spatial / X_0

    Ideal lightlike X:
        x = X_spatial / X_0
    """
    _set_precision(dps)
    tol = _default_tol() if snap_ideal_tol is None else mp.mpf(snap_ideal_tol)
    X, _ = _normalize_minkowski_point(point, snap_ideal_tol=tol)
    return [v / X[0] for v in X[1:]]


def minkowski_to_poincare(
    point: Sequence[Any],
    *,
    dps: int = 50,
    snap_ideal_tol: float | int | mp.mpf | None = None,
) -> list[mp.mpf]:
    """
    Convert a Minkowski / hyperboloid point to the Poincare ball.

    Finite X on the hyperboloid:
        p = X_spatial / (X_0 + 1)

    Ideal lightlike X:
        p = X_spatial / X_0
    """
    _set_precision(dps)
    tol = _default_tol() if snap_ideal_tol is None else mp.mpf(snap_ideal_tol)
    X, geometry = _normalize_minkowski_point(point, snap_ideal_tol=tol)
    if geometry == "finite":
        return [v / (X[0] + 1) for v in X[1:]]
    return [v / X[0] for v in X[1:]]


def minkowski_to_halfspace(
    point: Sequence[Any],
    *,
    dps: int = 50,
    snap_ideal_tol: float | int | mp.mpf | None = None,
) -> list[mp.mpf] | str:
    """
    Convert a Minkowski / hyperboloid point to upper half-space.

    Finite X on the hyperboloid:
        x_i = X_i / (X_0 - X_n),  1 <= i <= n-1,
        t   = 1 / (X_0 - X_n)

    Ideal lightlike X:
        x_i = X_i / (X_0 - X_n),  1 <= i <= n-1,   if X_0 != X_n
        infinity if X_0 = X_n
    """
    _set_precision(dps)
    tol = _default_tol() if snap_ideal_tol is None else mp.mpf(snap_ideal_tol)
    X, geometry = _normalize_minkowski_point(point, snap_ideal_tol=tol)
    den = X[0] - X[-1]
    if geometry == "finite":
        if den <= 0:
            raise ValueError("expected X_0 - X_n > 0 for a finite hyperboloid point")
        return [v / den for v in X[1:-1]] + [mp.mpf("1") / den]
    if abs(den) <= tol:
        return HALFSPACE_INFINITY
    return [v / den for v in X[1:-1]] + [mp.mpf("0")]


def convert_point(
    point: Any,
    from_model: str,
    to_model: str,
    *,
    dps: int = 50,
    snap_ideal_tol: float | int | mp.mpf | None = None,
    ambient_dim: int | None = None,
) -> Any:
    """
    Convert one point between the standard hyperbolic models.

    Supported model names:
        "klein", "poincare", "halfspace", "minkowski"

    For half-space, the point at infinity is represented by HALFSPACE_INFINITY.
    If you convert that point on its own, pass ambient_dim as well.
    """
    _set_precision(dps)
    src = _canonical_model_name(from_model)
    dst = _canonical_model_name(to_model)
    tol = _default_tol() if snap_ideal_tol is None else mp.mpf(snap_ideal_tol)

    if src == dst:
        if src == "minkowski":
            return _normalize_minkowski_point(point, snap_ideal_tol=tol)[0]
        if src == "halfspace" and _is_halfspace_infinity(point):
            return HALFSPACE_INFINITY
        if src in {"klein", "poincare"}:
            return _coerce_ball_point(point, snap_ideal_tol=tol, model_name=src)[0]
        h = [_to_mpf(v) for v in point]
        if len(h) < 1:
            raise ValueError("half-space point must have positive dimension")
        if abs(h[-1]) <= tol:
            h[-1] = mp.mpf("0")
        if h[-1] < 0:
            raise ValueError(f"half-space height must be nonnegative; got t = {h[-1]}")
        return h

    if src == "minkowski":
        X = _normalize_minkowski_point(point, snap_ideal_tol=tol)[0]
    elif src == "klein":
        X = klein_to_minkowski(point, dps=dps, snap_ideal_tol=tol)
    elif src == "poincare":
        X = poincare_to_minkowski(point, dps=dps, snap_ideal_tol=tol)
    else:
        # half-space
        if _is_halfspace_infinity(point):
            if ambient_dim is None:
                raise ValueError(
                    "convert_point cannot infer the ambient dimension from HALFSPACE_INFINITY alone; "
                    "pass ambient_dim=... or use convert_vertices when an infinity vertex is present"
                )
            X = halfspace_infinity_minkowski(int(ambient_dim))
        else:
            X = halfspace_to_minkowski(point, dps=dps, snap_ideal_tol=tol)

    if dst == "minkowski":
        return X
    if dst == "klein":
        return minkowski_to_klein(X, dps=dps, snap_ideal_tol=tol)
    if dst == "poincare":
        return minkowski_to_poincare(X, dps=dps, snap_ideal_tol=tol)
    return minkowski_to_halfspace(X, dps=dps, snap_ideal_tol=tol)


def convert_vertices(
    vertices: Sequence[Any],
    from_model: str,
    to_model: str,
    *,
    dps: int = 50,
    snap_ideal_tol: float | int | mp.mpf | None = None,
) -> list[Any]:
    """
    Convert a collection of vertices between the standard hyperbolic models.

    This is the preferred entry point when half-space infinity is present, because the ambient
    dimension can then be inferred from the other vertices.
    """
    _set_precision(dps)
    src = _canonical_model_name(from_model)
    dst = _canonical_model_name(to_model)
    tol = _default_tol() if snap_ideal_tol is None else mp.mpf(snap_ideal_tol)

    if src != "halfspace":
        return [convert_point(v, src, dst, dps=dps, snap_ideal_tol=tol) for v in vertices]

    # Half-space source: first infer the ambient dimension.
    dim: int | None = None
    for v in vertices:
        if _is_halfspace_infinity(v):
            continue
        vv = list(v)
        dim = len(vv)
        break
    if dim is None:
        raise ValueError("cannot infer the half-space dimension from only infinity vertices")

    minkowski_points: list[list[mp.mpf]] = []
    infinity_lift = [mp.mpf("1")] + [mp.mpf("0")] * (dim - 1) + [mp.mpf("1")]
    for v in vertices:
        if _is_halfspace_infinity(v):
            minkowski_points.append(infinity_lift[:])
        else:
            minkowski_points.append(halfspace_to_minkowski(v, dps=dps, snap_ideal_tol=tol))

    if dst == "minkowski":
        return minkowski_points
    if dst == "klein":
        return [minkowski_to_klein(v, dps=dps, snap_ideal_tol=tol) for v in minkowski_points]
    if dst == "poincare":
        return [minkowski_to_poincare(v, dps=dps, snap_ideal_tol=tol) for v in minkowski_points]
    if dst == "halfspace":
        return [minkowski_to_halfspace(v, dps=dps, snap_ideal_tol=tol) for v in minkowski_points]
    raise ValueError(f"unknown target model {to_model!r}")


def to_klein(
    point: Any,
    from_model: str,
    *,
    dps: int = 50,
    snap_ideal_tol: float | int | mp.mpf | None = None,
    ambient_dim: int | None = None,
) -> Any:
    return convert_point(
        point,
        from_model,
        "klein",
        dps=dps,
        snap_ideal_tol=snap_ideal_tol,
        ambient_dim=ambient_dim,
    )


def to_poincare(
    point: Any,
    from_model: str,
    *,
    dps: int = 50,
    snap_ideal_tol: float | int | mp.mpf | None = None,
    ambient_dim: int | None = None,
) -> Any:
    return convert_point(
        point,
        from_model,
        "poincare",
        dps=dps,
        snap_ideal_tol=snap_ideal_tol,
        ambient_dim=ambient_dim,
    )


def to_halfspace(
    point: Any,
    from_model: str,
    *,
    dps: int = 50,
    snap_ideal_tol: float | int | mp.mpf | None = None,
    ambient_dim: int | None = None,
) -> Any:
    return convert_point(
        point,
        from_model,
        "halfspace",
        dps=dps,
        snap_ideal_tol=snap_ideal_tol,
        ambient_dim=ambient_dim,
    )


def to_minkowski(
    point: Any,
    from_model: str,
    *,
    dps: int = 50,
    snap_ideal_tol: float | int | mp.mpf | None = None,
    ambient_dim: int | None = None,
) -> Any:
    return convert_point(
        point,
        from_model,
        "minkowski",
        dps=dps,
        snap_ideal_tol=snap_ideal_tol,
        ambient_dim=ambient_dim,
    )
