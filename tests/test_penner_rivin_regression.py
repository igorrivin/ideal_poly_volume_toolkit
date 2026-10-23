"""Regression tests for the Penner-Rivin holonomy / realization pipeline.

Covers three bugs found during the AMPS-224/225 audit (2026-07):

1. INF-sentinel coercion: build_penner_rivin_from_hull coerced the None
   (infinity) sentinel to NaN via np.array(dtype=complex), so cr_axby's
   INF branches never fired and any point set containing the ideal vertex
   at infinity produced NaN shears.

2. Cross-ratio slot arrangement: compute_shears_from_vertices used a
   Klein-four translate of the standard cross-ratio, reporting
   |shear| ~ 0.35 on provably zero-shear (arithmetic) configurations.

3. realize_angles_as_points placed vertices at isosceles-triangle apexes
   (correct apex angle, wrong other two angles) with an arbitrary side
   choice, producing 27-38 degree angle errors on Rivin-optimal angle
   structures; its success flag also ignored the angle error, and
   check_arithmeticity could silently certify the zero-shear structure
   without ever consuming coordinates.

Exact references used below:
  - Regular ideal tetrahedron: vertices {0, 1, e^{i pi/3}, INF}; all
    edge shears are exactly 0 and all Penner-Rivin holonomy traces are
    integers (the gluing uses only the integral matrices L, R, X(0)).
  - Regular ideal octahedron: vertices {0, +-1, +-i, INF}; same property.
  - Both properties are Moebius-invariant, so any Moebius image must give
    identical shears (used to pin the cross-ratio convention on an
    all-finite configuration).
"""

import cmath
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ideal_poly_volume_toolkit"))

from ideal_poly_volume_toolkit.rivin_holonomy import (  # noqa: E402
    check_arithmeticity,
    check_arithmeticity_from_vertices,
    compute_shears_from_vertices,
)
import pointset_to_fuchsian as pf  # noqa: E402


TETRAHEDRON = [0j, 1 + 0j, cmath.exp(1j * cmath.pi / 3), None]
OCTAHEDRON = [0j, 1 + 0j, 1j, -1 + 0j, -1j, None]


def _group(points):
    return pf.group_from_pointset(points)


def _trace_int_distances(gens):
    out = []
    for M in gens:
        t = M[0][0] + M[1][1]
        t = t.real if isinstance(t, complex) else t
        out.append(abs(t - round(t)))
    return out


@pytest.mark.parametrize("points,label", [
    (TETRAHEDRON, "regular ideal tetrahedron"),
    (OCTAHEDRON, "regular ideal octahedron"),
])
def test_inf_sentinel_zero_shear_integer_traces(points, label):
    """Configurations containing the ideal vertex at infinity must pass
    through the INF-safe cross-ratio path: shears < 1e-12, traces exactly
    integral (regression: None -> NaN coercion made this impossible)."""
    res = _group(points)
    shears = np.array([res["Z"][k] for k in res["Z"]], dtype=float)
    assert np.all(np.isfinite(shears)), f"{label}: NaN shears (INF bug)"
    assert np.abs(shears).max() < 1e-12, f"{label}: nonzero shears"
    dists = _trace_int_distances(res["Gamma_generators"])
    assert max(dists) < 1e-9, f"{label}: non-integral traces"


def test_standard_cross_ratio_convention_all_finite():
    """A Moebius image of the octahedron (all vertices finite) must have
    zero shears under compute_shears_from_vertices (regression: the old
    slot arrangement reported |shear| ~ 0.35 here)."""
    c = 2 + 0.5j
    finite = [1 / (z - c) for z in OCTAHEDRON[:-1]] + [0j]  # INF -> 0
    pts3 = np.vstack([pf.stereographic_inverse(z) for z in finite])
    from scipy.spatial import ConvexHull
    tris = [tuple(int(x) for x in s) for s in ConvexHull(pts3).simplices]
    shears = compute_shears_from_vertices(tris, finite)
    assert max(abs(v) for v in shears.values()) < 1e-12


def test_check_arithmeticity_requires_geometry():
    """The certificate must refuse to run without coordinates or shears
    (regression: the zero-shear default certified any triangulation)."""
    tris = [(0, 1, 2), (0, 2, 3)]
    with pytest.raises(ValueError):
        check_arithmeticity(tris)


def test_realization_attains_rivin_angles():
    """realize_angles_as_points must attain the optimized angle structure
    (regression: isosceles placement gave 27-38 degree errors)."""
    from ideal_poly_volume_toolkit.rivin_delaunay import (
        check_delaunay_realizability,
        optimize_hyperbolic_volume,
        realize_angles_as_points,
    )
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(10)
    pts = rng.random((10, 2))
    tris = [tuple(int(x) for x in s) for s in Delaunay(pts).simplices]
    rl = check_delaunay_realizability(tris, verbose=False)
    assert rl["realizable"]
    opt = optimize_hyperbolic_volume(
        tris, initial_angles=rl["angles_radians"], verbose=False)
    assert opt["success"]
    real = realize_angles_as_points(tris, opt["angles"], verbose=False)
    assert real["success"]
    assert real["triangulation_preserved"]
    assert real["angle_error_degrees"] < 1e-6
    assert real["closure_error"] < 1e-9


def test_require_interior_early_tie_rejection():
    """With require_interior=True the optimizer must reject a combinatorial
    type whose volume maximum is a boundary (cocircular-tie) maximum —
    quickly, at the first tie activation — and must pass a known
    interior-maximum type through unchanged."""
    from ideal_poly_volume_toolkit.rivin_delaunay import (
        check_delaunay_realizability,
        optimize_hyperbolic_volume,
    )
    from scipy.spatial import Delaunay

    def run(seed, require_interior):
        rng = np.random.default_rng(seed)
        pts = rng.random((50, 2))
        tris = [tuple(int(x) for x in s) for s in Delaunay(pts).simplices]
        rl = check_delaunay_realizability(tris, verbose=False)
        assert rl["realizable"]
        return optimize_hyperbolic_volume(
            tris, initial_angles=rl["angles_radians"], verbose=False,
            require_interior=require_interior)

    # seed 1002: boundary maximum (10 active ties without the flag)
    boundary_off = run(1002, require_interior=False)
    assert boundary_off["success"]
    assert boundary_off["n_active_constraints"] >= 1  # boundary max indeed

    boundary_on = run(1002, require_interior=True)
    assert not boundary_on["success"]
    assert boundary_on.get("rejected_tie") is True
    assert ("tie" in boundary_on["message"].lower()
            or "boundary maximum" in boundary_on["message"])
    # rejected at the FIRST tie activation: strictly fewer Newton
    # iterations than the full boundary convergence
    assert boundary_on["n_iterations"] < boundary_off["n_iterations"]

    # seed 1007: interior maximum — must pass through unchanged
    interior_on = run(1007, require_interior=True)
    assert interior_on["success"]
    assert not interior_on.get("rejected_tie", False)


def test_interior_optimum_zero_shear_diagnostic_is_not_an_open_certificate():
    """End-to-end: a combinatorial type whose volume maximum is interior
    to the Delaunay polytope realizes with ~zero shears, but its finite
    Delaunay disk must not be mislabeled an arithmetic closed surface."""
    from ideal_poly_volume_toolkit.rivin_delaunay import (
        check_delaunay_realizability,
        optimize_hyperbolic_volume,
        realize_angles_as_points,
    )
    from scipy.spatial import Delaunay

    # seed chosen so the optimum has no active Delaunay (tie) constraints
    rng = np.random.default_rng(1007)
    N = 50
    pts = rng.random((N, 2))
    tris = [tuple(int(x) for x in s) for s in Delaunay(pts).simplices]
    rl = check_delaunay_realizability(tris, verbose=False)
    opt = optimize_hyperbolic_volume(
        tris, initial_angles=rl["angles_radians"], verbose=False)
    real = realize_angles_as_points(tris, opt["angles"], verbose=False)
    assert real["success"]

    vc = np.zeros(N, dtype=complex)
    for i, v in enumerate(real["vertex_list"]):
        vc[v] = complex(real["points"][i, 0], real["points"][i, 1])
    open_shears = compute_shears_from_vertices(tris, vc)
    assert max(abs(s) for s in open_shears.values()) < 1e-10

    with pytest.raises(ValueError, match="incidence 1"):
        check_arithmeticity(tris, vertices_complex=vc, tolerance=1e-6)

    closed = check_arithmeticity_from_vertices(vc, tolerance=1e-6)
    assert closed["topology"]["n_vertices"] == N
    assert closed["topology"]["n_faces"] == 2 * N - 4
    assert closed["n_generators"] == N - 1
    assert not closed["is_arithmetic"]
    assert closed["arithmeticity_status"] == "not_certified"
