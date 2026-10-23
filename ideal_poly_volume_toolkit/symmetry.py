"""
Symmetry group computation for ideal polyhedra using nauty/pynauty.

Computes the automorphism group of the 1-skeleton (graph) of the polyhedron.

``audit_hyperbolic_symmetry`` is the dimension-independent, geometric entry
point.  It first computes the automorphism group of the full vertex--facet
incidence graph and then filters those combinatorial automorphisms by the
conformal Gram equations for the ideal null rays.  A direct Lorentz fit is
retained only as a diagnostic because it is ill-conditioned in strongly
boosted boundary charts.  The older ``compute_symmetry_group`` API below is
retained unchanged for compatibility.
"""

from collections import deque
from itertools import combinations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

try:
    import pynauty

    PYNAUTY_AVAILABLE = True
except ImportError:
    PYNAUTY_AVAILABLE = False


def _empty_hyperbolic_audit(max_group_order, tolerances):
    """Return the stable JSON-native result skeleton used on every path."""
    return {
        "schema_version": 1,
        "available": bool(PYNAUTY_AVAILABLE),
        "dimension": None,
        "n_vertices": None,
        "valid_geometry": False,
        "checks": {
            "shape": False,
            "finite": False,
            "unit": False,
            "distinct": False,
            "full_dimensional": False,
            "all_vertices_extreme": False,
            "simplicial": False,
            "strict_support": False,
        },
        "topology": {
            "affine_rank": None,
            "min_affine_singular_value": None,
            "max_unit_norm_error": None,
            "min_pairwise_distance": None,
            "n_facets": None,
            "min_support_margin": None,
            "max_incident_plane_residual": None,
            "max_vertices_on_support": None,
            "min_lift_singular_value": None,
            "lift_condition_number": None,
            "lift_condition_error_bound": None,
        },
        "combinatorial_group_order": None,
        "combinatorial_generators": [],
        "combinatorial_vertex_orbits": [],
        "geometric_group_order": None,
        "geometric_generators": [],
        "vertex_orbits": [],
        "orientation_preserving_count": None,
        "candidate_permutations_tested": 0,
        "geometric_candidate_count": 0,
        "indeterminate_permutations": 0,
        "max_group_order": int(max_group_order),
        "full_group_certified": False,
        "certified_nontrivial_symmetry": False,
        "residuals": {
            "max_accepted_isometry_residual": None,
            "max_accepted_scaling_residual": None,
            "max_accepted_mapping_residual": None,
            "max_accepted_lorentz_residual": None,
            "min_rejected_isometry_residual": None,
            "rejection_gap_threshold": float(10.0 * tolerances["isometry"]),
        },
        "tolerances": dict(tolerances),
        "failure_reason": None,
    }


def _validate_audit_controls(
    unit_tolerance,
    distinct_tolerance,
    support_tolerance,
    isometry_tolerance,
    max_group_order,
):
    tolerances = {
        "unit": unit_tolerance,
        "distinct": distinct_tolerance,
        "support": support_tolerance,
        "isometry": isometry_tolerance,
    }
    for name, value in tolerances.items():
        try:
            valid = np.isscalar(value) and np.isfinite(value) and value > 0
        except TypeError:
            valid = False
        if not valid:
            raise ValueError(f"{name}_tolerance must be a positive finite number")
        tolerances[name] = float(value)
    try:
        integral_group_cap = int(max_group_order)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_group_order must be a positive integer") from exc
    if (
        isinstance(max_group_order, (bool, np.bool_))
        or integral_group_cap != max_group_order
    ):
        raise ValueError("max_group_order must be a positive integer")
    max_group_order = integral_group_cap
    if max_group_order < 1:
        raise ValueError("max_group_order must be a positive integer")
    return tolerances, max_group_order


def _fail_audit(result, reason):
    result["failure_reason"] = str(reason)
    return result


def _strict_simplicial_facets(vertices, hull, support_tolerance):
    """Recover strict supporting facets without trusting Qhull diagonals."""
    dimension = vertices.shape[1]
    facets = {}
    min_support_margin = np.inf
    max_incident_residual = 0.0
    max_vertices_on_support = 0

    for equation in hull.equations:
        normal = np.asarray(equation[:-1], dtype=np.float64)
        normal_norm = float(np.linalg.norm(normal))
        if not np.isfinite(normal_norm) or normal_norm == 0.0:
            return None, None, "Qhull returned an invalid supporting hyperplane"
        signed = (vertices @ normal + float(equation[-1])) / normal_norm
        incident = tuple(
            int(index) for index in np.flatnonzero(np.abs(signed) <= support_tolerance)
        )
        max_vertices_on_support = max(max_vertices_on_support, len(incident))
        if len(incident) != dimension:
            return (
                None,
                {
                    "min_support_margin": None,
                    "max_incident_plane_residual": float(
                        np.max(np.abs(signed[list(incident)]))
                    )
                    if incident
                    else None,
                    "max_vertices_on_support": int(max_vertices_on_support),
                },
                "hull has a nonsimplicial, coplanar, or numerically "
                f"unresolved facet with {len(incident)} incident vertices",
            )

        incident_values = np.abs(signed[list(incident)])
        max_incident_residual = max(
            max_incident_residual, float(np.max(incident_values))
        )
        nonincident = np.ones(len(vertices), dtype=bool)
        nonincident[list(incident)] = False
        if np.any(signed[nonincident] > support_tolerance):
            return None, None, "Qhull supporting-plane orientation is inconsistent"
        margin = float(np.min(-signed[nonincident]))
        min_support_margin = min(min_support_margin, margin)
        if margin <= support_tolerance:
            return (
                None,
                {
                    "min_support_margin": float(margin),
                    "max_incident_plane_residual": float(max_incident_residual),
                    "max_vertices_on_support": int(max_vertices_on_support),
                },
                "hull has a non-strict or numerically unresolved supporting facet",
            )
        facets[tuple(sorted(incident))] = True

    facet_list = sorted(facets)
    if len(facet_list) != len(hull.simplices):
        return (
            None,
            {
                "min_support_margin": float(min_support_margin),
                "max_incident_plane_residual": float(max_incident_residual),
                "max_vertices_on_support": int(max_vertices_on_support),
            },
            "Qhull triangulated or duplicated a supporting facet",
        )
    diagnostics = {
        "min_support_margin": float(min_support_margin),
        "max_incident_plane_residual": float(max_incident_residual),
        "max_vertices_on_support": int(max_vertices_on_support),
    }
    return facet_list, diagnostics, None


def _nauty_group_order(mantissa, exponent):
    value = float(mantissa) * (10 ** int(exponent))
    nearest = round(value)
    if not np.isfinite(value) or abs(value - nearest) > 1e-7 * max(1.0, value):
        raise ValueError("pynauty returned a non-integral automorphism-group order")
    return int(nearest)


def _incidence_automorphisms(n_vertices, facets):
    """Return vertex-action generators and order of the full face incidence."""
    n_facets = len(facets)
    n_nodes = n_vertices + n_facets
    adjacency = {node: [] for node in range(n_nodes)}
    for facet_index, facet in enumerate(facets, start=n_vertices):
        for vertex in facet:
            adjacency[vertex].append(facet_index)
            adjacency[facet_index].append(vertex)
    graph = pynauty.Graph(
        number_of_vertices=n_nodes,
        directed=False,
        adjacency_dict=adjacency,
        vertex_coloring=[
            set(range(n_vertices)),
            set(range(n_vertices, n_nodes)),
        ],
    )
    generators, mantissa, exponent, orbit_labels, _ = pynauty.autgrp(graph)
    restricted = []
    seen = set()
    for generator in generators:
        permutation = tuple(int(generator[index]) for index in range(n_vertices))
        if permutation not in seen:
            restricted.append(permutation)
            seen.add(permutation)
    order = _nauty_group_order(mantissa, exponent)
    vertex_orbits = {}
    for vertex, label in enumerate(orbit_labels[:n_vertices]):
        vertex_orbits.setdefault(int(label), []).append(int(vertex))
    return restricted, order, sorted(vertex_orbits.values(), key=lambda orbit: orbit[0])


def _compose_permutations(left, right):
    """Return ``left o right`` for image-list permutations."""
    return tuple(left[right[index]] for index in range(len(left)))


def _enumerate_permutation_group(generators, degree, cap):
    identity = tuple(range(degree))
    generators = [tuple(generator) for generator in generators]
    elements = {identity}
    pending = deque([identity])
    while pending:
        element = pending.popleft()
        for generator in generators:
            product = _compose_permutations(element, generator)
            if product in elements:
                continue
            if len(elements) >= cap:
                raise OverflowError("permutation enumeration exceeded max_group_order")
            elements.add(product)
            pending.append(product)
    return elements


def _lorentz_fit_for_permutation(vertices, permutation):
    """Fit one ideal-vertex permutation and audit its conformal Gram data.

    For future null lifts ``v_i=(x_i,1)``, a permutation is induced by a
    future-preserving projective Lorentz map precisely when there are positive
    ray scales ``lambda_i`` for which

    ``<v_i,v_j> = lambda_i lambda_j <v_p(i),v_p(j)>``.

    The logarithmic ray-scale consistency residual is therefore the
    authoritative isometry residual below.  The explicitly fitted Lorentz
    matrix is retained only as a diagnostic; its raw ``A.T @ eta @ A``
    residual is *not* used for classification because that subtraction is
    badly conditioned after a large, perfectly valid Lorentz boost.  Chord
    squares compute the null inner products without the cancellation in
    ``1 - dot(x_i, x_j)``.  Orientation is computed separately from the exact
    action on the coherently oriented facet sphere.
    """
    n_vertices, dimension = vertices.shape
    lifts = np.column_stack((vertices, np.ones(n_vertices, dtype=np.float64)))
    pair_rows = []
    log_ratios = []
    for first, second in combinations(range(n_vertices), 2):
        source_delta = vertices[first] - vertices[second]
        target_delta = vertices[permutation[first]] - vertices[permutation[second]]
        source_product = 0.5 * float(np.dot(source_delta, source_delta))
        target_product = 0.5 * float(np.dot(target_delta, target_delta))
        if source_product <= 0.0 or target_product <= 0.0:
            return None
        row = np.zeros(n_vertices, dtype=np.float64)
        row[first] = 1.0
        row[second] = 1.0
        pair_rows.append(row)
        log_ratios.append(np.log(source_product) - np.log(target_product))
    scale_system = np.asarray(pair_rows, dtype=np.float64)
    log_ratios = np.asarray(log_ratios, dtype=np.float64)
    log_scales, _, scale_rank, _ = np.linalg.lstsq(scale_system, log_ratios, rcond=None)
    if scale_rank != n_vertices or not np.all(np.isfinite(log_scales)):
        return None
    scaling_residual = float(np.max(np.abs(scale_system @ log_scales - log_ratios)))
    mapping_residual = None
    lorentz_residual = None
    future_time_component = None
    determinant = None
    if float(np.max(np.abs(log_scales))) <= 700.0:
        scales = np.exp(log_scales)
        target_lifts = scales[:, None] * lifts[np.asarray(permutation, dtype=int)]
        lorentz_transpose, _, fit_rank, _ = np.linalg.lstsq(
            lifts, target_lifts, rcond=None
        )
        if fit_rank == dimension + 1 and np.all(np.isfinite(lorentz_transpose)):
            lorentz = lorentz_transpose.T
            mapped = lifts @ lorentz.T
            row_scales = np.maximum(1.0, np.linalg.norm(target_lifts, axis=1))
            mapping_residual = float(
                np.max(np.linalg.norm(mapped - target_lifts, axis=1) / row_scales)
            )
            metric = np.diag(np.r_[np.ones(dimension), -1.0])
            lorentz_residual = float(
                np.max(np.abs(lorentz.T @ metric @ lorentz - metric))
            )
            future_time_component = float(lorentz[-1, -1])
            determinant = float(np.linalg.det(lorentz))
    # This is the conformal-Gram consistency residual.  Do not fold the raw
    # matrix-fit residuals into it: exact Lorentz matrices can have enormous
    # Euclidean condition numbers in a strongly boosted chart.
    isometry_residual = scaling_residual
    if not np.isfinite(isometry_residual):
        return None
    diagnostics = [
        mapping_residual,
        lorentz_residual,
        future_time_component,
        determinant,
    ]
    if any(value is not None and not np.isfinite(value) for value in diagnostics):
        mapping_residual = None
        lorentz_residual = None
        future_time_component = None
        determinant = None
    return {
        "isometry_residual": float(isometry_residual),
        "scaling_residual": float(scaling_residual),
        "mapping_residual": mapping_residual,
        "lorentz_residual": lorentz_residual,
        "future_time_component": future_time_component,
        "determinant": determinant,
    }


def _facet_orientation_signs(facets, dimension):
    """Orient a simplicial sphere using only its exact facet incidences."""
    facets = [tuple(facet) for facet in facets]
    facet_set = set(facets)
    ridge_incidence = {}
    for facet in facets:
        if len(facet) != dimension:
            raise ValueError("facet has the wrong dimension")
        for omitted in range(dimension):
            ridge = facet[:omitted] + facet[omitted + 1 :]
            ridge_incidence.setdefault(ridge, []).append((facet, omitted))
    if any(len(incident) != 2 for incident in ridge_incidence.values()):
        raise ValueError("facet complex is not a closed pseudomanifold")

    seed = min(facets)
    signs = {seed: 1}
    pending = deque([seed])
    while pending:
        facet = pending.popleft()
        for omitted in range(dimension):
            ridge = facet[:omitted] + facet[omitted + 1 :]
            first, second = ridge_incidence[ridge]
            other, other_omitted = second if first[0] == facet else first
            required = -signs[facet] * (-1) ** (omitted + other_omitted)
            if other in signs:
                if signs[other] != required:
                    raise ValueError("facet complex is not coherently orientable")
                continue
            signs[other] = required
            pending.append(other)
    if set(signs) != facet_set:
        raise ValueError("facet dual graph is disconnected")
    return signs


def _tuple_sort_parity(values):
    inversions = sum(
        values[first] > values[second]
        for first in range(len(values))
        for second in range(first + 1, len(values))
    )
    return -1 if inversions % 2 else 1


def _boundary_orientation_sign(permutation, facet_signs):
    """Return the exact orientation action of a facet automorphism."""
    source = min(facet_signs)
    image_ordered = tuple(permutation[vertex] for vertex in source)
    image = tuple(sorted(image_ordered))
    if image not in facet_signs:
        raise ValueError("permutation does not preserve the facet complex")
    return facet_signs[source] * _tuple_sort_parity(image_ordered) * facet_signs[image]


def _max_optional(values):
    finite = [float(value) for value in values if value is not None]
    return max(finite) if finite else None


def _orbits_from_group(elements, degree):
    unseen = set(range(degree))
    orbits = []
    while unseen:
        seed = min(unseen)
        orbit = sorted({int(permutation[seed]) for permutation in elements})
        orbits.append(orbit)
        unseen.difference_update(orbit)
    return orbits


def _generators_from_elements(elements, degree):
    identity = tuple(range(degree))
    target = set(elements)
    generated = {identity}
    generators = []
    for permutation in sorted(target):
        if permutation in generated:
            continue
        generators.append(permutation)
        generated = _enumerate_permutation_group(
            generators, degree, max(1, len(target))
        )
        if generated == target:
            break
    if generated != target:
        raise ValueError("accepted geometric permutations do not form a group")
    return generators


def audit_hyperbolic_symmetry(
    vertices,
    *,
    unit_tolerance=1e-9,
    distinct_tolerance=1e-8,
    support_tolerance=1e-9,
    isometry_tolerance=1e-7,
    max_group_order=100_000,
):
    """Audit the full geometric symmetry group of ideal points in ``H^d``.

    The rows of ``vertices`` are boundary points on ``S^{d-1}``.  The audit
    validates a strict simplicial Euclidean hull, computes the automorphism
    group of its complete vertex--facet incidence graph, enumerates that group
    up to ``max_group_order``, and retains exactly those permutations whose
    projective null rays satisfy the conformal Gram/ray-scaling equations to
    ``isometry_tolerance``.  With full-rank lifts these equations are
    equivalent to a future-preserving Lorentz transformation.

    The return value contains only JSON-native values.  This is a numerical
    certificate at the requested tolerances, not an interval proof.
    """
    tolerances, max_group_order = _validate_audit_controls(
        unit_tolerance,
        distinct_tolerance,
        support_tolerance,
        isometry_tolerance,
        max_group_order,
    )
    result = _empty_hyperbolic_audit(max_group_order, tolerances)

    try:
        points = np.asarray(vertices, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        return _fail_audit(result, f"vertices cannot be converted to floats: {exc}")
    if points.ndim != 2:
        return _fail_audit(result, "vertices must be a two-dimensional N x d array")
    n_vertices, dimension = points.shape
    result["n_vertices"] = int(n_vertices)
    result["dimension"] = int(dimension)
    result["checks"]["shape"] = bool(dimension >= 2 and n_vertices >= dimension + 1)
    if not result["checks"]["shape"]:
        return _fail_audit(result, "vertices must have d >= 2 and N >= d + 1")

    finite = bool(np.all(np.isfinite(points)))
    result["checks"]["finite"] = finite
    if not finite:
        return _fail_audit(result, "vertices must contain only finite coordinates")

    coordinate_scales = np.max(np.abs(points), axis=1)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled_points = np.divide(
            points,
            coordinate_scales[:, None],
            out=np.zeros_like(points),
            where=coordinate_scales[:, None] != 0.0,
        )
        norms = coordinate_scales * np.linalg.norm(scaled_points, axis=1)
    if not np.all(np.isfinite(norms)):
        return _fail_audit(result, "vertex norms overflowed or are not finite")
    max_unit_error = float(np.max(np.abs(norms - 1.0)))
    result["topology"]["max_unit_norm_error"] = max_unit_error
    result["checks"]["unit"] = bool(max_unit_error <= tolerances["unit"])
    if not result["checks"]["unit"]:
        return _fail_audit(
            result,
            f"vertices are not unit points (maximum norm error {max_unit_error:.3e})",
        )
    # Work with exact unit representatives after recording and gating the
    # submitted norm error.  This keeps the null lifts genuinely null instead
    # of allowing the unit tolerance to dominate very small chordal products.
    points = points / norms[:, None]

    pairwise = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(pairwise, axis=2)
    upper = distances[np.triu_indices(n_vertices, 1)]
    min_distance = float(np.min(upper))
    result["topology"]["min_pairwise_distance"] = min_distance
    result["checks"]["distinct"] = bool(min_distance > tolerances["distinct"])
    if not result["checks"]["distinct"]:
        return _fail_audit(
            result,
            f"vertices are not pairwise distinct (minimum distance {min_distance:.3e})",
        )

    centered = points - np.mean(points, axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    rank_threshold = max(
        tolerances["support"],
        np.finfo(np.float64).eps * max(centered.shape) * float(singular_values[0]),
    )
    affine_rank = int(np.count_nonzero(singular_values > rank_threshold))
    result["topology"]["affine_rank"] = affine_rank
    result["topology"]["min_affine_singular_value"] = (
        float(singular_values[dimension - 1])
        if len(singular_values) >= dimension
        else None
    )
    result["checks"]["full_dimensional"] = bool(affine_rank == dimension)
    if not result["checks"]["full_dimensional"]:
        return _fail_audit(
            result,
            f"vertices are not full-dimensional (affine rank {affine_rank}, expected {dimension})",
        )

    lifts = np.column_stack((points, np.ones(n_vertices, dtype=np.float64)))
    lift_singular_values = np.linalg.svd(lifts, compute_uv=False)
    min_lift_singular = float(lift_singular_values[-1])
    if min_lift_singular <= 0.0 or not np.isfinite(min_lift_singular):
        return _fail_audit(result, "null-lift span is numerically singular")
    lift_condition = float(lift_singular_values[0] / min_lift_singular)
    lift_condition_error = float(64.0 * np.finfo(np.float64).eps * lift_condition)
    if not np.isfinite(lift_condition_error):
        return _fail_audit(result, "null-lift condition estimate overflowed")
    result["topology"]["min_lift_singular_value"] = min_lift_singular
    result["topology"]["lift_condition_number"] = lift_condition
    result["topology"]["lift_condition_error_bound"] = lift_condition_error

    try:
        hull = ConvexHull(points)
    except (QhullError, ValueError) as exc:
        return _fail_audit(result, f"convex hull failed: {exc}")
    hull_vertices = {int(vertex) for vertex in hull.vertices}
    result["checks"]["all_vertices_extreme"] = bool(
        hull_vertices == set(range(n_vertices))
    )
    if not result["checks"]["all_vertices_extreme"]:
        return _fail_audit(result, "not every submitted point is a hull vertex")

    facets, support_diagnostics, support_failure = _strict_simplicial_facets(
        points, hull, tolerances["support"]
    )
    if support_diagnostics is not None:
        result["topology"].update(support_diagnostics)
    if support_failure is not None:
        return _fail_audit(result, support_failure)
    result["checks"]["simplicial"] = True
    result["checks"]["strict_support"] = True
    result["topology"]["n_facets"] = int(len(facets))
    result["valid_geometry"] = True

    if not PYNAUTY_AVAILABLE:
        return _fail_audit(result, "pynauty is required for symmetry certification")

    try:
        generators, group_order, combinatorial_orbits = _incidence_automorphisms(
            n_vertices, facets
        )
    except Exception as exc:
        return _fail_audit(result, f"pynauty automorphism computation failed: {exc}")
    result["combinatorial_group_order"] = int(group_order)
    result["combinatorial_generators"] = [list(generator) for generator in generators]
    result["combinatorial_vertex_orbits"] = combinatorial_orbits

    if group_order > max_group_order:
        return _fail_audit(
            result,
            f"combinatorial group order {group_order} exceeds "
            f"max_group_order {max_group_order}",
        )
    try:
        candidates = _enumerate_permutation_group(
            generators, n_vertices, max_group_order
        )
    except OverflowError as exc:
        return _fail_audit(result, str(exc))
    if len(candidates) != group_order:
        return _fail_audit(
            result,
            "enumerated vertex action does not match pynauty group order "
            f"({len(candidates)} != {group_order})",
        )

    accepted = {}
    rejected_scores = []
    for permutation in sorted(candidates):
        fit = _lorentz_fit_for_permutation(points, permutation)
        result["candidate_permutations_tested"] += 1
        if fit is None:
            result["indeterminate_permutations"] += 1
            continue
        if fit["isometry_residual"] <= tolerances["isometry"]:
            accepted[permutation] = fit
        else:
            rejected_scores.append(fit["isometry_residual"])

    identity = tuple(range(n_vertices))
    if identity not in accepted:
        return _fail_audit(result, "identity permutation failed the Lorentz fit")
    accepted_fits = list(accepted.values())
    result["geometric_candidate_count"] = int(len(accepted_fits))
    result["residuals"] = {
        "max_accepted_isometry_residual": float(
            max(fit["isometry_residual"] for fit in accepted_fits)
        ),
        "max_accepted_scaling_residual": float(
            max(fit["scaling_residual"] for fit in accepted_fits)
        ),
        "max_accepted_mapping_residual": _max_optional(
            fit["mapping_residual"] for fit in accepted_fits
        ),
        "max_accepted_lorentz_residual": _max_optional(
            fit["lorentz_residual"] for fit in accepted_fits
        ),
        "min_rejected_isometry_residual": float(min(rejected_scores))
        if rejected_scores and np.isfinite(min(rejected_scores))
        else None,
        "rejection_gap_threshold": float(10.0 * tolerances["isometry"]),
    }

    # Threshold classification does not itself guarantee closure: for an
    # almost-symmetric configuration it is possible for some, but not all,
    # elements of a combinatorial subgroup to land just below the requested
    # tolerance.  In that case there is useful candidate-level information,
    # but no geometric *group* order to report.
    accepted_elements = set(accepted)
    try:
        geometric_generators = _generators_from_elements(accepted_elements, n_vertices)
    except (ValueError, OverflowError) as exc:
        return _fail_audit(
            result,
            "accepted geometric candidates do not form a subgroup at the "
            f"requested tolerance: {exc}",
        )

    result["geometric_group_order"] = int(len(accepted_elements))
    result["geometric_generators"] = [
        list(generator) for generator in geometric_generators
    ]
    result["vertex_orbits"] = _orbits_from_group(accepted_elements, n_vertices)
    try:
        facet_signs = _facet_orientation_signs(facets, dimension)
        result["orientation_preserving_count"] = int(
            sum(
                _boundary_orientation_sign(permutation, facet_signs) > 0
                for permutation in accepted_elements
            )
        )
    except ValueError as exc:
        return _fail_audit(result, f"facet orientation audit failed: {exc}")
    min_rejected = result["residuals"]["min_rejected_isometry_residual"]
    rejection_gap = result["residuals"]["rejection_gap_threshold"]
    if lift_condition_error > tolerances["isometry"]:
        result["failure_reason"] = (
            "null-lift span is too ill-conditioned for the requested "
            "isometry tolerance: estimated error bound "
            f"{lift_condition_error:.3e} exceeds {tolerances['isometry']:.3e}"
        )
    elif result["indeterminate_permutations"]:
        result["failure_reason"] = (
            f"{result['indeterminate_permutations']} candidate permutations had "
            "numerically indeterminate Lorentz fits"
        )
    elif min_rejected is not None and min_rejected <= rejection_gap:
        result["failure_reason"] = (
            "geometric-group classification is numerically ambiguous: "
            f"minimum rejected residual {min_rejected:.3e} does not exceed "
            f"the rejection-gap threshold {rejection_gap:.3e}"
        )
    else:
        result["full_group_certified"] = True
        result["certified_nontrivial_symmetry"] = bool(len(accepted_elements) > 1)
        result["failure_reason"] = None
    return result


def compute_symmetry_group(vertices_3d):
    """
    Compute the automorphism group of a polyhedron's 1-skeleton.

    Args:
        vertices_3d: N x 3 array of vertex coordinates (e.g., from lift_to_sphere_with_inf)

    Returns:
        dict with keys:
        - 'group_size': Total size of automorphism group
        - 'num_generators': Number of group generators
        - 'generators': List of permutations generating the group
        - 'orbits': Orbit partition of vertices
        - 'num_orbits': Number of vertex orbits
        - 'group_name': Human-readable group identification (if recognized)
        - 'available': Whether pynauty is available

    Raises:
        ImportError: If pynauty is not installed
    """
    if not PYNAUTY_AVAILABLE:
        return {
            "available": False,
            "error": "pynauty not installed. Install with: pip install pynauty",
        }

    # Get convex hull to extract 1-skeleton (edge graph)
    hull = ConvexHull(vertices_3d)

    # Build edge list from hull faces (simplices)
    edges = set()
    for simplex in hull.simplices:
        # Each face is a triangle with 3 edges
        v0, v1, v2 = simplex
        edges.add(tuple(sorted([v0, v1])))
        edges.add(tuple(sorted([v1, v2])))
        edges.add(tuple(sorted([v2, v0])))

    # Build adjacency list for pynauty
    n_vertices = len(vertices_3d)
    adjacency = {i: [] for i in range(n_vertices)}
    for v1, v2 in edges:
        adjacency[v1].append(v2)
        adjacency[v2].append(v1)

    # Create pynauty graph
    g = pynauty.Graph(
        number_of_vertices=n_vertices, directed=False, adjacency_dict=adjacency
    )

    # Compute automorphism group
    # Returns: (generators, grpsize1, grpsize2, orbits, numorbits)
    generators, grpsize1, grpsize2, orbits, numorbits = pynauty.autgrp(g)

    # Calculate total group size
    # nauty represents size as grpsize1 × 10^grpsize2
    if isinstance(grpsize2, (int, float)):
        group_size = int(grpsize1 * (10**grpsize2))
    else:
        group_size = int(grpsize1)

    # Identify common symmetry groups
    group_name = identify_group(group_size, n_vertices)

    return {
        "available": True,
        "group_size": group_size,
        "num_generators": len(generators),
        "generators": generators,
        "orbits": orbits,
        "num_orbits": numorbits,
        "group_name": group_name,
        "vertex_degrees": [len(adjacency[i]) for i in range(n_vertices)],
        "num_edges": len(edges),
        "num_faces": len(hull.simplices),
    }


def identify_group(size, n_vertices=None):
    """
    Identify common symmetry groups by order.

    Args:
        size: Order of the group
        n_vertices: Number of vertices (helps with identification)

    Returns:
        Human-readable group name
    """
    common_groups = {
        1: "C₁ (trivial, no symmetry)",
        2: "C₂ (single reflection/inversion)",
        3: "C₃ (3-fold rotation)",
        4: "V₄ (Klein four-group) or C₄ (4-fold rotation)",
        6: "S₃ ≅ D₃ (triangle symmetries)",
        8: "D₄ (square symmetries) or other order-8 group",
        12: "A₄ (tetrahedral rotations) or D₆ (hexagon symmetries)",
        24: "S₄ (octahedral symmetries) or other order-24 group",
        48: "Oh (full octahedral symmetries)",
        60: "A₅ (icosahedral rotations)",
        120: "Ih (full icosahedral symmetries) or S₅",
    }

    if size in common_groups:
        return common_groups[size]
    else:
        # Check for dihedral groups D_n (order 2n)
        if size % 2 == 0:
            n = size // 2
            return f"possibly D_{n} (dihedral) or other order-{size} group"
        # Check for cyclic groups C_n
        return f"possibly C_{size} (cyclic) or other order-{size} group"


def format_symmetry_report(sym_info):
    """
    Format symmetry information as a readable string.

    Args:
        sym_info: Dictionary returned by compute_symmetry_group

    Returns:
        Formatted string report
    """
    if not sym_info.get("available", False):
        return f"❌ Symmetry computation unavailable: {sym_info.get('error', 'Unknown error')}"

    report = []
    report.append("## Symmetry Group Analysis\n")
    report.append(f"**Group Order:** {sym_info['group_size']}")
    report.append(f"**Identification:** {sym_info['group_name']}")
    report.append(f"**Generators:** {sym_info['num_generators']}")
    report.append(f"**Vertex Orbits:** {sym_info['num_orbits']}")
    report.append("")
    report.append("**Graph Properties:**")
    report.append(f"- Vertices: {len(sym_info['vertex_degrees'])}")
    report.append(f"- Edges: {sym_info['num_edges']}")
    report.append(f"- Faces: {sym_info['num_faces']}")
    report.append(f"- Vertex degrees: {sym_info['vertex_degrees']}")

    if sym_info["num_generators"] > 0 and sym_info["num_generators"] <= 5:
        report.append("")
        report.append("**Generators (as vertex permutations):**")
        for i, gen in enumerate(sym_info["generators"], 1):
            report.append(f"- σ_{i}: {gen}")

    return "\n".join(report)
