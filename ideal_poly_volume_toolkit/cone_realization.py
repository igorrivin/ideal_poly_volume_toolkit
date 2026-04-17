"""
Inscribed realization of simplicial 4-polytopes via the cone/Volkov method.

Given the boundary simplicial complex of a 4-polytope (a triangulation of S³),
find edge lengths such that the cone from the origin has flat dihedral angles
(sum to 2π at every codimension-2 face).  Since cone edges are clamped to
length 1, this automatically produces an inscribed realization on S³.

This is the 4-dimensional analogue of Rivin's angle-based approach in 3D,
via Volkov's cone construction.
"""

import numpy as np
import torch
from scipy.spatial import ConvexHull
from itertools import combinations
from collections import defaultdict, Counter


# ============================================================================
# Dihedral angle computation for cone 4-simplices
# ============================================================================

def gram_from_edge_lengths(l_sq, idx_map, facet):
    """
    Build 4×4 Gram matrix for boundary vertices of a cone 4-simplex.

    Since all vertices are on S³ (cone edges = 1):
        G_{ij} = <v_i, v_j> = 1 - l²_{ij} / 2

    Parameters
    ----------
    l_sq : dict or tensor-backed mapping from edge -> squared length
    idx_map : dict mapping (i,j) -> index into parameter vector
    facet : tuple of 4 vertex indices (a, b, c, d)

    Returns
    -------
    G : (4, 4) tensor, Gram matrix
    """
    G = torch.ones(4, 4, dtype=torch.float64)
    for ii in range(4):
        for jj in range(ii + 1, 4):
            u, v = facet[ii], facet[jj]
            key = (min(u, v), max(u, v))
            lsq = l_sq[idx_map[key]]
            g = 1.0 - lsq / 2.0
            G[ii, jj] = g
            G[jj, ii] = g
    return G


def dihedral_angle_at_triangle(coords, tri_local, v3_local, v4_local):
    """
    Dihedral angle at a triangle in a 4-simplex, given local coordinates.

    The triangle is defined by 3 vertex indices (into coords).
    v3_local and v4_local are the two vertices NOT in the triangle but
    in the two 3-faces sharing it.

    For a cone 4-simplex {O, a, b, c, d} with O=origin:
    - "Cone triangle" {O, a, b}: tri vertices are O,a,b; the two
      opposite vertices are c,d.  tri_local gives indices of a,b
      in coords (O is implicitly at origin).
    - "Boundary triangle" {a, b, c}: tri vertices are a,b,c;
      the two opposite vertices are O and d.

    Parameters
    ----------
    coords : (4, 4) tensor — coordinates of a, b, c, d (O is origin)
    tri_local : tuple of local indices (0-3) for the 3 triangle vertices
                (for cone triangles, these are boundary vertex indices;
                 for boundary triangles, also boundary vertex indices)
    v3_local, v4_local : local indices of the two opposite vertices
                         (for boundary triangle, one of these is "O" = -1)

    Returns
    -------
    angle : scalar tensor
    """
    # For boundary triangles, one opposite vertex is O (the origin)
    # We handle this by passing coords_with_O where O = [0,0,0,0]

    # Get the triangle vertices
    t0, t1, t2 = tri_local

    # Basis for the affine plane of the triangle
    # Shift to t0 as origin
    e1 = coords[t1] - coords[t0]
    e2 = coords[t2] - coords[t0]

    # Project v3 and v4 onto orthogonal complement of span{e1, e2}
    # relative to t0
    p3 = coords[v3_local] - coords[t0]
    p4 = coords[v4_local] - coords[t0]

    # Gram matrix of the basis
    G = torch.stack([
        torch.stack([e1 @ e1, e1 @ e2]),
        torch.stack([e2 @ e1, e2 @ e2])
    ])

    # Solve for projection coefficients
    b3 = torch.stack([p3 @ e1, p3 @ e2])
    b4 = torch.stack([p4 @ e1, p4 @ e2])

    alpha3 = torch.linalg.solve(G, b3)
    alpha4 = torch.linalg.solve(G, b4)

    # Normal components
    n3 = p3 - alpha3[0] * e1 - alpha3[1] * e2
    n4 = p4 - alpha4[0] * e1 - alpha4[1] * e2

    # Dihedral angle
    cos_theta = (n3 @ n4) / (torch.norm(n3) * torch.norm(n4) + 1e-15)
    cos_theta = cos_theta.clamp(-1.0 + 1e-10, 1.0 - 1e-10)
    return torch.acos(cos_theta)


def compute_cone_simplex_angles(G):
    """
    Compute all dihedral angles of a cone 4-simplex from the Gram matrix
    of its 4 boundary vertices.

    The cone 4-simplex has vertices {O, v0, v1, v2, v3} where O is the
    origin and v0..v3 are on S³.

    Parameters
    ----------
    G : (4, 4) tensor — Gram matrix of v0, v1, v2, v3

    Returns
    -------
    boundary_angles : dict mapping (i,j,k) -> angle at boundary triangle
                      {vi, vj, vk} (local indices 0-3)
    cone_angles : dict mapping (i,j) -> angle at cone triangle {O, vi, vj}
                  (local indices 0-3)
    """
    # Reconstruct coordinates via Cholesky
    # Add small regularization for numerical stability
    G_reg = G + 1e-10 * torch.eye(4, dtype=G.dtype)
    try:
        L = torch.linalg.cholesky(G_reg)
    except Exception:
        # Not positive definite — return large penalty angles
        nan = torch.tensor(float('nan'), dtype=G.dtype)
        boundary_angles = {}
        cone_angles = {}
        for tri in combinations(range(4), 3):
            boundary_angles[tri] = nan
        for edge in combinations(range(4), 2):
            cone_angles[edge] = nan
        return boundary_angles, cone_angles

    coords = L  # rows are coordinates of v0, v1, v2, v3
    origin = torch.zeros(4, dtype=G.dtype)

    # Coordinates including origin as index -1
    # We'll use a 5-vertex array: [O, v0, v1, v2, v3]
    all_coords = torch.vstack([origin.unsqueeze(0), coords])
    # Index mapping: O=0, v0=1, v1=2, v2=3, v3=4

    boundary_angles = {}
    cone_angles = {}

    # Boundary triangles: {vi, vj, vk}, opposite vertices are O and vl
    for tri in combinations(range(4), 3):
        i, j, k = tri
        l = ({0, 1, 2, 3} - set(tri)).pop()
        # In all_coords: tri vertices are i+1, j+1, k+1
        # Opposite: O=0 and l+1
        e1 = all_coords[j+1] - all_coords[i+1]
        e2 = all_coords[k+1] - all_coords[i+1]
        p_O = all_coords[0] - all_coords[i+1]  # O - vi
        p_l = all_coords[l+1] - all_coords[i+1]  # vl - vi

        Gram = torch.stack([
            torch.stack([e1 @ e1, e1 @ e2]),
            torch.stack([e2 @ e1, e2 @ e2])
        ])

        b_O = torch.stack([p_O @ e1, p_O @ e2])
        b_l = torch.stack([p_l @ e1, p_l @ e2])

        alpha_O = torch.linalg.solve(Gram, b_O)
        alpha_l = torch.linalg.solve(Gram, b_l)

        n_O = p_O - alpha_O[0] * e1 - alpha_O[1] * e2
        n_l = p_l - alpha_l[0] * e1 - alpha_l[1] * e2

        cos_th = (n_O @ n_l) / (torch.norm(n_O) * torch.norm(n_l) + 1e-15)
        cos_th = cos_th.clamp(-1.0 + 1e-10, 1.0 - 1e-10)
        boundary_angles[tri] = torch.acos(cos_th)

    # Cone triangles: {O, vi, vj}, opposite vertices are vk and vl
    for edge in combinations(range(4), 2):
        i, j = edge
        others = sorted({0, 1, 2, 3} - set(edge))
        k, l = others

        # Triangle is {O, vi, vj} = {0, i+1, j+1} in all_coords
        e1 = all_coords[i+1] - all_coords[0]  # vi - O = vi
        e2 = all_coords[j+1] - all_coords[0]  # vj - O = vj
        p_k = all_coords[k+1] - all_coords[0]  # vk
        p_l = all_coords[l+1] - all_coords[0]  # vl

        Gram = torch.stack([
            torch.stack([e1 @ e1, e1 @ e2]),
            torch.stack([e2 @ e1, e2 @ e2])
        ])

        b_k = torch.stack([p_k @ e1, p_k @ e2])
        b_l = torch.stack([p_l @ e1, p_l @ e2])

        alpha_k = torch.linalg.solve(Gram, b_k)
        alpha_l = torch.linalg.solve(Gram, b_l)

        n_k = p_k - alpha_k[0] * e1 - alpha_k[1] * e2
        n_l = p_l - alpha_l[0] * e1 - alpha_l[1] * e2

        cos_th = (n_k @ n_l) / (torch.norm(n_k) * torch.norm(n_l) + 1e-15)
        cos_th = cos_th.clamp(-1.0 + 1e-10, 1.0 - 1e-10)
        cone_angles[edge] = torch.acos(cos_th)

    return boundary_angles, cone_angles


# ============================================================================
# Angle deficit loss
# ============================================================================

def build_incidence(n_vertices, edges, facets, triangles):
    """
    Build the incidence structure for the cone complex.

    Returns
    -------
    edge_to_idx : dict mapping (i,j) -> index in parameter vector
    tri_to_facets : dict mapping triangle -> list of (facet_idx, local_tri)
    cone_tri_to_facets : dict mapping edge -> list of (facet_idx, local_edge)
    facet_list : list of sorted 4-tuples
    """
    edge_list = sorted(set(tuple(sorted(e)) for e in edges))
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}

    facet_list = [tuple(sorted(f)) for f in facets]

    # For each boundary triangle, which facets contain it?
    tri_to_facets = defaultdict(list)
    for fi, f in enumerate(facet_list):
        for tri in combinations(f, 3):
            # Local indices within the facet
            local_tri = tuple(f.index(v) for v in tri)
            tri_to_facets[tri].append((fi, local_tri))

    # For each boundary edge, which facets contain it?
    # (These define the cone triangles {O, i, j})
    edge_to_facets = defaultdict(list)
    for fi, f in enumerate(facet_list):
        for edge in combinations(f, 2):
            local_edge = tuple(f.index(v) for v in edge)
            edge_to_facets[edge].append((fi, local_edge))

    return edge_to_idx, tri_to_facets, edge_to_facets, facet_list, edge_list


def _dihedral_angle_batch(base, e1, e2, p1, p2):
    """
    Batched dihedral angle at a triangle with basis vectors e1, e2.
    p1, p2 are the two opposite vertices (shifted by base).

    All inputs: (B, 4) tensors.
    Returns: (B,) tensor of angles.
    """
    # Project p1, p2 onto orthogonal complement of span{e1, e2}
    # Gram matrix of basis: (B, 2, 2)
    g11 = (e1 * e1).sum(-1)
    g12 = (e1 * e2).sum(-1)
    g22 = (e2 * e2).sum(-1)
    det = g11 * g22 - g12 * g12 + 1e-15

    # Projection coefficients for p1
    b1_1 = (p1 * e1).sum(-1)
    b1_2 = (p1 * e2).sum(-1)
    a1_1 = (g22 * b1_1 - g12 * b1_2) / det
    a1_2 = (g11 * b1_2 - g12 * b1_1) / det

    # Projection coefficients for p2
    b2_1 = (p2 * e1).sum(-1)
    b2_2 = (p2 * e2).sum(-1)
    a2_1 = (g22 * b2_1 - g12 * b2_2) / det
    a2_2 = (g11 * b2_2 - g12 * b2_1) / det

    # Normal components
    n1 = p1 - a1_1.unsqueeze(-1) * e1 - a1_2.unsqueeze(-1) * e2
    n2 = p2 - a2_1.unsqueeze(-1) * e1 - a2_2.unsqueeze(-1) * e2

    cos_th = (n1 * n2).sum(-1) / (
        torch.norm(n1, dim=-1) * torch.norm(n2, dim=-1) + 1e-15)
    cos_th = cos_th.clamp(-1.0 + 1e-10, 1.0 - 1e-10)
    return torch.acos(cos_th)


def precompute_angle_gather(facet_list, tri_to_facets, edge_to_facets):
    """
    Precompute index arrays for gathering angles into deficit sums.

    Returns structures that map each constraint (triangle or edge) to
    the specific (facet, local_face) entries whose angles must sum to 2π.
    """
    # For boundary triangles: 4 choose 3 = 4 local triangles per facet
    # Local triangle (i,j,k) in facet [a,b,c,d]: opposite vertex is the remaining one
    # Boundary angle at local tri (i,j,k): angle between face {O,vi,vj,vk} and {vi,vj,vk,vl}

    # For cone triangles: 4 choose 2 = 6 local edges per facet
    # Cone angle at local edge (i,j): angle between face {O,vi,vj,vk} and {O,vi,vj,vl}

    # We'll precompute which (facet_idx, local_type_idx) entries contribute
    # to each constraint, and build scatter indices.

    # Boundary triangle constraints
    btri_list = sorted(tri_to_facets.keys())
    btri_angles = []  # list of (facet_idx, local_tri_index_in_C43)
    btri_ptr = [0]  # CSR-like pointer
    # C(4,3) ordering
    c43 = list(combinations(range(4), 3))
    c43_map = {t: i for i, t in enumerate(c43)}
    for tri in btri_list:
        for fi, local_tri in tri_to_facets[tri]:
            btri_angles.append((fi, c43_map[local_tri]))
        btri_ptr.append(len(btri_angles))

    # Cone triangle (edge) constraints
    cedge_list = sorted(edge_to_facets.keys())
    cedge_angles = []
    cedge_ptr = [0]
    c42 = list(combinations(range(4), 2))
    c42_map = {e: i for i, e in enumerate(c42)}
    for edge in cedge_list:
        for fi, local_edge in edge_to_facets[edge]:
            cedge_angles.append((fi, c42_map[local_edge]))
        cedge_ptr.append(len(cedge_angles))

    return (btri_angles, btri_ptr, cedge_angles, cedge_ptr)


_cone_gather_cache = {}

def cone_loss(l_sq_params, edge_to_idx, facet_list,
              tri_to_facets, edge_to_facets):
    """
    Compute the angle deficit loss (vectorized).

    loss = Σ (angle_sum - 2π)² over all codimension-2 faces + PD barrier
    """
    TWO_PI = 2.0 * np.pi
    F = len(facet_list)

    # Build Gram matrices for all facets: (F, 4, 4)
    G_batch = torch.ones(F, 4, 4, dtype=torch.float64)
    for fi, facet in enumerate(facet_list):
        for ii in range(4):
            for jj in range(ii + 1, 4):
                key = (min(facet[ii], facet[jj]), max(facet[ii], facet[jj]))
                g = 1.0 - l_sq_params[edge_to_idx[key]] / 2.0
                G_batch[fi, ii, jj] = g
                G_batch[fi, jj, ii] = g

    # PD barrier via minimum eigenvalue
    eigvals = torch.linalg.eigvalsh(G_batch)  # (F, 4)
    min_eigs = eigvals[:, 0]
    pd_penalty = 100.0 * torch.nn.functional.softplus(0.01 - min_eigs).sum()

    # Cholesky to get coordinates: (F, 4, 4)
    G_reg = G_batch + 1e-8 * torch.eye(4, dtype=torch.float64).unsqueeze(0)
    try:
        L = torch.linalg.cholesky(G_reg)
    except Exception:
        # Return a loss connected to l_sq_params so gradients flow
        return pd_penalty + 1e6 * l_sq_params.sum() * 0 + 1e6

    coords = L  # (F, 4, 4): coords[fi, i, :] = position of vertex i in facet fi
    origin = torch.zeros(F, 1, 4, dtype=torch.float64)

    # All 5 vertices per simplex: [O, v0, v1, v2, v3]
    all5 = torch.cat([origin, coords], dim=1)  # (F, 5, 4)

    # Compute boundary triangle angles: C(4,3)=4 per facet
    # For triangle (i,j,k), opposite vertices are O (idx 0) and l (idx l+1)
    c43 = [(0,1,2,3), (0,1,3,2), (0,2,3,1), (1,2,3,0)]
    # Each entry: (i,j,k, opposite_l)
    boundary_angles = torch.zeros(F, 4, dtype=torch.float64)
    for tidx, (i, j, k, l) in enumerate(c43):
        base = all5[:, i+1, :]  # (F, 4)
        e1 = all5[:, j+1, :] - base
        e2 = all5[:, k+1, :] - base
        p_O = all5[:, 0, :] - base   # O - base
        p_l = all5[:, l+1, :] - base  # vl - base
        boundary_angles[:, tidx] = _dihedral_angle_batch(base, e1, e2, p_O, p_l)

    # Compute cone triangle angles: C(4,2)=6 per facet
    # For edge (i,j), opposite vertices are vk and vl
    c42 = [(0,1,2,3), (0,2,1,3), (0,3,1,2), (1,2,0,3), (1,3,0,2), (2,3,0,1)]
    cone_angles = torch.zeros(F, 6, dtype=torch.float64)
    for eidx, (i, j, k, l) in enumerate(c42):
        base = all5[:, 0, :]  # O = origin
        e1 = all5[:, i+1, :] - base  # = vi (since base = O)
        e2 = all5[:, j+1, :] - base  # = vj
        p_k = all5[:, k+1, :] - base  # = vk
        p_l = all5[:, l+1, :] - base  # = vl
        cone_angles[:, eidx] = _dihedral_angle_batch(base, e1, e2, p_k, p_l)

    # Gather and sum angles per constraint
    # Use precomputed gather structure
    global _cone_gather_cache
    cache_key = id(facet_list)
    if cache_key not in _cone_gather_cache:
        _cone_gather_cache[cache_key] = precompute_angle_gather(
            facet_list, tri_to_facets, edge_to_facets)
    btri_angles_idx, btri_ptr, cedge_angles_idx, cedge_ptr = _cone_gather_cache[cache_key]

    # Boundary triangle deficits — gather via indexing
    btri_deficits = []
    for ci in range(len(btri_ptr) - 1):
        angles = []
        for ai in range(btri_ptr[ci], btri_ptr[ci + 1]):
            fi, tidx = btri_angles_idx[ai]
            angles.append(boundary_angles[fi, tidx])
        angle_sum = torch.stack(angles).sum() if angles else boundary_angles[0, 0] * 0
        btri_deficits.append((angle_sum - TWO_PI) ** 2)
    boundary_loss = torch.stack(btri_deficits).sum() if btri_deficits \
        else boundary_angles.sum() * 0

    # Cone triangle deficits
    cedge_deficits = []
    for ci in range(len(cedge_ptr) - 1):
        angles = []
        for ai in range(cedge_ptr[ci], cedge_ptr[ci + 1]):
            fi, eidx = cedge_angles_idx[ai]
            angles.append(cone_angles[fi, eidx])
        angle_sum = torch.stack(angles).sum() if angles else cone_angles[0, 0] * 0
        cedge_deficits.append((angle_sum - TWO_PI) ** 2)
    cone_loss_val = torch.stack(cedge_deficits).sum() if cedge_deficits \
        else cone_angles.sum() * 0

    return boundary_loss + cone_loss_val + pd_penalty


# ============================================================================
# Coordinate recovery
# ============================================================================

def recover_coordinates(l_sq_params, edge_to_idx, n_vertices):
    """
    Recover vertex coordinates on S³ from edge lengths.

    Build the full N×N Gram matrix G_{ij} = 1 - l²_{ij}/2 for known edges,
    estimate unknown entries, then eigendecompose.

    Parameters
    ----------
    l_sq_params : 1D tensor or array of squared edge lengths
    edge_to_idx : dict mapping (i,j) -> index
    n_vertices : int

    Returns
    -------
    coords : (N, 4) numpy array on S³
    """
    N = n_vertices
    l_sq = l_sq_params.detach().numpy() if torch.is_tensor(l_sq_params) \
        else l_sq_params

    # Build partial Gram matrix
    G = np.eye(N)  # diagonal = 1 (unit sphere)
    known = np.zeros((N, N), dtype=bool)
    np.fill_diagonal(known, True)

    for (i, j), idx in edge_to_idx.items():
        g = 1.0 - l_sq[idx] / 2.0
        G[i, j] = g
        G[j, i] = g
        known[i, j] = True
        known[j, i] = True

    # For unknown entries, use shortest-path-based estimation
    # Simple approach: iterative averaging of unknown entries
    for _ in range(50):
        G_new = G.copy()
        for i in range(N):
            for j in range(i + 1, N):
                if not known[i, j]:
                    # Average over common neighbors
                    common = np.where(known[i] & known[j])[0]
                    if len(common) > 0:
                        # Estimate <vi, vj> from common neighbors
                        # This is approximate
                        est = np.mean([G[i, k] * G[j, k] for k in common])
                        G_new[i, j] = est
                        G_new[j, i] = est
        G = G_new

    # Symmetrize and project to nearest PSD matrix
    G = (G + G.T) / 2
    eigvals, eigvecs = np.linalg.eigh(G)

    # Take top 4 eigenvalues/vectors
    idx = np.argsort(eigvals)[::-1][:4]
    coords = eigvecs[:, idx] * np.sqrt(np.maximum(eigvals[idx], 0))

    # Normalize to S³
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    coords = coords / norms

    return coords


# ============================================================================
# Main solver
# ============================================================================

def realize_cone(n_vertices, edges, facets, triangles=None,
                 n_restarts=20, n_steps=500, lr=0.01, verbose=True):
    """
    Find an inscribed realization via the cone/Volkov method.

    Parameters
    ----------
    n_vertices : int
    edges : list of (i, j) pairs
    facets : list of 4-tuples
    triangles : list of 3-tuples (computed from facets if None)
    n_restarts : int
    n_steps : int
    lr : float
    verbose : bool

    Returns
    -------
    result : dict with 'success', 'points', 'details', 'n_attempts',
             'final_loss', 'max_deficit'
    """
    edges_sorted = sorted(set(tuple(sorted(e)) for e in edges))
    facets_sorted = [tuple(sorted(f)) for f in facets]

    if triangles is None:
        tri_set = set()
        for f in facets_sorted:
            for t in combinations(f, 3):
                tri_set.add(t)
        triangles = sorted(tri_set)

    edge_to_idx, tri_to_facets, edge_to_facets, facet_list, edge_list = \
        build_incidence(n_vertices, edges_sorted, facets_sorted, triangles)

    n_edges = len(edge_list)

    if verbose:
        n_tri_constraints = len(tri_to_facets)
        n_edge_constraints = len(edge_to_facets)
        print(f"Cone realization: {n_vertices} vertices, {n_edges} edge params, "
              f"{n_tri_constraints} tri + {n_edge_constraints} edge constraints")

    best_pts = None
    best_loss = float('inf')

    for restart in range(n_restarts):
        # Initialize edge lengths
        # For points on S³, edge lengths satisfy 0 < l² < 4
        # Start near typical values for well-spread points
        if restart == 0:
            # Use typical spacing: l² ≈ 2 (orthogonal) with some noise
            init_lsq = 1.5 + 0.5 * torch.rand(n_edges, dtype=torch.float64)
        else:
            # Random initialization in valid range
            init_lsq = 0.5 + 2.5 * torch.rand(n_edges, dtype=torch.float64)

        l_sq = init_lsq.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [l_sq], lr=lr, max_iter=5, line_search_fn='strong_wolfe')

        best_step_loss = float('inf')
        for step in range(n_steps):
            def closure():
                optimizer.zero_grad()
                # Clamp to valid range
                with torch.no_grad():
                    l_sq.data.clamp_(0.01, 3.99)
                loss = cone_loss(l_sq, edge_to_idx, facet_list,
                                 tri_to_facets, edge_to_facets)
                loss.backward()
                if l_sq.grad is not None:
                    l_sq.grad = l_sq.grad.contiguous()
                return loss

            loss_val = optimizer.step(closure)

            if loss_val is not None:
                lv = float(loss_val.detach()) if torch.is_tensor(loss_val) \
                    else float(loss_val)
                if lv < best_step_loss:
                    best_step_loss = lv
                if lv < 1e-10:
                    break

        # Recover coordinates
        with torch.no_grad():
            l_sq.data.clamp_(0.01, 3.99)

        final_loss = float(cone_loss(
            l_sq, edge_to_idx, facet_list,
            tri_to_facets, edge_to_facets).item())

        coords = recover_coordinates(l_sq, edge_to_idx, n_vertices)

        # Verify
        match, details = verify_realization(coords, facets_sorted)

        if verbose:
            status = "MATCH" if match else \
                f"miss={details['n_missing']} extra={details['n_extra']}"
            print(f"  restart {restart:3d}: loss={final_loss:.6f}  "
                  f"f={details['f_vector']}  {status}")

        if match:
            if verbose:
                print(f"  SUCCESS at restart {restart}")
            return {
                'success': True,
                'points': coords,
                'details': details,
                'n_attempts': restart + 1,
                'final_loss': final_loss,
            }

        if final_loss < best_loss:
            best_loss = final_loss
            best_pts = coords

    if verbose:
        print(f"  FAILED after {n_restarts} restarts (best loss={best_loss:.6f})")

    _, best_details = verify_realization(best_pts, facets_sorted)
    return {
        'success': False,
        'points': best_pts,
        'details': best_details,
        'n_attempts': n_restarts,
        'final_loss': best_loss,
    }


def verify_realization(pts, target_facets):
    """Check convex hull matches target facets."""
    try:
        hull = ConvexHull(pts)
    except Exception as e:
        return False, {'error': str(e), 'f_vector': [0, 0, 0, 0],
                       'n_missing': len(target_facets), 'n_extra': 0}

    got = set(tuple(sorted(s)) for s in hull.simplices)
    target = set(tuple(sorted(f)) for f in target_facets)

    edges = set()
    tris = set()
    for f in got:
        for e in combinations(f, 2): edges.add(e)
        for t in combinations(f, 3): tris.add(t)

    fv = [len(hull.vertices), len(edges), len(tris), len(got)]
    norms = np.linalg.norm(pts, axis=1)

    return len(got - target) == 0 and len(target - got) == 0, {
        'f_vector': fv,
        'n_missing': len(target - got),
        'n_extra': len(got - target),
        'inscribed': bool(np.allclose(norms, 1.0, atol=1e-4)),
        'norm_range': (float(norms.min()), float(norms.max())),
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Inscribed realization via cone/Volkov method")
    parser.add_argument('challenge', type=str)
    parser.add_argument('--restarts', type=int, default=20)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    with open(args.challenge) as f:
        challenge = json.load(f)

    N = challenge['n_vertices']
    edges = [tuple(e) for e in challenge['edges']]
    facets = [tuple(f) for f in challenge['facets']]
    tris = [tuple(t) for t in challenge.get('triangles', [])] or None

    print(f"N={N}  f={challenge['f_vector']}")

    result = realize_cone(N, edges, facets, triangles=tris,
                          n_restarts=args.restarts, n_steps=args.steps,
                          lr=args.lr)

    if result['success']:
        print(f"\nSUCCESS in {result['n_attempts']} attempts  "
              f"loss={result['final_loss']:.2e}")
        print(f"  f-vector: {result['details']['f_vector']}")
        print(f"  inscribed: {result['details']['inscribed']}")
        out = {'points': result['points'].tolist()}
        print(json.dumps(out))
    else:
        print(f"\nFAILED after {result['n_attempts']} attempts")
        print(f"  Best: {result['details']}")


if __name__ == '__main__':
    main()
