"""
Inscribed realization of simplicial 4-polytopes on S^3.

Given the face lattice (edges, facets) of a simplicial 4-polytope,
find N points on the unit 3-sphere whose convex hull realizes the
given combinatorial type.

Algorithm (following GPT-5.4 Pro's approach):
  1. Spectral initialization: embed the 1-skeleton via graph Laplacian
     eigenvectors into R^4, project onto S^3.
  2. Differentiable optimization: minimize a smooth surrogate for the
     facet-support inequalities using torch LBFGS on S^3.
  3. Exact combinatorial verification via scipy ConvexHull.

References:
  - Firsching, "Realizability and inscribability for simplicial
    polytopes via optimization", Math. Program. 166 (2017), 273-295.
  - Forman, "Morse theory for cell complexes", Adv. Math. 134 (1998),
    90-145.  (For higher-order Laplacian initialization.)
"""

import numpy as np
import torch
from scipy.spatial import ConvexHull
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from itertools import combinations
from collections import Counter


# ============================================================================
# Spectral initialization
# ============================================================================

def graph_laplacian_embedding(n_vertices, edges, dim=4):
    """
    Embed vertices into R^dim using the first nontrivial eigenvectors
    of the graph Laplacian, then project onto S^{dim-1}.

    Parameters
    ----------
    n_vertices : int
    edges : list of (i, j) pairs
    dim : int, embedding dimension (default 4)

    Returns
    -------
    pts : (n_vertices, dim) array on S^{dim-1}
    """
    n = n_vertices
    # Build adjacency
    row, col, data = [], [], []
    deg = np.zeros(n)
    for u, v in edges:
        row.extend([u, v])
        col.extend([v, u])
        data.extend([1.0, 1.0])
        deg[u] += 1
        deg[v] += 1

    A = csr_matrix((data, (row, col)), shape=(n, n))
    D = csr_matrix((deg, (range(n), range(n))), shape=(n, n))
    L = D - A

    # First dim+1 smallest eigenvalues (skip the trivial zero eigenvector)
    k = min(dim + 1, n - 1)
    eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')

    # Skip the constant eigenvector (eigenvalue ≈ 0)
    idx = np.argsort(eigenvalues)
    nontrivial = eigenvectors[:, idx[1:dim+1]]

    # If we got fewer than dim nontrivial eigenvectors, pad with noise
    if nontrivial.shape[1] < dim:
        pad = np.random.randn(n, dim - nontrivial.shape[1]) * 0.01
        nontrivial = np.hstack([nontrivial, pad])

    # Normalize to S^{dim-1}
    norms = np.linalg.norm(nontrivial, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return nontrivial / norms


def forman_1laplacian_embedding(n_vertices, edges, triangles, dim=4):
    """
    Embed vertices into R^dim using the combinatorial 1-Laplacian
    (Forman Laplacian on edges), projected back to vertices.

    The 1-Laplacian L_1 = B_1^T B_1 + B_2 B_2^T acts on 1-chains
    (edges). Its small eigenvectors encode edge-triangle incidence
    structure, which is closer to the facet constraints than the
    graph Laplacian alone.

    We project edge eigenvectors back to vertices by averaging over
    incident edges.

    Parameters
    ----------
    n_vertices : int
    edges : list of (i, j) pairs
    triangles : list of (i, j, k) triples
    dim : int

    Returns
    -------
    pts : (n_vertices, dim) array on S^{dim-1}
    """
    n = n_vertices
    edge_list = [tuple(sorted(e)) for e in edges]
    edge_idx = {e: i for i, e in enumerate(edge_list)}
    m = len(edge_list)

    # Boundary operator B1: vertices -> edges (m x n)
    B1_row, B1_col, B1_data = [], [], []
    for i, (u, v) in enumerate(edge_list):
        B1_row.extend([i, i])
        B1_col.extend([u, v])
        B1_data.extend([-1.0, 1.0])
    B1 = csr_matrix((B1_data, (B1_row, B1_col)), shape=(m, n))

    # Boundary operator B2: edges -> triangles
    tri_list = [tuple(sorted(t)) for t in triangles]
    p = len(tri_list)
    B2_row, B2_col, B2_data = [], [], []
    for j, (a, b, c) in enumerate(tri_list):
        # Boundary of triangle [a,b,c] = [b,c] - [a,c] + [a,b]
        for edge, sign in [((a, b), 1.0), ((a, c), -1.0), ((b, c), 1.0)]:
            if edge in edge_idx:
                B2_row.append(edge_idx[edge])
                B2_col.append(j)
                B2_data.append(sign)
    B2 = csr_matrix((B2_data, (B2_row, B2_col)), shape=(m, p))

    # 1-Laplacian: L1 = B1^T B1 + B2 B2^T  (acts on edge space R^m)
    # But we want vertex embedding, so use the "down" Laplacian B1^T B1
    # which is just the graph Laplacian, and the "up" Laplacian B2 B2^T
    # which encodes triangle structure.
    # For initialization, use the up-Laplacian eigenvectors projected
    # to vertices.
    L1_up = B2 @ B2.T

    k = min(dim + 1, m - 1)
    try:
        eigenvalues, eigenvectors = eigsh(L1_up, k=k, which='SM')
    except Exception:
        # Fall back to graph Laplacian
        return graph_laplacian_embedding(n_vertices, edges, dim)

    idx = np.argsort(eigenvalues)
    # Take nontrivial eigenvectors
    start = 0
    while start < len(idx) and eigenvalues[idx[start]] < 1e-10:
        start += 1
    if start >= len(idx):
        start = max(0, len(idx) - dim)
    edge_vecs = eigenvectors[:, idx[start:start+dim]]

    if edge_vecs.shape[1] < dim:
        pad = np.random.randn(m, dim - edge_vecs.shape[1]) * 0.01
        edge_vecs = np.hstack([edge_vecs, pad])

    # Project edge vectors to vertices: average over incident edges
    vertex_vecs = np.zeros((n, dim))
    vertex_count = np.zeros(n)
    for i, (u, v) in enumerate(edge_list):
        vertex_vecs[u] += edge_vecs[i]
        vertex_vecs[v] += edge_vecs[i]
        vertex_count[u] += 1
        vertex_count[v] += 1
    vertex_count = np.maximum(vertex_count, 1)
    vertex_vecs /= vertex_count[:, None]

    norms = np.linalg.norm(vertex_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return vertex_vecs / norms


# ============================================================================
# Differentiable loss on S^3
# ============================================================================

def build_facet_tensors(n_vertices, facets):
    """
    Precompute facet index tensors and non-incident vertex masks.

    Returns
    -------
    facet_indices : (F, 4) long tensor — vertex indices per facet
    non_incident : list of 1D long tensors — non-incident vertices per facet
    """
    F = len(facets)
    facet_indices = torch.zeros(F, 4, dtype=torch.long)
    non_incident = []

    all_verts = set(range(n_vertices))
    for i, f in enumerate(facets):
        facet_indices[i] = torch.tensor(sorted(f), dtype=torch.long)
        non_inc = sorted(all_verts - set(f))
        non_incident.append(torch.tensor(non_inc, dtype=torch.long))

    return facet_indices, non_incident


def realization_loss(pts, facet_indices, non_incident, edges,
                     margin=0.01, det_weight=0.1, repulsion_weight=0.01,
                     edge_weight=0.05, centroid_weight=0.1):
    """
    Differentiable loss for inscribed realization.

    Parameters
    ----------
    pts : (N, 4) tensor on S^3
    facet_indices : (F, 4) long tensor
    non_incident : list of 1D long tensors
    edges : (E, 2) long tensor
    margin : float, target margin for facet inequalities
    det_weight : float, weight for determinant barrier
    repulsion_weight : float, weight for vertex repulsion
    edge_weight : float, weight for edge/nonedge bias
    centroid_weight : float, weight for centroid penalty

    Returns
    -------
    loss : scalar tensor
    """
    N = pts.shape[0]
    F = facet_indices.shape[0]

    total_loss = torch.tensor(0.0, dtype=pts.dtype)

    # === Main facet constraint loss ===
    # For each facet F = {a,b,c,d}: solve P_F @ n_F = 1, then
    # require <n_F, p_j> < 1 for j not in F.
    ones4 = torch.ones(4, dtype=pts.dtype)

    for i in range(F):
        fi = facet_indices[i]
        P_F = pts[fi]  # (4, 4)

        # Solve P_F @ n_F = 1 for the support normal
        try:
            n_F = torch.linalg.solve(P_F, ones4)
        except Exception:
            total_loss = total_loss + 100.0
            continue

        # Inner products of non-incident vertices with n_F
        ni = non_incident[i]
        dots = pts[ni] @ n_F  # should all be < 1

        # Soft violation: penalize dots > 1 - margin via softplus
        violations = torch.nn.functional.softplus(dots - (1.0 - margin))
        total_loss = total_loss + violations.sum()

    # === Determinant barrier ===
    # Keep facet matrices well-conditioned
    if det_weight > 0:
        for i in range(F):
            fi = facet_indices[i]
            P_F = pts[fi]
            det = torch.det(P_F)
            # Penalize small |det|
            total_loss = total_loss - det_weight * torch.log(det.abs() + 1e-12)

    # === Repulsion ===
    # Prevent vertex collapse
    if repulsion_weight > 0:
        gram = pts @ pts.T
        # Penalize pairs that are too close (dot product too close to 1)
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        close = torch.nn.functional.softplus(gram[mask] - 0.95)
        total_loss = total_loss + repulsion_weight * close.sum()

    # === Edge / nonedge bias ===
    # Adjacent vertices should have larger dot products than non-adjacent
    if edge_weight > 0 and edges is not None:
        gram = pts @ pts.T if repulsion_weight == 0 else gram
        edge_dots = gram[edges[:, 0], edges[:, 1]]
        # Encourage edge dots to be larger (weakly)
        total_loss = total_loss - edge_weight * edge_dots.mean()

    # === Centroid penalty ===
    if centroid_weight > 0:
        centroid = pts.mean(dim=0)
        total_loss = total_loss + centroid_weight * (centroid ** 2).sum()

    return total_loss


# ============================================================================
# Batched facet loss (more efficient for large F)
# ============================================================================

def realization_loss_batched(pts, facet_indices, non_incident_padded,
                             non_incident_mask, edges,
                             margin=0.01, det_weight=0.1,
                             repulsion_weight=0.01, edge_weight=0.05,
                             centroid_weight=0.1):
    """
    Batched version of realization_loss for better GPU/vectorization.

    Parameters
    ----------
    pts : (N, 4) tensor
    facet_indices : (F, 4) long tensor
    non_incident_padded : (F, max_non_inc) long tensor
    non_incident_mask : (F, max_non_inc) bool tensor (True = valid)
    edges : (E, 2) long tensor
    """
    N = pts.shape[0]
    F = facet_indices.shape[0]

    # Gather facet vertex matrices: (F, 4, 4)
    P_F = pts[facet_indices]

    # Solve P_F @ n_F = 1 for all facets simultaneously
    ones_batch = torch.ones(F, 4, dtype=pts.dtype)
    try:
        n_F = torch.linalg.solve(P_F, ones_batch)  # (F, 4)
    except Exception:
        return torch.tensor(1e6, dtype=pts.dtype)

    # Gather non-incident vertex positions: (F, max_ni, 4)
    ni_pts = pts[non_incident_padded]  # may include garbage at masked positions

    # Dot products: (F, max_ni)
    dots = (ni_pts * n_F.unsqueeze(1)).sum(dim=-1)

    # Mask invalid entries
    dots = dots.masked_fill(~non_incident_mask, -1e6)

    # Soft violation
    violations = torch.nn.functional.softplus(dots - (1.0 - margin))
    violations = violations.masked_fill(~non_incident_mask, 0.0)
    facet_loss = violations.sum()

    # Determinant barrier
    dets = torch.det(P_F)
    det_loss = -det_weight * torch.log(dets.abs() + 1e-12).sum()

    # Repulsion
    gram = pts @ pts.T
    mask_upper = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    repulsion_loss = repulsion_weight * torch.nn.functional.softplus(
        gram[mask_upper] - 0.95).sum()

    # Edge bias
    edge_loss = torch.tensor(0.0, dtype=pts.dtype)
    if edge_weight > 0 and edges is not None:
        edge_dots = gram[edges[:, 0], edges[:, 1]]
        edge_loss = -edge_weight * edge_dots.mean()

    # Centroid
    centroid_loss = centroid_weight * (pts.mean(dim=0) ** 2).sum()

    return facet_loss + det_loss + repulsion_loss + edge_loss + centroid_loss


# ============================================================================
# Verification
# ============================================================================

def verify_realization(pts, target_facets):
    """
    Check if the convex hull of pts has exactly the target facets.

    Parameters
    ----------
    pts : (N, 4) numpy array
    target_facets : list of sorted 4-tuples

    Returns
    -------
    match : bool
    details : dict with f_vector, missing, extra facets
    """
    try:
        hull = ConvexHull(pts)
    except Exception as e:
        return False, {'error': str(e)}

    got_facets = set(tuple(sorted(s)) for s in hull.simplices)
    target_set = set(tuple(sorted(f)) for f in target_facets)

    missing = target_set - got_facets
    extra = got_facets - target_set

    # f-vector
    edges = set()
    tris = set()
    for f in got_facets:
        for e in combinations(f, 2):
            edges.add(e)
        for t in combinations(f, 3):
            tris.add(t)

    fv = [len(hull.vertices), len(edges), len(tris), len(got_facets)]

    # Degree sequence
    deg = Counter()
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    deg_seq = sorted([deg.get(i, 0) for i in range(len(pts))], reverse=True)

    # Norms
    norms = np.linalg.norm(pts, axis=1)

    match = len(missing) == 0 and len(extra) == 0
    return match, {
        'f_vector': fv,
        'degree_sequence': deg_seq,
        'n_missing': len(missing),
        'n_extra': len(extra),
        'inscribed': bool(np.allclose(norms, 1.0, atol=1e-6)),
        'norm_range': (float(norms.min()), float(norms.max())),
    }


# ============================================================================
# Main solver
# ============================================================================

def realize_inscribed(n_vertices, edges, facets, triangles=None,
                      n_restarts=50, n_steps=500, lr=0.02,
                      use_forman=False, verbose=True):
    """
    Find an inscribed realization of a simplicial 4-polytope on S^3.

    Parameters
    ----------
    n_vertices : int
    edges : list of (i, j) pairs
    facets : list of 4-tuples (tetrahedral facets)
    triangles : list of 3-tuples (2-faces), needed for Forman Laplacian
    n_restarts : int, number of random restarts
    n_steps : int, LBFGS iterations per restart
    lr : float, learning rate
    use_forman : bool, use Forman 1-Laplacian for initialization
    verbose : bool

    Returns
    -------
    result : dict with keys 'success', 'points', 'details', 'n_attempts'
    """
    N = n_vertices
    facets_sorted = [tuple(sorted(f)) for f in facets]
    edges_sorted = [tuple(sorted(e)) for e in edges]

    # Precompute facet tensors
    facet_idx, non_inc = build_facet_tensors(N, facets_sorted)
    edge_tensor = torch.tensor(edges_sorted, dtype=torch.long)

    # Pad non-incident lists for batched version
    max_ni = max(len(ni) for ni in non_inc)
    ni_padded = torch.zeros(len(facets_sorted), max_ni, dtype=torch.long)
    ni_mask = torch.zeros(len(facets_sorted), max_ni, dtype=torch.bool)
    for i, ni in enumerate(non_inc):
        ni_padded[i, :len(ni)] = ni
        ni_mask[i, :len(ni)] = True

    # Compute triangles from facets if not provided
    if triangles is None:
        tri_set = set()
        for f in facets_sorted:
            for t in combinations(f, 3):
                tri_set.add(t)
        triangles = sorted(tri_set)

    best_pts = None
    best_loss = float('inf')

    for restart in range(n_restarts):
        # --- Initialization ---
        if restart == 0:
            # Pure spectral (graph Laplacian)
            init = graph_laplacian_embedding(N, edges_sorted, dim=4)
        elif use_forman and restart == 1:
            # Forman 1-Laplacian
            init = forman_1laplacian_embedding(N, edges_sorted, triangles, dim=4)
        else:
            # Spectral + random rotation + perturbation
            if restart % 3 == 0:
                base = graph_laplacian_embedding(N, edges_sorted, dim=4)
            elif use_forman and restart % 3 == 1:
                base = forman_1laplacian_embedding(
                    N, edges_sorted, triangles, dim=4)
            else:
                base = np.random.randn(N, 4)
                base /= np.linalg.norm(base, axis=1, keepdims=True)

            # Random orthogonal rotation
            Q, _ = np.linalg.qr(np.random.randn(4, 4))
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1
            init = base @ Q.T

            # Perturbation
            noise_scale = 0.05 + 0.15 * np.random.rand()
            init += np.random.randn(N, 4) * noise_scale
            init /= np.linalg.norm(init, axis=1, keepdims=True)

        # --- Optimization ---
        # Optimize raw parameters; normalize inside loss
        raw = torch.tensor(init, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [raw], lr=lr, max_iter=5, line_search_fn='strong_wolfe')

        # Anneal margin from large to small
        for step in range(n_steps):
            margin = max(0.005, 0.1 * (1 - step / n_steps))
            det_w = max(0.01, 0.2 * (1 - step / n_steps))

            def closure():
                optimizer.zero_grad()
                normed = raw / torch.norm(raw, dim=1, keepdim=True)
                loss = realization_loss_batched(
                    normed, facet_idx, ni_padded, ni_mask, edge_tensor,
                    margin=margin, det_weight=det_w,
                    repulsion_weight=0.01, edge_weight=0.03,
                    centroid_weight=0.05)
                loss.backward()
                # Ensure gradient is contiguous for LBFGS
                if raw.grad is not None:
                    raw.grad = raw.grad.contiguous()
                return loss

            optimizer.step(closure)

        # --- Verification ---
        final_pts = raw.detach().numpy().copy()
        final_pts /= np.linalg.norm(final_pts, axis=1, keepdims=True)

        match, details = verify_realization(final_pts, facets_sorted)

        final_loss = float(realization_loss_batched(
            torch.tensor(final_pts, dtype=torch.float64),
            facet_idx, ni_padded, ni_mask, edge_tensor,
            margin=0.0, det_weight=0.0,
            repulsion_weight=0.0, edge_weight=0.0,
            centroid_weight=0.0).item())

        if verbose:
            status = "MATCH" if match else f"miss={details['n_missing']} extra={details['n_extra']}"
            print(f"  restart {restart:3d}: loss={final_loss:.4f}  "
                  f"f={details['f_vector']}  {status}")

        if match:
            if verbose:
                print(f"  SUCCESS at restart {restart}")
            return {
                'success': True,
                'points': final_pts,
                'details': details,
                'n_attempts': restart + 1,
                'final_loss': final_loss,
            }

        if final_loss < best_loss:
            best_loss = final_loss
            best_pts = final_pts

    # Failed
    if verbose:
        print(f"  FAILED after {n_restarts} restarts (best loss={best_loss:.4f})")

    _, best_details = verify_realization(best_pts, facets_sorted)
    return {
        'success': False,
        'points': best_pts,
        'details': best_details,
        'n_attempts': n_restarts,
        'final_loss': best_loss,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    """Command-line interface for inscribed realization."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Find inscribed realization of a simplicial 4-polytope")
    parser.add_argument('challenge', type=str,
                        help='Path to face_lattice JSON file')
    parser.add_argument('--restarts', type=int, default=50)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--forman', action='store_true',
                        help='Use Forman 1-Laplacian initialization')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    args = parser.parse_args()

    with open(args.challenge) as f:
        challenge = json.load(f)

    N = challenge['n_vertices']
    edges = [tuple(e) for e in challenge['edges']]
    facets = [tuple(f) for f in challenge['facets']]
    triangles = [tuple(t) for t in challenge.get('triangles', [])] or None

    print(f"Realizing N={N}  f={challenge['f_vector']}  "
          f"facets={len(facets)}  edges={len(edges)}")

    result = realize_inscribed(
        N, edges, facets, triangles=triangles,
        n_restarts=args.restarts, n_steps=args.steps,
        lr=args.lr, use_forman=args.forman)

    if result['success']:
        print(f"\nSUCCESS in {result['n_attempts']} attempts")
        print(f"  f-vector: {result['details']['f_vector']}")
        print(f"  inscribed: {result['details']['inscribed']}")

        out = {'points': result['points'].tolist()}
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(out, f, indent=2)
            print(f"  Saved to {args.output}")
        else:
            print(json.dumps(out))
    else:
        print(f"\nFAILED after {result['n_attempts']} attempts")
        print(f"  Best: {result['details']}")


if __name__ == '__main__':
    main()
