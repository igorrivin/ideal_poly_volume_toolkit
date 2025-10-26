"""
rivin_holonomy.py
-----------------
Reference implementation of the Penner–Rivin holonomy algorithm for an ideal triangulation
with optional shear data. Pure-Python (no external deps).

Model:
- Triangles are labeled 0..F-1, each with sides 0,1,2 in cyclic order (counterclockwise).
- Adjacency maps each oriented side (t, s) to (u, su, eid), where u is the triangle glued across side s,
  su is the side index in u, and eid is a global id for the underlying edge.
- Ori gives, for each global eid, a chosen orientation as ((t_from, s_from), (t_to, s_to)).
- Shears Z: dict eid -> real number (use 0.0 for zero-shear case).

Outputs:
- Generators as 2x2 matrices in SL(2,R) (project to PSL(2,R)).
"""

import math
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Any

def matmul(A, B):
    return [[A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
            [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]]

def matI():
    return [[1.0, 0.0], [0.0, 1.0]]

def matR():
    return [[1.0, 1.0], [-1.0, 0.0]]

def matL():
    return [[0.0, -1.0], [1.0, 1.0]]

def matX(z):
    ez = math.exp(z/2.0)
    return [[0.0, -ez], [1.0/ez, 0.0]]

class Triangulation:
    def __init__(self, F, adjacency, order, orientation):
        """
        F: number of triangles
        adjacency: dict (t, s) -> (u, su, eid)
        order: dict t -> list [0,1,2] (cyclic order of sides in triangle t)
        orientation: dict eid -> ((t_from, s_from), (t_to, s_to))
        """
        self.F = F
        self.Adj = dict(adjacency)
        self.Order = {t: list(order[t]) for t in order}
        self.Ori = dict(orientation)
        # basic sanity
        for (t,s), (u,su,eid) in self.Adj.items():
            rt = self.Adj.get((u,su))
            assert rt is not None and rt[0]==t and rt[1]==s and rt[2]==eid, "Adjacency must be symmetric"
        for t in range(F):
            assert t in self.Order and len(self.Order[t])==3, "Order missing for triangle {}".format(t)

    def dual_graph(self):
        """Return dual graph structure: dict t -> list of (u, s_in_t, s_in_u, eid)."""
        G = defaultdict(list)
        for (t,s), (u,su,eid) in self.Adj.items():
            G[t].append((u, s, su, eid))
        return G

def bfs_spanning_tree(G, root=0):
    parent = {root: None}
    parent_edge = {root: None}   # for v!=root: (eid, side_in_v, side_in_parent)
    Q = deque([root])
    while Q:
        v = Q.popleft()
        for (w, sv, sw, eid) in G[v]:
            if w not in parent:
                parent[w] = v
                parent_edge[w] = (eid, sw, sv)
                Q.append(w)
    return parent, parent_edge

def path_in_tree(parent, v, u):
    """Return list of vertices along the unique path v -> ... -> u in the tree."""
    Pv = []
    x = v
    while x is not None:
        Pv.append(x); x = parent[x]
    Pu = []
    x = u
    while x is not None:
        Pu.append(x); x = parent[x]
    i = len(Pv)-1; j = len(Pu)-1
    while i>=0 and j>=0 and Pv[i]==Pu[j]:
        i -= 1; j -= 1
    up = Pv[:i+1]
    down = list(reversed(Pu[:j+1]))
    return up + [Pv[i+1]] + down

def turn(order, side_in, side_out):
    i = order.index(side_in); j = order.index(side_out)
    d = (j - i) % 3
    if d == 1: return 'L'
    if d == 2: return 'R'
    raise ValueError("Degenerate turn inside triangle")

def tokens_for_path(T: Triangulation, parent, parent_edge, path):
    """
    Convert a path of triangles into a token list of 'L'/'R' and ('X', eid, sign).
    """
    tokens = []
    side_in_prev = None
    for k in range(len(path)-1):
        a, b = path[k], path[k+1]
        if parent[b] == a:
            eid, side_in_b, side_in_a = parent_edge[b]  # from a to b
        elif parent[a] == b:
            eid, side_in_a, side_in_b = parent_edge[a]  # from a to b
        else:
            raise RuntimeError("Non-tree step in tokens_for_path")
        if side_in_prev is not None:
            tokens.append(('T', a, side_in_prev, side_in_a))
        (tf, sf), (tt, st) = T.Ori[eid]
        if (tf == a and tt == b): sign = +1
        elif (tf == b and tt == a): sign = -1
        else: raise RuntimeError("Edge orientation inconsistent with step")
        tokens.append(('X', eid, sign))
        side_in_prev = side_in_b
    return _resolve_turns(T, tokens)

def _resolve_turns(T: Triangulation, tokens):
    resolved = []
    for tok in tokens:
        if isinstance(tok, tuple) and tok and tok[0]=='T':
            _, tri, s_in, s_out = tok
            resolved.append(turn(T.Order[tri], s_in, s_out))
        else:
            resolved.append(tok)
    return resolved

def tokens_to_matrix(tokens, Z):
    M = matI()
    for tok in tokens:
        if tok == 'L':
            M = matmul(M, matL())
        elif tok == 'R':
            M = matmul(M, matR())
        else:
            _, eid, sign = tok
            z = Z.get(eid, 0.0)
            if sign < 0: z = -z
            M = matmul(M, matX(z))
    return M

def generators_from_triangulation(T: Triangulation, Z, root=0):
    """
    Compute a generating set for the holonomy:
      one generator per non-tree dual edge.
    Returns: list of (u, v, tokens, matrix)
    """
    G = T.dual_graph()
    parent, parent_edge = bfs_spanning_tree(G, root=root)

    tree_edges = set()
    for v, p in parent.items():
        if p is None: continue
        tree_edges.add(tuple(sorted((v, p))))

    all_edges = set()
    for v, lst in G.items():
        for (w, sv, sw, eid) in lst:
            if v < w:
                all_edges.add((v, w, eid))

    gens = []
    for (u, v, eid) in all_edges:
        if tuple(sorted((u, v))) in tree_edges:
            continue
        path_u = path_in_tree(parent, root, u)
        path_v = path_in_tree(parent, root, v)

        tok1 = tokens_for_path(T, parent, parent_edge, path_u)          # root -> u
        (tf, sf), (tt, st) = T.Ori[eid]
        if (tf == u and tt == v): sign = +1
        elif (tf == v and tt == u): sign = -1
        else: raise RuntimeError("Orientation mismatch on non-tree edge")
        tok_mid = [('X', eid, sign)]
        tok2 = tokens_for_path(T, parent, parent_edge, list(reversed(path_v)))  # v -> root

        tokens = tok1 + tok_mid + tok2
        M = tokens_to_matrix(tokens, Z)
        gens.append((u, v, tokens, M))
    return gens

if __name__ == "__main__":
    print("rivin_holonomy.py loaded. Define a Triangulation(F, adjacency, order, orientation),")
    print("provide a shear dict Z (edge_id -> real), then call generators_from_triangulation(T, Z, root=0).")
    print("Each generator entry is (u, v, tokens, 2x2 matrix).")
