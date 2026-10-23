"""
pointset_to_fuchsian.py
-----------------------
Pipeline:
  (a) Start from a finite point set P in the complex plane (R^2).
  (b) Back-project to the unit sphere S^2 via inverse stereographic projection.
  (c) Compute the convex hull in R^3 to get an ideal polyhedron combinatorics.
  (d) Triangulate faces, compute complex shear–bend parameters via cross-ratios,
      feed shears (real parts) to the Penner–Rivin algorithm to get the Fuchsian group.
  (e) Since we get cross-ratios, recover exterior dihedral angles and (by a pulling triangulation)
      the hyperbolic volume via the Bloch–Wigner dilogarithm.

Requirements:
  - numpy
  - scipy (for 3D convex hull)
  - ideal_poly_volume_toolkit.geometry (for Bloch–Wigner)
  - trimesh, pyrender (optional: meshing & rendering)
  - rivin_holonomy.py in the same directory (Penner–Rivin core)
"""

from typing import List, Tuple, Dict, Any, Optional
import cmath
import math

import numpy as np

# Sentinel for infinity
INF = None

# Optional deps
try:
    from scipy.spatial import ConvexHull
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Local sibling import.  Keep the fallback for direct execution of this file,
# while normal package imports use the unambiguous relative path.
if __package__:
    from .geometry import bloch_wigner
    from .rivin_holonomy import Triangulation, generators_from_triangulation
else:  # pragma: no cover - direct-script compatibility
    from geometry import bloch_wigner
    from rivin_holonomy import Triangulation, generators_from_triangulation

# ---------- Stereographic and cross ratios ----------

def stereographic_inverse(z: complex) -> np.ndarray:
    if z is None or z is INF:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    x, y = z.real, z.imag
    d = x*x + y*y + 1.0
    return np.array([2*x/d, 2*y/d, (x*x + y*y - 1.0)/d], dtype=float)

def stereographic(p: np.ndarray) -> complex:
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    # North pole maps to infinity (None)
    if abs(z - 1.0) < 1e-14:
        return None
    denom = 1.0 - z
    if abs(denom) < 1e-15:
        return None
    return complex(x/denom, y/denom)

def cross_ratio(z1: complex, z2: complex, z3: complex, z4: complex) -> complex:
    return (z1 - z3) * (z2 - z4) / ((z1 - z4) * (z2 - z3))

def shear_bend_from_four(zx: complex, za: complex, zy: complex, zb: complex) -> complex:
    CR = cross_ratio(za, zx, zb, zy)
    if abs(CR.imag) < 1e-15 and CR.real < 0:
        CR += 1e-15j
    return cmath.log(CR)

def cr_axby(a, x, b, y):
    """
    Möbius-invariant cross ratio CR(a,x; b,y) = M(a)/M(b), M(z)=(z-x)/(z-y),
    allowing any of a,x,b,y to be INF (None). Returns complex.
    """
    # Degenerate edge (x=y) is not expected here.
    # Handle cases where either x or y is INF first.
    if x is INF and y is INF:
        raise ValueError("Degenerate configuration: x=y=INF")
    if x is INF:
        # M(z) ~ 1/(z - y)
        if a is INF or b is INF:
            # With x=INF, 'a' or 'b' being INF does not occur in our triangulated setup.
            raise ValueError("Unexpected INF pattern with x=INF.")
        return (b - y) / (a - y)
    if y is INF:
        # M(z) ~ (z - x)
        if a is INF or b is INF:
            raise ValueError("Unexpected INF pattern with y=INF.")
        return (a - x) / (b - x)
    # Now x,y are finite
    if a is INF and b is INF:
        return 1.0 + 0.0j  # M(a)=M(b)=1
    if a is INF:
        # 1 / ((b-x)/(b-y)) = (b - y)/(b - x)
        return (b - y) / (b - x)
    if b is INF:
        # (a-x)/(a-y)
        return (a - x) / (a - y)
    # All finite: standard expression
    return ((a - x)*(b - y)) / ((a - y)*(b - x))

def shear_bend_from_four_safe(zx, za, zy, zb):
    """
    Safe complex shear-bend with symbolic INF support:
      tau = log CR(za, zx; zb, zy).
    """
    CR = cr_axby(za, zx, zb, zy)
    # Avoid branch cut on negative real axis for numeric stability
    if abs(getattr(CR, "imag", 0.0)) < 1e-15 and getattr(CR, "real", 0.0) < 0:
        CR += 1e-15j
    return cmath.log(CR)

# ---------- Convex hull & faces ----------

def convex_hull_from_points_on_sphere(pts3: np.ndarray) -> Dict[str, Any]:
    if not _HAS_SCIPY:
        raise RuntimeError("SciPy (scipy.spatial.ConvexHull) is required for hull computation.")
    hull = ConvexHull(pts3)
    planes = {}
    for fi, (a,b,c,d) in enumerate(hull.equations):
        n = np.array([a,b,c], dtype=float)
        norm = np.linalg.norm(n)
        if norm == 0: continue
        n = n / norm
        key = tuple(np.round(np.append(n, d/norm), 8))
        group = planes.setdefault(key, {"normal": n, "triangles": []})
        group["triangles"].append(hull.simplices[fi])
    faces = []
    for group in planes.values():
        tris = group["triangles"]
        edge_count = {}
        for tri in tris:
            v = list(tri)
            for e in [(v[0],v[1]),(v[1],v[2]),(v[2],v[0])]:
                edge = tuple(sorted((int(e[0]), int(e[1]))))
                edge_count[edge] = edge_count.get(edge, 0) + 1

        # Internal diagonals occur twice in the triangulation of a coplanar
        # hull face.  Count them as undirected edges so their arbitrary Qhull
        # orientations cannot make them look like boundary edges.
        boundary = sorted(e for e, cnt in edge_count.items() if cnt == 1)
        if not boundary:
            continue

        adjacency = {}
        for u, v in boundary:
            adjacency.setdefault(u, []).append(v)
            adjacency.setdefault(v, []).append(u)
        if any(len(neighbours) != 2 for neighbours in adjacency.values()):
            raise ValueError("coplanar hull boundary is not a simple cycle")

        # Traverse deterministically, then orient the polygon consistently
        # with Qhull's outward plane normal.
        start = min(adjacency)
        first = min(adjacency[start])
        poly = [start, first]
        previous, current = start, first
        while True:
            candidates = [v for v in adjacency[current] if v != previous]
            if len(candidates) != 1:
                raise ValueError("coplanar hull boundary is not a simple cycle")
            nxt = candidates[0]
            if nxt == start:
                break
            if nxt in poly:
                raise ValueError("coplanar hull boundary closes prematurely")
            poly.append(nxt)
            previous, current = current, nxt

        if len(poly) != len(adjacency):
            raise ValueError("coplanar hull boundary has multiple components")

        polygon_normal = np.zeros(3, dtype=float)
        for i, u in enumerate(poly):
            polygon_normal += np.cross(pts3[u], pts3[poly[(i + 1) % len(poly)]])
        if np.dot(polygon_normal, group["normal"]) < 0:
            poly = [poly[0]] + list(reversed(poly[1:]))
        faces.append(poly)
    faces.sort(key=lambda face: tuple(face))
    edge_set=set()
    for f in faces:
        for i in range(len(f)):
            edge_set.add(tuple(sorted((f[i], f[(i+1)%len(f)]))))
    return {"vertices": pts3, "faces": faces, "edges": edge_set}

# ---------- Triangulation ----------

def triangulate_faces_with_face_id(faces: List[List[int]]):
    tris=[]; tri_face_id=[]
    for fi,f in enumerate(faces):
        if len(f)==3: tris.append(tuple(f)); tri_face_id.append(fi)
        else:
            a0=f[0]
            for i in range(1,len(f)-1):
                tris.append((a0,f[i],f[i+1])); tri_face_id.append(fi)
    return tris, tri_face_id

def build_penner_rivin_from_hull(pts3: np.ndarray, faces: List[List[int]]):
    N=pts3.shape[0]
    # Keep z as a plain list: stereographic() returns the INF sentinel (None)
    # for the north pole, and np.array(..., dtype=complex) silently coerces
    # None to NaN, so cr_axby's INF branches could never fire.
    z=[stereographic(pts3[i]) for i in range(N)]
    tris, tri_face_id = triangulate_faces_with_face_id(faces)
    Tverts=[tuple(t) for t in tris]

    def canonical_edge(u,v): return (u,v) if u<v else (v,u)
    oriented_to={}; edge_id={}; opp={}; edge_occ={}
    for ti,tri in enumerate(Tverts):
        for si in range(3):
            u,v=tri[si%3], tri[(si+1)%3]
            key=canonical_edge(u,v)
            edge_occ.setdefault(key, []).append((ti,si))
    for key,occ in edge_occ.items():
        if len(occ)!=2: continue
        (t1,s1),(t2,s2)=occ
        if key not in edge_id: edge_id[key]=len(edge_id)
        eid=edge_id[key]
        oriented_to[(t1,s1)]=(t2,s2,eid)
        oriented_to[(t2,s2)]=(t1,s1,eid)
        a=Tverts[t1][(s1+2)%3]; b=Tverts[t2][(s2+2)%3]
        opp[eid]=(a,b)
    orientation={}
    for key,eid in edge_id.items():
        (t1,s1),(t2,s2)=edge_occ[key]
        orientation[eid]=((t1,s1),(t2,s2))
    Adj={(t,s):(u,su,eid) for (t,s),(u,su,eid) in oriented_to.items()}
    Order={t:[0,1,2] for t in range(len(Tverts))}

    Z={}
    for key,eid in edge_id.items():
        (t1,s1),(t2,s2)=edge_occ[key]
        tri1=Tverts[t1]; tri2=Tverts[t2]
        u,v=tri1[s1%3], tri1[(s1+1)%3]
        a=tri1[(s1+2)%3]; b=tri2[(s2+2)%3]
        # INF-safe standard shear: tau = log CR(a, u; b, v) with
        # CR = M(a)/M(b), M(z) = (z-u)/(z-v).  (shear_bend_from_four used a
        # different cross-ratio slot arrangement and cannot handle INF.)
        tau=shear_bend_from_four_safe(z[u], z[a], z[v], z[b])
        Z[eid]=float(tau.real)

    T=Triangulation(F=len(Tverts), adjacency=Adj, order=Order, orientation=orientation)
    info={"edge_id":edge_id,"opposites":opp,"triangles":Tverts,"faces":faces,"stereo":z,"tri_face_id":tri_face_id}
    return T, Z, info

# ---------- High-level API ----------

def group_from_pointset(points: List[complex]) -> Dict[str,Any]:
    pts3 = np.vstack([stereographic_inverse(z) for z in points])
    hull = convex_hull_from_points_on_sphere(pts3)
    T, Z, info = build_penner_rivin_from_hull(pts3, hull["faces"])
    gens = generators_from_triangulation(T, Z, root=0)
    return {"Gamma_generators":[M for (_,_,_,M) in gens], "T":T, "Z":Z, "info":info, "faces":hull["faces"]}

def trace(M): return M[0][0]+M[1][1]

def invariant_trace_field_signature(generators):
    vals=[]
    for M in generators:
        M2=[[M[0][0]*M[0][0] + M[0][1]*M[1][0], M[0][0]*M[0][1] + M[0][1]*M[1][1]],
            [M[1][0]*M[0][0] + M[1][1]*M[1][0], M[1][0]*M[0][1] + M[1][1]*M[1][1]]]
        vals.append(trace(M2))
    def near_int(x,tol=1e-9):
        r=round(x); return (abs(x-r)<tol, r)
    flags=[near_int(v) for v in vals]
    all_int=all(f and float(r).is_integer() for f,r in flags)
    return {"traces_sq":vals, "all_integers":all_int, "rationality":flags}

# ---------- Dihedrals & Volume ----------

def edge_dihedrals(info: Dict[str,Any]) -> Dict[Tuple[int,int], float]:
    z=info["stereo"]; triangles=info["triangles"]; tri_face_id=info["tri_face_id"]; edge_id=info["edge_id"]
    dihedral={}
    for (u,v),eid in edge_id.items():
        occ=[]
        for t_idx, tri in enumerate(triangles):
            for s in range(3):
                a,b=tri[s%3], tri[(s+1)%3]
                if {a,b}=={u,v}: occ.append((t_idx,s))
        if len(occ)!=2: continue
        (t1,s1),(t2,s2)=occ
        if tri_face_id[t1]==tri_face_id[t2]:
            dihedral[tuple(sorted((u,v)))]=0.0; continue
        tri1=triangles[t1]; tri2=triangles[t2]
        x,y=tri1[s1%3], tri1[(s1+1)%3]
        a=tri1[(s1+2)%3]; b=tri2[(s2+2)%3]
        tau=shear_bend_from_four_safe(z[x], z[a], z[y], z[b])
        dihedral[tuple(sorted((u,v)))]=abs(float(tau.imag))
    return dihedral

def tetra_shape(z0, z1, z2, z3) -> complex:
    # Use INF-safe cross ratio if available
    CR = cr_axby(z0, z1, z2, z3)
    if abs(CR.imag) < 1e-15 and CR.real < 0: CR += 1e-15j
    return CR

def volume_from_hull(pts3: np.ndarray, faces: List[List[int]], apex: Optional[int]=None) -> float:
    z=[stereographic(p) for p in pts3]
    tris,_=triangulate_faces_with_face_id(faces)
    if apex is None: apex=0
    total=0.0
    for (a,b,c) in tris:
        if apex in (a,b,c): continue
        shp=tetra_shape(z[apex], z[a], z[b], z[c])
        total += abs(bloch_wigner(shp))
    return total

def volume_from_pointset(points: List[complex], apex_index: Optional[int]=None, add_north_pole: bool=False) -> float:
    """Compute volume using all hull simplices directly."""
    from scipy.spatial import ConvexHull
    pts3 = np.vstack([stereographic_inverse(z) for z in points])
    if add_north_pole:
        pts3 = np.vstack([pts3, np.array([[0.0,0.0,1.0]])])
    hull = ConvexHull(pts3)
    
    # Use all simplices
    z = [stereographic(p) for p in pts3]
    if apex_index is None:
        apex_index = 0
    
    total = 0.0
    for tri in hull.simplices:
        a, b, c = tri
        if apex_index in (a, b, c):
            continue
        shape = tetra_shape(z[apex_index], z[a], z[b], z[c])
        vol = abs(bloch_wigner(shape))
        total += vol
    
    return total

# ---------- Meshing & model conversion ----------

try:
    import trimesh as _trimesh
    _HAS_TRIMESH=True
except Exception:
    _HAS_TRIMESH=False

def klein_to_poincare(V: np.ndarray) -> np.ndarray:
    V=np.asarray(V,dtype=float)
    r2=np.sum(V*V,axis=1)
    s=np.sqrt(np.maximum(0.0,1.0-r2))
    denom=(1.0+s)[:,None]
    mask=(np.abs(r2-1.0)<1e-14)
    denom[mask]=1.0
    return V/denom

def triangulated_mesh_from_hull(pts3, faces, refinement=0, model="klein", as_trimesh=False):
    V=np.asarray(pts3,dtype=float).copy()
    tris,_=triangulate_faces_with_face_id(faces)
    F=np.array(tris,dtype=int)
    def refine_once(V,F):
        Vlist=V.tolist(); edge_mid={}
        def midpoint(i,j):
            a,b=min(i,j),max(i,j); key=(a,b)
            if key in edge_mid: return edge_mid[key]
            m=0.5*(V[a]+V[b]); idx=len(Vlist); Vlist.append(m.tolist()); edge_mid[key]=idx; return idx
        newF=[]
        for (i,j,k) in F:
            ij=midpoint(i,j); jk=midpoint(j,k); ki=midpoint(k,i)
            newF.extend([(i,ij,ki),(ij,j,jk),(ki,jk,k),(ij,jk,ki)])
        return np.array(Vlist,dtype=float), np.array(newF,dtype=int)
    for _ in range(max(0,refinement)):
        V,F=refine_once(V,F)
    if model.lower().startswith("poin"): V=klein_to_poincare(V)
    if as_trimesh and _HAS_TRIMESH: return _trimesh.Trimesh(vertices=V, faces=F, process=False)
    return V,F

def export_mesh_obj(filename: str, V: np.ndarray, F: np.ndarray):
    with open(filename,"w",encoding="utf-8") as f:
        for x,y,z in V: f.write(f"v {x:.10f} {y:.10f} {z:.10f}\n")
        for a,b,c in F: f.write(f"f {a+1} {b+1} {c+1}\n")
    return filename

def hull_to_mesh(points: List[complex], add_north_pole=False, refinement=0, model="poincare", as_trimesh=True):
    pts3=np.vstack([stereographic_inverse(z) for z in points])
    if add_north_pole:
        pts3=np.vstack([pts3, np.array([[0.0,0.0,1.0]])])
    hull=convex_hull_from_points_on_sphere(pts3)
    return triangulated_mesh_from_hull(pts3, hull["faces"], refinement=refinement, model=model, as_trimesh=as_trimesh)

# ---------- Volume gradients ----------

def _G_of_shape(z):
    """
    dVol = Im( G(z) * dz ), where
      G(z) = log|z|/(z-1) - log|1-z|/z.

    This is the standard differential of the Bloch-Wigner dilogarithm,

      dD(z) = log|z| d arg(1-z) - log|1-z| d arg(z).
    """
    if z is INF:
        return 0.0 + 0.0j
    z = complex(z)
    if z == 0.0 or z == 1.0:
        # Degenerate tetrahedra have zero volume and a singular differential.
        # Do not flatten merely near-degenerate shapes: their large, finite
        # derivatives provide an important direction away from collisions.
        return 0.0 + 0.0j

    # Preserve the first-order terms of these logarithms near their removable
    # cancellation points.  Directly evaluating abs(1-z), for example, rounds
    # away Re(z) when z is much smaller than one.
    x, y = z.real, z.imag
    z_minus_one = z - 1.0
    if abs(z_minus_one) < 0.5:
        dx = z_minus_one.real
        log_abs_z = 0.5 * math.log1p(2.0 * dx + dx * dx + y * y)
    else:
        log_abs_z = math.log(abs(z))
    if abs(z) < 0.5:
        log_abs_one_minus_z = 0.5 * math.log1p(-2.0 * x + x * x + y * y)
    else:
        log_abs_one_minus_z = math.log(abs(1.0 - z))

    return log_abs_z / z_minus_one - log_abs_one_minus_z / z

def _dlog_cr_coeffs(a, x, b, y):
    """
    Coefficients (S_a, S_x, S_b, S_y) such that d log z = S_a da + S_x dx + S_b db + S_y dy,
    for z = CR(a,x; b,y), with INF handled.
    """
    def inv_diff(u, v):
        if u is INF or v is INF:
            return 0.0 + 0.0j       # 1/(∞ - finite) = 0
        d = u - v
        if d == 0:
            # The logarithmic cross-ratio differential is singular for an
            # exact collision.  Preserve the existing zero convention only
            # at that endpoint; every nonzero separation uses the true
            # reciprocal, however small it is.
            return 0.0 + 0.0j
        return 1.0 / d

    # Reduced forms for INF cases
    if a is INF and b is INF:
        return 0j, 0j, 0j, 0j
    if a is INF:
        # z = (b - y)/(b - x)
        Sa = 0j
        Sb = inv_diff(b, y) - inv_diff(b, x)
        Sx = +inv_diff(b, x)
        Sy = -inv_diff(b, y)
        return Sa, Sx, Sb, Sy
    if b is INF:
        # z = (a - x)/(a - y)
        Sb = 0j
        Sa = inv_diff(a, x) - inv_diff(a, y)
        Sx = -inv_diff(a, x)
        Sy = +inv_diff(a, y)
        return Sa, Sx, Sb, Sy
    if x is INF:
        # z = (b - y)/(a - y)
        Sx = 0j
        Sb = inv_diff(b, y)
        Sa = -inv_diff(a, y)
        Sy = -inv_diff(b, y) + inv_diff(a, y)
        return Sa, Sx, Sb, Sy
    if y is INF:
        # z = (a - x)/(b - x)
        Sy = 0j
        Sa = inv_diff(a, x)
        Sb = -inv_diff(b, x)
        Sx = -inv_diff(a, x) + inv_diff(b, x)
        return Sa, Sx, Sb, Sy

    # General finite case
    Sa = inv_diff(a, x) - inv_diff(a, y)
    Sx = -inv_diff(a, x) + inv_diff(b, x)
    Sb = inv_diff(b, y) - inv_diff(b, x)
    Sy = -inv_diff(b, y) + inv_diff(a, y)
    return Sa, Sx, Sb, Sy

def volume_and_gradient_from_hull(pts3, faces, apex=None, variable_mask=None):
    """
    Volume and analytic gradient wrt planar coords of vertices.
    pts3: (N,3) on S^2; faces: polygon cycles (convex hull); apex: pulling vertex index.
    Returns (V, grad) where grad[i] = (∂V/∂x_i, ∂V/∂y_i).
    """
    # Note: We should use the hull simplices directly, not faces
    # This is a compatibility function - prefer volume_and_gradient_from_hull_simplices
    from scipy.spatial import ConvexHull
    hull = ConvexHull(pts3)
    return volume_and_gradient_from_hull_simplices(pts3, hull, apex, variable_mask)

def volume_and_gradient_from_hull_simplices(pts3, hull, apex=None, variable_mask=None):
    """
    Volume and analytic gradient using hull simplices directly.
    """
    zC = [stereographic(p) for p in pts3]          # complex chart; may be INF
    N  = len(zC)
    if apex is None:
        apex = 0
    tris = hull.simplices  # Use all simplices directly

    V = 0.0
    grad = np.zeros((N, 2), dtype=float)

    for (a_idx, b_idx, c_idx) in tris:
        if apex in (a_idx, b_idx, c_idx):
            continue
        z0, z1, z2, z3 = zC[apex], zC[a_idx], zC[b_idx], zC[c_idx]

        # shape for tetra (z0,z1,z2,z3) using CR(z2,z0; z3,z1)
        z = cr_axby(z2, z0, z3, z1)

        # The public volume is unsigned.  Its derivative therefore includes
        # the orientation sign of the signed Bloch-Wigner value.
        signed_volume = bloch_wigner(z)
        V += abs(signed_volume)

        # dVol = Im( G(z)*dz ), dz = z * d log z, d log z = S_a da + S_x dx + S_b db + S_y dy
        volume_sign = 1.0 if signed_volume > 0 else (-1.0 if signed_volume < 0 else 0.0)
        G  = volume_sign * _G_of_shape(z)
        Sa, Sx, Sb, Sy = _dlog_cr_coeffs(z2, z0, z3, z1)

        W_a = G * z * Sa
        W_x = G * z * Sx
        W_b = G * z * Sb
        W_y = G * z * Sy

        # Pack: Im(W)*dx + Re(W)*dy for each vertex
        if variable_mask is None or variable_mask[b_idx]:
            grad[b_idx, 0] += W_a.imag
            grad[b_idx, 1] += W_a.real
        if variable_mask is None or variable_mask[apex]:
            grad[apex, 0]  += W_x.imag
            grad[apex, 1]  += W_x.real
        if variable_mask is None or variable_mask[c_idx]:
            grad[c_idx, 0] += W_b.imag
            grad[c_idx, 1] += W_b.real
        if variable_mask is None or variable_mask[a_idx]:
            grad[a_idx, 0] += W_y.imag
            grad[a_idx, 1] += W_y.real

    return V, grad

def volume_and_gradient_from_pointset(points, apex_index=None, add_north_pole=False, variable_mask=None):
    """
    Wrapper: planar points -> hull -> volume and gradient.
    """
    from scipy.spatial import ConvexHull
    pts3 = np.vstack([stereographic_inverse(z) for z in points])
    if add_north_pole:
        pts3 = np.vstack([pts3, np.array([[0.0, 0.0, 1.0]])])
        if variable_mask is not None:
            variable_mask = variable_mask + [False]  # North pole is fixed
    hull = ConvexHull(pts3)
    return volume_and_gradient_from_hull_simplices(pts3, hull, apex=apex_index, variable_mask=variable_mask)

# ---------- Renderer with style knobs ----------

try:
    import pyrender as _pyrender
    _HAS_PYRENDER=True
except Exception:
    _HAS_PYRENDER=False

def _lambda_to_rgb_nm(lam_nm: float):
    lam=lam_nm
    if lam<380 or lam>700: return (0.0,0.0,0.0)
    if lam<440:
        t=(lam-380)/(440-380); r,g,b=(-t,0.0,1.0)
    elif lam<490:
        t=(lam-440)/(490-440); r,g,b=(0.0,t,1.0)
    elif lam<510:
        t=(lam-490)/(510-490); r,g,b=(0.0,1.0,1.0-t)
    elif lam<580:
        t=(lam-510)/(580-510); r,g,b=(t,1.0,0.0)
    elif lam<645:
        t=(lam-580)/(645-580); r,g,b=(1.0,1.0-t,0.0)
    else:
        t=(lam-645)/(700-645); r,g,b=(1.0,0.0,0.0)
    if lam<420: factor=0.3+0.7*(lam-380)/(420-380)
    elif lam>645: factor=0.3+0.7*(700-lam)/(700-645)
    else: factor=1.0
    gamma=0.8
    def correct(c): c=max(0.0,min(1.0,c*factor)); return c**gamma
    return (correct(r),correct(g),correct(b))

def _soap_iridescence_vertex_colors(mesh, camera_pos: np.ndarray,
                                    n_film: float = 1.33, n_env: float = 1.0,
                                    t0_nm: float = 500.0, t_amp_nm: float = 300.0,
                                    swirl_scale: float = 5.0,
                                    wavelengths_nm: Optional[List[float]] = None,
                                    alpha: float = 0.35):
    V=mesh.vertices; N=mesh.vertex_normals
    if len(N)==0:
        mesh.rezero(); mesh.remove_duplicate_faces(); N=mesh.vertex_normals
    cam=np.asarray(camera_pos,dtype=float).reshape(3)
    Vdir = cam[None,:]-V; L=np.linalg.norm(Vdir,axis=1)+1e-12; Vdir=Vdir/L[:,None]
    Nn=N/(np.linalg.norm(N,axis=1)[:,None]+1e-12)
    cos_theta=np.abs(np.sum(Nn*Vdir,axis=1))
    x,y,z=V[:,0],V[:,1],V[:,2]
    phi=np.arctan2(y,x)
    rho=np.linalg.norm(V,axis=1)+1e-12
    theta=np.arccos(np.clip(z/rho,-1,1))
    t_nm=t0_nm + t_amp_nm*(np.sin(swirl_scale*phi)*np.cos(0.5*swirl_scale*theta))
    lambdas=np.array(wavelengths_nm if wavelengths_nm else [450.0,530.0,610.0],dtype=float)
    r12=((n_env-n_film)/(n_env+n_film))**2; r23=r12; sqrt_term=math.sqrt(max(0.0,r12*r23))
    C=np.zeros((V.shape[0],3),dtype=float)
    for lam_nm in lambdas:
        delta=(4.0*math.pi*n_film/lam_nm)*t_nm*cos_theta
        R=r12+r23+2.0*sqrt_term*np.cos(delta); R=np.clip(R,0.0,1.0)
        rgb=np.array(_lambda_to_rgb_nm(lam_nm))[None,:]; C += (R[:,None])*rgb
    Cmax=np.maximum(1e-8, C.max(axis=1,keepdims=True)); C=np.clip(C/Cmax,0.0,1.0)
    A=np.full((V.shape[0],1), np.clip(alpha,0.0,1.0), dtype=float)
    rgba=np.concatenate([C,A],axis=1); return (rgba*255.0+0.5).astype(np.uint8)

def _look_at(eye, target, up=np.array([0,0,1.0])):
    eye=np.asarray(eye,dtype=float).reshape(3); target=np.asarray(target,dtype=float).reshape(3); up=np.asarray(up,dtype=float).reshape(3)
    z=eye-target; z=z/(np.linalg.norm(z)+1e-12); x=np.cross(up,z); x=x/(np.linalg.norm(x)+1e-12); y=np.cross(z,x)
    T=np.eye(4); T[:3,:3]=np.vstack([x,y,z]).T; T[:3,3]=eye; return T

def render_snapshot(points: List[complex],
                    add_north_pole: bool = False,
                    refinement: int = 1,
                    model: str = "poincare",
                    width: int = 1280, height: int = 960,
                    background: Tuple[float,float,float,float] = (1.0,1.0,1.0,1.0),
                    camera_pos: Optional[Tuple[float,float,float]] = None,
                    target: Optional[Tuple[float,float,float]] = None,
                    transparency_alpha: float = 0.35,
                    light_setup: str = "three_point",
                    soap_iridescence: bool = True,
                    # style knobs
                    base_rgba: Tuple[float,float,float,float] = (0.6,0.8,1.0,0.35),
                    metallic: float = 0.05, roughness: float = 0.05,
                    key_intensity: float = 4.0, fill_intensity: float = 1.5, rim_intensity: float = 2.5,
                    ring_count: int = 8, ring_radius: float = 4.5, ring_height: float = 2.0, ring_intensity: float = 2.0,
                    film_n: float = 1.33, film_t0_nm: float = 500.0, film_t_amp_nm: float = 300.0, film_swirl_scale: float = 5.0,
                    film_wavelengths_nm: Optional[List[float]] = None,
                    outfile: str = "render.png") -> str:
    if not (_HAS_TRIMESH and _HAS_PYRENDER):
        raise RuntimeError("render_snapshot requires trimesh and pyrender (pip install trimesh pyrender).")
    mesh = hull_to_mesh(points, add_north_pole=add_north_pole, refinement=refinement, model=model, as_trimesh=True)
    if camera_pos is None: camera_pos=(2.7,2.1,1.6)
    if target is None: target=(0.0,0.0,0.0)
    if soap_iridescence:
        vcols=_soap_iridescence_vertex_colors(mesh, np.array(camera_pos),
                                              n_film=film_n, t0_nm=film_t0_nm, t_amp_nm=film_t_amp_nm,
                                              swirl_scale=film_swirl_scale, wavelengths_nm=film_wavelengths_nm,
                                              alpha=transparency_alpha)
        mesh.visual.vertex_colors=vcols
        material = _pyrender.MetallicRoughnessMaterial(baseColorFactor=[1.0,1.0,1.0,max(0.05,transparency_alpha)],
                                                       metallicFactor=metallic, roughnessFactor=roughness,
                                                       alphaMode="BLEND", doubleSided=True)
    else:
        base=np.array(base_rgba,dtype=float); base[3]=transparency_alpha
        mesh.visual.vertex_colors=(base[None,:]*255.0+0.5).astype(np.uint8)
        material = _pyrender.MetallicRoughnessMaterial(baseColorFactor=[float(base[0]),float(base[1]),float(base[2]),max(0.05,transparency_alpha)],
                                                       metallicFactor=metallic, roughnessFactor=roughness,
                                                       alphaMode="BLEND", doubleSided=True)
    pm = _pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
    scene = _pyrender.Scene(bg_color=np.array(background,dtype=float))
    scene.add(pm)
    cam=_pyrender.PerspectiveCamera(yfov=np.deg2rad(45.0))
    scene.add(cam, pose=_look_at(np.array(camera_pos), np.array(target)))
    def add_three_point(scene):
        key=_pyrender.DirectionalLight(color=np.ones(3), intensity=float(key_intensity))
        fill=_pyrender.DirectionalLight(color=np.ones(3), intensity=float(fill_intensity))
        rim=_pyrender.DirectionalLight(color=np.ones(3), intensity=float(rim_intensity))
        scene.add(key, pose=_look_at(np.array([3,2,2]), np.zeros(3)))
        scene.add(fill, pose=_look_at(np.array([-2,3,1]), np.zeros(3)))
        scene.add(rim, pose=_look_at(np.array([-3,-2,2.5]), np.zeros(3)))
    def add_ring(scene, n=ring_count, radius=ring_radius, z=ring_height, intensity=ring_intensity):
        for k in range(int(n)):
            ang=2*math.pi*k/max(1,int(n)); pos=np.array([radius*math.cos(ang), radius*math.sin(ang), z])
            scene.add(_pyrender.PointLight(color=np.ones(3), intensity=float(intensity)), pose=_look_at(pos, np.zeros(3)))
    if light_setup=="ring": add_ring(scene)
    else: add_three_point(scene)
    r=_pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    color,_=r.render(scene, flags=_pyrender.RenderFlags.RGBA)
    import imageio; imageio.imwrite(outfile, color); r.delete(); return outfile
