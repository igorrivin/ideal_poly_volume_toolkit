import argparse, numpy as np
from math import sqrt
from ideal_poly_volume_toolkit.geometry import lift_to_sphere_with_inf, ideal_poly_volume_via_hull_project_back, ideal_poly_volume_via_delaunay

def unit(V): V = np.asarray(V, float); return V/np.linalg.norm(V, axis=1, keepdims=True)

def rotate_vec_a_to_b(a, b):
    a = a/np.linalg.norm(a); b = b/np.linalg.norm(b)
    v = np.cross(a, b); c = np.dot(a, b)
    if np.allclose(v, 0) and c > 0: return np.eye(3)
    if np.allclose(v, 0) and c < 0:
        axis = np.array([1,0,0]) if abs(a[0]) < 0.9 else np.array([0,1,0])
        v = np.cross(a, axis); v = v/np.linalg.norm(v)
        K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        return np.eye(3)+2*K@K
    s = np.linalg.norm(v); K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+K+K@K*((1-c)/(s*s))

def inverse_stereographic(X):
    x,y,z = X[:,0], X[:,1], X[:,2]; denom = (1-z); return (x/denom) + 1j*(y/denom)

def vertices_platonic(name):
    phi = (1+sqrt(5))/2
    if name == 'tetrahedron':
        V = np.array([[ 1, 1, 1],[-1,-1, 1],[-1, 1,-1],[ 1,-1,-1]], float)
    elif name == 'octahedron':
        V = np.array([[ 1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], float)
    elif name == 'cube':
        V = np.array([[ 1, 1, 1],[ 1, 1,-1],[ 1,-1, 1],[ 1,-1,-1],
                      [-1, 1, 1],[-1, 1,-1],[-1,-1, 1],[-1,-1,-1]], float)
    elif name == 'icosahedron':
        V = np.array([[0, 1, phi],[0,-1, phi],[0, 1,-phi],[0,-1,-phi],
                      [ 1, phi,0],[-1, phi,0],[ 1,-phi,0],[-1,-phi,0],
                      [ phi,0, 1],[ phi,0,-1],[-phi,0, 1],[-phi,0,-1]], float)
    elif name == 'dodecahedron':
        a = 1/phi; b = phi; V = []
        for s1 in (-1,1):
            for s2 in (-1,1):
                for s3 in (-1,1):
                    V.append([s1, s2, s3])
        for s1 in (-1,1):
            for s2 in (-1,1):
                V.append([0, s1*a, s2*b]); V.append([s1*a, s2*b, 0]); V.append([s1*b, 0, s2*a])
        V = np.array(V, float)
    else:
        raise ValueError('unknown solid')
    return unit(V)

def build_finite_complex(name):
    V = vertices_platonic(name)
    R = rotate_vec_a_to_b(V[0], np.array([0,0,1.0]))
    Vr = (R @ V.T).T
    Z = inverse_stereographic(Vr[1:])  # finite ones; ∞ is the dropped vertex
    return Z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solid', type=str, required=True,
                    choices=['tetrahedron','octahedron','cube','dodecahedron','icosahedron'])
    ap.add_argument('--mode', type=str, default='exact', choices=['fast','exact'])
    ap.add_argument('--dps', type=int, default=200)
    args = ap.parse_args()

    Z = build_finite_complex(args.solid)
    mode = 'eval_only' if args.mode == 'exact' else 'fast'

    vol_del  = ideal_poly_volume_via_delaunay(Z, mode=mode, dps=args.dps, series_terms=96)

    W = np.empty(len(Z)+1, dtype=np.complex128); W[0] = np.inf+0j; W[1:] = Z
    vol_hull = ideal_poly_volume_via_hull_project_back(W, index_inf=0, mode=mode, dps=args.dps, series_terms=96)

    print(f'[Regular ideal {args.solid}]')
    print(f'  volume via Delaunay:     {vol_del:.12f}')
    print(f'  volume via hull→project: {vol_hull:.12f}')
    print(f'  |difference|:            {abs(vol_del - vol_hull):.3e}')

if __name__ == '__main__':
    main()
