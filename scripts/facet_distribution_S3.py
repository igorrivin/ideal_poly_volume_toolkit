#!/usr/bin/env python3
"""
Study the distribution of facet counts for convex hulls of N random
points on S^3 ⊂ R^4.

For each N, sample many random configurations and record the full
f-vector (f0, f1, f2, f3).  For simplicial 4-polytopes (the generic
case), f3 = f1 - f0 by Dehn-Sommerville, so the distribution of f3
is determined by the edge count.

This tells us:
  1. How concentrated f3 is for fixed N
  2. What range of N produces a target f3
  3. Whether the distribution is singular enough to cause trouble
     for facet-constrained optimization
"""

import numpy as np
from scipy.spatial import ConvexHull
from itertools import combinations
from collections import Counter
from multiprocessing import Pool
import sys


def random_S3_points(n, rng):
    """Generate n uniform random points on S^3."""
    pts = rng.standard_normal((n, 4))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def f_vector(pts):
    """
    Compute the full f-vector (f0, f1, f2, f3) of the convex hull of
    points in R^4.
    """
    hull = ConvexHull(pts)
    n = len(pts)

    # hull.simplices = boundary facets (tetrahedra in 4D)
    f3 = len(hull.simplices)

    # Extract all faces
    edges = set()
    triangles = set()
    for simplex in hull.simplices:
        s = tuple(sorted(simplex))
        for i, j in combinations(s, 2):
            edges.add((i, j))
        for tri in combinations(s, 3):
            triangles.add(tri)

    f0 = n
    f1 = len(edges)
    f2 = len(triangles)

    return f0, f1, f2, f3


def sample_worker(args):
    """Generate one random configuration and return its f-vector."""
    n_vertices, seed = args
    rng = np.random.default_rng(seed)
    pts = random_S3_points(n_vertices, rng)
    try:
        fv = f_vector(pts)
        return fv
    except Exception:
        return None


def study_distribution(n_vertices, n_samples=2000, n_workers=32):
    """Sample f-vector distribution for N points on S^3."""
    work = [(n_vertices, 77777 + n_vertices * 1000 + i)
            for i in range(n_samples)]

    with Pool(processes=n_workers) as pool:
        results = pool.map(sample_worker, work)

    results = [r for r in results if r is not None]
    return results


def print_distribution(n_vertices, results):
    """Print summary statistics for the f-vector distribution."""
    f0s = [r[0] for r in results]
    f1s = [r[1] for r in results]
    f2s = [r[2] for r in results]
    f3s = [r[3] for r in results]

    # Check Euler characteristic
    eulers = [f0 - f1 + f2 - f3 for f0, f1, f2, f3 in results]

    # Check simpliciality: f2 should equal 2*f3 for simplicial
    simplicial = [f2 == 2 * f3 for _, _, f2, f3 in results]

    # Check Dehn-Sommerville: f3 = f1 - f0 for simplicial
    ds_check = [f3 == f1 - f0 for f0, f1, _, f3 in results]

    f3_arr = np.array(f3s)
    f1_arr = np.array(f1s)

    print(f"\n{'='*70}")
    print(f"N = {n_vertices} vertices on S^3  ({len(results)} samples)")
    print(f"{'='*70}")

    print(f"\n  All simplicial: {all(simplicial)}")
    print(f"  Euler χ = 0:    {all(e == 0 for e in eulers)}")
    print(f"  Dehn-Sommerville (f3=f1-f0): {all(ds_check)}")

    print(f"\n  f3 (facets):")
    print(f"    min    = {f3_arr.min()}")
    print(f"    max    = {f3_arr.max()}")
    print(f"    mean   = {f3_arr.mean():.2f}")
    print(f"    median = {np.median(f3_arr):.1f}")
    print(f"    std    = {f3_arr.std():.2f}")
    print(f"    range  = {f3_arr.max() - f3_arr.min()}")

    print(f"\n  f1 (edges):")
    print(f"    min    = {f1_arr.min()}")
    print(f"    max    = {f1_arr.max()}")
    print(f"    mean   = {f1_arr.mean():.2f}")
    print(f"    std    = {f1_arr.std():.2f}")

    # Distribution of f3
    counts = Counter(f3s)
    print(f"\n  f3 distribution:")
    for val in sorted(counts.keys()):
        pct = 100 * counts[val] / len(results)
        bar = '#' * int(pct)
        print(f"    {val:5d}: {counts[val]:5d}  ({pct:5.1f}%)  {bar}")

    # Upper bound theorem: max f3 for neighborly polytope
    # f3 = f0*(f0-3)/2 for neighborly 4-polytope
    f3_ubt = n_vertices * (n_vertices - 3) // 2
    # Lower bound theorem: f3 >= 3*f0 - 10 (for simplicial, f0 >= 6)
    f3_lbt = max(3 * n_vertices - 10, n_vertices - 1)  # simplex bound

    print(f"\n  Theoretical bounds:")
    print(f"    LBT: f3 >= {f3_lbt}")
    print(f"    UBT: f3 <= {f3_ubt} (neighborly)")
    print(f"    Observed range: [{f3_arr.min()}, {f3_arr.max()}]")
    print(f"    UBT utilization: {100*f3_arr.mean()/f3_ubt:.1f}%")


def main():
    vertex_counts = [6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100]

    if len(sys.argv) > 1:
        vertex_counts = [int(x) for x in sys.argv[1:]]

    n_samples = 2000

    print("Facet distribution for random ideal polytopes in H^4")
    print(f"(convex hulls of N uniform random points on S^3)")
    print(f"Samples per N: {n_samples}")

    for n in vertex_counts:
        results = study_distribution(n, n_samples=n_samples)
        print_distribution(n, results)

    # Summary table
    print(f"\n\n{'='*70}")
    print("Summary: mean f3 vs N")
    print(f"{'='*70}")
    print(f"{'N':>6s}  {'mean f3':>10s}  {'std f3':>8s}  {'min':>6s}  {'max':>6s}  {'f3/N':>6s}")
    for n in vertex_counts:
        results = study_distribution(n, n_samples=500)
        f3s = np.array([r[3] for r in results])
        print(f"{n:6d}  {f3s.mean():10.2f}  {f3s.std():8.2f}  "
              f"{f3s.min():6d}  {f3s.max():6d}  {f3s.mean()/n:6.2f}")


if __name__ == '__main__':
    main()
