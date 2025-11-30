"""
Combinatorial type enumeration for convex polyhedra inscribed in the sphere.

Generates random point sets on the sphere and counts distinct combinatorial types
using canonical graph labelings to avoid O(N^2) isomorphism checks.
"""

import numpy as np
from scipy.spatial import ConvexHull
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import os

try:
    import pynauty
    PYNAUTY_AVAILABLE = True
except ImportError:
    PYNAUTY_AVAILABLE = False


def generate_random_points_on_sphere(n_points, rng=None):
    """
    Generate n_points uniformly distributed on the unit sphere using Marsaglia's method.

    Args:
        n_points: Number of points to generate
        rng: numpy random generator (if None, uses default)

    Returns:
        Array of shape (n_points, 3) with points on unit sphere
    """
    if rng is None:
        rng = np.random.default_rng()

    # Marsaglia's method: uniform sampling on sphere
    # Generate points in [-1, 1]^2 and reject those outside unit circle
    points = []
    while len(points) < n_points:
        # Generate candidate points
        batch_size = max(n_points - len(points), 100)
        x = rng.uniform(-1, 1, batch_size)
        y = rng.uniform(-1, 1, batch_size)
        r_sq = x**2 + y**2

        # Keep only points inside unit circle
        mask = r_sq < 1
        x = x[mask]
        y = y[mask]
        r_sq = r_sq[mask]

        # Map to sphere: (x, y, r^2) -> (2x√(1-r^2), 2y√(1-r^2), 1-2r^2)
        sqrt_term = np.sqrt(1 - r_sq)
        sphere_x = 2 * x * sqrt_term
        sphere_y = 2 * y * sqrt_term
        sphere_z = 1 - 2 * r_sq

        for i in range(len(x)):
            if len(points) < n_points:
                points.append([sphere_x[i], sphere_y[i], sphere_z[i]])

    return np.array(points)


def extract_graph_from_hull(vertices_3d):
    """
    Extract the 1-skeleton (edge graph) from the convex hull of vertices.

    Args:
        vertices_3d: N x 3 array of vertex coordinates

    Returns:
        adjacency: dict mapping vertex index to list of adjacent vertex indices
        n_vertices: number of vertices
    """
    hull = ConvexHull(vertices_3d)

    # Build edge list from hull faces (simplices)
    edges = set()
    for simplex in hull.simplices:
        # Each face is a triangle with 3 edges
        v0, v1, v2 = simplex
        edges.add(tuple(sorted([v0, v1])))
        edges.add(tuple(sorted([v1, v2])))
        edges.add(tuple(sorted([v2, v0])))

    # Build adjacency list
    n_vertices = len(vertices_3d)
    adjacency = {i: [] for i in range(n_vertices)}
    for v1, v2 in edges:
        adjacency[v1].append(v2)
        adjacency[v2].append(v1)

    return adjacency, n_vertices


def compute_canonical_hash(adjacency, n_vertices):
    """
    Compute a canonical hash for a graph using pynauty.

    Isomorphic graphs will have the same canonical hash.

    Args:
        adjacency: dict mapping vertex index to list of adjacent vertices
        n_vertices: number of vertices in the graph

    Returns:
        tuple: canonical hash (hashable and comparable)
    """
    if not PYNAUTY_AVAILABLE:
        raise ImportError("pynauty not installed. Install with: pip install pynauty")

    # Create pynauty graph
    g = pynauty.Graph(number_of_vertices=n_vertices, directed=False, adjacency_dict=adjacency)

    # Get canonical labeling
    # canon_label returns a canonical permutation that maps the graph to a canonical form
    canonical_labeling = pynauty.canon_label(g)

    # Build canonical adjacency list using the relabeling
    # canonical_labeling[i] tells us what the new label of vertex i is
    canonical_edges = []
    for v1 in range(n_vertices):
        for v2 in adjacency[v1]:
            if v1 < v2:  # Only add each edge once
                # Map to canonical labels
                c1, c2 = canonical_labeling[v1], canonical_labeling[v2]
                canonical_edges.append(tuple(sorted([c1, c2])))

    # Sort edges for consistent ordering
    canonical_edges = tuple(sorted(canonical_edges))

    # Also include vertex degrees in canonical order
    canonical_degrees = tuple(sorted([len(adjacency[i]) for i in range(n_vertices)]))

    return (n_vertices, canonical_degrees, canonical_edges)


def enumerate_combinatorial_types(n_vertices, n_samples, seed=None, verbose=True):
    """
    Enumerate distinct combinatorial types of convex polyhedra with n_vertices on the sphere.

    Args:
        n_vertices: Number of vertices in each polyhedron
        n_samples: Number of random samples to generate
        seed: Random seed (for reproducibility)
        verbose: If True, print progress updates

    Returns:
        dict with keys:
        - 'n_vertices': number of vertices
        - 'n_samples': number of samples generated
        - 'n_types': number of distinct combinatorial types found
        - 'type_counts': dict mapping canonical hash to count
        - 'type_examples': dict mapping canonical hash to example vertex set
    """
    if not PYNAUTY_AVAILABLE:
        raise ImportError("pynauty not installed. Install with: pip install pynauty")

    rng = np.random.default_rng(seed)

    # Track unique types
    type_counts = defaultdict(int)
    type_examples = {}  # Store one example of each type

    # Track failures (non-simplicial hulls, degenerate cases)
    n_failures = 0

    for i in range(n_samples):
        if verbose and (i + 1) % max(1, n_samples // 10) == 0:
            print(f"  Sample {i + 1}/{n_samples}: {len(type_counts)} types found")

        try:
            # Generate random points on sphere
            vertices = generate_random_points_on_sphere(n_vertices, rng)

            # Extract graph from convex hull
            adjacency, n_verts = extract_graph_from_hull(vertices)

            # Compute canonical hash
            canonical_hash = compute_canonical_hash(adjacency, n_verts)

            # Track this type
            type_counts[canonical_hash] += 1

            # Store example if first time seeing this type
            if canonical_hash not in type_examples:
                type_examples[canonical_hash] = vertices.copy()

        except Exception as e:
            # Handle degenerate cases (coplanar points, numerical issues, etc.)
            n_failures += 1
            if verbose and n_failures == 1:
                print(f"  Warning: encountered degenerate case: {e}")

    if verbose:
        print(f"\nCompleted: {len(type_counts)} distinct types found")
        if n_failures > 0:
            print(f"  ({n_failures} degenerate cases skipped)")

    return {
        'n_vertices': n_vertices,
        'n_samples': n_samples,
        'n_types': len(type_counts),
        'type_counts': dict(type_counts),
        'type_examples': type_examples,
        'n_failures': n_failures,
    }


def format_enumeration_report(result):
    """
    Format enumeration results as a readable string.

    Args:
        result: Dictionary returned by enumerate_combinatorial_types

    Returns:
        Formatted string report
    """
    report = []
    report.append(f"\n{'='*60}")
    report.append(f"Combinatorial Type Enumeration: {result['n_vertices']} vertices")
    report.append(f"{'='*60}")
    report.append(f"Samples generated: {result['n_samples']}")
    report.append(f"Distinct types found: {result['n_types']}")
    report.append(f"Degenerate cases: {result['n_failures']}")
    report.append("")

    # Sort types by frequency
    sorted_types = sorted(result['type_counts'].items(), key=lambda x: x[1], reverse=True)

    report.append("Top 10 most common types:")
    for i, (type_hash, count) in enumerate(sorted_types[:10], 1):
        n_verts, degrees, edges = type_hash
        frequency = 100 * count / result['n_samples']
        report.append(f"  {i}. Type with degree sequence {degrees}: {count} times ({frequency:.2f}%)")

    if len(sorted_types) > 10:
        report.append(f"  ... and {len(sorted_types) - 10} more rare types")

    return "\n".join(report)


def _worker_enumerate_types(args):
    """
    Worker function for parallel enumeration.

    Each worker maintains its own local type_counts dictionary - no locking needed!
    Results are merged after all workers complete.

    Args:
        args: tuple of (n_vertices, n_samples_for_worker, worker_seed, worker_id, verbose)

    Returns:
        dict with local results from this worker
    """
    n_vertices, n_samples, seed, worker_id, verbose = args

    if not PYNAUTY_AVAILABLE:
        raise ImportError("pynauty not installed. Install with: pip install pynauty")

    # Each worker gets its own RNG seeded differently
    rng = np.random.default_rng(seed)

    # Local type tracking (no shared state - no locking!)
    type_counts = defaultdict(int)
    type_examples = {}
    n_failures = 0

    for i in range(n_samples):
        if verbose and (i + 1) % max(1, n_samples // 5) == 0:
            print(f"  Worker {worker_id}: {i + 1}/{n_samples} samples, {len(type_counts)} types")

        try:
            vertices = generate_random_points_on_sphere(n_vertices, rng)
            adjacency, n_verts = extract_graph_from_hull(vertices)
            canonical_hash = compute_canonical_hash(adjacency, n_verts)

            type_counts[canonical_hash] += 1

            if canonical_hash not in type_examples:
                type_examples[canonical_hash] = vertices.copy()

        except Exception as e:
            n_failures += 1
            if verbose and n_failures == 1:
                print(f"  Worker {worker_id}: Warning - degenerate case: {e}")

    return {
        'type_counts': dict(type_counts),
        'type_examples': type_examples,
        'n_failures': n_failures,
    }


def enumerate_combinatorial_types_parallel(n_vertices, n_samples, seed=None,
                                          n_workers=None, verbose=True):
    """
    Enumerate distinct combinatorial types using parallel processing.

    Uses a partition-and-merge strategy: each worker processes samples independently
    with no shared state (no locking!), then results are merged at the end.

    Args:
        n_vertices: Number of vertices in each polyhedron
        n_samples: Total number of random samples to generate
        seed: Random seed (for reproducibility)
        n_workers: Number of parallel workers (default: cpu_count())
        verbose: If True, print progress updates

    Returns:
        dict with keys:
        - 'n_vertices': number of vertices
        - 'n_samples': number of samples generated
        - 'n_types': number of distinct combinatorial types found
        - 'type_counts': dict mapping canonical hash to count
        - 'type_examples': dict mapping canonical hash to example vertex set
        - 'n_workers': number of workers used
    """
    if not PYNAUTY_AVAILABLE:
        raise ImportError("pynauty not installed. Install with: pip install pynauty")

    if n_workers is None:
        n_workers = cpu_count()

    if verbose:
        print(f"Using {n_workers} parallel workers")

    # Divide samples among workers
    samples_per_worker = n_samples // n_workers
    extra_samples = n_samples % n_workers

    # Generate independent seeds for each worker
    if seed is not None:
        rng = np.random.default_rng(seed)
        worker_seeds = [rng.integers(0, 2**32) for _ in range(n_workers)]
    else:
        worker_seeds = [None] * n_workers

    # Prepare worker arguments
    worker_args = []
    for i in range(n_workers):
        n_samples_for_worker = samples_per_worker + (1 if i < extra_samples else 0)
        worker_args.append((n_vertices, n_samples_for_worker, worker_seeds[i], i, verbose))

    # Run workers in parallel
    if verbose:
        print(f"Starting parallel enumeration...")

    with Pool(n_workers) as pool:
        worker_results = pool.map(_worker_enumerate_types, worker_args)

    # Merge results from all workers
    if verbose:
        print(f"Merging results from {n_workers} workers...")

    merged_type_counts = defaultdict(int)
    merged_type_examples = {}
    total_failures = 0

    for result in worker_results:
        # Merge type counts
        for type_hash, count in result['type_counts'].items():
            merged_type_counts[type_hash] += count

            # Keep one example (arbitrary which worker's example we keep)
            if type_hash not in merged_type_examples:
                merged_type_examples[type_hash] = result['type_examples'][type_hash]

        total_failures += result['n_failures']

    if verbose:
        print(f"\nCompleted: {len(merged_type_counts)} distinct types found")
        if total_failures > 0:
            print(f"  ({total_failures} degenerate cases skipped)")

    return {
        'n_vertices': n_vertices,
        'n_samples': n_samples,
        'n_types': len(merged_type_counts),
        'type_counts': dict(merged_type_counts),
        'type_examples': merged_type_examples,
        'n_failures': total_failures,
        'n_workers': n_workers,
    }
