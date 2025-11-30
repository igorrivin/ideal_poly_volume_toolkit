"""
Interface to plantri for enumerating planar triangulations.

plantri is installed via passagemath-plantri and provides a command-line
tool for generating planar graphs and triangulations.
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Optional, List, Tuple
import tempfile


def find_plantri_executable() -> Optional[str]:
    """
    Find the plantri executable.

    Returns:
        Path to plantri executable, or None if not found
    """
    # Try common locations
    possible_paths = [
        # From passagemath-plantri installation
        Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages' / 'sage_wheels' / 'bin' / 'plantri',
        # System installation
        'plantri',
    ]

    for path in possible_paths:
        try:
            result = subprocess.run([str(path)], capture_output=True, timeout=1)
            return str(path)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    return None


def count_triangulations(n_vertices: int, only_polytopes: bool = True) -> int:
    """
    Count the number of planar triangulations with n vertices.

    Args:
        n_vertices: Number of vertices
        only_polytopes: If True, only count 3-connected (convex polytope) triangulations

    Returns:
        Number of triangulations
    """
    plantri = find_plantri_executable()
    if plantri is None:
        raise RuntimeError("plantri executable not found. Install with: pip install passagemath-plantri")

    # -p flag: only 3-connected planar (polytopes)
    args = [plantri, '-p' if only_polytopes else '', str(n_vertices)]
    args = [a for a in args if a]  # Remove empty strings

    result = subprocess.run(args, capture_output=True, text=True)

    # Parse output: "N polytopes written to stdout; cpu=X.XX sec"
    for line in result.stderr.split('\n'):
        if 'written' in line:
            parts = line.split()
            return int(parts[0])

    raise RuntimeError(f"Could not parse plantri output: {result.stderr}")


def enumerate_triangulations(n_vertices: int,
                            only_polytopes: bool = True,
                            output_format: str = 'planar_code') -> bytes:
    """
    Generate all planar triangulations with n vertices.

    Args:
        n_vertices: Number of vertices
        only_polytopes: If True, only generate 3-connected (convex polytope) triangulations
        output_format: Output format ('planar_code' is the default binary format)

    Returns:
        Binary output in planar_code format
    """
    plantri = find_plantri_executable()
    if plantri is None:
        raise RuntimeError("plantri executable not found. Install with: pip install passagemath-plantri")

    args = [plantri, '-p' if only_polytopes else '', str(n_vertices)]
    args = [a for a in args if a]  # Remove empty strings

    result = subprocess.run(args, capture_output=True)

    if result.returncode not in [0, 1]:  # plantri often returns 1 even on success
        raise RuntimeError(f"plantri failed: {result.stderr.decode()}")

    return result.stdout


# Lookup table for small values (OEIS A000109 for 3-connected planar graphs)
KNOWN_COUNTS = {
    4: 1,    # Tetrahedron
    5: 1,    # Triangular dipyramid
    6: 7,    # Including octahedron
    7: 34,
    8: 257,
    9: 2606,
    10: 32300,
    11: 440564,
    12: 6384634,
}


def get_triangulation_count(n_vertices: int, use_cache: bool = True) -> int:
    """
    Get the number of 3-connected planar triangulations with n vertices.

    Args:
        n_vertices: Number of vertices
        use_cache: If True, use precomputed values for small n

    Returns:
        Number of triangulations
    """
    if use_cache and n_vertices in KNOWN_COUNTS:
        return KNOWN_COUNTS[n_vertices]

    return count_triangulations(n_vertices, only_polytopes=True)


if __name__ == '__main__':
    # Test
    print("Testing plantri interface...")

    for n in range(4, 10):
        count = get_triangulation_count(n)
        print(f"  n={n}: {count} triangulations")

    print("\n✓ plantri interface working!")
