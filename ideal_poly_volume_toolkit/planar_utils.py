"""
Utilities for working with planar graphs and embeddings.
"""

from typing import List, Tuple, Dict


def extract_faces_from_planar_embedding(n_vertices: int, adj_cyclic: Dict[int, List[int]]) -> List[Tuple[int, int, int]]:
    """
    Extract triangular faces from a planar embedding.

    In plantri's ASCII output format, the adjacency lists are given in CYCLIC ORDER
    around each vertex, representing the planar embedding. This function extracts
    the faces (triangles) from this embedding.

    Args:
        n_vertices: Number of vertices
        adj_cyclic: Adjacency dictionary where neighbors are in cyclic order

    Returns:
        List of triangular faces as (v0, v1, v2) tuples, sorted
    """
    faces_set = set()

    # For each directed edge (u, v), find the face to its right
    for u in range(n_vertices):
        neighbors = adj_cyclic[u]
        for i, v in enumerate(neighbors):
            # Start at edge u -> v
            # Find the face by following edges clockwise around the face boundary

            face = [u, v]
            current = v
            prev = u

            # Follow the face boundary
            while True:
                # At vertex 'current', we came from 'prev'
                # Find 'prev' in current's neighbor list
                current_neighbors = adj_cyclic[current]

                try:
                    prev_idx = current_neighbors.index(prev)
                except ValueError:
                    # Edge not found - broken embedding
                    break

                # Next vertex in face is the one BEFORE prev in cyclic order
                # (going clockwise around the face)
                next_v = current_neighbors[(prev_idx - 1) % len(current_neighbors)]

                if next_v == u:
                    # Completed the face
                    break

                face.append(next_v)
                prev = current
                current = next_v

                if len(face) > n_vertices:
                    # Sanity check - shouldn't happen for valid embeddings
                    break

            # Store face as sorted tuple for deduplication
            if len(face) == 3:
                face_tuple = tuple(sorted(face))
                faces_set.add(face_tuple)

    return sorted(list(faces_set))
