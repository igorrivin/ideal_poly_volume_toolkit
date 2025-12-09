#!/usr/bin/env python3
"""
Properly extract faces from plantri planar embedding output.

The key insight: plantri's adjacency format lists neighbors in CYCLIC ORDER
around each vertex, representing the planar embedding. We need to extract
the faces of this embedding, not just find triangular 3-cycles.
"""

import networkx as nx
from typing import List, Tuple, Dict

def extract_faces_from_planar_embedding(n_vertices: int, adj_cyclic: Dict[int, List[int]]) -> List[Tuple[int, int, int]]:
    """
    Extract triangular faces from a planar embedding.

    Args:
        n_vertices: Number of vertices
        adj_cyclic: Adjacency dictionary where neighbors are in cyclic order

    Returns:
        List of triangular faces as (v0, v1, v2) tuples
    """
    # Build NetworkX graph with planar embedding
    G = nx.Graph()

    # Add edges
    for v in range(n_vertices):
        for neighbor in adj_cyclic[v]:
            G.add_edge(v, neighbor)

    # Create planar embedding (mapping each vertex to cyclic order of neighbors)
    embedding = {v: adj_cyclic[v] for v in range(n_vertices)}

    # Set the planar embedding
    nx.set_edge_attributes(G, {}, 'embedding')

    # Extract faces using the combinatorial embedding
    # We'll traverse each edge in both directions and collect the faces
    faces_set = set()

    # For each directed edge (u, v), find the face to its right
    for u in range(n_vertices):
        neighbors = adj_cyclic[u]
        for i, v in enumerate(neighbors):
            # Start at edge u -> v
            # Find the next edge in the face by going to v and taking the
            # next edge clockwise from the reverse direction

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
                    # Sanity check - shouldn't happen
                    break

            # Store face as sorted tuple for deduplication
            if len(face) == 3:
                face_tuple = tuple(sorted(face))
                faces_set.add(face_tuple)

    return sorted(list(faces_set))


def test_face_extraction():
    """Test with a simple example: tetrahedron (K4)."""

    print("Testing face extraction with tetrahedron (K4)...")

    # Tetrahedron: 4 vertices, all connected
    # Planar embedding with vertices 0,1,2,3
    n = 4

    # One valid planar embedding of K4 (tetrahedron)
    adj = {
        0: [1, 2, 3],    # Neighbors of 0 in cyclic order
        1: [0, 3, 2],    # Neighbors of 1 in cyclic order
        2: [0, 1, 3],    # Neighbors of 2 in cyclic order
        3: [0, 2, 1],    # Neighbors of 3 in cyclic order
    }

    faces = extract_faces_from_planar_embedding(n, adj)

    print(f"Found {len(faces)} faces:")
    for face in faces:
        print(f"  {face}")

    # Tetrahedron should have 4 triangular faces
    assert len(faces) == 4, f"Expected 4 faces, got {len(faces)}"
    print("Test passed!")


if __name__ == '__main__':
    test_face_extraction()
