#!/usr/bin/env python3
"""
Visualize the 7-vertex configuration to understand the structure.
"""

import numpy as np
import matplotlib.pyplot as plt

# The configuration from our optimization
Z = [complex(0, 0), complex(1, 0),
     complex(-0.9959, 0.1942), complex(-0.4018, -0.9331),
     complex(0.8441, -1.2772), complex(0.3669, -0.5609)]

# Create plot
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Plot vertices
for i, z in enumerate(Z):
    ax.plot(z.real, z.imag, 'bo', markersize=10)
    ax.annotate(f'{i}', (z.real, z.imag), xytext=(5, 5), textcoords='offset points')

# The triangulation from our result
faces = [(0, 1, 6), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 5, 6), (1, 2, 6), (2, 3, 6)]

# Plot the triangles
for (a, b, c) in faces:
    triangle = [Z[a], Z[b], Z[c], Z[a]]
    xs = [z.real for z in triangle]
    ys = [z.imag for z in triangle]
    ax.plot(xs, ys, 'r-', alpha=0.5, linewidth=1)
    
    # Fill triangle with light color
    ax.fill(xs[:-1], ys[:-1], alpha=0.1, color='red')

ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('Real')
ax.set_ylabel('Imaginary')
ax.set_title('7-Vertex Configuration (Delaunay Triangulation)')

# Add circle at infinity for reference
circle = plt.Circle((0, 0), 2, fill=False, linestyle='--', color='gray', alpha=0.5)
ax.add_patch(circle)

plt.tight_layout()
plt.savefig('7vertex_triangulation.png', dpi=150)
print("Saved visualization to 7vertex_triangulation.png")

# Check convexity
print("\nAnalyzing configuration:")
print(f"Number of faces: {len(faces)}")
print(f"All vertices used: {sorted(set(sum(faces, ())))}")

# The issue: This is the Delaunay triangulation in the plane,
# not the convex hull of the lifted points!
print("\nNote: This appears to be a planar Delaunay triangulation,")
print("not the convex hull of an ideal polyhedron in hyperbolic space.")
print("The missing face explains why Euler characteristic = 1 instead of 2.")