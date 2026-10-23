"""Inspect gradient flow through the original angular parameterization."""

import torch
import numpy as np
from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    triangle_volume_from_points_torch,
)


def random_angles(K, rng):
    return 2 * np.pi * rng.random(K)


def build_Z(thetas: torch.Tensor) -> torch.Tensor:
    Z = torch.empty(thetas.numel() + 2, dtype=torch.complex128, device=thetas.device)
    Z[0] = 1 + 0j
    Z[1] = 0 + 0j
    Z[2:] = torch.exp(1j * thetas.to(torch.complex128))
    return Z


# Test with original implementation
rng = np.random.default_rng(0)
K = 3
thetas = torch.tensor(random_angles(K, rng), dtype=torch.float64, requires_grad=True)

print(f"Initial thetas: {thetas}")

# Build complex numbers and triangulate
with torch.no_grad():
    Z_np = build_Z(thetas).detach().numpy()
    idx = delaunay_triangulation_indices(Z_np)
    print(f"Triangulation has {len(idx)} triangles")
    print(f"Indices: {idx}")

# Compute volume with gradients
Z_t = build_Z(thetas)
total = torch.zeros((), dtype=torch.float64)

for i, (i_idx, j_idx, k_idx) in enumerate(idx):
    vol = triangle_volume_from_points_torch(
        Z_t[i_idx], Z_t[j_idx], Z_t[k_idx], series_terms=96
    )
    print(f"Triangle {i}: volume = {vol.item():.6f}")
    total = total + vol

print(f"\nTotal volume: {total.item()}")

# Compute gradients
loss = -total
loss.backward()
print(f"Gradients on thetas: {thetas.grad}")

# Check which vertices are involved in triangles
vertices_used = set()
for i, j, k in idx:
    vertices_used.update([i, j, k])
print(f"\nVertices used in triangulation: {sorted(vertices_used)}")

# Map back to thetas
print("Vertex mapping:")
print("0: Z[0] = 1+0j (fixed)")
print("1: Z[1] = 0+0j (fixed)")
for i, theta_idx in enumerate(range(K)):
    print(f"{i + 2}: Z[{i + 2}] = exp(i*thetas[{theta_idx}])")

# Explicitly check gradient flow by perturbing each theta
print("\n\nChecking gradient flow by finite differences:")
eps = 1e-6
for i in range(K):
    thetas_plus = thetas.detach().clone()
    thetas_plus[i] += eps

    Z_plus = build_Z(thetas_plus)
    vol_plus = torch.zeros((), dtype=torch.float64)
    for i_idx, j_idx, k_idx in idx:
        vol_plus = vol_plus + triangle_volume_from_points_torch(
            Z_plus[i_idx], Z_plus[j_idx], Z_plus[k_idx], series_terms=96
        )

    finite_diff_grad = (vol_plus.item() - total.item()) / eps
    print(f"Finite diff gradient for theta[{i}]: {finite_diff_grad:.6f}")
