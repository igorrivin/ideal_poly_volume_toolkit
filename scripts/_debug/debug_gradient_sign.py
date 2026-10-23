"""Manually compare Torch volume gradients with finite differences."""

import torch
import numpy as np
from ideal_poly_volume_toolkit.geometry import (
    triangle_volume_from_points_torch,
    _angles_for_triangle_torch,
    lob_fast,
)

# Test at theta = 0.5
theta = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

z0 = torch.tensor(0 + 0j, dtype=torch.complex128)
z1 = torch.tensor(1 + 0j, dtype=torch.complex128)
z2 = torch.exp(1j * theta.to(torch.complex128))

# Get the angles
a1, a2, a3 = _angles_for_triangle_torch(z0, z1, z2)
print("Triangle angles at theta=0.5:")
print(f"  a1 = {a1.item():.4f} rad ({a1.item() * 180 / np.pi:.2f}°)")
print(f"  a2 = {a2.item():.4f} rad ({a2.item() * 180 / np.pi:.2f}°)")
print(f"  a3 = {a3.item():.4f} rad ({a3.item() * 180 / np.pi:.2f}°)")
print(
    f"  Sum = {(a1 + a2 + a3).item():.4f} rad ({(a1 + a2 + a3).item() * 180 / np.pi:.2f}°)"
)

# Compute volume
volume = triangle_volume_from_points_torch(z0, z1, z2, series_terms=96)
print(f"\nVolume = {volume.item():.6f}")

# Compute gradient
volume.backward()
volume_gradient = theta.grad.item()
loss_gradient = -volume_gradient
print(f"Gradient of volume w.r.t. theta = {volume_gradient:.6f}")

# Now let's manually check which angle changes how
print("\n\nManual gradient check:")
eps = 1e-6

# Reset for manual computation
theta.grad = None
theta_plus = theta + eps
theta_minus = theta - eps

# Compute z2 at theta +/- eps
z2_plus = torch.exp(1j * theta_plus.to(torch.complex128))
z2_minus = torch.exp(1j * theta_minus.to(torch.complex128))

# Get angles at both points
a1_plus, a2_plus, a3_plus = _angles_for_triangle_torch(z0, z1, z2_plus)
a1_minus, a2_minus, a3_minus = _angles_for_triangle_torch(z0, z1, z2_minus)

# Volume at both points
vol_plus = lob_fast(a1_plus, 96) + lob_fast(a2_plus, 96) + lob_fast(a3_plus, 96)
vol_minus = lob_fast(a1_minus, 96) + lob_fast(a2_minus, 96) + lob_fast(a3_minus, 96)

fd_grad = (vol_plus - vol_minus) / (2 * eps)
print(f"Finite difference gradient = {fd_grad.item():.6f}")

# Check how each angle changes
print("\nAngle derivatives w.r.t. theta:")
print(f"  da1/dθ ≈ {((a1_plus - a1_minus) / (2 * eps)).item():.4f}")
print(f"  da2/dθ ≈ {((a2_plus - a2_minus) / (2 * eps)).item():.4f}")
print(f"  da3/dθ ≈ {((a3_plus - a3_minus) / (2 * eps)).item():.4f}")

# The issue might be with the loss = -volume in the optimizer
print("\n\nThe issue:")
print("In the optimizer, we use loss = -volume (to maximize)")
print(
    f"So gradient of loss w.r.t. theta = -{volume_gradient:.6f} = {loss_gradient:.6f}"
)
print("LBFGS moves in direction opposite to gradient of loss")
print(f"So it should move theta by approximately {-loss_gradient:+.6f}")

# But wait, let's check what the optimizer actually sees
theta2 = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
z0 = torch.tensor(0 + 0j, dtype=torch.complex128)
z1 = torch.tensor(1 + 0j, dtype=torch.complex128)
z2 = torch.exp(1j * theta2.to(torch.complex128))
volume2 = triangle_volume_from_points_torch(z0, z1, z2, series_terms=96)
loss = -volume2
loss.backward()
print("\nWhat the optimizer sees:")
print(f"  theta.grad after loss.backward() = {theta2.grad.item():.6f}")
