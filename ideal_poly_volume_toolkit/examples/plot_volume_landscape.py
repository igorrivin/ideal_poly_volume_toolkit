import numpy as np
import torch
from ideal_poly_volume_toolkit.geometry import triangle_volume_from_points_torch

# Plot the volume as a function of theta
thetas = np.linspace(-np.pi, 2*np.pi, 1000)
volumes = []
grads = []

for theta_val in thetas:
    theta = torch.tensor(theta_val, dtype=torch.float64, requires_grad=True)
    z0 = torch.tensor(0+0j, dtype=torch.complex128)
    z1 = torch.tensor(1+0j, dtype=torch.complex128)
    z2 = torch.exp(1j * theta.to(torch.complex128))
    
    volume = triangle_volume_from_points_torch(z0, z1, z2, series_terms=96)
    volumes.append(volume.item())
    
    # Compute gradient
    volume.backward()
    grads.append(theta.grad.item())

volumes = np.array(volumes)
grads = np.array(grads)

# Find maxima
max_idx = np.argmax(volumes)
print(f"Maximum volume: {volumes[max_idx]:.6f}")
print(f"at theta = {thetas[max_idx]:.6f} rad ({thetas[max_idx]*180/np.pi:.2f}°)")
print(f"Expected at theta = {np.pi/2:.6f} rad (90.00°)")

# Check the loss landscape around theta=0.5
idx_05 = np.argmin(np.abs(thetas - 0.5))
print(f"\nAt theta=0.5:")
print(f"  Volume: {volumes[idx_05]:.6f}")
print(f"  Gradient: {grads[idx_05]:.6f}")
print(f"  Should increase volume by moving right")

# Check for issues in the landscape
print(f"\nLandscape statistics:")
print(f"  Min volume: {np.min(volumes):.6f}")
print(f"  Max volume: {np.max(volumes):.6f}")
print(f"  Number of points with negative volume: {np.sum(volumes < 0)}")

# Find where gradient changes sign around 0.5
for i in range(idx_05-5, idx_05+5):
    print(f"  theta={thetas[i]:.4f}: vol={volumes[i]:.4f}, grad={grads[i]:.4f}")