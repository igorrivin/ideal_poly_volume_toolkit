"""Manually inspect gradient propagation through experimental builders."""

import torch
from ideal_poly_volume_toolkit.geometry import delaunay_triangulation_indices, lob_fast


def _angles_for_triangle_real(x1, y1, x2, y2, x3, y3):
    """Compute triangle angles while retaining real-coordinate gradients."""

    def angle_at(ax, ay, bx, by, cx, cy):
        ux = bx - ax
        uy = by - ay
        vx = cx - ax
        vy = cy - ay
        dot = ux * vx + uy * vy
        cross = torch.abs(ux * vy - uy * vx)
        return torch.atan2(cross, dot)

    return (
        angle_at(x1, y1, x2, y2, x3, y3),
        angle_at(x2, y2, x3, y3, x1, y1),
        angle_at(x3, y3, x1, y1, x2, y2),
    )


# Test gradient computation for the original version
def build_Z_complex(thetas):
    Z = torch.empty(thetas.numel() + 2, dtype=torch.complex128, device=thetas.device)
    Z[0] = 1 + 0j
    Z[1] = 0 + 0j
    Z[2:] = torch.exp(1j * thetas.to(torch.complex128))
    return Z


# Test gradient computation for the fixed version
def build_Z_real(thetas):
    real_parts = torch.zeros(
        thetas.numel() + 2, dtype=torch.float64, device=thetas.device
    )
    imag_parts = torch.zeros(
        thetas.numel() + 2, dtype=torch.float64, device=thetas.device
    )
    real_parts[0] = 1.0
    imag_parts[0] = 0.0
    real_parts[1] = 0.0
    imag_parts[1] = 0.0
    real_parts[2:] = torch.cos(thetas)
    imag_parts[2:] = torch.sin(thetas)
    return real_parts, imag_parts


# Simple test
thetas = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64, requires_grad=True)

print("Testing gradient flow for complex version:")
Z_complex = build_Z_complex(thetas)
# Try to compute a simple function and backprop
loss_complex = (Z_complex.real**2).sum() + (Z_complex.imag**2).sum()
print(f"Loss value: {loss_complex.item()}")
loss_complex.backward()
print(f"Gradients on thetas: {thetas.grad}")

# Reset
thetas.grad = None

print("\nTesting gradient flow for real version:")
real_parts, imag_parts = build_Z_real(thetas)
loss_real = (real_parts**2).sum() + (imag_parts**2).sum()
print(f"Loss value: {loss_real.item()}")
loss_real.backward()
print(f"Gradients on thetas: {thetas.grad}")

# Now test with actual triangle computation
print("\n\nTesting with actual triangle volume computation:")

# Reset
thetas = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64, requires_grad=True)

# Build points and get triangulation
with torch.no_grad():
    real_np, imag_np = build_Z_real(thetas)
    Z_np = real_np.numpy() + 1j * imag_np.numpy()
    idx = delaunay_triangulation_indices(Z_np)
    print(f"Triangulation indices: {idx}")

# Test if we can compute angles with gradients
real_parts, imag_parts = build_Z_real(thetas)

# Pick first triangle
i, j, k = idx[0]
a1, a2, a3 = _angles_for_triangle_real(
    real_parts[i],
    imag_parts[i],
    real_parts[j],
    imag_parts[j],
    real_parts[k],
    imag_parts[k],
)

print(f"\nAngles: {a1.item():.4f}, {a2.item():.4f}, {a3.item():.4f}")

# Test Lobachevsky function
lob1 = lob_fast(a1, 10)
print(f"Lob value: {lob1.item()}")

# Test full volume
volume = lob_fast(a1, 10) + lob_fast(a2, 10) + lob_fast(a3, 10)
print(f"Triangle volume: {volume.item()}")

# Compute gradients
loss = -volume
loss.backward()
print(f"Gradients on thetas: {thetas.grad}")
