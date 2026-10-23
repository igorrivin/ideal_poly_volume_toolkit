"""Manually reproduce the historical Torch LBFGS diagnostic."""

import torch
from ideal_poly_volume_toolkit.geometry import triangle_volume_from_points_torch, _LobFn

# Test if the issue is with the custom autograd function _LobFn
print("Testing if custom autograd function is the issue...\n")

# Test 1: Simple function with standard PyTorch ops
print("Test 1: Standard PyTorch function")
theta1 = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
opt1 = torch.optim.LBFGS([theta1], lr=1.0)


def closure1():
    opt1.zero_grad()
    loss = -(torch.sin(theta1) + 0.5 * torch.cos(2 * theta1))  # some smooth function
    loss.backward()
    print(
        f"  theta={theta1.item():.4f}, loss={loss.item():.4f}, grad={theta1.grad.item():.4f}"
    )
    return loss


print("Before:", theta1.item())
opt1.step(closure1)
print("After:", theta1.item())

# Test 2: Using the custom _LobFn directly
print("\n\nTest 2: Custom _LobFn autograd function")
theta2 = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
opt2 = torch.optim.LBFGS([theta2], lr=1.0)


def closure2():
    opt2.zero_grad()
    # Use the Lobachevsky function directly
    lob_val = _LobFn.apply(theta2, 96)
    loss = -lob_val
    loss.backward()
    print(
        f"  theta={theta2.item():.4f}, loss={loss.item():.4f}, grad={theta2.grad.item():.4f}"
    )
    return loss


print("Before:", theta2.item())
opt2.step(closure2)
print("After:", theta2.item())

# Test 3: Full triangle volume computation
print("\n\nTest 3: Full triangle volume (our actual case)")
theta3 = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
opt3 = torch.optim.LBFGS([theta3], lr=1.0)


def closure3():
    opt3.zero_grad()
    z0 = torch.tensor(0 + 0j, dtype=torch.complex128)
    z1 = torch.tensor(1 + 0j, dtype=torch.complex128)
    z2 = torch.exp(1j * theta3.to(torch.complex128))
    volume = triangle_volume_from_points_torch(z0, z1, z2, series_terms=96)
    loss = -volume
    loss.backward()
    print(
        f"  theta={theta3.item():.4f}, loss={loss.item():.4f}, grad={theta3.grad.item():.4f}"
    )
    return loss


print("Before:", theta3.item())
opt3.step(closure3)
print("After:", theta3.item())

# Test 4: Try with Adam optimizer instead
print("\n\nTest 4: Same problem with Adam optimizer")
theta4 = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
opt4 = torch.optim.Adam([theta4], lr=0.1)

print("Initial theta:", theta4.item())
for i in range(10):
    opt4.zero_grad()
    z0 = torch.tensor(0 + 0j, dtype=torch.complex128)
    z1 = torch.tensor(1 + 0j, dtype=torch.complex128)
    z2 = torch.exp(1j * theta4.to(torch.complex128))
    volume = triangle_volume_from_points_torch(z0, z1, z2, series_terms=96)
    loss = -volume
    loss.backward()
    opt4.step()
    if i % 3 == 0:
        print(
            f"  Step {i}: theta={theta4.item():.4f}, volume={volume.item():.4f}, grad={theta4.grad.item():.4f}"
        )

print("Final theta:", theta4.item())
